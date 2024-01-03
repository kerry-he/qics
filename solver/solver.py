import numpy as np
import math
import time

from utils import point, linear as lin
from solver.stepper.basic import BasicStepper
from solver.stepper.aggressive import AggressiveStepper
from solver.stepper.comb import CombinedStepper
from solver.syssolver import SysSolver

class Solver():
    def __init__(
        self, 
        model, 
        max_iter = 1000, 
        max_time = np.inf,
        feas_tol = 1e-8,
        gap_tol = 1e-8,
        verbose  = True,
        syssolver = None,
        stepper = None
    ):
        self.max_iter = max_iter
        self.max_time = max_time
        self.feas_tol = feas_tol
        self.gap_tol = gap_tol
        self.verbose = verbose

        self.model = model
        syssolver = SysSolver(model) if (syssolver is None) else syssolver
        self.stepper = CombinedStepper(syssolver, model) if (stepper is None) else stepper

        return
    
    def solve(self):
        self.setup_solver()
        # self.rescale_model()
        self.setup_point()

        while True:
            if self.step_and_check():
                break

        # Get solve data
        self.get_gap_feas()

        if self.verbose:
            if not self.status == "step_failure":
                print("%5d" % (self.num_iters), " %8.1e" % (self.mu), " %8.1e" % (self.point.tau), " %8.1e" % (self.point.kappa),
                    " | %10.3e" % (self.p_obj), " %10.3e" % (self.d_obj), " %10.3e" % (self.obj_gap), 
                    " | %10.3e" % (self.x_feas), " %10.3e" % (self.y_feas), " %10.3e" % (self.z_feas), end="")

            print()
            print("Opt value:  %.10f" % (self.p_obj))
            print("Tolerance:  %.10e" % (self.p_obj - self.d_obj))
            print("Solve time: %.10f" % (time.time() - self.solve_time), " seconds")
            print()

        return
    
    def step_and_check(self):
        # Check termination
        if self.num_iters >= self.max_iter:
            if self.verbose:
                print("Maximum iteration limit reached")
            return True
        
        if time.time() - self.solve_time >= self.max_time:
            if self.verbose:
                print("Maximum time limit reached")
            return True
        
        # Update barrier parameter mu
        self.calc_mu()

        # Get solve data
        self.get_gap_feas()      

        # Check optimality
        if abs(self.obj_gap) <= self.gap_tol:
            # Check feasibility
            if self.x_feas <= self.feas_tol and self.y_feas <= self.feas_tol and self.z_feas <= self.feas_tol:
                if self.verbose:
                    self.status = "solved"
                    print("Solved to desired tolerance")
                return True

        # Check infeasibility
        if self.point.tau <= self.feas_tol:
            if self.d_obj > 0:
                if self.verbose:
                    self.status = "primal_infeas"
                    print("Detected primal infeasibility")
                return True       

            if self.p_obj < 0:
                if self.verbose:
                    self.status = "dual_infeas"
                    print("Detected dual infeasibility")
                return True            

        if self.verbose:
            if self.num_iters % 20 == 0:
                print("=======================================================================================================================================================")
                print("%5s" % "iter", " %8s" % "mu", " %8s" % "tau", " %8s" % "kappa",
                    " | %10s" % "p_obj", " %10s" % "d_obj", " %10s" % "gap", 
                    " | %10s" % "x_feas", " %10s" % "y_feas", " %10s" % "z_feas",
                    " | %6s" % "step", "%10s" % "dir_tol", "%10s" % "prox", " %5s" % "alpha")
                print("=======================================================================================================================================================")                
            
            print("%5d" % (self.num_iters), " %8.1e" % (self.mu), " %8.1e" % (self.point.tau), " %8.1e" % (self.point.kappa),
                  " | %10.3e" % (self.p_obj), " %10.3e" % (self.d_obj), " %10.3e" % (self.obj_gap), 
                  " | %10.3e" % (self.x_feas), " %10.3e" % (self.y_feas), " %10.3e" % (self.z_feas), end="")
            
        # Step
        success = self.stepper.step(self.model, self.point, self.mu, self.verbose)

        if not success:
            if self.verbose:
                self.status = "step_failure"
                print("\nFailed to step")
                return True            

        self.num_iters += 1

        return False

    def setup_solver(self):
        self.num_iters = 0
        self.solve_time = time.time()
        return

    def setup_point(self):
        model = self.model
        self.point = point.Point(model)

        self.point.tau[0]   = 1.
        self.point.kappa[0] = 1.

        for (k, cone_k) in enumerate(model.cones):
            self.point.s[model.cone_idxs[k]] = cone_k.set_init_point()
            assert cone_k.get_feas()
            self.point.z_views[k][:] = -cone_k.get_grad()

        if model.use_G:
            self.point.x[:] = np.linalg.pinv(np.vstack((model.A, model.G))) @ np.vstack((model.b, model.h - self.point.s))
            self.point.y[:] = np.linalg.pinv(model.A.T) @ (-model.G.T @ self.point.z - model.c)
        else:
            self.point.x[:] = -(model.h - self.point.s)
            self.point.y[:] = np.linalg.pinv(model.A.T) @ (self.point.z - model.c)

        self.calc_mu()
        if not math.isclose(self.mu, 1.):
            print(f"Initial mu is {self.mu} but should be 1")

        return

    def calc_mu(self):
        self.mu = lin.inp(self.point.s, self.point.z) / self.model.nu
        return self.mu

    def get_gap_feas(self):
        # Get solve data
        self.p_obj   = lin.inp(self.model.c, self.point.x) / self.point.tau
        self.d_obj   = -(lin.inp(self.model.b, self.point.y) + lin.inp(self.model.h, self.point.z)) / self.point.tau
        self.obj_gap = self.p_obj - self.d_obj
        self.x_feas  = np.linalg.norm(self.model.A.T @ self.point.y + self.model.G.T @ self.point.z + self.model.c * self.point.tau, np.inf)
        self.y_feas  = np.linalg.norm(self.model.A @ self.point.x - self.model.b * self.point.tau, np.inf) if self.model.use_A else 0.0
        self.z_feas  = np.linalg.norm(self.model.G @ self.point.x - self.model.h * self.point.tau + self.point.s, np.inf)

    def rescale_model(self):
        model = self.model

        self.c_scale = np.zeros_like(model.c)
        self.b_scale = np.zeros_like(model.b)
        
        # Rescale c
        for i in range(model.n):
            self.c_scale[i] = np.sqrt(max(abs(model.c[i, 0]), np.max(np.abs(model.A[:, i])), np.max(np.abs(model.G[:, i]))))

        # Rescale b
        for i in range(model.p):
            self.b_scale[i] = np.sqrt(max(abs(model.b[i, 0]), np.max(np.abs(model.A[i, :]))))

        model.c[:, :] /= self.c_scale
        model.A[:, :] /= self.c_scale.T
        model.G[:, :] /= self.c_scale.T

        model.A[:, :] /= self.b_scale
        model.b[:, :] /= self.b_scale

        return