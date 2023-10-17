import numpy as np
import math
import time

from utils import point
from solver.stepper.basic import BasicStepper
from solver.stepper.aggressive import AggressiveStepper
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
        self.stepper = AggressiveStepper(syssolver, model) if (stepper is None) else stepper

        return
    
    def solve(self):
        self.setup_solver()
        self.setup_point()

        while True:
            if self.step_and_check():
                break

        # Get solve data
        self.p_obj   = np.dot(self.model.c[:, 0], self.point.x[:, 0]) / self.point.tau
        self.d_obj   = -(np.dot(self.model.b[:, 0], self.point.y[:, 0]) + np.dot(self.model.h[:, 0], self.point.z[:, 0])) / self.point.tau
        self.obj_gap = self.p_obj - self.d_obj
        self.x_feas  = np.linalg.norm(self.model.A.T @ self.point.y + self.model.G.T @ self.point.z + self.model.c * self.point.tau)
        self.y_feas  = np.linalg.norm(self.model.A @ self.point.x - self.model.b * self.point.tau)
        self.z_feas  = np.linalg.norm(self.model.G @ self.point.x - self.model.h * self.point.tau + self.point.s)

        if self.verbose:
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
        self.p_obj   = np.dot(self.model.c[:, 0], self.point.x[:, 0]) / self.point.tau
        self.d_obj   = -(np.dot(self.model.b[:, 0], self.point.y[:, 0]) + np.dot(self.model.h[:, 0], self.point.z[:, 0])) / self.point.tau
        self.obj_gap = self.p_obj - self.d_obj
        self.x_feas  = np.linalg.norm(self.model.A.T @ self.point.y + self.model.G.T @ self.point.z + self.model.c * self.point.tau)
        self.y_feas  = np.linalg.norm(self.model.A @ self.point.x - self.model.b * self.point.tau)
        self.z_feas  = np.linalg.norm(self.model.G @ self.point.x - self.model.h * self.point.tau + self.point.s)

        # Check optimality
        if abs(self.obj_gap) <= self.gap_tol:
            # Check feasibility
            if self.x_feas <= self.feas_tol and self.y_feas <= self.feas_tol and self.z_feas <= self.feas_tol:
                if self.verbose:
                    print("Solved to desired tolerance")
                return True

        # Check infeasibility
        if self.point.tau <= self.feas_tol:
            if self.d_obj > 0:
                if self.verbose:
                    print("Detected primal infeasibility")
                return True       

            if self.p_obj < 0:
                if self.verbose:
                    print("Detected dual infeasibility")
                return True            

        if self.verbose:
            if self.num_iters % 20 == 0:
                print("===========================================================================================================================================")
                print("%5s" % "iter", " %8s" % "mu", " %8s" % "tau", " %8s" % "kappa",
                    " | %10s" % "p_obj", " %10s" % "d_obj", " %10s" % "gap", 
                    " | %10s" % "x_feas", " %10s" % "y_feas", " %10s" % "z_feas",
                    " | %5s" % "step", "%10s" % "dir_tol", " %5s" % "alpha")
                print("===========================================================================================================================================")                
            
            print("%5d" % (self.num_iters), " %8.1e" % (self.mu), " %8.1e" % (self.point.tau), " %8.1e" % (self.point.kappa),
                  " | %10.3e" % (self.p_obj), " %10.3e" % (self.d_obj), " %10.3e" % (self.obj_gap), 
                  " | %10.3e" % (self.x_feas), " %10.3e" % (self.y_feas), " %10.3e" % (self.z_feas), end="")
            
        # Step
        self.stepper.step(self.model, self.point, self.mu, self.verbose)
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

        self.point.x[:] = np.linalg.pinv(model.G) @ (model.h - self.point.s)
        self.point.y[:] = np.linalg.pinv(model.A.T) @ (-model.G.T @ self.point.z - model.c)

        self.calc_mu()
        if not math.isclose(self.mu, 1.):
            print(f"Initial mu is {self.mu} but should be 1")

        return

    def calc_mu(self):
        self.mu = np.dot(self.point.s[:, 0], self.point.z[:, 0]) / self.model.nu
        return self.mu
