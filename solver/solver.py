import numpy as np
import math
import time

from utils import point, linear as lin
from solver.stepper.basic import BasicStepper
from solver.stepper.aggressive import AggressiveStepper
from solver.stepper.comb import CombinedStepper
from solver.syssolver import SysSolver

from utils import symmetric as sym

class Solver():
    def __init__(
        self, 
        model, 
        max_iter = 1000, 
        max_time = np.inf,
        tol_gap = 1e-8,
        tol_feas = 1e-8,
        tol_infeas = 1e-12,
        tol_ip = 1e-13,
        verbose  = True,
        subsolver = None,
        stepper = None,
        ir = True,
        sym = False
    ):
        self.max_iter = max_iter
        self.max_time = max_time        
        self.verbose = verbose

        self.tol_gap = tol_gap
        self.tol_feas = tol_feas
        self.tol_infeas = tol_infeas
        self.tol_ip = tol_ip

        self.model = model
        syssolver = SysSolver(model, subsolver=subsolver, ir=ir, sym=sym)
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
                    " | %10.3e" % (self.p_obj), " %10.3e" % (self.d_obj), " %10.3e" % (self.gap), 
                    " | %10.3e" % (self.x_feas), " %10.3e" % (self.y_feas), " %10.3e" % (self.z_feas), end="")

            print()
            print("Opt value:  %.10f" % (self.p_obj))
            print("Tolerance:  %.10e" % (self.gap))
            print("Solve time: %.10f" % (time.time() - self.solve_time), " seconds")
            print()

        return
    
    def step_and_check(self):
        # Check termination
        if self.num_iters >= self.max_iter:
            self.status = "max_iter"
            if self.verbose:
                print("Maximum iteration limit reached")
            return True
        
        if time.time() - self.solve_time >= self.max_time:
            self.status = "max_time"
            if self.verbose:
                print("Maximum time limit reached")
            return True
        
        # Update barrier parameter mu
        self.calc_mu()

        # Get solve data
        self.get_gap_feas()      

        # Check optimality
        if self.gap <= self.tol_gap:
            # Check feasibility
            if self.x_feas <= self.tol_feas and self.y_feas <= self.tol_feas and self.z_feas <= self.tol_feas:
                self.status = "solved"
                if self.verbose:
                    print("Solved to desired tolerance")
                return True

        # Check infeasibility
        if self.x_infeas <= self.tol_infeas:
            self.status = "primal_infeas"
            if self.verbose:
                print("Detected primal infeasibility")
            return True       

        if self.y_infeas <= self.tol_infeas and self.z_infeas <= self.tol_infeas:
            self.status = "dual_infeas"
            if self.verbose:
                print("Detected dual infeasibility")
            return True            
            
        # Check ill-posedness
        if self.mu <= self.tol_ip and self.point.tau <= self.tol_ip * min([1., self.point.kappa]):
            self.status = "ill_posed"
            if self.verbose:
                print("Detected ill-posed problem")     
            return True       

        if self.verbose:
            if self.num_iters % 20 == 0:
                print("===================================================================================================================================================================")
                print("%5s" % "iter", " %8s" % "mu", " %8s" % "tau", " %8s" % "kappa",
                    " | %10s" % "p_obj", " %10s" % "d_obj", " %10s" % "gap", 
                    " | %10s" % "x_feas", " %10s" % "y_feas", " %10s" % "z_feas",
                    " | %6s" % "step", "%10s" % "dir_tol", "%10s" % "prox", " %5s" % "alpha")
                print("===================================================================================================================================================================")                
            
            print("%5d" % (self.num_iters), " %8.1e" % (self.mu), " %8.1e" % (self.point.tau), " %8.1e" % (self.point.kappa),
                  " | %10.3e" % (self.p_obj), " %10.3e" % (self.d_obj), " %10.3e" % (self.gap),
                  " | %10.3e" % (self.x_feas), " %10.3e" % (self.y_feas), " %10.3e" % (self.z_feas), end="")
            
        # Step
        self.point, success = self.stepper.step(self.model, self.point, self.mu, self.verbose)

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

        self.point.tau   = 1.
        self.point.kappa = 1.

        for (k, cone_k) in enumerate(model.cones):
            self.point.S[k] = cone_k.set_init_point()
            assert cone_k.get_feas()
            self.point.Z[k] = -1 * cone_k.get_grad()

        if model.use_G:
            self.point.X[:] = np.linalg.pinv(np.vstack((model.A, model.G))) @ np.vstack((model.b, model.h - self.point.S.to_vec()))
            self.point.y[:] = np.linalg.pinv(model.A.T) @ (-model.G.T @ self.point.Z.to_vec() - model.c)
        else:
            self.point.X[:] = -(model.h - self.point.S.to_vec())
            # self.point.y[:] = np.linalg.pinv(model.A.todense().T) @ (self.point.Z.to_vec() - model.c)

        self.calc_mu()
        if not math.isclose(self.mu, 1.):
            print(f"Initial mu is {self.mu} but should be 1")

        return

    def calc_mu(self):
        self.mu = (self.point.S.inp(self.point.Z) + self.point.tau*self.point.kappa) / self.model.nu
        return self.mu

    def get_gap_feas(self):
        c = self.model.c
        b = self.model.b
        h = self.model.h
        A = self.model.A
        G = self.model.G

        x   = self.point.X
        y   = self.point.y
        z   = self.point.Z.to_vec()
        s   = self.point.S.to_vec()
        tau = self.point.tau

        c_max = np.linalg.norm(c, np.inf)
        b_max = np.linalg.norm(b, np.inf)
        h_max = np.linalg.norm(h, np.inf)

        # Get primal and dual objectives and optimality gap
        p_obj_tau =  lin.inp(c, x)
        d_obj_tau = -lin.inp(b, y) - lin.inp(h, z)

        self.p_obj = p_obj_tau / tau + self.model.offset
        self.d_obj = d_obj_tau / tau + self.model.offset
        self.gap   = min([self.point.Z.inp(self.point.S) / tau, abs(p_obj_tau - d_obj_tau)]) / max([tau, min([abs(p_obj_tau), abs(d_obj_tau)])])

        # Get primal and dual feasibilities
        x_res = A.T @ y + G.T @ z
        y_res =   A @ x          
        z_res =   G @ x       + s

        self.x_feas = np.linalg.norm(x_res + c * tau, np.inf) / (1. + c_max) / tau
        self.y_feas = np.linalg.norm(y_res - b * tau, np.inf) / (1. + b_max) / tau if self.model.use_A else 0.0
        self.z_feas = np.linalg.norm(z_res - h * tau, np.inf) / (1. + h_max) / tau

        # Get primal and dual infeasibilities
        self.x_infeas =  np.linalg.norm(x_res, np.inf) / d_obj_tau if (d_obj_tau > 0) else np.inf
        self.y_infeas = -np.linalg.norm(y_res, np.inf) / p_obj_tau if (p_obj_tau < 0) else np.inf
        self.z_infeas = -np.linalg.norm(z_res, np.inf) / p_obj_tau if (p_obj_tau < 0) else np.inf


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