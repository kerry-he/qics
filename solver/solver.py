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
        self.p_obj   =  np.dot(self.point.x[:, 0], self.model.c[:, 0])
        self.d_obj   = -np.dot(self.point.y[:, 0], self.model.b[:, 0])
        self.obj_gap =  self.p_obj - self.d_obj
        self.p_feas  =  np.linalg.norm(self.model.A @ self.point.x - self.model.b)
        self.d_feas  =  np.linalg.norm(self.model.A.T @ self.point.y + self.model.c - self.point.z)

        if self.verbose:
            print("%5d" % (self.num_iters), " %8.1e" % (self.mu),
                  " | %10.3e" % (self.p_obj), " %10.3e" % (self.d_obj), " %10.3e" % (self.obj_gap), 
                  " | %10.3e" % (self.p_feas), " %10.3e" % (self.d_feas))


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
        self.p_obj   =  np.dot(self.point.x[:, 0], self.model.c[:, 0])
        self.d_obj   = -np.dot(self.point.y[:, 0], self.model.b[:, 0])
        self.obj_gap =  self.p_obj - self.d_obj
        self.p_feas  =  np.linalg.norm(self.model.A @ self.point.x - self.model.b)
        self.d_feas  =  np.linalg.norm(self.model.A.T @ self.point.y + self.model.c - self.point.z)

        if abs(self.obj_gap) <= self.gap_tol and self.p_feas <= self.feas_tol and self.d_feas <= self.feas_tol:
            if self.verbose:
                print("Solved to desired tolerance")
            return True

        if self.verbose:
            if self.num_iters % 20 == 0:
                print("==========================================================================================================")
                print("%5s" % "iter", " %8s" % "mu",
                    " | %10s" % "p_obj", " %10s" % "d_obj", " %10s" % "gap", 
                    " | %10s" % "p_feas", " %10s" % "d_feas",
                    " | %5s" % "step", "%10s" % "dir_tol", " %5s" % "alpha")
                print("==========================================================================================================")                
            
            print("%5d" % (self.num_iters), " %8.1e" % (self.mu),
                  " | %10.3e" % (self.p_obj), " %10.3e" % (self.d_obj), " %10.3e" % (self.obj_gap), 
                  " | %10.3e" % (self.p_feas), " %10.3e" % (self.d_feas), end="")
            
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

        for (k, cone_k) in enumerate(model.cones):
            self.point.x[model.cone_idxs[k]] = cone_k.set_init_point()
            assert cone_k.get_feas()
            self.point.z_views[k][:] = -cone_k.get_grad()

        self.point.y[:] = np.linalg.pinv(model.A.T) @ (self.point.z - model.c)

        self.calc_mu()
        if not math.isclose(self.mu, 1.):
            print(f"Initial mu is {self.mu} but should be 1")

        return

    def calc_mu(self):
        self.mu = np.dot(self.point.x[:, 0], self.point.z[:, 0]) / self.model.nu
        return self.mu
