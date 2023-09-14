import numpy as np
import math
import time

from utils import point
from solver.stepper import Stepper
from solver.syssolver import SysSolver

class Solver():
    def __init__(
        self, 
        model, 
        max_iter = 100, 
        max_time = np.inf,
        verbose  = True,
        syssolver = None,
        stepper = None
    ):
        self.max_iter = max_iter
        self.max_time = max_time
        self.verbose = verbose

        self.model = model
        syssolver = SysSolver(model) if (syssolver is None) else syssolver
        self.stepper = Stepper(syssolver, model) if (stepper is None) else stepper

        return
    
    def solve(self):
        self.setup_solver()
        self.setup_point()

        while True:
            if self.step_and_check():
                break

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

        # Step
        self.stepper.step(self.model, self.point, self.mu)
        self.num_iters += 1

        if self.verbose:
            print("Iter", self.num_iters)
        
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

        self.point.y[:] = np.linalg.pinv(model.A.T) @ (model.c - self.point.z)

        self.calc_mu()
        if not math.isclose(self.mu, 1.):
            print(f"Initial mu is {self.mu} but should be 1")

        return

    def calc_mu(self):
        self.mu = np.dot(self.point.x[:, 0], self.point.z[:, 0]) / self.model.nu
        return self.mu
