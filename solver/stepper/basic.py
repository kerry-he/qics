import numpy as np
import math
from utils.point import Point
from utils import linear as lin

class BasicStepper():
    def __init__(self, syssolver, model):
        self.syssolver = syssolver
        self.cent_count = 2

        self.rhs = Point(model)
        
        return
    
    def step(self, model, point, mu):
        self.syssolver.update_lhs(model)

        if self.cent_count >= 2:
            self.update_rhs_pred(model, point)
            self.syssolver.solve_system(self.rhs, model)

            point = self.line_search(model, point)
            self.cent_count = 0
        else:
            self.update_rhs_cent(model, point, mu)
            self.syssolver.solve_system(self.rhs, model)

            point.vec[:] += self.syssolver.sol.vec / 84.
            self.cent_count += 1
        
        return point
    
    def line_search(self, model, point):
        alpha = 1.
        beta = 0.5
        eta = 1 / 6.

        dir = self.syssolver.sol
        next_point = Point(model)

        while True:
            next_point.vec[:] = point.vec + dir.vec * alpha
            mu = lin.inp(next_point.x, next_point.z) / model.nu
            if mu < 0:
                alpha *= beta
                continue

            rtmu = math.sqrt(mu)
            irtmu = np.reciprocal(rtmu)
            total_prox = 0.

            # Check feasibility
            in_prox = False
            for (k, cone_k) in enumerate(model.cones):
                cone_k.set_point(next_point.x_views[k] * irtmu)

                in_prox = False
                if cone_k.get_feas():
                    grad_k = cone_k.get_grad()
                    psi = next_point.z_views[k] + rtmu * grad_k
                    prod = cone_k.invhess_prod(psi)
                    total_prox += lin.inp(prod, psi)

                    in_prox = (total_prox < (eta*mu)**2)
                if not in_prox:
                    break
        
            # If feasible, return point
            if in_prox:
                point.vec[:] = next_point.vec[:]
                return point
        
            # Otherwise backtrack
            alpha *= beta

    def update_rhs_cent(self, model, point, mu):
        self.rhs.x.fill(0.)
        self.rhs.y.fill(0.)

        rtmu = math.sqrt(mu)
        for (k, cone_k) in enumerate(model.cones):
            z_k = point.z_views[k]
            grad_k = cone_k.get_grad()
            self.rhs.z_views[k][:] = -z_k - rtmu * grad_k

        return self.rhs

    def update_rhs_pred(self, model, point):
        self.rhs.x[:] = model.c - model.A.T @ point.y - point.z
        self.rhs.y[:] = model.b - model.A @ point.x

        for (rhs_z_k, z_k) in zip(self.rhs.z_views, point.z_views):
            rhs_z_k[:] = -z_k

        return self.rhs
