import numpy as np
import math
from utils.point import Point

class AggressiveStepper():
    def __init__(self, syssolver, model):
        self.syssolver = syssolver
        self.cent_count = 4
        self.prox = 0.0
        
        self.rhs = Point(model)
        
        return
    
    def step(self, model, point, mu, verbose):
        self.syssolver.update_lhs(model)

        eta = 0.0332
        N = 4

        if (self.prox < eta) or (self.cent_count >= N):
            self.update_rhs_pred(model, point)
            self.syssolver.solve_system(self.rhs, model)

            self.cent_count = 0

            if verbose:
                print("  | %5s" % "pred", end="")
        else:
            self.update_rhs_cent(model, point, mu)
            self.syssolver.solve_system(self.rhs, model)

            self.cent_count += 1

            if verbose:
                print("  | %5s" % "cent", end="")

        res = self.syssolver.apply_system(self.syssolver.sol, model)
        res.vec[:] -= self.rhs.vec
        print(" %10.3e" % (np.linalg.norm(res.vec)), end="")

        point = self.line_search(model, point, verbose)
        
        return point
    
    def line_search(self, model, point, verbose):
        alpha = 1.
        beta = 0.9
        eta = 0.99

        dir = self.syssolver.sol
        next_point = Point(model)

        while True:
            # Step point in direction and step size
            next_point.vec[:] = point.vec + dir.vec * alpha
            mu = np.dot(next_point.x[:, 0], next_point.z[:, 0]) / model.nu
            if mu < 0:
                alpha *= beta
                continue

            rtmu = math.sqrt(mu)
            irtmu = np.reciprocal(rtmu)
            self.prox = 0.

            # Check feasibility
            in_prox = False
            for (k, cone_k) in enumerate(model.cones):
                cone_k.set_point(next_point.x_views[k] * irtmu)

                in_prox = False
                if cone_k.get_feas():
                    grad_k = cone_k.get_grad()
                    psi = next_point.z_views[k] * irtmu + grad_k
                    prod = cone_k.invhess_prod(psi)
                    self.prox = max(self.prox, np.dot(prod[:, 0], psi[:, 0]))

                    in_prox = (self.prox < eta)
                if not in_prox:
                    break
        
            # If feasible, return point
            if in_prox:
                point.vec[:] = next_point.vec[:]

                if verbose:
                    print("  %5.3f" % (alpha))
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