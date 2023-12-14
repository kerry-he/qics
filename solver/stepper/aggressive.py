import numpy as np
import math
from utils.point import Point
from utils import linear as lin

class AggressiveStepper():
    def __init__(self, syssolver, model):
        self.syssolver = syssolver
        self.cent_count = 4
        self.prox = 0.0
        
        self.rhs = Point(model)
        self.dir = Point(model)
        self.res = Point(model)
        self.temp = Point(model)
        
        return
    
    def step(self, model, point, mu, verbose):
        self.syssolver.update_lhs(model)

        eta = 0.0332
        N = 4

        if (self.prox < eta) or (self.cent_count >= N):
            self.update_rhs_pred(model, point)
            self.cent_count = 0

            if verbose:
                print("  | %5s" % "pred", end="")
        else:
            self.update_rhs_cent(model, point, mu)
            self.cent_count += 1

            if verbose:
                print("  | %5s" % "cent", end="")

        res_norm = self.syssolver.solve_system_ir(self.dir, self.res, self.rhs, model, mu, point.tau)


        print(" %10.3e" % (res_norm), end="")

        point = self.line_search(model, point, verbose)
        
        return point
    
    def line_search(self, model, point, verbose):
        alpha = 1.
        beta = 0.9
        eta = 0.99

        dir = self.dir
        next_point = Point(model)

        while True:
            # Step point in direction and step size
            next_point.vec[:] = point.vec + dir.vec * alpha
            mu = lin.inp(next_point.s, next_point.z) / model.nu
            if mu < 0:
                alpha *= beta
                continue

            rtmu = math.sqrt(mu)
            irtmu = np.reciprocal(rtmu)
            self.prox = 0.

            # Check feasibility
            in_prox = False
            for (k, cone_k) in enumerate(model.cones):
                cone_k.set_point(next_point.s_views[k] * irtmu)

                in_prox = False
                if cone_k.get_feas():
                    grad_k = cone_k.get_grad()
                    psi = next_point.z_views[k] * irtmu + grad_k
                    prod = cone_k.invhess_prod(psi)
                    self.prox = max(self.prox, lin.inp(prod, psi))

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
        self.rhs.z.fill(0.)

        rtmu = math.sqrt(mu)
        for (k, cone_k) in enumerate(model.cones):
            z_k = point.z_views[k]
            grad_k = cone_k.get_grad()
            self.rhs.s_views[k][:] = -z_k - rtmu * grad_k

        self.rhs.tau[0]   = 0.
        self.rhs.kappa[0] = -point.kappa + mu / point.tau

        return self.rhs

    def update_rhs_pred(self, model, point):
        self.rhs.x[:] = -model.A.T @ point.y - model.G.T @ point.z - model.c * point.tau
        self.rhs.y[:] = model.A @ point.x - model.b * point.tau
        self.rhs.z[:] = model.G @ point.x - model.h * point.tau + point.s

        for (rhs_s_k, z_k) in zip(self.rhs.s_views, point.z_views):
            rhs_s_k[:] = -z_k

        self.rhs.tau[0]   = lin.inp(model.c, point.x) + lin.inp(model.b, point.y) + lin.inp(model.h, point.z) + point.kappa
        self.rhs.kappa[0] = -point.kappa

        return self.rhs