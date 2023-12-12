import numpy as np
import math
from utils.point import Point
from utils import linear as lin

alpha_sched = [0.999, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0.001, 0.000001]

class CombinedStepper():
    def __init__(self, syssolver, model):
        self.syssolver = syssolver
        self.prox = 0.0
        
        self.rhs        = Point(model)
        self.dir_c      = Point(model)
        self.dir_c_toa  = Point(model)
        self.dir_p      = Point(model)
        self.dir_p_toa  = Point(model)
        self.res        = Point(model)
        self.temp       = Point(model)
        
        return
    
    def step(self, model, point, mu, verbose):
        self.syssolver.update_lhs(model)

        if verbose:
            print("  | %5s" % "comb", end="")

        self.update_rhs_pred(model, point)
        self.dir_p.vec[:] = self.syssolver.solve_system(self.rhs, model, mu / point.tau / point.tau).vec
        self.res.vec[:] = self.rhs.vec - self.syssolver.apply_system(self.dir_p, model, mu / point.tau / point.tau).vec
        res_norm = np.linalg.norm(self.res.vec)

        self.update_rhs_pred_toa(model, point, mu, self.dir_p)
        self.dir_p_toa.vec[:] = self.syssolver.solve_system(self.rhs, model, mu / point.tau / point.tau).vec
        self.res.vec[:] = self.rhs.vec - self.syssolver.apply_system(self.dir_p_toa, model, mu / point.tau / point.tau).vec
        res_norm = max(np.linalg.norm(self.res.vec), res_norm)

        self.update_rhs_cent(model, point, mu)
        self.dir_c.vec[:] = self.syssolver.solve_system(self.rhs, model, mu / point.tau / point.tau).vec
        self.res.vec[:] = self.rhs.vec - self.syssolver.apply_system(self.dir_c, model, mu / point.tau / point.tau).vec
        res_norm = max(np.linalg.norm(self.res.vec), res_norm)

        self.update_rhs_cent_toa(model, point, mu, self.dir_c)
        self.dir_c_toa.vec[:] = self.syssolver.solve_system(self.rhs, model, mu / point.tau / point.tau).vec
        self.res.vec[:] = self.rhs.vec - self.syssolver.apply_system(self.dir_c_toa, model, mu / point.tau / point.tau).vec
        res_norm = max(np.linalg.norm(self.res.vec), res_norm)

        print(" %10.3e" % (res_norm), end="")

        point = self.line_search(model, point, verbose)
        
        return point
    
    def line_search(self, model, point, verbose):
        alpha_iter = 1
        beta = 0.9
        eta = 0.99

        dir_p = self.dir_p
        dir_c = self.dir_c
        dir_p_toa = self.dir_p_toa
        dir_c_toa = self.dir_c_toa
        next_point = Point(model)

        while True:
            alpha = alpha_sched[alpha_iter]
            # Step point in direction and step size
            next_point.vec[:] = point.vec + alpha * (dir_p.vec + alpha * dir_p_toa.vec) + (1.0 - alpha) * (dir_c.vec + (1.0 - alpha) * dir_c_toa.vec)
            mu = lin.inp(next_point.s, next_point.z) / model.nu
            if mu < 0:
                alpha_iter += 1
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
            alpha_iter += 1

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

    def update_rhs_cent_toa(self, model, point, mu, dir_c):
        self.rhs.x.fill(0.)
        self.rhs.y.fill(0.)
        self.rhs.z.fill(0.)

        rtmu = math.sqrt(mu)
        for (k, cone_k) in enumerate(model.cones):
            dir_c_s_k = dir_c.s_views[k]
            self.rhs.s_views[k][:] = -0.5 * cone_k.third_dir_deriv(dir_c_s_k) / rtmu

        self.rhs.tau[0]   = 0.
        self.rhs.kappa[0] = (dir_c.tau**2) / (point.tau**3) * mu

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

    def update_rhs_pred_toa(self, model, point, mu, dir_p):
        self.rhs.x.fill(0.)
        self.rhs.y.fill(0.)
        self.rhs.z.fill(0.)

        rtmu = math.sqrt(mu)
        for (k, cone_k) in enumerate(model.cones):
            dir_p_s_k = dir_p.s_views[k]
            self.rhs.s_views[k][:] = cone_k.hess_prod(dir_p_s_k) - 0.5 * cone_k.third_dir_deriv(dir_p_s_k) / rtmu

        self.rhs.tau[0]   = 0.
        self.rhs.kappa[0] = (dir_p.tau**2) / (point.tau**3) * mu

        return self.rhs