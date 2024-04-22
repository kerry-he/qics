import numpy as np
import math
from utils.point import Point
from utils import linear as lin
from utils import symmetric as sym

alpha_sched = [0.99, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0.01, 0.001]

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

        # Get prediction direction
        self.update_rhs_pred(model, point)
        res_norm = self.syssolver.solve_system_ir(self.dir_p, self.res, self.rhs, model, mu, point.tau)

        # Get TOA prediction direction
        self.update_rhs_pred_toa(model, point, mu, self.dir_p)
        temp_res_norm = self.syssolver.solve_system_ir(self.dir_p_toa, self.res, self.rhs, model, mu, point.tau)
        res_norm = max(temp_res_norm, res_norm)

        # Get centering direction
        self.update_rhs_cent(model, point, mu)
        temp_res_norm = self.syssolver.solve_system_ir(self.dir_c, self.res, self.rhs, model, mu, point.tau)
        res_norm = max(temp_res_norm, res_norm)

        # Get TOA centering direction
        self.update_rhs_cent_toa(model, point, mu, self.dir_c)
        temp_res_norm = self.syssolver.solve_system_ir(self.dir_c_toa, self.res, self.rhs, model, mu, point.tau)
        res_norm = max(temp_res_norm, res_norm)

        step_mode = "co_toa"
        point, alpha, success = self.line_search(model, point, step_mode)
        if not success:
            step_mode = "comb"
            point, alpha, success = self.line_search(model, point, step_mode)
            if not success:
                step_mode = "ce_toa"
                point, alpha, success = self.line_search(model, point, step_mode)
                if not success:
                    step_mode = "cent"
                    point, alpha, success = self.line_search(model, point, step_mode)
        
        if verbose:
            if success:
                print("  | %6s" % step_mode, "%10.3e" % (res_norm), "%10.3e" % (self.prox), " %5.3f" % (alpha))
            else:
                return False
        
        return True
    
    def line_search(self, model, point, mode="co_toa"):
        alpha_iter = 0
        eta = 0.99

        dir_p = self.dir_p
        dir_c = self.dir_c
        dir_p_toa = self.dir_p_toa
        dir_c_toa = self.dir_c_toa
        next_point = Point(model)

        while True:
            if alpha_iter >= len(alpha_sched):
                return point, alpha, False

            alpha = alpha_sched[alpha_iter]
            # Step point in direction and step size
            if mode == "co_toa":
                step = alpha * (dir_p + dir_p_toa * alpha) + (dir_c + dir_c_toa * (1.0 - alpha)) * (1.0 - alpha)
            elif mode == "comb":
                step = dir_p * alpha + dir_c * (1.0 - alpha)
            elif mode == "ce_toa":
                step = (dir_p + dir_p_toa * alpha) * alpha
            elif mode == "cent":
                step = dir_c * alpha

            next_point = point + step
            mu = (lin.inp(next_point.S, next_point.Z) + next_point.tau*next_point.kappa) / model.nu
            if mu < 0 or next_point.tau < 0 or next_point.kappa < 0:
                alpha_iter += 1
                continue

            rtmu = math.sqrt(mu)
            irtmu = np.reciprocal(rtmu)
            self.prox = 0.

            # Check feasibility
            in_prox = False
            for (k, cone_k) in enumerate(model.cones):
                in_prox = False

                cone_k.set_point_alt(next_point.S * irtmu)
                
                # Check if feasible
                if not cone_k.get_feas():
                    break

                grad_k = cone_k.get_grad()[1]
                psi = next_point.Z * irtmu + grad_k

                prod = cone_k.invhess_prod_alt(psi)
                self.prox = max(self.prox, lin.inp(prod, psi))
                in_prox = (self.prox < eta)
                if not in_prox:
                    break
        
            # If feasible, return point
            if in_prox:
                point.vec[:] = next_point.vec[:]
                return point, alpha, True
        
            # Otherwise backtrack
            alpha_iter += 1

    def update_rhs_cent(self, model, point, mu):
        self.rhs.x.fill(0.)
        self.rhs.y.fill(0.)
        self.rhs.z.fill(0.)

        self.rhs.X.fill(0.)
        self.rhs.Z.fill(0.)

        rtmu = math.sqrt(mu)
        for (k, cone_k) in enumerate(model.cones):
            z_k = point.z_views[k]
            grad_k, Grad_k = cone_k.get_grad()
            self.rhs.s_views[k][:] = -z_k - rtmu * grad_k
            self.rhs.S = -point.Z - rtmu * Grad_k

        self.rhs.tau[0]   = 0.
        self.rhs.kappa[0] = -point.kappa + mu / point.tau

        return self.rhs

    def update_rhs_cent_toa(self, model, point, mu, dir_c):
        self.rhs.x.fill(0.)
        self.rhs.y.fill(0.)
        self.rhs.z.fill(0.)

        self.rhs.X.fill(0.)
        self.rhs.Z.fill(0.)

        rtmu = math.sqrt(mu)
        for (k, cone_k) in enumerate(model.cones):
            dir_c_s_k = dir_c.s_views[k]
            tdd, TDD = cone_k.third_dir_deriv(dir_c_s_k)
            self.rhs.s_views[k][:] = -0.5 * tdd / rtmu
            self.rhs.S = -0.5 * TDD / rtmu

        self.rhs.tau[0]   = 0.
        self.rhs.kappa[0] = (dir_c.tau**2) / (point.tau**3) * mu

        return self.rhs

    def update_rhs_pred(self, model, point):
        self.rhs.x[:] = -model.A.T @ point.y - model.G.T @ point.z - model.c * point.tau
        self.rhs.y[:] = model.A @ point.x - model.b * point.tau
        self.rhs.z[:] = model.G @ point.x - model.h * point.tau + point.s

        self.rhs.X = point.Z - model.c_mtx * point.tau
        for (i, Ai) in enumerate(model.A_mtx):
            self.rhs.X -= Ai * point.y[i]
        
        self.rhs.Z = point.S - point.X

        for (rhs_s_k, z_k) in zip(self.rhs.s_views, point.z_views):
            rhs_s_k[:] = -z_k
        self.rhs.S = -point.Z

        self.rhs.tau[0]   = lin.inp(model.c, point.x) + lin.inp(model.b, point.y) + lin.inp(model.h, point.z) + point.kappa
        self.rhs.kappa[0] = -point.kappa

        return self.rhs

    def update_rhs_pred_toa(self, model, point, mu, dir_p):
        self.rhs.x.fill(0.)
        self.rhs.y.fill(0.)
        self.rhs.z.fill(0.)

        self.rhs.X.fill(0.)
        self.rhs.Z.fill(0.)        

        rtmu = math.sqrt(mu)
        for (k, cone_k) in enumerate(model.cones):
            dir_p_s_k = dir_p.s_views[k]
            tdd, TDD = cone_k.third_dir_deriv(dir_p_s_k)
            self.rhs.s_views[k][:] = cone_k.hess_prod(dir_p_s_k) - 0.5 * tdd / rtmu
        
        self.rhs.S = cone_k.hess_prod_alt(dir_p.S) - 0.5 * TDD / rtmu

        self.rhs.tau[0]   = 0.
        self.rhs.kappa[0] = (dir_p.tau**2) / (point.tau**3) * mu

        return self.rhs