import numpy as np
import math
from utils.point import Point
from utils import linear as lin
from utils import symmetric as sym

alpha_sched = [0.9999, 0.999, 0.99, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0.01, 0.001]

class CombinedStepper():
    def __init__(self, syssolver, model):
        self.syssolver = syssolver
        self.prox = 0.0
        
        self.rhs        = Point(model)
        self.dir_c      = Point(model)
        self.dir_c_toa  = Point(model)
        self.dir_p      = Point(model)
        self.dir_p_toa  = Point(model)
        self.temp       = Point(model)
        self.next_point = Point(model)
        
        return
    
    def step(self, model, point, mu, verbose):
        self.syssolver.update_lhs(model)

        # Get prediction direction
        self.update_rhs_pred(model, point)
        res_norm = self.syssolver.solve_system_ir(self.dir_p, self.rhs, model, mu, point.tau, point.kappa)

        # Get TOA prediction direction
        self.update_rhs_pred_toa(model, point, mu, self.dir_p)
        temp_res_norm = self.syssolver.solve_system_ir(self.dir_p_toa, self.rhs, model, mu, point.tau, point.kappa)
        res_norm = max(temp_res_norm, res_norm)

        # Get centering direction
        self.update_rhs_cent(model, point, mu)
        temp_res_norm = self.syssolver.solve_system_ir(self.dir_c, self.rhs, model, mu, point.tau, point.kappa)
        res_norm = max(temp_res_norm, res_norm)

        # Get TOA centering direction
        self.update_rhs_cent_toa(model, point, mu, self.dir_c)
        temp_res_norm = self.syssolver.solve_system_ir(self.dir_c_toa, self.rhs, model, mu, point.tau, point.kappa)
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
                return point, False
        
        return point, True
    
    def line_search(self, model, point, mode="co_toa"):
        alpha_iter = -1
        eta = 0.99

        dir_p = self.dir_p
        dir_c = self.dir_c
        dir_p_toa = self.dir_p_toa
        dir_c_toa = self.dir_c_toa
        next_point = Point(model)

        while True:
            alpha_iter += 1
            if alpha_iter >= len(alpha_sched):
                return point, alpha, False

            alpha = alpha_sched[alpha_iter]
            # Step point in direction and step size
            next_point.copy(point)
            if mode == "co_toa":
                # step = alpha * (dir_p + dir_p_toa * alpha) + (dir_c + dir_c_toa * (1.0 - alpha)) * (1.0 - alpha)
                next_point.axpy(alpha             , dir_p)
                next_point.axpy(alpha ** 2        , dir_p_toa)
                next_point.axpy((1.0 - alpha)     , dir_c)
                next_point.axpy((1.0 - alpha) ** 2, dir_c_toa)
            elif mode == "comb":
                # step = dir_p * alpha + dir_c * (1.0 - alpha)
                next_point.axpy(alpha        , dir_p)
                next_point.axpy((1.0 - alpha), dir_c)
            elif mode == "ce_toa":
                # step = (dir_p + dir_p_toa * alpha) * alpha
                next_point.axpy(alpha     ,  dir_c)
                next_point.axpy(alpha ** 2, dir_c_toa)
            elif mode == "cent":
                # step = dir_c * alpha
                next_point.axpy(alpha, dir_c)
            
            # Check that tau, kappa, mu are well defined
            # Make sure tau, kappa are positive
            taukap = next_point.tau * next_point.kappa
            if next_point.tau < 0 or next_point.kappa < 0 or taukap < 0:      
                continue
            # Compute barrier parameter mu, ensure it is positive
            sz = [s_k.inp(z_k) for (s_k, z_k) in zip(next_point.S, next_point.Z)]  
            mu = (sum(sz) + taukap) / model.nu
            if any(np.array(sz) < 0) or mu < 0:
                continue

            # Check cheap proximity conditions first
            if abs(taukap / mu - 1) > eta:
                continue
            nu  = [cone_k.get_nu() for cone_k in model.cones]
            rho = [np.abs(sz_k / mu - nu_k) / np.sqrt(nu_k) for (sz_k, nu_k) in zip(sz, nu)]
            if any(np.array(rho) > eta):
                continue

            rtmu = math.sqrt(mu)
            irtmu = np.reciprocal(rtmu)
            self.prox = 0.

            # Check feasibility
            in_prox = False
            for (k, cone_k) in enumerate(model.cones):
                in_prox = False

                cone_k.set_point(next_point.S[k], next_point.Z[k], irtmu)
                
                # Check if feasible
                if not cone_k.get_feas():
                    break

                prox_k = cone_k.prox()
                self.prox = max(self.prox, prox_k)
                in_prox = (self.prox < eta)
                if not in_prox:
                    break
        
            # If feasible, return point
            if in_prox:
                point = next_point
                return point, alpha, True

    def update_rhs_cent(self, model, point, mu):
        self.rhs.X *= 0
        self.rhs.y *= 0
        self.rhs.Z *= 0

        rtmu = math.sqrt(mu)
        for (k, cone_k) in enumerate(model.cones):
            self.rhs.S[k] = -1 * point.Z[k] - rtmu * cone_k.get_grad()

        self.rhs.tau   = 0.
        self.rhs.kappa = -point.kappa + mu / point.tau

        return self.rhs

    def update_rhs_cent_toa(self, model, point, mu, dir_c):
        self.rhs.X *= 0
        self.rhs.y *= 0
        self.rhs.Z *= 0

        rtmu = math.sqrt(mu)
        for (k, cone_k) in enumerate(model.cones):
            self.rhs.S[k] = -0.5 * cone_k.third_dir_deriv(dir_c.S[k]) / rtmu

        self.rhs.tau   = 0.
        self.rhs.kappa = (dir_c.tau**2) / (point.tau**3) * mu

        return self.rhs

    def update_rhs_pred(self, model, point):
        self.rhs.X = -model.A.T @ point.y - model.G.T @ point.Z.to_vec() - model.c * point.tau
        self.rhs.y = model.A @ point.X - model.b * point.tau
        self.rhs.Z.from_vec(model.G @ point.X - model.h * point.tau)
        self.rhs.Z += point.S

        self.rhs.S = -1 * point.Z

        self.rhs.tau   = lin.inp(model.c, point.X) + lin.inp(model.b, point.y) + lin.inp(model.h, point.Z.to_vec()) + point.kappa
        self.rhs.kappa = -point.kappa

        return self.rhs

    def update_rhs_pred_toa(self, model, point, mu, dir_p):
        self.rhs.X *= 0
        self.rhs.y *= 0
        self.rhs.Z *= 0 

        rtmu = math.sqrt(mu)
        for (k, cone_k) in enumerate(model.cones):
            if self.syssolver.sym:
                self.rhs.S[k] = 0.5 * cone_k.third_dir_deriv(dir_p.S[k], dir_p.Z[k]) / rtmu
            else:
                self.rhs.S[k] = cone_k.hess_prod(dir_p.S[k]) - 0.5 * cone_k.third_dir_deriv(dir_p.S[k]) / rtmu

        self.rhs.tau   = 0.
        if self.syssolver.sym:
            self.rhs.kappa = -dir_p.tau * dir_p.kappa / point.tau
        else:
            self.rhs.kappa = (dir_p.tau**2) / (point.tau**3) * mu

        return self.rhs