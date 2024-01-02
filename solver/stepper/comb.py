import numpy as np
import math
from utils.point import Point
from utils import linear as lin

alpha_sched = [0.999, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001]

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
        res_norm = np.linalg.norm(self.res.vec)
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

        point, alpha, success = self.line_search_comb(model, point)
        if not success:
            point, alpha = self.line_search_cent(model, point)
        
        if verbose:
            if success:
                print("  | %5s" % "comb", "%10.3e" % (res_norm), "%10.3e" % (self.prox), " %5.3f" % (alpha))
            else:
                print("  | %5s" % "cent", "%10.3e" % (res_norm), "%10.3e" % (self.prox), " %5.3f" % (alpha))

        
        return point
    
    def line_search_comb(self, model, point):
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
                in_prox = False

                cone_k.set_point(next_point.s_views[k] * irtmu)

                # Check cheap proximity conditions first
                inp_s_z_k = lin.inp(next_point.z_views[k], next_point.s_views[k])
                if inp_s_z_k <= 0:
                    # print("inp_s_z_k break")
                    break

                nu_k = cone_k.get_nu()
                rho_k = np.reciprocal(np.sqrt(nu_k)) * np.abs(inp_s_z_k / mu - nu_k)
                if rho_k >= eta:
                    # print("rho_k break")
                    break
                
                # Check if feasible
                if not cone_k.get_feas():
                    break

                grad_k = cone_k.get_grad()
                psi = next_point.z_views[k] * irtmu + grad_k

                # Check cheap inverse Hessian norm oracle
                if cone_k.norm_invhess(psi) >= eta:
                    # print("cheap hess break")
                    break

                prod = cone_k.invhess_prod(psi)
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

    def line_search_cent(self, model, point):
        alpha_iter = 0
        eta = 0.99

        dir_c = self.dir_c
        dir_c_toa = self.dir_c_toa
        next_point = Point(model)

        while True:
            alpha = alpha_sched[alpha_iter]
            # Step point in direction and step size
            next_point.vec[:] = point.vec + alpha * (dir_c.vec + alpha * dir_c_toa.vec)
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
                in_prox = False

                cone_k.set_point(next_point.s_views[k] * irtmu)

                # Check cheap proximity conditions first
                inp_s_z_k = lin.inp(next_point.z_views[k], next_point.s_views[k])
                if inp_s_z_k <= 0:
                    # print("inp_s_z_k break")
                    break

                nu_k = cone_k.get_nu()
                rho_k = np.reciprocal(np.sqrt(nu_k)) * np.abs(inp_s_z_k / mu - nu_k)
                if rho_k >= eta:
                    # print("rho_k break")
                    break
                
                # Check if feasible
                if not cone_k.get_feas():
                    break

                grad_k = cone_k.get_grad()
                psi = next_point.z_views[k] * irtmu + grad_k

                # Check cheap inverse Hessian norm oracle
                if cone_k.norm_invhess(psi) >= eta:
                    # print("cheap hess break")
                    break

                prod = cone_k.invhess_prod(psi)
                self.prox = max(self.prox, lin.inp(prod, psi))
                in_prox = (self.prox < eta)
                if not in_prox:
                    break
        
            # If feasible, return point
            if in_prox:
                point.vec[:] = next_point.vec[:]
                return point, alpha
        
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