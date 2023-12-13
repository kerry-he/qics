import numpy as np
import scipy as sp
from utils import point, linear as lin

# Solves the following square Newton system
#            - A'*y     - c*tau + z         = rx
#        A*x            - b*tau             = ry
#      -c'*x - b'*y                 - kappa = rtau
#     mu*H*x                    + z         = rz 
#                    mu/t^2*tau     + kappa = rkappa
# for (x, y, z, tau, kappa) given right-hand residuals (rx, ry, rz, rtau, rkappa)
# by using elimination.

class SysSolver():
    def __init__(self, model):
        self.sol = point.Point(model)
        return
    
    def update_lhs(self, model):
        # Precompute necessary objects on LHS of Newton system

        if model.use_G:
            HG = blk_hess_prod(model.G, model)
            GHG = model.G.T @ HG
            self.GHG_fact = lin.fact(GHG)

            GHGA = np.zeros((model.n, model.p))
            for i in range(model.p):
                GHGA[:, i] = lin.fact_solve(self.GHG_fact, model.A.T[:, i])
            AGHGA = model.A @ GHGA
            self.AGHGA_fact = lin.fact(AGHGA)

        else:
            HA = blk_invhess_prod(model.A.T, model)
            AHA = model.A @ HA
            self.AHA_fact = lin.fact(AHA)

        return

    def solve_system_ir(self, dir, res, rhs, model, mu, tau):
        temp = point.Point(model)

        # Solve system
        dir.vec[:] = self.solve_system(rhs, model, mu / tau / tau).vec

        # Check residuals of solve
        res.vec[:] = rhs.vec - self.apply_system(dir, model, mu / tau / tau).vec
        res_norm = np.linalg.norm(res.vec)

        # Iterative refinement
        iter_refine_count = 0
        while res_norm > 1e-8:
            self.solve_system(res, model, mu / tau / tau)
            temp.vec[:] = dir.vec + self.sol.vec
            res.vec[:] = rhs.vec - self.apply_system(temp, model, mu / tau / tau).vec
            res_norm_new = np.linalg.norm(res.vec)

            # Check if iterative refinement made things worse
            if res_norm_new > res_norm:
                break

            dir.vec[:] = temp.vec
            res_norm = res_norm_new

            iter_refine_count += 1

            if iter_refine_count > 5:
                break

        return res_norm


    def solve_system(self, rhs, model, mu_tau2):
        if model.use_G:
            return self.solve_system_with_G(rhs, model, mu_tau2)
        else:
            return self.solve_system_without_G(rhs, model, mu_tau2)
    

    def solve_system_without_G(self, rhs, model, mu_tau2):
        # Solve Newton system using elimination
        # NOTE: mu has already been accounted for in H

        temp = model.A @ (blk_invhess_prod(rhs.x + rhs.s, model) + rhs.z) + rhs.y
        y_r = lin.fact_solve(self.AHA_fact, temp)
        x_r = blk_invhess_prod(rhs.x + rhs.s - model.A.T @ y_r, model) + rhs.z

        temp = model.A @ blk_invhess_prod(model.c, model) + model.b
        y_b = lin.fact_solve(self.AHA_fact, temp)
        x_b = blk_invhess_prod(model.c - model.A.T @ y_b, model)

        self.sol.tau[0]   = rhs.tau + rhs.kappa + lin.inp(model.c, x_r) + lin.inp(model.b, y_r) 
        self.sol.tau[0]   = self.sol.tau[0] / (mu_tau2 + lin.inp(model.c, x_b) + lin.inp(model.b, y_b))

        self.sol.x[:]     = x_r - self.sol.tau[0] * x_b
        self.sol.y[:]     = y_r - self.sol.tau[0] * y_b
        self.sol.z[:]     = model.A.T @ self.sol.y[:] + model.c * self.sol.tau[0] - rhs.x
        self.sol.s[:]     = self.sol.x[:] - rhs.z
        self.sol.kappa[0] = rhs.kappa - mu_tau2 * self.sol.tau[0]

        return self.sol

    def solve_system_with_G(self, rhs, model, mu_tau2):
        # Solve Newton system using elimination
        # NOTE: mu has already been accounted for in H

        temp_vec = rhs.x - model.G.T @ (blk_hess_prod(rhs.z, model) + rhs.s)
        temp = model.A @ lin.fact_solve(self.GHG_fact, temp_vec) + rhs.y
        y_r = lin.fact_solve(self.AGHGA_fact, temp)
        x_r = lin.fact_solve(self.GHG_fact, temp_vec - model.A.T @ y_r)
        z_r = blk_hess_prod(rhs.z + model.G @ x_r, model) + rhs.s

        temp_vec = model.c - model.G.T @ blk_hess_prod(model.h, model)
        temp = model.A @ lin.fact_solve(self.GHG_fact, temp_vec) + model.b
        y_b = lin.fact_solve(self.AGHGA_fact, temp)
        x_b = lin.fact_solve(self.GHG_fact, temp_vec - model.A.T @ y_b)
        z_b = blk_hess_prod(model.h + model.G @ x_b, model)

        self.sol.tau[0]   = rhs.tau + rhs.kappa + lin.inp(model.c, x_r) + lin.inp(model.b, y_r) + lin.inp(model.h, z_r)
        self.sol.tau[0]   = self.sol.tau[0] / (mu_tau2 + lin.inp(model.c, x_b) + lin.inp(model.b, y_b) + lin.inp(model.h, z_b))

        self.sol.x[:]     = x_r - self.sol.tau[0] * x_b
        self.sol.y[:]     = y_r - self.sol.tau[0] * y_b
        self.sol.z[:]     = z_r - self.sol.tau[0] * z_b
        self.sol.s[:]     = blk_invhess_prod(rhs.s - self.sol.z, model)
        self.sol.kappa[0] = rhs.kappa - mu_tau2 * self.sol.tau[0]

        return self.sol

    
    def apply_system(self, rhs, model, mu_tau2):
        pnt = point.Point(model)

        pnt.x[:]     =  model.A.T @ rhs.y + model.G.T @ rhs.z + model.c * rhs.tau
        pnt.y[:]     = -model.A @ rhs.x + model.b * rhs.tau
        pnt.z[:]     = -model.G @ rhs.x + model.h * rhs.tau - rhs.s
        pnt.s[:]     =  blk_hess_prod(rhs.s, model) + rhs.z
        pnt.tau[0]   = -lin.inp(model.c, rhs.x) - lin.inp(model.b, rhs.y) - lin.inp(model.h, rhs.z) - rhs.kappa
        pnt.kappa[0] = mu_tau2 * rhs.tau + rhs.kappa

        return pnt

def blk_hess_prod(dirs, model):
    out = np.empty_like(dirs)

    for (cone_k, cone_idxs_k) in zip(model.cones, model.cone_idxs):
        out[cone_idxs_k, :] = cone_k.hess_prod(dirs[cone_idxs_k, :])
        
    return out

def blk_invhess_prod(dirs, model):
    out = np.empty_like(dirs)

    for (cone_k, cone_idxs_k) in zip(model.cones, model.cone_idxs):
        out[cone_idxs_k, :] = cone_k.invhess_prod(dirs[cone_idxs_k, :])

    return out