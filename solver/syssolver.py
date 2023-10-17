import numpy as np
import scipy as sp
from utils import point

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
        HA = blk_invhess_prod(model.A.T, model)
        AHA = model.A @ HA

        try:
            self.AHA_cho = sp.linalg.cho_factor(AHA)
        except np.linalg.LinAlgError:
            self.AHA_cho = None
            self.AHA_lu = sp.linalg.lu_factor(AHA)


        return
    
    def solve_system(self, rhs, model, mu_tau2):
        # Solve Newton system using elimination
        # NOTE: mu has already been accounted for in H

        temp = model.A @ (blk_invhess_prod(rhs.x + rhs.s, model) + rhs.z) + rhs.y
        y_r = sp.linalg.lu_solve(self.AHA_lu, temp) if self.AHA_cho is None else sp.linalg.cho_solve(self.AHA_cho, temp)
        x_r = blk_invhess_prod(rhs.x + rhs.s - model.A.T @ y_r, model) + rhs.z

        temp = model.A @ blk_invhess_prod(model.c, model) + model.b
        y_b = sp.linalg.lu_solve(self.AHA_lu, temp) if self.AHA_cho is None else sp.linalg.cho_solve(self.AHA_cho, temp)
        x_b = blk_invhess_prod(model.c - model.A.T @ y_b, model)

        self.sol.tau[0]   = rhs.tau + rhs.kappa + np.dot(model.c[:, 0], x_r[:, 0]) + np.dot(model.b[:, 0], y_r[:, 0]) 
        self.sol.tau[0]   = self.sol.tau[0] / (mu_tau2 + np.dot(model.c[:, 0], x_b[:, 0]) + np.dot(model.b[:, 0], y_b[:, 0]))

        self.sol.x[:]     = x_r - self.sol.tau[0] * x_b
        self.sol.y[:]     = y_r - self.sol.tau[0] * y_b
        self.sol.z[:]     = model.A.T @ self.sol.y[:] + model.c * self.sol.tau[0] - rhs.x
        self.sol.s[:]     = self.sol.x[:] - rhs.z
        self.sol.kappa[0] = rhs.kappa - mu_tau2 * self.sol.tau[0]

        return self.sol

    # def solve_system(self, rhs, model, mu_tau2):
    #     # Solve Newton system using elimination
    #     # NOTE: mu has already been accounted for in H

    #     rhs.z.fill(0.)

    #     temp = model.A @ (blk_invhess_prod(rhs.x + rhs.s, model) + rhs.z) + rhs.y
    #     y_r = sp.linalg.lu_solve(self.AHA_lu, temp) if self.AHA_cho is None else sp.linalg.cho_solve(self.AHA_cho, temp)
    #     x_r = blk_invhess_prod(rhs.x + rhs.s - model.A.T @ y_r, model) + rhs.z
    #     z_r = model.A.T @ y_r - rhs.x

    #     temp = model.A @ (blk_invhess_prod(model.c, model) + model.h) + model.b
    #     y_b = sp.linalg.lu_solve(self.AHA_lu, temp) if self.AHA_cho is None else sp.linalg.cho_solve(self.AHA_cho, temp)
    #     x_b = blk_invhess_prod(model.c - model.A.T @ y_b, model) + model.h
    #     z_b = model.A.T @ y_b - model.c

    #     self.sol.tau[0]   = rhs.tau + rhs.kappa + np.dot(model.c[:, 0], x_r[:, 0]) + np.dot(model.b[:, 0], y_r[:, 0]) + np.dot(model.h[:, 0], z_r[:, 0])
    #     self.sol.tau[0]   = self.sol.tau[0] / (mu_tau2 + np.dot(model.c[:, 0], x_b[:, 0]) + np.dot(model.b[:, 0], y_b[:, 0]) + np.dot(model.h[:, 0], z_b[:, 0]))

    #     self.sol.x[:]     = x_r - self.sol.tau[0] * x_b
    #     self.sol.y[:]     = y_r - self.sol.tau[0] * y_b
    #     self.sol.z[:]     = z_r - self.sol.tau[0] * z_b
    #     self.sol.s[:]     = blk_invhess_prod(rhs.s - self.sol.z, model)
    #     self.sol.kappa[0] = rhs.kappa - mu_tau2 * self.sol.tau[0]

    #     return self.sol

    
    def apply_system(self, rhs, model, mu_tau2):
        pnt = point.Point(model)

        pnt.x[:]     =  model.A.T @ rhs.y + model.G.T @ rhs.z + model.c * rhs.tau
        pnt.y[:]     = -model.A @ rhs.x + model.b * rhs.tau
        pnt.z[:]     = -model.G @ rhs.x + model.h * rhs.tau - rhs.s
        pnt.s[:]     =  blk_hess_prod(rhs.s, model) + rhs.z
        pnt.tau[0]   = -np.dot(model.c[:, 0], rhs.x[:, 0]) - np.dot(model.b[:, 0], rhs.y[:, 0]) - np.dot(model.h[:, 0], rhs.z[:, 0]) - rhs.kappa
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