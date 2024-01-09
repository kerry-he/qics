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
    def __init__(self, model, subsolver=None):
        self.sol = point.Point(model)

        if subsolver is None:
            if not model.use_G:
                self.subsolver = "elim"
            else:
                self.subsolver = "qrchol"
        else:
            self.subsolver = subsolver

        if self.subsolver == "qrchol" and model.use_A:
            self.Q, self.R = sp.linalg.qr(model.A.T)
            r = np.linalg.matrix_rank(model.A)
            self.R = self.R[:r, :]
            self.Q1 = self.Q[:, :r]
            self.Q2 = self.Q[:, r:]
            self.GQ1 = model.G @ self.Q1
            self.GQ2 = model.G @ self.Q2

        return
    
    def update_lhs(self, model):
        # Precompute necessary objects on LHS of Newton system

        if self.subsolver == "elim":
            if model.use_G:
                GHG = blk_hess_congruence(model.G, model)
                self.GHG_fact = lin.fact(GHG)

                if model.use_A:
                    GHGA = np.zeros((model.n, model.p))
                    for i in range(model.p):
                        GHGA[:, i] = lin.fact_solve(self.GHG_fact, model.A.T[:, i])
                    AGHGA = model.A @ GHGA
                    self.AGHGA_fact = lin.fact(AGHGA)

            elif model.use_A:
                AHA = blk_invhess_congruence(model.A.T, model)
                self.AHA_fact = lin.fact(AHA)

        if self.subsolver == "qrchol":
            if model.use_A and model.use_G:
                Q2GHGQ2 = blk_hess_congruence(self.GQ2, model)
                self.Q2GHGQ2_fact = lin.fact(Q2GHGQ2)

            elif model.use_A:
                Q2HQ2 = blk_hess_congruence(self.Q2, model)
                self.Q2HQ2_fact = lin.fact(Q2HQ2)
            
            elif model.use_G:
                GHG = blk_hess_congruence(model.G, model)
                self.GHG_fact = lin.fact(GHG)


        # Compute constant 3x3 subsystem
        self.x_b, self.y_b, self.z_b = self.solve_subsystem(model.c, model.b, blk_hess_prod(model.h, model), model)

        # Iterative refinement
        c_est, b_est, h_est = self.forward_subsystem(self.x_b, self.y_b, self.z_b, model)
        (x_res, y_res, z_res) = (model.c - c_est, model.b - b_est, model.h - h_est)
        res_norm = np.linalg.norm(np.vstack((x_res, y_res, z_res)))

        iter_refine_count = 0
        while res_norm > 1e-8:
            x_b2, y_b2, z_b2 = self.solve_subsystem(x_res, y_res, blk_hess_prod(z_res, model), model)      
            (x_temp, y_temp, z_temp) = (self.x_b + x_b2, self.y_b + y_b2, self.z_b + z_b2)
            c_est, b_est, h_est = self.forward_subsystem(x_temp, y_temp, z_temp, model)
            (x_res, y_res, z_res) = (model.c - c_est, model.b - b_est, model.h - h_est)
            res_norm_new = np.linalg.norm(np.vstack((x_res, y_res, z_res)))

            # Check if iterative refinement made things worse
            if res_norm_new > res_norm:
                break

            (self.x_b, self.y_b, self.z_b) = (x_temp, y_temp, z_temp)
            res_norm = res_norm_new

            iter_refine_count += 1

            if iter_refine_count > 5:
                break


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
        # Solve Newton system using elimination
        # NOTE: mu has already been accounted for in H

        x_r, y_r, z_r = self.solve_subsystem(rhs.x, rhs.y, blk_hess_prod(rhs.z, model) + rhs.s, model)
        c_est, b_est, h_est = self.forward_subsystem(x_r, y_r, z_r, model)
        (x_res, y_res, z_res) = (rhs.x - c_est, rhs.y - b_est, rhs.z + blk_invhess_prod(rhs.s, model) - h_est)
        res_norm = np.linalg.norm(np.vstack((x_res, y_res, z_res)))

        iter_refine_count = 0
        while res_norm > 1e-8:
            x_b2, y_b2, z_b2 = self.solve_subsystem(x_res, y_res, blk_hess_prod(z_res, model), model)      
            (x_temp, y_temp, z_temp) = (self.x_b + x_b2, self.y_b + y_b2, self.z_b + z_b2)
            c_est, b_est, h_est = self.forward_subsystem(x_temp, y_temp, z_temp, model)
            (x_res, y_res, z_res) = (rhs.x - c_est, rhs.y - b_est, rhs.z + blk_invhess_prod(rhs.s, model) - h_est)
            res_norm_new = np.linalg.norm(np.vstack((x_res, y_res, z_res)))

            # Check if iterative refinement made things worse
            if res_norm_new > res_norm:
                break

            (x_r, y_r, z_r) = (x_temp, y_temp, z_temp)
            res_norm = res_norm_new

            iter_refine_count += 1

            if iter_refine_count > 5:
                break        

        self.sol.tau[0]   = rhs.tau + rhs.kappa + lin.inp(model.c, x_r) + lin.inp(model.b, y_r) + lin.inp(model.h, z_r)
        self.sol.tau[0]   = self.sol.tau[0] / (mu_tau2 + lin.inp(model.c, self.x_b) + lin.inp(model.b, self.y_b) + lin.inp(model.h, self.z_b))

        self.sol.x[:]     = x_r - self.sol.tau[0] * self.x_b
        self.sol.y[:]     = y_r - self.sol.tau[0] * self.y_b
        self.sol.z[:]     = z_r - self.sol.tau[0] * self.z_b
        self.sol.s[:]     = -model.G @ self.sol.x[:] - rhs.z + self.sol.tau[0] * model.h
        self.sol.kappa[0] = rhs.kappa - mu_tau2 * self.sol.tau[0]

        return self.sol
        
    def solve_subsystem(self, rx, ry, Hrz, model):
        if self.subsolver == "elim":
            return self.solve_subsystem_elim(rx, ry, Hrz, model)
        elif self.subsolver == "qrchol":
            return self.solve_subsystem_qrchol(rx, ry, Hrz, model)

    def solve_subsystem_elim(self, rx, ry, Hrz, model):
        if model.use_A and model.use_G:
            temp_vec = rx - model.G.T @ Hrz
            temp = model.A @ lin.fact_solve(self.GHG_fact, temp_vec) + ry
            y = lin.fact_solve(self.AGHGA_fact, temp)
            x = lin.fact_solve(self.GHG_fact, temp_vec - model.A.T @ y)
            z = Hrz + blk_hess_prod(model.G @ x, model)

            return x, y, z
        
        if model.use_A and not model.use_G:
            temp = model.A @ (blk_invhess_prod(rx + Hrz, model)) + ry
            y = lin.fact_solve(self.AHA_fact, temp)
            x = blk_invhess_prod(rx + Hrz - model.A.T @ y, model)
            z = model.A.T @ y - rx

            return x, y, z
        
        if not model.use_A and model.use_G:
            x = lin.fact_solve(self.GHG_fact, rx - model.G.T @ Hrz)
            z = Hrz + blk_hess_prod(model.G @ x, model)
            y = np.zeros_like(model.b)

            return x, y, z
        
        if not model.use_A and not model.use_G:
            x = blk_invhess_prod(rx + Hrz, model)
            z = Hrz - blk_hess_prod(x, model)
            y = np.zeros_like(model.b)

            return x, y, z
    
    def solve_subsystem_qrchol(self, rx, ry, Hrz, model):
        if model.use_A and model.use_G:
            Q1x = -sp.linalg.solve_triangular(self.R.T, ry, lower=True)
            rhs = self.Q2.T @ (rx - model.G.T @ (Hrz + blk_hess_prod(self.GQ1 @ Q1x, model)))
            Q2x = lin.fact_solve(self.Q2GHGQ2_fact, rhs)
            x = self.Q @ np.vstack((Q1x, Q2x))

            z = Hrz + blk_hess_prod(model.G @ x, model)

            rhs = self.Q1.T @ (rx - model.G.T @ z)
            y = sp.linalg.solve_triangular(self.R, rhs, lower=False)

            return x, y, z

        if model.use_A and not model.use_G:
            Q1x = -sp.linalg.solve_triangular(self.R.T, ry, lower=True)
            rhs = self.Q2.T @ (rx + Hrz - blk_hess_prod(self.Q1 @ Q1x, model))
            Q2x = lin.fact_solve(self.Q2HQ2_fact, rhs)
            x = self.Q @ np.vstack((Q1x, Q2x))

            z = Hrz - blk_hess_prod(x, model)

            rhs = self.Q1.T @ (rx + z)
            y = sp.linalg.solve_triangular(self.R, rhs, lower=False)

            return x, y, z

        if not model.use_A and model.use_G:
            x = lin.fact_solve(self.GHG_fact, rx - model.G.T @ Hrz)
            z = Hrz + blk_hess_prod(model.G @ x, model)
            y = np.zeros_like(model.b)

            return x, y, z

        if not model.use_A and not model.use_G:
            x = blk_invhess_prod(rx + Hrz, model)
            z = Hrz - blk_hess_prod(x, model)
            y = np.zeros_like(model.b)

            return x, y, z

    def solve_subsystem_naive(self, rx, ry, rz, model):
        invH = blk_invhess_prod(np.eye(model.q), model)
        A1 = np.hstack((np.zeros((model.n, model.n)), model.A.T, model.G.T))
        A2 = np.hstack((-model.A, np.zeros((model.p, model.p + model.q))))
        A3 = np.hstack((-model.G, np.zeros((model.q, model.p)), invH))
        A = np.vstack((A1, A2, A3))

        sol = np.linalg.inv(A) @ np.vstack((rx, ry, rz))

        x = sol[:model.n, [0]]
        y = sol[model.n:model.n+model.p]
        z = sol[model.n+model.p:]

        return x, y, z    
    
    def forward_subsystem(self, rx, ry, rz, model):
        x = model.A.T @ ry + model.G.T @ rz
        y = -model.A @ rx
        z = -model.G @ rx + blk_invhess_prod(rz, model)

        return x, y, z
    
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

def blk_hess_congruence(dirs, model):

    p = dirs.shape[1]
    out = np.zeros((p, p))

    for (cone_k, cone_idxs_k) in zip(model.cones, model.cone_idxs):
        dirs_k = dirs[cone_idxs_k, :]

        if not cone_k.use_sqrt:
            H_dir_k = cone_k.hess_prod(dirs[cone_idxs_k, :])
            out += dirs_k.T @ H_dir_k
        else:
            H_dir_k = cone_k.sqrt_hess_prod(dirs[cone_idxs_k, :])
            out += H_dir_k.T @ H_dir_k

    return out

def blk_invhess_congruence(dirs, model):

    p = dirs.shape[1]
    out = np.zeros((p, p))

    for (cone_k, cone_idxs_k) in zip(model.cones, model.cone_idxs):
        dirs_k = dirs[cone_idxs_k, :]

        if not cone_k.use_sqrt:
            H_dir_k = cone_k.invhess_prod(dirs[cone_idxs_k, :])
            out += dirs_k.T @ H_dir_k
        else:
            H_dir_k = cone_k.sqrt_invhess_prod(dirs[cone_idxs_k, :])
            out += H_dir_k.T @ H_dir_k
    

    return out    