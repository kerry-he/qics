import numpy as np
import scipy as sp
from utils import point, linear as lin
from utils import symmetric as sym


# Solves the following square Newton system
#            - A'*y     - c*tau + z         = rx
#        A*x            - b*tau             = ry
#      -c'*x - b'*y                 - kappa = rtau
#     mu*H*x                    + z         = rz 
#                    mu/t^2*tau     + kappa = rkappa
# for (x, y, z, tau, kappa) given right-hand residuals (rx, ry, rz, rtau, rkappa)
# by using elimination.

class SysSolver():
    def __init__(self, model, subsolver=None, ir=True):
        self.sol = point.Point(model)
        self.ir = ir                    # Use iterative refinement or not

        self.pnt_model = point.Point(model)
        self.pnt_model.X = model.c_mtx
        self.pnt_model.y = model.b
        self.pnt_model.Z.fill(0.)

        self.pnt_res = point.Point(model)

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
        self.X_b, self.y_b, self.Z_b = self.solve_subsystem_elim(self.pnt_model, model)

        # Iterative refinement
        c_est, b_est, h_est = self.forward_subsystem(self.X_b, self.y_b, self.Z_b, model)
        (x_res, y_res, z_res) = (self.pnt_model.X - c_est, self.pnt_model.y - b_est, self.pnt_model.Z - h_est)
        res_norm = np.linalg.norm(np.hstack((x_res.ravel(), y_res.ravel(), z_res.ravel())))

        iter_refine_count = 0
        while res_norm > 1e-8:
            self.pnt_res.X = x_res
            self.pnt_res.y[:] = y_res
            self.pnt_res.Z = z_res
            x_b2, y_b2, z_b2 = self.solve_subsystem_elim(self.pnt_res, model)      
            (x_temp, y_temp, z_temp) = (self.X_b + x_b2, self.y_b + y_b2, self.Z_b + z_b2)
            c_est, b_est, h_est = self.forward_subsystem(x_temp, y_temp, z_temp, model)
            (x_res, y_res, z_res) = (self.pnt_model.X - c_est, self.pnt_model.y - b_est, self.pnt_model.Z - h_est)
            res_norm_new = np.linalg.norm(np.hstack((x_res.ravel(), y_res.ravel(), z_res.ravel())))

            # Check if iterative refinement made things worse
            if res_norm_new > res_norm:
                break

            (self.x_b, self.y_b, self.z_b) = (x_temp, y_temp, z_temp)
            res_norm = res_norm_new

            iter_refine_count += 1

            if iter_refine_count > 5:
                break


        return

    def solve_system_ir(self, res, rhs, model, mu, tau):
        temp = point.Point(model)

        # Solve system
        dir = self.solve_system(rhs, model, mu / tau / tau)

        # Check residuals of solve
        res = rhs - self.apply_system(dir, model, mu / tau / tau)
        res_norm = res.norm()

        # Iterative refinement
        if self.ir:
            iter_refine_count = 0
            while res_norm > 1e-8:
                dir_res = self.solve_system(res, model, mu / tau / tau)
                temp = dir + dir_res
                res = rhs - self.apply_system(temp, model, mu / tau / tau)
                res_norm_new = res.norm()

                # Check if iterative refinement made things worse
                if res_norm_new > res_norm:
                    break

                dir = temp
                res_norm = res_norm_new

                iter_refine_count += 1

                if iter_refine_count > 5:
                    break

        return dir, res_norm

    def solve_system(self, rhs, model, mu_tau2):
        # Solve Newton system using elimination
        # NOTE: mu has already been accounted for in H

        sol = point.Point(model)

        x_r, y_r, z_r = self.solve_subsystem_elim(rhs, model)
        if self.ir:
            c_est, b_est, h_est = self.forward_subsystem(x_r, y_r, z_r, model)
            (x_res, y_res, z_res) = (rhs.X - c_est, rhs.y - b_est, rhs.Z + model.cones[0].invhess_prod_alt(rhs.S) - h_est)
            res_norm = np.linalg.norm(np.hstack((x_res.ravel(), y_res.ravel(), z_res.ravel())))

            iter_refine_count = 0
            while res_norm > 1e-8:
                self.pnt_res.X = x_res
                self.pnt_res.y[:] = y_res
                self.pnt_res.Z = z_res                
                x_b2, y_b2, z_b2 = self.solve_subsystem_elim(self.pnt_res, model)      
                (x_temp, y_temp, z_temp) = (x_r + x_b2, y_r + y_b2, z_r + z_b2)
                c_est, b_est, h_est = self.forward_subsystem(x_temp, y_temp, z_temp, model)
                (x_res, y_res, z_res) = (rhs.X - c_est, rhs.y - b_est, rhs.Z + model.cones[0].invhess_prod_alt(rhs.S) - h_est)
                res_norm_new = np.linalg.norm(np.hstack((x_res.ravel(), y_res.ravel(), z_res.ravel())))

                # Check if iterative refinement made things worse
                if res_norm_new > res_norm:
                    break

                (x_r, y_r, z_r) = (x_temp, y_temp, z_temp)
                res_norm = res_norm_new

                iter_refine_count += 1

                if iter_refine_count > 5:
                    break        

        sol.tau   = rhs.tau + rhs.kappa + lin.inp(model.c_mtx, x_r) + lin.inp(model.b, y_r)
        sol.tau  /= (mu_tau2 + lin.inp(model.c_mtx, self.X_b) + lin.inp(model.b, self.y_b))

        sol.X     = x_r - sol.tau * self.X_b
        sol.y     = y_r - sol.tau * self.y_b
        sol.Z     = z_r - sol.tau * self.Z_b
        sol.S     = sol.X - rhs.Z
        sol.kappa = rhs.kappa - mu_tau2 * sol.tau

        return sol
        
    def solve_subsystem(self, rx, ry, Hrz, model):
        if self.subsolver == "elim":
            return self.solve_subsystem_elim(rx, ry, Hrz, model)
        elif self.subsolver == "qrchol":
            return self.solve_subsystem_qrchol(rx, ry, Hrz, model)

    def solve_subsystem_elim(self, rhs, model):
        # rx = rhs.x
        ry = rhs.y
        # rz = rhs.z
        # rs = rhs.s

        rX = rhs.X
        rZ = rhs.Z
        rS = rhs.S

        if model.use_A and model.use_G:
            temp_vec = rx - model.G.T @ (blk_hess_prod(rz, model) + rs)
            temp = model.A @ lin.fact_solve(self.GHG_fact, temp_vec) + ry
            y = lin.fact_solve(self.AGHGA_fact, temp)
            x = lin.fact_solve(self.GHG_fact, temp_vec - model.A.T @ y)
            z = (blk_hess_prod(rz, model) + rs) + blk_hess_prod(model.G @ x, model)

            return x, y, z
        
        if model.use_A and not model.use_G:
            Temp = rZ + model.cones[0].invhess_prod_alt(rX + rS)
            temp = np.zeros_like(ry)
            for (i, Ai) in enumerate(model.A_mtx):
                temp[i] = ry[i] + np.sum(Ai * Temp)
            # temp = model.A @ (rz + blk_invhess_prod(rx + rs, model)) + ry
            y = lin.fact_solve(self.AHA_fact, temp)

            A_T_y = np.zeros_like(rZ)
            for (i, Ai) in enumerate(model.A_mtx):
                A_T_y += y[i] * Ai
            X = rZ + model.cones[0].invhess_prod_alt(rX + rS - A_T_y)
            # x = rz + blk_invhess_prod(rx + rs - model.A.T @ y, model)

            Z = A_T_y - rX
            # z = model.A.T @ y - rx

            return X, y, Z
        
        if not model.use_A and model.use_G:
            x = lin.fact_solve(self.GHG_fact, rx - model.G.T @ (blk_hess_prod(rz, model) + rs))
            z = rs + blk_hess_prod(model.G @ x + rz, model)
            y = np.zeros_like(model.b)

            return x, y, z
        
        if not model.use_A and not model.use_G:
            x = rz + blk_invhess_prod(rx + rs, model)
            z = rs - blk_hess_prod(x - rz, model)
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
    
    def forward_subsystem(self, rX, ry, rZ, model):
        X = np.zeros_like(rX)
        for (i, Ai) in enumerate(model.A_mtx):
            X += ry[i] * Ai
        X -= rZ

        y = np.zeros_like(ry)
        for (i, Ai) in enumerate(model.A_mtx):
            y[i] = -np.sum(Ai * rX)

        Z = rX + model.cones[0].invhess_prod_alt(rZ)

        # x = model.A.T @ ry + model.G.T @ rz
        # y = -model.A @ rx
        # z = -model.G @ rx + blk_invhess_prod(rz, model)        

        return X, y, Z
    
    def apply_system(self, rhs, model, mu_tau2):
        pnt = point.Point(model)

        # pnt.x[:]     =  model.A.T @ rhs.y + model.G.T @ rhs.z + model.c * rhs.tau
        for (i, Ai) in enumerate(model.A_mtx):
            pnt.y[i] = -np.sum(Ai * rhs.X) + model.b[i] * rhs.tau        
        # pnt.y[:]     = -model.A @ rhs.x + model.b * rhs.tau
        # pnt.z[:]     = -model.G @ rhs.x + model.h * rhs.tau - rhs.s
        # pnt.s[:]     =  blk_hess_prod(rhs.s, model) + rhs.z
        pnt.tau   = -lin.inp(model.c_mtx, rhs.X) - lin.inp(model.b, rhs.y) - rhs.kappa
        pnt.kappa = mu_tau2 * rhs.tau + rhs.kappa

        pnt.X = model.c_mtx * rhs.tau - rhs.Z
        for (i, Ai) in enumerate(model.A_mtx): 
            pnt.X += Ai * rhs.y[i]
        pnt.Z = rhs.X - rhs.S
        pnt.S = model.cones[0].hess_prod_alt(rhs.S) + rhs.Z

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