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
    def __init__(self, model, subsolver=None, ir=True, sym=False):
        self.sol = point.Point(model)
        self.ir = ir                    # Use iterative refinement or not
        self.sym = sym

        self.pnt_model = point.Point(model)
        self.pnt_model.X = model.c_mtx
        self.pnt_model.y = model.b
        self.pnt_model.Z = model.h_mtx

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
                AHA = blk_invhess_congruence(model.A_mtx, model, self.sym)
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
        res_norm = lin.norm(np.array([x_res.norm(), lin.norm(y_res.data), z_res.norm()]))

        iter_refine_count = 0
        while res_norm > 1e-8:
            self.pnt_res.X = x_res
            self.pnt_res.y = y_res
            self.pnt_res.Z = z_res
            x_b2, y_b2, z_b2 = self.solve_subsystem_elim(self.pnt_res, model)      
            (x_temp, y_temp, z_temp) = (self.X_b + x_b2, self.y_b + y_b2, self.Z_b + z_b2)
            c_est, b_est, h_est = self.forward_subsystem(x_temp, y_temp, z_temp, model)
            (x_res, y_res, z_res) = (self.pnt_model.X - c_est, self.pnt_model.y - b_est, self.pnt_model.Z - h_est)
            res_norm_new = lin.norm(np.array([x_res.norm(), lin.norm(y_res.data), z_res.norm()]))

            # Check if iterative refinement made things worse
            if res_norm_new > res_norm:
                break

            (self.x_b, self.y_b, self.z_b) = (x_temp, y_temp, z_temp)
            res_norm = res_norm_new

            iter_refine_count += 1

            if iter_refine_count > 5:
                break


        return

    def solve_system_ir(self, res, rhs, model, mu, tau, kappa):
        temp = point.Point(model)

        # Solve system
        dir = self.solve_system(rhs, model, mu / tau / tau)

        # Check residuals of solve
        res = rhs - self.apply_system(dir, model, mu / tau / tau, kappa, tau)
        res_norm = res.norm()

        # Iterative refinement
        if self.ir:
            iter_refine_count = 0
            while res_norm > 1e-8:
                dir_res = self.solve_system(res, model, mu / tau / tau)
                temp = dir + dir_res
                res = rhs - self.apply_system(temp, model, mu / tau / tau, kappa, tau)
                res_norm_new = res.norm()

                # Check if iterative refinement made things worse
                if res_norm_new > res_norm:
                    break

                dir = temp
                res_norm = res_norm_new

                iter_refine_count += 1

                # if iter_refine_count > 5:
                #     break

        return dir, res_norm

    #@profile
    def solve_system(self, rhs, model, mu_tau2):
        # Solve Newton system using elimination
        # NOTE: mu has already been accounted for in H

        sol = point.Point(model)

        x_r, y_r, z_r = self.solve_subsystem_elim(rhs, model)
        if self.ir:
            c_est, b_est, h_est = self.forward_subsystem(x_r, y_r, z_r, model)
            (x_res, y_res, z_res) = (rhs.X - c_est, rhs.y - b_est, rhs.Z + blk_invhess_prod(rhs.S, model, self.sym) - h_est)
            res_norm = lin.norm(np.array([x_res.norm(), lin.norm(y_res.data), z_res.norm()]))

            iter_refine_count = 0
            while res_norm > 1e-8:
                self.pnt_res.X = x_res
                self.pnt_res.y[:] = y_res
                self.pnt_res.Z = z_res                
                x_b2, y_b2, z_b2 = self.solve_subsystem_elim(self.pnt_res, model)      
                (x_temp, y_temp, z_temp) = (x_r + x_b2, y_r + y_b2, z_r + z_b2)
                c_est, b_est, h_est = self.forward_subsystem(x_temp, y_temp, z_temp, model)
                (x_res, y_res, z_res) = (rhs.X - c_est, rhs.y - b_est, rhs.Z + blk_invhess_prod(rhs.S, model, self.sym) - h_est)
                res_norm_new = lin.norm(np.array([x_res.norm(), lin.norm(y_res.data), z_res.norm()]))

                # Check if iterative refinement made things worse
                if res_norm_new > res_norm:
                    break

                (x_r, y_r, z_r) = (x_temp, y_temp, z_temp)
                res_norm = res_norm_new

                # iter_refine_count += 1

                # if iter_refine_count > 5:
                #     break        

        sol.tau   = rhs.tau + rhs.kappa + model.c_mtx.inp(x_r) + lin.inp(model.b, y_r)
        sol.tau  /= (mu_tau2 + model.c_mtx.inp(self.X_b) + lin.inp(model.b, self.y_b))

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
        rX = rhs.X
        ry = rhs.y
        rZ = rhs.Z
        rS = rhs.S

        if model.use_A and model.use_G:
            temp_vec = rx - model.G.T @ (blk_hess_prod(rz, model, self.sym) + rs)
            temp = model.A @ lin.fact_solve(self.GHG_fact, temp_vec) + ry
            y = lin.fact_solve(self.AGHGA_fact, temp)
            x = lin.fact_solve(self.GHG_fact, temp_vec - model.A.T @ y)
            z = (blk_hess_prod(rz, model, self.sym) + rs) + blk_hess_prod(model.G @ x, model, self.sym)

            return x, y, z
        
        if model.use_A and not model.use_G:
            Temp = rZ + blk_invhess_prod(rX + rS, model, self.sym)
            temp = model.apply_A(Temp) + ry
            # for (i, Ai) in enumerate(model.A_mtx):
            #     temp[i] = ry[i] + np.sum(Ai[0] * Temp)
            # temp = model.A @ (rz + blk_invhess_prod(rx + rs, model)) + ry
            y = lin.fact_solve(self.AHA_fact, temp)

            A_T_y = model.apply_A_T(y)
            # np.zeros_like(rZ[0])
            # for (i, Ai) in enumerate(model.A_mtx):
            #     A_T_y += y[i] * Ai[0]
            X = rZ + blk_invhess_prod(rX + rS - A_T_y, model, self.sym)
            # x = rz + blk_invhess_prod(rx + rs - model.A.T @ y, model)

            Z = A_T_y - rX
            # z = model.A.T @ y - rx

            return X, y, Z
        
        if not model.use_A and model.use_G:
            x = lin.fact_solve(self.GHG_fact, rx - model.G.T @ (blk_hess_prod(rz, model, self.sym) + rs))
            z = rs + blk_hess_prod(model.G @ x + rz, model, self.sym)
            y = np.zeros_like(model.b)

            return x, y, z
        
        if not model.use_A and not model.use_G:
            x = rz + blk_invhess_prod(rx + rs, model, self.sym)
            z = rs - blk_hess_prod(x - rz, model, self.sym)
            y = np.zeros_like(model.b)

            return x, y, z
    
    def solve_subsystem_qrchol(self, rx, ry, Hrz, model):
        if model.use_A and model.use_G:
            Q1x = -sp.linalg.solve_triangular(self.R.T, ry, lower=True)
            rhs = self.Q2.T @ (rx - model.G.T @ (Hrz + blk_hess_prod(self.GQ1 @ Q1x, model, self.sym)))
            Q2x = lin.fact_solve(self.Q2GHGQ2_fact, rhs)
            x = self.Q @ np.vstack((Q1x, Q2x))

            z = Hrz + blk_hess_prod(model.G @ x, model, self.sym)

            rhs = self.Q1.T @ (rx - model.G.T @ z)
            y = sp.linalg.solve_triangular(self.R, rhs, lower=False)

            return x, y, z

        if model.use_A and not model.use_G:
            Q1x = -sp.linalg.solve_triangular(self.R.T, ry, lower=True)
            rhs = self.Q2.T @ (rx + Hrz - blk_hess_prod(self.Q1 @ Q1x, model, self.sym))
            Q2x = lin.fact_solve(self.Q2HQ2_fact, rhs)
            x = self.Q @ np.vstack((Q1x, Q2x))

            z = Hrz - blk_hess_prod(x, model, self.sym)

            rhs = self.Q1.T @ (rx + z)
            y = sp.linalg.solve_triangular(self.R, rhs, lower=False)

            return x, y, z

        if not model.use_A and model.use_G:
            x = lin.fact_solve(self.GHG_fact, rx - model.G.T @ Hrz)
            z = Hrz + blk_hess_prod(model.G @ x, model, self.sym)
            y = np.zeros_like(model.b)

            return x, y, z

        if not model.use_A and not model.use_G:
            x = blk_invhess_prod(rx + Hrz, model, self.sym)
            z = Hrz - blk_hess_prod(x, model, self.sym)
            y = np.zeros_like(model.b)

            return x, y, z

    def solve_subsystem_naive(self, rx, ry, rz, model):
        invH = blk_invhess_prod(np.eye(model.q), model, self.sym)
        A1 = np.hstack((np.zeros((model.n, model.n)), model.A.T, model.G.T))
        A2 = np.hstack((-model.A, np.zeros((model.p, model.p + model.q))))
        A3 = np.hstack((-model.G, np.zeros((model.q, model.p)), invH))
        A = np.vstack((A1, A2, A3))

        sol = np.linalg.inv(A) @ np.vstack((rx, ry, rz))

        x = sol[:model.n, [0]]
        y = sol[model.n:model.n+model.p]
        z = sol[model.n+model.p:]

        return x, y, z    
    
    #@profile
    def forward_subsystem(self, rX, ry, rZ, model):
        X = model.apply_A_T(ry) - rZ
        # X = [np.zeros_like(rX[0])]
        # for (i, Ai) in enumerate(model.A_mtx):
        #     X[0] += ry[i] * Ai[0]
        # X[0] -= rZ[0]

        y = -model.apply_A(rX)
        # y = np.zeros_like(ry)
        # for (i, Ai) in enumerate(model.A_mtx):
        #     y[i] = -np.sum(Ai[0] * rX[0])

        Z = rX + blk_invhess_prod(rZ, model, self.sym)

        # x = model.A.T @ ry + model.G.T @ rz
        # y = -model.A @ rx
        # z = -model.G @ rx + blk_invhess_prod(rz, model)        

        return X, y, Z
    
    #@profile
    def apply_system(self, rhs, model, mu_tau2, kappa, tau):
        pnt = point.Point(model)

        # pnt.x[:]     =  model.A.T @ rhs.y + model.G.T @ rhs.z + model.c * rhs.tau
        pnt.y = -model.apply_A(rhs.X) + model.b * rhs.tau
        # for (i, Ai) in enumerate(model.A_mtx):
        #     pnt.y[i] = -np.sum(Ai[0] * rhs.X[0]) + model.b[i] * rhs.tau        
        # pnt.y[:]     = -model.A @ rhs.x + model.b * rhs.tau
        # pnt.z[:]     = -model.G @ rhs.x + model.h * rhs.tau - rhs.s
        # pnt.s[:]     =  blk_hess_prod(rhs.s, model) + rhs.z
        pnt.tau   = -model.c_mtx.inp(rhs.X) - lin.inp(model.b, rhs.y) - rhs.kappa
        if self.sym:
            pnt.kappa = (kappa / tau) * rhs.tau + rhs.kappa
        else:
            pnt.kappa = mu_tau2 * rhs.tau + rhs.kappa

        pnt.X = model.apply_A_T(rhs.y) + model.c_mtx * rhs.tau - rhs.Z
        # pnt.X = [model.c_mtx[0] * rhs.tau - rhs.Z[0]]
        # for (i, Ai) in enumerate(model.A_mtx): 
        #     pnt.X[0] += Ai[0] * rhs.y[i]
        pnt.Z = rhs.X - rhs.S
        pnt.S = blk_hess_prod(rhs.S, model, self.sym) + rhs.Z

        return pnt

def blk_hess_prod(dirs, model, sym):
    out = dirs.zeros_like()

    for (k, cone_k) in enumerate(model.cones):
        if sym:
            out.data[k] = cone_k.nt_prod(dirs.data[k])
        else:
            out.data[k] = cone_k.hess_prod(dirs.data[k])
        
    return out

def blk_invhess_prod(dirs, model, sym):
    out = dirs.zeros_like()

    for (k, cone_k) in enumerate(model.cones):
        if sym:
            out.data[k] = cone_k.invnt_prod(dirs.data[k])
        else:
            out.data[k] = cone_k.invhess_prod(dirs.data[k])

    return out

# def blk_hess_congruence(dirs, model):

#     p = dirs.shape[1]
#     out = np.zeros((p, p))

#     for (cone_k, cone_idxs_k) in zip(model.cones, model.cone_idxs):
#         dirs_k = dirs[cone_idxs_k, :]

#         if not cone_k.use_sqrt:
#             H_dir_k = cone_k.hess_prod(dirs[cone_idxs_k, :])
#             out += dirs_k.T @ H_dir_k
#         else:
#             H_dir_k = cone_k.sqrt_hess_prod(dirs[cone_idxs_k, :])
#             out += H_dir_k.T @ H_dir_k

#     return out

def blk_invhess_congruence(dirs, model, sym):

    p = len(dirs)
    out = np.zeros((p, p))

    for (k, cone_k) in enumerate(model.cones):
        dirs_k = [dirs[i].data[k] for i in range(p)]
        if sym:
            out += cone_k.invnt_congr(dirs_k)
        else:
            out += cone_k.invhess_congr(dirs_k) 

    return out