import numpy as np
import scipy as sp
from utils import point, linear as lin
from utils import symmetric as sym
from cones import *


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

        self.cbh = point.PointXYZ(model)
        self.cbh.X = model.c
        self.cbh.y = model.b
        np.copyto(self.cbh.Z.vec, model.h)
        
        self.xyz_b = point.PointXYZ(model)
        self.xyz_r = point.PointXYZ(model)
        self.xyz_ir = point.PointXYZ(model)
        self.xyz_res = point.PointXYZ(model)

        self.rZ_HrS = lin.Vector(model.cones)
        self.vec_temp = lin.Vector(model.cones)
        self.vec_temp2 = lin.Vector(model.cones)
        self.pnt_res = point.Point(model)
        self.dir_ir = point.Point(model)
        self.res = point.Point(model)
        
        self.AHA_fact = None
        self.AHA_is_sparse = None

        self.subsolver = "elim"
        # if subsolver is None:
        #     if not model.use_G:
        #         self.subsolver = "elim"
        #     else:
        #         self.subsolver = "qrchol"
        # else:
        #     self.subsolver = subsolver

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
                # TODO: Check sparsity and if we can use CHOLMOD
                GHG = blk_hess_congruence(model.G_T_vec, model, self.sym)
                self.GHG_fact = lin.fact(GHG)

                if model.use_A:
                    GHGA = np.zeros((model.n, model.p))
                    for i in range(model.p):
                        GHGA[:, i] = lin.fact_solve(self.GHG_fact, model.A.T[:, i])
                    AGHGA = model.A @ GHGA
                    self.AGHGA_fact = lin.fact(AGHGA)

            elif model.use_A:
                # Check sparisty of 
                if self.AHA_is_sparse is None:
                    self.AHA_is_sparse = self.AHA_sparsity(model)

                if self.AHA_is_sparse:
                    H = blk_invhess_mtx(model)
                    AHA = (model.A @ H @ model.A_T).tocsc()
                    # AHA = sp_blk_invhess_congruence(model.A_vec, model, self.AHA_sp_is, self.AHA_sp_js, self.sym)
                else:
                    AHA = blk_invhess_congruence(model.A_vec, model, self.sym)
                
                self.AHA_fact = lin.fact(AHA, self.AHA_fact)

        if self.subsolver == "qrchol":
            if model.use_A and model.use_G:
                Q2GHGQ2 = blk_hess_congruence(self.GQ2, model)
                self.Q2GHGQ2_fact = lin.fact(Q2GHGQ2)

            elif model.use_A:
                Q2HQ2 = blk_hess_congruence(self.Q2, model)
                self.Q2HQ2_fact = lin.fact(Q2HQ2)
            
            elif model.use_G:
                GHG = blk_hess_congruence(model.G_T_vec, model, self.sym)
                self.GHG_fact = lin.fact(GHG)


        # Compute constant 3x3 subsystem
        self.solve_subsystem_elim(self.xyz_b, self.cbh, model)

        if self.ir:            
            self.forward_subsystem(self.xyz_res, self.xyz_b, model)
            self.xyz_res -= self.cbh
            res_norm = self.xyz_res.norm()
            
            iter_refine_count = 0
            while res_norm > 1e-8:
                self.solve_subsystem_elim(self.xyz_ir, self.xyz_res, model)
                self.xyz_ir *= -1
                self.xyz_ir += self.xyz_b
                self.forward_subsystem(self.xyz_res, self.xyz_ir, model)

                self.xyz_res -= self.cbh    
                res_norm_new = self.xyz_res.norm()

                # Check if iterative refinement made things worse
                if res_norm_new > res_norm:
                    break

                self.xyz_b.copy(self.xyz_ir)
                res_norm = res_norm_new

                iter_refine_count += 1

                if iter_refine_count > 1:
                    break        

        # if self.ir:
        #     # Iterative refinement
        #     c_est, b_est, h_est = self.forward_subsystem(self.X_b, self.y_b, self.Z_b, model)
        #     (x_res, y_res, z_res) = (self.pnt_model.X - c_est, self.pnt_model.y - b_est, self.pnt_model.Z - h_est)
        #     res_norm = lin.norm(np.array([lin.norm(x_res), lin.norm(y_res), z_res.norm()]))

        #     iter_refine_count = 0
        #     while res_norm > 1e-8:
        #         self.pnt_res.X = x_res
        #         self.pnt_res.y = y_res
        #         self.pnt_res.Z = z_res
        #         x_b2, y_b2, z_b2 = self.solve_subsystem_elim(self.pnt_res, model)      
        #         (x_temp, y_temp, z_temp) = (self.X_b + x_b2, self.y_b + y_b2, self.Z_b + z_b2)
        #         c_est, b_est, h_est = self.forward_subsystem(x_temp, y_temp, z_temp, model)
        #         (x_res, y_res, z_res) = (self.pnt_model.X - c_est, self.pnt_model.y - b_est, self.pnt_model.Z - h_est)
        #         res_norm_new = lin.norm(np.array([lin.norm(x_res), lin.norm(y_res), z_res.norm()]))

        #         # Check if iterative refinement made things worse
        #         if res_norm_new > res_norm:
        #             break

        #         (self.x_b, self.y_b, self.z_b) = (x_temp, y_temp, z_temp)
        #         res_norm = res_norm_new

        #         iter_refine_count += 1

        #         if iter_refine_count > 1:
        #             break


        return

    # @profile
    def solve_system_ir(self, dir, rhs, model, mu, tau, kappa):
        # Solve system
        self.solve_system(dir, rhs, model, mu / tau / tau, tau, kappa)
        res_norm = 0.0
        
        # Check residuals of solve
        if self.ir:
            self.apply_system(self.res, dir, model, mu / tau / tau, kappa, tau)
            self.res -= rhs
            res_norm = self.res.norm()

            # Iterative refinement
            iter_refine_count = 0
            while res_norm > 1e-8:
                self.solve_system(self.dir_ir, self.res, model, mu / tau / tau, tau, kappa)
                self.dir_ir *= -1
                self.dir_ir += dir
                self.apply_system(self.res, self.dir_ir, model, mu / tau / tau, kappa, tau)
                self.res -= rhs
                res_norm_new = self.res.norm()

                # Check if iterative refinement made things worse
                if res_norm_new > res_norm:
                    break

                dir.copy(self.dir_ir)
                res_norm = res_norm_new

                iter_refine_count += 1

                if iter_refine_count > 1:
                    break

        return res_norm

    def solve_system(self, sol, rhs, model, mu_tau2, tau, kappa):
        # Solve Newton system using elimination
        # NOTE: mu has already been accounted for in H
        
        self.xyz_res.X = rhs.X
        self.xyz_res.y = rhs.y
        self.xyz_res.Z.copy(rhs.Z)

        self.solve_subsystem_elim(self.xyz_r, self.xyz_res, model, rhs.S)
        if self.ir:
            # Precompute rZ + HrS residual
            blk_invhess_prod_ip(self.rZ_HrS, rhs.S, model, self.sym)
            self.rZ_HrS += rhs.Z            
            
            self.forward_subsystem(self.xyz_res, self.xyz_r, model)
            self.xyz_res.X -= rhs.X
            self.xyz_res.y -= rhs.y
            self.xyz_res.Z -= self.rZ_HrS
            res_norm = self.xyz_res.norm()
            
            iter_refine_count = 0
            while res_norm > 1e-8:
                self.solve_subsystem_elim(self.xyz_ir, self.xyz_res, model)
                self.xyz_ir *= -1
                self.xyz_ir += self.xyz_r
                self.forward_subsystem(self.xyz_res, self.xyz_ir, model)

                self.pnt_res.X -= rhs.X
                self.pnt_res.y -= rhs.y
                self.pnt_res.Z -= self.rZ_HrS      
                res_norm_new = self.xyz_res.norm()

                # Check if iterative refinement made things worse
                if res_norm_new > res_norm:
                    break

                self.xyz_r.copy(self.xyz_ir)
                res_norm = res_norm_new

                iter_refine_count += 1

                if iter_refine_count > 1:
                    break        

        tau_num = rhs.tau + rhs.kappa + self.cbh.inp(self.xyz_r)
        if self.sym:
            tau_den = (kappa / tau + self.cbh.inp(self.xyz_b))
        else:
            tau_den  = (mu_tau2 + self.cbh.inp(self.xyz_b))
        sol.tau = tau_num / tau_den

        sol.X     = self.xyz_r.X - sol.tau * self.xyz_b.X
        sol.y     = self.xyz_r.y - sol.tau * self.xyz_b.y
        sol.Z.copy(self.xyz_b.Z)
        sol.Z *= -sol.tau
        sol.Z += self.xyz_r.Z
        sol.S.from_vec(-model.G @ sol.X + sol.tau * model.h)
        sol.S -= rhs.Z
        
        if self.sym:
            sol.kappa = rhs.kappa - kappa / tau * sol.tau
        else:
            sol.kappa = rhs.kappa - mu_tau2 * sol.tau

        return sol
        
    def solve_subsystem(self, rx, ry, Hrz, model):
        if self.subsolver == "elim":
            return self.solve_subsystem_elim(rx, ry, Hrz, model)
        elif self.subsolver == "qrchol":
            return self.solve_subsystem_qrchol(rx, ry, Hrz, model)

    # @profile
    def solve_subsystem_elim(self, out, rhs, model, rS=None):
        rX = rhs.X
        ry = rhs.y
        rZ = rhs.Z

        if model.use_A and model.use_G:
            temp_vec = rx - model.G.T @ (blk_hess_prod(rz, model, self.sym) + rs)
            temp = model.A @ lin.fact_solve(self.GHG_fact, temp_vec) + ry
            y = lin.fact_solve(self.AGHGA_fact, temp)
            x = lin.fact_solve(self.GHG_fact, temp_vec - model.A.T @ y)
            z = (blk_hess_prod(rz, model, self.sym) + rs) + blk_hess_prod(model.G @ x, model, self.sym)

            return x, y, z
        
        if model.use_A and not model.use_G:
            # Solve for y: AHA y = A(rz + H \ (rx + rs)) + ry
            self.vec_temp.copy_from(rX)
            if rS is not None:
                self.vec_temp += rS
            blk_invhess_prod_ip(self.vec_temp2, self.vec_temp, model, self.sym)
            self.vec_temp2 += rZ
            out.X = self.vec_temp2.to_vec()
            temp = model.A @ out.X + ry
            out.y = lin.fact_solve(self.AHA_fact, temp)
            
            # Solve for z: z = At y - rx
            A_T_y = model.A_T @ out.y
            out.Z.from_vec(A_T_y - rX)

            # Solve for x: x = rz + H \ (rx + rs - At y)
            self.vec_temp.from_vec(A_T_y)
            blk_invhess_prod_ip(self.vec_temp2, self.vec_temp, model, self.sym)
            out.X -= self.vec_temp2.to_vec()
            


            return out
        
        if not model.use_A and model.use_G:
            # Solve for x: GHG x = rx - G'(H rz + rs)
            blk_hess_prod_ip(self.vec_temp, rZ, model, self.sym)
            if rS is not None:
                self.vec_temp += rS
            temp = rX - model.G_T @ self.vec_temp.to_vec()
            out.X = lin.fact_solve(self.GHG_fact, temp)

            # Solve for z: z = H(rz + Gx) + rs
            self.vec_temp.from_vec(model.G @ out.X)
            self.vec_temp += rZ
            blk_hess_prod_ip(out.Z, self.vec_temp, model, self.sym)
            if rS is not None:
                out.Z += rS

            # y is empty vector as p=0
            y = np.zeros_like(model.b)

            return out
        
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
    
    # @profile
    def forward_subsystem(self, out, rhs, model):
        rX = rhs.X
        ry = rhs.y
        rZ = rhs.Z
        
        out.X = model.A_T @ ry + model.G_T @ rZ.to_vec()
        # X = [np.zeros_like(rX[0])]
        # for (i, Ai) in enumerate(model.A_mtx):
        #     X[0] += ry[i] * Ai[0]
        # X[0] -= rZ[0]

        out.y = -model.A @ rX
        # y = np.zeros_like(ry)
        # for (i, Ai) in enumerate(model.A_mtx):
        #     y[i] = -np.sum(Ai[0] * rX[0])

        out.Z.from_vec(-model.G @ rX)
        out.Z += blk_invhess_prod_ip(self.vec_temp2, rZ, model, self.sym)

        # x = model.A.T @ ry + model.G.T @ rz
        # y = -model.A @ rx
        # z = -model.G @ rx + blk_invhess_prod(rz, model)        

        return out
    
    def apply_system(self, pnt, rhs, model, mu_tau2, kappa, tau):
        rhs_z_vec = rhs.Z.to_vec()

        # pnt.x[:]     =  model.A.T @ rhs.y + model.G.T @ rhs.z + model.c * rhs.tau
        pnt.y = -model.A @ rhs.X + model.b * rhs.tau
        # for (i, Ai) in enumerate(model.A_mtx):
        #     pnt.y[i] = -np.sum(Ai[0] * rhs.X[0]) + model.b[i] * rhs.tau        
        # pnt.y[:]     = -model.A @ rhs.x + model.b * rhs.tau
        # pnt.z[:]     = -model.G @ rhs.x + model.h * rhs.tau - rhs.s
        # pnt.s[:]     =  blk_hess_prod(rhs.s, model) + rhs.z
        pnt.tau   = -lin.inp(model.c, rhs.X) - lin.inp(model.b, rhs.y) - lin.inp(model.h, rhs_z_vec) - rhs.kappa
        if self.sym:
            pnt.kappa = (kappa / tau) * rhs.tau + rhs.kappa
        else:
            pnt.kappa = mu_tau2 * rhs.tau + rhs.kappa

        pnt.X = model.A_T @ rhs.y + model.G_T @ rhs_z_vec + model.c * rhs.tau
        # pnt.X = [model.c_mtx[0] * rhs.tau - rhs.Z[0]]
        # for (i, Ai) in enumerate(model.A_mtx): 
        #     pnt.X[0] += Ai[0] * rhs.y[i]
        pnt.Z.from_vec(-model.G @ rhs.X + model.h * rhs.tau)
        pnt.Z -= rhs.S
        blk_hess_prod_ip(pnt.S, rhs.S, model, self.sym)
        pnt.S += rhs.Z

        return pnt
    
    def AHA_sparsity(self, model):
        # Check if A is sparse
        if not sp.sparse.issparse(model.A):
            return False
        
        # Check if blocks are "small"
        H_block = []
        for cone_k in model.cones:
            dim_k = cone_k.dim
            # Cone is too big
            if isinstance(cone_k, nonnegorthant.Cone):
                H_block.append(sp.sparse.diags(np.random.rand(dim_k)))
            elif dim_k > model.q * 0.1:         # TODO: Determine a good threshold
                return False
            else:
                H_block.append(np.random.rand(dim_k, dim_k))

        # Check if AHA has a significant sparsity pattern
        H_dummy = sp.sparse.block_diag(H_block)
        AHA_dummy = model.A @ H_dummy @ model.A.T
        if AHA_dummy.nnz > model.q ** 1.5:              # TODO: Determine a good threshold
            return False
        
        # Get sparsity pattern for each cone
        self.AHA_sp_is = []
        self.AHA_sp_js = []
        
        for (k, cone_k) in enumerate(model.cones):
            Ak = model.A[:, model.cone_idxs[k]]
            Hk = sp.sparse.csr_array(H_block[k])
            AHAk = (Ak @ Hk @ Ak.T).tocoo()
            
            self.AHA_sp_is.append(AHAk.col)
            self.AHA_sp_js.append(AHAk.row)
        
        return True
            
            

def blk_hess_prod(dirs, model, sym):
    out = dirs.zeros_like()

    for (k, cone_k) in enumerate(model.cones):
        if sym:
            out[k] = cone_k.nt_prod(dirs[k])
        else:
            out[k] = cone_k.hess_prod(dirs[k])
        
    return out

def blk_invhess_prod(dirs, model, sym):
    out = dirs.zeros_like()

    for (k, cone_k) in enumerate(model.cones):
        if sym:
            out[k] = cone_k.invnt_prod(dirs[k])
        else:
            out[k] = cone_k.invhess_prod(dirs[k])

    return out

def blk_hess_prod_ip(out, dirs, model, sym):
    for (k, cone_k) in enumerate(model.cones):
        if sym:
            cone_k.nt_prod_ip(out[k], dirs[k])
        else:
            cone_k.hess_prod_ip(out[k], dirs[k])
    return out

def blk_invhess_prod_ip(out, dirs, model, sym):
    for (k, cone_k) in enumerate(model.cones):
        if sym:
            cone_k.invnt_prod_ip(out[k], dirs[k])
        else:
            cone_k.invhess_prod_ip(out[k], dirs[k])
    return out

def blk_hess_congruence(dirs, model, sym):

    p = len(dirs)
    out = np.zeros((p, p))

    for (k, cone_k) in enumerate(model.cones):
        dirs_k = [dirs[i].data[k] for i in range(p)]
        if sym:
            out += cone_k.nt_congr(dirs_k)
        else:
            out += cone_k.hess_congr(dirs_k) 

    return out

def blk_invhess_congruence(dirs, model, sym):

    p = len(dirs)
    out = np.zeros((p, p))

    for (k, cone_k) in enumerate(model.cones):
        dirs_k = [dirs[i].mat[k] for i in range(p)]
        if sym:
            out += cone_k.invnt_congr(dirs_k)
        else:
            out += cone_k.invhess_congr(dirs_k) 

    return out

def sp_blk_invhess_congruence(dirs, model, sp_is, sp_js, sym):

    p = len(dirs)
    out = sp.sparse.csc_matrix((p, p))

    for (k, cone_k) in enumerate(model.cones):
        dirs_k = [dirs[i].data[k] for i in range(p)]
        if sym:
            out += cone_k.sp_invnt_congr(dirs_k, sp_is[k], sp_js[k])
        else:
            out += cone_k.sp_invhess_congr(dirs_k, sp_is[k], sp_js[k]) 

    return out

def blk_invhess_mtx(model):

    H_blk = []

    for cone_k in model.cones:
        if sym:
            H_blk.append(cone_k.invnt_mtx())
        else:
            H_blk.append(cone_k.invhess_mtx())

    return sp.sparse.block_diag(H_blk, 'bsr')