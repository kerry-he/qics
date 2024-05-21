import numpy as np
import scipy as sp
from utils import linear as lin
from utils import vector as vec
from utils import symmetric as sym
from cones import *

# Solves the following square Newton system
#            - A'*y     - c*tau + z         = rx
#        A*x            - b*tau             = ry
#      -c'*x - b'*y                 - kap   = rtau
#     mu*H*x                    + z         = rz 
#                    mu/t^2*tau     + kap   = rkap
# for (x, y, z, tau, kap) given right-hand residuals (rx, ry, rz, rtau, rkap)
# by using elimination.

class SysSolver():
    def __init__(self, model, subsolver=None, ir=True, sym=False):
        self.sol = vec.Point(model)
        self.ir = ir                    # Use iterative refinement or not
        self.sym = sym

        self.cbh = vec.PointXYZ(model)
        self.cbh.x[:]     = model.c
        self.cbh.y[:]     = model.b
        self.cbh.z.vec[:] = model.h
        
        self.xyz_b = vec.PointXYZ(model)
        self.xyz_r = vec.PointXYZ(model)
        self.xyz_ir = vec.PointXYZ(model)
        self.xyz_res = vec.PointXYZ(model)

        self.rz_Hrs = vec.VecProduct(model.cones)
        self.vec_temp = vec.VecProduct(model.cones)
        self.vec_temp2 = vec.VecProduct(model.cones)
        self.pnt_res = vec.Point(model)
        self.dir_ir = vec.Point(model)
        self.res = vec.Point(model)
        
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
                GHG = blk_hess_congruence(model.G_T_views, model, self.sym)
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
                    AHA = (model.A @ H @ model.A_T).toarray()
                else:
                    AHA = blk_invhess_congruence(model.A_views, model, self.sym)
                
                self.AHA_fact = lin.fact(AHA, self.AHA_fact)

        if self.subsolver == "qrchol":
            if model.use_A and model.use_G:
                Q2GHGQ2 = blk_hess_congruence(self.GQ2, model)
                self.Q2GHGQ2_fact = lin.fact(Q2GHGQ2)

            elif model.use_A:
                Q2HQ2 = blk_hess_congruence(self.Q2, model)
                self.Q2HQ2_fact = lin.fact(Q2HQ2)
            
            elif model.use_G:
                GHG = blk_hess_congruence(model.G_T_views, model, self.sym)
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

                self.xyz_b.copy_from(self.xyz_ir)
                res_norm = res_norm_new

                iter_refine_count += 1

                if iter_refine_count > 1:
                    break

        return

    def solve_system_ir(self, dir, rhs, model, mu, tau, kap):
        # Solve system
        self.solve_system(dir, rhs, model, mu / tau / tau, tau, kap)
        res_norm = 0.0
        
        # Check residuals of solve
        if self.ir:
            self.apply_system(self.res, dir, model, mu / tau / tau, kap, tau)
            self.res -= rhs
            res_norm = self.res.norm()

            # Iterative refinement
            iter_refine_count = 0
            while res_norm > 1e-8:
                self.solve_system(self.dir_ir, self.res, model, mu / tau / tau, tau, kap)
                self.dir_ir *= -1
                self.dir_ir += dir
                self.apply_system(self.res, self.dir_ir, model, mu / tau / tau, kap, tau)
                self.res -= rhs
                res_norm_new = self.res.norm()

                # Check if iterative refinement made things worse
                if res_norm_new > res_norm:
                    break

                dir.copy_from(self.dir_ir)
                res_norm = res_norm_new

                iter_refine_count += 1

                if iter_refine_count > 1:
                    break

        return res_norm

    def solve_system(self, sol, rhs, model, mu_tau2, tau, kap):
        # Solve Newton system using elimination
        # NOTE: mu has already been accounted for in H
        
        self.xyz_res.vec[:] = rhs.xyz.vec

        self.solve_subsystem_elim(self.xyz_r, self.xyz_res, model, rhs.s)
        if self.ir:
            # Precompute rz + HrS residual
            blk_invhess_prod_ip(self.rz_Hrs, rhs.s, model, self.sym)
            self.rz_Hrs += rhs.z            
            
            self.forward_subsystem(self.xyz_res, self.xyz_r, model)
            self.xyz_res.x -= rhs.x
            self.xyz_res.y -= rhs.y
            self.xyz_res.z -= self.rz_Hrs
            res_norm = self.xyz_res.norm()
            
            iter_refine_count = 0
            while res_norm > 1e-8:
                self.solve_subsystem_elim(self.xyz_ir, self.xyz_res, model)
                self.xyz_ir *= -1
                self.xyz_ir += self.xyz_r
                self.forward_subsystem(self.xyz_res, self.xyz_ir, model)

                self.pnt_res.x -= rhs.x
                self.pnt_res.y -= rhs.y
                self.pnt_res.z -= self.rz_Hrs      
                res_norm_new = self.xyz_res.norm()

                # Check if iterative refinement made things worse
                if res_norm_new > res_norm:
                    break

                self.xyz_r.copy_from(self.xyz_ir)
                res_norm = res_norm_new

                iter_refine_count += 1

                if iter_refine_count > 1:
                    break        

        # Backsubstitute to obtain solutions for 6x6 block system
        # taunum := rtau + rkap + <c,xr> + <b,yr> + <h,zr>
        tau_num = rhs.tau + rhs.kap + self.cbh.inp(self.xyz_r)
        if self.sym:
            # tauden := kap/tau + <c,xb> + <b,yb> + <h,zb>
            tau_den = (kap / tau + self.cbh.inp(self.xyz_b))
        else:
            # tauden := mu/tau^2 + <c,xb> + <b,yb> + <h,zb>
            tau_den  = (mu_tau2 + self.cbh.inp(self.xyz_b))
        # dtau := taunum / tauden
        sol.tau[:] = tau_num / tau_den

        # (dx,dy,dz) := (xr,yr,zr) - dtau * (xb,yb,zb)
        sol.xyz.vec[:] = sp.linalg.blas.daxpy(self.xyz_b.vec, self.xyz_r.vec, a=-sol.tau[0, 0])

        # ds := -G @ dx + dtau * h - rz
        sol.s.vec[:] = -(model.G @ sol.x)
        sol.s.vec[:] = sp.linalg.blas.daxpy(model.h, sol.s.vec, a=sol.tau[0, 0])
        sol.s -= rhs.z
        
        if self.sym:
            # dkap := rkap - (kap/tau)*dtau
            sol.kap[:] = rhs.kap - kap / tau * sol.tau
        else:
            # dkap := rkap - (mu/tau^2)*dtau
            sol.kap[:] = rhs.kap - mu_tau2 * sol.tau

        return sol
        
    def solve_subsystem(self, rx, ry, Hrz, model):
        if self.subsolver == "elim":
            return self.solve_subsystem_elim(rx, ry, Hrz, model)
        elif self.subsolver == "qrchol":
            return self.solve_subsystem_qrchol(rx, ry, Hrz, model)

    def solve_subsystem_elim(self, out, rhs, model, rs=None):
        rx = rhs.x
        ry = rhs.y
        rz = rhs.z

        if model.use_A and model.use_G:
            temp_vec = rx - model.G.T @ (blk_hess_prod(rz, model, self.sym) + rs)
            temp = model.A @ lin.fact_solve(self.GHG_fact, temp_vec) + ry
            y = lin.fact_solve(self.AGHGA_fact, temp)
            x = lin.fact_solve(self.GHG_fact, temp_vec - model.A.T @ y)
            z = (blk_hess_prod(rz, model, self.sym) + rs) + blk_hess_prod(model.G @ x, model, self.sym)

            return x, y, z
        
        if model.use_A and not model.use_G:
            # Solve for y: AHA y := A(rz + H \ (rx + rs)) + ry
            self.vec_temp.copy_from(rx)
            if rs is not None:
                self.vec_temp += rs
            blk_invhess_prod_ip(self.vec_temp2, self.vec_temp, model, self.sym)
            self.vec_temp2 += rz
            out.x[:] = self.vec_temp2.vec
            temp = model.A @ out.x + ry
            out.y[:] = lin.fact_solve(self.AHA_fact, temp)
            
            # Solve for z: z := At y - rx
            A_T_y = model.A_T @ out.y
            out.z.copy_from(A_T_y - rx)

            # Solve for x: x := rz + H \ (rx + rs - At y)
            self.vec_temp.copy_from(A_T_y)
            blk_invhess_prod_ip(self.vec_temp2, self.vec_temp, model, self.sym)
            out.x[:] -= self.vec_temp2.vec

            return out
        
        if not model.use_A and model.use_G:
            # Solve for x: GHG x = rx - G'(H rz + rs)
            blk_hess_prod_ip(self.vec_temp, rz, model, self.sym)
            if rs is not None:
                self.vec_temp += rs
            temp = rx - model.G_T @ self.vec_temp.vec
            out.x[:] = lin.fact_solve(self.GHG_fact, temp)

            # Solve for z: z = H(rz + Gx) + rs
            self.vec_temp.copy_from(model.G @ out.x)
            self.vec_temp += rz
            blk_hess_prod_ip(out.z, self.vec_temp, model, self.sym)
            if rs is not None:
                out.z += rs

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
        rx = rhs.x
        ry = rhs.y
        rz = rhs.z
        
        out.x[:] = model.A_T @ ry + model.G_T @ rz.vec
        # X = [np.zeros_like(rx[0])]
        # for (i, Ai) in enumerate(model.A_mtx):
        #     X[0] += ry[i] * Ai[0]
        # X[0] -= rz[0]

        out.y[:] = -model.A @ rx
        # y = np.zeros_like(ry)
        # for (i, Ai) in enumerate(model.A_mtx):
        #     y[i] = -np.sum(Ai[0] * rx[0])

        out.z.copy_from(-model.G @ rx)
        out.z += blk_invhess_prod_ip(self.vec_temp2, rz, model, self.sym)

        # x = model.A.T @ ry + model.G.T @ rz
        # y = -model.A @ rx
        # z = -model.G @ rx + blk_invhess_prod(rz, model)        

        return out
    
    def apply_system(self, pnt, rhs, model, mu_tau2, kap, tau):
        rhs_z_vec = rhs.z.vec

        # pnt.x[:]     =  model.A.T @ rhs.y + model.G.T @ rhs.z + model.c * rhs.tau
        pnt.y[:] = -model.A @ rhs.x + model.b * rhs.tau
        # for (i, Ai) in enumerate(model.A_mtx):
        #     pnt.y[i] = -np.sum(Ai[0] * rhs.x[0]) + model.b[i] * rhs.tau        
        # pnt.y[:]     = -model.A @ rhs.x + model.b * rhs.tau
        # pnt.z[:]     = -model.G @ rhs.x + model.h * rhs.tau - rhs.s
        # pnt.s[:]     =  blk_hess_prod(rhs.s, model) + rhs.z
        pnt.tau[:]   = -lin.inp(model.c, rhs.x) - lin.inp(model.b, rhs.y) - lin.inp(model.h, rhs_z_vec) - rhs.kap
        if self.sym:
            pnt.kap[:] = (kap / tau) * rhs.tau + rhs.kap
        else:
            pnt.kap[:] = mu_tau2 * rhs.tau + rhs.kap

        pnt.x[:] = model.A_T @ rhs.y + model.G_T @ rhs_z_vec + model.c * rhs.tau
        # pnt.x = [model.c_mtx[0] * rhs.tau - rhs.z[0]]
        # for (i, Ai) in enumerate(model.A_mtx): 
        #     pnt.x[0] += Ai[0] * rhs.y[i]
        pnt.z.copy_from(-model.G @ rhs.x + model.h * rhs.tau)
        pnt.z -= rhs.s
        blk_hess_prod_ip(pnt.s, rhs.s, model, self.sym)
        pnt.s += rhs.z

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
        # self.AHA_sp_is = []
        # self.AHA_sp_js = []
        
        # for (k, cone_k) in enumerate(model.cones):
        #     Ak = model.A_views[k]
        #     Hk = sp.sparse.csr_array(H_block[k])
        #     AHAk = (Ak @ Hk @ Ak.T).tocoo()
            
        #     self.AHA_sp_is.append(AHAk.col)
        #     self.AHA_sp_js.append(AHAk.row)
        
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
    n = model.n
    out = np.zeros((n, n))

    for (k, cone_k) in enumerate(model.cones):
        if sym:
            out += cone_k.nt_congr(dirs[k])
        else:
            out += cone_k.hess_congr(dirs[k]) 

    return out

def blk_invhess_congruence(dirs, model, sym):
    p = model.p
    out = np.zeros((p, p))

    for (k, cone_k) in enumerate(model.cones):
        if sym:
            out += cone_k.invnt_congr(dirs[k])
        else:
            out += cone_k.invhess_congr(dirs[k]) 

    return out

def blk_invhess_mtx(model):

    H_blk = []

    for cone_k in model.cones:
        if sym:
            H_blk.append(cone_k.invnt_mtx())
        else:
            H_blk.append(cone_k.invhess_mtx())

    return sp.sparse.block_diag(H_blk, 'bsr')