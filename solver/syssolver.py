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
    def __init__(self, model, ir=True, sym=False):
        self.ir = ir                    # Use iterative refinement or not
        self.sym = sym

        self.cbh = vec.PointXYZ(model)
        self.cbh.x[:]     = model.c
        self.cbh.y[:]     = model.b
        self.cbh.z.vec[:] = model.h
        
        self.c_xyz = vec.PointXYZ(model)
        self.v_xyz = vec.PointXYZ(model)
        self.xyz_ir = vec.PointXYZ(model)
        self.xyz_res = vec.PointXYZ(model)

        self.rz_Hrs = vec.VecProduct(model.cones)
        self.vec_temp = vec.VecProduct(model.cones)
        self.vec_temp2 = vec.VecProduct(model.cones)
        self.pnt_res = vec.Point(model)
        self.dir_ir = vec.Point(model)
        self.res = vec.Point(model)

        self.H = None
        self.H_inv = None
        
        self.AHA_fact = None
        self.GHG_fact = None
        self.AHA_is_sparse = None
        self.GHG_is_sparse = None

        return
    
    def update_lhs(self, model):
        # Precompute necessary objects on LHS of Newton system

        if model.use_G:
            # Check sparisty of AHA
            if self.GHG_is_sparse is None:
                self.GHG_is_sparse = self.AHA_sparsity(model, model.G_T)

            if self.GHG_is_sparse:# and not model.is_lp:
                self.H_inv, _, _ = blk_invhess_mtx(model, self.Hrows, self.Hcols)
                self.H,     _, _ = blk_hess_mtx(model, self.Hrows, self.Hcols)
                GHG = (model.G_T @ self.H @ model.G).toarray()
            else:
                GHG = blk_hess_congruence(model.G_T_views, model, self.sym)                    

            self.GHG_fact = lin.fact(GHG)

            if model.use_A:
                GHGA = np.zeros((model.n, model.p))
                for i in range(model.p):
                    GHGA[:, i] = lin.fact_solve(self.GHG_fact, model.A.T[:, i])
                AGHGA = model.A @ GHGA
                self.AGHGA_fact = lin.fact(AGHGA)

        elif model.use_A:
            # Check sparisty of AHA
            if self.AHA_is_sparse is None:
                self.AHA_is_sparse = self.AHA_sparsity(model, model.A)

            if self.AHA_is_sparse:
                self.H_inv, _, _ = blk_invhess_mtx(model, self.Hrows, self.Hcols)
                self.H,     _, _ = blk_hess_mtx(model, self.Hrows, self.Hcols)
                AHA = (model.A @ self.H_inv @ model.A_T).toarray()
            else:
                AHA = blk_invhess_congruence(model.A_views, model, self.sym)
            
            self.AHA_fact = lin.fact(AHA, self.AHA_fact)

        # Compute constant 3x3 subsystem
        self.solve_sys_3(self.c_xyz, self.cbh, model)

        if self.ir:            
            self.apply_sys_3(self.xyz_res, self.c_xyz, model)
            self.xyz_res -= self.cbh
            res_norm = self.xyz_res.norm()
            
            iter_refine_count = 0
            while res_norm > 1e-8:
                self.solve_sys_3(self.xyz_ir, self.xyz_res, model)
                self.xyz_ir *= -1
                self.xyz_ir += self.c_xyz
                self.apply_sys_3(self.xyz_res, self.xyz_ir, model)

                self.xyz_res -= self.cbh
                res_norm_new = self.xyz_res.norm()

                # Check if iterative refinement made things worse
                if res_norm_new > res_norm:
                    break

                self.c_xyz.copy_from(self.xyz_ir)
                res_norm = res_norm_new

                iter_refine_count += 1

                if iter_refine_count > 1:
                    break

        return

    def solve_system_ir(self, dir, rhs, model, mu, tau, kap):
        # Solve system
        self.solve_sys_6(dir, rhs, model, mu / tau / tau, tau, kap)
        res_norm = 0.0
        
        # Check residuals of solve
        if self.ir:
            self.apply_sys_6(self.res, dir, model, mu / tau / tau, kap, tau)
            self.res -= rhs
            res_norm = self.res.norm()

            # Iterative refinement
            iter_refine_count = 0
            while res_norm > 1e-8:
                self.solve_sys_6(self.dir_ir, self.res, model, mu / tau / tau, tau, kap)
                self.dir_ir *= -1
                self.dir_ir += dir
                self.apply_sys_6(self.res, self.dir_ir, model, mu / tau / tau, kap, tau)
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

    def solve_sys_6(self, d, r, model, mu_tau2, tau, kap):
        # Compute (dx, dy, dz, ds, dtau, dkap) by solving the 
        # 6x6 block system
        #     [ rx ]    [      A'  G'  c ]  [ dx ]   [    ]
        #     [ ry ] := [ -A           b ]  [ dy ] - [    ]
        #     [ rz ]    [ -G           h ]  [ dz ]   [ ds ] 
        #     [rtau]    [ -c' -b' -h'    ]  [dtau]   [dkap]
        # and
        #       rs   :=  mu H(s) ds + dz
        #      rkap  :=  (mu / tau^2) dtau + dkap   
        # or, if NT scaling is used,
        #       rs   :=  H(w) ds + dz
        #      rkap  :=  (kap / tau) dtau + dkap
        # for a given (rx, ry, rz, rs, rtau, rkap).
        
        # First, solve the two reduced 3x3 subsystems 
        #     [ rx ]    [      A'  G' ]  [ dx ]
        #     [ ry ] := [ -A          ]  [ dy ]
        #     [ rz ]    [ -G     H^-1 ]  [ dz ]
        #               \____ = M ____/
        # for 
        #     1) (cx, cy, cz) := M \ (c, b, h)
        #     2) (vx, vy, vz) := M \ (rx, ry, rz + H \ rs)
        # (the first one has been precomputed)
        self.solve_sys_3(self.v_xyz, r.xyz, model, r.s)


        # Second, backsubstitute to obtain solutions for the full 6x6 system
        #     dtau := (rtau + rkap + c' vx + b' vy + h' vz) / (T + c' cx + b' cy + h' cz)
        #      dx  := vx - dtau cx
        #      dy  := vy - dtau cy
        #      dz  := vz - dtau cz
        #      ds  := -G dx + dtau * h - rz
        #     dkap := rkap - T * dtau
        # where T := mu/tau^2, or T := kap/tau if NT scaling is used

        # taunum := rtau + rkap + c' vx + b' vy + h' vz
        tau_num = r.tau + r.kap + self.cbh.inp(self.v_xyz)
        if self.sym:
            # tauden := kap / tau + c' cx + b' cy + h' cz
            tau_den = (kap / tau + self.cbh.inp(self.c_xyz))
        else:
            # tauden := mu / tau^2 + c' cx + b' cy + h' cz
            tau_den  = (mu_tau2 + self.cbh.inp(self.c_xyz))
        # dtau := taunum / tauden
        d.tau[:] = tau_num / tau_den

        # (dx, dy, dz) := (vx, vy, vz) - dtau * (cx, cy, cz)
        d.xyz.vec[:] = sp.linalg.blas.daxpy(self.c_xyz.vec, self.v_xyz.vec, a=-d.tau[0, 0])

        # ds := -G dx + dtau * h - rz
        np.multiply(model.h, d.tau[0, 0], out=d.s.vec)
        d.s.vec -= model.G @ d.x
        d.s.vec -= r.z.vec
        
        if self.sym:
            # dkap := rkap - (kap/tau) * dtau
            d.kap[:] = r.kap - kap / tau * d.tau
        else:
            # dkap := rkap - (mu/tau^2) * dtau
            d.kap[:] = r.kap - mu_tau2 * d.tau

        return d

    def solve_sys_3(self, d, r, model, rs=None):
        # Compute (dx, dy, dz) by solving the 3x3 block system
        #     [ rx ]   [    ]    [      A'  G' ]  [ dx ]
        #     [ ry ] + [    ] := [ -A          ]  [ dy ]
        #     [ rz ]   [H\rs]    [ -G     H^-1 ]  [ dz ]
        # where H = mu H(s), or H = H(w) if NT scaling is used,
        # for a given (rx, ry, rz, rs).

        # if model.use_A and model.use_G:
        #     # In the general case
        #     #     dy := (A (G'HG)^-1 A') \ [ry + A (G'HG) \ [rx - G' (H rz + rs)]]
        #     #     dx := (G'HG) \ [rx - G' (H rz + rs) - A' dy]
        #     #     dz := H (rz + G dx) + rs
        #
        #     temp_vec = rx - model.G.T @ (blk_hess_prod(rz, model, self.sym) + rs)
        #     temp = model.A @ lin.fact_solve(self.GHG_fact, temp_vec) + ry
        #     y = lin.fact_solve(self.AGHGA_fact, temp)
        #     x = lin.fact_solve(self.GHG_fact, temp_vec - model.A.T @ y)
        #     z = (blk_hess_prod(rz, model, self.sym) + rs) + blk_hess_prod(model.G @ x, model, self.sym)

        #     return x, y, z
        
        if model.use_A and not model.use_G:
            # If G = -I (or some easily invertible square diagonal scaling), then
            #     dy := AHA' \ [A (rz + H \ [rx + rs]) + ry]
            #     dx := rz + H \ [rx + rs - A' dy]
            #     dz := A' dy - rx

            # dy := AHA' \ [A (rz + H \ [rx + rs]) + ry]
            self.vec_temp.copy_from(r.x)
            if rs is not None:
                self.vec_temp += rs
            blk_invhess_prod_ip(self.vec_temp2, self.vec_temp, model, self.sym, self.H_inv)
            self.vec_temp2 += r.z
            d.x[:] = self.vec_temp2.vec
            temp = model.A @ d.x + r.y
            d.y[:] = lin.fact_solve(self.AHA_fact, temp)
            
            # dz := A' dy - rx
            A_T_y = model.A_T @ d.y
            d.z.copy_from(A_T_y - r.x)

            # dx := rz + H \ [rx + rs - A' dy]
            self.vec_temp.copy_from(A_T_y)
            blk_invhess_prod_ip(self.vec_temp2, self.vec_temp, model, self.sym, self.H_inv)
            d.x[:] -= self.vec_temp2.vec
        
        if not model.use_A and model.use_G:
            # If A = [] (i.e, no primal linear constraints), then
            #     dy := []            
            #     dx := G'HG \ [rx - G' (H rz + rs)]
            #     dz := H (rz + G dx) + rs

            # dx := GHG \ [rx - G' (H rz + rs)]
            blk_hess_prod_ip(self.vec_temp, r.z, model, self.sym, self.H)
            if rs is not None:
                self.vec_temp += rs
            temp = r.x - model.G_T @ self.vec_temp.vec
            d.x[:] = lin.fact_solve(self.GHG_fact, temp)

            # dz := H (rz + G dx) + rs
            self.vec_temp.copy_from(model.G @ d.x)
            self.vec_temp += r.z
            blk_hess_prod_ip(d.z, self.vec_temp, model, self.sym, self.H)
            if rs is not None:
                d.z += rs
        
        if not model.use_A and not model.use_G:
            # If both A = [] and G = -I, then
            #     dy := []            
            #     dx := rz + H \ [rx + rs]
            #     dz := H (rz - dx) + rs

            # dx := rz + H \ [rx + rs]
            self.vec_temp.vec[:] = r.x
            if rs is not None:
                self.vec_temp += rs
            blk_invhess_prod_ip(self.vec_temp2, self.vec_temp, model, self.sym, self.H_inv)
            d.x[:] = self.vec_temp2.vec
            d.x   += r.z.vec

            # dz := H (rz - dx) + rs
            self.vec_temp.vec[:] = r.z
            self.vec_temp.vec   -= d.x
            blk_hess_prod_ip(self.vec_temp2, self.vec_temp, model, self.sym, self.H)
            d.z.vec[:] = self.vec_temp2.vec
            if rs is not None:
                d.z.vec += rs
        
        return d

    
    def apply_sys_3(self, r, d, model):
        # Compute (rx, ry, rz) as a forwards pass 
        # of the 3x3 block system
        #     [ rx ]    [      A'  G'  ]  [ dx ]
        #     [ ry ] := [ -A           ]  [ dy ]
        #     [ rz ]    [ -G      H^-1 ]  [ dz ]
        # where H = mu H(s), or H = H(w) if NT scaling is used,
        # for a given (rx, ry, rz).

        # rx := A' dy + G' dz
        r.x[:] = model.A_T @ d.y
        r.x   += model.G_T @ d.z.vec
        
        # ry := -A dx
        r.y[:] = model.A @ d.x
        r.y   *= -1

        # pz := -G dx + H \ dz
        blk_invhess_prod_ip(r.z, d.z, model, self.sym, self.H_inv)
        r.z.vec -= model.G @ d.x

        return r
    
    def apply_sys_6(self, r, d, model, mu_tau2, kap, tau):
        # Compute (rx, ry, rz, rs, rtau, rkap) as a forwards pass 
        # of the 6x6 block system
        #     [ rx ]    [      A'  G'  c ]  [ dx ]   [    ]
        #     [ ry ] := [ -A           b ]  [ dy ] - [    ]
        #     [ rz ]    [ -G           h ]  [ dz ]   [ ds ] 
        #     [rtau]    [ -c' -b' -h'    ]  [dtau]   [dkap]
        # and
        #       rs   :=  mu H(s) ds + dz
        #      rkap  :=  (mu / tau^2) dtau + dkap   
        # or, if NT scaling is used,
        #       rs   :=  H(w) ds + dz
        #      rkap  :=  (kap / tau) dtau + dkap
        # for a given (dx, dy, dz, ds, dtau, dkap).

        # rx := A' dy + G' dz + c dtau
        np.multiply(model.c, d.tau[0, 0], out=r.x)
        r.x += model.A_T @ d.y
        r.x += model.G_T @ d.z.vec

        # ry := -A dx + b dtau
        np.multiply(model.b, d.tau[0, 0], out=r.y)
        r.y -= model.A @ d.x

        # rz := -G dx + h dtau - ds
        np.multiply(model.h, d.tau[0, 0], out=r.z.vec)
        r.z.vec -= model.G @ d.x
        r.z.vec -= d.s.vec

        # rs := mu H ds + dz
        blk_hess_prod_ip(r.s, d.s, model, self.sym, self.H)
        r.s.vec += d.z.vec

        # rtau := -c' dx - b' dy - h' dz - dkap
        r.tau[:] = -(model.c.T @ d.x) - (model.b.T @ d.y) - (model.h.T @ d.z.vec) - d.kap[0, 0]

        if self.sym:
            # rkap := (kap / tau) dtau + dkap
            r.kap[:] = (kap / tau) * d.tau[0, 0] + d.kap[0, 0]
        else:
            # rkap := (mu / tau^2) dtau + dkap   
            r.kap[:] = mu_tau2 * d.tau[0, 0] + d.kap[0, 0]

        return r
    
    def AHA_sparsity(self, model, A):
        # Check if A is sparse
        if not sp.sparse.issparse(A):
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
        H_dummy, self.Hcols, self.Hrows = blk_diag(H_block, None, None)
        AHA_dummy = A @ H_dummy @ A.T
        if AHA_dummy.nnz > model.q ** 1.5:              # TODO: Determine a good threshold
            return False
        
        return True

def blk_hess_prod_ip(out, dirs, model, sym, H):
    if H is not None:
        out.vec[:] = H @ dirs.vec
        return out

    for (k, cone_k) in enumerate(model.cones):
        if sym:
            cone_k.nt_prod_ip(out[k], dirs[k])
        else:
            cone_k.hess_prod_ip(out[k], dirs[k])
    return out

def blk_invhess_prod_ip(out, dirs, model, sym, H_inv):
    if H_inv is not None:
        out.vec[:] = H_inv @ dirs.vec
        return out

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

def blk_invhess_mtx(model, Hcols, Hrows):

    H_blks = []

    for cone_k in model.cones:
        if sym:
            H_blks.append(cone_k.invnt_mtx())
        else:
            H_blks.append(cone_k.invhess_mtx())

    return blk_diag(H_blks, Hcols, Hrows)

def blk_hess_mtx(model, Hcols, Hrows):

    H_blks = []

    for cone_k in model.cones:
        if sym:
            H_blks.append(cone_k.nt_mtx())
        else:
            H_blks.append(cone_k.hess_mtx())

    return blk_diag(H_blks, Hcols, Hrows)

def blk_diag(blks, cols, rows):
    # Precompute sparsity structure
    if cols is None and rows is None:
        # Count number of nonzero entries
        nnz = 0
        for blk in blks:
            n = blk.shape[0]
            if len(blk.shape) == 1:
                # Diagonal Hessian from nonnegative orthant
                nnz += n
            else:
                # Dense Hessian
                nnz += n * n

        cols = np.zeros(nnz, dtype='int32')
        rows = np.zeros(nnz, dtype='int32')

        t = 0   # Row/column counter
        s = 0   # Index counter
        for blk in blks:
            n = blk.shape[0]
            idxs = np.arange(t, t + n)
            if len(blk.shape) == 1:
                # Diagonal Hessian from nonnegative orthant
                cols[s : s + n] = idxs
                rows[s : s + n] = idxs
                s += n
            else:
                # Dense Hessian
                cols[s : s + n*n] = np.tile(idxs, n)
                rows[s : s + n*n] = np.repeat(idxs, n)
                s += n*n
            t += n
    
    vals = np.zeros(len(cols))
    s = 0
    for blk in blks:
        n = blk.shape[0]
        if len(blk.shape) == 1:
            # Diagonal Hessian from nonnegative orthant
            vals[s : s + n] = blk
            s += n
        else:
            # Dense Hessian
            vals[s : s + n*n] = blk.flat
            s += n*n

    return sp.sparse.csr_matrix((vals, (rows, cols))), cols, rows