import numpy as np
import scipy as sp
import numba as nb
import math
from utils import symmetric as sym
from utils import linear    as lin
from utils import mtxgrad   as mgrad

# Special cone for the symmetry reduced quantum rate distortion problem with entanglement fidelity distortion
# Matrix is composed of a (n*n - n) dimensional diagonal component y, and a (n x n) dense block component X.
class Cone():
    def __init__(self, n):
        # Dimension properties
        self.n = n                       # Dimension of input
        self.vn = sym.vec_dim(self.n)    # Dimension of vectorized system being traced out
        self.m = n * (n - 1)             # Dimension of diagonal component   

        self.dim = 1 + self.m + self.vn  # Dimension of the cone
        self.use_sqrt = False

        self.idx_y = slice(1, 1 + self.m)
        self.idx_X = slice(1 + self.m, self.dim)        

        # Update flags
        self.feas_updated        = False
        self.grad_updated        = False
        self.hess_aux_updated    = False
        self.invhess_aux_updated = False
        self.dder3_aux_updated   = False

        return
        
    def get_nu(self):
        return 1 + self.n * self.n
    
    def set_init_point(self):
        point = np.empty((self.dim, 1))
        point[0]               = 1 / self.n
        point[self.idx_y, 0]   = np.ones(self.m) / (self.n)
        point[self.idx_X, [0]] = sym.mat_to_vec(np.eye(self.n)) / (self.n)

        self.set_point(point)

        return point
    
    def set_point(self, point):
        assert np.size(point) == self.dim
        self.point = point

        self.t = point[0]
        self.y = point[self.idx_y, [0]]
        self.X = sym.vec_to_mat(point[self.idx_X, [0]])

        self.w = np.diag(self.X) + np.sum(self.y.reshape((self.n, self.n - 1)), 1)

        self.feas_updated        = False
        self.grad_updated        = False
        self.hess_aux_updated    = False
        self.invhess_aux_updated = False
        self.dder3_aux_updated   = False

        return

    def get_feas(self):
        if self.feas_updated:
            return self.feas
        
        self.feas_updated = True

        self.Dx, self.Ux = np.linalg.eigh(self.X)

        if any(self.Dx <= 0) or any(self.y <= 0):
            self.feas = False
            return self.feas
        
        self.log_Dx = np.log(self.Dx)

        self.log_X = (self.Ux * self.log_Dx) @ self.Ux.T
        self.log_y = np.log(self.y)
        self.log_w = np.log(self.w)

        entr_X = np.sum(self.Dx * self.log_Dx)
        entr_y = np.sum(self.y * self.log_y)
        entr_w = np.sum(self.w * self.log_w)

        self.z = self.t - (entr_X + entr_y - entr_w)

        self.feas = (self.z > 0)
        return self.feas

    def get_val(self):
        return -np.log(self.z) - np.sum(self.log_Dx) - np.sum(self.log_y)

    def get_grad(self):
        assert self.feas_updated

        if self.grad_updated:
            return self.grad
        
        self.inv_Dx = np.reciprocal(self.Dx)
        self.inv_X  = (self.Ux * self.inv_Dx) @ self.Ux.T
        self.inv_y = np.reciprocal(self.y)

        self.zi = np.reciprocal(self.z)
        self.DPhiX = self.log_X - np.diag(self.log_w)
        self.DPhiy = self.log_y - np.tile(self.log_w, (self.n - 1, 1)).T.reshape((-1, 1))

        self.grad     = np.empty((self.dim, 1))
        self.grad[0]  = -self.zi
        self.grad[self.idx_y] = self.zi * self.DPhiy - self.inv_y
        self.grad[self.idx_X] = sym.mat_to_vec(self.zi * self.DPhiX - self.inv_X)

        self.grad_updated = True
        return self.grad
    
    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        self.D1x_log = mgrad.D1_log(self.Dx, self.log_Dx)

        self.hess_aux_updated = True

        return

    def hess_prod(self, dirs):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        for j in range(p):
            Ht = dirs[0, j]
            Hx = sym.vec_to_mat(dirs[self.idx_X, [j]])
            Hy = dirs[self.idx_y, [j]]
            Hw = np.diag(Hx) + np.sum(Hy.reshape((self.n, self.n - 1)), 1)

            UxHxUx = self.Ux.T @ Hx @ self.Ux

            # Hessian product of conditional entropy
            D2PhiwH = Hw / self.w

            D2PhiXH = self.Ux @ (self.D1x_log * UxHxUx) @ self.Ux.T - np.diag(D2PhiwH)
            D2PhiyH = Hy / self.y - np.tile(D2PhiwH, (self.n - 1, 1)).T.reshape((-1, 1))

            # Hessian product of barrier function
            outt = (Ht - lin.inp(self.DPhiX, Hx) - lin.inp(self.DPhiy, Hy)) * self.zi * self.zi
            outX = -self.DPhiX * outt + D2PhiXH * self.zi + self.inv_X @ Hx @ self.inv_X
            outy = -self.DPhiy * outt + D2PhiyH * self.zi + Hy / self.y / self.y

            out[0, j] = outt
            out[self.idx_X, [j]] = sym.mat_to_vec(outX)
            out[self.idx_y, [j]] = outy

        return out
    
    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated

        irt2 = math.sqrt(0.5)

        D1x_inv = np.reciprocal(np.outer(self.Dx, self.Dx))
        self.D1x_comb_inv = 1 / (self.zi * self.D1x_log + D1x_inv)
        self.D1y_comb_inv = 1 / (self.zi / self.y + 1 / self.y / self.y)

        self.schur = np.zeros((self.n, self.n))

        for k in range(self.n):
            temp = self.Ux.T[:, [k]] @ self.Ux[[k], :]
            temp = self.Ux @ (self.D1x_comb_inv * temp) @ self.Ux.T
            self.schur[:, k] = -np.diag(temp)

        self.schur -= np.diag(np.sum(self.D1y_comb_inv.reshape((self.n, self.n - 1)), 1) - self.w / self.zi)
            
        self.schur_fact = lin.fact(self.schur)

        self.invhess_aux_updated = True

        return

    def invhess_prod(self, dirs):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        for j in range(p):
            Ht = dirs[0, j]
            Hx = sym.vec_to_mat(dirs[self.idx_X, [j]])
            Hy = dirs[self.idx_y, [j]]

            Wx = Hx + Ht * self.DPhiX
            Wy = Hy + Ht * self.DPhiy

            tempx = self.Ux @ (self.D1x_comb_inv * (self.Ux.T @ Wx @ self.Ux)) @ self.Ux.T
            tempy = Wy * self.D1y_comb_inv
            tempw = np.diag(tempx) + np.sum(tempy.reshape((self.n, self.n - 1)), 1)
            tempw = lin.fact_solve(self.schur_fact, tempw)

            tempx = Wx + np.diag(tempw)
            tempy = Wy + np.tile(tempw, (self.n - 1, 1)).T.reshape((-1, 1))
            
            outX = self.Ux @ (self.D1x_comb_inv * (self.Ux.T @ tempx @ self.Ux)) @ self.Ux.T
            outY = tempy * self.D1y_comb_inv            

            out[0, j] = Ht * self.z * self.z + lin.inp(outX, self.DPhiX) + lin.inp(outY, self.DPhiy)
            out[self.idx_X, [j]] = sym.mat_to_vec(outX)
            out[self.idx_y, [j]] = outY

        return out
    
    def update_dder3_aux(self):
        assert not self.dder3_aux_updated
        assert self.hess_aux_updated

        self.D2x_log = mgrad.D2_log(self.Dx, self.D1x_log)

        self.dder3_aux_updated = True

        return

    def third_dir_deriv(self, dirs):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.dder3_aux_updated:
            self.update_dder3_aux()

        Ht = dirs[0, :]
        Hx = sym.vec_to_mat(dirs[self.idx_X, :])
        Hy = dirs[self.idx_y, :]
        Hw = np.diag(Hx) + np.sum(Hy.reshape((self.n, self.n - 1)), 1)


        # Quantum conditional entropy oracles
        D2PhiwH  = Hw / self.w
        D3PhiwHH = - (Hw / self.w) ** 2

        UxHxUx   = self.Ux.T @ Hx @ self.Ux
        D2PhiXH  = self.Ux @ (self.D1x_log * UxHxUx) @ self.Ux.T - np.diag(D2PhiwH)
        D3PhiXHH = mgrad.scnd_frechet(self.D2x_log, UxHxUx, UxHxUx, self.Ux) - np.diag(D3PhiwHH)

        D2PhiyH  = Hy / self.y - np.tile(D2PhiwH, (self.n - 1, 1)).T.reshape((-1, 1))
        D3PhiyHH = - (Hy / self.y) ** 2 - np.tile(D3PhiwHH, (self.n - 1, 1)).T.reshape((-1, 1))

        # Third derivative of barrier
        DPhiXH = lin.inp(self.DPhiX, Hx)
        DPhiyH = lin.inp(self.DPhiy, Hy)
        D2PhiXHH = lin.inp(D2PhiXH, Hx)        
        D2PhiyHH = lin.inp(D2PhiyH, Hy)        
        chi = Ht - DPhiXH - DPhiyH
        chi2 = chi * chi

        # Third derivatives of barrier
        dder3_t = -2 * (self.zi**3) * chi2 - (self.zi**2) * (D2PhiXHH + D2PhiyHH)

        dder3_X  = -dder3_t * self.DPhiX
        dder3_X -=  2 * (self.zi**2) * chi * D2PhiXH
        dder3_X +=  self.zi * D3PhiXHH
        dder3_X -=  2 * self.inv_X @ Hx @ self.inv_X @ Hx @ self.inv_X

        dder3_y  = -dder3_t * self.DPhiy
        dder3_y -=  2 * (self.zi**2) * chi * D2PhiyH
        dder3_y +=  self.zi * D3PhiyHH
        dder3_y -=  2 * (Hy**2) / (self.y**3)

        dder3             = np.empty((self.dim, 1))
        dder3[0]          = dder3_t
        dder3[self.idx_X] = sym.mat_to_vec(dder3_X)
        dder3[self.idx_y] = dder3_y

        return dder3

    def norm_invhess(self, x):     
        return 0