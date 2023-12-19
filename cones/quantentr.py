import numpy as np
import scipy as sp
import numba as nb
import math
from utils import symmetric as sym
from utils import linear    as lin
from utils import mtxgrad   as mgrad

class QuantEntropy():
    def __init__(self, n):
        # Dimension properties
        self.n = n                          # Side dimension of system
        self.vn = sym.vec_dim(self.n)       # Vector dimension of system
        self.dim = 1 + self.vn              # Total dimension of cone

        # Update flags
        self.feas_updated        = False
        self.grad_updated        = False
        self.hess_aux_updated    = False
        self.invhess_aux_updated = False
        self.dder3_aux_updated   = False

        self.tr = sym.mat_to_vec(np.eye(self.n)).T

        return
        
    def get_nu(self):
        return 1 + self.n
    
    def set_init_point(self):
        point = np.empty((self.dim, 1))
        point[0]  = 1.
        point[1:] = sym.mat_to_vec(np.eye(self.n)) / self.n

        self.set_point(point)

        return point
    
    def set_point(self, point):
        assert np.size(point) == self.dim
        self.point = point

        self.t   = point[0, 0]
        self.X   = sym.vec_to_mat(point[1:])
        self.trX = np.trace(self.X)

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

        if any(self.Dx <= 0):
            self.feas = False
            return self.feas
        
        self.log_Dx = np.log(self.Dx)
        self.log_trX  = np.log(self.trX)

        entr_X   = lin.inp(self.Dx, self.log_Dx)
        entr_trX = self.trX * self.log_trX

        self.z = self.t - (entr_X - entr_trX)

        self.feas = (self.z > 0)
        return self.feas
    
    def get_grad(self):
        assert self.feas_updated

        if self.grad_updated:
            return self.grad
        
        log_X           = (self.Ux * self.log_Dx) @ self.Ux.T
        self.log_X      = sym.mat_to_vec(log_X)
        self.tr_log_trX = self.tr.T * self.log_trX

        self.inv_Dx = np.reciprocal(self.Dx)
        self.inv_X  = (self.Ux * self.inv_Dx) @ self.Ux.T

        self.zi   = np.reciprocal(self.z)
        self.DPhi = self.log_X - self.tr_log_trX
        self.DPhi_mat = log_X - np.eye(self.n) * self.log_trX

        self.grad     =  np.empty((self.dim, 1))
        self.grad[0]  = -self.zi
        self.grad[1:] =  self.zi * self.DPhi - sym.mat_to_vec(self.inv_X)

        self.grad_updated = True
        return self.grad
    
    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        self.D1x_log = mgrad.D1_log(self.Dx, self.log_Dx)

        self.D1x_inv = np.reciprocal(np.outer(self.Dx, self.Dx))
        self.D1x_comb = self.D1x_log * self.zi + self.D1x_inv

        self.hess_aux_updated = True

        return

    def hess_prod(self, dirs):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        for j in range(p):
            Ht     = dirs[0, j]
            Hx_vec = dirs[1:, [j]]
            Hx     = sym.vec_to_mat(Hx_vec)

            trH = np.trace(Hx)
            chi  = self.zi * self.zi * (Ht - lin.inp(Hx_vec, self.DPhi))

            UxHUx = self.Ux.T @ Hx @ self.Ux
            D2PhiH = self.Ux @ (self.D1x_comb * UxHUx) @ self.Ux.T

            # Hessian product of barrier function
            out[0, j]    =  chi
            out[1:, [j]] = -chi * self.DPhi + sym.mat_to_vec(D2PhiH) - (self.zi * trH / self.trX) * self.tr.T

        return out
    
    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated

        self.D1x_comb_inv = np.reciprocal(self.D1x_comb)
        self.Hinv_tr      = self.Ux @ np.diag(np.diag(self.D1x_comb_inv)) @ self.Ux.T
        self.tr_Hinv_tr   = np.trace(self.D1x_comb_inv)

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
            Hx = sym.vec_to_mat(dirs[1:, [j]])

            Wx = Hx + Ht * self.DPhi_mat

            UxWUx = self.Ux.T @ Wx @ self.Ux
            Hinv_W = self.Ux @ (self.D1x_comb_inv * UxWUx) @ self.Ux.T

            fac = self.zi * np.trace(Hinv_W) / (self.trX - self.zi * self.tr_Hinv_tr)
            temp = Hinv_W + fac * self.Hinv_tr
            temp = sym.mat_to_vec(temp)

            out[0, j] = Ht * self.z * self.z + lin.inp(temp, self.DPhi)
            out[1:, [j]] = temp

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

        Ht = dirs[0]
        Hx = sym.vec_to_mat(dirs[1:, [0]])
        
        trH = np.trace(Hx)
        UxHxUx = self.Ux.T @ Hx @ self.Ux

        # Quantum conditional entropy oracles
        D2PhiH  = self.Ux @ (self.D1x_log * UxHxUx) @ self.Ux.T
        D2PhiH -= (trH / self.trX) * np.eye(self.n)

        D3PhiHH  = mgrad.scnd_frechet(self.D2x_log, self.Ux, UxHxUx, UxHxUx)
        D3PhiHH += (trH / self.trX) ** 2 * np.eye(self.n)

        # Third derivative of barrier
        DPhiH = lin.inp(self.DPhi_mat, Hx)
        D2PhiHH = lin.inp(D2PhiH, Hx)
        chi = Ht - DPhiH

        dder3 = np.empty((self.dim, 1))
        dder3[0] = -2 * (self.zi**3) * (chi**2) - (self.zi**2) * D2PhiHH

        temp = -dder3[0] * self.DPhi_mat
        temp -= 2 * (self.zi**2) * chi * D2PhiH
        temp += self.zi * D3PhiHH
        temp -= 2 * self.inv_X @ Hx @ self.inv_X @ Hx @ self.inv_X
        dder3[1:] = sym.mat_to_vec(temp)

        return dder3

    def norm_invhess(self, x):
        return 0.0        