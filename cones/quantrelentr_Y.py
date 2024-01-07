import numpy as np
import scipy as sp
import math
from utils import symmetric as sym
from utils import linear    as lin
from utils import mtxgrad   as mgrad

class QuantRelEntropyY():
    def __init__(self, n, X):
        # Dimension properties
        self.n = n                          # Side dimension of system
        self.vn = sym.vec_dim(self.n)       # Vector dimension of system
        self.dim = 1 + self.vn              # Total dimension of cone
        self.use_sqrt = False

        self.X      = X
        assert np.all(np.linalg.eigvals(X) > 0)
        log_X       = sp.linalg.logm(X)
        self.entr_X = lin.inp(X, log_X)
        self.tr_X   = np.trace(X)

        self.tr = sym.mat_to_vec(np.eye(self.n)).T

        # Update flags
        self.feas_updated        = False
        self.grad_updated        = False
        self.hess_aux_updated    = False
        self.invhess_aux_updated = False
        self.dder3_aux_updated   = False

        return
        
    def get_nu(self):
        return 1 + self.n
    
    def set_init_point(self):
        point = np.empty((self.dim, 1))

        point[0]  = 1.0 + self.n * (self.entr_X + self.tr_X * np.log(self.n))
        point[1:] = sym.mat_to_vec(np.eye(self.n))        

        self.set_point(point)

        return point
    
    def set_point(self, point):
        assert np.size(point) == self.dim
        self.point = point

        self.t     = point[0]
        self.Y    = sym.vec_to_mat(point[1:])
        self.tr_Y = np.trace(self.Y)

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

        self.Dy, self.Uy = np.linalg.eigh(self.Y)

        if any(self.Dy <= 0):
            self.feas = False
            return self.feas
        
        self.log_Dy = np.log(self.Dy)

        self.log_Y = (self.Uy * self.log_Dy) @ self.Uy.T
        self.Phi   = (self.entr_X + self.tr_X * np.log(self.tr_Y) - lin.inp(self.X, self.log_Y))
        self.z     = self.t - self.tr_Y * self.Phi

        self.feas = (self.z > 0)
        return self.feas
    
    def get_val(self):
        assert self.feas_updated

        return -np.log(self.z) - np.sum(self.log_Dy)
    
    def get_grad(self):
        assert self.feas_updated

        if self.grad_updated:
            return self.grad
        
        self.inv_Dy = np.reciprocal(self.Dy)
        self.inv_Y  = (self.Uy * self.inv_Dy) @ self.Uy.T

        self.D1y_log = mgrad.D1_log(self.Dy, self.log_Dy)

        self.UyXUy = self.Uy.T @ self.X @ self.Uy

        self.zi               =  np.reciprocal(self.z)
        self.D1y_logUyXUy     =  self.D1y_log * self.UyXUy
        self.UyD1y_logUyXUyUy =  self.Uy @ self.D1y_logUyXUy @ self.Uy.T
        self.DPhi             = -self.tr_Y * self.UyD1y_logUyXUyUy
        self.DPhi            += np.eye(self.n) * (self.Phi + self.tr_X)
        self.DPhi_vec         = sym.mat_to_vec(self.DPhi)
        
        self.grad     = np.empty((self.dim, 1))
        self.grad[0]  = -self.zi
        self.grad[1:] = sym.mat_to_vec(self.zi * self.DPhi - self.inv_Y)

        self.grad_updated = True
        return self.grad
    
    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        self.D2y_log     = mgrad.D2_log(self.Dy, self.D1y_log)
        self.D2y_log_UXU = self.D2y_log * self.UyXUy

        self.hess_fac = self.tr_X / self.tr_Y * np.eye(self.n) - self.UyD1y_logUyXUyUy

        # Preparing other required variables
        self.zi2 = self.zi * self.zi

        self.hess_aux_updated = True

        return

    def hess_prod(self, dirs):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        for k in range(p):
            Ht = dirs[0, k]
            Hy = sym.vec_to_mat(dirs[1:, [k]])

            UyHyUy = self.Uy.T @ Hy @ self.Uy

            # Hessian product of conditional entropy
            D2PhiH  = -self.tr_Y * mgrad.scnd_frechet(self.D2y_log_UXU, self.Uy, UyHyUy)
            D2PhiH -=  np.trace(Hy) * self.UyD1y_logUyXUyUy
            D2PhiH +=  lin.inp(self.hess_fac, Hy) * np.eye(self.n)
            
            chi  = self.zi * self.zi * (Ht - lin.inp(Hy, self.DPhi))

            # Hessian product of barrier function
            out[0, k]    = chi
            out[1:, [k]] = sym.mat_to_vec(-chi * self.DPhi + self.zi * D2PhiH + self.inv_Y @ Hy @ self.inv_Y)

        return out
    
    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated

        rt2 = math.sqrt(2.0)

        self.z2 = self.z * self.z

        # Hessians of quantum relative entropy
        Hyy = -self.zi * self.tr_Y * mgrad.get_S_matrix(self.D2y_log_UXU, rt2)

        temp_hess = np.zeros((self.vn, self.vn))

        k = 0
        for j in range(self.n):
            for i in range(j + 1):
                # invYY
                Hyy[k, k] += np.reciprocal(self.Dy[i] * self.Dy[j])
                k += 1

        # Preparing other required variables
        self.Hyy_fact = lin.fact(Hyy)

        self.M = sym.mat_to_vec(self.D1y_logUyXUy)

        self.U = np.hstack((self.tr.T, self.M))
        self.V = np.vstack((self.tr_X / self.tr_Y * self.tr - self.M.T, -self.tr))
        self.Hyy_U = lin.fact_solve(self.Hyy_fact, self.U)

        self.mat = np.linalg.inv(self.z * np.eye(2) + self.V @ self.Hyy_U)

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

        temp_vec = np.empty((self.vn, p))   

        for k in range(p):
            Ht = dirs[0, k]
            Hy = sym.vec_to_mat(dirs[1:, [k]])

            Wy = Hy + Ht * self.DPhi

            temp = self.Uy.T @ Wy @ self.Uy
            temp_vec[:, [k]] = sym.mat_to_vec(temp)

        # temp_vec = self.hess_schur_inv @ temp_vec
        temp_vec = lin.fact_solve(self.Hyy_fact, temp_vec)

        for k in range(p):
            Ht = dirs[0, k]

            temp = temp_vec[:, [k]] - self.Hyy_U @ (self.mat @ (self.V @ temp_vec[:, [k]]))
            temp = sym.vec_to_mat(temp)
            temp = self.Uy @ temp @ self.Uy.T
            outY = sym.mat_to_vec(temp)

            outt = self.z2 * Ht + self.DPhi_vec.T @ outY

            out[0, k] = outt
            out[1:, [k]] = outY

        return out
    
    def update_dder3_aux(self):
        assert not self.dder3_aux_updated
        assert self.hess_aux_updated

        self.zi3 = self.zi2 * self.zi

        self.dder3_aux_updated = True

        return

    def third_dir_deriv(self, dirs):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.dder3_aux_updated:
            self.update_dder3_aux()

        Ht = dirs[0, :]
        Hy = sym.vec_to_mat(dirs[1:, :])
        tr_H = np.trace(Hy)

        out = np.empty((self.dim, 1))

        chi = Ht - lin.inp(self.DPhi, Hy)
        chi2 = chi * chi

        UyHyUy = self.Uy.T @ Hy @ self.Uy
        scnd_frechet = mgrad.scnd_frechet(self.D2y_log_UXU, self.Uy, UyHyUy)

        # Quantum relative entropy Hessians
        D2PhiH  = -self.tr_Y * scnd_frechet
        D2PhiH -=  tr_H * self.UyD1y_logUyXUyUy
        D2PhiH +=  lin.inp(self.hess_fac, Hy) * np.eye(self.n)

        D2PhiHH = lin.inp(Hy, D2PhiH)

        # Quantum relative entropy third order derivatives
        D3PhiHH  = -self.tr_Y * mgrad.thrd_frechet(self.D2y_log, self.Dy, self.Uy, self.UyXUy, UyHyUy, UyHyUy)
        D3PhiHH -=  2 * tr_H * scnd_frechet
        D3PhiHH -=  (lin.inp(Hy, scnd_frechet) + self.tr_X * (tr_H / self.tr_Y) ** 2) * np.eye(self.n)
        
        # Third derivatives of barrier
        dder3_t = -2 * self.zi3 * chi2 - self.zi2 * (D2PhiHH)

        dder3_Y  = -dder3_t * self.DPhi
        dder3_Y -=  2 * self.zi2 * chi * D2PhiH
        dder3_Y +=  self.zi * D3PhiHH
        dder3_Y -=  2 * self.inv_Y @ Hy @ self.inv_Y @ Hy @ self.inv_Y

        out[0]  = dder3_t
        out[1:] = sym.mat_to_vec(dder3_Y)

        return out

    def norm_invhess(self, x):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        Ht = x[0, :]
        Hy = sym.vec_to_mat(x[1:, :])

        Wy = Hy + Ht * self.DPhi

        # Estimate Hessian
        self.D1y_inv = np.reciprocal(np.outer(self.Dy, self.Dy))

        D2y_UXU = self.D2y_log * self.UyXUy
        S_est = np.zeros((self.n, self.n))
        for k in range(self.n):
            eigs_k = np.linalg.eigvalsh(D2y_UXU[k, :, :])
            y_smallest = np.min(eigs_k)
            S_est[k, :] += y_smallest
            S_est[:, k] += y_smallest

        Dyy = -S_est * self.zi * self.tr_Y + self.D1y_inv
        Dyy_inv = np.reciprocal(sym.mat_to_vec(Dyy))

        # Compute inverse
        self.M = sym.mat_to_vec(self.D1y_logUyXUy)

        self.U = np.hstack((self.tr.T, self.M))
        self.V = np.vstack((self.tr_X / self.tr_Y * self.tr - self.M.T, -self.tr))
        Hyy_U = Dyy_inv * self.U

        mat = np.linalg.inv(self.z * np.eye(2) + self.V @ Hyy_U)

        temp = self.Uy.T @ Wy @ self.Uy
        temp = Dyy_inv * sym.mat_to_vec(temp)
        temp = temp - Hyy_U @ (mat @ (self.V @ temp))
        temp = sym.vec_to_mat(temp)
        outY = self.Uy @ temp @ self.Uy.T

        outt = self.z * self.z * Ht + lin.inp(self.DPhi, outY)
        
        return lin.inp(outY, Hy) + (outt * Ht)