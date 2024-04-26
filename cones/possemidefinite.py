import numpy as np
import scipy as sp
import math
from utils import symmetric as sym

class Cone():
    def __init__(self, n, hermitian=False):
        # Dimension properties
        self.n  = n                                    # Side length of matrix
        self.hermitian = hermitian                     # Hermitian or symmetric vector space
        self.dim = sym.vec_dim(n, self.hermitian)      # Dimension of the cone
        self.use_sqrt = True

        # Update flags
        self.feas_updated = False
        self.grad_updated = False
        self.nt_aux_updated = False

        return
    
    def zeros(self):
        return np.zeros((self.n, self.n))
        
    def get_nu(self):
        return self.n
    
    def set_init_point(self):
        self.set_point(np.eye(self.n), np.eye(self.n))
        return self.X
    
    def set_point(self, point, dual=None):
        assert np.shape(point)[0] == self.n

        self.X = point
        self.Z = dual

        self.feas_updated = False
        self.grad_updated = False
        self.nt_aux_updated = False

        return
    
    def get_feas(self):
        if self.feas_updated:
            return self.feas
        
        self.feas_updated = True

        try:
            self.X_chol = np.linalg.cholesky(self.X)
            self.feas = True
        except np.linalg.linalg.LinAlgError:
            self.feas = False
            return self.feas

        if self.Z is not None:
            try:
                self.Z_chol = sp.linalg.cholesky(self.Z, lower=True, check_finite=False)
                self.feas = True
            except sp.linalg.LinAlgError:
                self.feas = False            

        return self.feas
    
    def get_val(self):
        (sign, logabsdet) = np.linalg.slogdet(self.X)
        return -sign * logabsdet
    
    def get_grad(self):
        assert self.feas_updated

        if self.grad_updated:
            return self.grad
        
        self.X_chol_inv = np.linalg.inv(self.X_chol)
        self.inv_X = self.X_chol_inv.T @ self.X_chol_inv
        self.grad  = -self.inv_X

        self.grad_updated = True
        return self.grad

    def hess_prod(self, H):
        assert self.grad_updated
        XHX = self.inv_X @ H @ self.inv_X
        return (XHX + XHX.T) / 2

    def sqrt_hess_prod(self, dirs):
        assert self.grad_updated

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        for j in range(p):
            H = sym.vec_to_mat(dirs[:, [j]], hermitian=self.hermitian)
            out[:, [j]] = sym.mat_to_vec(self.X_chol_inv @ H @ self.X_chol_inv.conj().T, hermitian=self.hermitian)

        return out
    
    def invhess_prod(self, H):
        XHX = self.X @ H @ self.X
        return (XHX + XHX.T) / 2

    def sqrt_invhess_prod(self, dirs):
        assert self.grad_updated

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        for j in range(p):
            H = sym.vec_to_mat(dirs[:, [j]], hermitian=self.hermitian)
            out[:, [j]] = sym.mat_to_vec(self.X_chol.conj().T @ H @ self.X_chol, hermitian=self.hermitian)

        return out

    def third_dir_deriv(self, dir1, dir2=None):
        assert self.grad_updated
        if dir2 is None:
            H = dir1
            XHXHX = self.inv_X @ H @ self.inv_X @ H @ self.inv_X
            return -(XHXHX + XHXHX.T)
        else:
            P = dir1
            D = dir2
            PD = P @ D
            temp = self.inv_X @ PD
            return -(temp + temp.T)

    def norm_invhess(self, x):
        return 0.0
    
    def invhess_congr(self, H):
        p = len(H)
        lhs = np.zeros((self.dim, p))

        for (i, Hi) in enumerate(H):
            lhs[:, [i]] = sym.mat_to_vec(self.X_chol.conj().T @ Hi @ self.X_chol)
        
        return lhs.T @ lhs    
    

    def nt_aux(self):
        assert not self.nt_aux_updated

        RL = self.Z_chol.T @ self.X_chol
        U, D, Vt = sp.linalg.svd(RL, check_finite=False)
        D_rt2 = np.sqrt(D)

        self.W_rt2 = self.X_chol @ (Vt.T / D_rt2)
        self.W = self.W_rt2 @ self.W_rt2.T

        self.W_irt2 = self.X_chol_inv.T @ (Vt.T * D_rt2)
        self.W_inv = self.W_irt2 @ self.W_irt2.T

        self.nt_aux_updated = True


    def nt_prod(self, H):
        if not self.nt_aux_updated:
            self.nt_aux()
        WHW = self.W_inv @ H @ self.W_inv
        return (WHW + WHW.T) / 2

    def invnt_prod(self, H):
        if not self.nt_aux_updated:
            self.nt_aux()
        WHW = self.W @ H @ self.W
        return (WHW + WHW.T) / 2
    
    def invnt_congr(self, H):
        if not self.nt_aux_updated:
            self.nt_aux()        
        p = len(H)
        lhs = np.zeros((self.dim, p))

        for (i, Hi) in enumerate(H):
            lhs[:, [i]] = sym.mat_to_vec(self.W_rt2.conj().T @ Hi @ self.W_rt2)
        
        return lhs.T @ lhs    