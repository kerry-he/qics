import numpy as np
import scipy as sp
import math
from utils import symmetric as sym
from utils import linear as lin

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
        return lin.Symmetric(self.n)
        
    def get_nu(self):
        return self.n
    
    def set_init_point(self):
        self.set_point(
            lin.Symmetric(np.eye(self.n)), 
            lin.Symmetric(np.eye(self.n))
        )
        return lin.Symmetric(self.X)
    
    def set_point(self, point, dual=None):
        self.X = point.data
        self.Z = dual.data

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
            return lin.Symmetric(self.grad)
        
        self.X_chol_inv = np.linalg.inv(self.X_chol)
        self.inv_X = self.X_chol_inv.T @ self.X_chol_inv
        self.grad  = -self.inv_X

        self.grad_updated = True
        return lin.Symmetric(self.grad)

    def hess_prod(self, H):
        assert self.grad_updated
        return lin.Symmetric(self.inv_X @ H.data @ self.inv_X)
    
    def invhess_prod(self, H):
        return lin.Symmetric(self.X @ H.data @ self.X)

    def third_dir_deriv(self, dir1, dir2=None):
        assert self.grad_updated
        if dir2 is None:
            H = dir1.data
            return lin.Symmetric(-2 * self.inv_X @ H @ self.inv_X @ H @ self.inv_X)
        else:
            P = dir1.data
            D = dir2.data
            PD = P @ D
            return lin.Symmetric(-2 * self.inv_X @ PD)

    def norm_invhess(self, x):
        return 0.0
    
    def invhess_congr(self, H):
        p = len(H)
        lhs = np.zeros((self.dim, p))

        for (i, Hi) in enumerate(H):
            lhs[:, [i]] = sym.mat_to_vec(self.X_chol.conj().T @ Hi.data @ self.X_chol)
        
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
        return lin.Symmetric(self.W_inv @ H.data @ self.W_inv)

    def invnt_prod(self, H):
        if not self.nt_aux_updated:
            self.nt_aux()
        return lin.Symmetric(self.W @ H.data @ self.W)
    
    def invnt_congr(self, H):
        if not self.nt_aux_updated:
            self.nt_aux()        
        p = len(H)
        lhs = np.zeros((self.dim, p))

        for (i, Hi) in enumerate(H):
            lhs[:, [i]] = sym.mat_to_vec(self.W_rt2.conj().T @ Hi.data @ self.W_rt2)
        
        return lhs.T @ lhs    