import numpy as np
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

        return
        
    def get_nu(self):
        return self.n
    
    def set_init_point(self):
        point = sym.mat_to_vec(np.eye(self.n), hermitian=self.hermitian)
        self.set_point(point)
        return point
    
    def set_point(self, point):
        assert np.size(point) == self.dim
        self.point = point

        self.X = sym.vec_to_mat(point[:, [0]], hermitian=self.hermitian)

        self.feas_updated = False
        self.grad_updated = False

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
    
    def get_val(self):
        (sign, logabsdet) = np.linalg.slogdet(self.X)
        return -sign * logabsdet
    
    def get_grad(self):
        assert self.feas_updated

        if self.grad_updated:
            return self.grad
        
        self.X_chol_inv = np.linalg.inv(self.X_chol)
        self.inv_X = self.X_chol_inv.conj().T @ self.X_chol_inv
        self.grad  = -sym.mat_to_vec(self.inv_X, hermitian=self.hermitian)

        self.grad_updated = True
        return self.grad

    def hess_prod(self, dirs):
        assert self.grad_updated

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        for j in range(p):
            H = sym.vec_to_mat(dirs[:, [j]], hermitian=self.hermitian)
            out[:, [j]] = sym.mat_to_vec(self.inv_X @ H @ self.inv_X, hermitian=self.hermitian)

        return out

    def sqrt_hess_prod(self, dirs):
        assert self.grad_updated

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        for j in range(p):
            H = sym.vec_to_mat(dirs[:, [j]], hermitian=self.hermitian)
            out[:, [j]] = sym.mat_to_vec(self.X_chol_inv @ H @ self.X_chol_inv.conj().T, hermitian=self.hermitian)

        return out
    
    def invhess_prod(self, dirs):
        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        for j in range(p):
            H = sym.vec_to_mat(dirs[:, [j]], hermitian=self.hermitian)
            out[:, [j]] = sym.mat_to_vec(self.X @ H @ self.X, hermitian=self.hermitian)

        return out

    def sqrt_invhess_prod(self, dirs):
        assert self.grad_updated

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        for j in range(p):
            H = sym.vec_to_mat(dirs[:, [j]], hermitian=self.hermitian)
            out[:, [j]] = sym.mat_to_vec(self.X_chol.conj().T @ H @ self.X_chol, hermitian=self.hermitian)

        return out        

    def third_dir_deriv(self, dirs):
        assert self.grad_updated
        H = sym.vec_to_mat(dirs, hermitian=self.hermitian)
        return -2 * sym.mat_to_vec(self.inv_X @ H @ self.inv_X @ H @ self.inv_X, hermitian=self.hermitian)

    def norm_invhess(self, x):
        return 0.0