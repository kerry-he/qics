import numpy as np
import scipy as sp
from utils import symmetric as sym
from utils import linear as lin

class Cone():
    def __init__(self, n, hermitian=False):
        # Dimension properties
        self.n  = n                                    # Side length of matrix
        self.hermitian = hermitian                     # Hermitian or symmetric vector space
        self.dim = sym.vec_dim(n, self.hermitian)      # Dimension of the cone
        self.use_sqrt = True
        
        self.grad = lin.Symmetric(self.n)
        self.temp = lin.Symmetric(self.n)

        # Update flags
        self.feas_updated = False
        self.grad_updated = False
        self.nt_aux_updated = False
        self.congr_aux_updated = False

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
            self.X_chol = sp.linalg.cholesky(self.X, lower=True, check_finite=False)
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
        
        self.X_chol_inv, info = sp.linalg.lapack.dtrtri(self.X_chol, lower=True)
        self.inv_X = self.X_chol_inv.T @ self.X_chol_inv
        self.grad.data = -self.inv_X

        self.grad_updated = True
        return self.grad

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
        self.temp.data = self.W_inv @ H.data @ self.W_inv
        return self.temp

    def invnt_prod(self, H):
        if not self.nt_aux_updated:
            self.nt_aux()
        self.temp.data = self.W @ H.data @ self.W
        return self.temp
    
    def congr_aux(self, A):
        # Check if A matrix is sparse, and build data, col, row arrays if so
        self.A_is_sparse = all([sp.sparse.issparse(Ai.data) for Ai in A])
        if self.A_is_sparse:
            self.A_data = np.array([Ai.data.tocoo().data for Ai in A])
            self.A_cols = np.array([Ai.data.tocoo().col for Ai in A])
            self.A_rows = np.array([Ai.data.tocoo().row for Ai in A])

    def invnt_congr(self, A):
        if not self.nt_aux_updated:
            self.nt_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)
                    
        p = len(A)
        
        # If constraint matrix is sparse, then do:
        #     For all j=1,...,p
        #         Compute W Aj W
        #         For all i=1,...j
        #             M_ij = <Ai, W Aj W>
        #         End
        #     End
        if self.A_is_sparse:
            WAjW = np.zeros((self.n, self.n))
            out = np.zeros((p, p))

            for (j, Aj) in enumerate(A):
                AjW  = Aj.data @ self.W
                WAjW = self.W.conj().T @ AjW          #TODO: Use gemmt to only compute upper triangular 
                out[:j+1, j] = np.sum(WAjW[self.A_rows[:j+1], self.A_cols[:j+1]] * self.A_data[:j+1], 1)                
            return out

        # If constraint matrix is not all sparse, then do:
        #     For all j=1,...,p
        #         (A W^1/2 ox W^1/2)_j = W^1/2 Aj W^1/2
        #     End
        #     (A W^1/2 ox W^1/2) @ (W^1/2 ox W^1/2 A)
        lhs = np.zeros((self.dim, p))

        for (j, Aj) in enumerate(A):
            AjW  = Aj.data @ self.W_rt2
            WAjW = self.W_rt2.conj().T @ AjW
            lhs[:, [j]] = sym.mat_to_vec(WAjW, hermitian=self.hermitian)
        
        return lhs.T @ lhs