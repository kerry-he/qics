import numpy as np
import scipy as sp
import numba as nb
from utils import symmetric as sym
from utils import linear as lin

class Cone():
    def __init__(self, n, hermitian=False):
        # Dimension properties
        self.n  = n                                    # Side length of matrix
        self.hermitian = hermitian                     # Hermitian or symmetric vector space
        self.dim = sym.vec_dim(n, self.hermitian)      # Dimension of the cone
        
        self.grad = lin.Symmetric(self.n)
        self.temp = lin.Symmetric(self.n)
        self.temp_mat = np.zeros((n, n))

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
    
    def set_point(self, point, dual=None, a=True):
        self.X = point.data * a
        self.Z = dual.data * a

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
    
    def hess_prod_ip(self, out, H):
        if not self.nt_aux_updated:
            self.nt_aux()
        XHX = self.inv_X @ H.data @ self.inv_X
        out.data = (XHX + XHX.T) / 2
        return out    
    
    def invhess_prod_ip(self, out, H):
        XHX = self.X @ H.data @ self.X
        out.data = (XHX + XHX.T) / 2
        return out
    
    def third_dir_deriv(self, dir1, dir2=None):
        if not self.grad_updated:
            self.get_grad()
        if dir2 is None:
            H = dir1.data
            XHX_2 = self.inv_X @ H @ self.X_chol_inv.T
            return lin.Symmetric(-2 * XHX_2 @ XHX_2.T)
        else:
            P = dir1.data
            D = dir2.data
            PD = P @ D
            return lin.Symmetric(-2 * self.inv_X @ PD)
    
    def prox(self):
        assert self.feas_updated
        XZX_I = self.X_chol.conj().T @ self.Z @ self.X_chol
        XZX_I.flat[::self.n+1] -= 1
        return np.linalg.norm(XZX_I) ** 2
    
    def nt_aux(self):
        assert not self.nt_aux_updated
        if not self.grad_updated:
            grad_k = self.get_grad()   

        self.Z_chol = sp.linalg.cholesky(self.Z, lower=True, check_finite=False)

        RL = self.Z_chol.T @ self.X_chol
        U, D, Vt = sp.linalg.svd(RL, check_finite=False)
        D_rt2 = np.sqrt(D)

        self.W_rt2 = self.X_chol @ (Vt.T / D_rt2)
        self.W = self.W_rt2 @ self.W_rt2.T

        self.W_irt2 = self.X_chol_inv.T @ (Vt.T * D_rt2)
        self.W_inv = self.W_irt2 @ self.W_irt2.T

        self.nt_aux_updated = True
    
    def nt_prod_ip(self, out, H):
        if not self.nt_aux_updated:
            self.nt_aux()
        WHW = self.W_inv @ H.data @ self.W_inv
        out.data = (WHW + WHW.T) / 2
        return out
    
    def invnt_prod_ip(self, out, H):
        if not self.nt_aux_updated:
            self.nt_aux()
        WHW = self.W @ H.data @ self.W
        out.data = (WHW + WHW.T) / 2
        return out    
    
    def congr_aux(self, A):
        assert not self.congr_aux_updated
        # Check if A matrix is sparse, and build data, col, row arrays if so
        self.A_is_sparse = all([sp.sparse.issparse(Ai.data) for Ai in A])
        if self.A_is_sparse:
            # Check aggregate sparsity structure
            if sum([Ai.data for Ai in A]).nnz > self.n ** 1.5:
                # If aggregate structure is not sparse enough
                self.A_data = ragged_to_array([Ai.data.tocoo().data for Ai in A])
                self.A_cols = ragged_to_array([Ai.data.tocoo().col for Ai in A])
                self.A_rows = ragged_to_array([Ai.data.tocoo().row for Ai in A])
                
                self.congr_mode = 1
            else:
                # If aggergate structure is sparse
                # Construct sparse marix representation of A
                self.A = sp.sparse.vstack([Ai.data.reshape((1, -1)) for Ai in A], format="csr")
                
                # Find where zero columns in A are
                A_where_nz = np.where(self.A.getnnz(0))[0]
                self.A = self.A[:, A_where_nz]
                
                # Get corresponding coordinates to nonzero columns of A
                ijs = [(i, j) for j in range(self.n) for i in range(self.n)]
                ijs = [ijs[idx] for idx in A_where_nz]
                self.A_is = [ij[0] for ij in ijs]
                self.A_js = [ij[1] for ij in ijs]
                
                self.congr_mode = 2
        else:
            # Otherwise just use dense linear algebra
            self.congr_mode = 0

        self.congr_aux_updated = True

    def invnt_congr(self, A):
        if not self.nt_aux_updated:
            self.nt_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)
                    
        p = len(A)
        
        if self.congr_mode == 0:
            # If constraint matrix is dense, then do:
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

        elif self.congr_mode == 1:
            # If constraint matrix is sparse, then do:
            #     For all j=1,...,p
            #         Compute W Aj W
            #         For all i=1,...j
            #             M_ij = <Ai, W Aj W>
            #         End
            #     End            
            out = np.zeros((p, p))
            
            for (j, Aj) in enumerate(A):
                AjW  = Aj.data.dot(self.W)
                WAjW = self.W @ AjW
                out[:j+1, j] = np.sum(WAjW[self.A_rows[:j+1], self.A_cols[:j+1]] * self.A_data[:j+1], 1)
            return out
        
        elif self.congr_mode == 2:
            # If constraint matrix is has sparse aggregate structure
            WW = self.W[np.ix_(self.A_is, self.A_is)] * self.W[np.ix_(self.A_js, self.A_js)]
            return self.A @ WW @ self.A.T
        
    def invhess_congr(self, A):
        if not self.nt_aux_updated:
            self.nt_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)
                    
        p = len(A)
        
        if self.congr_mode == 0:
            # If constraint matrix is dense, then do:
            #     For all j=1,...,p
            #         (A X^1/2 ox X^1/2)_j = X^1/2 Aj X^1/2
            #     End
            #     (A X^1/2 ox X^1/2) @ (X^1/2 ox X^1/2 A)
            lhs = np.zeros((self.dim, p))

            for (j, Aj) in enumerate(A):
                AjX  = Aj.data @ self.X_chol
                XAjX = self.X_chol.conj().T @ AjX
                lhs[:, [j]] = sym.mat_to_vec(XAjX, hermitian=self.hermitian)
            
            return lhs.T @ lhs

        elif self.congr_mode == 1:
            # If constraint matrix is sparse, then do:
            #     For all j=1,...,p
            #         Compute X Aj X
            #         For all i=1,...j
            #             M_ij = <Ai, X Aj X>
            #         End
            #     End            
            out = np.zeros((p, p))
            
            for (j, Aj) in enumerate(A):
                AjX  = Aj.data.dot(self.X)
                XAjX = self.X @ AjX
                out[:j+1, j] = np.sum(XAjX[self.A_roXs[:j+1], self.A_cols[:j+1]] * self.A_data[:j+1], 1)
            return out
        
        elif self.congr_mode == 2:
            # If constraint matrix is has sparse aggregate structure
            XX = self.X[np.ix_(self.A_is, self.A_is)] * self.X[np.ix_(self.A_js, self.A_js)]
            return self.A @ XX @ self.A.T

def ragged_to_array(ragged):
    p = len(ragged)
    ns = [xi.size for xi in ragged]
    n = max(ns)
    array = np.zeros((p, n), dtype=ragged[0].dtype)
    mask = np.ones((p, n), dtype=bool)
    
    for i in range(p):
        array[i, :ns[i]] = ragged[i]
        mask[i, :ns[i]] = 0
        
    # array = np.ma.masked_array(array, mask)
        
    return array