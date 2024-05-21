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
        self.dim = n * n
        self.type = ['s']
        
        # Update flags
        self.feas_updated = False
        self.grad_updated = False
        self.nt_aux_updated = False
        self.congr_aux_updated = False

        self.dgesdd_lwork = sp.linalg.lapack.dgesdd_lwork(n, n)

        return

    def get_nu(self):
        return self.n
    
    def set_init_point(self):
        self.set_point(
            np.eye(self.n), 
            np.eye(self.n)
        )
        return self.X
    
    def set_point(self, point, dual=None, a=True):
        self.X = point * a
        self.Z = dual * a

        self.feas_updated = False
        self.grad_updated = False
        self.nt_aux_updated = False

        return
    
    def get_feas(self):
        if self.feas_updated:
            return self.feas
        
        self.feas_updated = True
        
        # Try to perform Cholesky factorizations to check PSD
        self.X_chol, info = sp.linalg.lapack.dpotrf(self.X, lower=True)
        if info != 0:
            self.feas = False
            return self.feas
        
        self.Z_chol, info = sp.linalg.lapack.dpotrf(self.Z, lower=True)
        if info != 0:
            self.feas = False
            return self.feas        

        self.feas = True
        return self.feas
    
    def get_val(self):
        (sign, logabsdet) = np.linalg.slogdet(self.X)
        return -sign * logabsdet
    
    def get_grad(self):
        assert self.feas_updated

        if self.grad_updated:
            return self.grad
        
        self.X_chol_inv, info = sp.linalg.lapack.dtrtri(self.X_chol, lower=True)
        self.X_inv = self.X_chol_inv.T @ self.X_chol_inv
        # self.X_inv = sp.linalg.blas.dsyrk(alpha=True, a=self.X_chol_inv, trans=True)
        self.grad = -self.X_inv

        self.grad_updated = True
        return self.grad
    
    def hess_prod_ip(self, out, H):
        if not self.grad_updated:
            self.get_grad()
        XHX = self.X_inv @ H @ self.X_inv
        out[:] = (XHX + XHX.T) / 2
        return out    
    
    def invhess_prod_ip(self, out, H):
        XHX = self.X @ H @ self.X
        out[:] = (XHX + XHX.T) / 2
        return out
    
    def third_dir_deriv(self, dir1, dir2=None):
        if not self.grad_updated:
            self.get_grad()
        if dir2 is None:
            XHX_2 = self.X_inv @ dir1 @ self.X_chol_inv.T
            return -2 * XHX_2 @ XHX_2.T
        else:
            PD = dir1 @ dir2
            XiPD = self.X_inv @ PD
            return -XiPD - XiPD.T
    
    def prox(self):
        assert self.feas_updated
        XZX_I = self.X_chol.conj().T @ self.Z @ self.X_chol
        XZX_I.flat[::self.n+1] -= 1
        return np.linalg.norm(XZX_I) ** 2
    
    # @profile
    def nt_aux(self):
        assert not self.nt_aux_updated
        if not self.grad_updated:
            self.get_grad()   

        RL = self.Z_chol.T @ self.X_chol
        U, D, Vt, _ = sp.linalg.lapack.dgesdd(RL, lwork=self.dgesdd_lwork)
        D_rt2 = np.sqrt(D)

        self.W_rt2 = self.X_chol @ (Vt.T / D_rt2)
        self.W = self.W_rt2 @ self.W_rt2.T

        self.W_irt2 = self.X_chol_inv.T @ (Vt.T * D_rt2)
        self.W_inv = self.W_irt2 @ self.W_irt2.T

        self.nt_aux_updated = True
    
    def nt_prod_ip(self, out, H):
        if not self.nt_aux_updated:
            self.nt_aux()
        WHW = self.W_inv @ H @ self.W_inv
        out[:] = np.maximum(WHW, WHW.T)
    
    def invnt_prod_ip(self, out, H):
        if not self.nt_aux_updated:
            self.nt_aux()
        WHW = self.W @ H @ self.W
        out[:] = np.maximum(WHW, WHW.T)
    
    def invhess_mtx(self):
        return lin.kron(self.X, self.X)
    
    def invnt_mtx(self):
        if not self.nt_aux_updated:
            self.nt_aux()        
        return lin.kron(self.W, self.W)    
    
    def congr_aux(self, A):
        assert not self.congr_aux_updated
        # Check if A matrix is sparse, and build data, col, row arrays if so
        if sp.sparse.issparse(A) and sum(A.getnnz(0) > 0) < self.n ** 1.5:
            # If aggergate structure is sparse
            # Find where zero columns in A are
            A_where_nz = np.where(A.getnnz(0))[0]
            self.A_nz = A[:, A_where_nz]
            
            # Get corresponding coordinates to nonzero columns of A
            ijs = [(i, j) for j in range(self.n) for i in range(self.n)]
            ijs = [ijs[idx] for idx in A_where_nz]
            self.A_is = [ij[0] for ij in ijs]
            self.A_js = [ij[1] for ij in ijs]
            
            self.congr_mode = 1
        else:
            # Loop through, get sparse and non-sparse structures
            self.A_sp_data = []
            self.A_sp_cols = []
            self.A_sp_rows = []
            self.A_sp_idxs = []
            self.A_ds_idxs = []
            self.A_ds_mtx  = []
            self.As = []
            
            for i in range(A.shape[0]):
                Ai = A[i].reshape((self.n, self.n))
                self.As.append(Ai)

                if sp.sparse.issparse(Ai) and Ai.nnz < 10:
                    # If aggregate structure is not sparse enough
                    self.A_sp_data.append(Ai.tocoo().data)
                    self.A_sp_cols.append(Ai.tocoo().col)
                    self.A_sp_rows.append(Ai.tocoo().row)
                    self.A_sp_idxs.append(i)
                else:
                    self.A_ds_idxs.append(i)
                    if sp.sparse.issparse(Ai):
                        self.A_ds_mtx.append(Ai.toarray().ravel())
                    else:
                        self.A_ds_mtx.append(Ai.ravel())
                    
            self.A_sp_data = ragged_to_array(self.A_sp_data)
            self.A_sp_cols = ragged_to_array(self.A_sp_cols)
            self.A_sp_rows = ragged_to_array(self.A_sp_rows)
            self.A_ds_mtx  = np.array(self.A_ds_mtx)
                    
            self.congr_mode = 0
            
        self.congr_aux_updated = True

    def nt_congr(self, A):
        if not self.nt_aux_updated:
            self.nt_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)
                    
        p = A.shape[0]
        
        if self.congr_mode == 0:
            out = np.zeros((p, p))
            
            # Compute SPARSE x SPARSE     
            if len(self.A_sp_idxs) > 0:        
                for (j, t) in enumerate(self.A_sp_idxs):
                    ts = self.A_sp_idxs[:j+1]
                    
                    AjW  = A[t].data.dot(self.W_inv)
                    WAjW = self.W_inv @ AjW
                    out[ts, t] = np.sum(WAjW[self.A_sp_rows[:j+1], self.A_sp_cols[:j+1]] * self.A_sp_data[:j+1], 1)
            
            lhs = np.zeros((self.dim, len(self.A_ds_idxs)))
            
            # Compute SPARSE x DENSE
            if len(self.A_sp_idxs) > 0 and len(self.A_ds_idxs) > 0:
                for (j, t) in enumerate(self.A_ds_idxs):
                    AjW  = A[t].data @ self.W_inv
                    WAjW = self.W_inv @ AjW
                    
                    out[self.A_sp_idxs, t] = np.sum(WAjW[self.A_sp_rows, self.A_sp_cols] * self.A_sp_data, 1)
                    out[t, self.A_sp_idxs] = out[self.A_sp_idxs, t]
                
                    lhs[:, j] = WAjW.flat
                    
                out[np.ix_(self.A_ds_idxs, self.A_ds_idxs)] = self.A_ds_mtx @ lhs
                
            # Compute DENSE x DENSE
            if len(self.A_ds_idxs) > 0 and len(self.A_sp_idxs) == 0:
                for (j, t) in enumerate(self.A_ds_idxs):
                    AjW  = self.As[j] @ self.W_irt2
                    WAjW = self.W_irt2.conj().T @ AjW
                    lhs[:, j] = WAjW.flat   
                out = lhs.T @ lhs
                
            return out

        elif self.congr_mode == 1:

            # W_sub_((i,j), (k,l)) = W_(i,l)
            # WW_((i,j), (k,l)) = W_(i,l) * W_(j,k)
            # for all (i,j) in the aggregate sparsity pattern

            # If constraint matrix has sparse aggregate structure
            W_sub = self.W_inv[np.ix_(self.A_is, self.A_js)]
            WW = W_sub * W_sub.T
            return self.A_nz @ WW @ self.A_nz.T

    def invnt_congr(self, A):
        if not self.nt_aux_updated:
            self.nt_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)
                    
        p = A.shape[0]
        
        if self.congr_mode == 0:
            out = np.zeros((p, p))
            
            # Compute SPARSE x SPARSE     
            if len(self.A_sp_idxs) > 0:        
                for (j, t) in enumerate(self.A_sp_idxs):
                    ts = self.A_sp_idxs[:j+1]
                    
                    AjW  = A[t].data.dot(self.W)
                    WAjW = self.W @ AjW
                    out[ts, t] = np.sum(WAjW[self.A_sp_rows[:j+1], self.A_sp_cols[:j+1]] * self.A_sp_data[:j+1], 1)
            
            lhs = np.zeros((self.dim, len(self.A_ds_idxs)))
            
            # Compute SPARSE x DENSE
            if len(self.A_sp_idxs) > 0 and len(self.A_ds_idxs) > 0:
                for (j, t) in enumerate(self.A_ds_idxs):
                    AjW  = A[t].data @ self.W
                    WAjW = self.W.conj().T @ AjW
                    
                    out[self.A_sp_idxs, t] = np.sum(WAjW[self.A_sp_rows, self.A_sp_cols] * self.A_sp_data, 1)
                    out[t, self.A_sp_idxs] = out[self.A_sp_idxs, t]
                
                    lhs[:, j] = WAjW.flat
                    
                out[np.ix_(self.A_ds_idxs, self.A_ds_idxs)] = self.A_ds_mtx @ lhs
                
            # Compute DENSE x DENSE
            if len(self.A_ds_idxs) > 0 and len(self.A_sp_idxs) == 0:
                for (j, t) in enumerate(self.A_ds_idxs):
                    AjW  = self.As[j] @ self.W_rt2
                    WAjW = self.W_rt2.conj().T @ AjW
                    lhs[:, j] = WAjW.flat   
                out = lhs.T @ lhs
                
            return out

        elif self.congr_mode == 1:

            # W_sub_((i,j), (k,l)) = W_(i,l)
            # WW_((i,j), (k,l)) = W_(i,l) * W_(j,k)
            # for all (i,j) in the aggregate sparsity pattern

            # If constraint matrix has sparse aggregate structure
            W_sub = self.W[np.ix_(self.A_is, self.A_js)]
            WW = W_sub * W_sub.T
            return self.A_nz @ WW @ self.A_nz.T

    def hess_congr(self, A):
        if not self.grad_updated:
            self.get_grad()        
        if not self.congr_aux_updated:
            self.congr_aux(A)
                    
        p = A.shape[0]
        
        if self.congr_mode == 0:
            out = np.zeros((p, p))
            
            # Compute SPARSE x SPARSE     
            if len(self.A_sp_idxs) > 0:        
                for (j, t) in enumerate(self.A_sp_idxs):
                    ts = self.A_sp_idxs[:j+1]
                    
                    AjX  = A[t].data.dot(self.X_inv)
                    XAjX = self.X_inv @ AjX
                    out[ts, t] = np.sum(XAjX[self.A_sp_rows[:j+1], self.A_sp_cols[:j+1]] * self.A_sp_data[:j+1], 1)
            
            lhs = np.zeros((self.dim, len(self.A_ds_idxs)))
            
            # Compute SPARSE x DENSE
            if len(self.A_sp_idxs) > 0 and len(self.A_ds_idxs) > 0:
                for (j, t) in enumerate(self.A_ds_idxs):
                    AjX  = A[t].data @ self.X_inv
                    XAjX = self.X_inv @ AjX
                    
                    out[self.A_sp_idxs, t] = np.sum(XAjX[self.A_sp_rows, self.A_sp_cols] * self.A_sp_data, 1)
                    out[t, self.A_sp_idxs] = out[self.A_sp_idxs, t]
                
                    lhs[:, j] = XAjX.flat
                    
                out[np.ix_(self.A_ds_idxs, self.A_ds_idxs)] = self.A_ds_mtx @ lhs
                
            # Compute DENSE x DENSE
            if len(self.A_ds_idxs) > 0 and len(self.A_sp_idxs) == 0:
                for (j, t) in enumerate(self.A_ds_idxs):
                    AjX  = self.As[j] @ self.X_chol_inv
                    XAjX = self.X_chol_inv.conj().T @ AjX
                    lhs[:, j] = XAjX.flat
                out = lhs.T @ lhs
                
            return out

        elif self.congr_mode == 1:
            # If constraint matrix has sparse aggregate structure
            X_sub = self.X_inv[np.ix_(self.A_is, self.A_js)]
            XX = X_sub * X_sub.T
            return self.A_nz @ XX @ self.A_nz.T

    def invhess_congr(self, A):
        if not self.congr_aux_updated:
            self.congr_aux(A)
                    
        p = A.shape[0]
        
        if self.congr_mode == 0:
            out = np.zeros((p, p))
            
            # Compute SPARSE x SPARSE     
            if len(self.A_sp_idxs) > 0:        
                for (j, t) in enumerate(self.A_sp_idxs):
                    ts = self.A_sp_idxs[:j+1]
                    
                    AjX  = A[t].data.dot(self.X)
                    XAjX = self.X @ AjX
                    out[ts, t] = np.sum(XAjX[self.A_sp_rows[:j+1], self.A_sp_cols[:j+1]] * self.A_sp_data[:j+1], 1)
            
            lhs = np.zeros((self.dim, len(self.A_ds_idxs)))
            
            # Compute SPARSE x DENSE
            if len(self.A_sp_idxs) > 0 and len(self.A_ds_idxs) > 0:
                for (j, t) in enumerate(self.A_ds_idxs):
                    AjX  = A[t].data @ self.X
                    XAjX = self.X.conj().T @ AjX
                    
                    out[self.A_sp_idxs, t] = np.sum(XAjX[self.A_sp_rows, self.A_sp_cols] * self.A_sp_data, 1)
                    out[t, self.A_sp_idxs] = out[self.A_sp_idxs, t]
                
                    lhs[:, j] = XAjX.flat
                    
                out[np.ix_(self.A_ds_idxs, self.A_ds_idxs)] = self.A_ds_mtx @ lhs
                
            # Compute DENSE x DENSE
            if len(self.A_ds_idxs) > 0 and len(self.A_sp_idxs) == 0:
                for (j, t) in enumerate(self.A_ds_idxs):
                    AjX  = self.As[j] @ self.X_chol
                    XAjX = self.X_chol.conj().T @ AjX
                    lhs[:, j] = XAjX.flat   
                out = lhs.T @ lhs
                
            return out

        elif self.congr_mode == 1:
            # If constraint matrix has sparse aggregate structure
            X_sub = self.X[np.ix_(self.A_is, self.A_js)]
            XX = X_sub * X_sub.T
            return self.A_nz @ XX @ self.A_nz.T
    
    def comb_dir(self, dS, dZ, sigma, mu):
        if not self.nt_aux_updated:
            self.nt_aux()

        # V = W^-T x = W z where W z = R^T Z R
        V = self.W_rt2.T @ self.Z @ self.W_rt2

        # lhs = -VoV - (W^-T ds)o(W dz) + sigma*mu*I
        WdS = self.W_irt2.T @ dS.data @ self.W_irt2
        WdZ = self.W_rt2.T @ dZ.data @ self.W_rt2
        temp = WdS @ WdZ

        lhs = -V @ V.T - 0.5*(temp + temp.T) + sigma*mu*np.eye(self.n)
        
        eig, vec = np.linalg.eigh(V)

        temp = vec.T @ lhs @ vec
        temp = 2 * temp / np.add.outer(eig, eig)
        temp = vec @ temp @ vec.T

        return lin.Symmetric(self.W_irt2 @ temp @ self.W_irt2.T)
    
def ragged_to_array(ragged):
    p = len(ragged)
    if p == 0:
        return np.array([])
    
    
    ns = [xi.size for xi in ragged]
    n = max(ns)
    array = np.zeros((p, n), dtype=ragged[0].dtype)
    mask = np.ones((p, n), dtype=bool)
    
    for i in range(p):
        array[i, :ns[i]] = ragged[i]
        mask[i, :ns[i]] = 0
        
    return array