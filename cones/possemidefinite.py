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
        self.grad = -self.X_inv

        self.grad_updated = True
        return self.grad
    
    def hess_prod_ip(self, out, H):
        if not self.grad_updated:
            self.get_grad()
        XHX = self.X_inv @ H @ self.X_inv
        np.add(XHX, XHX.T, out=out)
        out *= 0.5
        return out    
    
    def invhess_prod_ip(self, out, H):
        XHX = self.X @ H @ self.X
        np.add(XHX, XHX.T, out=out)
        out *= 0.5
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
    
    def nt_aux(self):
        assert not self.nt_aux_updated
        if not self.grad_updated:
            self.get_grad()   

        # Compute the symmeterized point
        # Lambda = R' Z R = R^-1 S R^-T
        RL = self.Z_chol.T @ self.X_chol
        _, self.Lambda, Vt, _ = sp.linalg.lapack.dgesdd(RL, lwork=self.dgesdd_lwork)
        D_rt2 = np.sqrt(self.Lambda)

        # Compute the scaling point W = R R'
        self.R = self.X_chol @ (Vt.T / D_rt2)
        self.W = self.R @ self.R.T

        self.R_inv = (self.X_chol_inv.T @ (Vt.T * D_rt2)).T
        self.W_inv = self.R_inv.T @ self.R_inv

        self.nt_aux_updated = True
    
    def nt_prod_ip(self, out, H):
        if not self.nt_aux_updated:
            self.nt_aux()
        WHW = self.W_inv @ H @ self.W_inv
        np.add(WHW, WHW.T, out=out)
        out *= 0.5
    
    def invnt_prod_ip(self, out, H):
        if not self.nt_aux_updated:
            self.nt_aux()
        WHW = self.W @ H @ self.W
        np.add(WHW, WHW.T, out=out)
        out *= 0.5
    
    def invhess_mtx(self):
        return lin.kron(self.X, self.X)
    
    def invnt_mtx(self):
        if not self.nt_aux_updated:
            self.nt_aux()        
        return lin.kron(self.W, self.W)
    
    def hess_mtx(self):
        return lin.kron(self.X_inv, self.X_inv)
    
    def nt_mtx(self):
        if not self.nt_aux_updated:
            self.nt_aux()        
        return lin.kron(self.W_inv, self.W_inv)        
    
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
        return self.base_congr(A, self.W_inv, self.R_inv.T)

    def invnt_congr(self, A):
        if not self.nt_aux_updated:
            self.nt_aux()
        return self.base_congr(A, self.W, self.R)

    def hess_congr(self, A):
        if not self.grad_updated:
            self.get_grad()
        return self.base_congr(A, self.X_inv, self.X_chol_inv)

    def invhess_congr(self, A):
        return self.base_congr(A, self.X, self.X_chol)
        
    def base_congr(self, A, X, X_rt2):
        if not self.congr_aux_updated:
            self.congr_aux(A)
                    
        p = A.shape[0]
        
        if self.congr_mode == 0:
            out = np.zeros((p, p))
            
            # Compute SPARSE x SPARSE     
            if len(self.A_sp_idxs) > 0:        
                for (j, t) in enumerate(self.A_sp_idxs):
                    ts = self.A_sp_idxs[:j+1]
                    
                    AjX  = self.As[t].dot(X)
                    XAjX = X @ AjX
                    out[ts, t] = np.sum(XAjX[self.A_sp_rows[:j+1], self.A_sp_cols[:j+1]] * self.A_sp_data[:j+1], 1)
            
            lhs = np.zeros((self.dim, len(self.A_ds_idxs)))
            
            # Compute SPARSE x DENSE
            if len(self.A_sp_idxs) > 0 and len(self.A_ds_idxs) > 0:
                for (j, t) in enumerate(self.A_ds_idxs):
                    AjX  = self.As[t] @ X
                    XAjX = X.conj().T @ AjX
                    
                    out[self.A_sp_idxs, t] = np.sum(XAjX[self.A_sp_rows, self.A_sp_cols] * self.A_sp_data, 1)
                    out[t, self.A_sp_idxs] = out[self.A_sp_idxs, t]
                
                    lhs[:, j] = XAjX.flat
                    
                out[np.ix_(self.A_ds_idxs, self.A_ds_idxs)] = self.A_ds_mtx @ lhs
                
            # Compute DENSE x DENSE
            if len(self.A_ds_idxs) > 0 and len(self.A_sp_idxs) == 0:
                for (j, t) in enumerate(self.A_ds_idxs):
                    AjX       = self.As[j] @ X_rt2
                    XAjX      = X_rt2.conj().T @ AjX
                    lhs[:, j] = XAjX.flat   
                out = lhs.T @ lhs
                
            return out

        elif self.congr_mode == 1:
            # If constraint matrix has sparse aggregate structure
            # W_sub_((i,j), (k,l)) = W_(i,l)
            # WW_((i,j), (k,l)) = W_(i,l) * W_(j,k)
            # for all (i,j) in the aggregate sparsity pattern            
            X_sub = X[np.ix_(self.A_is, self.A_js)]
            XX = X_sub * X_sub.T
            return self.A_nz @ XX @ self.A_nz.T

    def comb_dir(self, out, dS, dZ, sigma_mu):
        # Compute the residual for rs where rs is given as the lhs of
        #     Lambda o (W dz + W^-T ds) = -Lambda o Lambda - (W^-T ds_a) o (W dz_a) 
        #                                 + sigma * mu * I
        # which is rearranged into the form H ds + dz = rs, i.e.,
        #     rs := W^-1 [ Lambda \ (-Lambda o Lambda - (W^-T ds_a) o (W dz_a) + sigma*mu I) ]
        # See: [Section 5.4]https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf

        if not self.nt_aux_updated:
            self.nt_aux()        

        # Compute (W^-T ds_a) o (W dz_a) = (R^-1 dS R^-T) o (R^T dZ R)
        #                                = 0.5 * ([R^-1 dS dZ R] + [R^T dZ dS R^-T])
        temp1 = self.R_inv @ dS
        temp2 = dZ @ self.R
        temp3 = temp1 @ temp2        
        np.add(temp3, temp3.T, out=temp1)
        temp1 *= -0.5

        # Compute -Lambda o Lambda - [ ... ] + sigma*mu I
        # Note that Lambda is a diagonal matrix
        temp1.flat[::self.n+1] -= np.square(self.Lambda)
        temp1.flat[::self.n+1] += sigma_mu

        # Compute Lambda \ [ ... ]
        # Since Lambda is diagonal, the solution to the Sylvester equation 
        #     find  X  s.t.  0.5 * (Lambda X + X Lambda) = B
        # is given by
        #     X = B .* (2 / [Lambda_ii + Lambda_jj]_ij)
        Gamma = np.add.outer(self.Lambda, self.Lambda)
        temp1 /= Gamma

        # Compute W^-1 [ ... ] = R^-T [... ] R^-1
        temp = self.R_inv.T @ temp1 @ self.R_inv
        np.add(temp, temp.T, out=out)

    def step_to_boundary(self, dS, dZ):
        # Compute the maximum step alpha in [0, 1] we can take such that 
        #     S + alpha dS >= 0
        #     Z + alpha dZ >= 0  
        # See: [Section 8.3]https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf

        if not self.nt_aux_updated:
            self.nt_aux()                
        Lambda_irt2 = np.reciprocal(np.sqrt(self.Lambda))

        # Compute rho := H(lambda)^1/2 W^-T dS 
        #              = Lambda^-1/2 R^-1 dS R^-T Lambda^-1/2
        rho = self.R_inv @ dS @ self.R_inv.T
        rho *= Lambda_irt2.reshape((-1, 1))
        rho *= Lambda_irt2.reshape(( 1,-1))

        # Compute sig := H(lambda)^1/2 W dS 
        #              = Lambda^-1/2 R^T dS R Lambda^-1/2
        sig = self.R.T @ dZ @ self.R
        sig *= Lambda_irt2.reshape((-1, 1))
        sig *= Lambda_irt2.reshape(( 1,-1))        

        # Compute minimum eigenvalues of rho and sig
        min_eig_rho = sp.linalg.lapack.dsyevr(rho, compute_v=False, range='I', iu=1)[0][0]
        min_eig_sig = sp.linalg.lapack.dsyevr(sig, compute_v=False, range='I', iu=1)[0][0]

        # Maximum step is given by 
        #     alpha := 1 / max(0, -min(eig(rho)), -min(eig(sig)))
        # Clamp this step between 0 and 1
        if min_eig_rho >= 0 and min_eig_sig >= 0:
            return 1.
        else:
            return 1. / max(-min_eig_rho, -min_eig_sig)
    
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