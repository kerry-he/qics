import numpy as np
import scipy as sp
import numba as nb
import itertools
from utils import symmetric as sym
from utils import linear as lin

import time

class Cone():
    def __init__(self, n, hermitian=False):
        # Dimension properties
        self.n  = n                                    # Side length of matrix
        self.hermitian = hermitian                     # Hermitian or symmetric vector space

        self.dim   = n * n      if (not hermitian) else 2 * n * n
        self.type  = ['s']      if (not hermitian) else ['h']
        self.dtype = np.float64 if (not hermitian) else np.complex128
        
        # Update flags
        self.feas_updated = False
        self.grad_updated = False
        self.nt_aux_updated = False
        self.congr_aux_updated = False

        # Get LAPACK operators
        self.X = np.eye(self.n, dtype=self.dtype)

        self.cho_fact  = sp.linalg.lapack.get_lapack_funcs("potrf", (self.X,))
        self.cho_inv   = sp.linalg.lapack.get_lapack_funcs("trtri", (self.X,))
        self.svd       = sp.linalg.lapack.get_lapack_funcs("gesdd", (self.X,))
        self.eigvalsh  = sp.linalg.lapack.get_lapack_funcs("heevr", (self.X,)) if self.hermitian else sp.linalg.lapack.get_lapack_funcs("syevr", (self.X,))
        self.svd_lwork = sp.linalg.lapack.get_lapack_funcs("gesdd_lwork", (self.X,))(n, n)

        return

    def get_nu(self):
        return self.n
    
    def set_init_point(self):
        self.set_point(
            np.eye(self.n, dtype=self.dtype), 
            np.eye(self.n, dtype=self.dtype)
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
        self.X_chol, info = self.cho_fact(self.X, lower=True)
        if info != 0:
            self.feas = False
            return self.feas
        
        self.Z_chol, info = self.cho_fact(self.Z, lower=True)
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
        
        self.X_chol_inv, _ = self.cho_inv(self.X_chol, lower=True)
        self.X_inv = self.X_chol_inv.conj().T @ self.X_chol_inv
        self.grad = -self.X_inv

        self.grad_updated = True
        return self.grad
    
    def hess_prod_ip(self, out, H):
        if not self.grad_updated:
            self.get_grad()
        XHX = self.X_inv @ H @ self.X_inv
        np.add(XHX, XHX.conj().T, out=out)
        out *= 0.5
        return out    
    
    def invhess_prod_ip(self, out, H):
        XHX = self.X @ H @ self.X
        np.add(XHX, XHX.conj().T, out=out)
        out *= 0.5
        return out
    
    def invhess_mtx(self):
        return lin.kron(self.X, self.X)
    
    def hess_mtx(self):
        return lin.kron(self.X_inv, self.X_inv)    

    def hess_congr(self, A):
        if not self.grad_updated:
            self.get_grad()
        return self.base_congr(A, self.X_inv, self.X_chol_inv)

    def invhess_congr(self, A):
        return self.base_congr(A, self.X, self.X_chol)    
    
    def congr_aux(self, A):
        assert not self.congr_aux_updated

        if sp.sparse.issparse(A):
            # Count the number of nonzeros in each row of A / for each Ai
            # and sort A by number of nonzero elements
            A_nnz = A.getnnz(1)

            # Split A into 3 groups: 1) sparse; 2) sparse-ish; 3) dense
            # TODO: Determine good thresholds
            A_1_idxs = np.where((A_nnz >  0)      & (A_nnz < self.n))[0]
            A_2_idxs = np.where((A_nnz >= self.n) & (A_nnz < self.n))[0]
            A_3_idxs = np.where((A_nnz >= self.n))[0]

            # Sort each of these by the number of nonzero entries
            self.A_1_idxs = A_1_idxs[np.argsort(A_nnz[A_1_idxs])]
            self.A_2_idxs = A_2_idxs[np.argsort(A_nnz[A_2_idxs])]
            self.A_3_idxs = A_3_idxs[np.argsort(A_nnz[A_3_idxs])]

            def lil_to_array(ragged):
                # Converts a list of lists (with possibly different lengths) 
                # into a numpy array padded with zeros
                return np.array(list(itertools.zip_longest(*ragged, fillvalue=0))).T
            
            def triu_idx_to_ij(idx):
                # Converts upper triangular indices to (i,j) coordinates
                #     [ 0  1  3       ]         [ (0,0)  (0,1)  (0,2)       ]
                #     [    2  4  ...  ]   -->   [        (1,1)  (1,2)  ...  ]
                #     [       5       ]         [               (2,2)       ]
                # https://stackoverflow.com/questions/40950460/how-to-convert-triangular-matrix-indexes-in-to-row-column-coordinates
                j = (np.ceil(np.sqrt(2 * (idx + 1) + 0.25) - 0.5) - 1).astype('int32')
                i = (idx - (j + 1) * j / 2).astype('int32')
                return i, j

            # Prepare things we need for Strategy 1:
            if len(self.A_1_idxs) > 0:

                A_1 = A[self.A_1_idxs]  

                triu_indices = np.array([j + i*self.n for j in range(self.n) for i in range(j + 1)])

                if self.hermitian:
                    # Create arrays where real and complex are combined
                    A_1_real = A_1[:, ::2][:, triu_indices]
                    A_1_imag = A_1[:, 1::2][:, triu_indices]
                    A_1_lil  = (A_1_real + A_1_imag*1j).tolil()

                    # Get number of nonzeros for each Ai (to account for ragged arrays)
                    self.A_1x_nnzs = A_1_lil.getnnz(1)

                    # Get rows and columns of nonzeros of Ai
                    rowcols = lil_to_array(A_1_lil.rows)
                    self.A_1x_rows, self.A_1x_cols = triu_idx_to_ij(rowcols)

                    # Get values of nonzeros of Ai, and scale off-diagonal elements to account
                    # for us only using upper triangular nonzeros
                    self.A_1x_data = lil_to_array(A_1_lil.data)
                    self.A_1x_data[self.A_1x_cols != self.A_1x_rows] *= 2

                    if len(self.A_2_idxs) > 0 or len(self.A_3_idxs) > 0:
                        # Create arrays where real and complex are split to do easy inner products with
                        A_1_real = A_1_real.tolil()
                        A_1_imag = A_1_imag.tolil()

                        # Get number of nonzeros for each Ai (to account for ragged arrays)
                        A_1_r_nnzs    = A_1_real.getnnz(1)
                        A_1_i_nnzs    = A_1_imag.getnnz(1)
                        self.A_1_nnzs = np.array(A_1_r_nnzs) + np.array(A_1_i_nnzs)

                         # Get rows and columns of nonzeros of Ai
                        rowcols = lil_to_array(A_1_real.rows + A_1_imag.rows)
                        self.A_1_rows, self.A_1_cols = triu_idx_to_ij(rowcols)

                        # Shift columns so that 
                        #     - Even column corresond to real entries
                        #     - Odd columns correpsond to imaginary entries
                        imag_ones     = [[0] * nnz_r + [1] * nnz_i for (nnz_r, nnz_i) in zip(A_1_r_nnzs, A_1_i_nnzs)]
                        imag_ones     = lil_to_array(imag_ones)
                        self.A_1_cols = self.A_1_cols * 2 + imag_ones
                        
                        # Get values of nonzeros of Ai, and scale off-diagonal elements to account
                        # for us only using upper triangular nonzeros
                        self.A_1_data = lil_to_array(A_1_real.data + A_1_imag.data)
                        self.A_1_data[self.A_1_cols != 2*self.A_1_rows] *= 2
                else:
                    A_1_lil = A_1[:, triu_indices].tolil()

                    # Get number of nonzeros for each Ai (to account for ragged arrays)
                    self.A_1_nnzs = A_1_lil.getnnz(1)

                    # Get rows and columns of nonzeros of Ai
                    rowcols = lil_to_array(A_1_lil.rows)
                    self.A_1_rows, self.A_1_cols = triu_idx_to_ij(rowcols)

                    # Get values of nonzeros of Ai, and scale off-diagonal elements to account
                    # for us only using upper triangular nonzeros
                    self.A_1_data = lil_to_array(A_1_lil.data)
                    self.A_1_data[self.A_1_cols != self.A_1_rows] *= 2

            # Prepare things we need for Strategy 2:
            if len(self.A_2_idxs) > 0:

                A_2 = A[self.A_2_idxs]

                # First turn rows of A into sparse CSR matrices Ai so we can do 
                # efficient matrix multiplications X Ai X
                if self.hermitian:
                    A_2_real = A_2[:, ::2]
                    A_2_imag = A_2[:, 1::2]
                    A_2 = A_2_real + A_2_imag*1j
                A_2_lil = A_2.tolil()
                A_2_data = [np.array(data_k)                      for data_k in A_2_lil.data]
                A_2_cols = [np.array(idxs_k, dtype=int)  % self.n for idxs_k in A_2_lil.rows]
                A_2_rows = [np.array(idxs_k, dtype=int) // self.n for idxs_k in A_2_lil.rows]
                self.Ai_2 = [sp.sparse.csr_array((data, (row, col)), shape=(self.n, self.n))
                                for (data, row, col) in zip(A_2_data, A_2_rows, A_2_cols)]

                # Now get indices of sparse matrices so we can do efficient inner product
                triu_indices = np.array([j + i*self.n for j in range(self.n) for i in range(j + 1)])

                if self.hermitian:
                    A_2_real = A_2_real[:, triu_indices].tolil()
                    A_2_imag = A_2_imag[:, triu_indices].tolil()

                    # Get number of nonzeros for each Ai (to account for ragged arrays)
                    A_2_r_nnzs    = A_2_real.getnnz(1)
                    A_2_i_nnzs    = A_2_imag.getnnz(1)                    
                    self.A_2_nnzs = np.array(A_2_r_nnzs) + np.array(A_2_i_nnzs)

                    # Get rows and columns of nonzeros of Ai
                    rowcols = lil_to_array(A_2_real.rows + A_2_imag.rows)
                    self.A_2_rows, self.A_2_cols = triu_idx_to_ij(rowcols)

                    # Shift columns so that 
                    #     - Even column corresond to real entries
                    #     - Odd columns correpsond to imaginary entries
                    imag_ones     = [[0] * nnz_r + [1] * nnz_i for (nnz_r, nnz_i) in zip(A_2_r_nnzs, A_2_i_nnzs)]
                    imag_ones     = lil_to_array(imag_ones)
                    self.A_2_cols = self.A_2_cols * 2 + imag_ones
                    
                    # Get values of nonzeros of Ai, and scale off-diagonal elements to account
                    # for us only using upper triangular nonzeros
                    self.A_2_data = lil_to_array(A_2_real.data + A_2_imag.data)
                    self.A_2_data[self.A_2_cols != 2*self.A_2_rows] *= 2
                else:
                    A_2_lil = A_2[:, triu_indices].tolil()

                    # Get number of nonzeros for each Ai (to account for ragged arrays)
                    self.A_2_nnzs = A_2_lil.getnnz(1)

                    # Get rows and columns of nonzeros of Ai
                    rowcols = lil_to_array(A_2_lil.rows)
                    self.A_2_rows, self.A_2_cols = triu_idx_to_ij(rowcols)

                    # Get values of nonzeros of Ai, and scale off-diagonal elements to account
                    # for us only using upper triangular nonzeros
                    self.A_2_data = lil_to_array(A_2_lil.data)
                    self.A_2_data[self.A_2_cols != self.A_2_rows] *= 2

            # Prepare things we need for Strategy 3:
            if len(self.A_3_idxs) > 0:
                self.AHA_3_3_idxs = np.ix_(self.A_3_idxs, self.A_3_idxs)

                # Turn rows of A into dense matrices Ai
                self.A_3 = A[self.A_3_idxs].toarray()
                if self.hermitian:
                    self.Ai_3 = [Ai.reshape((-1, 2)).view(dtype=np.complex128).reshape(self.n, self.n) for Ai in self.A_3]
                else:
                    self.Ai_3 = [Ai.reshape((self.n, self.n)) for Ai in self.A_3]                

        else:
            # A and all Ai are dense matrices
            # Just need to convert the rows of A into dense matrices
            self.A_1_idxs = np.array([])
            self.A_2_idxs = np.array([])
            self.A_3_idxs = np.arange(A.shape[0])
            if self.hermitian:
                self.Ai_3 = [Ai.reshape((-1, 2)).view(dtype=np.complex128).reshape(self.n, self.n) for Ai in A]
            else:
                self.Ai_3 = [Ai.reshape((self.n, self.n)) for Ai in A]
            
        self.congr_aux_updated = True
    
    @profile
    def base_congr(self, A, X, X_rt2):
        if not self.congr_aux_updated:
            self.congr_aux(A)

        p = A.shape[0]
        out = np.zeros((p, p))

        # Compute sparse-sparse component
        if len(self.A_1_idxs) > 0:
            # Use fast Numba compiled functions when A is extremely sparse 
            if self.hermitian:
                AHA_complex(out, self.A_1x_rows, self.A_1x_cols, self.A_1x_data, self.A_1x_nnzs, X, self.A_1_idxs)
            else:
                AHA(out, self.A_1_rows, self.A_1_cols, self.A_1_data, self.A_1_nnzs, X, self.A_1_idxs)

        if len(self.A_2_idxs) > 0:
            for (j, t) in enumerate(self.A_2_idxs):
                # Compute X Aj X for sparse Aj
                ts = self.A_2_idxs[:j+1]
                AjX  = self.Ai_2[j].dot(X)
                XAjX = X @ AjX
                XAjX = XAjX.view(dtype=np.float64)

                # Efficient inner product <Ai, X Aj X> for sparse Ai
                out[ts, t] = np.sum(XAjX[self.A_2_rows[:j+1], self.A_2_cols[:j+1]] * self.A_2_data[:j+1], 1)
                out[t, ts] = out[ts, t]

                if len(self.A_1_idxs) > 0:
                    out[self.A_1_idxs, t] = np.sum(XAjX[self.A_1_rows, self.A_1_cols] * self.A_1_data, 1)
                    out[t, self.A_1_idxs] = out[self.A_1_idxs, t]          

        if (len(self.A_1_idxs) > 0 or len(self.A_2_idxs) > 0) and len(self.A_3_idxs) > 0:
            lhs = np.zeros((len(self.A_3_idxs), self.dim))

            for (j, t) in enumerate(self.A_3_idxs):
                # Compute X Aj X for sparse Aj
                AjX  = self.Ai_3[j].dot(X)
                XAjX = X @ AjX
                XAjX = XAjX.view(dtype=np.float64)
                lhs[j] = XAjX.reshape(-1)

                if len(self.A_1_idxs) > 0:
                    out[self.A_1_idxs, t] = np.sum(XAjX[self.A_1_rows, self.A_1_cols] * self.A_1_data, 1)
                    out[t, self.A_1_idxs] = out[self.A_1_idxs, t]

                if len(self.A_2_idxs) > 0:
                    out[self.A_2_idxs, t] = np.sum(XAjX[self.A_2_rows, self.A_2_cols] * self.A_2_data, 1)
                    out[t, self.A_2_idxs] = out[self.A_2_idxs, t]

            out[self.AHA_3_3_idxs] = self.A_3 @ lhs.T
            
        # If all of Ai are dense, then it is slightly faster to use a strategy using
        # symmetric matrix multiplication, i.e., for Cholesky factor X = LL'
        #     AHA = A (X kr X) A' 
        #         = A (L kr L) (L' kr L') A' 
        #         = [A (L kr L)] [A (L kr L)]'
        # where 
        #     [A (L kr L)]_j = vec(L' Aj L)
        if len(self.A_1_idxs) == 0 and len(self.A_2_idxs) == 0 and len(self.A_3_idxs) > 0:
            lhs = np.zeros((len(self.A_3_idxs), self.dim))

            for j in range(len(self.A_3_idxs)):
                # Compute L Aj L' for dense Aj
                AjX    = self.Ai_3[j] @ X_rt2
                XAjX   = X_rt2.conj().T @ AjX
                lhs[j] = XAjX.view(dtype=np.float64).reshape(-1)
            # Compute symmetric matrix multiplication [A (L kr L)] [A (L kr L)]'
            return lhs @ lhs.conj().T
            
        return out
    
    def third_dir_deriv(self, dir1, dir2=None):
        if not self.grad_updated:
            self.get_grad()
        if dir2 is None:
            XHX_2 = self.X_inv @ dir1 @ self.X_chol_inv.conj().T
            return -2 * XHX_2 @ XHX_2.conj().T
        else:
            PD = dir1 @ dir2
            XiPD = self.X_inv @ PD
            return -XiPD - XiPD.conj().T
    
    def prox(self):
        assert self.feas_updated
        XZX_I = self.X_chol.conj().T @ self.Z @ self.X_chol
        XZX_I.flat[::self.n+1] -= 1
        return np.linalg.norm(XZX_I) ** 2    

    # ========================================================================
    # Functions specific to symmetric cones for NT scaling
    # ========================================================================
    # Computes the NT scaling point W and scaled variable Lambda such that
    #     H(W)[S] = Z  <==> Lambda := P^-T(S) = P(Z)
    # where H(W) = V^T V. To obtain for for the PSD cone, first let compute the SVD
    #     U D V^T = Z_chol^T S_chol
    # Then compute 
    #     R    := S_chol V D^-1/2     = Z_chol^-T U D^1/2
    #     R^-1 := D^1/2 V^T S_chol^-1 = D^-1/2 U^T Z_chol^T
    # Then we can find the scaling point as
    #     W    := R R^T
    #           = S^1/2 (S^1/2 Z S^1/2)^-1/2 S^1/2 
    #           = Z^-1/2 (Z^1/2 S Z^1/2)^1/2 Z^1/2 (i.e., geometric mean of Z and S)
    #     W^-1 := R^-T R^-1
    # and the scaled point as
    #     Lambda := D
    # Also, we have the linear transformations given by
    #     H(W)[S] = W^-1 S W^-1
    #     P^-T(S) = R^-1 S R^-T
    #     P(Z)    = R^T Z R
    # See: [Section 4.3]https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf

    def nt_aux(self):
        assert not self.nt_aux_updated
        # Take the SVD of Z_chol^T S_chol to get scaled point Lambda := D
        RL = self.Z_chol.conj().T @ self.X_chol
        _, self.Lambda, Vt, _ = self.svd(RL, lwork=self.svd_lwork)
        D_rt2 = np.sqrt(self.Lambda)

        # Compute the scaling point as
        #    R := S_chol V D^-1/2, and
        #    W := R R^T
        self.R = self.X_chol @ (Vt.conj().T / D_rt2)
        self.W = self.R @ self.R.conj().T

        # Compute the inverse scaling point as
        #     R^-1 := D^1/2 V^T S_chol^-1, and
        #     W^-1 := R^-T R^-1
        self.X_chol_inv, _ = self.cho_inv(self.X_chol, lower=True)        
        self.R_inv = (self.X_chol_inv.conj().T @ (Vt.conj().T * D_rt2)).conj().T
        self.W_inv = self.R_inv.conj().T @ self.R_inv

        self.nt_aux_updated = True

    def nt_prod_ip(self, out, H):
        if not self.nt_aux_updated:
            self.nt_aux()
        WHW = self.W_inv @ H @ self.W_inv
        np.add(WHW, WHW.conj().T, out=out)
        out *= 0.5
    
    def invnt_prod_ip(self, out, H):
        if not self.nt_aux_updated:
            self.nt_aux()
        WHW = self.W @ H @ self.W
        np.add(WHW, WHW.conj().T, out=out)
        out *= 0.5    

    def invnt_mtx(self):
        if not self.nt_aux_updated:
            self.nt_aux()        
        return lin.kron(self.W, self.W)
    
    def nt_mtx(self):
        if not self.nt_aux_updated:
            self.nt_aux()        
        return lin.kron(self.W_inv, self.W_inv)
    
    def nt_congr(self, A):
        if not self.nt_aux_updated:
            self.nt_aux()
        return self.base_congr(A, self.W_inv, self.R_inv.T)

    def invnt_congr(self, A):
        if not self.nt_aux_updated:
            self.nt_aux()
        return self.base_congr(A, self.W, self.R)

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
        np.add(temp3, temp3.conj().T, out=temp1)
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
        temp = self.R_inv.conj().T @ temp1 @ self.R_inv
        np.add(temp, temp.conj().T, out=out)

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
        rho  = self.R_inv @ dS @ self.R_inv.conj().T
        rho *= Lambda_irt2.reshape((-1, 1))
        rho *= Lambda_irt2.reshape(( 1,-1))

        # Compute sig := H(lambda)^1/2 W dS 
        #              = Lambda^-1/2 R^T dS R Lambda^-1/2
        sig  = self.R.conj().T @ dZ @ self.R
        sig *= Lambda_irt2.reshape((-1, 1))
        sig *= Lambda_irt2.reshape(( 1,-1))

        # Compute minimum eigenvalues of rho and sig
        min_eig_rho = self.eigvalsh(rho, compute_v=False, range='I', iu=1)[0][0]
        min_eig_sig = self.eigvalsh(sig, compute_v=False, range='I', iu=1)[0][0]

        # Maximum step is given by 
        #     alpha := 1 / max(0, -min(eig(rho)), -min(eig(sig)))
        # Clamp this step between 0 and 1
        if min_eig_rho >= 0 and min_eig_sig >= 0:
            return 1.
        else:
            return 1. / max(-min_eig_rho, -min_eig_sig)
        

# ============================================================================
# Numba functions for computing Schur complement matrix when A is very sparse
# ============================================================================
import numba as nb

@nb.njit(parallel=True, fastmath=True)
def AHA(
        out,
        A_rows,
        A_cols,
        A_vals,
        A_nnz,
        X,
        indices,
    ):
    # Computes the congruence transform A (X kron X) A' when A is very sparse
    # See https://link.springer.com/article/10.1007/BF02614319

    # We can cut the amount of operations in half by exploiting symetry of A and X as follows
    # (AHA)_ij = SUM_a,b (Ai)_ab (SUM_c,d (Aj)_cd X_ac X_db) 
    #          = SUM_a,b (Ai)_ab (  [SUM_c=d (Aj)_cd X_ac X_db] 
    #                             + [SUM_c<d (Aj)_cd X_ac X_db] 
    #                             + [SUM_c>d (Aj)_cd X_ac X_db]  ) 
    #          = SUM_a,b (Ai)_ab (  [SUM_c=d (Aj)_cd X_ac X_db] 
    #                             + [SUM_c<d (Aj)_cd (X_ac X_db + X_ad X_cb)]  )
    #          = [SUM_a=b (Ai)_ab ( ... )] + [SUM_a<b (Ai)_ab ( ... )] + [SUM_a>b (Ai)_ab ( ... )]
    #          = [SUM_a=b (Ai)_ab ( ... )] + 2 [SUM_a<b (Ai)_ab ( ... )]
    # Note that we assume off-diagonal entries of Ai have been scaled by 2
    # Also note that we assume only upper triangular elements are given to us so c < d

    p = A_rows.shape[0]

    # Loop through upper triangular entries of the Schur complement matrix (AHA)_ij
    for j in nb.prange(p):
        for i in nb.prange(j + 1):
            I = indices[i]
            J = indices[j]

            tmp1 = 0.

            # Loop over nonzero entries of Ai
            for alpha in range(A_nnz[i]):
                a = A_rows[i, alpha]
                b = A_cols[i, alpha]

                tmp2 = 0.
                tmp3 = 0.

                # Loop over nonzero entries of Aj
                for beta in range(A_nnz[j]):
                    c = A_rows[j, beta]
                    d = A_cols[j, beta]

                    if c < d:
                        tmp2 += A_vals[j, beta] * (X[a, c] * X[d, b] + X[a, d] * X[c, b])
                    else:
                        tmp3 += A_vals[j, beta] * X[a, c] * X[d, b]

                tmp1 += A_vals[i, alpha] * (0.5 * tmp2 + tmp3)
                    
            if I <= J:
                out[I, J] = tmp1
            else:
                out[J, I] = tmp1

@nb.njit(parallel=True, fastmath=True)
def AHA_complex(
        out,
        A_rows,
        A_cols,
        A_vals,
        A_nnz,
        X,
        indices,
    ):
    # Computes the congruence transform A (X kron X) A' when A is very sparse
    # See https://link.springer.com/article/10.1007/BF02614319

    # We can cut the amount of operations in half by exploiting symetry of A and X as follows
    # (AHA)_ij = SUM_a,b (Ai)_ab* (SUM_c,d (Aj)_cd X_ac X_db) 
    #          = SUM_a,b (Ai)_ab* (  [SUM_c=d (Aj)_cd X_ac X_db] 
    #                              + [SUM_c>d (Aj)_cd X_ac X_db] 
    #                              + [SUM_c<d (Aj)_cd X_ac X_db]  ) 
    #          = SUM_a,b (Ai)_ab* (  [SUM_c=d (Aj)_cd X_ac X_db] 
    #                              + [SUM_c<d (Aj)_cd X_ac X_db + (Aj)_cd* X_ad X_cb]  )
    #          = [SUM_a=b (Ai)_ab ( ... )] + [SUM_a<b (Ai)_ab ( ... )] + [SUM_a>b (Ai)_ab ( ... )]
    #          = [SUM_a=b (Ai)_ab ( ... )] + [SUM_a>b (Ai)_ab ( ... ) + (Ai)_ab* ( ... )]
    # Also note that off-diagonal entries of Ai have been scaled by 2    

    p = A_rows.shape[0]

    # Loop through each entry of the Schur complement matrix (AHA)_ij
    for j in nb.prange(p):
        for i in nb.prange(j + 1):
            I = indices[i]
            J = indices[j]

            tmp1 = 0.
            for alpha in range(A_nnz[i]):
                a = A_rows[i, alpha]
                b = A_cols[i, alpha]

                tmp2 = 0.
                tmp3 = 0.
                for beta in range(A_nnz[j]):
                    c = A_rows[j, beta]
                    d = A_cols[j, beta]

                    if c < d:
                        tmp2 +=         A_vals[j, beta]  * X[a, c] * X[d, b]
                        tmp2 += np.conj(A_vals[j, beta]) * X[a, d] * X[c, b]
                    else:
                        tmp3 += A_vals[j, beta].real * X[a, c] * X[d, b]

                # Do addition slightly differently to guarantee a real number
                # i.e., just take the inner product between Ai and X Aj X by
                #     SUM_ab Re[(X Aj X)_ab] * Re[(Ai)_ab] + 2 Im[(X Aj X)_ab] * Im[(Ai)_ab]
                tmp3 += 0.5 * tmp2
                tmp1 += A_vals[i, alpha].real * tmp3.real
                tmp1 += A_vals[i, alpha].imag * tmp3.imag
                    
            if I <= J:
                out[I, J] = tmp1
            else:
                out[J, I] = tmp1