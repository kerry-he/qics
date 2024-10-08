import numpy as np
import scipy as sp
import numba as nb
import itertools

import qics._utils.linalg as lin
from qics.cones.base import SymCone


class PosSemidefinite(SymCone):
    r"""A class representing a positive semidefinite cone defined either
    on real symmetric matrices

    .. math::

        \mathbb{S}^n_+ = \{ X \in \mathbb{S}^n : X \succeq 0 \},

    or complex Hermitian matrices

    .. math::

        \mathbb{H}^n_+ = \{ X \in \mathbb{H}^n : X \succeq 0 \}.

    Parameters
    ----------
    n : :obj:`int`
        Dimension of the matrix :math:`X`.
    iscomplex : :obj:`bool`
        Whether the matrix :math:`X` is defined over :math:`\mathbb{H}^n`
        (``True``), or restricted to :math:`\mathbb{S}^n` (``False``). The
        default is ``False``.
    """

    def __init__(self, n, iscomplex=False):
        self.n = n
        self.iscomplex = iscomplex

        self.nu = n  # Barrier parameter

        if iscomplex:
            self.dim = [2 * n * n]
            self.type = ["h"]
            self.dtype = np.complex128
        else:
            self.dim = [n * n]
            self.type = ["s"]
            self.dtype = np.float64

        # Update flags
        self.feas_updated = False
        self.grad_updated = False
        self.nt_aux_updated = False
        self.congr_aux_updated = False

        # Get LAPACK operators
        from scipy.linalg.lapack import get_lapack_funcs

        X = np.eye(self.n, dtype=self.dtype)
        self.cho_fact = get_lapack_funcs("potrf", (X,))
        self.cho_inv = get_lapack_funcs("trtri", (X,))
        self.svd = get_lapack_funcs("gesdd", (X,))
        self.svd_lwork = get_lapack_funcs("gesdd_lwork", (X,))(n, n)
        if iscomplex:
            self.eigvalsh = get_lapack_funcs("heevr", (X,))
        else:
            self.eigvalsh = get_lapack_funcs("syevr", (X,))

        return

    def get_iscomplex(self):
        return self.iscomplex

    def get_init_point(self, out):
        point = [np.eye(self.n, dtype=self.dtype)]

        self.set_point(point, point)

        out[0][:] = self.X
        return out

    def set_point(self, primal, dual=None, a=True):
        self.X = primal[0] * a
        self.Z = dual[0] * a if (dual is not None) else None

        self.feas_updated = False
        self.grad_updated = False
        self.nt_aux_updated = False

    def set_dual(self, dual, a=True):
        self.Z = dual[0] * a

    def get_feas(self):
        if self.feas_updated:
            return self.feas

        self.feas_updated = True

        # Check that X is PSD by trying to Cholesky factor it
        self.X_chol, info = self.cho_fact(self.X, lower=True)
        if info != 0:
            self.feas = False
            return self.feas

        # Check that Z is PSD by trying to Cholesky factor it
        if self.Z is not None:
            self.Z_chol, info = self.cho_fact(self.Z, lower=True)
            if info != 0:
                self.feas = False
                return self.feas

        self.feas = True
        return self.feas
    
    def get_dual_feas(self):
        self.Z_chol, info = self.cho_fact(self.Z, lower=True)
        return info == 0

    def get_val(self):
        (sign, logabsdet) = np.linalg.slogdet(self.X)
        return -sign * logabsdet

    def update_grad(self):
        assert not self.grad_updated

        self.X_chol_inv, _ = self.cho_inv(self.X_chol, lower=True)
        self.X_inv = self.X_chol_inv.conj().T @ self.X_chol_inv
        self.grad = [-self.X_inv]

        self.grad_updated = True

    def hess_prod_ip(self, out, H):
        if not self.grad_updated:
            self.update_grad()
        XHX = self.X_inv @ H[0] @ self.X_inv
        np.add(XHX, XHX.conj().T, out=out[0])
        out[0] *= 0.5
        return out

    def hess_congr(self, A):
        if not self.grad_updated:
            self.update_grad()
        return self.base_congr(A, self.X_inv, self.X_chol_inv.conj().T)

    def invhess_prod_ip(self, out, H):
        XHX = self.X @ H[0] @ self.X
        np.add(XHX, XHX.conj().T, out=out[0])
        out[0] *= 0.5
        return out

    def invhess_congr(self, A):
        return self.base_congr(A, self.X, self.X_chol)

    def base_congr(self, A, X, X_rt2):
        # Generalized function to compute the matrix [<Ai, X Aj X>]_ij for a
        # a given matrix X
        if not self.congr_aux_updated:
            self.congr_aux(A)

        (n, p, p_ds) = (self.n, A.shape[0], len(self.A_ds_idxs))
        out = np.zeros((p, p))

        if len(self.A_sp_idxs) > 0:
            # Compute sparse-sparse component using Numba compiled functions
            if self.iscomplex:
                _sparse_congr_complex(out, self.A_sp_rows, self.A_sp_cols, 
                    self.A_sp_data, self.A_sp_nnzs, X, self.A_sp_idxs)
            else:
                _sparse_congr(out, self.A_sp_rows, self.A_sp_cols, 
                    self.A_sp_data, self.A_sp_nnzs, X, self.A_sp_idxs)

            # Compute sparse-dense and dense-dense components
            if p_ds > 0:
                work = self.work
                
                # Compute X Aj X for all dense Aj
                lhs = np.zeros((p_ds, self.dim[0]))
                lhs_view = lhs.reshape((p_ds, n, -1)).view(self.dtype)
                lin.congr_multi(lhs_view, X, self.Ai_ds, work=work)

                # Compute inner products <Ai, X Aj X> for all dense Aj
                temp = lin.x_dot_dense(self.A_triu, lhs[:, self.triu_idxs].T)
                out[:, self.A_ds_idxs] = temp
                out[self.A_ds_idxs, :] = temp.T
        else:
            # If there are no (nonzero) sparse Aj, then compute congruence using
            # the symmetric multiplication [A (L' kr L)] [A (L' kr L)]'
            work = self.work

            # Compute L' Aj L
            lhs = np.zeros((p_ds, self.dim[0]))
            lhs_view = lhs.reshape((p_ds, n, -1)).view(self.dtype)
            lin.congr_multi(lhs_view, X_rt2.conj().T, self.Ai_ds, work=work)

            # Compute inner products <L' Ai L, L' Aj L>
            out[self.A_ds_ds_idxs] = lhs @ lhs.T

        return out

    def third_dir_deriv_axpy(self, out, H, a=True):
        if not self.grad_updated:
            self.update_grad()

        XHX_2 = self.X_inv @ H[0] @ self.X_chol_inv.conj().T
        out[0] -= 2 * a * XHX_2 @ XHX_2.conj().T
        return out

    def prox(self):
        assert self.feas_updated
        XZX_I = self.X_chol.conj().T @ self.Z @ self.X_chol
        XZX_I.flat[:: self.n + 1] -= 1
        return np.linalg.norm(XZX_I) ** 2

    # ==========================================================================
    # Functions specific to symmetric cones for NT scaling
    # ==========================================================================
    # Computes the NT scaling point W and scaled variable Lambda such that
    #     H(W)[S] = Z  <==> Lambda := P^-T(S) = P(Z)
    # where H(W) = V^T V. To obtain for for the PSD cone, first let compute the
    # SVD
    #     U D V^T = Z_chol^T S_chol
    # Then compute
    #     R    := S_chol V D^-1/2     = Z_chol^-T U D^1/2
    #     R^-1 := D^1/2 V^T S_chol^-1 = D^-1/2 U^T Z_chol^T
    # Then we can find the scaling point as
    #     W    := R R^T
    #           = S^1/2 (S^1/2 Z S^1/2)^-1/2 S^1/2
    #           = Z^-1/2 (Z^1/2 S Z^1/2)^1/2 Z^1/2 (i.e., geo. mean of Z and S)
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
        #    R^-1 := D^1/2 V^T S_chol^-1, and
        #    W^-1 := R^-T R^-1
        self.X_chol_inv, _ = self.cho_inv(self.X_chol, lower=True)
        self.R_inv = (self.X_chol_inv.conj().T @ (Vt.conj().T * D_rt2)).conj().T
        self.W_inv = self.R_inv.conj().T @ self.R_inv

        self.nt_aux_updated = True

    def nt_prod_ip(self, out, H):
        if not self.nt_aux_updated:
            self.nt_aux()
        WHW = self.W_inv @ H[0] @ self.W_inv
        np.add(WHW, WHW.conj().T, out=out[0])
        out[0] *= 0.5

    def nt_congr(self, A):
        if not self.nt_aux_updated:
            self.nt_aux()
        return self.base_congr(A, self.W_inv, self.R_inv.conj().T)

    def invnt_prod_ip(self, out, H):
        if not self.nt_aux_updated:
            self.nt_aux()
        WHW = self.W @ H[0] @ self.W
        np.add(WHW, WHW.conj().T, out=out[0])
        out[0] *= 0.5

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
        #               = 0.5 * ([R^-1 dS dZ R] + [R^T dZ dS R^-T])
        temp1 = self.R_inv @ dS[0]
        temp2 = dZ[0] @ self.R
        temp3 = temp1 @ temp2
        np.add(temp3, temp3.conj().T, out=temp1)
        temp1 *= -0.5

        # Compute -Lambda o Lambda - [ ... ] + sigma*mu I
        # Note that Lambda is a diagonal matrix
        temp1.flat[:: self.n + 1] -= np.square(self.Lambda)
        temp1.flat[:: self.n + 1] += sigma_mu

        # Compute Lambda \ [ ... ]
        # Since Lambda is diagonal, the solution to the Sylvester equation
        #     find  X  s.t.  0.5 * (Lambda X + X Lambda) = B
        # is given by
        #     X = B .* (2 / [Lambda_ii + Lambda_jj]_ij)
        Gamma = np.add.outer(self.Lambda, self.Lambda)
        temp1 /= Gamma

        # Compute W^-1 [ ... ] = R^-T [... ] R^-1
        temp = self.R_inv.conj().T @ temp1 @ self.R_inv
        np.add(temp, temp.conj().T, out=out[0])

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
        rho = self.R_inv @ dS @ self.R_inv.conj().T
        rho *= Lambda_irt2.reshape((-1, 1))
        rho *= Lambda_irt2.reshape((1, -1))

        # Compute sig := H(lambda)^1/2 W dS
        #              = Lambda^-1/2 R^T dS R Lambda^-1/2
        sig = self.R.conj().T @ dZ @ self.R
        sig *= Lambda_irt2.reshape((-1, 1))
        sig *= Lambda_irt2.reshape((1, -1))

        # Compute minimum eigenvalues of rho and sig
        min_eig_rho = self.eigvalsh(rho, compute_v=False, range="I", iu=1)[0][0]
        min_eig_sig = self.eigvalsh(sig, compute_v=False, range="I", iu=1)[0][0]

        # Maximum step is given by
        #     alpha := 1 / max(0, -min(eig(rho)), -min(eig(sig)))
        # Clamp this step between 0 and 1
        if min_eig_rho >= 0 and min_eig_sig >= 0:
            return 1.0
        else:
            return 1.0 / max(-min_eig_rho, -min_eig_sig)

    # ==========================================================================
    # Auxilliary functions
    # ==========================================================================
    def congr_aux(self, A):
        assert not self.congr_aux_updated

        n = self.n

        if sp.sparse.issparse(A):
            A = A.tocsr()

            # Split A into sparse and dense groups
            A_nnz = A.getnnz(1)
            self.A_sp_idxs = np.where((A_nnz > 0) & (A_nnz < n))[0]
            self.A_ds_idxs = np.where((A_nnz >= n))[0]

            self.A_sp_idxs = self.A_sp_idxs[np.argsort(A_nnz[self.A_sp_idxs])]
            self.A_ds_ds_idxs = np.ix_(self.A_ds_idxs, self.A_ds_idxs)

            # Prepare things we need for Strategy 1, i.e., sparse-sparse 
            # congruence using Numba functions
            if len(self.A_sp_idxs) > 0:
                A_sp = A[self.A_sp_idxs]

                triu_idxs = _get_triu_idxs(n)

                if self.iscomplex:
                    A_sp_real = A_sp[:, ::2][:, triu_idxs]
                    A_sp_imag = A_sp[:, 1::2][:, triu_idxs]
                    A_sp_lil = (A_sp_real + A_sp_imag * 1j).tolil()
                else:
                    A_sp_lil = A_sp[:, triu_idxs].tolil()

                # Get number of nonzeros for each sparse Ai (to account for
                # ragged arrays)
                self.A_sp_nnzs = A_sp_lil.getnnz(1)

                # Get rows and columns of nonzeros of sparse Ai
                rowcols = _lil_to_array(A_sp_lil.rows)
                self.A_sp_rows, self.A_sp_cols = _triu_idx_to_ij(rowcols)

                # Get values of nonzeros of sparse Ai, and scale off-diagonal 
                # elements to account for only using upper triangular nonzeros
                self.A_sp_data = _lil_to_array(A_sp_lil.data)
                self.A_sp_data[self.A_sp_cols != self.A_sp_rows] *= 2

            # Prepare things we need for Strategy 2, i.e., sparse-dense and
            # dense-dense congruence by computing X Aj X for all dense Aj, then
            # <Ai, X Aj X> for all Ai and dense Aj
            if len(self.A_ds_idxs) > 0:
                A_ds = A[self.A_ds_idxs, :]

                # Turn rows of A into matrices Ai
                if self.iscomplex:
                    A_ds = A_ds[:, ::2] + A_ds[:, 1::2] * 1j
                A_ds = A_ds.toarray()

                self.Ai_ds = np.array([Ai.reshape((n, n)) for Ai in A_ds])
                self.work = np.zeros_like(self.Ai_ds)

                # Get upper triangular slices of A so we can more efficiently
                # do the inner product <Ai, X Aj X> as a matrix multiplication
                if len(self.A_sp_idxs) > 0:
                    # Scale upper triangular elements by 2 to account for only
                    # using upper triangular elements
                    scale = 2 * np.ones(self.dim[0])
                    if self.iscomplex:
                        scale[:: 2 * n + 2] = 1
                    else:
                        scale[:: n + 1] = 1
                    self.triu_idxs = _get_triu_idxs(n, self.iscomplex)
                    self.A_triu = lin.scale_axis(A, scale_cols=scale).tocsr()
                    self.A_triu = self.A_triu[:, self.triu_idxs].tocoo()

        else:
            # A and all Ai are dense matrices
            # Just need to convert the rows of A into dense matrices
            from qics.vectorize import vec_to_mat
            A = np.ascontiguousarray(A)

            self.A_sp_idxs = np.array([])
            self.A_ds_idxs = np.arange(A.shape[0])
            self.A_ds_ds_idxs = np.ix_(self.A_ds_idxs, self.A_ds_idxs)

            self.Ai_ds = np.array([vec_to_mat(Ak, self.iscomplex) for Ak in A])    
            self.work = np.zeros_like(self.Ai_ds)

        self.congr_aux_updated = True


def _lil_to_array(ragged):
    # Converts a list of lists (with possibly different lengths) into a numpy
    # array padded with zeros
    padded_list = list(itertools.zip_longest(*ragged, fillvalue=0))
    return np.array(padded_list).T


def _triu_idx_to_ij(idx):
    # Converts upper triangular indices to (i,j) coordinates
    #     [ 0  1  3       ]         [ (0,0)  (0,1)  (0,2)       ]
    #     [    2  4  ...  ]   -->   [        (1,1)  (1,2)  ...  ]
    #     [       5       ]         [               (2,2)       ]
    # See: https://stackoverflow.com/questions/40950460/
    j = np.ceil(np.sqrt(2 * (idx + 1) + 0.25) - 0.5) - 1
    i = idx - (j + 1) * j / 2
    return i.astype("int32"), j.astype("int32")


def _get_triu_idxs(n, iscomplex=False):
    # Gets indices of a vectorized matrix corresponding to the upper triangular
    # elements of the matrix
    if iscomplex:
        diag = [2 * (i + i * n) for i in range(n)]
        triu_real = [2 * (j + i * n) for j in range(n) for i in range(j)]
        triu_imag = [2 * (j + i * n) + 1 for j in range(n) for i in range(j)]
        return np.array(diag + triu_real + triu_imag)
    else:
        return np.array([j + i * n for j in range(n) for i in range(j + 1)])    


# ============================================================================
# Numba functions for computing Schur complement matrix when A is very sparse
# ============================================================================

@nb.njit(parallel=True, fastmath=True)
def _sparse_congr(out, A_rows, A_cols, A_vals, A_nnz, X, indices):
    # Computes the congruence transform A (X kron X) A' when A is very sparse
    # See https://link.springer.com/article/10.1007/BF02614319

    # We can cut the amount of operations in half by exploiting symetry of A and
    # X as follows
    # (AHA)_ij = Σ_a,b (Ai)_ab (Σ_c,d (Aj)_cd X_ac X_db)
    #          = Σ_a,b (Ai)_ab (  [Σ_c=d (Aj)_cd X_ac X_db]
    #                             + [Σ_c<d (Aj)_cd X_ac X_db]
    #                             + [Σ_c>d (Aj)_cd X_ac X_db]  )
    #          = Σ_a,b (Ai)_ab (  [Σ_c=d (Aj)_cd X_ac X_db]
    #                             + [Σ_c<d (Aj)_cd (X_ac X_db + X_ad X_cb)]  )
    #          = [Σ_a=b (Ai)_ab ( ... )] + [Σ_a<b (Ai)_ab ( ... )] + [Σ_a>b (Ai)_ab ( ... )]
    #          = [Σ_a=b (Ai)_ab ( ... )] + 2 [Σ_a<b (Ai)_ab ( ... )]
    # Note that we assume off-diagonal entries of Ai have been scaled by 2.
    # Also note that we assume only upper triangular elements are given to us so
    # c < d.

    n = A_rows.shape[0]

    # Loop through upper triangular entries of the Schur complement matrix (AHA)_ij
    for j in nb.prange(n):
        for i in nb.prange(j + 1):
            i_AHA = indices[i]
            j_AHA = indices[j]

            tmp1 = 0.0

            # Loop over nonzero entries of Ai
            for alpha in range(A_nnz[i]):
                a = A_rows[i, alpha]
                b = A_cols[i, alpha]

                tmp2 = 0.0
                tmp3 = 0.0

                # Loop over nonzero entries of Aj
                for beta in range(A_nnz[j]):
                    c = A_rows[j, beta]
                    d = A_cols[j, beta]

                    if c < d:
                        tmp2 += A_vals[j, beta] * (
                            X[a, c] * X[d, b] + X[a, d] * X[c, b]
                        )
                    else:
                        tmp3 += A_vals[j, beta] * X[a, c] * X[d, b]

                tmp1 += A_vals[i, alpha] * (0.5 * tmp2 + tmp3)

            if i_AHA <= j_AHA:
                out[i_AHA, j_AHA] = tmp1
            else:
                out[j_AHA, i_AHA] = tmp1


@nb.njit(parallel=True, fastmath=True)
def _sparse_congr_complex(out, A_rows, A_cols, A_vals, A_nnz, X, indices):
    # Computes the congruence transform A (X kron X) A' when A is very sparse
    # See https://link.springer.com/article/10.1007/BF02614319

    # We can cut the amount of operations in half by exploiting symetry of A and
    # X as follows
    # (AHA)_ij = Σ_a,b (Ai)_ab* (Σ_c,d (Aj)_cd X_ac X_db)
    #          = Σ_a,b (Ai)_ab* (  [Σ_c=d (Aj)_cd X_ac X_db]
    #                              + [Σ_c>d (Aj)_cd X_ac X_db]
    #                              + [Σ_c<d (Aj)_cd X_ac X_db]  )
    #          = Σ_a,b (Ai)_ab* (  [Σ_c=d (Aj)_cd X_ac X_db]
    #                              + [Σ_c<d (Aj)_cd X_ac X_db + (Aj)_cd* X_ad X_cb]  )
    #          = [Σ_a=b (Ai)_ab ( ... )] + [Σ_a<b (Ai)_ab ( ... )] + [Σ_a>b (Ai)_ab ( ... )]
    #          = [Σ_a=b (Ai)_ab ( ... )] + [Σ_a>b (Ai)_ab ( ... ) + (Ai)_ab* ( ... )]
    # Also note that off-diagonal entries of Ai have been scaled by 2

    n = A_rows.shape[0]

    # Loop through each entry of the Schur complement matrix (AHA)_ij
    for j in nb.prange(n):
        for i in nb.prange(j + 1):
            i_AHA = indices[i]
            j_AHA = indices[j]

            tmp1 = 0.0
            for alpha in range(A_nnz[i]):
                a = A_rows[i, alpha]
                b = A_cols[i, alpha]

                tmp2 = 0.0
                tmp3 = 0.0
                for beta in range(A_nnz[j]):
                    c = A_rows[j, beta]
                    d = A_cols[j, beta]

                    if c < d:
                        tmp2 += A_vals[j, beta] * X[a, c] * X[d, b]
                        tmp2 += np.conj(A_vals[j, beta]) * X[a, d] * X[c, b]
                    else:
                        tmp3 += A_vals[j, beta].real * X[a, c] * X[d, b]

                # Do addition slightly differently to guarantee a real number
                # i.e., just take the inner product between Ai and X Aj X by
                #     Σ_ab Re[(X Aj X)_ab] * Re[(Ai)_ab] + 2 Im[(X Aj X)_ab] * Im[(Ai)_ab]
                tmp3 += 0.5 * tmp2
                tmp1 += A_vals[i, alpha].real * tmp3.real
                tmp1 += A_vals[i, alpha].imag * tmp3.imag

            if i_AHA <= j_AHA:
                out[i_AHA, j_AHA] = tmp1
            else:
                out[j_AHA, i_AHA] = tmp1
