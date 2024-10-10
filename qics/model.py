# Copyright (c) 2024, Kerry He, James Saunderson, and Hamza Fawzi

# This Python package QICS is licensed under the MIT license; see LICENSE.md 
# file in the root directory or at https://github.com/kerry-he/qics

import numpy as np
import scipy as sp

import qics.cones
import qics.point
from qics._utils import linalg as la


class Model:
    r"""A class representing an instance of the standard form primal

    .. math::

        \min_{x \in \mathbb{R}^n} &&& c^\top x

        \text{s.t.} &&& b - Ax = 0

         &&& h - Gx \in \mathcal{K},

    and dual

    .. math::

        \max_{y \in \mathbb{R}^p, z \in \mathbb{R}^q} &&& 
        -b^\top y - h^\top z

        \text{s.t.} &&& c + A^\top y + G^\top z = 0

         &&& z \in \mathcal{K}_*,

    conic programs, where :math:`c\in\mathbb{R}^n`, 
    :math:`b\in\mathbb{R}^p`, :math:`h\in\mathbb{R}^q`, 
    :math:`A\in\mathbb{R}^{p\times n}`, :math:`G\in\mathbb{R}^{q\times n}`,
    and :math:`\mathcal{K}\subset\mathbb{R}^{q}` is a convex, proper cone 
    with dual cone :math:`\mathcal{K}_*\subset\mathbb{R}^{q}`.


    Parameters
    ----------
    c : :class:`~numpy.ndarray`
        2D :obj:`~numpy.float64` array of size ``(n, 1)`` representing the
        linear objective :math:`c`.
    A : :class:`~numpy.ndarray` or :class:`~scipy.sparse.sparray`, optional
        2D :obj:`~numpy.float64` array of size ``(p, 1)`` representing 
        linear equality constraint matrix :math:`A`. The default is
        ``numpy.empty((0, n))``, i.e., there are no linear equalitiy
        constraints.
    b : :class:`~numpy.ndarray`, optional
        2D :obj:`~numpy.float64` array of size ``(p, 1)`` representing
        linear equality constraint vector :math:`b`. The default is
        ``numpy.zeros((p, 1))``, i.e., :math:`b=0`.
    G : :class:`~numpy.ndarray` or :class:`~scipy.sparse.sparray`, optional
        2D :obj:`~numpy.float64` array of size ``(q, n)`` representing
        linear cone constraint matrix :math:`G`. The default is 
        ``-scipy.sparse.eye(n)``, i.e., cone constraints are of the
        simplified form :math:`x+h\in\mathcal{K}`.
    h : :class:`~numpy.ndarray`, optional
        2D :obj:`~numpy.float64` array of size ``(q, 1)`` representing
        linear cone constraint vector :math:`h`. The default is 
        ``numpy.zeros((q, 1))``, i.e., :math:`h=0`.
    cones : :class:`list` of :mod:`~qics.cones`, optional
        Cartesian product of cones :math:`\mathcal{K}`. Default is ``[]``
        i.e., there are no conic constraints.
    offset : :class:`float`, optional
        Constant offset term to add to the objective function. Default is
        ``0``.
    """

    def __init__(
        self,
        c,
        A=None,
        b=None,
        G=None,
        h=None,
        cones=None,
        offset=0.0
    ):
        # Intiialize model parameters and default values for missing data
        self.n_orig = self.n = np.size(c)
        self.p_orig = self.p = np.size(b) if (b is not None) else 0
        self.q_orig = self.q = np.size(h) if (h is not None) else self.n

        # Make copies of everything so we don't overwrite data matrices
        self.c = c.copy()
        self.A = A.copy() if (A is not None) else np.empty((0, self.n))
        self.b = b.copy() if (b is not None) else np.empty((0, 1))
        self.G = G.copy() if (G is not None) else -sp.sparse.eye(self.n).tocsr()
        self.h = h.copy() if (h is not None) else np.zeros((self.n, 1))
        self.cones = cones
        self.offset = offset

        # Barrier parameter
        self.nu = 1 + sum([cone.nu for cone in cones])

        # Get properties of the problem
        self.issymmetric = all([cone.get_issymmetric() for cone in self.cones])
        self.iscomplex = any([cone.get_iscomplex() for cone in self.cones])        

        # Check if model uses A or G matrices
        self.use_G = not _is_like_eye(self.G)
        self.use_A = (A is not None) and (np.prod(A.shape) > 0)

    def _preprocess(self, use_invhess=False, init_pnt=None):
        SPARSE_THRESHOLD = 0.01
        cone_idxs = self.cone_idxs = _build_cone_idxs(self.q, self.cones)

        # Sparsify A and G if they are sufficiently sparse
        self.A = _sparsify(self.A, SPARSE_THRESHOLD, "csr")
        if self.use_G:
            self.G = _sparsify(self.G, SPARSE_THRESHOLD, "csr")

        # Restructure to allow for avoiding inverse Hessian oracles
        if self.use_G and not use_invhess:
            self._restructure(init_pnt)

        # Rescale model
        self._rescale()

        # Precompute transposes of A and G for faster sparse operations
        self.A_T = self.A.T.tocsr() if sp.sparse.issparse(self.A) else self.A.T
        self.G_T = self.G.T.tocsr() if sp.sparse.issparse(self.G) else self.G.T

        # Get slices of A or G matrices correpsonding to each cone
        # and some other handy precomputations
        if self.use_G:
            self.G_T_views = [self.G_T[:, idxs_k] for idxs_k in cone_idxs]
            self.G_T_views = _sparsify(self.G_T_views, SPARSE_THRESHOLD)

            # Need a dense A' to do Cholesky solves on
            if sp.sparse.issparse(self.A):
                self.A_T_dense = self.A_T.toarray()
            else:
                self.A_T_dense = self.A_T

            issparse_list = [sp.sparse.issparse(Gk) for Gk in self.G_T_views]
            self.issparse = any(issparse_list)

        elif self.use_A:
            # After rescaling, G is an easily invertible square diagonal matrix
            self.G_inv = np.reciprocal(self.G.diagonal()).reshape((-1, 1))

            self.A_invG = la.scale_axis(self.A.copy(), scale_cols=self.G_inv)
            if sp.sparse.issparse(self.A_invG):
                self.A_invG = self.A_invG.tocsr()

            self.A_invG_views = [self.A_invG[:, idxs_k] for idxs_k in cone_idxs]
            self.A_invG_views = _sparsify(self.A_invG_views, SPARSE_THRESHOLD)

            issparse_list = [sp.sparse.issparse(Ak) for Ak in self.A_invG_views]
            self.issparse = any(issparse_list)

        else:
            self.G_inv = np.reciprocal(self.G.diagonal()).reshape((-1, 1))
            self.issparse = True

        return

    def _restructure(self, init_pnt=None):
        # Restructures the conic program into
        #     min  <c,x1>
        #     s.t  A*x1 = b,  x2 = 0,  x3 = 1
        #          h1*x2 + h2*x3 - G*x1 ∈ K
        # where h1 is an interior point of K and h2 = h. This allows us to 
        # solve problems using the cone K'={x : G*x ∈ K}.

        n = self.n
        self.x_offset = np.zeros((n, 1))

        # Add variable x2 and constraint x2 = 0 if necessary
        # Find an interior point of K and normalize it
        if init_pnt is None:
            s_init = qics.point.VecProduct(self.cones)
            s_init.vec.fill(np.nan)
        else:
            # If user gives us an inital s, then use this instead
            s_init = init_pnt.s

        for k, cone_k in enumerate(self.cones):
            if any(np.isnan(s_init.vecs[k])):
                cone_k.get_init_point(s_init[k])
        s_norm = np.sum(np.abs(s_init.vec))

        G_temp = _hstack((self.G, -s_init.vec / s_norm))
        if la.is_full_col_rank(G_temp):
            A_new_col = sp.sparse.coo_matrix((self.p, 1))
            A_new_row = sp.sparse.coo_matrix(([1.], ([0], [n])), (1, n+1))
            self.c = np.vstack((self.c, np.array([[0.]])))
            self.A = _vstack((_hstack((self.A, A_new_col)), A_new_row))
            self.b = np.vstack((self.b, np.array([[0.]])))
            self.G = G_temp

            self.x_offset = np.vstack((self.x_offset, np.array([[0.]])))

            self.use_A = True
            (n, _) = (self.n, self.p) = (self.n + 1, self.p + 1)

        # Add variable x3 and constraint x3 = 1 if necessary
        if np.any(self.h):
            h_norm = np.sum(np.abs(self.h))
            G_temp = _hstack((self.G, -self.h / h_norm))
            if la.is_full_col_rank(G_temp):
                A_new_col = sp.sparse.coo_matrix((self.p, 1))
                A_new_row = sp.sparse.coo_matrix(([1.], ([0], [n])), (1, n+1))
                self.c = np.vstack((self.c, np.array([[0.]])))
                self.A = _vstack((_hstack((self.A, A_new_col)), A_new_row))
                self.b = np.vstack((self.b, np.array([[h_norm]])))
                self.G = G_temp
                self.h = np.zeros((self.q, 1))

                self.x_offset = np.vstack((self.x_offset, np.array([[0.]])))

                self.use_A = True
                (n, _) = (self.n, self.p) = (self.n + 1, self.p + 1)
            else:
                self.x_offset = sp.sparse.linalg.lsqr(self.G, self.h)[0]
                self.x_offset = self.x_offset.reshape((-1, 1))
                self.offset += (self.c.T @ self.x_offset)[0, 0]
                self.b = self.b - self.A @ self.x_offset
                self.h = np.zeros((self.q, 1))
        
        if self.x_offset is None:
            self.x_offset = np.zeros((n, 1))

    def _rescale(self):
        # Rescale c
        self.c_scale = np.maximum.reduce([np.abs(self.c.ravel()), 
                                          la.abs_max(self.A, axis=0), 
                                          la.abs_max(self.G, axis=0)])
        self.c_scale = np.sqrt(self.c_scale).reshape((-1, 1))

        # Rescale b
        self.b_scale = np.maximum.reduce([np.abs(self.b.ravel()), 
                                          la.abs_max(self.A, axis=1)])
        self.b_scale = np.sqrt(self.b_scale).reshape((-1, 1))

        # Rescale h
        # Note we can only scale each cone by a positive factor, and
        # we can't scale each individual variable by a different factor
        # (except for the nonnegative orthant)
        self.h_scale = np.zeros((self.q, 1))
        h_absmax = np.abs(self.h.ravel())
        G_absmax = la.abs_max(self.G, axis=1)
        for k, cone_k in enumerate(self.cones):
            idxs = self.cone_idxs[k]
            if isinstance(cone_k, qics.cones.NonNegOrthant):
                self.h_scale[idxs, 0] = np.maximum.reduce([h_absmax[idxs], 
                                                           G_absmax[idxs]])
            else:
                self.h_scale[idxs, 0] = np.max([h_absmax[idxs], G_absmax[idxs]])
        self.h_scale = np.sqrt(self.h_scale)

        # Ensure there are no divide by zeros
        EPS = np.finfo(self.b_scale.dtype).eps
        self.c_scale[self.c_scale < EPS] = 1.0
        self.b_scale[self.b_scale < EPS] = 1.0
        self.h_scale[self.h_scale < EPS] = 1.0

        # Rescale data
        self.c /= self.c_scale
        self.b /= self.b_scale
        self.h /= self.h_scale
        self.A = la.scale_axis(self.A,
                               scale_rows=np.reciprocal(self.b_scale),
                               scale_cols=np.reciprocal(self.c_scale))
        self.G = la.scale_axis(self.G,
                               scale_rows=np.reciprocal(self.h_scale),
                               scale_cols=np.reciprocal(self.c_scale))

        return


def _is_like_eye(A, tol=1e-10):
    if A.shape[0] != A.shape[1]:
        return False
    n = A.shape[0]
    if sp.sparse.issparse(A):
        A_minus_eye = sp.sparse.eye(n) - abs(A)
        return sp.sparse.linalg.norm(A_minus_eye) < tol
    else:
        A_minus_eye = np.eye(n) - abs(A)
        return np.linalg.norm(A_minus_eye) < tol


def _vstack(tup):
    if isinstance(tup, tuple):
        tup = list(tup)

    if sp.sparse.issparse(tup[0]):
        for k in range(1, len(tup)):
            tup[k] = sp.sparse.coo_matrix(tup[k])
        return sp.sparse.vstack(tup)
    else:
        for k in range(1, len(tup)):
            if sp.sparse.issparse(tup[k]):
                tup[k] = tup[k].toarray()
        return np.vstack(tup)


def _hstack(tup):
    if isinstance(tup, tuple):
        tup = list(tup)

    if sp.sparse.issparse(tup[0]):
        for k in range(1, len(tup)):
            tup[k] = sp.sparse.coo_matrix(tup[k])
        return sp.sparse.hstack(tup)
    else:
        for k in range(1, len(tup)):
            if sp.sparse.issparse(tup[k]):
                tup[k] = tup[k].toarray()
        return np.hstack(tup)


def _build_cone_idxs(n, cones):
    cone_idxs = []
    prev_idx = 0
    for cone in cones:
        dim_k = np.sum(cone.dim)
        cone_idxs.append(slice(prev_idx, prev_idx + dim_k))
        prev_idx += dim_k
    assert prev_idx == n
    return cone_idxs


def _sparsify(A, threshold, format="coo"):
    def sparsify_single(A, threshold, format):
        if A.size == 0:
            if sp.sparse.issparse(A):
                A = A.toarray()
            return A

        if sp.sparse.issparse(A):
            if A.nnz / np.prod(A.shape) > threshold:
                return A.toarray()
            else:
                if format == "coo":
                    return A.tocoo()
                elif format == "csr":
                    return A.tocsr()
        else:
            if np.count_nonzero(A) / A.size < threshold:
                if format == "coo":
                    return sp.sparse.coo_matrix(A)
                elif format == "csr":
                    return sp.sparse.csr_matrix(A)
        return A

    if isinstance(A, list):
        return [sparsify_single(A_k, threshold, format) for A_k in A]
    else:
        return sparsify_single(A, threshold, format)
