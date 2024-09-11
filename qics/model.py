import numpy as np
import scipy as sp

import qics.cones
from qics._utils import linalg


class Model:
    r"""A class representing an instance of the primal.

    .. math::

        \min_{x \in \mathbb{R}^n} &&& c^\top x

        \text{s.t.} &&& b - Ax = 0

         &&& h - Gx \in \mathcal{K}

    and dual

    .. math::

        \max_{y \in \mathbb{R}^p, z \in \mathbb{R}^q} &&& -b^\top y - h^\top z

        \text{s.t.} &&& c + A^\top y + G^\top z = 0

         &&& z \in \mathcal{K}_*

    standard form conic programs, where :math:`c \in \mathbb{R}^n`,
    :math:`b \in \mathbb{R}^p`, :math:`h \in \mathbb{R}^q`,
    :math:`A \in \mathbb{R}^{p \times n}`, :math:`G \in \mathbb{R}^{q \times n}`,
    and :math:`\mathcal{K} \subset \mathbb{R}^{q}` is a convex, proper cone with dual
    cone :math:`\mathcal{K}_* \subset \mathbb{R}^{q}`.


    Parameters
    ----------
    c : (n, 1) ndarray
        Float array representing the linear objective.
    A : (p, n) ndarray or scipy.sparse.sparray, optional
        Float array representing linear equality constraints. Default is empty matrix.
    b : (p, 1) ndarray, optional
        Float array representing linear equality constraints. Default is ``0``.
    G : (q, n) ndarray or scipy.sparse.sparray, optional
        Float array representing linear cone constraints. Default is ``-I``.
    h : (q, 1) ndarray, optional
        Float array representing linear cone constraints. Default is ``0``.
    cones : list, optional
        List of :class:`qics.cones` representing the Cartesian product of cones
        :math:`\mathcal{K}`. Default is empty set.
    offset : float, optional
        Constant offset term to add to the objective function. Default is ``0``.
    """

    def __init__(self, c, A=None, b=None, G=None, h=None, cones=None, offset=0.0):
        SPARSE_THRESHOLD = 0.01

        # Intiialize model parameters and default values for missing data
        self.n = np.size(c)
        self.p = np.size(b) if (b is not None) else 0
        self.q = np.size(h) if (h is not None) else self.n

        self.c_raw = c
        self.A_raw = A if (A is not None) else np.empty((0, self.n))
        self.b_raw = b if (b is not None) else np.empty((0, 1))
        self.G_raw = G if (G is not None) else -sp.sparse.eye(self.n).tocsr()
        self.h_raw = h if (h is not None) else np.zeros((self.n, 1))

        self.c = c.copy()
        self.A = A.copy() if (A is not None) else np.empty((0, self.n))
        self.b = b.copy() if (b is not None) else np.empty((0, 1))
        self.G = G.copy() if (G is not None) else -sp.sparse.eye(self.n).tocsr()
        self.h = h.copy() if (h is not None) else np.zeros((self.n, 1))
        self.cones = cones

        self.use_G = (G is not None) and (
            self.n != self.q or (np.linalg.norm(np.eye(self.n) + self.G) > 1e-10)
        )
        self.use_A = (A is not None) and (np.prod(A.shape) > 0)

        self.A = sparsify(self.A, SPARSE_THRESHOLD, "csr")
        self.G = sparsify(self.G, SPARSE_THRESHOLD, "csr") if self.use_G else self.G

        self.cone_idxs = build_cone_idxs(self.q, cones)
        self.nu = 1 + sum([cone.nu for cone in cones])

        self.offset = offset

        # Rescale model
        self.rescale_model()

        self.A_T = self.A.T.tocsr() if sp.sparse.issparse(self.A) else self.A.T
        self.G_T = self.G.T.tocsr() if sp.sparse.issparse(self.G) else self.G.T

        # Get slices of A or G matrices correpsonding to each cone
        if self.use_G:
            self.G_T_views = sparsify(
                [self.G_T[:, idxs_k] for idxs_k in self.cone_idxs], SPARSE_THRESHOLD
            )
            self.A_T_dense = (
                self.A_T.toarray() if sp.sparse.issparse(self.A_T) else self.A_T
            )
            self.A_coo = self.A.tocoo() if sp.sparse.issparse(self.A) else self.A
            self.issparse = any([sp.sparse.issparse(G_T_k) for G_T_k in self.G_T_views])
        elif self.use_A:
            # After rescaling, G is some easily invertible square diagonal matrix
            self.G_inv = -self.c_scale.reshape((-1, 1))
            self.A_invG = linalg.scale_axis(self.A.copy(), scale_cols=self.G_inv)
            self.A_invG = (
                self.A_invG.tocsr() if sp.sparse.issparse(self.A_invG) else self.A_invG
            )
            self.A_invG_views = sparsify(
                [self.A_invG[:, idxs_k] for idxs_k in self.cone_idxs], SPARSE_THRESHOLD
            )
            self.issparse = any(
                [sp.sparse.issparse(A_invG_k) for A_invG_k in self.A_invG_views]
            )
        else:
            self.G_inv = np.reciprocal(self.G.diagonal()).reshape((-1, 1))
            self.issparse = True

        self.issymmetric = all([cone_k.get_issymmetric() for cone_k in cones])
        self.iscomplex = any([cone_k.get_iscomplex() for cone_k in cones])

        return

    def rescale_model(self):
        # Rescale c
        self.c_scale = np.sqrt(
            np.maximum.reduce(
                [
                    np.abs(self.c.reshape(-1)),
                    linalg.abs_max(self.A, axis=0),
                    linalg.abs_max(self.G, axis=0),
                ]
            )
        )

        # Rescale b
        self.b_scale = np.sqrt(
            np.maximum.reduce(
                [np.abs(self.b.reshape(-1)), linalg.abs_max(self.A, axis=1)]
            )
        )

        # Rescale h
        # Note we can only scale each cone by a positive factor, and
        # we can't scale each individual variable by a different factor
        # (except for the nonnegative orthant)
        self.h_scale = np.zeros(self.q)
        h_absmax = np.abs(self.h.reshape(-1))
        G_absmax_row = linalg.abs_max(self.G, axis=1)
        for k, cone_k in enumerate(self.cones):
            idxs = self.cone_idxs[k]
            if isinstance(cone_k, qics.cones.NonNegOrthant):
                self.h_scale[idxs] = np.sqrt(
                    np.maximum.reduce([h_absmax[idxs], G_absmax_row[idxs]])
                )
            else:
                self.h_scale[idxs] = np.sqrt(
                    np.max([h_absmax[idxs], G_absmax_row[idxs]])
                )

        # Ensure there are no divide by zeros
        self.c_scale[self.c_scale < np.finfo(self.c_scale.dtype).eps] = 1.0
        self.b_scale[self.b_scale < np.finfo(self.b_scale.dtype).eps] = 1.0
        self.h_scale[self.h_scale < np.finfo(self.h_scale.dtype).eps] = 1.0

        # Rescale data
        self.c /= self.c_scale.reshape((-1, 1))
        self.b /= self.b_scale.reshape((-1, 1))
        self.h /= self.h_scale.reshape((-1, 1))

        self.A = linalg.scale_axis(
            self.A,
            scale_rows=np.reciprocal(self.b_scale),
            scale_cols=np.reciprocal(self.c_scale),
        )

        self.G = linalg.scale_axis(
            self.G,
            scale_rows=np.reciprocal(self.h_scale),
            scale_cols=np.reciprocal(self.c_scale),
        )

        return


def build_cone_idxs(n, cones):
    cone_idxs = []
    prev_idx = 0
    for cone in cones:
        dim_k = np.sum(cone.dim)
        cone_idxs.append(slice(prev_idx, prev_idx + dim_k))
        prev_idx += dim_k
    assert prev_idx == n
    return cone_idxs


def sparsify(A, threshold, format="coo"):
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


def complex_to_real(model):
    if not model.iscomplex:
        return model

    from qics.vectorize import mat_to_vec, vec_to_mat

    def _c2r_matrix(G, model, factor=1.0):
        # Loop through columns
        Gc = []
        for i in range(G.shape[1]):
            # Loop through cones
            Gc_i = []
            for j, cone_j in enumerate(model.cones):
                G_ij = G[model.cone_idxs[j], [i]]
                # Loop through subvectors (if necessary)
                if isinstance(cone_j.dim, list):
                    Gc_ij = []
                    idxs = np.insert(np.cumsum(cone_j.dim), 0, 0)
                    for k in range(len(cone_j.dim)):
                        Gc_ijk = G_ij[idxs[k] : idxs[k + 1]]
                        if cone_j.type[k] == "h":
                            Gc_ijk_mtx = vec_to_mat(Gc_ijk, iscomplex=True)
                            Gc_ijk_mtx_real = np.block(
                                [
                                    [Gc_ijk_mtx.real, -Gc_ijk_mtx.imag],
                                    [Gc_ijk_mtx.imag, Gc_ijk_mtx.real],
                                ]
                            )
                            Gc_ijk = mat_to_vec(Gc_ijk_mtx_real)
                        Gc_ij += [Gc_ijk]

                    Gc_ij = np.vstack(Gc_ij)

                else:
                    Gc_ij = G_ij
                    if cone_j.type == "h":
                        Gc_ij_mtx = vec_to_mat(Gc_ij, iscomplex=True)
                        Gc_ij_mtx_real = np.block(
                            [
                                [Gc_ij_mtx.real, -Gc_ij_mtx.imag],
                                [Gc_ij_mtx.imag, Gc_ij_mtx.real],
                            ]
                        )
                        Gc_ij = mat_to_vec(Gc_ij_mtx_real)

                Gc_i += [Gc_ij]
            Gc += [np.vstack(Gc_i)]
        Gc = np.hstack(Gc)

        return Gc

    cones = []
    for cone_k in model.cones:
        if isinstance(cone_k, qics.cones.NonNegOrthant):
            cones += [qics.cones.NonNegOrthant(cone_k.n)]
        if isinstance(cone_k, qics.cones.PosSemidefinite):
            if cone_k.get_iscomplex():
                cones += [qics.cones.PosSemidefinite(2 * cone_k.n)]
            else:
                cones += [qics.cones.PosSemidefinite(cone_k.n)]
        if isinstance(cone_k, qics.cones.QuantEntr):
            if cone_k.get_iscomplex():
                cones += [qics.cones.QuantEntr(2 * cone_k.n)]
            else:
                cones += [qics.cones.QuantEntr(cone_k.n)]
        if isinstance(cone_k, qics.cones.QuantRelEntr):
            if cone_k.get_iscomplex():
                cones += [qics.cones.QuantRelEntr(2 * cone_k.n)]
            else:
                cones += [qics.cones.QuantRelEntr(cone_k.n)]

    if model.use_G:
        # Need to split A into [-G; -A] and b into [-h; -b]
        # and uncompact G, h
        G = _c2r_matrix(model.G_raw, model)
        h = _c2r_matrix(model.h_raw, model)
        return Model(
            c=model.c_raw / 2,
            A=model.A_raw,
            b=model.b_raw,
            G=G,
            h=h,
            cones=cones,
            offset=model.offset,
        )
    else:
        # No G, just need to uncompact c and A
        c = _c2r_matrix(model.c_raw, model, factor=0.5)
        A = _c2r_matrix(model.A_raw.T, model, factor=0.5).T
        return Model(c=c, A=A, b=model.b_raw, cones=cones, offset=model.offset)
