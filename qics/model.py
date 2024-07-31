import numpy as np
import scipy as sp

from qics.utils import sparse
import qics.cones
class Model():
    """A class representing an instance of the primal

    .. math::

        \\min_{x \\in \\mathbb{R}^n} &&& c^\\top x

        \\text{s.t.} &&& b - Ax = 0

         &&& h - Gx \\in \\mathcal{K}

    and dual

    .. math::

        \\max_{y \\in \\mathbb{R}^p, z \\in \\mathbb{R}^q} &&& -b^\\top y - h^\\top z

        \\text{s.t.} &&& c + A^\\top y + G^\\top z = 0

         &&& z \\in \\mathcal{K}_*

    standard form conic programs, where :math:`c \\in \\mathbb{R}^n`, :math:`b \\in \\mathbb{R}^p`, 
    :math:`h \\in \\mathbb{R}^q`, :math:`A \\in \\mathbb{R}^{p \\times n}`, :math:`G \\in \\mathbb{R}^{q \\times n}`, 
    and :math:`\\mathcal{K} \\subset \\mathbb{R}^{q}` is a convex, proper cone with dual cone :math:`\\mathcal{K}_* \\subset \\mathbb{R}^{q}`.

         
    Parameters
    ----------
    c : (n, 1) ndarray
        Float array representing linear objective
    A : (p, n) ndarray, optional
        Float array representing linear equality constraints. Default is empty matrix.
    b : (p, 1) ndarray, optional
        Float array representing linear equality constraints. Default is ``0``.
    G : (q, n) ndarray, optional
        Float array representing linear cone constraints. Default is ``-I``.
    h : (q, 1) ndarray, optional
        Float array representing linear cone constraints. Default is ``0``.
    cones : list, optional
        List of cone classes representing the Cartesian product of cones :math:`\\mathcal{K}`. Default is empty set.
    offset : float, optional
        Constant offset term to add to the objective function. Default is ``0``.
    """
    def __init__(self, c, A=None, b=None, G=None, h=None, cones=None, offset=0.0):
        # Intiialize model parameters and default values for missing data
        self.n = np.size(c)
        self.p = np.size(b) if (b is not None) else 0
        self.q = np.size(h) if (h is not None) else self.n
    
        self.c = c.copy()
        self.A = A.copy() if (A is not None) else  np.empty((0, self.n))
        self.b = b.copy() if (b is not None) else  np.empty((0, 1))
        self.G = G.copy() if (G is not None) else -sp.sparse.eye(self.n).tocsr()
        self.h = h.copy() if (h is not None) else  np.zeros((self.n, 1))
        self.cones = cones

        self.use_G = (G is not None)
        self.use_A = (A is not None) and (A.size > 0)

        self.cone_idxs = build_cone_idxs(self.q, cones)
        self.nu = 1 + sum([cone.nu for cone in cones])

        self.offset = offset
        
        # Rescale model
        self.rescale_model()
        
        self.A_T = self.A.T.tocsr() if sp.sparse.issparse(self.A) else self.A.T
        self.G_T = self.G.T.tocsr() if sp.sparse.issparse(self.G) else self.G.T
        
        # Get sclices of A or G matrices correpsonding to each cone
        if self.use_G:
            self.G_T_views = [self.G_T[:, idxs_k] for idxs_k in self.cone_idxs]
        elif self.use_A:
            # After rescaling, G is some easily invertible square diagonal matrix
            self.G_inv = -self.c_scale.reshape((-1, 1))
            self.A_invG = sparse.scale_axis(self.A.copy(), scale_cols=self.G_inv)
            self.A_invG_T = self.A_invG.T.tocsr() if sp.sparse.issparse(self.A_invG) else self.A_invG.T
            self.A_invG_views = [self.A_invG[:, idxs_k] for idxs_k in self.cone_idxs]

        self.issymmetric = all([cone_k.get_issymmetric() for cone_k in cones])
        self.iscomplex   = any([cone_k.get_iscomplex()   for cone_k in cones])
        
        return
    
    def rescale_model(self):        
        # Rescale c
        self.c_scale = np.sqrt(np.maximum.reduce([
            np.abs(self.c.reshape(-1)),
            sparse.abs_max(self.A, axis=0),
            sparse.abs_max(self.G, axis=0)
        ]))

        # Rescale b
        self.b_scale = np.sqrt(np.maximum.reduce([
            np.abs(self.b.reshape(-1)),
            sparse.abs_max(self.A, axis=1)
        ]))

        # Rescale h
        # Note we can only scale each cone by a positive factor, and 
        # we can't scale each individual variable by a different factor
        # (except for the nonnegative orthant)
        self.h_scale = np.zeros(self.q)
        h_absmax = np.abs(self.h.reshape(-1))
        G_absmax_row = sparse.abs_max(self.G, axis=1)
        for (k, cone_k) in enumerate(self.cones):
            idxs = self.cone_idxs[k]
            if isinstance(cone_k, qics.cones.NonNegOrthant):
                self.h_scale[idxs] = np.sqrt(np.maximum.reduce([
                    h_absmax[idxs],
                    G_absmax_row[idxs]
                ]))
            else:
                self.h_scale[idxs] = np.sqrt(np.max([
                    h_absmax[idxs],
                    G_absmax_row[idxs]
                ]))

        # Ensure there are no divide by zeros
        self.c_scale[self.c_scale < np.finfo(self.c_scale.dtype).eps] = 1.
        self.b_scale[self.b_scale < np.finfo(self.b_scale.dtype).eps] = 1.
        self.h_scale[self.h_scale < np.finfo(self.h_scale.dtype).eps] = 1.

        # Rescale data
        self.c /= self.c_scale.reshape((-1, 1))
        self.b /= self.b_scale.reshape((-1, 1))
        self.h /= self.h_scale.reshape((-1, 1))

        self.A = sparse.scale_axis(self.A, 
            scale_rows = np.reciprocal(self.b_scale), 
            scale_cols = np.reciprocal(self.c_scale)
        )

        self.G = sparse.scale_axis(self.G, 
            scale_rows = np.reciprocal(self.h_scale), 
            scale_cols = np.reciprocal(self.c_scale)
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