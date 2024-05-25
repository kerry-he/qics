import numpy as np
import scipy as sp

from utils import symmetric as sym
from utils import linear as lin
from utils import sparse
from cones import *

class Model():
    def __init__(self, c, A=None, b=None, G=None, h=None, cones=None, offset=0.0):
        # Intiialize model parameters and default values for missing data
        self.n = np.size(c)
        self.p = np.size(b) if (b is not None) else 0
        self.q = np.size(h) if (h is not None) else self.n
    
        self.c = c
        self.A = A if (A is not None) else  np.empty((0, self.n))
        self.b = b if (b is not None) else  np.empty((0, 1))
        self.G = G if (G is not None) else -sp.sparse.identity(self.n).tocsr()
        self.h = h if (h is not None) else  np.zeros((self.n, 1))
        self.cones = cones

        self.use_G = (G is not None)
        self.use_A = (A is not None) and (A.size > 0)

        self.cone_idxs = build_cone_idxs(self.q, cones)
        self.nu = 1 if (len(cones) == 0) else (1 + sum([cone.get_nu() for cone in cones]))

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
            self.G_inv = np.reciprocal(self.G.diagonal()).reshape((-1, 1))

            G_inv_A = sparse.scale_axis(self.A.copy(), scale_cols=self.G_inv)
            self.G_inv_A_views = [G_inv_A[:, idxs_k] for idxs_k in self.cone_idxs]

        self.sym = True
        for cone_k in cones:
            self.sym = self.sym and (isinstance(cone_k, nonnegorthant.Cone) or isinstance(cone_k, possemidefinite.Cone))
        
        return
    
    def rescale_model(self):        
        # Rescale c
        self.c_scale = np.maximum.reduce([
            np.abs(self.c.reshape(-1)),
            sparse.abs_max(self.A, axis=0),
            sparse.abs_max(self.G, axis=0)
        ])

        # Rescale b
        self.b_scale = np.maximum.reduce([
            np.abs(self.b.reshape(-1)),
            sparse.abs_max(self.A, axis=1)
        ])

        # Rescale h
        self.h_scale = np.maximum.reduce([
            np.abs(self.h.reshape(-1)),
            sparse.abs_max(self.G, axis=1)
        ])

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
        dim_k = cone.dim
        cone_idxs.append(slice(prev_idx, prev_idx + dim_k))
        prev_idx += dim_k
    assert prev_idx == n
    return cone_idxs