import numpy as np
import scipy as sp

from utils import symmetric as sym
from utils import linear as lin
from cones import *

class Model():
    def __init__(self, c, A=None, b=None, G=None, h=None, cones=None, offset=0.0):
        self.n = np.size(c)
        self.p = np.size(b) if (b is not None) else 0
        self.q = np.size(h) if (h is not None) else self.n
    
        self.c = c
        self.A = A if (A is not None) else  np.empty((0, self.n))
        self.b = b if (b is not None) else  np.empty((0, 1))
        self.G = G if (G is not None) else -sp.sparse.identity(self.n)
        self.h = h if (h is not None) else  np.zeros((self.n, 1))
        self.cones = cones

        self.use_G = (G is not None)
        self.use_A = (A is not None) and (A.size > 0)

        self.cone_idxs = build_cone_idxs(self.q, cones)
        self.nu = 1 if (len(cones) == 0) else (1 + sum([cone.get_nu() for cone in cones]))

        self.offset = offset
        
        self.A_T = self.A.T.tocsr()
        self.G_T = self.G.T.tocsr()
        
        if self.use_G:
            self.G_T_views = [self.G_T[:, idxs_k] for idxs_k in self.cone_idxs]
        elif self.use_A:
            self.A_views = [self.A[:, idxs_k] for idxs_k in self.cone_idxs]

        self.sym = True
        for cone_k in cones:
            self.sym = self.sym and (isinstance(cone_k, nonnegorthant.Cone) or isinstance(cone_k, possemidefinite.Cone))
        
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