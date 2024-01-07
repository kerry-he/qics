import numpy as np
import scipy as sp

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
        self.nu = 0 if (len(cones) == 0) else sum((cone.get_nu() for cone in cones))

        self.offset = offset

        return
    
def build_cone_idxs(n, cones):
    cone_idxs = []
    prev_idx = 0
    for (i, cone) in enumerate(cones):
        dim = cone.dim
        cone_idxs.append(slice(prev_idx, prev_idx + dim))
        prev_idx += dim
    assert prev_idx == n
    return cone_idxs