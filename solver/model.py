import numpy as np
import scipy as sp

from utils import symmetric as sym
from utils import linear as lin

class Model():
    def __init__(self, c, A=None, b=None, G=None, h=None, cones=None, offset=0.0, c_mtx=None, A_mtx=None, h_mtx=None):
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

        self.c_mtx = lin.Vector([c_mtx])
        self.h_mtx = lin.Vector([lin.Symmetric(c_mtx.n)])
        self.A_mtx = A_mtx

        return
    
    def apply_A(self, x):
        b = np.zeros((self.p, 1))
        for (j, Aj) in enumerate(self.A_mtx):
            b[j] = x.inp(Aj)
        return b

    def apply_A_T(self, y):
        c = lin.Vector([cone_k.zeros() for cone_k in self.cones])
        for (j, Aj) in enumerate(self.A_mtx):
            for (i, Aji) in enumerate(Aj):
                c[i] += y[j, 0] * Aji
        return c
    
    def get_A_mtx(self):
        # Convert list of A correspoding to linear constaints
        # <x1, Ai1> + <x2, Ai2> + ... + <xn, Ain> = bi, for all i=1,...,p
        # to its corresponding matrix expression

        dims = [ci.vn for ci in self.c_mtx]   # Dimensions of vector spaces
        p = len(self.A)                         # Number of constaints
        n = sum(dims)                           # Total real vector dimension
        out = np.zeros((p, n))

        for (j, Aj) in enumerate(self.A_mtx):
            out[j, :] = Aj.to_vec()

        return out 

def build_cone_idxs(n, cones):
    cone_idxs = []
    prev_idx = 0
    for (i, cone) in enumerate(cones):
        dim = cone.dim
        cone_idxs.append(slice(prev_idx, prev_idx + dim))
        prev_idx += dim
    assert prev_idx == n
    return cone_idxs