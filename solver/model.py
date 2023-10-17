import numpy as np

class Model():
    def __init__(self, c, A, b, G=None, h=None, cones=None):
        self.n = np.size(c)
        self.p = np.size(b)
        self.q = np.size(h) if (h is not None) else self.n
    
        self.c = c
        self.A = A
        self.b = b
        self.G = G if (G is not None) else -np.eye(self.n)
        self.h = h if (h is not None) else np.zeros((self.n, 1))
        self.cones = cones

        self.use_G = (G is None)

        self.cone_idxs = build_cone_idxs(self.q, cones)
        self.nu = 0 if (len(cones) == 0) else sum((cone.get_nu() for cone in cones))

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