import numpy as np

class Model():
    def __init__(self, c, A, b, cones):
        self.c = c
        self.A = A
        self.b = b
        self.cones = cones

        self.n = np.size(c)
        self.p = np.size(b)

        self.cone_idxs = build_cone_idxs(self.n, cones)
        self.nu = 0 if (len(cones) == 0) else sum((cone.get_nu() for cone in cones))

        return
    
def build_cone_idxs(n, cones):
    cone_idxs = []
    prev_idx = 0
    for (i, cone) in enumerate(cones):
        dim = cone.dim
        cone_idxs.append(range(prev_idx, prev_idx + dim))
        prev_idx += dim
    assert prev_idx == n
    return cone_idxs