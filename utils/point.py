import numpy as np

class Point():
    def __init__(self, model):

        (n, p, q) = (model.n, model.p, model.q)

        self.vec = np.empty((n + p + q + q + 2, 1))

        self.x     = self.vec[:n]
        self.y     = self.vec[n : n+p]
        self.z     = self.vec[n+p : n+p+q]
        self.s     = self.vec[n+p+q : n+p+q+q]
        self.tau   = self.vec[n+p+q+q : n+p+q+q+1]
        self.kappa = self.vec[n+p+q+q+1 : n+p+q+q+2]

        self.s_views = [self.s[idxs] for idxs in model.cone_idxs]
        self.z_views = [self.z[idxs] for idxs in model.cone_idxs]

        return