import numpy as np

class Point():
    def __init__(self, model):
        (n, p) = (model.n, model.p)

        self.vec = np.empty((n + p + n + 2, 1))

        self.x     = self.vec[:n]
        self.y     = self.vec[n : n+p]
        self.z     = self.vec[n+p : n+p+n]
        self.tau   = self.vec[n+p+n : n+p+n+1]
        self.kappa = self.vec[n+p+n+1 : n+p+n+2]

        self.x_views = [self.x[idxs] for idxs in model.cone_idxs]
        self.z_views = [self.z[idxs] for idxs in model.cone_idxs]

        return