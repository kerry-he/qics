import numpy as np

class Point():
    def __init__(self, model):
        (n, p) = (model.n, model.p)

        self.vec = np.empty((n + p + n, 1))

        self.x = self.vec[:n]
        self.y = self.vec[n : n + p]
        self.z = self.vec[n + p:]

        self.x_views = [self.x[idxs] for idxs in model.cone_idxs]
        self.z_views = [self.z[idxs] for idxs in model.cone_idxs]

        return