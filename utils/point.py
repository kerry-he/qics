import numpy as np
from utils import linear as lin

class Point():
    def __init__(self, model):

        # (n, p, q) = (model.n, model.p, model.q)

        # self.vec = np.empty((n + p + q + q + 2, 1))

        # self.X     = self.vec[:n]
        # self.y     = self.vec[n : n+p]
        # self.Z     = self.vec[n+p : n+p+q]
        # self.S     = self.vec[n+p+q : n+p+q+q]
        # self.tau   = self.vec[n+p+q+q : n+p+q+q+1]
        # self.kappa = self.vec[n+p+q+q+1 : n+p+q+q+2]

        # self.s_views = [self.S[idxs] for idxs in model.cone_idxs]
        # self.z_views = [self.Z[idxs] for idxs in model.cone_idxs]

        #TODO: Make x allow for G neq -I
        self.X     = [cone_k.zeros() for cone_k in model.cones]
        self.y     = np.zeros((model.p, 1))
        self.Z     = [cone_k.zeros() for cone_k in model.cones]
        self.S     = [cone_k.zeros() for cone_k in model.cones]
        self.tau   = 0.
        self.kappa = 0.

        self.model = model

        return
    
    def __add__(self, other):
        point = Point(self.model)

        point.X     = [x1 + x2 for (x1, x2) in zip(self.X, other.X)]
        point.y     = self.y + other.y
        point.Z     = [z1 + z2 for (z1, z2) in zip(self.Z, other.Z)]
        point.S     = [s1 + s2 for (s1, s2) in zip(self.S, other.S)]
        point.tau   = self.tau + other.tau
        point.kappa = self.kappa + other.kappa

        return point
    
    def __sub__(self, other):
        point = Point(self.model)

        point.X     = [x1 - x2 for (x1, x2) in zip(self.X, other.X)]
        point.y     = self.y - other.y
        point.Z     = [z1 - z2 for (z1, z2) in zip(self.Z, other.Z)]
        point.S     = [s1 - s2 for (s1, s2) in zip(self.S, other.S)]
        point.tau   = self.tau - other.tau
        point.kappa = self.kappa - other.kappa

        return point    
    
    def __mul__(self, a):
        point = Point(self.model)

        point.X     = [a * x for x in self.X]
        point.y     =  a * self.y
        point.Z     = [a * z for z in self.Z]
        point.S     = [a * s for s in self.S]
        point.tau   =  a * self.tau
        point.kappa =  a * self.kappa

        return point
    
    __rmul__ = __mul__
    
    def norm(self):
        return np.linalg.norm(np.array([
            lin.norm(self.X),
            lin.norm(self.y),
            lin.norm(self.Z),
            lin.norm(self.S),
            self.tau, 
            self.kappa
        ]))