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
        self.X     = np.zeros((model.n, 1))
        self.y     = np.zeros((model.p, 1))
        self.Z     = lin.Vector([cone_k.zeros() for cone_k in model.cones])
        self.S     = lin.Vector([cone_k.zeros() for cone_k in model.cones])
        self.tau   = 0.
        self.kappa = 0.

        self.model = model

        return
    
    def __add__(self, other):
        point = Point(self.model)

        point.X     = self.X + other.X
        point.y     = self.y + other.y
        point.Z     = self.Z + other.Z
        point.S     = self.S + other.S
        point.tau   = self.tau + other.tau
        point.kappa = self.kappa + other.kappa

        return point
    
    def __iadd__(self, other):
        self.X     += other.X
        self.y     += other.y
        self.Z     += other.Z
        self.S     += other.S
        self.tau   += other.tau
        self.kappa += other.kappa
        return self
    
    def axpy(self, a, other):
        self.X     += a * other.X
        self.y     += a * other.y
        self.Z.axpy(a, other.Z)
        self.S.axpy(a, other.S)
        self.tau   += a * other.tau
        self.kappa += a * other.kappa
        return self
     
    def copy(self, other):
        np.copyto(self.X, other.X)
        np.copyto(self.y, other.y)
        self.Z.copy(other.Z)
        self.S.copy(other.S)
        self.tau   = other.tau
        self.kappa = other.kappa
    
    def __sub__(self, other):
        point = Point(self.model)

        point.X     = self.X - other.X
        point.y     = self.y - other.y
        point.Z     = self.Z - other.Z
        point.S     = self.S - other.S
        point.tau   = self.tau - other.tau
        point.kappa = self.kappa - other.kappa

        return point    
    
    def __isub__(self, other):
        self.X     -= other.X
        self.y     -= other.y
        self.Z     -= other.Z
        self.S     -= other.S
        self.tau   -= other.tau
        self.kappa -= other.kappa
        return self    
    
    def __mul__(self, a):
        point = Point(self.model)

        point.X     = a * self.X
        point.y     = a * self.y
        point.Z     = a * self.Z
        point.S     = a * self.S
        point.tau   = a * self.tau
        point.kappa = a * self.kappa

        return point
    
    def __imul__(self, a):
        self.X     *= a
        self.y     *= a
        self.Z     *= a
        self.S     *= a
        self.tau   *= a
        self.kappa *= a
        return self    
    
    __rmul__ = __mul__
    
    def norm(self):
        return np.linalg.norm(np.array([
            lin.norm(self.X),
            lin.norm(self.y),
            self.Z.norm(),
            self.S.norm(),
            self.tau, 
            self.kappa
        ]))
        
class PointXYZ():
    def __init__(self, model):

        #TODO: Make x allow for G neq -I
        self.X     = np.zeros((model.n, 1))
        self.y     = np.zeros((model.p, 1))
        self.Z     = lin.Vector([cone_k.zeros() for cone_k in model.cones])

        self.model = model

        return
    
    def __add__(self, other):
        point = Point(self.model)

        point.X     = self.X + other.X
        point.y     = self.y + other.y
        point.Z     = self.Z + other.Z

        return point
    
    def __iadd__(self, other):
        self.X     += other.X
        self.y     += other.y
        self.Z     += other.Z
        return self
    
    def axpy(self, a, other):
        self.X     += a * other.X
        self.y     += a * other.y
        self.Z.axpy(a, other.Z)
        return self
     
    def copy(self, other):
        np.copyto(self.X, other.X)
        np.copyto(self.y, other.y)
        self.Z.copy(other.Z)
    
    def __sub__(self, other):
        point = Point(self.model)

        point.X     = self.X - other.X
        point.y     = self.y - other.y
        point.Z     = self.Z - other.Z

        return point    
    
    def __isub__(self, other):
        self.X     -= other.X
        self.y     -= other.y
        self.Z     -= other.Z
        return self    
    
    def __mul__(self, a):
        point = Point(self.model)

        point.X     = a * self.X
        point.y     = a * self.y
        point.Z     = a * self.Z

        return point
    
    def __imul__(self, a):
        self.X     *= a
        self.y     *= a
        self.Z     *= a
        return self    
    
    __rmul__ = __mul__
    
    def norm(self):
        return np.linalg.norm(np.array([
            lin.norm(self.X),
            lin.norm(self.y),
            self.Z.norm(),
        ]))
        
    def inp(self, other):
        return lin.inp(self.X, other.X) + lin.inp(self.y, other.y) + self.Z.inp(other.Z)