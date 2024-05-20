import numpy as np
import scipy as sp
import numba as nb
from utils import linear as lin

class Point():
    def __init__(self, model):
        (n, p, q) = (model.n, model.p, model.q)

        self.vec = np.zeros((n + p + q + q + 2, 1))

        self.X     = self.vec[:n]
        self.y     = self.vec[n : n+p]
        self.Z     = lin.Vector(model.cones, self.vec[n+p : n+p+q])
        self.S     = lin.Vector(model.cones, self.vec[n+p+q : n+p+q+q])
        self.tau   = self.vec[n+p+q+q : n+p+q+q+1]
        self.kappa = self.vec[n+p+q+q+1 : n+p+q+q+2]

        # self.s_views = [self.S[idxs] for idxs in model.cone_idxs]
        # self.z_views = [self.Z[idxs] for idxs in model.cone_idxs]

        return
    
    def __iadd__(self, other):
        self.X     += other.X
        self.y     += other.y
        self.Z     += other.Z
        self.S     += other.S
        self.tau   += other.tau
        self.kappa += other.kappa
        return self
    
    def __isub__(self, other):
        self.X     -= other.X
        self.y     -= other.y
        self.Z     -= other.Z
        self.S     -= other.S
        self.tau   -= other.tau
        self.kappa -= other.kappa
        return self    
    
    def __imul__(self, a):
        self.X     *= a
        self.y     *= a
        self.Z     *= a
        self.S     *= a
        self.tau   *= a
        self.kappa *= a
        return self    
    
    def axpy(self, a, other):
        self.vec[:] = sp.linalg.blas.daxpy(other.vec, self.vec, a=a)
        return self
     
    def copy_from(self, other):
        np.copyto(self.vec, other.vec)
        return self
        
    def norm(self, order=None):
        return np.linalg.norm(self.vec, ord=order)
        
class PointXYZ():
    def __init__(self, model):
        (n, p, q) = (model.n, model.p, model.q)

        self.vec = np.zeros((n + p + q, 1))

        self.X     = self.vec[:n]
        self.y     = self.vec[n : n+p]
        self.Z     = lin.Vector(model.cones, self.vec[n+p : n+p+q])

        return
    
    def __iadd__(self, other):
        self.X     += other.X
        self.y     += other.y
        self.Z     += other.Z
        return self

    def __isub__(self, other):
        self.X     -= other.X
        self.y     -= other.y
        self.Z     -= other.Z
        return self    

    def __imul__(self, a):
        self.X     *= a
        self.y     *= a
        self.Z     *= a
        return self
    
    def axpy(self, a, other):
        self.vec[:] = sp.linalg.blas.daxpy(other.vec, self.vec, a=a)
        return self
     
    def copy_from(self, other):
        np.copyto(self.vec, other.vec)
        return self
    
    def norm(self, order=None):
        return np.linalg.norm(self.vec, ord=order)
        
    def inp(self, other):
        return (self.vec.T @ other.vec)[0, 0]