import numpy as np
from utils import symmetric as sym

class Point():
    def __init__(self, model):

        self.model = model

        (n, p, q) = (model.n, model.p, model.q)

        self.vec = np.zeros((n + p + q + q + 2, 1))

        # self.x     = self.vec[:n]
        self.y     = self.vec[n : n+p]
        # self.z     = self.vec[n+p : n+p+q]
        # self.s     = self.vec[n+p+q : n+p+q+q]
        self.tau   = self.vec[n+p+q+q : n+p+q+q+1]
        self.kappa = self.vec[n+p+q+q+1 : n+p+q+q+2]

        self.y     = np.zeros((p, 1))
        self.tau   = 0.
        self.kappa = 0.
        
        self.X = sym.vec_to_mat(self.x)
        self.S = sym.vec_to_mat(self.s)
        self.Z = sym.vec_to_mat(self.z)

        return
    
    def __add__(self, other):
        point = Point(self.model)
        
        point.X = self.X + other.X
        point.y = self.y + other.y
        point.Z = self.Z + other.Z
        point.S = self.S + other.S
        point.tau = self.tau + other.tau
        point.kappa = self.kappa + other.kappa

        return point
    
    def __sub__(self, other):
        point = Point(self.model)

        point.X = self.X - other.X
        point.y = self.y - other.y
        point.Z = self.Z - other.Z
        point.S = self.S - other.S
        point.tau = self.tau - other.tau
        point.kappa = self.kappa - other.kappa

        return point
    
    def __mul__(self, other):
        point = Point(self.model)

        point.X = self.X * other
        point.y = self.y * other
        point.Z = self.Z * other
        point.S = self.S * other
        point.tau = self.tau * other
        point.kappa = self.kappa * other

        return point
    
    __rmul__ = __mul__
    
    def norm(self):
        return np.sqrt(
            np.sum(self.X * self.X) + 
            np.sum(self.y * self.y) + 
            np.sum(self.Z * self.Z) +
            np.sum(self.S * self.S) +
            self.tau * self.tau + self.kappa * self.kappa
        )