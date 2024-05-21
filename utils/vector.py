import numpy as np
import scipy as sp
import numba as nb
from utils import linear as lin

class Vector():
    # Base vector class represented by a numpy array called 'vec'
    def __init__(self):
        self.vec = None

    def __iadd__(self, other):
        self.vec[:] = sp.linalg.blas.daxpy(other.vec, self.vec, a=1)
        return self
    
    def __isub__(self, other):
        self.vec[:] = sp.linalg.blas.daxpy(other.vec, self.vec, a=-1)
        return self    
    
    def __imul__(self, a):
        self.vec *= a
        return self    
    
    def axpy(self, a, other):
        self.vec[:] = sp.linalg.blas.daxpy(other.vec, self.vec, a=a)
        return self
     
    def copy_from(self, other):
        if isinstance(other, np.ndarray):
            np.copyto(self.vec, other)
        else:
            np.copyto(self.vec, other.vec)
        return self 
        
    def norm(self, order=None):
        return np.linalg.norm(self.vec, ord=order)
    
    def inp(self, other):
        return (self.vec.T @ other.vec)[0, 0]
    
    def fill(self, a):
        self.vec.fill(a)
        return self
    
class Point(Vector):
    def __init__(self, model):
        # Vector class containing the variables involved in a homogeneous 
        # self-dual embedding of a primal-dual conic program
        (n, p, q) = (model.n, model.p, model.q)

        # Initialize vector
        self.vec = np.zeros((n + p + q + q + 2, 1))

        # Build views of vector
        self.xyz = PointXYZ(model, self.vec[:n+p+q])
        self.x   = self.xyz.x
        self.y   = self.xyz.y
        self.z   = self.xyz.z
        self.s   = VecProduct(model.cones, self.vec[n+p+q : n+p+q+q])
        self.tau = self.vec[n+p+q+q : n+p+q+q+1]
        self.kap = self.vec[n+p+q+q+1 : n+p+q+q+2]

        return
    
class PointXYZ(Vector):
    def __init__(self, model, vec=None):
        # Vector class containing the (x,y,z) variables invovled in a 
        # primal-dual conic program
        (n, p, q) = (model.n, model.p, model.q)

        # Initialize vector
        if vec is not None:
            # If view of vector is already given, use that 
            assert vec.size == n + p + q
            self.vec = vec            
        else:
            # Otherwise allocate a new vector
            self.vec = np.zeros((n + p + q, 1))

        # Build views of vector
        self.x = self.vec[:n]
        self.y = self.vec[n : n+p]
        self.z = VecProduct(model.cones, self.vec[n+p : n+p+q])

        return
    
class VecProduct(Vector):
    def __init__(self, cones, vec=None):
        # Vector class of a Cartesian product of vector spaces corresponding
        # to a list of cones
        self.dims  = []
        self.types = []
        for cone_k in cones:
            self.dims.append(cone_k.dim)
            self.types.extend(cone_k.type)
        self.dim = sum(self.dims)

        # Initialize vector
        if vec is not None:
            # If view of vector is already given, use that 
            assert vec.size == self.dim
            self.vec = vec            
        else:
            # Otherwise allocate a new vector
            self.vec = np.zeros((self.dim, 1))

        # Build views of vector
        self.vecs = []
        self.mats = []
        t = 0
        for (dim_k, type_k) in zip(self.dims, self.types):
            self.vecs.append(self.vec[t:t+dim_k])

            if type_k == 'r':
                # Real vector
                self.mats.append(self.vecs[-1])
            elif type_k == 's':
                # Symmetric matrix
                n_k = int(np.sqrt(dim_k))
                self.mats.append(self.vecs[-1].reshape((n_k, n_k)))
            elif type_k == 'h':
                # Hermitian matrix
                n_k = int(np.sqrt(dim_k // 2))
                self.mats.append(self.vecs[-1].reshape((-1, 2)).view(dtype=np.complex128).reshape(n_k, n_k))
            t += dim_k

    def __getitem__(self, key):
        return self.mats[key]