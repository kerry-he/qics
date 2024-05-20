import numpy as np
import scipy as sp

from utils import symmetric as sym
from utils import linear as lin
from cones import *

class Model():
    def __init__(self, c, A=None, b=None, G=None, h=None, cones=None, offset=0.0):
        self.n = np.size(c)
        self.p = np.size(b) if (b is not None) else 0
        self.q = np.size(h) if (h is not None) else self.n
    
        self.c = c
        self.A = A if (A is not None) else  np.empty((0, self.n))
        self.b = b if (b is not None) else  np.empty((0, 1))
        self.G = G if (G is not None) else -sp.sparse.identity(self.n)
        self.h = h if (h is not None) else  np.zeros((self.n, 1))
        self.cones = cones

        self.use_G = (G is not None)
        self.use_A = (A is not None) and (A.size > 0)

        self.cone_idxs = build_cone_idxs(self.q, cones)
        self.nu = 1 if (len(cones) == 0) else (1 + sum([cone.get_nu() for cone in cones]))

        self.offset = offset
        
        self.A_T = self.A.T
        self.G_T = self.G.T
        
        if self.use_G:
            self.G_T_vec = op_vec_to_mat(G.T, cones)
        elif self.use_A:
            self.A_vec = op_vec_to_mat(A, cones)

        self.sym = True
        for cone_k in cones:
            self.sym = self.sym and (isinstance(cone_k, nonnegorthant.Cone) or isinstance(cone_k, possemidefinite.Cone))
        
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

def op_mat_to_vec(op_mat):
    # Obtain the matrix representation of the linear map
    #     <A_i1, x_1> + <A_i2, x_1> + ... + <A_in, x_1> = b_i
    # for i=1,...,p.
    
    p = len(op_mat)
    n = op_mat[0].get_vn()
    
    op_vec = np.zeros((p, n))
    for i in range(p):
        op_vec[[i], :] = op_mat[i].to_vec()
        
    if np.count_nonzero(op_vec) < p * n / 0.05:
        op_vec = sp.sparse.csr_array(op_vec)
        
    return op_vec

def op_vec_to_mat(op_vec, cones):
    # Obtain the matrices A_ij for the matrix representation of
    #     <A_i1, x_1> + <A_i2, x_1> + ... + <A_in, x_1> = b_i
    # for i=1,...,p.
    
    p = op_vec.shape[0]
    op_mat = [lin.Vector(cones) for _ in range(p)]
    
    if sp.sparse.issparse(op_vec):
        op_vec_dense = op_vec.toarray()        
    
    for i in range(p):
        if not sp.sparse.issparse(op_vec):
            op_mat[i].from_vec(op_vec[[i], :].T)
        else:
            np.copyto(op_mat[i].vec, op_vec_dense[[i], :].T)
            # op_mat[i].from_vec(op_vec_dense[[i], :].T)
            # op_mat[i].to_sparse()
        
    return op_mat