import numpy as np
import scipy as sp
from utils import linear as lin

class Cone():
    def __init__(self, dim):
        # Dimension properties
        self.dim = dim
        self.grad = lin.Real(self.dim)
        
        self.congr_aux_updated = False
        return
    
    def zeros(self):
        return lin.Real(self.dim)
        
    def get_nu(self):
        return self.dim
    
    def set_init_point(self):
        self.set_point(
            lin.Real(np.ones((self.dim, 1))), 
            lin.Real(np.ones((self.dim, 1)))
        )
        return lin.Real(self.x)
    
    def set_point(self, point, dual=None, a=True):
        self.x = point.data * a
        self.z = dual.data * a
        return
    
    def get_feas(self):
        return np.all(np.greater(self.x, 0)) and np.all(np.greater(self.z, 0))
    
    def get_val(self):
        return -np.sum(np.log(self.x))    
    
    def get_grad(self):
        self.grad.data = -np.reciprocal(self.x)
        return self.grad

    def hess_prod_ip(self, out, H):
        out.data = H.data / (self.x**2)
        return out    

    def invhess_prod_ip(self, out, H):
        out.data = H.data * (self.x**2)
        return out
    
    def third_dir_deriv(self, dir1, dir2=None):
        if dir2 is None:
            return lin.Real(-2 * (dir1.data*dir1.data) / (self.x*self.x*self.x))
        else:
            return lin.Real(-2 * dir1.data * dir2.data / self.x)
    
    def prox(self):
        return np.linalg.norm(self.x * self.z - 1, np.inf)
        
    def nt_prod_ip(self, out, H):
        out.data = H.data * self.z / self.x
        return out
    
    def invnt_prod_ip(self, out, H):
        out.data = H.data * self.x / self.z
        return out        

    def congr_aux(self, A):
        assert not self.congr_aux_updated
        # Check if A matrix is sparse, and build data, col, row arrays if so
        p = len(A)
        n = A[0].data.shape[0]
        
        nnz = 0
        self.A = np.zeros((p, n))
        for i in range(p):
            if sp.sparse.issparse(A[i].data):
                self.A[[i], :] = A[i].data.toarray().T
                nnz += A[i].data.nnz
            else:
                self.A[[i], :] = A[i].data.T
                nnz += n
                
        if nnz < n * p * 0.05:
            self.A = sp.sparse.csr_matrix(self.A)
                
        self.congr_aux_updated = True 

    def hess_congr(self, A):
        if not self.congr_aux_updated:
            self.congr_aux(A)
            
        if sp.sparse.issparse(self.A):
            d = sp.sparse.diags_array(np.reciprocal(self.x.ravel()))
            Ax = d @ self.A.T
            return sp.sparse.csc_matrix(Ax.T @ Ax)
        else:
            Ax = self.A.T / self.x
            return Ax.T @ Ax     

    def invhess_congr(self, A):
        if not self.congr_aux_updated:
            self.congr_aux(A)
            
        if sp.sparse.issparse(self.A):
            d = sp.sparse.diags_array(self.x.ravel())
            Ax = d @ self.A.T
            return sp.sparse.csc_matrix(Ax.T @ Ax)
        else:
            Ax = self.x * self.A.T
            return Ax.T @ Ax     

    def nt_congr(self, A):
        if not self.congr_aux_updated:
            self.congr_aux(A)
            
        if sp.sparse.issparse(self.A):
            d = sp.sparse.diags_array(np.sqrt(self.z / self.x).ravel())
            Ax = d @ self.A.T
            return sp.sparse.csc_matrix(Ax.T @ Ax)
        else:
            Ax = np.sqrt(self.z / self.x) * self.A.T
            return Ax.T @ Ax 

    def invnt_congr(self, A):
        if not self.congr_aux_updated:
            self.congr_aux(A)
            
        if sp.sparse.issparse(self.A):
            d = sp.sparse.diags_array(np.sqrt(self.x / self.z).ravel())
            Ax = d @ self.A.T
            return sp.sparse.csc_matrix(Ax.T @ Ax)
        else:
            Ax = np.sqrt(self.x / self.z) * self.A.T
            return Ax.T @ Ax 

    def sp_invhess_congr(self, A, sp_is, sp_js):
        return self.invnt_congr(A)
        
    def sp_invnt_congr(self, A, sp_is, sp_js):
        return self.invnt_congr(A)