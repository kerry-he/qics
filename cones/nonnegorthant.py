import numpy as np
import scipy as sp
from utils import linear as lin

class Cone():
    def __init__(self, dim):
        # Dimension properties
        self.dim = dim
        self.type = ['r']

        self.Ax = None
        
        self.congr_aux_updated = False
        return
        
    def get_nu(self):
        return self.dim
    
    def set_init_point(self):
        self.set_point(
            np.ones((self.dim, 1)), 
            np.ones((self.dim, 1))
        )
        return self.x
    
    def set_point(self, point, dual=None, a=True):
        self.x = point * a
        self.z = dual * a
        return
    
    def get_feas(self):
        return np.all(np.greater(self.x, 0)) and np.all(np.greater(self.z, 0))
    
    def get_val(self):
        return -np.sum(np.log(self.x))    
    
    def get_grad(self):
        return -np.reciprocal(self.x)

    def hess_prod_ip(self, out, H):
        out[:] = H / (self.x**2)
        return out    

    def invhess_prod_ip(self, out, H):
        out[:] = H * (self.x**2)
        return out
    
    def third_dir_deriv(self, dir1, dir2=None):
        if dir2 is None:
            return -2 * (dir1*dir1) / (self.x*self.x*self.x)
        else:
            return -2 * dir1 * dir2 / self.x
    
    def prox(self):
        return np.linalg.norm(self.x * self.z - 1, np.inf)
        
    def nt_prod_ip(self, out, H):
        out[:] = H * self.z / self.x
        return out
    
    def invnt_prod_ip(self, out, H):
        out[:] = H * self.x / self.z
        return out

    def hess_congr(self, A):
        if sp.sparse.issparse(A):
            if self.Ax is None:
                self.Ax = A.copy()
            self.Ax.data = A.data * np.take(np.reciprocal(self.x), A.indices)
            return self.Ax @ self.Ax.T
        else:
            Ax = A.T / self.x
            return Ax.T @ Ax

    def invhess_congr(self, A):            
        if sp.sparse.issparse(A):
            if self.Ax is None:
                self.Ax = A.copy()
            self.Ax.data = A.data * np.take(self.x, A.indices)
            return self.Ax @ self.Ax.T
        else:
            Ax = self.x * A.T
            return Ax.T @ Ax     

    def nt_congr(self, A):
        if sp.sparse.issparse(A):
            if self.Ax is None:
                self.Ax = A.copy()
            self.Ax.data = A.data * np.take(np.sqrt(self.z / self.x), A.indices)
            return self.Ax @ self.Ax.T               
        else:
            Ax = np.sqrt(self.z / self.x) * A.T
            return Ax.T @ Ax 

    def invnt_congr(self, A):
        if sp.sparse.issparse(A):
            if self.Ax is None:
                self.Ax = A.copy()
            self.Ax.data = A.data * np.take(np.sqrt(self.x / self.z), A.indices)
            return self.Ax @ self.Ax.T
        else:
            Ax = np.sqrt(self.x / self.z) * A.T
            return Ax.T @ Ax 
        
    def invhess_mtx(self):
        return (self.x * self.x).reshape((-1,))
    
    def invnt_mtx(self):
        return (self.x / self.z).reshape((-1,))
    
    def hess_mtx(self):
        return np.reciprocal(self.x * self.x).reshape((-1,))
    
    def nt_mtx(self):
        return (self.z / self.x).reshape((-1,))
    
    def comb_dir(self, out, ds, dz, sigma_mu):
        # Compute the residual for rs where rs is given as the lhs of
        #     Lambda o (W dz + W^-T ds) = -Lambda o Lambda - (W^-T ds_a) o (W dz_a) 
        #                                 + sigma * mu * 1
        # which is rearranged into the form H ds + dz = rs, i.e.,
        #     rs := W^-1 [ Lambda \ (-Lambda o Lambda - (W^-T ds_a) o (W dz_a) + sigma*mu 1) ]
        # See: [Section 5.4]https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf        
        out[:] = (sigma_mu - ds*dz) / self.x - self.z
    
    def step_to_boundary(self, ds, dz):
        # Compute the maximum step alpha in [0, 1] we can take such that 
        #     s + alpha ds >= 0
        #     z + alpha dz >= 0  
        # See: [Section 8.3]https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf

        # Compute rho := ds / s and sig := dz / z
        min_rho = np.min(ds / self.x)
        min_sig = np.min(dz / self.z)

        # Maximum step is given by 
        #     alpha := 1 / max(0, -min(rho), -min(sig))
        # Clamp this step between 0 and 1        
        if min_rho >= 0 and min_sig >= 0:
            return 1.
        else:
            return 1. / max(-min_rho, -min_sig)