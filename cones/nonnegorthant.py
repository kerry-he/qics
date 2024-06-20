import numpy as np
import scipy as sp
from utils import linear as lin

class Cone():
    def __init__(self, dim):
        # Dimension properties
        self.dim = dim
        self.type = 'r'

        self.Ax = None

        self.congr_aux_updated = False
        return

    def get_nu(self):
        return self.dim
    
    def get_init_point(self, out):
        self.set_point(
            np.ones((self.dim, 1)), 
            np.ones((self.dim, 1))
        )

        out[:] = self.x
        return out

    def set_point(self, point, dual=None, a=True):
        self.x = point * a
        self.z = dual * a
        return

    def get_feas(self):
        return np.all(np.greater(self.x, 0)) and np.all(np.greater(self.z, 0))

    def get_val(self):
        return -np.sum(np.log(self.x))    

    def get_grad(self, out):
        out[:] = -np.reciprocal(self.x)
        return out

    def hess_prod_ip(self, out, H):
        out[:] = H / (self.x**2)
        return out    

    def invhess_prod_ip(self, out, H):
        out[:] = H * (self.x**2)
        return out

    def hess_mtx(self):
        return np.reciprocal(self.x * self.x).reshape((-1,))

    def invhess_mtx(self):
        return (self.x * self.x).reshape((-1,))    

    def hess_congr(self, A):
        return self.base_congr(A, np.reciprocal(self.x))

    def invhess_congr(self, A):
        return self.base_congr(A, self.x)

    def base_congr(self, A, x):
        if sp.sparse.issparse(A):
            if self.Ax is None:
                self.Ax = A.copy()
            self.Ax.data = A.data * np.take(x, A.indices)
            return self.Ax @ self.Ax.T
        else:
            Ax = x * A.T
            return Ax.T @ Ax

    def third_dir_deriv_axpy(self, out, dir1, dir2=None, a=True):
        if dir2 is None:
            out -= 2 * a * dir1 * dir1 / (self.x*self.x*self.x)
            return out
        else:
            out -= 2 * a * dir1 * dir2 / self.x
            return out

    def prox(self):
        return np.linalg.norm(self.x * self.z - 1, np.inf)

    # ========================================================================
    # Functions specific to symmetric cones for NT scaling
    # ========================================================================
    # We want the NT scaling point w and scaled variable lambda such that
    #     H(w) s = z  <==> lambda := W^-T s = W z
    # where H(w) = W^T W. For the nonnegative orthant, we have
    #     w      = s.^1/2 ./ z.^1/2
    #     lambda = z.^1/2 .* s.^1/2
    # Also, we have the linear transformations given by
    #     H(w) ds = ds .* s ./ z
    #     W^-T ds = ds .* z.^1/2 ./ s.^1/2
    #     W    dz = dz .* s.^1/2 ./ z.^1/2
    # See: [Section 4.1]https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf

    def nt_prod_ip(self, out, H):
        out[:] = H * self.z / self.x
        return out

    def invnt_prod_ip(self, out, H):
        out[:] = H * self.x / self.z
        return out

    def nt_mtx(self):
        return (self.z / self.x).reshape((-1,))
    
    def invnt_mtx(self):
        return (self.x / self.z).reshape((-1,))

    def nt_congr(self, A):
        return self.base_congr(A, np.sqrt(self.z / self.x))

    def invnt_congr(self, A):
        return self.base_congr(A, np.sqrt(self.x / self.z))    

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