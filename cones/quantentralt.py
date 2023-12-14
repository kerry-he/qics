import numpy as np
import scipy as sp
import numba as nb
import math
from utils import symmetric as sym

class QuantEntropy():
    def __init__(self, n):
        # Dimension properties
        self.n = n                          # Side dimension of system
        self.vn = sym.vec_dim(self.n)       # Vector dimension of system
        self.dim = 2 + self.vn              # Total dimension of cone

        # Update flags
        self.feas_updated        = False
        self.grad_updated        = False
        self.hess_aux_updated    = False
        self.invhess_aux_updated = False
        self.dder3_aux_updated   = False

        return
        
    def get_nu(self):
        return 2 + self.n
    
    def set_init_point(self):
        point = np.empty((self.dim, 1))
        point[0] = 1.
        point[1] = 1.
        point[2:] = sym.mat_to_vec(np.eye(self.n)) / self.n

        self.set_point(point)

        return point
    
    def set_point(self, point):
        assert np.size(point) == self.dim
        self.point = point

        self.t = point[0, 0]
        self.y = point[1, 0]
        self.X = sym.vec_to_mat(point[2:])

        self.feas_updated        = False
        self.grad_updated        = False
        self.hess_aux_updated    = False
        self.invhess_aux_updated = False
        self.dder3_aux_updated   = False

        return
    
    def get_feas(self):
        if self.feas_updated:
            return self.feas
        
        self.feas_updated = True

        self.Dx, self.Ux = np.linalg.eigh(self.X)
        self.Xy = self.X / self.y
        self.Dxy = self.Dx / self.y

        if any(self.Dx <= 0) or self.y <= 0:
            self.feas = False
            return self.feas
        
        self.log_Dxy = np.log(self.Dxy)
        self.log_Xy  = (self.Ux * self.log_Dxy) @ self.Ux.T
        self.Phi     = sym.inner(self.Xy, self.log_Xy)
        self.z       = self.t - self.y * self.Phi

        self.feas = (self.z > 0)
        return self.feas
    
    def get_grad(self):
        assert self.feas_updated

        if self.grad_updated:
            return self.grad
        
        self.inv_Dx = np.reciprocal(self.Dx)
        self.inv_X  = (self.Ux * self.inv_Dx) @ self.Ux.T

        self.zi    = np.reciprocal(self.z)
        self.yi    = np.reciprocal(self.y)
        self.DPhi  = self.log_Xy + np.eye(self.n)
        self.sigma = self.Phi - sym.inner(self.DPhi, self.X / self.y)

        self.grad     =  np.empty((self.dim, 1))
        self.grad[0]  = -self.zi
        self.grad[1]  =  self.zi * self.sigma - self.yi 
        self.grad[2:] =  sym.mat_to_vec(self.zi * self.DPhi - self.inv_X)

        self.grad_updated = True
        return self.grad
    
    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        self.D1xy_log = D1_log(self.Dxy, self.log_Dxy)

        self.hess_aux_updated = True

        return

    def hess_prod(self, dirs):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        for j in range(p):
            Ht = dirs[0, j]
            Hy = dirs[1, j]
            Hx = sym.vec_to_mat(dirs[2:, [j]])

            chi = self.zi * (Ht - Hy*self.sigma - sym.inner(Hx, self.log_Xy) - np.trace(Hx))
            xi  = self.yi * (Hx - Hy * self.Xy)

            UxXiUx = self.Ux.T @ xi @ self.Ux
            D2PhiH = self.Ux @ (self.D1xy_log * UxXiUx) @ self.Ux.T

            # Hessian product of barrier function
            out[0, j] = self.zi * chi
            out[1, j] = -self.zi * self.sigma * chi - self.zi * sym.inner(D2PhiH, self.Xy) + Hy*self.yi*self.yi
            temp = -self.zi * chi * self.DPhi + self.zi * D2PhiH + self.inv_X @ Hx @ self.inv_X
            out[2:, [j]] = sym.mat_to_vec(temp)

        return out
    
    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated

        D1x_inv = np.reciprocal(np.outer(self.Dx, self.Dx))
        self.D1xy_comb_inv = 1 / (self.D1xy_log*self.zi*self.yi + D1x_inv)

        UxDPhiUx = self.Ux.T @ self.DPhi @ self.Ux
        self.alpha = self.Ux @ (self.D1xy_comb_inv * UxDPhiUx) @ self.Ux.T

        temp = np.diag(self.D1xy_comb_inv) * np.diag(self.D1xy_log) * self.Dx
        self.gamma = self.yi*self.yi*self.zi * self.Ux @ np.diag(temp) @ self.Ux.T

        self.k1 = self.z*self.z + sym.inner(self.DPhi, self.alpha)
        self.k2 = self.sigma + sym.inner(self.DPhi, self.gamma)
        self.k3 = self.yi*self.yi + self.yi * sym.inner(self.gamma, self.inv_X)

        self.invhess_aux_updated = True

        return

    def invhess_prod(self, dirs):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        for j in range(p):
            Ht = dirs[0, j]
            Hy = dirs[1, j]
            Hx = sym.vec_to_mat(dirs[2:, [j]])

            UxHxUx = self.Ux.T @ Hx @ self.Ux
            temp = self.Ux @ (self.D1xy_comb_inv * UxHxUx) @ self.Ux.T

            out[1, j] = (self.k2 * Ht + Hy + sym.inner(Hx, self.gamma)) / self.k3
            out[0, j] = self.k1 * Ht + self.k2 * out[1, j] + sym.inner(self.alpha, Hx)
            temp = Ht * self.alpha + out[1, j] * self.gamma + temp
            out[2:, [j]] = sym.mat_to_vec(temp)

        return out

@nb.njit
def D1_log(D, log_D):
    eps = np.finfo(np.float64).eps
    rteps = np.sqrt(eps)

    n = D.size
    D1 = np.empty((n, n))
    
    for j in range(n):
        for i in range(j):
            d_ij = D[i] - D[j]
            if abs(d_ij) < rteps:
                D1[i, j] = 2 / (D[i] + D[j])
            else:
                D1[i, j] = (log_D[i] - log_D[j]) / d_ij
            D1[j, i] = D1[i, j]

        D1[j, j] = np.reciprocal(D[j])

    return D1
