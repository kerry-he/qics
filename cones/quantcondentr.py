import numpy as np
from utils import symmetric as sym

class QuantCondEntropy():
    def __init__(self, n, m):
        # Dimension properties
        self.n  = n          # Dimension of system 1
        self.m  = m          # Dimension of system 2
        self.nm = n * m      # Total dimension of bipartite system

        self.vn  = sym.vec_dim(n)           # Dimension of vectorized system 1
        self.vnm = sym.vec_dim(self.nm)     # Dimension of vectorized bipartite system

        self.dim = 1 + self.vnm     # Dimension of the cone

        # Update flags
        self.feas_updated           = False
        self.grad_updated           = False
        self.hess_aux_updated       = False
        self.invhess_aux_updated    = False

        return
        
    def get_nu(self):
        return 1 + self.nm
    
    def get_point(self, point):
        assert np.size(point) == self.dim
        self.point = point

        self.t = point[0]
        self.X = sym.vec_to_mat(point[1:])
        self.Y = sym.p_tr(self.X, 0, (self.n, self.m))

        self.feas_updated           = False
        self.grad_updated           = False
        self.hess_aux_updated       = False
        self.invhess_aux_updated    = False

        return
    
    def get_feas(self):
        if self.feas_updated:
            return self.feas
        
        self.feas_updated = True

        self.Dx, self.Ux = np.linalg.eig(self.X)
        self.Dy, self.Uy = np.linalg.eig(self.Y)

        if any(self.Dx <= 0):
            self.feas = False
            return self.feas
        
        self.log_Dx = np.log(self.Dx)
        self.log_Dy = np.log(self.Dy)

        self.log_X = (self.Ux * self.log_Dx) @ self.Ux.T
        self.log_Y = (self.Uy * self.log_Dy) @ self.Uy.T
        self.log_XY = self.log_X - np.kron(np.eye(self.n), self.log_Y)
        self.z = self.t - sym.inner(self.X, self.log_XY)

        self.feas = (self.z > 0)
        return self.feas
    
    def get_grad(self):
        assert self.feas_updated

        if self.grad_updated:
            return self.grad
        
        self.inv_Dx = np.reciprocal(self.Dx)
        self.inv_X  = (self.Ux * self.inv_Dx) @ self.Ux.T

        self.zi = np.reciprocal(self.z)
        self.DPhi = self.log_XY

        self.grad = np.empty((self.dim, 1))
        self.grad[0] = self.zi
        self.grad[1:] = sym.mat_to_vec(self.zi * self.DPhi - self.inv_X)

        print(self.zi * self.DPhi - self.inv_X)

        self.grad_updated = True
        return self.grad
    
    def update_hessprod_aux(self):
        assert ~self.hess_aux_updated
        assert self.grad_updated

        self.D1x_log = D1_log(self.Dx, self.log_Dx)
        self.D1y_log = D1_log(self.Dy, self.log_Dy)

        self.hess_aux_updated = True

        return

    def hess_prod(self, dirs):
        assert self.grad_updated
        if ~self.hess_aux_updated:
            self.update_hessprod_aux()

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        for j in range(p):
            Ht = dirs[0, j]
            HX = sym.vec_to_mat(dirs[1:, j])
            HY = sym.p_tr(HX, 0, (self.n, self.m))

            UxHxUx = self.Ux.T @ HX @ self.Ux
            UyHyUy = self.Uy.T @ HY @ self.Uy

            # Hessian product of conditional entropy
            D2PhiH = self.Ux @ (self.D1x_log * UxHxUx) @ self.Ux.T
            D2PhiH -= np.kron(np.eye(self.n), self.Uy @ (self.D1y_log * UyHyUy) @ self.Uy.T)

            # Hessian product of barrier function
            out[0, j] = (Ht - sym.inner(self.DPhi, HX)) * self.zi * self.zi
            temp = -self.DPhi * out[0, j] + D2PhiH * self.zi + self.inv_X @ HX @ self.inv_X
            out[1:, [j]] = sym.mat_to_vec(temp)

            print(temp)

        return out


def D1_log(D, log_D):
    eps = np.finfo(np.float64).eps
    rteps = np.sqrt(eps)

    n = np.size(D)
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