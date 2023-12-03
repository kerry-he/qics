import numpy as np
import scipy as sp
import numba as nb
import math
from utils import symmetric as sym

class QuantRelEntropy():
    def __init__(self, n):
        # Dimension properties
        self.n = n                          # Side dimension of system
        self.vn = sym.vec_dim(self.n)       # Vector dimension of system
        self.dim = 1 + 2 * self.vn          # Total dimension of cone

        self.idx_X = slice(1, 1 + self.vn)
        self.idx_Y = slice(1 + self.vn, 1 + 2 * self.vn)

        # Update flags
        self.feas_updated        = False
        self.grad_updated        = False
        self.hess_aux_updated    = False
        self.invhess_aux_updated = False
        self.dder3_aux_updated   = False

        return
        
    def get_nu(self):
        return 1 + 2 * self.n
    
    def set_init_point(self):
        point = np.empty((self.dim, 1))
        point[0] = 1.
        point[self.idx_X] = sym.mat_to_vec(np.eye(self.n)) / self.n
        point[self.idx_Y] = sym.mat_to_vec(np.eye(self.n)) / self.n

        self.set_point(point)

        return point
    
    def set_point(self, point):
        assert np.size(point) == self.dim
        self.point = point

        self.t = point[0]
        self.X = sym.vec_to_mat(point[self.idx_X])
        self.Y = sym.vec_to_mat(point[self.idx_Y])

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
        self.Dy, self.Uy = np.linalg.eigh(self.Y)

        if any(self.Dx <= 0) or any(self.Dy <= 0):
            self.feas = False
            return self.feas
        
        self.log_Dx = np.log(self.Dx)
        self.log_Dy = np.log(self.Dy)

        self.log_X = (self.Ux * self.log_Dx) @ self.Ux.T
        self.log_Y = (self.Uy * self.log_Dy) @ self.Uy.T
        self.log_XY = self.log_X - self.log_Y
        self.z = self.t - sym.inner(self.X, self.log_XY)

        self.feas = (self.z > 0)
        return self.feas
    
    def get_grad(self):
        assert self.feas_updated

        if self.grad_updated:
            return self.grad
        
        self.inv_Dx = np.reciprocal(self.Dx)
        self.inv_Dy = np.reciprocal(self.Dy)
        self.inv_X  = (self.Ux * self.inv_Dx) @ self.Ux.T
        self.inv_Y  = (self.Uy * self.inv_Dy) @ self.Uy.T

        self.D1y_log = D1_log(self.Dy, self.log_Dy)

        self.UyXUy = self.Uy.T @ self.X @ self.Uy

        self.zi    = np.reciprocal(self.z)
        self.DPhiX = self.log_XY + np.eye(self.n)
        self.DPhiY = -self.Uy @ (self.D1y_log * self.UyXUy) @ self.Uy.T

        self.grad             = np.empty((self.dim, 1))
        self.grad[0]          = -self.zi
        self.grad[self.idx_X] = sym.mat_to_vec(self.zi * self.DPhiX - self.inv_X)
        self.grad[self.idx_Y] = sym.mat_to_vec(self.zi * self.DPhiY - self.inv_Y)

        self.grad_updated = True
        return self.grad
    
    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        irt2 = math.sqrt(0.5)

        self.D1x_log = D1_log(self.Dx, self.log_Dx)
        self.D2y_log = D2_log(self.Dy, self.D1y_log)

        # Hessians of quantum relative entropy
        D2PhiXX = np.empty((self.vn, self.vn))
        D2PhiXY = np.empty((self.vn, self.vn))
        D2PhiYY = np.empty((self.vn, self.vn))

        invXX = np.empty((self.vn, self.vn))
        invYY = np.empty((self.vn, self.vn))        

        k = 0
        for j in range(self.n):
            for i in range(j + 1):
                # D2PhiXX
                UxHUx = np.outer(self.Ux[i, :], self.Ux[j, :])
                if i != j:
                    UxHUx = UxHUx + UxHUx.T
                    UxHUx *= irt2
                temp = self.Ux @ (self.D1x_log * UxHUx) @ self.Ux.T
                D2PhiXX[:, [k]] = sym.mat_to_vec(temp)

                # D2PhiXY
                UyHUy = np.outer(self.Uy[i, :], self.Uy[j, :])
                if i != j:
                    UyHUy = UyHUy + UyHUy.T
                    UyHUy *= irt2
                temp = -self.Uy @ (self.D1y_log * UyHUy) @ self.Uy.T
                D2PhiXY[:, [k]] = sym.mat_to_vec(temp)

                # D2PhiYY
                temp = -scnd_frechet(self.D2y_log, self.Uy, UyHUy, self.UyXUy)
                D2PhiYY[:, [k]] = sym.mat_to_vec(temp)

                # invXX and invYY
                temp = np.outer(self.inv_X[i, :], self.inv_X[j, :])
                if i != j:
                    temp = temp + temp.T
                    temp *= irt2
                invXX[:, [k]] = sym.mat_to_vec(temp)

                temp = np.outer(self.inv_Y[i, :], self.inv_Y[j, :])
                if i != j:
                    temp = temp + temp.T
                    temp *= irt2
                invYY[:, [k]] = sym.mat_to_vec(temp)                

                k += 1

        # Preparing other required variables
        zi2 = self.zi * self.zi
        DPhiX_vec = sym.mat_to_vec(self.DPhiX)
        DPhiY_vec = sym.mat_to_vec(self.DPhiY)

        self.hess = np.empty((self.dim, self.dim))
        self.hess[0, 0] = zi2
        self.hess[self.idx_X, [0]] = -zi2 * DPhiX_vec
        self.hess[self.idx_Y, [0]] = -zi2 * DPhiY_vec 
        self.hess[[0], self.idx_X] = self.hess[self.idx_X, [0]].T
        self.hess[[0], self.idx_Y] = self.hess[self.idx_Y, [0]].T

        self.hess[self.idx_X, self.idx_Y] = zi2 * np.outer(DPhiX_vec, DPhiY_vec) + self.zi * D2PhiXY
        self.hess[self.idx_Y, self.idx_X] = self.hess[self.idx_X, self.idx_Y].T

        self.hess[self.idx_X, self.idx_X] = zi2 * np.outer(DPhiX_vec, DPhiX_vec) + self.zi * D2PhiXX + invXX
        self.hess[self.idx_Y, self.idx_Y] = zi2 * np.outer(DPhiY_vec, DPhiY_vec) + self.zi * D2PhiYY + invYY

        self.hess_aux_updated = True

        return

    def hess_prod(self, dirs):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        return self.hess @ dirs
    
    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated

        # try:
        #     self.hess_cho = sp.linalg.cho_factor(self.hess)
        # except np.linalg.LinAlgError:
        #     self.hess_cho = None
        #     self.hess_lu = sp.linalg.lu_factor(self.hess)

        self.hess_inv = np.linalg.inv(self.hess)

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

        # for j in range(p):
        #     H = dirs[:, j]
        #     out[:, j] = sp.linalg.lu_solve(self.hess_lu, H) if self.hess_cho is None else sp.linalg.cho_solve(self.hess_cho, H)

        out = self.hess_inv @ dirs

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

@nb.njit
def D2_log(D, D1):
    eps = np.finfo(np.float64).eps
    rteps = np.sqrt(eps)

    n = D.size
    D2 = np.zeros((n, n, n))

    for k in range(n):
        for j in range(k + 1):
            for i in range(j + 1):
                d_jk = D[j] - D[k]
                if abs(d_jk) < rteps:
                    d_ij = D[i] - D[j]
                    if abs(d_ij) < rteps:
                        t = ((3 / (D[i] + D[j] + D[k]))**2) / -2
                    else:
                        t = (D1[i, j] - D1[j, k]) / d_ij
                else:
                    t = (D1[i, j] - D1[i, k]) / d_jk

                D2[i, j, k] = t
                D2[i, k, j] = t
                D2[j, i, k] = t
                D2[j, k, i] = t
                D2[k, i, j] = t
                D2[k, j, i] = t

    return D2

def scnd_frechet(D2, U, UHU, UXU):
    if D2.shape[0] <= 40:
        return scnd_frechet_single(D2, U, UHU, UXU)
    else:
        return scnd_frechet_parallel(D2, U, UHU, UXU)
    
@nb.njit
def scnd_frechet_single(D2, U, UHU, UXU):
    n = U.shape[0]
    out = np.empty((n, n))

    D2_UXU = D2 * UXU

    for k in range(n):
        out[:, k] = np.ascontiguousarray(D2_UXU[k, :, :]) @ np.ascontiguousarray(UHU[k, :])

    out = out + out.T
    out = U @ out @ U.T

    return out

@nb.njit(parallel=True)
def scnd_frechet_parallel(D2, U, UHU, UXU):
    n = U.shape[0]
    out = np.empty((n, n))

    for k in nb.prange(n):
        D2_UXU = D2[k, :, :] * UXU
        out[:, k] = D2_UXU @ UHU[k, :]

    out = out + out.T
    out = U @ out @ U.T

    return out