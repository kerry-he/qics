import numpy as np
import scipy as sp
import numba as nb
import math
from utils import symmetric as sym, linear as lin

class QuantMutualInf():
    def __init__(self, V, no):
        # Dimension properties
        self.V = V                  # Define channel using Stinespring representation
        self.ni, N = np.shape(V)    # Get input dimension
        self.no = no                # Get output dimension
        self.ne = N // no           # Get environment dimension
        
        self.vni = sym.vec_dim(self.ni)     # Get input vector dimension
        self.vno = sym.vec_dim(self.no)     # Get output vector dimension
        self.vne = sym.vec_dim(self.ne)     # Get environment vector dimension

        self.dim = 2 + self.vni     # Total dimension of cone

        # Update flags
        self.feas_updated        = False
        self.grad_updated        = False
        self.hess_aux_updated    = False
        self.invhess_aux_updated = False
        self.dder3_aux_updated   = False

        return
        
    def get_nu(self):
        return 2 + self.ni
    
    def set_init_point(self):
        point = np.empty((self.dim, 1))
        point[0] = 1.
        point[1:] = sym.mat_to_vec(np.eye(self.nin)) / self.nin

        self.set_point(point)

        return point
    
    def set_point(self, point):
        assert np.size(point) == self.dim
        self.point = point

        self.t = point[0]
        self.X = sym.vec_to_mat(point[1:])
        self.Y = sym.vec_to_mat(self.Nc @ point[1:])

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

        if any(self.Dx <= 0):
            self.feas = False
            return self.feas
        
        self.log_Dx = np.log(self.Dx)
        self.log_Dy = np.log(self.Dy)

        self.z = self.t - (lin.inp(self.Dx, self.log_Dx) - lin.inp(self.Dy, self.log_Dy))

        self.feas = (self.z > 0)
        return self.feas
    
    def get_grad(self):
        assert self.feas_updated

        if self.grad_updated:
            return self.grad
        
        self.log_X = (self.Ux * self.log_Dx) @ self.Ux.T
        self.log_Y = (self.Uy * self.log_Dy) @ self.Uy.T
        self.Nc_log_Y = sym.vec_to_mat(self.Nc.T @ sym.mat_to_vec(self.log_Y))

        self.inv_Dx = np.reciprocal(self.Dx)
        self.inv_X  = (self.Ux * self.inv_Dx) @ self.Ux.T

        self.UyXUy = self.Uy.T @ self.X @ self.Uy

        self.zi   = np.reciprocal(self.z)
        self.DPhi = self.log_X - self.Nc_log_Y

        self.grad     = np.empty((self.dim, 1))
        self.grad[0]  = -self.zi
        self.grad[1:] = sym.mat_to_vec(self.zi * self.DPhi - self.inv_X)

        self.grad_updated = True
        return self.grad
    
    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        irt2 = math.sqrt(0.5)

        self.D1x_log = D1_log(self.Dx, self.log_Dx)
        self.D1y_log = D1_log(self.Dy, self.log_Dy)

        # Hessians of quantum relative entropy
        D2PhiX = np.empty((self.vnin, self.vnin))
        D2PhiY = np.empty((self.vnenv, self.vnenv))
        invXX  = np.empty((self.vnin, self.vnin))

        k = 0
        for j in range(self.nin):
            for i in range(j + 1):
                # D2Phi
                UxHUx = np.outer(self.Ux[i, :], self.Ux[j, :])
                if i != j:
                    UxHUx = UxHUx + UxHUx.T
                    UxHUx *= irt2
                temp = self.Ux @ (self.D1x_log * UxHUx) @ self.Ux.T
                D2PhiX[:, [k]] = sym.mat_to_vec(temp)

                # invXX
                temp = np.outer(self.inv_X[i, :], self.inv_X[j, :])
                if i != j:
                    temp = temp + temp.T
                    temp *= irt2
                invXX[:, [k]] = sym.mat_to_vec(temp)

                k += 1

        k = 0
        for j in range(self.nenv):
            for i in range(j + 1):
                UyHUy = np.outer(self.Uy[i, :], self.Uy[j, :])
                if i != j:
                    UyHUy = UyHUy + UyHUy.T
                    UyHUy *= irt2
                temp = self.Uy @ (self.D1y_log * UyHUy) @ self.Uy.T
                D2PhiY[:, [k]] = sym.mat_to_vec(temp)                

                k += 1

        # Preparing other required variables
        zi2 = self.zi * self.zi
        DPhi_vec = sym.mat_to_vec(self.DPhi)
        D2Phi = D2PhiX - self.Nc.T @ D2PhiY @ self.Nc

        self.hess = np.empty((self.dim, self.dim))
        self.hess[0, 0] = zi2
        self.hess[1:, [0]] = -zi2 * DPhi_vec
        self.hess[[0], 1:] = self.hess[1:, [0]].T
        self.hess[1:, 1:] = zi2 * np.outer(DPhi_vec, DPhi_vec) + self.zi * D2Phi + invXX

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