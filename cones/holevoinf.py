import numpy as np
import scipy as sp
import numba as nb
import math
from utils import symmetric as sym, linear as lin, quantum as quant

class HolevoInf():
    def __init__(self, X_list):
        # Dimension properties
        self.X_list = X_list                # List of quantum states
        self.N, self.n, _ = np.shape(X_list) # Get number of states
        self.vN = sym.vec_dim(self.N)
        
        self.dim = 1 + self.n       # Total dimension of cone

        # Precompute constants
        self.SX_list = np.array([[quant.quantEntropy(Xi)] for Xi in self.X_list])

        # Update flags
        self.feas_updated        = False
        self.grad_updated        = False
        self.hess_aux_updated    = False
        self.invhess_aux_updated = False
        self.dder3_aux_updated   = False

        return
        
    def get_nu(self):
        return 1 + self.n
    
    def set_init_point(self):
        point = np.empty((self.dim, 1))
        point[0] = 1.
        point[1:] = 1. / self.n

        self.set_point(point)

        return point
    
    def set_point(self, point):
        assert np.size(point) == self.dim
        self.point = point

        self.t = point[0]
        self.p = point[1:]
        self.X = np.sum(np.reshape(self.p, (self.n, 1, 1)) * self.X_list, 0)

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

        if any(self.p <= 0):
            self.feas = False
            return self.feas
            
        self.Dx, self.Ux = np.linalg.eigh(self.X)
        self.log_Dx      = np.log(self.Dx)
        self.sum_p = np.sum(self.p)

        entr_X = lin.inp(self.Dx, self.log_Dx)
        entr_p = self.sum_p * np.log(self.sum_p)

        self.z = self.t - (entr_X - entr_p + lin.inp(self.p, self.SX_list))

        self.feas = (self.z > 0)
        return self.feas

    def get_grad(self):
        assert self.feas_updated

        if self.grad_updated:
            return self.grad
        
        log_X = (self.Ux * self.log_Dx) @ self.Ux.T

        self.zi   = np.reciprocal(self.z)
        self.DPhi = np.array([[lin.inp(Xi, log_X)] for Xi in self.X_list]) - np.log(self.sum_p) + self.SX_list

        self.grad     =  np.empty((self.dim, 1))
        self.grad[0]  = -self.zi
        self.grad[1:] =  self.zi * self.DPhi - np.reciprocal(self.p)

        self.grad_updated = True
        return self.grad

    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        irt2 = math.sqrt(0.5)

        D1x_log = D1_log(self.Dx, self.log_Dx)
        sqrt_D1x_log = np.sqrt(sym.mat_to_vec(D1x_log, rt2=1.0))

        # Hessians of quantum relative entropy
        UXU_list = np.empty((self.n, self.vN))
        for i in range(self.n):
            UXU_list[[i], :] = (sym.mat_to_vec(self.Ux.T @ self.X_list[i] @ self.Ux) * sqrt_D1x_log).T

        D2Phi = UXU_list @ UXU_list.T
        D2Phi -= np.reciprocal(self.sum_p)

        # Preparing other required variables
        zi2 = self.zi * self.zi
        invXX = np.diag(np.reciprocal(np.square(self.p[:, 0])))

        self.hess = np.empty((self.dim, self.dim))
        self.hess[0, 0] = zi2
        self.hess[1:, [0]] = -zi2 * self.DPhi
        self.hess[[0], 1:] = self.hess[1:, [0]].T
        self.hess[1:, 1:] = zi2 * np.outer(self.DPhi, self.DPhi) + self.zi * D2Phi + invXX

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

        self.hess_fact = lin.fact(self.hess)
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
            out[:, j] = lin.fact_solve(self.hess_fact, dirs[:, j])

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