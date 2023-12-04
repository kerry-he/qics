import numpy as np
import scipy as sp
import numba as nb
import math
from utils import symmetric as sym, linear as lin

class QuantMutualInf():
    def __init__(self, V, no):
        # Dimension properties
        self.V = V                  # Define channel using Stinespring representation
        N, self.ni = np.shape(V)    # Get input dimension
        self.no = no                # Get output dimension
        self.ne = N // no           # Get environment dimension
        
        self.vni = sym.vec_dim(self.ni)     # Get input vector dimension
        self.vno = sym.vec_dim(self.no)     # Get output vector dimension
        self.vne = sym.vec_dim(self.ne)     # Get environment vector dimension

        self.dim = 1 + self.vni     # Total dimension of cone

        # Build linear maps of quantum channels
        self.N  = np.zeros((self.vno, self.vni))  # Quantum channel
        self.Nc = np.zeros((self.vne, self.vni))  # Complementary channel
        k = -1
        for j in range(self.ni):
            for i in range(j + 1):
                k += 1
            
                VHV = np.outer(self.V[:, i], self.V[:, j])
                if i != j:
                    VHV = VHV + VHV.T
                    VHV *= math.sqrt(0.5)
                
                trE_VHV = sym.p_tr(VHV, 1, (self.no, self.ne))
                trB_VHV = sym.p_tr(VHV, 0, (self.no, self.ne))

                self.N[:, [k]]  = sym.mat_to_vec(trE_VHV)
                self.Nc[:, [k]] = sym.mat_to_vec(trB_VHV)
        self.tr = sym.mat_to_vec(np.eye(self.ni)).T
        

        # Update flags
        self.feas_updated        = False
        self.grad_updated        = False
        self.hess_aux_updated    = False
        self.invhess_aux_updated = False
        self.dder3_aux_updated   = False

        return
        
    def get_nu(self):
        return 1 + self.ni
    
    def set_init_point(self):
        point = np.empty((self.dim, 1))
        point[0] = 1.
        point[1:] = sym.mat_to_vec(np.eye(self.ni)) / self.ni

        self.set_point(point)

        return point
    
    def set_point(self, point):
        assert np.size(point) == self.dim
        self.point = point

        self.t   = point[0]
        self.X   = sym.vec_to_mat(point[1:])
        self.NX  = sym.vec_to_mat(self.N @ point[1:])
        self.NcX = sym.vec_to_mat(self.Nc @ point[1:])
        self.trX = np.trace(self.X)

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

        self.Dx, self.Ux     = np.linalg.eigh(self.X)
        if any(self.Dx <= 0):
            self.feas = False
            return self.feas

        self.Dnx, self.Unx   = np.linalg.eigh(self.NX)
        self.Dncx, self.Uncx = np.linalg.eigh(self.NcX)
        
        self.log_Dx   = np.log(self.Dx)
        self.log_Dnx  = np.log(self.Dnx)
        self.log_Dncx = np.log(self.Dncx)
        self.log_trX  = np.log(self.trX)

        entr_X   = lin.inp(self.Dx, self.log_Dx)
        entr_NX  = lin.inp(self.Dnx, self.log_Dnx)
        entr_NcX = lin.inp(self.Dncx, self.log_Dncx)
        entr_trX = self.trX * self.log_trX

        self.z = self.t - (entr_X + entr_NX - entr_NcX - entr_trX)

        self.feas = (self.z > 0)
        return self.feas
    
    def get_grad(self):
        assert self.feas_updated

        if self.grad_updated:
            return self.grad
        
        log_X   = (self.Ux * self.log_Dx) @ self.Ux.T
        log_NX  = (self.Unx * self.log_Dnx) @ self.Unx.T
        log_NcX = (self.Uncx * self.log_Dncx) @ self.Uncx.T
        self.log_X      = sym.mat_to_vec(log_X)
        self.N_log_NX   = self.N.T @ sym.mat_to_vec(log_NX)
        self.Nc_log_NcX = self.Nc.T @ sym.mat_to_vec(log_NcX)
        self.tr_log_trX = self.tr.T * self.log_trX

        self.inv_Dx = np.reciprocal(self.Dx)
        self.inv_X  = (self.Ux * self.inv_Dx) @ self.Ux.T

        self.zi   = np.reciprocal(self.z)
        self.DPhi = self.log_X + self.N_log_NX - self.Nc_log_NcX - self.tr_log_trX

        self.grad     =  np.empty((self.dim, 1))
        self.grad[0]  = -self.zi
        self.grad[1:] =  self.zi * self.DPhi - sym.mat_to_vec(self.inv_X)

        self.grad_updated = True
        return self.grad
    
    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        irt2 = math.sqrt(0.5)

        self.D1x_log   = D1_log(self.Dx, self.log_Dx)
        self.D1nx_log  = D1_log(self.Dnx, self.log_Dnx)
        self.D1ncx_log = D1_log(self.Dncx, self.log_Dncx)

        # Hessians of quantum relative entropy
        D2PhiX = np.empty((self.vni, self.vni))
        invXX  = np.empty((self.vni, self.vni))

        k = 0
        for j in range(self.ni):
            for i in range(j + 1):
                # D2Phi
                UHU = np.outer(self.Ux[i, :], self.Ux[j, :])
                if i != j:
                    UHU = UHU + UHU.T
                    UHU *= irt2
                temp = self.Ux @ (self.D1x_log * UHU) @ self.Ux.T
                D2PhiX[:, [k]] = sym.mat_to_vec(temp)

                # invXX
                temp = np.outer(self.inv_X[i, :], self.inv_X[j, :])
                if i != j:
                    temp = temp + temp.T
                    temp *= irt2
                invXX[:, [k]] = sym.mat_to_vec(temp)

                k += 1

        sqrt_D1nx_log = np.sqrt(sym.mat_to_vec(self.D1nx_log, 1.0))
        UnUnN = np.empty((self.vni, self.vno))
        for k in range(self.vni):
            N = sym.vec_to_mat(self.N[:, [k]])
            UnNUn = self.Unx.T @ N @ self.Unx
            UnUnN[[k], :] = (sym.mat_to_vec(UnNUn) * sqrt_D1nx_log).T
        D2PhiNX = UnUnN @ UnUnN.T

        sqrt_D1ncx_log = np.sqrt(sym.mat_to_vec(self.D1ncx_log, 1.0))
        UcnUcnN = np.empty((self.vni, self.vne))
        for k in range(self.vni):
            Nc = sym.vec_to_mat(self.Nc[:, [k]])
            UcnNcUcn = self.Uncx.T @ Nc @ self.Uncx
            UcnUcnN[[k], :] = (sym.mat_to_vec(UcnNcUcn) * sqrt_D1ncx_log).T
        D2PhiNcX = UcnUcnN @ UcnUcnN.T

        # Preparing other required variables
        zi2 = self.zi * self.zi
        D2Phi = D2PhiX + D2PhiNX - D2PhiNcX - self.tr.T @ self.tr / self.trX

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