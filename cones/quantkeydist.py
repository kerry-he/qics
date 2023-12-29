import numpy as np
import scipy as sp
import numba as nb
import math
from utils import symmetric as sym
from utils import linear    as lin
from utils import mtxgrad   as mgrad

class QuantKeyDist():
    def __init__(self, K_list, Z_list):
        # Dimension properties
        self.K_list = K_list                    # Get K Kraus operators
        self.Z_list = Z_list                    # Get Z Kraus operators
        self.no, self.ni = np.shape(K_list[0])  # Get input and output dimension
        
        self.vni = sym.vec_dim(self.ni)     # Get input vector dimension
        self.vno = sym.vec_dim(self.no)     # Get output vector dimension

        self.dim = 1 + self.vni             # Total dimension of cone

        # Reduce systems
        KK = np.zeros((self.no, self.no))
        for K in self.K_list:
            KK += K @ K.T
        ZKKZ = np.zeros((self.no, self.no))
        for Z in self.Z_list:
            ZKKZ += Z @ KK @ Z.T

        Dkk, Ukk     = np.linalg.eigh(KK)
        Dzkkz, Uzkkz = np.linalg.eigh(ZKKZ)

        KKnzidx   = np.where(Dkk > np.finfo(Dkk.dtype).eps)[0]
        ZKKZnzidx = np.where(Dzkkz > np.finfo(Dzkkz.dtype).eps)[0]

        Qkk   = Ukk[:, KKnzidx]
        Qzkkz = Ukk[:, ZKKZnzidx]

        self.nk   = np.size(KKnzidx)
        self.nzk  = np.size(ZKKZnzidx)
        self.vnk  = sym.vec_dim(self.nk)
        self.vnzk = sym.vec_dim(self.nzk)

        # Build linear maps of quantum channels
        self.K  = np.zeros((self.vnk, self.vni))
        self.ZK = np.zeros((self.vnzk, self.vni))

        k = -1
        for j in range(self.ni):
            for i in range(j + 1):
                k += 1
                
                KHK = np.zeros((self.no, self.no))
                for K in self.K_list:
                    KHK += np.outer(K[:, i], K[:, j])

                if i != j:
                    KHK = KHK + KHK.T
                    KHK *= math.sqrt(0.5)

                self.K[:, [k]] = sym.mat_to_vec(Qkk.T @ KHK @ Qkk)
                
                ZKHKZ = np.zeros((self.no, self.no))
                for Z in self.Z_list:
                    ZKHKZ += Z @ KHK @ Z.T
                self.ZK[:, [k]] = sym.mat_to_vec(Qzkkz.T @ ZKHKZ @ Qzkkz)
        

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
        self.KX  = sym.vec_to_mat(self.K @ point[1:])
        self.ZKX = sym.vec_to_mat(self.ZK @ point[1:])

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

        self.Dkx, self.Ukx   = np.linalg.eigh(self.KX)
        self.Dzkx, self.Uzkx = np.linalg.eigh(self.ZKX)
        
        self.log_Dkx  = np.log(self.Dkx)
        self.log_Dzkx = np.log(self.Dzkx)

        entr_KX  = lin.inp(self.Dkx, self.log_Dkx)
        entr_ZKX = lin.inp(self.Dzkx, self.log_Dzkx)

        self.z = self.t - (entr_KX - entr_ZKX)

        self.feas = (self.z > 0)
        return self.feas
    
    def get_grad(self):
        assert self.feas_updated

        if self.grad_updated:
            return self.grad
        
        log_KX  = (self.Ukx * self.log_Dkx) @ self.Ukx.T
        log_ZKX = (self.Uzkx * self.log_Dzkx) @ self.Uzkx.T
        self.K_log_KX   = self.K.T @ sym.mat_to_vec(log_KX)
        self.ZK_log_ZKX = self.ZK.T @ sym.mat_to_vec(log_ZKX)

        self.inv_Dx = np.reciprocal(self.Dx)
        self.inv_X  = (self.Ux * self.inv_Dx) @ self.Ux.T

        self.zi   = np.reciprocal(self.z)
        self.DPhi = self.K_log_KX - self.ZK_log_ZKX

        self.grad     =  np.empty((self.dim, 1))
        self.grad[0]  = -self.zi
        self.grad[1:] =  self.zi * self.DPhi - sym.mat_to_vec(self.inv_X)

        self.grad_updated = True
        return self.grad
    
    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        irt2 = math.sqrt(0.5)

        self.D1kx_log  = mgrad.D1_log(self.Dkx, self.log_Dkx)
        self.D1zkx_log = mgrad.D1_log(self.Dzkx, self.log_Dzkx)

        # Hessians of quantum relative entropy
        invXX  = np.empty((self.vni, self.vni))
        k = 0
        for j in range(self.ni):
            for i in range(j + 1):
                # invXX
                temp = np.outer(self.inv_X[i, :], self.inv_X[j, :])
                if i != j:
                    temp = temp + temp.T
                    temp *= irt2
                invXX[:, [k]] = sym.mat_to_vec(temp)

                k += 1

        sqrt_D1kx_log = np.sqrt(sym.mat_to_vec(self.D1kx_log, 1.0))
        UnUnK = np.empty((self.vni, self.vnk))
        for k in range(self.vni):
            K = sym.vec_to_mat(self.K[:, [k]])
            UnKUn = self.Ukx.T @ K @ self.Ukx
            UnUnK[[k], :] = (sym.mat_to_vec(UnKUn) * sqrt_D1kx_log).T
        D2PhiKX = UnUnK @ UnUnK.T

        sqrt_D1zkx_log = np.sqrt(sym.mat_to_vec(self.D1zkx_log, 1.0))
        UxkUxkZK = np.empty((self.vni, self.vnzk))
        for k in range(self.vni):
            ZK = sym.vec_to_mat(self.ZK[:, [k]])
            UzkZKUzk = self.Uzkx.T @ ZK @ self.Uzkx
            UxkUxkZK[[k], :] = (sym.mat_to_vec(UzkZKUzk) * sqrt_D1zkx_log).T
        D2PhiZKX = UxkUxkZK @ UxkUxkZK.T

        # Preparing other required variables
        zi2 = self.zi * self.zi
        D2Phi = D2PhiKX - D2PhiZKX

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

    def update_dder3_aux(self):
        assert not self.dder3_aux_updated
        assert self.hess_aux_updated

        self.D2kx_log   = mgrad.D2_log(self.Dkx, self.D1kx_log)
        self.D2zkx_log  = mgrad.D2_log(self.Dzkx, self.D1zkx_log)

        self.dder3_aux_updated = True

        return

    def third_dir_deriv(self, dirs):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.dder3_aux_updated:
            self.update_dder3_aux()

        Ht   = dirs[0]
        Hx   = sym.vec_to_mat(dirs[1:, [0]])
        Hkx  = sym.vec_to_mat(self.K  @ dirs[1:, [0]])
        Hzkx = sym.vec_to_mat(self.ZK @ dirs[1:, [0]])
        Hx_vec = dirs[1:, [0]]

        UkxHkxUkx    = self.Ukx.T @ Hkx @ self.Ukx
        UzkxHzkxUzkx = self.Uzkx.T @ Hzkx @ self.Uzkx

        # Quantum conditional entropy oracles
        D2PhiH  = self.K.T  @ sym.mat_to_vec(self.Ukx @ (self.D1kx_log * UkxHkxUkx) @ self.Ukx.T)
        D2PhiH -= self.ZK.T @ sym.mat_to_vec(self.Uzkx @ (self.D1zkx_log * UzkxHzkxUzkx) @ self.Uzkx.T)

        D3PhiHH  = self.K.T  @ sym.mat_to_vec(mgrad.scnd_frechet(self.D2kx_log, self.Ukx, UkxHkxUkx, UkxHkxUkx))
        D3PhiHH -= self.ZK.T @ sym.mat_to_vec(mgrad.scnd_frechet(self.D2zkx_log, self.Uzkx, UzkxHzkxUzkx, UzkxHzkxUzkx))

        # Third derivative of barrier
        DPhiH = lin.inp(self.DPhi, Hx_vec)
        D2PhiHH = lin.inp(D2PhiH, Hx_vec)
        chi = Ht - DPhiH

        dder3 = np.empty((self.dim, 1))
        dder3[0] = -2 * (self.zi**3) * (chi**2) - (self.zi**2) * D2PhiHH

        temp = -dder3[0] * self.DPhi
        temp -= 2 * (self.zi**2) * chi * D2PhiH
        temp += self.zi * D3PhiHH
        temp -= 2 * sym.mat_to_vec(self.inv_X @ Hx @ self.inv_X @ Hx @ self.inv_X)
        dder3[1:] = temp

        return dder3
    
    def norm_invhess(self, x):
        return 0.0    