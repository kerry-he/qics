import numpy as np
import scipy as sp
import numba as nb
import math
from utils import symmetric as sym
from utils import linear    as lin
from utils import mtxgrad   as mgrad

class QuantKeyDist():
    def __init__(self, K_list, Z_list=None, ZK_list=None, protocol=None, hermitian=False):
        # Dimension properties
        self.ni = K_list[0].shape[1]    # Get input dimension
        self.hermitian = hermitian      # Is the problem complex-valued 
        self.protocol = protocol        # Special oracles available for dprBB84 and DMCV protocols
        
        self.vni = sym.vec_dim(self.ni, hermitian)     # Get input vector dimension

        self.dim = 1 + self.vni             # Total dimension of cone
        self.use_sqrt = False

        if protocol is None:
            # If there is no protocol, then do standard facial reduction on both G and ZG 
            ZK_list = [Z @ K for K in K_list for Z in Z_list] if ZK_list is None else ZK_list
            self.K_list_blk  = [facial_reduction(K_list,  hermitian=hermitian)]
            self.ZK_list_blk = [facial_reduction(ZK_list, hermitian=hermitian)]

        elif protocol == "dprBB84" or protocol == "dprBB84_fast":
            # For dprBB84 protocol, G and ZG are both projectors in the computational basis
            span_idx     = np.sort(np.array([np.where(K)[0] for K in K_list]).ravel())
            self.K_list  = [K[span_idx, :] for K in K_list]
            self.ZK_list = [(Z @ K)[span_idx, :] for K in K_list for Z in Z_list]

            self.K_mat_idx  = np.array([np.sort(np.where(K)[1]) for K in self.K_list])
            self.ZK_mat_idx = np.array([np.sort(np.where(ZK)[1]) for ZK in self.ZK_list])
            self.K_vec_idx  = np.array([mat_to_vec_idx(mat_idx, hermitian=hermitian) for mat_idx in self.K_mat_idx])
            self.ZK_vec_idx = np.array([mat_to_vec_idx(mat_idx, hermitian=hermitian) for mat_idx in self.ZK_mat_idx])
            self.K_v        = [K[np.where(K)][0] for K in self.K_list]
            self.ZK_v       = [ZK[np.where(ZK)][0] for ZK in self.ZK_list]

            ZK_list = [Z @ K for K in K_list for Z in Z_list] if ZK_list is None else ZK_list
            self.K_list_blk  = [[K[np.where(K)[0], :]] for K in K_list]
            self.ZK_list_blk = [[ZK[np.where(ZK)[0], :]] for ZK in ZK_list]
            
            self.cond_est = 0.0

        elif protocol == "DMCV":
            # For DMCV protocol, Z maps onto block diagonal matrix with 4 blocks
            self.K_list_blk = [facial_reduction(K_list, hermitian=hermitian)]
            self.ZK_list = [Z @ K for K in K_list for Z in Z_list]
            self.ZK_list_blk = [[ZK[i::4, :]] for (i, ZK) in enumerate(self.ZK_list)]

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
        point[1:] = sym.mat_to_vec(np.eye(self.ni), hermitian=self.hermitian)

        self.set_point(point)

        return point
    
    def set_point(self, point):
        assert np.size(point) == self.dim
        self.point = point

        self.t   = point[0]
        self.X   = sym.vec_to_mat(point[1:], hermitian=self.hermitian)
        self.KX_blk  = [sym.congr_map(self.X, K_list)  for K_list  in self.K_list_blk]
        self.ZKX_blk = [sym.congr_map(self.X, ZK_list) for ZK_list in self.ZK_list_blk]

        self.feas_updated        = False
        self.grad_updated        = False
        self.hess_aux_updated    = False
        self.invhess_aux_updated = False
        self.dder3_aux_updated   = False

        if self.protocol == "dprBB84_fast" and self.cond_est >= 1e15:
            self.protocol = "dprBB84"

        return
    
    def get_feas(self):
        if self.feas_updated:
            return self.feas
        
        self.feas_updated = True

        # Eigendecomposition of X
        self.Dx, self.Ux = np.linalg.eigh(self.X)
        if any(self.Dx <= 0):
            self.feas = False
            return self.feas

        # Eigendecomposition of G(X)
        DUkx_blk = [np.linalg.eigh(KX) for KX in self.KX_blk]
        self.Dkx_blk = [DUkx[0] for DUkx in DUkx_blk]
        self.Ukx_blk = [DUkx[1] for DUkx in DUkx_blk]

        if any([any(Dkx <= 0) for Dkx in self.Dkx_blk]):
            self.feas = False
            return self.feas        

        # Eigendecomposition of Z(G(X))
        DUzkx_blk = [np.linalg.eigh(ZKX) for ZKX in self.ZKX_blk]
        self.Dzkx_blk = [DUzkx[0] for DUzkx in DUzkx_blk]
        self.Uzkx_blk = [DUzkx[1] for DUzkx in DUzkx_blk]

        if any([any(Dzkx <= 0) for Dzkx in self.Dzkx_blk]):
            self.feas = False
            return self.feas

        # Compute feasibility
        self.log_Dkx_blk  = [np.log(D) for D in self.Dkx_blk]
        self.log_Dzkx_blk = [np.log(D) for D in self.Dzkx_blk]

        entr_KX  = sum([lin.inp(D, log_D) for (D, log_D) in zip(self.Dkx_blk,  self.log_Dkx_blk)])
        entr_ZKX = sum([lin.inp(D, log_D) for (D, log_D) in zip(self.Dzkx_blk, self.log_Dzkx_blk)])

        self.z = self.t - (entr_KX - entr_ZKX)

        self.feas = (self.z > 0)
        return self.feas

    def get_val(self):
        assert self.feas_updated

        return -np.log(self.z) - np.sum(np.log(self.Dx))

    def get_grad(self):
        assert self.feas_updated

        if self.grad_updated:
            return self.grad

        log_KX  = [(U * log_D) @ U.conj().T for (U, log_D) in zip(self.Ukx_blk,  self.log_Dkx_blk)]
        log_ZKX = [(U * log_D) @ U.conj().T for (U, log_D) in zip(self.Uzkx_blk, self.log_Dzkx_blk)]

        self.K_log_KX   = sum([sym.congr_map(log_X, K_list, adjoint=True) for (log_X, K_list) in zip(log_KX, self.K_list_blk)])
        self.ZK_log_ZKX = sum([sym.congr_map(log_X, K_list, adjoint=True) for (log_X, K_list) in zip(log_ZKX, self.ZK_list_blk)])

        self.inv_Dx = np.reciprocal(self.Dx)
        self.inv_X  = (self.Ux * self.inv_Dx) @ self.Ux.conj().T

        self.zi   = np.reciprocal(self.z)
        self.DPhi = sym.mat_to_vec(self.K_log_KX - self.ZK_log_ZKX, hermitian=self.hermitian)

        self.grad     =  np.empty((self.dim, 1))
        self.grad[0]  = -self.zi
        self.grad[1:] =  self.zi * self.DPhi - sym.mat_to_vec(self.inv_X, hermitian=self.hermitian)

        self.grad_updated = True
        return self.grad
    
    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        self.zi2 = self.zi * self.zi

        self.D1kx_log_blk  = [mgrad.D1_log(D, log_D) for (D, log_D) in zip(self.Dkx_blk,  self.log_Dkx_blk)]
        self.D1zkx_log_blk = [mgrad.D1_log(D, log_D) for (D, log_D) in zip(self.Dzkx_blk, self.log_Dzkx_blk)]

        self.hess_aux_updated = True

        return
    
    def hess_prod(self, dirs):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        for k in range(p):
            Ht = dirs[0, k]
            Hx = sym.vec_to_mat(dirs[1:, [k]], hermitian=self.hermitian)
            Hx_vec = dirs[1:, [k]]

            KH_blk  = [sym.congr_map(Hx, K_list)  for K_list  in self.K_list_blk]
            ZKH_blk = [sym.congr_map(Hx, ZK_list) for ZK_list in self.ZK_list_blk]

            UkKHUk_blk    = [U.conj().T @ H @ U for (H, U) in zip(KH_blk, self.Ukx_blk)]
            UkzZKHUkz_blk = [U.conj().T @ H @ U for (H, U) in zip(ZKH_blk, self.Uzkx_blk)]

            # Hessian product of conditional entropy
            D2PhiH  = sum([sym.congr_map(U @ (D1 * UHU) @ U.conj().T, K_list, adjoint=True) 
                          for (U, D1, UHU, K_list) in zip(self.Ukx_blk, self.D1kx_log_blk, UkKHUk_blk, self.K_list_blk)])
            D2PhiH -= sum([sym.congr_map(U @ (D1 * UHU) @ U.conj().T, K_list, adjoint=True) 
                          for (U, D1, UHU, K_list) in zip(self.Uzkx_blk, self.D1zkx_log_blk, UkzZKHUkz_blk, self.ZK_list_blk)])
            
            # Hessian product of barrier function
            out[0, k] = (Ht - lin.inp(self.DPhi, Hx_vec)) * self.zi2

            out[1:, [k]]  = -out[0, k] * self.DPhi
            out[1:, [k]] +=  sym.mat_to_vec(self.zi * D2PhiH + self.inv_X @ Hx @ self.inv_X, hermitian=self.hermitian)

        return out
    
    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated

        irt2 = math.sqrt(0.5)

        if self.protocol is None:
            # Default computation of QKD Hessian
            self.hess  = kronecker_matrix(self.inv_X, hermitian=self.hermitian)
            self.hess += sum([frechet_matrix_alt(U, D1, K_list=K_list, hermitian=self.hermitian) * self.zi
                          for (U, D1, K_list) in zip(self.Ukx_blk, self.D1kx_log_blk, self.K_list_blk)])
            self.hess -= sum([frechet_matrix_alt(U, D1, K_list=K_list, hermitian=self.hermitian) * self.zi
                          for (U, D1, K_list) in zip(self.Uzkx_blk, self.D1zkx_log_blk, self.ZK_list_blk)])

            self.hess_fact = lin.fact(self.hess)
        
        elif self.protocol == "dprBB84":
            self.update_invhessprod_aux_dprBB84()

        elif self.protocol == "dprBB84_fast":
            self.update_invhessprod_aux_dprBB84_fast()

        elif self.protocol == "DMCV":
            self.update_invhessprod_aux_DMCV()

        self.invhess_aux_updated = True

        return
    
    def update_invhessprod_aux_dprBB84(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated

        # Hessians of quantum relative entropy
        self.hess = kronecker_matrix(self.inv_X, hermitian=self.hermitian)

        def block_frechet_matrix(mat_idx_list, vec_idx_list, v_list, c, out):
            for (mat_idx, vec_idx, v) in zip(mat_idx_list, vec_idx_list, v_list):
                v2 = v * v
                v4 = v2 * v2

                Kx = self.X[np.ix_(mat_idx, mat_idx)] * v2
                Dkx, Ukx = np.linalg.eigh(Kx)
                D1kx_log = mgrad.D1_log(Dkx, np.log(Dkx))

                # Build matrix
                Hkx = frechet_matrix(Ukx, D1kx_log, hermitian=self.hermitian) * v4

                out[np.ix_(vec_idx, vec_idx)] += Hkx * c
            
            return out

        block_frechet_matrix(self.K_mat_idx,  self.K_vec_idx,  self.K_v,   self.zi, self.hess)
        block_frechet_matrix(self.ZK_mat_idx, self.ZK_vec_idx, self.ZK_v, -self.zi, self.hess)

        self.hess_fact = lin.fact(self.hess)

        return
    
    def update_invhessprod_aux_dprBB84_fast(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated


        # Get subblock of X kron X 
        nK = self.K_vec_idx[0].size

        X00 = self.X[np.ix_(self.K_mat_idx[0], self.K_mat_idx[0])]
        X01 = self.X[np.ix_(self.K_mat_idx[1], self.K_mat_idx[0])]
        X11 = self.X[np.ix_(self.K_mat_idx[1], self.K_mat_idx[1])]
        
        small_XX = np.empty((nK * 2, nK * 2))
        small_XX[:nK, :nK] = kronecker_matrix(X00, hermitian=self.hermitian)
        small_XX[nK:, nK:] = kronecker_matrix(X11, hermitian=self.hermitian)
        small_XX[nK:, :nK] = kronecker_matrix(X01, hermitian=self.hermitian)
        small_XX[:nK, nK:] = small_XX[nK:, :nK].T


        # Get Hessian corresponding to QRE
        def block_frechet_matrix(mat_idx_list, vec_idx_list, v_list, c, out):
            for (mat_idx, vec_idx, v) in zip(mat_idx_list, vec_idx_list, v_list):
                v2 = v * v
                v4 = v2 * v2

                Kx = self.X[np.ix_(mat_idx, mat_idx)] * v2
                Dkx, Ukx = np.linalg.eigh(Kx)
                D1kx_log = mgrad.D1_log(Dkx, np.log(Dkx))

                # Build matrix
                Hkx = frechet_matrix(Ukx, D1kx_log, hermitian=self.hermitian) * v4

                out[np.ix_(vec_idx, vec_idx)] += Hkx * c
            
            return out

        temp = np.zeros((self.vni, self.vni))
        block_frechet_matrix(self.K_mat_idx,  self.K_vec_idx,  self.K_v,   self.zi, temp)
        block_frechet_matrix(self.ZK_mat_idx, self.ZK_vec_idx, self.ZK_v, -self.zi, temp)
        temp = temp[np.ix_(self.K_vec_idx.ravel(), self.K_vec_idx.ravel())]


        # Determine if low-rank component needs further reduction 
        M1 = temp[:nK, :nK]
        M2 = temp[nK:, nK:]

        D1, U1 = np.linalg.eigh(M1)
        idx = np.where(D1 <= 1e-10)[0]
        D1 = np.delete(D1, idx)
        U1 = np.delete(U1, idx, 1)

        D2, U2 = np.linalg.eigh(M2)
        idx = np.where(D2 <= 1e-10)[0]
        D2 = np.delete(D2, idx)
        U2 = np.delete(U2, idx, 1)

        self.U12 = sp.linalg.block_diag(U1, U2)
        D12_inv = np.hstack((np.reciprocal(D1), np.reciprocal(D2)))


        # Compute and factor Schur complement
        self.schur = self.U12.T @ small_XX @ self.U12
        self.schur[np.diag_indices_from(self.schur)] += D12_inv

        self.schur_fact = lin.fact(self.schur)


        # Estimate condition number
        self.cond_est = (np.max(self.Dx) + np.max(D12_inv)) / (np.min(self.Dx) + np.min(D12_inv))

        return    

    def update_invhessprod_aux_DMCV(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated

        # Hessians of quantum relative entropy
        self.hess = kronecker_matrix(self.inv_X, hermitian=self.hermitian)
                
        def block_frechet_matrix(K_list, c, out):
            for K in K_list:
                Kx = sym.congr_map(self.X, K)
                Dkx, Ukx = np.linalg.eigh(Kx)
                D1kx_log = mgrad.D1_log(Dkx, np.log(Dkx))

                # Build matrix
                Hkx = frechet_matrix_alt(Ukx, D1kx_log, K_list=K, hermitian=True)
                out += Hkx * c
            
            return out

        block_frechet_matrix(self.K_list_blk,   self.zi, self.hess)
        block_frechet_matrix(self.ZK_list_blk, -self.zi, self.hess)

        self.hess_fact = lin.fact(self.hess)

        return

    def invhess_prod(self, dirs):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        if self.protocol == "dprBB84_fast":
            return self.invhess_prod_dprBB84_fast(dirs)
        
        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        Ht = dirs[[0], :]
        Hx = dirs[1:, :]

        out[1:, :] = lin.fact_solve(self.hess_fact, Hx + Ht * self.DPhi)
        out[[0], :] = self.z * self.z * Ht + np.sum(out[1:, :] * self.DPhi, axis=0)

        return out
    
    def invhess_prod_dprBB84_fast(self, dirs):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        # if not self.invhess_aux_updated:
        #     self.update_invhessprod_aux()

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        Ht = dirs[[0], :]
        Hx = dirs[1:, :]
        Wx = Hx + Ht * self.DPhi

        temp_vec = np.zeros((self.K_vec_idx.ravel().size, p))

        for k in range(p):
            Wx_k = sym.vec_to_mat(Wx[:, [k]], hermitian=self.hermitian)

            temp = self.X @ Wx_k @ self.X
            temp = sym.mat_to_vec(temp, hermitian=self.hermitian)
            temp_vec[:, [k]] = temp[self.K_vec_idx.ravel()]

        temp_vec = self.U12.T @ temp_vec
        temp_vec = lin.fact_solve(self.schur_fact, temp_vec)
        temp_vec = self.U12 @ temp_vec

        Wx[self.K_vec_idx.ravel()] -= temp_vec

        for k in range(p):
            Wx_k = sym.vec_to_mat(Wx[:, [k]], hermitian=self.hermitian)

            temp = self.X @ Wx_k @ self.X

            outX = sym.mat_to_vec(temp, hermitian=self.hermitian)
            outt = self.z * self.z * Ht[0, k] + self.DPhi.T @ outX

            out[0, k] = outt
            out[1:, [k]] = outX

        return out

    def update_dder3_aux(self):
        assert not self.dder3_aux_updated
        assert self.hess_aux_updated

        self.D2kx_log_blk  = [mgrad.D2_log(D, D1) for (D, D1) in zip(self.Dkx_blk,  self.D1kx_log_blk)]
        self.D2zkx_log_blk = [mgrad.D2_log(D, D1) for (D, D1) in zip(self.Dzkx_blk, self.D1zkx_log_blk)]

        self.dder3_aux_updated = True

        return

    def third_dir_deriv(self, dirs):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.dder3_aux_updated:
            self.update_dder3_aux()

        Ht   = dirs[0]
        Hx   = sym.vec_to_mat(dirs[1:, [0]], hermitian=self.hermitian)
        Hx_vec = dirs[1:, [0]]

        KH_blk  = [sym.congr_map(Hx, K_list) for K_list in self.K_list_blk]
        ZKH_blk = [sym.congr_map(Hx, ZK_list) for ZK_list in self.ZK_list_blk]

        UkKHUk_blk    = [U.conj().T @ H @ U for (H, U) in zip(KH_blk, self.Ukx_blk)]
        UkzZKHUkz_blk = [U.conj().T @ H @ U for (H, U) in zip(ZKH_blk, self.Uzkx_blk)]

        # Quantum conditional entropy oracles
        D2PhiH  = sum([sym.congr_map(U @ (D1 * UHU) @ U.conj().T, K_list, adjoint=True) 
                        for (U, D1, UHU, K_list) in zip(self.Ukx_blk, self.D1kx_log_blk, UkKHUk_blk, self.K_list_blk)])
        D2PhiH -= sum([sym.congr_map(U @ (D1 * UHU) @ U.conj().T, K_list, adjoint=True) 
                        for (U, D1, UHU, K_list) in zip(self.Uzkx_blk, self.D1zkx_log_blk, UkzZKHUkz_blk, self.ZK_list_blk)])

        D3PhiHH  = sum([sym.congr_map(mgrad.scnd_frechet(D2 * UHU, UHU, U=U), K_list, adjoint=True)
                        for (U, D2, UHU, K_list) in zip(self.Ukx_blk, self.D2kx_log_blk, UkKHUk_blk, self.K_list_blk)])
        D3PhiHH -= sum([sym.congr_map(mgrad.scnd_frechet(D2 * UHU, UHU, U=U), K_list, adjoint=True)
                        for (U, D2, UHU, K_list) in zip(self.Uzkx_blk, self.D2zkx_log_blk, UkzZKHUkz_blk, self.ZK_list_blk)])

        # Third derivative of barrier
        DPhiH = lin.inp(self.DPhi, Hx_vec)
        D2PhiHH = lin.inp(D2PhiH, Hx)
        chi = Ht - DPhiH

        dder3 = np.zeros((self.dim, 1))
        dder3[0] = -2 * (self.zi**3) * (chi**2) - (self.zi**2) * D2PhiHH

        dder3[1:] = -dder3[0] * self.DPhi
        temp  = -2 * (self.zi**2) * chi * D2PhiH
        temp += self.zi * D3PhiHH
        temp -= 2 * self.inv_X @ Hx @ self.inv_X @ Hx @ self.inv_X
        dder3[1:] += sym.mat_to_vec(temp, hermitian=self.hermitian)

        return dder3
    
    def norm_invhess(self, x):
        return 0.0
    

def facial_reduction(K_list, hermitian=False):
    # For a set of Kraus operators i.e., SUM_i K_i @ X @ K_i.T, returns a set of
    # Kraus operators which preserves positive definiteness
    nk = K_list[0].shape[0]

    # Pass identity matrix (maximally mixed state) through the Kraus operators
    KK = np.zeros((nk, nk), 'complex128') if hermitian else np.zeros((nk, nk))
    for K in K_list:
        KK += K @ K.conj().T

    # Determine if output is low rank, in which case we need to perform facial reduction
    Dkk, Ukk = np.linalg.eigh(KK)
    KKnzidx = np.where(Dkk > 1e-12)[0]
    nk_fr = np.size(KKnzidx)

    if nk == nk_fr:
        return K_list
    
    # Perform facial reduction
    Qkk = Ukk[:, KKnzidx]
    K_list_fr = [Qkk.conj().T @ K for K in K_list]

    return K_list_fr

def mat_to_vec_idx(mat_idx, hermitian=False):
    # Get indices
    n  = mat_idx.size
    vn = sym.vec_dim(n, hermitian=hermitian)
    vec_idx = np.zeros((vn,), dtype='uint64')

    k = 0
    for j in range(n):
        for i in range(j):
            (I, J) = (mat_idx[i], mat_idx[j])
            vec_idx[k] = 2*I + J*J
            k += 1

            if hermitian:
                vec_idx[k] = 2*I + J*J + 1
                k += 1
        
        J = mat_idx[j]
        vec_idx[k] = 2*J + J*J
        k += 1

    return vec_idx

def frechet_matrix(U, D1, K_list=None, hermitian=True):
    # Build matrix corresponding to linear map H -> U @ [D1 * (U' @ H @ U)] @ U'
    KU_list = [U] if (K_list is None) else [K.conj().T @ U for K in K_list]
    
    n   = KU_list[0].shape[0]
    vn  = sym.vec_dim(n, hermitian=hermitian)
    rt2 = np.sqrt(2.0)
    out = np.zeros((vn, vn))

    k = 0
    for j in range(n):
        for i in range(j):
            UHU = sum([KU.conj().T[:, [i]] @ KU[[j], :] for KU in KU_list]) / rt2
            D_H = sym.congr_map(D1 * UHU, KU_list)
            out[:, [k]]     = sym.mat_to_vec(D_H + D_H.conj().T, rt2, hermitian)
            D_H            *= 1j
            out[:, [k + 1]] = sym.mat_to_vec(D_H + D_H.conj().T, rt2, hermitian)
            k += 2

        UHU         = sum([KU.conj().T[:, [j]] @ KU[[j], :] for KU in KU_list])
        D_H         = sym.congr_map(D1 * UHU, KU_list)
        out[:, [k]] = sym.mat_to_vec(D_H, rt2, hermitian)
        k += 1

    return out


def frechet_matrix_alt(U, D1, K_list=None, hermitian=True):
    # Build matrix corresponding to linear map H -> U @ [D1 * (U' @ H @ U)] @ U'
    KU_list = [U] if (K_list is None) else [K.conj().T @ U for K in K_list]
    D1_rt2 = np.sqrt(D1)
    
    n, m = KU_list[0].shape
    vn   = sym.vec_dim(n, hermitian=hermitian)
    vm   = sym.vec_dim(m, hermitian=hermitian)
    rt2  = np.sqrt(2.0)

    fact = np.zeros((vm, vn))

    k = 0
    for j in range(n):
        for i in range(j):
            UHU = sum([KU.conj().T[:, [i]] @ KU[[j], :] for KU in KU_list]) / rt2
            D_H = D1_rt2 * UHU
            fact[:, [k]]     = sym.mat_to_vec(D_H + D_H.conj().T, rt2, hermitian)
            D_H            *= 1j
            fact[:, [k + 1]] = sym.mat_to_vec(D_H + D_H.conj().T, rt2, hermitian)
            k += 2

        UHU         = sum([KU.conj().T[:, [j]] @ KU[[j], :] for KU in KU_list])
        D_H         = D1_rt2 * UHU
        fact[:, [k]] = sym.mat_to_vec(D_H, rt2, hermitian)
        k += 1

    return fact.T @ fact


def kronecker_matrix(X, hermitian=False):
    # Build matrix corresponding to linear map H -> X @ H @ X'
    n    = X.shape[0]
    vn   = sym.vec_dim(n, hermitian=hermitian)
    rt2 = np.sqrt(2.0)
    out  = np.zeros((vn, vn))

    k = 0
    for j in range(n):
        for i in range(j + 1):
            temp = X[:, [i]] @ X.conj().T[[j], :]
            if i != j:
                temp /= rt2
                out[:, [k]] = sym.mat_to_vec(temp + temp.conj().T, rt2, hermitian=hermitian)
                k += 1
                if hermitian:
                    temp *= 1j
                    out[:, [k]] = sym.mat_to_vec(temp + temp.conj().T, rt2, hermitian=hermitian)                        
                    k += 1
            else:
                out[:, [k]] = sym.mat_to_vec(temp, rt2, hermitian=hermitian)
                k += 1

    return out