import numpy as np
import scipy as sp
import numba as nb
import math
from utils import symmetric as sym
from utils import linear    as lin
from utils import mtxgrad   as mgrad
from utils import quantum   as quant

class Cone():
    def __init__(self, K_list, Z_list, hermitian=False):
        # Dimension properties
        self.ni = K_list[0].shape[1]    # Get input dimension
        self.nu = 1 + self.ni           # Barrier parameter
        self.hermitian = hermitian      # Is the problem complex-valued 
        
        self.vni = sym.vec_dim(self.ni, hermitian)     # Get input vector dimension

        self.dim   = [1, self.ni*self.ni] if (not hermitian) else [1, 2*self.ni*self.ni]
        self.type  = ['r', 's']           if (not hermitian) else ['r', 'h']
        self.dtype = np.float64           if (not hermitian) else np.complex128       

        # Always block the ZK operator as Z maps to block matrices
        self.K_list_blk  = [facial_reduction(K_list)]
        self.ZK_list_blk = [facial_reduction([K[np.where(Z)[0], :] for K in K_list]) for Z in Z_list]

        # Update flags
        self.feas_updated        = False
        self.grad_updated        = False
        self.hess_aux_updated    = False
        self.invhess_aux_updated = False
        self.dder3_aux_updated   = False
        self.congr_aux_updated = False
        self.invhess_aux_aux_updated = False

        if self.hermitian:
            self.diag_indices = np.append(0, np.cumsum([i for i in range(3, 2*self.ni+1, 2)]))
            self.triu_indices = np.empty(self.ni*self.ni, dtype=int)
            self.scale        = np.empty(self.ni*self.ni)
            k = 0
            for j in range(self.ni):
                for i in range(j):
                    self.triu_indices[k]     = 2 * (j + i*self.ni)
                    self.triu_indices[k + 1] = 2 * (j + i*self.ni) + 1
                    self.scale[k:k+2]        = np.sqrt(2.)
                    k += 2
                self.triu_indices[k] = 2 * (j + j*self.ni)
                self.scale[k]        = 1.
                k += 1
        else:
            self.diag_indices = np.append(0, np.cumsum([i for i in range(2, self.ni+1, 1)]))
            self.triu_indices = np.array([j + i*self.ni for j in range(self.ni) for i in range(j + 1)])
            self.scale = np.array([1 if i==j else np.sqrt(2.) for j in range(self.ni) for i in range(j + 1)])        

        return

    def get_init_point(self, out):
        KK_blk   = [sym.congr_map(np.eye(self.ni), K_list)  for K_list  in self.K_list_blk]
        ZKKZ_blk = [sym.congr_map(np.eye(self.ni), ZK_list) for ZK_list in self.ZK_list_blk]

        entr_KK   = -sum([quant.quantEntropy(KK)   for KK   in KK_blk])
        entr_ZKKZ = -sum([quant.quantEntropy(ZKKZ) for ZKKZ in ZKKZ_blk])

        t0 = 1. + (entr_KK - entr_ZKKZ)

        point = [
            np.array([[t0]]), 
            np.eye(self.ni, dtype=self.dtype)
        ]

        self.set_point(point, point)

        out[0][:] = point[0]
        out[1][:] = point[1]

        return out
    
    def set_point(self, point, dual, a=True):
        self.t = point[0] * a
        self.X = point[1] * a

        self.t_d = dual[0] * a
        self.X_d = dual[1] * a

        self.KX_blk  = [sym.congr_map(self.X, K_list)  for K_list  in self.K_list_blk]
        self.ZKX_blk = [sym.congr_map(self.X, ZK_list) for ZK_list in self.ZK_list_blk]

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

        self.z = self.t[0, 0] - (entr_KX - entr_ZKX)

        self.feas = (self.z > 0)
        return self.feas

    def get_val(self):
        assert self.feas_updated

        return -np.log(self.z) - np.sum(np.log(self.Dx))

    def update_grad(self):
        assert self.feas_updated
        assert not self.grad_updated
        
        log_KX  = [(U * log_D) @ U.conj().T for (U, log_D) in zip(self.Ukx_blk,  self.log_Dkx_blk)]
        log_ZKX = [(U * log_D) @ U.conj().T for (U, log_D) in zip(self.Uzkx_blk, self.log_Dzkx_blk)]

        self.K_log_KX   = sum([sym.congr_map(log_X, K_list, adjoint=True) for (log_X, K_list) in zip(log_KX, self.K_list_blk)])
        self.ZK_log_ZKX = sum([sym.congr_map(log_X, K_list, adjoint=True) for (log_X, K_list) in zip(log_ZKX, self.ZK_list_blk)])

        self.inv_Dx = np.reciprocal(self.Dx)
        inv_X_rt2   = self.Ux * np.sqrt(self.inv_Dx)
        self.inv_X  = inv_X_rt2 @ inv_X_rt2.conj().T

        self.zi   = np.reciprocal(self.z)
        self.DPhi = self.K_log_KX - self.ZK_log_ZKX

        self.grad = [
           -self.zi,
            self.zi * self.DPhi - self.inv_X,
        ]

        self.grad_updated = True

    def get_grad(self, out):
        assert self.feas_updated
        if not self.grad_updated:
            self.update_grad()

        out[0][:] = self.grad[0]
        out[1][:] = self.grad[1]
        
        return out
    
    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        self.zi2 = self.zi * self.zi

        self.D1kx_log_blk  = [mgrad.D1_log(D, log_D) for (D, log_D) in zip(self.Dkx_blk,  self.log_Dkx_blk)]
        self.D1zkx_log_blk = [mgrad.D1_log(D, log_D) for (D, log_D) in zip(self.Dzkx_blk, self.log_Dzkx_blk)]

        self.hess_aux_updated = True

        return
    
    def hess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        # Computes Hessian product of the QKD barrier with a single vector (Ht, Hx)
        # See hess_congr() for additional comments

        (Ht, Hx) = H

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
        out[0][:] = (Ht - lin.inp(self.DPhi, Hx)) * self.zi2

        out_X     = -out[0] * self.DPhi
        out_X    += self.zi * D2PhiH
        out_X    += self.inv_X @ Hx @ self.inv_X
        out_X     = (out_X + out_X.conj().T) * 0.5
        out[1][:] = out_X

        return out

    def congr_aux(self, A):
        assert not self.congr_aux_updated

        self.At     = A[:, 0]
        self.Ax_vec = A[:, 1 + self.triu_indices] * self.scale

        self.work0 = np.empty_like(self.Ax_vec)
        self.work1 = np.empty_like(self.Ax_vec)

        self.congr_aux_updated = True

    def hess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)            

        p = A.shape[0]
        lhs = np.zeros((p, sum(self.dim)))

        for k in range(p):
            Ht = self.At[k]
            Hx = self.Ax[k]

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
            outt = (Ht - lin.inp(self.DPhi, Hx)) * self.zi2

            outX  = -outt * self.DPhi
            outX +=  self.zi * D2PhiH + self.inv_X @ Hx @ self.inv_X

            lhs[:, 0] = outt
            lhs[:, 1:] = outX.reshape((p, -1)).view(dtype=np.float64)

        # Multiply A (H A')
        return lhs @ A.T

    def update_invhessprod_aux_aux(self):
        assert not self.invhess_aux_aux_updated

        rt2 = np.sqrt(0.5)
        self.E = np.zeros((self.vn, self.n, self.n), dtype=self.dtype)
        k = 0
        for j in range(self.n):
            for i in range(j):
                self.E[k, i, j] = rt2
                self.E[k, j, i] = rt2
                k += 1
                if self.hermitian:
                    self.E[k, i, j] = rt2 *  1j
                    self.E[k, j, i] = rt2 * -1j
                    k += 1
            self.E[k, j, j] = 1.
            k += 1

        self.hess = np.empty((2*self.vn, 2*self.vn))

        self.work4  = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work5  = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work6  = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work7  = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work8  = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work9  = np.empty((self.n, self.n, self.n), dtype=self.dtype)
        self.work10 = np.empty((self.n, 1, self.n), dtype=self.dtype)

        self.invhess_aux_aux_updated = True

    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated

        self.z2 = self.z * self.z

        # Default computation of QKD Hessian
        self.hess  = kronecker_matrix(self.inv_X, hermitian=self.hermitian)
        self.hess += sum([frechet_matrix_alt(U, D1, K_list=K_list, hermitian=self.hermitian) * self.zi
                        for (U, D1, K_list) in zip(self.Ukx_blk, self.D1kx_log_blk, self.K_list_blk)])
        self.hess -= sum([frechet_matrix_alt(U, D1, K_list=K_list, hermitian=self.hermitian) * self.zi
                        for (U, D1, K_list) in zip(self.Uzkx_blk, self.D1zkx_log_blk, self.ZK_list_blk)])

        self.DPhi_vec = self.DPhi.view(dtype=np.float64).reshape(-1, 1)[self.triu_indices] * self.scale.reshape(-1, 1)

        self.hess_fact = lin.fact(self.hess)

        self.invhess_aux_updated = True

        return

    def invhess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        # Computes inverse Hessian product of the QKD barrier with a single vector (Ht, Hx)
        # See invhess_congr() for additional comments

        (Ht, Hx) = H
        work = Hx + Ht * self.DPhi

        # Inverse Hessian products with respect to X
        temp_vec = work.view(dtype=np.float64).reshape((-1, 1))[self.triu_indices]
        temp_vec *= self.scale.reshape((-1, 1))

        temp_vec = lin.fact_solve(self.hess_fact, temp_vec)

        work.fill(0.)
        temp_vec[self.diag_indices] *= 0.5
        temp_vec /= self.scale.reshape((-1, 1))
        work.view(dtype=np.float64).reshape((-1, 1))[self.triu_indices] = temp_vec
        work += work.conj().T

        out[1][:] = work

        # Inverse Hessian products with respect to t
        out[0][:] = self.z2 * Ht + lin.inp(self.DPhi, out[1])

        return out

    def invhess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)            

        # The inverse Hessian product applied on (Ht, Hx, Hy) for the QRE barrier is 
        #     (X, Y) =  M \ (Wx, Wy)
        #         t  =  z^2 Ht + <DPhi(X, Y), (X, Y)>
        # where (Wx, Wy) = [(Hx, Hy) + Ht DPhi(X, Y)],
        #     M = Vxy [ 1/z log^[1](Dx) + Dx^-1 kron Dx^-1  -1/z (Ux'Uy kron Ux'Uy) log^[1](Dy) ]
        #             [-1/z log^[1](Dy) (Uy'Ux kron Uy'Ux)      -1/z Sy + Dy^-1 kron Dy^-1      ] Vxy'
        # and 
        #     Vxy = [ Ux kron Ux             ]
        #           [             Uy kron Uy ]
        #
        # To solve linear systems with M, we simplify it by doing block elimination, in which case we get
        #     Uy' Y Uy = S \ ({Uy' Wy Uy} - [1/z log^[1](Dy) (Uy'Ux kron Uy'Ux) (1/z log^[1](Dx) + Dx^-1 kron Dx^-1)^-1 {Ux' Wx Ux}])
        #     Ux' X Ux = -(1/z log^[1](Dx) + Dx^-1 kron Dx^-1)^-1 [{Ux' Wx Ux} + 1/z (Ux'Uy kron Ux'Uy) log^[1](Dy) Y]
        # where S is the Schur complement matrix of M.
        
        # ====================================================================
        # Inverse Hessian products with respect to X
        # ====================================================================
        # Compute Wx
        np.outer(self.At, self.DPhi_vec, out=self.work1)
        np.add(self.Ax_vec, self.work1, out=self.work0)

        # Solve system
        lhsX = lin.fact_solve(self.hess_fact, self.work0.T)

        # ====================================================================
        # Inverse Hessian products with respect to t
        # ====================================================================
        lhst  = self.z2 * self.At 
        lhst += (lhsX.T @ self.DPhi_vec).ravel()
        
        # Multiply A (H A')
        return np.outer(lhst, self.At) + lhsX.T @ self.Ax_vec.T

    def update_dder3_aux(self):
        assert not self.dder3_aux_updated
        assert self.hess_aux_updated

        self.zi3 = self.zi2 * self.zi

        self.D2kx_log_blk  = [mgrad.D2_log(D, D1) for (D, D1) in zip(self.Dkx_blk,  self.D1kx_log_blk)]
        self.D2zkx_log_blk = [mgrad.D2_log(D, D1) for (D, D1) in zip(self.Dzkx_blk, self.D1zkx_log_blk)]

        self.dder3_aux_updated = True

        return

    def third_dir_deriv_axpy(self, out, dir, a=True):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.dder3_aux_updated:
            self.update_dder3_aux()

        (Ht, Hx) = dir

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
        DPhiH = lin.inp(self.DPhi, Hx)
        D2PhiHH = lin.inp(D2PhiH, Hx)
        chi = Ht - DPhiH
        chi2 = chi * chi

        dder3_t = -2 * self.zi3 * chi2 - self.zi2 * D2PhiHH

        dder3_X  = -dder3_t * self.DPhi
        dder3_X -= 2 * self.zi2 * chi * D2PhiH
        dder3_X += self.zi * D3PhiHH
        dder3_X -= 2 * self.inv_X @ Hx @ self.inv_X @ Hx @ self.inv_X
        dder3_X  = (dder3_X + dder3_X.conj().T) * 0.5

        out[0][:] += dder3_t * a
        out[1][:] += dder3_X * a

        return out
    
    def prox(self):
        assert self.feas_updated
        if not self.grad_updated:
            self.update_grad()
        psi = [
            self.t_d + self.grad[0],
            self.X_d + self.grad[1]
        ]
        temp = [np.zeros((1, 1)), np.zeros((self.ni, self.ni), dtype=self.dtype)]
        self.invhess_prod_ip(temp, psi)
        return lin.inp(temp[0], psi[0]) + lin.inp(temp[1], psi[1]) 

def facial_reduction(K_list):
    # For a set of Kraus operators i.e., SUM_i K_i @ X @ K_i.T, returns a set of
    # Kraus operators which preserves positive definiteness
    nk = K_list[0].shape[0]

    # Pass identity matrix (maximally mixed state) through the Kraus operators
    KK = sum([K @ K.conj().T for K in K_list])

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
            out[:, [k]] = sym.mat_to_vec(D_H + D_H.conj().T, rt2, hermitian)
            k += 1

            if hermitian:
                D_H *= 1j
                out[:, [k]] = sym.mat_to_vec(D_H + D_H.conj().T, rt2, hermitian)
                k += 1

        UHU = sum([KU.conj().T[:, [j]] @ KU[[j], :] for KU in KU_list])
        D_H = sym.congr_map(D1 * UHU, KU_list)
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
            fact[:, [k]] = sym.mat_to_vec(D_H + D_H.conj().T, rt2, hermitian)
            k += 1

            if hermitian:
                D_H *= 1j
                fact[:, [k]] = sym.mat_to_vec(D_H + D_H.conj().T, rt2, hermitian)
                k += 1

        UHU = sum([KU.conj().T[:, [j]] @ KU[[j], :] for KU in KU_list])
        D_H = D1_rt2 * UHU
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
        for i in range(j):
            temp = X[:, [i]] @ X.conj().T[[j], :] / rt2
            out[:, [k]] = sym.mat_to_vec(temp + temp.conj().T, rt2, hermitian=hermitian)
            k += 1

            if hermitian:
                temp *= 1j
                out[:, [k]] = sym.mat_to_vec(temp + temp.conj().T, rt2, hermitian=hermitian)                        
                k += 1

        temp = X[:, [j]] @ X.conj().T[[j], :]
        out[:, [k]] = sym.mat_to_vec(temp, rt2, hermitian=hermitian)
        k += 1

    return out