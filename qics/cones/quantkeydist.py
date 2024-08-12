import numpy as np
import qics._utils.linalg as lin
import qics._utils.gradient as grad
from qics.cones.base import Cone

class QuantKeyDist(Cone):
    """A class representing a quantum key distribution cone

    .. math::
    
        \\mathcal{K}_{\\text{qkd}} = \\text{cl}\\{ (t, X) \\in \\mathbb{R} \\times \\mathbb{H}^n_{++} : t \\geq -S(\\mathcal{G}(X)) + S(\\mathcal{Z}(\\mathcal{G}(X))) \\},
        
    with barrier function

    .. math::
    
        (t, X) \\mapsto -\\log( t + S(\\mathcal{G}(X)) - S(\\mathcal{Z}(\\mathcal{G}(X))) ) - \\log \\det(X),
        
    where

    .. math::

        S(X) = -\\text{tr}[X \\log(X)],
        
    is the quantum (von Neumann) entropy function, :math:`\\mathcal{G}:\\mathbb{H}^n\\rightarrow\\mathbb{H}^{mr}` is a positive linear map, 
    and :math:`\\mathcal{Z}:\\mathbb{H}^{mr}\\rightarrow\\mathbb{H}^{mr}` is a pinching map that maps off-diagonal blocks to zero.

    Parameters
    ----------
    K_list : list of ndarray
        List of Kraus operators :math:`\\{ K_i \\in \\mathbb{C}^{mr \\times n } \\}_{i=1}^l` corresponding to :math:`\\mathcal{G}` such that 
        
        .. math::

            \\mathcal{G}(X) = \\sum_{i=1}^l K_i X K_i^\\dagger.
        
    Z_info : int or list of ndarray
        Specify the Kraus operators :math:`\\{ Z_i \\in \\mathbb{C}^{mr \\times mr } \\}_{i=1}^r` corresponding to :math:`\\mathcal{Z}` such that

        .. math::

            \\mathcal{Z}(Y) = \\sum_{i=1}^r Z_i Y Z_i^\\dagger.

        Note that these pinching map Kraus operators should be of the form

        .. math::

            Z_i = | i \\rangle \\langle i | \\otimes \\mathbb{I}_m,

        for all :math:`i=1,\\ldots,r`. Can either specify the dimension :math:`r` if ``Z_info`` is type ``int``, or directly specify the 
        Kraus operators :math:`\\{ Z_i \\}_{i=1}^r` if ``Z_info`` is type ``list``.
        
    iscomplex : bool
        Whether the matrix is symmetric :math:`X \\in \\mathbb{S}^n` (False) or Hermitian :math:`X \\in \\mathbb{H}^n` (True). Default is False.
    """    
    def __init__(self, K_list, Z_info, iscomplex=False):
        # Dimension properties
        self.n = K_list[0].shape[1]    # Get input dimension
        self.nu = 1 + self.n           # Barrier parameter
        self.iscomplex = iscomplex      # Is the problem complex-valued 
        
        self.vn = self.n*self.n if iscomplex else self.n*(self.n+1)//2      # Compact dimension of system

        self.dim   = [1, self.n*self.n] if (not iscomplex) else [1, 2*self.n*self.n]
        self.type  = ['r', 's']         if (not iscomplex) else ['r', 'h']
        self.dtype = np.float64         if (not iscomplex) else np.complex128       

        # Always block the ZK operator as Z maps to block matrices
        self.K_list_blk  = [facial_reduction(K_list)]
        if isinstance(Z_info, int):
            m = K_list[0].shape[0] // Z_info
            assert m * Z_info == K_list[0].shape[0]
            self.ZK_list_blk = [facial_reduction([K[i*m : (i+1)*m, :] for K in K_list]) for i in range(Z_info)]
        else:
            self.ZK_list_blk = [facial_reduction([K[np.where(Z)[0], :] for K in K_list]) for Z in Z_info]

        self.nk  = [K_list[0].shape[0] for K_list in self.K_list_blk]
        self.nzk = [K_list[0].shape[0] for K_list in self.ZK_list_blk]

        # Update flags
        self.feas_updated        = False
        self.grad_updated        = False
        self.hess_aux_updated    = False
        self.invhess_aux_updated = False
        self.dder3_aux_updated   = False
        self.congr_aux_updated = False
        self.invhess_aux_aux_updated = False

        self.precompute_mat_vec()     

        return

    def get_iscomplex(self):
        return self.iscomplex

    def get_init_point(self, out):
        KK_blk   = [apply_kraus(np.eye(self.n), K_list)  for K_list  in self.K_list_blk]
        ZKKZ_blk = [apply_kraus(np.eye(self.n), ZK_list) for ZK_list in self.ZK_list_blk]

        from qics.quantum import quant_entropy
        entr_KK   = sum([quant_entropy(KK)   for KK   in KK_blk])
        entr_ZKKZ = sum([quant_entropy(ZKKZ) for ZKKZ in ZKKZ_blk])

        t0 = 1. + (-entr_KK + entr_ZKKZ)

        point = [
            np.array([[t0]]), 
            np.eye(self.n, dtype=self.dtype)
        ]

        self.set_point(point, point)

        out[0][:] = point[0]
        out[1][:] = point[1]

        return out
    
    def get_feas(self):
        if self.feas_updated:
            return self.feas
        
        self.feas_updated = True

        (self.t, self.X) = self.primal

        # Eigendecomposition of X
        self.Dx, self.Ux = np.linalg.eigh(self.X)
        if any(self.Dx <= 0):
            self.feas = False
            return self.feas

        # Eigendecomposition of G(X)        
        self.KX_blk  = [apply_kraus(self.X, K_list)  for K_list  in self.K_list_blk]

        DUkx_blk     = [np.linalg.eigh(KX) for KX in self.KX_blk]
        self.Dkx_blk = [DUkx[0] for DUkx in DUkx_blk]
        self.Ukx_blk = [DUkx[1] for DUkx in DUkx_blk]

        if any([any(Dkx <= 0) for Dkx in self.Dkx_blk]):
            self.feas = False
            return self.feas        

        # Eigendecomposition of Z(G(X))
        self.ZKX_blk = [apply_kraus(self.X, ZK_list) for ZK_list in self.ZK_list_blk]        

        DUzkx_blk     = [np.linalg.eigh(ZKX) for ZKX in self.ZKX_blk]
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

        self.K_log_KX   = sum([apply_kraus(log_X, K_list, adjoint=True) for (log_X, K_list) in zip(log_KX, self.K_list_blk)])
        self.ZK_log_ZKX = sum([apply_kraus(log_X, K_list, adjoint=True) for (log_X, K_list) in zip(log_ZKX, self.ZK_list_blk)])

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
    
    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        self.zi2 = self.zi * self.zi

        self.D1kx_log_blk  = [grad.D1_log(D, log_D) for (D, log_D) in zip(self.Dkx_blk,  self.log_Dkx_blk)]
        self.D1zkx_log_blk = [grad.D1_log(D, log_D) for (D, log_D) in zip(self.Dzkx_blk, self.log_Dzkx_blk)]

        self.hess_aux_updated = True

        return
    
    def hess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        # Computes Hessian product of the QKD barrier with a single vector (Ht, Hx)
        # See hess_congr() for additional comments

        (Ht, Hx) = H

        KH_blk  = [apply_kraus(Hx, K_list)  for K_list  in self.K_list_blk]
        ZKH_blk = [apply_kraus(Hx, ZK_list) for ZK_list in self.ZK_list_blk]

        UkKHUk_blk    = [U.conj().T @ H @ U for (H, U) in zip(KH_blk, self.Ukx_blk)]
        UkzZKHUkz_blk = [U.conj().T @ H @ U for (H, U) in zip(ZKH_blk, self.Uzkx_blk)]

        # Hessian product of conditional entropy
        D2PhiH  = sum([apply_kraus(U @ (D1 * UHU) @ U.conj().T, K_list, adjoint=True) 
                        for (U, D1, UHU, K_list) in zip(self.Ukx_blk, self.D1kx_log_blk, UkKHUk_blk, self.K_list_blk)])
        D2PhiH -= sum([apply_kraus(U @ (D1 * UHU) @ U.conj().T, K_list, adjoint=True) 
                        for (U, D1, UHU, K_list) in zip(self.Uzkx_blk, self.D1zkx_log_blk, UkzZKHUkz_blk, self.ZK_list_blk)])
        
        # Hessian product of barrier function
        out[0][:] = (Ht - lin.inp(self.DPhi, Hx)) * self.zi2

        out_X     = -out[0] * self.DPhi
        out_X    += self.zi * D2PhiH
        out_X    += self.inv_X @ Hx @ self.inv_X
        out_X     = (out_X + out_X.conj().T) * 0.5
        out[1][:] = out_X

        return out

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
        temp_vec = work.view(dtype=np.float64).reshape((-1, 1))[self.triu_idxs]
        temp_vec *= self.scale.reshape((-1, 1))

        temp_vec = lin.cho_solve(self.hess_fact, temp_vec)

        work.fill(0.)
        temp_vec[self.diag_idxs] *= 0.5
        temp_vec /= self.scale.reshape((-1, 1))
        work.view(dtype=np.float64).reshape((-1, 1))[self.triu_idxs] = temp_vec
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
        lhsX = lin.cho_solve(self.hess_fact, self.work0.T)

        # ====================================================================
        # Inverse Hessian products with respect to t
        # ====================================================================
        lhst  = self.z2 * self.At 
        lhst += (lhsX.T @ self.DPhi_vec).ravel()
        
        # Multiply A (H A')
        return np.outer(lhst, self.At) + lhsX.T @ self.Ax_vec.T

    def third_dir_deriv_axpy(self, out, H, a=True):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.dder3_aux_updated:
            self.update_dder3_aux()

        (Ht, Hx) = H

        KH_blk  = [apply_kraus(Hx, K_list) for K_list in self.K_list_blk]
        ZKH_blk = [apply_kraus(Hx, ZK_list) for ZK_list in self.ZK_list_blk]

        UkKHUk_blk    = [U.conj().T @ H @ U for (H, U) in zip(KH_blk, self.Ukx_blk)]
        UkzZKHUkz_blk = [U.conj().T @ H @ U for (H, U) in zip(ZKH_blk, self.Uzkx_blk)]

        # Quantum conditional entropy oracles
        D2PhiH  = sum([apply_kraus(U @ (D1 * UHU) @ U.conj().T, K_list, adjoint=True) 
                        for (U, D1, UHU, K_list) in zip(self.Ukx_blk, self.D1kx_log_blk, UkKHUk_blk, self.K_list_blk)])
        D2PhiH -= sum([apply_kraus(U @ (D1 * UHU) @ U.conj().T, K_list, adjoint=True) 
                        for (U, D1, UHU, K_list) in zip(self.Uzkx_blk, self.D1zkx_log_blk, UkzZKHUkz_blk, self.ZK_list_blk)])

        D3PhiHH  = sum([apply_kraus(grad.scnd_frechet(D2 * UHU, UHU, U=U), K_list, adjoint=True)
                        for (U, D2, UHU, K_list) in zip(self.Ukx_blk, self.D2kx_log_blk, UkKHUk_blk, self.K_list_blk)])
        D3PhiHH -= sum([apply_kraus(grad.scnd_frechet(D2 * UHU, UHU, U=U), K_list, adjoint=True)
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

    # ========================================================================
    # Auxilliary functions
    # ========================================================================
    def congr_aux(self, A):
        assert not self.congr_aux_updated

        self.At     = A[:, 0]
        self.Ax_vec = A[:, 1 + self.triu_idxs] * self.scale

        self.work0 = np.empty_like(self.Ax_vec)
        self.work1 = np.empty_like(self.Ax_vec)

        self.congr_aux_updated = True

    def hess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()                

        vec = self.At - self.DPhi_vec.T @ self.Ax_vec.T
        vec *= self.zi
        
        out = self.Ax_vec @ self.hess @ self.Ax_vec.T
        out += np.outer(vec, vec)
        return out

    def update_invhessprod_aux_aux(self):
        assert not self.invhess_aux_aux_updated

        self.work2  = [np.zeros((self.vn, self.n, nk), dtype=self.dtype) for nk in self.nk]
        self.work2b  = [np.zeros((self.vn, nk, self.n), dtype=self.dtype) for nk in self.nk]
        self.work3  = [np.zeros((self.vn, nk, nk), dtype=self.dtype) for nk in self.nk]
        self.work3b  = [np.zeros((self.vn, nk, nk), dtype=self.dtype) for nk in self.nk]
        self.work4  = [np.zeros((self.vn, self.n, nzk), dtype=self.dtype) for nzk in self.nzk]
        self.work4b  = [np.zeros((self.vn, nzk, self.n), dtype=self.dtype) for nzk in self.nzk]
        self.work5  = [np.zeros((self.vn, nzk, nzk), dtype=self.dtype) for nzk in self.nzk]
        self.work5b  = [np.zeros((self.vn, nzk, nzk), dtype=self.dtype) for nzk in self.nzk]
        self.work6  = np.zeros((self.vn, self.n, self.n), dtype=self.dtype)
        self.work7  = np.zeros((self.vn, self.n, self.n), dtype=self.dtype)
        self.work8  = np.zeros((self.vn, self.n, self.n), dtype=self.dtype)

        self.invhess_aux_aux_updated = True

    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated
        if not self.invhess_aux_aux_updated:
            self.update_invhessprod_aux_aux()

        self.z2 = self.z * self.z
        self.DPhi_vec = self.DPhi.view(dtype=np.float64).reshape(-1, 1)[self.triu_idxs] * self.scale.reshape(-1, 1)

        # Get X^-1 kron X^-1 
        lin.congr_multi(self.work8, self.inv_X, self.E, work=self.work7)

        # Get S(G(X)) Hessians
        for (U, D1, K_list, work0, work1, work2, work3) in zip(self.Ukx_blk, self.D1kx_log_blk, self.K_list_blk, self.work2, self.work2b, self.work3, self.work3b):
            KU_list = [K.conj().T @ U for K in K_list]      
            work2 *= 0
            for KU in KU_list:
                lin.congr_multi(work3, KU.conj().T, self.E, work=work1)
                work2 += work3

            work2 *= D1 * self.zi

            for KU in KU_list:
                lin.congr_multi(self.work7, KU, work2, work=work0)
                self.work8 += self.work7

        # Get S(Z(G(X))) Hessians
        for (U, D1, K_list, work0, work1, work2, work3) in zip(self.Uzkx_blk, self.D1zkx_log_blk, self.ZK_list_blk, self.work4, self.work4b, self.work5, self.work5b):
            KU_list = [K.conj().T @ U for K in K_list]      
            work2 *= 0
            for KU in KU_list:
                lin.congr_multi(work3, KU.conj().T, self.E, work=work1)
                work2 += work3

            work2 *= D1 * self.zi

            for KU in KU_list:
                lin.congr_multi(self.work7, KU, work2, work=work0)
                self.work8 -= self.work7             

        # Get Hessian and factorize
        self.hess  = self.work8.view(dtype=np.float64).reshape((self.vn, -1))[:, self.triu_idxs]
        self.hess *= self.scale
        self.hess_fact = lin.cho_fact(self.hess.copy())

        self.invhess_aux_updated = True

        return

    def update_dder3_aux(self):
        assert not self.dder3_aux_updated
        assert self.hess_aux_updated

        self.zi3 = self.zi2 * self.zi

        self.D2kx_log_blk  = [grad.D2_log(D, D1) for (D, D1) in zip(self.Dkx_blk,  self.D1kx_log_blk)]
        self.D2zkx_log_blk = [grad.D2_log(D, D1) for (D, D1) in zip(self.Dzkx_blk, self.D1zkx_log_blk)]

        self.dder3_aux_updated = True

        return

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

def apply_kraus(x, Klist, adjoint=False):
    # Compute congruence map
    if adjoint:
        return sum([K.conj().T @ x @ K for K in Klist])   
    else:
        return sum([K @ x @ K.conj().T for K in Klist])