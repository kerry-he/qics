import numpy as np
import scipy as sp
import numba as nb
import math
from utils import symmetric as sym
from utils import linear    as lin
from utils import mtxgrad   as mgrad

class Cone():
    def __init__(self, n0, n1, sys, hermitian=False):
        # Dimension properties
        self.n0 = n0          # Dimension of system 0
        self.n1 = n1          # Dimension of system 1
        self.N  = n0 * n1     # Total dimension of bipartite system
        self.hermitian = hermitian

        self.sys   = sys                       # System being traced out
        self.n = n0 if (sys == 1) else n1      # Dimension of system not traced out

        self.vn = sym.vec_dim(self.n, hermitian=hermitian)      # Dimension of vectorized system being traced out
        self.vN = sym.vec_dim(self.N, hermitian=hermitian)          # Dimension of vectorized bipartite system

        self.dim   = [1, self.N*self.N] if (not hermitian) else [1, 2*self.N*self.N]
        self.type  = ['r', 's']         if (not hermitian) else ['r', 'h']
        self.dtype = np.float64         if (not hermitian) else np.complex128        

        # Update flags
        self.feas_updated            = False
        self.grad_updated            = False
        self.hess_aux_updated        = False
        self.invhess_aux_updated     = False
        self.invhess_aux_aux_updated = False
        self.congr_aux_updated       = False
        self.dder3_aux_updated       = False

        return
        
    def get_nu(self):
        return 1 + self.N
    
    def get_init_point(self, out):
        point = [
            np.array([[1.]]), 
            np.eye(self.N, dtype=self.dtype)
        ]

        self.set_point(point, point)

        out[0][:] = point[0]
        out[1][:] = point[1]

        return out
    
    def set_point(self, point, dual, a=True):
        self.t = point[0] * a
        self.X = point[1] * a
        self.Y = sym.p_tr(self.X, self.sys, (self.n0, self.n1))

        self.t_d = dual[0] * a
        self.X_d = dual[1] * a        

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

        self.log_X = (self.Ux * self.log_Dx) @ self.Ux.conj().T
        self.log_X = (self.log_X + self.log_X.conj().T) * 0.5
        self.log_Y = (self.Uy * self.log_Dy) @ self.Uy.conj().T
        self.log_Y = (self.log_Y + self.log_Y.conj().T) * 0.5
        
        self.log_XY = self.log_X - sym.i_kr(self.log_Y, self.sys, (self.n0, self.n1))
        self.z = self.t[0, 0] - lin.inp(self.X, self.log_XY)

        self.feas = (self.z > 0)
        return self.feas
    
    def get_val(self):
        return -np.log(self.z) - np.sum(self.log_Dx)
    
    def update_grad(self):
        assert self.feas_updated
        assert not self.grad_updated
        
        self.inv_Dx = np.reciprocal(self.Dx)
        inv_X_rt2 = self.Ux * np.sqrt(self.inv_Dx)
        self.inv_X = inv_X_rt2 @ inv_X_rt2.conj().T

        self.zi = np.reciprocal(self.z)
        self.DPhi = self.log_XY

        self.grad = [
           -self.zi,
            self.zi * self.DPhi - self.inv_X
        ]        

        self.grad_updated = True

    def get_grad(self, out):
        assert self.feas_updated
        if not self.grad_updated:
            self.update_grad()

        out[0][:] = self.grad[0]
        out[1][:] = self.grad[1]
        
        return out

    def hess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        # Computes Hessian product of the QCE barrier with a single vector (Ht, Hx)
        # See hess_congr() for additional comments

        (Ht, Hx) = H
        Hy = sym.p_tr(Hx, self.sys, (self.n0, self.n1))

        UxHxUx = self.Ux.conj().T @ Hx @ self.Ux
        UyHyUy = self.Uy.conj().T @ Hy @ self.Uy

        # Hessian product of conditional entropy
        D2PhiH = self.Ux @ (self.D1x_log * UxHxUx) @ self.Ux.conj().T
        D2PhiH -= sym.i_kr(self.Uy @ (self.D1y_log * UyHyUy) @ self.Uy.conj().T, self.sys, (self.n0, self.n1))

        # Hessian product of barrier function
        out[0][:] = (Ht - lin.inp(self.DPhi, Hx)) * self.zi2

        out_X     = -out[0] * self.DPhi
        out_X    +=  self.zi * D2PhiH
        out_X    +=  self.inv_X @ Hx @ self.inv_X
        out[1][:] = (out_X + out_X.conj().T) * 0.5

        return out

    def hess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)            

        p = A.shape[0]
        lhs = np.zeros((p, sum(self.dim)))

        # Precompute Hessian products for quantum conditional entropy
        # D2Phi(X)[Hx] =  Ux [log^[1](Dx) .* (Ux'     Hx  Ux)] Ux' 
        #              - [Uy [log^[1](Dy) .* (Uy' PTr(Hx) Uy)] Uy'] kron I
        sym.p_tr_multi(self.work1, self.Ax, self.sys, (self.n0, self.n1))
        lin.congr(self.work2, self.Uy.conj().T, self.work1, self.work3)
        self.work2 *= self.D1y_log
        lin.congr(self.work1, self.Uy, self.work2, self.work3)
        sym.i_kr_multi(self.Work0, self.work1, self.sys, (self.n0, self.n1))

        lin.congr(self.Work2, self.Ux.conj().T, self.Ax, self.Work3)
        self.Work2 *= self.D1x_log
        lin.congr(self.Work1, self.Ux, self.Work2, self.Work3)

        self.Work1 -= self.Work0

        # ====================================================================
        # Hessian products with respect to t
        # ====================================================================
        # D2_tt F(t, X)[Ht] =  Ht / z^2
        # D2_tX F(t, X)[Hx] = -DPhi(X)[Hx] / z^2
        outt  = self.At - (self.Ax.view(dtype=np.float64).reshape((p, 1, -1)) @ self.DPhi.view(dtype=np.float64).reshape((-1, 1))).ravel()
        outt *= self.zi2

        lhs[:, 0] = outt

        # ====================================================================
        # Hessian products with respect to X
        # ====================================================================
        # D2_Xt F(t, X)[Ht] = -Ht DPhi(X)[Hx] / z^2
        # D2_XX F(t, X)[Hx] =  DPhi(X)[Hx] DPhi(X) / z^2 + D2Phi(X)[Hx] / z + X^-1 Hx X^-1
        self.Work1 *= self.zi
        lin.congr(self.Work3, self.inv_X, self.Ax, self.Work2)
        self.Work1 += self.Work3
        self.Work1 -= np.outer(outt, self.DPhi).reshape((p, self.N, self.N))

        lhs[:, 1:] = self.Work1.reshape((p, -1)).view(dtype=np.float64)
        
        # Multiply A (H A')
        return lhs @ A.T

    def invhess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        # Computes inverse Hessian product of the QCE barrier with a single vector (Ht, Hx)
        # See invhess_congr() for additional comments

        (Ht, Hx) = H
        Wx = Hx + Ht * self.DPhi

        UxWxUx = self.Ux.conj().T @ Wx @ self.Ux
        Hxx_inv_x = self.Ux @ (self.D1x_comb_inv * UxWxUx) @ self.Ux.conj().T
        rhs_y = -sym.p_tr(Hxx_inv_x, self.sys, (self.n0, self.n1))
        temp = sym.mat_to_vec(rhs_y, hermitian=self.hermitian)
        H_inv_g_y = lin.fact_solve(self.Hy_KHxK_fact, temp)
        temp = sym.i_kr(sym.vec_to_mat(H_inv_g_y, hermitian=self.hermitian), self.sys, (self.n0, self.n1))

        temp = self.Ux.conj().T @ temp @ self.Ux
        H_inv_w_x = Hxx_inv_x - self.Ux @ (self.D1x_comb_inv * temp) @ self.Ux.conj().T
        H_inv_w_x = (H_inv_w_x + H_inv_w_x.conj().T) * 0.5

        out[0][:] = Ht * self.z2 + lin.inp(H_inv_w_x, self.DPhi)
        out[1][:] = H_inv_w_x

        return out

    def invhess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)            

        # The inverse Hessian product applied on (Ht, Hx) for the QCE barrier is 
        #     X = M \ Wx
        #     t = z^2 Ht + <DPhi(X), X>
        # where Wx = Hx + Ht DPhi(X)
        #     M = D2S(X) - PTr' D2S(PTr(X)) PTr
        #
        # Treating [PTr' D2S(PTr(X)) PTr] as a low-rank perturbation of D2S(X), we can solve 
        # linear systems with M by using the matrix inversion lemma
        #     X = [D2S(X)^-1 - D2S(X)^-1 PTr' N^-1 PTr D2S(X)^-1] Wx
        # where
        #     N = D2S(PTr(X))^-1 - PTr D2S(X)^-1 PTr'

        p = A.shape[0]
        lhs = np.zeros((p, sum(self.dim)))

        # ====================================================================
        # Inverse Hessian products with respect to X
        # ====================================================================
        # Compute Wx
        np.outer(self.At, self.DPhi, out=self.Work2.reshape((p, -1)))
        np.add(self.Ax, self.Work2, out=self.Work0)

        # Apply D2S(X)^-1 = (Ux kron Ux) log^[1](Dx) (Ux' kron Ux')
        lin.congr(self.Work2, self.Ux.conj().T, self.Work0, self.Work3)
        self.Work2 *= self.D1x_comb_inv
        lin.congr(self.Work0, self.Ux, self.Work2, self.Work3)
        # Apply PTr
        sym.p_tr_multi(self.work1, self.Work0, self.sys, (self.n0, self.n1))
        self.work1 *= -1

        # Solve linear system N \ ( ... )
        # Convert matrices to truncated real vectors
        work  = self.work1.view(dtype=np.float64).reshape((p, -1))[:, self.triu_indices]
        work *= self.scale
        # Solve system
        work = lin.fact_solve(self.Hy_KHxK_fact, work.T)
        # Expand truncated real vectors back into matrices
        self.work1.fill(0.)
        work[self.diag_indices, :] *= 0.5
        work /= self.scale.reshape((-1, 1))
        self.work1.view(dtype=np.float64).reshape((p, -1))[:, self.triu_indices] = work.T
        self.work1 += self.work1.conj().transpose((0, 2, 1))

        # Apply PTr' = IKr
        sym.i_kr_multi(self.Work1, self.work1, self.sys, (self.n0, self.n1))
        # Apply D2S(X)^-1 = (Ux kron Ux) log^[1](Dx) (Ux' kron Ux')
        lin.congr(self.Work2, self.Ux.conj().T, self.Work1, self.Work3)
        self.Work2 *= self.D1x_comb_inv
        lin.congr(self.Work1, self.Ux, self.Work2, self.Work3)

        # Subtract previous expression from D2S(X)^-1 Wx to get X
        self.Work0 -= self.Work1
        lhs[:, 1:] = self.Work0.reshape((p, -1)).view(dtype=np.float64)

        # ====================================================================
        # Inverse Hessian products with respect to t
        # ====================================================================
        outt  = self.z2 * self.At
        outt += (self.Work0.view(dtype=np.float64).reshape((p, 1, -1)) @ self.DPhi.view(dtype=np.float64).reshape((-1, 1))).ravel()
        lhs[:, 0] = outt

        # Multiply A (H A')
        return lhs @ A.T

    def third_dir_deriv_axpy(self, out, dir, a=True):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.dder3_aux_updated:
            self.update_dder3_aux()

        (Ht, Hx) = dir
        Hy = sym.p_tr(Hx, self.sys, (self.n0, self.n1))

        UxHxUx = self.Ux.conj().T @ Hx @ self.Ux
        UyHyUy = self.Uy.conj().T @ Hy @ self.Uy

        # Quantum conditional entropy oracles
        D2PhiH = self.Ux @ (self.D1x_log * UxHxUx) @ self.Ux.conj().T
        D2PhiH -= sym.i_kr(self.Uy @ (self.D1y_log * UyHyUy) @ self.Uy.conj().T, self.sys, (self.n0, self.n1))

        D3PhiHH = mgrad.scnd_frechet(self.D2x_log, UxHxUx, UxHxUx, self.Ux)
        D3PhiHH -= sym.i_kr(mgrad.scnd_frechet(self.D2y_log, UyHyUy, UyHyUy, self.Uy), self.sys, (self.n0, self.n1))

        # Third derivative of barrier
        DPhiH = lin.inp(self.DPhi, Hx)
        D2PhiHH = lin.inp(D2PhiH, Hx)
        chi = Ht - DPhiH

        dder3_t = -2 * (self.zi**3) * (chi**2) - (self.zi**2) * D2PhiHH

        dder3_X = -dder3_t * self.DPhi
        dder3_X -= 2 * (self.zi**2) * chi * D2PhiH
        dder3_X += self.zi * D3PhiHH
        dder3_X -= 2 * self.inv_X @ Hx @ self.inv_X @ Hx @ self.inv_X
        dder3_X = (dder3_X + dder3_X.conj().T) * 0.5

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
        temp = [np.zeros((1, 1)), np.zeros((self.N, self.N), dtype=self.dtype)]
        self.invhess_prod_ip(temp, psi)
        return lin.inp(temp[0], psi[0]) + lin.inp(temp[1], psi[1])

    # ========================================================================
    # Auxilliary functions
    # ========================================================================
    def congr_aux(self, A):
        assert not self.congr_aux_updated

        p = A.shape[0]

        self.At = A[:, 0]
        Ax = np.ascontiguousarray(A[:, 1:])

        if self.hermitian:
            self.Ax = np.array([Ax_k.reshape((-1, 2)).view(dtype=np.complex128).reshape((self.N, self.N)) for Ax_k in Ax])
        else:
            self.Ax = np.array([Ax_k.reshape((self.N, self.N)) for Ax_k in Ax])

        self.Work0 = np.empty_like(self.Ax, dtype=self.dtype)
        self.Work1 = np.empty_like(self.Ax, dtype=self.dtype)
        self.Work2 = np.empty_like(self.Ax, dtype=self.dtype)
        self.Work3 = np.empty_like(self.Ax, dtype=self.dtype)

        self.work1 = np.empty((p, self.n, self.n), dtype=self.dtype)
        self.work2 = np.empty((p, self.n, self.n), dtype=self.dtype)
        self.work3 = np.empty((p, self.n, self.n), dtype=self.dtype)

        self.congr_aux_updated = True

    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        self.D1x_log = mgrad.D1_log(self.Dx, self.log_Dx)
        self.D1y_log = mgrad.D1_log(self.Dy, self.log_Dy)

        # Preparing other required variables
        self.zi2 = self.zi * self.zi        

        self.hess_aux_updated = True

        return

    def update_dder3_aux(self):
        assert not self.dder3_aux_updated
        assert self.hess_aux_updated

        self.D2x_log = mgrad.D2_log(self.Dx, self.D1x_log)
        self.D2y_log = mgrad.D2_log(self.Dy, self.D1y_log)

        self.dder3_aux_updated = True

        return

    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated
        if not self.invhess_aux_aux_updated:
            self.update_invhessprod_aux_aux()

        irt2 = math.sqrt(0.5)
        self.z2 = self.z * self.z

        D1x_inv = np.reciprocal(np.outer(self.Dx, self.Dx))
        self.D1x_comb_inv = 1 / (self.D1x_log*self.zi + D1x_inv)
        sqrt_D1x_comb_inv = np.sqrt(self.D1x_comb_inv)

        UxK = np.empty((self.vn, self.vN))
        Hy_inv = np.empty((self.vn, self.vn))

        k = 0
        for j in range(self.n):
            for i in range(j):
                # Build UxK matrix
                if self.sys == 0:
                    lhs = self.Ux.conj().T[:, slice(i, self.vN, self.n1)]
                    rhs = self.Ux[slice(j, self.vN, self.n1), :]
                    UxK_k = (lhs @ rhs) * irt2
                else:
                    lhs = self.Ux.conj().T[:, self.n1*i:self.n1*(i+1)]
                    rhs = self.Ux[self.n1*j:self.n1*(j+1), :]
                    UxK_k = (lhs @ rhs) * irt2
                UxK_k *= sqrt_D1x_comb_inv
                UxK[[k], :] = (sym.mat_to_vec(UxK_k + UxK_k.conj().T, hermitian=self.hermitian)).T

                # Build Hyy^-1 matrix
                UyH_k = self.Uy.conj().T[:, [i]] @ self.Uy[[j], :] * irt2
                UyH_k = self.Uy @ (UyH_k * self.z / self.D1y_log) @ self.Uy.conj().T
                Hy_inv[:, [k]] = sym.mat_to_vec(UyH_k + UyH_k.conj().T, hermitian=self.hermitian)

                k += 1

                if self.hermitian:
                    UxK_k *= 1j
                    UxK[[k], :] = (sym.mat_to_vec(UxK_k + UxK_k.conj().T, hermitian=self.hermitian)).T

                    UyH_k *= 1j
                    Hy_inv[:, [k]] = sym.mat_to_vec(UyH_k + UyH_k.conj().T, hermitian=self.hermitian)

                    k += 1

            # Build UxK matrix
            if self.sys == 0:
                lhs = self.Ux.conj().T[:, slice(j, self.vN, self.n1)]
                rhs = self.Ux[slice(j, self.vN, self.n1), :]
                UxK_k = lhs @ rhs
            else:
                lhs = self.Ux.conj().T[:, self.n1*j:self.n1*(j+1)]
                rhs = self.Ux[self.n1*j:self.n1*(j+1), :]
                UxK_k = lhs @ rhs
            UxK_k *= sqrt_D1x_comb_inv
            UxK[[k], :] = (sym.mat_to_vec(UxK_k, hermitian=self.hermitian)).T

            UyH_k = self.Uy.conj().T[:, [j]] @ self.Uy[[j], :]
            UyH_k = self.Uy @ (UyH_k * self.z / self.D1y_log) @ self.Uy.conj().T
            Hy_inv[:, [k]] = sym.mat_to_vec(UyH_k, hermitian=self.hermitian)

            k += 1


        KHxK = UxK @ UxK.T
        Hy_KHxK = Hy_inv - KHxK
        self.Hy_KHxK_fact = lin.fact(Hy_KHxK)

        self.invhess_aux_updated = True

        return

    def update_invhessprod_aux_aux(self):
        assert not self.invhess_aux_aux_updated

        if self.hermitian:
            self.diag_indices = np.append(0, np.cumsum([i for i in range(3, 2*self.n+1, 2)]))
            self.triu_indices = np.empty(self.n*self.n, dtype=int)
            self.scale        = np.empty(self.n*self.n)
            k = 0
            for j in range(self.n):
                for i in range(j):
                    self.triu_indices[k]     = 2 * (j + i*self.n)
                    self.triu_indices[k + 1] = 2 * (j + i*self.n) + 1
                    self.scale[k:k+2]        = np.sqrt(2.)
                    k += 2
                self.triu_indices[k] = 2 * (j + j*self.n)
                self.scale[k]        = 1.
                k += 1
        else:
            self.diag_indices = np.append(0, np.cumsum([i for i in range(2, self.n+1, 1)]))
            self.triu_indices = np.array([j + i*self.n for j in range(self.n) for i in range(j + 1)])
            self.scale = np.array([1 if i==j else np.sqrt(2.) for j in range(self.n) for i in range(j + 1)])

        self.invhess_aux_aux_updated = True