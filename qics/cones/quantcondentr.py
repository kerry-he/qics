import numpy as np
import qics.utils.linear as lin
import qics.utils.mtxgrad as mgrad
import qics.utils.symmetric as sym
from qics.cones.base import Cone

class QuantCondEntr(Cone):
    """A class representing a quantum conditional entropy cone
    
        K = { (t, X) ∈ R x H^n0n1 : t >= S(X) - S(pTr(X)), X ⪰ 0 },
        
    with barrier function
    
        F(t, X) = -log(t - S(X) + S(pTr(X))) - logdet(X),
        
    where

        S(X) = -tr[X log(X)],
        
    is the quantum (von Neumann) entropy, and pTr is the partial trace.
    """    
    def __init__(self, n0, n1, sys, iscomplex=False):
        """Initialize a QuantCondEntr instance

        Parameters
        ----------
        n0 : int
            Dimension of the first system.
        n1 : int
            Dimension of the second system.
        sys : int
            Which system is being traced out by the partial trace (either 0 or 1).
        iscomplex : bool
            Whether the matrix X is symmetric (False) or Hermitian (True). Default is False.
        """              
        # Dimension properties
        self.n0 = n0          # Dimension of system 0
        self.n1 = n1          # Dimension of system 1
        self.N  = n0 * n1     # Total dimension of bipartite system
        self.nu = 1 + self.N  # Barrier parameter
        self.iscomplex = iscomplex

        self.sys   = sys                       # System being traced out
        self.n = n0 if (sys == 1) else n1      # Dimension of system not traced out
        self.m = n1 if (sys == 1) else n0      # Dimension of system traced out

        self.vn = self.n*self.n if iscomplex else self.n*(self.n+1)//2      # Compact dimension of vectorized system being traced out

        self.dim   = [1, self.N*self.N] if (not iscomplex) else [1, 2*self.N*self.N]
        self.type  = ['r', 's']         if (not iscomplex) else ['r', 'h']
        self.dtype = np.float64         if (not iscomplex) else np.complex128        

        # Update flags
        self.feas_updated            = False
        self.grad_updated            = False
        self.hess_aux_updated        = False
        self.invhess_aux_updated     = False
        self.invhess_aux_aux_updated = False
        self.congr_aux_updated       = False
        self.dder3_aux_updated       = False

        self.precompute_mat_vec()

        return

    def get_iscomplex(self):
        return self.iscomplex

    def get_init_point(self, out):
        # This gives the central point satisfying x = -F'(x)
        a = self.N * np.log(self.m)**2 + 1
        b = -(2. + (1 + self.N) * self.N * np.log(self.m)**2)

        t0 = np.sqrt((-b - np.sqrt(b*b - 4*a)) / (2*a))
        x0 = np.sqrt((1 + self.N - t0*t0) / self.N)

        point = [
            np.array([[t0]]), 
            np.eye(self.N, dtype=self.dtype) * x0
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
        self.Y = sym.p_tr(self.X, self.sys, (self.n0, self.n1))

        self.Dx, self.Ux = np.linalg.eigh(self.X)
        self.Dy, self.Uy = np.linalg.eigh(self.Y)

        if any(self.Dx <= 0) or any(self.Dy <= 0):
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
        self.work2 *= self.D1y_log * self.zi
        lin.congr(self.work1, self.Uy, self.work2, self.work3)
        sym.i_kr_multi(self.Work0, self.work1, self.sys, (self.n0, self.n1))

        lin.congr(self.Work2, self.Ux.conj().T, self.Ax, self.Work3)
        self.Work2 *= self.D1x_comb
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
        # D2_Xt F(t, X)[Ht] = -Ht DPhi(X) / z^2
        # D2_XX F(t, X)[Hx] =  DPhi(X)[Hx] DPhi(X) / z^2 + D2Phi(X)[Hx] / z + X^-1 Hx X^-1
        np.outer(outt, self.DPhi, out=self.Work0.reshape(p, -1))
        self.Work1 -= self.Work0

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

        # Apply D2S(X)^-1 = (Ux kron Ux) log^[1](Dx) (Ux' kron Ux')
        UxWxUx = self.Ux.conj().T @ Wx @ self.Ux
        Hxx_inv_x = self.Ux @ (self.D1x_comb_inv * UxWxUx) @ self.Ux.conj().T
        work = -sym.p_tr(Hxx_inv_x, self.sys, (self.n0, self.n1))

        # Solve linear system N \ ( ... )
        temp = work.view(dtype=np.float64).reshape((-1, 1))[self.triu_idxs]
        temp *= self.scale.reshape((-1, 1))

        H_inv_g_y = lin.cho_solve(self.Hy_KHxK_fact, temp)

        work.fill(0.)
        H_inv_g_y[self.diag_idxs] *= 0.5
        H_inv_g_y /= self.scale.reshape((-1, 1))
        work.view(dtype=np.float64).reshape((-1, 1))[self.triu_idxs] = H_inv_g_y
        work += work.conj().T

        # Apply PTr' = IKr
        temp = sym.i_kr(work, self.sys, (self.n0, self.n1))
        # Apply D2S(X)^-1 = (Ux kron Ux) log^[1](Dx) (Ux' kron Ux')
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
        #     M = 1/z D2S(X) - 1/z PTr' D2S(PTr(X)) PTr + X^-1 kron X^-1
        #       = (Ux kron Ux) (1/z log + inv)^[1](Dx) (Ux' kron Ux')
        #         - 1/z PTr' (Uy kron Uy) log^[1](Dy) (Uy' kron Uy') PTr
        #
        # Treating [PTr' D2S(PTr(X)) PTr] as a low-rank perturbation of D2S(X), we can solve 
        # linear systems with M by using the matrix inversion lemma
        #     X = [D2S(X)^-1 - D2S(X)^-1 PTr' N^-1 PTr D2S(X)^-1] Wx
        # where
        #     N = 1/z (Uy kron Uy) [log^[1](Dy)]^-1 (Uy' kron Uy') 
        #         - PTr (Ux kron Ux) [(1/z log + inv)^[1](Dx)]^-1  (Ux' kron Ux') PTr'

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
        work  = self.work1.view(dtype=np.float64).reshape((p, -1))[:, self.triu_idxs]
        work *= self.scale
        # Solve system
        work = lin.cho_solve(self.Hy_KHxK_fact, work.T)
        # Expand truncated real vectors back into matrices
        self.work1.fill(0.)
        work[self.diag_idxs, :] *= 0.5
        work /= self.scale.reshape((-1, 1))
        self.work1.view(dtype=np.float64).reshape((p, -1))[:, self.triu_idxs] = work.T
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

    def third_dir_deriv_axpy(self, out, H, a=True):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.dder3_aux_updated:
            self.update_dder3_aux()

        (Ht, Hx) = H
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

    # ========================================================================
    # Auxilliary functions
    # ========================================================================
    def congr_aux(self, A):
        assert not self.congr_aux_updated

        p = A.shape[0]

        self.At = A[:, 0]
        Ax = np.ascontiguousarray(A[:, 1:])

        if self.iscomplex:
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

        D1x_inv       = np.reciprocal(np.outer(self.Dx, self.Dx))
        self.D1x_log  = mgrad.D1_log(self.Dx, self.log_Dx)
        self.D1x_comb = self.zi * self.D1x_log + D1x_inv
        
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

        # Precompute and factorize the matrix
        #     N = 1/z (Uy kron Uy) [log^[1](Dy)]^-1 (Uy' kron Uy')
        #         - PTr (Ux kron Ux) [(1/z log + inv)^[1](Dx)]^-1  (Ux' kron Ux') PTr'
        # which we will need to solve linear systems with the Hessian of our barrier function

        self.z2 = self.z * self.z
        self.D1x_comb_inv = np.reciprocal(self.D1x_comb)

        # Get [1/z (Uy kron Uy) [log^[1](Dy)]^-1 (Uy' kron Uy')] matrix
        # Begin with (Uy' kron Uy')
        lin.congr(self.work8, self.Uy.conj().T, self.E, work=self.work7)
        # Apply z [log^[1](Dy)]^-1
        self.work8 *= (self.z * np.reciprocal(self.D1y_log))
        # Apply (Uy kron Uy)
        lin.congr(self.work6, self.Uy, self.work8, work=self.work7)

        # Get [PTr (Ux kron Ux) [(1/z log + inv)^[1](Dx)]^-1  (Ux' kron Ux') PTr'] matrix
        # Begin with [(Ux' kron Ux') PTr']
        temp = self.Ux.T.reshape(self.N, self.n0, self.n1)
        if self.sys == 1:
            lhs = np.copy(temp.conj().transpose(1, 0, 2))  # self.n0, self.N, self.n1
            rhs = np.copy(temp.transpose(1, 2, 0))         # self.n0, self.n1, self.N
        else:
            lhs = np.copy(temp.conj().transpose(2, 0, 1))  # self.n1, self.N, self.n0
            rhs = np.copy(temp.transpose(2, 1, 0))         # self.n1, self.n0, self.N

        np.matmul(lhs, rhs, out=self.Work9)
        self.Work8[self.diag_idxs] = self.Work9
        rhs *= np.sqrt(0.5)
        t = 0
        for j in range(self.n):
            np.matmul(lhs[j], rhs[:j], out=self.Work9[:j])

            if self.iscomplex:
                np.add(self.Work9[:j], self.Work9[:j].conj().transpose(0, 2, 1), out=self.Work8[t : t+2*j : 2])
                np.subtract(self.Work9[:j], self.Work9[:j].conj().transpose(0, 2, 1), out=self.Work8[t+1 : t+2*j+1 : 2])
                self.Work8[t+1 : t+2*j+1 : 2] *= -1j
                t += 2*j + 1
            else:
                np.add(self.Work9[:j], self.Work9[:j].transpose(0, 2, 1), out=self.Work8[t : t+j])
                t += j + 1
        # Apply [(1/z log + inv)^[1](Dx)]^-1/2
        self.Work8 *= self.D1x_comb_inv
        # Apply PTr (Ux kron Ux)
        lin.congr(self.Work6, self.Ux, self.Work8, work=self.Work7)
        sym.p_tr_multi(self.work7, self.Work6, self.sys, (self.n0, self.n1))

        # Subtract to obtain N then Cholesky factor
        self.work6 -= self.work7
        work  = self.work6.view(dtype=np.float64).reshape((self.vn, -1))[:, self.triu_idxs]
        work *= self.scale
        self.Hy_KHxK_fact = lin.cho_fact(work) 

        self.invhess_aux_updated = True

        return

    def update_invhessprod_aux_aux(self):
        assert not self.invhess_aux_aux_updated

        self.work6  = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work7  = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work8  = np.empty((self.vn, self.n, self.n), dtype=self.dtype)

        self.Work6  = np.empty((self.vn, self.N, self.N), dtype=self.dtype)
        self.Work7  = np.empty((self.vn, self.N, self.N), dtype=self.dtype)
        self.Work8  = np.empty((self.vn, self.N, self.N), dtype=self.dtype)
        self.Work9  = np.empty((self.n, self.N, self.N), dtype=self.dtype)

        self.invhess_aux_aux_updated = True