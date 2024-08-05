import numpy as np
import qics.utils.linalg as lin
import qics.utils.gradient as grad
from qics.cones.base import Cone, get_central_ray_relentr

class QuantRelEntr(Cone):
    """A class representing a classical relative entropy cone

    .. math::
    
        \\mathcal{K}_{\\text{qre}} = \\text{cl}\\{ (t, X, Y) \\in \\mathbb{R} \\times \\mathbb{H}^n_{++} \\times \\mathbb{H}^n_{++} : t \\geq S(X \\| Y) \\},
        
    with barrier function
    
    .. math::

        (t, X, Y) \\mapsto -\\log(t - S(x \\| y)) - \\log \\det(X) - \\log \\det(Y),
        
    where

    .. math::

        S(X \\| Y) = \\text{tr}[X \\log(X)] - \\text{tr}[X \\log(Y)],
        
    is the quantum (Umegaki) relative entropy function.

    Parameters
    ----------
    n : int
        Dimension of the (n, n) matrices :math:`X` and :math:`Y`.
    iscomplex : bool
        Whether the matrices symmetric :math:`X,Y \\in \\mathbb{S}^n` (False) or Hermitian :math:`X,Y \\in \\mathbb{H}^n` (True). Default is False.
    """    
    def __init__(self, n, iscomplex=False):      
        # Dimension properties
        self.n  = n               # Side dimension of system
        self.nu = 1 + 2 * self.n  # Barrier parameter

        self.iscomplex = iscomplex                      # Hermitian or symmetric vector space
        self.vn = n*n if iscomplex else n*(n+1)//2      # Compact dimension of system

        self.dim   = [1, n*n, n*n]   if (not iscomplex) else [1, 2*n*n, 2*n*n]
        self.type  = ['r', 's', 's'] if (not iscomplex) else ['r', 'h', 'h']
        self.dtype = np.float64      if (not iscomplex) else np.complex128

        self.idx_X = slice(1, 1 + self.dim[1])
        self.idx_Y = slice(1 + self.dim[1], sum(self.dim))

        # Update flags
        self.feas_updated            = False
        self.grad_updated            = False
        self.hess_aux_updated        = False
        self.invhess_aux_updated     = False
        self.invhess_aux_aux_updated = False
        self.dder3_aux_updated       = False
        self.congr_aux_updated       = False

        self.precompute_mat_vec()

        return

    def get_iscomplex(self):
        return self.iscomplex

    def get_init_point(self, out):
        (t0, x0, y0) = get_central_ray_relentr(self.n)

        point = [
            np.array([[t0]]), 
            np.eye(self.n, dtype=self.dtype) * x0,
            np.eye(self.n, dtype=self.dtype) * y0,
        ]

        self.set_point(point, point)

        out[0][:] = point[0]
        out[1][:] = point[1]
        out[2][:] = point[2]

        return out

    def get_feas(self):
        if self.feas_updated:
            return self.feas
        
        self.feas_updated = True

        (self.t, self.X, self.Y) = self.primal

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

        self.log_XY = self.log_X - self.log_Y
        self.z = self.t[0, 0] - lin.inp(self.X, self.log_XY)

        self.feas = (self.z > 0)
        return self.feas

    def get_val(self):
        assert self.feas_updated

        return -np.log(self.z) - np.sum(self.log_Dx) - np.sum(self.log_Dy)

    def update_grad(self):
        assert self.feas_updated
        assert not self.grad_updated
        
        self.inv_Dx = np.reciprocal(self.Dx)
        self.inv_Dy = np.reciprocal(self.Dy)

        inv_X_rt2 = self.Ux * np.sqrt(self.inv_Dx)
        self.inv_X = inv_X_rt2 @ inv_X_rt2.conj().T
        inv_Y_rt2 = self.Uy * np.sqrt(self.inv_Dy)
        self.inv_Y = inv_Y_rt2 @ inv_Y_rt2.conj().T        

        self.D1y_log = grad.D1_log(self.Dy, self.log_Dy)

        self.UyXUy = self.Uy.conj().T @ self.X @ self.Uy

        self.zi    = np.reciprocal(self.z)
        self.DPhiX = self.log_XY + np.eye(self.n)
        self.DPhiY = -self.Uy @ (self.D1y_log * self.UyXUy) @ self.Uy.conj().T
        self.DPhiY = (self.DPhiY + self.DPhiY.conj().T) * 0.5

        self.grad = [
           -self.zi,
            self.zi * self.DPhiX - self.inv_X,
            self.zi * self.DPhiY - self.inv_Y
        ]

        self.grad_updated = True

    def hess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        # Computes Hessian product of the QRE barrier with a single vector (Ht, Hx, Hy)
        # See hess_congr() for additional comments

        (Ht, Hx, Hy) = H

        UxHxUx = self.Ux.conj().T @ Hx @ self.Ux
        UyHxUy = self.Uy.conj().T @ Hx @ self.Uy
        UyHyUy = self.Uy.conj().T @ Hy @ self.Uy

        # Hessian product of relative entropy
        D2PhiXXH =  self.Ux @ (self.D1x_log * UxHxUx) @ self.Ux.conj().T
        D2PhiXYH = -self.Uy @ (self.D1y_log * UyHyUy) @ self.Uy.conj().T
        D2PhiYXH = -self.Uy @ (self.D1y_log * UyHxUy) @ self.Uy.conj().T
        D2PhiYYH = -grad.scnd_frechet(self.D2y_log_UXU, UyHyUy, U=self.Uy)
        
        # Hessian product of barrier function
        out[0][:] = (Ht - lin.inp(self.DPhiX, Hx) - lin.inp(self.DPhiY, Hy)) * self.zi2

        out_X     = -out[0] * self.DPhiX
        out_X    +=  self.zi * (D2PhiXYH + D2PhiXXH)
        out_X    +=  self.inv_X @ Hx @ self.inv_X
        out_X     = (out_X + out_X.conj().T) * 0.5
        out[1][:] = out_X

        out_Y     = -out[0] * self.DPhiY
        out_Y    +=  self.zi * (D2PhiYXH + D2PhiYYH)
        out_Y    +=  self.inv_Y @ Hy @ self.inv_Y
        out_Y     = (out_Y + out_Y.conj().T) * 0.5
        out[2][:] = out_Y

        return out

    def hess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)

        p = A.shape[0]
        lhs = np.zeros((p, sum(self.dim)))

        # Precompute Hessian products for quantum relative entropy
        # D2_XX Phi(X, Y) [Hx] =  Ux [log^[1](Dx) .* (Ux' Hx Ux)] Ux'
        # D2_YX Phi(X, Y) [Hx] = -Uy [log^[1](Dy) .* (Uy' Hx Uy)] Uy'
        # D2_XY Phi(X, Y) [Hy] = -Uy [log^[1](Dy) .* (Uy' Hy Uy)] Uy'
        # D2_YY Phi(X, Y) [Hy] = -Uy [SUM_k log_k^[2](Dy) .* ([Uy' X Uy]_k [Uk' Hy Uy]_k' + [Uy' Hy Uy]_k [Uk' X Uy]_k') ] Uy'
        lin.congr_multi(self.work1, self.Ux.conj().T, self.Ax, self.work2)
        self.work1 *= self.D1x_comb
        lin.congr_multi(self.D2PhiXXH, self.Ux, self.work1, self.work2)

        lin.congr_multi(self.work1, self.Uy.conj().T, self.Ax, self.work2)
        self.work1 *= -self.zi * self.D1y_log
        lin.congr_multi(self.D2PhiYXH, self.Uy, self.work1, self.work2)

        lin.congr_multi(self.work1, self.Uy.conj().T, self.Ay, self.work2)
        grad.scnd_frechet_multi(self.D2PhiYYH, self.D2y_comb, self.work1, U=self.Uy, work1=self.work2, work2=self.work3, work3=self.work5)

        self.work1 *= self.D1y_log * self.zi
        lin.congr_multi(self.D2PhiXYH, self.Uy, self.work1, self.work2)

        # ====================================================================
        # Hessian products with respect to t
        # ====================================================================
        # D2_tt F(t, X, Y)[Ht] = Ht / z^2
        # D2_tX F(t, X, Y)[Hx] = -(D_X Phi(X, Y) [Hx]) / z^2
        # D2_tY F(t, X, Y)[Hy] = -(D_Y Phi(X, Y) [Hy]) / z^2
        outt  = self.At - (self.Ax.view(dtype=np.float64).reshape((p, 1, -1)) @ self.DPhiX.view(dtype=np.float64).reshape((-1, 1))).ravel()
        outt -= (self.Ay.view(dtype=np.float64).reshape((p, 1, -1)) @ self.DPhiY.view(dtype=np.float64).reshape((-1, 1))).ravel()
        outt *= self.zi2

        lhs[:, 0] = outt

        # ====================================================================
        # Hessian products with respect to X
        # ====================================================================
        # D2_Xt F(t, X, Y)[Ht] = -Ht (D_X Phi(X, Y)) / z^2
        # D2_XX F(t, X, Y)[Hx] = (D_X Phi(X, Y) [Hx]) D_X Phi(X, Y) / z^2 + (D2_XX Phi(X, Y) [Hx]) / z + X^-1 Hx X^-1
        # D2_XY F(t, X, Y)[Hy] = (D_Y Phi(X, Y) [Hy]) D_X Phi(X, Y) / z^2 + (D2_XY Phi(X, Y) [Hy]) / z
        np.subtract(self.D2PhiXXH, self.D2PhiXYH, out=self.work1)
        np.outer(outt, self.DPhiX, out=self.work2.reshape((p, -1)))
        self.work1 -= self.work2

        lhs[:, self.idx_X] = self.work1.reshape((p, -1)).view(dtype=np.float64)

        # ====================================================================
        # Hessian products with respect to Y
        # ====================================================================
        # D2_Yt F(t, X, Y)[Ht] = -Ht (D_X Phi(X, Y)) / z^2
        # D2_YX F(t, X, Y)[Hx] = (D_X Phi(X, Y) [Hx]) D_Y Phi(X, Y) / z^2 + (D2_YX Phi(X, Y) [Hx]) / z
        # D2_YY F(t, X, Y)[Hy] = (D_Y Phi(X, Y) [Hy]) D_Y Phi(X, Y) / z^2 + (D2_YY Phi(X, Y) [Hy]) / z + Y^-1 Hy Y^-1
        np.add(self.D2PhiYYH, self.D2PhiYXH, out=self.work1)
        np.outer(outt, self.DPhiY, out=self.work2.reshape((p, -1)))
        self.work1 -= self.work2

        lhs[:, self.idx_Y] = self.work1.reshape((p, -1)).view(dtype=np.float64)

        # Multiply A (H A')
        return lhs @ A.T        
    
    def invhess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        # Computes inverse Hessian product of the QRE barrier with a single vector (Ht, Hx, Hy)
        # See invhess_congr() for additional comments

        (Ht, Hx, Hy) = H

        Wx = Hx + Ht * self.DPhiX
        Wy = Hy + Ht * self.DPhiY

        # Inverse Hessian products with respect to Y
        temp = self.Ux.conj().T @ Wx @ self.Ux
        temp = self.UyUx @ (self.D1x_comb_inv * temp) @ self.UyUx.conj().T
        temp = -self.Uy @ (self.zi * self.D1y_log * temp) @ self.Uy.conj().T
        temp = self.Uy.conj().T @ (Wy - temp) @ self.Uy

        temp_vec = temp.view(dtype=np.float64).reshape((-1, 1))[self.triu_idxs]
        temp_vec *= self.scale.reshape((-1, 1))

        temp_vec = lin.cho_solve(self.hess_schur_fact, temp_vec)

        temp.fill(0.)
        temp_vec[self.diag_idxs] *= 0.5
        temp_vec /= self.scale.reshape((-1, 1))
        temp.view(dtype=np.float64).reshape((-1, 1))[self.triu_idxs] = temp_vec
        temp += temp.conj().T

        temp = self.Uy @ temp @ self.Uy.conj().T
        temp = (temp + temp.conj().T) * 0.5
        out[2][:] = temp

        # Inverse Hessian products with respect to X
        temp = self.Uy.conj().T @ out[2] @ self.Uy
        temp = -self.Uy @ (self.zi * self.D1y_log * temp) @ self.Uy.conj().T
        temp = Wx - temp
        temp = self.Ux.conj().T @ temp @ self.Ux
        temp = self.Ux @ (self.D1x_comb_inv * temp) @ self.Ux.conj().T
        temp = (temp + temp.conj().T) * 0.5
        out[1][:] = temp

        # Inverse Hessian products with respect to t
        out[0][:] = self.z2 * Ht + lin.inp(self.DPhiX, out[1]) + lin.inp(self.DPhiY, out[2])

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

        p = self.Ax.shape[0]
        lhs = np.empty((p, sum(self.dim)))
        
        # ====================================================================
        # Inverse Hessian products with respect to Y
        # ====================================================================
        # Compute ({Uy' Wy Uy} - [1/z log^[1](Dy) (Uy'Ux kron Uy'Ux) (1/z log^[1](Dx) + Dx^-1 kron Dx^-1)^-1 {Ux' Wx Ux}])
        # Compute Ux' Wx Ux
        np.outer(self.At, self.DPhiX, out=self.work2.reshape((p, -1)))
        np.add(self.Ax, self.work2, out=self.work0)
        lin.congr_multi(self.work2, self.Ux.conj().T, self.work0, self.work3)
        # Apply (1/z log^[1](Dx) + Dx^-1 kron Dx^-1)^-1
        self.work2 *= self.D1x_comb_inv
        # Apply (Uy'Ux kron Uy'Ux)
        lin.congr_multi(self.work1, self.UyUx, self.work2, self.work3)
        # Apply -1/z log^[1](Dy)
        self.work1 *= (-self.zi * self.D1y_log)
        # Compute Uy' Wy Uy and subtract previous expression
        np.outer(self.At, self.DPhiY, out=self.work2.reshape((p, -1)))
        np.add(self.Ay, self.work2, out=self.work3)
        lin.congr_multi(self.work2, self.Uy.conj().T, self.work3, self.work4)
        self.work2 -= self.work1

        # Solve the linear system S \ ( ... ) to obtain Uy' Y Uy
        # Convert matrices to truncated real vectors
        work  = self.work2.view(dtype=np.float64).reshape((p, -1))[:, self.triu_idxs]
        work *= self.scale
        # Solve system
        work = lin.cho_solve(self.hess_schur_fact, work.T)
        # Expand truncated real vectors back into matrices
        self.work1.fill(0.)
        work[self.diag_idxs, :] *= 0.5
        work /= self.scale.reshape((-1, 1))
        self.work1.view(dtype=np.float64).reshape((p, -1))[:, self.triu_idxs] = work.T
        self.work1 += self.work1.conj().transpose((0, 2, 1))

        # Recover Y
        lin.congr_multi(self.work4, self.Uy, self.work1, self.work2)
        lhs[:, self.idx_Y] = self.work4.reshape((p, -1)).view(dtype=np.float64)

        # ====================================================================
        # Inverse Hessian products with respect to X
        # ====================================================================
        # Apply -1/z log^[1](Dy)
        self.work1 *= (-self.zi * self.D1y_log)
        # Apply (Uy kron Uy)
        lin.congr_multi(self.work2, self.Uy, self.work1, self.work3)
        # Subtract Wx from previous expression
        self.work0 -= self.work2
        # Apply (Ux kron Ux)
        lin.congr_multi(self.work1, self.Ux.conj().T, self.work0, self.work3)
        # Apply (1/z log^[1](Dx) + Dx^-1 kron Dx^-1)^-1 to obtian Ux' X Ux
        self.work1 *= self.D1x_comb_inv

        # Recover X
        lin.congr_multi(self.work2, self.Ux, self.work1, self.work3)
        lhs[:, self.idx_X] = self.work2.reshape((p, -1)).view(dtype=np.float64)

        # ====================================================================
        # Inverse Hessian products with respect to t
        # ====================================================================
        outt  = self.z2 * self.At 
        outt += (self.work2.view(dtype=np.float64).reshape((p, 1, -1)) @ self.DPhiX.view(dtype=np.float64).reshape((-1, 1))).ravel()
        outt += (self.work4.view(dtype=np.float64).reshape((p, 1, -1)) @ self.DPhiY.view(dtype=np.float64).reshape((-1, 1))).ravel()
        lhs[:, 0] = outt

        # Multiply A (H A')
        return lhs @ A.T

    def third_dir_deriv_axpy(self, out, H, a=True):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.dder3_aux_updated:
            self.update_dder3_aux()

        (Ht, Hx, Hy) = H

        chi = Ht[0, 0] - lin.inp(self.DPhiX, Hx) - lin.inp(self.DPhiY, Hy)
        chi2 = chi * chi

        UxHxUx = self.Ux.conj().T @ Hx @ self.Ux
        UyHyUy = self.Uy.conj().T @ Hy @ self.Uy
        UyHxUy = self.Uy.conj().T @ Hx @ self.Uy

        # Quantum relative entropy Hessians
        D2PhiXXH =  self.Ux @ (self.D1x_log * UxHxUx) @ self.Ux.conj().T
        D2PhiXYH = -self.Uy @ (self.D1y_log * UyHyUy) @ self.Uy.conj().T
        D2PhiYXH = -self.Uy @ (self.D1y_log * UyHxUy) @ self.Uy.conj().T
        D2PhiYYH = -grad.scnd_frechet(self.D2y_log_UXU, UyHyUy, U=self.Uy)

        D2PhiXHH = lin.inp(Hx, D2PhiXXH + D2PhiXYH)
        D2PhiYHH = lin.inp(Hy, D2PhiYXH + D2PhiYYH)

        # Quantum relative entropy third order derivatives
        D3PhiXXX =  grad.scnd_frechet(self.D2x_log, UxHxUx, UxHxUx, self.Ux)
        D3PhiXYY = -grad.scnd_frechet(self.D2y_log, UyHyUy, UyHyUy, self.Uy)

        D3PhiYYX = -grad.scnd_frechet(self.D2y_log, UyHyUy, UyHxUy, self.Uy)
        D3PhiYXY = D3PhiYYX
        D3PhiYYY = -grad.thrd_frechet(self.Dy, self.D2y_log, 2*(self.inv_Dy**3), self.Uy, self.UyXUy, UyHyUy, UyHyUy)
        
        # Third derivatives of barrier
        dder3_t = -2 * self.zi3 * chi2 - self.zi2 * (D2PhiXHH + D2PhiYHH)

        dder3_X  = -dder3_t * self.DPhiX
        dder3_X -=  2 * self.zi2 * chi * (D2PhiXXH + D2PhiXYH)
        dder3_X +=  self.zi * (D3PhiXXX + D3PhiXYY)
        dder3_X -=  2 * self.inv_X @ Hx @ self.inv_X @ Hx @ self.inv_X
        dder3_X  = (dder3_X + dder3_X.conj().T) * 0.5

        dder3_Y  = -dder3_t * self.DPhiY
        dder3_Y -=  2 * self.zi2 * chi * (D2PhiYXH + D2PhiYYH)
        dder3_Y +=  self.zi * (D3PhiYYX + D3PhiYXY + D3PhiYYY)
        dder3_Y -=  2 * self.inv_Y @ Hy @ self.inv_Y @ Hy @ self.inv_Y
        dder3_Y  = (dder3_Y + dder3_Y.conj().T) * 0.5

        out[0][:] += dder3_t * a
        out[1][:] += dder3_X * a
        out[2][:] += dder3_Y * a

        return out
    
    # ========================================================================
    # Auxilliary functions
    # ========================================================================
    def congr_aux(self, A):
        assert not self.congr_aux_updated

        self.At = A[:, 0]
        Ax = np.ascontiguousarray(A[:, self.idx_X])
        Ay = np.ascontiguousarray(A[:, self.idx_Y])

        if self.iscomplex:
            self.Ax = np.array([Ax_k.reshape((-1, 2)).view(dtype=np.complex128).reshape((self.n, self.n)) for Ax_k in Ax])
            self.Ay = np.array([Ay_k.reshape((-1, 2)).view(dtype=np.complex128).reshape((self.n, self.n)) for Ay_k in Ay])            
        else:
            self.Ax = np.array([Ax_k.reshape((self.n, self.n)) for Ax_k in Ax])
            self.Ay = np.array([Ay_k.reshape((self.n, self.n)) for Ay_k in Ay])

        self.work0 = np.empty_like(self.Ax, dtype=self.dtype)
        self.work1 = np.empty_like(self.Ax, dtype=self.dtype)
        self.work2 = np.empty_like(self.Ax, dtype=self.dtype)
        self.work3 = np.empty_like(self.Ax, dtype=self.dtype)
        self.work4 = np.empty_like(self.Ax, dtype=self.dtype)
        self.work5 = np.empty((self.Ax.shape[::-1]), dtype=self.dtype)

        self.D2PhiXXH = np.empty_like(self.Ax, dtype=self.dtype)
        self.D2PhiYXH = np.empty_like(self.Ax, dtype=self.dtype)
        self.D2PhiXYH = np.empty_like(self.Ay, dtype=self.dtype)
        self.D2PhiYYH = np.empty_like(self.Ay, dtype=self.dtype)

        self.congr_aux_updated = True

    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        D1x_inv       = np.reciprocal(np.outer(self.Dx, self.Dx))
        self.D1x_log  = grad.D1_log(self.Dx, self.log_Dx)
        self.D1x_comb = self.zi * self.D1x_log + D1x_inv

        self.D2y_log = grad.D2_log(self.Dy, self.D1y_log)
        self.D2y_log_UXU = self.D2y_log * self.UyXUy
        self.D2y_comb    = -self.D2y_log * (self.zi * self.UyXUy + np.eye(self.n))        

        # Preparing other required variables
        self.zi2 = self.zi * self.zi

        self.hess_aux_updated = True

        return

    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated
        if not self.invhess_aux_aux_updated:
            self.update_invhessprod_aux_aux()

        # Precompute and factorize the Schur complement matrix
        #     S = (-1/z Sy + Dy^-1 kron Dy^-1)
        #         - [1/z^2 log^[1](Dy) (Uy'Ux kron Uy'Ux) [(1/z log + inv)^[1](Dx)]^-1 (Ux'Uy kron Ux'Uy) log^[1](Dy)]
        # where
        #     (Sy)_ij,kl = delta_kl (Uy' X Uy)_ij log^[2]_ijl(Dy) + delta_ij (Uy' X Uy)_kl log^[2]_jkl(Dy)        
        # which we will need to solve linear systems with the Hessian of our barrier function

        self.z2           = self.z * self.z
        self.UyUx         = self.Uy.conj().T @ self.Ux
        self.D1x_comb_inv = np.reciprocal(self.D1x_comb)
        
        # Get [-1/z Sy + Dy^-1 kron Dy^-1] matrix
        hess_schur = grad.get_S_matrix(self.D2y_comb, np.sqrt(2.0), iscomplex=self.iscomplex)

        # Get [1/z^2 log^[1](Dy) (Uy'Ux kron Uy'Ux) [(1/z log + inv)^[1](Dx)]^-1 (Ux'Uy kron Ux'Uy) log^[1](Dy)] matrix
        # Begin with log^[1](Dy)
        if self.iscomplex:
            work = self.D1y_log * 1j
            work.view(np.float64).reshape(-1)[self.tril_idxs] *= -1
            work += self.D1y_log

            worku = work.view(np.float64).reshape(-1)[self.triu_idxs] / self.scale
            workl = work.view(np.float64).reshape(-1)[self.tril_idxs] / self.scale

            self.E.view(np.float64).reshape(self.vn, -1)[range(self.vn), self.triu_idxs] = worku
            self.E.view(np.float64).reshape(self.vn, -1)[range(self.vn), self.tril_idxs] = workl
        else:
            work = self.D1y_log.reshape(-1)[self.triu_idxs] / self.scale
            self.E.reshape(self.vn, -1)[range(self.vn), self.triu_idxs] = work
            self.E.reshape(self.vn, -1)[range(self.vn), self.tril_idxs] = work
        # Apply (Ux'Uy kron Ux'Uy)
        lin.congr_multi(self.work8, self.UyUx.conj().T, self.E, work=self.work7)
        # Apply [(1/z log + inv)^[1](Dx)]^-1
        self.work8 *= self.D1x_comb_inv
        # Apply (Uy'Ux kron Uy'Ux)
        lin.congr_multi(self.work6, self.UyUx, self.work8, work=self.work7)
        # Apply (1/z^2 log^[1](Dy))
        self.work6 *= self.D1y_log
        work  = self.work6.view(dtype=np.float64).reshape((self.vn, -1))[:, self.triu_idxs]
        work *= (self.zi2 * self.scale)

        # Subtract to obtain Schur complement then Cholesky factor
        hess_schur -= work
        self.hess_schur_fact = lin.cho_fact(hess_schur)
        self.invhess_aux_updated = True

        return

    def update_invhessprod_aux_aux(self):
        assert not self.invhess_aux_aux_updated

        if self.iscomplex:
            self.tril_idxs = np.empty(self.n*self.n, dtype=int)
            self.tril_idxs = np.empty(self.n*self.n, dtype=int)
            k = 0
            for j in range(self.n):
                for i in range(j):
                    self.tril_idxs[k]     = 2 * (i + j*self.n)
                    self.tril_idxs[k + 1] = 2 * (i + j*self.n) + 1
                    k += 2
                self.tril_idxs[k] = 2 * (j + j*self.n)
                k += 1
        else:
            self.tril_idxs = np.array([i + j*self.n for j in range(self.n) for i in range(j + 1)])

        self.work6  = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work7  = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work8  = np.empty((self.vn, self.n, self.n), dtype=self.dtype)

        self.invhess_aux_aux_updated = True

    def update_dder3_aux(self):
        assert not self.dder3_aux_updated
        assert self.hess_aux_updated

        self.zi3 = self.zi2 * self.zi
        self.D2x_log = grad.D2_log(self.Dx, self.D1x_log)

        self.dder3_aux_updated = True

        return