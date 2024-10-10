# Copyright (c) 2024, Kerry He, James Saunderson, and Hamza Fawzi

# This Python package QICS is licensed under the MIT license; see LICENSE.md 
# file in the root directory or at https://github.com/kerry-he/qics

import numpy as np
import scipy as sp
import qics._utils.linalg as lin
import qics._utils.gradient as grad
from qics.cones.base import Cone, get_central_ray_relentr
from qics.vectorize import get_full_to_compact_op


class QuantRelEntr(Cone):
    r"""A class representing a quantum relative entropy cone

    .. math::

        \mathcal{QRE}_{n} = \text{cl}\{ (t, X, Y) \in \mathbb{R} \times
        \mathbb{H}^n_{++} \times \mathbb{H}^n_{++} : t \geq S(X \| Y) \},

    where

    .. math::

        S(X \| Y) = \text{tr}[X \log(X) - X \log(Y)],

    is the quantum (Umegaki) relative entropy function.

    Parameters
    ----------
    n : :obj:`int`
        Dimension of the matrices :math:`X` and :math:`Y`.
    iscomplex : :obj:`bool`
        Whether the matrices :math:`X` and :math:`Y` are defined over
        :math:`\mathbb{H}^n` (``True``), or restricted to 
        :math:`\mathbb{S}^n` (``False``). The default is ``False``.

    See also
    --------
    ClassRelEntr : Classical relative entropy
    QuantEntr : (Homogenized) quantum entropy cone
    QuantCondEntr : Quantum conditional entropy cone
    QuantKeyDist : Quantum key distribution cone 
    """

    def __init__(self, n, iscomplex=False):
        self.n = n
        self.iscomplex = iscomplex

        self.nu = 1 + 2 * n  # Barrier parameter

        if iscomplex:
            self.vn = n * n
            self.dim = [1, 2 * n * n, 2 * n * n]
            self.type = ["r", "h", "h"]
            self.dtype = np.complex128
        else:
            self.vn = n * (n + 1) // 2
            self.dim = [1, n * n, n * n]
            self.type = ["r", "s", "s"]
            self.dtype = np.float64

        self.idx_X = slice(1, 1 + self.dim[1])
        self.idx_Y = slice(1 + self.dim[1], sum(self.dim))

        # Update flags
        self.feas_updated = False
        self.grad_updated = False
        self.hess_aux_updated = False
        self.invhess_aux_updated = False
        self.invhess_aux_aux_updated = False
        self.dder3_aux_updated = False
        self.congr_aux_updated = False

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

        # Check that X and Y are positive definite
        self.Dx, self.Ux = np.linalg.eigh(self.X)
        self.Dy, self.Uy = np.linalg.eigh(self.Y)

        if any(self.Dx <= 0) or any(self.Dy <= 0):
            self.feas = False
            return self.feas

        # Check that t > S(X||Y)
        self.log_Dx = np.log(self.Dx)
        self.log_Dy = np.log(self.Dy)

        self.log_X = (self.Ux * self.log_Dx) @ self.Ux.conj().T
        self.log_X = (self.log_X + self.log_X.conj().T) * 0.5
        self.log_Y = (self.Uy * self.log_Dy) @ self.Uy.conj().T
        self.log_Y = (self.log_Y + self.log_Y.conj().T) * 0.5

        self.log_XY = self.log_X - self.log_Y
        self.z = self.t[0, 0] - lin.inp(self.X, self.log_XY)

        self.feas = self.z > 0
        return self.feas

    def get_val(self):
        assert self.feas_updated
        return -np.log(self.z) - np.sum(self.log_Dx) - np.sum(self.log_Dy)

    def update_grad(self):
        assert self.feas_updated
        assert not self.grad_updated

        # Compute gradients of quantum relative entropy
        # D_X S(X||Y) = log(X) - log(Y) + I
        self.DPhiX = self.log_XY + np.eye(self.n)
        # D_Y S(X||Y) = -Uy * [log^[1](Dy) .* (Uy' X Uy)] Uy'
        self.D1y_log = grad.D1_log(self.Dy, self.log_Dy)
        self.UyXUy = self.Uy.conj().T @ self.X @ self.Uy
        self.DPhiY = -self.Uy @ (self.D1y_log * self.UyXUy) @ self.Uy.conj().T
        self.DPhiY = (self.DPhiY + self.DPhiY.conj().T) * 0.5

        # Compute X^-1 and Y^-1
        inv_Dx = np.reciprocal(self.Dx)
        inv_X_rt2 = self.Ux * np.sqrt(inv_Dx)
        self.inv_X = inv_X_rt2 @ inv_X_rt2.conj().T

        inv_Dy = np.reciprocal(self.Dy)
        inv_Y_rt2 = self.Uy * np.sqrt(inv_Dy)
        self.inv_Y = inv_Y_rt2 @ inv_Y_rt2.conj().T

        # Compute gradient of barrier function
        self.zi = np.reciprocal(self.z)

        self.grad = [
            -self.zi,
            self.zi * self.DPhiX - self.inv_X,
            self.zi * self.DPhiY - self.inv_Y,
        ]

        self.grad_updated = True

    def hess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        (Ht, Hx, Hy) = H

        UxHxUx = self.Ux.conj().T @ Hx @ self.Ux
        UyHxUy = self.Uy.conj().T @ Hx @ self.Uy
        UyHyUy = self.Uy.conj().T @ Hy @ self.Uy

        # Hessian product of quantum relative entropy
        # D2_XX S(X||Y)[Hx] =  Ux [log^[1](Dx) .* (Ux' Hx Ux)] Ux'
        D2PhiXXH = self.Ux @ (self.D1x_log * UxHxUx) @ self.Ux.conj().T
        # D2_YX S(X||Y)[Hx] = -Uy [log^[1](Dy) .* (Uy' Hx Uy)] Uy'
        D2PhiXYH = -self.Uy @ (self.D1y_log * UyHyUy) @ self.Uy.conj().T
        # D2_XY S(X||Y)[Hy] = -Uy [log^[1](Dy) .* (Uy' Hy Uy)] Uy'
        D2PhiYXH = -self.Uy @ (self.D1y_log * UyHxUy) @ self.Uy.conj().T
        # D2_YY S(X||Y)[Hy] = -Uy [Σ_k log_k^[2](Dy) .* ... ] Uy'
        D2PhiYYH = -grad.scnd_frechet(self.D2y_log_UXU, UyHyUy, U=self.Uy)

        # ======================================================================
        # Hessian products with respect to t
        # ======================================================================
        # D2_t F(t, X, Y)[Ht, Hx, Hy] 
        #         = (Ht - D_X S(X||Y)[Hx] - D_Y S(X||Y)[Hy]) / z^2
        out_t = (Ht - lin.inp(self.DPhiX, Hx) - lin.inp(self.DPhiY, Hy))
        out_t *= self.zi2
        out[0][:] = out_t

        # ======================================================================
        # Hessian products with respect to X
        # ======================================================================
        # D2_X F(t, X, Y)[Ht, Hx, Hy] 
        #         = -D2_t F(t, X, Y)[Ht, Hx, Hy] * D_X S(X||Y)
        #           + (D2_XX S(X||Y)[Hx] + D2_XY S(X||Y)[Hy]) / z
        #           + X^-1 Hx X^-1
        out_X = -out_t * self.DPhiX
        out_X += self.zi * (D2PhiXYH + D2PhiXXH)
        out_X += self.inv_X @ Hx @ self.inv_X
        out[1][:] = (out_X + out_X.conj().T) * 0.5

        # ======================================================================
        # Hessian products with respect to Y
        # ======================================================================
        # D2_Y F(t, X, Y)[Ht, Hx, Hy] 
        #         = -D2_t F(t, X, Y)[Ht, Hx, Hy] * D_Y S(X||Y)
        #           + (D2_YX S(X||Y)[Hx] + D2_YY S(X||Y)[Hy]) / z
        #           + Y^-1 Hy Y^-1
        out_Y = -out_t * self.DPhiY
        out_Y += self.zi * (D2PhiYXH + D2PhiYYH)
        out_Y += self.inv_Y @ Hy @ self.inv_Y
        out[2][:] = (out_Y + out_Y.conj().T) * 0.5

        return out
         
    def hess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.congr_aux_updated:
            self.update_congr_aux(A)

        p = A.shape[0]
        lhs = np.zeros((p, sum(self.dim)))

        work0, work1, work2 = self.work0, self.work1, self.work2
        work3, work4, work5 = self.work3, self.work4, self.work5

        # ======================================================================
        # Hessian products with respect to t
        # ======================================================================
        # D2_t F(t, X, Y)[Ht, Hx, Hy] 
        #         = (Ht - D_X S(X||Y)[Hx] - D_Y S(X||Y)[Hy]) / z^2
        DPhiX_vec = self.DPhiX.view(np.float64).reshape((-1, 1))
        DPhiY_vec = self.DPhiY.view(np.float64).reshape((-1, 1))
        out_t = self.At - (self.Ax_vec @ DPhiX_vec).ravel()
        out_t -= (self.Ay_vec @ DPhiY_vec).ravel()
        out_t *= self.zi2

        lhs[:, 0] = out_t

        # ======================================================================
        # Hessian products with respect to Y
        # ======================================================================
        # Hessian products of quantum relative entropy
        # D2_YX S(X||Y)[Hx] = -Uy [log^[1](Dy) .* (Uy' Hx Uy)] Uy'
        lin.congr_multi(work1, self.Uy.conj().T, self.Ax, work2)
        work1 *= -self.zi * self.D1y_log
        lin.congr_multi(work0, self.Uy, work1, work2)
        # D2_YY S(X||Y)[Hy] = -Uy [Σ_k log_k^[2](Dy) .* ... ] Uy'
        lin.congr_multi(work1, self.Uy.conj().T, self.Ay, work2)
        grad.scnd_frechet_multi(work4, self.D2y_comb, work1, U=self.Uy, 
                                work1=work2, work2=work3, work3=work5)

        # Hessian product of barrier function
        # D2_Y F(t, X, Y)[Ht, Hx, Hy] 
        #         = -D2_t F(t, X, Y)[Ht, Hx, Hy] * D_Y S(X||Y)
        #           + (D2_YX S(X||Y)[Hx] + D2_YY S(X||Y)[Hy]) / z
        #           + Y^-1 Hy Y^-1
        work0 += work4
        np.outer(out_t, self.DPhiY, out=work2.reshape((p, -1)))
        work0 -= work2

        lhs[:, self.idx_Y] = work0.reshape((p, -1)).view(np.float64)

        # ======================================================================
        # Hessian products with respect to X
        # ======================================================================
        # Hessian products of quantum relative entropy
        # D2_XY S(X||Y)[Hy] = -Uy [log^[1](Dy) .* (Uy' Hy Uy)] Uy'
        work1 *= self.D1y_log * self.zi
        lin.congr_multi(work0, self.Uy, work1, work2)
        # D2_XX S(X||Y)[Hx] =  Ux [log^[1](Dx) .* (Ux' Hx Ux)] Ux'
        lin.congr_multi(work1, self.Ux.conj().T, self.Ax, work2)
        work1 *= self.D1x_comb
        lin.congr_multi(work3, self.Ux, work1, work2)

        # Hessian product of barrier function
        # D2_X F(t, X, Y)[Ht, Hx, Hy] 
        #         = -D2_t F(t, X, Y)[Ht, Hx, Hy] * D_X S(X||Y)
        #           + (D2_XX S(X||Y)[Hx] + D2_XY S(X||Y)[Hy]) / z
        #           + X^-1 Hx X^-1
        work3 -= work0
        np.outer(out_t, self.DPhiX, out=work2.reshape((p, -1)))
        work3 -= work2

        lhs[:, self.idx_X] = work3.reshape((p, -1)).view(np.float64)

        # Multiply A (H A')
        return lin.dense_dot_x(lhs, A.T)

    def invhess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        # See comments in invhess_congr for further details about how this
        # inverse Hessian product is performed

        (Ht, Hx, Hy) = H

        Wx = Hx + Ht * self.DPhiX
        Wy = Hy + Ht * self.DPhiY

        # ======================================================================
        # Inverse Hessian products with respect to Y
        # ======================================================================
        # Compute RHS of S \ ( ... ) expression
        # Compute Ux' Wx Ux
        temp = self.Ux.conj().T @ Wx @ self.Ux
         # Apply (Uy'Ux ⊗ Uy'Ux) (1/z log^[1](Dx) + Dx^-1 ⊗ Dx^-1)^-1
        temp = self.UyUx @ (self.D1x_comb_inv * temp) @ self.UyUx.conj().T
        # Apply -1/z log^[1](Dy)
        temp = -self.zi * self.D1y_log * temp
        # Compute Uy' Wy Uy and subtract previous expression
        temp = self.Uy.conj().T @ Wy @ self.Uy - temp
        
        # Solve the linear system S \ ( ... ) to obtain Uy' Y Uy
        # Convert matrices to compact real vectors
        temp_vec = temp.view(np.float64).reshape((-1, 1))
        temp_vec = self.F2C_op @ temp_vec
        # Solve system
        temp_vec = lin.cho_solve(self.hess_schur_fact, temp_vec)
        # Expand compact real vectors back into full matrices
        temp_vec = self.F2C_op.T @ temp_vec
        temp = temp_vec.T.view(self.dtype).reshape((self.n, self.n))

        # Recover Y from Uy' Y Uy
        out_Y = self.Uy @ temp @ self.Uy.conj().T
        out[2][:] = (out_Y + out_Y.conj().T) * 0.5

        # ======================================================================
        # Inverse Hessian products with respect to X
        # ======================================================================
        # Apply -(Uy ⊗ Uy) (1/z log^[1](Dy))
        temp = -self.Uy @ (self.zi * self.D1y_log * temp) @ self.Uy.conj().T
        # Subtract Wx from previous expression
        temp = Wx - temp
        # Apply (Ux ⊗ Ux)
        temp = self.Ux.conj().T @ temp @ self.Ux
        # Apply (Ux' ⊗ Ux') (1/z log^[1](Dx) + Dx^-1 ⊗ Dx^-1)^-1 to obtain X
        out_X = self.Ux @ (self.D1x_comb_inv * temp) @ self.Ux.conj().T
        out[1][:] = (out_X + out_X.conj().T) * 0.5

        # ======================================================================
        # Inverse Hessian products with respect to t
        # ======================================================================
        # t = z^2 Ht + <DPhi(X, Y), (X, Y)>
        out_t = self.z2 * Ht
        out_t += lin.inp(self.DPhiX, out_X)
        out_t += lin.inp(self.DPhiY, out_Y)
        out[0][:] = out_t

        return out

    def invhess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()
        if not self.congr_aux_updated:
            self.update_congr_aux(A)

        # The inverse Hessian product applied on (Ht, Hx, Hy) for the QRE
        # barrier is
        #     (X, Y) =  M \ (Wx, Wy)
        #         t  =  z^2 Ht + <DPhi(X, Y), (X, Y)>
        # where (Wx, Wy) = [(Hx, Hy) + Ht DPhi(X, Y)],
        #     M = Vxy [ 1/z log^[1](Dx) + Dx^-1 ⊗ Dx^-1  -1/z (Ux'Uy ⊗ Ux'Uy) log^[1](Dy) ]
        #             [-1/z log^[1](Dy) (Uy'Ux ⊗ Uy'Ux)      -1/z Sy + Dy^-1 ⊗ Dy^-1      ] Vxy'
        # and
        #     Vxy = [ Ux ⊗ Ux             ]
        #           [             Uy ⊗ Uy ]
        #
        # To solve linear systems with M, we simplify it by doing block
        # elimination, in which case we get
        #     Uy' Y Uy = S \ ({Uy' Wy Uy} - [1/z log^[1](Dy) (Uy'Ux ⊗ Uy'Ux) (1/z log^[1](Dx) + Dx^-1 ⊗ Dx^-1)^-1 {Ux' Wx Ux}])
        #     Ux' X Ux = -(1/z log^[1](Dx) + Dx^-1 ⊗ Dx^-1)^-1 [{Ux' Wx Ux} + 1/z (Ux'Uy ⊗ Ux'Uy) log^[1](Dy) Y]
        # where S is the Schur complement matrix of M.

        p = self.Ax.shape[0]
        lhs = np.empty((p, sum(self.dim)))

        work0, work1 = self.work0, self.work1
        work2, work3, work4 = self.work2, self.work3, self.work4

        # ======================================================================
        # Inverse Hessian products with respect to Y
        # ======================================================================
        # Compute RHS of S \ ( ... ) expression
        # Compute Ux' Wx Ux
        np.outer(self.At, self.DPhiX, out=work2.reshape((p, -1)))
        np.add(self.Ax, work2, out=work0)
        lin.congr_multi(work2, self.Ux.conj().T, work0, work3)
        # Apply (1/z log^[1](Dx) + Dx^-1 ⊗ Dx^-1)^-1
        work2 *= self.D1x_comb_inv
        # Apply (Uy'Ux ⊗ Uy'Ux)
        lin.congr_multi(work1, self.UyUx, work2, work3)
        # Apply -1/z log^[1](Dy)
        work1 *= -self.zi * self.D1y_log
        # Compute Uy' Wy Uy and subtract previous expression
        np.outer(self.At, self.DPhiY, out=work2.reshape((p, -1)))
        np.add(self.Ay, work2, out=work3)
        lin.congr_multi(work2, self.Uy.conj().T, work3, work4)
        work2 -= work1

        # Solve the linear system S \ ( ... ) to obtain Uy' Y Uy
        # Convert matrices to compact real vectors
        work = work2.view(np.float64).reshape((p, -1)).T
        work = lin.x_dot_dense(self.F2C_op, work)
        # Solve system
        work = lin.cho_solve(self.hess_schur_fact, work)
        # Expand compact real vectors back into full matrices
        work = lin.x_dot_dense(self.F2C_op.T, work)
        work1.view(np.float64).reshape((p, -1))[:] = work.T

        # Recover Y from Uy' Y Uy
        lin.congr_multi(work4, self.Uy, work1, work2)
        out_Y = work4.reshape((p, -1)).view(np.float64)
        lhs[:, self.idx_Y] = out_Y

        # ======================================================================
        # Inverse Hessian products with respect to X
        # ======================================================================
        # Apply -1/z log^[1](Dy)
        work1 *= -self.zi * self.D1y_log
        # Apply (Uy ⊗ Uy)
        lin.congr_multi(work2, self.Uy, work1, work3)
        # Subtract Wx from previous expression
        work0 -= work2
        # Apply (Ux ⊗ Ux)
        lin.congr_multi(work1, self.Ux.conj().T, work0, work3)
        # Apply (1/z log^[1](Dx) + Dx^-1 ⊗ Dx^-1)^-1 to obtian Ux' X Ux
        work1 *= self.D1x_comb_inv

        # Recover X from Ux' X Ux
        lin.congr_multi(work2, self.Ux, work1, work3)
        out_X = work2.reshape((p, -1)).view(np.float64)
        lhs[:, self.idx_X] = out_X

        # ======================================================================
        # Inverse Hessian products with respect to t
        # ======================================================================
        DPhiX_vec = self.DPhiX.view(np.float64).reshape((-1, 1))
        DPhiY_vec = self.DPhiY.view(np.float64).reshape((-1, 1))

        # t = z^2 Ht + <DS(X||Y), (X, Y)>
        outt = self.z2 * self.At
        outt += (out_X @ DPhiX_vec).ravel()
        outt += (out_Y @ DPhiY_vec).ravel()
        lhs[:, 0] = outt

        # Multiply A (H A')
        return lin.dense_dot_x(lhs, A.T)

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
        D3_log_Y = 2 * np.power(self.Dy, -3)

        # Quantum relative entropy Hessians
        D2PhiXXH = self.Ux @ (self.D1x_log * UxHxUx) @ self.Ux.conj().T
        D2PhiXYH = -self.Uy @ (self.D1y_log * UyHyUy) @ self.Uy.conj().T
        D2PhiYXH = -self.Uy @ (self.D1y_log * UyHxUy) @ self.Uy.conj().T
        D2PhiYYH = -grad.scnd_frechet(self.D2y_log_UXU, UyHyUy, U=self.Uy)

        D2PhiXHH = lin.inp(Hx, D2PhiXXH + D2PhiXYH)
        D2PhiYHH = lin.inp(Hy, D2PhiYXH + D2PhiYYH)

        # Quantum relative entropy third order derivatives
        D3PhiXXX = grad.scnd_frechet(self.D2x_log, UxHxUx, UxHxUx, self.Ux)
        D3PhiXYY = -grad.scnd_frechet(self.D2y_log, UyHyUy, UyHyUy, self.Uy)

        D3PhiYYX = -grad.scnd_frechet(self.D2y_log, UyHyUy, UyHxUy, self.Uy)
        D3PhiYXY = D3PhiYYX
        D3PhiYYY = -grad.thrd_frechet(self.Dy, self.D2y_log, D3_log_Y, self.Uy,
                                      self.UyXUy, UyHyUy)

        # Third derivatives of barrier
        dder3_t = -2 * self.zi3 * chi2 - self.zi2 * (D2PhiXHH + D2PhiYHH)

        dder3_X = -dder3_t * self.DPhiX
        dder3_X -= 2 * self.zi2 * chi * (D2PhiXXH + D2PhiXYH)
        dder3_X += self.zi * (D3PhiXXX + D3PhiXYY)
        dder3_X -= 2 * self.inv_X @ Hx @ self.inv_X @ Hx @ self.inv_X
        dder3_X = (dder3_X + dder3_X.conj().T) * 0.5

        dder3_Y = -dder3_t * self.DPhiY
        dder3_Y -= 2 * self.zi2 * chi * (D2PhiYXH + D2PhiYYH)
        dder3_Y += self.zi * (D3PhiYYX + D3PhiYXY + D3PhiYYY)
        dder3_Y -= 2 * self.inv_Y @ Hy @ self.inv_Y @ Hy @ self.inv_Y
        dder3_Y = (dder3_Y + dder3_Y.conj().T) * 0.5

        out[0][:] += dder3_t * a
        out[1][:] += dder3_X * a
        out[2][:] += dder3_Y * a

        return out

    # ==========================================================================
    # Auxilliary functions
    # ==========================================================================
    def update_congr_aux(self, A):
        assert not self.congr_aux_updated

        from qics.vectorize import vec_to_mat
        iscomplex = self.iscomplex

        # Get slices and views of A matrix to be used in congruence computations
        if sp.sparse.issparse(A):
            A = A.tocsr()
        self.Ax_vec = A[:, self.idx_X]
        self.Ay_vec = A[:, self.idx_Y]

        if sp.sparse.issparse(A):
            A = A.toarray()
        Ax_dense = np.ascontiguousarray(A[:, self.idx_X])
        Ay_dense = np.ascontiguousarray(A[:, self.idx_Y])
        self.At = A[:, 0]
        self.Ax = np.array([vec_to_mat(Ax_k, iscomplex) for Ax_k in Ax_dense])
        self.Ay = np.array([vec_to_mat(Ay_k, iscomplex) for Ay_k in Ay_dense])

        # Preallocate matrices we will need when performing these congruences
        self.work0 = np.empty_like(self.Ax)
        self.work1 = np.empty_like(self.Ax)
        self.work2 = np.empty_like(self.Ax)
        self.work3 = np.empty_like(self.Ax)
        self.work4 = np.empty_like(self.Ax)
        self.work5 = np.empty((self.Ax.shape[::-1]), dtype=self.dtype)

        self.congr_aux_updated = True

    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        D1x_inv = np.reciprocal(np.outer(self.Dx, self.Dx))
        self.D1x_log = grad.D1_log(self.Dx, self.log_Dx)
        self.D1x_comb = self.zi * self.D1x_log + D1x_inv

        self.D2y_log = grad.D2_log(self.Dy, self.D1y_log)
        self.D2y_log_UXU = self.D2y_log * self.UyXUy
        self.D2y_comb = -self.D2y_log * (self.zi * self.UyXUy + np.eye(self.n))

        # Preparing other required variables
        self.zi2 = self.zi * self.zi

        self.hess_aux_updated = True

    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated
        if not self.invhess_aux_aux_updated:
            self.update_invhessprod_aux_aux()

        # Precompute and factorize the Schur complement matrix
        #     S = (-1/z Sy + Dy^-1 ⊗ Dy^-1)
        #         - [1/z^2 log^[1](Dy) (Uy'Ux ⊗ Uy'Ux) [(1/z log + inv)^[1](Dx)]^-1 (Ux'Uy ⊗ Ux'Uy) log^[1](Dy)]
        # where
        #     (Sy)_ij,kl = delta_kl (Uy' X Uy)_ij log^[2]_ijl(Dy) + delta_ij (Uy' X Uy)_kl log^[2]_jkl(Dy)
        # which we will need to solve linear systems with the Hessian of our
        # barrier function

        work6, work7, work8 = self.work6, self.work7, self.work8

        self.z2 = self.z * self.z
        self.UyUx = self.Uy.conj().T @ self.Ux
        self.D1x_comb_inv = np.reciprocal(self.D1x_comb)

        # ======================================================================
        # Get first term in S matrix, i.e., [-1/z Sy + Dy^-1 ⊗ Dy^-1]
        # ======================================================================
        rt2 = np.sqrt(2.0)
        hess_schur = grad.get_S_matrix(self.D2y_comb, rt2, self.iscomplex)

        # ======================================================================
        # Get second term in S matrix, i.e., [1/z^2 log^[1](Dy) ... ]
        # ======================================================================
        # Apply log^[1](Dy) to computational basis
        work6[:] = self.E
        work6[self.Ek, self.Ei, self.Ej] *= self.D1y_log[self.Ei, self.Ej]
        # Apply (Ux'Uy ⊗ Ux'Uy)
        lin.congr_multi(work8, self.UyUx.conj().T, work6, work=work7)
        # Apply [(1/z log + inv)^[1](Dx)]^-1
        work8 *= self.D1x_comb_inv
        # Apply (Uy'Ux ⊗ Uy'Ux)
        lin.congr_multi(work6, self.UyUx, work8, work=work7)
        # Apply (1/z^2 log^[1](Dy)) and reshape into square symmetric matrix
        work6 *= self.D1y_log
        work = work6.view(np.float64).reshape((self.vn, -1))
        work = lin.x_dot_dense(self.F2C_op, work.T)
        work *= self.zi2

        # Subtract the two terms to obtain Schur complement then Cholesky factor
        hess_schur -= work
        self.hess_schur_fact = lin.cho_fact(hess_schur)
        
        self.invhess_aux_updated = True

    def update_invhessprod_aux_aux(self):
        # This auxiliary function should only ever be called once
        assert not self.invhess_aux_aux_updated

        self.precompute_computational_basis()
        self.Ek, self.Ei, self.Ej = np.where(self.E)

        self.F2C_op = get_full_to_compact_op(self.n, self.iscomplex)
        
        self.work6 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work7 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work8 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)

        self.invhess_aux_aux_updated = True

    def update_dder3_aux(self):
        assert not self.dder3_aux_updated
        assert self.hess_aux_updated

        self.zi3 = self.zi2 * self.zi
        self.D2x_log = grad.D2_log(self.Dx, self.D1x_log)

        self.dder3_aux_updated = True
