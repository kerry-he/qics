# Copyright (c) 2024, Kerry He, James Saunderson, and Hamza Fawzi

# This Python package QICS is licensed under the MIT license; see LICENSE.md
# file in the root directory or at https://github.com/kerry-he/qics

import numpy as np
import scipy as sp

from qics._utils.gradient import (
    D1_f,
    D2_f,
    scnd_frechet,
    scnd_frechet_multi,
    thrd_frechet,
)
from qics._utils.linalg import (
    cho_fact,
    cho_solve,
    congr_multi,
    dense_dot_x,
    inp,
    x_dot_dense,
)
from qics.cones.base import Cone
from qics.vectorize import get_full_to_compact_op, vec_to_mat


class RenyiEntr(Cone):
    r"""A class representing the epigraph of the (homogenized) Renyi entropy,
    i.e., for some :math:`\alpha\in[0, 1)`,

    .. math::

        \mathcal{RE}_{n} = \text{cl}\{ (t,X,Y) \in \mathbb{R} \times \mathbb{H}^n_{++}
        \times \mathbb{H}^n_{++} : t \geq u D_\alpha(u^{-1}X \| u^{-1}Y) \},

    where

    .. math::

        D_\alpha(X\|Y)=\frac{1}{\alpha-1}\log(\text{tr}[X^\alpha Y^{1-\alpha}]),

    is the :math:`\alpha`-Renyi entropy.

    Parameters
    ----------
    n : :obj:`int`
        Dimension of the matrices :math:`X` and :math:`Y`.
    alpha : :obj:`float`
        The exponent :math:`\alpha` used to parameterize the Renyi entropy.
    iscomplex : :obj:`bool`
        Whether the matrices :math:`X` and :math:`Y` are defined over
        :math:`\mathbb{H}^n` (``True``), or restricted to
        :math:`\mathbb{S}^n` (``False``). The default is ``False``.

    See also
    --------
    SandRenyiEntr : SandwichedRenyi entropy
    TrRenyiEntr : Trace function used to define the Renyi entropy
    QuantRelEntr : Quantum relative entropy

    """

    def __init__(self, n, alpha, iscomplex=False):
        assert 0 <= alpha and alpha < 1

        self.n = n
        self.alpha = alpha
        self.iscomplex = iscomplex

        self.nu = 2 + 2 * n  # Barrier parameter

        if iscomplex:
            self.vn = n * n
            self.dim = [1, 1, 2 * n * n, 2 * n * n]
            self.type = ["r", "r", "h", "h"]
            self.dtype = np.complex128
        else:
            self.vn = n * (n + 1) // 2
            self.dim = [1, 1, n * n, n * n]
            self.type = ["r", "r", "s", "s"]
            self.dtype = np.float64

        self.idx_X = slice(2, 2 + self.dim[2])
        self.idx_Y = slice(2 + self.dim[2], sum(self.dim))

        # Get function handles for g(x)=x^α
        # and their first, second and third derivatives
        a = alpha
        self.g = lambda x: np.power(x, a)
        self.dg = lambda x: np.power(x, a - 1) * a
        self.d2g = lambda x: np.power(x, a - 2) * (a * (a - 1))
        self.d3g = lambda x: np.power(x, a - 3) * (a * (a - 1) * (a - 2))

        # Get function handles for h(x)=x^β where β=1-α
        # and their first, second and third derivatives
        b = 1 - alpha
        self.h = lambda x: np.power(x, b)
        self.dh = lambda x: np.power(x, b - 1) * b
        self.d2h = lambda x: np.power(x, b - 2) * (b * (b - 1))
        self.d3h = lambda x: np.power(x, b - 3) * (b * (b - 1) * (b - 2))

        # Get sparse operator to convert from full to compact vectorizations
        self.F2C_op = get_full_to_compact_op(n, iscomplex)

        # Update flags
        self.feas_updated = False
        self.grad_updated = False
        self.hess_aux_updated = False
        self.invhess_aux_updated = False
        self.invhess_aux_aux_updated = False
        self.congr_aux_updated = False

        return

    def get_iscomplex(self):
        return self.iscomplex

    def get_init_point(self, out):
        (t0, u0, x0, y0) = self.get_central_ray()

        point = [
            np.array([[t0]]),
            np.array([[u0]]),
            np.eye(self.n, dtype=self.dtype) * x0,
            np.eye(self.n, dtype=self.dtype) * y0,
        ]

        self.set_point(point, point)

        out[0][:] = point[0]
        out[1][:] = point[1]
        out[2][:] = point[2]
        out[3][:] = point[3]

        return out

    def get_feas(self):
        if self.feas_updated:
            return self.feas

        self.feas_updated = True

        (self.t, self.u, self.X, self.Y) = self.primal

        # Check that u, X, and Y are positive
        self.Dx, self.Ux = np.linalg.eigh(self.X)
        self.Dy, self.Uy = np.linalg.eigh(self.Y)

        if self.u <= 0 or any(self.Dx <= 0) or any(self.Dy <= 0):
            self.feas = False
            return self.feas

        # Construct X^α and Y^β
        Dgx = self.g(self.Dx)
        gX = (self.Ux * Dgx) @ self.Ux.conj().T
        self.gX = (gX + gX.conj().T) / 2

        Dhy = self.h(self.Dy)
        hY = (self.Uy * Dhy) @ self.Uy.conj().T
        self.hY = (hY + hY.conj().T) / 2

        # Check that t > log( tr[ X^α Y^β ] ) / (α - 1)
        self.Tr = np.sum(self.gX * self.hY.conj()).real
        self.z = (self.t - self.u * np.log(self.Tr / self.u) / (self.alpha - 1))[0, 0]

        self.feas = self.z > 0
        return self.feas

    def get_val(self):
        assert self.feas_updated
        func = -np.log(self.z)
        func -= np.sum(np.log(self.Dx)) + np.sum(np.log(self.Dy)) + np.log(self.u[0, 0])
        return func

    def update_grad(self):
        assert self.feas_updated
        assert not self.grad_updated

        # Precompute useful expressions
        self.D1x_g = D1_f(self.Dx, self.g(self.Dx), self.dg(self.Dx))
        self.D1y_h = D1_f(self.Dy, self.h(self.Dy), self.dh(self.Dy))

        self.Ux_hY_Ux = self.Ux.conj().T @ self.hY @ self.Ux
        self.Uy_gX_Uy = self.Uy.conj().T @ self.gX @ self.Uy

        # Compute gradients of trace function
        # D_X Ψ(X, Y) = Dg(X)[h(Y)]
        self.DTrX = self.Ux @ (self.D1x_g * self.Ux_hY_Ux) @ self.Ux.conj().T
        self.DTrX = (self.DTrX + self.DTrX.conj().T) * 0.5
        # D_Y Ψ(X, Y) = Dh(Y)[g(X)]
        self.DTrY = self.Uy @ (self.D1y_h * self.Uy_gX_Uy) @ self.Uy.conj().T
        self.DTrY = (self.DTrY + self.DTrY.conj().T) * 0.5

        # Compute gradients of Renyi entropy
        # D_u S(u, X, Y) = (log(Ψ / u) - 1) / (α - 1)
        self.DPhiu = (np.log(self.Tr / self.u) - 1) / (self.alpha - 1)
        # D_X S(u, X, Y) = (D_X Ψ(X, Y) / Ψ) / (α - 1)
        self.DPhiX = (self.u * self.DTrX / self.Tr) / (self.alpha - 1)
        # D_Y S(u, X, Y) = (D_Y Ψ(X, Y) / Ψ) / (α - 1)
        self.DPhiY = (self.u * self.DTrY / self.Tr) / (self.alpha - 1)

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
            self.zi * self.DPhiu - 1 / self.u,
            self.zi * self.DPhiX - self.inv_X,
            self.zi * self.DPhiY - self.inv_Y,
        ]

        self.grad_updated = True

    def hess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        (Ht, Hu, Hx, Hy) = H

        UHxU = self.Ux.conj().T @ Hx @ self.Ux
        UHyU = self.Uy.conj().T @ Hy @ self.Uy

        # Hessian product of trace function
        # D2_XX Ψ(X, Y)[Hx] = D2g(X)[h(Y), Hx]
        D2TrXXH = scnd_frechet(self.D2x_g, self.Ux_hY_Ux, UHxU, U=self.Ux)
        # D2_XY Ψ(X, Y)[Hy] = Dg(X)[Dh(Y)[Hy]]
        work = self.UxUy @ (self.D1y_h * UHyU) @ self.UxUy.conj().T
        D2TrXYH = self.Ux @ (self.D1x_g * work) @ self.Ux.conj().T
        # D2_YX Ψ(X, Y)[Hx] = Dh(Y)[Dg(X)[Hx]]
        work = self.UxUy.conj().T @ (self.D1x_g * UHxU) @ self.UxUy
        D2TrYXH = self.Uy @ (self.D1y_h * work) @ self.Uy.conj().T
        # D2_YY Ψ(X, Y)[Hy] = D2h(Y)[g(X), Hy]
        D2TrYYH = scnd_frechet(self.D2y_h, self.Uy_gX_Uy, UHyU, U=self.Uy)

        D2TrXH = D2TrXXH + D2TrXYH
        D2TrYH = D2TrYXH + D2TrYYH

        # Hessian product of Renyi entropy
        rho = Hu - ((inp(self.DTrX, Hx) + inp(self.DTrY, Hy)) * self.u) / self.Tr
        # D2_u S(X, Y)[(Hu, Hx, Hy)] = (DΨ(X,Y)[(Hx,Hy)]/Ψ - Hu/u) / (α-1)
        D2PhiuH = -rho / self.u / (self.alpha - 1)
        # D2_X S(X, Y)[(Hu, Hx, Hy)]
        #   = ((Hu/Ψ - u/Ψ^2 DΨ(X,Y)[(Hx,Hy)]) D_X Ψ + u/Ψ D2_X Ψ(X,Y)[(Hx,Hy)]) / (α-1)
        D2PhiXH = (self.DTrX * rho + self.u * D2TrXH) / self.Tr / (self.alpha - 1)
        # D2_Y S(X, Y)[(Hu, Hx, Hy)]
        #   = ((Hu/Ψ - u/Ψ^2 DΨ(X,Y)[(Hx,Hy)]) D_Y Ψ + u/Ψ D2_Y Ψ(X,Y)[(Hx,Hy)]) / (α-1)
        D2PhiYH = (self.DTrY * rho + self.u * D2TrYH) / self.Tr / (self.alpha - 1)

        # ======================================================================
        # Hessian products with respect to t
        # ======================================================================
        # D2_t F(t, u, X, Y)[Ht, Hu, Hx, Hy]
        #    = (Ht - D_u S(u, X, Y)[Hu] - D_X S(u, X, Y)[Hx] - D_Y S(u, X, Y)[Hy]) / z^2
        out_t = Ht - self.DPhiu * Hu - inp(self.DPhiX, Hx) - inp(self.DPhiY, Hy)
        out_t *= self.zi2
        out[0][:] = out_t

        # ======================================================================
        # Hessian products with respect to u
        # ======================================================================
        # D2_u F(t, u, X, Y)[Ht, Hu, Hx, Hy]
        #    = -D2_t F(t, u, X, Y)[Ht, Hu, Hx, Hy] * D_u S(u, X, Y)
        #      + D2_u Ψ(X, Y)[(Hu, Hx, Hy)] / z + Hu / u^2
        out_u = -out_t * self.DPhiu + self.zi * D2PhiuH + Hu / self.u / self.u
        out[1][:] = out_u

        # ======================================================================
        # Hessian products with respect to X
        # ======================================================================
        # D2_X F(t, u, X, Y)[Ht, Hu, Hx, Hy]
        #    = -D2_t F(t, u, X, Y)[Ht, Hu, Hx, Hy] * D_X S(u, X, Y)
        #      + D2_X Ψ(X, Y)[(Hu, Hx, Hy)] / z + X^-1 Hx X^-1
        out_X = -out_t * self.DPhiX + self.zi * D2PhiXH + self.inv_X @ Hx @ self.inv_X
        out_X = (out_X + out_X.conj().T) * 0.5
        out[2][:] = out_X

        # ==================================================================
        # Hessian products with respect to Y
        # ==================================================================
        # D2_Y F(t, u, X, Y)[Ht, Hu, Hx, Hy]
        #    = -D2_t F(t, u, X, Y)[Ht, Hu, Hx, Hy] * D_Y S(u, X, Y)
        #      + D2_Y Ψ(X, Y)[(Hu, Hx, Hy)] / z + Y^-1 Hy Y^-1
        out_Y = -out_t * self.DPhiY + self.zi * D2PhiYH + self.inv_Y @ Hy @ self.inv_Y
        out_Y = (out_Y + out_Y.conj().T) * 0.5
        out[3][:] = out_Y

        return out

    def hess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)

        p = A.shape[0]
        lhs = np.empty((p, sum(self.dim)))

        work0, work1 = self.work0, self.work1
        work2, work3 = self.work2, self.work3
        work4, work5, work6 = self.work4, self.work5, self.work6

        DTrX_vec = self.DTrX.view(np.float64).reshape((-1, 1))
        DTrY_vec = self.DTrY.view(np.float64).reshape((-1, 1))

        # ======================================================================
        # Hessian products with respect to t
        # ======================================================================
        # D2_t F(t, u, X, Y)[Ht, Hu, Hx, Hy]
        #    = (Ht - D_u S(u, X, Y)[Hu] - D_X S(u, X, Y)[Hx] - D_Y S(u, X, Y)[Hy]) / z^2
        DPhiX_vec = self.DPhiX.view(np.float64).reshape((-1, 1))
        DPhiY_vec = self.DPhiY.view(np.float64).reshape((-1, 1))

        out_t = self.At - (self.Au * self.DPhiu).ravel()
        out_t -= (self.Ax_vec @ DPhiX_vec + self.Ay_vec @ DPhiY_vec).ravel()
        out_t *= self.zi2

        lhs[:, 0] = out_t

        # ======================================================================
        # Hessian products with respect to u
        # ======================================================================
        # Hessian product of Renyi entropy
        # D2_u S(X, Y)[(Hu, Hx, Hy)] = (DΨ(X,Y)[(Hx,Hy)]/Ψ - Hu/u) / (α-1)
        rho = self.Ax_vec @ DTrX_vec + self.Ay_vec @ DTrY_vec
        rho *= -self.u / self.Tr
        rho += self.Au
        D2PhiuH = -rho / (self.u[0, 0] * (self.alpha - 1))

        # Hessian product of barrier function
        # D2_u F(t, u, X, Y)[Ht, Hu, Hx, Hy]
        #    = -D2_t F(t, u, X, Y)[Ht, Hu, Hx, Hy] * D_u S(u, X, Y)
        #      + D2_u Ψ(X, Y)[(Hu, Hx, Hy)] / z + Hu / u^2
        out_u = -out_t * self.DPhiu[0, 0]
        out_u += self.zi * D2PhiuH.ravel()
        out_u += (self.Au / (self.u * self.u)).ravel()

        lhs[:, 1] = out_u

        # ==================================================================
        # Hessian products with respect to X
        # ==================================================================
        # Hessian products of trace function
        # D2_XX Ψ(X, Y)[Hx] = D2g(X)[h(Y), Hx]
        congr_multi(work2, self.Ux.conj().T, self.Ax, work=work4)
        scnd_frechet_multi(work5, self.D2x_g, work2, self.Ux_hY_Ux, U=self.Ux,
                           work1=work3, work2=work4, work3=work6)  # fmt: skip

        # D2_XY Ψ(X, Y)[Hy] = Dg(X)[Dh(Y)[Hy]]
        congr_multi(work1, self.Uy.conj().T, self.Ay, work=work4)
        np.multiply(work1, self.D1y_h, out=work0)
        congr_multi(work3, self.UxUy, work0, work=work4)
        work3 *= self.D1x_g
        congr_multi(work0, self.Ux, work3, work=work4)
        work5 += work0

        # Hessian products of Renyi entropy
        # D2_X S(X, Y)[(Hu, Hx, Hy)]
        #   = ((Hu/Ψ - u/Ψ^2 DΨ(X,Y)[(Hx,Hy)]) D_X Ψ + u/Ψ D2_X Ψ(X,Y)[(Hx,Hy)]) / (α-1)
        work5 *= self.u
        np.outer(rho, self.DTrX, out=work0.reshape((p, -1)))
        work5 += work0
        work5 /= self.Tr * (self.alpha - 1)

        # Hessian product of barrier function
        # D2_X F(t, u, X, Y)[Ht, Hu, Hx, Hy]
        #    = -D2_t F(t, u, X, Y)[Ht, Hu, Hx, Hy] * D_X S(u, X, Y)
        #      + D2_X Ψ(X, Y)[(Hu, Hx, Hy)] / z + X^-1 Hx X^-1
        work5 *= self.zi
        np.outer(out_t, self.DPhiX, out=work3.reshape((p, -1)))
        work5 -= work3
        congr_multi(work3, self.inv_X, self.Ax, work=work3)
        work5 += work3

        lhs[:, self.idx_X] = work5.reshape((p, -1)).view(np.float64)

        # ======================================================================
        # Hessian products with respect to Y
        # ======================================================================
        # Hessian products of trace function
        # D2_YY Ψ(X, Y)[Hy] = D2h(Y)[g(X), Hy]
        scnd_frechet_multi(work5, self.D2y_h, work1, self.Uy_gX_Uy, U=self.Uy,
                           work1=work3, work2=work4, work3=work6)  # fmt: skip

        # D2_YX Ψ(X, Y)[Hx] = Dh(Y)[Dg(X)[Hx]]
        work2 *= self.D1x_g
        congr_multi(work3, self.UxUy.conj().T, work2, work=work4)
        work3 *= self.D1y_h
        congr_multi(work0, self.Uy, work3, work=work4)
        work5 += work0

        # Hessian products of Renyi entropy
        # D2_Y S(X, Y)[(Hu, Hx, Hy)]
        #   = ((Hu/Ψ - u/Ψ^2 DΨ(X,Y)[(Hx,Hy)]) D_Y Ψ + u/Ψ D2_Y Ψ(X,Y)[(Hx,Hy)]) / (α-1)
        work5 *= self.u
        np.outer(rho, self.DTrY, out=work0.reshape((p, -1)))
        work5 += work0
        work5 /= self.Tr * (self.alpha - 1)

        # Hessian product of barrier function
        # D2_Y F(t, u, X, Y)[Ht, Hu, Hx, Hy]
        #    = -D2_t F(t, u, X, Y)[Ht, Hu, Hx, Hy] * D_Y S(u, X, Y)
        #      + D2_Y Ψ(X, Y)[(Hu, Hx, Hy)] / z + Y^-1 Hy Y^-1
        work5 *= self.zi
        np.outer(out_t, self.DPhiY, out=work1.reshape((p, -1)))
        work5 -= work1
        congr_multi(work1, self.inv_Y, self.Ay, work=work4)
        work5 += work1

        lhs[:, self.idx_Y] = work5.reshape((p, -1)).view(np.float64)

        # Multiply A (H A')
        return dense_dot_x(lhs, A.T)

    def invhess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        (Ht, Hu, Hx, Hy) = H

        # Compute Wu
        Wu = Hu + Ht * self.DPhiu

        # Compute Wx and get compact vectorization
        Wx = Hx + Ht * self.DPhiX
        Wx_vec = Wx.view(np.float64).reshape(-1, 1)
        Wx_cvec = self.F2C_op @ Wx_vec

        # Compute Wy and get compact vectorization
        Wy = Hy + Ht * self.DPhiY
        Wy_vec = Wy.view(np.float64).reshape(-1, 1)
        Wy_cvec = self.F2C_op @ Wy_vec

        # Solve for (u, X, Y) =  M \ (Wu, Wx, Wy)
        Wuxy_cvec = np.vstack((Wu, Wx_cvec, Wy_cvec))
        out_uXY = cho_solve(self.hess_fact, Wuxy_cvec)
        out_u = out_uXY[0]
        out_XY = out_uXY[1:].reshape(2, -1)

        out[1][:] = out_u

        out_X = self.F2C_op.T @ out_XY[0]
        out_X = out_X.view(self.dtype).reshape((self.n, self.n))
        out[2][:] = (out_X + out_X.conj().T) * 0.5

        out_Y = self.F2C_op.T @ out_XY[1]
        out_Y = out_Y.view(self.dtype).reshape((self.n, self.n))
        out[3][:] = (out_Y + out_Y.conj().T) * 0.5

        # Solve for t = z^2 Ht + <DPhi(u, X, Y), (u, X, Y)>
        out_t = self.z2 * Ht + out_u * self.DPhiu
        out_t += inp(out_X, self.DPhiX)
        out_t += inp(out_Y, self.DPhiY)
        out[0][:] = out_t

        return out

    def invhess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)

        # The inverse Hessian product applied on (Ht, Hx, Hy) for the SRE
        # barrier is
        #     (u, X, Y) =  M \ (Wu, Wx, Wy)
        #         t  =  z^2 Ht + <DPhi(u, X, Y), (u, X, Y)>
        # where (Wu, Wx, Wy) = (Hu, Hx, Hy) + Ht DPhi(u, X, Y) and
        #     M = 1/z [ D2uuPhi D2uxPhi D2uyPhi ] + [ 1 / u^2                         ]
        #             [ D2uxPhi D2xxPhi D2xyPhi ] + [         X^1 ⊗ X^-1             ]
        #             [ D2uyPhi D2yxPhi D2yyPhi ]   [                     Y^1 ⊗ Y^-1 ]

        # Compute (Wu, Wx, Wy)
        np.outer(self.DPhi_cvec, self.At, out=self.work)
        self.work += self.Auxy_cvec.T

        # Solve for (u, X, Y) =  M \ (Wu, Wx, Wy)
        out_uxy = cho_solve(self.hess_fact, self.work)

        # Solve for t = z^2 Ht + <DPhi(u, X, Y), (u, X, Y)>
        out_t = self.z2 * self.At.reshape(-1, 1) + out_uxy.T @ self.DPhi_cvec

        # Multiply A (H A')
        return x_dot_dense(self.Auxy_cvec, out_uxy) + np.outer(self.At, out_t)

    def third_dir_deriv_axpy(self, out, H, a=True):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        (Ht, Hu, Hx, Hy) = H

        self.zi3 = self.zi2 * self.zi

        UHxU = self.Ux.conj().T @ Hx @ self.Ux
        UHyU = self.Uy.conj().T @ Hy @ self.Uy

        # First derivatives
        DTrXH = inp(Hx, self.DTrX)
        DTrYH = inp(Hy, self.DTrY)
        DTrH = DTrXH + DTrYH

        # Hessian product of trace function
        # D2_XX Ψ(X, Y)[Hx] = D2g(X)[h(Y), Hx]
        D2TrXXH = scnd_frechet(self.D2x_g, self.Ux_hY_Ux, UHxU, U=self.Ux)
        # D2_XY Ψ(X, Y)[Hy] = Dg(X)[Dh(Y)[Hy]]
        work = self.UxUy @ (self.D1y_h * UHyU) @ self.UxUy.conj().T
        D2TrXYH = self.Ux @ (self.D1x_g * work) @ self.Ux.conj().T
        # D2_YX Ψ(X, Y)[Hx] = Dh(Y)[Dg(X)[Hx]]
        work = self.UxUy.conj().T @ (self.D1x_g * UHxU) @ self.UxUy
        D2TrYXH = self.Uy @ (self.D1y_h * work) @ self.Uy.conj().T
        # D2_YY Ψ(X, Y)[Hy] = D2h(Y)[g(X), Hy]
        D2TrYYH = scnd_frechet(self.D2y_h, self.Uy_gX_Uy, UHyU, U=self.Uy)

        D2TrXH = D2TrXXH + D2TrXYH
        D2TrYH = D2TrYXH + D2TrYYH
        D2TrHH = inp(Hx, D2TrXH) + inp(Hy, D2TrYH)

        # Hessian product of Renyi entropy
        rho = Hu - ((inp(self.DTrX, Hx) + inp(self.DTrY, Hy)) * self.u) / self.Tr
        # D2_u S(X, Y)[(Hu, Hx, Hy)] = (DΨ(X,Y)[(Hx,Hy)]/Ψ - Hu/u) / (α-1)
        D2PhiuH = -rho / self.u / (self.alpha - 1)
        # D2_X S(X, Y)[(Hu, Hx, Hy)]
        #   = ((Hu/Ψ - u/Ψ^2 DΨ(X,Y)[(Hx,Hy)]) D_X Ψ + u/Ψ D2_X Ψ(X,Y)[(Hx,Hy)]) / (α-1)
        D2PhiXH = (self.DTrX * rho + self.u * D2TrXH) / self.Tr / (self.alpha - 1)
        # D2_Y S(X, Y)[(Hu, Hx, Hy)]
        #   = ((Hu/Ψ - u/Ψ^2 DΨ(X,Y)[(Hx,Hy)]) D_Y Ψ + u/Ψ D2_Y Ψ(X,Y)[(Hx,Hy)]) / (α-1)
        D2PhiYH = (self.DTrY * rho + self.u * D2TrYH) / self.Tr / (self.alpha - 1)

        D2PhiuHH = inp(Hu, D2PhiuH)
        D2PhiXHH = inp(Hx, D2PhiXH)
        D2PhiYHH = inp(Hy, D2PhiYH)

        # Third order derivatives of trace function
        # Second derivatives of D_X Ψ(X, Y)
        D3TrXXX = thrd_frechet(self.Dx, self.D2x_g, self.d3g(self.Dx), self.Ux, 
                                self.Ux_hY_Ux, UHxU)  # fmt: skip

        work = self.UxUy @ (self.D1y_h * UHyU) @ self.UxUy.conj().T
        D3TrXXY = scnd_frechet(self.D2x_g, UHxU, work, U=self.Ux)

        D3TrXYX = D3TrXXY

        work = scnd_frechet(self.D2y_h, UHyU, UHyU, U=self.UxUy)
        D3TrXYY = self.Ux @ (self.D1x_g * work) @ self.Ux.conj().T

        # Second derivatives of D_Y Ψ(X, Y)
        D3TrYYY = thrd_frechet(self.Dy, self.D2y_h, self.d3h(self.Dy), self.Uy, 
                                self.Uy_gX_Uy, UHyU)  # fmt: skip

        work = self.UxUy.conj().T @ (self.D1x_g * UHxU) @ self.UxUy
        D3TrYYX = scnd_frechet(self.D2y_h, UHyU, work, U=self.Uy)

        D3TrYXY = D3TrYYX

        work = scnd_frechet(self.D2x_g, UHxU, UHxU, U=self.UxUy.conj().T)
        D3TrYXX = self.Uy @ (self.D1y_h * work) @ self.Uy.conj().T

        D3TrX = D3TrXXX + D3TrXXY + D3TrXYX + D3TrXYY
        D3TrY = D3TrYYY + D3TrYYX + D3TrYXY + D3TrYXX

        # Third order derivatives of Renyi entropy
        eta = 2 * self.u * DTrH * DTrH / self.Tr - 2 * Hu * DTrH - self.u * D2TrHH
        eta /= self.Tr * self.Tr

        D3PhiuHH = (Hu / self.u) ** 2 + (D2TrHH - DTrH * DTrH / self.Tr) / self.Tr
        D3PhiuHH /= self.alpha - 1

        D3PhiXHH = (2 * rho * D2TrXH + self.u * D3TrX) / self.Tr + self.DTrX * eta
        D3PhiXHH /= self.alpha - 1

        D3PhiYHH = (2 * rho * D2TrYH + self.u * D3TrY) / self.Tr + self.DTrY * eta
        D3PhiYHH /= self.alpha - 1

        # Third derivatives of barrier function
        chi = (Ht - self.DPhiu * Hu - inp(self.DPhiX, Hx) - inp(self.DPhiY, Hy))[0, 0]
        chi2 = chi * chi

        dder3_t = -2 * self.zi3 * chi2 - self.zi2 * (D2PhiuHH + D2PhiXHH + D2PhiYHH)

        dder3_u = -dder3_t * self.DPhiu
        dder3_u -= 2 * self.zi2 * chi * D2PhiuH
        dder3_u += self.zi * D3PhiuHH
        dder3_u -= 2 * Hu * Hu / self.u / self.u / self.u

        dder3_X = -dder3_t * self.DPhiX
        dder3_X -= 2 * self.zi2 * chi * D2PhiXH
        dder3_X += self.zi * D3PhiXHH
        dder3_X -= 2 * self.inv_X @ Hx @ self.inv_X @ Hx @ self.inv_X
        dder3_X = (dder3_X + dder3_X.conj().T) * 0.5

        dder3_Y = -dder3_t * self.DPhiY
        dder3_Y -= 2 * self.zi2 * chi * D2PhiYH
        dder3_Y += self.zi * D3PhiYHH
        dder3_Y -= 2 * self.inv_Y @ Hy @ self.inv_Y @ Hy @ self.inv_Y
        dder3_Y = (dder3_Y + dder3_Y.conj().T) * 0.5

        out[0][:] += dder3_t * a
        out[1][:] += dder3_u * a
        out[2][:] += dder3_X * a
        out[3][:] += dder3_Y * a

        return out

    # ========================================================================
    # Auxilliary functions
    # ========================================================================
    def congr_aux(self, A):
        assert not self.congr_aux_updated

        iscomplex = self.iscomplex

        # Get slices and views of A matrix to be used in congruence computations
        if sp.sparse.issparse(A):
            A = A.tocsr()
        Au = A[:, [1]]
        self.Ax_vec = A[:, self.idx_X]
        self.Ay_vec = A[:, self.idx_Y]
        Ax_cvec = (self.F2C_op @ self.Ax_vec.T).T
        Ay_cvec = (self.F2C_op @ self.Ay_vec.T).T
        if sp.sparse.issparse(A):
            self.Auxy_cvec = sp.sparse.hstack((Au, Ax_cvec, Ay_cvec), format="coo")
        else:
            self.Auxy_cvec = np.hstack((Au, Ax_cvec, Ay_cvec))

        if sp.sparse.issparse(A):
            A = A.toarray()
        Ax_dense = np.ascontiguousarray(A[:, self.idx_X])
        Ay_dense = np.ascontiguousarray(A[:, self.idx_Y])
        self.At = A[:, 0]
        self.Au = A[:, [1]]
        self.Ax = np.array([vec_to_mat(Ax_k, iscomplex) for Ax_k in Ax_dense])
        self.Ay = np.array([vec_to_mat(Ay_k, iscomplex) for Ay_k in Ay_dense])

        # Preallocate matrices we will need when performing these congruences
        self.work = np.empty_like(self.Auxy_cvec.T)

        self.work0 = np.empty_like(self.Ax)
        self.work1 = np.empty_like(self.Ax)
        self.work2 = np.empty_like(self.Ax)
        self.work3 = np.empty_like(self.Ax)
        self.work4 = np.empty_like(self.Ax)
        self.work5 = np.empty_like(self.Ax)

        self.work6 = np.empty((self.Ax.shape[::-1]), dtype=self.dtype)

        self.congr_aux_updated = True

    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        self.D2x_g = D2_f(self.Dx, self.D1x_g, self.d2g(self.Dx))
        self.D2y_h = D2_f(self.Dy, self.D1y_h, self.d2h(self.Dy))

        # Preparing other required variables
        self.UxUy = self.Ux.conj().T @ self.Uy
        self.zi2 = self.zi * self.zi

        self.hess_aux_updated = True

    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated
        if not self.invhess_aux_aux_updated:
            self.update_invhessprod_aux_aux()

        # Precompute and factorize the matrix
        #     M = 1/z [ D2uuPhi D2uxPhi D2uyPhi ] + [ 1 / u^2                         ]
        #             [ D2uxPhi D2xxPhi D2xyPhi ] + [         X^1 ⊗ X^-1             ]
        #             [ D2uyPhi D2yxPhi D2yyPhi ]   [                     Y^1 ⊗ Y^-1 ]

        self.z2 = self.z * self.z

        work10, work11, work12 = self.work10, self.work11, self.work12
        work13, work14, work15 = self.work13, self.work14, self.work15

        # Precompute compact vectorizations of derivatives
        DTrX_vec = self.DTrX.view(np.float64).reshape(-1, 1)
        DTrX_cvec = (self.F2C_op @ DTrX_vec).reshape(-1, 1, 1)
        DTrY_vec = self.DTrY.view(np.float64).reshape(-1, 1)
        DTrY_cvec = (self.F2C_op @ DTrY_vec).reshape(-1, 1, 1)

        DPhiX_vec = self.DPhiX.view(np.float64).reshape(-1, 1)
        DPhiX_cvec = self.F2C_op @ DPhiX_vec
        DPhiY_vec = self.DPhiY.view(np.float64).reshape(-1, 1)
        DPhiY_cvec = self.F2C_op @ DPhiY_vec
        self.DPhi_cvec = np.vstack((self.DPhiu, DPhiX_cvec, DPhiY_cvec))

        # ======================================================================
        # Construct blocks of Hessian corresponding to u
        # ======================================================================
        Huu = -self.zi / (self.u * (self.alpha - 1)) + 1 / self.u / self.u
        Hux = self.zi * DTrX_cvec.ravel() / self.Tr / (self.alpha - 1)
        Huy = self.zi * DTrY_cvec.ravel() / self.Tr / (self.alpha - 1)

        # ======================================================================
        # Construct XX block of Hessian, i.e., (D2xxPhi + X^-1 ⊗ X^-1)
        # ======================================================================
        # Hessian products of trace function
        # D2_XX Ψ(X, Y)[Hx] = D2g(X)[h(Y), Hx]
        congr_multi(work14, self.Ux.conj().T, self.E, work=work13)
        scnd_frechet_multi(work11, self.D2x_g, work14, self.Ux_hY_Ux, U=self.Ux,
                           work1=work12, work2=work13, work3=work15)  # fmt: skip

        # Hessian product of Renyi entropy
        # D2_XX S(X, Y)[Hx]
        #   = (u/Ψ D2_XX Ψ(X, Y)[Hx] - u/Ψ^2 D_X Ψ(X, Y)[Hx] D_X Ψ) / (α - 1)
        np.multiply(DTrX_cvec, self.DTrX.reshape(1, self.n, self.n), out=work13)
        work13 /= self.Tr
        work11 -= work13
        work11 *= self.zi * self.u / self.Tr / (self.alpha - 1)

        # X^-1 Eij X^-1
        congr_multi(work12, self.inv_X, self.E, work=work13)
        work12 += work11
        # Vectorize matrices as compact vectors to get square matrix
        work = work12.view(np.float64).reshape((self.vn, -1))
        Hxx = x_dot_dense(self.F2C_op, work.T)

        # ======================================================================
        # Construct YY block of Hessian, i.e., (D2yyPhi + Y^-1 ⊗ Y^-1)
        # ======================================================================
        # Hessian products of trace function
        # D2_YY Ψ(X, Y)[Hy] = D2h(Y)[g(X), Hy]
        congr_multi(work14, self.Uy.conj().T, self.E, work=work13)
        scnd_frechet_multi(work10, self.D2y_h, work14, self.Uy_gX_Uy, U=self.Uy,
                           work1=work12, work2=work13, work3=work15)  # fmt: skip

        # Hessian product of Renyi entropy
        # D2_YY S(X, Y)[Hy]
        #   = (u/Ψ D2_YY Ψ(X, Y)[Hy] - u/Ψ^2 D_Y Ψ(X, Y)[Hy] D_Y Ψ) / (α - 1)
        np.multiply(DTrY_cvec, self.DTrY.reshape(1, self.n, self.n), out=work13)
        work13 /= self.Tr
        work10 -= work13
        work10 *= self.zi * self.u / self.Tr / (self.alpha - 1)

        # Y^1 Eij Y^-1
        congr_multi(work12, self.inv_Y, self.E, work=work13)
        work12 += work10
        # Vectorize matrices as compact vectors to get square matrix
        work = work12.view(np.float64).reshape((self.vn, -1))
        Hyy = x_dot_dense(self.F2C_op, work.T)

        # ======================================================================
        # Construct XY block of Hessian, i.e., D2xyPhi
        # ======================================================================
        # Hessian products of trace function
        # D2_XY Ψ(X, Y)[Hy] = Dg(X)[Dh(Y)[Hy]]
        work14 *= self.D1y_h
        congr_multi(work12, self.UxUy, work14, work=work13)
        work12 *= self.D1x_g
        congr_multi(work14, self.Ux, work12, work=work13)

        # Hessian product of Renyi entropy
        # D2_XX S(X, Y)[Hx]
        #   = (u/Ψ D2_XY Ψ(X, Y)[Hy] - u/Ψ^2 D_Y Ψ(X, Y)[Hy] D_X Ψ) / (α - 1)
        np.multiply(DTrY_cvec, self.DTrX.reshape(1, self.n, self.n), out=work13)
        work13 /= self.Tr
        work14 -= work13
        work14 *= self.zi * self.u / self.Tr / (self.alpha - 1)

        # Vectorize matrices as compact vectors to get square matrix
        work = work14.view(np.float64).reshape((self.vn, -1))
        Hxy = x_dot_dense(self.F2C_op, work.T)

        # Construct Hessian and factorize
        Hxx = (Hxx + Hxx.conj().T) * 0.5
        Hyy = (Hyy + Hyy.conj().T) * 0.5

        self.hess[0, 0] = Huu
        self.hess[0, 1 : 1 + self.vn] = Hux
        self.hess[0, 1 + self.vn :] = Huy
        self.hess[1 : 1 + self.vn, 0] = Hux
        self.hess[1 + self.vn :, 0] = Huy

        self.hess[1 : 1 + self.vn, 1 : 1 + self.vn] = Hxx
        self.hess[1 + self.vn :, 1 + self.vn :] = Hyy
        self.hess[1 + self.vn :, 1 : 1 + self.vn] = Hxy.T
        self.hess[1 : 1 + self.vn, 1 + self.vn :] = Hxy

        self.hess_fact = cho_fact(self.hess)
        self.invhess_aux_updated = True

        return

    def update_invhessprod_aux_aux(self):
        assert not self.invhess_aux_aux_updated

        self.precompute_computational_basis()

        self.hess = np.empty((1 + 2 * self.vn, 1 + 2 * self.vn))

        self.work10 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work11 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work12 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work13 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work14 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work15 = np.empty((self.n, self.n, self.vn), dtype=self.dtype)

        self.invhess_aux_aux_updated = True

    def get_central_ray(self):
        # Solve a 4-dimensional nonlinear system of equations to get the central
        # point of the barrier function
        n, alpha = self.n, self.alpha
        (t, u, x, y) = (1.0 + n * self.g(1.0), 1.0, 1.0, 1.0)

        for _ in range(20):
            # Precompute some useful things
            x_a, y_b = np.power(x, alpha), np.power(y, 1 - alpha)
            z = t - u * np.log(n * x_a * y_b / u) / (alpha - 1)
            zi = 1 / z
            zi2 = zi * zi

            du = (np.log(n * x_a * y_b / u) - 1) / (alpha - 1)
            dx = alpha / (alpha - 1) * u / x
            dy = -u / y

            d2du2 = -1 / (alpha - 1) / u
            d2dudx = alpha / (alpha - 1) / x
            d2dudy = -1 / y
            d2dx2 = -alpha / (alpha - 1) * u / x / x
            d2dy2 = 0
            d2dxdy = u / y / y

            # Get gradient
            g = np.array([t - zi, 
                          u + du * zi - 1 / u, 
                          n * x + dx * zi - n / x, 
                          n * y + dy * zi - n / y])  # fmt: skip

            # Get Hessian
            (Htt, Htu, Htx, Hty) = (zi2, -zi2 * du, -zi2 * dx, -zi2 * dy)
            Huu = zi2 * dx * dx + zi * d2du2 + 1 / u / u
            Hux = zi2 * du * dx + zi * d2dudx
            Huy = zi2 * du * dy + zi * d2dudy
            Hxx = zi2 * dx * dx + zi * d2dx2 + n / x / x
            Hyy = zi2 * dy * dy + zi * d2dy2 + n / y / y
            Hxy = zi2 * dx * dy + zi * d2dxdy

            H = np.array([[Htt + 1, Htu, Htx, Hty],
                          [Htu, Huu + 1, Hux, Huy],
                          [Htx, Hux, Hxx + n, Hxy],
                          [Hty, Huy, Hxy, Hyy + n]])  # fmt: skip

            # Perform Newton step
            delta = -np.linalg.solve(H, g)
            decrement = -np.dot(delta, g)

            # Check feasible
            (t1, u1, x1, y1) = (t + delta[0], u + delta[1], x + delta[2], y + delta[3])
            x_a, y_b = np.power(x, alpha), np.power(y, 1 - alpha)
            if x1 < 0 or y1 < 0 or t1 < u1 * np.log(n * x_a * y_b / u1) / (alpha - 1):
                # Exit if not feasible and return last feasible point
                break

            (t, u, x, y) = (t1, u1, x1, y1)

            # Exit if decrement is small, i.e., near optimality
            if decrement / 2.0 <= 1e-12:
                break

        return (t, u, x, y)
