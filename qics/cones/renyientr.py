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
from qics.cones.base import Cone, get_perspective_derivatives
from qics.vectorize import get_full_to_compact_op, vec_to_mat


class RenyiEntr(Cone):
    r"""A class representing the epigraph or hypograph of the trace function used to 
    define the Renyi entropy, i.e.,

    .. math::

        \mathcal{RE}_{n, \alpha} = \text{cl} \{ (t, X, Y) \in \mathbb{R} \times
        \mathbb{H}^n_{++} \times \mathbb{H}^n_{++} : t \geq -\text{tr}[ 
        X^\alpha Y^{1-\alpha} ] \},

    when :math:`\alpha\in[0, 1]`, and

    .. math::

        \mathcal{RE}_{n, \alpha} = \text{cl} \{ (t, X, Y) \in \mathbb{R} \times
        \mathbb{H}^n_{++} \times \mathbb{H}^n_{++} : t \geq \text{tr}[ 
        X^\alpha Y^{1-\alpha} ] \},

    when :math:`\alpha\in[-1, 0] \cup [1, 2]`.

    Parameters
    ----------
    n : :obj:`int`
        Dimension of the matrices :math:`X` and :math:`Y`.
    alpha : :obj:`float`
        The exponent :math:`\alpha` used to parameterize the sandwiched Renyi entropy.
    iscomplex : :obj:`bool`
        Whether the matrices :math:`X` and :math:`Y` are defined over
        :math:`\mathbb{H}^n` (``True``), or restricted to
        :math:`\mathbb{S}^n` (``False``). The default is ``False``.

    See also
    --------
    TrSandRenyiEntr : Sandwiched Renyi entropy
    QuantRelEntr : Quantum relative entropy

    Notes
    -----
    The Renyi entropy is actually defined as the function

    .. math::

        D_\alpha(X \| Y) = \frac{1}{\alpha - 1} \log(\Psi_\alpha(X, Y)),

    where

    .. math::

        \Psi_\alpha(X, Y) = \text{tr}[ X^\alpha Y^{1-\alpha} ].

    Note that :math:`\Psi_\alpha` is jointly concave for :math:`\alpha\in[1/2, 1]`, and
    jointly convex for :math:`\alpha\in[-1, 0] \cup [1, 2]`, whereas :math:`D_\alpha` is 
    jointly convex for :math:`\alpha\in[0, 1)`, but is neither convex nor concave for 
    :math:`\alpha\in[-1, 0) \cup (1, 2]`.

    Note that due to monotonicity of :math:`x \mapsto \log(x)`, we can minimize the
    sandwiched Renyi entropy by using the identities

    .. math::

        \min_{(X,Y)\in\mathcal{C}} D_\alpha(X \| Y)  = \frac{1}{\alpha - 1} 
        \log\left( \max_{(X,Y)\in\mathcal{C}} \Psi_\alpha(X, Y) \right),

    if :math:`\alpha\in[0, 1)`, and

    .. math::

        \min_{(X,Y)\in\mathcal{C}} D_\alpha(X \| Y)  = \frac{1}{\alpha - 1} 
        \log\left( \min_{(X,Y)\in\mathcal{C}} \Psi_\alpha(X, Y) \right),
    
    if :math:`\alpha\in(1, 2]`. Similarly, we can maximize the sandwiched Renyi entropy
    by using the identities

    .. math::

        \max_{(X,Y)\in\mathcal{C}} D_\alpha(X \| Y)  = \frac{1}{\alpha - 1}
        \log\left( \min_{(X,Y)\in\mathcal{C}} \Psi_\alpha(X, Y) \right),

    for :math:`\alpha\in[-1, 0]`.

    """

    def __init__(self, n, alpha, iscomplex=False):
        assert -1 <= alpha and alpha <= 2

        self.n = n
        self.alpha = alpha
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

        # Get function handles for g(x)=x^α
        # and their first, second and third derivatives
        perspective_derivatives = get_perspective_derivatives(alpha)
        self.g, self.dg, self.d2g, self.d3g = perspective_derivatives["g"]

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
        (t0, x0, y0) = self.get_central_ray()

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

        # Construct X^α and Y^β
        Dgx = self.g(self.Dx)
        gX = (self.Ux * Dgx) @ self.Ux.conj().T
        self.gX = (gX + gX.conj().T) / 2

        Dhy = self.h(self.Dy)
        hY = (self.Uy * Dhy) @ self.Uy.conj().T
        self.hY = (hY + hY.conj().T) / 2

        # Check that t > tr[ X^α Y^β ]
        self.z = self.t[0, 0] - np.sum(self.gX * self.hY.conj()).real

        self.feas = self.z > 0
        return self.feas

    def get_val(self):
        assert self.feas_updated
        return -np.log(self.z) - np.sum(np.log(self.Dx)) - np.sum(np.log(self.Dy))

    def update_grad(self):
        assert self.feas_updated
        assert not self.grad_updated

        # Precompute useful expressions
        self.D1x_g = D1_f(self.Dx, self.g(self.Dx), self.dg(self.Dx))
        self.D1y_h = D1_f(self.Dy, self.h(self.Dy), self.dh(self.Dy))

        self.Ux_hY_Ux = self.Ux.conj().T @ self.hY @ self.Ux
        self.Uy_gX_Uy = self.Uy.conj().T @ self.gX @ self.Uy

        # Compute gradients of sandwiched Renyi entropy
        # D_X Ψ(X, Y) = Dg(X)[h(Y)]
        self.DPhiX = self.Ux @ (self.D1x_g * self.Ux_hY_Ux) @ self.Ux.conj().T
        self.DPhiX = (self.DPhiX + self.DPhiX.conj().T) * 0.5
        # D_Y Ψ(X, Y) = Dh(Y)[g(X)]
        self.DPhiY = self.Uy @ (self.D1y_h * self.Uy_gX_Uy) @ self.Uy.conj().T
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

        UHxU = self.Ux.conj().T @ Hx @ self.Ux
        UHyU = self.Uy.conj().T @ Hy @ self.Uy

        # Hessian product of sandwiched Renyi entropy
        # D2_XX Ψ(X, Y)[Hx] = D2g(X)[h(Y), Hx]
        D2PhiXXH = scnd_frechet(self.D2x_g, self.Ux_hY_Ux, UHxU, U=self.Ux)
        # D2_XY Ψ(X, Y)[Hy] = Dg(X)[Dh(Y)[Hy]]
        work = self.UxUy @ (self.D1y_h * UHyU) @ self.UxUy.conj().T
        D2PhiXYH = self.Ux @ (self.D1x_g * work) @ self.Ux.conj().T
        # D2_YX Ψ(X, Y)[Hx] = Dh(Y)[Dg(X)[Hx]]
        work = self.UxUy.conj().T @ (self.D1x_g * UHxU) @ self.UxUy
        D2PhiYXH = self.Uy @ (self.D1y_h * work) @ self.Uy.conj().T
        # D2_YY Ψ(X, Y)[Hy] = D2h(Y)[g(X), Hy]
        D2PhiYYH = scnd_frechet(self.D2y_h, self.Uy_gX_Uy, UHyU, U=self.Uy)

        # ======================================================================
        # Hessian products with respect to t
        # ======================================================================
        # D2_t F(t, X, Y)[Ht, Hx, Hy] = (Ht - D_X Ψ(X, Y)[Hx] - D_Y Ψ(X, Y)[Hy]) / z^2
        out_t = Ht - inp(self.DPhiX, Hx) - inp(self.DPhiY, Hy)
        out_t *= self.zi2
        out[0][:] = out_t

        # ======================================================================
        # Hessian products with respect to X
        # ======================================================================
        # D2_X F(t, X, Y)[Ht, Hx, Hy] = -D2_t F(t, X, Y)[Ht, Hx, Hy] * D_X Ψ(X, Y)
        #                               + (D2_XX Ψ(X, Y)[Hx] + D2_XY Ψ(X, Y)[Hy]) / z
        #                               + X^-1 Hx X^-1
        out_X = -out_t * self.DPhiX
        out_X += self.zi * (D2PhiXYH + D2PhiXXH)
        out_X += self.inv_X @ Hx @ self.inv_X
        out_X = (out_X + out_X.conj().T) * 0.5
        out[1][:] = out_X

        # ==================================================================
        # Hessian products with respect to Y
        # ==================================================================
        # Hessian product of barrier function
        # D2_Y F(t, X, Y)[Ht, Hx, Hy] = -D2_t F(t, X, Y)[Ht, Hx, Hy] * D_Y Ψ(X, Y)
        #                               + (D2_YX Ψ(X, Y)[Hx] + D2_YY Ψ(X, Y)[Hy]) / z
        #                               + Y^-1 Hy Y^-1
        out_Y = -out_t * self.DPhiY
        out_Y += self.zi * (D2PhiYXH + D2PhiYYH)
        out_Y += self.inv_Y @ Hy @ self.inv_Y
        out_Y = (out_Y + out_Y.conj().T) * 0.5
        out[2][:] = out_Y

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

        # ======================================================================
        # Hessian products with respect to t
        # ======================================================================
        # D2_t F(t, X, Y)[Ht, Hx, Hy] = (Ht - D_X Ψ(X, Y)[Hx] - D_Y Ψ(X, Y)[Hy]) / z^2
        DPhiX_vec = self.DPhiX.view(np.float64).reshape((-1, 1))
        DPhiY_vec = self.DPhiY.view(np.float64).reshape((-1, 1))

        out_t = self.At - (self.Ax_vec @ DPhiX_vec).ravel()
        out_t -= (self.Ay_vec @ DPhiY_vec).ravel()
        out_t *= self.zi2

        lhs[:, 0] = out_t

        # ==================================================================
        # Hessian products with respect to X
        # ==================================================================
        # Hessian products of Renyi entropy
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

        # Hessian product of barrier function
        # D2_X F(t, X, Y)[Ht, Hx, Hy] = -D2_t F(t, X, Y)[Ht, Hx, Hy] * D_X Ψ(X, Y)
        #                               + (D2_XX Ψ(X, Y)[Hx] + D2_XY Ψ(X, Y)[Hy]) / z
        #                               + X^-1 Hx X^-1
        work5 *= self.zi
        np.outer(out_t, self.DPhiX, out=work3.reshape((p, -1)))
        work5 -= work3
        congr_multi(work4, self.inv_X, self.Ax, work=work3)
        work5 += work4

        lhs[:, self.idx_X] = work5.reshape((p, -1)).view(np.float64)

        # ======================================================================
        # Hessian products with respect to Y
        # ======================================================================
        # Hessian products of sandwiched Renyi entropy
        # D2_YY Ψ(X, Y)[Hy] = D2h(Y)[g(X), Hy]
        scnd_frechet_multi(work5, self.D2y_h, work1, self.Uy_gX_Uy, U=self.Uy,
                           work1=work3, work2=work4, work3=work6)  # fmt: skip

        # D2_YX Ψ(X, Y)[Hx] = Dh(Y)[Dg(X)[Hx]]
        work2 *= self.D1x_g
        congr_multi(work3, self.UxUy.conj().T, work2, work=work4)
        work3 *= self.D1y_h
        congr_multi(work0, self.Uy, work3, work=work4)
        work5 += work0

        # Hessian product of barrier function
        # D2_Y F(t, X, Y)[Ht, Hx, Hy] = -D2_t F(t, X, Y)[Ht, Hx, Hy] * D_Y Ψ(X, Y)
        #                               + (D2_YX Ψ(X, Y)[Hx] + D2_YY Ψ(X, Y)[Hy]) / z
        #                               + Y^-1 Hy Y^-1
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

        (Ht, Hx, Hy) = H

        # Compute Wx and get compact vectorization
        Wx = Hx + Ht * self.DPhiX
        Wx_vec = Wx.view(np.float64).reshape(-1, 1)
        Wx_cvec = self.F2C_op @ Wx_vec

        # Compute Wy and get compact vectorization
        Wy = Hy + Ht * self.DPhiY
        Wy_vec = Wy.view(np.float64).reshape(-1, 1)
        Wy_cvec = self.F2C_op @ Wy_vec

        # Solve for (X, Y) =  M \ (Wx, Wy)
        Wxy_cvec = np.vstack((Wx_cvec, Wy_cvec))
        out_XY = cho_solve(self.hess_fact, Wxy_cvec)
        out_XY = out_XY.reshape(2, -1)

        out_X = self.F2C_op.T @ out_XY[0]
        out_X = out_X.view(self.dtype).reshape((self.n, self.n))
        out[1][:] = (out_X + out_X.conj().T) * 0.5

        out_Y = self.F2C_op.T @ out_XY[1]
        out_Y = out_Y.view(self.dtype).reshape((self.n, self.n))
        out[2][:] = (out_Y + out_Y.conj().T) * 0.5

        # Solve for t = z^2 Ht + <DPhi(X, Y), (X, Y)>
        out_t = self.z2 * Ht
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
        #     (X, Y) =  M \ (Wx, Wy)
        #         t  =  z^2 Ht + <DPhi(X, Y), (X, Y)>
        # where (Wx, Wy) = (Hx, Hy) + Ht DPhi(X, Y) and
        #     M = 1/z [ D2xxPhi D2xyPhi ] + [ X^1 ⊗ X^-1              ]
        #             [ D2yxPhi D2yyPhi ]   [              Y^1 ⊗ Y^-1 ]

        # Compute (Wx, Wy)
        np.outer(self.DPhi_cvec, self.At, out=self.work)
        self.work += self.Axy_cvec.T

        # Solve for (X, Y) =  M \ (Wx, Wy)
        out_xy = cho_solve(self.hess_fact, self.work)

        # Solve for t = z^2 Ht + <DPhi(X, Y), (X, Y)>
        out_t = self.z2 * self.At.reshape(-1, 1) + out_xy.T @ self.DPhi_cvec

        # Multiply A (H A')
        return x_dot_dense(self.Axy_cvec, out_xy) + np.outer(self.At, out_t)

    def third_dir_deriv_axpy(self, out, H, a=True):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        (Ht, Hx, Hy) = H

        chi = Ht[0, 0] - inp(self.DPhiX, Hx) - inp(self.DPhiY, Hy)
        chi2 = chi * chi

        self.zi3 = self.zi2 * self.zi

        UHxU = self.Ux.conj().T @ Hx @ self.Ux
        UHyU = self.Uy.conj().T @ Hy @ self.Uy

        # Hessian product of sandwiched Renyi entropy
        # D2_XX Ψ(X, Y)[Hx] = D2g(X)[h(Y), Hx]
        D2PhiXXH = scnd_frechet(self.D2x_g, self.Ux_hY_Ux, UHxU, U=self.Ux)
        # D2_XY Ψ(X, Y)[Hy] = Dg(X)[Dh(Y)[Hy]]
        work = self.UxUy @ (self.D1y_h * UHyU) @ self.UxUy.conj().T
        D2PhiXYH = self.Ux @ (self.D1x_g * work) @ self.Ux.conj().T
        # D2_YX Ψ(X, Y)[Hx] = Dh(Y)[Dg(X)[Hx]]
        work = self.UxUy.conj().T @ (self.D1x_g * UHxU) @ self.UxUy
        D2PhiYXH = self.Uy @ (self.D1y_h * work) @ self.Uy.conj().T
        # D2_YY Ψ(X, Y)[Hy] = D2h(Y)[g(X), Hy]
        D2PhiYYH = scnd_frechet(self.D2y_h, self.Uy_gX_Uy, UHyU, U=self.Uy)

        D2PhiXHH = inp(Hx, D2PhiXXH + D2PhiXYH)
        D2PhiYHH = inp(Hy, D2PhiYXH + D2PhiYYH)

        # Trace noncommutative perspective third order derivatives

        # Second derivatives of D_X Ψ(X, Y)
        D3PhiXXX = thrd_frechet(self.Dx, self.D2x_g, self.d3g(self.Dx), self.Ux, 
                                self.Ux_hY_Ux, UHxU)  # fmt: skip
        
        work = self.UxUy @ (self.D1y_h * UHyU) @ self.UxUy.conj().T
        D3PhiXXY = scnd_frechet(self.D2x_g, UHxU, work, U=self.Ux)

        D3PhiXYX = D3PhiXXY

        work = scnd_frechet(self.D2y_h, UHyU, UHyU, U=self.UxUy)
        D3PhiXYY = self.Ux @ (self.D1x_g * work) @ self.Ux.conj().T

        # Second derivatives of D_Y Ψ(X, Y)
        D3PhiYYY = thrd_frechet(self.Dy, self.D2y_h, self.d3h(self.Dy), self.Uy, 
                                self.Uy_gX_Uy, UHyU)  # fmt: skip
        
        work = self.UxUy.conj().T @ (self.D1x_g * UHxU) @ self.UxUy
        D3PhiYYX = scnd_frechet(self.D2y_h, UHyU, work, U=self.Uy)

        D3PhiYXY = D3PhiYYX

        work = scnd_frechet(self.D2x_g, UHxU, UHxU, U=self.UxUy.conj().T)
        D3PhiYXX = self.Uy @ (self.D1y_h * work) @ self.Uy.conj().T

        # Third derivatives of barrier
        dder3_t = -2 * self.zi3 * chi2 - self.zi2 * (D2PhiXHH + D2PhiYHH)

        dder3_X = -dder3_t * self.DPhiX
        dder3_X -= 2 * self.zi2 * chi * (D2PhiXXH + D2PhiXYH)
        dder3_X += self.zi * (D3PhiXXX + D3PhiXXY + D3PhiXYX + D3PhiXYY)
        dder3_X -= 2 * self.inv_X @ Hx @ self.inv_X @ Hx @ self.inv_X
        dder3_X = (dder3_X + dder3_X.conj().T) * 0.5

        dder3_Y = -dder3_t * self.DPhiY
        dder3_Y -= 2 * self.zi2 * chi * (D2PhiYXH + D2PhiYYH)
        dder3_Y += self.zi * (D3PhiYYY + D3PhiYYX + D3PhiYXY + D3PhiYXX)
        dder3_Y -= 2 * self.inv_Y @ Hy @ self.inv_Y @ Hy @ self.inv_Y
        dder3_Y = (dder3_Y + dder3_Y.conj().T) * 0.5

        out[0][:] += dder3_t * a
        out[1][:] += dder3_X * a
        out[2][:] += dder3_Y * a

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
        self.Ax_vec = A[:, self.idx_X]
        self.Ay_vec = A[:, self.idx_Y]
        Ax_cvec = (self.F2C_op @ self.Ax_vec.T).T
        Ay_cvec = (self.F2C_op @ self.Ay_vec.T).T
        if sp.sparse.issparse(A):
            self.Axy_cvec = sp.sparse.hstack((Ax_cvec, Ay_cvec), format="coo")
        else:
            self.Axy_cvec = np.hstack((Ax_cvec, Ay_cvec))

        if sp.sparse.issparse(A):
            A = A.toarray()
        Ax_dense = np.ascontiguousarray(A[:, self.idx_X])
        Ay_dense = np.ascontiguousarray(A[:, self.idx_Y])
        self.At = A[:, 0]
        self.Ax = np.array([vec_to_mat(Ax_k, iscomplex) for Ax_k in Ax_dense])
        self.Ay = np.array([vec_to_mat(Ay_k, iscomplex) for Ay_k in Ay_dense])

        # Preallocate matrices we will need when performing these congruences
        self.work = np.empty_like(self.Axy_cvec.T)

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
        #     M = 1/z [ D2xxPhi D2xyPhi ] + [ X^1 ⊗ X^-1               ]
        #             [ D2yxPhi D2yyPhi ]   [               Y^1 ⊗ Y^-1 ]

        self.z2 = self.z * self.z

        work11, work12 = self.work11, self.work12
        work13, work14, work15 = self.work13, self.work14, self.work15

        # Precompute compact vectorizations of derivatives
        DPhiX_cvec = self.DPhiX.view(np.float64).reshape(-1, 1)
        DPhiX_cvec = self.F2C_op @ DPhiX_cvec

        DPhiY_cvec = self.DPhiY.view(np.float64).reshape(-1, 1)
        DPhiY_cvec = self.F2C_op @ DPhiY_cvec

        self.DPhi_cvec = np.vstack((DPhiX_cvec, DPhiY_cvec))

        # ======================================================================
        # Construct XX block of Hessian, i.e., (D2xxPhi + X^-1 ⊗ X^-1)
        # ======================================================================
        # D2_XX Ψ(X, Y)[Hx] = D2g(X)[h(Y), Hx]
        congr_multi(work14, self.Ux.conj().T, self.E, work=work13)
        scnd_frechet_multi(work11, self.D2x_g, work14, self.Ux_hY_Ux, U=self.Ux,
                           work1=work12, work2=work13, work3=work15)  # fmt: skip
        work11 *= self.zi
        # X^-1 Eij X^-1
        congr_multi(work12, self.inv_X, self.E, work=work13)
        work12 += work11
        # Vectorize matrices as compact vectors to get square matrix
        work = work12.view(np.float64).reshape((self.vn, -1))
        Hxx = x_dot_dense(self.F2C_op, work.T)

        # ======================================================================
        # Construct YY block of Hessian, i.e., (D2yyxPhi + Y^-1 ⊗ Y^-1)
        # ======================================================================
        # D2_YY Ψ(X, Y)[Hy] = D2h(Y)[g(X), Hy]
        congr_multi(work14, self.Uy.conj().T, self.E, work=work13)
        scnd_frechet_multi(work11, self.D2y_h, work14, self.Uy_gX_Uy, U=self.Uy,
                           work1=work12, work2=work13, work3=work15)  # fmt: skip
        work11 *= self.zi
        # Y^1 Eij Y^-1
        congr_multi(work12, self.inv_Y, self.E, work=work13)
        work12 += work11
        # Vectorize matrices as compact vectors to get square matrix
        work = work12.view(np.float64).reshape((self.vn, -1))
        Hyy = x_dot_dense(self.F2C_op, work.T)

        # ======================================================================
        # Construct XY block of Hessian, i.e., D2xyPhi
        # ======================================================================
        # D2_XY Ψ(X, Y)[Hy] = Dg(X)[Dh(Y)[Hy]]
        work14 *= self.D1y_h
        congr_multi(work12, self.UxUy, work14, work=work13)
        work12 *= self.D1x_g
        congr_multi(work14, self.Ux, work12, work=work13)
        work14 *= self.zi
        # Vectorize matrices as compact vectors to get square matrix
        work = work14.view(np.float64).reshape((self.vn, -1))
        Hxy = x_dot_dense(self.F2C_op, work.T)

        # Construct Hessian and factorize
        Hxx = (Hxx + Hxx.conj().T) * 0.5
        Hyy = (Hyy + Hyy.conj().T) * 0.5

        self.hess[: self.vn, : self.vn] = Hxx
        self.hess[self.vn :, self.vn :] = Hyy
        self.hess[self.vn :, : self.vn] = Hxy.T
        self.hess[: self.vn, self.vn :] = Hxy

        self.hess_fact = cho_fact(self.hess)
        self.invhess_aux_updated = True

        return

    def update_invhessprod_aux_aux(self):
        assert not self.invhess_aux_aux_updated

        self.precompute_computational_basis()

        self.hess = np.empty((2 * self.vn, 2 * self.vn))

        self.work11 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work12 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work13 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work14 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work15 = np.empty((self.n, self.n, self.vn), dtype=self.dtype)

        self.invhess_aux_aux_updated = True

    def get_central_ray(self):
        # Solve a 3-dimensional nonlinear system of equations to get the central
        # point of the barrier function
        n, alpha = self.n, self.alpha
        (t, x, y) = (1.0 + n * self.g(1.0), 1.0, 1.0)

        for _ in range(10):
            # Precompute some useful things
            z = t - n * self.g(x) * (y ** (1 - alpha))
            zi = 1 / z
            zi2 = zi * zi

            dx = self.dg(x) * (y ** (1 - alpha))
            dy = self.g(x) * (1 - alpha) * (y ** (-alpha))

            d2dx2 = self.d2g(x) * (y ** (1 - alpha))
            d2dy2 = self.g(x) * (1 - alpha) * (-alpha) * (y ** (-alpha - 1))
            d2dxdy = self.dg(x) * (1 - alpha) * (y ** (-alpha))

            # Get gradient
            g = np.array([t - zi, 
                          n * x + n * dx * zi - n / x, 
                          n * y + n * dy * zi - n / y])  # fmt: skip

            # Get Hessian
            (Htt, Htx, Hty) = (zi2, -n * zi2 * dx, -n * zi2 * dy)
            Hxx = n * n * zi2 * dx * dx + n * zi * d2dx2 + n / x / x
            Hyy = n * n * zi2 * dy * dy + n * zi * d2dy2 + n / y / y
            Hxy = n * n * zi2 * dx * dy + n * zi * d2dxdy

            H = np.array([[Htt + 1, Htx, Hty],
                          [Htx, Hxx + n, Hxy],
                          [Hty, Hxy, Hyy + n]])  # fmt: skip

            # Perform Newton step
            delta = -np.linalg.solve(H, g)
            decrement = -np.dot(delta, g)

            # Check feasible
            (t1, x1, y1) = (t + delta[0], x + delta[1], y + delta[2])
            if x1 < 0 or y1 < 0 or t1 < n * self.g(x) * (y1 ** (1 - alpha)):
                # Exit if not feasible and return last feasible point
                break

            (t, x, y) = (t1, x1, y1)

            # Exit if decrement is small, i.e., near optimality
            if decrement / 2.0 <= 1e-12:
                break

        return (t, x, y)
