# Copyright (c) 2024, Kerry He, James Saunderson, and Hamza Fawzi

# This Python package QICS is licensed under the MIT license; see LICENSE.md
# file in the root directory or at https://github.com/kerry-he/qics

import numpy as np
import scipy as sp

from qics._utils.linalg import dense_dot_x
from qics.cones.base import Cone, get_central_ray_relentr


class ClassRelEntr(Cone):
    r"""A class representing a classical relative entropy cone

    .. math::

        \mathcal{CRE}_{n} = \text{cl}\{ (t, x, y) \in \mathbb{R} \times
        \mathbb{R}^n_{++} \times \mathbb{R}^n_{++} : t \geq H(x \| y) \},

    where

    .. math::

        H(x \| y) = \sum_{i=1}^n x_i \log(x_i / y_i),

    is the classical relative entropy function (Kullback-Leibler
    divergence).

    Parameters
    ----------
    n : :obj:`int`
        Dimension of the vectors :math:`x` and :math:`y`, i.e., how many
        terms are in the classical relative entropy function.

    See also
    --------
    ClassEntr : (Homogenized) classical entropy cone
    QuantRelEntr : Quantum relative entropy cone
    """

    def __init__(self, n):
        self.n = n

        self.nu = 1 + 2 * self.n  # Barrier parameter

        self.dim = [1, n, n]
        self.type = ["r", "r", "r"]

        self.idx_X = slice(1, 1 + n)
        self.idx_Y = slice(1 + n, 1 + 2 * n)

        # Update flags
        self.feas_updated = False
        self.grad_updated = False
        self.congr_aux_updated = False
        self.hess_aux_updated = False
        self.invhess_aux_updated = False
        self.dder3_aux_updated = False

        return

    def get_init_point(self, out):
        (t0, x0, y0) = get_central_ray_relentr(self.n)

        point = [
            np.array([[t0]]),
            np.ones((self.n, 1)) * x0,
            np.ones((self.n, 1)) * y0,
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

        (self.t, self.x, self.y) = self.primal

        # Check that x and y are strictly positive
        if any(self.x <= 0) or any(self.y <= 0):
            self.feas = False
            return self.feas

        # Check that t > H(x||y)
        self.log_x = np.log(self.x)
        self.log_y = np.log(self.y)

        self.z = (self.t - (self.x.T @ (self.log_x - self.log_y)))[0, 0]

        self.feas = self.z > 0
        return self.feas

    def get_val(self):
        return -np.log(self.z) - np.sum(self.log_x) - np.sum(self.log_y)

    def update_grad(self):
        assert self.feas_updated
        assert not self.grad_updated

        # Compute gradients of classical relative entropy
        # D_x H(x||y) = log(x) - log(y) + 1
        self.DPhiX = self.log_x - self.log_y + 1
        # D_y H(x||y) = -x / y
        self.DPhiY = -self.x / self.y

        # Compute 1 / x and 1 / y
        self.xi = np.reciprocal(self.x)
        self.yi = np.reciprocal(self.y)

        # Compute gradient of barrier function
        self.zi = np.reciprocal(self.z)

        self.grad = [
            -self.zi,
            self.zi * self.DPhiX - self.xi,
            self.zi * self.DPhiY - self.yi,
        ]

        self.grad_updated = True

    def hess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        (Ht, Hx, Hy) = H

        # Hessian product of classical relative entropy
        D2PhiXH = Hx * self.xi - Hy * self.yi
        D2PhiYH = -Hx * self.yi + Hy * self.x * self.yi2

        # Hessian product of barrier function
        out[0][:] = (Ht - Hx.T @ self.DPhiX - Hy.T @ self.DPhiY) * self.zi2
        out[1][:] = -out[0] * self.DPhiX + self.zi * D2PhiXH + Hx * self.xi2
        out[2][:] = -out[0] * self.DPhiY + self.zi * D2PhiYH + Hy * self.yi2

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

        # ======================================================================
        # Hessian products with respect to t
        # ======================================================================
        # D2_t F(t, x, y)[Ht, Hx, Hy]
        #         = (Ht - D_x H(x||y)[Hx] - D_y H(x||y)[Hy]) / z^2
        outt = self.At - (self.Ax @ self.DPhiX).ravel()
        outt -= (self.Ay @ self.DPhiY).ravel()
        outt *= self.zi2

        lhs[:, 0] = outt

        # ======================================================================
        # Hessian products with respect to x
        # ======================================================================
        # Precompute Hessian products for classical relative entropy
        # D2_xx Phi(x, y) [Hx] =  Hx / x
        np.multiply(self.Ax, self.Hxx.T, out=work0)
        # D2_xy Phi(x, y) [Hx] = -Hx / y
        np.multiply(self.Ay, self.Hxy.T, out=work1)

        # Hessian product of barrier function
        # D2_x F(t, x, y)[Ht, Hx, Hy]
        #         = -D2_t F(t, x, y)[Ht, Hx, Hy] * D_x H(x||y)
        #           + (D2_xx H(x||y)[Hx] + D2_xy H(x||y)[Hy]) / z
        #           + Hx / x^2
        work0 += work1
        np.outer(outt, self.DPhiX, out=work1)
        work0 -= work1

        lhs[:, self.idx_X] = work0

        # ======================================================================
        # Hessian products with respect to y
        # ======================================================================
        # Precompute Hessian products for classical relative entropy
        # D2_yx Phi(x, y) [Hy] = -Hy / y
        np.multiply(self.Ax, self.Hxy.T, out=work0)
        # D2_yy Phi(x, y) [Hy] =  Hy * x / y^2
        np.multiply(self.Ay, self.Hyy.T, out=work1)

        # Hessian product of barrier function
        # D2_y F(t, x, y)[Ht, Hx, Hy]
        #         = -D2_t F(t, x, y)[Ht, Hx, Hy] * D_y H(x||y)
        #           + (D2_yx H(x||y)[Hx] + D2_yy H(x||y)[Hy]) / z
        #           + Hy / y^2
        work0 += work1
        np.outer(outt, self.DPhiY, out=work1)
        work0 -= work1

        lhs[:, self.idx_Y] = work0

        # Multiply A (H A')
        return dense_dot_x(lhs, A.T)

    def invhess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        (Ht, Hx, Hy) = H

        Wx = Hx + Ht * self.DPhiX
        Wy = Hy + Ht * self.DPhiY

        # Inverse Hessian product of classical relative entropy
        outX = self.Hxx_inv * Wx + self.Hxy_inv * Wy
        outY = self.Hxy_inv * Wx + self.Hyy_inv * Wy

        # Inverse Hessian product of barrier function
        out[0][:] = Ht * self.z2 + outX.T @ self.DPhiX + outY.T @ self.DPhiY
        out[1][:] = outX
        out[2][:] = outY

        return out

    def invhess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)

        # The inverse Hessian product applied on (Ht, Hx, Hy) for the CRE
        # barrier is
        #     (x, y) =  M \ (Wx, Wy)
        #         t  =  z^2 Ht + <DPhi(x, y), (x, y)>
        # where (Wx, Wy) = [(Hx, Hy) + Ht DPhi(x, y)]
        #     M = [  diag(1/zx + 1/x^2)       -diag(1/zy)     ] = [ Hxx Hxy ]
        #         [     -diag(1/zy)      diag(x/zy^2 + 1/y^2) ]   [ Hxy Hyy ]
        # The inverse of a block matrix with diagonal blocks is another block
        # matrix with diaognal blocks
        #     M^-1 = [ (Hxx - Hxy^2 * Hyy^-1)^-1  (Hxy - Hxx Hyy Hxy^-1)^-1 ]
        #            [ (Hxy - Hxx Hyy Hxy^-1)^-1  (Hyy - Hxy^2 * Hxx^-1)^-1 ]
        #          = [ Hxx_inv  Hxy_inv ]
        #            [ Hxy_inv  Hyy_inv ]

        p = A.shape[0]
        lhs = np.empty((p, sum(self.dim)))

        work0, work1 = self.work0, self.work1
        work2, work3, work4 = self.work2, self.work3, self.work4

        # Compute Wx = Hx + Ht D_x H(x||y)
        np.outer(self.At, self.DPhiX, out=work4)
        work4 += self.Ax
        # Compute Wy = Hy + Ht D_y H(x||y)
        np.outer(self.At, self.DPhiY, out=work3)
        work3 += self.Ay

        # ======================================================================
        # Inverse Hessian products with respect to x
        # ======================================================================
        # x = Hxx_inv Wx + Hxy_inv Wy
        np.multiply(work4, self.Hxx_inv.T, out=work0)
        np.multiply(work3, self.Hxy_inv.T, out=work1)
        work0 += work1
        lhs[:, self.idx_X] = work0

        # ======================================================================
        # Inverse Hessian products with respect to y
        # ======================================================================
        # y = Hxy_inv Wx + Hyy_inv Wy
        np.multiply(work4, self.Hxy_inv.T, out=work1)
        np.multiply(work3, self.Hyy_inv.T, out=work2)
        work1 += work2
        lhs[:, self.idx_Y] = work1

        # ======================================================================
        # Inverse Hessian products with respect to t
        # ======================================================================
        # t = z^2 Ht + <DPhi(x, y), (x, y)>
        outt = self.z2 * self.At
        outt += (work0 @ self.DPhiX).ravel()
        outt += (work1 @ self.DPhiY).ravel()
        lhs[:, 0] = outt

        return dense_dot_x(lhs, A.T)

    def third_dir_deriv_axpy(self, out, H, a=True):
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.dder3_aux_updated:
            self.update_dder3_aux()

        (Ht, Hx, Hy) = H

        Hx2 = Hx * Hx
        Hy2 = Hy * Hy

        chi = (Ht - self.DPhiX.T @ Hx - self.DPhiY.T @ Hy)[0, 0]
        chi2 = chi * chi

        # Classical relative entropy Hessians
        D2PhiXH = Hx * self.xi - Hy * self.yi
        D2PhiYH = -Hx * self.yi + Hy * self.x * self.yi2

        D2PhiXHH = Hx.T @ D2PhiXH
        D2PhiYHH = Hy.T @ D2PhiYH

        # Classical relative entropy third order derivatives
        D3PhiXHH = -Hx2 * self.xi2 + Hy2 * self.yi2
        D3PhiYHH = 2 * Hy * (Hx - Hy * self.x * self.yi) * self.yi2

        # Third derivatives of barrier
        dder3_t = -2 * self.zi3 * chi2 - self.zi2 * (D2PhiXHH + D2PhiYHH)

        dder3_x = -dder3_t * self.DPhiX
        dder3_x -= 2 * self.zi2 * chi * D2PhiXH
        dder3_x += self.zi * D3PhiXHH
        dder3_x -= 2 * Hx2 * self.xi3

        dder3_y = -dder3_t * self.DPhiY
        dder3_y -= 2 * self.zi2 * chi * D2PhiYH
        dder3_y += self.zi * D3PhiYHH
        dder3_y -= 2 * Hy2 * self.yi3

        out[0][:] += dder3_t * a
        out[1][:] += dder3_x * a
        out[2][:] += dder3_y * a

        return out

    # ==========================================================================
    # Auxilliary functions
    # ==========================================================================
    def congr_aux(self, A):
        assert not self.congr_aux_updated

        if sp.sparse.issparse(A):
            A = A.toarray()

        self.At = A[:, 0]
        self.Ax = np.ascontiguousarray(A[:, self.idx_X])
        self.Ay = np.ascontiguousarray(A[:, self.idx_Y])

        self.work0 = np.empty_like(self.Ax)
        self.work1 = np.empty_like(self.Ax)
        self.work2 = np.empty_like(self.Ax)
        self.work3 = np.empty_like(self.Ax)
        self.work4 = np.empty_like(self.Ax)

        self.congr_aux_updated = True

    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        self.zi2 = self.zi * self.zi
        self.xi2 = self.xi * self.xi
        self.yi2 = self.yi * self.yi

        self.Hxx = self.zi * self.xi + self.xi2
        self.Hxy = -self.zi * self.yi
        self.Hyy = (self.zi * self.x + 1) * self.yi2

        self.hess_aux_updated = True

    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated

        self.z2 = self.z * self.z

        self.Hxx_inv = np.reciprocal(self.Hxx - self.Hxy * self.Hxy / self.Hyy)
        self.Hxy_inv = np.reciprocal(self.Hxy - self.Hxx * self.Hyy / self.Hxy)
        self.Hyy_inv = np.reciprocal(self.Hyy - self.Hxy * self.Hxy / self.Hxx)

    def update_dder3_aux(self):
        assert not self.dder3_aux_updated
        assert self.hess_aux_updated

        self.zi3 = self.zi * self.zi2
        self.xi3 = self.xi * self.xi2
        self.yi3 = self.yi * self.yi2

        self.dder3_aux_updated = True
