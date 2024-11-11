# Copyright (c) 2024, Kerry He, James Saunderson, and Hamza Fawzi

# This Python package QICS is licensed under the MIT license; see LICENSE.md
# file in the root directory or at https://github.com/kerry-he/qics

import numpy as np
import scipy as sp

from qics._utils.gradient import (
    D1_log,
    D1_f,
    D2_log,
    D2_f,
    get_S_matrix,
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
from qics.cones.base import Cone, get_central_ray_relentr, get_perspective_derivatives
from qics.vectorize import get_full_to_compact_op, vec_to_mat


class SandwichedRenyiEntr(Cone):

    def __init__(self, n, alpha, iscomplex=False):
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

        # Get function handles for g(x)=x^alpha
        # and their first, second and third derivatives
        perspective_derivatives = get_perspective_derivatives(alpha)
        self.g, self.dg, self.d2g, self.d3g = perspective_derivatives["g"]

        b = (1 - alpha) / alpha
        self.h = lambda x : np.power(x, b)
        self.dh = lambda x : np.power(x, b - 1) * b
        self.d2h = lambda x : np.power(x, b - 2) * (b * (b - 1))
        self.d3h = lambda x : np.power(x, b - 3) * (b * (b - 1) * (b - 2))

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

        # Construct (X^(b/2) Y X^(b/2)) and (Y^1/2 X^b Y^1/2)
        # and double check they are also PSD (in case of numerical errors)
        beta = (1 - self.alpha) / self.alpha

        beta_Dx = np.power(self.Dx, beta)
        beta2_Dx = np.power(self.Dx, beta / 2)
        beta2_X = self.Ux * np.sqrt(beta_Dx)
        beta4_X = self.Ux * np.sqrt(beta2_Dx)
        ibeta4_X = self.Ux / np.sqrt(beta2_Dx)
        self.beta_X = beta2_X @ beta2_X.conj().T
        self.beta2_X = beta4_X @ beta4_X.conj().T
        self.ibeta2_X = ibeta4_X @ ibeta4_X.conj().T

        rt2_Dy = np.sqrt(self.Dy)
        rt4_Y = self.Uy * np.sqrt(rt2_Dy)
        self.rt2_Y = rt4_Y @ rt4_Y.conj().T

        XYX = self.beta2_X @ self.Y @ self.beta2_X
        YXY = self.rt2_Y @ self.beta_X @ self.rt2_Y

        self.Dxyx, self.Uxyx = np.linalg.eigh(XYX)
        self.Dyxy, self.Uyxy = np.linalg.eigh(YXY)

        if any(self.Dxyx <= 0) or any(self.Dyxy <= 0):
            self.feas = False
            return self.feas

        # Check that t > tr[ ( X^(b/2) Y X^(b/2) )^a ]
        self.g_Dxyx = self.g(self.Dxyx)
        self.z = self.t[0, 0] - np.sum(self.g_Dxyx)

        self.feas = self.z > 0
        return self.feas

    def get_val(self):
        assert self.feas_updated
        return -np.log(self.z) - np.sum(np.log(self.Dx)) - np.sum(np.log(self.Dy))

    def update_grad(self):
        assert self.feas_updated
        assert not self.grad_updated

        # Compute gradients of sandwiched Renyi entropy
        # D_X f(X, Y) = Ux ( h^[1](Dx) .* [Ux' Y^½ a*( Y^½ X^b Y^½ )^(a-1) Y^½ Ux] ) Ux'
        self.D1x_h = D1_f(self.Dx, self.h(self.Dx), self.dh(self.Dx))

        self.dg_Dyxy = self.dg(self.Dyxy)
        self.dg_YXY = (self.Uyxy * self.dg_Dyxy) @ self.Uyxy.conj().T
        work = self.Ux.conj().T @ self.rt2_Y @ self.dg_YXY @ self.rt2_Y @ self.Ux
        work = self.Ux @ (self.D1x_h * work) @ self.Ux.conj().T
        self.DPhiX = (work + work.conj().T) * 0.5

        # D_Y f(X, Y) = X^b/2 a*( X^b/2 Y X^b/2 )^(a-1) X^b/2
        self.dg_Dxyx = self.dg(self.Dxyx)
        self.dg_XYX = (self.Uxyx * self.dg_Dxyx) @ self.Uxyx.conj().T
        self.DPhiY = self.beta2_X @ self.dg_XYX @ self.beta2_X
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

        self.D1x_h = D1_f(self.Dx, self.h(self.Dx), self.dh(self.Dx))
        self.D2x_h = D2_f(self.Dx, self.D1x_h, self.d2h(self.Dx))

        self.D1xyx_g = D1_f(self.Dxyx, self.g(self.Dxyx), self.dg(self.Dxyx))
        self.D1yxy_dg = D1_f(self.Dyxy, self.dg(self.Dyxy), self.d2g(self.Dyxy))
        self.D1xyx_dg = D1_f(self.Dxyx, self.dg(self.Dxyx), self.d2g(self.Dxyx))

        # Hessian product of trace operator perspective
        # D2_XX D(X, Y)[Hx]
        work = self.D1x_h * (self.Ux.conj().T @ Hx @ self.Ux)
        work = self.Uyxy.conj().T @ self.rt2_Y @ self.Ux @ work @ self.Ux.conj().T @ self.rt2_Y @ self.Uyxy
        work = self.Ux.conj().T @ self.rt2_Y @ self.Uyxy @ (self.D1yxy_dg * work) @ self.Uyxy.conj().T @ self.rt2_Y @ self.Ux
        D2PhiXXH = self.Ux @ (self.D1x_h * work) @ self.Ux.conj().T

        work = self.Ux.conj().T @ self.rt2_Y @ self.dg_YXY @ self.rt2_Y @ self.Ux
        D2PhiXXH += scnd_frechet(self.D2x_h, work, self.Ux.conj().T @ Hx @ self.Ux, U=self.Ux)

        # D2_XY D(X, Y)[Hy]

        work = self.D1xyx_g * (self.Uxyx.conj().T @ self.beta2_X @ Hy @ self.beta2_X @ self.Uxyx)
        work = self.Ux.conj().T @ self.ibeta2_X @ self.Uxyx @ work @ self.Uxyx.conj().T @ self.ibeta2_X @ self.Ux
        D2PhiXYH = self.alpha * self.Ux @ (work * self.D1x_h) @ self.Ux.conj().T

        # D2_YX D(X, Y)[Hx]
        work = self.D1x_h * (self.Ux.conj().T @ Hx @ self.Ux)
        work = self.Uxyx.conj().T @ self.ibeta2_X @ self.Ux @ work @ self.Ux.conj().T @ self.ibeta2_X @ self.Uxyx
        D2PhiYXH = self.alpha * self.beta2_X @ self.Uxyx @ (self.D1xyx_g * work) @ self.Uxyx.conj().T @ self.beta2_X

        # D2_YY D(X, Y)[Hy]
        work = self.Uxyx.conj().T @ self.beta2_X @ Hy @ self.beta2_X @ self.Uxyx
        D2PhiYYH = self.beta2_X @ self.Uxyx @ (work * self.D1xyx_dg) @ self.Uxyx.conj().T @ self.beta2_X

        # ======================================================================
        # Hessian products with respect to t
        # ======================================================================
        # D2_t F(t, X, Y)[Ht, Hx, Hy]
        #         = (Ht - D_X trPg(X, Y)[Hx] - D_Y trPg(X, Y)[Hy]) / z^2
        out_t = Ht - inp(self.DPhiX, Hx) - inp(self.DPhiY, Hy)
        out_t *= self.zi2
        out[0][:] = out_t

        # ======================================================================
        # Hessian products with respect to X
        # ======================================================================
        # D2_X F(t, X, Y)[Ht, Hx, Hy]
        #         = -D2_t F(t, X, Y)[Ht, Hx, Hy] * D_X trPg(X, Y)
        #           + (D2_XX trPg(X, Y)[Hx] + D2_XY trPg(X, Y)[Hy]) / z
        #           + X^-1 Hx X^-1
        out_X = -out_t * self.DPhiX
        out_X += self.zi * (D2PhiXYH + D2PhiXXH)
        out_X += self.inv_X @ Hx @ self.inv_X
        out_X = (out_X + out_X.conj().T) * 0.5
        out[1][:] = out_X

        # ==================================================================
        # Hessian products with respect to Y
        # ==================================================================
        # Hessian product of barrier function
        # D2_Y F(t, X, Y)[Ht, Hx, Hy]
        #         = -D2_t F(t, X, Y)[Ht, Hx, Hy] * D_Y trPg(X, Y)
        #           + (D2_YX trPg(X, Y)[Hx] + D2_YY trPg(X, Y)[Hy]) / z
        #           + Y^-1 Hy Y^-1
        out_Y = -out_t * self.DPhiY
        out_Y += self.zi * (D2PhiYXH + D2PhiYYH)
        out_Y += self.inv_Y @ Hy @ self.inv_Y
        out_Y = (out_Y + out_Y.conj().T) * 0.5
        out[2][:] = out_Y

        return out

    def hess_congr(self, A):
        pass

    def invhess_prod_ip(self, out, H):
        pass

    def invhess_congr(self, A):
        pass

    def third_dir_deriv_axpy(self, out, H, a=True):
        pass

    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        # Preparing other required variables
        self.zi2 = self.zi * self.zi

        self.hess_aux_updated = True