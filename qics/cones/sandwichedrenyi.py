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

        # Get sparse operator to convert from full to compact vectorizations
        self.F2C_op = get_full_to_compact_op(n, iscomplex)

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
        irt4_Y = self.Uy / np.sqrt(rt2_Dy)
        self.rt2_Y = rt4_Y @ rt4_Y.conj().T
        self.irt2_Y = irt4_Y @ irt4_Y.conj().T

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
        # D2_t F(t, X, Y)[Ht, Hx, Hy]
        #         = (Ht - D_X trPg(X, Y)[Hx] - D_Y trPg(X, Y)[Hy]) / z^2
        DPhiX_vec = self.DPhiX.view(np.float64).reshape((-1, 1))
        DPhiY_vec = self.DPhiY.view(np.float64).reshape((-1, 1))

        out_t = self.At - (self.Ax_vec @ DPhiX_vec).ravel()
        out_t -= (self.Ay_vec @ DPhiY_vec).ravel()
        out_t *= self.zi2

        lhs[:, 0] = out_t

        # ======================================================================
        # Hessian products with respect to X
        # ======================================================================
        # Hessian products of trace operator perspective

        # Hxx
        # work = self.D1x_h * (self.Ux.conj().T @ Hx @ self.Ux)
        # work = self.Uyxy.conj().T @ self.rt2_Y @ self.Ux @ work @ self.Ux.conj().T @ self.rt2_Y @ self.Uyxy
        # work = self.Ux.conj().T @ self.rt2_Y @ self.Uyxy @ (self.D1yxy_dg * work) @ self.Uyxy.conj().T @ self.rt2_Y @ self.Ux
        # D2PhiXXH = self.Ux @ (self.D1x_h * work) @ self.Ux.conj().T
        congr_multi(work2, self.Ux.conj().T, self.Ax, work=work4)
        np.multiply(work2, self.D1x_h, out=work1)
        congr_multi(work5, self.Uyxy.conj().T @ self.rt2_Y @ self.Ux, work1, work=work4)
        work5 *= self.D1yxy_dg
        congr_multi(work1, self.Ux.conj().T @ self.rt2_Y @ self.Uyxy, work5, work=work4)
        work1 *= self.D1x_h
        congr_multi(work5, self.Ux, work1, work=work4)
        # work = self.Ux.conj().T @ self.rt2_Y @ self.dg_YXY @ self.rt2_Y @ self.Ux
        # D2PhiXXH += scnd_frechet(self.D2x_h, work, self.Ux.conj().T @ Hx @ self.Ux, U=self.Ux)
        work = self.Ux.conj().T @ self.rt2_Y @ self.dg_YXY @ self.rt2_Y @ self.Ux
        scnd_frechet_multi(work1, self.D2x_h, work2, work, U=self.Ux,
                                work1=work3, work2=work4, work3=work6)  # fmt: skip
        work5 += work1

        # Hxy
        # work = self.D1xyx_g * (self.Uxyx.conj().T @ self.beta2_X @ Hy @ self.beta2_X @ self.Uxyx)
        # work = self.Ux.conj().T @ self.ibeta2_X @ self.Uxyx @ work @ self.Uxyx.conj().T @ self.ibeta2_X @ self.Ux
        # D2PhiXYH = self.alpha * self.Ux @ (work * self.D1x_h) @ self.Ux.conj().T
        congr_multi(work3, self.Uxyx.conj().T @ self.beta2_X, self.Ay, work=work4)
        np.multiply(work3, self.D1xyx_g, out=work0)
        congr_multi(work1, self.Ux.conj().T @ self.ibeta2_X @ self.Uxyx, work0, work=work4)
        work1 *= self.alpha * self.D1x_h
        congr_multi(work0, self.Ux, work1, work=work4)
        work5 += work0

        # Hessian product of barrier function
        # D2_X F(t, X, Y)[Ht, Hx, Hy]
        #         = -D2_t F(t, X, Y)[Ht, Hx, Hy] * D_X trPg(X, Y)
        #           + (D2_XX trPg(X, Y)[Hx] + D2_XY trPg(X, Y)[Hy]) / z
        #           + X^-1 Hx X^-1
        work5 *= self.zi
        np.outer(out_t, self.DPhiX, out=work1.reshape((p, -1)))
        work5 -= work1
        congr_multi(work1, self.inv_X, self.Ax, work=work4)
        work5 += work1

        lhs[:, self.idx_X] = work5.reshape((p, -1)).view(np.float64)

        # ==================================================================
        # Hessian products with respect to Y
        # ==================================================================
        # Hessian products of trace operator perspective

        # D2_YX D(X, Y)[Hx]
        # work = self.D1x_h * (self.Ux.conj().T @ Hx @ self.Ux)
        # work = self.Uxyx.conj().T @ self.ibeta2_X @ self.Ux @ work @ self.Ux.conj().T @ self.ibeta2_X @ self.Uxyx
        # D2PhiYXH = self.alpha * self.beta2_X @ self.Uxyx @ (self.D1xyx_g * work) @ self.Uxyx.conj().T @ self.beta2_X
        work2 *= self.D1x_h
        congr_multi(work0, self.Uxyx.conj().T @ self.ibeta2_X @ self.Ux, work2, work=work4)
        work0 *= self.alpha * self.D1xyx_g
        congr_multi(work5, self.beta2_X @ self.Uxyx, work0, work=work4)

        # D2_YY D(X, Y)[Hy]
        # work = slef.Uxyx.conj().T @ self.beta2_X @ Hy @ self.beta2_X @ self.Uxyx
        # D2PhiYYH = self.beta2_X @ self.Uxyx @ (work * self.D1xyx_dg) @ self.Uxyx.conj().T @ self.beta2_X
        work3 *= self.D1xyx_dg
        congr_multi(work1, self.beta2_X @ self.Uxyx, work3, work=work4)
        work5 += work1

        # Hessian product of barrier function
        # D2_Y F(t, X, Y)[Ht, Hx, Hy]
        #         = -D2_t F(t, X, Y)[Ht, Hx, Hy] * D_Y trPg(X, Y)
        #           + (D2_YX trPg(X, Y)[Hx] + D2_YY trPg(X, Y)[Hy]) / z
        #           + Y^-1 Hy Y^-1
        work5 *= self.zi
        np.outer(out_t, self.DPhiY, out=work1.reshape((p, -1)))
        work5 -= work1
        congr_multi(work1, self.inv_Y, self.Ay, work=work3)
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

        # The inverse Hessian product applied on (Ht, Hx, Hy) for the OPT
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
        if not self.dder3_aux_updated:
            self.update_dder3_aux()

        (Ht, Hx, Hy) = H


        chi = Ht[0, 0] - inp(self.DPhiX, Hx) - inp(self.DPhiY, Hy)
        chi2 = chi * chi


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

        D2PhiXHH = inp(Hx, D2PhiXXH + D2PhiXYH)
        D2PhiYHH = inp(Hy, D2PhiYXH + D2PhiYYH)

        # Trace noncommutative perspective third order derivatives
        self.D2xyx_dg = D2_f(self.Dxyx, self.D1xyx_dg, self.d3g(self.Dxyx))
        self.D2xyx_g = D2_f(self.Dxyx, self.D1xyx_g, self.d2g(self.Dxyx))

        self.D1yxy_g = D1_f(self.Dyxy, self.g(self.Dyxy), self.dg(self.Dyxy))
        self.D2yxy_g = D2_f(self.Dyxy, self.D1yxy_g, self.d2g(self.Dyxy))
        self.D2yxy_dg = D2_f(self.Dyxy, self.D1yxy_dg, self.d3g(self.Dyxy))

        # Second derivatives of D_X trPg(X, Y)
        work = self.Ux.conj().T @ self.rt2_Y @ self.dg_YXY @ self.rt2_Y @ self.Ux
        work2 = self.Ux.conj().T @ Hx @ self.Ux
        D3PhiXXX = thrd_frechet(self.Dx, self.D2x_h, self.d3h(self.Dx), self.Ux, work, work2)

        work = self.rt2_Y @ self.Ux @ (self.D1x_h * (self.Ux.conj().T @ Hx @ self.Ux)) @ self.Ux.conj().T @ self.rt2_Y
        work = self.Uyxy @ (self.D1yxy_dg * (self.Uyxy.conj().T @ work @ self.Uyxy)) @ self.Uyxy.conj().T
        work = self.Ux.conj().T @ self.rt2_Y @ work @ self.rt2_Y @ self.Ux
        work2 = self.Ux.conj().T @ Hx @ self.Ux
        D3PhiXXX += 2 * scnd_frechet(self.D2x_h, work, work2, U=self.Ux)

        work = self.Ux.conj().T @ Hx @ self.Ux
        work = scnd_frechet(self.D2x_h, work, work, U=self.Uyxy.conj().T @ self.rt2_Y @ self.Ux)
        work = self.rt2_Y @ self.Uyxy @ (self.D1yxy_dg * work) @ self.Uyxy.conj().T @ self.rt2_Y
        D3PhiXXX += self.Ux @ (self.D1x_h * (self.Ux.conj().T @ work @ self.Ux)) @ self.Ux.conj().T

        work = self.rt2_Y @ self.Ux @ (self.D1x_h * (self.Ux.conj().T @ Hx @ self.Ux)) @ self.Ux.conj().T @ self.rt2_Y
        work = self.Uyxy.conj().T @ work @ self.Uyxy
        work = scnd_frechet(self.D2yxy_dg, work, work, U=self.Ux.conj().T @ self.rt2_Y @ self.Uyxy)
        D3PhiXXX += self.Ux @ (self.D1x_h * work) @ self.Ux.conj().T

        ##

        work = self.Ux @ (self.D1x_h * (self.Ux.conj().T @ Hx @ self.Ux)) @ self.Ux.conj().T
        work = self.Uyxy.conj().T @ self.rt2_Y @ work @ self.rt2_Y @ self.Uyxy
        work2 = self.Uyxy.conj().T @ self.irt2_Y @ Hy @ self.irt2_Y @ self.Uyxy
        work = scnd_frechet(self.D2yxy_g, work, work2, U=self.Ux.conj().T @ self.rt2_Y @ self.Uyxy)
        D3PhiXXY = self.Ux @ (self.D1x_h * work) @ self.Ux.conj().T
        work = self.Ux.conj().T @ self.rt2_Y @ self.Uyxy @ (self.D1yxy_g * work2) @ self.Uyxy.conj().T @ self.rt2_Y @ self.Ux
        work2 = self.Ux.conj().T @ Hx @ self.Ux
        D3PhiXXY += scnd_frechet(self.D2x_h, work, work2, U=self.Ux)
        D3PhiXXY *= self.alpha

        D3PhiXYX = D3PhiXXY

        work = self.Uxyx.conj().T @ self.beta2_X @ Hy @ self.beta2_X @ self.Uxyx
        work = scnd_frechet(self.D2xyx_g, work, work, U=self.Ux.conj().T @ self.ibeta2_X @ self.Uxyx)
        D3PhiXYY = self.alpha * self.Ux @ (self.D1x_h * work) @ self.Ux.conj().T

        # Second derivatives of D_Y trPg(X, Y)
        work = self.Uxyx.conj().T @ self.beta2_X @ Hy @ self.beta2_X @ self.Uxyx
        D3PhiYYY = scnd_frechet(self.D2xyx_dg, work, work, U=self.beta2_X @ self.Uxyx)

        work2 = self.D1x_h * (self.Ux.conj().T @ Hx @ self.Ux)
        work2 = self.Uxyx.conj().T @ self.ibeta2_X @ self.Ux @ work2 @ self.Ux.conj().T @ self.ibeta2_X @ self.Uxyx
        D3PhiYYX = self.alpha * scnd_frechet(self.D2xyx_g, work, work2, U=self.beta2_X @ self.Uxyx)
        
        D3PhiYXY = D3PhiYYX

        work = self.D1x_h * (self.Ux.conj().T @ Hx @ self.Ux)
        work = self.Uyxy.conj().T @ self.rt2_Y @ self.Ux @ work @ self.Ux.conj().T @ self.rt2_Y @ self.Uyxy
        D3PhiYXX = scnd_frechet(self.D2yxy_g, work, work, U=self.irt2_Y @ self.Uyxy)
        work = self.Ux.conj().T @ Hx @ self.Ux
        work = scnd_frechet(self.D2x_h, work, work, U=self.Uyxy.conj().T @ self.rt2_Y @ self.Ux)
        D3PhiYXX += self.irt2_Y @ self.Uyxy @ (self.D1yxy_g * work) @ self.Uyxy.conj().T @ self.irt2_Y
        D3PhiYXX *= self.alpha

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

        self.D1x_h = D1_f(self.Dx, self.h(self.Dx), self.dh(self.Dx))
        self.D2x_h = D2_f(self.Dx, self.D1x_h, self.d2h(self.Dx))

        self.D1xyx_g = D1_f(self.Dxyx, self.g(self.Dxyx), self.dg(self.Dxyx))
        self.D1yxy_dg = D1_f(self.Dyxy, self.dg(self.Dyxy), self.d2g(self.Dyxy))
        self.D1xyx_dg = D1_f(self.Dxyx, self.dg(self.Dxyx), self.d2g(self.Dxyx))

        # Preparing other required variables
        self.zi2 = self.zi * self.zi

        self.hess_aux_updated = True

    def update_dder3_aux(self):
        assert not self.dder3_aux_updated
        assert self.hess_aux_updated

        self.zi3 = self.zi2 * self.zi

        self.dder3_aux_updated = True

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

        work10, work11, work12 = self.work10, self.work11, self.work12
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
        # work = self.D1x_h * (self.Ux.conj().T @ Hx @ self.Ux)
        # work = self.Uyxy.conj().T @ self.rt2_Y @ self.Ux @ work @ self.Ux.conj().T @ self.rt2_Y @ self.Uyxy
        # work = self.Ux.conj().T @ self.rt2_Y @ self.Uyxy @ (self.D1yxy_dg * work) @ self.Uyxy.conj().T @ self.rt2_Y @ self.Ux
        # D2PhiXXH = self.Ux @ (self.D1x_h * work) @ self.Ux.conj().T
        congr_multi(work11, self.Ux.conj().T, self.E, work=work13)
        np.multiply(work11, self.D1x_h, out=work14)
        congr_multi(work12, self.Uyxy.conj().T @ self.rt2_Y @ self.Ux, work14, work=work13)
        work12 *= self.D1yxy_dg
        congr_multi(work14, self.Ux.conj().T @ self.rt2_Y @ self.Uyxy, work12, work=work13)
        work14 *= self.D1x_h
        congr_multi(work10, self.Ux, work14, work=work13)
        # work = self.Ux.conj().T @ self.rt2_Y @ self.dg_YXY @ self.rt2_Y @ self.Ux
        # D2PhiXXH += scnd_frechet(self.D2x_h, work, self.Ux.conj().T @ Hx @ self.Ux, U=self.Ux)
        work = self.Ux.conj().T @ self.rt2_Y @ self.dg_YXY @ self.rt2_Y @ self.Ux
        scnd_frechet_multi(work14, self.D2x_h, work11, work, U=self.Ux,
                                work1=work12, work2=work13, work3=work15)  # fmt: skip
        work10 += work14
        work10 *= self.zi
        # X^1 Eij X^-1
        congr_multi(work14, self.inv_X, self.E, work=work13)
        work14 += work10
        # Vectorize matrices as compact vectors to get square matrix
        work = work14.view(np.float64).reshape((self.vn, -1))
        Hxx = x_dot_dense(self.F2C_op, work.T)

        # ======================================================================
        # Construct YY block of Hessian, i.e., (D2yyPhi + Y^-1 ⊗ Y^-1)
        # ======================================================================
        # work = self.Uxyx.conj().T @ self.beta2_X @ Hy @ self.beta2_X @ self.Uxyx
        # D2PhiYYH = self.beta2_X @ self.Uxyx @ (work * self.D1xyx_dg) @ self.Uxyx.conj().T @ self.beta2_X
        congr_multi(work14, self.Uxyx.conj().T @ self.beta2_X, self.E, work=work13)
        np.multiply(work14, self.D1xyx_dg, out=work10)
        congr_multi(work11, self.beta2_X @ self.Uxyx, work10, work=work13)
        work11 *= self.zi
        # Y^-1 Eij Y^-1
        congr_multi(work12, self.inv_Y, self.E, work=work13)
        work12 += work11
        # Vectorize matrices as compact vectors to get square matrix
        work = work12.view(np.float64).reshape((self.vn, -1))
        Hyy = x_dot_dense(self.F2C_op, work.T)

        # ======================================================================
        # Construct XY block of Hessian, i.e., D2yxPhi
        # ======================================================================
        # work = self.D1xyx_g * (self.Uxyx.conj().T @ self.beta2_X @ Hy @ self.beta2_X @ self.Uxyx)
        # work = self.Ux.conj().T @ self.ibeta2_X @ self.Uxyx @ work @ self.Uxyx.conj().T @ self.ibeta2_X @ self.Ux
        # D2PhiXYH = self.alpha * self.Ux @ (work * self.D1x_h) @ self.Ux.conj().T
        work14 *= self.D1xyx_g
        congr_multi(work12, self.Ux.conj().T @ self.ibeta2_X @ self.Uxyx, work14, work=work13)
        work12 *= self.alpha * self.D1x_h
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

        self.work10 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work11 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work12 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work13 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work14 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work15 = np.empty((self.n, self.n, self.vn), dtype=self.dtype)

        self.invhess_aux_aux_updated = True
