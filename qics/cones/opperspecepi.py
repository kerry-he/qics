# Copyright (c) 2024, Kerry He, James Saunderson, and Hamza Fawzi

# This Python package QICS is licensed under the MIT license; see LICENSE.md 
# file in the root directory or at https://github.com/kerry-he/qics

import numpy as np
import scipy as sp
import qics._utils.linalg as lin
import qics._utils.gradient as grad
from qics.cones.base import Cone, get_perspective_derivatives
from qics.vectorize import get_full_to_compact_op


class OpPerspecEpi(Cone):
    r"""A class representing a operator perspective epigraph cone

    .. math::

        \mathcal{OPE}_{n,g} = \text{cl}\{ (T, X, Y) \in \mathbb{H}^n \times
        \mathbb{H}^n_{++}\times\mathbb{H}^n_{++} : T \succeq P_g(X, Y) \},

    for an operator concave function
    :math:`g:(0,\infty)\rightarrow\mathbb{R}`, where

    .. math::

        P_g(X, Y) = X^{1/2} g(X^{-1/2} Y X^{-1/2}) X^{1/2},

    is the operator perspective of :math:`g`.

    Parameters
    ----------
    n : :obj:`int`
        Dimension of the matrices :math:`T`, :math:`X`, and :math:`Y`.
    func : :obj:`string` or :obj:`float`
        Choice for the function :math:`g`. Can be defined in the following
        ways.

        - :math:`g(x) = -\log(x)` if ``func="log"``
        - :math:`g(x) = -x^p` if ``func=p`` is a :obj:`float` where 
          :math:`p\in(0, 1)`
        - :math:`g(x) = x^p` if ``func=p`` is a :obj:`float` where 
          :math:`p\in[-1, 0)\cup(1, 2)`

    iscomplex : :obj:`bool`
        Whether the matrix :math:`T`, :math:`X`, and :math:`Y` is defined
        over :math:`\mathbb{H}^n` (``True``), or restricted to 
        :math:`\mathbb{S}^n` (``False``). The default is ``False``.

    See also
    --------
    OpPerspecTr : Trace operator perspective cone

    Notes
    -----
    We do not support operator perspectives for ``p=0``, ``p=1``, and
    ``p=2`` as these functions are more efficiently modelled using 
    just the positive semidefinite cone. 

    - When :math:`g(x)=x^0`, :math:`P_g(X, Y)=X`.
    - When :math:`g(x)=x^1`, :math:`P_g(X, Y)=Y`.
    - When :math:`g(x)=x^2`, :math:`P_g(X, Y)=YX^{-1}Y`, which can be 
      modelled using the Schur complement lemma, i.e., if :math:`X\succ 0`,
      then

      .. math::

          \begin{bmatrix} X & Y \\ Y & T \end{bmatrix} \succeq 0 \qquad 
          \Longleftrightarrow \qquad T \succeq YX^{-1}Y.
    """

    def __init__(self, n, func, iscomplex=False):
        self.n = n
        self.func = func
        self.iscomplex = iscomplex

        self.nu = 3 * self.n  # Barrier parameter

        if iscomplex:
            self.vn = n * n
            self.dim = [2 * n * n, 2 * n * n, 2 * n * n]
            self.type = ["h", "h", "h"]
            self.dtype = np.complex128
        else:
            self.vn = n * (n + 1) // 2
            self.dim = [n * n, n * n, n * n]
            self.type = ["s", "s", "s"]
            self.dtype = np.float64

        self.idx_T = slice(0, self.dim[0])
        self.idx_X = slice(self.dim[0], 2 * self.dim[0])
        self.idx_Y = slice(2 * self.dim[0], 3 * self.dim[0])

        # Get function handles for g(x), h(x)=x*g(1/x), x*g(x), and x*h(x)
        # and their first, second and third derivatives
        perspective_derivatives = get_perspective_derivatives(func)
        self.g, self.dg, self.d2g, self.d3g = perspective_derivatives["g"]
        self.h, self.dh, self.d2h, self.d3h = perspective_derivatives["h"]
        self.xg, self.dxg, self.d2xg, self.d3xg = perspective_derivatives["xg"]
        self.xh, self.dxh, self.d2xh, self.d3xh = perspective_derivatives["xh"]

        # Get LAPACK operators
        X = np.eye(n, dtype=self.dtype)
        self.cho_fact = sp.linalg.lapack.get_lapack_funcs("potrf", (X,))
        self.cho_inv = sp.linalg.lapack.get_lapack_funcs("trtri", (X,))

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
            np.eye(self.n, dtype=self.dtype) * t0,
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

        (self.T, self.X, self.Y) = self.primal

        # Check that X and Y are positive definite
        self.Dx, self.Ux = np.linalg.eigh(self.X)
        self.Dy, self.Uy = np.linalg.eigh(self.Y)

        if any(self.Dx <= 0) or any(self.Dy <= 0):
            self.feas = False
            return self.feas

        # Construct (X^-1/2 Y X^-1/2) and (Y^-1/2 X Y^-1/2)
        # and double check they are also PSD (in case of numerical errors)
        rt2_Dx = np.sqrt(self.Dx)
        rt4_X = self.Ux * np.sqrt(rt2_Dx)
        irt4_X = self.Ux / np.sqrt(rt2_Dx)
        self.rt2_X = rt4_X @ rt4_X.conj().T
        self.irt2_X = irt4_X @ irt4_X.conj().T

        rt2_Dy = np.sqrt(self.Dy)
        rt4_Y = self.Uy * np.sqrt(rt2_Dy)
        irt4_Y = self.Uy / np.sqrt(rt2_Dy)
        self.rt2_Y = rt4_Y @ rt4_Y.conj().T
        self.irt2_Y = irt4_Y @ irt4_Y.conj().T

        XYX = self.irt2_X @ self.Y @ self.irt2_X
        YXY = self.irt2_Y @ self.X @ self.irt2_Y

        self.Dxyx, self.Uxyx = np.linalg.eigh(XYX)
        self.Dyxy, self.Uyxy = np.linalg.eigh(YXY)

        if any(self.Dxyx <= 0) or any(self.Dyxy <= 0):
            self.feas = False
            return self.feas

        # Check that T ≻ Pg(X, Y)
        self.g_Dxyx = self.g(self.Dxyx)
        self.h_Dyxy = self.h(self.Dyxy)
        g_XYX = (self.Uxyx * self.g_Dxyx) @ self.Uxyx.conj().T
        self.Z = self.T - self.rt2_X @ g_XYX @ self.rt2_X

        # Try to perform Cholesky factorization to check PSD
        self.Z_chol, info = self.cho_fact(self.Z, lower=True)
        self.feas = info == 0

        return self.feas

    def get_val(self):
        assert self.feas_updated
        (sgn, abslogdet_Z) = np.linalg.slogdet(self.Z)
        logdet_Z = sgn * abslogdet_Z
        return -logdet_Z - np.sum(np.log(self.Dx)) - np.sum(np.log(self.Dy))

    def update_grad(self):
        assert self.feas_updated
        assert not self.grad_updated

        # Compute X^-1, Y^-1, and Z^-1
        inv_Dx = np.reciprocal(self.Dx)
        inv_X_rt2 = self.Ux * np.sqrt(inv_Dx)
        self.inv_X = inv_X_rt2 @ inv_X_rt2.conj().T

        inv_Dy = np.reciprocal(self.Dy)
        inv_Y_rt2 = self.Uy * np.sqrt(inv_Dy)
        self.inv_Y = inv_Y_rt2 @ inv_Y_rt2.conj().T

        self.Z_chol_inv, _ = self.cho_inv(self.Z_chol, lower=True)
        self.inv_Z = self.Z_chol_inv.conj().T @ self.Z_chol_inv

        # Precompute some useful expressions
        self.irt2Y_Uyxy = self.irt2_Y @ self.Uyxy
        self.irt2X_Uxyx = self.irt2_X @ self.Uxyx

        self.rt2Y_Uyxy = self.rt2_Y @ self.Uyxy
        self.rt2X_Uxyx = self.rt2_X @ self.Uxyx

        self.UyxyYZYUyxy = self.rt2Y_Uyxy.conj().T @ self.inv_Z @ self.rt2Y_Uyxy
        self.UxyxXZXUxyx = self.rt2X_Uxyx.conj().T @ self.inv_Z @ self.rt2X_Uxyx

        # Compute adjoint derivatives of operator perspective
        self.D1yxy_h = grad.D1_f(self.Dyxy, self.h_Dyxy, self.dh(self.Dyxy))
        if self.func == "log":
            self.D1xyx_g = -grad.D1_log(self.Dxyx, -self.g_Dxyx)
        else:
            self.D1xyx_g = grad.D1_f(self.Dxyx, self.g_Dxyx, self.dg(self.Dxyx))
        # D_X Pg(X, Y)*[Z^-1] = Y^-½ Dh(Y^-½ X Y^-½)[Y^½ Z^-1 Y^½] Y^-½
        work = self.D1yxy_h * self.UyxyYZYUyxy
        work = self.irt2Y_Uyxy @ work @ self.irt2Y_Uyxy.conj().T
        DPhiX = (work + work.conj().T) * 0.5
        # D_Y Pg(X, Y)*[Z^-1] = X^-½ Dg(X^-½ Y X^-½)[X^½ Z^-1 X^½] X^-½
        work = self.D1xyx_g * self.UxyxXZXUxyx
        work = self.irt2X_Uxyx @ work @ self.irt2X_Uxyx.conj().T
        DPhiY = (work + work.conj().T) * 0.5

        # Compute gradient of barrier function
        self.grad = [-self.inv_Z, DPhiX - self.inv_X, DPhiY - self.inv_Y]

        self.grad_updated = True

    def hess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        (Ht, Hx, Hy) = H

        D2yxy_h, D2yxy_xh = self.D2yxy_h, self.D2yxy_xh
        D2xyx_g, D2xyx_xg = self.D2xyx_g, self.D2xyx_xg
        UyxyYZYUyxy, UxyxXZXUxyx = self.UyxyYZYUyxy, self.UxyxXZXUxyx
        rt2Y_Uyxy, irt2Y_Uyxy = self.rt2Y_Uyxy, self.irt2Y_Uyxy
        rt2X_Uxyx, irt2X_Uxyx = self.rt2X_Uxyx, self.irt2X_Uxyx

        # ======================================================================
        # Hessian products with respect to t
        # ======================================================================
        # D2_T F(T, X, Y)[Ht, Hx, Hy] 
        #     = Z^-1 (Ht - D_X Pg(X, Y)[Hx] - D_Y Pg(X, Y)[Hy]) Z^-1 := Ξ
        # where
        #   D_X Pg(X, Y)[Hx] = Y^½ Dh(Y^-½ X Y^-½)[Y^-½ Hx Y^-½] Y^½
        #   D_Y Pg(X, Y)[Hy] = X^½ Dg(X^-½ Y X^-½)[X^-½ Hy X^-½] X^½

        # D_X Pg(X, Y)[Hx] 
        work = self.D1yxy_h * (irt2Y_Uyxy.conj().T @ Hx @ irt2Y_Uyxy)
        DxPhiHx = rt2Y_Uyxy @ work @ rt2Y_Uyxy.conj().T
        # D_Y Pg(X, Y)[Hy] 
        work = self.D1xyx_g * (irt2X_Uxyx.conj().T @ Hy @ irt2X_Uxyx)
        DyPhiHy = rt2X_Uxyx @ work @ rt2X_Uxyx.conj().T

        # Hessian product of barrier function D2_T F(T, X, Y)[Ht, Hx, Hy]
        out_T = self.inv_Z @ (Ht - DxPhiHx - DyPhiHy) @ self.inv_Z
        out[0][:] = (out_T + out_T.conj().T) * 0.5

        # ======================================================================
        # Hessian products with respect to X
        # ======================================================================
        # D2_X F(T, X, Y)[Ht, Hx, Hy]
        #     = -D_X Pg(X, Y)*[Ξ] + D2_X Pg(X, Y)[Hx, Hy]*[Z^-1] + X^-1 Hx X^-1
        # where
        #   D_X Pg(X, Y)*[Ξ] = Y^-½ Dh(Y^-½ X Y^-½)[Y^½ Ξ Y^½] Y^-½
        #   D2_XX Pg(X, Y)[Hx]*[Z^-1] 
        #     = Y^-½ D2h(Y^-½ X Y^-½)[Y^½ Z^-1 Y^½, Y^-½ Hx Y^-½] Y^-½
        #   D2_XY Pg(X, Y)[Hy]*[Z^-1]
        #     = -Y^-½ D2xh(Y^-½ X Y^-½)[Y^½ Z^-1 Y^½, Y^-½ Hy Y^-½] Y^-½
        #       + Y^-½ Dh(Y^-½ X Y^-½)[Y^½ Z^-1 Hy Y^-½ + Y^-½ Hy Z^-1 Y^½] Y^-½

        # D_X Pg(X, Y)*[Ξ] and second part of D2_XY Pg(X, Y)[Hy]*[Z^-1]
        work = self.inv_Z @ Hy @ self.inv_Y
        work = rt2Y_Uyxy.conj().T @ (work + work.conj().T - out_T) @ rt2Y_Uyxy
        out_X = irt2Y_Uyxy @ (self.D1yxy_h * work) @ irt2Y_Uyxy.conj().T
        # D2_XX Pg(X, Y)[Hx]*[Z^-1]
        work = irt2Y_Uyxy.conj().T @ Hx @ irt2Y_Uyxy
        out_X += grad.scnd_frechet(D2yxy_h, UyxyYZYUyxy, work, U=irt2Y_Uyxy)
        # First part of D2_XY Pg(X, Y)[Hy]*[Z^-1]
        work = irt2Y_Uyxy.conj().T @ Hy @ irt2Y_Uyxy
        out_X -= grad.scnd_frechet(D2yxy_xh, UyxyYZYUyxy, work, U=irt2Y_Uyxy)
        # X^-1 Hx X^-1
        out_X += self.inv_X @ Hx @ self.inv_X

        # Hessian product of barrier function D2_X F(T, X, Y)[Ht, Hx, Hy]
        out[1][:] = (out_X + out_X.conj().T) * 0.5

        # ======================================================================
        # Hessian products with respect to Y
        # ======================================================================
        # D2_Y F(T, X, Y)[Ht, Hx, Hy]
        #     = -D_Y Pg(X, Y)*[Ξ] + D2_Y Pg(X, Y)[Hx, Hy]*[Z^-1] + Y^-1 Hy Y^-1
        # where
        #   D_Y Pg(X, Y)*[Ξ] = X^-½ Dg(X^-½ Y X^-½)[X^½ Ξ X^½] X^-½
        #   D2_YX Pg(X, Y)[Hx]*[Z^-1]
        #     = -X^-½ D2xg(X^-½ Y X^-½)[X^½ Z^-1 X^½, X^-½ Hx X^-½] X^-½
        #       + X^-½ Dg(X^-½ Y X^-½)[X^½ Z^-1 Hx X^-½ + X^-½ Hx Z^-1 X^½] X^-½
        #   D2_YY Pg(X, Y)[Hy]*[Z^-1] 
        #     = X^-½ D2g(X^-½ Y X^-½)[X^½ Z^-1 X^½, X^-½ Hy X^-½] X^-½
        
        # D_Y Pg(X, Y)*[Ξ] and second part of D2_YX Pg(X, Y)[Hx]*[Z^-1]
        work = self.inv_Z @ Hx @ self.inv_X
        work = rt2X_Uxyx.conj().T @ (work + work.conj().T - out_T) @ rt2X_Uxyx
        out_Y = irt2X_Uxyx @ (self.D1xyx_g * work) @ irt2X_Uxyx.conj().T
        # D2_YY Pg(X, Y)[Hy]*[Z^-1]
        work = irt2X_Uxyx.conj().T @ Hy @ irt2X_Uxyx
        out_Y += grad.scnd_frechet(D2xyx_g, UxyxXZXUxyx, work, U=irt2X_Uxyx)
        # First part of D2_YX Pg(X, Y)[Hx]*[Z^-1]
        work = irt2X_Uxyx.conj().T @ Hx @ irt2X_Uxyx
        out_Y -= grad.scnd_frechet(D2xyx_xg, UxyxXZXUxyx, work, U=irt2X_Uxyx)
        # Y^-1 Hy Y^-1
        out_Y += self.inv_Y @ Hy @ self.inv_Y

        # Hessian product of barrier function D2_Y F(T, X, Y)[Ht, Hx, Hy]
        out[2][:] = (out_Y + out_Y.conj().T) * 0.5

        return out

    def hess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)

        p = A.shape[0]
        lhs = np.empty((p, sum(self.dim)))

        D2yxy_h, D2yxy_xh = self.D2yxy_h, self.D2yxy_xh
        D2xyx_g, D2xyx_xg = self.D2xyx_g, self.D2xyx_xg
        UyxyYZYUyxy, UxyxXZXUxyx = self.UyxyYZYUyxy, self.UxyxXZXUxyx
        rt2Y_Uyxy, irt2Y_Uyxy = self.rt2Y_Uyxy, self.irt2Y_Uyxy
        rt2X_Uxyx, irt2X_Uxyx = self.rt2X_Uxyx, self.irt2X_Uxyx

        work0, work1 = self.work0, self.work1, 
        work2, work3 = self.work2, self.work3
        work4, work5, work6 = self.work4, self.work5, self.work6


        # ======================================================================
        # Hessian products with respect to t
        # ======================================================================
        # D2_T F(T, X, Y)[Ht, Hx, Hy] 
        #     = Z^-1 (Ht - D_X Pg(X, Y)[Hx] - D_Y Pg(X, Y)[Hy]) Z^-1 := Ξ
        # where
        #   D_X Pg(X, Y)[Hx] = Y^½ Dh(Y^-½ X Y^-½)[Y^-½ Hx Y^-½] Y^½
        #   D_Y Pg(X, Y)[Hy] = X^½ Dg(X^-½ Y X^-½)[X^-½ Hy X^-½] X^½

        # D_X Pg(X, Y)[Hx]
        lin.congr_multi(work2, irt2Y_Uyxy.conj().T, self.Ax, work=work3)
        work2 *= self.D1yxy_h
        lin.congr_multi(work1, rt2Y_Uyxy, work2, work=work3)
        # D_Y Pg(X, Y)[Hy]
        lin.congr_multi(work2, irt2X_Uxyx.conj().T, self.Ay, work=work3)
        work2 *= self.D1xyx_g
        lin.congr_multi(work0, rt2X_Uxyx, work2, work=work3)

        # Hessian product of barrier function D2_T F(T, X, Y)[Ht, Hx, Hy]
        work0 += work1
        np.subtract(self.At, work0, out=work2)
        lin.congr_multi(work0, self.inv_Z, work2, work=work3)

        lhs[:, self.idx_T] = work0.reshape((p, -1)).view(np.float64)

        # ======================================================================
        # Hessian products with respect to X
        # ======================================================================
        # D2_X F(T, X, Y)[Ht, Hx, Hy]
        #     = -D_X Pg(X, Y)*[Ξ] + D2_X Pg(X, Y)[Hx, Hy]*[Z^-1] + X^-1 Hx X^-1
        # where
        #   D_X Pg(X, Y)*[Ξ] = Y^-½ Dh(Y^-½ X Y^-½)[Y^½ Ξ Y^½] Y^-½
        #   D2_XX Pg(X, Y)[Hx]*[Z^-1] 
        #     = Y^-½ D2h(Y^-½ X Y^-½)[Y^½ Z^-1 Y^½, Y^-½ Hx Y^-½] Y^-½
        #   D2_XY Pg(X, Y)[Hy]*[Z^-1]
        #     = -Y^-½ D2xh(Y^-½ X Y^-½)[Y^½ Z^-1 Y^½, Y^-½ Hy Y^-½] Y^-½
        #       + Y^-½ Dh(Y^-½ X Y^-½)[Y^½ Z^-1 Hy Y^-½ + Y^-½ Hy Z^-1 Y^½] Y^-½

        # D_X Pg(X, Y)*[Ξ] and second part of D2_XY Pg(X, Y)[Hy]*[Z^-1]
        lin.congr_multi(work2, self.inv_Z, self.Ay, work=work3, B=self.inv_Y)
        np.add(work2, work2.conj().transpose(0, 2, 1), out=work1)
        work1 -= work0
        lin.congr_multi(work2, rt2Y_Uyxy.conj().T, work1, work=work3)
        work2 *= self.D1yxy_h
        lin.congr_multi(work1, irt2Y_Uyxy, work2, work=work3)
        # D2_XX Pg(X, Y)[Hx]*[Z^-1]
        lin.congr_multi(work2, irt2Y_Uyxy.conj().T, self.Ax, work=work3)
        grad.scnd_frechet_multi(work5, D2yxy_h, work2, UyxyYZYUyxy,
            U=irt2Y_Uyxy, work1=work3, work2=work4, work3=work6)
        work1 += work5
        # First part of D2_XY Pg(X, Y)[Hy]*[Z^-1]
        lin.congr_multi(work2, irt2Y_Uyxy.conj().T, self.Ay, work=work3)
        grad.scnd_frechet_multi(work5, D2yxy_xh, work2, UyxyYZYUyxy,
            U=irt2Y_Uyxy, work1=work3, work2=work4, work3=work6)
        work1 -= work5            
        # X^-1 Hx X^-1
        lin.congr_multi(work2, self.inv_X, self.Ax, work=work3)
        work1 += work2
        
        # Hessian product of barrier function D2_X F(T, X, Y)[Ht, Hx, Hy]
        lhs[:, self.idx_X] = work1.reshape((p, -1)).view(np.float64)

        # ======================================================================
        # Hessian products with respect to Y
        # ======================================================================
        # D2_Y F(T, X, Y)[Ht, Hx, Hy]
        #     = -D_Y Pg(X, Y)*[Ξ] + D2_Y Pg(X, Y)[Hx, Hy]*[Z^-1] + Y^-1 Hy Y^-1
        # where
        #   D_Y Pg(X, Y)*[Ξ] = X^-½ Dg(X^-½ Y X^-½)[X^½ Ξ X^½] X^-½
        #   D2_YX Pg(X, Y)[Hx]*[Z^-1]
        #     = -X^-½ D2xg(X^-½ Y X^-½)[X^½ Z^-1 X^½, X^-½ Hx X^-½] X^-½
        #       + X^-½ Dg(X^-½ Y X^-½)[X^½ Z^-1 Hx X^-½ + X^-½ Hx Z^-1 X^½] X^-½
        #   D2_YY Pg(X, Y)[Hy]*[Z^-1] 
        #     = X^-½ D2g(X^-½ Y X^-½)[X^½ Z^-1 X^½, X^-½ Hy X^-½] X^-½
        
        # D_Y Pg(X, Y)*[Ξ] and second part of D2_YX Pg(X, Y)[Hx]*[Z^-1]
        lin.congr_multi(work2, self.inv_Z, self.Ax, work=work3, B=self.inv_X)
        np.add(work2, work2.conj().transpose(0, 2, 1), out=work1)
        work1 -= work0
        lin.congr_multi(work2, rt2X_Uxyx.conj().T, work1, work=work3)
        work2 *= self.D1xyx_g
        lin.congr_multi(work1, irt2X_Uxyx, work2, work=work3)
        # D2_YY Pg(X, Y)[Hy]*[Z^-1]
        lin.congr_multi(work2, irt2X_Uxyx.conj().T, self.Ay, work=work3)
        grad.scnd_frechet_multi(work5, D2xyx_g, work2, UxyxXZXUxyx,
            U=irt2X_Uxyx, work1=work3, work2=work4, work3=work6)
        work1 += work5
        # First part of D2_YX Pg(X, Y)[Hx]*[Z^-1]
        lin.congr_multi(work2, irt2X_Uxyx.conj().T, self.Ax, work=work3)
        grad.scnd_frechet_multi(work5, D2xyx_xg, work2, UxyxXZXUxyx,
            U=irt2X_Uxyx, work1=work3, work2=work4, work3=work6)
        work1 -= work5   
        # X^-1 Hx X^-1
        lin.congr_multi(work2, self.inv_Y, self.Ay, work=work3)
        work1 += work2

        # Hessian product of barrier function D2_Y F(T, X, Y)[Ht, Hx, Hy]
        lhs[:, self.idx_Y] = work1.reshape((p, -1)).view(np.float64)
        
        # Multiply A (H A')
        return lin.dense_dot_x(lhs, A.T)

    def invhess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        (Ht, Hx, Hy) = H

        rt2Y_Uyxy, irt2Y_Uyxy = self.rt2Y_Uyxy, self.irt2Y_Uyxy
        rt2X_Uxyx, irt2X_Uxyx = self.rt2X_Uxyx, self.irt2X_Uxyx

        # ======================================================================
        # Inverse Hessian products with respect to X and Y
        # ======================================================================
        # Compute Wx = Hx + D_X Pg(X, Y)*[Ht]
        #            = Hx + Y^-½ Dh(Y^-½ X Y^-½)[Y^½ Ht Y^½] Y^-½
        work = rt2Y_Uyxy.conj().T @ Ht @ rt2Y_Uyxy
        Wx = Hx + irt2Y_Uyxy @ (self.D1yxy_h * work) @ irt2Y_Uyxy.conj().T
        Wx_vec = Wx.view(np.float64).reshape(-1, 1)
        Wx_vec = self.F2C_op @ Wx_vec

        # Compute Wy = Hy + D_Y Pg(X, Y)*[Ht]
        #            = Hy + X^-½ Dg(X^-½ Y X^-½)[X^½ Ht X^½] X^-½
        work = rt2X_Uxyx.conj().T @ Ht @ rt2X_Uxyx
        Wy = Hy + irt2X_Uxyx @ (self.D1xyx_g * work) @ irt2X_Uxyx.conj().T
        Wy_vec = Wy.view(np.float64).reshape(-1, 1)
        Wy_vec = self.F2C_op @ Wy_vec

        # Solve linear system (ΔX, ΔY) = M \ (Wx, Wy)
        Wxy_vec = np.vstack((Wx_vec, Wy_vec))
        out_Xy = lin.cho_solve(self.hess_fact, Wxy_vec)
        out_Xy = out_Xy.reshape(2, -1)

        # Recover ΔX as matrices from compact vectors
        out_X = self.F2C_op.T @ out_Xy[0]
        out_X = out_X.view(self.dtype).reshape((self.n, self.n))
        out[1][:] = (out_X + out_X.conj().T) * 0.5

        # Recover ΔY as matrices from compact vectors
        out_Y = self.F2C_op.T @ out_Xy[1]
        out_Y = out_Y.view(self.dtype).reshape((self.n, self.n))
        out[2][:] = (out_Y + out_Y.conj().T) * 0.5

        # ======================================================================
        # Inverse Hessian products with respect to Z
        # ======================================================================
        # Compute Z Ht Z
        outT = self.Z @ Ht @ self.Z
        # Compute D_X Pg(X, Y)[ΔX] = Y^½ Dh(Y^-½ X Y^-½)[Y^-½ ΔX Y^-½] Y^½
        work = irt2Y_Uyxy.conj().T @ out[1] @ irt2Y_Uyxy
        outT += rt2Y_Uyxy @ (self.D1yxy_h * work) @ rt2Y_Uyxy.conj().T
        # Compute D_Y Pg(X, Y)[ΔY] = X^½ Dg(X^-½ Y X^-½)[X^-½ ΔY X^-½] X^½
        work = irt2X_Uxyx.conj().T @ out[2] @ irt2X_Uxyx
        outT += rt2X_Uxyx @ (self.D1xyx_g * work) @ rt2X_Uxyx.conj().T

        # Δt = Z Ht Z + DPg(X, Y)[(ΔX, ΔY)]
        out[0][:] = (outT + outT.conj().T) * 0.5

        return out
    
    def invhess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)

        # The inverse Hessian product applied on (Ht, Hx, Hy) for the OPE
        # barrier is
        #     (X, Y) =  M \ (Wx, Wy)
        #         t  =  Z Ht Z + DPhi[(X, Y)]
        # where (Wx, Wy) = (Hx, Hy) + DPhi*[Ht],
        #     M = [ D2xxPhi*[Z^-1]  D2xyPhi*[Z^-1] ] + [ X^-1 ⊗ X^-1              ]
        #         [ D2yxPhi*[Z^-1]  D2yyPhi*[Z^-1] ]   [              Y^-1 ⊗ Y^-1 ]

        p = A.shape[0]

        rt2Y_Uyxy, irt2Y_Uyxy = self.rt2Y_Uyxy, self.irt2Y_Uyxy
        rt2X_Uxyx, irt2X_Uxyx = self.rt2X_Uxyx, self.irt2X_Uxyx

        work0, work1 = self.work0, self.work1, 
        work2, work3, work7 = self.work2, self.work3, self.work7

        # ======================================================================
        # Inverse Hessian products with respect to X and Y
        # ======================================================================
        # Compute Wx = Hx + D_X Pg(X, Y)*[Ht]
        #            = Hx + Y^-½ Dh(Y^-½ X Y^-½)[Y^½ Ht Y^½] Y^-½
        lin.congr_multi(work0, rt2Y_Uyxy.conj().T, self.At, work=work2)
        work0 *= self.D1yxy_h
        lin.congr_multi(work1, irt2Y_Uyxy, work0, work=work2)
        work1 += self.Ax

        # Compute Wy = Hy + D_Y Pg(X, Y)*[Ht]
        #            = Hy + X^-½ Dg(X^-½ Y X^-½)[X^½ Ht X^½] X^-½
        lin.congr_multi(work3, rt2X_Uxyx.conj().T, self.At, work=work2)
        work3 *= self.D1xyx_g
        lin.congr_multi(work0, irt2X_Uxyx, work3, work=work2)
        work0 += self.Ay

        # Solve linear system (ΔX, ΔY) = M \ (Wx, Wy)
        # Convert matrices to compact real vectors
        Wx_vec = work1.view(np.float64).reshape((p, -1))
        Wy_vec = work0.view(np.float64).reshape((p, -1))
        work7[:self.vn] = lin.x_dot_dense(self.F2C_op, Wx_vec.T)
        work7[self.vn:] = lin.x_dot_dense(self.F2C_op, Wy_vec.T)
        # Solve system
        sol = lin.cho_solve(self.hess_fact, work7)

        # Multiply Axy (H A')xy
        out = lin.dense_dot_x(sol.T, self.Axy_cvec.T).T

        # ======================================================================
        # Inverse Hessian products with respect to Z
        # ======================================================================
        # Δt = Z Ht Z + DPg(X, Y)[(ΔX, ΔY)]
        # Compute Z Ht Z
        lin.congr_multi(work0, self.Z, self.At, work=work3)
        
        # Recover ΔX as matrices from compact vectors
        work = lin.x_dot_dense(self.F2C_op.T, sol[:self.vn])
        work1.view(np.float64).reshape((p, -1))[:] = work.T
        # Compute D_X Pg(X, Y)[ΔX] = Y^½ Dh(Y^-½ X Y^-½)[Y^-½ ΔX Y^-½] Y^½
        lin.congr_multi(work2, irt2Y_Uyxy.conj().T, work1, work=work3)
        work2 *= self.D1yxy_h
        lin.congr_multi(work1, rt2Y_Uyxy, work2, work=work3)
        work0 += work1

        # Recover Y as matrices from compact vectors
        work = lin.x_dot_dense(self.F2C_op.T, sol[self.vn:])
        work1.view(np.float64).reshape((p, -1))[:] = work.T
        # Compute D_Y Pg(X, Y)[ΔY] = X^½ Dg(X^-½ Y X^-½)[X^-½ ΔY X^-½] X^½
        lin.congr_multi(work2, irt2X_Uxyx.conj().T, work1, work=work3)
        work2 *= self.D1xyx_g
        lin.congr_multi(work1, rt2X_Uxyx, work2, work=work3)
        work0 += work1

        # Multiply At (H A')t
        out_t = work0.view(np.float64).reshape((p, -1))
        out += lin.x_dot_dense(self.At_vec, out_t.T)

        return out

    def third_dir_deriv_axpy(self, out, H, a=True):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        (Ht, Hx, Hy) = H

        Dyxy, Dxyx, Uyxy, Uxyx = self.Dyxy, self.Dxyx, self.Uyxy, self.Uxyx
        D2yxy_h, D2yxy_xh = self.D2yxy_h, self.D2yxy_xh
        D2xyx_g, D2xyx_xg = self.D2xyx_g, self.D2xyx_xg
        UyxyYZYUyxy, UxyxXZXUxyx = self.UyxyYZYUyxy, self.UxyxXZXUxyx
        rt2Y_Uyxy, irt2Y_Uyxy = self.rt2Y_Uyxy, self.irt2Y_Uyxy
        rt2X_Uxyx, irt2X_Uxyx = self.rt2X_Uxyx, self.irt2X_Uxyx

        UyxyYHxYUyxy = irt2Y_Uyxy.conj().T @ Hx @ irt2Y_Uyxy
        UxyxXHyXUxyx = irt2X_Uxyx.conj().T @ Hy @ irt2X_Uxyx
        UxyxXHxXUxyx = irt2X_Uxyx.conj().T @ Hx @ irt2X_Uxyx
        UyxyYHyYUyxy = irt2Y_Uyxy.conj().T @ Hy @ irt2Y_Uyxy

        # Noncommutative perspective gradients
        DxPhiHx = rt2Y_Uyxy @ (self.D1yxy_h * UyxyYHxYUyxy) @ rt2Y_Uyxy.conj().T
        DyPhiHy = rt2X_Uxyx @ (self.D1xyx_g * UxyxXHyXUxyx) @ rt2X_Uxyx.conj().T

        Chi = Ht - DxPhiHx - DyPhiHy

        # Noncommutative perspective Hessians
        D2xxPhiHxHx = grad.scnd_frechet(D2yxy_h, UyxyYHxYUyxy, UyxyYHxYUyxy, 
                                        U=rt2Y_Uyxy)

        D2yyPhiHyHy = grad.scnd_frechet(D2xyx_g, UxyxXHyXUxyx, UxyxXHyXUxyx, 
                                        U=rt2X_Uxyx)

        work = self.D1xyx_g * UxyxXHyXUxyx
        work = Hx @ irt2X_Uxyx @ work @ rt2X_Uxyx.conj().T
        D2xyPhiHxHy = work + work.conj().T
        D2xyPhiHxHy -= grad.scnd_frechet(D2xyx_xg, UxyxXHxXUxyx, UxyxXHyXUxyx, 
                                         U=rt2X_Uxyx)

        # ======================================================================
        # Third order derivative with respect to T
        # ======================================================================
        work = Chi @ self.inv_Z @ Chi
        work = 2 * work + D2xxPhiHxHx + 2 * D2xyPhiHxHy + D2yyPhiHyHy
        dder3_T = -self.inv_Z @ work @ self.inv_Z

        out[0][:] += dder3_T * a

        # ======================================================================
        # Third order derivative with respect to X
        # ======================================================================
        # -D_X Pg(X, Y)*[D3_T F(T, X, Y)[Ht, Hx, Hy]]
        work = rt2Y_Uyxy.conj().T @ -dder3_T @ rt2Y_Uyxy
        dder3_X = irt2Y_Uyxy @ (self.D1yxy_h * work) @ irt2Y_Uyxy.conj().T

        # -2 * D2_XX Pg(X, Y)[Hx]*[Z^-1 Chi Z^-1]
        work2 = -2 * self.inv_Z @ Chi @ self.inv_Z
        work = rt2Y_Uyxy.conj().T @ work2 @ rt2Y_Uyxy
        dder3_X += grad.scnd_frechet(D2yxy_h, work, UyxyYHxYUyxy, U=irt2Y_Uyxy)

        # -2 * D2_XY Pg(X, Y)[Hy]*[Z^-1 Chi Z^-1]
        work = self.D1xyx_g * UxyxXHyXUxyx
        work = irt2X_Uxyx @ work @ rt2X_Uxyx.conj().T @ work2
        dder3_X += work + work.conj().T
        work = rt2X_Uxyx.conj().T @ work2 @ rt2X_Uxyx
        dder3_X -= grad.scnd_frechet(D2xyx_xg, work, UxyxXHyXUxyx, U=irt2X_Uxyx)

        # D3_XXX Pg(X, Y)[Hx, Hx]*[Z^-1]
        dder3_X += grad.thrd_frechet(Dyxy, D2yxy_h, self.d3h(Dyxy), irt2Y_Uyxy, 
                                     UyxyYZYUyxy, UyxyYHxYUyxy)

        # 2 * D3_XXY Pg(X, Y)[Hx, Hy]*[Z^-1]
        work = rt2Y_Uyxy.conj().T @ self.inv_Z @ Hy @ irt2Y_Uyxy
        work += work.conj().T
        dder3_X += 2 * grad.scnd_frechet(D2yxy_h, work, UyxyYHxYUyxy, 
                                         U=irt2Y_Uyxy)
        work = self.rt2Y_Uyxy.conj().T @ self.inv_Z @ self.rt2Y_Uyxy
        dder3_X -= 2 * grad.thrd_frechet(Dyxy, D2yxy_xh, self.d3xh(Dyxy),
            irt2Y_Uyxy, work, UyxyYHyYUyxy, UyxyYHxYUyxy)

        # D3_XYY Pg(X, Y)[Hy, Hy]*[Z^-1]
        work = grad.scnd_frechet(D2xyx_g, UxyxXHyXUxyx, UxyxXHyXUxyx, U=Uxyx)
        work =  self.irt2_X @ work @ self.rt2_X @ self.inv_Z
        dder3_X += work + work.conj().T
        dder3_X -= grad.thrd_frechet(Dxyx, D2xyx_xg, self.d3xg(Dxyx),
                                     irt2X_Uxyx, UxyxXZXUxyx, UxyxXHyXUxyx)

        # -2 * X^-1 Hx X^-1 Hx X^-1
        dder3_X -= 2 * self.inv_X @ Hx @ self.inv_X @ Hx @ self.inv_X

        out[1][:] += dder3_X * a

        # ======================================================================
        # Third order derivative with respect to Y
        # ======================================================================
        # -D_Y Pg(X, Y)*[D3_T F(T, X, Y)[Ht, Hx, Hy]]
        work = rt2X_Uxyx.conj().T @ -dder3_T @ rt2X_Uxyx
        dder3_Y = irt2X_Uxyx @ (self.D1xyx_g * work) @ irt2X_Uxyx.conj().T

        # -2 * D2_YY Pg(X, Y)[Hy]*[Z^-1 Chi Z^-1]
        work2 = -2 * self.inv_Z @ Chi @ self.inv_Z
        work = rt2X_Uxyx.conj().T @ work2 @ rt2X_Uxyx
        dder3_Y += grad.scnd_frechet(D2xyx_g, work, UxyxXHyXUxyx, U=irt2X_Uxyx)

        # -2 * D2_XY Pg(X, Y)[Hy]*[Z^-1 Chi Z^-1]
        work = self.D1yxy_h * UyxyYHxYUyxy
        work =  irt2Y_Uyxy @ work @ rt2Y_Uyxy.conj().T @ work2
        dder3_Y += work + work.conj().T
        work = rt2Y_Uyxy.conj().T @ work2 @ rt2Y_Uyxy
        dder3_Y -= grad.scnd_frechet(D2yxy_xh, work, UyxyYHxYUyxy, U=irt2Y_Uyxy)

        # D3_YYY Pg(X, Y)[Hy, Hy]*[Z^-1]
        dder3_Y += grad.thrd_frechet(Dxyx, D2xyx_g, self.d3g(Dxyx), irt2X_Uxyx,
                                     UxyxXZXUxyx, UxyxXHyXUxyx)

        # 2 * D3_YYX Pg(X, Y)[Hx, Hy]*[Z^-1]
        work = rt2X_Uxyx.conj().T @ self.inv_Z @ Hx @ irt2X_Uxyx
        work += work.conj().T
        dder3_Y += 2 * grad.scnd_frechet(D2xyx_g, work, UxyxXHyXUxyx, 
                                         U=irt2X_Uxyx)
        work = rt2X_Uxyx.conj().T @ self.inv_Z @ rt2X_Uxyx
        dder3_Y -= 2 * grad.thrd_frechet(Dxyx, D2xyx_xg, self.d3xg(Dxyx),
            irt2X_Uxyx, work, UxyxXHxXUxyx, UxyxXHyXUxyx)

        # D3_YXX Pg(X, Y)[Hx, Hx]*[Z^-1]
        work = grad.scnd_frechet(D2yxy_h, UyxyYHxYUyxy, UyxyYHxYUyxy, U=Uyxy)
        work = self.irt2_Y @ work @ self.rt2_Y @ self.inv_Z
        dder3_Y += work + work.conj().T
        dder3_Y -= grad.thrd_frechet(Dyxy, D2yxy_xh, self.d3xh(Dyxy),
                                      irt2Y_Uyxy, UyxyYZYUyxy, UyxyYHxYUyxy)

        # -2 * Y^-1 Hy Y^-1 Hy Y^-1
        dder3_Y -= 2 * self.inv_Y @ Hy @ self.inv_Y @ Hy @ self.inv_Y

        out[2][:] += dder3_Y * a

        return out

    # ==========================================================================
    # Auxilliary functions
    # ==========================================================================
    def congr_aux(self, A):
        assert not self.congr_aux_updated

        from qics.vectorize import vec_to_mat
        iscomplex = self.iscomplex

        # Get slices and views of A matrix to be used in congruence computations
        if sp.sparse.issparse(A):
            A = A.tocsr()        
        Ax_cvec = (self.F2C_op @ A[:, self.idx_X].T).T
        Ay_cvec = (self.F2C_op @ A[:, self.idx_Y].T).T
        if sp.sparse.issparse(A):
            self.At_vec = A[:, self.idx_T].tocoo()
            self.Axy_cvec = sp.sparse.hstack((Ax_cvec, Ay_cvec), format="coo")
        else:
            self.At_vec = np.ascontiguousarray(A[:, self.idx_T])
            self.Axy_cvec = np.hstack((Ax_cvec, Ay_cvec))

        if sp.sparse.issparse(A):
            A = A.toarray()
        At_dense = np.ascontiguousarray(A[:, self.idx_T])
        Ax_dense = np.ascontiguousarray(A[:, self.idx_X])
        Ay_dense = np.ascontiguousarray(A[:, self.idx_Y])
        self.At = np.array([vec_to_mat(At_k, iscomplex) for At_k in At_dense])
        self.Ax = np.array([vec_to_mat(Ax_k, iscomplex) for Ax_k in Ax_dense])
        self.Ay = np.array([vec_to_mat(Ay_k, iscomplex) for Ay_k in Ay_dense])

        # Preallocate matrices we will need when performing these congruences
        p = A.shape[0]
        self.work0 = np.empty_like(self.At)
        self.work1 = np.empty_like(self.At)
        self.work2 = np.empty_like(self.At)
        self.work3 = np.empty_like(self.At)
        self.work4 = np.empty_like(self.At)
        self.work5 = np.empty_like(self.At)
        self.work6 = np.empty((self.At.shape[::-1]), dtype=self.dtype)
        self.work7 = np.empty((2 * self.vn, p))

        self.congr_aux_updated = True

    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        Dyxy, Dxyx = self.Dyxy, self.Dxyx

        self.D1yxy_xh = grad.D1_f(Dyxy, self.xh(Dyxy), self.dxh(Dyxy))
        self.D1xyx_xg = grad.D1_f(Dxyx, self.xg(Dxyx), self.dxg(Dxyx))

        self.D2yxy_h = grad.D2_f(Dyxy, self.D1yxy_h, self.d2h(Dyxy))
        self.D2xyx_g = grad.D2_f(Dxyx, self.D1xyx_g, self.d2g(Dxyx))
        self.D2yxy_xh = grad.D2_f(Dyxy, self.D1yxy_xh, self.d2xh(Dyxy))
        self.D2xyx_xg = grad.D2_f(Dxyx, self.D1xyx_xg, self.d2xg(Dxyx))

        self.hess_aux_updated = True

        return

    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated
        if not self.invhess_aux_aux_updated:
            self.update_invhessprod_aux_aux()

        # Precompute and factorize the Schur complement matrix
        #     M = [ D2xxPhi*[Z^-1]  D2xyPhi*[Z^-1] ] + [ X^-1 ⊗ X^-1              ]
        #         [ D2yxPhi*[Z^-1]  D2yyPhi*[Z^-1] ]   [              Y^-1 ⊗ Y^-1 ]

        D2xyx_g, D2xyx_xg = self.D2xyx_g, self.D2xyx_xg
        D2yxy_h, irt2Y_Uyxy = self.D2yxy_h, self.irt2Y_Uyxy
        UyxyYZYUyxy, UxyxXZXUxyx = self.UyxyYZYUyxy, self.UxyxXZXUxyx
        rt2X_Uxyx, irt2X_Uxyx = self.rt2X_Uxyx, self.irt2X_Uxyx

        work10, work11 = self.work10, self.work11
        work12, work13, work14 = self.work12, self.work13, self.work14

        # ======================================================================
        # Construct XX block of Hessian, i.e., (D2xxPhi'[Z^-1] + X^1 ⊗ X^-1)
        # ======================================================================
        # D2_XX Pg(X, Y)[Eij]*[Z^-1] 
        #     = Y^-½ D2h(Y^-½ X Y^-½)[Y^½ Z^-1 Y^½, Y^-½ Eij Y^-½] Y^-½
        lin.congr_multi(work14, irt2Y_Uyxy.conj().T, self.E, work=work12)
        grad.scnd_frechet_multi(work11, D2yxy_h, work14, UyxyYZYUyxy,
            U=irt2Y_Uyxy, work1=work12, work2=work13, work3=work10)
        # X^1 Eij X^-1
        lin.congr_multi(work14, self.inv_X, self.E, work=work13)
        work14 += work11
        # Vectorize matrices as compact vectors to get square matrix
        work = work14.view(np.float64).reshape((self.vn, -1))
        Hxx = lin.x_dot_dense(self.F2C_op, work.T)

        # ======================================================================
        # Construct YY block of Hessian, i.e., (D2yyPhi'[Z^-1] + Y^1 ⊗ Y^-1)
        # ======================================================================
        # D2_YY Pg(X, Y)[Eij]*[Z^-1] 
        #     = X^-½ D2g(X^-½ Y X^-½)[X^½ Z^-1 X^½, X^-½ Eij X^-½] X^-½
        lin.congr_multi(work14, irt2X_Uxyx.conj().T, self.E, work=work13)
        grad.scnd_frechet_multi(work11, D2xyx_g, work14, UxyxXZXUxyx,
            U=irt2X_Uxyx, work1=work12, work2=work13, work3=work10)
        # Y^1 Eij Y^-1
        lin.congr_multi(work12, self.inv_Y, self.E, work=work13)
        work12 += work11
        # Vectorize matrices as compact vectors to get square matrix
        work = work12.view(np.float64).reshape((self.vn, -1))
        Hyy = lin.x_dot_dense(self.F2C_op, work.T)

        # ======================================================================
        # Construct YX block of Hessian, i.e., D2yxPhi'[Z^-1]
        # ======================================================================
        # D2_YX Pg(X, Y)[Eij]*[Z^-1]
        #   = -X^-½ D2xg(X^-½ Y X^-½)[X^½ Z^-1 X^½, X^-½ Eij X^-½] X^-½
        #     + X^-½ Dg(X^-½ Y X^-½)[X^½ Z^-1 Eij X^-½ + X^-½ Eij Z^-1 X^½] X^-½
        # Compute first term, i.e., -X^-½ D2xg(X^-½ Y X^-½)[ ... ] X^-½
        grad.scnd_frechet_multi(work11, D2xyx_xg, work14, UxyxXZXUxyx,
            U=irt2X_Uxyx, work1=work12, work2=work13, work3=work10)
        # Compute second term, i.e., X^-½ Dg(X^-½ Y X^½)[ ... ] X^-½
        work14 *= self.D1xyx_g
        lin.congr_multi(work12, irt2X_Uxyx, work14, work=work13,
            B=self.inv_Z @ rt2X_Uxyx)
        np.add(work12, work12.conj().transpose(0, 2, 1), out=work13)
        work13 -= work11
        # Vectorize matrices as compact vectors to get square matrix
        work = work13.view(np.float64).reshape((self.vn, -1))
        Hxy = lin.x_dot_dense(self.F2C_op, work.T)

        # Construct Hessian and Cholesky factor
        Hxx = (Hxx + Hxx.T) * 0.5
        Hyy = (Hyy + Hyy.T) * 0.5

        self.hess[: self.vn, : self.vn] = Hxx
        self.hess[self.vn :, self.vn :] = Hyy
        self.hess[self.vn :, : self.vn] = Hxy.T
        self.hess[: self.vn, self.vn :] = Hxy

        self.hess_fact = lin.cho_fact(self.hess)
        self.invhess_aux_updated = True

        return

    def update_invhessprod_aux_aux(self):
        assert not self.invhess_aux_aux_updated

        self.precompute_computational_basis()

        self.work10 = np.empty((self.n, self.n, self.vn), dtype=self.dtype)
        self.work11 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work12 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work13 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work14 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)

        self.hess = np.empty((2 * self.vn, 2 * self.vn))

        self.invhess_aux_aux_updated = True

    def get_central_ray(self):
        # Solve a 3-dimensional nonlinear system of equations to get the central
        # point of the barrier function
        (t, x, y) = (1.0 + self.g(1.0), 1.0, 1.0)  # Initial point

        for _ in range(10):
            # Precompute some useful things
            z = t - x * self.g(y / x)
            zi = 1 / z
            zi2 = zi * zi

            dx = self.dh(x / y)
            dy = self.dg(y / x)

            d2dx2 = self.d2h(x / y) / y
            d2dy2 = self.d2g(y / x) / x
            d2dxdy = -d2dy2 * y / x

            # Get gradient
            g = np.array([t - zi, x + dx * zi - 1 / x, y + dy * zi - 1 / y])

            # Get Hessian
            (Htt, Htx, Hty) = (zi2, -zi2 * dx, -zi2 * dy)
            Hxx = zi2 * dx * dx + zi * d2dx2 + 1 / x / x
            Hyy = zi2 * dy * dy + zi * d2dy2 + 1 / y / y
            Hxy = zi2 * dx * dy + zi * d2dxdy

            H = np.array([[Htt + 1, Htx, Hty],
                          [Htx, Hxx + 1, Hxy],
                          [Hty, Hxy, Hyy + 1]])

            # Perform Newton step
            delta = -np.linalg.solve(H, g)
            decrement = -np.dot(delta, g)

            # Check feasible
            (t1, x1, y1) = (t + delta[0], x + delta[1], y + delta[2])
            if x1 < 0 or y1 < 0 or t1 < x1 * self.g(y1 / x1):
                # Exit if not feasible and return last feasible point
                break

            (t, x, y) = (t1, x1, y1)
            
            # Exit if decrement is small, i.e., near optimality
            if decrement / 2.0 <= 1e-12:
                break

        return (t, x, y)
