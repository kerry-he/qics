import numpy as np
import scipy as sp
import qics._utils.linalg as lin
import qics._utils.gradient as grad
from qics.cones.base import Cone, get_perspective_derivatives
from qics.vectorize import get_full_to_compact_op


class OpPerspecTr(Cone):
    r"""A class representing a trace operator perspective epigraph cone

    .. math::

        \mathcal{OPT}_{n, g} = \text{cl}\{ (t, X, Y) \in \mathbb{R} \times
        \mathbb{H}^n_{++} \times \mathbb{H}^n_{++} : 
        t \geq \text{tr}[P_g(X, Y)] \},

    for an operator concave function 
    :math:`g:(0, \infty)\rightarrow\mathbb{R}`, where

    .. math::

        P_g(X, Y) = X^{1/2} g(X^{-1/2} Y X^{-1/2}) X^{1/2},

    is the operator perspective of :math:`g`.

    Parameters
    ----------
    n : :obj:`int`
        Dimension of the matrices :math:`X` and :math:`Y`.
    func : :obj:`string` or :obj:`float`
        Choice for the function :math:`g`. Can be defined in the following
        ways.

        - :math:`g(x) = -\log(x)` if ``func="log"``
        - :math:`g(x) = -x^p` if ``func=p`` is a :obj:`float` where 
          :math:`p\in(0, 1)`
        - :math:`g(x) = x^p` if ``func=p`` is a :obj:`float` where 
          :math:`p\in(-1, 0)\cup(1, 2)`

    iscomplex : :obj:`bool`
        Whether the matrices :math:`X` and :math:`Y` are defined over
        :math:`\mathbb{H}^n` (``True``), or restricted to 
        :math:`\mathbb{S}^n` (``False``). The default is ``False``.

    See also
    --------
    OpPerspecEpi : Operator perspective epigraph

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

        self.nu = 1 + 2 * self.n  # Barrier parameter

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

        # Get function handles for g(x), h(x)=x*g(1/x), x*g(x), and x*h(x)
        # and their first, second and third derivatives
        perspective_derivatives = get_perspective_derivatives(func)
        self.g, self.dg, self.d2g, self.d3g = perspective_derivatives["g"]
        self.h, self.dh, self.d2h, self.d3h = perspective_derivatives["h"]
        self.xg, self.dxg, self.d2xg, self.d3xg = perspective_derivatives["xg"]
        self.xh, self.dxh, self.d2xh, self.d3xh = perspective_derivatives["xh"]

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

        # Check that t > tr[Pg(X, Y)]
        self.g_Dxyx = self.g(self.Dxyx)
        self.h_Dyxy = self.h(self.Dyxy)
        g_XYX = (self.Uxyx * self.g_Dxyx) @ self.Uxyx.conj().T
        self.z = self.t[0, 0] - lin.inp(self.X, g_XYX)

        self.feas = self.z > 0
        return self.feas

    def get_val(self):
        assert self.feas_updated
        log_z = np.log(self.z)
        return -log_z - np.sum(np.log(self.Dx)) - np.sum(np.log(self.Dy))

    def update_grad(self):
        assert self.feas_updated
        assert not self.grad_updated

        # Compute X^-1 and Y^-1
        inv_Dx = np.reciprocal(self.Dx)
        inv_X_rt2 = self.Ux * np.sqrt(inv_Dx)
        self.inv_X = inv_X_rt2 @ inv_X_rt2.conj().T

        inv_Dy = np.reciprocal(self.Dy)
        inv_Y_rt2 = self.Uy * np.sqrt(inv_Dy)
        self.inv_Y = inv_Y_rt2 @ inv_Y_rt2.conj().T

        # Precompute useful expressions
        self.UyxyYUyxy = UyxyYUyxy = self.Uyxy.conj().T @ self.Y @ self.Uyxy
        self.UxyxXUxyx = UxyxXUxyx = self.Uxyx.conj().T @ self.X @ self.Uxyx

        self.irt2Y_Uyxy = irt2Y_Uyxy = self.irt2_Y @ self.Uyxy
        self.irt2X_Uxyx = irt2X_Uxyx = self.irt2_X @ self.Uxyx

        self.rt2Y_Uyxy = self.rt2_Y @ self.Uyxy
        self.rt2X_Uxyx = self.rt2_X @ self.Uxyx

        # Compute derivatives of trace operator perspective
        self.D1yxy_h = grad.D1_f(self.Dyxy, self.h_Dyxy, self.dh(self.Dyxy))
        if self.func == "log":
            self.D1xyx_g = -grad.D1_log(self.Dxyx, -self.g_Dxyx)
        else:
            self.D1xyx_g = grad.D1_f(self.Dxyx, self.g_Dxyx, self.dg(self.Dxyx))
        # D_X trPg(X, Y) = Y^-½ Dh(Y^-½ X Y^-½)[Y] Y^-½
        work = irt2Y_Uyxy @ (self.D1yxy_h * UyxyYUyxy) @ irt2Y_Uyxy.conj().T
        self.DPhiX = (work + work.conj().T) * 0.5
        # D_Y trPg(X, Y) = X^-½ Dg(X^-½ Y X^-½)[X] X^-½
        work = irt2X_Uxyx @ (self.D1xyx_g * UxyxXUxyx) @ irt2X_Uxyx.conj().T
        self.DPhiY = (work + work.conj().T) * 0.5

        # Precompute compact vectorizations of derivatives
        DPhiX_cvec = self.DPhiX.view(np.float64).reshape(-1, 1)
        DPhiX_cvec = self.F2C_op @ DPhiX_cvec

        DPhiY_cvec = self.DPhiY.view(np.float64).reshape(-1, 1)
        DPhiY_cvec = self.F2C_op @ DPhiY_cvec

        self.DPhi_cvec = np.vstack((DPhiX_cvec, DPhiY_cvec))

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

        D2yxy_h, D2yxy_xh = self.D2yxy_h, self.D2yxy_xh
        D2xyx_g, D2xyx_xg = self.D2xyx_g, self.D2xyx_xg
        UyxyYUyxy, UxyxXUxyx = self.UyxyYUyxy, self.UxyxXUxyx
        rt2Y_Uyxy, irt2Y_Uyxy = self.rt2Y_Uyxy, self.irt2Y_Uyxy
        rt2X_Uxyx, irt2X_Uxyx = self.rt2X_Uxyx, self.irt2X_Uxyx

        UyxyYHxYUyxy = self.irt2Y_Uyxy.conj().T @ Hx @ self.irt2Y_Uyxy
        UxyxXHyXUxyx = self.irt2X_Uxyx.conj().T @ Hy @ self.irt2X_Uxyx

        # Hessian product of trace operator perspective
        # D2_XX trPg(X, Y)[Hx] = Y^-½ D2h(Y^-½ X Y^½)[Y, Y^-½ Hx Y^-½] Y^-½
        D2PhiXXH = grad.scnd_frechet(D2yxy_h, UyxyYUyxy, UyxyYHxYUyxy, 
                                     U=irt2Y_Uyxy)
        # D2_XY trPg(X, Y)[Hy]
        #     = -X^-½ D2xg(X^-½ Y X^-½)[X, X^-½ Hy X^-½] X^-½
        #       + X^½ Dg(X^-½ Y X-^½)[X^-½ Hy X^-½] X^-½
        #       + X^-½ Dg(X^-½ Y X-^½)[X^-½ Hy X^-½] X^½
        work = self.D1xyx_g * UxyxXHyXUxyx
        D2PhiXYH = irt2X_Uxyx @ work @ rt2X_Uxyx.conj().T
        D2PhiXYH += D2PhiXYH.conj().T
        D2PhiXYH -= grad.scnd_frechet(D2xyx_xg, UxyxXUxyx, UxyxXHyXUxyx, 
                                      U=irt2X_Uxyx)
        # D2_YX trPg(X, Y)[Hx]
        #     = -Y^-½ D2xh(Y^-½ X Y^-½)[Y, Y^-½ Hx Y^-½] Y^-½
        #       + Y^½ Dh(Y^-½ X Y-^½)[Y^-½ Hx Y^-½] Y^-½
        #       + Y^-½ Dh(Y^-½ X Y-^½)[Y^-½ Hx Y^-½] Y^½
        work = self.D1yxy_h * UyxyYHxYUyxy
        D2PhiYXH = irt2Y_Uyxy @ work @ rt2Y_Uyxy.conj().T
        D2PhiYXH += D2PhiYXH.conj().T
        D2PhiYXH -= grad.scnd_frechet(D2yxy_xh, UyxyYUyxy, UyxyYHxYUyxy, 
                                      U=irt2Y_Uyxy)
        # D2_YY trPg(X, Y)[Hy] = X^-½ D2g(X^-½ Y X^½)[X, X^-½ Hy X^-½] X^-½
        D2PhiYYH = grad.scnd_frechet(D2xyx_g, UxyxXUxyx, UxyxXHyXUxyx, 
                                     U=irt2X_Uxyx)

        # ======================================================================
        # Hessian products with respect to t
        # ======================================================================
        # D2_t F(t, X, Y)[Ht, Hx, Hy] 
        #         = (Ht - D_X S(X||Y)[Hx] - D_Y S(X||Y)[Hy]) / z^2
        out_t = Ht - lin.inp(self.DPhiX, Hx) - lin.inp(self.DPhiY, Hy)
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
        #         = -D2_t F(t, X, Y)[Ht, Hx, Hy] * D_Y S(X||Y)
        #           + (D2_YX S(X||Y)[Hx] + D2_YY S(X||Y)[Hy]) / z
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

        D2yxy_h, D2yxy_xh = self.D2yxy_h, self.D2yxy_xh
        D2xyx_g, D2xyx_xg = self.D2xyx_g, self.D2xyx_xg
        UyxyYUyxy, UxyxXUxyx = self.UyxyYUyxy, self.UxyxXUxyx
        rt2Y_Uyxy, irt2Y_Uyxy = self.rt2Y_Uyxy, self.irt2Y_Uyxy
        rt2X_Uxyx, irt2X_Uxyx = self.rt2X_Uxyx, self.irt2X_Uxyx

        work0, work1 = self.work0, self.work1, 
        work2, work3 = self.work2, self.work3
        work4, work5, work6 = self.work4, self.work5, self.work6

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
        # Hessian products with respect to X
        # ======================================================================
        # Hessian products of trace operator perspective
        # D2_XX trPg(X, Y)[Hx] = Y^-½ D2h(Y^-½ X Y^½)[Y, Y^-½ Hx Y^-½] Y^-½
        lin.congr_multi(work0, irt2Y_Uyxy.conj().T, self.Ax, work=work3)
        grad.scnd_frechet_multi(work5, D2yxy_h, work0, UyxyYUyxy, U=irt2Y_Uyxy,
            work1=work3, work2=work4, work3=work6)
        # D2_XY trPg(X, Y)[Hy]
        #     = -X^-½ D2xg(X^-½ Y X^-½)[X, X^-½ Hy X^-½] X^-½
        #       + X^½ Dg(X^-½ Y X-^½)[X^-½ Hy X^-½] X^-½
        #       + X^-½ Dg(X^-½ Y X-^½)[X^-½ Hy X^-½] X^½
        # Second and third terms, i.e., X^½ [ ... ] X^-½ + X^-½ [ ... ] X^½
        lin.congr_multi(work1, irt2X_Uxyx.conj().T, self.Ay, work=work3)
        np.multiply(work1, self.D1xyx_g, out=work2)
        lin.congr_multi(work3, irt2X_Uxyx, work2, work=work4, B=rt2X_Uxyx)
        np.add(work3, work3.conj().transpose(0, 2, 1), out=work2)
        work5 += work2
        # First term, i.e., -X^-½ D2xg(X^-½ Y X^-½)[X, X^-½ Hy X^-½] X^-½
        grad.scnd_frechet_multi(work2, D2xyx_xg, work1, UxyxXUxyx, 
            U=irt2X_Uxyx, work1=work3, work2=work4, work3=work6)
        work5 -= work2

        # Hessian product of barrier function
        # D2_X F(t, X, Y)[Ht, Hx, Hy] 
        #         = -D2_t F(t, X, Y)[Ht, Hx, Hy] * D_X trPg(X, Y)
        #           + (D2_XX trPg(X, Y)[Hx] + D2_XY trPg(X, Y)[Hy]) / z
        #           + X^-1 Hx X^-1
        work5 *= self.zi
        np.outer(out_t, self.DPhiX, out=work2.reshape((p, -1)))
        work5 -= work2
        lin.congr_multi(work2, self.inv_X, self.Ax, work=work3)
        work5 += work2

        lhs[:, self.idx_X] = work5.reshape((p, -1)).view(np.float64)

        # ==================================================================
        # Hessian products with respect to Y
        # ==================================================================
        # Hessian products of trace operator perspective
        # D2_YY trPg(X, Y)[Hy] = X^-½ D2g(X^-½ Y X^½)[X, X^-½ Hy X^-½] X^-½
        lin.congr_multi(work1, irt2X_Uxyx.conj().T, self.Ay, work=work3)
        grad.scnd_frechet_multi(work5, D2xyx_g, work1, UxyxXUxyx, U=irt2X_Uxyx, 
            work1=work3, work2=work4, work3=work6)

        # D2_YX trPg(X, Y)[Hx]
        #     = -Y^-½ D2xh(Y^-½ X Y^-½)[Y, Y^-½ Hx Y^-½] Y^-½
        #       + Y^½ Dh(Y^-½ X Y-^½)[Y^-½ Hx Y^-½] Y^-½
        #       + Y^-½ Dh(Y^-½ X Y-^½)[Y^-½ Hx Y^-½] Y^½
        # Second and third terms, i.e., Y^½ [ ... ] Y^-½ + Y^-½ [ ... ] Y^½
        lin.congr_multi(work0, irt2Y_Uyxy.conj().T, self.Ax, work=work3)
        np.multiply(work0, self.D1yxy_h, out=work2)
        lin.congr_multi(work3, irt2Y_Uyxy, work2, work=work4, B=rt2Y_Uyxy)
        np.add(work3, work3.conj().transpose(0, 2, 1), out=work2)
        work5 += work2
        # First term, i.e., -Y^-½ D2xh(Y^-½ X Y^-½)[Y, Y^-½ Hx Y^-½] Y^-½
        grad.scnd_frechet_multi(work2, D2yxy_xh, work0, UyxyYUyxy, U=irt2Y_Uyxy,
            work1=work3, work2=work4, work3=work6)
        work5 -= work2

        # Hessian product of barrier function
        # D2_Y F(t, X, Y)[Ht, Hx, Hy] 
        #         = -D2_t F(t, X, Y)[Ht, Hx, Hy] * D_Y S(X||Y)
        #           + (D2_YX S(X||Y)[Hx] + D2_YY S(X||Y)[Hy]) / z
        #           + Y^-1 Hy Y^-1
        work5 *= self.zi
        np.outer(out_t, self.DPhiY, out=work2.reshape((p, -1)))
        work5 -= work2
        lin.congr_multi(work2, self.inv_Y, self.Ay, work=work3)
        work5 += work2

        lhs[:, self.idx_Y] = work5.reshape((p, -1)).view(np.float64)

        # Multiply A (H A')
        return lin.dense_dot_x(lhs, A.T)

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
        out_XY = lin.cho_solve(self.hess_fact, Wxy_cvec)
        out_XY = out_XY.reshape(2, -1)
        
        out_X = self.F2C_op.T @ out_XY[0]
        out_X = out_X.view(self.dtype).reshape((self.n, self.n))
        out[1][:] = (out_X + out_X.conj().T) * 0.5

        out_Y = self.F2C_op.T @ out_XY[1]
        out_Y = out_Y.view(self.dtype).reshape((self.n, self.n))
        out[2][:] = (out_Y + out_Y.conj().T) * 0.5

        # Solve for t = z^2 Ht + <DPhi(X, Y), (X, Y)>
        out_t = self.z2 * Ht
        out_t += lin.inp(out_X, self.DPhiX)
        out_t += lin.inp(out_Y, self.DPhiY)
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
        out_xy = lin.cho_solve(self.hess_fact, self.work)

        # Solve for t = z^2 Ht + <DPhi(X, Y), (X, Y)>
        out_t = self.z2 * self.At.reshape(-1, 1) + out_xy.T @ self.DPhi_cvec

        # Multiply A (H A')
        return lin.x_dot_dense(self.Axy_cvec, out_xy) + np.outer(self.At, out_t)

    def third_dir_deriv_axpy(self, out, H, a=True):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.dder3_aux_updated:
            self.update_dder3_aux()

        (Ht, Hx, Hy) = H

        Dyxy, Dxyx, Uyxy, Uxyx = self.Dyxy, self.Dxyx, self.Uyxy, self.Uxyx
        D2yxy_h, D2yxy_xh = self.D2yxy_h, self.D2yxy_xh
        D2xyx_g, D2xyx_xg = self.D2xyx_g, self.D2xyx_xg
        UyxyYUyxy, UxyxXUxyx = self.UyxyYUyxy, self.UxyxXUxyx
        rt2Y_Uyxy, irt2Y_Uyxy = self.rt2Y_Uyxy, self.irt2Y_Uyxy
        rt2X_Uxyx, irt2X_Uxyx = self.rt2X_Uxyx, self.irt2X_Uxyx

        chi = Ht[0, 0] - lin.inp(self.DPhiX, Hx) - lin.inp(self.DPhiY, Hy)
        chi2 = chi * chi
        
        UyxyYHxYUyxy = irt2Y_Uyxy.conj().T @ Hx @ irt2Y_Uyxy
        UxyxXHyXUxyx = irt2X_Uxyx.conj().T @ Hy @ irt2X_Uxyx
        UxyxXHxXUxyx = irt2X_Uxyx.conj().T @ Hx @ irt2X_Uxyx
        UyxyYHyYUyxy = irt2Y_Uyxy.conj().T @ Hy @ irt2Y_Uyxy

        # Trace noncommutative perspective Hessians
        D2PhiXXH = grad.scnd_frechet(D2yxy_h, UyxyYUyxy, UyxyYHxYUyxy, 
                                     U=irt2Y_Uyxy)

        work = self.D1xyx_g * UxyxXHyXUxyx
        D2PhiXYH = irt2X_Uxyx @ work @ rt2X_Uxyx.conj().T
        D2PhiXYH += D2PhiXYH.conj().T
        D2PhiXYH -= grad.scnd_frechet(D2xyx_xg, UxyxXUxyx, UxyxXHyXUxyx, 
                                      U=irt2X_Uxyx)

        work = self.D1yxy_h * UyxyYHxYUyxy
        D2PhiYXH = irt2Y_Uyxy @ work @ rt2Y_Uyxy.conj().T
        D2PhiYXH += D2PhiYXH.conj().T
        D2PhiYXH -= grad.scnd_frechet(D2yxy_xh, UyxyYUyxy, UyxyYHxYUyxy, 
                                      U=irt2Y_Uyxy)

        D2PhiYYH = grad.scnd_frechet(D2xyx_g, UxyxXUxyx, UxyxXHyXUxyx, 
                                     U=irt2X_Uxyx)

        D2PhiXHH = lin.inp(Hx, D2PhiXXH + D2PhiXYH)
        D2PhiYHH = lin.inp(Hy, D2PhiYXH + D2PhiYYH)

        # Trace noncommutative perspective third order derivatives
        # Second derivatives of D_X trPg(X, Y)
        D3PhiXXX = grad.thrd_frechet(Dyxy, D2yxy_h, self.d3h(Dyxy), irt2Y_Uyxy,
            UyxyYUyxy, UyxyYHxYUyxy)

        work = rt2Y_Uyxy.conj().T @ Hy @ irt2Y_Uyxy
        work = work + work.conj().T
        D3PhiXXY = grad.scnd_frechet(D2yxy_h, work, UyxyYHxYUyxy, U=irt2Y_Uyxy)
        D3PhiXXY -= grad.thrd_frechet(Dyxy, D2yxy_xh, self.d3xh(Dyxy),
            irt2Y_Uyxy, UyxyYUyxy, UyxyYHxYUyxy, UyxyYHyYUyxy)
        D3PhiXYX = D3PhiXXY

        work = grad.scnd_frechet(D2xyx_g, UxyxXHyXUxyx, UxyxXHyXUxyx, U=Uxyx)
        D3PhiXYY = self.irt2_X @ work @ self.rt2_X
        D3PhiXYY += D3PhiXYY.conj().T
        D3PhiXYY -= grad.thrd_frechet(Dxyx, D2xyx_xg, self.d3xg(Dxyx),
            irt2X_Uxyx, UxyxXUxyx, UxyxXHyXUxyx)

        # Second derivatives of D_Y trPg(X, Y)
        D3PhiYYY = grad.thrd_frechet(Dxyx, D2xyx_g, self.d3g(Dxyx), irt2X_Uxyx,
            UxyxXUxyx, UxyxXHyXUxyx)

        work = rt2X_Uxyx.conj().T @ Hx @ irt2X_Uxyx
        work = work + work.conj().T
        D3PhiYYX = grad.scnd_frechet(D2xyx_g, work, UxyxXHyXUxyx, U=irt2X_Uxyx)
        D3PhiYYX -= grad.thrd_frechet(Dxyx, D2xyx_xg, self.d3xg(Dxyx),
            irt2X_Uxyx, UxyxXUxyx, UxyxXHyXUxyx, UxyxXHxXUxyx)
        D3PhiYXY = D3PhiYYX

        work = grad.scnd_frechet(D2yxy_h, UyxyYHxYUyxy, UyxyYHxYUyxy, U=Uyxy)
        D3PhiYXX = self.irt2_Y @ work @ self.rt2_Y
        D3PhiYXX += D3PhiYXX.conj().T
        D3PhiYXX -= grad.thrd_frechet(Dyxy, D2yxy_xh, self.d3xh(Dyxy),
            irt2Y_Uyxy, UyxyYUyxy, UyxyYHxYUyxy)

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

        from qics.vectorize import vec_to_mat
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

        Dyxy, Dxyx = self.Dyxy, self.Dxyx

        self.D1yxy_xh = grad.D1_f(Dyxy, self.xh(Dyxy), self.dxh(Dyxy))
        self.D1xyx_xg = grad.D1_f(Dxyx, self.xg(Dxyx), self.dxg(Dxyx))

        self.D2yxy_h = grad.D2_f(Dyxy, self.D1yxy_h, self.d2h(Dyxy))
        self.D2xyx_g = grad.D2_f(Dxyx, self.D1xyx_g, self.d2g(Dxyx))
        self.D2yxy_xh = grad.D2_f(Dyxy, self.D1yxy_xh, self.d2xh(Dyxy))
        self.D2xyx_xg = grad.D2_f(Dxyx, self.D1xyx_xg, self.d2xg(Dxyx))

        # Preparing other required variables
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

        D2xyx_g, D2xyx_xg = self.D2xyx_g, self.D2xyx_xg
        D2yxy_h, irt2Y_Uyxy = self.D2yxy_h, self.irt2Y_Uyxy
        UyxyYUyxy, UxyxXUxyx = self.UyxyYUyxy, self.UxyxXUxyx
        rt2X_Uxyx, irt2X_Uxyx = self.rt2X_Uxyx, self.irt2X_Uxyx

        work10, work11 = self.work10, self.work11
        work12, work13, work14 = self.work12, self.work13, self.work14

        # ======================================================================
        # Construct XX block of Hessian, i.e., (D2xxPhi + X^1 ⊗ X^-1)
        # ======================================================================
        # D2_XX trPg(X, Y)[Hx] = Y^-½ D2h(Y^-½ X Y^½)[Y, Y^-½ Hx Y^-½] Y^-½
        lin.congr_multi(work14, irt2Y_Uyxy.conj().T, self.E, work=work13)
        grad.scnd_frechet_multi(work11, D2yxy_h, work14, UyxyYUyxy,
            U=irt2Y_Uyxy, work1=work12, work2=work13, work3=work10)
        work11 *= self.zi
        # X^1 Eij X^-1
        lin.congr_multi(work14, self.inv_X, self.E, work=work13)
        work14 += work11
        # Vectorize matrices as compact vectors to get square matrix
        work = work14.view(np.float64).reshape((self.vn, -1))
        Hxx = lin.x_dot_dense(self.F2C_op, work.T)    

        # ======================================================================
        # Construct YY block of Hessian, i.e., (D2yyPhi + Y^1 ⊗ Y^-1)
        # ======================================================================
        # D2_YY trPg(X, Y)[Hy] = X^-½ D2g(X^-½ Y X^½)[X, X^-½ Hy X^-½] X^-½
        lin.congr_multi(work14, irt2X_Uxyx.conj().T, self.E, work=work13)
        grad.scnd_frechet_multi(work11, D2xyx_g, work14, UxyxXUxyx,
            U=irt2X_Uxyx, work1=work12, work2=work13, work3=work10)
        work11 *= self.zi
        # Y^1 Eij Y^-1
        lin.congr_multi(work12, self.inv_Y, self.E, work=work13)
        work12 += work11
        # Vectorize matrices as compact vectors to get square matrix
        work = work12.view(np.float64).reshape((self.vn, -1))
        Hyy = lin.x_dot_dense(self.F2C_op, work.T)    

        # ======================================================================
        # Construct XY block of Hessian, i.e., D2yxPhi
        # ======================================================================
        # D2_XY trPg(X, Y)[Hy]
        #     = -X^-½ D2xg(X^-½ Y X^-½)[X, X^-½ Hy X^-½] X^-½
        #       + X^½ Dg(X^-½ Y X-^½)[X^-½ Hy X^-½] X^-½
        #       + X^-½ Dg(X^-½ Y X-^½)[X^-½ Hy X^-½] X^½
        # First term, i.e., -X^-½ D2xg(X^-½ Y X^-½)[X, X^-½ Hy X^-½] X^-½
        grad.scnd_frechet_multi(work11, D2xyx_xg, work14, UxyxXUxyx,
            U=irt2X_Uxyx, work1=work12, work2=work13, work3=work10)
        # Second and third terms, i.e., X^½ [ ... ] X^-½ + X^-½ [ ... ] X^½
        work14 *= self.D1xyx_g
        lin.congr_multi(work12, irt2X_Uxyx, work14, work=work13, B=rt2X_Uxyx)
        np.add(work12, work12.conj().transpose(0, 2, 1), out=work13)
        work13 -= work11
        work13 *= self.zi
        # Vectorize matrices as compact vectors to get square matrix
        work = work13.view(np.float64).reshape((self.vn, -1))
        Hxy = lin.x_dot_dense(self.F2C_op, work.T)

        # Construct Hessian and factorize
        Hxx = (Hxx + Hxx.conj().T) * 0.5
        Hyy = (Hyy + Hyy.conj().T) * 0.5

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

        self.hess = np.empty((2 * self.vn, 2 * self.vn))

        self.work10 = np.empty((self.n, self.n, self.vn), dtype=self.dtype)
        self.work11 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work12 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work13 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work14 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)

        self.invhess_aux_aux_updated = True

    def update_dder3_aux(self):
        assert not self.dder3_aux_updated
        assert self.hess_aux_updated

        self.zi3 = self.zi2 * self.zi

        self.dder3_aux_updated = True

    def get_central_ray(self):
        # Solve a 3-dimensional nonlinear system of equations to get the central
        # point of the barrier function
        n = self.n
        (t, x, y) = (1.0 + n * self.g(1.0), 1.0, 1.0)

        for _ in range(10):
            # Precompute some useful things
            z = t - n * x * self.g(y / x)
            zi = 1 / z
            zi2 = zi * zi

            dx = self.dh(x / y)
            dy = self.dg(y / x)

            d2dx2 = self.d2h(x / y) / y
            d2dy2 = self.d2g(y / x) / x
            d2dxdy = -d2dy2 * y / x

            # Get gradient
            g = np.array([t - zi, 
                          n * x + n * dx * zi - n / x, 
                          n * y + n * dy * zi - n / y])

            # Get Hessian
            (Htt, Htx, Hty) = (zi2, -n * zi2 * dx, -n * zi2 * dy)
            Hxx = n * n * zi2 * dx * dx + n * zi * d2dx2 + n / x / x
            Hyy = n * n * zi2 * dy * dy + n * zi * d2dy2 + n / y / y
            Hxy = n * n * zi2 * dx * dy + n * zi * d2dxdy

            H = np.array([[Htt + 1, Htx, Hty],
                          [Htx, Hxx + n, Hxy],
                          [Hty, Hxy, Hyy + n]])

            # Perform Newton step
            delta = -np.linalg.solve(H, g)
            decrement = -np.dot(delta, g)

            # Check feasible
            (t1, x1, y1) = (t + delta[0], x + delta[1], y + delta[2])
            if x1 < 0 or y1 < 0 or t1 < n * x1 * self.g(y1 / x1):
                # Exit if not feasible and return last feasible point
                break

            (t, x, y) = (t1, x1, y1)

            # Exit if decrement is small, i.e., near optimality
            if decrement / 2.0 <= 1e-12:
                break

        return (t, x, y)
