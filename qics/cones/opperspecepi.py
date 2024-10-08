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
        # Dimension properties
        self.n = n  # Side dimension of system
        self.nu = 3 * self.n  # Barrier parameter

        self.iscomplex = iscomplex  # Hermitian or symmetric vector space
        self.vn = (
            n * n if iscomplex else n * (n + 1) // 2
        )  # Compact dimension of system

        self.dim = (
            [n * n, n * n, n * n]
            if (not iscomplex)
            else [2 * n * n, 2 * n * n, 2 * n * n]
        )
        self.type = ["s", "s", "s"] if (not iscomplex) else ["h", "h", "h"]
        self.dtype = np.float64 if (not iscomplex) else np.complex128

        self.idx_T = slice(0, self.dim[0])
        self.idx_X = slice(self.dim[0], 2 * self.dim[0])
        self.idx_Y = slice(2 * self.dim[0], 3 * self.dim[0])

        # Get LAPACK operators
        self.X = np.eye(self.n, dtype=self.dtype)

        self.cho_fact = sp.linalg.lapack.get_lapack_funcs("potrf", (self.X,))
        self.cho_inv = sp.linalg.lapack.get_lapack_funcs("trtri", (self.X,))

        # Update flags
        self.feas_updated = False
        self.grad_updated = False
        self.hess_aux_updated = False
        self.invhess_aux_updated = False
        self.invhess_aux_aux_updated = False
        self.dder3_aux_updated = False
        self.congr_aux_updated = False

        (
            self.g,
            self.dg,
            self.d2g,
            self.d3g,
            self.xg,
            self.dxg,
            self.d2xg,
            self.d3xg,
            self.h,
            self.dh,
            self.d2h,
            self.d3h,
            self.xh,
            self.dxh,
            self.d2xh,
            self.d3xh,
        ) = get_perspective_derivatives(func)
        self.func = func

        self.F2C_op = get_full_to_compact_op(self.n, self.iscomplex)

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

        # Check that X and Y are PSD
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

        # Check that t > tr[X^0.5 g(X^-1/2 Y X^-1/2) X^0.5]
        self.g_Dxyx = self.g(self.Dxyx)
        self.h_Dyxy = self.h(self.Dyxy)
        g_XYX = (self.Uxyx * self.g_Dxyx) @ self.Uxyx.conj().T

        self.Z = self.T - self.rt2_X @ g_XYX @ self.rt2_X

        # Try to perform Cholesky factorization to check PSD
        self.Z_chol, info = self.cho_fact(self.Z, lower=True)
        if info != 0:
            self.feas = False
            return self.feas
        self.feas = True

        return self.feas

    def get_val(self):
        assert self.feas_updated
        (sgn, logabsdet_Z) = np.linalg.slogdet(self.Z)
        return -(sgn * logabsdet_Z) - np.sum(np.log(self.Dx)) - np.sum(np.log(self.Dy))

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

        # Precompute useful expressions
        self.irt2Y_Uyxy = self.irt2_Y @ self.Uyxy
        self.irt2X_Uxyx = self.irt2_X @ self.Uxyx

        self.rt2Y_Uyxy = self.rt2_Y @ self.Uyxy
        self.rt2X_Uxyx = self.rt2_X @ self.Uxyx

        self.UyxyYZYUyxy = self.rt2Y_Uyxy.conj().T @ self.inv_Z @ self.rt2Y_Uyxy
        self.UxyxXZXUxyx = self.rt2X_Uxyx.conj().T @ self.inv_Z @ self.rt2X_Uxyx

        # Compute derivatives of Pg(X, Y)
        self.D1yxy_h = grad.D1_f(self.Dyxy, self.h_Dyxy, self.dh(self.Dyxy))
        if self.func == "log":
            self.D1xyx_g = -grad.D1_log(self.Dxyx, -self.g_Dxyx)
        else:
            self.D1xyx_g = grad.D1_f(self.Dxyx, self.g_Dxyx, self.dg(self.Dxyx))

        self.DPhiX = (
            self.irt2Y_Uyxy
            @ (self.D1yxy_h * self.UyxyYZYUyxy)
            @ self.irt2Y_Uyxy.conj().T
        )
        self.DPhiX = (self.DPhiX + self.DPhiX.conj().T) * 0.5
        self.DPhiY = (
            self.irt2X_Uxyx
            @ (self.D1xyx_g * self.UxyxXZXUxyx)
            @ self.irt2X_Uxyx.conj().T
        )
        self.DPhiY = (self.DPhiY + self.DPhiY.conj().T) * 0.5

        self.grad = [-self.inv_Z, self.DPhiX - self.inv_X, self.DPhiY - self.inv_Y]

        self.grad_updated = True

    def hess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        # Computes Hessian product of Pg(X, Y) barrier with a single vector (Ht, Hx, Hy)

        (Ht, Hx, Hy) = H

        work = self.irt2Y_Uyxy.conj().T @ Hx @ self.irt2Y_Uyxy
        DxPhiHx = self.rt2Y_Uyxy @ (self.D1yxy_h * work) @ self.rt2Y_Uyxy.conj().T

        work = self.irt2X_Uxyx.conj().T @ Hy @ self.irt2X_Uxyx
        DyPhiHy = self.rt2X_Uxyx @ (self.D1xyx_g * work) @ self.rt2X_Uxyx.conj().T

        # Hessian product for T
        out[0][:] = self.inv_Z @ (Ht - DxPhiHx - DyPhiHy) @ self.inv_Z

        # Hessian product for X
        work = self.inv_Z @ Hy @ self.inv_Y
        work = (
            self.rt2Y_Uyxy.conj().T @ (work + work.conj().T - out[0]) @ self.rt2Y_Uyxy
        )
        outX = self.irt2Y_Uyxy @ (self.D1yxy_h * work) @ self.irt2Y_Uyxy.conj().T

        work = self.irt2Y_Uyxy.conj().T @ Hx @ self.irt2Y_Uyxy
        outX += grad.scnd_frechet(
            self.D2yxy_h, self.UyxyYZYUyxy, work, U=self.irt2Y_Uyxy
        )

        work = self.irt2Y_Uyxy.conj().T @ Hy @ self.irt2Y_Uyxy
        outX -= grad.scnd_frechet(
            self.D2yxy_xh, self.UyxyYZYUyxy, work, U=self.irt2Y_Uyxy
        )

        outX += self.inv_X @ Hx @ self.inv_X

        out[1][:] = (outX + outX.conj().T) * 0.5

        # Hessian product for Y
        work = self.inv_Z @ Hx @ self.inv_X
        work = (
            self.rt2X_Uxyx.conj().T @ (work + work.conj().T - out[0]) @ self.rt2X_Uxyx
        )
        outY = self.irt2X_Uxyx @ (self.D1xyx_g * work) @ self.irt2X_Uxyx.conj().T

        work = self.irt2X_Uxyx.conj().T @ Hy @ self.irt2X_Uxyx
        outY += grad.scnd_frechet(
            self.D2xyx_g, self.UxyxXZXUxyx, work, U=self.irt2X_Uxyx
        )

        work = self.irt2X_Uxyx.conj().T @ Hx @ self.irt2X_Uxyx
        outY -= grad.scnd_frechet(
            self.D2xyx_xg, self.UxyxXZXUxyx, work, U=self.irt2X_Uxyx
        )

        outY += self.inv_Y @ Hy @ self.inv_Y

        out[2][:] = (outY + outY.conj().T) * 0.5

        return out

    def hess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)

        # The Hessian is of the form
        #     H = [ 0  0 ] + [      Z^-1 kron Z^-1           -(Z^-1 kron Z^-1) DxPhi   ]
        #         [ 0  M ]   [ -DPhi' (Z^-1 kron Z^-1)    DPhi' (Z^-1 kron Z^-1) DxPhi ]
        #
        #       = [ 0  0 ] + [      L^-T kron L^-T     ] [      L^-T kron L^-T     ]'
        #         [ 0  M ]   [ -DPhi' (L^-T kron L^-T) ] [ -DPhi' (L^-T kron L^-T) ]
        # where Z = L L' and M is precomputed in update_invhess_aux()

        p = A.shape[0]
        lhs = np.empty((p, sum(self.dim)))

        # DxPhiHx
        lin.congr_multi(self.work2, self.irt2Y_Uyxy.conj().T, self.Ax, work=self.work3)
        self.work2 *= self.D1yxy_h
        lin.congr_multi(self.work1, self.rt2Y_Uyxy, self.work2, work=self.work3)

        # DyPhiHy
        lin.congr_multi(self.work2, self.irt2X_Uxyx.conj().T, self.Ay, work=self.work3)
        self.work2 *= self.D1xyx_g
        lin.congr_multi(self.work0, self.rt2X_Uxyx, self.work2, work=self.work3)

        # Hessian product for T
        self.work0 += self.work1
        np.subtract(self.At, self.work0, out=self.work2)
        lin.congr_multi(self.work0, self.inv_Z, self.work2, work=self.work3)

        lhs[:, self.idx_T] = self.work0.reshape((p, -1)).view(dtype=np.float64)

        # Hessian product for X
        lin.congr_multi(self.work2, self.inv_Z, self.Ay, work=self.work3, B=self.inv_Y)
        np.add(self.work2, self.work2.conj().transpose(0, 2, 1), out=self.work1)
        self.work1 -= self.work0
        lin.congr_multi(self.work2, self.rt2Y_Uyxy.conj().T, self.work1, work=self.work3)
        self.work2 *= self.D1yxy_h
        lin.congr_multi(self.work1, self.irt2Y_Uyxy, self.work2, work=self.work3)

        lin.congr_multi(self.work2, self.irt2Y_Uyxy.conj().T, self.Ax, work=self.work3)
        grad.scnd_frechet_multi(
            self.work5,
            self.D2yxy_h,
            self.work2,
            self.UyxyYZYUyxy,
            U=self.irt2Y_Uyxy,
            work1=self.work3,
            work2=self.work4,
            work3=self.work6,
        )
        self.work1 += self.work5

        lin.congr_multi(self.work2, self.irt2Y_Uyxy.conj().T, self.Ay, work=self.work3)
        grad.scnd_frechet_multi(
            self.work5,
            self.D2yxy_xh,
            self.work2,
            self.UyxyYZYUyxy,
            U=self.irt2Y_Uyxy,
            work1=self.work3,
            work2=self.work4,
            work3=self.work6,
        )
        self.work1 -= self.work5            

        lin.congr_multi(self.work2, self.inv_X, self.Ax, work=self.work3)
        self.work1 += self.work2

        lhs[:, self.idx_X] = self.work1.reshape((p, -1)).view(dtype=np.float64)


        # Hessian product for Y
        lin.congr_multi(self.work2, self.inv_Z, self.Ax, work=self.work3, B=self.inv_X)
        np.add(self.work2, self.work2.conj().transpose(0, 2, 1), out=self.work1)
        self.work1 -= self.work0
        lin.congr_multi(self.work2, self.rt2X_Uxyx.conj().T, self.work1, work=self.work3)
        self.work2 *= self.D1xyx_g
        lin.congr_multi(self.work1, self.irt2X_Uxyx, self.work2, work=self.work3)

        lin.congr_multi(self.work2, self.irt2X_Uxyx.conj().T, self.Ay, work=self.work3)
        grad.scnd_frechet_multi(
            self.work5,
            self.D2xyx_g,
            self.work2,
            self.UxyxXZXUxyx,
            U=self.irt2X_Uxyx,
            work1=self.work3,
            work2=self.work4,
            work3=self.work6,
        )
        self.work1 += self.work5

        lin.congr_multi(self.work2, self.irt2X_Uxyx.conj().T, self.Ax, work=self.work3)
        grad.scnd_frechet_multi(
            self.work5,
            self.D2xyx_xg,
            self.work2,
            self.UxyxXZXUxyx,
            U=self.irt2X_Uxyx,
            work1=self.work3,
            work2=self.work4,
            work3=self.work6,
        )
        self.work1 -= self.work5   

        lin.congr_multi(self.work2, self.inv_Y, self.Ay, work=self.work3)
        self.work1 += self.work2

        lhs[:, self.idx_Y] = self.work1.reshape((p, -1)).view(dtype=np.float64)
        

        return lin.dense_dot_x(lhs, A.T)

    def invhess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        # Computes inverse Hessian product of Pg(X, Y) barrier with a single vector (Ht, Hx, Hy)
        # See invhess_congr() for additional comments

        (Ht, Hx, Hy) = H

        # ====================================================================
        # Inverse Hessian products with respect to (X, Y)
        # ====================================================================
        work = self.rt2Y_Uyxy.conj().T @ Ht @ self.rt2Y_Uyxy
        Wx = Hx + self.irt2Y_Uyxy @ (self.D1yxy_h * work) @ self.irt2Y_Uyxy.conj().T
        Wx_vec = Wx.view(dtype=np.float64).reshape(-1, 1)
        Wx_vec = self.F2C_op @ Wx_vec

        work = self.rt2X_Uxyx.conj().T @ Ht @ self.rt2X_Uxyx
        Wy = Hy + self.irt2X_Uxyx @ (self.D1xyx_g * work) @ self.irt2X_Uxyx.conj().T
        Wy_vec = Wy.view(dtype=np.float64).reshape(-1, 1)
        Wy_vec = self.F2C_op @ Wy_vec

        Wxy_vec = np.vstack((Wx_vec, Wy_vec))
        outxy = lin.cho_solve(self.hess_fact, Wxy_vec)
        outxy = outxy.reshape(2, -1)

        outX = self.F2C_op.T @ outxy[0]
        outX = outX.view(dtype=self.dtype).reshape((self.n, self.n))
        out[1][:] = (outX + outX.conj().T) * 0.5

        outY = self.F2C_op.T @ outxy[1]
        outY = outY.view(dtype=self.dtype).reshape((self.n, self.n))
        out[2][:] = (outY + outY.conj().T) * 0.5

        # ====================================================================
        # Inverse Hessian products with respect to Z
        # ====================================================================
        outT = self.Z @ Ht @ self.Z

        work = self.irt2Y_Uyxy.conj().T @ out[1] @ self.irt2Y_Uyxy
        outT += self.rt2Y_Uyxy @ (self.D1yxy_h * work) @ self.rt2Y_Uyxy.conj().T

        work = self.irt2X_Uxyx.conj().T @ out[2] @ self.irt2X_Uxyx
        outT += self.rt2X_Uxyx @ (self.D1xyx_g * work) @ self.rt2X_Uxyx.conj().T

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

        # The inverse Hessian product applied on (Ht, Hx, Hy) for the QRE barrier is
        #     (X, Y) =  M \ (Wx, Wy)
        #         t  =  Z Ht Z + DPhi[(X, Y)]
        # where (Wx, Wy) = (Hx, Hy) + DPhi'[Ht],
        #     M = [ D2xxPhi'[Z^-1]  D2xyPhi'[Z^-1] ] + [ X^1 kron X^-1               ]
        #         [ D2yxPhi'[Z^-1]  D2yyPhi'[Z^-1] ]   [               Y^1 kron Y^-1 ]

        p = A.shape[0]

        # ====================================================================
        # Inverse Hessian products with respect to (X, Y)
        # ====================================================================
        # Compute Wx
        lin.congr_multi(self.work0, self.rt2Y_Uyxy.conj().T, self.At, work=self.work2)
        self.work0 *= self.D1yxy_h
        lin.congr_multi(self.work1, self.irt2Y_Uyxy, self.work0, work=self.work2)
        self.work1 += self.Ax

        # Compute Wy
        lin.congr_multi(self.work3, self.rt2X_Uxyx.conj().T, self.At, work=self.work2)
        self.work3 *= self.D1xyx_g
        lin.congr_multi(self.work0, self.irt2X_Uxyx, self.work3, work=self.work2)
        self.work0 += self.Ay

        # Solve linear system (X, Y) = M \ (Wx, Wy)
        Wx_vec = self.work1.view(dtype=np.float64).reshape((p, -1))
        Wy_vec = self.work0.view(dtype=np.float64).reshape((p, -1))
        self.work15[:self.vn] = lin.x_dot_dense(self.F2C_op, Wx_vec.T)
        self.work15[self.vn:] = lin.x_dot_dense(self.F2C_op, Wy_vec.T)

        sol = lin.cho_solve(self.hess_fact, self.work15)

        # Multiply Axy (H A')xy
        out = lin.dense_dot_x(sol.T, self.A_compact.T).T

        # ====================================================================
        # Inverse Hessian products with respect to Z
        # ====================================================================
        # Compute Z Ht Z
        lin.congr_multi(self.work0, self.Z, self.At, work=self.work3)

        # Compute DxPhi[X]
        # Recover X as matrices from compact vectors
        work = lin.x_dot_dense(self.F2C_op.T, sol[:self.vn])
        self.work1.view(dtype=np.float64).reshape((p, -1))[:] = work.T

        lin.congr_multi(
            self.work2, self.irt2Y_Uyxy.conj().T, self.work1, work=self.work3
        )
        self.work2 *= self.D1yxy_h
        lin.congr_multi(self.work1, self.rt2Y_Uyxy, self.work2, work=self.work3)

        self.work0 += self.work1

        # Compute DyPhi[Y]
        # Recover Y as matrices from compact vectors
        work = lin.x_dot_dense(self.F2C_op.T, sol[self.vn:])
        self.work1.view(dtype=np.float64).reshape((p, -1))[:] = work.T

        lin.congr_multi(
            self.work2, self.irt2X_Uxyx.conj().T, self.work1, work=self.work3
        )
        self.work2 *= self.D1xyx_g
        lin.congr_multi(self.work1, self.rt2X_Uxyx, self.work2, work=self.work3)

        self.work0 += self.work1

        # Multiply At (H A')t
        out += (
            self.At.view(dtype=np.float64).reshape((p, -1))
            @ self.work0.view(dtype=np.float64).reshape((p, -1)).T
        )

        return out

    def third_dir_deriv_axpy(self, out, H, a=True):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.dder3_aux_updated:
            self.update_dder3_aux()

        (Ht, Hx, Hy) = H

        UyxyYHxYUyxy = self.irt2Y_Uyxy.conj().T @ Hx @ self.irt2Y_Uyxy
        UxyxXHyXUxyx = self.irt2X_Uxyx.conj().T @ Hy @ self.irt2X_Uxyx
        UxyxXHxXUxyx = self.irt2X_Uxyx.conj().T @ Hx @ self.irt2X_Uxyx
        UyxyYHyYUyxy = self.irt2Y_Uyxy.conj().T @ Hy @ self.irt2Y_Uyxy

        DxPhiHx = (
            self.rt2Y_Uyxy @ (self.D1yxy_h * UyxyYHxYUyxy) @ self.rt2Y_Uyxy.conj().T
        )
        DyPhiHy = (
            self.rt2X_Uxyx @ (self.D1xyx_g * UxyxXHyXUxyx) @ self.rt2X_Uxyx.conj().T
        )

        D2xxPhiHxHx = grad.scnd_frechet(
            self.D2yxy_h, UyxyYHxYUyxy, UyxyYHxYUyxy, U=self.rt2Y_Uyxy
        )
        D2yyPhiHyHy = grad.scnd_frechet(
            self.D2xyx_g, UxyxXHyXUxyx, UxyxXHyXUxyx, U=self.rt2X_Uxyx
        )

        work = (
            Hx
            @ self.irt2X_Uxyx
            @ (self.D1xyx_g * UxyxXHyXUxyx)
            @ self.rt2X_Uxyx.conj().T
        )
        D2xyPhiHxHy = work + work.conj().T
        D2xyPhiHxHy -= grad.scnd_frechet(
            self.D2xyx_xg, UxyxXHxXUxyx, UxyxXHyXUxyx, U=self.rt2X_Uxyx
        )

        # Third order derivative with respect to T
        work = Ht - DxPhiHx - DyPhiHy
        work = work @ self.inv_Z @ work
        dder3_T = (
            -self.inv_Z
            @ (2 * work + D2xxPhiHxHx + 2 * D2xyPhiHxHy + D2yyPhiHyHy)
            @ self.inv_Z
        )

        # Third order derivative with respect to X
        work = self.rt2Y_Uyxy.conj().T @ -dder3_T @ self.rt2Y_Uyxy
        dder3_X = self.irt2Y_Uyxy @ (self.D1yxy_h * work) @ self.irt2Y_Uyxy.conj().T

        work2 = -2 * self.inv_Z @ (Ht - DxPhiHx - DyPhiHy) @ self.inv_Z
        work = self.rt2Y_Uyxy.conj().T @ work2 @ self.rt2Y_Uyxy
        dder3_X += grad.scnd_frechet(
            self.D2yxy_h, work, UyxyYHxYUyxy, U=self.irt2Y_Uyxy
        )

        work = (
            self.irt2X_Uxyx
            @ (self.D1xyx_g * UxyxXHyXUxyx)
            @ self.rt2X_Uxyx.conj().T
            @ work2
        )
        dder3_X += work + work.conj().T
        work = self.rt2X_Uxyx.conj().T @ work2 @ self.rt2X_Uxyx
        dder3_X -= grad.scnd_frechet(
            self.D2xyx_xg, work, UxyxXHyXUxyx, U=self.irt2X_Uxyx
        )

        dder3_X += grad.thrd_frechet(
            self.Dyxy,
            self.D2yxy_h,
            self.d3h(self.Dyxy),
            self.irt2Y_Uyxy,
            self.UyxyYZYUyxy,
            UyxyYHxYUyxy,
        )

        work = self.rt2Y_Uyxy.conj().T @ self.inv_Z @ Hy @ self.irt2Y_Uyxy
        work += work.conj().T
        dder3_X += 2 * grad.scnd_frechet(
            self.D2yxy_h, work, UyxyYHxYUyxy, U=self.irt2Y_Uyxy
        )
        work = self.rt2Y_Uyxy.conj().T @ self.inv_Z @ self.rt2Y_Uyxy
        dder3_X -= 2 * grad.thrd_frechet(
            self.Dyxy,
            self.D2yxy_xh,
            self.d3xh(self.Dyxy),
            self.irt2Y_Uyxy,
            work,
            UyxyYHyYUyxy,
            UyxyYHxYUyxy,
        )

        work = (
            self.irt2_X
            @ grad.scnd_frechet(self.D2xyx_g, UxyxXHyXUxyx, UxyxXHyXUxyx, U=self.Uxyx)
            @ self.rt2_X
            @ self.inv_Z
        )
        dder3_X += work + work.conj().T
        dder3_X -= grad.thrd_frechet(
            self.Dxyx,
            self.D2xyx_xg,
            self.d3xg(self.Dxyx),
            self.irt2X_Uxyx,
            self.UxyxXZXUxyx,
            UxyxXHyXUxyx,
        )

        dder3_X -= 2 * self.inv_X @ Hx @ self.inv_X @ Hx @ self.inv_X

        # Third order derivative with respect to Y
        work = self.rt2X_Uxyx.conj().T @ -dder3_T @ self.rt2X_Uxyx
        dder3_Y = self.irt2X_Uxyx @ (self.D1xyx_g * work) @ self.irt2X_Uxyx.conj().T

        work2 = -2 * self.inv_Z @ (Ht - DxPhiHx - DyPhiHy) @ self.inv_Z
        work = self.rt2X_Uxyx.conj().T @ work2 @ self.rt2X_Uxyx
        dder3_Y += grad.scnd_frechet(
            self.D2xyx_g, work, UxyxXHyXUxyx, U=self.irt2X_Uxyx
        )

        work = (
            self.irt2Y_Uyxy
            @ (self.D1yxy_h * UyxyYHxYUyxy)
            @ self.rt2Y_Uyxy.conj().T
            @ work2
        )
        dder3_Y += work + work.conj().T
        work = self.rt2Y_Uyxy.conj().T @ work2 @ self.rt2Y_Uyxy
        dder3_Y -= grad.scnd_frechet(
            self.D2yxy_xh, work, UyxyYHxYUyxy, U=self.irt2Y_Uyxy
        )

        dder3_Y += grad.thrd_frechet(
            self.Dxyx,
            self.D2xyx_g,
            self.d3g(self.Dxyx),
            self.irt2X_Uxyx,
            self.UxyxXZXUxyx,
            UxyxXHyXUxyx,
        )

        work = self.rt2X_Uxyx.conj().T @ self.inv_Z @ Hx @ self.irt2X_Uxyx
        work += work.conj().T
        dder3_Y += 2 * grad.scnd_frechet(
            self.D2xyx_g, work, UxyxXHyXUxyx, U=self.irt2X_Uxyx
        )
        work = self.rt2X_Uxyx.conj().T @ self.inv_Z @ self.rt2X_Uxyx
        dder3_Y -= 2 * grad.thrd_frechet(
            self.Dxyx,
            self.D2xyx_xg,
            self.d3xg(self.Dxyx),
            self.irt2X_Uxyx,
            work,
            UxyxXHxXUxyx,
            UxyxXHyXUxyx,
        )

        work = (
            self.irt2_Y
            @ grad.scnd_frechet(self.D2yxy_h, UyxyYHxYUyxy, UyxyYHxYUyxy, U=self.Uyxy)
            @ self.rt2_Y
            @ self.inv_Z
        )
        dder3_Y += work + work.conj().T
        dder3_Y -= grad.thrd_frechet(
            self.Dyxy,
            self.D2yxy_xh,
            self.d3xh(self.Dyxy),
            self.irt2Y_Uyxy,
            self.UyxyYZYUyxy,
            UyxyYHxYUyxy,
        )

        dder3_Y -= 2 * self.inv_Y @ Hy @ self.inv_Y @ Hy @ self.inv_Y

        out[0][:] += dder3_T * a
        out[1][:] += dder3_X * a
        out[2][:] += dder3_Y * a

        return out

    # ========================================================================
    # Auxilliary functions
    # ========================================================================
    def congr_aux(self, A):
        assert not self.congr_aux_updated

        p = A.shape[0]

        if sp.sparse.issparse(A):
            A = A.tocsr()        

        self.Ax_compact = (self.F2C_op @ A[:, self.idx_X].T).T
        self.Ay_compact = (self.F2C_op @ A[:, self.idx_Y].T).T
        if sp.sparse.issparse(A):
            self.A_compact = sp.sparse.hstack(
                (self.Ax_compact, self.Ay_compact), format="coo"
            )
        else:
            self.A_compact = np.hstack((self.Ax_compact, self.Ay_compact))

        if sp.sparse.issparse(A):
            A = A.toarray()

        self.At_vec = np.ascontiguousarray(A[:, self.idx_T])
        self.Ax_vec = np.ascontiguousarray(A[:, self.idx_X])
        self.Ay_vec = np.ascontiguousarray(A[:, self.idx_Y])

        if self.iscomplex:
            self.At = np.array(
                [
                    At_k.reshape((-1, 2))
                    .view(dtype=np.complex128)
                    .reshape((self.n, self.n))
                    for At_k in self.At_vec
                ]
            )
            self.Ax = np.array(
                [
                    Ax_k.reshape((-1, 2))
                    .view(dtype=np.complex128)
                    .reshape((self.n, self.n))
                    for Ax_k in self.Ax_vec
                ]
            )
            self.Ay = np.array(
                [
                    Ay_k.reshape((-1, 2))
                    .view(dtype=np.complex128)
                    .reshape((self.n, self.n))
                    for Ay_k in self.Ay_vec
                ]
            )
        else:
            self.At = np.array([At_k.reshape((self.n, self.n)) for At_k in self.At_vec])
            self.Ax = np.array([Ax_k.reshape((self.n, self.n)) for Ax_k in self.Ax_vec])
            self.Ay = np.array([Ay_k.reshape((self.n, self.n)) for Ay_k in self.Ay_vec])

        self.work0 = np.empty_like(self.At)
        self.work1 = np.empty_like(self.At)
        self.work2 = np.empty_like(self.At)
        self.work3 = np.empty_like(self.At)
        self.work4 = np.empty_like(self.At)
        self.work5 = np.empty_like(self.At)

        self.work6 = np.empty((self.At.shape[::-1]), dtype=self.dtype)

        self.work15 = np.empty((2 * self.vn, p))

        self.congr_aux_updated = True

    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        self.D1yxy_xh = grad.D1_f(self.Dyxy, self.xh(self.Dyxy), self.dxh(self.Dyxy))
        self.D1xyx_xg = grad.D1_f(self.Dxyx, self.xg(self.Dxyx), self.dxg(self.Dxyx))

        self.D2yxy_h = grad.D2_f(self.Dyxy, self.D1yxy_h, self.d2h(self.Dyxy))
        self.D2xyx_g = grad.D2_f(self.Dxyx, self.D1xyx_g, self.d2g(self.Dxyx))
        self.D2yxy_xh = grad.D2_f(self.Dyxy, self.D1yxy_xh, self.d2xh(self.Dyxy))
        self.D2xyx_xg = grad.D2_f(self.Dxyx, self.D1xyx_xg, self.d2xg(self.Dxyx))

        self.hess_aux_updated = True

        return

    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated
        if not self.invhess_aux_aux_updated:
            self.update_invhessprod_aux_aux()

        # Precompute and factorize the Schur complement matrix
        #     M = [ D2xxPhi'[Z^-1]  D2xyPhi'[Z^-1] ] + [ X^1 kron X^-1               ]
        #         [ D2yxPhi'[Z^-1]  D2yyPhi'[Z^-1] ]   [               Y^1 kron Y^-1 ]

        # Make Hxx = (D2xxPhi'[Z^-1] + X^1 kron X^-1) block
        # D2xxPhi'[Z^-1]
        lin.congr_multi(self.work14, self.irt2Y_Uyxy.conj().T, self.E, work=self.work12)
        grad.scnd_frechet_multi(
            self.work11,
            self.D2yxy_h,
            self.work14,
            self.UyxyYZYUyxy,
            U=self.irt2Y_Uyxy,
            work1=self.work12,
            work2=self.work13,
            work3=self.work10,
        )
        # X^1 kron X^-1
        lin.congr_multi(self.work14, self.inv_X, self.E, work=self.work13)
        self.work14 += self.work11
        # Vectorize matrices as compact vectors
        work = self.work14.view(dtype=np.float64).reshape((self.vn, -1))
        Hxx = lin.x_dot_dense(self.F2C_op, work.T)        

        # Make Hyy = (D2yyPhi'[Z^-1] + Y^1 kron Y^-1) block
        # D2yyPhi'[Z^-1]
        lin.congr_multi(self.work14, self.irt2X_Uxyx.conj().T, self.E, work=self.work13)
        grad.scnd_frechet_multi(
            self.work11,
            self.D2xyx_g,
            self.work14,
            self.UxyxXZXUxyx,
            U=self.irt2X_Uxyx,
            work1=self.work12,
            work2=self.work13,
            work3=self.work10,
        )
        # Y^1 kron Y^-1
        lin.congr_multi(self.work12, self.inv_Y, self.E, work=self.work13)
        self.work12 += self.work11
        # Vectorize matrices as compact vectors
        work = self.work12.view(dtype=np.float64).reshape((self.vn, -1))
        Hyy = lin.x_dot_dense(self.F2C_op, work.T)

        # Make Hyx = D2yxPhi'[Z^-1] block
        # Make -D2(xg) component
        grad.scnd_frechet_multi(
            self.work11,
            self.D2xyx_xg,
            self.work14,
            self.UxyxXZXUxyx,
            U=self.irt2X_Uxyx,
            work1=self.work12,
            work2=self.work13,
            work3=self.work10,
        )
        # Make Dg + Dg' component
        self.work14 *= self.D1xyx_g
        lin.congr_multi(
            self.work12,
            self.irt2X_Uxyx,
            self.work14,
            work=self.work13,
            B=self.inv_Z @ self.rt2X_Uxyx,
        )
        np.add(self.work12, self.work12.conj().transpose(0, 2, 1), out=self.work13)
        self.work13 -= self.work11
        # Vectorize matrices as compact vectors
        work = self.work13.view(dtype=np.float64).reshape((self.vn, -1))
        Hxy = lin.x_dot_dense(self.F2C_op, work.T)

        # Construct Hessian and factorize
        Hxx = (Hxx + Hxx.T) * 0.5
        Hyy = (Hyy + Hyy.T) * 0.5

        self.hess[: self.vn, : self.vn] = Hxx
        self.hess[self.vn :, self.vn :] = Hyy
        self.hess[self.vn :, : self.vn] = Hxy.T
        self.hess[: self.vn, self.vn :] = Hxy

        self.hess_fact = lin.cho_fact(self.hess.copy())
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

    def update_dder3_aux(self):
        assert not self.dder3_aux_updated
        assert self.hess_aux_updated

        self.dder3_aux_updated = True

        return

    def get_central_ray(self):
        # Solve a 3-dimensional system to get central point
        (t, x, y) = (1.0 + self.g(1.0), 1.0, 1.0)

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
            H = np.array(
                [
                    [zi2, -zi2 * dx, -zi2 * dy],
                    [
                        -zi2 * dx,
                        zi2 * dx * dx + zi * d2dx2 + 1 / x / x,
                        zi2 * dx * dy + zi * d2dxdy,
                    ],
                    [
                        -zi2 * dy,
                        zi2 * dx * dy + zi * d2dxdy,
                        zi2 * dy * dy + zi * d2dy2 + 1 / y / y,
                    ],
                ]
            ) + np.eye(3)

            # Perform Newton step
            delta = -np.linalg.solve(H, g)
            decrement = -np.dot(delta, g)

            # Check feasible
            (t1, x1, y1) = (t + delta[0], x + delta[1], y + delta[2])
            if x1 < 0 or y1 < 0 or t1 < x1 * self.g(y1 / x1):
                break

            (t, x, y) = (t1, x1, y1)

            if decrement / 2.0 <= 1e-12:
                break

        return (t, x, y)
