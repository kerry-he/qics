import numpy as np
import scipy as sp
import qics._utils.linalg as lin
import qics._utils.gradient as grad
from qics.cones.base import Cone, get_central_ray_entr


class QuantEntr(Cone):
    r"""A class representing a (homogenized) quantum entropy cone

    .. math::

        \mathcal{QE}_{n} = \text{cl}\{ (t, u, X) \in \mathbb{R} \times
        \mathbb{R}_{++} \times \mathbb{H}^n_{++} : t \geq -u S(u^{-1}X) \},

    where

    .. math::

        S(X) = -\text{tr}[X \log(X)],

    is the quantum (von Neumann) entropy function.

    Parameters
    ----------
    n : :obj:`int`
        Dimension of the matrix :math:`X`.
    iscomplex : :obj:`bool`
        Whether the matrix :math:`X` is defined over :math:`\mathbb{H}^n`
        (``True``), or restricted to :math:`\mathbb{S}^n` (``False``). The
        default is ``False``.

    See also
    --------
    ClassEntr : (Homogenized) classical entropy cone
    QuantRelEntr : Quantum relative entropy cone

    Notes
    -----
    The epigraph of the quantum entropy can be obtained by enforcing the
    linear constraint :math:`u=1`.
    """

    def __init__(self, n, iscomplex=False):
        # Dimension properties
        self.n = n
        self.iscomplex = iscomplex

        self.nu = 2 + self.n  # Barrier parameter

        if iscomplex:
            self.dim = [1, 1, 2 * n * n]
            self.type = ["r", "r", "h"]
            self.dtype = np.complex128
        else:
            self.dim = [1, 1, n * n]
            self.type = ["r", "r", "s"]
            self.dtype = np.float64

        # Update flags
        self.feas_updated = False
        self.grad_updated = False
        self.hess_aux_updated = False
        self.invhess_aux_updated = False
        self.congr_aux_updated = False
        self.dder3_aux_updated = False

        return

    def get_iscomplex(self):
        return self.iscomplex

    def get_init_point(self, out):
        (t0, u0, x0) = get_central_ray_entr(self.n)

        point = [
            np.array([[t0]]),
            np.array([[u0]]),
            np.eye(self.n, dtype=self.dtype) * x0,
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

        (self.t, self.u, self.X) = self.primal

        # Check that u and X are positive (definite)
        self.Dx, self.Ux = np.linalg.eigh(self.X)

        if (self.u <= 0) or any(self.Dx <= 0):
            self.feas = False
            return self.feas

        # Check that t > -u S(X/u) = tr[X log(X)] - tr[X] log(u)
        self.trX = np.trace(self.X).real
        self.log_Dx = np.log(self.Dx)
        self.log_u = np.log(self.u[0, 0])

        entr_X = lin.inp(self.Dx, self.log_Dx)
        entr_Xu = self.trX * self.log_u
        self.z = self.t[0, 0] - (entr_X - entr_Xu)

        self.feas = self.z > 0
        return self.feas

    def get_val(self):
        return -np.log(self.z) - np.sum(self.log_Dx) - self.log_u

    def update_grad(self):
        assert self.feas_updated
        assert not self.grad_updated

        # Compute gradients of quantum entropy
        # D_u S(u, X) = -tr[X] / u
        self.ui = np.reciprocal(self.u)
        self.DPhiu = -self.trX * self.ui
        # D_X S(u, X) = log(X) + (1 - log(u)) I
        log_X = (self.Ux * self.log_Dx) @ self.Ux.conj().T
        log_X = (log_X + log_X.conj().T) * 0.5
        self.DPhiX = log_X + np.eye(self.n) * (1.0 - self.log_u)

        # Compute X^-1
        inv_Dx = np.reciprocal(self.Dx)
        inv_X_rt2 = self.Ux * np.sqrt(inv_Dx)
        self.inv_X = inv_X_rt2 @ inv_X_rt2.conj().T

        # Compute gradient of barrier function
        self.zi = np.reciprocal(self.z)

        self.grad = [
            -self.zi,
            self.zi * self.DPhiu - self.ui,
            self.zi * self.DPhiX - self.inv_X,
        ]

        self.grad_updated = True

    def hess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        (Ht, Hu, Hx) = H

        UxHxUx = self.Ux.conj().T @ Hx @ self.Ux

        # Hessian product of quantum entropy
        # D2_uu S(u, X) [Hu] = tr[X] Hu / u^2
        D2PhiuuH = self.trX * Hu * self.ui2
        # D2_uX S(u, X) [Hx] = -tr[Hx] / u
        D2PhiuXH = -np.trace(Hx).real * self.ui
        # D2_Xu S(u, X) [Hu] = -I Hu / u
        D2PhiXuH = -np.eye(self.n) * Hu * self.ui
        # D2_XX S(u, X) [Hx] =  Ux [log^[1](Dx) .* (Ux' Hx Ux)] Ux'
        D2PhiXXH = self.Ux @ (self.D1x_log * UxHxUx) @ self.Ux.conj().T

        # ======================================================================
        # Hessian products with respect to t
        # ======================================================================
        # D2_t F(t, u, X)[Ht, Hu, Hx]
        #         = (Ht - D_u S(u, X)[Hu] - D_X S(u, X)[Hx]) / z^2
        out[0][:] = (Ht - self.DPhiu * Hu - lin.inp(self.DPhiX, Hx)) * self.zi2

        # ======================================================================
        # Hessian products with respect to u
        # ======================================================================
        # Hessian product of barrier function
        # D2_u F(t, u, X)[Ht, Hu, Hx]
        #         = -D2_t F(t, u, X)[Ht, Hu, Hx] * D_u S(u, X)
        #           + (D2_uu S(u, X)[Hu] + D2_uX S(u, X)[Hx]) / z
        #           + Hu / u^2
        out_u = -out[0] * self.DPhiu
        out_u += self.zi * (D2PhiuuH + D2PhiuXH)
        out_u += Hu * self.ui2
        out[1][:] = out_u

        # ======================================================================
        # Hessian products with respect to X
        # ======================================================================
        # D2_X F(t, u, X)[Ht, Hu, Hx]
        #         = -D2_t F(t, u, X)[Ht, Hu, Hx] * D_X S(u, X)
        #           + (D2_Xu S(u, X)[Hu] + D2_XX S(u, X)[Hx]) / z
        #           + X^-1 Hx X^-1
        out_X = -out[0] * self.DPhiX
        out_X += self.zi * (D2PhiXuH + D2PhiXXH)
        out_X += self.inv_X @ Hx @ self.inv_X
        out_X = (out_X + out_X.conj().T) * 0.5
        out[2][:] = out_X

        return out

    def hess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)

        p = A.shape[0]
        lhs = np.zeros((p, sum(self.dim)))

        work1, work2, work3 = self.work1, self.work2, self.work3

        # ======================================================================
        # Hessian products with respect to t
        # ======================================================================
        # D2_t F(t, u, X)[Ht, Hu, Hx]
        #         = (Ht - D_u S(u, X)[Hu] - D_X S(u, X)[Hx]) / z^2
        DPhiX_vec = self.DPhiX.view(np.float64).reshape((-1, 1))
        out_t = self.At - (self.Ax_vec @ DPhiX_vec).ravel()
        out_t -= self.Au * self.DPhiu[0, 0]
        out_t *= self.zi2

        lhs[:, 0] = out_t

        # ======================================================================
        # Hessian products with respect to u
        # ======================================================================
        # Hessian products for quantum entropy
        # D2_uu S(u, X) [Hu] = tr[X] Hu / u^2
        D2PhiuuH = (self.trX * self.ui2) * self.Au
        # D2_uX S(u, X) [Hx] = -tr[Hx] / u
        D2PhiuXH = np.trace(self.Ax, axis1=1, axis2=2).real * self.ui

        # Hessian product of barrier function
        # D2_u F(t, u, X)[Ht, Hu, Hx]
        #         = -D2_t F(t, u, X)[Ht, Hu, Hx] * D_u S(u, X)
        #           + (D2_uu S(u, X)[Hu] + D2_uX S(u, X)[Hx]) / z
        #           + Hu / u^2
        out_u = -out_t * self.DPhiu
        out_u += self.zi * (D2PhiuuH - D2PhiuXH)
        out_u += self.Au * self.ui2

        lhs[:, 1] = out_u

        # ====================================================================
        # Hessian products with respect to X
        # ====================================================================
        # Hessian products for quantum entropy
        # D2_XX S(u, X) [Hx] =  Ux [log^[1](Dx) .* (Ux' Hx Ux)] Ux'
        lin.congr_multi(self.work1, self.Ux.conj().T, self.Ax, work2)
        work1 *= self.D1x_comb
        lin.congr_multi(work3, self.Ux, work1, work2)
        # D2_Xu S(u, X) [Hu] = -I Hu / u
        work = (self.zi * self.ui[0, 0]) * self.Au.reshape(-1, 1)
        work3[:, range(self.n), range(self.n)] -= work

        # Hessian product of barrier function
        # D2_X F(t, u, X)[Ht, Hu, Hx]
        #         = -D2_t F(t, u, X)[Ht, Hu, Hx] * D_X S(u, X)
        #           + (D2_Xu S(u, X)[Hu] + D2_XX S(u, X)[Hx]) / z
        #           + X^-1 Hx X^-1
        np.outer(out_t, self.DPhiX, out=work1.reshape((p, -1)))
        work3 -= work1

        lhs[:, 2:] = work3.reshape((p, -1)).view(np.float64)

        # Multiply A (H A')
        return lin.dense_dot_x(lhs, A.T)

    def invhess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        (Ht, Hu, Hx) = H

        Wu = Hu + Ht * self.DPhiu
        Wx = Hx + Ht * self.DPhiX

        work = self.Ux.conj().T @ Wx @ self.Ux
        N_inv_Wx = self.Ux @ (self.D1x_comb_inv * work) @ self.Ux.conj().T
        N_inv_Wx = (N_inv_Wx + N_inv_Wx.conj().T) * 0.5

        # ======================================================================
        # Inverse Hessian products with respect to u
        # ======================================================================
        # u = (Wu + tr[N \ Wx] / z) / ((1 + tr[X]/z) / u^2 - tr[N \ I] / z^2)
        out_u = Wu * self.uz2 + np.trace(N_inv_Wx).real * self.uz
        out_u /= (self.z2 + self.trX * self.z) - self.tr_N_inv_I
        out[1][:] = out_u

        # ======================================================================
        # Inverse Hessian products with respect to X
        # ======================================================================
        # X = (N \ Wx) + u / z (N \ I)
        out_X = N_inv_Wx + (out_u / self.uz) * self.N_inv_I
        out[2][:] = out_X

        # ======================================================================
        # Inverse Hessian products with respect to t
        # ======================================================================
        # t = z^2 Ht + <DS(u, X), (u, X)>
        out_t = self.z2 * Ht + self.DPhiu * out_u + lin.inp(self.DPhiX, out_X)
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

        # The inverse Hessian product applied on (Ht, Hu, Hx) for the QE barrier
        # is
        #     (u, X) =  M \ (Wu, Wx)
        #         t  =  z^2 Ht + <DPhi(u, X), (u, X)>
        # where (Wu, Wx) = [(Hu, Hx) + Ht DPhi(u, X)]
        #     M = [ (1 + tr[X]/z) / u^2   -vec(I)' / zu ]
        #         [    -vec(I) / zu              N      ]
        # and
        #     N = (Ux kron Ux) (1/z log + inv)^[1](Dx) (Ux' kron Ux')
        #
        # To solve linear systems with M, we simplify it by doing block
        # elimination, in which case we get
        #     u = (Wu + tr[N \ Wx] / z) / ((1 + tr[X]/z) / u^2 - tr[N \ I] / z^2)
        #     X = (N \ Wx) + u / z (N \ I)

        p = A.shape[0]
        lhs = np.zeros((p, sum(self.dim)))

        work1, work2, work3 = self.work1, self.work2, self.work3

        # Compute Wu
        Wu = self.Au + self.At * self.DPhiu[0, 0]
        # Compute Wx
        np.outer(self.At, self.DPhiX, out=work2.reshape((p, -1)))
        np.add(self.Ax, work2, out=work1)

        # Compute N \ Wx
        lin.congr_multi(work2, self.Ux.conj().T, work1, work3)
        work2 *= self.D1x_comb_inv
        lin.congr_multi(work1, self.Ux, work2, work3)

        # ======================================================================
        # Inverse Hessian products with respect to u
        # ======================================================================
        # u = (Wu + tr[N \ Wx] / z) / ((1 + tr[X]/z) / u^2 - tr[N \ I] / z^2)
        tr_N_inv_Wx = np.trace(work1, axis1=1, axis2=2).real
        out_u = Wu * self.uz2[0, 0] + tr_N_inv_Wx * self.uz[0, 0]
        out_u /= (self.z2 + self.trX * self.z) - self.tr_N_inv_I
        lhs[:, 1] = out_u

        # ======================================================================
        # Inverse Hessian products with respect to X
        # ======================================================================
        # X = (N \ Wx) + u / z (N \ I)
        np.outer(out_u / self.uz, self.N_inv_I, out=work2.reshape((p, -1)))
        work1 += work2
        out_X = work1.reshape((p, -1)).view(np.float64)
        lhs[:, 2:] = out_X

        # ======================================================================
        # Inverse Hessian products with respect to t
        # ======================================================================
        DPhiX_vec = self.DPhiX.view(np.float64).reshape((-1, 1))

        # t = z^2 Ht + <DS(u, X), (u, X)>
        out_t = self.z2 * self.At
        out_t += out_u * self.DPhiu[0, 0]
        out_t += (out_X @ DPhiX_vec).ravel()
        lhs[:, 0] = out_t

        # Multiply A (H A')
        return lin.dense_dot_x(lhs, A.T)

    def third_dir_deriv_axpy(self, out, H, a=True):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.dder3_aux_updated:
            self.update_dder3_aux()

        (Ht, Hu, Hx) = H

        Hu2 = Hu * Hu
        trHx = np.trace(Hx).real

        chi = (Ht - self.DPhiu * Hu)[0, 0] - lin.inp(self.DPhiX, Hx)
        chi2 = chi * chi

        UxHxUx = self.Ux.conj().T @ Hx @ self.Ux

        # Quantum entropy Hessians
        D2PhiuuH = self.trX * Hu * self.ui2
        D2PhiuXH = -np.trace(Hx).real * self.ui
        D2PhiXuH = -np.eye(self.n) * Hu * self.ui
        D2PhiXXH = self.Ux @ (self.D1x_log * UxHxUx) @ self.Ux.conj().T

        D2PhiuHH = lin.inp(Hu, D2PhiuXH + D2PhiuuH)
        D2PhiXHH = lin.inp(Hx, D2PhiXXH + D2PhiXuH)

        # Quantum entropy third order derivatives
        D3Phiuuu = -2 * Hu2 * self.trX * self.ui3
        D3PhiuXu = Hu * trHx * self.ui2
        D3PhiuuX = D3PhiuXu

        D3PhiXXX = grad.scnd_frechet(self.D2x_log, UxHxUx, UxHxUx, self.Ux)
        D3PhiXuu = (Hu2 * self.ui2) * np.eye(self.n)

        # Third derivatives of barrier
        dder3_t = -2 * self.zi3 * chi2 - self.zi2 * (D2PhiXHH + D2PhiuHH)

        dder3_u = -dder3_t * self.DPhiu
        dder3_u -= 2 * self.zi2 * chi * (D2PhiuXH + D2PhiuuH)
        dder3_u += self.zi * (D3PhiuuX + D3PhiuXu + D3Phiuuu)
        dder3_u -= 2 * Hu2 * self.ui3

        dder3_X = -dder3_t * self.DPhiX
        dder3_X -= 2 * self.zi2 * chi * (D2PhiXXH + D2PhiXuH)
        dder3_X += self.zi * (D3PhiXXX + D3PhiXuu)
        dder3_X -= 2 * self.inv_X @ Hx @ self.inv_X @ Hx @ self.inv_X
        dder3_X = (dder3_X + dder3_X.conj().T) * 0.5

        out[0][:] += dder3_t * a
        out[1][:] += dder3_u * a
        out[2][:] += dder3_X * a

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
        self.Ax_vec = A[:, 2:]

        if sp.sparse.issparse(A):
            A = A.toarray()
        Ax_dense = np.ascontiguousarray(A[:, 2:])
        self.At = A[:, 0]
        self.Au = A[:, 1]
        self.Ax = np.array([vec_to_mat(Ax_k, iscomplex) for Ax_k in Ax_dense])

        # Preallocate matrices we will need when performing these congruences
        self.work1 = np.empty_like(self.Ax)
        self.work2 = np.empty_like(self.Ax)
        self.work3 = np.empty_like(self.Ax)

        self.congr_aux_updated = True

    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        self.zi2 = self.zi * self.zi
        self.ui2 = self.ui * self.ui

        D1x_inv = np.reciprocal(np.outer(self.Dx, self.Dx))
        self.D1x_log = grad.D1_log(self.Dx, self.log_Dx)
        self.D1x_comb = self.zi * self.D1x_log + D1x_inv

        self.hess_aux_updated = True

    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated

        self.z2 = self.z * self.z
        self.uz = self.u * self.z
        self.uz2 = self.uz * self.uz
        self.D1x_comb_inv = np.reciprocal(self.D1x_comb)

        N_inv_I_rt2 = self.Ux * np.sqrt(np.diag(self.D1x_comb_inv))
        self.N_inv_I = N_inv_I_rt2 @ N_inv_I_rt2.conj().T
        self.tr_N_inv_I = np.trace(self.N_inv_I).real

        self.invhess_aux_updated = True

    def update_dder3_aux(self):
        assert not self.dder3_aux_updated
        assert self.hess_aux_updated

        self.zi3 = self.zi * self.zi2
        self.ui3 = self.ui * self.ui2
        self.D2x_log = grad.D2_log(self.Dx, self.D1x_log)

        self.dder3_aux_updated = True
