# Copyright (c) 2024, Kerry He, James Saunderson, and Hamza Fawzi

# This Python package QICS is licensed under the MIT license; see LICENSE.md
# file in the root directory or at https://github.com/kerry-he/qics

import numpy as np
import scipy as sp
import qics._utils.linalg as lin
import qics._utils.gradient as grad
from qics.quantum import p_tr, p_tr_multi, i_kr, i_kr_multi
from qics.cones.base import Cone
from qics.vectorize import get_full_to_compact_op


class QuantCondEntr(Cone):
    r"""A class representing a quantum conditional entropy cone defined on
    :math:`k` subsystems, where the :math:`i`-th subsystem has dimension
    :math:`n_i`, and the :math:`i`-th subsystem is being traced out, i.e.,

    .. math::

        \mathcal{QCE}_{\{n_i\}, j} = \text{cl}\{ (t, X) \in \mathbb{R}
        \times \mathbb{H}^{\Pi_in_i}_{++} : t \geq -S(X) + S(\text{tr}_i(X)) \},

    where

    .. math::

        S(X) = -\text{tr}[X \log(X)],

    is the quantum (von Neumann) entropy, and :math:`\text{tr}_i` is the
    partial trace on the :math:`i`-th subsystem.

    Parameters
    ----------
    dims : :obj:`tuple` of :obj:`int` or :obj:`list` of :obj:`int`
        List of dimensions :math:`\{ n_i \}_{i=0}^{k-1}` of the :math:`k`
        subsystems.
    sys : :obj:`int` or :obj:`tuple` of :obj:`int` or :obj:`list` of :obj:`int`
        Which systems are being traced out by the partial trace. Can define
        multiple subsystems to trace out.
    iscomplex : :obj:`bool`
        Whether the matrix :math:`X` is defined over :math:`\mathbb{H}^n`
        (``True``), or restricted to :math:`\mathbb{S}^n` (``False``). The
        default is ``False``.

    See also
    --------
    QuantRelEntr : Quantum relative entropy cone

    Notes
    -----
    The quantum conditional entropy can also be modelled by the quantum
    relative entropy by noting the identity

    .. math::

        S(X \| \mathbb{I} \otimes \text{tr}_1(X))
        = -S(X) + S(\text{tr}_1(X)).

    However, the cone oracles for the quantum conditional entropy cone are
    much more efficient than those for the quantum relative entropy cone,
    so it is recommended to use the quantum conditional entropy cone where
    possible.
    """

    def __init__(self, dims, sys, iscomplex=False):
        if isinstance(sys, int):
            sys = [sys]
        if isinstance(sys, tuple) or isinstance(sys, set):
            sys = list(sys)

        assert all(
            [sys_k < len(dims) for sys_k in sys]
        ), "Invalid subsystems specified, exceeds total number of dimensions provided."

        self.dims = dims
        self.sys = sys
        self.iscomplex = iscomplex

        self.N = np.prod(dims)  # Total dim. of multipartite system
        self.m = np.prod([dims[k] for k in sys])  # Dim. of system traced out
        self.n = self.N // self.m  # Dim. of system after partial trace

        self.nu = 1 + self.N  # Barrier parameter

        if iscomplex:
            self.vn = self.n * self.n
            self.dim = [1, 2 * self.N * self.N]
            self.type = ["r", "h"]
            self.dtype = np.complex128
        else:
            self.vn = self.n * (self.n + 1) // 2
            self.dim = [1, self.N * self.N]
            self.type = ["r", "s"]
            self.dtype = np.float64

        # Update flags
        self.feas_updated = False
        self.grad_updated = False
        self.hess_aux_updated = False
        self.invhess_aux_updated = False
        self.invhess_aux_aux_updated = False
        self.congr_aux_updated = False
        self.dder3_aux_updated = False

        return

    def get_iscomplex(self):
        return self.iscomplex

    def get_init_point(self, out):
        a = self.N * np.log(self.m) ** 2 + 1
        b = -(2.0 + (1 + self.N) * self.N * np.log(self.m) ** 2)

        t0 = np.sqrt((-b - np.sqrt(b * b - 4 * a)) / (2 * a))
        x0 = np.sqrt((1 + self.N - t0 * t0) / self.N)

        point = [np.array([[t0]]), np.eye(self.N, dtype=self.dtype) * x0]

        self.set_point(point, point)

        out[0][:] = point[0]
        out[1][:] = point[1]

        return out

    def get_feas(self):
        if self.feas_updated:
            return self.feas

        self.feas_updated = True

        (self.t, self.X) = self.primal
        self.Y = p_tr(self.X, self.dims, self.sys)

        # Check that X and Y are positive definite
        #   Note that Y = pTr(X) should be positive definite if X is,
        #   but check just to be sure that we can safely take logarithms of Y.
        self.Dx, self.Ux = np.linalg.eigh(self.X)
        self.Dy, self.Uy = np.linalg.eigh(self.Y)

        if any(self.Dx <= 0) or any(self.Dy <= 0):
            self.feas = False
            return self.feas

        # Check that t > -S(X) + S(pTr(X))
        self.log_Dx = np.log(self.Dx)
        self.log_Dy = np.log(self.Dy)

        log_X = (self.Ux * self.log_Dx) @ self.Ux.conj().T
        log_X = (log_X + log_X.conj().T) * 0.5
        log_Y = (self.Uy * self.log_Dy) @ self.Uy.conj().T
        log_Y = (log_Y + log_Y.conj().T) * 0.5

        self.log_XY = log_X - i_kr(log_Y, self.dims, self.sys)
        self.z = self.t[0, 0] - lin.inp(self.X, self.log_XY)

        self.feas = self.z > 0
        return self.feas

    def get_val(self):
        return -np.log(self.z) - np.sum(self.log_Dx)

    def update_grad(self):
        assert self.feas_updated
        assert not self.grad_updated

        # Compute gradients of quantum conditional entropy
        # DPhi(X) = log(X) - I ⊗ pTr(X)
        self.DPhi = self.log_XY

        # Compute X^-1
        inv_Dx = np.reciprocal(self.Dx)
        inv_X_rt2 = self.Ux * np.sqrt(inv_Dx)
        self.inv_X = inv_X_rt2 @ inv_X_rt2.conj().T

        # Compute gradient of barrier function
        self.zi = np.reciprocal(self.z)
        self.grad = [-self.zi, self.zi * self.DPhi - self.inv_X]

        self.grad_updated = True

    def hess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        (Ht, Hx) = H
        Hy = p_tr(Hx, self.dims, self.sys)

        UxHxUx = self.Ux.conj().T @ Hx @ self.Ux
        UyHyUy = self.Uy.conj().T @ Hy @ self.Uy

        # Hessian products for quantum conditional entropy
        # D2Phi(X)[Hx] = Ux [log^[1](Dx) .* (Ux' Hx Ux)] Ux'
        #                - I ⊗ [Uy [log^[1](Dy) .* (Uy' pTr(Hx) Uy)] Uy']
        D2PhiH = self.Ux @ (self.D1x_log * UxHxUx) @ self.Ux.conj().T
        work = self.Uy @ (self.D1y_log * UyHyUy) @ self.Uy.conj().T
        D2PhiH -= i_kr(work, self.dims, self.sys)

        # ======================================================================
        # Hessian products with respect to t
        # ======================================================================
        # D2_t F(t, X)[Ht, Hx] = (Ht - D_X Phi(X)[Hx]) / z^2
        out[0][:] = (Ht - lin.inp(self.DPhi, Hx)) * self.zi2

        # ======================================================================
        # Hessian products with respect to X
        # ======================================================================
        # D2_X F(t, X)[Ht, Hx] = -D2_t F(t, X)[Ht, Hx] * DPhi(X)
        #                        + D2Phi(X)[Hx] / z + X^-1 Hx X^-1
        out_X = -out[0] * self.DPhi
        out_X += self.zi * D2PhiH
        out_X += self.inv_X @ Hx @ self.inv_X
        out[1][:] = (out_X + out_X.conj().T) * 0.5

        return out

    def hess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)

        p = A.shape[0]
        lhs = np.zeros((p, sum(self.dim)))

        Work0, Work1 = self.Work0, self.Work1
        Work2, Work3 = self.Work2, self.Work3

        # ======================================================================
        # Hessian products with respect to t
        # ======================================================================
        # D2_t F(t, X)[Ht, Hx] = (Ht - D_X Phi(X)[Hx]) / z^2
        DPhi_vec = self.DPhi.view(np.float64).reshape((-1, 1))
        out_t = self.At - (self.Ax_vec @ DPhi_vec).ravel()
        out_t *= self.zi2
        lhs[:, 0] = out_t

        # ======================================================================
        # Hessian products with respect to X
        # ======================================================================
        # Hessian products for quantum conditional entropy
        # D2Phi(X)[Hx] = Ux [log^[1](Dx) .* (Ux' Hx Ux)] Ux'
        #                - I ⊗ [Uy [log^[1](Dy) .* (Uy' pTr(Hx) Uy)] Uy']
        # Compute first term, i.e., Ux [log^[1](Dx) .* (Ux' Hx Ux)] Ux'
        lin.congr_multi(Work2, self.Ux.conj().T, self.Ax, Work3)
        Work2 *= self.D1x_comb
        lin.congr_multi(Work1, self.Ux, Work2, Work3)
        # Compute second term, i.e., I ⊗ [Uy [log^[1](Dy) .* (Uy' Hy Uy)] Uy']
        p_tr_multi(self.work1, self.Ax, self.dims, self.sys)  # Compute pTr(Ax)
        lin.congr_multi(self.work2, self.Uy.conj().T, self.work1, self.work3)
        self.work2 *= self.D1y_log * self.zi
        lin.congr_multi(self.work1, self.Uy, self.work2, self.work3)
        i_kr_multi(Work0, self.work1, self.dims, self.sys)
        # Subtract the two terms to obtain D2Phi(X)[Hx]
        Work1 -= Work0

        # Hessian product of barrier function
        # D2_X F(t, X)[Ht, Hx] = -D2_t F(t, X)[Ht, Hx] * DPhi(X)
        #                        + D2Phi(X)[Hx] / z + X^-1 Hx X^-1
        np.outer(out_t, self.DPhi, out=Work0.reshape(p, -1))
        Work1 -= Work0
        out_X = Work1.reshape((p, -1)).view(np.float64)
        lhs[:, 1:] = out_X

        # Multiply A (H A')
        return lin.dense_dot_x(lhs, A.T)

    def invhess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        # Computes inverse Hessian product of the QCE barrier with a single vector (Ht, Hx)
        # See invhess_congr() for additional comments

        (Ht, Hx) = H
        Wx = Hx + Ht * self.DPhi

        # ======================================================================
        # Inverse Hessian products with respect to X
        # ======================================================================
        # X = [D2S(X)^-1 - D2S(X)^-1 pTr' N^-1 pTr D2S(X)^-1] Wx
        # Apply D2S(X)^-1 = (Ux ⊗ Ux) log^[1](Dx) (Ux' ⊗ Ux')
        UxWxUx = self.Ux.conj().T @ Wx @ self.Ux
        Hinv_x = self.Ux @ (self.D1x_comb_inv * UxWxUx) @ self.Ux.conj().T
        work = -p_tr(Hinv_x, self.dims, self.sys)

        # Solve linear system N \ ( ... )
        # Convert matrices to compact real vectors
        temp_vec = work.view(np.float64).reshape((-1, 1))
        temp_vec = self.F2C_op @ temp_vec
        # Solve system
        temp_vec = lin.cho_solve(self.hess_schur_fact, temp_vec)
        # Expand compact real vectors back into full matrices
        temp_vec = self.F2C_op.T @ temp_vec
        work = temp_vec.T.view(self.dtype).reshape((self.n, self.n))

        # Apply pTr' = IKr
        temp = i_kr(work, self.dims, self.sys)
        # Apply D2S(X)^-1 = (Ux ⊗ Ux) log^[1](Dx) (Ux' ⊗ Ux')
        temp = self.Ux.conj().T @ temp @ self.Ux
        out_X = Hinv_x - self.Ux @ (self.D1x_comb_inv * temp) @ self.Ux.conj().T
        out[1][:] = (out_X + out_X.conj().T) * 0.5

        # ======================================================================
        # Inverse Hessian products with respect to t
        # ======================================================================
        # z^2 Ht + <DPhi(X), X>
        out[0][:] = Ht * self.z2 + lin.inp(out_X, self.DPhi)

        return out

    def invhess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)

        # The inverse Hessian product applied on (Ht, Hx) for the QCE barrier is
        #     X = M \ Wx
        #     t = z^2 Ht + <DPhi(X), X>
        # where Wx = Hx + Ht DPhi(X)
        #     M = 1/z D2S(X) - 1/z pTr' D2S(pTr(X)) pTr + X^-1 ⊗ X^-1
        #       = (Ux ⊗ Ux) (1/z log + inv)^[1](Dx) (Ux' ⊗ Ux')
        #         - 1/z pTr' (Uy ⊗ Uy) log^[1](Dy) (Uy' ⊗ Uy') pTr
        #
        # Treating [pTr' D2S(pTr(X)) pTr] as a low-rank perturbation of D2S(X),
        # we can solve linear systems with M by using the matrix inversion lemma
        #     X = [D2S(X)^-1 - D2S(X)^-1 pTr' N^-1 pTr D2S(X)^-1] Wx
        # where
        #     N = 1/z (Uy ⊗ Uy) [log^[1](Dy)]^-1 (Uy' ⊗ Uy')
        #         - pTr (Ux ⊗ Ux) [(1/z log + inv)^[1](Dx)]^-1  (Ux' ⊗ Ux') pTr'

        p = A.shape[0]
        lhs = np.zeros((p, sum(self.dim)))

        Work0, Work1, work1 = self.Work0, self.Work1, self.work1
        Work2, Work3 = self.Work2, self.Work3

        # ======================================================================
        # Inverse Hessian products with respect to X
        # ======================================================================
        # X = [D2S(X)^-1 - D2S(X)^-1 pTr' N^-1 pTr D2S(X)^-1] Wx
        # Compute Wx = Hx + Ht DPhi(X)
        np.outer(self.At, self.DPhi, out=Work2.reshape((p, -1)))
        np.add(self.Ax, Work2, out=Work0)

        # Apply D2S(X)^-1 = (Ux ⊗ Ux) log^[1](Dx) (Ux' ⊗ Ux')
        lin.congr_multi(Work2, self.Ux.conj().T, Work0, Work3)
        Work2 *= self.D1x_comb_inv
        lin.congr_multi(Work0, self.Ux, Work2, Work3)
        # Apply pTr
        p_tr_multi(work1, Work0, self.dims, self.sys)
        work1 *= -1

        # Solve linear system N \ ( ... )
        # Convert matrices to compact real vectors
        work = work1.view(np.float64).reshape((p, -1)).T
        work = lin.x_dot_dense(self.F2C_op, work)
        # Solve system
        work = lin.cho_solve(self.hess_schur_fact, work)
        # Expand compact real vectors back into full matrices
        work = lin.x_dot_dense(self.F2C_op.T, work)
        work1.view(np.float64).reshape((p, -1))[:] = work.T

        # Apply pTr' = iKr
        i_kr_multi(Work1, work1, self.dims, self.sys)
        # Apply D2S(X)^-1 = (Ux ⊗ Ux) log^[1](Dx) (Ux' ⊗ Ux')
        lin.congr_multi(Work2, self.Ux.conj().T, Work1, Work3)
        Work2 *= self.D1x_comb_inv
        lin.congr_multi(Work1, self.Ux, Work2, Work3)

        # Subtract previous expression from D2S(X)^-1 Wx to get X
        Work0 -= Work1
        out_X = Work0.reshape((p, -1)).view(np.float64)
        lhs[:, 1:] = out_X

        # ======================================================================
        # Inverse Hessian products with respect to t
        # ======================================================================
        # z^2 Ht + <DPhi(X), X>
        DPhi_vec = self.DPhi.view(np.float64).reshape((-1, 1))
        out_t = self.z2 * self.At
        out_t += (out_X @ DPhi_vec).ravel()
        lhs[:, 0] = out_t

        # Multiply A (H A')
        return lin.dense_dot_x(lhs, A.T)

    def third_dir_deriv_axpy(self, out, H, a=True):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.dder3_aux_updated:
            self.update_dder3_aux()

        (Ht, Hx) = H
        Hy = p_tr(Hx, self.dims, self.sys)

        UxHxUx = self.Ux.conj().T @ Hx @ self.Ux
        UyHyUy = self.Uy.conj().T @ Hy @ self.Uy

        # Quantum conditional entropy Hessians
        D2PhiH = self.Ux @ (self.D1x_log * UxHxUx) @ self.Ux.conj().T
        work = self.Uy @ (self.D1y_log * UyHyUy) @ self.Uy.conj().T
        D2PhiH -= i_kr(work, self.dims, self.sys)

        # Quantum conditional entropy third order derivatives
        D3PhiHH = grad.scnd_frechet(self.D2x_log, UxHxUx, UxHxUx, self.Ux)
        work = grad.scnd_frechet(self.D2y_log, UyHyUy, UyHyUy, self.Uy)
        D3PhiHH -= i_kr(work, self.dims, self.sys)

        # Third derivative of barrier
        DPhiH = lin.inp(self.DPhi, Hx)
        D2PhiHH = lin.inp(D2PhiH, Hx)
        chi = Ht - DPhiH

        dder3_t = -2 * (self.zi**3) * (chi**2) - (self.zi**2) * D2PhiHH

        dder3_X = -dder3_t * self.DPhi
        dder3_X -= 2 * (self.zi**2) * chi * D2PhiH
        dder3_X += self.zi * D3PhiHH
        dder3_X -= 2 * self.inv_X @ Hx @ self.inv_X @ Hx @ self.inv_X
        dder3_X = (dder3_X + dder3_X.conj().T) * 0.5

        out[0][:] += dder3_t * a
        out[1][:] += dder3_X * a

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
        self.Ax_vec = A[:, 1:]

        if sp.sparse.issparse(A):
            A = A.toarray()
        Ax_dense = np.ascontiguousarray(A[:, 1:])
        self.At = A[:, 0]
        self.Ax = np.array([vec_to_mat(Ax_k, iscomplex) for Ax_k in Ax_dense])

        # Preallocate matrices we will need when performing these congruences
        self.Work0 = np.empty_like(self.Ax)
        self.Work1 = np.empty_like(self.Ax)
        self.Work2 = np.empty_like(self.Ax)
        self.Work3 = np.empty_like(self.Ax)

        p = A.shape[0]
        self.work1 = np.empty((p, self.n, self.n), dtype=self.dtype)
        self.work2 = np.empty((p, self.n, self.n), dtype=self.dtype)
        self.work3 = np.empty((p, self.n, self.n), dtype=self.dtype)

        self.congr_aux_updated = True

    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        D1x_inv = np.reciprocal(np.outer(self.Dx, self.Dx))
        self.D1x_log = grad.D1_log(self.Dx, self.log_Dx)
        self.D1x_comb = self.zi * self.D1x_log + D1x_inv

        self.D1y_log = grad.D1_log(self.Dy, self.log_Dy)

        # Preparing other required variables
        self.zi2 = self.zi * self.zi

        self.hess_aux_updated = True

    def update_dder3_aux(self):
        assert not self.dder3_aux_updated
        assert self.hess_aux_updated

        self.D2x_log = grad.D2_log(self.Dx, self.D1x_log)
        self.D2y_log = grad.D2_log(self.Dy, self.D1y_log)

        self.dder3_aux_updated = True

    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated
        if not self.invhess_aux_aux_updated:
            self.update_invhessprod_aux_aux()

        # Precompute and factorize the matrix
        #     N = z (Uy ⊗ Uy) [log^[1](Dy)]^-1 (Uy' ⊗ Uy')
        #         - pTr (Ux ⊗ Ux) [(1/z log + inv)^[1](Dx)]^-1  (Ux' ⊗ Ux') pTr'
        # which we will need to solve linear systems with the Hessian of our
        # barrier function

        self.z2 = self.z * self.z
        self.D1x_comb_inv = np.reciprocal(self.D1x_comb)

        work6, work7, work8 = self.work6, self.work7, self.work8
        Work6, Work7 = self.Work6, self.Work7
        Work8, Work9 = self.Work8, self.Work9

        # ======================================================================
        # Get first term, i.e., [z (Uy ⊗ Uy) [log^[1](Dy)]^-1 (Uy' ⊗ Uy')]
        # ======================================================================
        # Begin with (Uy' ⊗ Uy')
        lin.congr_multi(work8, self.Uy.conj().T, self.E, work=work7)
        # Apply z [log^[1](Dy)]^-1
        work8 *= self.z * np.reciprocal(self.D1y_log)
        # Apply (Uy ⊗ Uy)
        lin.congr_multi(work6, self.Uy, work8, work=work7)

        # ======================================================================
        # Get second term, i.e., [pTr (Ux ⊗ Ux) ...]
        # ======================================================================
        # Begin with [(Ux' ⊗ Ux') pTr']
        # Permute columns of Ux' so subsystems we are tracing out are in front
        temp = self.Ux.T.reshape(self.N, *self.dims)
        temp = np.transpose(temp, self.reordered_dims)
        # Obtain groups of columns of Ux' corresponding to
        #   Ux' (I ⊗ Eij) Ux = (Ux' [I ⊗ ei]) ([I ⊗ ej'] Ux)
        temp = temp.reshape(self.N, self.m, self.n)
        lhs = np.copy(temp.conj().transpose(2, 0, 1))
        rhs = np.copy(temp.transpose(2, 1, 0))
        # Use these groups of columns of Ux' to compute [(Ux' ⊗ Ux') pTr']
        # Compute the entries corresponding to Eii first
        np.matmul(lhs, rhs, out=Work9)
        Work8[self.diag_idxs] = Work9
        # Now compute entries corresponding to Eij for i =/= j
        rhs *= np.sqrt(0.5)
        t = 0
        for j in range(self.n):
            np.matmul(lhs[j], rhs[:j], out=Work9[:j])
            Work9_T = Work9[:j].conj().transpose(0, 2, 1)
            if self.iscomplex:
                # Get symmetric and skew-symmetric matrices corresponding to
                # real and complex basis vectors
                np.add(Work9[:j], Work9_T, out=Work8[t : t + 2 * j : 2])
                np.subtract(Work9[:j], Work9_T, out=Work8[t + 1 : t + 2 * j + 1 : 2])
                Work8[t + 1 : t + 2 * j + 1 : 2] *= -1j
                t += 2 * j + 1
            else:
                # Symmeterize matrix
                np.add(Work9[:j], Work9_T, out=Work8[t : t + j])
                t += j + 1

        # Apply [(1/z log + inv)^[1](Dx)]^-1/2
        Work8 *= self.D1x_comb_inv
        # Apply pTr (Ux ⊗ Ux)
        lin.congr_multi(Work6, self.Ux, Work8, work=Work7)
        p_tr_multi(work7, Work6, self.dims, self.sys)

        # Subtract the two terms to obtain Schur complement then Cholesky factor
        work6 -= work7
        work = work6.view(np.float64).reshape((self.vn, -1))
        hess_schur = lin.x_dot_dense(self.F2C_op, work.T)
        self.hess_schur_fact = lin.cho_fact(hess_schur)

        self.invhess_aux_updated = True

    def update_invhessprod_aux_aux(self):
        assert not self.invhess_aux_aux_updated

        self.precompute_computational_basis()

        if self.iscomplex:
            self.diag_idxs = np.cumsum([i for i in range(3, 2 * self.n + 1, 2)])
            self.diag_idxs = np.append(0, self.diag_idxs)
        else:
            self.diag_idxs = np.cumsum([i for i in range(2, self.n + 1, 1)])
            self.diag_idxs = np.append(0, self.diag_idxs)

        not_sys = list(set(range(len(self.dims))) - set(self.sys))
        reordered_dims = self.sys + not_sys
        self.reordered_dims = [0] + [k + 1 for k in reordered_dims]

        self.F2C_op = get_full_to_compact_op(self.n, self.iscomplex)

        self.work6 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work7 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work8 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)

        self.Work6 = np.empty((self.vn, self.N, self.N), dtype=self.dtype)
        self.Work7 = np.empty((self.vn, self.N, self.N), dtype=self.dtype)
        self.Work8 = np.empty((self.vn, self.N, self.N), dtype=self.dtype)
        self.Work9 = np.empty((self.n, self.N, self.N), dtype=self.dtype)

        self.invhess_aux_aux_updated = True
