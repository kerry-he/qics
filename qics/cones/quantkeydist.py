import numpy as np
import scipy as sp
import qics._utils.linalg as lin
import qics._utils.gradient as grad
from qics.cones.base import Cone


class QuantKeyDist(Cone):
    r"""A class representing a quantum key distribution cone

    .. math::

        \mathcal{K}_{\text{qkd}} = \text{cl}\{ (t, X) \in \mathbb{R} \times
        \mathbb{H}^n_{++}:t \geq -S(\mathcal{G}(X)) + S(\mathcal{Z}(\mathcal{G}(X))) \},

    where

    .. math::

        S(X) = -\text{tr}[X \log(X)],

    is the quantum (von Neumann) entropy function,
    :math:`\mathcal{G}:\mathbb{H}^n\rightarrow\mathbb{H}^{mr}` is a positive linear
    map, and :math:`\mathcal{Z}:\mathbb{H}^{mr}\rightarrow\mathbb{H}^{mr}` is a
    pinching map that maps off-diagonal blocks to zero.

    Parameters
    ----------
    G_info : int or list of ndarray
        Defines the linear map :math:`\mathcal{G}`. If ``G_info`` is an ``int``, then
        :math:`\mathcal{G}(X)=X` and this argument specifies the dimension of
        :math:`X`. If ``G_info`` is a ``list`` of ``ndarray``, then this argument
        specifies the list of Kraus operators
        :math:`\{ K_i \in \mathbb{C}^{mr \times n } \}_{i=1}^l` corresponding to
        :math:`\mathcal{G}` such that

        .. math::

            \mathcal{G}(X) = \sum_{i=1}^l K_i X K_i^\dagger.

    Z_info : int or tuple or list(ndarray)
        Defines the linear map :math:`\mathcal{Z}`. There are three ways the user can
        specify this argument. If ``Z_info`` is an ``int``, then this argument specifies
        the block-structure which is being zeroed out, i.e., defines :math:`r` where
        
        .. math::

            \mathcal{Z}(Y) = \sum_{i=1}^r (| i \rangle \langle i | \otimes \mathbb{I}_m) Y (| i \rangle \langle i | \otimes \mathbb{I}_m).

        If ``Z_info`` is a ``list`` of ``ndarray``, then this argument directly
        specifies the Kraus operator corresponding to :math:`\mathcal{Z}`.

    iscomplex : bool
        Whether the matrix is symmetric :math:`X \in \mathbb{S}^n` (False) or
        Hermitian :math:`X \in \mathbb{H}^n` (True). Default is False.
    """

    def __init__(self, G_info, Z_info, iscomplex=False):
        # Process G_info
        if isinstance(G_info, int):
            self.n = G_info  # Input dimension
            self.N = G_info  # Output dimension
            self.K_list_raw = [np.eye(self.N)]
            self.G_is_Id = True
        else:
            self.n = G_info[0].shape[1]
            self.N = G_info[0].shape[0]
            self.K_list_raw = G_info
            self.G_is_Id = False
            if G_info[0].shape[0] == G_info[0].shape[1]:
                if np.linalg.norm(G_info[0] - np.eye(G_info[0].shape[0])) < 1e-10:
                    self.G_is_Id = True

        # Process Z_info
        if isinstance(Z_info, int):
            self.r = Z_info
            self.m = self.N // self.r
            assert self.m * Z_info == self.N
            self.Z_list_raw = [np.zeros((self.N, self.N)) for _ in range(self.r)]
            for k in range(self.r):
                range_k = np.arange(k * self.m, (k + 1) * self.m)
                self.Z_list_raw[k][range_k, range_k] = 1.0
            self.Z_idxs = [
                np.arange(i * self.m, (i + 1) * self.m) for i in range(self.r)
            ]
        elif isinstance(Z_info, tuple):
            (dims, sys) = Z_info
            r = np.meshgrid(*[range(dims[k]) for k in sys])
            r = list(np.array(r).reshape(len(self.subsystems), -1).T)
            self.Z_list_raw = []
            for i in range(len(r)):
                Z_i = np.array([1])
                counter = 0
                for (k, dimk) in enumerate(self.dimensions):
                    if k in self.subsystems:
                        Z_ik = np.zeros(dimk[0])
                        Z_ik[r[i][counter]] = 1
                        Z_i = np.kron(Z_i, Z_ik)
                        counter += 1
                    else:
                        Z_i = np.kron(Z_i, np.ones(dimk[0]))
                self.Z_list_raw += [np.diag(Z_i)]
            self.Z_idxs = [np.where(Z)[0] for Z in Z_info]    
        else:
            self.r = len(Z_info)
            self.m = self.N // self.r
            self.Z_list_raw = Z_info
            self.Z_idxs = [np.where(Z)[0] for Z in Z_info]

        # Dimension properties
        self.nu = 1 + self.n  # Barrier parameter
        self.iscomplex = iscomplex  # Is the problem complex-valued

        self.vn = self.n * self.n if iscomplex else self.n * (self.n + 1) // 2
        self.vm = (
            self.m * self.m if iscomplex else self.m * (self.m + 1) // 2
        )  # Compact dimension of system

        self.dim = [1, self.n * self.n] if (not iscomplex) else [1, 2 * self.n * self.n]
        self.type = ["r", "s"] if (not iscomplex) else ["r", "h"]
        self.dtype = np.float64 if (not iscomplex) else np.complex128

        # Facial reduction
        self.K_list_blk = [facial_reduction(self.K_list_raw)]
        self.ZK_list_blk = [
            facial_reduction([K[Z_idxs_k, :] for K in self.K_list_raw])
            for Z_idxs_k in self.Z_idxs
        ]

        self.Nk = [K_list[0].shape[0] for K_list in self.K_list_blk]
        self.Nzk = [K_list[0].shape[0] for K_list in self.ZK_list_blk]

        # Update flags
        self.feas_updated = False
        self.grad_updated = False
        self.hess_aux_updated = False
        self.invhess_aux_updated = False
        self.dder3_aux_updated = False
        self.congr_aux_updated = False
        self.invhess_aux_aux_updated = False

        if self.G_is_Id:
            self.precompute_mat_vec(self.m)
        else:
            self.precompute_mat_vec()

        return

    def get_iscomplex(self):
        return self.iscomplex

    def get_init_point(self, out):
        KK_blk = [apply_kraus(np.eye(self.n), K_list) for K_list in self.K_list_blk]
        ZKKZ_blk = [
            apply_kraus(np.eye(self.n), ZK_list) for ZK_list in self.ZK_list_blk
        ]

        from qics.quantum import quant_entropy

        entr_KK = sum([quant_entropy(KK) for KK in KK_blk])
        entr_ZKKZ = sum([quant_entropy(ZKKZ) for ZKKZ in ZKKZ_blk])

        f0 = -entr_KK + entr_ZKKZ
        t0 = f0 / 2 + np.sqrt(1 + f0 * f0 / 4)

        point = [np.array([[t0]]), np.eye(self.n, dtype=self.dtype)]

        self.set_point(point, point)

        out[0][:] = point[0]
        out[1][:] = point[1]

        return out

    def get_feas(self):
        if self.feas_updated:
            return self.feas

        self.feas_updated = True

        (self.t, self.X) = self.primal

        # Eigendecomposition of X
        self.Dx, self.Ux = np.linalg.eigh(self.X)
        if any(self.Dx <= 0):
            self.feas = False
            return self.feas

        # Eigendecomposition of G(X)
        self.KX_blk = [apply_kraus(self.X, K_list) for K_list in self.K_list_blk]

        DUkx_blk = [np.linalg.eigh(KX) for KX in self.KX_blk]
        self.Dkx_blk = [DUkx[0] for DUkx in DUkx_blk]
        self.Ukx_blk = [DUkx[1] for DUkx in DUkx_blk]

        if any([any(Dkx <= 0) for Dkx in self.Dkx_blk]):
            self.feas = False
            return self.feas

        # Eigendecomposition of Z(G(X))
        self.ZKX_blk = [apply_kraus(self.X, ZK_list) for ZK_list in self.ZK_list_blk]

        DUzkx_blk = [np.linalg.eigh(ZKX) for ZKX in self.ZKX_blk]
        self.Dzkx_blk = [DUzkx[0] for DUzkx in DUzkx_blk]
        self.Uzkx_blk = [DUzkx[1] for DUzkx in DUzkx_blk]

        if any([any(Dzkx <= 0) for Dzkx in self.Dzkx_blk]):
            self.feas = False
            return self.feas

        # Compute feasibility
        self.log_Dkx_blk = [np.log(D) for D in self.Dkx_blk]
        self.log_Dzkx_blk = [np.log(D) for D in self.Dzkx_blk]

        entr_KX = sum(
            [lin.inp(D, log_D) for (D, log_D) in zip(self.Dkx_blk, self.log_Dkx_blk)]
        )
        entr_ZKX = sum(
            [lin.inp(D, log_D) for (D, log_D) in zip(self.Dzkx_blk, self.log_Dzkx_blk)]
        )

        self.z = self.t[0, 0] - (entr_KX - entr_ZKX)

        self.feas = self.z > 0
        return self.feas

    def get_val(self):
        assert self.feas_updated

        return -np.log(self.z) - np.sum(np.log(self.Dx))

    def update_grad(self):
        assert self.feas_updated
        assert not self.grad_updated

        log_KX = [
            (U * log_D) @ U.conj().T
            for (U, log_D) in zip(self.Ukx_blk, self.log_Dkx_blk)
        ]
        log_ZKX = [
            (U * log_D) @ U.conj().T
            for (U, log_D) in zip(self.Uzkx_blk, self.log_Dzkx_blk)
        ]

        self.K_log_KX = sum(
            [
                apply_kraus(log_X, K_list, adjoint=True)
                for (log_X, K_list) in zip(log_KX, self.K_list_blk)
            ]
        )
        self.ZK_log_ZKX = sum(
            [
                apply_kraus(log_X, K_list, adjoint=True)
                for (log_X, K_list) in zip(log_ZKX, self.ZK_list_blk)
            ]
        )

        self.inv_Dx = np.reciprocal(self.Dx)
        inv_X_rt2 = self.Ux * np.sqrt(self.inv_Dx)
        self.inv_X = inv_X_rt2 @ inv_X_rt2.conj().T

        self.zi = np.reciprocal(self.z)
        self.DPhi = self.K_log_KX - self.ZK_log_ZKX

        self.grad = [
            -self.zi,
            self.zi * self.DPhi - self.inv_X,
        ]

        self.grad_updated = True

    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        self.zi2 = self.zi * self.zi

        self.D1kx_log_blk = [
            grad.D1_log(D, log_D) for (D, log_D) in zip(self.Dkx_blk, self.log_Dkx_blk)
        ]
        self.D1zkx_log_blk = [
            grad.D1_log(D, log_D)
            for (D, log_D) in zip(self.Dzkx_blk, self.log_Dzkx_blk)
        ]

        self.hess_aux_updated = True

        return

    def hess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        # Computes Hessian product of the QKD barrier with a single vector (Ht, Hx)
        # See hess_congr() for additional comments

        (Ht, Hx) = H

        KH_blk = [apply_kraus(Hx, K_list) for K_list in self.K_list_blk]
        ZKH_blk = [apply_kraus(Hx, ZK_list) for ZK_list in self.ZK_list_blk]

        UkKHUk_blk = [U.conj().T @ H @ U for (H, U) in zip(KH_blk, self.Ukx_blk)]
        UkzZKHUkz_blk = [U.conj().T @ H @ U for (H, U) in zip(ZKH_blk, self.Uzkx_blk)]

        # Hessian product of conditional entropy
        D2PhiH = sum(
            [
                apply_kraus(U @ (D1 * UHU) @ U.conj().T, K_list, adjoint=True)
                for (U, D1, UHU, K_list) in zip(
                    self.Ukx_blk, self.D1kx_log_blk, UkKHUk_blk, self.K_list_blk
                )
            ]
        )
        D2PhiH -= sum(
            [
                apply_kraus(U @ (D1 * UHU) @ U.conj().T, K_list, adjoint=True)
                for (U, D1, UHU, K_list) in zip(
                    self.Uzkx_blk, self.D1zkx_log_blk, UkzZKHUkz_blk, self.ZK_list_blk
                )
            ]
        )

        # Hessian product of barrier function
        out[0][:] = (Ht - lin.inp(self.DPhi, Hx)) * self.zi2

        out_X = -out[0] * self.DPhi
        out_X += self.zi * D2PhiH
        out_X += self.inv_X @ Hx @ self.inv_X
        out_X = (out_X + out_X.conj().T) * 0.5
        out[1][:] = out_X

        return out

    def hess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        if self.G_is_Id:
            p = A.shape[0]
            lhs = np.zeros((p, sum(self.dim)))

            # Precompute Hessian products for quantum conditional entropy
            # D2Phi(X)[Hx] =  Ux [log^[1](Dx) .* (Ux'     Hx  Ux)] Ux'
            #              - [Uy [log^[1](Dy) .* (Uy' PTr(Hx) Uy)] Uy'] kron I
            self.Work0 *= 0
            for Z_idxs_k, U, D1_log in zip(
                self.Z_idxs, self.Uzkx_blk, self.D1zkx_log_blk
            ):
                temp = self.Ax[np.ix_(np.arange(p), Z_idxs_k, Z_idxs_k)]
                lin.congr_multi(self.work2, U.conj().T, temp, self.work3)
                self.work2 *= D1_log * self.zi
                lin.congr_multi(self.work1, U, self.work2, self.work3)
                self.Work0[np.ix_(np.arange(p), Z_idxs_k, Z_idxs_k)] = self.work1

            lin.congr_multi(self.Work2, self.Ux.conj().T, self.Ax, self.Work3)
            self.Work2 *= self.D1x_comb
            lin.congr_multi(self.Work1, self.Ux, self.Work2, self.Work3)

            self.Work1 -= self.Work0

            # ====================================================================
            # Hessian products with respect to t
            # ====================================================================
            # D2_tt F(t, X)[Ht] =  Ht / z^2
            # D2_tX F(t, X)[Hx] = -DPhi(X)[Hx] / z^2
            outt = (
                self.At
                - (
                    self.Ax.view(dtype=np.float64).reshape((p, 1, -1))
                    @ self.DPhi.view(dtype=np.float64).reshape((-1, 1))
                ).ravel()
            )
            outt *= self.zi2

            lhs[:, 0] = outt

            # ====================================================================
            # Hessian products with respect to X
            # ====================================================================
            # D2_Xt F(t, X)[Ht] = -Ht DPhi(X) / z^2
            # D2_XX F(t, X)[Hx] =  DPhi(X)[Hx] DPhi(X) / z^2 + D2Phi(X)[Hx] / z + X^-1 Hx X^-1
            np.outer(outt, self.DPhi, out=self.Work0.reshape(p, -1))
            self.Work1 -= self.Work0

            lhs[:, 1:] = self.Work1.reshape((p, -1)).view(dtype=np.float64)

            # Multiply A (H A')
            return lin.dense_dot_x(lhs, A.T)

        else:
            vec = self.At - lin.dense_dot_x(self.DPhi_vec.T, self.Ax_vec.T)
            vec *= self.zi

            temp = lin.dense_dot_x(self.hess, self.Ax_vec.T)
            out = lin.dense_dot_x(temp.T, self.Ax_vec.T)
            out += np.outer(vec, vec)
            return out

    def invhess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        # Computes inverse Hessian product of the QKD barrier with a single vector (Ht, Hx)
        # See invhess_congr() for additional comments

        if self.G_is_Id:
            (Ht, Hx) = H
            Wx = Hx + Ht * self.DPhi

            work = np.zeros((self.r * self.vm))

            # Apply D2S(X)^-1 = (Ux kron Ux) log^[1](Dx) (Ux' kron Ux')
            UxWxUx = self.Ux.conj().T @ Wx @ self.Ux
            Hxx_inv_x = self.Ux @ (self.D1x_comb_inv * UxWxUx) @ self.Ux.conj().T
            for k, Z_idxs_k in enumerate(self.Z_idxs):
                temp = Hxx_inv_x[np.ix_(Z_idxs_k, Z_idxs_k)]
                work[k * self.vm : (k + 1) * self.vm] = (
                    temp.view(dtype=np.float64).reshape(-1)[self.triu_idxs] * self.scale
                )
            work *= -1

            work = lin.cho_solve(self.schur_fact, work)

            Work3 = np.zeros((self.n, self.n), dtype=self.dtype)
            for k, Z_idxs_k in enumerate(self.Z_idxs):
                work_k = work[k * self.vm : (k + 1) * self.vm]
                work_k[self.diag_idxs] *= 0.5
                work_k /= self.scale

                work2_k = np.zeros((self.m, self.m), dtype=self.dtype)
                work2_k.view(dtype=np.float64).reshape(-1)[self.triu_idxs] = work_k
                Work3[np.ix_(Z_idxs_k, Z_idxs_k)] = work2_k
            Work3 += Work3.conj().T

            # Apply D2S(X)^-1 = (Ux kron Ux) log^[1](Dx) (Ux' kron Ux')
            temp = self.Ux.conj().T @ Work3 @ self.Ux
            H_inv_w_x = (
                Hxx_inv_x - self.Ux @ (self.D1x_comb_inv * temp) @ self.Ux.conj().T
            )
            H_inv_w_x = (H_inv_w_x + H_inv_w_x.conj().T) * 0.5

            out[0][:] = Ht * self.z2 + lin.inp(H_inv_w_x, self.DPhi)
            out[1][:] = H_inv_w_x

            return out

        else:
            (Ht, Hx) = H
            work = Hx + Ht * self.DPhi

            # Inverse Hessian products with respect to X
            temp_vec = work.view(dtype=np.float64).reshape((-1, 1))[self.triu_idxs]
            temp_vec *= self.scale.reshape((-1, 1))

            temp_vec = lin.cho_solve(self.hess_fact, temp_vec)

            work.fill(0.0)
            temp_vec[self.diag_idxs] *= 0.5
            temp_vec /= self.scale.reshape((-1, 1))
            work.view(dtype=np.float64).reshape((-1, 1))[self.triu_idxs] = temp_vec
            work += work.conj().T

            out[1][:] = work

            # Inverse Hessian products with respect to t
            out[0][:] = self.z2 * Ht + lin.inp(self.DPhi, out[1])

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
        #         t  =  z^2 Ht + <DPhi(X, Y), (X, Y)>
        # where (Wx, Wy) = [(Hx, Hy) + Ht DPhi(X, Y)],
        #     M = Vxy [ 1/z log^[1](Dx) + Dx^-1 kron Dx^-1  -1/z (Ux'Uy kron Ux'Uy) log^[1](Dy) ]
        #             [-1/z log^[1](Dy) (Uy'Ux kron Uy'Ux)      -1/z Sy + Dy^-1 kron Dy^-1      ] Vxy'
        # and
        #     Vxy = [ Ux kron Ux             ]
        #           [             Uy kron Uy ]
        #
        # To solve linear systems with M, we simplify it by doing block elimination, in which case we get
        #     Uy' Y Uy = S \ ({Uy' Wy Uy} - [1/z log^[1](Dy) (Uy'Ux kron Uy'Ux) (1/z log^[1](Dx) + Dx^-1 kron Dx^-1)^-1 {Ux' Wx Ux}])
        #     Ux' X Ux = -(1/z log^[1](Dx) + Dx^-1 kron Dx^-1)^-1 [{Ux' Wx Ux} + 1/z (Ux'Uy kron Ux'Uy) log^[1](Dy) Y]
        # where S is the Schur complement matrix of M.

        if self.G_is_Id:
            p = A.shape[0]
            lhs = np.zeros((p, sum(self.dim)))

            work = np.zeros((self.r * self.vm, p))

            # ====================================================================
            # Inverse Hessian products with respect to X
            # ====================================================================
            # Compute Wx
            np.outer(self.At, self.DPhi, out=self.Work2.reshape((p, -1)))
            np.add(self.Ax, self.Work2, out=self.Work0)

            # Apply D2S(X)^-1 = (Ux kron Ux) log^[1](Dx) (Ux' kron Ux')
            lin.congr_multi(self.Work2, self.Ux.conj().T, self.Work0, self.Work3)
            self.Work2 *= self.D1x_comb_inv
            lin.congr_multi(self.Work0, self.Ux, self.Work2, self.Work3)
            # Apply PTr
            for k, Z_idxs_k in enumerate(self.Z_idxs):
                temp = self.Work0[np.ix_(np.arange(p), Z_idxs_k, Z_idxs_k)]
                work[k * self.vm : (k + 1) * self.vm] = (
                    temp.view(dtype=np.float64).reshape((p, -1))[:, self.triu_idxs]
                    * self.scale
                ).T
            work *= -1

            # Solve linear system N \ ( ... )
            work = lin.cho_solve(self.schur_fact, work)
            # Expand truncated real vectors back into matrices
            self.Work1.fill(0.0)
            for k, Z_idxs_k in enumerate(self.Z_idxs):
                work_k = work[k * self.vm : (k + 1) * self.vm]
                work_k[self.diag_idxs, :] *= 0.5
                work_k /= self.scale.reshape((-1, 1))

                work2_k = np.zeros((p, self.m, self.m), dtype=self.dtype)
                work2_k.view(dtype=np.float64).reshape((p, -1))[:, self.triu_idxs] = (
                    work_k.T
                )
                self.Work1[np.ix_(np.arange(p), Z_idxs_k, Z_idxs_k)] = work2_k
            self.Work1 += self.Work1.conj().transpose((0, 2, 1))

            # Apply D2S(X)^-1 = (Ux kron Ux) log^[1](Dx) (Ux' kron Ux')
            lin.congr_multi(self.Work2, self.Ux.conj().T, self.Work1, self.Work3)
            self.Work2 *= self.D1x_comb_inv
            lin.congr_multi(self.Work1, self.Ux, self.Work2, self.Work3)

            # Subtract previous expression from D2S(X)^-1 Wx to get X
            self.Work0 -= self.Work1
            lhs[:, 1:] = self.Work0.reshape((p, -1)).view(dtype=np.float64)

            # ====================================================================
            # Inverse Hessian products with respect to t
            # ====================================================================
            outt = self.z2 * self.At
            outt += (
                self.Work0.view(dtype=np.float64).reshape((p, 1, -1))
                @ self.DPhi.view(dtype=np.float64).reshape((-1, 1))
            ).ravel()
            lhs[:, 0] = outt

            # Multiply A (H A')
            return lin.dense_dot_x(lhs, A.T)

        else:
            # ====================================================================
            # Inverse Hessian products with respect to X
            # ====================================================================
            # Compute Wx
            np.outer(self.At, self.DPhi_vec, out=self.work1)
            if sp.sparse.issparse(self.Ax_vec):
                self.work0 = self.work1 + self.Ax_vec
            else:
                np.add(self.Ax_vec, self.work1, out=self.work0)

            # Solve system
            lhsX = lin.cho_solve(self.hess_fact, self.work0.T)

            # ====================================================================
            # Inverse Hessian products with respect to t
            # ====================================================================
            lhst = self.z2 * self.At
            lhst += (lhsX.T @ self.DPhi_vec).ravel()

            # Multiply A (H A')
            return np.outer(lhst, self.At) + lin.dense_dot_x(lhsX.T, self.Ax_vec.T)

    def third_dir_deriv_axpy(self, out, H, a=True):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.dder3_aux_updated:
            self.update_dder3_aux()

        (Ht, Hx) = H

        KH_blk = [apply_kraus(Hx, K_list) for K_list in self.K_list_blk]
        ZKH_blk = [apply_kraus(Hx, ZK_list) for ZK_list in self.ZK_list_blk]

        UkKHUk_blk = [U.conj().T @ H @ U for (H, U) in zip(KH_blk, self.Ukx_blk)]
        UkzZKHUkz_blk = [U.conj().T @ H @ U for (H, U) in zip(ZKH_blk, self.Uzkx_blk)]

        # Quantum conditional entropy oracles
        D2PhiH = sum(
            [
                apply_kraus(U @ (D1 * UHU) @ U.conj().T, K_list, adjoint=True)
                for (U, D1, UHU, K_list) in zip(
                    self.Ukx_blk, self.D1kx_log_blk, UkKHUk_blk, self.K_list_blk
                )
            ]
        )
        D2PhiH -= sum(
            [
                apply_kraus(U @ (D1 * UHU) @ U.conj().T, K_list, adjoint=True)
                for (U, D1, UHU, K_list) in zip(
                    self.Uzkx_blk, self.D1zkx_log_blk, UkzZKHUkz_blk, self.ZK_list_blk
                )
            ]
        )

        D3PhiHH = sum(
            [
                apply_kraus(grad.scnd_frechet(D2 * UHU, UHU, U=U), K_list, adjoint=True)
                for (U, D2, UHU, K_list) in zip(
                    self.Ukx_blk, self.D2kx_log_blk, UkKHUk_blk, self.K_list_blk
                )
            ]
        )
        D3PhiHH -= sum(
            [
                apply_kraus(grad.scnd_frechet(D2 * UHU, UHU, U=U), K_list, adjoint=True)
                for (U, D2, UHU, K_list) in zip(
                    self.Uzkx_blk, self.D2zkx_log_blk, UkzZKHUkz_blk, self.ZK_list_blk
                )
            ]
        )

        # Third derivative of barrier
        DPhiH = lin.inp(self.DPhi, Hx)
        D2PhiHH = lin.inp(D2PhiH, Hx)
        chi = Ht - DPhiH
        chi2 = chi * chi

        dder3_t = -2 * self.zi3 * chi2 - self.zi2 * D2PhiHH

        dder3_X = -dder3_t * self.DPhi
        dder3_X -= 2 * self.zi2 * chi * D2PhiH
        dder3_X += self.zi * D3PhiHH
        dder3_X -= 2 * self.inv_X @ Hx @ self.inv_X @ Hx @ self.inv_X
        dder3_X = (dder3_X + dder3_X.conj().T) * 0.5

        out[0][:] += dder3_t * a
        out[1][:] += dder3_X * a

        return out

    # ========================================================================
    # Auxilliary functions
    # ========================================================================
    def congr_aux(self, A):
        assert not self.congr_aux_updated

        p = A.shape[0]

        if self.G_is_Id:
            if sp.sparse.issparse(A):
                A = A.toarray()

            self.At = A[:, 0]
            Ax = np.ascontiguousarray(A[:, 1:])

            if self.iscomplex:
                self.Ax = np.array(
                    [
                        Ax_k.reshape((-1, 2))
                        .view(dtype=np.complex128)
                        .reshape((self.n, self.n))
                        for Ax_k in Ax
                    ]
                )
            else:
                self.Ax = np.array([Ax_k.reshape((self.n, self.n)) for Ax_k in Ax])

            self.Work0 = np.zeros_like(self.Ax, dtype=self.dtype)
            self.Work1 = np.zeros_like(self.Ax, dtype=self.dtype)
            self.Work2 = np.zeros_like(self.Ax, dtype=self.dtype)
            self.Work3 = np.zeros_like(self.Ax, dtype=self.dtype)

            self.work1 = np.empty((p, self.m, self.m), dtype=self.dtype)
            self.work2 = np.empty((p, self.m, self.m), dtype=self.dtype)
            self.work3 = np.empty((p, self.m, self.m), dtype=self.dtype)
        else:
            if sp.sparse.issparse(A):
                A = A.tocsr()
            self.At = A[:, 0].toarray().flatten() if sp.sparse.issparse(A) else A[:, 0]
            self.Ax_vec = lin.scale_axis(
                A[:, 1 + self.triu_idxs], scale_cols=self.scale
            )

            self.work0 = np.zeros(self.Ax_vec.shape)
            self.work1 = np.zeros(self.Ax_vec.shape)

        self.congr_aux_updated = True

    def update_invhessprod_aux_aux(self):
        assert not self.invhess_aux_aux_updated

        if self.G_is_Id:
            self.work6 = np.zeros((self.vm, self.m, self.m), dtype=self.dtype)
            self.work7 = np.zeros((self.vm, self.m, self.m), dtype=self.dtype)
            self.work8 = np.zeros((self.vm, self.m, self.m), dtype=self.dtype)

            # Computational basis for symmetric/Hermitian matrices
            rt2 = np.sqrt(0.5)
            self.E_blk = np.zeros((self.r, self.vm, self.n, self.n), dtype=self.dtype)
            for b, Z_idxs_k in enumerate(self.Z_idxs):
                k = 0
                for j_subblk in range(self.m):
                    j = Z_idxs_k[j_subblk]
                    for i_subblk in range(j_subblk):
                        i = Z_idxs_k[i_subblk]
                        self.E_blk[b, k, i, j] = rt2
                        self.E_blk[b, k, j, i] = rt2
                        k += 1
                        if self.iscomplex:
                            self.E_blk[b, k, i, j] = rt2 * 1j
                            self.E_blk[b, k, j, i] = rt2 * -1j
                            k += 1
                    self.E_blk[b, k, j, j] = 1.0
                    k += 1
            self.E_blk = self.E_blk.reshape((-1, self.n, self.n))

            self.Work6 = np.zeros((self.r * self.vm, self.n, self.n), dtype=self.dtype)
            self.Work7 = np.zeros((self.r * self.vm, self.n, self.n), dtype=self.dtype)
            self.Work8 = np.zeros((self.r * self.vm, self.n, self.n), dtype=self.dtype)
        else:
            self.work2 = [
                np.zeros((self.vn, self.n, nk), dtype=self.dtype) for nk in self.Nk
            ]
            self.work2b = [
                np.zeros((self.vn, nk, self.n), dtype=self.dtype) for nk in self.Nk
            ]
            self.work3 = [
                np.zeros((self.vn, nk, nk), dtype=self.dtype) for nk in self.Nk
            ]
            self.work3b = [
                np.zeros((self.vn, nk, nk), dtype=self.dtype) for nk in self.Nk
            ]
            self.work4 = [
                np.zeros((self.vn, self.n, nzk), dtype=self.dtype) for nzk in self.Nzk
            ]
            self.work4b = [
                np.zeros((self.vn, nzk, self.n), dtype=self.dtype) for nzk in self.Nzk
            ]
            self.work5 = [
                np.zeros((self.vn, nzk, nzk), dtype=self.dtype) for nzk in self.Nzk
            ]
            self.work5b = [
                np.zeros((self.vn, nzk, nzk), dtype=self.dtype) for nzk in self.Nzk
            ]
            self.work6 = np.zeros((self.vn, self.n, self.n), dtype=self.dtype)
            self.work7 = np.zeros((self.vn, self.n, self.n), dtype=self.dtype)
            self.work8 = np.zeros((self.vn, self.n, self.n), dtype=self.dtype)

        self.invhess_aux_aux_updated = True

    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated
        if not self.invhess_aux_aux_updated:
            self.update_invhessprod_aux_aux()

        self.z2 = self.z * self.z

        if self.G_is_Id:
            # Precompute and factorize the matrix
            #     N = 1/z (Uy kron Uy) [log^[1](Dy)]^-1 (Uy' kron Uy')
            #         - PTr (Ux kron Ux) [(1/z log + inv)^[1](Dx)]^-1  (Ux' kron Ux') PTr'
            # which we will need to solve linear systems with the Hessian of our barrier function

            D1x_inv = np.reciprocal(np.outer(self.Dx, self.Dx))
            self.D1x_comb = self.zi * self.D1kx_log_blk[0] + D1x_inv
            self.D1x_comb_inv = np.reciprocal(self.D1x_comb)

            self.schur = np.zeros((self.r * self.vm, self.r * self.vm))

            # Get [1/z (Uy kron Uy) [log^[1](Dy)]^-1 (Uy' kron Uy')] matrix
            for k, (U, D1_log) in enumerate(zip(self.Uzkx_blk, self.D1zkx_log_blk)):
                # Begin with (Uy' kron Uy')
                lin.congr_multi(self.work8, U.conj().T, self.E, work=self.work7)
                # Apply z [log^[1](Dy)]^-1
                self.work8 *= self.z * np.reciprocal(D1_log)
                # Apply (Uy kron Uy)
                lin.congr_multi(self.work6, U, self.work8, work=self.work7)

                self.schur[
                    k * self.vm : (k + 1) * self.vm, k * self.vm : (k + 1) * self.vm
                ] = (
                    self.work6.view(dtype=np.float64).reshape((self.vm, -1))[
                        :, self.triu_idxs
                    ]
                    * self.scale
                )

            # Get [PTr (Ux kron Ux) [(1/z log + inv)^[1](Dx)]^-1  (Ux' kron Ux') PTr'] matrix
            # Begin with [(Ux' kron Ux') PTr']
            lin.congr_multi(self.Work8, self.Ux.conj().T, self.E_blk, work=self.Work7)
            self.Work8 *= self.D1x_comb_inv
            lin.congr_multi(self.Work6, self.Ux, self.Work8, work=self.Work7)

            for k, Z_idxs_k in enumerate(self.Z_idxs):
                temp = self.Work6[
                    np.ix_(np.arange(self.r * self.vm), Z_idxs_k, Z_idxs_k)
                ]
                self.schur[:, k * self.vm : (k + 1) * self.vm] -= (
                    temp.view(dtype=np.float64).reshape((self.r * self.vm, -1))[
                        :, self.triu_idxs
                    ]
                    * self.scale
                )

            # Subtract to obtain N then Cholesky factor
            self.schur_fact = lin.cho_fact(self.schur)

        else:
            self.DPhi_vec = self.DPhi.view(dtype=np.float64).reshape(-1, 1)[
                self.triu_idxs
            ] * self.scale.reshape(-1, 1)

            # Get X^-1 kron X^-1
            lin.congr_multi(self.work8, self.inv_X, self.E, work=self.work7)

            # Get S(G(X)) Hessians
            for U, D1, K_list, work0, work1, work2, work3 in zip(
                self.Ukx_blk,
                self.D1kx_log_blk,
                self.K_list_blk,
                self.work2,
                self.work2b,
                self.work3,
                self.work3b,
            ):
                KU_list = [K.conj().T @ U for K in K_list]
                work2 *= 0
                for KU in KU_list:
                    lin.congr_multi(work3, KU.conj().T, self.E, work=work1)
                    work2 += work3

                work2 *= D1 * self.zi

                for KU in KU_list:
                    lin.congr_multi(self.work7, KU, work2, work=work0)
                    self.work8 += self.work7

            # Get S(Z(G(X))) Hessians
            for U, D1, K_list, work0, work1, work2, work3 in zip(
                self.Uzkx_blk,
                self.D1zkx_log_blk,
                self.ZK_list_blk,
                self.work4,
                self.work4b,
                self.work5,
                self.work5b,
            ):
                KU_list = [K.conj().T @ U for K in K_list]
                work2 *= 0
                for KU in KU_list:
                    lin.congr_multi(work3, KU.conj().T, self.E, work=work1)
                    work2 += work3

                work2 *= D1 * self.zi

                for KU in KU_list:
                    lin.congr_multi(self.work7, KU, work2, work=work0)
                    self.work8 -= self.work7

            # Get Hessian and factorize
            self.hess = self.work8.view(dtype=np.float64).reshape((self.vn, -1))[
                :, self.triu_idxs
            ]
            self.hess *= self.scale
            self.hess_fact = lin.cho_fact(self.hess.copy())

        self.invhess_aux_updated = True

        return

    def update_dder3_aux(self):
        assert not self.dder3_aux_updated
        assert self.hess_aux_updated

        self.zi3 = self.zi2 * self.zi

        self.D2kx_log_blk = [
            grad.D2_log(D, D1) for (D, D1) in zip(self.Dkx_blk, self.D1kx_log_blk)
        ]
        self.D2zkx_log_blk = [
            grad.D2_log(D, D1) for (D, D1) in zip(self.Dzkx_blk, self.D1zkx_log_blk)
        ]

        self.dder3_aux_updated = True

        return


def facial_reduction(K_list):
    # For a set of Kraus operators i.e., SUM_i K_i @ X @ K_i.T, returns a set of
    # Kraus operators which preserves positive definiteness
    nk = K_list[0].shape[0]

    # Pass identity matrix (maximally mixed state) through the Kraus operators
    KK = sum([K @ K.conj().T for K in K_list])

    # Determine if output is low rank, in which case we need to perform facial reduction
    Dkk, Ukk = np.linalg.eigh(KK)
    KKnzidx = np.where(Dkk > 1e-12)[0]
    nk_fr = np.size(KKnzidx)

    if nk == nk_fr:
        return K_list

    # Perform facial reduction
    Qkk = Ukk[:, KKnzidx]
    K_list_fr = [Qkk.conj().T @ K for K in K_list]

    return K_list_fr


def apply_kraus(x, Klist, adjoint=False):
    # Compute congruence map
    if adjoint:
        return sum([K.conj().T @ x @ K for K in Klist])
    else:
        return sum([K @ x @ K.conj().T for K in Klist])
