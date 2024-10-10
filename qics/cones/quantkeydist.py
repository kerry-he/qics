# Copyright (c) 2024, Kerry He, James Saunderson, and Hamza Fawzi

# This Python package QICS is licensed under the MIT license; see LICENSE.md 
# file in the root directory or at https://github.com/kerry-he/qics

import numpy as np
import scipy as sp
import qics._utils.linalg as lin
import qics._utils.gradient as grad
from qics.cones.base import Cone
from qics.vectorize import get_full_to_compact_op


class QuantKeyDist(Cone):
    r"""A class representing a quantum key distribution cone

    .. math::

        \mathcal{QKD}_{\mathcal{G},\mathcal{Z}} = 
        \text{cl}\{ (t, X) \in \mathbb{R} \times \mathbb{H}^n_{++} :
        t \geq -S(\mathcal{G}(X)) +  S(\mathcal{Z}(\mathcal{G}(X))) \},

    where

    .. math::

        S(X) = -\text{tr}[X \log(X)],

    is the quantum (von Neumann) entropy function,
    :math:`\mathcal{G}:\mathbb{H}^n\rightarrow\mathbb{H}^{mr}` is a
    positive linear map, and 
    :math:`\mathcal{Z}:\mathbb{H}^{mr}\rightarrow\mathbb{H}^{mr}` is a
    pinching map that maps off-diagonal blocks to zero.

    Parameters
    ----------
    G_info : :obj:`int` or :obj:`list` of :class:`~numpy.ndarray`
        Defines the linear map :math:`\mathcal{G}`. There are two ways to
        specify this linear map.

        - If ``G_info`` is an :obj:`int`, then :math:`\mathcal{G}` is the
          identity map, i.e., :math:`\mathcal{G}(X)=X`, and ``G_info`` 
          specifies the dimension of :math:`X`.
        - If ``G_info`` is a :obj:`list` of :class:`~numpy.ndarray`, then
          ``G_info`` specifies the Kraus operators 
          :math:`K_i \in \mathbb{C}^{mr \times n }` corresponding to
          :math:`\mathcal{G}` such that

          .. math::

              \mathcal{G}(X) = \sum_{i} K_i X K_i^\dagger.
        
    Z_info : :obj:`int` or :obj:`tuple` or :obj:`list` of :class:`~numpy.ndarray`
    
        Defines the pinching map :math:`\mathcal{Z}`, which is of the form
        
        .. math::

            \mathcal{Z}(Y) = \sum_{i} Z_i Y Z_i^\dagger.

        There are three ways to specify this linear map. 

        - If ``Z_info`` is an :obj:`int`, then 
          :math:`Z_i=|i \rangle\langle i| \otimes\mathbb{I}` for
          :math:`i=1,\ldots,r`, where ``r=Z_info``.

        - If ``Z_info`` is a :obj:`tuple` of the form ``(dims, sys)``,
          where ``dims=(n0, n1)`` is a :obj:`tuple` of :obj:`int` and
          ``sys`` is an :obj:`int`, then

          - :math:`Z_i=|i \rangle\langle i| \otimes\mathbb{I}_{n_1}`
            for :math:`i=1,\ldots,n_0` if ``sys=0``, and
          - :math:`Z_i=\mathbb{I}_{n_0}\otimes |i \rangle\langle i|` for 
            :math:`i=1,\ldots,n_1` if ``sys=1``. 
          
          We generalize this definition to the case where ``dims`` and ``sys`` 
          are lists of any length.

        - If ``Z_info`` is a :obj:`list` of :class:`~numpy.ndarray`, then
          ``Z_info`` directly specifies the Kraus operators 
          :math:`Z_i \in \mathbb{C}^{mr \times mr}`.

          .. warning:: 
          
              If ``Z_info`` is specified in this way, the user themselves
              must ensure that the Kraus operators they provide correpsond
              to a valid pinching map.

    iscomplex : :obj:`bool`
        Whether the matrix :math:`X` is defined over :math:`\mathbb{H}^n`
        (``True``), or restricted to :math:`\mathbb{S}^n` (``False``). The
        default is ``False``.

    See also
    --------
    QuantRelEntr : Quantum relative entropy cone

    Notes
    -----
    The quantum key distribution cone can also be modelled by the quantum
    relative entropy by noting the identity

    .. math::

        S(\mathcal{G}(X) \| \mathcal{Z}(\mathcal{G}(X))) 
        = -S(\mathcal{G}(X)) +  S(\mathcal{Z}(\mathcal{G}(X))).

    However, the cone oracles for the quantum key distribution cone are
    more efficient than those for the quantum relative entropy cone 
    (especially when :math:`\mathcal{G}` is the idenity map), so it is
    recommended to use the quantum key distribution cone where possible.
    """

    def __init__(self, G_info, Z_info, iscomplex=False):
        self._process_G_info(G_info)
        self._process_Z_info(Z_info)
        self.iscomplex = iscomplex

        self.nu = 1 + self.n  # Barrier parameter

        if iscomplex:
            self.vn = self.n * self.n
            self.vm = self.m * self.m
            self.dim = [1, 2 * self.n * self.n]
            self.type = ["r", "h"]
            self.dtype = np.complex128
        else:
            self.vn = self.n * (self.n + 1) // 2
            self.vm = self.m * (self.m + 1) // 2
            self.dim = [1, self.n * self.n]
            self.type = ["r", "s"]
            self.dtype = np.float64

        # Facial reduction on G(X) and Z(G(X))
        ZK_list_raw = [[K[Z, :] for K in self.K_list_raw] for Z in self.Z_idxs]
        self.K_list_blk = [facial_reduction(self.K_list_raw)]
        self.ZK_list_blk = [facial_reduction(ZK) for ZK in ZK_list_raw]

        self.Nk = [K_list[0].shape[0] for K_list in self.K_list_blk]
        self.Nzk = [ZK_list[0].shape[0] for ZK_list in self.ZK_list_blk]
        
        # Update flags
        self.feas_updated = False
        self.grad_updated = False
        self.hess_aux_updated = False
        self.invhess_aux_updated = False
        self.dder3_aux_updated = False
        self.invhess_aux_aux_updated = False
        self.congr_aux_updated = False
        self.hess_congr_aux_updated = False
        self.invhess_congr_aux_updated = False

        if self.G_is_Id:
            self.precompute_computational_basis(self.m)
            self.F2C_op = get_full_to_compact_op(self.m, iscomplex)
        else:
            self.precompute_computational_basis(self.n)
            self.F2C_op = get_full_to_compact_op(self.n, iscomplex)

        return
    
    def _process_G_info(self, G_info):
        if isinstance(G_info, int):
            # Define G(X) as the identity map
            self.n = n = G_info  # Input dimension of X
            self.N = N = G_info  # Output dimension of G(X)

            self.K_list_raw = [np.eye(self.N)]
            self.G_is_Id = True

        else:
            # Define G(X) using given Kraus operators
            self.n = n = G_info[0].shape[1]  # Input dimension of X
            self.N = N = G_info[0].shape[0]  # Output dimension of G(X)
            assert all([Ki.shape == (N, n) for Ki in G_info]), "Kraus " \
                "operators specified by G_info must have the same dimensions."

            self.K_list_raw = G_info
            self.G_is_Id = len(G_info) == 1 and n == N \
                           and np.linalg.norm(G_info[0] - np.eye(n)) < 1e-10

    def _process_Z_info(self, Z_info):
        N = self.N

        if isinstance(Z_info, int):
            # Define block structure for Z(X) with r x r blocks of size m x m
            self.r = r = Z_info
            self.m = m = N // r
            assert m * r == N, "Number of blocks r specified by Z_info " \
                "must be an integer factor of the dimension of G(X)."

            self.Z_list_raw = [np.zeros((N, N)) for _ in range(r)]
            for k in range(r):
                range_k = np.arange(k * m, (k + 1) * m)
                self.Z_list_raw[k][range_k, range_k] = 1.0
            self.Z_idxs = [np.arange(i * m, (i + 1) * m) for i in range(r)]

        elif isinstance(Z_info, tuple):
            # Define custom block structure for a given subsystem structure
            (dims, sys) = Z_info
            if isinstance(sys, int):
                sys = [sys]
            if isinstance(sys, tuple) or isinstance(sys, set):
                sys = list(sys)
            assert np.prod(dims) == N, "Total dimension of subsystems must " \
                "equal the dimension of G(X)."
            assert all([sys_k < len(dims) for sys_k in sys]), "Invalid " \
                "subsystems specified, exceeds total number of dimensions " \
                "provided."
            
            self.r = np.prod([dims[k] for k in sys])
            self.m = N // self.r

            idxs = np.meshgrid(*[range(dims[k]) for k in sys])
            idxs = list(np.array(idxs).reshape(len(sys), -1).T)
            self.Z_list_raw = []
            for i in range(len(idxs)):
                Z_i = np.array([1])
                counter = 0
                for k, dim_k in enumerate(dims):
                    if k in sys:
                        Z_ik = np.zeros(dim_k)
                        Z_ik[idxs[i][counter]] = 1
                        Z_i = np.kron(Z_i, Z_ik)
                        counter += 1
                    else:
                        Z_i = np.kron(Z_i, np.ones(dim_k))
                self.Z_list_raw += [np.diag(Z_i)]
            self.Z_idxs = [np.where(Z)[0] for Z in self.Z_list_raw]
        else:
            # Define Z(X) using given Kraus operators
            self.r = len(Z_info)
            self.m = self.N // self.r
            assert all([Zi.shape == (N, N) for Zi in Z_info]), "Kraus " \
                "operators specified by Z_info must have the same dimensions."

            self.Z_list_raw = Z_info
            self.Z_idxs = [np.where(Z)[0] for Z in Z_info]

    def get_iscomplex(self):
        return self.iscomplex

    def get_init_point(self, out):
        from qics.quantum import entropy

        eye = np.eye(self.n)
        GI_blk = [apply_kraus(eye, K_list) for K_list in self.K_list_blk]
        ZGI_blk = [apply_kraus(eye, ZK_list) for ZK_list in self.ZK_list_blk]

        entr_GI = sum([entropy(KK) for KK in GI_blk])
        entr_ZGI = sum([entropy(ZKKZ) for ZKKZ in ZGI_blk])

        f0 = -entr_GI + entr_ZGI
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

        (self.t, self.X) = (t, X) = self.primal

        # Check that X is positive definite
        self.Dx, self.Ux = np.linalg.eigh(X)
        if any(self.Dx <= 0):
            self.feas = False
            return self.feas

        # Check that G(X) is positive definite
        #   Note that G(X) should be positive definite if X is, but check 
        #   just to be sure that we can safely take logarithms of G(X).
        GX_blk = [apply_kraus(X, K_list) for K_list in self.K_list_blk]

        DUgx_blk = [np.linalg.eigh(GX) for GX in GX_blk]
        self.Dgx_blk = [DUgx[0] for DUgx in DUgx_blk]
        self.Ugx_blk = [DUgx[1] for DUgx in DUgx_blk]

        if any([any(Dgx <= 0) for Dgx in self.Dgx_blk]):
            self.feas = False
            return self.feas

        # Check that Z(G(X)) is positive definite
        #   Note that Z(G(X)) should be positive definite if X is, but check 
        #   just to be sure that we can safely take logarithms of Z(G(X)).
        ZGX_blk = [apply_kraus(X, ZG_list) for ZG_list in self.ZK_list_blk]

        DUzgx_blk = [np.linalg.eigh(ZGX) for ZGX in ZGX_blk]
        self.Dzgx_blk = [DUzkx[0] for DUzkx in DUzgx_blk]
        self.Uzgx_blk = [DUzkx[1] for DUzkx in DUzgx_blk]

        if any([any(Dzkx <= 0) for Dzkx in self.Dzgx_blk]):
            self.feas = False
            return self.feas

        # Check that t > -S(G(X)) + S(Z(G(X)))
        self.log_Dgx_blk = [np.log(D) for D in self.Dgx_blk]
        self.log_Dzgx_blk = [np.log(D) for D in self.Dzgx_blk]

        entr_GX = [lin.inp(D, log_D) 
                   for (D, log_D) in zip(self.Dgx_blk, self.log_Dgx_blk)]
        entr_ZGX = [lin.inp(D, log_D) 
                    for (D, log_D) in zip(self.Dzgx_blk, self.log_Dzgx_blk)]
        self.z = t[0, 0] - (sum(entr_GX) - sum(entr_ZGX))

        self.feas = self.z > 0
        return self.feas

    def get_val(self):
        assert self.feas_updated

        return -np.log(self.z) - np.sum(np.log(self.Dx))

    def update_grad(self):
        assert self.feas_updated
        assert not self.grad_updated

        # Compute gradients of quantum relative entropy
        # DPhi(X) = G'[log(G(X)) + I] - G'Z'[log(G(Z(X))) + I]
        log_GX_blk = [(U * logD) @ U.conj().T + np.eye(n)
            for (U, logD, n) in zip(self.Ugx_blk, self.log_Dgx_blk, self.Nk)]
        log_ZGX_blk = [(U * logD) @ U.conj().T + np.eye(n)
            for (U, logD, n) in zip(self.Uzgx_blk, self.log_Dzgx_blk, self.Nzk)]

        G_log_GX = sum([apply_kraus(logX, K_list, adjoint=True)
            for (logX, K_list) in zip(log_GX_blk, self.K_list_blk)])
        ZG_log_ZGX = sum([apply_kraus(logX, ZK_list, adjoint=True)
            for (logX, ZK_list) in zip(log_ZGX_blk, self.ZK_list_blk)])
        
        self.DPhi = G_log_GX - ZG_log_ZGX

        # Compute X^-1
        inv_Dx = np.reciprocal(self.Dx)
        inv_X_rt2 = self.Ux * np.sqrt(inv_Dx)
        self.inv_X = inv_X_rt2 @ inv_X_rt2.conj().T

        # Compute gradient of barrier function
        self.zi = np.reciprocal(self.z)
        self.grad = [-self.zi, self.zi * self.DPhi - self.inv_X]

        self.grad_updated = True

    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        self.zi2 = self.zi * self.zi

        self.D1gx_log_blk = [grad.D1_log(D, log_D) 
            for (D, log_D) in zip(self.Dgx_blk, self.log_Dgx_blk)]
        self.D1zgx_log_blk = [grad.D1_log(D, log_D)
            for (D, log_D) in zip(self.Dzgx_blk, self.log_Dzgx_blk)]

        if self.G_is_Id:
            D1x_inv = np.reciprocal(np.outer(self.Dx, self.Dx))
            self.D1x_comb = self.zi * self.D1gx_log_blk[0] + D1x_inv

        self.hess_aux_updated = True

        return

    def hess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        (Ht, Hx) = H

        K_list_blk, ZK_list_blk = self.K_list_blk, self.ZK_list_blk
        Ugx_blk, Uzgx_blk = self.Ugx_blk, self.Uzgx_blk
        D1gx_log_blk, D1zgx_log_blk = self.D1gx_log_blk, self.D1zgx_log_blk

        GH_blk = [apply_kraus(Hx, K_list) for K_list in K_list_blk]
        ZGH_blk = [apply_kraus(Hx, ZK_list) for ZK_list in ZK_list_blk]

        UGHU_blk = [U.conj().T @ H @ U for (H, U) in zip(GH_blk, Ugx_blk)]
        UZGHU_blk = [U.conj().T @ H @ U for (H, U) in zip(ZGH_blk, Uzgx_blk)]

        # Hessian products for quantum key distribution
        # D2Phi(X)[H] 
        #     = G'(Ugx [log^[1](Dgx) .* (Ugx' G(H) Ugx)] Ugx')
        #       - G'(Z'(Uzgx [log^[1](Dzgx) .* (Uzgx' Z(G(H)) Uzgx)] Uzgx'))
        D2PhiH = np.zeros_like(self.X)
        # First term, i.e., G'(Ugx [log^[1](Dgx) .* (Ugx' G(H) Ugx)] Ugx')
        for (k, (U_k, D1_k)) in enumerate(zip(Ugx_blk, D1gx_log_blk)):
            temp = U_k @ (D1_k * UGHU_blk[k]) @ U_k.conj().T
            D2PhiH += apply_kraus(temp, K_list_blk[k], adjoint=True)
        # Second term, i.e., -G'(Z'(Uzgx [ ... ] Uzgx'))
        for (k, (U_k, D1_k)) in enumerate(zip(Uzgx_blk, D1zgx_log_blk)):
            temp = U_k @ (D1_k * UZGHU_blk[k]) @ U_k.conj().T
            D2PhiH -= apply_kraus(temp, ZK_list_blk[k], adjoint=True)

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
        if not self.hess_congr_aux_updated:
            self.update_hess_congr_aux(A)

        p = A.shape[0]
        lhs = np.zeros((p, sum(self.dim)))

        Z_idxs = self.Z_idxs
        K_list_blk, ZK_list_blk = self.K_list_blk, self.ZK_list_blk
        Ugx_blk, Uzgx_blk = self.Ugx_blk, self.Uzgx_blk
        D1gx_log_blk, D1zgx_log_blk = self.D1gx_log_blk, self.D1zgx_log_blk

        Work0, Work1 = self.Work0, self.Work1
        Work2, Work3 = self.Work2, self.Work3

        # ======================================================================
        # Hessian products with respect to t
        # ======================================================================
        # D2_t F(t, X)[Ht, Hx] = (Ht - D_X Phi(X)[Hx]) / z^2
        Ax_vec = self.Ax.view(np.float64).reshape((p, 1, -1))
        DPhi_vec = self.DPhi.view(np.float64).reshape((-1, 1))
        outt = self.At - (Ax_vec @ DPhi_vec).ravel()
        outt *= self.zi2

        lhs[:, 0] = outt

        # ======================================================================
        # Hessian products with respect to X
        # ======================================================================
        # Hessian products for quantum key distribution
        # D2Phi(X)[H] 
        #     = G'(Ugx [log^[1](Dgx) .* (Ugx' G(H) Ugx)] Ugx')
        #       - G'(Z'(Uzgx [log^[1](Dzgx) .* (Uzgx' Z(G(H)) Uzgx)] Uzgx'))
        if self.G_is_Id:
            work1, work2, work3 = self.work1, self.work2, self.work2

            # If G(X) = X, then we can do some more efficient slicing operations
            # Compute second term, i.e., -Z'(Uzx [log^[1](Dzx) .* ... ] Uzx')
            Work0 *= 0
            for Z_idxs_k, U_k, D1_log_k in zip(Z_idxs, Uzgx_blk, D1zgx_log_blk):
                # Apply Z(Ax), i.e., extract k-th submatrix from Ax
                temp = self.Ax[np.ix_(np.arange(p), Z_idxs_k, Z_idxs_k)]
                # Compute Uzx [log^[1](Dzx) .* (Uzgx' Z(Ax) Uzx)] Uzx'
                lin.congr_multi(work2, U_k.conj().T, temp, work3)
                work2 *= D1_log_k * self.zi
                lin.congr_multi(work1, U_k, work2, work3)
                # Apply Z'( ... ), i.e., place result in k-th submatrix
                Work0[np.ix_(np.arange(p), Z_idxs_k, Z_idxs_k)] = work1

            # Compute first term, i.e., Ux [log^[1](Dx) .* (Ux' G(H) Ux)] Ux'
            # Also combine this with 1/z ( ... ) + X^-1 Ax X^-1 
            lin.congr_multi(Work2, self.Ux.conj().T, self.Ax, Work3)
            self.Work2 *= self.D1x_comb
            lin.congr_multi(Work1, self.Ux, Work2, Work3)

            # Subtract two terms to get D2Phi(X)[H]
            Work1 -= Work0
        else:
            Work1 *= 0

            # Get first term, i.e., G'(Ugx [log^[1](Dgx) .* ... ] Ugx')
            for k in range(len(K_list_blk)):
                worka, workb = self.Work4[k], self.Work4b[k]
                workc, workd = self.Work5[k], self.Work5b[k]
                workc *= 0

                KU_list = [K.conj().T @ Ugx_blk[k] for K in K_list_blk[k]]
                # Apply Ugx' G(Ax) Ugx
                for KU in KU_list:
                    lin.congr_multi(workd, KU.conj().T, self.Ax, work=workb)
                    workc += workd
                # Apply log^[1](Dgx) .* ( ... )
                workc *= D1gx_log_blk[k] * self.zi
                # Apply G'(Ugx [ ... ] Ugx')
                for KU in KU_list:
                    lin.congr_multi(Work0, KU, workc, work=worka)
                    Work1 += Work0

            # Get second term, i.e., G'(Z'(Uzgx [log^[1](Dzgx) .* ... ] Uzgx'))
            for k in range(len(ZK_list_blk)):
                worka, workb = self.Work6[k], self.Work6b[k]
                workc, workd = self.Work7[k], self.Work7b[k]
                workc *= 0

                KU_list = [K.conj().T @ Uzgx_blk[k] for K in ZK_list_blk[k]]
                # Apply Uzgx' Z(G(H)) Uzgx
                for KU in KU_list:
                    lin.congr_multi(workd, KU.conj().T, self.Ax, work=workb)
                    workc += workd
                # Apply log^[1](Dzgx) .* ( ... )
                workc *= D1zgx_log_blk[k] * self.zi
                # Apply G'(Z'(Uzgx [ ... ] Uzgx'))
                for KU in KU_list:
                    lin.congr_multi(Work0, KU, workc, work=worka)
                    Work1 -= Work0

            # Compute X^-1 Ax X^-1
            lin.congr_multi(Work0, self.inv_X, self.Ax, Work2)
            Work1 += Work0

        # Hessian product of barrier function
        # D2_X F(t, X)[Ht, Hx] = -D2_t F(t, X)[Ht, Hx] * DPhi(X)
        #                        + D2Phi(X)[Hx] / z + X^-1 Hx X^-1
        np.outer(outt, self.DPhi, out=Work0.reshape(p, -1))
        Work1 -= Work0

        lhs[:, 1:] = Work1.reshape((p, -1)).view(np.float64)

        # Multiply A (H A')
        return lin.dense_dot_x(lhs, A.T)

    def invhess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        (Ht, Hx) = H

        Wx = Hx + Ht * self.DPhi

        # ======================================================================
        # Inverse Hessian products with respect to X
        # ======================================================================
        # Compute X = M \ Wx, where we solve this using two different strategies
        if self.G_is_Id:
            # If G(X) = X, then solve M using the matrix inversion lemma
            work = np.zeros((self.r * self.vm))

            # Apply D2S(X)^-1 = (Ux⊗Ux) (1/z log + inv)^[1](Dx)^-1 (Ux'⊗Ux')
            temp = self.Ux.conj().T @ Wx @ self.Ux
            out_X = self.Ux @ (self.D1x_comb_inv * temp) @ self.Ux.conj().T
            # Apply Z( ... ), i.e., extract k-th submatrix then get compact
            # vectorization of this submatrix            
            for k, Z_idxs_k in enumerate(self.Z_idxs):
                temp_k = out_X[np.ix_(Z_idxs_k, Z_idxs_k)]
                temp_vec = temp_k.view(np.float64).reshape(-1)
                work[k * self.vm : (k + 1) * self.vm] = self.F2C_op @ temp_vec
            work *= -1
            # Solve linear system N \ ( ... )
            work = lin.cho_solve(self.schur_fact, work)
            # Apply Z'( ... ), i.e., recover submatrix from compact 
            # vectorization then place in k-th submatrix
            Work3 = np.zeros((self.n, self.n), dtype=self.dtype)
            for k, Z_idxs_k in enumerate(self.Z_idxs):
                work_k = work[k * self.vm : (k + 1) * self.vm]
                work_k = self.F2C_op.T @ work_k
                work_k = work_k.view(self.dtype).reshape((self.m, self.m))
                Work3[np.ix_(Z_idxs_k, Z_idxs_k)] = work_k

            # Apply D2S(X)^-1 = (Ux ⊗ Ux) log^[1](Dx) (Ux' ⊗ Ux') and subtract
            # from (D2S(X)^-1 Wx) to get X
            temp = self.Ux.conj().T @ Work3 @ self.Ux
            out_X -= self.Ux @ (self.D1x_comb_inv * temp) @ self.Ux.conj().T

        else:
            # Otherwise, directly build M and Cholesky factor to solve M \ Wx
            # Convert matrices to compact real vectors
            temp_vec = Wx.view(np.float64).reshape((-1, 1))
            temp_vec = self.F2C_op @ temp_vec
            # Solve system
            temp_vec = lin.cho_solve(self.hess_fact, temp_vec)
            # Expand compact real vectors back into full matrices
            temp_vec = self.F2C_op.T @ temp_vec
            out_X = temp_vec.T.view(self.dtype).reshape((self.n, self.n))

        out[1][:] = (out_X + out_X.conj().T) * 0.5

        # ======================================================================
        # Inverse Hessian products with respect to t
        # ======================================================================
        # Compute t = z^2 Ht + <DPhi(X), X>
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
        if not self.invhess_congr_aux_updated:
            self.update_invhess_congr_aux(A)            

        # The inverse Hessian product applied on (Ht, Hx, Hy) for the OPT
        # barrier is
        #     X = M \ Wx
        #     t = z^2 Ht + <DPhi(X), X>
        # where Wx = Hx + Ht DPhi(X) and M = 1/z D2Phi + X^1 ⊗ X^-1

        if self.G_is_Id:
            # When G(X) = X,  we can write the expression for M as
            #   M = 1/z D2S(X) - 1/z Z' D2S(Z(X)) Z + X^-1 ⊗ X^-1
            #     = (Ux ⊗ Ux) (1/z log + inv)^[1](Dx) (Ux' ⊗ Ux')
            #       - 1/z Z' (Uzx ⊗ Uzx) log^[1](Dzx) (Uzx' ⊗ Uzx') Z
            # Treating [Z' D2S(Z(X)) Z] as a low-rank perturbation of D2S(X), we
            # can solve linear systems with M by using the matrix inversion
            # lemma
            #   X = [D2S(X)^-1 - D2S(X)^-1 Z' N^-1 Z D2S(X)^-1] Wx
            # where
            #   N = 1/z (Uzx ⊗ Uzx) [log^[1](Dzx)]^-1 (Uzx' ⊗ Uzx')
            #       - Z (Ux ⊗ Ux) [(1/z log + inv)^[1](Dx)]^-1  (Ux' ⊗ Ux') Z'

            p = A.shape[0]
            lhs = np.zeros((p, sum(self.dim)))

            work = np.zeros((self.r * self.vm, p))

            Work0, Work1 = self.Work0, self.Work1
            Work2, Work3 = self.Work2, self.Work3

            # ==================================================================
            # Inverse Hessian products with respect to X
            # ==================================================================
            # Compute Wx = Hx + Ht DPhi(X)
            np.outer(self.At, self.DPhi, out=Work2.reshape((p, -1)))
            np.add(self.Ax, Work2, out=Work0)

            # Apply D2S(X)^-1 = (Ux⊗Ux) (1/z log + inv)^[1](Dx)^-1 (Ux'⊗Ux')
            lin.congr_multi(Work2, self.Ux.conj().T, Work0, Work3)
            Work2 *= self.D1x_comb_inv
            lin.congr_multi(Work0, self.Ux, Work2, Work3)

            # Apply Z( ... ), i.e., extract k-th submatrix then get compact
            # vectorization of this submatrix
            for k, Z_idxs_k in enumerate(self.Z_idxs):
                temp = Work0[np.ix_(np.arange(p), Z_idxs_k, Z_idxs_k)]
                temp = temp.view(np.float64).reshape((p, -1)).T
                temp = lin.x_dot_dense(self.F2C_op, temp)
                work[k * self.vm : (k + 1) * self.vm] = temp
            work *= -1
            # Solve linear system N \ ( ... )
            work = lin.cho_solve(self.schur_fact, work)
            # Apply Z'( ... ), i.e., recover submatrix from compact 
            # vectorization then place in k-th submatrix
            Work1.fill(0.0)
            for k, Z_idxs_k in enumerate(self.Z_idxs):
                work_k = work[k * self.vm : (k + 1) * self.vm]
                work_k = lin.x_dot_dense(self.F2C_op.T, work_k)
                work_k = work_k.T.view(self.dtype).reshape(p, self.m, self.m)
                Work1[np.ix_(np.arange(p), Z_idxs_k, Z_idxs_k)] = work_k

            # Apply D2S(X)^-1 = (Ux⊗Ux) (1/z log + inv)^[1](Dx)^-1 (Ux'⊗Ux')
            lin.congr_multi(Work2, self.Ux.conj().T, Work1, Work3)
            Work2 *= self.D1x_comb_inv
            lin.congr_multi(Work1, self.Ux, Work2, Work3)

            # Subtract previous expression from (D2S(X)^-1 Wx) to get X
            Work0 -= Work1
            out_X = Work0.reshape((p, -1)).view(np.float64)
            lhs[:, 1:] = out_X

            # ==================================================================
            # Inverse Hessian products with respect to t
            # ==================================================================
            # Compute t = z^2 Ht + <DPhi(X), X>
            DPhi_vec = self.DPhi.view(np.float64).reshape((-1, 1))

            out_t = self.z2 * self.At
            out_t += (out_X @ DPhi_vec).ravel()
            lhs[:, 0] = out_t

            # Multiply A (H A')
            return lin.dense_dot_x(lhs, A.T)

        else:
            # Otherwise, we will directly build M and Cholesky factor it to 
            # solve X = M \ Wx

            # ==================================================================
            # Inverse Hessian products with respect to X
            # ==================================================================
            # Compute Wx = Hx + Ht DPhi(X)
            np.outer(self.At, self.DPhi_cvec, out=self.work0)
            self.work0 += self.Ax_cvec

            # Solve system X = M \ Wx
            out_X = lin.cho_solve(self.hess_fact, self.work0.T)

            # ==================================================================
            # Inverse Hessian products with respect to t
            # ==================================================================
            # Compute t = z^2 Ht + <DPhi(X), X>
            out_t = self.z2 * self.At
            out_t += (out_X.T @ self.DPhi_cvec).ravel()

            # Multiply A (H A')
            out = np.outer(out_t, self.At)
            out += lin.x_dot_dense(self.Ax_cvec, out_X)
            return out

    def third_dir_deriv_axpy(self, out, H, a=True):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.dder3_aux_updated:
            self.update_dder3_aux()

        (Ht, Hx) = H

        K_list_blk, ZK_list_blk = self.K_list_blk, self.ZK_list_blk
        Ugx_blk, Uzgx_blk = self.Ugx_blk, self.Uzgx_blk
        D1gx_log_blk, D1zgx_log_blk = self.D1gx_log_blk, self.D1zgx_log_blk
        D2gx_log_blk, D2zgx_log_blk = self.D2gx_log_blk, self.D2zgx_log_blk

        GH_blk = [apply_kraus(Hx, K_list) for K_list in K_list_blk]
        ZGH_blk = [apply_kraus(Hx, ZK_list) for ZK_list in ZK_list_blk]

        UGHU_blk = [U.conj().T @ H @ U for (H, U) in zip(GH_blk, Ugx_blk)]
        UZGHU_blk = [U.conj().T @ H @ U for (H, U) in zip(ZGH_blk, Uzgx_blk)]

        # Quantum key distribution Hessians
        D2PhiH = np.zeros_like(self.X)
        # Hessians of S(G(X))
        for (k, (U_k, D1_k)) in enumerate(zip(Ugx_blk, D1gx_log_blk)):
            temp = U_k @ (D1_k * UGHU_blk[k]) @ U_k.conj().T
            D2PhiH += apply_kraus(temp, K_list_blk[k], adjoint=True)
        # Hessians of S(Z(G(X)))
        for (k, (U_k, D1_k)) in enumerate(zip(Uzgx_blk, D1zgx_log_blk)):
            temp = U_k @ (D1_k * UZGHU_blk[k]) @ U_k.conj().T
            D2PhiH -= apply_kraus(temp, ZK_list_blk[k], adjoint=True)

        # Quantum key distribution third order derivatives
        D3PhiHH = np.zeros_like(self.X)
        # Third order derivatives of S(G(X))
        for (k, (U_k, D2_k)) in enumerate(zip(Ugx_blk, D2gx_log_blk)):
            temp = grad.scnd_frechet(D2_k * UGHU_blk[k], UGHU_blk[k], U=U_k)
            D3PhiHH += apply_kraus(temp, K_list_blk[k], adjoint=True)
        # Third order derivatives of S(Z(G(X)))
        for (k, (U_k, D2_k)) in enumerate(zip(Uzgx_blk, D2zgx_log_blk)):
            temp = grad.scnd_frechet(D2_k * UZGHU_blk[k], UZGHU_blk[k], U=U_k)
            D3PhiHH -= apply_kraus(temp, ZK_list_blk[k], adjoint=True)

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

        self.congr_aux_updated = True

    def update_hess_congr_aux(self, A):
        assert not self.hess_congr_aux_updated

        p = A.shape[0]

        self.Work0 = np.zeros_like(self.Ax, dtype=self.dtype)
        self.Work1 = np.zeros_like(self.Ax, dtype=self.dtype)
        self.Work2 = np.zeros_like(self.Ax, dtype=self.dtype)
        self.Work3 = np.zeros_like(self.Ax, dtype=self.dtype)

        if self.G_is_Id:
            self.work1 = np.empty((p, self.m, self.m), dtype=self.dtype)
            self.work2 = np.empty((p, self.m, self.m), dtype=self.dtype)
            self.work3 = np.empty((p, self.m, self.m), dtype=self.dtype)
        else:
            n, Nk, Nzk, dtype = self.n, self.Nk, self.Nzk, self.dtype
            self.Work4 = [np.zeros((p, n, nk), dtype=dtype) for nk in Nk]
            self.Work4b = [np.zeros((p, nk, n), dtype=dtype) for nk in Nk]
            self.Work5 = [np.zeros((p, nk, nk), dtype=dtype) for nk in Nk]
            self.Work5b = [np.zeros((p, nk, nk), dtype=dtype) for nk in Nk]
            self.Work6 = [np.zeros((p, n, nzk), dtype=dtype) for nzk in Nzk]
            self.Work6b = [np.zeros((p, nzk, n), dtype=dtype) for nzk in Nzk]
            self.Work7 = [np.zeros((p, nzk, nzk), dtype=dtype) for nzk in Nzk]
            self.Work7b = [np.zeros((p, nzk, nzk), dtype=dtype) for nzk in Nzk]

        self.hess_congr_aux_updated = True

    def update_invhess_congr_aux(self, A):
        assert not self.invhess_congr_aux_updated

        if self.G_is_Id:
            self.Work0 = np.zeros_like(self.Ax, dtype=self.dtype)
            self.Work1 = np.zeros_like(self.Ax, dtype=self.dtype)
            self.Work2 = np.zeros_like(self.Ax, dtype=self.dtype)
            self.Work3 = np.zeros_like(self.Ax, dtype=self.dtype)
        else:
            if sp.sparse.issparse(A):
                A = A.tocsr()
            self.Ax_cvec = (self.F2C_op @ A[:, 1:].T).T
            if sp.sparse.issparse(A):
                self.Ax_cvec = self.Ax_cvec.tocoo()

            self.work0 = np.zeros(self.Ax_cvec.shape)
            self.work1 = np.zeros(self.Ax_cvec.shape)

        self.invhess_congr_aux_updated = True

    def update_invhessprod_aux_aux(self):
        assert not self.invhess_aux_aux_updated

        if self.G_is_Id:
            self.work6 = np.zeros((self.vm, self.m, self.m), dtype=self.dtype)
            self.work7 = np.zeros((self.vm, self.m, self.m), dtype=self.dtype)
            self.work8 = np.zeros((self.vm, self.m, self.m), dtype=self.dtype)

            # Computational basis for symmetric/Hermitian matrices
            rt2 = np.sqrt(0.5)
            n, r, m, vm = self.n, self.r, self.m, self.vm
            self.E_blk = np.zeros((r, vm, n, n), dtype=self.dtype)
            for b, Z_idxs_k in enumerate(self.Z_idxs):
                k = 0
                for j_subblk in range(m):
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

            self.Work6 = np.zeros((r * vm, n, n), dtype=self.dtype)
            self.Work7 = np.zeros((r * vm, n, n), dtype=self.dtype)
            self.Work8 = np.zeros((r * vm, n, n), dtype=self.dtype)
        else:
            dtype = self.dtype
            n, vn, Nk, Nzk = self.n, self.vn, self.Nk, self.Nzk
            self.work2 = [np.zeros((vn, n, nk), dtype=dtype) for nk in Nk]
            self.work2b = [np.zeros((vn, nk, n), dtype=dtype) for nk in Nk]
            self.work3 = [np.zeros((vn, nk, nk), dtype=dtype) for nk in Nk]
            self.work3b = [np.zeros((vn, nk, nk), dtype=dtype) for nk in Nk]
            self.work4 = [np.zeros((vn, n, nzk), dtype=dtype) for nzk in Nzk]
            self.work4b = [np.zeros((vn, nzk, n), dtype=dtype) for nzk in Nzk]
            self.work5 = [np.zeros((vn, nzk, nzk), dtype=dtype) for nzk in Nzk]
            self.work5b = [np.zeros((vn, nzk, nzk), dtype=dtype) for nzk in Nzk]
            self.work6 = np.zeros((vn, n, n), dtype=dtype)
            self.work7 = np.zeros((vn, n, n), dtype=dtype)
            self.work8 = np.zeros((vn, n, n), dtype=dtype)

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
            #   N = z (Uzx ⊗ Uzx) [log^[1](Dzx)]^-1 (Uzx' ⊗ Uzx')
            #       - Z (Ux ⊗ Ux) [(1/z log + inv)^[1](Dx)]^-1  (Ux' ⊗ Ux') Z'
            # which we will need to solve linear systems with the Hessian of
            # our barrier function

            self.D1x_comb_inv = np.reciprocal(self.D1x_comb)

            r, vm = self.r, self.vm
            Uzgx_blk, D1zgx_log_blk = self.Uzgx_blk, self.D1zgx_log_blk

            work6, work7, work8 = self.work6, self.work7, self.work8
            Work6, Work7, Work8 = self.Work6, self.Work7, self.Work8

            self.schur = np.zeros((r * vm, r * vm))

            # ==================================================================
            # Get first term, i.e., 1/z (Uzx ⊗ Uzx) ... (Uzx' ⊗ Uzx')
            # ==================================================================
            for k, (U, D1_log) in enumerate(zip(Uzgx_blk, D1zgx_log_blk)):
                # Begin with (Uzx' ⊗ Uzx')
                lin.congr_multi(work8, U.conj().T, self.E, work=work7)
                # Apply z [log^[1](Dzx)]^-1
                work8 *= self.z * np.reciprocal(D1_log)
                # Apply (Uzx ⊗ Uzx)
                lin.congr_multi(work6, U, work8, work=work7)

                work = work6.view(np.float64).reshape((vm, -1))
                work = lin.x_dot_dense(self.F2C_op, work.T)
                
                self.schur[k * vm : (k + 1) * vm, k * vm : (k + 1) * vm] = work

            # ==================================================================
            # Get second term, i.e., Z (Ux ⊗ Ux) ...  (Ux' ⊗ Ux') Z'
            # ==================================================================
            # Begin with (Ux' ⊗ Ux') Z'
            lin.congr_multi(Work8, self.Ux.conj().T, self.E_blk, work=Work7)
            # Apply [(1/z log + inv)^[1](Dx)]^-1
            Work8 *= self.D1x_comb_inv
            # Apply (Ux ⊗ Ux)
            lin.congr_multi(Work6, self.Ux, Work8, work=Work7)
            # Apply Z, i.e., extract submatrices
            for k, Z_idxs_k in enumerate(self.Z_idxs):
                temp = Work6[np.ix_(np.arange(r * vm), Z_idxs_k, Z_idxs_k)]
                temp = temp.view(np.float64).reshape((r * vm, -1))
                temp = lin.x_dot_dense(self.F2C_op, temp.T).T

                self.schur[:, k * vm : (k + 1) * vm] -= temp

            # Subtract terms to obtain N then Cholesky factor
            self.schur_fact = lin.cho_fact(self.schur)

        else:
            # Precompute and factorize the matrix
            #   M = 1/z G' (Ugx ⊗ Ugx) log^[1](Dgx) (Ugx' ⊗ Ugx') G
            #       - 1/z G'Z' (Uzgx ⊗ Uzgx) log^[1](Dzgx) (Uzgx' ⊗ Uzgx') ZG
            #       +  X^-1 ⊗ X^-1
            # which we will need to solve linear systems with the Hessian of
            # our barrier function

            K_list_blk, ZK_list_blk = self.K_list_blk, self.ZK_list_blk
            Ugx_blk, Uzgx_blk = self.Ugx_blk, self.Uzgx_blk
            D1gx_log_blk, D1zgx_log_blk = self.D1gx_log_blk, self.D1zgx_log_blk

            work7, work8 = self.work7, self.work8

            DPhi_vec = self.DPhi.view(np.float64).reshape(-1, 1)
            self.DPhi_cvec = self.F2C_op @ DPhi_vec

            # ==================================================================
            # Get third term, i.e., X^-1 ⊗ X^-1
            # ==================================================================
            lin.congr_multi(work8, self.inv_X, self.E, work=work7)

            # ==================================================================
            # Get first term, i.e., 1/z G' (Ugx ⊗ Ugx) ... (Ugx' ⊗ Ugx') G
            # ==================================================================
            for k in range(len(K_list_blk)):
                worka, workb = self.work2[k], self.work2b[k]
                workc, workd = self.work3[k], self.work3b[k]
                workc *= 0

                KU_list = [K.conj().T @ Ugx_blk[k] for K in K_list_blk[k]]
                # Apply Ugx' G(Ax) Ugx
                for KU in KU_list:
                    lin.congr_multi(workd, KU.conj().T, self.E, work=workb)
                    workc += workd
                # Apply log^[1](Dgx) .* ( ... )
                workc *= D1gx_log_blk[k] * self.zi
                # Apply G'(Ugx [ ... ] Ugx')
                for KU in KU_list:
                    lin.congr_multi(work7, KU, workc, work=worka)
                    work8 += work7

            # ==================================================================
            # Get second term, i.e., 1/z G'Z' [ ... ] ZG
            # ==================================================================
            for k in range(len(ZK_list_blk)):
                worka, workb = self.work4[k], self.work4b[k]
                workc, workd = self.work5[k], self.work5b[k]
                workc *= 0

                KU_list = [K.conj().T @ Uzgx_blk[k] for K in ZK_list_blk[k]]
                # Apply Uzgx' Z(G(Ax)) Uzgx
                for KU in KU_list:
                    lin.congr_multi(workd, KU.conj().T, self.E, work=workb)
                    workc += workd
                # Apply log^[1](Dzgx) .* ( ... )
                workc *= D1zgx_log_blk[k] * self.zi
                # Apply G'Z'((Uzgx [ ... ] Uzgx'))
                for KU in KU_list:
                    lin.congr_multi(work7, KU, workc, work=worka)
                    work8 -= work7

            # Get Hessian and factorize
            work = work8.view(np.float64).reshape((self.vn, -1))
            self.hess = lin.x_dot_dense(self.F2C_op, work.T)
            self.hess = (self.hess + self.hess.T) * 0.5
            self.hess_fact = lin.cho_fact(self.hess)

        self.invhess_aux_updated = True

    def update_dder3_aux(self):
        assert not self.dder3_aux_updated
        assert self.hess_aux_updated

        self.zi3 = self.zi2 * self.zi

        self.D2gx_log_blk = [grad.D2_log(D, D1) 
            for (D, D1) in zip(self.Dgx_blk, self.D1gx_log_blk)]
        self.D2zgx_log_blk = [grad.D2_log(D, D1) 
            for (D, D1) in zip(self.Dzgx_blk, self.D1zgx_log_blk)]

        self.dder3_aux_updated = True

        return


def facial_reduction(K_list):
    # For a set of Kraus operators i.e., Σ_i K_i @ X @ K_i.T, returns a set of
    # Kraus operators which preserves positive definiteness
    nk = K_list[0].shape[0]

    # Pass identity matrix (maximally mixed state) through the Kraus operators
    KK = sum([K @ K.conj().T for K in K_list])

    # Determine if output is low rank, in which case we need to perform facial
    # reduction
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
