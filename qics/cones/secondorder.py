import numpy as np
import scipy as sp
from qics.cones.base import SymCone


class SecondOrder(SymCone):
    r"""A class representing a second order cone

    .. math::

        \mathcal{Q}_{n+1} = \{ (t, x) \in \mathbb{R} \times \mathbb{R}^{n}
        : t \geq \| x \|_2 \}.

    Parameters
    ----------
    n : :obj:`int`
        Dimension of the vector :math:`x`, i.e., how many terms are in the
        Euclidean norm.
    """

    def __init__(self, n):
        # Dimension properties
        self.n = n
        self.nu = 1

        self.dim = [1, n]
        self.type = ["r", "r"]

        self.grad_updated = False
        self.congr_aux_updated = False
        self.nt_aux_updated = False
        return

    def get_init_point(self, out):
        point = [
            np.array([[1.0]]),
            np.zeros((self.n, 1)),
        ]

        self.set_point(point, point)

        out[0][:] = point[0]
        out[1][:] = point[1]

        return out

    def get_feas(self):
        self.feas_updated = True

        self.x = self.primal
        self.z = self.dual

        if self.x[0] <= np.sqrt(self.x[1].T @ self.x[1]):
            self.feas = False
            return self.feas

        if self.z is not None:
            if self.z[0] <= np.sqrt(self.z[1].T @ self.z[1]):
                self.feas = False
                return self.feas

        self.feas = True
        return self.feas

    def get_dual_feas(self):
        self.z = self.dual
        return self.z[0] > np.sqrt(self.z[1].T @ self.z[1])

    def get_val(self):
        return -0.5 * np.log(_soc_res(self.x))

    def update_grad(self):
        assert self.feas_updated
        assert not self.grad_updated

        self.x_res = _soc_res(self.x)
        self.x_res_inv = 1 / self.x_res

        self.grad = [-self.x[0] * self.x_res_inv, self.x[1] * self.x_res_inv]

        self.grad_updated = True

    def hess_prod_ip(self, out, H):
        assert self.grad_updated

        (Ht, Hx) = H

        x_res_inv2 = self.x_res_inv * self.x_res_inv
        coeff = 2 * x_res_inv2 * (self.x[0] * Ht - self.x[1].T @ Hx)

        out[0][:] = coeff * self.x[0] - Ht * self.x_res_inv
        out[1][:] = -coeff * self.x[1] + Hx * self.x_res_inv

        return out

    def hess_congr(self, A):
        assert self.grad_updated
        if not self.congr_aux_updated:
            self.congr_aux(A)

        # First term
        lhs = self.x[0] * self.At.T - self.x[1].T @ self.Ax.T
        lhs *= np.sqrt(2.0) * self.x_res_inv
        out = lhs.T @ lhs

        # Second term
        out -= (self.At @ self.At.T) * self.x_res_inv
        out += (self.Ax @ self.Ax.T) * self.x_res_inv

        return out

    def invhess_prod_ip(self, out, H):
        assert self.grad_updated

        (Ht, Hx) = H

        coeff = 2 * (self.x[0] * Ht + self.x[1].T @ Hx)

        out[0][:] = coeff * self.x[0] - Ht * self.x_res
        out[1][:] = coeff * self.x[1] + Hx * self.x_res

        return out

    def invhess_congr(self, A):
        assert self.grad_updated
        if not self.congr_aux_updated:
            self.congr_aux(A)

        # First term
        lhs = self.x[0] * self.At.T + self.x[1].T @ self.Ax.T
        lhs *= np.sqrt(2.0)
        out = lhs.T @ lhs

        # Second term
        out -= (self.At @ self.At.T) * self.x_res
        out += (self.Ax @ self.Ax.T) * self.x_res

        return out

    def third_dir_deriv_axpy(self, out, H, a=True):
        assert self.grad_updated

        (Ht, Hx) = H

        x_res_inv2 = self.x_res_inv * self.x_res_inv
        x_res_inv3 = self.x_res_inv * x_res_inv2

        # Gradients of (t, x) -> t*t - <x, x>
        DPhit = 2 * self.x[0]
        DPhix = -2 * self.x[1]

        DPhitH = Ht * DPhit
        DPhixH = Hx.T @ DPhix

        # Hessians of (t, x) -> t*t - <x, x>
        D2PhittH = 2 * Ht
        D2PhixxH = -2 * Hx

        D2PhitHH = Ht * D2PhittH
        D2PhixHH = Hx.T @ D2PhixxH

        # Third order derivatives of barrier
        coeff1 = DPhitH + DPhixH
        coeff2 = x_res_inv2 * (D2PhitHH + D2PhixHH)
        coeff2 -= 2 * x_res_inv3 * coeff1 * coeff1
        coeff2 *= 0.5

        dder3_t = coeff2 * DPhit
        dder3_t += coeff1 * x_res_inv2 * D2PhittH

        dder3_x = coeff2 * DPhix
        dder3_x += coeff1 * x_res_inv2 * D2PhixxH

        out[0][:] += dder3_t * a
        out[1][:] += dder3_x * a

        return out

    # ========================================================================
    # Functions specific to symmetric cones for NT scaling
    # ========================================================================
    # We want the NT scaling point w and scaled variable lambda such that
    #     H(w) s = z  <==> lambda := W^-T s = W z
    # where H(w) = W^T W. For the nonnegative orthant, we have
    #     w      = s.^1/2 ./ z.^1/2
    #     lambda = z.^1/2 .* s.^1/2
    # Also, we have the linear transformations given by
    #     H(w) ds = ds .* s ./ z
    #     W^-T ds = ds .* z.^1/2 ./ s.^1/2
    #     W    dz = dz .* s.^1/2 ./ z.^1/2
    # See: [Section 4.1]https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf

    def nt_aux(self):
        assert not self.nt_aux_updated
        self.x_res = _soc_res(self.x)
        self.z_res = _soc_res(self.z)

        self.rt_x_res = np.sqrt(self.x_res)
        self.rt_z_res = np.sqrt(self.z_res)

        x_bar = [self.x[0] / self.rt_x_res, self.x[1] / self.rt_x_res]
        z_bar = [self.z[0] / self.rt_z_res, self.z[1] / self.rt_z_res]

        xz_bar = x_bar[0] * z_bar[0] + x_bar[1].T @ z_bar[1]
        gamma = np.sqrt((1.0 + xz_bar) / 2.0)

        # Scaling point
        self.w_res = self.rt_x_res / self.rt_z_res
        self.rt_w_res = np.sqrt(self.w_res)
        self.w_bar = [
            (x_bar[0] + z_bar[0]) / (2 * gamma),
            (x_bar[1] - z_bar[1]) / (2 * gamma),
        ]
        self.rt_w_bar = [
            (1 + self.w_bar[0]) / np.sqrt(2 * (self.w_bar[0] + 1)),
            self.w_bar[1] / np.sqrt(2 * (self.w_bar[0] + 1)),
        ]
        self.w = [self.w_bar[0] * self.rt_w_res, self.w_bar[1] * self.rt_w_res]

        # Scaled variable
        temp = (gamma + z_bar[0]) * x_bar[1] + (gamma + x_bar[0]) * z_bar[1]
        self.lmbda_bar = [gamma, temp / (x_bar[0] + z_bar[0] + 2 * gamma)]
        self.lmbda = [
            self.lmbda_bar[0] * np.sqrt(self.rt_x_res * self.rt_z_res),
            self.lmbda_bar[1] * np.sqrt(self.rt_x_res * self.rt_z_res),
        ]
        self.lmbda_res = _soc_res(self.lmbda)
        self.rt_lmbda_res = np.sqrt(self.lmbda_res)

        self.nt_aux_updated = True

    def nt_prod_ip(self, out, H):
        if not self.nt_aux_updated:
            self.nt_aux()

        (Ht, Hx) = H

        w_res_inv2 = 1 / (self.w_res * self.w_res)
        coeff = 2 * w_res_inv2 * (self.w[0] * Ht - self.w[1].T @ Hx)

        out[0][:] = coeff * self.w[0] - Ht / self.w_res
        out[1][:] = -coeff * self.w[1] + Hx / self.w_res

        return out

    def nt_congr(self, A):
        if not self.nt_aux_updated:
            self.nt_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)

        # First term
        lhs = self.w[0] * self.At.T - self.w[1].T @ self.Ax.T
        lhs *= np.sqrt(2.0) / self.w_res
        out = lhs.T @ lhs

        # Second term
        out -= (self.At @ self.At.T) / self.w_res
        out += (self.Ax @ self.Ax.T) / self.w_res

        return out

    def invnt_prod_ip(self, out, H):
        if not self.nt_aux_updated:
            self.nt_aux()

        (Ht, Hx) = H

        coeff = 2 * (self.w[0] * Ht + self.w[1].T @ Hx)

        out[0][:] = coeff * self.w[0] - Ht * self.w_res
        out[1][:] = coeff * self.w[1] + Hx * self.w_res

        return out

    def invnt_congr(self, A):
        if not self.nt_aux_updated:
            self.nt_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)

        # First term
        lhs = self.w[0] * self.At.T + self.w[1].T @ self.Ax.T
        lhs *= np.sqrt(2.0)
        out = lhs.T @ lhs

        # Second term
        out -= (self.At @ self.At.T) * self.w_res
        out += (self.Ax @ self.Ax.T) * self.w_res

        return out

    def comb_dir(self, out, ds, dz, sigma_mu):
        # Compute the residual for rs where rs is given as the lhs of
        #     Lambda o (W dz + W^-T ds) = -Lambda o Lambda - (W^-T ds_a) o (W dz_a)
        #                                 + sigma * mu * 1
        # which is rearranged into the form H ds + dz = rs, i.e.,
        #     rs := W^-1 [ Lambda \ (-Lambda o Lambda - (W^-T ds_a) o (W dz_a) + sigma*mu 1) ]
        # See: [Section 5.4]https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf
        if not self.nt_aux_updated:
            self.nt_aux()

        # Compute (W^-T ds_a) o (W dz_a)
        rtw_ds = 2 * (self.rt_w_bar[0] * ds[0] - self.rt_w_bar[1].T @ ds[1])
        W_ds = [
            (rtw_ds * self.rt_w_bar[0] - ds[0]) / self.rt_w_res,
            (-rtw_ds * self.rt_w_bar[1] + ds[1]) / self.rt_w_res,
        ]

        rt2_dz = 2 * (self.rt_w_bar[0] * dz[0] + self.rt_w_bar[1].T @ dz[1])
        W_dz = [
            (rt2_dz * self.rt_w_bar[0] - dz[0]) * self.rt_w_res,
            (rt2_dz * self.rt_w_bar[1] + dz[1]) * self.rt_w_res,
        ]

        Wds_Wdz = _soc_prod(W_ds, W_dz)

        # Compute -Lambda o Lambda - [ ... ] + sigma*mu I
        lmbda_lmbda = _soc_prod(self.lmbda, self.lmbda)
        rhs = [
            -lmbda_lmbda[0] - Wds_Wdz[0] + sigma_mu,
            -lmbda_lmbda[1] - Wds_Wdz[1],
        ]

        # Compute Lambda \ [ ... ]
        lmbda_rhs = self.lmbda[1].T @ rhs[1]
        temp = self.lmbda_res * rhs[1] + lmbda_rhs * self.lmbda[1]
        work = [
            (self.lmbda[0] * rhs[0] - lmbda_rhs) / self.lmbda_res,
            (-rhs[0] * self.lmbda[1] + temp / self.lmbda[0]) / self.lmbda_res,
        ]

        # Compute W^-1 [ ... ]
        temp = 2 * (self.rt_w_bar[0] * work[0] - self.rt_w_bar[1].T @ work[1])
        out[0][:] = (temp * self.rt_w_bar[0] - work[0]) / self.rt_w_res
        out[1][:] = (-temp * self.rt_w_bar[1] + work[1]) / self.rt_w_res

    def step_to_boundary(self, ds, dz):
        # Compute the maximum step alpha in [0, 1] we can take such that
        #     s + alpha ds >= 0
        #     z + alpha dz >= 0
        # See: [Section 8.3]https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf
        if not self.nt_aux_updated:
            self.nt_aux()

        rtw_ds = 2 * (self.rt_w_bar[0] * ds[0] - self.rt_w_bar[1].T @ ds[1])
        W_ds = [
            (rtw_ds * self.rt_w_bar[0] - ds[0]) / self.rt_w_res,
            (-rtw_ds * self.rt_w_bar[1] + ds[1]) / self.rt_w_res,
        ]

        rtw_dz = 2 * (self.rt_w_bar[0] * dz[0] + self.rt_w_bar[1].T @ dz[1])
        W_dz = [
            (rtw_dz * self.rt_w_bar[0] - dz[0]) * self.rt_w_res,
            (rtw_dz * self.rt_w_bar[1] + dz[1]) * self.rt_w_res,
        ]

        temp = self.lmbda_bar[0] * W_ds[0] - self.lmbda_bar[1].T @ W_ds[1]
        temp2 = (temp + W_ds[0]) / (1 + self.lmbda_bar[0]) * self.lmbda_bar[1]
        rho = [temp / self.rt_lmbda_res, (W_ds[1] - temp2) / self.rt_lmbda_res]

        temp = self.lmbda_bar[0] * W_dz[0] - self.lmbda_bar[1].T @ W_dz[1]
        temp2 = (temp + W_dz[0]) / (1 + self.lmbda_bar[0]) * self.lmbda_bar[1]
        sig = [temp / self.rt_lmbda_res, (W_dz[1] - temp2) / self.rt_lmbda_res]

        rho_step = (rho[0] - np.sqrt(rho[1].T @ rho[1]))[0, 0]
        sig_step = (sig[0] - np.sqrt(sig[1].T @ sig[1]))[0, 0]

        if rho_step >= 0 and sig_step >= 0:
            return 1.0
        else:
            return 1.0 / max(-rho_step, -sig_step)

    # ========================================================================
    # Auxilliary functions
    # ========================================================================
    def congr_aux(self, A):
        assert not self.congr_aux_updated

        if sp.sparse.issparse(A):
            A = A.tocsr()

        self.At = A[:, [0]]
        self.Ax = A[:, 1:]

        self.congr_aux_updated = True


def _soc_res(x):
    return (x[0] * x[0] - x[1].T @ x[1])[0, 0]


def _soc_prod(x, y):
    return [x[0] * y[0] + x[1].T @ y[1], x[0] * y[1] + y[0] * x[1]]
