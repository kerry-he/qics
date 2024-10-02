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
        (self.t, self.x) = self.primal
        (self.u, self.z) = self.dual

        if self.t <= np.sqrt(self.x.T @ self.x):
            self.feas = False
            return self.feas

        if self.u <= np.sqrt(self.z.T @ self.z):
            self.feas = False
            return self.feas
    
        self.feas = True
        return self.feas    
    
    def get_dual_feas(self):
        return self.u > np.sqrt(self.z.T @ self.z)

    def get_val(self):
        return -0.5 * np.log(self.t * self.t - self.x.T @ self.x)[0, 0]

    def update_grad(self):
        assert self.feas_updated
        assert not self.grad_updated

        self.slack = (self.t * self.t - self.x.T @ self.x)[0, 0]
        self.slack_inv = 1 / self.slack

        self.grad = [-self.t * self.slack_inv, self.x * self.slack_inv]

        self.grad_updated = True

    def hess_prod_ip(self, out, H):
        assert self.grad_updated

        (Ht, Hx) = H

        w = self.t * Ht - self.x.T @ Hx
        coeff = 2 * w * self.slack_inv * self.slack_inv

        out[0][:] = coeff * self.t - Ht * self.slack_inv
        out[1][:] = -coeff * self.x + Hx * self.slack_inv

        return out

    def hess_congr(self, A):
        assert self.grad_updated
        if not self.congr_aux_updated:
            self.congr_aux(A)

        # First term
        lhs = self.t * self.At.T - self.x.T @ self.Ax.T
        lhs *= np.sqrt(2.0) * self.slack_inv
        out = lhs.T @ lhs

        # Second term
        out -= (self.At @ self.At.T) * self.slack_inv
        out += (self.Ax @ self.Ax.T) * self.slack_inv

        return out

    def invhess_prod_ip(self, out, H):
        assert self.grad_updated

        (Ht, Hx) = H

        w = self.t * Ht + self.x.T @ Hx
        coeff = 2 * w

        out[0][:] = coeff * self.t - Ht * self.slack
        out[1][:] = coeff * self.x + Hx * self.slack

        return out

    def invhess_congr(self, A):
        assert self.grad_updated
        if not self.congr_aux_updated:
            self.congr_aux(A)

        # First term
        lhs = self.t * self.At.T + self.x.T @ self.Ax.T
        lhs *= np.sqrt(2.0)
        out = lhs.T @ lhs

        # Second term
        out -= (self.At @ self.At.T) * self.slack
        out += (self.Ax @ self.Ax.T) * self.slack

        return out

    def third_dir_deriv_axpy(self, out, H, a=True):
        assert self.grad_updated

        (Ht, Hx) = H

        self.slack_inv2 = self.slack_inv * self.slack_inv
        self.slack_inv3 = self.slack_inv * self.slack_inv2

        # Gradients of (t, x) -> t*t - <x, x>
        DPhit = 2 * self.t
        DPhix = -2 * self.x

        DPhitH = Ht * DPhit
        DPhixH = Hx.T @ DPhix

        # Hessians of (t, x) -> t*t - <x, x>
        D2PhittH = 2 * Ht
        D2PhixxH = -2 * Hx

        D2PhitHH = Ht * D2PhittH
        D2PhixHH = Hx.T @ D2PhixxH

        # Third order derivatives of barrier
        coeff1 = DPhitH + DPhixH
        coeff2 = (
            self.slack_inv2 * (D2PhitHH + D2PhixHH) - 2 * self.slack_inv3 * coeff1 * coeff1
        ) * 0.5

        dder3_t = coeff2 * DPhit
        dder3_t += coeff1 * self.slack_inv2 * D2PhittH

        dder3_x = coeff2 * DPhix
        dder3_x += coeff1 * self.slack_inv2 * D2PhixxH

        out[0][:] += dder3_t * a
        out[1][:] += dder3_x * a

        return out

    # TODO: Add in symmetric functionality
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
        self.slack = (self.t * self.t - self.x.T @ self.x)[0, 0]
        self.dual_slack = (self.u * self.u - self.z.T @ self.z)[0, 0]

        t_norm = self.t / np.sqrt(self.slack)
        x_norm = self.x / np.sqrt(self.slack)
        u_norm = self.u / np.sqrt(self.dual_slack)
        z_norm = self.z / np.sqrt(self.dual_slack)

        gamma = np.sqrt( (1. + t_norm*u_norm + x_norm.T @ z_norm) / 2. )

        # Scaling point
        self.w_bar = [
            (t_norm + u_norm) / (2*gamma),
            (x_norm - z_norm) / (2*gamma)
        ]
        v_norm = (t_norm + u_norm) / (2*gamma)
        w_norm = (x_norm - z_norm) / (2*gamma)
        self.scaling_slack = np.sqrt(self.slack / self.dual_slack)

        self.v = v_norm * np.sqrt(self.scaling_slack)
        self.w = w_norm * np.sqrt(self.scaling_slack)

        self.W_bar = np.vstack((
            np.hstack((v_norm, w_norm.T)),
            np.hstack((w_norm, np.eye(self.n) + w_norm@w_norm.T / (1 + v_norm)))
        ))
        self.W = self.W_bar * np.power(self.slack / self.dual_slack, 0.25)

        # Scaled variable
        self.lmbda_bar = [
            gamma,
            ((gamma + u_norm)*x_norm + (gamma + t_norm)*z_norm) / (t_norm + u_norm + 2*gamma)
        ]

        self.lmbda = [
            self.lmbda_bar[0] * np.sqrt(np.sqrt(self.slack) * np.sqrt(self.dual_slack)),
            self.lmbda_bar[1] * np.sqrt(np.sqrt(self.slack) * np.sqrt(self.dual_slack))
        ]

        self.rt2_w_bar = [
            (1 + self.w_bar[0]) / np.sqrt(2 * (self.w_bar[0] + 1)),
            self.w_bar[1] / np.sqrt(2 * (self.w_bar[0] + 1))
        ]

        self.nt_aux_updated = True

    def nt_prod_ip(self, out, H):
        if not self.nt_aux_updated:
            self.nt_aux()

        (Ht, Hx) = H

        w = self.v * Ht - self.w.T @ Hx
        coeff = 2 * w / self.scaling_slack / self.scaling_slack

        out[0][:] = coeff * self.v - Ht / self.scaling_slack
        out[1][:] = -coeff * self.w + Hx / self.scaling_slack

        return out

    def nt_congr(self, A):
        if not self.nt_aux_updated:
            self.nt_aux()        
        if not self.congr_aux_updated:
            self.congr_aux(A)

        # First term
        lhs = self.v * self.At.T - self.w.T @ self.Ax.T
        lhs *= np.sqrt(2.0) / self.scaling_slack
        out = lhs.T @ lhs

        # Second term
        out -= (self.At @ self.At.T) / self.scaling_slack
        out += (self.Ax @ self.Ax.T) / self.scaling_slack

        return out

    def invnt_prod_ip(self, out, H):
        if not self.nt_aux_updated:
            self.nt_aux()

        (Ht, Hx) = H

        w = self.v * Ht + self.w.T @ Hx
        coeff = 2 * w

        out[0][:] = coeff * self.v - Ht * self.scaling_slack
        out[1][:] = coeff * self.w + Hx * self.scaling_slack

        return out

    def invnt_congr(self, A):
        if not self.nt_aux_updated:
            self.nt_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)

        # First term
        lhs = self.v * self.At.T + self.w.T @ self.Ax.T
        lhs *= np.sqrt(2.0)
        out = lhs.T @ lhs

        # Second term
        out -= (self.At @ self.At.T) * self.scaling_slack
        out += (self.Ax @ self.Ax.T) * self.scaling_slack

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
        temp = 2 * (self.rt2_w_bar[0] * ds[0] - self.rt2_w_bar[1].T @ ds[1])
        W_ds = [
            (temp * self.rt2_w_bar[0] - ds[0]) / np.power(self.slack / self.dual_slack, 0.25),
           (-temp * self.rt2_w_bar[1] + ds[1]) / np.power(self.slack / self.dual_slack, 0.25)
        ]
        
        temp = 2 * (self.rt2_w_bar[0] * dz[0] + self.rt2_w_bar[1].T @ dz[1])
        W_dz = [
            (temp * self.rt2_w_bar[0] - dz[0]) * np.power(self.slack / self.dual_slack, 0.25),
            (temp * self.rt2_w_bar[1] + dz[1]) * np.power(self.slack / self.dual_slack, 0.25)
        ]

        Wds_Wdz = [
            W_ds[0] * W_dz[0] + W_ds[1].T @ W_dz[1],
            W_ds[0] * W_dz[1] + W_ds[1] * W_dz[0]
        ]

        # Compute -Lambda o Lambda - [ ... ] + sigma*mu I
        lambda_lambda = [
            self.lmbda[0] * self.lmbda[0] + self.lmbda[1].T @ self.lmbda[1],
            self.lmbda[0] * self.lmbda[1] + self.lmbda[1] * self.lmbda[0]
        ]
        
        rhs = [
            -lambda_lambda[0] - Wds_Wdz[0] + sigma_mu,
            -lambda_lambda[1] - Wds_Wdz[1]
        ]

        # Compute Lambda \ [ ... ]
        lmbda_slack = self.lmbda[0] * self.lmbda[0] - self.lmbda[1].T @ self.lmbda[1]
        temp = lmbda_slack * rhs[1] + (self.lmbda[1].T @ rhs[1]) * self.lmbda[1]
        work = [
            (self.lmbda[0] * rhs[0] - self.lmbda[1].T @ rhs[1]) / lmbda_slack,
            (-rhs[0] * self.lmbda[1] + temp / self.lmbda[0]) / lmbda_slack
        ]

        # Compute W^-1 [ ... ]
        temp = 2 * (self.rt2_w_bar[0] * work[0] - self.rt2_w_bar[1].T @ work[1])
        out[0][:] =  (temp * self.rt2_w_bar[0] - work[0]) / np.power(self.slack / self.dual_slack, 0.25)
        out[1][:] = (-temp * self.rt2_w_bar[1] + work[1]) / np.power(self.slack / self.dual_slack, 0.25)

    def step_to_boundary(self, ds, dz):
        # Compute the maximum step alpha in [0, 1] we can take such that
        #     s + alpha ds >= 0
        #     z + alpha dz >= 0
        # See: [Section 8.3]https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf
        if not self.nt_aux_updated:
            self.nt_aux()
        
        lmbda_slack = self.lmbda[0] * self.lmbda[0] - self.lmbda[1].T @ self.lmbda[1]

        temp = 2 * (self.rt2_w_bar[0] * ds[0] - self.rt2_w_bar[1].T @ ds[1])
        W_ds = [
            (temp * self.rt2_w_bar[0] - ds[0]) / np.power(self.slack / self.dual_slack, 0.25),
           (-temp * self.rt2_w_bar[1] + ds[1]) / np.power(self.slack / self.dual_slack, 0.25)
        ]
        
        temp = 2 * (self.rt2_w_bar[0] * dz[0] + self.rt2_w_bar[1].T @ dz[1])
        W_dz = [
            (temp * self.rt2_w_bar[0] - dz[0]) * np.power(self.slack / self.dual_slack, 0.25),
            (temp * self.rt2_w_bar[1] + dz[1]) * np.power(self.slack / self.dual_slack, 0.25)
        ]

        temp = self.lmbda_bar[0] * W_ds[0] - self.lmbda_bar[1].T @ W_ds[1]
        rho = [
            temp / np.sqrt(lmbda_slack),
            (W_ds[1] - (temp + W_ds[0]) / (1 + self.lmbda_bar[0]) * self.lmbda_bar[1]) / np.sqrt(lmbda_slack)
        ]

        temp = self.lmbda_bar[0] * W_dz[0] - self.lmbda_bar[1].T @ W_dz[1]
        sig = [
            temp / np.sqrt(lmbda_slack),
            (W_dz[1] - (temp + W_dz[0]) / (1 + self.lmbda_bar[0]) * self.lmbda_bar[1]) / np.sqrt(lmbda_slack)
        ]

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
