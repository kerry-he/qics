import numpy as np
import scipy as sp
from qics.cones.base import Cone


class SecondOrder(Cone):
    """A class representing a second order cone

    .. math::

        \\mathcal{Q}_{n+1} = \\{ (t, x) \\in \\mathbb{R} \\times \\mathbb{R}^{n} : t \\geq \\| x \\|_2 \\},

    with barrier function

    .. math::

        (t, x) \\mapsto -\\log(t^2 - x^\\top x).

    Parameters
    ----------
    n : int
        Dimension of the vector :math:`x`, i.e., how many terms are in the Euclidean 
        norm.
    """

    def __init__(self, n):
        # Dimension properties
        self.n = n
        self.nu = 1

        self.dim = [1, n]
        self.type = ["r", "r"]

        self.congr_aux_updated = False
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
        return self.t > np.sqrt(self.x.T @ self.x)

    def get_val(self):
        return -0.5 * np.log(self.t * self.t - self.x.T @ self.x)[0, 0]

    def update_grad(self):
        assert self.feas_updated
        assert not self.grad_updated

        self.z = (self.t * self.t - self.x.T @ self.x)[0, 0]
        self.zi = 1 / self.z

        self.grad = [-self.t * self.zi, self.x * self.zi]

        self.grad_updated = True

    def hess_prod_ip(self, out, H):
        assert self.grad_updated

        (Ht, Hx) = H

        w = self.t * Ht - self.x.T @ Hx
        coeff = 2 * w * self.zi * self.zi

        out[0][:] = coeff * self.t - Ht * self.zi
        out[1][:] = -coeff * self.x + Hx * self.zi

        return out

    def hess_congr(self, A):
        assert self.grad_updated
        if not self.congr_aux_updated:
            self.congr_aux(A)

        # First term
        lhs = self.t * self.At.T - self.x.T @ self.Ax.T
        lhs *= np.sqrt(2.0) * self.zi
        out = lhs.T @ lhs

        # Second term
        out -= (self.At @ self.At.T) * self.zi
        out += (self.Ax @ self.Ax.T) * self.zi

        return out

    def invhess_prod_ip(self, out, H):
        assert self.grad_updated

        (Ht, Hx) = H

        w = self.t * Ht + self.x.T @ Hx
        coeff = 2 * w

        out[0][:] = coeff * self.t - Ht * self.z
        out[1][:] = coeff * self.x + Hx * self.z

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
        out -= (self.At @ self.At.T) * self.z
        out += (self.Ax @ self.Ax.T) * self.z

        return out

    def third_dir_deriv_axpy(self, out, H, a=True):
        assert self.grad_updated

        (Ht, Hx) = H

        self.zi2 = self.zi * self.zi
        self.zi3 = self.zi * self.zi2

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
            self.zi2 * (D2PhitHH + D2PhixHH) - 2 * self.zi3 * coeff1 * coeff1
        ) * 0.5

        dder3_t = coeff2 * DPhit
        dder3_t += coeff1 * self.zi2 * D2PhittH

        dder3_x = coeff2 * DPhix
        dder3_x += coeff1 * self.zi2 * D2PhixxH

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

    def nt_prod_ip(self, out, H):
        pass

    def nt_congr(self, A):
        pass

    def invnt_prod_ip(self, out, H):
        pass

    def invnt_congr(self, A):
        pass

    def comb_dir(self, out, ds, dz, sigma_mu):
        # Compute the residual for rs where rs is given as the lhs of
        #     Lambda o (W dz + W^-T ds) = -Lambda o Lambda - (W^-T ds_a) o (W dz_a)
        #                                 + sigma * mu * 1
        # which is rearranged into the form H ds + dz = rs, i.e.,
        #     rs := W^-1 [ Lambda \ (-Lambda o Lambda - (W^-T ds_a) o (W dz_a) + sigma*mu 1) ]
        # See: [Section 5.4]https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf
        pass

    def step_to_boundary(self, ds, dz):
        # Compute the maximum step alpha in [0, 1] we can take such that
        #     s + alpha ds >= 0
        #     z + alpha dz >= 0
        # See: [Section 8.3]https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf
        pass

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
