import numpy as np
import scipy as sp
from qics.cones.base import SymCone


class NonNegOrthant(SymCone):
    r"""A class representing a nonnegative orthant

    .. math::

        \mathbb{R}^n_+ = \{ x \in \mathbb{R}^n : x \geq 0 \}.

    Parameters
    ----------
    n : :obj:`int`
        Dimension of the cone.
    """

    def __init__(self, n):
        self.n = n

        self.nu = n  # Barrier parameter

        self.dim = [n]
        self.type = ["r"]

        self.congr_aux_updated = False

        return

    def get_init_point(self, out):
        self.set_point([np.ones((self.n, 1))], [np.ones((self.n, 1))])

        out[0][:] = self.x

        return out

    def set_point(self, primal, dual=None, a=True):
        self.x = primal[0] * a
        self.z = dual[0] * a if (dual is not None) else None

    def set_dual(self, dual, a=True):
        self.z = dual[0] * a

    def get_feas(self):
        if np.any(np.less_equal(self.x, 0)):
            return False

        if self.z is None and np.any(np.less_equal(self.z, 0)):
            return False

        return True

    def get_dual_feas(self):
        return np.all(np.greater(self.z, 0))

    def get_val(self):
        return -np.sum(np.log(self.x))

    def grad_ip(self, out):
        out[0][:] = -np.reciprocal(self.x)
        return out

    def hess_prod_ip(self, out, H):
        out[0][:] = H[0] / (self.x**2)
        return out

    def hess_congr(self, A):
        return self.base_congr(A, np.reciprocal(self.x))

    def invhess_prod_ip(self, out, H):
        out[0][:] = H[0] * (self.x**2)
        return out

    def invhess_congr(self, A):
        return self.base_congr(A, self.x)

    def base_congr(self, A, x):
        if sp.sparse.issparse(A):
            if not hasattr(self, "Ax"):
                self.Ax = A.copy()
            self.Ax.data = A.data * np.take(x, A.col)
            return self.Ax @ self.Ax.T
        else:
            Ax = x * A.T
            return Ax.T @ Ax

    def third_dir_deriv_axpy(self, out, H, a=True):
        out[0] -= 2 * a * H[0] * H[0] / (self.x * self.x * self.x)
        return out

    def prox(self):
        return np.linalg.norm(self.x * self.z - 1, np.inf)

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
        out[0][:] = H[0] * self.z / self.x
        return out

    def nt_congr(self, A):
        return self.base_congr(A, np.sqrt(self.z / self.x))

    def invnt_prod_ip(self, out, H):
        out[0][:] = H[0] * self.x / self.z
        return out

    def invnt_congr(self, A):
        return self.base_congr(A, np.sqrt(self.x / self.z))

    def comb_dir(self, out, ds, dz, sigma_mu):
        # Compute the residual for rs where rs is given as the lhs of
        #     Lambda o (W dz + W^-T ds) = -Lambda o Lambda - (W^-T ds_a) o (W dz_a)
        #                                 + sigma * mu * 1
        # which is rearranged into the form H ds + dz = rs, i.e.,
        #     rs := W^-1 [ Lambda \ (-Lambda o Lambda - (W^-T ds_a) o (W dz_a) + sigma*mu 1) ]
        # See: [Section 5.4]https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf
        out[0][:] = (sigma_mu - ds[0] * dz[0]) / self.x - self.z

    def step_to_boundary(self, ds, dz):
        # Compute the maximum step alpha in [0, 1] we can take such that
        #     s + alpha ds >= 0
        #     z + alpha dz >= 0
        # See: [Section 8.3]https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf

        # Compute rho := ds / s and sig := dz / z
        min_rho = np.min(ds / self.x)
        min_sig = np.min(dz / self.z)

        # Maximum step is given by
        #     alpha := 1 / max(0, -min(rho), -min(sig))
        # Clamp this step between 0 and 1
        if min_rho >= 0 and min_sig >= 0:
            return 1.0
        else:
            return 1.0 / max(-min_rho, -min_sig)
