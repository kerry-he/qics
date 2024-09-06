import numpy as np
import qics._utils.linalg as lin


class Cone:
    """Base class for cones"""

    def __init__(self):
        pass

    def get_issymmetric(self):
        return False

    def get_iscomplex(self):
        return False

    def zeros(self):
        out = []
        for dim_k, type_k in zip(self.dim, self.type):
            if type_k == "r":
                out += [np.zeros((dim_k, 1))]
            elif type_k == "s":
                n_k = int(np.sqrt(dim_k))
                out += [np.zeros((n_k, n_k))]
            elif type_k == "h":
                n_k = int(np.sqrt(dim_k // 2))
                out += [np.zeros((n_k, n_k), dtype=np.complex128)]
        return out

    def prox(self):
        assert self.feas_updated
        if not self.grad_updated:
            self.update_grad()

        # Proximity measure is given by <psi, H^-1 psi>
        # where psi = z/mu + g(s)
        psi = [dual_k + grad_k for (dual_k, grad_k) in zip(self.dual, self.grad)]

        H_psi = self.zeros()
        self.invhess_prod_ip(H_psi, psi)

        return sum([lin.inp(H_psi_k, psi_k) for (H_psi_k, psi_k) in zip(H_psi, psi)])

    def set_point(self, primal, dual, a=True):
        self.primal = [primal_k * a for primal_k in primal]
        self.dual = [dual_k * a for dual_k in dual]

        self.feas_updated = False
        self.grad_updated = False
        self.hess_aux_updated = False
        self.invhess_aux_updated = False
        self.dder3_aux_updated = False

    def grad_ip(self, out):
        assert self.feas_updated
        if not self.grad_updated:
            self.update_grad()

        for out_k, grad_k in zip(out, self.grad):
            out_k[:] = grad_k

        return out

    def precompute_mat_vec(self, n=None):
        n = self.n if (n is None) else n
        vn = n * n if self.iscomplex else n * (n + 1) // 2
        # Indices to convert to and from compact vectors and matrices
        if self.iscomplex:
            self.diag_idxs = np.append(
                0, np.cumsum([i for i in range(3, 2 * n + 1, 2)])
            )
            self.triu_idxs = np.empty(n * n, dtype=int)
            self.scale = np.empty(n * n)
            k = 0
            for j in range(n):
                for i in range(j):
                    self.triu_idxs[k] = 2 * (j + i * n)
                    self.triu_idxs[k + 1] = 2 * (j + i * n) + 1
                    self.scale[k : k + 2] = np.sqrt(2.0)
                    k += 2
                self.triu_idxs[k] = 2 * (j + j * n)
                self.scale[k] = 1.0
                k += 1
        else:
            self.diag_idxs = np.append(0, np.cumsum([i for i in range(2, n + 1, 1)]))
            self.triu_idxs = np.array(
                [j + i * n for j in range(n) for i in range(j + 1)]
            )
            self.scale = np.array(
                [1 if i == j else np.sqrt(2.0) for j in range(n) for i in range(j + 1)]
            )

        # Computational basis for symmetric/Hermitian matrices
        rt2 = np.sqrt(0.5)
        self.E = np.zeros((vn, n, n), dtype=self.dtype)
        k = 0
        for j in range(n):
            for i in range(j):
                self.E[k, i, j] = rt2
                self.E[k, j, i] = rt2
                k += 1
                if self.iscomplex:
                    self.E[k, i, j] = rt2 * 1j
                    self.E[k, j, i] = rt2 * -1j
                    k += 1
            self.E[k, j, j] = 1.0
            k += 1

    # Functions that the child class has to implement
    def get_init_point(self, out):
        """Returns a central primal-dual point (s0, z0) satisfying

             z0 = -F'(s0)

        and stores this point in-place in out.

        Parameters
        ----------
        out : ndarray or list of ndarray
            Preallocated vector to store the central point in place.
        """
        pass

    def get_feas(self):
        """Returns whether current primal point s is in the interior of the cone K.

        Returns
        ----------
        bool
            Whether current primal point s is in the interior of the cone K.
        """
        pass

    def update_grad(self):
        """Compute the gradient F'(s) of the barrier function and store in self.grad."""
        pass

    def hess_prod_ip(self, out, H):
        """Compute the Hessian product D2F(s)[H] of the barrier function in the
        direction of H, and store this in-place in out.

        Parameters
        ----------
        out : ndarray or list of ndarray
            Preallocated vector to store the Hessian product in place.
        H : ndarray or list of ndarray
            The direction to compute the second derivative in.
        """
        pass

    def hess_congr(self, A):
        """Compute the congruence transform AH(s)A' with the Hessian matrix H of the barrier function.

        Parameters
        ----------
        A : ndarray
            Matrix of size (p, n), where n is the dimension of the cone K.
            Should correspond to a linear constraint matrix from the conic
            program definition.

        Returns
        ----------
        ndarray
            The matrix product AH(s)A'.
        """
        pass

    def invhess_prod_ip(self, out, H):
        """Compute the inverse Hessian product D2F(s)^-1[H] of the barrier function in the
        direction of H, and store this in-place in out.

        Parameters
        ----------
        out : ndarray or list of ndarray
            Preallocated vector to store the inverse Hessian product in place.
        H : ndarray or list of ndarray
            The direction to compute the inverse second derivative in.
        """
        pass

    def invhess_congr(self, A):
        """Compute the congruence transform A(H(s)^-1)A' with the inverse Hessian
        matrix H(s)^-1 of the barrier function.

        Parameters
        ----------
        A : ndarray
            Matrix of size (p, n), where n is the dimension of the cone K.
            Should correspond to a linear constraint matrix from the conic
            program definition.

        Returns
        ----------
        ndarray
            The matrix product A(H(s)^-1)A'.
        """
        pass

    def third_dir_deriv_axpy(self, out, H, a=True):
        """Compute the third directional derivative in direction H and add
        it to an existing vector, i.e.,

            out <-- out + a * D3F(s)[H, H]

        Parameters
        ----------
        out : ndarray or list of ndarray
            Vector to add the third directional derivative to.
        H : ndarray or list of ndarray
            The direction to compute the third directional derivative in.
        a : float, optional
            Amount to scale the third directional derivative by. Default is 1.
        """
        pass


class SymCone(Cone):
    """Base class for symmetric cones. These cones have a NT scaling
    point w and scaled variable lambda such that

        H(w) s = z  <==> lambda := W^-T s = W z

    where H(w) = W^T W.
    """

    def get_issymmetric(self):
        return True

    # Functions that the child class has to implement
    def nt_prod_ip(self, out, H):
        """Compute the Hessian product D2F(w)[H] of the barrier function in the
        direction of H, and store this in-place in out.

        Parameters
        ----------
        out : ndarray or list of ndarray
            Preallocated vector to store the Hessian product in place.
        H : ndarray or list of ndarray
            The direction to compute the second derivative in.
        """
        pass

    def nt_congr(self, A):
        """Compute the congruence transform AH(w)A' with the Hessian matrix H of the barrier function.

        Parameters
        ----------
        A : ndarray
            Matrix of size (p, n), where n is the dimension of the cone K.
            Should correspond to a linear constraint matrix from the conic
            program definition.

        Returns
        ----------
        ndarray
            The matrix product AH(w)A'.
        """
        pass

    def invnt_prod_ip(self, out, H):
        """Compute the inverse Hessian product D2F(w)^-1[H] of the barrier function in the
        direction of H, and store this in-place in out.

        Parameters
        ----------
        out : ndarray or list of ndarray
            Preallocated vector to store the inverse Hessian product in place.
        H : ndarray or list of ndarray
            The direction to compute the inverse second derivative in.
        """
        pass

    def invnt_congr(self, A):
        """Compute the congruence transform A(H(w)^-1)A' with the inverse Hessian
        matrix H(w)^-1 of the barrier function.

        Parameters
        ----------
        A : ndarray
            Matrix of size (p, n), where n is the dimension of the cone K.
            Should correspond to a linear constraint matrix from the conic
            program definition.

        Returns
        ----------
        ndarray
            The matrix product A(H(w)^-1)A'.
        """
        pass

    def comb_dir(self, out, ds, dz, sigma_mu):
        """Compute the residual rs where rs is given as the lhs of

            Lambda o (W dz + W^-T ds) = -Lambda o Lambda - (W^-T ds_a) o (W dz_a)
                                        + sigma * mu * I

        which is rearranged into the form H ds + dz = rs, i.e.,

            rs := W^-1 [ Lambda \ (-Lambda o Lambda - (W^-T ds_a) o (W dz_a) + sigma*mu I) ]

        See: [Section 5.4]https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf

        Parameters
        ----------
        out : ndarray or list of ndarray
            Preallocated vector to store the residual in place.
        ds : ndarray or list of ndarray
            Vector representing the primal step direction.
        dz : ndarray or list of ndarray
            Vector representing the dual step direction.
        sigma_mu : float
            Value of the product simga*mu.
        """
        pass

    def step_to_boundary(self, ds, dz):
        """Compute the maximum step alpha in [0, 1] we can take such that

            s + alpha ds >= 0
            z + alpha dz >= 0

        See: [Section 8.3]https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf

        Parameters
        ----------
        ds : ndarray or list of ndarray
            Vector representing the primal step direction.
        dz : ndarray or list of ndarray
            Vector representing the dual step direction.
        """
        pass


def get_central_ray_relentr(x_dim):
    if x_dim <= 10:
        return central_rays_relentr[x_dim - 1, :]

    # use nonlinear fit for higher dimensions
    rtx_dim = np.sqrt(x_dim)
    if x_dim <= 20:
        t = 1.2023 / rtx_dim - 0.0150
        x = -0.3057 / rtx_dim + 0.9720
        y = 0.4320 / rtx_dim + 1.0125
    else:
        t = 1.1513 / rtx_dim - 0.0069
        x = -0.4247 / rtx_dim + 0.9961
        y = 0.4873 / rtx_dim + 1.0008

    return [t, x, y]


central_rays_relentr = np.array(
    [
        [0.827838399, 0.805102002, 1.290927710],
        [0.708612491, 0.818070436, 1.256859152],
        [0.622618845, 0.829317078, 1.231401007],
        [0.558111266, 0.838978355, 1.211710886],
        [0.508038610, 0.847300430, 1.196018952],
        [0.468039614, 0.854521306, 1.183194752],
        [0.435316653, 0.860840990, 1.172492396],
        [0.408009282, 0.866420016, 1.163403373],
        [0.384838611, 0.871385497, 1.155570329],
        [0.364899121, 0.875838067, 1.148735192],
    ]
)


def get_central_ray_entr(x_dim):
    if x_dim <= 10:
        return central_rays_entr[x_dim - 1, :]

    # use nonlinear fit for higher dimensions
    t = np.exp(-0.9985 * np.log(x_dim) + 0.6255)
    u = 1.6990 - np.exp(-1.0556 * np.log(x_dim) - 0.9553)
    x = 1.0000 - np.exp(-0.9919 * np.log(x_dim) - 0.8646)

    return [t, u, x]


central_rays_entr = np.array(
    [
        [0.827838399, 1.290927710, 0.805102002],
        [0.645835061, 1.452200439, 0.858490216],
        [0.508869537, 1.536239134, 0.890883666],
        [0.412577808, 1.582612762, 0.911745085],
        [0.344114873, 1.610404042, 0.926086795],
        [0.293923498, 1.628354048, 0.936489181],
        [0.255930312, 1.640678700, 0.944357105],
        [0.226334911, 1.649564614, 0.950507042],
        [0.202708200, 1.656225672, 0.955442068],
        [0.183450062, 1.661379149, 0.959487644],
    ]
)


def get_perspective_derivatives(func):
    if func == "log":
        g = lambda x: -np.log(x)
        dg = lambda x: -np.reciprocal(x)
        d2g = lambda x: np.reciprocal(x * x)
        d3g = lambda x: -np.reciprocal(x * x * x) * 2.0

        xg = lambda x: -x * np.log(x)
        dxg = lambda x: -np.log(x) - 1.0
        d2xg = lambda x: -np.reciprocal(x)
        d3xg = lambda x: np.reciprocal(x * x)

        h = lambda x: x * np.log(x)
        dh = lambda x: np.log(x) + 1.0
        d2h = lambda x: np.reciprocal(x)
        d3h = lambda x: -np.reciprocal(x * x)

        xh = lambda x: x * x * np.log(x)
        dxh = lambda x: 2.0 * x * np.log(x) + x
        d2xh = lambda x: 2.0 * np.log(x) + 3.0
        d3xh = lambda x: 2 * np.reciprocal(x)
    elif isinstance(func, (int, float)):
        alpha = func
        if alpha > 0 and alpha < 1:
            sgn = -1
        elif (alpha > 1 and alpha < 2) or (alpha > -1 and alpha < 0):
            sgn = 1

        g = lambda x: sgn * np.power(x, alpha)
        dg = lambda x: sgn * np.power(x, alpha - 1) * alpha
        d2g = lambda x: sgn * np.power(x, alpha - 2) * (alpha * (alpha - 1))
        d3g = (
            lambda x: sgn * np.power(x, alpha - 3) * (alpha * (alpha - 1) * (alpha - 2))
        )

        xg = lambda x: sgn * np.power(x, alpha + 1)
        dxg = lambda x: sgn * np.power(x, alpha) * (alpha + 1)
        d2xg = lambda x: sgn * np.power(x, alpha - 1) * ((alpha + 1) * alpha)
        d3xg = (
            lambda x: sgn * np.power(x, alpha - 2) * ((alpha + 1) * alpha * (alpha - 1))
        )

        beta = 1.0 - alpha
        h = lambda x: sgn * np.power(x, beta)
        dh = lambda x: sgn * np.power(x, beta - 1) * beta
        d2h = lambda x: sgn * np.power(x, beta - 2) * (beta * (beta - 1))
        d3h = lambda x: sgn * np.power(x, beta - 3) * (beta * (beta - 1) * (beta - 2))

        xh = lambda x: sgn * np.power(x, beta + 1)
        dxh = lambda x: sgn * np.power(x, beta) * (beta + 1)
        d2xh = lambda x: sgn * np.power(x, beta - 1) * ((beta + 1) * beta)
        d3xh = lambda x: sgn * np.power(x, beta - 2) * ((beta + 1) * beta * (beta - 1))

    return (g, dg, d2g, d3g, xg, dxg, d2xg, d3xg, h, dh, d2h, d3h, xh, dxh, d2xh, d3xh)
