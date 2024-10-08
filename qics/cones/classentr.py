import numpy as np
import scipy as sp
import qics._utils.linalg as lin
from qics.cones.base import Cone, get_central_ray_entr


class ClassEntr(Cone):
    r"""A class representing a (homogenized) classical entropy cone

    .. math::

        \mathcal{CE}_{n} = \text{cl}\{ (t, u, x) \in \mathbb{R} \times
        \mathbb{R}_{++} \times \mathbb{R}^n_{++} : t \geq -u H(u^{-1}x) \},

    where

    .. math::

        H(x) = -\sum_{i=1}^n x_i \log(x_i),

    is the classical (Shannon) entropy function. 

    Parameters
    ----------
    n : :obj:`int`
        Dimension of the vector :math:`x`, i.e., how many terms are in the
        classical entropy function.

    See also
    --------
    ClassRelEntr : Classical relative entropy cone
    QuantEntr : (Homogenized) quantum entropy cone

    Notes
    -----
    The epigraph of the classical entropy can be obtained by enforcing the
    linear constraint :math:`u=1`. 
    
    Additionally, the exponential cone

    .. math::

        \mathcal{E}=\{ (x,y,z)\in\mathbb{R}_+\times\mathbb{R}_+
        \times\mathbb{R} : y \geq x \exp(z/x) \},

    can be modelled by realizing that if :math:`(x,y,z)\in\mathcal{E}`,
    then :math:`(-z, y, x)\in\mathcal{CE}_1`.
    """

    def __init__(self, n):
        self.n = n

        self.nu = 2 + self.n  # Barrier parameter

        self.dim = [1, 1, n]
        self.type = ["r", "r", "r"]

        # Update flags
        self.feas_updated = False
        self.grad_updated = False
        self.hess_aux_updated = False
        self.invhess_aux_updated = False
        self.dder3_aux_updated = False
        self.congr_aux_updated = False

        return

    def get_init_point(self, out):
        (t0, u0, x0) = get_central_ray_entr(self.n)

        point = [
            np.array([[t0]]),
            np.array([[u0]]),
            np.ones((self.n, 1)) * x0,
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

        (self.t, self.u, self.x) = self.primal

        # Check that u and x are strictly positive
        if (self.u <= 0) or any(self.x <= 0):
            self.feas = False
            return self.feas

        # Check that t > -u H(x/u) = Σ_i xi log(xi) - (Σ_i xi) log(u)
        self.sum_x = np.sum(self.x)
        self.log_x = np.log(self.x)
        self.log_u = np.log(self.u[0, 0])

        entr_x = self.x.T @ self.log_x
        entr_xu = self.sum_x * self.log_u
        self.z = (self.t - (entr_x - entr_xu))[0, 0]

        self.feas = self.z > 0
        return self.feas

    def get_val(self):
        return -np.log(self.z) - np.sum(self.log_u) - np.sum(self.log_x)

    def update_grad(self):
        assert self.feas_updated
        assert not self.grad_updated

        # Compute gradients of classical entropy
        # D_u H(u, x) = -Σ_i xi / u
        self.ui = np.reciprocal(self.u)
        self.DPhiu = -self.sum_x * self.ui
        # D_x H(u, x) = log(x) + (1 - log(u))
        self.DPhiX = self.log_x + (1.0 - self.log_u)

        # Compute 1 / x
        self.xi = np.reciprocal(self.x)

        # Compute gradient of barrier function
        self.zi = np.reciprocal(self.z)

        self.grad = [
            -self.zi,
            self.zi * self.DPhiu - self.ui,
            self.zi * self.DPhiX - self.xi,
        ]

        self.grad_updated = True

    def hess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        (Ht, Hu, Hx) = H

        # Hessian product of classical entropy
        D2PhiuH = Hu * self.sum_x * self.ui2 - np.sum(Hx) * self.ui
        D2PhixH = -Hu * self.ui + Hx * self.xi

        # Hessian product of barrier function
        out[0][:] = (Ht - Hu * self.DPhiu - Hx.T @ self.DPhiX) * self.zi2
        out[1][:] = -out[0] * self.DPhiu + self.zi * D2PhiuH + Hu * self.ui2
        out[2][:] = -out[0] * self.DPhiX + self.zi * D2PhixH + Hx * self.xi2

        return out

    def hess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)

        p = A.shape[0]
        lhs = np.zeros((p, sum(self.dim)))

        work1, work2 = self.work1, self.work2

        # ======================================================================
        # Hessian products with respect to t
        # ======================================================================
        # D2_t F(t, u, x)[Ht, Hu, Hx] 
        #         = (Ht - D_u H(u, x)[Hu] - D_x H(u, x)[Hx]) / z^2
        out_t = self.At - self.Au * self.DPhiu[0, 0]
        out_t -= (self.Ax @ self.DPhiX).ravel()
        out_t *= self.zi2

        lhs[:, 0] = out_t

        # ======================================================================
        # Hessian products with respect to u
        # ======================================================================
        # Hessian products for classical entropy
        # D2_uu Phi(u, x) [Hu] =  sum(x) Hu / u^2
        D2PhiuH = self.Huu * self.Au
        # D2_ux Phi(u, x) [Hx] = -sum(Hx) / u
        D2PhiuH += self.Hux * np.sum(self.Ax, axis=1)

        # Hessian product of barrier function
        # D2_u F(t, u, x)[Ht, Hu, Hx] 
        #         = -D2_t F(t, u, x)[Ht, Hu, Hx] * D_u H(u, x)
        #           + (D2_uu H(u, x)[Hu] + D2_ux H(u, x)[Hx]) / z
        #           + Hu / u^2
        out_u = -out_t * self.DPhiu
        out_u += D2PhiuH

        lhs[:, 1] = out_u

        # ======================================================================
        # Hessian products with respect to x
        # ======================================================================
        # Hessian products for classical entropy
        # D2_xx Phi(u, x) [Hx] =  Hx / x
        np.multiply(self.Hxx.T, self.Ax, out=work1)
        # D2_xu Phi(u, x) [Hu] = -Hu / u
        work1 += self.Hux * self.Au.reshape(-1, 1)

        # Hessian product of barrier function
        # D2_x F(t, u, x)[Ht, Hu, Hx] 
        #         = -D2_t F(t, u, x)[Ht, Hu, Hx] * D_x H(u, x)
        #           + (D2_xu H(u, x)[Hu] + D2_xx H(u, x)[Hx]) / z
        #           + Hx / x^2
        np.outer(out_t, self.DPhiX, out=work2)
        work1 -= work2

        lhs[:, 2:] = work1

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

        # Inverse Hessian product of classical entropy
        out_u = self.rho * (Wu - self.Hxx_inv_Hux.T @ Wx)
        out_x = self.Hxx_inv * Wx - out_u * self.Hxx_inv_Hux

        # Inverse Hessian product of barrier function
        out[0][:] = Ht * self.z2 + out_u * self.DPhiu + out_x.T @ self.DPhiX
        out[1][:] = out_u
        out[2][:] = out_x

        return out

    def invhess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)

        # The inverse Hessian product applied on (Ht, Hu, Hx) for the CE barrier
        # is
        #     (u, x) =  M \ (Wu, Wx)
        #         t  =  z^2 Ht + <DPhi(u, x), (u, x)>
        # where (Wu, Wx) = [(Hu, Hx) + Ht DPhi(u, x)]
        #     M = [ (1 + sum(x)/z) / u^2        -1' / zu       ] = [ a  b']
        #         [        -1 / zu         diag(1/zx + 1/x^2)  ]   [ b  D ]
        #
        # To solve linear systems with M, we simplify it by doing block
        # elimination, in which case we get
        #     u = (Wu - b' D^-1 Wx) / (a - b' D^-1 b)
        #     x = D^-1 (Wx - Wu b)

        p = A.shape[0]
        lhs = np.zeros((p, sum(self.dim)))

        # Compute Wu
        Wu = self.Au + self.At * self.DPhiu[0, 0]

        # Compute Wx
        np.outer(self.At, self.DPhiX, out=self.work1)
        self.work1 += self.Ax

        # ======================================================================
        # Inverse Hessian products with respect to u
        # ======================================================================
        # u = (Wu - b' D^-1 Wx) / (a - b' D^-1 b)
        out_u = self.rho * (Wu - (self.work1 @ self.Hxx_inv_Hux).ravel())
        lhs[:, 1] = out_u

        # ======================================================================
        # Inverse Hessian products with respect to x
        # ======================================================================
        # x = D^-1 (Wx - Wu b)
        self.work1 *= self.Hxx_inv.T
        np.outer(out_u, self.Hxx_inv_Hux, out=self.work2)
        self.work1 -= self.work2
        lhs[:, 2:] = self.work1

        # ======================================================================
        # Inverse Hessian products with respect to t
        # ======================================================================
        # t = z^2 Ht + <DH(u, x), (u, x)>
        out_t = self.z2 * self.At
        out_t += out_u * self.DPhiu[0, 0]
        out_t += (self.work1 @ self.DPhiX).ravel()
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
        Hx2 = Hx * Hx
        sum_Hx = np.sum(Hx)

        chi = (Ht - self.DPhiu * Hu - self.DPhiX.T @ Hx)[0, 0]
        chi2 = chi * chi

        # Classical entropy Hessians
        D2PhiuH = Hu * self.sum_x * self.ui2 - np.sum(Hx) * self.ui
        D2PhixH = -Hu * self.ui + Hx * self.xi

        D2PhiuHH = Hu * D2PhiuH
        D2PhixHH = Hx.T @ D2PhixH

        # Classical entropy third order derivatives
        D3PhiuHH = -2 * Hu2 * self.sum_x * self.ui3
        D3PhiuHH += 2 * Hu * sum_Hx * self.ui2

        D3PhixHH = -Hx2 * self.xi2
        D3PhixHH += Hu2 * self.ui2

        # Third derivatives of barrier
        dder3_t = -2 * self.zi3 * chi2 - self.zi2 * (D2PhixHH + D2PhiuHH)

        dder3_u = -dder3_t * self.DPhiu
        dder3_u -= 2 * self.zi2 * chi * D2PhiuH
        dder3_u += self.zi * D3PhiuHH
        dder3_u -= 2 * Hu2 * self.ui3

        dder3_x = -dder3_t * self.DPhiX
        dder3_x -= 2 * self.zi2 * chi * D2PhixH
        dder3_x += self.zi * D3PhixHH
        dder3_x -= 2 * Hx2 * self.xi3

        out[0][:] += dder3_t * a
        out[1][:] += dder3_u * a
        out[2][:] += dder3_x * a

        return out

    # ==========================================================================
    # Auxilliary functions
    # ==========================================================================
    def congr_aux(self, A):
        assert not self.congr_aux_updated

        if sp.sparse.issparse(A):
            A = A.toarray()

        self.At = A[:, 0]
        self.Au = A[:, 1]
        self.Ax = A[:, 2:]

        self.work1 = np.empty_like(self.Ax)
        self.work2 = np.empty_like(self.Ax)

        self.congr_aux_updated = True

    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        self.zi2 = self.zi * self.zi
        self.ui2 = self.ui * self.ui
        self.xi2 = self.xi * self.xi

        self.Huu = (self.zi * self.sum_x + 1.0) * self.ui2
        self.Hux = -self.zi * self.ui[0, 0]
        self.Hxx = self.zi * self.xi + self.xi2

        self.hess_aux_updated = True

    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated

        self.Hxx_inv = np.reciprocal(self.Hxx)
        self.Hxx_inv_Hux = self.Hxx_inv * self.Hux
        self.rho = 1.0 / (self.Huu - np.sum(self.Hxx_inv_Hux) * self.Hux)[0, 0]

        self.z2 = self.z * self.z

        self.invhess_aux_updated = True

    def update_dder3_aux(self):
        assert not self.dder3_aux_updated
        assert self.hess_aux_updated

        self.zi3 = self.zi * self.zi2
        self.ui3 = self.ui * self.ui2
        self.xi3 = self.xi * self.xi2

        self.dder3_aux_updated = True
