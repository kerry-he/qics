import numpy as np
from utils import linear as lin

class Cone():
    def __init__(self, n):
        # Dimension properties
        self.n = n # Dimension of system

        self.dim   = [1, 1, n]
        self.type  = ['r', 'r', 'r']

        # Update flags
        self.feas_updated        = False
        self.grad_updated        = False
        self.hess_aux_updated    = False
        self.invhess_aux_updated = False
        self.dder3_aux_updated   = False
        self.congr_aux_updated   = False

        return
        
    def get_nu(self):
        return 2 + self.n
    
    def get_init_point(self, out):
        (t0, x0, u0) = get_central_ray_relentr(self.n)

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
    
    def set_point(self, point, dual, a=True):
        self.t = point[0] * a
        self.u = point[1] * a
        self.x = point[2] * a

        self.t_d = dual[0] * a
        self.u_d = dual[1] * a
        self.x_d = dual[2] * a

        self.feas_updated        = False
        self.grad_updated        = False
        self.hess_aux_updated    = False
        self.invhess_aux_updated = False
        self.dder3_aux_updated   = False

        return
    
    def get_feas(self):
        if self.feas_updated:
            return self.feas
        
        self.feas_updated = True

        if (self.u <= 0) or any(self.x <= 0):
            self.feas = False
            return self.feas
        
        self.sum_x     = np.sum(self.x)
        self.log_x     = np.log(self.x)
        self.log_u     = np.log(self.u[0, 0])
        self.log_sum_x = np.log(self.sum_x)

        entr_x   = self.x.T @ self.log_x
        entr_xu  = self.sum_x * self.log_u

        self.z = (self.t - (entr_x - entr_xu))[0, 0]

        self.feas = (self.z > 0)
        return self.feas
    
    def update_grad(self):
        assert self.feas_updated
        assert not self.grad_updated

        self.zi = np.reciprocal(self.z)
        self.ui = np.reciprocal(self.u)
        self.xi = np.reciprocal(self.x)

        self.DPhiu = -self.sum_x * self.ui
        self.DPhiX = self.log_x + (1. - self.log_u)

        self.grad = [
           -self.zi,
            self.zi * self.DPhiu - self.ui,
            self.zi * self.DPhiX - self.xi
        ]        

        self.grad_updated = True

    def get_grad(self, out):
        assert self.feas_updated
        if not self.grad_updated:
            self.update_grad()

        out[0][:] = self.grad[0]
        out[1][:] = self.grad[1]
        out[2][:] = self.grad[2]
        
        return out

    def hess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        # Computes Hessian product of the CE barrier with a single vector (Ht, Hu, Hx)
        # See hess_congr() for additional comments

        (Ht, Hu, Hx) = H

        # Hessian product of classical entropy
        D2PhiuH =  Hu * self.sum_x * self.ui2 - np.sum(Hx) * self.ui
        D2PhixH = -Hu * self.ui               +        Hx  * self.xi
        
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

        # Precompute Hessian products for quantum entropy
        # D2_uu Phi(u, x) [Hu] =  sum(x) Hu / u^2
        # D2_ux Phi(u, x) [Hx] = -sum(Hx) / u
        # D2_xu Phi(u, x) [Hu] = -Hu / u
        # D2_xx Phi(u, x) [Hx] =  Hx / x
        D2PhiuH  = self.Huu * self.Au
        D2PhiuH += self.Hux * np.sum(self.Ax, axis=1)

        np.multiply(self.Hxx.T, self.Ax, out=self.work1)
        self.work1 += self.Hux * self.Au.reshape(-1, 1)

        # ====================================================================
        # Hessian products with respect to t
        # ====================================================================
        # D2_tt F(t, u, x)[Ht] = Ht / z^2
        # D2_tu F(t, u, x)[Hu] = -(D_u Phi(u, x) [Hu]) / z^2
        # D2_tx F(t, u, x)[Hx] = -(D_X Phi(u, x) [Hx]) / z^2
        outt  = self.At - self.Au * self.DPhiu[0, 0]
        outt -= (self.Ax @ self.DPhiX).ravel()
        outt *= self.zi2

        lhs[:, 0] = outt

        # ====================================================================
        # Hessian products with respect to u
        # ====================================================================
        # D2_ut F(t, u, x)[Ht] = -Ht (D_u Phi(u, x)) / z^2
        # D2_uu F(t, u, x)[Hu] = (D_u Phi(u, x) [Hu]) D_u Phi(u, x) / z^2 + (D2_uu Phi(u, x) [Hu]) / z + Hu / u^2
        # D2_ux F(t, u, x)[Hx] = (D_x Phi(u, x) [Hx]) D_u Phi(u, x) / z^2 + (D2_ux Phi(u, x) [Hx]) / z
        outu  = -outt * self.DPhiu
        outu += D2PhiuH

        lhs[:, 1] = outu

        # ====================================================================
        # Hessian products with respect to x
        # ====================================================================
        # D2_ut F(t, u, X)[Ht] = -Ht (D_u Phi(u, x)) / z^2
        # D2_xu F(t, u, X)[Hu] = (D_u Phi(u, x) [Hu]) D_x Phi(u, x) / z^2 + (D2_xu Phi(u, x) [Hu]) / z
        # D2_xx F(t, u, X)[Hx] = (D_x Phi(u, x) [Hx]) D_x Phi(u, x) / z^2 + (D2_xx Phi(u, x) [Hx]) / z + Hx / x^2
        np.outer(outt, self.DPhiX, out=self.work2)
        self.work1 -= self.work2

        lhs[:, 2:] = self.work1

        # Multiply A (H A')
        return lhs @ A.T

    def invhess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        # Computes Hessian product of the CE barrier with a single vector (Ht, Hu, Hx)
        # See invhess_congr() for additional comments

        (Ht, Hu, Hx) = H

        Wu = Hu + Ht * self.DPhiu
        Wx = Hx + Ht * self.DPhiX

        # Hessian product of classical entropy
        outu = self.rho * (Wu - self.Hxx_inv_Hux.T @ Wx)
        outx = self.Hxx_inv * Wx - outu * self.Hxx_inv_Hux
        
        # Hessian product of barrier function
        out[0][:] = Ht * self.z2 + outu * self.DPhiu + outx.T @ self.DPhiX
        out[1][:] = outu
        out[2][:] = outx

        return out

    def invhess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)

        # The inverse Hessian product applied on (Ht, Hu, Hx) for the CE barrier is 
        #     (u, x) =  M \ (Wu, Wx)
        #         t  =  z^2 Ht + <DPhi(u, x), (u, x)>
        # where (Wu, Wx) = [(Hu, Hx) + Ht DPhi(u, x)]
        #     M = [ (1 + sum(x)/z) / u^2        -1' / zu       ] = [ a  b']
        #         [        -1 / zu         diag(1/zx + 1/x^2)  ]   [ b  D ]
        #
        # To solve linear systems with M, we simplify it by doing block elimination, in which case we get
        #     u = (Wu - b' D^-1 Wx) / (a - b' D^-1 b)
        #     x = D^-1 (Wx - Wu b)

        p = A.shape[0]
        lhs = np.zeros((p, sum(self.dim)))

        # Compute Wu
        Wu = self.Au + self.At * self.DPhiu[0, 0]

        # Compute Wx
        np.outer(self.At, self.DPhiX, out=self.work1)
        self.work1 += self.Ax

        # ====================================================================
        # Inverse Hessian products with respect to u
        # ====================================================================
        outu = self.rho * (Wu - (self.work1 @ self.Hxx_inv_Hux).ravel())
        lhs[:, 1] = outu

        # ====================================================================
        # Inverse Hessian products with respect to x
        # ====================================================================
        self.work1 *= self.Hxx_inv.T
        np.outer(outu, self.Hxx_inv_Hux, out=self.work2)
        self.work1 -= self.work2
        lhs[:, 2:] = self.work1

        # ====================================================================
        # Inverse Hessian products with respect to t
        # ====================================================================
        outt  = self.z2 * self.At 
        outt += outu * self.DPhiu[0, 0]
        outt += (self.work1 @ self.DPhiX).ravel()
        lhs[:, 0] = outt
        
        # Multiply A (H A')
        return lhs @ A.T

    def third_dir_deriv_axpy(self, out, dir, a=True):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.dder3_aux_updated:
            self.update_dder3_aux()

        (Ht, Hu, Hx) = dir

        Hu2    = Hu * Hu
        Hx2    = Hx * Hx
        sum_Hx = np.sum(Hx)

        chi = (Ht - self.DPhiu * Hu - self.DPhiX.T @ Hx)[0, 0]
        chi2 = chi * chi

        # Classical entropy Hessians
        D2PhiuH =  Hu * self.sum_x * self.ui2 - np.sum(Hx) * self.ui
        D2PhixH = -Hu * self.ui               +        Hx  * self.xi

        D2PhiuHH = Hu   * D2PhiuH
        D2PhixHH = Hx.T @ D2PhixH

        # Classical entropy third order derivatives
        D3PhiuHH  = -2 * Hu2 * self.sum_x * self.ui3
        D3PhiuHH +=  2 * Hu * sum_Hx * self.ui2

        D3PhixHH  = -Hx2 * self.xi2
        D3PhixHH +=  Hu2 * self.ui2

        # Third derivatives of barrier
        dder3_t = -2 * self.zi3 * chi2 - self.zi2 * (D2PhixHH + D2PhiuHH)

        dder3_u  = -dder3_t * self.DPhiu
        dder3_u -=  2 * self.zi2 * chi * D2PhiuH
        dder3_u +=  self.zi * D3PhiuHH
        dder3_u -=  2 * Hu2 * self.ui3

        dder3_x  = -dder3_t * self.DPhiX
        dder3_x -=  2 * self.zi2 * chi * D2PhixH
        dder3_x +=  self.zi * D3PhixHH
        dder3_x -=  2 * Hx2 * self.xi3

        out[0][:] += dder3_t * a
        out[1][:] += dder3_u * a
        out[2][:] += dder3_x * a

        return out

    def prox(self):
        assert self.feas_updated
        if not self.grad_updated:
            self.update_grad()
        psi = [
            self.t_d + self.grad[0],
            self.u_d + self.grad[1],
            self.x_d + self.grad[2]
        ]
        temp = [np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((self.n, 1))]
        self.invhess_prod_ip(temp, psi)
        return lin.inp(temp[0], psi[0]) + lin.inp(temp[1], psi[1]) + lin.inp(temp[2], psi[2])

    # ========================================================================
    # Auxilliary functions
    # ========================================================================
    def congr_aux(self, A):
        assert not self.congr_aux_updated

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

        self.Huu = (self.zi * self.sum_x + 1.) * self.ui2
        self.Hux = -self.zi * self.ui[0, 0]
        self.Hxx =  self.zi * self.xi + self.xi2 

        self.hess_aux_updated = True

        return
    
    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated

        self.Hxx_inv     = np.reciprocal(self.Hxx)
        self.Hxx_inv_Hux = self.Hxx_inv * self.Hux
        self.rho         = 1. / (self.Huu - np.sum(self.Hxx_inv_Hux) * self.Hux)[0, 0]

        self.z2 = self.z * self.z

        self.invhess_aux_updated = True

    def update_dder3_aux(self):
        assert not self.dder3_aux_updated
        assert self.hess_aux_updated

        self.zi3 = self.zi * self.zi2
        self.ui3 = self.ui * self.ui2
        self.xi3 = self.xi * self.xi2

        self.dder3_aux_updated = True

        return

def get_central_ray_relentr(x_dim):
    if x_dim <= 10:
        return central_rays_relentr[x_dim - 1, :]
    
    # use nonlinear fit for higher dimensions
    rtx_dim = np.sqrt(x_dim)
    if x_dim <= 20:
        t = 1.2023 / rtx_dim - 0.015
        x = -0.3057 / rtx_dim + 0.972
        y = 0.432 / rtx_dim + 1.0125
    else:
        t = 1.1513 / rtx_dim - 0.0069
        x = -0.4247 / rtx_dim + 0.9961
        y = 0.4873 / rtx_dim + 1.0008

    return [t, x, y]

central_rays_relentr = np.array([
    [0.827838399065679, 0.805102001584795, 1.290927709856958],
    [0.708612491381680, 0.818070436209846, 1.256859152780596],
    [0.622618845069333, 0.829317078332457, 1.231401007595669],
    [0.558111266369854, 0.838978355564968, 1.211710886507694],
    [0.508038610665358, 0.847300430936648, 1.196018952086134],
    [0.468039614334303, 0.854521306762642, 1.183194752717249],
    [0.435316653088949, 0.860840990717540, 1.172492396103674],
    [0.408009282337263, 0.866420016062860, 1.163403373278751],
    [0.384838611966541, 0.871385497883771, 1.155570329098724],
    [0.364899121739834, 0.875838067970643, 1.148735192195660]
])