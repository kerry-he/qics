import numpy as np
from utils import symmetric as sym
from utils import linear    as lin
from utils import mtxgrad   as mgrad

class Cone():
    def __init__(self, n, hermitian=False):
        # Dimension properties
        self.n = n                          # Side dimension of system
        self.hermitian = hermitian

        self.dim   = [1, 1, n*n]     if (not hermitian) else [1, 1, 2*n*n]
        self.type  = ['r', 'r', 's'] if (not hermitian) else ['r', 'r', 'h']
        self.dtype = np.float64      if (not hermitian) else np.complex128

        # Update flags
        self.feas_updated        = False
        self.grad_updated        = False
        self.hess_aux_updated    = False
        self.invhess_aux_updated = False
        self.congr_aux_updated   = False
        self.dder3_aux_updated   = False

        return

    def get_nu(self):
        return 2 + self.n
    
    def get_init_point(self, out):
        (t0, x0, u0) = get_central_ray_relentr(self.n)

        point = [
            np.array([[t0]]),
            np.array([[u0]]), 
            np.eye(self.n, dtype=self.dtype) * x0,
        ]

        self.set_point(point, point)

        out[0][:] = point[0]
        out[1][:] = point[1]
        out[2][:] = point[2]

        return out
    
    def set_point(self, point, dual, a=True):
        self.t = point[0] * a
        self.u = point[1] * a
        self.X = point[2] * a

        self.t_d = dual[0] * a
        self.u_d = dual[1] * a
        self.X_d = dual[2] * a

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

        self.Dx, self.Ux = np.linalg.eigh(self.X)

        if (self.u <= 0) or any(self.Dx <= 0):
            self.feas = False
            return self.feas
        
        self.trX    = np.trace(self.X).real
        self.log_Dx = np.log(self.Dx)
        self.log_u  = np.log(self.u[0, 0])

        entr_X  = lin.inp(self.Dx, self.log_Dx)
        entr_Xu = self.trX * self.log_u
        self.z  = self.t[0, 0] - (entr_X - entr_Xu)

        self.feas = (self.z > 0)
        return self.feas

    def get_val(self):
        return -np.log(self.z) - np.sum(self.log_Dx) - self.log_u
    
    def update_grad(self):
        assert self.feas_updated
        assert not self.grad_updated

        self.log_X = (self.Ux * self.log_Dx) @ self.Ux.conj().T
        self.log_X = (self.log_X + self.log_X.conj().T) * 0.5

        self.inv_Dx = np.reciprocal(self.Dx)
        inv_X_rt2 = self.Ux * np.sqrt(self.inv_Dx)
        self.inv_X = inv_X_rt2 @ inv_X_rt2.conj().T

        self.zi    = np.reciprocal(self.z)
        self.ui    = np.reciprocal(self.u)
        self.DPhiu = -self.trX * self.ui
        self.DPhiX = self.log_X + np.eye(self.n) * (1 - self.log_u)

        self.grad = [
           -self.zi,
            self.zi * self.DPhiu - self.ui,
            self.zi * self.DPhiX - self.inv_X
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
    
    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        self.zi2 = self.zi * self.zi
        self.ui2 = self.ui * self.ui

        D1x_inv       = np.reciprocal(np.outer(self.Dx, self.Dx))
        self.D1x_log  = mgrad.D1_log(self.Dx, self.log_Dx)
        self.D1x_comb = self.zi * self.D1x_log + D1x_inv

        self.hess_aux_updated = True

        return

    def hess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        # Computes Hessian product of the QE barrier with a single vector (Ht, Hu, Hx)
        # See hess_congr() for additional comments

        (Ht, Hu, Hx) = H

        UxHxUx = self.Ux.conj().T @ Hx @ self.Ux

        # Hessian product of quantum entropy
        D2PhiuuH = self.trX * Hu * self.ui2
        D2PhiuXH = -np.trace(Hx).real * self.ui
        D2PhiXuH = -np.eye(self.n) * Hu * self.ui
        D2PhiXXH = self.Ux @ (self.D1x_log * UxHxUx) @ self.Ux.conj().T
        
        # Hessian product of barrier function
        out[0][:] = (Ht - self.DPhiu * Hu - lin.inp(self.DPhiX, Hx)) * self.zi2

        out_u     = -out[0] * self.DPhiu
        out_u    +=  self.zi * (D2PhiuuH + D2PhiuXH)
        out_u    +=  Hu * self.ui2
        out[1][:] = out_u

        out_X     = -out[0] * self.DPhiX
        out_X    +=  self.zi * (D2PhiXuH + D2PhiXXH)
        out_X    +=  self.inv_X @ Hx @ self.inv_X
        out_X     = (out_X + out_X.conj().T) * 0.5
        out[2][:] = out_X

        return out

    def congr_aux(self, A):
        assert not self.congr_aux_updated

        self.At = A[:, 0]
        self.Au = A[:, 1]
        Ax = np.ascontiguousarray(A[:, 2:])

        if self.hermitian:
            self.Ax = np.array([Ax_k.reshape((-1, 2)).view(dtype=np.complex128).reshape((self.n, self.n)) for Ax_k in Ax])
        else:
            self.Ax = np.array([Ax_k.reshape((self.n, self.n)) for Ax_k in Ax])

        self.work1 = np.empty_like(self.Ax, dtype=self.dtype)
        self.work2 = np.empty_like(self.Ax, dtype=self.dtype)
        self.work3 = np.empty_like(self.Ax, dtype=self.dtype)

        self.congr_aux_updated = True

    def hess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)            

        p = A.shape[0]
        lhs = np.zeros((p, sum(self.dim)))

        # Precompute Hessian products for quantum entropy
        # D2_uu Phi(u, X) [Hu] =  tr[X] Hu / u^2
        # D2_uX Phi(u, X) [Hx] = -tr[Hx] / u
        # D2_Xu Phi(u, X) [Hu] = -I Hu / u
        # D2_XX Phi(u, X) [Hx] =  Ux [log^[1](Dx) .* (Ux' Hx Ux)] Ux'
        D2PhiuuH = (self.trX * self.ui2) * self.Au
        D2PhiuXH = np.trace(self.Ax, axis1=1, axis2=2).real * self.ui
        # D2PhiXXH
        lin.congr(self.work1, self.Ux.conj().T, self.Ax, self.work2)
        self.work1 *= self.D1x_comb
        lin.congr(self.work3, self.Ux, self.work1, self.work2)
        # D2PhiXuH
        self.work3[:, range(self.n), range(self.n)] -= (self.zi * self.ui[0, 0]) * self.Au.reshape(-1, 1)

        # ====================================================================
        # Hessian products with respect to t
        # ====================================================================
        # D2_tt F(t, u, X)[Ht] = Ht / z^2
        # D2_tu F(t, u, X)[Hu] = -(D_u Phi(u, X) [Hu]) / z^2
        # D2_tX F(t, u, X)[Hx] = -(D_X Phi(u, X) [Hx]) / z^2
        outt  = self.At - (self.Ax.view(dtype=np.float64).reshape((p, 1, -1)) @ self.DPhiX.view(dtype=np.float64).reshape((-1, 1))).ravel()
        outt -= self.Au * self.DPhiu[0, 0]
        outt *= self.zi2

        lhs[:, 0] = outt

        # ====================================================================
        # Hessian products with respect to u
        # ====================================================================
        # D2_ut F(t, u, X)[Ht] = -Ht (D_u Phi(u, X) [Hu]) / z^2
        # D2_uu F(t, u, X)[Hu] = (D_u Phi(u, X) [Hu]) D_u Phi(u, X) / z^2 + (D2_uu Phi(u, X) [Hu]) / z + Hu / u^2
        # D2_uX F(t, u, X)[Hx] = (D_X Phi(u, X) [Hx]) D_u Phi(u, X) / z^2 + (D2_uX Phi(u, X) [Hx]) / z
        outu  = -outt * self.DPhiu
        outu += self.zi * (D2PhiuuH - D2PhiuXH)
        outu += self.Au * self.ui2

        lhs[:, 1] = outu

        # ====================================================================
        # Hessian products with respect to X
        # ====================================================================
        # D2_Xt F(t, u, X)[Ht] = -Ht (D_X Phi(u, X) [Hx]) / z^2
        # D2_Xu F(t, u, X)[Hu] = (D_u Phi(u, X) [Hu]) D_X Phi(u, X) / z^2 + (D2_Xu Phi(u, X) [Hu]) / z
        # D2_XX F(t, u, X)[Hx] = (D_X Phi(u, X) [Hx]) D_X Phi(u, X) / z^2 + (D2_XX Phi(u, X) [Hx]) / z + X^-1 Hx X^-1
        np.outer(outt, self.DPhiX, out=self.work1.reshape((p, -1)))
        self.work3 -= self.work1

        lhs[:, 2:] = self.work3.reshape((p, -1)).view(dtype=np.float64)

        return lhs @ A.T
    
    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated

        self.z2 = self.z * self.z
        self.D1x_comb_inv = np.reciprocal(self.D1x_comb)

        self.invhess_aux_updated = True

        return

    def invhess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        # Computes Hessian product of the QE barrier with a single vector (Ht, Hu, Hx)
        # See hess_congr() for additional comments

        # The inverse Hessian product applied on (Ht, Hx, Hy) for the QRE barrier is 
        #     (u, X) =  M \ (Wu, Wx)
        #         t  =  z^2 Ht + <DPhi(u, X), (u, X)>
        # where (Wu, Wx) = [(Hu, Hx) + Ht DPhi(u, X)]
        #     M = [ (1 + tr[X]/z) / u^2   -vec(I)' / z ]
        #         [     -vec(I) / z             N      ]
        # and
        #     N = (Ux kron Ux) (1/z log + inv)^[1](Dx) (Ux' kron Ux')
        #
        # To solve linear systems with M, we simplify it by doing block elimination, in which case we get
        #     u = (Wu + tr[N \ Wx] / z) / ((1 + tr[X]/z) / u^2 - tr[N \ I] / z^2)
        #     X = (N \ Wx) + u/z (N \ I)
        # where S is the Schur complement matrix of M.

        (Ht, Hu, Hx) = H

        # ====================================================================
        # Inverse Hessian products with respect to u
        # ====================================================================
        Wu = Hu + Ht * self.DPhiu
        Wx = Hx + Ht * self.DPhiX

        N_inv_Wx = self.Ux.conj().T @ Wx @ self.Ux
        N_inv_Wx = self.Ux @ (self.D1x_comb_inv * N_inv_Wx) @ self.Ux.conj().T

        N_inv_I = self.Ux * np.sqrt(np.diag(self.D1x_comb_inv))
        N_inv_I = N_inv_I @ N_inv_I.conj().T

        uz = self.u * self.z
        uz2 = uz * uz
        out[1][:] = (Wu * uz2 + np.trace(N_inv_Wx).real * uz) / ((self.z2 + self.trX * self.z) - np.trace(N_inv_I).real)

        # ====================================================================
        # Inverse Hessian products with respect to X
        # ====================================================================    
        out[2][:] = N_inv_Wx + out[1] * self.zi * self.ui * N_inv_I

       # ====================================================================
        # Inverse Hessian products with respect to t
        # ====================================================================
        out[0][:] = self.z2 * Ht + self.DPhiu * out[1] + lin.inp(self.DPhiX, out[2])

        return out

    def invhess_prod(self, dirs):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        for j in range(p):
            Ht = dirs[0, j]
            Hx = sym.vec_to_mat(dirs[1:, [j]], hermitian=self.hermitian)

            Wx = Hx + Ht * self.DPhi_mat

            UxWUx = self.Ux.conj().T @ Wx @ self.Ux
            Hinv_W = self.Ux @ (self.D1x_comb_inv * UxWUx) @ self.Ux.conj().T

            fac = self.zi * np.trace(Hinv_W).real / (self.trX - self.zi * self.tr_Hinv_tr)
            temp = Hinv_W + fac * self.Hinv_tr
            temp = sym.mat_to_vec(temp, hermitian=self.hermitian)

            out[0, j] = Ht * self.z * self.z + lin.inp(temp, self.DPhi)
            out[1:, [j]] = temp

        return out

    def invhess_prod(self, dirs):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        for j in range(p):
            Ht = dirs[0, j]
            Hx = sym.vec_to_mat(dirs[1:, [j]], hermitian=self.hermitian)

            Wx = Hx + Ht * self.DPhi_mat

            UxWUx = self.Ux.conj().T @ Wx @ self.Ux
            Hinv_W = self.Ux @ (self.D1x_comb_inv * UxWUx) @ self.Ux.conj().T

            fac = self.zi * np.trace(Hinv_W).real / (self.trX - self.zi * self.tr_Hinv_tr)
            temp = Hinv_W + fac * self.Hinv_tr
            temp = sym.mat_to_vec(temp, hermitian=self.hermitian)

            out[0, j] = Ht * self.z * self.z + lin.inp(temp, self.DPhi)
            out[1:, [j]] = temp

        return out

    def update_dder3_aux(self):
        assert not self.dder3_aux_updated
        assert self.hess_aux_updated

        self.zi3 = self.zi * self.zi2
        self.ui3 = self.ui * self.ui2
        self.D2x_log = mgrad.D2_log(self.Dx, self.D1x_log)

        self.dder3_aux_updated = True

        return

    def third_dir_deriv_axpy(self, out, dir, a=True):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.dder3_aux_updated:
            self.update_dder3_aux()

        (Ht, Hu, Hx) = dir

        Hu2 = Hu * Hu
        trHx = np.trace(Hx).real

        chi = (Ht - self.DPhiu * Hu)[0, 0] - lin.inp(self.DPhiX, Hx)
        chi2 = chi * chi

        UxHxUx = self.Ux.conj().T @ Hx @ self.Ux

        # Quantum relative entropy Hessians
        D2PhiuuH = self.trX * Hu * self.ui2
        D2PhiuXH = -np.trace(Hx).real * self.ui
        D2PhiXuH = -np.eye(self.n) * Hu * self.ui
        D2PhiXXH = self.Ux @ (self.D1x_log * UxHxUx) @ self.Ux.conj().T

        D2PhiuHH = lin.inp(Hu, D2PhiuXH + D2PhiuuH)
        D2PhiXHH = lin.inp(Hx, D2PhiXXH + D2PhiXuH)

        # Quantum relative entropy third order derivatives
        D3Phiuuu = -2 * Hu2 * self.trX * self.ui3
        D3PhiuXu = Hu * trHx * self.ui2
        D3PhiuuX = D3PhiuXu

        D3PhiXXX = mgrad.scnd_frechet(self.D2x_log, UxHxUx, UxHxUx, self.Ux)
        D3PhiXuu = (Hu2 * self.ui2) * np.eye(self.n)

        # Third derivatives of barrier
        dder3_t = -2 * self.zi3 * chi2 - self.zi2 * (D2PhiXHH + D2PhiuHH)

        dder3_u  = -dder3_t * self.DPhiu
        dder3_u -=  2 * self.zi2 * chi * (D2PhiuXH + D2PhiuuH)
        dder3_u +=  self.zi * (D3PhiuuX + D3PhiuXu + D3Phiuuu)
        dder3_u -=  2 * Hu2 * self.ui3

        dder3_X  = -dder3_t * self.DPhiX
        dder3_X -=  2 * self.zi2 * chi * (D2PhiXXH + D2PhiXuH)
        dder3_X +=  self.zi * (D3PhiXXX + D3PhiXuu)
        dder3_X -=  2 * self.inv_X @ Hx @ self.inv_X @ Hx @ self.inv_X
        dder3_X  = (dder3_X + dder3_X.conj().T) * 0.5

        out[0][:] += dder3_t * a
        out[1][:] += dder3_u * a
        out[2][:] += dder3_X * a

        return out

    def prox(self):
        assert self.feas_updated
        if not self.grad_updated:
            self.update_grad()
        psi = [
            self.t_d + self.grad[0],
            self.u_d + self.grad[1],
            self.X_d + self.grad[2]
        ]
        temp = [np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((self.n, self.n), dtype=self.dtype)]
        self.invhess_prod_ip(temp, psi)
        return lin.inp(temp[0], psi[0]) + lin.inp(temp[1], psi[1]) + lin.inp(temp[2], psi[2])

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