import numpy as np
import scipy as sp
import numba as nb
import math
from utils import symmetric as sym
from utils import linear    as lin
from utils import mtxgrad   as mgrad

class Cone():
    def __init__(self, n0, n1, sys, hermitian=False):
        # Dimension properties
        self.n0 = n0          # Dimension of system 0
        self.n1 = n1          # Dimension of system 1
        self.N  = n0 * n1     # Total dimension of bipartite system
        self.hermitian = hermitian

        self.sys   = sys                       # System being traced out
        self.n_sys = n0 if (sys == 1) else n1  # Dimension of system not traced out

        self.vn = sym.vec_dim(self.n_sys, hermitian=hermitian)      # Dimension of vectorized system being traced out
        self.vN = sym.vec_dim(self.N, hermitian=hermitian)          # Dimension of vectorized bipartite system

        self.dim = 1 + self.vN                 # Dimension of the cone
        self.use_sqrt = False

        # Update flags
        self.feas_updated        = False
        self.grad_updated        = False
        self.hess_aux_updated    = False
        self.invhess_aux_updated = False
        self.dder3_aux_updated   = False

        return
        
    def get_nu(self):
        return 1 + self.N
    
    def set_init_point(self):
        point = np.empty((self.dim, 1))
        point[0] = 1.
        point[1:] = sym.mat_to_vec(np.eye(self.N), hermitian=self.hermitian)

        self.set_point(point)

        return point
    
    def set_point(self, point):
        assert np.size(point) == self.dim
        self.point = point

        self.t = point[0]
        self.X = sym.vec_to_mat(point[1:, [0]], hermitian=self.hermitian)
        self.Y = sym.p_tr(self.X, self.sys, (self.n0, self.n1), hermitian=self.hermitian)

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
        self.Dy, self.Uy = np.linalg.eigh(self.Y)

        if any(self.Dx <= 0):
            self.feas = False
            return self.feas
        
        self.log_Dx = np.log(self.Dx)
        self.log_Dy = np.log(self.Dy)

        self.log_X = (self.Ux * self.log_Dx) @ self.Ux.conj().T
        self.log_Y = (self.Uy * self.log_Dy) @ self.Uy.conj().T
        self.log_XY = self.log_X - sym.i_kr(self.log_Y, self.sys, (self.n0, self.n1))
        self.z = self.t - lin.inp(self.X, self.log_XY)

        self.feas = (self.z > 0)
        return self.feas
    
    def get_val(self):
        return -np.log(self.z) - np.sum(self.log_Dx)
    
    def get_grad(self):
        assert self.feas_updated

        if self.grad_updated:
            return self.grad
        
        self.inv_Dx = np.reciprocal(self.Dx)
        self.inv_X  = (self.Ux * self.inv_Dx) @ self.Ux.conj().T

        self.zi = np.reciprocal(self.z)
        self.DPhi = self.log_XY

        self.grad     = np.empty((self.dim, 1))
        self.grad[0]  = -self.zi
        self.grad[1:] = sym.mat_to_vec(self.zi * self.DPhi - self.inv_X, hermitian=self.hermitian)

        self.grad_updated = True
        return self.grad
    
    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        self.D1x_log = mgrad.D1_log(self.Dx, self.log_Dx)
        self.D1y_log = mgrad.D1_log(self.Dy, self.log_Dy)

        self.hess_aux_updated = True

        return

    def hess_prod(self, dirs):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        for j in range(p):
            Ht = dirs[0, j]
            Hx = sym.vec_to_mat(dirs[1:, [j]], hermitian=self.hermitian)
            Hy = sym.p_tr(Hx, self.sys, (self.n0, self.n1), hermitian=self.hermitian)

            UxHxUx = self.Ux.conj().T @ Hx @ self.Ux
            UyHyUy = self.Uy.conj().T @ Hy @ self.Uy

            # Hessian product of conditional entropy
            D2PhiH = self.Ux @ (self.D1x_log * UxHxUx) @ self.Ux.conj().T
            D2PhiH -= sym.i_kr(self.Uy @ (self.D1y_log * UyHyUy) @ self.Uy.conj().T, self.sys, (self.n0, self.n1))

            # Hessian product of barrier function
            out[0, j] = (Ht - lin.inp(self.DPhi, Hx)) * self.zi * self.zi
            temp = -self.DPhi * out[0, j] + D2PhiH * self.zi + self.inv_X @ Hx @ self.inv_X
            out[1:, [j]] = sym.mat_to_vec(temp, hermitian=self.hermitian)

        return out
    
    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated

        irt2 = math.sqrt(0.5)

        D1x_inv = np.reciprocal(np.outer(self.Dx, self.Dx))
        self.D1x_comb_inv = 1 / (self.D1x_log*self.zi + D1x_inv)
        sqrt_D1x_comb_inv = np.sqrt(self.D1x_comb_inv)

        UxK = np.empty((self.vn, self.vN))
        Hy_inv = np.empty((self.vn, self.vn))

        k = 0
        for j in range(self.n_sys):
            for i in range(j):
                # Build UxK matrix
                if self.sys == 0:
                    lhs = self.Ux.conj().T[:, slice(i, self.vN, self.n1)]
                    rhs = self.Ux[slice(j, self.vN, self.n1), :]
                    UxK_k = (lhs @ rhs) * irt2
                else:
                    lhs = self.Ux.conj().T[:, self.n1*i:self.n1*(i+1)]
                    rhs = self.Ux[self.n1*j:self.n1*(j+1), :]
                    UxK_k = (lhs @ rhs) * irt2
                UxK_k *= sqrt_D1x_comb_inv
                UxK[[k], :] = (sym.mat_to_vec(UxK_k + UxK_k.conj().T, hermitian=self.hermitian)).T

                # Build Hyy^-1 matrix
                UyH_k = self.Uy.conj().T[:, [i]] @ self.Uy[[j], :] * irt2
                UyH_k = self.Uy @ (UyH_k * self.z / self.D1y_log) @ self.Uy.conj().T
                Hy_inv[:, [k]] = sym.mat_to_vec(UyH_k + UyH_k.conj().T, hermitian=self.hermitian)

                k += 1

                if self.hermitian:
                    UxK_k *= 1j
                    UxK[[k], :] = (sym.mat_to_vec(UxK_k + UxK_k.conj().T, hermitian=self.hermitian)).T

                    UyH_k *= 1j
                    Hy_inv[:, [k]] = sym.mat_to_vec(UyH_k + UyH_k.conj().T, hermitian=self.hermitian)

                    k += 1

            # Build UxK matrix
            if self.sys == 0:
                lhs = self.Ux.conj().T[:, slice(j, self.vN, self.n1)]
                rhs = self.Ux[slice(j, self.vN, self.n1), :]
                UxK_k = lhs @ rhs
            else:
                lhs = self.Ux.conj().T[:, self.n1*j:self.n1*(j+1)]
                rhs = self.Ux[self.n1*j:self.n1*(j+1), :]
                UxK_k = lhs @ rhs
            UxK_k *= sqrt_D1x_comb_inv
            UxK[[k], :] = (sym.mat_to_vec(UxK_k, hermitian=self.hermitian)).T

            UyH_k = self.Uy.conj().T[:, [j]] @ self.Uy[[j], :]
            UyH_k = self.Uy @ (UyH_k * self.z / self.D1y_log) @ self.Uy.conj().T
            Hy_inv[:, [k]] = sym.mat_to_vec(UyH_k, hermitian=self.hermitian)

            k += 1


        KHxK = UxK @ UxK.T
        Hy_KHxK = Hy_inv - KHxK
        self.Hy_KHxK_fact = lin.fact(Hy_KHxK)

        self.invhess_aux_updated = True

        return

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

            Wx = Hx + Ht * self.DPhi

            UxWxUx = self.Ux.conj().T @ Wx @ self.Ux
            Hxx_inv_x = self.Ux @ (self.D1x_comb_inv * UxWxUx) @ self.Ux.conj().T
            rhs_y = -sym.p_tr(Hxx_inv_x, self.sys, (self.n0, self.n1), hermitian=self.hermitian)
            temp = sym.mat_to_vec(rhs_y, hermitian=self.hermitian)
            H_inv_g_y = lin.fact_solve(self.Hy_KHxK_fact, temp)
            temp = sym.i_kr(sym.vec_to_mat(H_inv_g_y, hermitian=self.hermitian), self.sys, (self.n0, self.n1))

            temp = self.Ux.conj().T @ temp @ self.Ux
            H_inv_w_x = Hxx_inv_x - self.Ux @ (self.D1x_comb_inv * temp) @ self.Ux.conj().T

            out[0, j] = Ht * self.z * self.z + lin.inp(H_inv_w_x, self.DPhi)
            out[1:, [j]] = sym.mat_to_vec(H_inv_w_x, hermitian=self.hermitian)

        return out
    
    def update_dder3_aux(self):
        assert not self.dder3_aux_updated
        assert self.hess_aux_updated

        self.D2x_log = mgrad.D2_log(self.Dx, self.D1x_log)
        self.D2y_log = mgrad.D2_log(self.Dy, self.D1y_log)

        self.dder3_aux_updated = True

        return

    def third_dir_deriv(self, dirs):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.dder3_aux_updated:
            self.update_dder3_aux()

        Ht = dirs[0]
        Hx = sym.vec_to_mat(dirs[1:, [0]], hermitian=self.hermitian)
        Hy = sym.p_tr(Hx, self.sys, (self.n0, self.n1), hermitian=self.hermitian)

        UxHxUx = self.Ux.conj().T @ Hx @ self.Ux
        UyHyUy = self.Uy.conj().T @ Hy @ self.Uy

        # Quantum conditional entropy oracles
        D2PhiH = self.Ux @ (self.D1x_log * UxHxUx) @ self.Ux.conj().T
        D2PhiH -= sym.i_kr(self.Uy @ (self.D1y_log * UyHyUy) @ self.Uy.conj().T, self.sys, (self.n0, self.n1))

        D3PhiHH = mgrad.scnd_frechet(self.D2x_log, UxHxUx, UxHxUx, self.Ux)
        D3PhiHH -= sym.i_kr(mgrad.scnd_frechet(self.D2y_log, UyHyUy, UyHyUy, self.Uy), self.sys, (self.n0, self.n1))

        # Third derivative of barrier
        DPhiH = lin.inp(self.DPhi, Hx)
        D2PhiHH = lin.inp(D2PhiH, Hx)
        chi = Ht - DPhiH

        dder3 = np.empty((self.dim, 1))
        dder3[0] = -2 * (self.zi**3) * (chi**2) - (self.zi**2) * D2PhiHH

        temp = -dder3[0] * self.DPhi
        temp -= 2 * (self.zi**2) * chi * D2PhiH
        temp += self.zi * D3PhiHH
        temp -= 2 * self.inv_X @ Hx @ self.inv_X @ Hx @ self.inv_X
        dder3[1:] = sym.mat_to_vec(temp, hermitian=self.hermitian)

        return dder3

    def norm_invhess(self, x):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        D1x_inv = np.reciprocal(np.outer(self.Dx, self.Dx))
        self.D1x_comb_inv = np.reciprocal(self.D1x_log*self.zi + D1x_inv)

        Ht = x[0, :]
        Hx = sym.vec_to_mat(x[1:, :], hermitian=self.hermitian)

        Wx = Hx + Ht * self.DPhi

        UxWxUx = self.Ux.conj().T @ Wx @ self.Ux
        outX = self.Ux @ (self.D1x_comb_inv * UxWxUx) @ self.Ux.conj().T
        outt = Ht * self.z * self.z + lin.inp(outX, self.DPhi)
        
        return lin.inp(outX, Hx) + (outt * Ht)