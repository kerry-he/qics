import numpy as np
import scipy as sp
import math
from utils import symmetric as sym
from utils import linear    as lin
from utils import mtxgrad   as mgrad

class Cone():
    def __init__(self, n, hermitian=False):
        # Dimension properties
        self.n = n                                      # Side dimension of system
        self.hermitian = hermitian                      # Hermitian or symmetric vector space
        self.vn = sym.vec_dim(self.n, self.hermitian)   # Vector dimension of system
        self.dim = 1 + 2 * self.vn                      # Total dimension of cone
        self.use_sqrt = False



        self.dim   = [1, n*n, n*n]   if (not hermitian) else [1, 2*n*n, 2*n*n]
        self.type  = ['r', 's', 's'] if (not hermitian) else ['r', 'h', 'h']
        self.dtype = np.float64      if (not hermitian) else np.complex128


        self.idx_X = slice(1, 1 + self.vn)
        self.idx_Y = slice(1 + self.vn, self.dim)

        # Update flags
        self.feas_updated        = False
        self.grad_updated        = False
        self.hess_aux_updated    = False
        self.invhess_aux_updated = False
        self.dder3_aux_updated   = False
        self.congr_aux_updated   = False

        return
        
    def get_nu(self):
        return 1 + 2 * self.n
    
    def get_init_point(self, out):
        (t0, x0, y0) = get_central_ray_relentr(self.n)

        point = [
            t0, 
            np.eye(self.n, dtype=self.dtype) * x0,
            np.eye(self.n, dtype=self.dtype) * y0,
        ]

        self.set_point(point, point)

        out[0][:] = point[0]
        out[1][:] = point[1]
        out[2][:] = point[2]

        return out
    
    def set_point(self, point, dual=None, a=True):

        self.t = point[0] * a
        self.X = point[1] * a
        self.Y = point[2] * a

        self.t_d = dual[0] * a
        self.X_d = dual[1] * a
        self.Y_d = dual[2] * a        

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

        if any(self.Dx <= 0) or any(self.Dy <= 0):
            self.feas = False
            return self.feas
        
        self.log_Dx = np.log(self.Dx)
        self.log_Dy = np.log(self.Dy)

        self.log_X = (self.Ux * self.log_Dx) @ self.Ux.conj().T
        self.log_Y = (self.Uy * self.log_Dy) @ self.Uy.conj().T
        self.log_XY = self.log_X - self.log_Y
        self.z = self.t - lin.inp(self.X, self.log_XY)

        self.feas = (self.z > 0)
        return self.feas

    def get_val(self):
        assert self.feas_updated

        return -np.log(self.z) - np.sum(self.log_Dx) - np.sum(self.log_Dy)

    def get_grad(self, out=None):
        assert self.feas_updated

        if self.grad_updated:
            if out is not None:
                out[0][:] = self.grad[0]
                out[1][:] = self.grad[1]
                out[2][:] = self.grad[2]
                return out
            else:            
                return self.grad
        
        self.inv_Dx = np.reciprocal(self.Dx)
        self.inv_Dy = np.reciprocal(self.Dy)
        self.inv_X  = (self.Ux * self.inv_Dx) @ self.Ux.conj().T
        self.inv_Y  = (self.Uy * self.inv_Dy) @ self.Uy.conj().T

        self.D1y_log = mgrad.D1_log(self.Dy, self.log_Dy)

        self.UyXUy = self.Uy.conj().T @ self.X @ self.Uy

        self.zi    = np.reciprocal(self.z)
        self.DPhiX = self.log_XY + np.eye(self.n)
        self.DPhiY = -self.Uy @ (self.D1y_log * self.UyXUy) @ self.Uy.conj().T

        self.grad = [
           -self.zi,
            self.zi * self.DPhiX - self.inv_X,
            self.zi * self.DPhiY - self.inv_Y
        ]

        self.grad_updated = True

        if out is not None:
            out[0][:] = self.grad[0]
            out[1][:] = self.grad[1]
            out[2][:] = self.grad[2]
            return out
        else:
            return self.grad
    
    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        self.D1x_log = mgrad.D1_log(self.Dx, self.log_Dx)
        self.D2y_log = mgrad.D2_log(self.Dy, self.D1y_log)
        self.D2y_log_UXU = self.D2y_log * self.UyXUy

        # Preparing other required variables
        self.zi2 = self.zi * self.zi

        self.hess_aux_updated = True

        return

    def hess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        (Ht, Hx, Hy) = H

        UxHxUx = self.Ux.conj().T @ Hx @ self.Ux
        UyHxUy = self.Uy.conj().T @ Hx @ self.Uy
        UyHyUy = self.Uy.conj().T @ Hy @ self.Uy

        # Hessian product of conditional entropy
        D2PhiXXH =  self.Ux @ (self.D1x_log * UxHxUx) @ self.Ux.conj().T
        D2PhiXYH = -self.Uy @ (self.D1y_log * UyHyUy) @ self.Uy.conj().T
        D2PhiYXH = -self.Uy @ (self.D1y_log * UyHxUy) @ self.Uy.conj().T
        D2PhiYYH = -mgrad.scnd_frechet(self.D2y_log_UXU, UyHyUy, U=self.Uy)
        
        # Hessian product of barrier function
        out[0][:] = (Ht - lin.inp(self.DPhiX, Hx) - lin.inp(self.DPhiY, Hy)) * self.zi2

        out[1][:] = -out[0] * self.DPhiX
        out[1]   +=  self.zi * (D2PhiXYH + D2PhiXXH)
        out[1]   +=  self.inv_X @ Hx @ self.inv_X

        out[2][:] = -out[0] * self.DPhiY
        out[2]   +=  self.zi * (D2PhiYXH + D2PhiYYH)
        temp = self.inv_Y @ Hy @ self.inv_Y
        out[2]   +=  temp

        return out

    def congr_aux(self, A):
        assert not self.congr_aux_updated

        self.At = A[:, 0]
        Ax = A[:, 1 : 1+self.dim[1]]
        Ay = A[:, 1+self.dim[1] : 1+self.dim[1]+self.dim[2]]

        self.Ax = np.array([Ax_k.reshape((self.n, self.n)) for Ax_k in Ax])
        self.Ay = np.array([Ay_k.reshape((self.n, self.n)) for Ay_k in Ay])

        self.congr_aux_updated = True

    def hess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)

        p = A.shape[0]
        lhs = np.zeros((p, sum(self.dim)))

        UxHxUx = self.Ux.conj().T @ self.Ax @ self.Ux
        UyHxUy = self.Uy.conj().T @ self.Ax @ self.Uy
        UyHyUy = self.Uy.conj().T @ self.Ay @ self.Uy

        # Hessian product of conditional entropy
        D2PhiXXH =  self.Ux @ (self.D1x_log * UxHxUx) @ self.Ux.conj().T
        D2PhiXYH = -self.Uy @ (self.D1y_log * UyHyUy) @ self.Uy.conj().T
        D2PhiYXH = -self.Uy @ (self.D1y_log * UyHxUy) @ self.Uy.conj().T
        D2PhiYYH = -mgrad.scnd_frechet_multi(self.D2y_log_UXU, UyHyUy, U=self.Uy)
        # D2PhiYYH = np.array([-mgrad.scnd_frechet(self.D2y_log_UXU, UyHyUy_k, U=self.Uy) for UyHyUy_k in UyHyUy])
        
        # Hessian product of barrier function
        outt = (self.At - np.sum(self.DPhiX * self.Ax, axis=(1, 2)) - np.sum(self.DPhiY * self.Ay, axis=(1, 2))) * self.zi2

        outX  = -np.outer(outt, self.DPhiX).reshape((p, self.n, self.n))
        outX +=  self.zi * (D2PhiXYH + D2PhiXXH)
        outX +=  self.inv_X @ self.Ax @ self.inv_X

        outY  = -np.outer(outt, self.DPhiY).reshape((p, self.n, self.n))
        outY +=  self.zi * (D2PhiYXH + D2PhiYYH)
        outY +=  self.inv_Y @ self.Ay @ self.inv_Y

        lhs[:, 0] = outt
        lhs[:, 1 : 1+self.dim[1]] = outX.reshape((p, -1))
        lhs[:, 1+self.dim[1] : 1+self.dim[1]+self.dim[2]] = outY.reshape((p, -1))    

        return lhs @ A.T

    def hess_prod(self, dirs):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        for k in range(p):
            Ht = dirs[0, k]
            Hx = sym.vec_to_mat(dirs[self.idx_X, [k]], hermitian=self.hermitian)
            Hy = sym.vec_to_mat(dirs[self.idx_Y, [k]], hermitian=self.hermitian)

            UxHxUx = self.Ux.conj().T @ Hx @ self.Ux
            UyHxUy = self.Uy.conj().T @ Hx @ self.Uy
            UyHyUy = self.Uy.conj().T @ Hy @ self.Uy

            # Hessian product of conditional entropy
            D2PhiXXH =  self.Ux @ (self.D1x_log * UxHxUx) @ self.Ux.conj().T
            D2PhiXYH = -self.Uy @ (self.D1y_log * UyHyUy) @ self.Uy.conj().T
            D2PhiYXH = -self.Uy @ (self.D1y_log * UyHxUy) @ self.Uy.conj().T
            D2PhiYYH = -mgrad.scnd_frechet(self.D2y_log_UXU, UyHyUy, U=self.Uy)
            
            # Hessian product of barrier function
            out[0, k] = (Ht - lin.inp(self.DPhiX, Hx) - lin.inp(self.DPhiY, Hy)) * self.zi2

            out[self.idx_X, [k]]  = -out[0, k] * self.DPhiX_vec
            out[self.idx_X, [k]] +=  sym.mat_to_vec(self.zi * D2PhiXXH + self.inv_X @ Hx @ self.inv_X, hermitian=self.hermitian)
            out[self.idx_X, [k]] +=  self.zi * sym.mat_to_vec(D2PhiXYH, hermitian=self.hermitian)

            out[self.idx_Y, [k]]  = -out[0, k] * self.DPhiY_vec
            out[self.idx_Y, [k]] +=  self.zi * sym.mat_to_vec(D2PhiYXH, hermitian=self.hermitian)
            out[self.idx_Y, [k]] +=  sym.mat_to_vec(self.zi * D2PhiYYH + self.inv_Y @ Hy @ self.inv_Y, hermitian=self.hermitian)

        return out
    
    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated

        rt2 = math.sqrt(2.0)
        irt2 = math.sqrt(0.5)

        self.z2 = self.z * self.z

        self.D1x_inv = np.reciprocal(np.outer(self.Dx, self.Dx))
        self.D1x_comb_inv = np.reciprocal(self.zi * self.D1x_log + self.D1x_inv)

        # Hessians of quantum relative entropy
        Hxy_Hxx_Hxy = np.empty((self.vn, self.vn))

        self.UyUx = self.Uy.conj().T @ self.Ux

        # D2PhiYY
        Hyy = -mgrad.get_S_matrix(self.D2y_log * (self.zi * self.UyXUy + np.eye(self.n)), rt2, hermitian=self.hermitian)

        # @TODO: Investigate how to make these loops faster
        k = 0
        for j in range(self.n):
            for i in range(j + 1):
                # Hyx @ Hxx @ Hyx   np.outer(self.UyUx[i, :], self.UyUx[j, :])
                temp = self.UyUx.conj().T[:, [i]] @ self.UyUx[[j], :]
                temp = self.D1x_comb_inv * temp
                temp = self.UyUx @ temp @ self.UyUx.conj().T
                temp = self.D1y_log * temp
                if i != j:
                    temp *= (irt2 * self.zi2 * self.D1y_log[i, j])

                    Hxy_Hxx_Hxy[:, [k]] = sym.mat_to_vec(temp + temp.conj().T, hermitian=self.hermitian)

                    if self.hermitian:
                        k += 1
                        temp *= 1j
                        Hxy_Hxx_Hxy[:, [k]] = sym.mat_to_vec(temp + temp.conj().T, hermitian=self.hermitian)
                else:
                    temp *= (self.zi2 * self.D1y_log[i, j])
                    Hxy_Hxx_Hxy[:, [k]] = sym.mat_to_vec(temp, hermitian=self.hermitian)

                k += 1

        # Preparing other required variables
        hess_schur = Hyy - Hxy_Hxx_Hxy
        # self.hess_schur_inv = np.linalg.inv(hess_schur)
        self.hess_schur_fact = lin.fact(hess_schur)

        self.invhess_aux_updated = True

        return
    

    def invhess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        (Ht, Hx, Hy) = H

        Wx = Hx + Ht * self.DPhiX
        Wy = Hy + Ht * self.DPhiY

        temp = self.Ux.conj().T @ Wx @ self.Ux
        temp = self.UyUx @ (self.D1x_comb_inv * temp) @ self.UyUx.conj().T
        temp = -self.Uy @ (self.zi * self.D1y_log * temp) @ self.Uy.conj().T
        temp = self.Uy.conj().T @ (Wy - temp) @ self.Uy
        temp_vec = sym.mat_to_vec(temp, hermitian=self.hermitian)


        temp_vec = lin.fact_solve(self.hess_schur_fact, temp_vec)


        temp = sym.vec_to_mat(temp_vec, hermitian=self.hermitian)
        outY = self.Uy @ temp @ self.Uy.conj().T

        temp = self.Uy.conj().T @ outY @ self.Uy
        temp = -self.Uy @ (self.zi * self.D1y_log * temp) @ self.Uy.conj().T
        temp = Wx - temp
        temp = self.Ux.conj().T @ temp @ self.Ux
        outX = self.Ux @ (self.D1x_comb_inv * temp) @ self.Ux.conj().T

        outt = self.z2 * Ht + lin.inp(self.DPhiX, outX) + lin.inp(self.DPhiY, outY)

        out[0][:] = outt
        out[1][:] = outX
        out[2][:] = outY

        return out
    

    def invhess_prod(self, dirs):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))    

        temp_vec = np.empty((self.vn, p))   
        Wx_list = np.empty((p, self.n, self.n), dtype='complex128') if self.hermitian else np.empty((p, self.n, self.n))

        for k in range(p):
            Ht = dirs[0, k]
            Hx = sym.vec_to_mat(dirs[self.idx_X, [k]], hermitian=self.hermitian)
            Hy = sym.vec_to_mat(dirs[self.idx_Y, [k]], hermitian=self.hermitian)

            Wx = Hx + Ht * self.DPhiX
            Wy = Hy + Ht * self.DPhiY
            Wx_list[k, :, :] = Wx

            temp = self.Ux.conj().T @ Wx @ self.Ux
            temp = self.UyUx @ (self.D1x_comb_inv * temp) @ self.UyUx.conj().T
            temp = -self.Uy @ (self.zi * self.D1y_log * temp) @ self.Uy.conj().T
            temp = self.Uy.conj().T @ (Wy - temp) @ self.Uy
            temp_vec[:, [k]] = sym.mat_to_vec(temp, hermitian=self.hermitian)

        # temp_vec = self.hess_schur_inv @ temp_vec
        temp_vec = lin.fact_solve(self.hess_schur_fact, temp_vec)

        for k in range(p):
            Ht = dirs[0, k]
            Wx = Wx_list[k, :, :]

            temp = sym.vec_to_mat(temp_vec[:, [k]], hermitian=self.hermitian)
            temp = self.Uy @ temp @ self.Uy.conj().T
            outY = sym.mat_to_vec(temp, hermitian=self.hermitian)

            temp = self.Uy.conj().T @ temp @ self.Uy
            temp = -self.Uy @ (self.zi * self.D1y_log * temp) @ self.Uy.conj().T
            temp = Wx - temp
            temp = self.Ux.conj().T @ temp @ self.Ux
            temp = self.Ux @ (self.D1x_comb_inv * temp) @ self.Ux.conj().T
            outX = sym.mat_to_vec(temp, hermitian=self.hermitian)

            outt = self.z2 * Ht + self.DPhiX_vec.conj().T @ outX + self.DPhiY_vec.conj().T @ outY

            out[0, k] = outt
            out[self.idx_X, [k]] = outX
            out[self.idx_Y, [k]] = outY

        return out
    
    def update_dder3_aux(self):
        assert not self.dder3_aux_updated
        assert self.hess_aux_updated

        self.zi3 = self.zi2 * self.zi
        self.D2x_log = mgrad.D2_log(self.Dx, self.D1x_log)

        self.dder3_aux_updated = True

        return

    def third_dir_deriv_axpy(self, out, dir, a=True):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.dder3_aux_updated:
            self.update_dder3_aux()

        (Ht, Hx, Hy) = dir

        chi = Ht[0, 0] - lin.inp(self.DPhiX, Hx) - lin.inp(self.DPhiY, Hy)
        chi2 = chi * chi

        UxHxUx = self.Ux.conj().T @ Hx @ self.Ux
        UyHyUy = self.Uy.conj().T @ Hy @ self.Uy
        UyHxUy = self.Uy.conj().T @ Hx @ self.Uy

        # Quantum relative entropy Hessians
        D2PhiXXH =  self.Ux @ (self.D1x_log * UxHxUx) @ self.Ux.conj().T
        D2PhiXYH = -self.Uy @ (self.D1y_log * UyHyUy) @ self.Uy.conj().T
        D2PhiYXH = -self.Uy @ (self.D1y_log * UyHxUy) @ self.Uy.conj().T
        D2PhiYYH = -mgrad.scnd_frechet(self.D2y_log_UXU, UyHyUy, U=self.Uy)

        D2PhiXHH = lin.inp(Hx, D2PhiXXH + D2PhiXYH)
        D2PhiYHH = lin.inp(Hy, D2PhiYXH + D2PhiYYH)

        # Quantum relative entropy third order derivatives
        D3PhiXXX =  mgrad.scnd_frechet(self.D2x_log, UxHxUx, UxHxUx, self.Ux)
        D3PhiXYY = -mgrad.scnd_frechet(self.D2y_log, UyHyUy, UyHyUy, self.Uy)

        D3PhiYYX = -mgrad.scnd_frechet(self.D2y_log, UyHyUy, UyHxUy, self.Uy)
        D3PhiYXY = D3PhiYYX
        D3PhiYYY = -mgrad.thrd_frechet(self.D2y_log, self.Dy, self.Uy, self.UyXUy, UyHyUy, UyHyUy)
        
        # Third derivatives of barrier
        dder3_t = -2 * self.zi3 * chi2 - self.zi2 * (D2PhiXHH + D2PhiYHH)

        dder3_X  = -dder3_t * self.DPhiX
        dder3_X -=  2 * self.zi2 * chi * (D2PhiXXH + D2PhiXYH)
        dder3_X +=  self.zi * (D3PhiXXX + D3PhiXYY)
        dder3_X -=  2 * self.inv_X @ Hx @ self.inv_X @ Hx @ self.inv_X

        dder3_Y  = -dder3_t * self.DPhiY
        dder3_Y -=  2 * self.zi2 * chi * (D2PhiYXH + D2PhiYYH)
        dder3_Y +=  self.zi * (D3PhiYYX + D3PhiYXY + D3PhiYYY)
        dder3_Y -=  2 * self.inv_Y @ Hy @ self.inv_Y @ Hy @ self.inv_Y

        out[0][:] += dder3_t * a
        out[1][:] += dder3_X * a
        out[2][:] += dder3_Y * a

        return out
    
    def prox(self):
        assert self.feas_updated
        if not self.grad_updated:
            self.get_grad()
        psi = (
            self.t_d + self.grad[0],
            self.X_d + self.grad[1],
            self.Y_d + self.grad[2]
        )
        temp = [np.zeros((1, 1)), np.zeros((self.n, self.n)), np.zeros((self.n, self.n))]
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