import numpy as np
import scipy as sp
import math
from utils import symmetric as sym
from utils import linear    as lin
from utils import mtxgrad   as mgrad

class QuantRelEntropy():
    def __init__(self, n):
        # Dimension properties
        self.n = n                          # Side dimension of system
        self.vn = sym.vec_dim(self.n)       # Vector dimension of system
        self.dim = 1 + 2 * self.vn          # Total dimension of cone

        self.idx_X = slice(1, 1 + self.vn)
        self.idx_Y = slice(1 + self.vn, 1 + 2 * self.vn)

        # Update flags
        self.feas_updated        = False
        self.grad_updated        = False
        self.hess_aux_updated    = False
        self.invhess_aux_updated = False
        self.dder3_aux_updated   = False

        return
        
    def get_nu(self):
        return 1 + 2 * self.n
    
    def set_init_point(self):
        point = np.empty((self.dim, 1))
        point[0] = 1.
        point[self.idx_X] = sym.mat_to_vec(np.eye(self.n)) / self.n
        point[self.idx_Y] = sym.mat_to_vec(np.eye(self.n)) / self.n

        self.set_point(point)

        return point
    
    def set_point(self, point):
        assert np.size(point) == self.dim
        self.point = point

        self.t = point[0]
        self.X = sym.vec_to_mat(point[self.idx_X])
        self.Y = sym.vec_to_mat(point[self.idx_Y])

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

        self.log_X = (self.Ux * self.log_Dx) @ self.Ux.T
        self.log_Y = (self.Uy * self.log_Dy) @ self.Uy.T
        self.log_XY = self.log_X - self.log_Y
        self.z = self.t - sym.inner(self.X, self.log_XY)

        self.feas = (self.z > 0)
        return self.feas
    
    def get_grad(self):
        assert self.feas_updated

        if self.grad_updated:
            return self.grad
        
        self.inv_Dx = np.reciprocal(self.Dx)
        self.inv_Dy = np.reciprocal(self.Dy)
        self.inv_X  = (self.Ux * self.inv_Dx) @ self.Ux.T
        self.inv_Y  = (self.Uy * self.inv_Dy) @ self.Uy.T

        self.D1y_log = mgrad.D1_log(self.Dy, self.log_Dy)

        self.UyXUy = self.Uy.T @ self.X @ self.Uy

        self.zi    = np.reciprocal(self.z)
        self.DPhiX = self.log_XY + np.eye(self.n)
        self.DPhiY = -self.Uy @ (self.D1y_log * self.UyXUy) @ self.Uy.T

        self.grad             = np.empty((self.dim, 1))
        self.grad[0]          = -self.zi
        self.grad[self.idx_X] = sym.mat_to_vec(self.zi * self.DPhiX - self.inv_X)
        self.grad[self.idx_Y] = sym.mat_to_vec(self.zi * self.DPhiY - self.inv_Y)

        self.grad_updated = True
        return self.grad
    
    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        self.D1x_log = mgrad.D1_log(self.Dx, self.log_Dx)
        self.D2y_log = mgrad.D2_log(self.Dy, self.D1y_log)

        # Preparing other required variables
        self.zi2 = self.zi * self.zi
        self.DPhiX_vec = sym.mat_to_vec(self.DPhiX)
        self.DPhiY_vec = sym.mat_to_vec(self.DPhiY)

        self.hess_aux_updated = True

        return

    def hess_prod(self, dirs):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        for k in range(p):
            Ht = dirs[0, k]
            Hx = sym.vec_to_mat(dirs[self.idx_X, [k]])
            Hy = sym.vec_to_mat(dirs[self.idx_Y, [k]])

            UxHxUx = self.Ux.T @ Hx @ self.Ux
            UyHxUy = self.Uy.T @ Hx @ self.Uy
            UyHyUy = self.Uy.T @ Hy @ self.Uy

            # Hessian product of conditional entropy
            D2PhiXXH =  self.Ux @ (self.D1x_log * UxHxUx) @ self.Ux.T
            D2PhiXYH = -self.Uy @ (self.D1y_log * UyHyUy) @ self.Uy.T
            D2PhiYXH = -self.Uy @ (self.D1y_log * UyHxUy) @ self.Uy.T
            D2PhiYYH = -mgrad.scnd_frechet(self.D2y_log, self.Uy, UyHyUy, self.UyXUy)
            
            # Hessian product of barrier function
            out[0, k] = (Ht - lin.inp(self.DPhiX, Hx) - lin.inp(self.DPhiY, Hy)) * self.zi2

            out[self.idx_X, [k]]  = -out[0, k] * self.DPhiX_vec
            out[self.idx_X, [k]] +=  sym.mat_to_vec(self.zi * D2PhiXXH + self.inv_X @ Hx @ self.inv_X)
            out[self.idx_X, [k]] +=  self.zi * sym.mat_to_vec(D2PhiXYH)

            out[self.idx_Y, [k]]  = -out[0, k] * self.DPhiY_vec
            out[self.idx_Y, [k]] +=  self.zi * sym.mat_to_vec(D2PhiYXH)
            out[self.idx_Y, [k]] +=  sym.mat_to_vec(self.zi * D2PhiYYH + self.inv_Y @ Hy @ self.inv_Y)

        return out
    
    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated

        rt2 = math.sqrt(2.0)
        irt2 = math.sqrt(0.5)

        self.D1x_inv = np.reciprocal(np.outer(self.Dx, self.Dx))
        self.D1x_comb_inv = np.reciprocal(self.zi * self.D1x_log + self.D1x_inv)

        # Hessians of quantum relative entropy
        Hxy_Hxx_Hxy = np.empty((self.vn, self.vn))

        self.UyUx = self.Uy.T @ self.Ux

        # D2PhiYY
        Hyy = -self.zi * mgrad.get_S_matrix(self.D2y_log * self.UyXUy, rt2)

        # @TODO: Investigate how to make these loops faster
        k = 0
        for j in range(self.n):
            for i in range(j + 1):
                # Hyx @ Hxx @ Hyx
                temp = np.outer(self.UyUx[i, :], self.UyUx[j, :])
                if i != j:
                    temp = temp + temp.T
                    temp *= (irt2 * self.zi2 * self.D1y_log[i, j])
                else:
                    temp *= (self.zi2 * self.D1y_log[i, j])
                temp = self.D1x_comb_inv * temp
                temp = self.UyUx @ temp @ self.UyUx.T
                temp = self.D1y_log * temp

                Hxy_Hxx_Hxy[:, [k]] = sym.mat_to_vec(temp)

                # invYY
                Hyy[k, k] += np.reciprocal(self.Dy[i] * self.Dy[j])

                k += 1

        # Preparing other required variables
        hess_schur = Hyy - (Hxy_Hxx_Hxy + Hxy_Hxx_Hxy.T) / 2
        # self.hess_schur_inv = np.linalg.inv(hess_schur)
        self.hess_schur_fact = lin.fact(hess_schur)

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

        temp_vec = np.empty((self.vn, p))   
        Wx_list = np.empty((p, self.n, self.n))

        for k in range(p):
            Ht = dirs[0, k]
            Hx = sym.vec_to_mat(dirs[self.idx_X, [k]])
            Hy = sym.vec_to_mat(dirs[self.idx_Y, [k]])

            Wx = Hx + Ht * self.DPhiX
            Wy = Hy + Ht * self.DPhiY
            Wx_list[k, :, :] = Wx

            temp = self.Ux.T @ Wx @ self.Ux
            temp = self.UyUx @ (self.D1x_comb_inv * temp) @ self.UyUx.T
            temp = -self.Uy @ (self.zi * self.D1y_log * temp) @ self.Uy.T
            temp = self.Uy.T @ (Wy - temp) @ self.Uy
            temp_vec[:, [k]] = sym.mat_to_vec(temp)

        # temp_vec = self.hess_schur_inv @ temp_vec
        temp_vec = lin.fact_solve(self.hess_schur_fact, temp_vec)

        for k in range(p):
            Ht = dirs[0, k]
            Wx = Wx_list[k, :, :]

            temp = sym.vec_to_mat(temp_vec[:, [k]])
            temp = self.Uy @ temp @ self.Uy.T
            outY = sym.mat_to_vec(temp)

            temp = self.Uy.T @ temp @ self.Uy
            temp = -self.Uy @ (self.zi * self.D1y_log * temp) @ self.Uy.T
            temp = Wx - temp
            temp = self.Ux.T @ temp @ self.Ux
            temp = self.Ux @ (self.D1x_comb_inv * temp) @ self.Ux.T
            outX = sym.mat_to_vec(temp)

            outt = self.z * self.z * Ht + lin.inp(self.DPhiX_vec, outX) + lin.inp(self.DPhiY_vec, outY)      

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

    def third_dir_deriv(self, dirs):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.dder3_aux_updated:
            self.update_dder3_aux()

        Ht = dirs[0, :]
        Hx = sym.vec_to_mat(dirs[self.idx_X, :])
        Hy = sym.vec_to_mat(dirs[self.idx_Y, :])

        out = np.empty((self.dim, 1))

        chi = Ht - lin.inp(self.DPhiX, Hx) - lin.inp(self.DPhiY, Hy)
        chi2 = chi * chi

        UxHxUx = self.Ux.T @ Hx @ self.Ux
        UyHyUy = self.Uy.T @ Hy @ self.Uy
        UyHxUy = self.Uy.T @ Hx @ self.Uy

        # Quantum relative entropy Hessians
        D2PhiXXH =  self.Ux @ (self.D1x_log * UxHxUx) @ self.Ux.T
        D2PhiXYH = -self.Uy @ (self.D1y_log * UyHyUy) @ self.Uy.T
        D2PhiYXH = -self.Uy @ (self.D1y_log * UyHxUy) @ self.Uy.T
        D2PhiYYH = -mgrad.scnd_frechet(self.D2y_log, self.Uy, UyHyUy, self.UyXUy)

        D2PhiXHH = lin.inp(Hx, D2PhiXXH + D2PhiXYH)
        D2PhiYHH = lin.inp(Hy, D2PhiYXH + D2PhiYYH)

        # Quantum relative entropy third order derivatives
        D3PhiXXX =  mgrad.scnd_frechet(self.D2x_log, self.Ux, UxHxUx, UxHxUx)
        D3PhiXYY = -mgrad.scnd_frechet(self.D2y_log, self.Uy, UyHyUy, UyHyUy)

        D3PhiYYX = -mgrad.scnd_frechet(self.D2y_log, self.Uy, UyHyUy, UyHxUy)
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

        out[0]          = dder3_t
        out[self.idx_X] = sym.mat_to_vec(dder3_X)
        out[self.idx_Y] = sym.mat_to_vec(dder3_Y)

        return out

    def norm_invhess(self, x):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        Ht = x[0, :]
        Hx = sym.vec_to_mat(x[self.idx_X, :])
        Hy = sym.vec_to_mat(x[self.idx_Y, :])

        Wx = Hx + Ht * self.DPhiX
        Wy = Hy + Ht * self.DPhiY

        UWxU = self.Uy.T @ Wx @ self.Uy
        UWyU = self.Uy.T @ Wy @ self.Uy

        # Estimate Hessian
        self.D1x_inv = np.reciprocal(np.outer(self.Dx, self.Dx))
        self.D1y_inv = np.reciprocal(np.outer(self.Dy, self.Dy))
        self.D1x_comb = self.zi * self.D1x_log + self.D1x_inv
        x_largest = np.max(self.D1x_comb)

        D2y_UXU = self.D2y_log * self.UyXUy
        S_est = np.zeros((self.n, self.n))
        for k in range(self.n):
            eigs_k = np.linalg.eigvalsh(D2y_UXU[k, :, :])
            y_smallest = np.min(eigs_k)
            S_est[k, :] += y_smallest
            S_est[:, k] += y_smallest

        Dxy = -self.D1y_log * self.zi
        Dyy = -S_est * self.zi + self.D1y_inv

        schur = Dyy - Dxy * Dxy / x_largest

        temp = UWyU - Dxy * UWxU / x_largest
        tempY = temp / schur
        outY = self.Uy @ tempY @ self.Uy.T
        outX = (Wx - (self.Uy @ (Dxy * tempY) @ self.Uy.T)) / x_largest
        outt = self.z * self.z * Ht + lin.inp(self.DPhiX, outX) + lin.inp(self.DPhiY, outY)
        
        return lin.inp(outX, Hx) + lin.inp(outY, Hy) + (outt * Ht)