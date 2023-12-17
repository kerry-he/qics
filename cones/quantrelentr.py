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

        irt2 = math.sqrt(0.5)

        self.D1x_inv = np.reciprocal(np.outer(self.Dx, self.Dx))

        self.D1x_comb_inv = np.reciprocal(self.zi * self.D1x_log + self.D1x_inv)
        self.D1x_comb_inv_sqrt = np.sqrt(self.D1x_comb_inv)

        # Hessians of quantum relative entropy
        D2PhiYY = np.empty((self.vn, self.vn))
        Hyy_Hxx = np.empty((self.vn, self.vn))

        self.Hxx_inv = np.empty((self.vn, self.vn))

        invYY = np.empty((self.vn, self.vn))

        self.UyUx = self.Uy.T @ self.Ux

        k = 0
        for j in range(self.n):
            for i in range(j + 1):
                # Hyx @ Hxx^-0.5
                # UxHUx = np.outer(self.Ux[i, :], self.Ux[j, :])
                # if i != j:
                #     UxHUx = UxHUx + UxHUx.T
                #     UxHUx *= irt2
                # temp = self.Ux @ (self.D1x_comb_inv_sqrt * UxHUx) @ self.Ux.T
                # temp = self.Uy.T @ temp @ self.Uy
                # temp = -self.Uy @ (self.zi * self.D1y_log * temp) @ self.Uy.T
                # Hyy_Hxx[:, [k]] = sym.mat_to_vec(temp) 

                temp = np.outer(self.UyUx.T[i, :], self.UyUx.T[j, :])
                if i != j:
                    temp = temp + temp.T
                    temp *= irt2
                temp = -self.Uy @ (self.zi * self.D1y_log * temp) @ self.Uy.T
                Hyy_Hxx[:, [k]] = sym.mat_to_vec(temp) * self.D1x_comb_inv_sqrt[i, j]

                # D2PhiYY
                UyHUy = np.outer(self.Uy[i, :], self.Uy[j, :])
                if i != j:
                    UyHUy = UyHUy + UyHUy.T
                    UyHUy *= irt2
                temp = -mgrad.scnd_frechet(self.D2y_log, self.Uy, UyHUy, self.UyXUy)
                D2PhiYY[:, [k]] = sym.mat_to_vec(temp)

                # invXX and invYY
                temp = np.outer(self.inv_Y[i, :], self.inv_Y[j, :])
                if i != j:
                    temp = temp + temp.T
                    temp *= irt2
                invYY[:, [k]] = sym.mat_to_vec(temp)

                k += 1

        # Preparing other required variables
        Hyy = self.zi * D2PhiYY + invYY
        hess_schur = Hyy - Hyy_Hxx @ Hyy_Hxx.T
        self.hess_schur_inv = np.linalg.inv(hess_schur)

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

        Ht = dirs[0, :]
        Hx = dirs[1:self.vn+1, :]
        Hy = dirs[self.vn+1:, :]

        Wx = Hx + Ht * self.DPhiX_vec
        Wy = Hy + Ht * self.DPhiY_vec

        Wx_mat = sym.vec_to_mat_multi(Wx)
        temp = self.Ux.T @ Wx_mat @ self.Ux
        temp = self.UyUx @ (self.D1x_comb_inv * temp) @ self.UyUx.T
        temp = -self.Uy @ (self.zi * self.D1y_log * temp) @ self.Uy.T
        temp_vec = sym.mat_to_vec_multi(temp)
        outY = self.hess_schur_inv @ (Wy - temp_vec)

        temp2 = sym.vec_to_mat_multi(outY)
        temp = self.Uy.T @ temp2 @ self.Uy
        temp = -self.Uy @ (self.zi * self.D1y_log * temp) @ self.Uy.T
        temp = Wx_mat - temp
        temp = self.Ux.T @ temp @ self.Ux
        temp = self.Ux @ (self.D1x_comb_inv * temp) @ self.Ux.T
        outX = sym.mat_to_vec_multi(temp)

        outt = self.z * self.z * Ht + lin.inp(self.DPhiX_vec, outX) + lin.inp(self.DPhiY_vec, outY)      

        out[0, :] = outt
        out[self.idx_X, :] = outX
        out[self.idx_Y, :] = outY          

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