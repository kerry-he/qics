import numpy as np
import math
from utils import symmetric as sym

class QuantRelEntropy():
    def __init__(self, n):
        # Dimension properties
        self.n                              # Side dimension of system
        self.vn = sym.vec_dim(self.n_sys)   # Vector dimension of system
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
        point[self.idx_X] = sym.mat_to_vec(np.eye(self.N)) / self.N
        point[self.idx_Y] = sym.mat_to_vec(np.eye(self.N)) / self.N

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

        self.Dx, self.Ux = np.linalg.eig(self.X)
        self.Dy, self.Uy = np.linalg.eig(self.Y)

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
        self.inv_Dx = np.reciprocal(self.Dy)
        self.inv_X  = (self.Ux * self.inv_Dx) @ self.Ux.T
        self.inv_Y  = (self.Uy * self.inv_Dy) @ self.Uy.T

        self.D1y_log = D1_log(self.Dy, self.log_Dy)

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
        assert ~self.hess_aux_updated
        assert self.grad_updated

        self.D1x_log = D1_log(self.Dx, self.log_Dx)
        self.D2y_log = D2_log(self.Dy, self.D1y_log)

        self.D2PhiXX = np.empty((self.vn, self.vn))
        self.D2PhiXY = np.empty((self.vn, self.vn))
        self.D2PhiYY = np.empty((self.vn, self.vn))

        self.hess_aux_updated = True

        return

    def hess_prod(self, dirs):
        assert self.grad_updated
        if ~self.hess_aux_updated:
            self.update_hessprod_aux()

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        for j in range(p):
            Ht = dirs[0, j]
            Hx = sym.vec_to_mat(dirs[1:, j])
            Hy = sym.p_tr(Hx, self.sys, (self.n0, self.n1))

            UxHxUx = self.Ux.T @ Hx @ self.Ux
            UyHyUy = self.Uy.T @ Hy @ self.Uy

            # Hessian product of conditional entropy
            D2PhiH = self.Ux @ (self.D1x_log * UxHxUx) @ self.Ux.T
            D2PhiH -= sym.i_kr(self.Uy @ (self.D1y_log * UyHyUy) @ self.Uy.T, self.sys, (self.n0, self.n1))

            # Hessian product of barrier function
            out[0, j] = (Ht - sym.inner(self.DPhi, Hx)) * self.zi * self.zi
            temp = -self.DPhi * out[0, j] + D2PhiH * self.zi + self.inv_X @ Hx @ self.inv_X
            out[1:, [j]] = sym.mat_to_vec(temp)

        return out
    
    def update_invhessprod_aux(self):
        assert ~self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated

        irt2 = math.sqrt(0.5)

        D1x_inv = 1 / np.outer(self.Dx, self.Dx)
        self.D1x_comb_inv = 1 / (self.D1x_log*self.zi + D1x_inv)

        UxK = np.empty((self.vN, self.vn))
        Hy_inv = np.empty((self.vn, self.vn))

        k = 0
        for j in range(self.n_sys):
            for i in range(j + 1):
                UxK_k = np.zeros((self.N, self.N))

                # Build UxK matrix
                if self.sys == 0:
                    for l in range(self.n0):
                        lhs = self.Ux[self.n1*l + i, :]
                        rhs = self.Ux[self.n1*l + j, :]
                        UxK_k += np.outer(lhs, rhs)
                else:
                    for l in range(self.n1):
                        lhs = self.Ux[self.n1*i + l, :]
                        rhs = self.Ux[self.n1*j + l, :]
                        UxK_k += np.outer(lhs, rhs)

                if i != j:
                    UxK_k = UxK_k + UxK_k.T
                    UxK_k *= irt2

                UxK[:, [k]] = sym.mat_to_vec(UxK_k)

                # Build Hyy^-1 matrix
                lhs = self.Uy[i, :]
                rhs = self.Uy[j, :]
                UyH_k = np.outer(lhs, rhs)

                if i != j:
                    UyH_k = UyH_k + UyH_k.T
                    UyH_k *= irt2

                temp = self.Uy @ (UyH_k * self.z / self.D1y_log) @ self.Uy.T
                Hy_inv[:, [k]] = sym.mat_to_vec(temp)

                k += 1

        temp = sym.mat_to_vec(self.D1x_comb_inv, 1.0)
        UxK_temp = UxK * temp
        KHxK = UxK.T @ UxK_temp
        self.Hy_KHxK = Hy_inv - KHxK

        self.invhess_aux_updated = True

        return

    def invhess_prod(self, dirs):
        assert self.grad_updated
        if ~self.hess_aux_updated:
            self.update_hessprod_aux()
        if ~self.invhess_aux_updated:
            self.update_invhessprod_aux()

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        for j in range(p):
            Ht = dirs[0, j]
            Hx = sym.vec_to_mat(dirs[1:, j])

            Wx = Hx + Ht * self.DPhi

            UxWxUx = self.Ux.T @ Wx @ self.Ux
            Hxx_inv_x = self.Ux @ (self.D1x_comb_inv * UxWxUx) @ self.Ux.T
            rhs_y = -sym.p_tr(Hxx_inv_x, self.sys, (self.n0, self.n1))
            H_inv_g_y = np.linalg.solve(self.Hy_KHxK, sym.mat_to_vec(rhs_y))
            temp = sym.i_kr(sym.vec_to_mat(H_inv_g_y), self.sys, (self.n0, self.n1))

            temp = self.Ux.T @ temp @ self.Ux
            H_inv_w_x = Hxx_inv_x - self.Ux @ (self.D1x_comb_inv * temp) @ self.Ux.T

            out[0, j] = Ht * self.z * self.z + sym.inner(H_inv_w_x, self.DPhi)
            out[1:, [j]] = sym.mat_to_vec(H_inv_w_x)

        return out


def D1_log(D, log_D):
    eps = np.finfo(np.float64).eps
    rteps = np.sqrt(eps)

    n = np.size(D)
    D1 = np.empty((n, n))
    
    for j in range(n):
        for i in range(j):
            d_ij = D[i] - D[j]
            if abs(d_ij) < rteps:
                D1[i, j] = 2 / (D[i] + D[j])
            else:
                D1[i, j] = (log_D[i] - log_D[j]) / d_ij
            D1[j, i] = D1[i, j]

        D1[j, j] = np.reciprocal(D[j])

    return D1

def D2_log(D, D1):
    eps = np.finfo(np.float64).eps
    rteps = np.sqrt(eps)

    n = np.size(D)
    D2 = np.zeros((n, n, n))

    for k in range(n):
        for j in range(k + 1):
            for i in range(j + 1):
                d_jk = D[j] - D[k]
                if abs(d_jk) < rteps:
                    d_ij = D[i] - D[j]
                    if abs(d_ij) < rteps:
                        t = ((3 / (D[i] + D[j] + D[k]))**2) / -2
                    else:
                        t = (D1[i, j] - D1[j, k]) / d_ij
                else:
                    t = (D1[i, j] - D1[i, k]) / d_jk

                D2[i, j, k] = t
                D2[i, k, j] = t
                D2[j, i, k] = t
                D2[j, k, i] = t
                D2[k, i, j] = t
                D2[k, j, i] = t

    return D2