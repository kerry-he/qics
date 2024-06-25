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

        self.dim   = [1, n*n, n*n]   if (not hermitian) else [1, 2*n*n, 2*n*n]
        self.type  = ['r', 's', 's'] if (not hermitian) else ['r', 'h', 'h']
        self.dtype = np.float64      if (not hermitian) else np.complex128

        self.idx_X = slice(1, 1 + self.dim[1])
        self.idx_Y = slice(1 + self.dim[1], sum(self.dim))

        # Update flags
        self.feas_updated            = False
        self.grad_updated            = False
        self.hess_aux_updated        = False
        self.invhess_aux_updated     = False
        self.invhess_aux_aux_updated = False
        self.dder3_aux_updated       = False
        self.congr_aux_updated       = False

        return
        
    def get_nu(self):
        return 1 + 2 * self.n
    
    def get_init_point(self, out):
        (t0, x0, y0) = get_central_ray_relentr(self.n)

        point = [
            np.array([[t0]]), 
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
        self.z = self.t[0, 0] - lin.inp(self.X, self.log_XY)

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
        Ax = np.ascontiguousarray(A[:, self.idx_X])
        Ay = np.ascontiguousarray(A[:, self.idx_Y])

        if self.hermitian:
            self.Ax = np.array([Ax_k.reshape((-1, 2)).view(dtype=np.complex128).reshape((self.n, self.n)) for Ax_k in Ax])
            self.Ay = np.array([Ay_k.reshape((-1, 2)).view(dtype=np.complex128).reshape((self.n, self.n)) for Ay_k in Ay])            
        else:
            self.Ax = np.array([Ax_k.reshape((self.n, self.n)) for Ax_k in Ax])
            self.Ay = np.array([Ay_k.reshape((self.n, self.n)) for Ay_k in Ay])

        self.work1 = np.empty_like(self.Ax, dtype=self.dtype)
        self.work2 = np.empty_like(self.Ax, dtype=self.dtype)
        self.work3 = np.empty_like(self.Ax, dtype=self.dtype)
        self.work4 = np.empty((self.Ax.shape[::-1]), dtype=self.dtype)

        self.D2PhiXXH = np.empty_like(self.Ax, dtype=self.dtype)
        self.D2PhiYXH = np.empty_like(self.Ax, dtype=self.dtype)
        self.D2PhiXYH = np.empty_like(self.Ay, dtype=self.dtype)
        self.D2PhiYYH = np.empty_like(self.Ay, dtype=self.dtype)

        self.congr_aux_updated = True

    def hess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)

        p = A.shape[0]
        lhs = np.zeros((p, sum(self.dim)))

        # Hessian products for quantum relative entropy:
        #    D2_XX Phi(X, Y) [Hx] =  Ux [log^[1](Dx) .* (Ux' Hx Ux)] Ux'
        #    D2_YX Phi(X, Y) [Hx] = -Uy [log^[1](Dy) .* (Uy' Hx Uy)] Uy'
        #    D2_XY Phi(X, Y) [Hy] = -Uy [log^[1](Dy) .* (Uy' Hy Uy)] Uy'
        #    D2_YY Phi(X, Y) [Hy] = -Uy [SUM_k log_k^[2](Dy) .* ([Uy' X Uy]_k [Uk' Hy Uy]_k' + [Uy' Hy Uy]_k [Uk' X Uy]_k') ] Uy'
        congr(self.work1, self.Ux.conj().T, self.Ax, self.work2)
        self.work1 *= self.D1x_log
        congr(self.D2PhiXXH, self.Ux, self.work1, self.work2)

        congr(self.work1, self.Uy.conj().T, self.Ax, self.work2)
        self.work1 *= self.D1y_log
        congr(self.D2PhiYXH, self.Uy, self.work1, self.work2)

        congr(self.work1, self.Uy.conj().T, self.Ay, self.work2)
        mgrad.scnd_frechet_multi(self.D2PhiYYH, self.D2y_log_UXU, self.work1, U=self.Uy, work1=self.work2, work2=self.work3, work3=self.work4)

        self.work1 *= self.D1y_log
        congr(self.D2PhiXYH, self.Uy, self.work1, self.work2)

        # Hessian products with respect to t
        #    D2_tt F(t, X, Y)[Ht] = Ht / z^2
        #    D2_tX F(t, X, Y)[Hx] = -(D_X Phi(X, Y) [Hx]) / z^2
        #    D2_tY F(t, X, Y)[Hy] = -(D_Y Phi(X, Y) [Hy]) / z^2
        outt  = self.At - (self.Ax.view(dtype=np.float64).reshape((p, 1, -1)) @ self.DPhiX.view(dtype=np.float64).reshape((-1, 1))).ravel()
        outt -= (self.Ay.view(dtype=np.float64).reshape((p, 1, -1)) @ self.DPhiY.view(dtype=np.float64).reshape((-1, 1))).ravel()
        outt *= self.zi2

        lhs[:, 0] = outt

        # Hessian products with respect to X
        #    D2_Xt F(t, X, Y)[Ht] = -Ht (D_X Phi(X, Y) [Hx]) / z^2
        #    D2_XX F(t, X, Y)[Hx] = (D_X Phi(X, Y) [Hx]) D_X Phi(X, Y) / z^2 + (D2_XX Phi(X, Y) [Hx]) / z + X^-1 Hx X^-1
        #    D2_XY F(t, X, Y)[Hy] = (D_Y Phi(X, Y) [Hy]) D_X Phi(X, Y) / z^2 + (D2_XY Phi(X, Y) [Hy]) / z
        np.subtract(self.D2PhiXXH, self.D2PhiXYH, out=self.work1)
        self.work1 *= self.zi
        congr(self.work3, self.inv_X, self.Ax, self.work2)
        self.work1 += self.work3
        self.work1 -= np.outer(outt, self.DPhiX).reshape((p, self.n, self.n))

        lhs[:, self.idx_X] = self.work1.reshape((p, -1)).view(dtype=np.float64)

        # Hessian products with respect to Y
        #    D2_Yt F(t, X, Y)[Ht] = -Ht (D_X Phi(X, Y) [Hx]) / z^2
        #    D2_YX F(t, X, Y)[Hx] = (D_X Phi(X, Y) [Hx]) D_Y Phi(X, Y) / z^2 + (D2_YX Phi(X, Y) [Hx]) / z
        #    D2_YY F(t, X, Y)[Hy] = (D_Y Phi(X, Y) [Hy]) D_Y Phi(X, Y) / z^2 + (D2_YY Phi(X, Y) [Hy]) / z + Y^-1 Hy Y^-1
        np.add(self.D2PhiYYH, self.D2PhiYXH, out=self.work1)
        self.work1 *= -self.zi
        congr(self.work3, self.inv_Y, self.Ay, self.work2)
        self.work1 += self.work3
        self.work1 -= np.outer(outt, self.DPhiY).reshape((p, self.n, self.n))

        lhs[:, self.idx_Y] = self.work1.reshape((p, -1)).view(dtype=np.float64)

        # Multiply A (H A')
        return lhs @ A.T

    def update_invhessprod_aux_aux(self):
        assert not self.invhess_aux_aux_updated

        self.triu_1_indices = np.array([j + i*self.n for j in range(self.n) for i in range(j)])

        if self.hermitian:
            self.diag_indices = np.append(0, np.cumsum([i for i in range(3, 2*self.n+1, 2)]))
            self.triu_indices = np.empty(self.n*self.n, dtype=int)
            self.scale        = np.empty(self.n*self.n)
            k = 0
            for j in range(self.n):
                for i in range(j):
                    self.triu_indices[k]     = 2 * (j + i*self.n)
                    self.triu_indices[k + 1] = 2 * (j + i*self.n) + 1
                    self.scale[k:k+2]        = np.sqrt(2.)
                    k += 2
                self.triu_indices[k] = 2 * (j + j*self.n)
                self.scale[k]        = 1.
                k += 1
        else:
            self.diag_indices = np.append(0, np.cumsum([i for i in range(2, self.n+1, 1)]))
            self.triu_indices = np.array([j + i*self.n for j in range(self.n) for i in range(j + 1)])
            self.scale = np.array([1 if i==j else np.sqrt(2.) for j in range(self.n) for i in range(j + 1)])
        self.invhess_aux_aux_updated = True

        self.work5 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work6 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work7 = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work8 = np.empty((self.n, self.n, self.n), dtype=self.dtype)
        self.work9 = np.empty((self.n, 1, self.n), dtype=self.dtype)
    
    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated
        if not self.invhess_aux_aux_updated:
            self.update_invhessprod_aux_aux()

        self.z2           = self.z * self.z
        self.UyUx         = self.Uy.conj().T @ self.Ux
        self.D1x_inv      = np.reciprocal(np.outer(self.Dx, self.Dx))
        self.D1x_comb_inv = np.reciprocal(self.zi * self.D1x_log + self.D1x_inv)

        # Hessian is a block matrix
        #         [ 1/z log^[1](Dx) + Dx^-1 kron Dx^-1  -1/z (Ux'Uy kron Ux'Uy) log^[1](Dy) ]
        #     Wxy [-1/z log^[1](Dy) (Uy'Ux kron Uy'Ux)      -1/z Sy + Dy^-1 kron Dy^-1      ] Wxy'
        # where 
        #           [ Ux kron Ux             ]
        #     Wxy = [             Uy kron Uy ]
        # and
        #     (Sy)_ij,kl = delta_kl (Uy' X Uy)_ij log^[2]_ijl(Dy) + delta_ij (Uy' X Uy)_kl log^[2]_jkl(Dy)
        # To solve linear systems with this matrix, we simplify it by doing block elimination,
        # in which case we just need the Schur complement matrix, which we compute below.
        
        # Get (1/z Sy + Dy^-1 kron Dy^-1) matrix
        hess_schur = -mgrad.get_S_matrix(self.D2y_log * (self.zi * self.UyXUy + np.eye(self.n)), np.sqrt(2.0), hermitian=self.hermitian)

        # Get [1/z^2 log^[1](Dy) (Uy'Ux kron Uy'Ux) [(1/z log + inv)^[1](Dx)]^-1 (Ux'Uy kron Ux'Uy) log^[1](Dy)] matrix
        # Begin with (Ux'Uy kron Ux'Uy) log^[1](Dy)
        np.multiply(self.UyUx.reshape(self.n, 1, self.n).T, self.D1y_log.flat[::self.n+1], out=self.work9.T)
        np.matmul(self.UyUx.conj().reshape(self.n, self.n, 1), self.work9, out=self.work8)
        self.work7[self.diag_indices] = self.work8
        t = 0
        for j in range(self.n):
            np.multiply(self.UyUx[:j].reshape((j, 1, self.n)).T, np.sqrt(0.5) * self.D1y_log[j, :j], out=self.work9[:j].T)
            np.matmul(self.UyUx[[j]].conj().T, self.work9[:j], out=self.work8[:j])

            if self.hermitian:
                np.add(self.work8[:j], self.work8[:j].conj().transpose(0, 2, 1), out=self.work7[t : t+2*j : 2])
                np.subtract(self.work8[:j], self.work8[:j].conj().transpose(0, 2, 1), out=self.work7[t+1 : t+2*j+1 : 2])
                self.work7[t+1 : t+2*j+1 : 2] *= -1j
                t += 2*j + 1
            else:
                np.add(self.work8[:j], self.work8[:j].transpose(0, 2, 1), out=self.work7[t : t+j])
                t += j + 1
        # Apply [(1/z log + inv)^[1](Dx)]^-1
        self.work7 *= self.D1x_comb_inv
        # Apply (Uy'Ux kron Uy'Ux)
        congr(self.work5, self.UyUx, self.work7, work=self.work6)
        # Apply (1/z^2 log^[1](Dy))
        self.work5 *= self.D1y_log
        work  = self.work5.view(dtype=np.float64).reshape((self.vn, -1))[:, self.triu_indices]
        work *= (self.zi2 * self.scale)

        # Subtract to obtain Schur complement then Cholesky factor
        hess_schur -= work
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
    
    def invhess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)            

        p = self.Ax.shape[0]
        out = np.empty((sum(self.dim), p))    

        temp_vec = np.empty((self.vn, p))   
        Wx_list = np.empty((p, self.n, self.n), dtype=self.dtype)

        for k in range(p):
            Ht = self.At[k]
            Hx = self.Ax[k]
            Hy = self.Ay[k]

            Wx = Hx + Ht * self.DPhiX
            Wy = Hy + Ht * self.DPhiY
            Wx_list[k, :, :] = Wx

            temp = self.Ux.conj().T @ Wx @ self.Ux
            temp = self.UyUx @ (self.D1x_comb_inv * temp) @ self.UyUx.conj().T
            temp = -self.Uy @ (self.zi * self.D1y_log * temp) @ self.Uy.conj().T
            temp = self.Uy.conj().T @ (Wy - temp) @ self.Uy
            temp_vec[:, [k]] = sym.mat_to_vec(temp, hermitian=self.hermitian)

        temp_vec = lin.fact_solve(self.hess_schur_fact, temp_vec)

        for k in range(p):
            Ht = self.At[k]
            Wx = Wx_list[k, :, :]

            temp = sym.vec_to_mat(temp_vec[:, [k]], hermitian=self.hermitian)
            temp = self.Uy @ temp @ self.Uy.conj().T
            outY = temp

            temp = self.Uy.conj().T @ temp @ self.Uy
            temp = -self.Uy @ (self.zi * self.D1y_log * temp) @ self.Uy.conj().T
            temp = Wx - temp
            temp = self.Ux.conj().T @ temp @ self.Ux
            temp = self.Ux @ (self.D1x_comb_inv * temp) @ self.Ux.conj().T
            outX = temp

            outt = self.z2 * Ht + lin.inp(self.DPhiX, outX) + lin.inp(self.DPhiY, outY)

            out[0, k] = outt
            out[self.idx_X, [k]] = outX.view(dtype=np.float64).reshape(-1, 1)
            out[self.idx_Y, [k]] = outY.view(dtype=np.float64).reshape(-1, 1)

        return out.T @ A.T


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
        temp = [np.zeros((1, 1)), np.zeros((self.n, self.n), dtype=self.dtype), np.zeros((self.n, self.n), dtype=self.dtype)]
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

def congr(out, A, X, work):
    n = A.shape[0]
    np.matmul(A, X, out=work)
    np.matmul(work.reshape((-1, n)), A.conj().T, out=out.reshape((-1, n)))
    return out