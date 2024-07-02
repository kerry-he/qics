import numpy as np
from utils import linear    as lin
from utils import mtxgrad   as mgrad
from utils import symmetric as sym

class Cone():
    def __init__(self, n, hermitian=False):
        # Dimension properties
        self.n  = n               # Side dimension of system
        self.nu = 1 + 2 * self.n  # Barrier parameter

        self.hermitian = hermitian                      # Hermitian or symmetric vector space
        self.vn = n*n if hermitian else n*(n+1)//2      # Compact dimension of system

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
    
    def set_point(self, point, dual, a=True):
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
        
        rt2_Dx      = np.sqrt(self.Dx)
        rt4_X       = self.Ux * np.sqrt(rt2_Dx)
        irt4_X      = self.Ux / np.sqrt(rt2_Dx)
        self.rt2_X  = rt4_X @ rt4_X.conj().T
        self.irt2_X = irt4_X @ irt4_X.conj().T

        rt2_Dy      = np.sqrt(self.Dy)
        rt4_Y       = self.Uy * np.sqrt(rt2_Dy)
        irt4_Y      = self.Uy / np.sqrt(rt2_Dy)
        self.rt2_Y  = rt4_Y @ rt4_Y.conj().T
        self.irt2_Y = irt4_Y @ irt4_Y.conj().T        

        self.XYX = self.irt2_X @ self.Y @ self.irt2_X
        self.YXY = self.irt2_Y @ self.X @ self.irt2_Y

        self.Dxyx, self.Uxyx = np.linalg.eigh(self.XYX)
        self.Dyxy, self.Uyxy = np.linalg.eigh(self.YXY)

        if any(self.Dxyx <= 0) or any(self.Dyxy <= 0):
            self.feas = False
            return self.feas        
        
        self.log_Dxyx  = np.log(self.Dxyx)
        self.log_Dyxy  = np.log(self.Dyxy)
        self.entr_Dxyx = self.Dxyx * self.log_Dxyx
        self.entr_Dyxy = self.Dyxy * self.log_Dyxy

        self.log_XYX  = (self.Uxyx * self.log_Dxyx) @ self.Uxyx.conj().T
        self.log_XYX  = (self.log_XYX + self.log_XYX.conj().T) * 0.5

        self.z = self.t[0, 0] - lin.inp(self.X, self.log_XYX)

        self.feas = (self.z > 0)
        return self.feas

    def get_val(self):
        assert self.feas_updated

        return -np.log(self.z) - np.sum(np.log(self.Dx)) - np.sum(np.log(self.Dy))

    def update_grad(self):
        assert self.feas_updated
        assert not self.grad_updated
        
        self.inv_Dx = np.reciprocal(self.Dx)
        self.inv_Dy = np.reciprocal(self.Dy)

        inv_X_rt2 = self.Ux * np.sqrt(self.inv_Dx)
        self.inv_X = inv_X_rt2 @ inv_X_rt2.conj().T
        inv_Y_rt2 = self.Uy * np.sqrt(self.inv_Dy)
        self.inv_Y = inv_Y_rt2 @ inv_Y_rt2.conj().T        

        self.D1yxy_entr = mgrad.D1_entr(self.Dyxy, self.log_Dyxy, self.entr_Dyxy)
        self.D1xyx_log  = mgrad.D1_log(self.Dxyx, self.log_Dxyx)

        self.UyxyYUyxy = self.Uyxy.conj().T @ self.Y @ self.Uyxy
        self.UxyxXUxyx = self.Uxyx.conj().T @ self.X @ self.Uxyx

        self.zi    =  np.reciprocal(self.z)
        self.DPhiX = -self.irt2_Y @ self.Uyxy @ (self.D1yxy_entr * self.UyxyYUyxy) @ self.Uyxy.conj().T @ self.irt2_Y
        self.DPhiY =  self.irt2_X @ self.Uxyx @ (self.D1xyx_log  * self.UxyxXUxyx) @ self.Uxyx.conj().T @ self.irt2_X

        self.grad = [
           -self.zi,
            self.zi * self.DPhiX - self.inv_X,
            self.zi * self.DPhiY - self.inv_Y
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
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()            

        # Computes Hessian product of the QRE barrier with a single vector (Ht, Hx, Hy)
        # See hess_congr() for additional comments

        (Ht, Hx, Hy) = H

        UyxyYHxYUyxy = self.Uyxy.conj().T @ self.irt2_Y @ Hx @ self.irt2_Y @ self.Uyxy
        UxyxXHyXUxyx = self.Uxyx.conj().T @ self.irt2_X @ Hy @ self.irt2_X @ self.Uxyx
        UxyxXHxXUxyx = self.Uxyx.conj().T @ self.irt2_X @ Hx @ self.irt2_X @ self.Uxyx

        # Hessian product of relative entropy
        D2PhiXXH = -mgrad.scnd_frechet(self.D2yxy_entr_UYU, UyxyYHxYUyxy, U=self.irt2_Y @ self.Uyxy)

        D2PhiXYH  = self.irt2_X @ self.Uxyx @ (self.D1xyx_log * UxyxXHyXUxyx) @ self.Uxyx.conj().T @ self.rt2_X
        D2PhiXYH += D2PhiXYH.conj().T
        D2PhiXYH -= mgrad.scnd_frechet(self.D2xyx_entr_UXU, UxyxXHyXUxyx, U=self.irt2_X @ self.Uxyx)

        work      = self.Uxyx.conj().T @ self.rt2_X @ Hx @ self.irt2_X @ self.Uxyx
        work     += work.conj().T
        D2PhiYXH  = self.irt2_X @ self.Uxyx @ (self.D1xyx_log * work) @ self.Uxyx.conj().T @ self.irt2_X
        D2PhiYXH -= mgrad.scnd_frechet(self.D2xyx_entr_UXU, UxyxXHxXUxyx, U=self.irt2_X @ self.Uxyx)

        D2PhiYYH  = mgrad.scnd_frechet(self.D2xyx_log_UXU, UxyxXHyXUxyx, U=self.irt2_X @ self.Uxyx)
        
        # Hessian product of barrier function
        out[0][:] = (Ht - lin.inp(self.DPhiX, Hx) - lin.inp(self.DPhiY, Hy)) * self.zi2

        out_X     = -out[0] * self.DPhiX
        out_X    +=  self.zi * (D2PhiXYH + D2PhiXXH)
        out_X    +=  self.inv_X @ Hx @ self.inv_X
        out_X     = (out_X + out_X.conj().T) * 0.5
        out[1][:] = out_X

        out_Y     = -out[0] * self.DPhiY
        out_Y    +=  self.zi * (D2PhiYXH + D2PhiYYH)
        out_Y    +=  self.inv_Y @ Hy @ self.inv_Y
        out_Y     = (out_Y + out_Y.conj().T) * 0.5
        out[2][:] = out_Y

        return out

    def hess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)

        p = A.shape[0]
        lhs = np.zeros((p, sum(self.dim)))

        # Precompute Hessian products for quantum relative entropy
        # D2_XX Phi(X, Y) [Hx] =  Ux [log^[1](Dx) .* (Ux' Hx Ux)] Ux'
        # D2_YX Phi(X, Y) [Hx] = -Uy [log^[1](Dy) .* (Uy' Hx Uy)] Uy'
        # D2_XY Phi(X, Y) [Hy] = -Uy [log^[1](Dy) .* (Uy' Hy Uy)] Uy'
        # D2_YY Phi(X, Y) [Hy] = -Uy [SUM_k log_k^[2](Dy) .* ([Uy' X Uy]_k [Uk' Hy Uy]_k' + [Uy' Hy Uy]_k [Uk' X Uy]_k') ] Uy'
        lin.congr(self.work1, self.Ux.conj().T, self.Ax, self.work2)
        self.work1 *= self.D1x_comb
        lin.congr(self.D2PhiXXH, self.Ux, self.work1, self.work2)

        lin.congr(self.work1, self.Uy.conj().T, self.Ax, self.work2)
        self.work1 *= -self.zi * self.D1y_log
        lin.congr(self.D2PhiYXH, self.Uy, self.work1, self.work2)

        lin.congr(self.work1, self.Uy.conj().T, self.Ay, self.work2)
        mgrad.scnd_frechet_multi(self.D2PhiYYH, self.D2y_comb, self.work1, U=self.Uy, work1=self.work2, work2=self.work3, work3=self.work5)

        self.work1 *= self.D1y_log * self.zi
        lin.congr(self.D2PhiXYH, self.Uy, self.work1, self.work2)

        # ====================================================================
        # Hessian products with respect to t
        # ====================================================================
        # D2_tt F(t, X, Y)[Ht] = Ht / z^2
        # D2_tX F(t, X, Y)[Hx] = -(D_X Phi(X, Y) [Hx]) / z^2
        # D2_tY F(t, X, Y)[Hy] = -(D_Y Phi(X, Y) [Hy]) / z^2
        outt  = self.At - (self.Ax.view(dtype=np.float64).reshape((p, 1, -1)) @ self.DPhiX.view(dtype=np.float64).reshape((-1, 1))).ravel()
        outt -= (self.Ay.view(dtype=np.float64).reshape((p, 1, -1)) @ self.DPhiY.view(dtype=np.float64).reshape((-1, 1))).ravel()
        outt *= self.zi2

        lhs[:, 0] = outt

        # ====================================================================
        # Hessian products with respect to X
        # ====================================================================
        # D2_Xt F(t, X, Y)[Ht] = -Ht (D_X Phi(X, Y)) / z^2
        # D2_XX F(t, X, Y)[Hx] = (D_X Phi(X, Y) [Hx]) D_X Phi(X, Y) / z^2 + (D2_XX Phi(X, Y) [Hx]) / z + X^-1 Hx X^-1
        # D2_XY F(t, X, Y)[Hy] = (D_Y Phi(X, Y) [Hy]) D_X Phi(X, Y) / z^2 + (D2_XY Phi(X, Y) [Hy]) / z
        np.subtract(self.D2PhiXXH, self.D2PhiXYH, out=self.work1)
        np.outer(outt, self.DPhiX, out=self.work2.reshape((p, -1)))
        self.work1 -= self.work2

        lhs[:, self.idx_X] = self.work1.reshape((p, -1)).view(dtype=np.float64)

        # ====================================================================
        # Hessian products with respect to Y
        # ====================================================================
        # D2_Yt F(t, X, Y)[Ht] = -Ht (D_X Phi(X, Y)) / z^2
        # D2_YX F(t, X, Y)[Hx] = (D_X Phi(X, Y) [Hx]) D_Y Phi(X, Y) / z^2 + (D2_YX Phi(X, Y) [Hx]) / z
        # D2_YY F(t, X, Y)[Hy] = (D_Y Phi(X, Y) [Hy]) D_Y Phi(X, Y) / z^2 + (D2_YY Phi(X, Y) [Hy]) / z + Y^-1 Hy Y^-1
        np.add(self.D2PhiYYH, self.D2PhiYXH, out=self.work1)
        np.outer(outt, self.DPhiY, out=self.work2.reshape((p, -1)))
        self.work1 -= self.work2

        lhs[:, self.idx_Y] = self.work1.reshape((p, -1)).view(dtype=np.float64)

        # Multiply A (H A')
        return lhs @ A.T        
    
    def invhess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        # Computes inverse Hessian product of the QRE barrier with a single vector (Ht, Hx, Hy)
        # See invhess_congr() for additional comments

        (Ht, Hx, Hy) = H

        vec = np.vstack((Ht, sym.mat_to_vec(Hx), sym.mat_to_vec(Hy)))
        sol = lin.fact_solve(self.hess_fact, vec)

        out[0][:] = sol[0]
        out[1][:] = sym.vec_to_mat(sol[1:1+self.vn])
        out[2][:] = sym.vec_to_mat(sol[1+self.vn:])

        return out
    
    def invhess_congr(self, A):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()
        if not self.congr_aux_updated:
            self.congr_aux(A)            

        # The inverse Hessian product applied on (Ht, Hx, Hy) for the QRE barrier is 
        #     (X, Y) =  M \ (Wx, Wy)
        #         t  =  z^2 Ht + <DPhi(X, Y), (X, Y)>
        # where (Wx, Wy) = [(Hx, Hy) + Ht DPhi(X, Y)]
        #     M = Vxy [ 1/z log^[1](Dx) + Dx^-1 kron Dx^-1  -1/z (Ux'Uy kron Ux'Uy) log^[1](Dy) ]
        #             [-1/z log^[1](Dy) (Uy'Ux kron Uy'Ux)      -1/z Sy + Dy^-1 kron Dy^-1      ] Vxy'
        # and 
        #     Vxy = [ Ux kron Ux             ]
        #           [             Uy kron Uy ]
        #
        # To solve linear systems with M, we simplify it by doing block elimination, in which case we get
        #     Uy' Y Uy = S \ ({Uy' Wy Uy} - [1/z log^[1](Dy) (Uy'Ux kron Uy'Ux) (1/z log^[1](Dx) + Dx^-1 kron Dx^-1)^-1 {Ux' Wx Ux}])
        #     Ux' X Ux = -(1/z log^[1](Dx) + Dx^-1 kron Dx^-1)^-1 [{Ux' Wx Ux} + 1/z (Ux'Uy kron Ux'Uy) log^[1](Dy) Y]
        # where S is the Schur complement matrix of M.

        p = self.Ax.shape[0]
        lhs = np.empty((p, sum(self.dim)))
        
        # ====================================================================
        # Inverse Hessian products with respect to Y
        # ====================================================================
        # Compute ({Uy' Wy Uy} - [1/z log^[1](Dy) (Uy'Ux kron Uy'Ux) (1/z log^[1](Dx) + Dx^-1 kron Dx^-1)^-1 {Ux' Wx Ux}])
        # Compute Ux' Wx Ux
        np.outer(self.At, self.DPhiX, out=self.work2.reshape((p, -1)))
        np.add(self.Ax, self.work2, out=self.work0)
        lin.congr(self.work2, self.Ux.conj().T, self.work0, self.work3)
        # Apply (1/z log^[1](Dx) + Dx^-1 kron Dx^-1)^-1
        self.work2 *= self.D1x_comb_inv
        # Apply (Uy'Ux kron Uy'Ux)
        lin.congr(self.work1, self.UyUx, self.work2, self.work3)
        # Apply -1/z log^[1](Dy)
        self.work1 *= (-self.zi * self.D1y_log)
        # Compute Uy' Wy Uy and subtract previous expression
        np.outer(self.At, self.DPhiY, out=self.work2.reshape((p, -1)))
        np.add(self.Ay, self.work2, out=self.work3)
        lin.congr(self.work2, self.Uy.conj().T, self.work3, self.work4)
        self.work2 -= self.work1

        # Solve the linear system S \ ( ... ) to obtain Uy' Y Uy
        # Convert matrices to truncated real vectors
        work  = self.work2.view(dtype=np.float64).reshape((p, -1))[:, self.triu_indices]
        work *= self.scale
        # Solve system
        work = lin.fact_solve(self.hess_schur_fact, work.T)
        # Expand truncated real vectors back into matrices
        self.work1.fill(0.)
        work[self.diag_indices, :] *= 0.5
        work /= self.scale.reshape((-1, 1))
        self.work1.view(dtype=np.float64).reshape((p, -1))[:, self.triu_indices] = work.T
        self.work1 += self.work1.conj().transpose((0, 2, 1))

        # Recover Y
        lin.congr(self.work4, self.Uy, self.work1, self.work2)
        lhs[:, self.idx_Y] = self.work4.reshape((p, -1)).view(dtype=np.float64)

        # ====================================================================
        # Inverse Hessian products with respect to X
        # ====================================================================
        # Apply -1/z log^[1](Dy)
        self.work1 *= (-self.zi * self.D1y_log)
        # Apply (Uy kron Uy)
        lin.congr(self.work2, self.Uy, self.work1, self.work3)
        # Subtract Wx from previous expression
        self.work0 -= self.work2
        # Apply (Ux kron Ux)
        lin.congr(self.work1, self.Ux.conj().T, self.work0, self.work3)
        # Apply (1/z log^[1](Dx) + Dx^-1 kron Dx^-1)^-1 to obtian Ux' X Ux
        self.work1 *= self.D1x_comb_inv

        # Recover X
        lin.congr(self.work2, self.Ux, self.work1, self.work3)
        lhs[:, self.idx_X] = self.work2.reshape((p, -1)).view(dtype=np.float64)

        # ====================================================================
        # Inverse Hessian products with respect to t
        # ====================================================================
        outt  = self.z2 * self.At 
        outt += (self.work2.view(dtype=np.float64).reshape((p, 1, -1)) @ self.DPhiX.view(dtype=np.float64).reshape((-1, 1))).ravel()
        outt += (self.work4.view(dtype=np.float64).reshape((p, 1, -1)) @ self.DPhiY.view(dtype=np.float64).reshape((-1, 1))).ravel()
        lhs[:, 0] = outt

        # Multiply A (H A')
        return lhs @ A.T

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
        dder3_X  = (dder3_X + dder3_X.conj().T) * 0.5

        dder3_Y  = -dder3_t * self.DPhiY
        dder3_Y -=  2 * self.zi2 * chi * (D2PhiYXH + D2PhiYYH)
        dder3_Y +=  self.zi * (D3PhiYYX + D3PhiYXY + D3PhiYYY)
        dder3_Y -=  2 * self.inv_Y @ Hy @ self.inv_Y @ Hy @ self.inv_Y
        dder3_Y  = (dder3_Y + dder3_Y.conj().T) * 0.5

        out[0][:] += dder3_t * a
        out[1][:] += dder3_X * a
        out[2][:] += dder3_Y * a

        return out
    
    def prox(self):
        assert self.feas_updated
        if not self.grad_updated:
            self.update_grad()
        psi = [
            self.t_d + self.grad[0],
            self.X_d + self.grad[1],
            self.Y_d + self.grad[2]
        ]
        temp = [np.zeros((1, 1)), np.zeros((self.n, self.n), dtype=self.dtype), np.zeros((self.n, self.n), dtype=self.dtype)]
        self.invhess_prod_ip(temp, psi)
        return lin.inp(temp[0], psi[0]) + lin.inp(temp[1], psi[1]) + lin.inp(temp[2], psi[2])
    
    # ========================================================================
    # Auxilliary functions
    # ========================================================================
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

        self.work0 = np.empty_like(self.Ax, dtype=self.dtype)
        self.work1 = np.empty_like(self.Ax, dtype=self.dtype)
        self.work2 = np.empty_like(self.Ax, dtype=self.dtype)
        self.work3 = np.empty_like(self.Ax, dtype=self.dtype)
        self.work4 = np.empty_like(self.Ax, dtype=self.dtype)
        self.work5 = np.empty((self.Ax.shape[::-1]), dtype=self.dtype)

        self.D2PhiXXH = np.empty_like(self.Ax, dtype=self.dtype)
        self.D2PhiYXH = np.empty_like(self.Ax, dtype=self.dtype)
        self.D2PhiXYH = np.empty_like(self.Ay, dtype=self.dtype)
        self.D2PhiYYH = np.empty_like(self.Ay, dtype=self.dtype)

        self.congr_aux_updated = True

    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        self.D1xyx_entr = mgrad.D1_entr(self.Dxyx, self.log_Dxyx, self.entr_Dxyx)

        self.D2xyx_entr = mgrad.D2_entr(self.Dxyx, self.D1xyx_entr)
        self.D2yxy_entr = mgrad.D2_entr(self.Dyxy, self.D1yxy_entr)
        self.D2xyx_log  = mgrad.D2_log(self.Dxyx, self.D1xyx_log)

        self.D2xyx_entr_UXU = self.D2xyx_entr * self.UxyxXUxyx
        self.D2yxy_entr_UYU = self.D2yxy_entr * self.UyxyYUyxy
        self.D2xyx_log_UXU  = self.D2xyx_log  * self.UxyxXUxyx

        # Preparing other required variables
        self.zi2 = self.zi * self.zi

        self.hess_aux_updated = True

        return

    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated
        if not self.invhess_aux_aux_updated:
            self.update_invhessprod_aux_aux()

        # Precompute and factorize the Schur complement matrix
        #     S = (-1/z Sy + Dy^-1 kron Dy^-1)
        #         - [1/z^2 log^[1](Dy) (Uy'Ux kron Uy'Ux) [(1/z log + inv)^[1](Dx)]^-1 (Ux'Uy kron Ux'Uy) log^[1](Dy)]
        # where
        #     (Sy)_ij,kl = delta_kl (Uy' X Uy)_ij log^[2]_ijl(Dy) + delta_ij (Uy' X Uy)_kl log^[2]_jkl(Dy)        
        # which we will need to solve linear systems with the Hessian of our barrier function

        Hxx = np.zeros((self.vn, self.vn))
        Hxy = np.zeros((self.vn, self.vn))
        Hyy = np.zeros((self.vn, self.vn))

        # Make Hxx block
        k = 0
        for j in range(self.n):
            for i in range(j + 1):
                Hx = np.zeros((self.n, self.n))
                if i == j:
                    Hx[i, j] = 1.
                else:
                    Hx[i, j] = np.sqrt(0.5)
                    Hx[j, i] = np.sqrt(0.5)
                
                UyxyYHxYUyxy = self.Uyxy.conj().T @ self.irt2_Y @ Hx @ self.irt2_Y @ self.Uyxy

                D2PhiXXH  = -self.zi * mgrad.scnd_frechet(self.D2yxy_entr_UYU, UyxyYHxYUyxy, U=self.irt2_Y @ self.Uyxy)
                D2PhiXXH +=  self.zi2 * self.DPhiX * lin.inp(self.DPhiX, Hx)
                D2PhiXXH +=  self.inv_X @ Hx @ self.inv_X

                Hxx[:, [k]] = sym.mat_to_vec(D2PhiXXH, hermitian=self.hermitian)

                k += 1

        # Make Hxy block
        k = 0
        for j in range(self.n):
            for i in range(j + 1):
                Hy = np.zeros((self.n, self.n))
                if i == j:
                    Hy[i, j] = 1.
                else:
                    Hy[i, j] = np.sqrt(0.5)
                    Hy[j, i] = np.sqrt(0.5)
                
                UxyxXHyXUxyx = self.Uxyx.conj().T @ self.irt2_X @ Hy @ self.irt2_X @ self.Uxyx

                D2PhiXYH  = self.zi * self.irt2_X @ self.Uxyx @ (self.D1xyx_log * UxyxXHyXUxyx) @ self.Uxyx.conj().T @ self.rt2_X
                D2PhiXYH += D2PhiXYH.conj().T
                D2PhiXYH -= self.zi * mgrad.scnd_frechet(self.D2xyx_entr_UXU, UxyxXHyXUxyx, U=self.irt2_X @ self.Uxyx)
                D2PhiXYH += self.zi2 * self.DPhiX * lin.inp(self.DPhiY, Hy)

                Hxy[:, [k]] = sym.mat_to_vec(D2PhiXYH, hermitian=self.hermitian)

                k += 1           

        # Make Hyy block
        k = 0
        for j in range(self.n):
            for i in range(j + 1):
                Hy = np.zeros((self.n, self.n))
                if i == j:
                    Hy[i, j] = 1.
                else:
                    Hy[i, j] = np.sqrt(0.5)
                    Hy[j, i] = np.sqrt(0.5)
                
                UxyxXHyXUxyx = self.Uxyx.conj().T @ self.irt2_X @ Hy @ self.irt2_X @ self.Uxyx

                D2PhiYYH  = self.zi * mgrad.scnd_frechet(self.D2xyx_log_UXU, UxyxXHyXUxyx, U=self.irt2_X @ self.Uxyx)
                D2PhiYYH += self.zi2 * self.DPhiY * lin.inp(self.DPhiY, Hy)
                D2PhiYYH += self.inv_Y @ Hy @ self.inv_Y

                Hyy[:, [k]] = sym.mat_to_vec(D2PhiYYH, hermitian=self.hermitian)

                k += 1

        self.hess = np.block([
            [self.zi2,  -self.zi2 * sym.mat_to_vec(self.DPhiX).T, -self.zi2 * sym.mat_to_vec(self.DPhiY).T],
            [-self.zi2 * sym.mat_to_vec(self.DPhiX), Hxx,   Hxy], 
            [-self.zi2 * sym.mat_to_vec(self.DPhiY), Hxy.T, Hyy]
        ])

        # Subtract to obtain Schur complement then Cholesky factor
        self.hess_fact = lin.fact(self.hess.copy())
        self.invhess_aux_updated = True

        return

    def update_invhessprod_aux_aux(self):
        assert not self.invhess_aux_aux_updated

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

        # self.work6  = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        # self.work7  = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        # self.work8  = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        # self.work9  = np.empty((self.n, self.n, self.n), dtype=self.dtype)
        # self.work10 = np.empty((self.n, 1, self.n), dtype=self.dtype)

        self.invhess_aux_aux_updated = True

    def update_dder3_aux(self):
        assert not self.dder3_aux_updated
        assert self.hess_aux_updated

        self.zi3 = self.zi2 * self.zi
        self.D2x_log = mgrad.D2_log(self.Dx, self.D1x_log)

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