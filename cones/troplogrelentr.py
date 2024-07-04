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

        self.z = self.t[0, 0] + lin.inp(self.X, self.log_XYX)

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

        self.irt2Y_Uyxy = self.irt2_Y @ self.Uyxy
        self.irt2X_Uxyx = self.irt2_X @ self.Uxyx

        self.zi    =  np.reciprocal(self.z)
        self.DPhiX =  self.irt2_Y @ self.Uyxy @ (self.D1yxy_entr * self.UyxyYUyxy) @ self.Uyxy.conj().T @ self.irt2_Y
        self.DPhiX = (self.DPhiX + self.DPhiX.conj().T) * 0.5
        self.DPhiY = -self.irt2_X @ self.Uxyx @ (self.D1xyx_log  * self.UxyxXUxyx) @ self.Uxyx.conj().T @ self.irt2_X
        self.DPhiY = (self.DPhiY + self.DPhiY.conj().T) * 0.5

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
        D2PhiXXH =  mgrad.scnd_frechet(self.D2yxy_entr_UYU, UyxyYHxYUyxy, U=self.irt2_Y @ self.Uyxy)

        D2PhiXYH  = -self.irt2_X @ self.Uxyx @ (self.D1xyx_log * UxyxXHyXUxyx) @ self.Uxyx.conj().T @ self.rt2_X
        D2PhiXYH += D2PhiXYH.conj().T
        D2PhiXYH += mgrad.scnd_frechet(self.D2xyx_entr_UXU, UxyxXHyXUxyx, U=self.irt2_X @ self.Uxyx)

        work      = self.Uxyx.conj().T @ self.rt2_X @ Hx @ self.irt2_X @ self.Uxyx
        work     += work.conj().T
        D2PhiYXH  = -self.irt2_X @ self.Uxyx @ (self.D1xyx_log * work) @ self.Uxyx.conj().T @ self.irt2_X
        D2PhiYXH += mgrad.scnd_frechet(self.D2xyx_entr_UXU, UxyxXHxXUxyx, U=self.irt2_X @ self.Uxyx)

        D2PhiYYH  = -mgrad.scnd_frechet(self.D2xyx_log_UXU, UxyxXHyXUxyx, U=self.irt2_X @ self.Uxyx)
        
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
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()
    
        return self.A_compact @ self.hess @ self.A_compact.T
    
    def invhess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        # Computes inverse Hessian product of the QRE barrier with a single vector (Ht, Hx, Hy)
        # See invhess_congr() for additional comments

        (Ht, Hx, Hy) = H

        vec = np.vstack((Ht, sym.mat_to_vec(Hx, hermitian=self.hermitian), sym.mat_to_vec(Hy, hermitian=self.hermitian)))
        sol = lin.fact_solve(self.hess_fact, vec)

        out[0][:] = sol[0]
        out[1][:] = sym.vec_to_mat(sol[1:1+self.vn], hermitian=self.hermitian)
        out[2][:] = sym.vec_to_mat(sol[1+self.vn:], hermitian=self.hermitian)

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

        lhs = lin.fact_solve(self.hess_fact, self.A_compact.T)

        # Multiply A (H A')
        return self.A_compact @ lhs

    def third_dir_deriv_axpy(self, out, dir, a=True):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.dder3_aux_updated:
            self.update_dder3_aux()

        (Ht, Hx, Hy) = dir

        chi = Ht[0, 0] - lin.inp(self.DPhiX, Hx) - lin.inp(self.DPhiY, Hy)
        chi2 = chi * chi

        UyxyYHxYUyxy = self.Uyxy.conj().T @ self.irt2_Y @ Hx @ self.irt2_Y @ self.Uyxy
        UxyxXHyXUxyx = self.Uxyx.conj().T @ self.irt2_X @ Hy @ self.irt2_X @ self.Uxyx
        UxyxXHxXUxyx = self.Uxyx.conj().T @ self.irt2_X @ Hx @ self.irt2_X @ self.Uxyx
        UyxyYHyYUyxy = self.Uyxy.conj().T @ self.irt2_Y @ Hy @ self.irt2_Y @ self.Uyxy

        # Hessian product of relative entropy
        D2PhiXXH =  mgrad.scnd_frechet(self.D2yxy_entr_UYU, UyxyYHxYUyxy, U=self.irt2_Y @ self.Uyxy)

        D2PhiXYH  = -self.irt2_X @ self.Uxyx @ (self.D1xyx_log * UxyxXHyXUxyx) @ self.Uxyx.conj().T @ self.rt2_X
        D2PhiXYH += D2PhiXYH.conj().T
        D2PhiXYH += mgrad.scnd_frechet(self.D2xyx_entr_UXU, UxyxXHyXUxyx, U=self.irt2_X @ self.Uxyx)

        work      = self.Uxyx.conj().T @ self.rt2_X @ Hx @ self.irt2_X @ self.Uxyx
        work     += work.conj().T
        D2PhiYXH  = -self.irt2_X @ self.Uxyx @ (self.D1xyx_log * work) @ self.Uxyx.conj().T @ self.irt2_X
        D2PhiYXH += mgrad.scnd_frechet(self.D2xyx_entr_UXU, UxyxXHxXUxyx, U=self.irt2_X @ self.Uxyx)

        D2PhiYYH  = -mgrad.scnd_frechet(self.D2xyx_log_UXU, UxyxXHyXUxyx, U=self.irt2_X @ self.Uxyx)

        D2PhiXHH = lin.inp(Hx, D2PhiXXH + D2PhiXYH)
        D2PhiYHH = lin.inp(Hy, D2PhiYXH + D2PhiYYH)

        # Quantum relative entropy third order derivatives
        # X: XX XY YX YY
        D3PhiXXX = mgrad.thrd_frechet(self.Dyxy, self.D2yxy_entr, -self.Dyxy**-2, self.irt2_Y @ self.Uyxy, self.UyxyYUyxy, UyxyYHxYUyxy, UyxyYHxYUyxy)

        work2     = self.Uyxy.conj().T @ self.rt2_Y @ Hy @ self.irt2_Y @ self.Uyxy
        work2    += work2.conj().T
        D3PhiXXY  = mgrad.scnd_frechet(self.D2yxy_entr, work2, UyxyYHxYUyxy, U=self.irt2_Y @ self.Uyxy)
        D3PhiXXY -= mgrad.thrd_frechet(self.Dyxy, self.D2yxy_x2logx, 2*self.Dyxy**-1, self.irt2_Y @ self.Uyxy, self.UyxyYUyxy, UyxyYHxYUyxy, UyxyYHyYUyxy)
        D3PhiXYX  = D3PhiXXY

        D3PhiXYY  = -self.irt2_X @ mgrad.scnd_frechet(self.D2xyx_log, UxyxXHyXUxyx, UxyxXHyXUxyx, U=self.Uxyx) @ self.rt2_X
        D3PhiXYY += D3PhiXYY.conj().T
        D3PhiXYY += mgrad.thrd_frechet(self.Dxyx, self.D2xyx_entr, -self.Dxyx**-2, self.irt2_X @ self.Uxyx, self.UxyxXUxyx, UxyxXHyXUxyx, UxyxXHyXUxyx)

        
        # Y: YY YX XY XX
        D3PhiYYY = -mgrad.thrd_frechet(self.Dxyx, self.D2xyx_log, 2*self.Dxyx**-3, self.irt2_X @ self.Uxyx, self.UxyxXUxyx, UxyxXHyXUxyx, UxyxXHyXUxyx)

        D3PhiYYX  = -mgrad.scnd_frechet(self.D2xyx_log, work, UxyxXHyXUxyx, U=self.irt2_X @ self.Uxyx)
        D3PhiYYX += mgrad.thrd_frechet(self.Dxyx, self.D2xyx_entr, -self.Dxyx**-2, self.irt2_X @ self.Uxyx, self.UxyxXUxyx, UxyxXHxXUxyx, UxyxXHyXUxyx)
        D3PhiYXY  = D3PhiYYX
    
        D3PhiYXX  = self.irt2_Y @ mgrad.scnd_frechet(self.D2yxy_entr, UyxyYHxYUyxy, UyxyYHxYUyxy, U=self.Uyxy) @ self.rt2_Y
        D3PhiYXX += D3PhiYXX.conj().T
        D3PhiYXX -= mgrad.thrd_frechet(self.Dyxy, self.D2yxy_x2logx, 2*self.Dyxy**-1, self.irt2_Y @ self.Uyxy, self.UyxyYUyxy, UyxyYHxYUyxy, UyxyYHxYUyxy)

        
        # Third derivatives of barrier
        dder3_t = -2 * self.zi3 * chi2 - self.zi2 * (D2PhiXHH + D2PhiYHH)

        dder3_X  = -dder3_t * self.DPhiX
        dder3_X -=  2 * self.zi2 * chi * (D2PhiXXH + D2PhiXYH)
        dder3_X +=  self.zi * (D3PhiXXX + D3PhiXXY + D3PhiXYX + D3PhiXYY)
        dder3_X -=  2 * self.inv_X @ Hx @ self.inv_X @ Hx @ self.inv_X
        dder3_X  = (dder3_X + dder3_X.conj().T) * 0.5

        dder3_Y  = -dder3_t * self.DPhiY
        dder3_Y -=  2 * self.zi2 * chi * (D2PhiYXH + D2PhiYYH)
        dder3_Y +=  self.zi * (D3PhiYYY + D3PhiYYX + D3PhiYXY + D3PhiYXX)
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
        self.Ax_vec = np.ascontiguousarray(A[:, self.idx_X])
        self.Ay_vec = np.ascontiguousarray(A[:, self.idx_Y])

        if self.hermitian:
            self.Ax = np.array([Ax_k.reshape((-1, 2)).view(dtype=np.complex128).reshape((self.n, self.n)) for Ax_k in self.Ax_vec])
            self.Ay = np.array([Ay_k.reshape((-1, 2)).view(dtype=np.complex128).reshape((self.n, self.n)) for Ay_k in self.Ay_vec])            
        else:
            self.Ax = np.array([Ax_k.reshape((self.n, self.n)) for Ax_k in self.Ax_vec])
            self.Ay = np.array([Ay_k.reshape((self.n, self.n)) for Ay_k in self.Ay_vec])

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
        Hyx = np.zeros((self.vn, self.vn))
        Hyy = np.zeros((self.vn, self.vn))

        # Make Hxx block
        # D2PhiXXH  = self.zi * mgrad.scnd_frechet(self.D2yxy_entr_UYU, UyxyYHxYUyxy, U=self.irt2_Y @ self.Uyxy)
        lin.congr(self.work8, self.irt2Y_Uyxy.conj().T, self.E, work=self.work7)
        mgrad.scnd_frechet_multi(self.work5, self.D2yxy_entr_UYU, self.work8, U=self.irt2Y_Uyxy, work1=self.work6, work2=self.work7, work3=self.work4)
        self.work5 *= self.zi

        # D2PhiXXH += self.inv_X @ Hx @ self.inv_X
        lin.congr(self.work8, self.inv_X, self.E, work=self.work7)
        self.work8 += self.work5

        Hxx  = self.work8.view(dtype=np.float64).reshape((self.vn, -1))[:, self.triu_indices]
        Hxx *= self.scale

        # D2PhiXXH += self.zi2 * self.DPhiX * lin.inp(self.DPhiX, Hx)
        DPhiX_vec = self.DPhiX.view(dtype=np.float64).reshape(-1)[self.triu_indices] * self.scale
        Hxx += np.outer(DPhiX_vec * self.zi, DPhiX_vec * self.zi)


        # Make Hxy block
        # D2PhiXYH += self.zi * mgrad.scnd_frechet(self.D2xyx_entr_UXU, UxyxXHyXUxyx, U=self.irt2_X @ self.Uxyx)
        lin.congr(self.work8, self.irt2X_Uxyx.conj().T, self.E, work=self.work7)
        mgrad.scnd_frechet_multi(self.work5, self.D2xyx_entr_UXU, self.work8, U=self.irt2X_Uxyx, work1=self.work6, work2=self.work7, work3=self.work4)

        # D2PhiXYH  = -self.zi * self.irt2_X @ self.Uxyx @ (self.D1xyx_log * UxyxXHyXUxyx) @ self.Uxyx.conj().T @ self.rt2_X
        # D2PhiXYH += D2PhiXYH.conj().T
        self.work8 *= self.D1xyx_log
        lin.congr(self.work6, self.irt2X_Uxyx, self.work8, work=self.work7, B=self.rt2_X @ self.Uxyx)
        np.add(self.work6, self.work6.conj().transpose(0, 2, 1), out=self.work7)
        
        self.work5 -= self.work7
        self.work5 *= self.zi

        Hyx  = self.work5.view(dtype=np.float64).reshape((self.vn, -1))[:, self.triu_indices]
        Hyx *= self.scale

        # D2PhiXYH += self.zi2 * self.DPhiX * lin.inp(self.DPhiY, Hy)
        DPhiY_vec = self.DPhiY.view(dtype=np.float64).reshape(-1)[self.triu_indices] * self.scale
        Hyx += np.outer(DPhiY_vec * self.zi, DPhiX_vec * self.zi)


        # Make Hyy block
        # D2PhiYYH  = -self.zi * mgrad.scnd_frechet(self.D2xyx_log_UXU, UxyxXHyXUxyx, U=self.irt2_X @ self.Uxyx)
        lin.congr(self.work8, self.irt2X_Uxyx.conj().T, self.E, work=self.work7)
        mgrad.scnd_frechet_multi(self.work5, self.D2xyx_log_UXU, self.work8, U=self.irt2X_Uxyx, work1=self.work6, work2=self.work7, work3=self.work4)
        self.work5 *= self.zi

        # D2PhiXXH += self.inv_Y @ Hy @ self.inv_Y
        lin.congr(self.work8, self.inv_Y, self.E, work=self.work7)
        self.work8 -= self.work5

        Hyy  = self.work8.view(dtype=np.float64).reshape((self.vn, -1))[:, self.triu_indices]
        Hyy *= self.scale

        # D2PhiXXH += self.zi2 * self.DPhiY * lin.inp(self.DPhiY, Hy)
        Hyy += np.outer(DPhiY_vec * self.zi, DPhiY_vec * self.zi)

        # Construct Hessian and factorize
        self.hess = np.block([
            [self.zi2,  -self.zi2 * sym.mat_to_vec(self.DPhiX, hermitian=self.hermitian).T, -self.zi2 * sym.mat_to_vec(self.DPhiY, hermitian=self.hermitian).T],
            [-self.zi2 * sym.mat_to_vec(self.DPhiX, hermitian=self.hermitian), Hxx, Hyx.T], 
            [-self.zi2 * sym.mat_to_vec(self.DPhiY, hermitian=self.hermitian), Hyx, Hyy]
        ])

        self.hess += self.hess.T
        self.hess *= 0.5

        self.hess_fact = lin.fact(self.hess)
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

        rt2 = np.sqrt(0.5)
        self.E = np.zeros((self.vn, self.n, self.n))
        k = 0
        for j in range(self.n):
            for i in range(j):
                self.E[k, i, j] = rt2
                self.E[k, j, i] = rt2
                k += 1
                if self.hermitian:
                    self.E[k, i, j] = rt2 *  1j
                    self.E[k, j, i] = rt2 * -1j
                    k += 1
            self.E[k, j, j] = 1.
            k += 1

        self.Ax_compact = self.Ax_vec[:, self.triu_indices]
        self.Ay_compact = self.Ay_vec[:, self.triu_indices]

        self.Ax_compact *= self.scale
        self.Ay_compact *= self.scale

        self.A_compact = np.hstack((self.At.reshape((-1, 1)), self.Ax_compact, self.Ay_compact))

        self.work4  = np.empty((self.n, self.n, self.vn), dtype=self.dtype)
        self.work5  = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work6  = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work7  = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work8  = np.empty((self.vn, self.n, self.n), dtype=self.dtype)
        self.work9  = np.empty((self.n, self.n, self.n), dtype=self.dtype)
        self.work10 = np.empty((self.n, 1, self.n), dtype=self.dtype)

        self.invhess_aux_aux_updated = True

    def update_dder3_aux(self):
        assert not self.dder3_aux_updated
        assert self.hess_aux_updated

        self.zi3 = self.zi2 * self.zi

        self.D1yxy_x2logx = mgrad.D1_f(self.Dyxy, self.Dyxy**2 * self.log_Dyxy, self.Dyxy + 2*self.entr_Dyxy)
        self.D2yxy_x2logx = mgrad.D2_f(self.Dyxy, self.D1yxy_x2logx, 3. + 2*self.log_Dyxy)

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