import numpy as np
import scipy as sp
from utils import linear    as lin
from utils import mtxgrad   as mgrad
from utils import symmetric as sym

class Cone():
    def __init__(self, n, hermitian=False):
        # Dimension properties
        self.n  = n               # Side dimension of system
        self.nu = 3 * self.n  # Barrier parameter

        self.hermitian = hermitian                      # Hermitian or symmetric vector space
        self.vn = n*n if hermitian else n*(n+1)//2      # Compact dimension of system

        self.dim   = [n*n, n*n, n*n]   if (not hermitian) else [2*n*n, 2*n*n, 2*n*n]
        self.type  = ['s', 's', 's'] if (not hermitian) else ['h', 'h', 'h']
        self.dtype = np.float64      if (not hermitian) else np.complex128

        self.idx_T = slice(0, self.dim[0])
        self.idx_X = slice(self.dim[0], 2 * self.dim[0])
        self.idx_Y = slice(2 * self.dim[0], 3 * self.dim[0])

        # Get LAPACK operators
        self.X = np.eye(self.n, dtype=self.dtype)

        self.cho_fact  = sp.linalg.lapack.get_lapack_funcs("potrf", (self.X,))
        self.cho_inv   = sp.linalg.lapack.get_lapack_funcs("trtri", (self.X,))

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
            np.eye(self.n, dtype=self.dtype) * t0 / self.n,
            np.eye(self.n, dtype=self.dtype) * x0,
            np.eye(self.n, dtype=self.dtype) * y0,
        ]

        self.set_point(point, point)

        out[0][:] = point[0]
        out[1][:] = point[1]
        out[2][:] = point[2]

        return out
    
    def set_point(self, point, dual, a=True):
        self.T = point[0] * a
        self.X = point[1] * a
        self.Y = point[2] * a

        self.T_d = dual[0] * a
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

        self.Z = self.T + self.rt2_X @ self.log_XYX @ self.rt2_X

        # Try to perform Cholesky factorization to check PSD
        self.Z_chol, info = self.cho_fact(self.Z, lower=True)
        if info != 0:
            self.feas = False
            return self.feas
        self.feas = True

        return self.feas

    def get_val(self):
        assert self.feas_updated
        (sgn, logabsdet_Z) = np.linalg.slogdet(self.Z)
        return -(sgn * logabsdet_Z) - np.sum(np.log(self.Dx)) - np.sum(np.log(self.Dy))

    def update_grad(self):
        assert self.feas_updated
        assert not self.grad_updated
        
        self.Z_chol_inv, _ = self.cho_inv(self.Z_chol, lower=True)
        self.inv_Z = self.Z_chol_inv.conj().T @ self.Z_chol_inv

        self.inv_Dx = np.reciprocal(self.Dx)
        self.inv_Dy = np.reciprocal(self.Dy)

        inv_X_rt2 = self.Ux * np.sqrt(self.inv_Dx)
        self.inv_X = inv_X_rt2 @ inv_X_rt2.conj().T
        inv_Y_rt2 = self.Uy * np.sqrt(self.inv_Dy)
        self.inv_Y = inv_Y_rt2 @ inv_Y_rt2.conj().T        

        self.D1yxy_entr = mgrad.D1_entr(self.Dyxy, self.log_Dyxy, self.entr_Dyxy)
        self.D1xyx_log  = mgrad.D1_log(self.Dxyx, self.log_Dxyx)

        self.irt2Y_Uyxy = self.irt2_Y @ self.Uyxy
        self.irt2X_Uxyx = self.irt2_X @ self.Uxyx

        self.UyxyYUyxy = self.Uyxy.conj().T @ self.Y @ self.Uyxy
        self.UxyxXUxyx = self.Uxyx.conj().T @ self.X @ self.Uxyx

        self.UyxyYZYUyxy = self.Uyxy.conj().T @ self.rt2_Y @ self.inv_Z @ self.rt2_Y @ self.Uyxy
        self.UxyxXZXUxyx = self.Uxyx.conj().T @ self.rt2_X @ self.inv_Z @ self.rt2_X @ self.Uxyx

        self.DPhiX =  self.irt2_Y @ self.Uyxy @ (self.D1yxy_entr * self.UyxyYZYUyxy) @ self.Uyxy.conj().T @ self.irt2_Y
        self.DPhiX = (self.DPhiX + self.DPhiX.conj().T) * 0.5
        self.DPhiY = -self.irt2_X @ self.Uxyx @ (self.D1xyx_log  * self.UxyxXZXUxyx) @ self.Uxyx.conj().T @ self.irt2_X
        self.DPhiY = (self.DPhiY + self.DPhiY.conj().T) * 0.5

        self.grad = [
           -self.inv_Z,
            self.DPhiX - self.inv_X,
            self.DPhiY - self.inv_Y
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

        # Computes Hessian product of the QRE barrier with a single vector (Ht, Hx, Hy)
        # See hess_congr() for additional comments

        (Ht, Hx, Hy) = H

        work    = self.Uyxy.conj().T @ self.irt2_Y @ Hx @ self.irt2_Y @ self.Uyxy
        DxPhiHx = -self.rt2_Y @ self.Uyxy @ (self.D1yxy_entr * work) @ self.Uyxy.conj().T @ self.rt2_Y

        work    = self.Uxyx.conj().T @ self.irt2_X @ Hy @ self.irt2_X @ self.Uxyx
        DyPhiHy = self.rt2_X @ self.Uxyx @ (self.D1xyx_log * work) @ self.Uxyx.conj().T @ self.rt2_X
        
        # Hessian product for T
        out[0][:] = self.inv_Z @ (Ht + DxPhiHx + DyPhiHy) @ self.inv_Z

        # Hessian product for X
        work = self.Uyxy.conj().T @ self.rt2_Y @ self.inv_Z @ Ht @ self.inv_Z @ self.rt2_Y @ self.Uyxy
        Hxt  = -self.irt2_Y @ self.Uyxy @ (self.D1yxy_entr * work) @ self.Uyxy.conj().T @ self.irt2_Y

        work = self.Uyxy.conj().T @ self.rt2_Y @ self.inv_Z @ DxPhiHx @ self.inv_Z @ self.rt2_Y @ self.Uyxy
        Hxx  = -self.irt2_Y @ self.Uyxy @ (self.D1yxy_entr * work) @ self.Uyxy.conj().T @ self.irt2_Y

        work  = self.Uyxy.conj().T @ self.irt2_Y @ Hx @ self.irt2_Y @ self.Uyxy
        Hxx  += mgrad.scnd_frechet(self.D2yxy_entr, self.UyxyYZYUyxy, work, U=self.irt2_Y @ self.Uyxy)

        Hxx += self.inv_X @ Hx @ self.inv_X

        work = self.Uyxy.conj().T @ self.rt2_Y @ self.inv_Z @ DyPhiHy @ self.inv_Z @ self.rt2_Y @ self.Uyxy
        Hxy  = -self.irt2_Y @ self.Uyxy @ (self.D1yxy_entr * work) @ self.Uyxy.conj().T @ self.irt2_Y

        work = self.Uxyx.conj().T @ self.irt2_X @ Hy @ self.irt2_X @ self.Uxyx
        work = self.irt2_X @ self.Uxyx @ (self.D1xyx_log * work) @ self.Uxyx.conj().T @ self.rt2_X @ self.inv_Z
        Hxy -= (work + work.conj().T)

        work = self.Uxyx.conj().T @ self.irt2_X @ Hy @ self.irt2_X @ self.Uxyx
        Hxy += mgrad.scnd_frechet(self.D2xyx_entr, self.UxyxXZXUxyx, work, U=self.irt2_X @ self.Uxyx)

        out[1][:] = Hxt + Hxx + Hxy

        # Hessian product for Y
        work = self.Uxyx.conj().T @ self.rt2_X @ self.inv_Z @ Ht @ self.inv_Z @ self.rt2_X @ self.Uxyx
        Hyt  = self.irt2_X @ self.Uxyx @ (self.D1xyx_log * work) @ self.Uxyx.conj().T @ self.irt2_X

        work = self.Uxyx.conj().T @ self.rt2_X @ self.inv_Z @ DxPhiHx @ self.inv_Z @ self.rt2_X @ self.Uxyx
        Hyx  = self.irt2_X @ self.Uxyx @ (self.D1xyx_log  * work) @ self.Uxyx.conj().T @ self.irt2_X

        work  = self.Uxyx.conj().T @ self.rt2_X @ self.inv_Z @ Hx @ self.irt2_X @ self.Uxyx
        work += work.conj().T
        Hyx  -= self.irt2_X @ self.Uxyx @ (self.D1xyx_log * work) @ self.Uxyx.conj().T @ self.irt2_X

        work = self.Uxyx.conj().T @ self.irt2_X @ Hx @ self.irt2_X @ self.Uxyx
        Hyx += mgrad.scnd_frechet(self.D2xyx_entr, self.UxyxXZXUxyx, work, U=self.irt2_X @ self.Uxyx)

        work = self.Uxyx.conj().T @ self.rt2_X @ self.inv_Z @ DyPhiHy @ self.inv_Z @ self.rt2_X @ self.Uxyx
        Hyy  = self.irt2_X @ self.Uxyx @ (self.D1xyx_log * work) @ self.Uxyx.conj().T @ self.irt2_X

        work  = self.Uxyx.conj().T @ self.irt2_X @ Hy @ self.irt2_X @ self.Uxyx
        Hyy  -= mgrad.scnd_frechet(self.D2xyx_log, self.UxyxXZXUxyx, work, U=self.irt2_X @ self.Uxyx)

        Hyy += self.inv_Y @ Hy @ self.inv_Y

        out[2][:] = Hyt + Hyx + Hyy

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

        vec = np.vstack((sym.mat_to_vec(Ht, hermitian=self.hermitian), sym.mat_to_vec(Hx, hermitian=self.hermitian), sym.mat_to_vec(Hy, hermitian=self.hermitian)))
        sol = lin.fact_solve(self.hess_fact, vec)

        out[0][:] = sym.vec_to_mat(sol[:self.vn], hermitian=self.hermitian)
        out[1][:] = sym.vec_to_mat(sol[self.vn:2*self.vn], hermitian=self.hermitian)
        out[2][:] = sym.vec_to_mat(sol[2*self.vn:], hermitian=self.hermitian)

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

        UyxyYHxYUyxy = self.Uyxy.conj().T @ self.irt2_Y @ Hx @ self.irt2_Y @ self.Uyxy
        UxyxXHyXUxyx = self.Uxyx.conj().T @ self.irt2_X @ Hy @ self.irt2_X @ self.Uxyx
        UxyxXHxXUxyx = self.Uxyx.conj().T @ self.irt2_X @ Hx @ self.irt2_X @ self.Uxyx
        UyxyYHyYUyxy = self.Uyxy.conj().T @ self.irt2_Y @ Hy @ self.irt2_Y @ self.Uyxy        

        DxPhiHx = -self.rt2_Y @ self.Uyxy @ (self.D1yxy_entr * UyxyYHxYUyxy) @ self.Uyxy.conj().T @ self.rt2_Y
        DyPhiHy =  self.rt2_X @ self.Uxyx @ (self.D1xyx_log  * UxyxXHyXUxyx) @ self.Uxyx.conj().T @ self.rt2_X

        D2xxPhiHxHx = -mgrad.scnd_frechet(self.D2yxy_entr, UyxyYHxYUyxy, UyxyYHxYUyxy, U=self.rt2_Y @ self.Uyxy)
        D2yyPhiHyHy =  mgrad.scnd_frechet(self.D2xyx_log,  UxyxXHyXUxyx, UxyxXHyXUxyx, U=self.rt2_X @ self.Uxyx)

        work = Hx @ self.irt2_X @ self.Uxyx @ (self.D1xyx_log * UxyxXHyXUxyx) @ self.Uxyx.conj().T @ self.rt2_X
        D2xyPhiHxHy  = work + work.conj().T
        D2xyPhiHxHy -= mgrad.scnd_frechet(self.D2xyx_entr, UxyxXHxXUxyx, UxyxXHyXUxyx, U=self.rt2_X @ self.Uxyx)

        # T
        dder3_t  = -2 * self.inv_Z @ Ht @ self.inv_Z @ Ht @ self.inv_Z
        dder3_t += -2 * (self.inv_Z @ DxPhiHx @ self.inv_Z @ Ht @ self.inv_Z + self.inv_Z @ Ht @ self.inv_Z @ DxPhiHx @ self.inv_Z)
        dder3_t += -2 * (self.inv_Z @ DyPhiHy @ self.inv_Z @ Ht @ self.inv_Z + self.inv_Z @ Ht @ self.inv_Z @ DyPhiHy @ self.inv_Z)
        
        dder3_t += self.inv_Z @ D2xxPhiHxHx @ self.inv_Z - 2 * self.inv_Z @ DxPhiHx @ self.inv_Z @ DxPhiHx @ self.inv_Z
        dder3_t += self.inv_Z @ D2yyPhiHyHy @ self.inv_Z - 2 * self.inv_Z @ DyPhiHy @ self.inv_Z @ DyPhiHy @ self.inv_Z
        dder3_t += 2 * (self.inv_Z @ D2xyPhiHxHy @ self.inv_Z - self.inv_Z @ DxPhiHx @ self.inv_Z @ DyPhiHy @ self.inv_Z - self.inv_Z @ DyPhiHy @ self.inv_Z @ DxPhiHx @ self.inv_Z)


        # X 
        # TT
        work    = self.Uyxy.conj().T @ self.rt2_Y @ self.inv_Z @ Ht @ self.inv_Z @ Ht @ self.inv_Z @ self.rt2_Y @ self.Uyxy
        dder3_X = 2 * self.irt2_Y @ self.Uyxy @ (self.D1yxy_entr * work) @ self.Uyxy.conj().T @ self.irt2_Y

        # 2TX
        work = self.Uyxy.conj().T @ self.rt2_Y @ self.inv_Z @ Ht @ self.inv_Z @ self.rt2_Y @ self.Uyxy
        dder3_X -= 2 * mgrad.scnd_frechet(self.D2yxy_entr, work, UyxyYHxYUyxy, U=self.irt2_Y @ self.Uyxy)
        work = self.Uyxy.conj().T @ self.rt2_Y @ self.inv_Z @ Ht @ self.inv_Z @ DxPhiHx @ self.inv_Z @ self.rt2_Y @ self.Uyxy
        work += work.conj().T
        dder3_X += 2 * self.irt2_Y @ self.Uyxy @ (self.D1yxy_entr * work) @ self.Uyxy.conj().T @ self.irt2_Y

        #2TY
        work = self.inv_Z @ Ht @ self.inv_Z
        work2 = self.irt2_X @ self.Uxyx @ (self.D1xyx_log * UxyxXHyXUxyx) @ self.Uxyx.conj().T @ self.rt2_X @ work
        dder3_X += 2 * (work2 + work2.conj().T)
        work = self.Uxyx.conj().T @ self.rt2_X @ work @ self.rt2_X @ self.Uxyx
        dder3_X -= 2 * mgrad.scnd_frechet(self.D2xyx_entr, work, UxyxXHyXUxyx, U=self.irt2_X @ self.Uxyx)

        work = self.Uyxy.conj().T @ self.rt2_Y @ self.inv_Z @ Ht @ self.inv_Z @ DyPhiHy @ self.inv_Z @ self.rt2_Y @ self.Uyxy
        work += work.conj().T
        dder3_X += 2 * self.irt2_Y @ self.Uyxy @ (self.D1yxy_entr * work) @ self.Uyxy.conj().T @ self.irt2_Y

        # XX
        work = self.Uyxy.conj().T @ self.rt2_Y @ self.inv_Z @ (D2xxPhiHxHx - 2 * DxPhiHx @ self.inv_Z @ DxPhiHx) @ self.inv_Z @ self.rt2_Y @ self.Uyxy
        dder3_X -= self.irt2_Y @ self.Uyxy @ (self.D1yxy_entr * work) @ self.Uyxy.conj().T @ self.irt2_Y
        work = self.Uyxy.conj().T @ self.rt2_Y @ self.inv_Z @ DxPhiHx @ self.inv_Z @ self.rt2_Y @ self.Uyxy
        dder3_X -= 2 * mgrad.scnd_frechet(self.D2yxy_entr, work, UyxyYHxYUyxy, U=self.irt2_Y @ self.Uyxy)
        dder3_X += mgrad.thrd_frechet(self.Dyxy, self.D2yxy_entr, -self.Dyxy**-2, self.irt2_Y @ self.Uyxy, UyxyYHxYUyxy, UyxyYHxYUyxy, self.UyxyYZYUyxy)
        dder3_X -= 2 * self.inv_X @ Hx @ self.inv_X @ Hx @ self.inv_X

        # 2XY
        work = self.Uyxy.conj().T @ self.rt2_Y @ self.inv_Z @ D2xyPhiHxHy @ self.inv_Z @ self.rt2_Y @ self.Uyxy
        dder3_X -= 2 * self.irt2_Y @ self.Uyxy @ (self.D1yxy_entr * work) @ self.Uyxy.conj().T @ self.irt2_Y

        work = self.inv_Z @ DxPhiHx @ self.inv_Z
        work2 = self.irt2_X @ self.Uxyx @ (self.D1xyx_log * UxyxXHyXUxyx) @ self.Uxyx.conj().T @ self.rt2_X @ work
        dder3_X += 2 * (work2 + work2.conj().T)
        work = self.Uxyx.conj().T @ self.rt2_X @ work @ self.rt2_X @ self.Uxyx
        dder3_X -= 2 * mgrad.scnd_frechet(self.D2xyx_entr, work, UxyxXHyXUxyx, U=self.irt2_X @ self.Uxyx)

        work = self.Uyxy.conj().T @ self.rt2_Y @ self.inv_Z @ DxPhiHx @ self.inv_Z @ DyPhiHy @ self.inv_Z @ self.rt2_Y @ self.Uyxy
        work += work.conj().T
        dder3_X += 2 * self.irt2_Y @ self.Uyxy @ (self.D1yxy_entr * work) @ self.Uyxy.conj().T @ self.irt2_Y

        work = self.Uyxy.conj().T @ self.rt2_Y @ self.inv_Z @ Hy @ self.irt2_Y @ self.Uyxy
        work += work.conj().T
        dder3_X += 2 * mgrad.scnd_frechet(self.D2yxy_entr, work, UyxyYHxYUyxy, U=self.irt2_Y @ self.Uyxy)
        work = self.Uyxy.conj().T @ self.rt2_Y @ self.inv_Z @ self.rt2_Y @ self.Uyxy
        dder3_X -= 2 * mgrad.thrd_frechet(self.Dyxy, self.D2yxy_x2logx, 2*self.Dyxy**-1, self.irt2_Y @ self.Uyxy, work, UyxyYHyYUyxy, UyxyYHxYUyxy)

        work = self.Uyxy.conj().T @ self.rt2_Y @ self.inv_Z @ DyPhiHy @ self.inv_Z @ self.rt2_Y @ self.Uyxy
        dder3_X -= 2 * mgrad.scnd_frechet(self.D2yxy_entr, work, UyxyYHxYUyxy, U=self.irt2_Y @ self.Uyxy)

        # YY
        work = self.inv_Z @ DyPhiHy @ self.inv_Z
        work2 = self.irt2_X @ self.Uxyx @ (self.D1xyx_log * UxyxXHyXUxyx) @ self.Uxyx.conj().T @ self.rt2_X @ work
        dder3_X += 2 * (work2 + work2.conj().T)
        work = self.Uxyx.conj().T @ self.rt2_X @ work @ self.rt2_X @ self.Uxyx
        dder3_X -= 2 * mgrad.scnd_frechet(self.D2xyx_entr, work, UxyxXHyXUxyx, U=self.irt2_X @ self.Uxyx)

        work = self.Uyxy.conj().T @ self.rt2_Y @ self.inv_Z @ DyPhiHy @ self.inv_Z @ DyPhiHy @ self.inv_Z @ self.rt2_Y @ self.Uyxy
        dder3_X += 2 * self.irt2_Y @ self.Uyxy @ (self.D1yxy_entr * work) @ self.Uyxy.conj().T @ self.irt2_Y

        work = self.irt2_X @ mgrad.scnd_frechet(self.D2xyx_log, UxyxXHyXUxyx, UxyxXHyXUxyx, U=self.Uxyx) @ self.rt2_X @ self.inv_Z
        dder3_X -= work + work.conj().T
        dder3_X += mgrad.thrd_frechet(self.Dxyx, self.D2xyx_entr, -self.Dxyx**-2, self.irt2_X @ self.Uxyx, self.UxyxXZXUxyx, UxyxXHyXUxyx, UxyxXHyXUxyx)

        work = self.Uyxy.conj().T @ self.rt2_Y @ self.inv_Z @ D2yyPhiHyHy @ self.inv_Z @ self.rt2_Y @ self.Uyxy
        dder3_X -= self.irt2_Y @ self.Uyxy @ (self.D1yxy_entr * work) @ self.Uyxy.conj().T @ self.irt2_Y


        # Y
        # TT
        work    = self.Uxyx.conj().T @ self.rt2_X @ self.inv_Z @ Ht @ self.inv_Z @ Ht @ self.inv_Z @ self.rt2_X @ self.Uxyx
        dder3_Y = -2 * self.irt2_X @ self.Uxyx @ (self.D1xyx_log * work) @ self.Uxyx.conj().T @ self.irt2_X

        #2TX
        work = self.Uxyx.conj().T @ self.rt2_X @ self.inv_Z @ Ht @ self.inv_Z @ Hx @ self.irt2_X @ self.Uxyx
        work += work.conj().T
        dder3_Y += 2 * self.irt2_X @ self.Uxyx @ (self.D1xyx_log * work) @ self.Uxyx.conj().T @ self.irt2_X
        work = self.Uxyx.conj().T @ self.rt2_X @ self.inv_Z @ Ht @ self.inv_Z @ self.rt2_X @ self.Uxyx
        dder3_Y -= 2 * mgrad.scnd_frechet(self.D2xyx_entr, work, UxyxXHxXUxyx, U=self.irt2_X @ self.Uxyx)

        work = self.Uxyx.conj().T @ self.rt2_X @ self.inv_Z @ Ht @ self.inv_Z @ DxPhiHx @ self.inv_Z @ self.rt2_X @ self.Uxyx
        work += work.conj().T
        dder3_Y -= 2 * self.irt2_X @ self.Uxyx @ (self.D1xyx_log * work) @ self.Uxyx.conj().T @ self.irt2_X

        # 2TY
        work = self.Uxyx.conj().T @ self.rt2_X @ self.inv_Z @ Ht @ self.inv_Z @ self.rt2_X @ self.Uxyx
        dder3_Y += 2 * mgrad.scnd_frechet(self.D2xyx_log, work, UxyxXHyXUxyx, U=self.irt2_X @ self.Uxyx)
        work = self.Uxyx.conj().T @ self.rt2_X @ self.inv_Z @ Ht @ self.inv_Z @ DyPhiHy @ self.inv_Z @ self.rt2_X @ self.Uxyx
        work += work.conj().T
        dder3_Y -= 2 * self.irt2_X @ self.Uxyx @ (self.D1xyx_log * work) @ self.Uxyx.conj().T @ self.irt2_X

        # YY
        work = self.Uxyx.conj().T @ self.rt2_X @ self.inv_Z @ (D2yyPhiHyHy - 2 * DyPhiHy @ self.inv_Z @ DyPhiHy) @ self.inv_Z @ self.rt2_X @ self.Uxyx
        dder3_Y += self.irt2_X @ self.Uxyx @ (self.D1xyx_log * work) @ self.Uxyx.conj().T @ self.irt2_X
        work = self.Uxyx.conj().T @ self.rt2_X @ self.inv_Z @ DyPhiHy @ self.inv_Z @ self.rt2_X @ self.Uxyx
        dder3_Y += 2 * mgrad.scnd_frechet(self.D2xyx_log, work, UxyxXHyXUxyx, U=self.irt2_X @ self.Uxyx)
        dder3_Y -= mgrad.thrd_frechet(self.Dxyx, self.D2xyx_log, 2*self.Dxyx**-3, self.irt2_X @ self.Uxyx, UxyxXHyXUxyx, UxyxXHyXUxyx, self.UxyxXZXUxyx)
        dder3_Y -= 2 * self.inv_Y @ Hy @ self.inv_Y @ Hy @ self.inv_Y

        # XY
        work = self.Uxyx.conj().T @ self.rt2_X @ self.inv_Z @ D2xyPhiHxHy @ self.inv_Z @ self.rt2_X @ self.Uxyx
        dder3_Y += 2 * self.irt2_X @ self.Uxyx @ (self.D1xyx_log * work) @ self.Uxyx.conj().T @ self.irt2_X

        work = self.inv_Z @ DyPhiHy @ self.inv_Z
        work2 = self.Uxyx.conj().T @ self.rt2_X @ work @ Hx @ self.irt2_X @ self.Uxyx
        work2 += work2.conj().T
        dder3_Y += 2 * self.irt2_X @ self.Uxyx @ (self.D1xyx_log * work2) @ self.Uxyx.conj().T @ self.irt2_X
        work = self.Uxyx.conj().T @ self.rt2_X @ work @ self.rt2_X @ self.Uxyx
        dder3_Y -= 2 * mgrad.scnd_frechet(self.D2xyx_entr, work, UxyxXHxXUxyx, U=self.irt2_X @ self.Uxyx)

        work = self.Uxyx.conj().T @ self.rt2_X @ self.inv_Z @ DxPhiHx @ self.inv_Z @ DyPhiHy @ self.inv_Z @ self.rt2_X @ self.Uxyx
        work += work.conj().T
        dder3_Y -= 2 * self.irt2_X @ self.Uxyx @ (self.D1xyx_log * work) @ self.Uxyx.conj().T @ self.irt2_X

        work = self.Uxyx.conj().T @ self.rt2_X @ self.inv_Z @ Hx @ self.irt2_X @ self.Uxyx
        work += work.conj().T
        dder3_Y -= 2 * mgrad.scnd_frechet(self.D2xyx_log, work, UxyxXHyXUxyx, U=self.irt2_X @ self.Uxyx)
        work = self.Uxyx.conj().T @ self.rt2_X @ self.inv_Z @ self.rt2_X @ self.Uxyx
        dder3_Y += 2 * mgrad.thrd_frechet(self.Dxyx, self.D2xyx_entr, -self.Dxyx**-2, self.irt2_X @ self.Uxyx, work, UxyxXHyXUxyx, UxyxXHxXUxyx)

        work = self.Uxyx.conj().T @ self.rt2_X @ self.inv_Z @ DxPhiHx @ self.inv_Z @ self.rt2_X @ self.Uxyx
        dder3_Y += 2 * mgrad.scnd_frechet(self.D2xyx_log, work, UxyxXHyXUxyx, U=self.irt2_X @ self.Uxyx)

        # XX
        work = self.inv_Z @ DxPhiHx @ self.inv_Z
        work2 = self.Uxyx.conj().T @ self.rt2_X @ work @ Hx @ self.irt2_X @ self.Uxyx
        work2 += work2.conj().T
        dder3_Y += 2 * self.irt2_X @ self.Uxyx @ (self.D1xyx_log * work2) @ self.Uxyx.conj().T @ self.irt2_X
        work = self.Uxyx.conj().T @ self.rt2_X @ work @ self.rt2_X @ self.Uxyx
        dder3_Y -= 2 * mgrad.scnd_frechet(self.D2xyx_entr, work, UxyxXHxXUxyx, U=self.irt2_X @ self.Uxyx)

        work = self.Uxyx.conj().T @ self.rt2_X @ self.inv_Z @ DxPhiHx @ self.inv_Z @ DxPhiHx @ self.inv_Z @ self.rt2_X @ self.Uxyx
        dder3_Y -= 2 * self.irt2_X @ self.Uxyx @ (self.D1xyx_log * work) @ self.Uxyx.conj().T @ self.irt2_X

        work = self.irt2_Y @ mgrad.scnd_frechet(self.D2yxy_entr, UyxyYHxYUyxy, UyxyYHxYUyxy, U=self.Uyxy) @ self.rt2_Y @ self.inv_Z
        dder3_Y += work + work.conj().T
        dder3_Y -= mgrad.thrd_frechet(self.Dyxy, self.D2yxy_x2logx, 2*self.Dyxy**-1, self.irt2_Y @ self.Uyxy, self.UyxyYZYUyxy, UyxyYHxYUyxy, UyxyYHxYUyxy)

        work = self.Uxyx.conj().T @ self.rt2_X @ self.inv_Z @ D2xxPhiHxHx @ self.inv_Z @ self.rt2_X @ self.Uxyx
        dder3_Y += self.irt2_X @ self.Uxyx @ (self.D1xyx_log * work) @ self.Uxyx.conj().T @ self.irt2_X

        out[0][:] += dder3_t * a
        out[1][:] += dder3_X * a
        out[2][:] += dder3_Y * a

        return out
    
    def prox(self):
        assert self.feas_updated
        if not self.grad_updated:
            self.update_grad()
        psi = [
            self.T_d + self.grad[0],
            self.X_d + self.grad[1],
            self.Y_d + self.grad[2]
        ]
        temp = [np.zeros((self.n, self.n), dtype=self.dtype), np.zeros((self.n, self.n), dtype=self.dtype), np.zeros((self.n, self.n), dtype=self.dtype)]
        self.invhess_prod_ip(temp, psi)
        return lin.inp(temp[0], psi[0]) + lin.inp(temp[1], psi[1]) + lin.inp(temp[2], psi[2])
    
    # ========================================================================
    # Auxilliary functions
    # ========================================================================
    def congr_aux(self, A):
        assert not self.congr_aux_updated

        self.At_vec = np.ascontiguousarray(A[:, self.idx_T])
        self.Ax_vec = np.ascontiguousarray(A[:, self.idx_X])
        self.Ay_vec = np.ascontiguousarray(A[:, self.idx_Y])

        # if self.hermitian:
        #     self.Ax = np.array([Ax_k.reshape((-1, 2)).view(dtype=np.complex128).reshape((self.n, self.n)) for Ax_k in self.Ax_vec])
        #     self.Ay = np.array([Ay_k.reshape((-1, 2)).view(dtype=np.complex128).reshape((self.n, self.n)) for Ay_k in self.Ay_vec])            
        # else:
        #     self.Ax = np.array([Ax_k.reshape((self.n, self.n)) for Ax_k in self.Ax_vec])
        #     self.Ay = np.array([Ay_k.reshape((self.n, self.n)) for Ay_k in self.Ay_vec])

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

        self.Htt = np.zeros((self.vn, self.vn))
        self.Htx = np.zeros((self.vn, self.vn))
        self.Hty = np.zeros((self.vn, self.vn))
        self.Hxx = np.zeros((self.vn, self.vn))
        self.Hxy = np.zeros((self.vn, self.vn))
        self.Hyy = np.zeros((self.vn, self.vn))

        # Htt block
        lin.congr(self.work8, self.inv_Z, self.E, work=self.work7)
        self.Htt  = self.work8.view(dtype=np.float64).reshape((self.vn, -1))[:, self.triu_indices]
        self.Htt *= self.scale

        # Htx block
        lin.congr(self.work8, self.Uyxy.conj().T @ self.irt2_Y, self.E, work=self.work7)
        self.work8 *= -self.D1yxy_entr
        lin.congr(self.work6, self.inv_Z @ self.rt2_Y @ self.Uyxy, self.work8, work=self.work7)
        iZ_DxPhiHx_iZ = self.work6.copy()

        self.Htx  = self.work6.view(dtype=np.float64).reshape((self.vn, -1))[:, self.triu_indices]
        self.Htx *= self.scale
        self.Htx  = self.Htx.T

        # Hty block
        lin.congr(self.work8, self.Uxyx.conj().T @ self.irt2_X, self.E, work=self.work7)
        self.work8 *= self.D1xyx_log
        lin.congr(self.work5, self.inv_Z @ self.rt2_X @ self.Uxyx, self.work8, work=self.work7)
        iZ_DyPhiHy_iZ = self.work5.copy()     

        self.Hty  = self.work5.view(dtype=np.float64).reshape((self.vn, -1))[:, self.triu_indices]
        self.Hty *= self.scale
        self.Hty  = self.Hty.T

        # Hxx block
        # work = self.Uyxy.conj().T @ self.rt2_Y @ self.inv_Z @ DxPhiHx @ self.inv_Z @ self.rt2_Y @ self.Uyxy
        # Hxx  = -self.irt2_Y @ self.Uyxy @ (self.D1yxy_entr * work) @ self.Uyxy.conj().T @ self.irt2_Y
        lin.congr(self.work8, self.Uyxy.conj().T @ self.rt2_Y, iZ_DxPhiHx_iZ, work=self.work7)
        self.work8 *= -self.D1yxy_entr
        lin.congr(self.work6, self.irt2_Y @ self.Uyxy, self.work8, work=self.work7)

        # work  = self.Uyxy.conj().T @ self.irt2_Y @ Hx @ self.irt2_Y @ self.Uyxy
        # Hxx  += mgrad.scnd_frechet(self.D2yxy_entr, self.UyxyYZYUyxy, work, U=self.irt2_Y @ self.Uyxy)
        lin.congr(self.work8, self.Uyxy.conj().T @ self.irt2_Y, self.E, work=self.work7)
        mgrad.scnd_frechet_multi(self.work5, self.D2yxy_entr, self.work8, self.UyxyYZYUyxy, U=self.irt2_Y @ self.Uyxy, work1=self.work5, work2=self.work7, work3=self.work4)
        self.work6 += self.work5

        # Hxx += self.inv_X @ Hx @ self.inv_X
        lin.congr(self.work5, self.inv_X, self.E, work=self.work7)
        self.work6 += self.work5

        self.Hxx  = self.work6.view(dtype=np.float64).reshape((self.vn, -1))[:, self.triu_indices]
        self.Hxx *= self.scale


        # Hyy block
        # work = self.Uxyx.conj().T @ self.rt2_X @ self.inv_Z @ DyPhiHy @ self.inv_Z @ self.rt2_X @ self.Uxyx
        # Hyy  = self.irt2_X @ self.Uxyx @ (self.D1xyx_log * work) @ self.Uxyx.conj().T @ self.irt2_X
        lin.congr(self.work8, self.Uxyx.conj().T @ self.rt2_X, iZ_DyPhiHy_iZ, work=self.work7)
        self.work8 *= self.D1xyx_log
        lin.congr(self.work6, self.irt2_X @ self.Uxyx, self.work8, work=self.work7)

        # work  = self.Uxyx.conj().T @ self.irt2_X @ Hy @ self.irt2_X @ self.Uxyx
        # Hyy  -= mgrad.scnd_frechet(self.D2xyx_log, self.UxyxXZXUxyx, work, U=self.irt2_X @ self.Uxyx)
        lin.congr(self.work8, self.Uxyx.conj().T @ self.irt2_X, self.E, work=self.work7)
        mgrad.scnd_frechet_multi(self.work5, self.D2xyx_log, self.work8, self.UxyxXZXUxyx, U=self.irt2_X @ self.Uxyx, work1=self.work5, work2=self.work7, work3=self.work4)
        self.work6 -= self.work5

        # Hyy += self.inv_Y @ Hy @ self.inv_Y
        lin.congr(self.work5, self.inv_Y, self.E, work=self.work7)
        self.work6 += self.work5

        self.Hyy  = self.work6.view(dtype=np.float64).reshape((self.vn, -1))[:, self.triu_indices]
        self.Hyy *= self.scale

        # Hxy block
        # work = self.Uyxy.conj().T @ self.rt2_Y @ self.inv_Z @ DyPhiHy @ self.inv_Z @ self.rt2_Y @ self.Uyxy
        # Hxy  = -self.irt2_Y @ self.Uyxy @ (self.D1yxy_entr * work) @ self.Uyxy.conj().T @ self.irt2_Y
        lin.congr(self.work8, self.Uyxy.conj().T @ self.rt2_Y, iZ_DyPhiHy_iZ, work=self.work7)
        self.work8 *= -self.D1yxy_entr
        lin.congr(self.work6, self.irt2_Y @ self.Uyxy, self.work8, work=self.work7)

        # work = self.Uxyx.conj().T @ self.irt2_X @ Hy @ self.irt2_X @ self.Uxyx
        # work = self.irt2_X @ self.Uxyx @ (self.D1xyx_log * work) @ self.Uxyx.conj().T @ self.rt2_X @ self.inv_Z
        # Hxy -= (work + work.conj().T)
        lin.congr(self.work8, self.Uxyx.conj().T @ self.irt2_X, self.E, work=self.work7)
        self.work8 *= self.D1xyx_log
        lin.congr(self.work5, self.irt2_X @ self.Uxyx, self.work8, work=self.work7, B=self.inv_Z @ self.rt2_X @ self.Uxyx)
        self.work6 -= self.work5
        self.work6 -= self.work5.conj().transpose(0, 2, 1)

        # work = self.Uxyx.conj().T @ self.irt2_X @ Hy @ self.irt2_X @ self.Uxyx
        # Hxy += mgrad.scnd_frechet(self.D2xyx_entr, self.UxyxXZXUxyx, work, U=self.irt2_X @ self.Uxyx)
        lin.congr(self.work8, self.Uxyx.conj().T @ self.irt2_X, self.E, work=self.work7)
        mgrad.scnd_frechet_multi(self.work5, self.D2xyx_entr, self.work8, self.UxyxXZXUxyx, U=self.irt2_X @ self.Uxyx, work1=self.work5, work2=self.work7, work3=self.work4)
        self.work6 += self.work5

        self.Hxy  = self.work6.view(dtype=np.float64).reshape((self.vn, -1))[:, self.triu_indices]
        self.Hxy *= self.scale
        self.Hxy  = self.Hxy.T



        # Construct Hessian and factorize
        self.hess = np.block([
            [self.Htt,   self.Htx,   self.Hty],
            [self.Htx.T, self.Hxx,   self.Hxy], 
            [self.Hty.T, self.Hxy.T, self.Hyy]
        ])

        # self.hess += self.hess.T
        # self.hess *= 0.5

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
        self.E = np.zeros((self.vn, self.n, self.n), dtype=self.dtype)
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

        self.At_compact = self.At_vec[:, self.triu_indices]
        self.Ax_compact = self.Ax_vec[:, self.triu_indices]
        self.Ay_compact = self.Ay_vec[:, self.triu_indices]

        self.At_compact *= self.scale
        self.Ax_compact *= self.scale
        self.Ay_compact *= self.scale

        self.A_compact = np.hstack((self.At_compact, self.Ax_compact, self.Ay_compact))

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