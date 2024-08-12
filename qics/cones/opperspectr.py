import numpy as np
import qics._utils.linalg as lin
import qics._utils.gradient as grad
from qics.cones.base import Cone, get_perspective_derivatives

class OpPerspecTr(Cone):
    """A class representing a trace operator perspective epigraph cone

    .. math::
    
        \\mathcal{K}_{\\text{op,tr}}^g = \\text{cl}\\{ (t, X, Y) \\in \\mathbb{R} \\times \\mathbb{H}^n_+ \\times \\mathbb{H}^n_+ : t \\geq \\text{tr}[P_g(X, Y)] \\},
        
    for an operator concave function :math:`g:[0, \\infty)\\rightarrow\\mathbb{R}`, with barrier function

    .. math::
    
        (t, X, Y) \\mapsto -\\log(t - \\text{tr}[P_g(X, Y)]) - \\log\\det(X) - \\log\\det(Y),
        
    where

    .. math::

        P_g(X, Y) = X^{1/2} g(X^{-1/2} Y X^{-1/2}) X^{1/2},
        
    is the operator perspective of :math:`g`.

    Parameters
    ----------
    n : int
        Dimension of the (n, n) matrices :math:`X` and :math:`Y`.
    func : string or float
        Choice for the function g. Can either be

        - ``"log"``             : :math:`g(x) = -\\log(x)`
        - ``(0, 1)``            : :math:`g(x) = -x^p`
        - ``(-1, 0) or (1, 2)`` : :math:`g(x) = x^p`

    iscomplex : bool
        Whether the matrices symmetric :math:`X,Y \\in \\mathbb{S}^n` (False) or Hermitian :math:`X,Y \\in \\mathbb{H}^n` (True). Default is False.
    """    
    def __init__(self, n, func, iscomplex=False):        
        # Dimension properties
        self.n  = n               # Side dimension of system
        self.nu = 1 + 2 * self.n  # Barrier parameter

        self.iscomplex = iscomplex                      # Hermitian or symmetric vector space
        self.vn = n*n if iscomplex else n*(n+1)//2      # Compact dimension of system

        self.dim   = [1, n*n, n*n]   if (not iscomplex) else [1, 2*n*n, 2*n*n]
        self.type  = ['r', 's', 's'] if (not iscomplex) else ['r', 'h', 'h']
        self.dtype = np.float64      if (not iscomplex) else np.complex128

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

        (
            self.g,  self.dg,  self.d2g,  self.d3g, 
            self.xg, self.dxg, self.d2xg, self.d3xg,
            self.h,  self.dh,  self.d2h,  self.d3h,
            self.xh, self.dxh, self.d2xh, self.d3xh
        ) = get_perspective_derivatives(func)

        self.precompute_mat_vec()

        return

    def get_iscomplex(self):
        return self.iscomplex

    def get_init_point(self, out):
        (t0, x0, y0) = self.get_central_ray()

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
    
    def get_feas(self):
        if self.feas_updated:
            return self.feas
        
        self.feas_updated = True

        (self.t, self.X, self.Y) = self.primal

        # Check that X and Y are PSD
        self.Dx, self.Ux = np.linalg.eigh(self.X)
        self.Dy, self.Uy = np.linalg.eigh(self.Y)

        if any(self.Dx <= 0) or any(self.Dy <= 0):
            self.feas = False
            return self.feas
        
        # Construct (X^-1/2 Y X^-1/2) and (Y^-1/2 X Y^-1/2) 
        # and double check they are also PSD (in case of numerical errors)
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

        XYX = self.irt2_X @ self.Y @ self.irt2_X
        YXY = self.irt2_Y @ self.X @ self.irt2_Y

        self.Dxyx, self.Uxyx = np.linalg.eigh(XYX)
        self.Dyxy, self.Uyxy = np.linalg.eigh(YXY)

        if any(self.Dxyx <= 0) or any(self.Dyxy <= 0):
            self.feas = False
            return self.feas
        
        # Check that t > tr[X^0.5 g(X^-1/2 Y X^-1/2) X^0.5]
        self.g_Dxyx  = self.g(self.Dxyx)
        self.h_Dyxy  = self.h(self.Dyxy)
        g_XYX  = (self.Uxyx * self.g_Dxyx) @ self.Uxyx.conj().T

        self.z = self.t[0, 0] - lin.inp(self.X, g_XYX)

        self.feas = (self.z > 0)
        return self.feas

    def get_val(self):
        assert self.feas_updated

        return -np.log(self.z) - np.sum(np.log(self.Dx)) - np.sum(np.log(self.Dy))

    def update_grad(self):
        assert self.feas_updated
        assert not self.grad_updated
        
        # Compute X^-1 and Y^-1
        inv_Dx = np.reciprocal(self.Dx)
        inv_X_rt2 = self.Ux * np.sqrt(inv_Dx)
        self.inv_X = inv_X_rt2 @ inv_X_rt2.conj().T

        inv_Dy = np.reciprocal(self.Dy)
        inv_Y_rt2 = self.Uy * np.sqrt(inv_Dy)
        self.inv_Y = inv_Y_rt2 @ inv_Y_rt2.conj().T

        # Precompute useful expressions
        self.UyxyYUyxy = self.Uyxy.conj().T @ self.Y @ self.Uyxy
        self.UxyxXUxyx = self.Uxyx.conj().T @ self.X @ self.Uxyx

        self.irt2Y_Uyxy = self.irt2_Y @ self.Uyxy
        self.irt2X_Uxyx = self.irt2_X @ self.Uxyx

        self.zi =  np.reciprocal(self.z)

        # Compute derivatives of tr[Pg(X, Y)]
        self.D1yxy_h = grad.D1_f(self.Dyxy, self.h_Dyxy, self.dh(self.Dyxy))
        self.D1xyx_g = grad.D1_f(self.Dxyx, self.g_Dxyx, self.dg(self.Dxyx))

        self.DPhiX = self.irt2Y_Uyxy @ (self.D1yxy_h * self.UyxyYUyxy) @ self.irt2Y_Uyxy.conj().T
        self.DPhiX = (self.DPhiX + self.DPhiX.conj().T) * 0.5
        self.DPhiY = self.irt2X_Uxyx @ (self.D1xyx_g  * self.UxyxXUxyx) @ self.irt2X_Uxyx.conj().T
        self.DPhiY = (self.DPhiY + self.DPhiY.conj().T) * 0.5

        self.DPhiX_vec = self.DPhiX.view(dtype=np.float64).reshape(-1, 1)[self.triu_idxs] * self.scale.reshape(-1, 1)
        self.DPhiY_vec = self.DPhiY.view(dtype=np.float64).reshape(-1, 1)[self.triu_idxs] * self.scale.reshape(-1, 1)
        self.DPhi_vec = np.vstack((self.DPhiX_vec, self.DPhiY_vec))

        self.grad = [
           -self.zi,
            self.zi * self.DPhiX - self.inv_X,
            self.zi * self.DPhiY - self.inv_Y
        ]

        self.grad_updated = True

    def hess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        # Computes Hessian product of tr[Pg(X, Y)] barrier with a single vector (Ht, Hx, Hy)

        (Ht, Hx, Hy) = H

        UyxyYHxYUyxy = self.irt2Y_Uyxy.conj().T @ Hx @ self.irt2Y_Uyxy
        UxyxXHyXUxyx = self.irt2X_Uxyx.conj().T @ Hy @ self.irt2X_Uxyx

        # Hessian product of tr[Pg(X, Y)]
        D2PhiXXH = grad.scnd_frechet(self.D2yxy_h, self.UyxyYUyxy, UyxyYHxYUyxy, U=self.irt2Y_Uyxy)

        D2PhiXYH  = self.irt2X_Uxyx @ (self.D1xyx_g * UxyxXHyXUxyx) @ self.Uxyx.conj().T @ self.rt2_X
        D2PhiXYH += D2PhiXYH.conj().T
        D2PhiXYH -= grad.scnd_frechet(self.D2xyx_xg, self.UxyxXUxyx, UxyxXHyXUxyx, U=self.irt2X_Uxyx)

        D2PhiYXH  = self.irt2Y_Uyxy @ (self.D1yxy_h * UyxyYHxYUyxy) @ self.Uyxy.conj().T @ self.rt2_Y
        D2PhiYXH += D2PhiYXH.conj().T
        D2PhiYXH -= grad.scnd_frechet(self.D2yxy_xh, self.UyxyYUyxy, UyxyYHxYUyxy, U=self.irt2Y_Uyxy)

        D2PhiYYH = grad.scnd_frechet(self.D2xyx_g, self.UxyxXUxyx, UxyxXHyXUxyx, U=self.irt2X_Uxyx)
        
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
    
        vec = self.At - self.DPhiX_vec.T @ self.Ax_compact.T - self.DPhiY_vec.T @ self.Ay_compact.T
        vec *= self.zi
        
        out = self.A_compact @ self.hess @ self.A_compact.T
        out += np.outer(vec, vec)
        return out
    
    def invhess_prod_ip(self, out, H):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        # Computes inverse Hessian product of tr[Pg(X, Y)] barrier with a single vector (Ht, Hx, Hy)
        # See invhess_congr() for additional comments

        (Ht, Hx, Hy) = H

        Wx      = Hx + Ht * self.DPhiX
        Wx_vec  = Wx.view(dtype=np.float64).reshape(-1)[self.triu_idxs]
        Wx_vec *= self.scale

        Wy      = Hy + Ht * self.DPhiY
        Wy_vec  = Wy.view(dtype=np.float64).reshape(-1)[self.triu_idxs]
        Wy_vec *= self.scale

        Wxy_vec = np.hstack((Wx_vec, Wy_vec))
        outxy   = lin.cho_solve(self.hess_fact, Wxy_vec)

        outxy = outxy.reshape(2, -1)
        outxy[:, self.diag_idxs] *= 0.5
        outxy /= self.scale

        outX = np.zeros_like(Wx)
        outX.view(dtype=np.float64).reshape(-1)[self.triu_idxs] = outxy[0]
        out[1][:] = outX + outX.conj().T

        outY = np.zeros_like(Wx)
        outY.view(dtype=np.float64).reshape(-1)[self.triu_idxs] = outxy[1]
        out[2][:] = outY + outY.conj().T

        out[0][:] = self.z2 * Ht + lin.inp(out[1], self.DPhiX) + lin.inp(out[2], self.DPhiY)

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
        # where (Wx, Wy) = (Hx, Hy) + Ht DPhi(X, Y) and
        #     M = 1/z [ D2xxPhi D2xyPhi ] + [ X^1 kron X^-1               ]
        #             [ D2yxPhi D2yyPhi ]   [               Y^1 kron Y^-1 ]

        # Compute (Wx, Wy)
        np.outer(self.DPhi_vec, self.At, out=self.work)
        self.work += self.A_compact.T
        
        # Solve for (X, Y) =  M \ (Wx, Wy)
        lhsxy = lin.cho_solve(self.hess_fact, self.work)
        # Solve for t = z^2 Ht + <DPhi(X, Y), (X, Y)>
        lhst  = self.z2 * self.At.reshape(-1, 1) + lhsxy.T @ self.DPhi_vec

        # Multiply A (H A')
        return self.A_compact @ lhsxy + np.outer(self.At, lhst)

    def third_dir_deriv_axpy(self, out, H, a=True):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.dder3_aux_updated:
            self.update_dder3_aux()

        (Ht, Hx, Hy) = H

        chi = Ht[0, 0] - lin.inp(self.DPhiX, Hx) - lin.inp(self.DPhiY, Hy)
        chi2 = chi * chi

        UyxyYHxYUyxy = self.Uyxy.conj().T @ self.irt2_Y @ Hx @ self.irt2_Y @ self.Uyxy
        UxyxXHyXUxyx = self.Uxyx.conj().T @ self.irt2_X @ Hy @ self.irt2_X @ self.Uxyx
        UxyxXHxXUxyx = self.Uxyx.conj().T @ self.irt2_X @ Hx @ self.irt2_X @ self.Uxyx
        UyxyYHyYUyxy = self.Uyxy.conj().T @ self.irt2_Y @ Hy @ self.irt2_Y @ self.Uyxy

        # Hessian products of tr[Pg(X, Y)]
        D2PhiXXH = grad.scnd_frechet(self.D2yxy_h, self.UyxyYUyxy, UyxyYHxYUyxy, U=self.irt2Y_Uyxy)

        D2PhiXYH  = self.irt2_X @ self.Uxyx @ (self.D1xyx_g * UxyxXHyXUxyx) @ self.Uxyx.conj().T @ self.rt2_X
        D2PhiXYH += D2PhiXYH.conj().T
        D2PhiXYH -= grad.scnd_frechet(self.D2xyx_xg, self.UxyxXUxyx, UxyxXHyXUxyx, U=self.irt2X_Uxyx)

        D2PhiYXH  = self.irt2_Y @ self.Uyxy @ (self.D1yxy_h * UyxyYHxYUyxy) @ self.Uyxy.conj().T @ self.rt2_Y
        D2PhiYXH += D2PhiYXH.conj().T
        D2PhiYXH -= grad.scnd_frechet(self.D2yxy_xh, self.UyxyYUyxy, UyxyYHxYUyxy, U=self.irt2Y_Uyxy)

        D2PhiYYH = grad.scnd_frechet(self.D2xyx_g, self.UxyxXUxyx, UxyxXHyXUxyx, U=self.irt2X_Uxyx)

        D2PhiXHH = lin.inp(Hx, D2PhiXXH + D2PhiXYH)
        D2PhiYHH = lin.inp(Hy, D2PhiYXH + D2PhiYYH)

        # Operator perspective third order derivatives
        # Second derivatives of DxPhi
        D3PhiXXX = grad.thrd_frechet(self.Dyxy, self.D2yxy_h, self.d3h(self.Dyxy), self.irt2Y_Uyxy, self.UyxyYUyxy, UyxyYHxYUyxy)

        work      = self.Uyxy.conj().T @ self.rt2_Y @ Hy @ self.irt2Y_Uyxy
        D3PhiXXY  = grad.scnd_frechet(self.D2yxy_h, work + work.conj().T, UyxyYHxYUyxy, U=self.irt2Y_Uyxy)
        D3PhiXXY -= grad.thrd_frechet(self.Dyxy, self.D2yxy_xh, self.d3xh(self.Dyxy), self.irt2Y_Uyxy, self.UyxyYUyxy, UyxyYHxYUyxy, UyxyYHyYUyxy)
        D3PhiXYX  = D3PhiXXY

        D3PhiXYY  = self.irt2_X @ grad.scnd_frechet(self.D2xyx_g, UxyxXHyXUxyx, UxyxXHyXUxyx, U=self.Uxyx) @ self.rt2_X
        D3PhiXYY += D3PhiXYY.conj().T
        D3PhiXYY -= grad.thrd_frechet(self.Dxyx, self.D2xyx_xg, self.d3xg(self.Dxyx), self.irt2X_Uxyx, self.UxyxXUxyx, UxyxXHyXUxyx)

        # Second derivatives of DyPhi
        D3PhiYYY = grad.thrd_frechet(self.Dxyx, self.D2xyx_g, self.d3g(self.Dxyx), self.irt2X_Uxyx, self.UxyxXUxyx, UxyxXHyXUxyx)

        work      = self.Uxyx.conj().T @ self.rt2_X @ Hx @ self.irt2X_Uxyx
        D3PhiYYX  = grad.scnd_frechet(self.D2xyx_g, work + work.conj().T, UxyxXHyXUxyx, U=self.irt2X_Uxyx)
        D3PhiYYX -= grad.thrd_frechet(self.Dxyx, self.D2xyx_xg, self.d3xg(self.Dxyx), self.irt2X_Uxyx, self.UxyxXUxyx, UxyxXHyXUxyx, UxyxXHxXUxyx)
        D3PhiYXY  = D3PhiYYX
    
        D3PhiYXX  = self.irt2_Y @ grad.scnd_frechet(self.D2yxy_h, UyxyYHxYUyxy, UyxyYHxYUyxy, U=self.Uyxy) @ self.rt2_Y
        D3PhiYXX += D3PhiYXX.conj().T
        D3PhiYXX -= grad.thrd_frechet(self.Dyxy, self.D2yxy_xh, self.d3xh(self.Dyxy), self.irt2Y_Uyxy, self.UyxyYUyxy, UyxyYHxYUyxy)
        
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
    
    # ========================================================================
    # Auxilliary functions
    # ========================================================================
    def congr_aux(self, A):
        assert not self.congr_aux_updated

        self.At = A[:, 0]
        self.Ax_vec = np.ascontiguousarray(A[:, self.idx_X])
        self.Ay_vec = np.ascontiguousarray(A[:, self.idx_Y])

        self.Ax_compact = self.Ax_vec[:, self.triu_idxs]
        self.Ay_compact = self.Ay_vec[:, self.triu_idxs]

        self.Ax_compact *= self.scale
        self.Ay_compact *= self.scale
        self.A_compact = np.hstack((self.Ax_compact, self.Ay_compact))

        self.work = np.empty_like(self.A_compact.T)

        self.congr_aux_updated = True

    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        self.D1yxy_xh = grad.D1_f(self.Dyxy, self.xh(self.Dyxy), self.dxh(self.Dyxy))
        self.D1xyx_xg = grad.D1_f(self.Dxyx, self.xg(self.Dxyx), self.dxg(self.Dxyx))

        self.D2yxy_h  = grad.D2_f(self.Dyxy, self.D1yxy_h, self.d2h(self.Dyxy))
        self.D2xyx_g  = grad.D2_f(self.Dxyx, self.D1xyx_g, self.d2g(self.Dxyx))
        self.D2yxy_xh = grad.D2_f(self.Dyxy, self.D1yxy_xh, self.d2xh(self.Dyxy))
        self.D2xyx_xg = grad.D2_f(self.Dxyx, self.D1xyx_xg, self.d2xg(self.Dxyx))

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

        # Precompute and factorize the matrix
        #     M = 1/z [ D2xxPhi D2xyPhi ] + [ X^1 kron X^-1               ]
        #             [ D2yxPhi D2yyPhi ]   [               Y^1 kron Y^-1 ]        

        self.z2 = self.z * self.z

        # Make Hxx block
        # D2PhiXXH  = self.zi * grad.scnd_frechet(self.D2yxy_h, self.UyxyYUyxy, UyxyYHxYUyxy, U=self.irt2Y_Uyxy)
        lin.congr_multi(self.work8, self.irt2Y_Uyxy.conj().T, self.E, work=self.work7)
        grad.scnd_frechet_multi(self.work5, self.D2yxy_h * self.UyxyYUyxy, self.work8, U=self.irt2Y_Uyxy, work1=self.work6, work2=self.work7, work3=self.work4)
        self.work5 *= self.zi

        # D2PhiXXH += self.inv_X @ Hx @ self.inv_X
        lin.congr_multi(self.work8, self.inv_X, self.E, work=self.work7)
        self.work8 += self.work5

        Hxx  = self.work8.view(dtype=np.float64).reshape((self.vn, -1))[:, self.triu_idxs]
        Hxx *= self.scale

        # Make Hyy block
        # D2PhiYYH  = self.zi * grad.scnd_frechet(self.D2xyx_g, self.UxyxXUxyx, UxyxXHyXUxyx, U=self.irt2X_Uxyx)
        lin.congr_multi(self.work8, self.irt2X_Uxyx.conj().T, self.E, work=self.work7)
        grad.scnd_frechet_multi(self.work5, self.D2xyx_g * self.UxyxXUxyx, self.work8, U=self.irt2X_Uxyx, work1=self.work6, work2=self.work7, work3=self.work4)
        self.work5 *= self.zi

        # D2PhiYYH += self.inv_Y @ Hy @ self.inv_Y
        lin.congr_multi(self.work6, self.inv_Y, self.E, work=self.work7)
        self.work6 += self.work5

        Hyy  = self.work6.view(dtype=np.float64).reshape((self.vn, -1))[:, self.triu_idxs]
        Hyy *= self.scale

        # Make Hxy block
        # D2PhiXYH -= self.zi * grad.scnd_frechet(self.D2xyx_xg, self.UxyxXUxyx, UxyxXHyXUxyx, U=self.irt2X_Uxyx)
        grad.scnd_frechet_multi(self.work5, self.D2xyx_xg * self.UxyxXUxyx, self.work8, U=self.irt2X_Uxyx, work1=self.work6, work2=self.work7, work3=self.work4)

        # D2PhiXYH  = self.zi * self.irt2_X @ self.Uxyx @ (self.D1xyx_g * UxyxXHyXUxyx) @ self.Uxyx.conj().T @ self.rt2_X
        # D2PhiXYH += D2PhiXYH.conj().T
        self.work8 *= self.D1xyx_g
        lin.congr_multi(self.work6, self.irt2X_Uxyx, self.work8, work=self.work7, B=self.rt2_X @ self.Uxyx)
        np.add(self.work6, self.work6.conj().transpose(0, 2, 1), out=self.work7)
        
        self.work7 -= self.work5
        self.work7 *= self.zi

        Hyx  = self.work7.view(dtype=np.float64).reshape((self.vn, -1))[:, self.triu_idxs]
        Hyx *= self.scale

        # Construct Hessian and factorize
        Hxx = (Hxx + Hxx.conj().T) * 0.5
        Hyy = (Hyy + Hyy.conj().T) * 0.5
        self.hess[:self.vn, :self.vn] = Hxx
        self.hess[self.vn:, self.vn:] = Hyy
        self.hess[self.vn:, :self.vn] = Hyx
        self.hess[:self.vn, self.vn:] = Hyx.T

        self.hess_fact = lin.cho_fact(self.hess.copy())
        self.invhess_aux_updated = True

        return

    def update_invhessprod_aux_aux(self):
        assert not self.invhess_aux_aux_updated

        self.hess = np.empty((2*self.vn, 2*self.vn))

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

        self.dder3_aux_updated = True

        return
    
    def get_central_ray(self):
        # Solve a 3-dimensional system to get central point
        n = self.n
        (t, x, y) = (1. + n*self.g(1.), 1., 1.)

        for _ in range(10):
            # Precompute some useful things
            z   = t - n * x * self.g(y / x)
            zi  = 1 / z
            zi2 = zi * zi

            dx = self.dh(x / y)
            dy = self.dg(y / x)

            d2dx2  =  self.d2h(x / y) / y
            d2dy2  =  self.d2g(y / x) / x
            d2dxdy = -d2dy2 * y / x

            # Get gradient
            g = np.array([
                t - zi,
                n * x + n * dx * zi - n / x,
                n * y + n * dy * zi - n / y
            ])
            
            # Get Hessian
            H = np.array([
                [zi2,       -n*zi2*dx,                          -n*zi2*dy                         ],
                [-n*zi2*dx, n*n*zi2*dx*dx + n*zi*d2dx2 + n/x/x, n*n*zi2*dx*dy + n*zi*d2dxdy       ],
                [-n*zi2*dy, n*n*zi2*dx*dy + n*zi*d2dxdy,        n*n*zi2*dy*dy + n*zi*d2dy2 + n/y/y]
            ]) + np.diag([1, n, n])

            # Perform Newton step
            delta     = -np.linalg.solve(H, g)
            decrement = -np.dot(delta, g)

            # Check feasible
            (t1, x1, y1) = (t + delta[0], x + delta[1], y + delta[2])
            if x1 < 0 or y1 < 0 or t1 < n * x1 * self.g(y1 / x1):
                break
            
            (t, x, y) = (t1, x1, y1)

            if decrement / 2. <= 1e-12:
                break

        return (t, x, y)    