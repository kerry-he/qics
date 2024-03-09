import numpy as np
from utils import linear as lin

class Cone():
    def __init__(self, n):
        # Dimension properties
        self.n = n                          # Dimension of system
        self.dim = 1 + 2 * self.n           # Total dimension of cone
        self.use_sqrt = False

        self.idx_X = slice(1, 1 + self.n)
        self.idx_Y = slice(1 + self.n, 1 + 2 * self.n)        

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

        (t0, x0, y0) = get_central_ray_relentr(self.n)

        point[0] = t0
        point[self.idx_X] = x0
        point[self.idx_Y] = y0

        self.set_point(point)

        return point
    
    def set_point(self, point):
        assert np.size(point) == self.dim
        self.point = point

        self.t = point[0, 0]
        self.x = point[self.idx_X]
        self.y = point[self.idx_Y]

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

        if any(self.x <= 0) or any(self.y <= 0):
            self.feas = False
            return self.feas
        
        self.log_x = np.log(self.x)
        self.log_y = np.log(self.y)

        self.z = self.t - (self.x.T @ (self.log_x - self.log_y))

        self.feas = (self.z > 0)
        return self.feas
    
    def get_val(self):
        return -np.log(self.z) - np.sum(self.log_x) - np.sum(self.log_y)
    
    def get_grad(self):
        assert self.feas_updated

        if self.grad_updated:
            return self.grad
        
        self.zi    = np.reciprocal(self.z)
        self.DPhiX = self.log_x - self.log_y + 1
        self.DPhiY = -self.x / self.y

        self.grad             =  np.empty((self.dim, 1))
        self.grad[0]          = -self.zi
        self.grad[self.idx_X] =  self.zi * self.DPhiX - np.reciprocal(self.x)
        self.grad[self.idx_Y] =  self.zi * self.DPhiY - np.reciprocal(self.y)

        self.grad_updated = True
        return self.grad

    def hess_prod(self, dirs):
        assert self.grad_updated

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        for j in range(p):
            Ht = dirs[0, j]
            Hx = dirs[self.idx_X, [j]]
            Hy = dirs[self.idx_Y, [j]]

            chi  = self.zi * self.zi * (Ht - Hx.T @ self.DPhiX - Hy.T @ self.DPhiY)
            D2PhiXH =  Hx / self.x - Hy / self.y
            D2PhiYH = -Hx / self.y + Hy * self.x / self.y / self.y

            # Hessian product of barrier function
            out[0, j]            =  chi
            out[self.idx_X, [j]] = -chi * self.DPhiX + self.zi * D2PhiXH + Hx / (self.x * self.x)
            out[self.idx_Y, [j]] = -chi * self.DPhiY + self.zi * D2PhiYH + Hy / (self.y * self.y)

        return out
    
    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated

        self.Hxx_inv = np.reciprocal((1 / self.x + self.zi) / self.x)
        self.HxyHyy = -self.zi * self.Hxx_inv / self.y
        self.Hyy_HxyHxxHxy_inv = self.y * self.y / (1 + self.zi * self.x - self.zi * self.zi * self.Hxx_inv)

        return

    def invhess_prod(self, dirs):
        assert self.grad_updated
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        for j in range(p):
            Ht = dirs[0, j]
            Hx = dirs[self.idx_X, [j]]
            Hy = dirs[self.idx_Y, [j]]

            Wx = Hx + Ht * self.DPhiX
            Wy = Hy + Ht * self.DPhiY

            outY = self.Hyy_HxyHxxHxy_inv * (Wy - self.HxyHyy * Wx)
            outX = self.Hxx_inv * (Wx + self.zi * outY / self.y)

            out[0, j] = Ht * self.z * self.z + outX.T @ self.DPhiX + outY.T @ self.DPhiY
            out[self.idx_X, [j]] = outX
            out[self.idx_Y, [j]] = outY

        return out

    def third_dir_deriv(self, dirs):
        assert self.grad_updated

        Ht = dirs[0, 0]
        Hx = dirs[self.idx_X, [0]]
        Hy = dirs[self.idx_Y, [0]]
        
        sum_H = np.sum(Hx)

        # Quantum conditional entropy oracles
        D2PhiXH =  Hx / self.x - Hy / self.y
        D2PhiYH = -Hx / self.y + Hy * self.x / self.y / self.y

        D2PhiXHH = Hx.T @ D2PhiXH
        D2PhiYHH = Hy.T @ D2PhiYH

        D3PhiXHH = -(Hx / self.x) ** 2 + (Hy / self.y) ** 2
        D3PhiYHH = 2 * Hy * (Hx - Hy * self.x / self.y) / self.y / self.y

        # Third derivatives of barrier
        chi = Ht - lin.inp(self.DPhiX, Hx) - lin.inp(self.DPhiY, Hy)
        chi2 = chi * chi

        dder3_t = -2 * (self.zi**3) * chi2 - (self.zi**2) * (D2PhiXHH + D2PhiYHH)

        dder3_X  = -dder3_t * self.DPhiX
        dder3_X -=  2 * (self.zi**2) * chi * D2PhiXH
        dder3_X +=  self.zi * D3PhiXHH
        dder3_X -=  2 * (Hx**2) / (self.x**3)

        dder3_Y  = -dder3_t * self.DPhiY
        dder3_Y -=  2 * (self.zi**2) * chi * D2PhiYH
        dder3_Y +=  self.zi * D3PhiYHH
        dder3_Y -=  2 * (Hy**2) / (self.y**3)

        dder3             = np.empty((self.dim, 1))
        dder3[0]          = dder3_t
        dder3[self.idx_X] = dder3_X
        dder3[self.idx_Y] = dder3_Y

        return dder3

    def norm_invhess(self, x):
        return 0.0
    
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