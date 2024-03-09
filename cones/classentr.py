import numpy as np
from utils import linear as lin

class Cone():
    def __init__(self, n):
        # Dimension properties
        self.n = n                          # Dimension of system
        self.dim = 1 + self.n               # Total dimension of cone
        self.use_sqrt = False

        # Update flags
        self.feas_updated        = False
        self.grad_updated        = False
        self.hess_aux_updated    = False

        return
        
    def get_nu(self):
        return 1 + self.n
    
    def set_init_point(self):
        point = np.empty((self.dim, 1))

        (t0, x0) = get_central_ray_entr(self.n)

        point[0]  = t0
        point[1:] = x0

        self.set_point(point)

        return point
    
    def set_point(self, point):
        assert np.size(point) == self.dim
        self.point = point

        self.t   = point[0, 0]
        self.x   = point[1:]
        self.sum_x = np.sum(self.x)

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

        if any(self.x <= 0):
            self.feas = False
            return self.feas
        
        self.log_x = np.log(self.x)
        self.log_sum_x  = np.log(self.sum_x)

        entr_x   = self.x.T @ self.log_x
        entr_sum_x = self.sum_x * self.log_sum_x

        self.z = self.t - (entr_x - entr_sum_x)

        self.feas = (self.z > 0)
        return self.feas
    
    def get_grad(self):
        assert self.feas_updated

        if self.grad_updated:
            return self.grad
        
        self.zi   = np.reciprocal(self.z)
        self.DPhi = self.log_x - self.log_sum_x

        self.grad     =  np.empty((self.dim, 1))
        self.grad[0]  = -self.zi
        self.grad[1:] =  self.zi * self.DPhi - np.reciprocal(self.x)

        self.grad_updated = True
        return self.grad

    def hess_prod(self, dirs):
        assert self.grad_updated

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        for j in range(p):
            Ht = dirs[0, j]
            Hx = dirs[1:, [j]]

            sum_Hx = np.sum(Hx)
            chi  = self.zi * self.zi * (Ht - Hx.T @ self.DPhi)
            D2PhiH = Hx / self.x - sum_Hx / self.sum_x

            # Hessian product of barrier function
            out[0, j]    =  chi
            out[1:, [j]] = -chi * self.DPhi + self.zi * D2PhiH + Hx / (self.x * self.x)

        return out
    
    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated

        self.H_inv_1 = np.reciprocal(self.zi / self.x + 1 / self.x / self.x)

        return

    def invhess_prod(self, dirs):
        assert self.grad_updated
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        for j in range(p):
            Ht = dirs[0, j]
            Hx = dirs[1:, [j]]

            Wx = Hx + Ht * self.DPhi

            fac = (self.H_inv_1.T @ Wx) / (self.z * self.sum_x - np.sum(self.H_inv_1))
            outX = self.H_inv_1 * Wx + fac * self.H_inv_1

            out[0, j] = Ht * self.z * self.z + lin.inp(outX, self.DPhi)
            out[1:, [j]] = outX

        return out

    def third_dir_deriv(self, dirs):
        assert self.grad_updated

        Ht = dirs[0]
        Hx = dirs[1:, [0]]
        
        sum_H = np.sum(Hx)

        # Quantum conditional entropy oracles
        D2PhiH = Hx / self.x - sum_H / self.sum_x

        D3PhiHH = - (Hx / self.x) ** 2 + (sum_H / self.sum_x) ** 2

        # Third derivative of barrier
        DPhiH = self.DPhi.T @ Hx
        D2PhiHH = D2PhiH.T @ Hx
        chi = Ht - DPhiH

        dder3 = np.empty((self.dim, 1))
        dder3[0] = -2 * (self.zi**3) * (chi**2) - (self.zi**2) * D2PhiHH

        dder3[1:] = -dder3[0] * self.DPhi
        dder3[1:] -= 2 * (self.zi**2) * chi * D2PhiH
        dder3[1:] += self.zi * D3PhiHH
        dder3[1:] -= 2 * (Hx**2) / (self.x**3)

        return dder3

    def norm_invhess(self, x):
        return 0.0
    
def get_central_ray_entr(x_dim):
    if x_dim <= 10:
        return central_rays_entr[x_dim - 1, :]
    
    # use nonlinear fit for higher dimensions
    t0 = np.power(2.1031 * x_dim - 5.3555, -0.865)
    x0 = 1 + np.reciprocal(1.9998 * x_dim + 0.626)

    return np.array([t0, x0])

central_rays_entr = np.array([
    [1.000000000000000, 1.00000000000000],
    [0.474515353116114, 1.17788691016068],
    [0.248466010640737, 1.14575369296145],
    [0.157680362921038, 1.11525072080430],
    [0.112130291129889, 1.09428983209783],
    [0.085532730884352, 1.07955514941305],
    [0.068371512284588, 1.06873258290068],
    [0.056503495284181, 1.06047202668235],
    [0.047869373521682, 1.05397177503172],
    [0.041340796367988, 1.04872736869751]
])