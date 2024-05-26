import numpy as np
import math
from utils.vector import Point
from utils import linear as lin
from utils import symmetric as sym
from cones import *


class CVXOPTStepper():
    def __init__(self, syssolver, model):
        self.syssolver = syssolver
        self.prox = 0.0
        
        self.rhs        = Point(model)
        self.dir_a      = Point(model)
        self.dir_comb   = Point(model)
        self.temp       = Point(model)
        self.next_point = Point(model)
        
        return
    
    def step(self, model, point, xyztau_res, mu, verbose):
        self.syssolver.update_lhs(model, point, mu)

        # Step 2: Get affine direction
        self.update_rhs_affine(model, point)
        res_norm = self.syssolver.solve_sys(self.dir_a, self.rhs)

        # Step 3: Step size and centering parameter (use bisection)
        alpha_u = 1.0
        alpha_l = 0.0
        alpha = 0.5
        next_point = self.next_point
        for k in range(10):
            # Take step
            next_point.vec[:] = point.vec + alpha * self.dir_a.vec

            # Check if new point is feasible
            is_feas = True
            is_feas = is_feas and (next_point.tau > 0 and next_point.kap > 0)
            for (k, cone_k) in enumerate(model.cones): 
                if isinstance(cone_k, possemidefinite.Cone):
                    S_eig = np.linalg.eigvalsh(next_point.s[k])
                    Z_eig = np.linalg.eigvalsh(next_point.z[k])
                    is_feas = is_feas and (all(S_eig > 0) and all(Z_eig > 0))
                else:
                    is_feas = is_feas and (all(next_point.s[k] > 0) and all(next_point.z[k] > 0))

            if not is_feas:
                alpha_u = alpha
            else:
                alpha_l = alpha
            
            alpha = 0.5 * (alpha_u + alpha_l)

        alpha = alpha_l
        sigma = (1 - alpha) ** 3

        # Step 4: Combined direction
        self.update_rhs_comb(model, point, mu, self.dir_a, sigma)
        temp_res_norm = self.syssolver.solve_sys(self.dir_comb, self.rhs)
        res_norm = max(temp_res_norm, res_norm)

        # Step 5: Line search
        alpha_u = 1.0
        alpha_l = 0.0
        alpha = 0.5
        for k in range(10):
            # Take step
            next_point.vec[:] = point.vec + alpha * self.dir_comb.vec

            # Check if new point is feasible
            is_feas = True
            is_feas = is_feas and (next_point.tau > 0 and next_point.kap > 0)
            for (k, cone_k) in enumerate(model.cones): 
                if isinstance(cone_k, possemidefinite.Cone):
                    S_eig = np.linalg.eigvalsh(next_point.s[k])
                    Z_eig = np.linalg.eigvalsh(next_point.z[k])
                    is_feas = is_feas and (all(S_eig > 0) and all(Z_eig > 0))
                else:
                    is_feas = is_feas and (all(next_point.s[k] > 0) and all(next_point.z[k] > 0))

            if not is_feas:
                alpha_u = alpha
            else:
                alpha_l = alpha
            
            alpha = 0.5 * (alpha_u + alpha_l)
        
        alpha = alpha_l * 0.99
        success = (alpha > 0.0) # All steps were failures

        point.vec += alpha * self.dir_comb.vec
        for (k, cone_k) in enumerate(model.cones):
            cone_k.set_point(point.s[k], point.z[k])
            assert cone_k.get_feas()
        
        if verbose:
            if success:
                print("  | %6.4f" % sigma, "%10.3e" % (res_norm), "%10.3e" % (self.prox), " %5.3f" % (alpha))
            else:
                return point, False
        
        return point, True
    
    def update_rhs_affine(self, model, point):
        self.rhs.x[:] = -model.A.T @ point.y - model.G.T @ point.z.vec - model.c * point.tau
        self.rhs.y[:] = model.A @ point.x - model.b * point.tau
        self.rhs.z.vec[:] = (model.G @ point.x - model.h * point.tau)
        self.rhs.z.vec += point.s.vec

        self.rhs.s.vec[:] = -1 * point.z.vec

        self.rhs.tau[:] = lin.inp(model.c, point.x) + lin.inp(model.b, point.y) + lin.inp(model.h, point.z.vec) + point.kap
        self.rhs.kap[:] = -point.kap

        return self.rhs

    def update_rhs_comb(self, model, point, mu, dir_a, sigma):
        self.rhs.x[:] = (-model.A.T @ point.y - model.G.T @ point.z.vec - model.c * point.tau) * (1 - sigma)
        self.rhs.y[:] = (model.A @ point.x - model.b * point.tau) * (1 - sigma)
        self.rhs.z.vec[:] = ((model.G @ point.x - model.h * point.tau) * (1 - sigma))
        self.rhs.z.vec += point.s.vec * (1 - sigma)

        for (k, cone_k) in enumerate(model.cones):
            self.rhs.s[k][:] = cone_k.comb_dir(dir_a.s[k], dir_a.z[k], sigma, mu)
            # self.rhs.s[k] = -1 * point.z[k] - sigma*mu*cone_k.get_grad()

        self.rhs.tau[:] = (lin.inp(model.c, point.x) + lin.inp(model.b, point.y) + lin.inp(model.h, point.z.vec) + point.kap) * (1 - sigma)
        self.rhs.kap[:] = -point.kap + (-dir_a.kap*dir_a.tau + sigma*mu) / point.tau

        return self.rhs