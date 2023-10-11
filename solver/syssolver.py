import numpy as np
import scipy as sp
from utils import point

# Solves the following square Newton system
#               -A' * y + z = rx
#        A * x              = ry
#     mu*H * x          + z = rz 
# for (x, y, z) given right-hand residuals (rx, ry, rz)
# by using elimination.

class SysSolver():
    def __init__(self, model):
        self.sol = point.Point(model)
        return
    
    def update_lhs(self, model):
        # Precompute necessary objects on LHS of Newton system
        HA = blk_invhess_prod(model.A.T, model)
        AHA = model.A @ HA
        self.AHA_lu = sp.linalg.cho_factor(AHA)

        return
    
    def solve_system(self, rhs, model):
        # Solve Newton system using elimination
        # NOTE: mu has already been accounted for in H

        temp = model.A @ blk_invhess_prod(rhs.z - rhs.x, model) - rhs.y
        self.sol.y[:] = sp.linalg.cho_solve(self.AHA_lu, temp)
        self.sol.x[:] = blk_invhess_prod(rhs.z - rhs.x - model.A.T @ self.sol.y, model)
        self.sol.z[:] = rhs.x + model.A.T @ self.sol.y

        # res = self.apply_system(self.sol, model)
        # res.vec[:] -= rhs.vec
        # print(np.linalg.norm(res.vec))

        return self.sol
    
    def apply_system(self, rhs, model):
        pnt = point.Point(model)

        pnt.x[:] = -model.A.T @ rhs.y + rhs.z
        pnt.y[:] = model.A @ rhs.x
        pnt.z[:] = blk_hess_prod(rhs.x, model) + rhs.z

        return pnt

def blk_hess_prod(dirs, model):
    out = np.empty_like(dirs)

    for (cone_k, cone_idxs_k) in zip(model.cones, model.cone_idxs):
        out[cone_idxs_k, :] = cone_k.hess_prod(dirs[cone_idxs_k, :])

    return out

def blk_invhess_prod(dirs, model):
    out = np.empty_like(dirs)

    for (cone_k, cone_idxs_k) in zip(model.cones, model.cone_idxs):
        out[cone_idxs_k, :] = cone_k.invhess_prod(dirs[cone_idxs_k, :])

    return out