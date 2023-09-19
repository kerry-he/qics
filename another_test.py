import numpy as np
import scipy as sp

from cones import *
from utils import symmetric as sym
from utils import point
from solver import model, syssolver

# np.random.seed(1)

A = np.array([[1.,  0.,  1.], [-1.,  1.,  0.]])
b = np.array([[1.], [1.]])
c = np.array([[1.], [2.], [0.]])

cones = [nonnegorthant.NonNegOrthant(3)]
model = model.Model(c, A, b, cones)

model.cones[0].set_point(np.random.rand(3, 1))

syssolver = syssolver.SysSolver(model)

rhs = point.Point(model)
rhs.vec[:] = np.random.rand(3 + 2 + 3, 1)

syssolver.update_lhs(model)
sol = syssolver.solve_system(rhs, model)
res = syssolver.apply_system(sol, model)

print(res.vec - rhs.vec)
