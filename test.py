import numpy as np
import scipy as sp

from cones import *
from utils import symmetric as sym
from solver import model, solver

np.random.seed(1)

A = np.array([[1.,  0.,  1.,  1.], [-1.,  1.,  0., -1.]])
b = np.array([[1.], [1.]])
c = np.array([[1.], [2.], [0.], [-1.]])

cones = [nonnegorthant.NonNegOrthant(4)]
model = model.Model(c, A, b, cones)
solver = solver.Solver(model, max_iter=20)

solver.solve()

print(solver.point.x)