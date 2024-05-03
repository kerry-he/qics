import numpy as np
import scipy as sp

from cones import *
from utils import symmetric as sym
from solver import model, solver

np.random.seed(1)

A = np.array([[1.,  0.,  1.,  1.], [-1.,  1.,  0., -1.]])
b = np.array([[1.], [1.]])
c = np.array([[1.], [2.], [0.], [-1.]])

cones = [nonnegorthant.Cone(4)]
model = model.Model(c, A, b, cones=cones)
solver = solver.Solver(model, sym=True)

solver.solve()