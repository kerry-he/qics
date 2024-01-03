import numpy as np
import scipy as sp

from cones import *
from utils import symmetric as sym
from solver import model, solver

np.random.seed(1)

A1 = sym.mat_to_vec(np.array([[1., 0., 1.], [0., 3., 7.], [1., 7., 5.]]))
A2 = sym.mat_to_vec(np.array([[0., 2., 8.], [2., 6., 0.], [8., 0., 4.]]))
A = np.vstack((A1.T, A2.T))
b = np.array([[11.], [19.]])
c = sym.mat_to_vec(np.array([[1., 2., 3.], [2., 9., 0.], [3., 0., 7.]]))

cones = [possemidefinite.PosSemiDefinite(3)]
model = model.Model(c, A, b, cones=cones)
solver = solver.Solver(model, subsolver="qrchol")

solver.solve()