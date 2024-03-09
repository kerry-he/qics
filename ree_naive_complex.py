import numpy as np
import scipy as sp
import math

import cProfile

from cones import *
from utils import symmetric as sym
from solver import model, solver
from utils import quantum as quant

np.random.seed(1)
np.set_printoptions(threshold=np.inf)

# Problem data
n = 6
m = 6
N = n * m
vn = sym.vec_dim(n, hermitian=True)
vN = sym.vec_dim(N, hermitian=True)
X = quant.randDensityMatrix(N, hermitian=True)

# Build problem model
A = np.hstack((np.zeros((1, 1)), sym.mat_to_vec(np.eye(N), hermitian=True).T))
b = np.ones((1, 1))

c = np.zeros((1 + vN, 1))
c[0] = 1.0

p_transpose = sym.lin_to_mat(lambda x : sym.p_transpose(x, 1, (n, m)), n*m, n*m, hermitian=True)

G0 = np.hstack((np.ones((1, 1)), np.zeros((1, vN))))
G1 = np.hstack((np.zeros((vN, 1)), np.zeros((vN, vN))))
G2 = np.hstack((np.zeros((vN, 1)), np.eye(vN)))
G3 = np.hstack((np.zeros((vN, 1)), p_transpose))
G = -np.vstack((G0, G1, G2, G3))

h = np.zeros((1 + 3*vN, 1))
h[1:1+vN] = sym.mat_to_vec(X, hermitian=True)


# Input into model and solve
cones = [quantrelentr.Cone(N, hermitian=True), possemidefinite.Cone(N, hermitian=True)]
model = model.Model(c, A, b, G, h, cones=cones)
solver = solver.Solver(model,)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")