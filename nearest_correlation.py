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
n = 100
vn = sym.vec_dim(n)
# M = 2 * np.eye(n)
# M = quant.randDensityMatrix(n)
M = np.random.rand(n, n)
M = (M @ M.T)
M = M / np.max(np.diag(M))

# Build problem model
A = np.zeros((n, 1 + vn))
for i in range(n):
    H = np.zeros((n, n))
    H[i, i] = 1.0
    A[[i], 1:] = sym.mat_to_vec(H).T
b = np.ones((n, 1))

c = np.zeros((1 + vn, 1))
c[0] = 1.0 / n

# Input into model and solve
cones = [quantrelentr_Y.QuantRelEntropyY(n, M)]
model = model.Model(c, A, b, cones=cones, offset=-np.trace(M) * np.log(n))
solver = solver.Solver(model)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")