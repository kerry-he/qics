import numpy as np
import scipy as sp
import math

import cProfile

from cones import *
from utils import symmetric as sym, quantum as quant
from solver import model, solver

np.random.seed(1)
np.set_printoptions(threshold=np.inf)

# Problem data
n = 100
vn = sym.vec_dim(n)
# M = 2 * np.eye(n)
# M = quant.randDensityMatrix(n)
M = np.random.rand(n, 10)
M = (M @ M.T)
M = M / np.max(np.diag(M))

# Build problem model
A = np.zeros((0, n))
b = np.zeros((0, 1))

c = np.zeros((n, 1))
c[0] = 1.0 / n

G1 = np.hstack((np.ones((1, 1)), np.zeros((1, n - 1))))         # QRE t
G2 = np.zeros((vn, n))                                          # QRE Y
for i in range(n - 1):
    H = np.zeros((n, n))
    H[i, i + 1] = np.sqrt(0.5)
    H[i + 1, i] = np.sqrt(0.5)
    G2[:, [1 + i]] = sym.mat_to_vec(H)
G = -np.vstack((G1, G2))

h = np.zeros((1 + vn, 1))
h[1:]  = sym.mat_to_vec(np.eye(n))

# Input into model and solve
cones = [quantrelentr_Y.Cone(n, M)]
model = model.Model(c, A, b, G, h, cones=cones, offset=-np.trace(M) * np.log(n))
solver = solver.Solver(model)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")