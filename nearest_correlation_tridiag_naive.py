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
M = quant.randDensityMatrix(n)
# M = np.random.rand(n, n)
# M = (M @ M.T)
# M = M / np.max(np.diag(M))

# Build problem model
A = np.zeros((0, n))
b = np.zeros((0, 1))

c = np.zeros((n, 1))
c[0] = 1.

G1 = np.hstack((np.ones((1, 1)), np.zeros((1, n - 1))))         # QRE t
G2 = np.hstack((np.zeros((vn, 1)), np.zeros((vn, n - 1))))      # QRE X
G3 = np.zeros((vn, n))                                          # QRE Y
for i in range(n - 1):
    H = np.zeros((n, n))
    H[i, i + 1] = np.sqrt(0.5)
    H[i + 1, i] = np.sqrt(0.5)
    G3[:, [1 + i]] = sym.mat_to_vec(H)
G = -np.vstack((G1, G2, G3))

h = np.zeros((1 + 2 * vn, 1))
h[1:vn+1] = sym.mat_to_vec(M)
h[vn+1:]  = sym.mat_to_vec(np.eye(n))

# Input into model and solve
cones = [quantrelentr.QuantRelEntropy(n)]
model = model.Model(c, A, b, G, h, cones=cones)
solver = solver.Solver(model)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")