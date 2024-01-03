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
n = 50
vn = sym.vec_dim(n)
# M = 2 * np.eye(n)
M = quant.randDensityMatrix(n)

# Build problem model
A = np.zeros((n, 1 + vn))
for i in range(n):
    H = np.zeros((n, n))
    H[i, i] = 1.0
    A[[i], 1:] = sym.mat_to_vec(H).T
b = np.ones((n, 1))

c = np.zeros((1 + vn, 1))
c[0] = 1.

G1 = np.hstack((np.ones((1, 1)), np.zeros((1, vn))))
G2 = np.hstack((np.zeros((vn, 1)), np.zeros((vn, vn))))
G3 = np.hstack((np.zeros((vn, 1)), np.eye(vn)))
G = -np.vstack((G1, G2, G3))

h = np.zeros((1 + 2 * vn, 1))
h[1:vn+1] = sym.mat_to_vec(M)

# Input into model and solve
cones = [quantrelentr.QuantRelEntropy(n)]
model = model.Model(c, A, b, G, h, cones=cones)
solver = solver.Solver(model)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")