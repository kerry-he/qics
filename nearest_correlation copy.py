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
n = 60
vn = sym.vec_dim(n)
# M = 2 * np.eye(n)
M = quant.randDensityMatrix(n)
# M = np.random.rand(n, n)
# M = (M @ M.T)
# M = M / np.max(np.diag(M))
# M = np.random.rand(n, 3)
# M = M @ M.T

# Build problem model
A = np.hstack((np.zeros((vn, 1)), np.eye(vn)))
b = sym.mat_to_vec(np.eye(n))

c = np.zeros((1 + vn, 1))
c[0] = 1.0

# Input into model and solve
cones = [quantrelentr_Y.QuantRelEntropyY(n, M, cg=True)]
model = model.Model(c, A, b, cones=cones, offset=-np.trace(M) * np.log(n))
solver = solver.Solver(model, subsolver="qrchol")

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")