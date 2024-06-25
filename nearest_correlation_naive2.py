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
hermitian = False
dtype = np.float64 if (not hermitian) else np.complex128

n = 10
vn = n*n if (not hermitian) else 2*n*n
M = quant.randDensityMatrix(n, hermitian=hermitian)

# Build problem model
Adiag = np.zeros((n, vn))
for i in range(n):
    H = np.zeros((n, n), dtype=dtype)
    H[i, i] = 1.0
    Adiag[i] = H.view(dtype=np.float64).reshape(-1)

A1 = np.hstack((np.zeros((vn, 1)), np.eye(vn), np.zeros((vn, vn))))
A2 = np.hstack((np.zeros((n, 1)), np.zeros((n, vn)), Adiag))
A  = np.vstack((A1, A2))

b = np.ones((vn + n, 1))
b[:vn] = M.view(dtype=np.float64).reshape(-1, 1)

c = np.zeros((1 + 2*vn, 1))
c[0] = 1.

# Input into model and solve
cones = [quantrelentr.Cone(n, hermitian=hermitian)]
model = model.Model(c, A, b, cones=cones)
solver = solver.Solver(model)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")