import numpy as np
import scipy as sp
import math

import cProfile

from cones import *
from utils import symmetric as sym
from utils import quantum
from solver import model, solver

np.random.seed(1)
np.set_printoptions(threshold=np.inf)

# Define dimensions
n = 4
sn = sym.vec_dim(n)

# cq channel capacity problem data
alphabet = [quantum.randDensityMatrix(n) for i in range(n)]
alphabet_vec = np.hstack([sym.mat_to_vec(rho) for rho in alphabet])
entr_alphabet = np.array([quantum.quantEntropy(rho) for rho in alphabet])

# Build problem model
A1 = np.hstack((alphabet_vec, np.zeros((sn, 2)), -np.eye(sn)))
A2 = np.hstack((np.ones((1, n)), np.zeros((1, 2 + sn))))
A3 = np.hstack((np.zeros((1, n)), np.zeros((1, 1)), np.ones((1, 1)), np.zeros((1, sn))))
A  = np.vstack((A1, A2, A3))

b = np.zeros((sn + 1 + 1, 1))
b[-2] = 1.
b[-1] = 1.

c = np.zeros((n + 2 + sn, 1))
c[:n, 0] = entr_alphabet
c[n] = 1.

# Input into model and solve
cones = [nonnegorthant.NonNegOrthant(n), quantentr.QuantEntropy(n)]
model = model.Model(c, A, b, cones)
solver = solver.Solver(model)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")