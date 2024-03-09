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
n = 128
sn = sym.vec_dim(n)

# cq channel capacity problem data
alphabet = np.array([quantum.randDensityMatrix(n) for i in range(n)])

# Build problem model
A = np.hstack((np.zeros((1, 1)), np.ones((1, n))))
b = np.ones((1, 1))

c = np.zeros((n + 1, 1))
c[0] = 1.

# Input into model and solve
cones = [holevoinf.HolevoInf(alphabet)]
model = model.Model(c, A, b, cones=cones)
solver = solver.Solver(model)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")