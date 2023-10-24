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
nin  = 16
nout = 16
nenv = 16

sin = sym.vec_dim(nin)
sout = sym.vec_dim(nout)
sout_env = sym.vec_dim(nout * nenv)

# ea channel capacity problem data
V = quantum.randStinespringOperator(nin, nout, nenv)
tr = sym.mat_to_vec(np.eye(nin)).T

# Build problem model
A = np.hstack((np.zeros((1, 1)), tr))
b = np.ones((1, 1))

c = np.zeros((1 + sin, 1))
c[0] = 1.

# Input into model and solve
cones = [quantmutualinf.QuantMutualInf(V, nout)]
model = model.Model(c, A, b, cones=cones)
solver = solver.Solver(model)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")