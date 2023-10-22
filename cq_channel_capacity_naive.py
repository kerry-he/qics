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
alphabet = [quantum.randDensityMatrix(n) for i in range(n)]
alphabet_vec = np.hstack([sym.mat_to_vec(rho) for rho in alphabet])
entr_alphabet = np.array([quantum.quantEntropy(rho) for rho in alphabet])

# Build problem model
A = np.hstack((np.ones((1, n)), np.zeros((1, 1))))
b = np.ones((1, 1))

c = np.zeros((n + 1, 1))
c[:n, 0] = entr_alphabet
c[n] = 1.

G1 = np.hstack((np.eye(n), np.zeros((n, 1))))
G2 = np.hstack((np.zeros((1, n)), np.ones((1, 1))))
G3 = np.zeros((1, n + 1))
G4 = np.hstack((alphabet_vec, np.zeros((sn, 1))))
G = -np.vstack((G1, G2, G3, G4))

h = np.zeros((n + 1 + 1 + sn, 1))
h[n + 1] = 1.

# Input into model and solve
cones = [nonnegorthant.NonNegOrthant(n), quantentr.QuantEntropy(n)]
model = model.Model(c, A, b, G, h, cones=cones)
solver = solver.Solver(model, max_iter=78)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")