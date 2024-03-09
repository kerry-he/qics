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
n = 7
m = 7
N = n * m
vn = sym.vec_dim(n)
vN = sym.vec_dim(N)
X = quant.randDensityMatrix(N)

# Build problem model
A = np.hstack((np.zeros((1, 1)), sym.mat_to_vec(np.eye(N)).T))
b = np.ones((1, 1))

c = np.zeros((1 + vN, 1))
c[0] = 1.0

G = np.zeros((1 + vN*2, 1 + vN))
G[:vN+1, :] = -np.eye(vN + 1)
G1 = np.zeros((vN, 1 + vN))
for J in range(N):
    for I in range(J + 1):
        # Get product space indices
        i = I // m
        j = I - i * m

        k = J // m
        l = J - k * m

        I_pt = i * m + l
        J_pt = k * m + j

        if I_pt > J_pt:
            I_pt, J_pt = J_pt, I_pt

        k_in  = I + (J * (J + 1)) // 2
        k_out = I_pt + (J_pt * (J_pt + 1)) // 2

        G[vN + k_out, 1 + k_in] = -1.0

h = np.zeros((1 + 2*vN, 1))

# Input into model and solve
cones = [quantrelentr_Y.Cone(N, X, cg=True), possemidefinite.Cone(N)]
model = model.Model(c, A, b, G, h, cones=cones)
solver = solver.Solver(model, subsolver="elim")

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")