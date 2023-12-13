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
VV = np.zeros((sout_env, sin))
trE_VV = np.zeros((sout, sin))
k = -1
for j in range(nin):
    for i in range(j + 1):
        k += 1
    
        H = np.zeros((nin, nin))
        if i == j:
            H[i, j] = 1
        else:
            H[i, j] = H[j, i] = math.sqrt(0.5)
        
        VHV = V @ H @ V.T
        VV[:, [k]] = sym.mat_to_vec(VHV)

        trE_VHV = sym.p_tr(VHV, 1, (nout, nenv))
        trE_VV[:, [k]] = sym.mat_to_vec(trE_VHV)

# Build problem model
A = np.hstack((np.zeros((1, 2)), tr))
b = np.ones((1, 1))

c = np.zeros((2 + sin, 1))
c[0:2] = 1.

G1 = np.hstack((np.ones((1, 1)), np.zeros((1, 1 + sin))))                   # t_cond
G2 = np.hstack((np.zeros((sout_env, 2)), VV))                               # X_cond
G3 = np.hstack((np.zeros((1, 1)), np.ones((1, 1)), np.zeros((1, sin))))     # t_entr
G4 = np.hstack((np.zeros((sout, 2)), trE_VV))                               # X_entr
G5 = np.hstack((np.zeros((sin, 2)), np.eye(sin)))                           # PSD
G = -np.vstack((G1, G2, G3, G4, G5))

h = np.zeros((1 + sout_env + 1 + sout + sin, 1))

# Input into model and solve
cones = [quantcondentr.QuantCondEntropy(nout, nenv, 0), quantentr.QuantEntropy(nout), possemidefinite.PosSemiDefinite(nin)]
model = model.Model(c, A, b, G, h, cones=cones)
solver = solver.Solver(model, max_iter=30)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")