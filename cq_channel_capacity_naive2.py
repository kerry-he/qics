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

def get_eye(n, sn, iscomplex):
    dtype = np.float64 if (not iscomplex) else np.complex128
    eye = np.zeros((n*n, sn)) if (not iscomplex) else np.zeros((2*n*n, sn))
    k = 0
    for j in range(n):
        for i in range(j + 1):    
            H = np.zeros((n, n), dtype=dtype)
            if i == j:
                H[i, j] = 1
                eye[:, k] = H.view(dtype=np.float64).reshape(-1)
                k += 1
            else:
                H[i, j] = H[j, i] = math.sqrt(0.5)
                eye[:, k] = H.view(dtype=np.float64).reshape(-1)
                k += 1
                if iscomplex:
                    H[i, j] = 1j * math.sqrt(0.5)
                    H[j, i] = -1j * math.sqrt(0.5)
                    eye[:, k] = H.view(dtype=np.float64).reshape(-1)
                    k += 1
    return eye

# Define dimensions
iscomplex = True
n = 64
vn = sym.vec_dim(n, iscomplex=iscomplex)
sn = n*n if (not iscomplex) else 2*n*n

# cq channel capacity problem data
alphabet = [quantum.randDensityMatrix(n, iscomplex=iscomplex) for i in range(n)]
alphabet_vec = np.hstack([sym.mat_to_vec(rho, iscomplex=iscomplex) for rho in alphabet])
entr_alphabet = np.array([quantum.quantEntropy(rho) for rho in alphabet])

eye = get_eye(n, vn, iscomplex)

# Build problem model
A1 = np.hstack((np.ones((1, n)), np.zeros((1, 2)), np.zeros((1, sn))))         # SUM pi = 1
A2 = np.hstack((np.zeros((1, n + 1)), np.ones((1, 1)), np.zeros((1, sn))))
A3 = np.hstack((alphabet_vec, np.zeros((vn, 2)), -eye.T))
A = np.vstack((A1, A2, A3))

b = np.zeros((2 + vn, 1))
b[0:2] = 1

c = np.zeros((n + 2 + sn, 1))
c[:n, 0] = entr_alphabet
c[n] = 1.

# Input into model and solve
cones = [nonnegorthant.Cone(n), quantentr.Cone(n, iscomplex=iscomplex)]
model = model.Model(c, A, b, cones=cones)
solver = solver.Solver(model)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")