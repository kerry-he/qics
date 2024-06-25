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

def get_eye(n, sn, hermitian):
    dtype = np.float64 if (not hermitian) else np.complex128
    eye = np.zeros((sn, n*n))
    k = 0
    for j in range(n):
        for i in range(j):
            H = np.zeros((n, n), dtype=dtype)
            H[i, j] = H[j, i] = np.sqrt(0.5)
            eye[k] = H.view(dtype=np.float64).reshape(-1)
            k += 1
            if hermitian:
                H[i, j] =  np.sqrt(0.5) * 1j
                H[j, i] = -np.sqrt(0.5) * 1j
                eye[k] = H.view(dtype=np.float64).reshape(-1)
                k += 1
        H = np.zeros((n, n), dtype=dtype)
        H[j, j] = 1.
        eye[k] = H.view(dtype=np.float64).reshape(-1)
        k += 1
            
    return eye


# Problem data
hermitian = False
dtype = np.float64 if (not hermitian) else np.complex128

n = 50
vn = n*n if (not hermitian) else 2*n*n
sn = sym.vec_dim(n, hermitian=hermitian)
M = quant.randDensityMatrix(n, hermitian=hermitian)

# Build problem model
Adiag = np.zeros((n, vn))
for i in range(n):
    H = np.zeros((n, n), dtype=dtype)
    H[i, i] = 1.0
    Adiag[i] = H.view(dtype=np.float64).reshape(-1)

eye = get_eye(n, sn, hermitian=hermitian)

A1 = np.hstack((np.zeros((sn, 1)), eye, np.zeros((sn, vn))))
A2 = np.hstack((np.zeros((n, 1)), np.zeros((n, vn)), Adiag))
A  = np.vstack((A1, A2))

b = np.ones((sn + n, 1))
b[:sn] = sym.mat_to_vec(M, hermitian=hermitian)

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