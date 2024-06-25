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
    eye = np.zeros((n*n, sn)) if (not hermitian) else np.zeros((2*n*n, sn))
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
                if hermitian:
                    H[i, j] = 1j * math.sqrt(0.5)
                    H[j, i] = -1j * math.sqrt(0.5)
                    eye[:, k] = H.view(dtype=np.float64).reshape(-1)
                    k += 1
    return eye

# Problem data
hermitian = False
dtype = np.float64 if (not hermitian) else np.complex128

n = 50
vn = sym.vec_dim(n, hermitian=hermitian)
sn = n*n if (not hermitian) else 2*n*n
M = quant.randDensityMatrix(n, hermitian=hermitian)


eye = get_eye(n, vn, hermitian)


# Build problem model
A = np.zeros((n, 1 + vn))
for i in range(n):
    H = np.zeros((n, n), dtype=dtype)
    H[i, i] = 1.0
    A[[i], 1:] = sym.mat_to_vec(H, hermitian=hermitian).T
b = np.ones((n, 1))

c = np.zeros((1 + vn, 1))
c[0] = 1.

G1 = np.hstack((np.ones((1, 1)), np.zeros((1, vn))))
G2 = np.hstack((np.zeros((sn, 1)), np.zeros((sn, vn))))
G3 = np.hstack((np.zeros((sn, 1)), eye))
G = -np.vstack((G1, G2, G3))

h = np.zeros((1 + 2 * sn, 1))
h[1:sn+1] = M.view(dtype=np.float64).reshape(-1, 1)

# Input into model and solve
cones = [quantrelentr.Cone(n, hermitian=hermitian)]
model = model.Model(c, A, b, G, h, cones=cones)
solver = solver.Solver(model)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")