import numpy as np
import scipy as sp
import math

import cProfile

from cones import *
from utils import symmetric as sym
from utils import quantum as qu
from solver import model, solver

def purify(eig):
    n = np.size(eig)

    vec = np.zeros((n*n, 1))
    for (i, ii) in enumerate(range(0, n*n, n + 1)):
        vec[ii] = math.sqrt(eig[i])

    return vec @ vec.T

def get_tr2(n, sn, vN, iscomplex):
    dtype = np.float64 if (not iscomplex) else np.complex128
    tr2 = np.zeros((sn, vN))
    k = 0
    for j in range(n):
        for i in range(j + 1):
            H = np.zeros((n, n), dtype=dtype)
            if i == j:
                H[i, j] = 1
                I_H = sym.i_kr(H, 1, (n, n))
                tr2[k, :] = I_H.view(dtype=np.float64).reshape(1, -1)         
                k += 1       
            else:
                H[i, j] = H[j, i] = math.sqrt(0.5)
                I_H = sym.i_kr(H, 1, (n, n))
                tr2[k, :] = I_H.view(dtype=np.float64).reshape(1, -1)
                k += 1
                
                if iscomplex:
                    H[i, j] = math.sqrt(0.5) * 1j
                    H[j, i] = -math.sqrt(0.5) * 1j
                    I_H = sym.i_kr(H, 1, (n, n))
                    tr2[k, :] = I_H.view(dtype=np.float64).reshape(1, -1)
                    k += 1 

    return tr2


np.random.seed(1)
np.set_printoptions(threshold=np.inf)

# Define dimensions
iscomplex = True

n = 16
N = n * n
sn = sym.vec_dim(n, iscomplex=iscomplex)
vn = n*n if (not iscomplex) else 2*n*n
sN = sym.vec_dim(N, iscomplex=iscomplex)
vN = N*N if (not iscomplex) else 2*N*N

# Rate-distortion problem data
rho    = qu.randDensityMatrix(n, iscomplex=iscomplex)
entr_A = qu.quantEntropy(rho)

Delta = (np.eye(n*n) - qu.purify(rho)).view(dtype=np.float64).reshape(1, -1)
D = 0.5

# Build problem model
tr2 = get_tr2(n, sn, vN, iscomplex)

A1 = np.hstack((np.zeros((sn, 1)), tr2, np.zeros((sn, 1))))
A2 = np.hstack((np.zeros((1, 1)), Delta, np.ones((1, 1))))
A = np.vstack((A1, A2))

b = np.zeros((sn + 1, 1))
b[:sn] = sym.mat_to_vec(rho, iscomplex=iscomplex)
b[-1] = D

c = np.zeros((vN + 2, 1))
c[0] = 1.

# Input into model and solve
cones = [quantcondentr.Cone(n, n, 0, iscomplex=iscomplex), nonnegorthant.Cone(1)]
model = model.Model(c, A, b, cones=cones, offset=entr_A)
solver = solver.Solver(model)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")