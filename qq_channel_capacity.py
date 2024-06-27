import numpy as np
import scipy as sp
import math

import cProfile

from cones import *
from utils import symmetric as sym
from utils import quantum as qu
from solver import model, solver

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

np.random.seed(1)
np.set_printoptions(threshold=np.inf)

# Define dimensions
hermitian = True
dtype = np.complex128 if hermitian else np.float64
ni = 8
no = 8
ne = 8

# Define random instance of qq channel capacity problem
V, W = qu.randDegradableChannel(ni, no, ne, hermitian=hermitian)

sni  = sym.vec_dim(ni, hermitian=hermitian)
snei = sym.vec_dim(ne*ni, hermitian=hermitian)
vni  = 2*ni*ni if hermitian else ni*ni
vnei = 2*ne*ni*ne*ni if hermitian else ne*ni*ne*ni

eye = get_eye(ni, sni, hermitian=hermitian)
tr  = sym.mat_to_vec(np.eye(ni, dtype=dtype), hermitian=hermitian)
WNW = np.zeros((vnei, sni))
k = 0
for j in range(ni):
    for i in range(j):
        H = np.zeros((ni, ni), dtype=dtype)
        H[i, j] = H[j, i] = np.sqrt(0.5)
        out = W @ sym.p_tr(V @ H @ V.conj().T, 1, (no, ne)) @ W.conj().T
        out = (out + out.conj().T) * 0.5
        WNW[:, k] = out.view(dtype=np.float64).reshape(-1)
        k += 1

        if hermitian:
            H[i, j] = np.sqrt(0.5) * 1j
            H[j, i] = -np.sqrt(0.5) * 1j
            out = W @ sym.p_tr(V @ H @ V.conj().T, 1, (no, ne)) @ W.conj().T
            out = (out + out.conj().T) * 0.5
            WNW[:, k] = out.view(dtype=np.float64).reshape(-1)
            k += 1
    H = np.zeros((ni, ni), dtype=dtype)
    H[j, j] = 1
    out = W @ sym.p_tr(V @ H @ V.conj().T, 1, (no, ne)) @ W.conj().T
    out = (out + out.conj().T) * 0.5
    WNW[:, k] = out.view(dtype=np.float64).reshape(-1)
    k += 1

# Build problem model
A = np.hstack((np.zeros((1, 1)), tr.T))        
b = np.ones((1, 1)) 

c = np.zeros((1 + sni, 1))
c[0] = 1.


G1 = np.hstack((np.ones((1, 1)), np.zeros((1, sni))))
G2 = np.hstack((np.zeros((vnei, 1)), WNW))
# G3 = np.hstack((np.zeros((vni, 1)), eye))
G = -np.vstack((G1, G2))

h = np.zeros((1 + vnei, 1))

# Input into model and solve
cones = [quantcondentr.Cone(ne, ni, 1, hermitian=hermitian)]
model = model.Model(c, A, b, G, h, cones=cones)
solver = solver.Solver(model)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")