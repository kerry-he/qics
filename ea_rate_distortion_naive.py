import numpy as np
import scipy as sp
import math

import cProfile

from cones import *
from utils import symmetric as sym
from solver import model, solver

def purify(eig):
    n = np.size(eig)

    vec = np.zeros((n*n, 1))
    for (i, ii) in enumerate(range(0, n*n, n + 1)):
        vec[ii] = math.sqrt(eig[i])

    return vec @ vec.T

def get_ikr_tr1(n, sN):
    N = n * n
    tr2 = np.zeros((N*N, sN))
    k = -1
    for j in range(N):
        for i in range(j + 1):
            k += 1
        
            H = np.zeros((N, N))
            if i == j:
                H[i, j] = 1
            else:
                H[i, j] = H[j, i] = math.sqrt(0.5)
            
            I_H = sym.i_kr(sym.p_tr(H, 0, (n, n)), 0, (n, n))
            tr2[:, k] = I_H.ravel()

    return tr2

def get_tr2(n, sn, sN):
    tr2 = np.zeros((sn, sN))
    k = -1
    for j in range(n):
        for i in range(j + 1):
            k += 1
        
            H = np.zeros((n, n))
            if i == j:
                H[i, j] = 1
            else:
                H[i, j] = H[j, i] = math.sqrt(0.5)
            
            I_H = sym.i_kr(H, 1, (n, n))
            tr2[[k], :] = sym.mat_to_vec(I_H).T

    return tr2

def get_eye(n, sn):
    eye = np.zeros((n*n, sn))
    k = -1
    for j in range(n):
        for i in range(j + 1):
            k += 1
        
            H = np.zeros((n, n))
            if i == j:
                H[i, j] = 1
            else:
                H[i, j] = H[j, i] = math.sqrt(0.5)
            
            eye[:, k] = H.ravel()
    return eye


np.random.seed(1)
np.set_printoptions(threshold=np.inf)

# Define dimensions
n = 6
N = n * n
sn = sym.vec_dim(n)
sN = sym.vec_dim(N)

# Rate-distortion problem data
eig_A = np.random.rand(n)
eig_A /= np.sum(eig_A)
rho_A = np.diag(eig_A)
rho_AR = purify(eig_A)
entr_A = -np.sum(eig_A * np.log(eig_A))

Delta = sym.mat_to_vec(np.eye(N) - rho_AR)
D = 0.5

# Build problem model
tr2 = get_tr2(n, sn, sN)
ikr_tr1 = get_ikr_tr1(n, sN)
eye = get_eye(N, sN)

A = np.hstack((np.zeros((sn, 1)), tr2))

b = sym.mat_to_vec(rho_A)

c = np.zeros((sN + 1, 1))
c[0] = 1.

G1 = np.hstack((np.ones((1, 1)), np.zeros((1, sN))))     # t_qre
G2 = np.hstack((np.zeros((N*N, 1)), eye))                 # X_qre
G3 = np.hstack((np.zeros((N*N, 1)), ikr_tr1))             # Y_qre
G4 = np.hstack((np.zeros((1, 1)), -Delta.T))             # LP
G = -np.vstack((G1, G2, G3, G4))

h = np.zeros((1 + N*N*2 + 1, 1))
h[-1] = D

# Input into model and solve
cones = [quantrelentr.Cone(N), nonnegorthant.Cone(1)]
model = model.Model(c, A, b, G, h, cones=cones, offset=entr_A)
solver = solver.Solver(model)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")