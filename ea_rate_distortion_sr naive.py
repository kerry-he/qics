import numpy as np
import scipy as sp
import math

import cProfile

from cones import *
from utils import symmetric as sym
from solver import model, solver
from utils import linear as lin

def purify(eig):
    n = np.size(eig)

    vec = np.zeros((n*n, 1))
    for (i, ii) in enumerate(range(0, n*n, n + 1)):
        vec[ii] = math.sqrt(eig[i])

    return vec @ vec.T


np.random.seed(1)
np.set_printoptions(threshold=np.inf)

# Define dimensions
n = 16
N = n * n
vn = sym.vec_dim(n)
vN = sym.vec_dim(N)

# Rate-distortion problem data
eig_A = np.random.rand(n)
eig_A /= np.sum(eig_A)
rho_A = np.diag(eig_A)
rho_AR = purify(eig_A)
entr_A = -np.sum(eig_A * np.log(eig_A))

Delta = np.eye(N) - rho_AR
Delta_X = sym.mat_to_vec(Delta[::n+1, ::n+1])
D = 0.5

# Build problem model
tr2 = sym.lin_to_mat(lambda x : x, n, n, hermitian=False)
IDX = np.zeros((n, n), 'uint64')
temp = np.arange(n*(n-1)).reshape((n, n-1)).T
IDX[1:, :] += np.tril(temp).astype('uint64')
IDX[:-1, :] += np.triu(temp, 1).astype('uint64')
A_y = np.zeros((n, n*(n-1)))
A_X = np.zeros((n, vn))
for i in range(n):
    idx = IDX[i]
    idx = np.delete(idx, i)
    A_y[i, idx] = 1.

    temp = np.zeros((n, n))
    temp[i, i] = 1.
    A_X[[i], :] = sym.mat_to_vec(temp).T

A = np.hstack((np.zeros((n, 1)), np.zeros((n, 1)), A_y, A_X))                         # Partial trace constraint
b = eig_A.reshape((-1, 1))

G3_y = np.zeros((n*(n-1), n*(n-1)))
G3_X = np.zeros((n*(n-1), vn))
k = 0
for i in range(n):
    for j in range(n-1):
        idx = IDX.T[i]
        idx = np.delete(idx, i)
        G3_y[k, idx] = 1.

        temp = np.zeros((n, n))
        temp[i, i] = 1.
        G3_X[[k], :] = sym.mat_to_vec(temp).T

        k += 1

G6_y = np.zeros((vn, n*(n-1)))
G6_X = np.zeros((vn, vn))
k = 0
for j in range(n):
    for i in range(j):
        k += 1

    idx = IDX.T[j]
    idx = np.delete(idx, j)
    G6_y[k, idx] = 1.

    temp = np.zeros((n, n))
    temp[j, j] = 1.
    G6_X[[k], :] = sym.mat_to_vec(temp).T

    k += 1


Gtest = np.zeros((vN, vn + n*(n-1)))

k = 0
p_k = 0
for j in range(N):
    for i in range(j + 1):
        G_Y = np.zeros((1, n*(n-1)))
        G_X = np.zeros((1, vn))

        # Idx should be in RHO
        if (i % (n + 1) == 0) and (j % (n + 1) == 0):
            I = i // (n + 1)
            J = j // (n + 1)
            
            G_X = np.zeros((n, n))
            G_X[I, J] = 1.0 if (I == J) else np.sqrt(0.5)
            G_X[J, I] = G_X[I, J]
            G_X = sym.mat_to_vec(G_X).T
        
        elif i == j:
            G_Y[0, p_k] = 1.0
            p_k += 1

        Gtest[k, :] = -np.hstack((G_Y, G_X))

        k += 1


G1 = -np.hstack((np.ones((1, 1)),  np.zeros((1, 1)), np.zeros((1, n*(n-1) + vn))))                              # t
G2 = -np.hstack((np.zeros((n*(n-1), 1)), np.zeros((n*(n-1), 1)), np.eye((n*(n-1))), np.zeros((n*(n-1), vn))))   # p
G3 = -np.hstack((np.zeros((n*(n-1), 1)), np.zeros((n*(n-1), 1)), G3_y, G3_X))                                   # q
G4 = -np.hstack((np.zeros((1, 1)),  np.ones((1, 1)), np.zeros((1, n*(n-1) + vn))))                              # t
G5 = -np.hstack((np.zeros((vn, 1)), np.zeros((vn, 1)), np.zeros((vn, n*(n-1))), np.eye(vn)))                    # X
G6 = -np.hstack((np.zeros((vn, 1)), np.zeros((vn, 1)), G6_y, G6_X))                                             # Y
G7 = np.hstack((np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, n*(n-1))), Delta_X.T))                          # Distortion
G = np.vstack((G1, G2, G3, G4, G5, G6, G7))

h = np.zeros((1 + 2*n*(n-1) + 1 + 2*vn + 1, 1))
h[-1] = D

c = np.zeros((1 + n*(n-1) + vn + 1, 1))
c[0] = 1.
c[1] = 1.


# Input into model and solve
cones = [classrelentr.ClassRelEntropy(n*(n-1)), quantrelentr.QuantRelEntropy(n), nonnegorthant.NonNegOrthant(1)]
model = model.Model(c, A, b, G, h, cones=cones, offset=entr_A)
solver = solver.Solver(model)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")