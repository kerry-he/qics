import numpy as np
import scipy as sp
import math

from cones import *
from utils import symmetric as sym
from solver import model, solver

def purify(eig):
    n = np.size(eig)

    vec = np.zeros((n*n, 1))
    for (i, ii) in enumerate(range(0, n*n, n + 1)):
        vec[ii] = math.sqrt(eig[i])

    return vec @ vec.T

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


np.random.seed(1)
np.set_printoptions(threshold=np.inf)

# Define dimensions
n = 2
N = n * n
sn = sym.vec_dim(n)
sN = sym.vec_dim(N)

# Rate-distortion problem data
eig_A = np.random.rand(n)
eig_A /= np.sum(eig_A)
rho_A = np.diag(eig_A)
rho_AR = purify(eig_A)

Delta = sym.mat_to_vec(np.eye(N) - rho_AR)
D = 0.5

# Build problem model
tr2 = get_tr2(n, sn, sN)

A1 = np.hstack((np.zeros((sn, 1)), tr2, np.zeros((sn, 1))))
A2 = np.hstack((np.zeros((1, 1)), Delta.T, np.ones((1, 1))))
A = np.vstack((A1, A2))

b = np.zeros((sn + 1, 1))
b[:sn] = sym.mat_to_vec(rho_A)
b[-1] = D

c = np.zeros((sN + 2, 1))
c[0] = 1.

# Input into model and solve
cones = [quantcondentr.QuantCondEntropy(n, n, 0), nonnegorthant.NonNegOrthant(1)]
model = model.Model(c, A, b, cones)
solver = solver.Solver(model, max_iter=100)

solver.solve()

print(b)
print(A @ solver.point.x)
print(sym.vec_to_mat(solver.point.x[1:-1]))
