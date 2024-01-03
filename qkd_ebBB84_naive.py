import numpy as np
import scipy as sp
import math

import cProfile

from cones import *
from utils import symmetric as sym
from solver import model, solver

np.random.seed(1)
np.set_printoptions(threshold=np.inf)

# Problem solves the type
# (QKD)     min    f(rho) = D( K(rho) || Z(K(rho)) )  (quantum relative entropy)
#           s.t.   Gamma(rho) = gamma                 (affine constraint)
#                  rho >= 0                           (rho positive semidefinite)
# Obtained from https://math.uwaterloo.ca/~hwolkowi/henry/reports/ZGNQKDmainsolverUSEDforPUBLCNJuly31/

# Problem data
data = sp.io.loadmat('examples/ebBB84.mat')
gamma = data['gamma']
Gamma = data['Gamma'][:, 0]
Klist = data['Klist'][0, :]
Zlist = data['Zlist'][0, :]

no, ni = np.shape(Klist[0])
nc = np.size(gamma)

vni = sym.vec_dim(ni)
vno = sym.vec_dim(no)

K_op = np.zeros((vno, vni))
ZK_op = np.zeros((vno, vni))
Gamma_op = np.array([sym.mat_to_vec(G).T[0] for G in Gamma])

k = -1
for j in range(ni):
    for i in range(j + 1):
        k += 1
    
        H = np.zeros((ni, ni))
        if i == j:
            H[i, j] = 1
        else:
            H[i, j] = H[j, i] = math.sqrt(0.5)
        
        KHK = np.zeros((no, no))
        for K in Klist:
            KHK += K @ H @ K.T
        K_op[:, [k]] = sym.mat_to_vec(KHK)
        
        ZKHKZ = np.zeros((no, no))
        for Z in Zlist:
            ZKHKZ += Z @ KHK @ Z.T
        ZK_op[:, [k]] = sym.mat_to_vec(ZKHKZ)


# Build problem model
A = np.hstack((np.zeros((nc, 1)), Gamma_op))
b = gamma

c = np.zeros((1 + vni, 1))
c[0] = 1.

G1 = np.hstack((np.ones((1, 1)), np.zeros((1, vni))))
G2 = np.hstack((np.zeros((vno, 1)), K_op))
G3 = np.hstack((np.zeros((vno, 1)), ZK_op))
G = -np.vstack((G1, G2, G3))

h = np.zeros((1 + 2 * vno, 1))

# Input into model and solve
cones = [quantrelentr.QuantRelEntropy(no)]
model = model.Model(c, A, b, G, h, cones=cones)
solver = solver.Solver(model)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")