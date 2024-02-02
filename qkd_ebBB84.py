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
data = sp.io.loadmat('examples/DMCV.mat')
gamma = data['gamma']
Gamma = data['Gamma'][:, 0]
Klist = data['Klist'][0, :]
Zlist = data['Zlist'][0, :]
hermitian = True

no, ni = np.shape(Klist[0])
nc = np.size(gamma)

vni = sym.vec_dim(ni, hermitian=hermitian)
vno = sym.vec_dim(no, hermitian=hermitian)

Gamma_op = np.array([sym.mat_to_vec(G, hermitian=hermitian).T[0] for G in Gamma])

# nc -= 1
# Gamma_op = Gamma_op[:-1, :]
# gamma = gamma[:-1]

# Build problem model
A = np.hstack((np.zeros((nc, 1)), Gamma_op))
b = gamma

c = np.zeros((1 + vni, 1))
c[0] = 1.

# Input into model and solve
cones = [quantkeydist.QuantKeyDist(Klist, Zlist, hermitian=hermitian)]
model = model.Model(c, A, b, cones=cones)
solver = solver.Solver(model)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")