import numpy as np
import scipy as sp
import math

import cProfile

from cones import *
from utils import symmetric as sym
from solver import model, solver

np.random.seed(1)
np.set_printoptions(suppress=True, edgeitems=30, linewidth=100000, precision=2)
# Problem solves the type
# (QKD)     min    f(rho) = D( K(rho) || Z(K(rho)) )  (quantum relative entropy)
#           s.t.   Gamma(rho) = gamma                 (affine constraint)
#                  rho >= 0                           (rho positive semidefinite)
# Obtained from https://math.uwaterloo.ca/~hwolkowi/henry/reports/ZGNQKDmainsolverUSEDforPUBLCNJuly31/

# Problem data
data   = sp.io.loadmat('problems/quant_key_rate/qkd_dprBB84_3_02_15.mat')
gamma  = data['raw']['gamma_fr'][0, 0]
Gamma  = data['raw']['Gamma_fr'][0, 0][0]
Klist  = data['raw']['Klist_fr'][0, 0][0]
ZKlist = data['raw']['ZKlist_fr'][0, 0][0]
Klist_raw = data['raw']['Klist'][0, 0][0]
Zlist_raw = data['raw']['Zlist'][0, 0][0]
hermitian = bool(data['cones'][0, 0][0, 0]['complex'][0, 0])

no, ni = np.shape(Klist[0])
nc = np.size(gamma)

vni = sym.vec_dim(ni, hermitian=hermitian)
vno = sym.vec_dim(no, hermitian=hermitian)

Gamma_op = np.array([sym.mat_to_vec(G, hermitian=hermitian).T[0] for G in Gamma])

# Build problem model
A = np.hstack((np.zeros((nc, 1)), Gamma_op))
b = gamma

c = np.zeros((1 + vni, 1))
c[0] = 1.

# Input into model and solve
cones = [quantkeydist.Cone(Klist_raw, Zlist_raw, protocol="dprBB84_fast", hermitian=hermitian)]
model = model.Model(c, A, b, cones=cones)
solver = solver.Solver(model)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")