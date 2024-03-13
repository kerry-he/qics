import numpy as np
import scipy as sp
import math

import cProfile

from cones import *
from utils import symmetric as sym
from solver import model, solver

def heisenberg(delta, L):
    sx = np.array([[0., 1.], [1., 0.]])
    sy = np.array([[0., -1.j], [1.j, 0.]])
    sz = np.array([[1., 0.], [0., -1.]])
    h = -(np.kron(sx,sx) + np.kron(sy,sy) + delta*np.kron(sz,sz))

    return ( np.kron(h,np.eye(2**(L-2))) ).real


np.random.seed(1)
np.set_printoptions(threshold=np.inf)

# Build MED problem
#   min tr(h*rho)
#   s.t. rho >= 0, trace(rho) == 1
#        Tr_1 rho == Tr_L rho
#        S(L|1...L-1) >= 0

L = 7
H = heisenberg(-1, L) # Hamiltonian
dims = 2*np.ones(L)

N = 2**L
vN = sym.vec_dim(N)

m = 2**(L-1)
vm = sym.vec_dim(m)


tr1   = sym.lin_to_mat(lambda x : sym.p_tr(x, 0, (2, m)), N, m)
trend = sym.lin_to_mat(lambda x : sym.p_tr(x, 1, (m, 2)), N, m)
Id    = sym.mat_to_vec(np.eye(N))

A1 = np.hstack((np.zeros((vm, 1)), tr1 - trend))
A2 = np.hstack((np.zeros((1, 1)), Id.T))
A3 = np.hstack((np.ones((1, 1)), np.zeros((1, vN))))
A = np.vstack((A1, A2, A3))

b = np.zeros((vm + 2, 1))
b[-2] = 1.

c = np.vstack((np.zeros((1, 1)), sym.mat_to_vec(H)))

# Input into model and solve
cones = [quantcondentr.Cone(m, 2, 1)]
model = model.Model(c, A, b, cones=cones, offset=0.0)
solver = solver.Solver(model)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")