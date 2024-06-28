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

def get_ptr(n0, n1, sys):
    n = n0 if sys == 0 else n1      # System being traced out
    m = n1 if sys == 0 else n0      # Remaining system
    N = n0 * n1
    vN = N*N 
    sm = sym.vec_dim(m)
    ptr = np.zeros((sm, vN))
    k = 0
    for j in range(m):
        for i in range(j + 1):
            H = np.zeros((m, m))
            if i == j:
                H[i, j] = 1
                I_H = sym.i_kr(H, sys, (n0, n1))
                ptr[k, :] = I_H.reshape(-1)         
                k += 1       
            else:
                H[i, j] = H[j, i] = math.sqrt(0.5)
                I_H = sym.i_kr(H, sys, (n0, n1))
                ptr[k, :] = I_H.reshape(-1)
                k += 1

    return ptr

np.random.seed(1)
np.set_printoptions(threshold=np.inf)

# Build MED problem
#   min tr(h*rho)
#   s.t. rho >= 0, trace(rho) == 1
#        Tr_1 rho == Tr_L rho
#        S(L|1...L-1) >= 0
L = 2
H = heisenberg(-1, L) # Hamiltonian
dims = 2*np.ones(L)

N = 2**L
vN = N*N

m = 2**(L-1)
vm = sym.vec_dim(m)

tr1   = get_ptr(2, m, 0)
trend = get_ptr(m, 2, 1)
Id    = np.eye(N).reshape((-1, 1))

A1 = np.hstack((np.zeros((vm, 1)), tr1 - trend))
A2 = np.hstack((np.zeros((1, 1)), Id.T))
A3 = np.hstack((np.ones((1, 1)), np.zeros((1, vN))))
A = np.vstack((A1, A2, A3))

b = np.zeros((vm + 2, 1))
b[-2] = 1.

c = np.vstack((np.zeros((1, 1)), H.reshape((-1, 1))))

# Input into model and solve
cones = [quantcondentr.Cone(m, 2, 1)]
model = model.Model(c, A, b, cones=cones, offset=0.0)
solver = solver.Solver(model)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")