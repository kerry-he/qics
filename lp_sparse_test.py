import cProfile

import numpy as np
import scipy as sp

from cones import *
from utils import symmetric as sym
from solver import model, solver

np.random.seed(2)

n = 2          # Size of SDP blocks
L = 2500      # Number of SDP blocks
p = 200       # Number of constraints
# n = 10
# L = 500
# p = 200
vn = n * n

A_temp = sp.sparse.random(p, n*(n+1)*L // 2, 0.0001).tocsr().toarray()
A = np.zeros((p, vn*L))
for i in range(p):
    t = 0
    u = 0
    for k in range(L):
        A_vec = A_temp[[i], u:u+n*(n+1)//2]
        A[[i], t:t+vn] = sym.vec_to_mat(A_vec.T).ravel()
        
        t += vn
        u += n*(n+1) // 2
A = sp.sparse.csr_array(A)
A = A[A.getnnz(1)>0]
p = A.shape[0]

b = np.random.rand(p, 1)

t = 0
c = np.random.rand(vn*L, 1)
for k in range(L):
    c_k = np.random.randn(n, n)
    c_k = c_k + c_k.T
    c[t:t+vn, 0] = c_k.ravel()
    t += vn

cones = []
for i in range(L):
    cones.append(possemidefinite.Cone(n))

model = model.Model(c, A, b, cones=cones)
solver = solver.Solver(model, sym=True, ir=True)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")
