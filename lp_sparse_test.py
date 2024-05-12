import cProfile

import numpy as np
import scipy as sp

from cones import *
from utils import symmetric as sym
from solver import model, solver

np.random.seed(2)

n = 4
L = 1
p = 4
# n = 10
# L = 500
# p = 200
vn = n * n

A = np.zeros((p, vn))
for i in range(p):
    H = sp.sparse.random(n, n, 0.25)
    H = H + H.T
    A[[i], :] = H.reshape(-1, 1).toarray().T
# A = sp.sparse.random(p, vn*L, 0.5).tocsr()#.toarray()
# A = A[A.getnnz(1)>0]
# p = A.shape[0]
# A = A.toarray()
A = sp.sparse.csr_array(A)
b = np.random.rand(p, 1)
c = np.random.rand(n, n)
c = c + c.T
c = c.reshape((-1, 1))

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

# A = np.array([[1.,  0.,  1.,  1.], [-1.,  1.,  0., -1.]])
# b = np.array([[1.], [1.]])
# c = np.array([[1.], [.], [0.], [-1.]])

# cones = [nonnegorthant.Cone(4)]
# model = model.Model(c, A, b, cones=cones)
# solver = solver.Solver(model, sym=True)

# solver.solve()