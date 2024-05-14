import cProfile

import numpy as np
import scipy as sp

from cones import *
from utils import symmetric as sym
from solver import model, solver

np.random.seed(3)

n = 50000
p = 2000

A = sp.sparse.random(p, n, 0.0001).tocsr()#.toarray()
A = A[A.getnnz(1)>0]
p = A.shape[0]
# A = A.toarray()
b = np.random.rand(p, 1)
c = np.random.rand(n, 1)

cones = [nonnegorthant.Cone(n)]
model = model.Model(c, A, b, cones=cones)
solver = solver.Solver(model, sym=True, ir=True)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")

# A = np.array([[1.,  0.,  1.,  1.], [-1.,  1.,  0., -1.]])
# b = np.array([[1.], [1.]])
# c = np.array([[1.], [2.], [0.], [-1.]])

# cones = [nonnegorthant.Cone(4)]
# model = model.Model(c, A, b, cones=cones)
# solver = solver.Solver(model, sym=True)

# solver.solve()