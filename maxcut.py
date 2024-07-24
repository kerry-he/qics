import numpy as np
import scipy as sp

import cProfile

import qics

np.random.seed(1)

n = 20

A_is = [i for i in range(n)]
A_js = [i + i*n for i in range(n)]
A_vs = [1. for i in range(n)]
A = sp.sparse.csr_array((A_vs, (A_is, A_js)), shape=(n, n*n))

b = np.ones((n, 1))
C = np.random.randn(n, n)
C = C + C.T
c = C.reshape((-1, 1))

cones = [qics.cones.PosSemidefinite(n)]
model = qics.Model(c=c,  A=A,   b=b, cones=cones)
# model = model.Model(c=-b, G=A.T, h=c, cones=cones)
solver = qics.Solver(model, ir=True)

profiler = cProfile.Profile()
profiler.enable()

out = solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")
