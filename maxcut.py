import numpy as np
import scipy as sp

import cProfile

from cones import *
from utils import symmetric as sym, linear as lin
from solver import model, solver

np.random.seed(1)

n = 1000

A_is = [i for i in range(n)]
A_js = [i + i*n for i in range(n)]
A_vs = [1. for i in range(n)]
A = sp.sparse.csr_array((A_vs, (A_is, A_js)), shape=(n, n*n))

b = np.ones((n, 1))
C = np.random.randn(n, n)
C = C + C.T
c = C.reshape((-1, 1))

cones = [possemidefinite.Cone(n)]
model = model.Model(c=c,  A=A,   b=b, cones=cones)
# model = model.Model(c=-b, G=A.T, h=c, cones=cones)
solver = solver.Solver(model, sym=True, ir=True)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")

# Solve using CVXOPT and MOSEK
from utils.other_solvers import cvxopt_solve_sdp, mosek_solve_sdp

sol = cvxopt_solve_sdp([-C], b, A, [n])
print("optval: ", sol['gap']) 
print("time:   ", sol['time'])   

sol = mosek_solve_sdp([-C], b, A, [n])