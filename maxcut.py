import numpy as np
import scipy as sp

import cProfile

from cones import *
from utils import symmetric as sym, linear as lin
from solver import model, solver

np.random.seed(1)

n = 300

# A = np.zeros((n,n*(n+1)//2))
# for i in range(n):
#     A[i,i+i*(i+1)//2] = 1
# A = sp.sparse.csr_array(A)

A = np.zeros((n,n*n))
for i in range(n):
    A[i,i+i*n] = 1
A = sp.sparse.csr_array(A)


b = np.ones((n, 1))
C = np.random.randn(n,n)
C = C+C.T
# c = sym.mat_to_vec(C)
c = C.reshape((-1, 1))

cones = [possemidefinite.Cone(n)]
model = model.Model(c, A, b, cones=cones)
solver = solver.Solver(model, sym=True, ir=True)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")

# Solve using MOSEK
import sys
from mosek.fusion import Model, Matrix, Domain, Expr, ObjectiveSense, ProblemStatus
M = Model("maxcut")
X = M.variable(Domain.inPSDCone(n))
M.constraint(X.diag(),Domain.equalsTo(1.0))
M.objective(ObjectiveSense.Minimize,Expr.sum(Expr.mulElm(C,X)))
#M.setSolverParam("numThreads", 1)
M.setLogHandler(sys.stdout)
M.solve()