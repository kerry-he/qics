import numpy as np
import scipy as sp

import cProfile

from cones import *
from utils import symmetric as sym, linear as lin
from solver import model, solver

np.random.seed(1)

n = 400
m = 4
vn = n * (n + 1) // 2
vm = m * (m + 1) // 2
p = n // m

A = np.zeros((vm*p, vn))
b = np.ones((vm*p, 1))
t = 0
for k in range(p):
    for j in range(m):
        for i in range(j):
            H = np.zeros((n, n))
            H[k*m + i, k*m + j] = 1
            H[k*m + j, k*m + i] = 1
            A[[t]] = sym.mat_to_vec(H).T
            b[t] = 0
            t += 1
            
        H = np.zeros((n, n))
        H[k*m + j, k*m + j] = 1
        A[[t]] = sym.mat_to_vec(H).T
        b[t] = 1
        t += 1
A = sp.sparse.csr_array(A)

C = np.random.randn(n,n)
C = C+C.T
c = sym.mat_to_vec(C)

cones = [possemidefinite.Cone(n)]
model = model.Model(c, A, b, cones=cones)
solver = solver.Solver(model, sym=True, ir=True)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")

# # Solve using MOSEK
# import sys
# from mosek.fusion import Model, Matrix, Domain, Expr, ObjectiveSense, ProblemStatus
# M = Model("maxcut")
# X = M.variable(Domain.inPSDCone(n))
# M.constraint(X.diag(), Domain.equalsTo(1.0))
# M.objective(ObjectiveSense.Minimize, Expr.sum(Expr.mulElm(C, X)))
# #M.setSolverParam("numThreads", 1)
# M.setLogHandler(sys.stdout)
# M.solve()