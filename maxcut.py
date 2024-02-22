import numpy as np
import scipy as sp

from cones import *
from utils import symmetric as sym
from solver import model, solver

np.random.seed(1)

n = 10

A = np.zeros((n,n*(n+1)//2))
for i in range(n):
    A[i,i+i*(i+1)//2] = 1
b = np.ones((n,1))
C = np.random.randn(n,n)
C = C+C.T
c = sym.mat_to_vec(C)

cones = [possemidefinite.PosSemiDefinite(n)]
model = model.Model(c, A, b, cones=cones)
solver = solver.Solver(model)
solver.solve()

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