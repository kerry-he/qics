import numpy as np
import scipy as sp

import cProfile

from cones import *
from utils import symmetric as sym, linear as lin
from solver import model, solver

np.random.seed(1)

n = 100

A_is = [i for i in range(n)]
A_js = [2*i + 2*i*n for i in range(n)]
A_vs = [1. for i in range(n)]
A = sp.sparse.csr_array((A_vs, (A_is, A_js)), shape=(n, 2*n*n))

b = np.ones((n, 1))
C = np.random.randn(n, n) + np.random.randn(n, n)*1j
C = C + C.conj().T
c = C.view(dtype=np.float64).reshape(-1, 1)

cones = [possemidefinite.Cone(n, hermitian=True)]
model = model.Model(c=c,  A=A,   b=b, cones=cones)
# model = model.Model(c=-b, G=A.T, h=c, cones=cones)
solver = solver.Solver(model, sym=True, ir=True)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")

# # Solve using CVXOPT and MOSEK


# C = np.block([[C.real, C.imag], [-C.imag, C.real]]) * 0.5
# #MAXCUT constraints
# A_is = [i for i in range(n)] + [i for i in range(n)]
# A_js = [i + 2*i*n for i in range(2*n)]
# A_vs = [1. for i in range(2*n)]
# b    = [2. for i in range(n)]

# # Hermiticity constraints
# A_is += [i for i in range(n, n + n*(n+1)//2)] + [i for i in range(n, n + n*(n+1)//2)]
# A_js += [i + 2*j*n for j in range(n) for i in range(j + 1)] + [j + 2*i*n for j in range(n) for i in range(j + 1)]
# A_vs += [1. for i in range(n*(n+1))]

# A_is += [i for i in range(n, n + n*(n+1)//2)] + [i for i in range(n, n + n*(n+1)//2)]
# A_js += [2*n*n + n + i + 2*j*n for j in range(n) for i in range(j + 1)] + [2*n*n + n + j + 2*i*n for j in range(n) for i in range(j + 1)]
# A_vs += [-1. for i in range(n*(n+1))]
# b    += [0. for i in range(n*(n+1)//2)]

# A_is += [i for i in range(n + n*(n+1)//2, n + n*(n+1))] + [i for i in range(n + n*(n+1)//2, n + n*(n+1))]
# A_js += [n + i + 2*j*n for j in range(n) for i in range(j + 1)] + [n + j + 2*i*n for j in range(n) for i in range(j + 1)]
# A_vs += [1. for i in range(n*(n+1))]

# A_is += [i for i in range(n, n + n*(n+1)//2)] + [i for i in range(n, n + n*(n+1)//2)]
# A_js += [2*n*n + i + 2*j*n for j in range(n) for i in range(j + 1)] + [2*n*n + j + 2*i*n for j in range(n) for i in range(j + 1)]
# A_vs += [1. for i in range(n*(n+1))]
# b    += [0. for i in range(n*(n+1)//2)]

# b = np.array(b).reshape((-1, 1))

# A = sp.sparse.coo_array((A_vs, (A_is, A_js)), shape=(n+n*(n+1), 4*n*n))
# A.sum_duplicates()
# A = A.tocsr()

# from utils.other_solvers import cvxopt_solve_sdp, mosek_solve_sdp

# # sol = cvxopt_solve_sdp([-C], b, A, [2*n])
# # print("optval: ", sol['gap']) 
# # print("time:   ", sol['time'])   

# sol = mosek_solve_sdp([-C], b, A, [2*n])


import cvxopt
import numpy
import picos

# Make the output reproducible.
cvxopt.setseed(1)

# Generate an arbitrary rank-deficient hermitian matrix M.
M = picos.Constant("M", C)

# Define the problem.
P = picos.Problem()
U = picos.HermitianVariable("U", n)
P.set_objective("min", (U | M))
P.add_constraint(picos.maindiag(U) == 1)
P.add_constraint(U >> 0)

print(P)

# Solve the problem.
P.solve(solver="mosek", verbose=True)