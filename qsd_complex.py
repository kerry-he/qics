import numpy as np
import scipy as sp

import cProfile

from cones import *
from utils import symmetric as sym, linear as lin
from utils import quantum as qu
from solver import model, solver

#     max_Ei    p1 tr[E1 rho1] + ... + pn tr[En rhon]
#     s.t.      Ei >= 0                 for i = 1,...,n+1
#               E1 + ... + E_n+1 = I
#               <Ei, rhoj> = 0          for i,j = 1,...,n, i =/=j

np.random.seed(1)

unambiguous = False

n = 50
p = 10

if unambiguous:
    # Add another variable for the failure outcome
    p += 1

if unambiguous:
    # Only do unambiguous state discrimination for 
    rhos = [qu.randPureDensityMatrix(n, iscomplex=True) for _ in range(p)]
else:
    rhos = [qu.randDensityMatrix(n, iscomplex=True) for _ in range(p)]
ps = np.random.rand(p)
if unambiguous:
    ps[-1] = 0
ps = ps / np.sum(ps)

# Write in SDP form
C = [(-p * rho).view(dtype=np.float64).reshape(-1, 1) for (p, rho) in zip(rhos, ps)]
c = np.array(C).reshape(-1, 1)

A = []
b = []
for j in range(n):
    for i in range(j):
        temp = [sp.sparse.coo_matrix(([1., 1.], ([i, j], [j, i])), shape=[n, n], dtype=np.complex128).toarray().view(dtype=np.float64).reshape(-1) for _ in range(p)]
        temp = np.array(temp).reshape(-1)
        A.append(temp)
        b.append(0.)

        temp = [sp.sparse.coo_matrix(([1.j, -1.j], ([i, j], [j, i])), shape=[n, n], dtype=np.complex128).toarray().view(dtype=np.float64).reshape(-1) for _ in range(p)]
        temp = np.array(temp).reshape(-1)
        A.append(temp)
        b.append(0.)        

    temp = [sp.sparse.coo_matrix(([1.], ([j], [j])), shape=[n, n], dtype=np.complex128).toarray().view(dtype=np.float64).reshape(-1) for _ in range(p)]
    temp = np.array(temp).reshape(-1)
    A.append(temp)
    b.append(1.)

# Optionally add in unambiguous state discimination constraints
if unambiguous:
    for i in range(p - 1):
        for j in range(p - 1):
            if i != j:
                temp = []
                for k in range(p):
                    if k != i:
                        temp.append(np.zeros((2*n*n,)))
                    else:
                        temp.append(rhos[j].view(dtype=np.float64).reshape(-1))
                temp = np.array(temp).reshape(-1)
                A.append(temp)
                b.append(0.)

A = np.array(A)
A = sp.sparse.csr_matrix(A)
b = np.array(b).reshape((-1, 1))

cones = [possemidefinite.Cone(n, iscomplex=True) for _ in range(p)]
model = model.Model(c=c,  A=A,   b=b, cones=cones)
# model = model.Model(c=-b, G=A.T, h=c, cones=cones)
solver = solver.Solver(model, sym=True, ir=True)

profiler = cProfile.Profile()
profiler.enable()

solver.solve()

profiler.disable()
profiler.dump_stats("example.stats")