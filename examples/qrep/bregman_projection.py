import numpy as np
import scipy as sp

import qics
from qics.vectorize import mat_to_vec

## Bregman projection
#   min  S(X||Y) = -S(X) - tr[X log(Y)]
#   s.t. <X, Ai> = bi    for    i = 1,...,p
#        tr[X] = 1

np.random.seed(1)

n = 5
p = 2

# Generate random matrix Y to project
Y = np.random.randn(n, n) + np.random.randn(n, n) * 1j
Y = Y @ Y.conj().T
tr_Y = np.trace(Y).real

# Define objective function
ct = np.array([[1.0]])
cu = np.array([[0.0]])
cX = -sp.linalg.logm(Y) - np.eye(n)
c = np.vstack((ct, cu, mat_to_vec(cX)))

# Build linear constraints
# u = 1
A1 = np.hstack((np.array([[0.0, 1.0]]), np.zeros((1, 2 * n * n))))
b1 = np.array([[1.0]])
# <X, Ai> = bi for randomly generated Ai, bi
A2 = np.zeros((p, 2 + 2 * n * n))
for i in range(p):
    Ai = np.random.randn(n, n) + np.random.rand(n, n) * 1j
    A2[[i], 2:] = mat_to_vec(Ai + Ai.conj().T).T
b2 = np.random.randn(p, 1)

A = np.vstack((A1, A2))
b = np.vstack((b1, b2))

# Define cones to optimize over
cones = [qics.cones.QuantEntr(n, iscomplex=True)]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, cones=cones, offset=tr_Y)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
