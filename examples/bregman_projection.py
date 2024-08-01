import numpy as np
import scipy as sp
import qics
import qics.utils.quantum as qu

## Bregman projection
#   min  S(X||Y) = -S(X) - tr[X log(Y)]
#   s.t. <X, Ai> = bi    for    i = 1,...,p
#        tr[X] = 1

n = 10
p = 3
iscomplex = False
dtype = np.complex128 if iscomplex else np.float64

Y = qu.rand_density_matrix(n, iscomplex=iscomplex) * np.random.rand()

# Define objective function
c1 = np.array([[1.]])
c2 = np.array([[0.]])
c3 = sp.linalg.logm(Y).view(dtype=np.float64).reshape(-1, 1)
c  = np.vstack((c1, c2, c3))

# Build linear constraints
# <X, Ai> = bi for randomly generated Ai, bi
A1 = np.zeros((p, 2 + n*n))
for i in range(p):
    X = np.random.randn(n, n)
    if iscomplex:
        X = X + np.random.rand(n, n) * 1j
    A1[i, 2:] = (X + X.conj().T).view(dtype=np.float64).reshape(-1)
b1 = np.random.randn(p, 1)
# tr[X] = 1
A2 = np.hstack((np.zeros((1, 2)), np.eye(n, dtype=dtype).view(dtype=np.float64).reshape(1, -1)))
b2 = np.array([[1.]])
# u = 1
A3 = np.hstack((np.zeros((1, 1)), np.ones((1, 1)), np.zeros((1, n*n))))
b3 = np.array([[1.]])

A = np.vstack((A1, A2, A3))
b = np.vstack((b1, b2, b3))

# Input into model and solve
cones = [qics.cones.QuantEntr(n, iscomplex=iscomplex)]

# Initialize model and solver objects
model  = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
out = solver.solve()