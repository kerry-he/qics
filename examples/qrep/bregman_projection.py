import numpy as np
import scipy as sp

import qics
from qics.vectorize import lin_to_mat, vec_dim, mat_to_vec

np.random.seed(1)

n = 5
vn = vec_dim(n, iscomplex=True)

# Generate random positive semidefinite matrix Y to project
Y = np.random.randn(n, n) + np.random.randn(n, n)*1j
Y = Y @ Y.conj().T
trY = np.trace(Y).real

# Model problem using primal variables (t, u, X)
# Define objective function
c = np.block([[1.0], [0.0], [mat_to_vec(-sp.linalg.logm(Y) - np.eye(n))]])

# Build linear constraints
trace = lin_to_mat(lambda X: np.trace(X), (n, 1), iscomplex=True)

A = np.block([
    [0.0, 1.0, np.zeros((1, vn))],  # u = 1
    [0.0, 0.0, trace            ]   # tr[X] = 1
])

b = np.array([[1.0], [0.0]])

# Define cones to optimize over
cones = [qics.cones.QuantEntr(n, iscomplex=True)]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, cones=cones, offset=trY)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
