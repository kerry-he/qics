import numpy as np

import qics
from qics.quantum.random import density_matrix
from qics.vectorize import eye, mat_to_vec, vec_dim

np.random.seed(1)

n = 4
vn = vec_dim(n, iscomplex=True)
cn = vec_dim(n, iscomplex=True, compact=True)

# Model problem using primal variables (T, X, Y)
# Define random problem data
alpha = 0.25
rho = density_matrix(n, iscomplex=True)
sigma = density_matrix(n, iscomplex=True)

# Define objective function
c_T = (1 - alpha) * mat_to_vec(sigma)
c_X = np.zeros((vn, 1))
c_Y = alpha * mat_to_vec(rho)
c = np.block([[c_T], [c_X], [c_Y]])

# Build linear constraint X = I
A = np.block([[np.zeros((cn, vn)), eye(n, iscomplex=True), np.zeros((cn, vn))]])
b = mat_to_vec(np.eye(n, dtype=np.complex128), compact=True)

# Define cones to optimize over
cones = [qics.cones.OpPerspecEpi(n, alpha / (alpha - 1), iscomplex=True)]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
