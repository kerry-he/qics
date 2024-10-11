import numpy

import qics
from qics.quantum.random import density_matrix
from qics.vectorize import lin_to_mat, mat_to_vec

numpy.random.seed(1)

n = 2

# Generate random problem data
rho = density_matrix(n, iscomplex=True)
sig = density_matrix(n, iscomplex=True)

# Model problem using primal variable M
# Define objective function
eye_n = numpy.eye(n, dtype=numpy.complex128)
zero_n = numpy.zeros((n, n), dtype=numpy.complex128)
C = numpy.block([[zero_n, eye_n], [eye_n, zero_n]])
c = -0.5 * qics.vectorize.mat_to_vec(C)

# Build linear constraints
# M11 = rho
A1 = lin_to_mat(lambda X: X[:n, :n], (2 * n, n), iscomplex=True)
b1 = mat_to_vec(rho, compact=True)

# M22 = sig
A2 = lin_to_mat(lambda X: X[n:, n:], (2 * n, n), iscomplex=True)
b2 = mat_to_vec(sig, compact=True)

A = numpy.vstack((A1, A2))
b = numpy.vstack((b1, b2))

# Define cones to optimize over
cones = [qics.cones.PosSemidefinite(2 * n, iscomplex=True)]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
