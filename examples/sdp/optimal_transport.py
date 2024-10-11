import numpy

import qics
from qics.quantum import p_tr
from qics.quantum.random import density_matrix
from qics.vectorize import lin_to_mat, mat_to_vec

numpy.random.seed(1)

n = m = 2

# Generate random problem data
rho_A = density_matrix(n, iscomplex=True)
rho_B = density_matrix(m, iscomplex=True)

# Model problem using primal variable X
# Generate random objective function
C = numpy.random.randn(n * m, n * m) + numpy.random.randn(n * m, n * m) * 1j
c = mat_to_vec(C + C.conj().T)

# Build linear constraints
# tr_A(X) = rho_A
A1 = lin_to_mat(lambda X: p_tr(X, (n, m), 1), (n * m, n), iscomplex=True)
b1 = mat_to_vec(rho_A, compact=True)

# tr_B(X) = rho_B
A2 = lin_to_mat(lambda X: p_tr(X, (n, m), 0), (n * m, m), iscomplex=True)
b2 = mat_to_vec(rho_B, compact=True)

A = numpy.vstack((A1, A2))
b = numpy.vstack((b1, b2))

# Define cones to optimize over
cones = [qics.cones.PosSemidefinite(n * m, iscomplex=True)]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
