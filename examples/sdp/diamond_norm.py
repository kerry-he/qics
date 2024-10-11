import numpy

import qics
from qics.quantum import i_kr
from qics.quantum.random import choi_operator
from qics.vectorize import lin_to_mat, mat_to_vec, vec_dim

numpy.random.seed(1)

n = 4
N = n * n
vn = vec_dim(n, iscomplex=True)
vN = vec_dim(N, iscomplex=True)
cN = vec_dim(N, iscomplex=True, compact=True)

# Generate random problem data
J1 = choi_operator(n, iscomplex=True)
J2 = choi_operator(n, iscomplex=True)
J = J1 - J2

# Model problem using primal variables (M, rho, sig)
# Define objective function
C_M = numpy.block([[numpy.zeros((N, N)), J], [J.conj().T, numpy.zeros((N, N))]])

c_M = -0.5 * mat_to_vec(C_M)
c_rho = numpy.zeros((vn, 1))
c_sig = numpy.zeros((vn, 1))
c = numpy.vstack((c_M, c_rho, c_sig))

# Build linear constraints
tr_mat = lin_to_mat(lambda X: numpy.trace(X), (n, 1), iscomplex=True)
i_kr_mat = lin_to_mat(lambda X: i_kr(X, (n, n), 0), (n, N), iscomplex=True)
submat_11_mat = lin_to_mat(lambda X: X[:N, :N], (2 * N, N), iscomplex=True)
submat_22_mat = lin_to_mat(lambda X: X[N:, N:], (2 * N, N), iscomplex=True)

# M11 = I ⊗ rho
A1 = numpy.hstack((submat_11_mat, -i_kr_mat, numpy.zeros((cN, vn))))
b1 = numpy.zeros((cN, 1))

# M22 = I ⊗ sig
A2 = numpy.hstack((submat_22_mat, numpy.zeros((cN, vn)), -i_kr_mat))
b2 = numpy.zeros((cN, 1))

# tr[rho] = 1
A3 = numpy.hstack((numpy.zeros((1, 4 * vN)), tr_mat, numpy.zeros((1, vn))))
b3 = numpy.array([[1.0]])

# tr[sig] = 1
A4 = numpy.hstack((numpy.zeros((1, 4 * vN)), numpy.zeros((1, vn)), tr_mat))
b4 = numpy.array([[1.0]])

A = numpy.vstack((A1, A2, A3, A4))
b = numpy.vstack((b1, b2, b3, b4))

# Define cones to optimize over
cones = [
    qics.cones.PosSemidefinite(2 * N, iscomplex=True),  # M ⪰ 0
    qics.cones.PosSemidefinite(n, iscomplex=True),  # rho ⪰ 0
    qics.cones.PosSemidefinite(n, iscomplex=True),  # sig ⪰ 0
]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
