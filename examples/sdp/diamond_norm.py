import numpy as np

import qics
import qics.quantum as qu
import qics.vectorize as vec

np.random.seed(1)

n = 2
N = n * n

J1 = qu.random.choi_operator(n, iscomplex=True)
J2 = qu.random.choi_operator(n, iscomplex=True)
J = J1 - J2

# Define objective function
c1 = -0.5 * vec.mat_to_vec(
    np.block([[np.zeros((N, N)), J], [J.conj().T, np.zeros((N, N))]])
)
c2 = np.zeros((2 * n * n, 1))
c3 = np.zeros((2 * n * n, 1))
c = np.vstack((c1, c2, c3))

# Build linear constraints
vN = vec.vec_dim(N, iscomplex=True, compact=True)
submtx_11 = vec.lin_to_mat(lambda X: X[:N, :N], (2 * N, N), iscomplex=True)
submtx_22 = vec.lin_to_mat(lambda X: X[N:, N:], (2 * N, N), iscomplex=True)
i_kr = vec.lin_to_mat(lambda X: qu.i_kr(X, (n, n), 0), (n, N), iscomplex=True)
tr = vec.mat_to_vec(np.eye(n, dtype=np.complex128)).T
# I ⊗ rho block
A1 = np.hstack((submtx_11, -i_kr, np.zeros((vN, 2 * n * n))))
b1 = np.zeros((vN, 1))
# I ⊗ sig block
A2 = np.hstack((submtx_22, np.zeros((vN, 2 * n * n)), -i_kr))
b2 = np.zeros((vN, 1))
# tr[rho] = 1
A3 = np.hstack((np.zeros((1, 8 * N * N)), tr, np.zeros((1, 2 * n * n))))
b3 = np.array([[1.0]])
# tr[sig] = 1
A4 = np.hstack((np.zeros((1, 8 * N * N)), np.zeros((1, 2 * n * n)), tr))
b4 = np.array([[1.0]])

A = np.vstack((A1, A2, A3, A4))
b = np.vstack((b1, b2, b3, b4))

# Define cones to optimize over
cones = [
    qics.cones.PosSemidefinite(2 * n * n, iscomplex=True),
    qics.cones.PosSemidefinite(n, iscomplex=True),
    qics.cones.PosSemidefinite(n, iscomplex=True),
]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
