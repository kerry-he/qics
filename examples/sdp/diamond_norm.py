import numpy as np

import qics
from qics.quantum import i_kr
from qics.quantum.random import choi_operator
from qics.vectorize import lin_to_mat, mat_to_vec, vec_dim

np.random.seed(1)

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
C_M = np.block([[np.zeros((N, N)), J], [J.conj().T, np.zeros((N, N))]])

c_M = -0.5 * mat_to_vec(C_M)
c_rho = np.zeros((vn, 1))
c_sig = np.zeros((vn, 1))
c = np.block([[c_M], [c_rho], [c_sig]])

# Build linear constraints
trace = lin_to_mat(lambda X: np.trace(X), (n, 1), iscomplex=True)
ikr_1 = lin_to_mat(lambda X: i_kr(X, (n, n), 0), (n, N), iscomplex=True)
submat_11 = lin_to_mat(lambda X: X[:N, :N], (2 * N, N), iscomplex=True)
submat_22 = lin_to_mat(lambda X: X[N:, N:], (2 * N, N), iscomplex=True)

A = np.block([
    [submat_11,             -ikr_1,             np.zeros((cN, vn))],  # M11 = I ⊗ rho
    [submat_22,             np.zeros((cN, vn)), -ikr_1            ],  # M22 = I ⊗ sig
    [np.zeros((1, 4 * vN)), trace,              np.zeros((1, vn)) ],  # tr[rho] = 1
    [np.zeros((1, 4 * vN)), np.zeros((1, vn)),  trace             ]   # tr[sig] = 1
])  # fmt: skip

b = np.block([[np.zeros((cN, 1))], [np.zeros((cN, 1))], [1.0], [1.0]])

# Define cones to optimize over
cones = [
    qics.cones.PosSemidefinite(2 * N, iscomplex=True),  # M ⪰ 0
    qics.cones.PosSemidefinite(n, iscomplex=True),      # rho ⪰ 0
    qics.cones.PosSemidefinite(n, iscomplex=True),      # sig ⪰ 0
]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
