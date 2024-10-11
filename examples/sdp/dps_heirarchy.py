import numpy as np

import qics
from qics.quantum import p_tr, partial_transpose, swap
from qics.vectorize import eye, lin_to_mat, mat_to_vec, vec_dim

n = 2
n2 = n * n
n3 = n * n * n

vn3 = vec_dim(n3)
cn2 = vec_dim(n2, compact=True)
cn3 = vec_dim(n3, compact=True)

# Define an entangled quantum state
rho = 0.5 * np.array([[1.0, 0.0, 0.0, 1.0],
                      [0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0, 1.0]])  # fmt: skip

# Model problem using primal variables (rho_aB, sigma_aB, omega_aB)
# Define objective function
c = np.zeros((3 * vn3, 1))

# Build linear constraints
trace = lin_to_mat(lambda X: np.trace(X), (n3, 1))
ptr_b2 = lin_to_mat(lambda X: p_tr(X, (n, n, n), 2), (n3, n2))
swap_b1b2 = lin_to_mat(lambda X: swap(X, (n, n, n), 1, 2), (n3, n3))
T_b2 = lin_to_mat(lambda X: partial_transpose(X, (n2, n), 1), (n3, n3))
T_b1b2 = lin_to_mat(lambda X: partial_transpose(X, (n, n2), 1), (n3, n3))

A = np.block([
    [ptr_b2,              np.zeros((cn2, vn3)), np.zeros((cn2, vn3))],  # tr_b2(rho_aB) = rho
    [swap_b1b2 - eye(n3), np.zeros((cn3, vn3)), np.zeros((cn3, vn3))],  # swap_b1b2(rho_aB) = rho_aB
    [trace,               np.zeros((1, vn3)),   np.zeros((1, vn3))  ],  # tr[rho_aB] = 1
    [T_b2,                -eye(n3),             np.zeros((cn3, vn3))],  # sigma_aB = T_b2(rho_aB)
    [T_b1b2,              np.zeros((cn3, vn3)), -eye(n3)            ]   # omega_aB = T_b1b2(rho_aB)
])  # fmt: skip

b = np.block([
    [mat_to_vec(rho, compact=True)], 
    [np.zeros((cn3, 1))], 
    [1.0], 
    [np.zeros((cn3, 1))], 
    [np.zeros((cn3, 1))]
])  # fmt: skip

# Define cones to optimize over
cones = [
    qics.cones.PosSemidefinite(n3),  # rho_aB ⪰ 0
    qics.cones.PosSemidefinite(n3),  # sigma_aB = T_b2(rho_aB) ⪰ 0
    qics.cones.PosSemidefinite(n3),  # omega_aB = T_b1b2(rho_aB) ⪰ 0
]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
