# Copyright (c) 2024, Kerry He, James Saunderson, and Hamza Fawzi

# This Python package QICS is licensed under the MIT license; see LICENSE.md
# file in the root directory or at https://github.com/kerry-he/qics

import numpy as np

import qics
from qics.vectorize import eye, lin_to_mat, mat_to_vec, vec_dim

np.random.seed(1)

n = 10
m = 20
k = 5
eps = 1e-6

vn = vec_dim(n)
cn = vec_dim(n, compact=True)

# Define random problem data
A_dat = 1 / (n**0.25) * np.random.randn(n, m)

# Model problem using primal variables (t, z, cvec(Y))
# Define objective function
c_t = np.ones((1, 1))
c_z = np.zeros((m, 1))
c_Y = mat_to_vec(np.eye(n), compact=True) * np.log(eps)
c = np.block([[c_t], [c_z], [c_Y]])

# Build linear constraint Σ_i zi = k
A = np.block([0.0, np.ones((1, m)), np.zeros((1, cn))])
b = np.array([[k]])

# Build linear cone constraints
trace = lin_to_mat(lambda X: np.trace(X), (n, 1), compact=(True, False))
AdiagA = np.hstack([mat_to_vec(A_dat[:, [i]] @ A_dat[:, [i]].T) for i in range(m)])

G = np.block([
    [0.0,               np.zeros((1, m)),  trace            ],  # tr[Y] <= k
    [np.zeros((m, 1)),  -np.eye(m),        np.zeros((m, cn))],  # 0 <= z
    [np.zeros((m, 1)),  np.eye(m),         np.zeros((m, cn))],  # z <= 1
    [np.zeros((vn, 1)), np.zeros((vn, m)), eye(n).T         ],  # Y <= I
    [-1.0,              np.zeros((1, m)),  np.zeros((1, cn))],  # t_ore = t
    [np.zeros((vn, 1)), np.zeros((vn, m)), -eye(n).T        ],  # X_ore = Y
    [np.zeros((vn, 1)), -AdiagA,           -eye(n).T * eps  ]   # Y_ore = Adiag(z)A' + eY
])  # fmt: skip

h = np.block([
    [k],
    [np.zeros((m, 1))],
    [np.ones((m, 1))],
    [mat_to_vec(np.eye(n))],
    [0.0],
    [np.zeros((vn, 1))],
    [np.zeros((vn, 1))]
])  # fmt: skip

# Define cones to optimize over
cones = [
    qics.cones.NonNegOrthant(1 + m + m),  # (tr[Y], -z, z - 1) >= 0
    qics.cones.PosSemidefinite(n),  # Y ⪰ 0
    qics.cones.OpPerspecTr(n, "log"),  # (t, Y, Adiag(z)A' + eY) ∈ ORE
]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones, offset=-n * np.log(eps))
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
