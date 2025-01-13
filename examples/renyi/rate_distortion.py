# Copyright (c) 2024, Kerry He, James Saunderson, and Hamza Fawzi

# This Python package QICS is licensed under the MIT license; see LICENSE.md
# file in the root directory or at https://github.com/kerry-he/qics

import numpy as np

import qics
from qics.quantum import p_tr, purify
from qics.vectorize import lin_to_mat, vec_dim, mat_to_vec, eye

np.random.seed(1)

n = 4
N = n * n

vN = vec_dim(N)
cN = vec_dim(N, compact=True)
cn = vec_dim(n, compact=True)

# Define random problem data
D = 0.25
Delta = np.eye(N) - purify(np.eye(n) / n)
Delta_vec = mat_to_vec(Delta, compact=True)

# Model problem using primal variables (t, sigB)
# Define objective function
c = np.vstack((np.array([[1.0]]), np.zeros((cN, 1))))

# Build linear constraint tr[X] = 1
ptrace = lin_to_mat(lambda X: p_tr(X, (n, n), 1), (N, n), compact=(True, True))
A = np.block([[np.zeros((cn, 1)), ptrace]])
b = mat_to_vec(np.eye(n) / n, compact=True)

# Build conic linear constraints
I_kr = lin_to_mat(
    lambda X: np.kron(np.eye(n), p_tr(X, (n, n), 0)), (N, N), compact=(True, False)
)

G = np.block([
    [-1.0,              np.zeros((1, cN))],   # t_sre = t
    [np.zeros((vN, 1)), -eye(N).T        ],   # X_sre = rhoAB
    [np.zeros((vN, 1)), -I_kr            ],   # Y_sre = tauA x sigB
    [0.0,                Delta_vec.T     ]
])  # fmt: skip

h = np.block([
    [0.0              ], 
    [np.zeros((vN, 1))], 
    [np.zeros((vN, 1))],
    [D                ], 
])  # fmt: skip

# Define cones to optimize over
alpha = 1.01
cones = [qics.cones.SandQuasiEntr(N, alpha), qics.cones.NonNegOrthant(1)]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
