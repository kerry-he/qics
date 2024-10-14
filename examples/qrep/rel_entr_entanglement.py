# Copyright (c) 2024, Kerry He, James Saunderson, and Hamza Fawzi

# This Python package QICS is licensed under the MIT license; see LICENSE.md
# file in the root directory or at https://github.com/kerry-he/qics

import numpy as np

import qics
from qics.quantum import partial_transpose
from qics.quantum.random import density_matrix
from qics.vectorize import eye, lin_to_mat, mat_to_vec, vec_dim

np.random.seed(1)

n1 = 2
n2 = 3
N = n1 * n2

vN = vec_dim(N, iscomplex=True)
cN = vec_dim(N, iscomplex=True, compact=True)

# Generate random quantum state
C = density_matrix(N, iscomplex=True)
C_cvec = mat_to_vec(C, compact=True)

# Model problem using primal variables (t, X, Y, Z)
# Define objective function
c = np.block([[1.0], [np.zeros((vN, 1))], [np.zeros((vN, 1))], [np.zeros((vN, 1))]])

# Build linear constraints
trace = lin_to_mat(lambda X: np.trace(X), (N, 1), True)
ptranspose = lin_to_mat(lambda X: partial_transpose(X, (n1, n2), 1), (N, N), True)

A = np.block([
    [np.zeros((cN, 1)), eye(N, True),       np.zeros((cN, vN)), np.zeros((cN, vN))],  # X = C
    [np.zeros((1, 1)),  np.zeros((1, vN)),  trace,              np.zeros((1, vN)) ],  # tr[Y] = 1
    [np.zeros((cN, 1)), np.zeros((cN, vN)), ptranspose,         -eye(N, True)     ]   # T2(Y) = Z
])  # fmt: skip

b = np.block([[C_cvec], [1.0], [np.zeros((cN, 1))]])

# Input into model and solve
cones = [
    qics.cones.QuantRelEntr(N, iscomplex=True),     # (t, X, Y) ∈ QRE
    qics.cones.PosSemidefinite(N, iscomplex=True),  # Z = T2(Y) ⪰ 0
]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
