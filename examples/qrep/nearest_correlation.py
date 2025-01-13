# Copyright (c) 2024, Kerry He, James Saunderson, and Hamza Fawzi

# This Python package QICS is licensed under the MIT license; see LICENSE.md
# file in the root directory or at https://github.com/kerry-he/qics

import numpy as np

import qics
from qics.vectorize import vec_dim, mat_to_vec, eye

np.random.seed(1)

n = 5

vn = vec_dim(n)
cn = vec_dim(n, compact=True)

# Generate random positive semidefinite matrix C
C = np.random.randn(n, n)
C = C @ C.T / n
C_cvec = mat_to_vec(C, compact=True)

# Model problem using primal variables (t, X, Y)
# Define objective function
c = np.block([[1.0], [np.zeros((vn, 1))], [np.zeros((vn, 1))]])

# Build linear constraints
diag = np.zeros((n, vn))
diag[np.arange(n), np.arange(0, vn, n + 1)] = 1.0

A = np.block([
    [np.zeros((cn, 1)), eye(n),            np.zeros((cn, vn))],  # X = C
    [np.zeros((n, 1)),  np.zeros((n, vn)), diag              ]   # Yii = 1
])  # fmt: skip

b = np.block([[C_cvec], [np.ones((n, 1))]])

# Define cones to optimize over
cones = [qics.cones.QuantRelEntr(n)]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
