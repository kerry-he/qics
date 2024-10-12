# Copyright (c) 2024, Kerry He, James Saunderson, and Hamza Fawzi

# This Python package QICS is licensed under the MIT license; see LICENSE.md
# file in the root directory or at https://github.com/kerry-he/qics

import numpy as np

import qics

np.random.seed(1)

n = 5
m = 4
vn = qics.vectorize.vec_dim(n, iscomplex=True)

# Generate random linear objective function
U = np.random.randn(n, m) + np.random.randn(n, m)*1j
v = np.random.randn(n)
C = np.diag(v) @ (np.eye(n) - U @ U.conj().T) @ np.diag(v)
c = qics.vectorize.mat_to_vec(C)

# Build linear constraints  Xii = 1 for all i
A = np.zeros((n, vn))
A[np.arange(n), np.arange(0, vn, 2 * n + 2)] = 1.

b = np.ones((n, 1))

# Define cones to optimize over
cones = [qics.cones.PosSemidefinite(n, iscomplex=True)]

# Initialize model and solver objects
model  = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
