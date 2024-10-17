# Copyright (c) 2024, Kerry He, James Saunderson, and Hamza Fawzi

# This Python package QICS is licensed under the MIT license; see LICENSE.md
# file in the root directory or at https://github.com/kerry-he/qics

import numpy as np

import qics
from qics.quantum.random import density_matrix
from qics.vectorize import lin_to_mat, mat_to_vec

np.random.seed(1)

n = 2

# Generate random problem data
rho = density_matrix(n, iscomplex=True)
sig = density_matrix(n, iscomplex=True)

rho_cvec = mat_to_vec(rho, compact=True)
sig_cvec = mat_to_vec(sig, compact=True)

# Model problem using primal variable M
# Define objective function
eye_n = np.eye(n, dtype=np.complex128)
zero_n = np.zeros((n, n), dtype=np.complex128)
C = np.block([[zero_n, eye_n], [eye_n, zero_n]])
c = -0.5 * qics.vectorize.mat_to_vec(C)

# Build linear constraints M11 = rho and M22 = sig
submat_11 = lin_to_mat(lambda X: X[:n, :n], (2 * n, n), iscomplex=True)
submat_22 = lin_to_mat(lambda X: X[n:, n:], (2 * n, n), iscomplex=True)

A = np.block([[submat_11], [submat_22]])
b = np.block([[rho_cvec], [sig_cvec]])

# Define cones to optimize over
cones = [qics.cones.PosSemidefinite(2 * n, iscomplex=True)]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
