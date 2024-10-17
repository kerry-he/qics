# Copyright (c) 2024, Kerry He, James Saunderson, and Hamza Fawzi

# This Python package QICS is licensed under the MIT license; see LICENSE.md
# file in the root directory or at https://github.com/kerry-he/qics

import numpy as np

import qics
from qics.quantum import p_tr
from qics.quantum.random import density_matrix
from qics.vectorize import lin_to_mat, mat_to_vec

np.random.seed(1)

n = m = 2

# Generate random problem data
rho_A = density_matrix(n, iscomplex=True)
rho_B = density_matrix(m, iscomplex=True)

rho_A_cvec = mat_to_vec(rho_A, compact=True)
rho_B_cvec = mat_to_vec(rho_B, compact=True)

# Model problem using primal variable X
# Generate random objective function
C = np.random.randn(n * m, n * m) + np.random.randn(n * m, n * m) * 1j
c = mat_to_vec(C + C.conj().T)

# Build linear constraints tr_A(X) = rho_A and tr_B(X) = rho_B
ptr_A = lin_to_mat(lambda X: p_tr(X, (n, m), 1), (n * m, n), iscomplex=True)
ptr_B = lin_to_mat(lambda X: p_tr(X, (n, m), 0), (n * m, m), iscomplex=True)

A = np.block([[ptr_A], [ptr_B]])
b = np.block([[rho_A_cvec], [rho_B_cvec]])

# Define cones to optimize over
cones = [qics.cones.PosSemidefinite(n * m, iscomplex=True)]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
