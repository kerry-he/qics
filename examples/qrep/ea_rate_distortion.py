# Copyright (c) 2024, Kerry He, James Saunderson, and Hamza Fawzi

# This Python package QICS is licensed under the MIT license; see LICENSE.md
# file in the root directory or at https://github.com/kerry-he/qics

import numpy as np

import qics
from qics.quantum import p_tr, purify, entropy
from qics.quantum.random import density_matrix
from qics.vectorize import lin_to_mat, vec_dim, mat_to_vec

np.random.seed(1)

n = 4
N = n * n
vN = vec_dim(N, iscomplex=True)
cn = vec_dim(n, iscomplex=True, compact=True)

# Define random problem data
rho = density_matrix(n, iscomplex=True)
entr_rho = entropy(rho)
rho_cvec = mat_to_vec(rho, compact=True)

D = 0.25
Delta = np.eye(N) - purify(rho)
Delta_vec = mat_to_vec(Delta)

# Model problem using primal variables (t, X, d)
# Define objective function
c = np.block([[1.0], [np.zeros((vN, 1))], [0.0]])

# Build linear constraint matrices
tr_B = lin_to_mat(lambda X: p_tr(X, (n, n), 1), (N, n), iscomplex=True)

A = np.block([
    [np.zeros((cn, 1)), tr_B,        np.zeros((cn, 1))],  # tr_B[X] = rho
    [0.0,               Delta_vec.T, 1.0              ]   # d = D - <Delta, X>
])  # fmt: skip

b = np.block([[rho_cvec], [D]])

# Define cones to optimize over
cones = [
    qics.cones.QuantCondEntr((n, n), 0, iscomplex=True),  # (t, X) ∈ QCE
    qics.cones.NonNegOrthant(1),  # d = D - <Delta, X> >= 0
]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, cones=cones, offset=entr_rho)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
