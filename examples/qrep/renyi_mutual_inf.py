# Copyright (c) 2024, Kerry He, James Saunderson, and Hamza Fawzi

# This Python package QICS is licensed under the MIT license; see LICENSE.md
# file in the root directory or at https://github.com/kerry-he/qics

import numpy as np

import qics
from qics.quantum.random import density_matrix
from qics.vectorize import lin_to_mat, vec_dim, mat_to_vec

np.random.seed(1)

n = 4
N = n * n

vN = vec_dim(N, iscomplex=True)
cn = vec_dim(n, compact=True, iscomplex=True)

# Define random problem data
rhoAB = density_matrix(N, iscomplex=True)
tauA = density_matrix(n, iscomplex=True)

# Model problem using primal variables (t, sigB)
# Define objective function
c = np.vstack((np.array([[1.0]]), np.zeros((cn, 1))))

# Build linear constraint tr[X] = 1
trace = lin_to_mat(lambda X: np.trace(X), (n, 1), compact=(True, True), iscomplex=True)
A = np.block([[0.0, trace]])
b = np.array([[1.0]])

# Build conic linear constraints
tauA_kron = lin_to_mat(lambda X: np.kron(tauA, X), (n, N), compact=(True, False), iscomplex=True)

G = np.block([
    [-1.0,              np.zeros((1, cn)) ],   # t_sre = t
    [np.zeros((vN, 1)), np.zeros((vN, cn))],   # X_sre = rhoAB
    [np.zeros((vN, 1)), -tauA_kron         ],  # Y_sre = tauA x sigB
])  # fmt: skip

h = np.block([
    [0.0], 
    [mat_to_vec(rhoAB)], 
    [np.zeros((vN, 1))], 
])  # fmt: skip

# Define cones to optimize over
cones = [qics.cones.SandRenyiEntr(N, 0.75, True)]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
solver = qics.Solver(model, verbose=3)

# Solve problem
info = solver.solve()
