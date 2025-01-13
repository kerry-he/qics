# Copyright (c) 2024, Kerry He, James Saunderson, and Hamza Fawzi

# This Python package QICS is licensed under the MIT license; see LICENSE.md
# file in the root directory or at https://github.com/kerry-he/qics

import numpy as np

import qics
from qics.vectorize import mat_to_vec

qx = 0.25
qz = 0.75

X0 = np.array([[0.5, 0.5], [0.5, 0.5]])
X1 = np.array([[0.5, -0.5], [-0.5, 0.5]])
Z0 = np.array([[1.0, 0.0], [0.0, 0.0]])
Z1 = np.array([[0.0, 0.0], [0.0, 1.0]])

# Model problem using primal variables (t, X)
# Define objective function
c = np.vstack((np.array([[1.0]]), np.zeros((16, 1))))

# Build linear constraints <Ai, X> = bi for all i
Ax = np.kron(X0, X1) + np.kron(X1, X0)
Az = np.kron(Z0, Z1) + np.kron(Z1, Z0)
A_mats = [np.eye(4), Ax, Az]

A = np.block([[0.0, mat_to_vec(Ak).T] for Ak in A_mats])
b = np.array([[1.0], [qx], [qz]])

# Input into model and solve
cones = [qics.cones.QuantKeyDist(4, 2)]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
