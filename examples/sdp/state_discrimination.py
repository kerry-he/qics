# Copyright (c) 2024, Kerry He, James Saunderson, and Hamza Fawzi

# This Python package QICS is licensed under the MIT license; see LICENSE.md
# file in the root directory or at https://github.com/kerry-he/qics

import numpy as np

import qics

n = 2
p1 = p2 = 0.5

eye_mtx = qics.vectorize.eye(n)
rho_1 = qics.vectorize.mat_to_vec(np.array([[1.0, 0.0], [0.0, 0.0]]))
rho_2 = qics.vectorize.mat_to_vec(np.array([[0.0, 0.0], [0.0, 1.0]]))

# Model problem using primal variables (E1, E2)
# Define objective function
c = -np.block([[p1 * rho_1], [p2 * rho_2]])

# Build linear constraint E1 + E2 = I
A = np.block([eye_mtx, eye_mtx])
b = qics.vectorize.mat_to_vec(np.eye(n), compact=True)

# Define cones to optimize over
cones = [qics.cones.PosSemidefinite(n), qics.cones.PosSemidefinite(n)]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
