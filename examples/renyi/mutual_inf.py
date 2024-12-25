# Copyright (c) 2024, Kerry He, James Saunderson, and Hamza Fawzi

# This Python package QICS is licensed under the MIT license; see LICENSE.md
# file in the root directory or at https://github.com/kerry-he/qics

import numpy as np

import qics
import qics.quantum
from qics.quantum.random import density_matrix
from qics.vectorize import lin_to_mat, vec_dim, mat_to_vec, vec_to_mat

np.random.seed(1)

n = 4
m = 4
N = n * m

vN = vec_dim(N, iscomplex=True)
cm = vec_dim(m, compact=True, iscomplex=True)

# Define random problem data
rhoAB = density_matrix(N, iscomplex=True)
tauA = qics.quantum.p_tr(rhoAB, (n, m), 1)

# Model problem using primal variables (t, sigB)
# Define objective function
c = np.vstack((np.array([[1.0]]), np.zeros((cm, 1))))

# Build linear constraint tr[X] = 1
trace = lin_to_mat(lambda X: np.trace(X), (m, 1), compact=(True, True), iscomplex=True)
A = np.block([[0.0, trace]])
b = np.array([[1.0]])

# Build conic linear constraints
tauA_kron = lin_to_mat(lambda X: np.kron(tauA, X), (m, N), compact=(True, False), iscomplex=True)

G = np.block([
    [-1.0,              np.zeros((1, cm)) ],   # t_sre = t
    [np.zeros((vN, 1)), np.zeros((vN, cm))],   # X_sre = rhoAB
    [np.zeros((vN, 1)), -tauA_kron         ],  # Y_sre = tauA x sigB
])  # fmt: skip

h = np.block([
    [0.0], 
    [mat_to_vec(rhoAB)], 
    [np.zeros((vN, 1))], 
])  # fmt: skip

# Define cones to optimize over
alpha = 1.5
cones = [qics.cones.SandRenyiEntr(N, alpha, True)]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
solver = qics.Solver(model, verbose=3)

# Solve problem
info = solver.solve()


# Check if satisfies identity by: https://doi.org/10.1063/1.4964755
def mpower(X, p):
    D, U = np.linalg.eigh(X)
    return (U * np.power(D, p)) @ U.conj().T

sigB = vec_to_mat(info["x_opt"][1:], iscomplex=True, compact=True)

temp = mpower(np.kron(tauA, sigB), (1-alpha)/(2*alpha))
temp = mpower(temp @ rhoAB @ temp, alpha)

temp = qics.quantum.p_tr(temp, (n, m), 0) / np.trace(temp)

print(np.linalg.norm(sigB - temp))