import numpy as np

import qics
from qics.quantum import entropy
from qics.quantum.random import density_matrix
from qics.vectorize import mat_to_vec, vec_dim

np.random.seed(1)

n = m = 16
vn = vec_dim(n, iscomplex=True)

# Generate random problem data
rhos = [density_matrix(n, iscomplex=True) for _ in range(m)]
rho_vecs = np.hstack(([mat_to_vec(rho) for rho in rhos]))

# Model problem using primal variables (p, t)
# Define objective function
c_p = np.array([[-entropy(rho)] for rho in rhos])
c_t = 1.0
c = np.block([[c_p], [c_t]])

# Build linear constraint Σ_i pi = 1
A = np.block([np.ones((1, m)), 0.0])
b = np.array([[1.0]])

# Build linear cone constraints
G = np.block([
    [-np.eye(m),       np.zeros((m, 1)) ],  # x_nn = p
    [np.zeros((1, m)), -np.ones((1, 1)) ],  # t_qe = t
    [np.zeros((1, m)), np.zeros((1, 1)) ],  # u_qe = 1
    [-rho_vecs,        np.zeros((vn, 1))]   # X_qe = Σ_i pi N(Xi)
])  # fmt: skip

h = np.block([[np.zeros((m, 1))], [0.0], [1.0], [np.zeros((vn, 1))]])

# Define cones to optimize over
cones = [
    qics.cones.NonNegOrthant(n),  # p >= 0
    qics.cones.QuantEntr(n, iscomplex=True),  # (t, 1, Σ_i pi N(Xi)) ∈ QE
]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
