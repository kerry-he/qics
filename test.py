import numpy as np

import qics
import qics.quantum
from qics.quantum.random import density_matrix
from qics.vectorize import lin_to_mat, vec_dim, mat_to_vec, vec_to_mat

np.random.seed(1)

n = 4
m = 4
N = n * m
alpha = 0.75

vN = vec_dim(N, iscomplex=True)
cm = vec_dim(m, compact=True, iscomplex=True)

# Define random problem data
rhoAB = density_matrix(N, iscomplex=True)
rhoA = qics.quantum.p_tr(rhoAB, (n, m), 1)

# Model problem using primal variables (t, sigB)
# Define objective function
c = np.vstack((np.array([[1.0]]), np.zeros((cm, 1))))

# Build linear constraint tr[X] = 1
trace = lin_to_mat(lambda X: np.trace(X), (m, 1), compact=(True, True), iscomplex=True)
A = np.block([[0.0, trace]])
b = np.array([[1.0]])

# Build conic linear constraints
rhoA_kron = lin_to_mat(lambda X: np.kron(rhoA, X), (m, N), compact=(True, False),
                        iscomplex=True)

G = np.block([
    [-1.0,              np.zeros((1, cm)) ],  # t_tre = t
    [np.zeros((vN, 1)), np.zeros((vN, cm))],  # X_tre = rhoAB
    [np.zeros((vN, 1)), -rhoA_kron        ],  # Y_tre = rhoA x sigB
])

h = np.block([
    [0.0              ],
    [mat_to_vec(rhoAB)],
    [np.zeros((vN, 1))],
])

# Define cones to optimize over
cones = [qics.cones.SandQuasiEntr(N, alpha, True)]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()


# Check if solution satisfies identity
def mpower(X, p):
    D, U = np.linalg.eigh(X)
    return (U * np.power(D, p)) @ U.conj().T

LHS = vec_to_mat(info["x_opt"][1:], iscomplex=True, compact=True)

temp = mpower(np.kron(rhoA, LHS), (1-alpha)/(2*alpha))
temp = mpower(temp @ rhoAB @ temp, alpha)
RHS = qics.quantum.p_tr(temp, (n, m), 0) / np.trace(temp)

print("LHS of identity is:")
print(np.round(LHS, 3))
print("RHS of identity is:")
print(np.round(RHS, 3))