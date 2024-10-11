import numpy as np

import qics

## Quantum key rate
#   min  S( G(X) || Z(G(X)) )
#   s.t. <Ai, X> = bi
#        X >= 0

qx = qz = 0.5

# Define objective function
c = np.vstack((np.array([[1.0]]), np.zeros((16, 1))))

# Build linear constraints
X0 = np.array([[0.5, 0.5], [0.5, 0.5]])
X1 = np.array([[0.5, -0.5], [-0.5, 0.5]])
Z0 = np.array([[1.0, 0.0], [0.0, 0.0]])
Z1 = np.array([[0.0, 0.0], [0.0, 1.0]])

Ax = np.kron(X0, X1) + np.kron(X1, X0)
Az = np.kron(Z0, Z1) + np.kron(Z1, Z0)

A = np.vstack(
    (
        np.hstack((np.array([[0.0]]), np.eye(4).reshape(1, -1))),
        np.hstack((np.array([[0.0]]), Ax.reshape(1, -1))),
        np.hstack((np.array([[0.0]]), Az.reshape(1, -1))),
    )
)

b = np.array([[1.0, qx, qz]]).T

# Input into model and solve
cones = [qics.cones.QuantKeyDist([np.eye(4)], 2)]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
