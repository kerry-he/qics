import numpy as np
import qics

## Quantum key rate
#   min  S( G(X) || Z(G(X)) )
#   s.t. <Ai, X> = bi
#        X >= 0

n = 4

# Data for ebBB84 obtained from: 
# https://www.math.uwaterloo.ca/~hwolkowi/henry/reports/ZGNQKDmainsolverUSEDforPUBLCNJuly31/
K_list = [
    np.array([
        [.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, .1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .1, 0]
    ]).T,

    np.array([
        [0, .45, 0, 0, 0, .45, 0, 0, 0, .45, 0, 0, 0, -.45, 0, 0],
        [0, 0, 0, .45, 0, 0, 0, .45, 0, 0, 0, .45, 0, 0, 0, -.45],
        [0, .45, 0, 0, 0, .45, 0, 0, 0, -.45, 0, 0, 0, .45, 0, 0],
        [0, 0, 0, .45, 0, 0, 0, .45, 0, 0, 0, -.45, 0, 0, 0, .45]
    ]).T
]

Gamma_list = [
    np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [ 0, 0, 0, 0]]),
    np.array([[1, 0, 0,-1], [0, 1,-1, 0], [0,-1, 1, 0], [-1, 0, 0, 1]]) / 2,
    np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0,-1], [ 0, 0,-1, 0]]),
    np.array([[0, 0, 1, 0], [0, 0, 0,-1], [1, 0, 0, 0], [ 0,-1, 0, 0]]),
    np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [ 0, 0, 0, 1]])
]

gamma = np.array([[0.03, 0.03, 0, 0, 1]]).T

# Define objective function
c = np.zeros((1 + n*n, 1))
c[0] = 1.

# Build linear constraints
A = np.hstack((np.zeros((5, 1)), np.array(Gamma_list).reshape(-1, n*n)))
b = gamma

# Input into model and solve
cones = [qics.cones.QuantKeyDist(K_list, 2)]

# Initialize model and solver objects
model  = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
out = solver.solve()