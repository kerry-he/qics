import numpy as np
import qics
import qics.vectorize as vec
import qics.quantum as qu

## Classical-quantum channel capacity
#   max  S(Σ_i pi N(Xi)) - Σ_i pi S(N(Xi))
#   s.t. Σ_i pi = 1
#        p >= 0

np.random.seed(1)

n = m = 16

rhos = [qu.random.density_matrix(n, iscomplex=True) for i in range(n)]

# Define objective function
# where x = ({pi}, t) and c = ({-S(N(Xi))}, 1)
c1 = np.array([[-qu.quant_entropy(rho)] for rho in rhos])
c2 = np.array([[1.0]])
c  = np.vstack((c1, c2))

# Build linear constraint Σ_i pi = 1
A = np.hstack((np.ones((1, n)), np.zeros((1, 1))))
b = np.ones((1, 1))

# Build linear cone constraints
# x_nn = p
G1 = np.hstack((-np.eye(n), np.zeros((n, 1))))
h1 = np.zeros((n, 1))
# t_qe = t
G2 = np.hstack((np.zeros((1, n)), -np.ones((1, 1))))
h2 = np.zeros((1, 1))
# u_qe = 1
G3 = np.hstack((np.zeros((1, n)), np.zeros((1, 1))))
h3 = np.ones((1, 1))
# X_qe = Σ_i pi N(Xi)
rhos_vec = np.hstack(([vec.mat_to_vec(rho) for rho in rhos]))
G4 = np.hstack((-rhos_vec, np.zeros((2*n*n, 1))))
h4 = np.zeros((2*n*n, 1))

G = np.vstack((G1, G2, G3, G4))
h = np.vstack((h1, h2, h3, h4))

# Input into model and solve
cones = [
    qics.cones.NonNegOrthant(n), 
    qics.cones.QuantEntr(n, iscomplex=True)
]

# Initialize model and solver objects
model  = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
