import numpy as np
import qics
import qics.utils.symmetric as sym
import qics.utils.quantum as qu

## Classical-quantum channel capacity
#   max  S(Σ_i pi N(Xi)) - Σ_i pi S(N(Xi))
#   s.t. Σ_i pi = 1
#        p >= 0

n = 64
iscomplex = False

alphabet = [qu.rand_density_matrix(n, iscomplex=iscomplex) for i in range(n)]
vn       = sym.vec_dim(n, iscomplex=iscomplex, compact=False)

# Define objective function, where x = ({pi}, t) and c = ({-S(N(Xi))}, 1)
c1 = np.array([[-qu.quant_entropy(rho)] for rho in alphabet])
c2 = np.array([[1.]])
c  = np.vstack((c1, c2))

# Build linear constraint Σ_i pi = 1
A = np.hstack((np.ones((1, n)), np.zeros((1, 1))))
b = np.ones((1, 1))

# Build linear cone constraints
# p >= 0
G1 = np.hstack((-np.eye(n), np.zeros((n, 1))))
h1 = np.zeros((n, 1))
# t = t
G2 = np.hstack((np.zeros((1, n)), -np.ones((1, 1))))
h2 = np.zeros((1, 1))
# u = 1
G3 = np.hstack((np.zeros((1, n)), np.zeros((1, 1))))
h3 = np.ones((1, 1))
# X = Σ_i pi N(Xi)
alphabet_vec = np.hstack(([sym.mat_to_vec(rho, iscomplex=iscomplex, compact=False) for rho in alphabet]))
G4 = np.hstack((-alphabet_vec, np.zeros((vn, 1))))
h4 = np.zeros((vn, 1))

G = np.vstack((G1, G2, G3, G4))
h = np.vstack((h1, h2, h3, h4))

# Input into model and solve
cones = [
    qics.cones.NonNegOrthant(n), 
    qics.cones.QuantEntr(n, iscomplex=iscomplex)
]

# Initialize model and solver objects
model  = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
solver = qics.Solver(model)

# Solve problem
out = solver.solve()