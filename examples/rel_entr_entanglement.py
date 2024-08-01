import numpy as np
import qics
import qics.utils.symmetric as sym
import qics.utils.quantum as qu

## Relative entropy of entanglement
#   min  S(X||Y)
#   s.t. tr[Y] = 1
#        T2(Y) >= 0

n0 = 2
n1 = 3
iscomplex = False

N = n0 * n1
X = qu.rand_density_matrix(N, iscomplex=iscomplex)

sN = sym.vec_dim(N, iscomplex=iscomplex)
vN = sym.vec_dim(N, iscomplex=iscomplex, compact=False)

# Define objective function, where x = (t, triu[Y]) and c = (1, 0)
c1 = np.array(([[1.]]))
c2 = np.zeros((sN, 1))
c  = np.vstack((c1, c2))

# Build linear constraint tr[Y] = 1
A = np.hstack((np.zeros((1, 1)), sym.mat_to_vec(np.eye(N), iscomplex=iscomplex).T))
b = np.ones((1, 1))

# Build linear cone constraints
# t = t
G1 = np.hstack((-np.ones((1, 1)), np.zeros((1, sN))))
h1 = np.zeros((1, 1))
# X = X (const)
G2 = np.hstack((np.zeros((vN, 1)), np.zeros((vN, sN))))
h2 = sym.mat_to_vec(X, iscomplex=iscomplex, compact=False)
# Y = Y
eye = sym.lin_to_mat(lambda X : X, (N, N), iscomplex=iscomplex, compact=(True, False))
G3 = np.hstack((np.zeros((vN, 1)), -eye))
h3 = np.zeros((vN, 1))
# T2(Y) >= 0
p_transpose = sym.lin_to_mat(lambda X : sym.p_transpose(X, 1, (n0, n1)), (N, N), iscomplex=iscomplex, compact=(True, False))
G4 = np.hstack((np.zeros((vN, 1)), -p_transpose))
h4 = np.zeros((vN, 1))

G = np.vstack((G1, G2, G3, G4))
h = np.vstack((h1, h2, h3, h4))

# Input into model and solve
cones = [
    qics.cones.QuantRelEntr(N, iscomplex=iscomplex), 
    qics.cones.PosSemidefinite(N, iscomplex=iscomplex)
]

# Initialize model and solver objects
model  = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
solver = qics.Solver(model)

# Solve problem
out = solver.solve()