import numpy as np
import qics
import qics.utils.symmetric as sym
import qics.utils.quantum as qu

## Nearest correlation matrix
#   min  S(X||Y)
#   s.t. Y_ii = 1

n = 25
iscomplex = False

X = qu.rand_density_matrix(n, iscomplex=iscomplex)

sn = sym.vec_dim(n, iscomplex=iscomplex)
vn = sym.vec_dim(n, iscomplex=iscomplex, compact=False)

# Define objective function, where x = (t, triu[Y]) and c = (1, 0)
c1 = np.array(([[1.]]))
c2 = np.zeros((sn, 1))
c  = np.vstack((c1, c2))

# Build linear constraint Y_ii = 1
diag_idxs = np.arange(3, 2*n, 2) if iscomplex else np.arange(2, 1+n)
diag_idxs = np.insert(np.cumsum(diag_idxs), 0, 0)
A = np.zeros((n, 1 + sn))
A[np.arange(n), 1 + diag_idxs] = 1.

b = np.ones((n, 1))

# Build linear cone constraints
# t = t
G1 = np.hstack((-np.ones((1, 1)), np.zeros((1, sn))))
h1 = np.zeros((1, 1))
# X = X (const)
G2 = np.hstack((np.zeros((vn, 1)), np.zeros((vn, sn))))
h2 = sym.mat_to_vec(X, iscomplex=iscomplex, compact=False)
# Y = Y
eye = sym.lin_to_mat(lambda X : X, (n, n), iscomplex=iscomplex, compact=(True, False))
G3 = np.hstack((np.zeros((vn, 1)), -eye))
h3 = np.zeros((vn, 1))

G = np.vstack((G1, G2, G3))
h = np.vstack((h1, h2, h3))

# Input into model and solve
cones = [qics.cones.QuantRelEntr(n, iscomplex=iscomplex)]

# Initialize model and solver objects
model  = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
solver = qics.Solver(model)

# Solve problem
out = solver.solve()