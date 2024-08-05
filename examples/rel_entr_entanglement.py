import numpy as np
import qics
import qics.utils.symmetric as sym
import qics.utils.quantum as qu

## Relative entropy of entanglement
#   min  S(X||Y)
#   s.t. tr[Y] = 1
#        T2(Y) >= 0

np.random.seed(1)

n1 = 2
n2 = 3
N  = n1 * n2

# Generate random (complex) quantum state
C = qu.rand_density_matrix(N, iscomplex=True)

# Define objective function
ct = np.array(([[1.]]))
cX = np.zeros((2*N*N, 1))
cY = np.zeros((2*N*N, 1))
cZ = np.zeros((2*N*N, 1))
c  = np.vstack((ct, cX, cY, cZ))

# Build linear constraints
# X = C
sN = sym.vec_dim(N, iscomplex=True, compact=True)
A1 = np.hstack((
    np.zeros((sN, 1)),
    sym.eye(N, iscomplex=True), 
    np.zeros((sN, 2*N*N)),
    np.zeros((sN, 2*N*N)),
))
b1 = sym.mat_to_vec(C, compact=True)
# tr[Y] = 1
A2 = np.hstack((
    np.zeros((1, 1)), 
    np.zeros((1, 2*N*N)), 
    sym.mat_to_vec(np.eye(N, dtype=np.complex128)).T, 
    np.zeros((1, 2*N*N))
))
b2 = np.array([[1.]])
# T2(Y) = Z
p_transpose = sym.lin_to_mat(
    lambda X : sym.p_transpose(X, (n1, n2), 1), 
    (N, N), iscomplex=True
)
A3 = np.hstack((
    np.zeros((1, 1)), 
    np.zeros((1, 2*N*N)),
    p_transpose, 
    -sym.eye(N, iscomplex=True)
))
b3 = np.zeros((sN, 1))

A = np.vstack((A1, A2, A3))
b = np.vstack((b1, b2, b3))

# Input into model and solve
cones = [
    qics.cones.QuantRelEntr(N, iscomplex=True), 
    qics.cones.PosSemidefinite(N, iscomplex=True)
]

# Initialize model and solver objects
model  = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()