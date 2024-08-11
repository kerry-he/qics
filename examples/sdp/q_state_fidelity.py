import numpy as np
import qics
import qics.utils.symmetric as sym
import qics.utils.quantum as qu

np.random.seed(1)

n = 2

rho = qu.rand_density_matrix(n, iscomplex=True)
sig = qu.rand_density_matrix(n, iscomplex=True)

# Define objective function
c = -0.5 * sym.mat_to_vec(np.block([
    [np.zeros((n, n)), np.eye(n)],
    [np.eye(n), np.zeros((n, n))]
]).astype(np.complex128))

# Build linear constraints
A = np.vstack((
    sym.lin_to_mat(lambda X : X[:n, :n], (2*n, n), iscomplex=True),
    sym.lin_to_mat(lambda X : X[n:, n:], (2*n, n), iscomplex=True)
))

b = np.vstack((
    sym.mat_to_vec(rho, compact=True),
    sym.mat_to_vec(sig, compact=True)
))

# Define cones to optimize over
cones = [qics.cones.PosSemidefinite(2*n, iscomplex=True)]

# Initialize model and solver objects
model  = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()