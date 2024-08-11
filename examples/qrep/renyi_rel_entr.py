import numpy as np
import qics
import qics.utils.symmetric as sym
import qics.utils.quantum as qu

np.random.seed(1)

n = 4
alpha = 0.25

rho   = qu.rand_density_matrix(n, iscomplex=True)
sigma = qu.rand_density_matrix(n, iscomplex=True)

# Define objective function
cT = (1 - alpha) * sym.mat_to_vec(sigma)
cX = np.zeros((2*n*n, 1))
cY = alpha * sym.mat_to_vec(rho)
c = np.vstack((cT, cX, cY))

# Build linear constraint matrices
vn = sym.vec_dim(n, compact=True, iscomplex=True)
# X = I
A = np.hstack((
    np.zeros((vn, 2*n*n)), 
    sym.eye(n, iscomplex=True), 
    np.zeros((vn, 2*n*n))
))
b = sym.mat_to_vec(np.eye(n, dtype=np.complex128), compact=True)

# Define cones to optimize over
cones = [qics.cones.OpPerspecEpi(n, alpha/(alpha - 1), iscomplex=True)]

# Initialize model and solver objects
model  = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()