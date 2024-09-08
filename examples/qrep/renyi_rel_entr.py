import numpy as np
import qics
import qics.vectorize as vec
import qics.quantum as qu

np.random.seed(1)

n = 4
alpha = 0.25

rho = qu.random.density_matrix(n, iscomplex=True)
sigma = qu.random.density_matrix(n, iscomplex=True)

# Define objective function
cT = (1 - alpha) * vec.mat_to_vec(sigma)
cX = np.zeros((2 * n * n, 1))
cY = alpha * vec.mat_to_vec(rho)
c = np.vstack((cT, cX, cY))

# Build linear constraint matrices
vn = vec.vec_dim(n, compact=True, iscomplex=True)
# X = I
A = np.hstack(
    (np.zeros((vn, 2 * n * n)), vec.eye(n, iscomplex=True), np.zeros((vn, 2 * n * n)))
)
b = vec.mat_to_vec(np.eye(n, dtype=np.complex128), compact=True)

# Define cones to optimize over
cones = [qics.cones.OpPerspecEpi(n, alpha / (alpha - 1), iscomplex=True)]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
