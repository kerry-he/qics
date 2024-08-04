import numpy as np
import scipy as sp
import qics

n = 5

# Generate random linear objective function
c = np.array([
    [-2.,  1.,  1.,  0.,  0.],
    [ 1., -3.,  1.,  1.,  0.],
    [ 1.,  1., -3.,  0.,  1.],
    [ 0.,  1.,  0., -2.,  1.],
    [ 0.,  0.,  1.,  1., -2.]
]).reshape(-1, 1)

# Build linear constraints A corresponding to Xii=1
A = np.zeros((n, n, n))
A[range(n), range(n), range(n)] = 1.
A = sp.sparse.csr_matrix(A.reshape(n, -1))

b = np.ones((n, 1))

# Define cones to optimize over
cones = [qics.cones.PosSemidefinite(n)]

# Initialize model and solver objects
model  = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
out = solver.solve()

print("Optimal matrix variable X is: ")
print(out["x_opt"].reshape(n, n))