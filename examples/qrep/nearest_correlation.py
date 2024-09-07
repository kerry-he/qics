import numpy as np
import qics
import qics.vectorize as vec

## Nearest correlation matrix
#   min  S(X||Y)
#   s.t. Y_ii = 1

n = 5

# Generate random matrix C
C = np.random.randn(n, n)
C = C @ C.T

# Define objective function
ct = np.array(([[1.]]))
cX = np.zeros((n*n, 1))
cY = np.zeros((n*n, 1))
c  = np.vstack((ct, cX, cY))

# Build linear constraints
# X = C
sn = vec.vec_dim(n, compact=True)
A1 = np.hstack((np.zeros((sn, 1)), vec.eye(n), np.zeros((sn, n*n))))
b1 = vec.mat_to_vec(C, compact=True)
# Yii = 1
A2 = np.zeros((n, 1 + 2*n*n))
A2[range(n), range(1 + n*n, 1 + 2*n*n, n+1)] = 1.
b2 = np.ones((n, 1))

A = np.vstack((A1, A2))
b = np.vstack((b1, b2))

# Define cones to optimize over
cones = [qics.cones.QuantRelEntr(n)]

# Initialize model and solver objects
model  = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()