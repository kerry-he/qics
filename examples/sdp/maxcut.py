import numpy as np
import scipy as sp
import qics

## SDP relaxation of max-cut
#   min  <C,X>
#   s.t. X_ii = 1    for    i = 1,...,n.
#        X >= 0

n = 200
iscomplex = False

# Generate random linear objective function
C = np.random.randn(n, n)
if iscomplex:
    C = C + np.random.randn(n, n)*1j
C = C + C.conj().T
c = C.view(dtype=np.float64).reshape(-1, 1)

# Build linear constraints
step = 2 if iscomplex else 1
A_is = [i for i in range(n)]
A_js = [i*step + i*n*step for i in range(n)]
A_vs = [1. for i in range(n)]
A = sp.sparse.csr_matrix((A_vs, (A_is, A_js)), shape=(n, n*n*step))

b = np.ones((n, 1))

# Define cones to optimize over
cones = [qics.cones.PosSemidefinite(n, iscomplex=iscomplex)]

# Initialize model and solver objects
model  = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()