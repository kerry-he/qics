import numpy
import qics

numpy.random.seed(1)

n = 5
m = 4
vn = qics.vectorize.vec_dim(n, iscomplex=True)

# Generate random linear objective function
U = numpy.random.randn(n, m) + numpy.random.randn(n, m)*1j
v = numpy.random.randn(n)
C = numpy.diag(v) @ (numpy.eye(n) - U @ U.conj().T) @ numpy.diag(v)
c = qics.vectorize.mat_to_vec(C)

# Build linear constraints  Xii = 1 for all i
A = numpy.zeros((n, vn))
A[numpy.arange(n), numpy.arange(0, vn, 2 * n + 2)] = 1.

b = numpy.ones((n, 1))

# Define cones to optimize over
cones = [qics.cones.PosSemidefinite(n, iscomplex=True)]

# Initialize model and solver objects
model  = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
