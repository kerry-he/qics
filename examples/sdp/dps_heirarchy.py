import numpy
import qics

n  = 2
n2 = n * n
n3 = n * n * n

vn2 = qics.vectorize.vec_dim(n2, compact=True)
vn3 = qics.vectorize.vec_dim(n3, compact=True)

rho_ab = 0.5 * numpy.array([
    [1., 0., 0., 1.],
    [0., 0., 0., 0.],
    [0., 0., 0., 0.],
    [1., 0., 0., 1.]
])

# Define objective function
c = numpy.zeros((3*n3*n3, 1))

# Build linear constraints
# rho_ab1 = tr_b2(rho_aB)
tr_b2 = qics.vectorize.lin_to_mat(
    lambda X : qics.quantum.p_tr(X, (n, n, n), 2), (n3, n2))
A1 = numpy.hstack((tr_b2, numpy.zeros((vn2, 2*n3*n3))))
b1 = qics.vectorize.mat_to_vec(rho_ab, compact=True)
# rho_aB = swap_b1,b2(rho_aB)
swap = qics.vectorize.lin_to_mat(
    lambda X : qics.quantum.swap(X, (n, n, n), 1, 2), (n3, n3))
A2 = numpy.hstack((swap - qics.vectorize.eye(n3), numpy.zeros((vn3, 2*n3*n3))))
b2 = numpy.zeros((vn3, 1))
# tr[rho_aB] = 1
tr = qics.vectorize.mat_to_vec(numpy.eye(n3)).T
A3 = numpy.hstack((tr, numpy.zeros((1, 2*n3*n3))))
b3 = numpy.array([[1.]])
# Y = T_b2(rho_aB)
T_b2 = qics.vectorize.lin_to_mat(
    lambda X : qics.quantum.partial_transpose(X, (n2, n), 1), (n3, n3))
A4 = numpy.hstack((T_b2, -qics.vectorize.eye(n3), numpy.zeros((vn3, n3*n3))))
b4 = numpy.zeros((vn3, 1))
# Z = T_b1b2(rho_aB)
T_b1b2 = qics.vectorize.lin_to_mat(
    lambda X : qics.quantum.partial_transpose(X, (n, n2), 1), (n3, n3))
A5 = numpy.hstack((T_b1b2, numpy.zeros((vn3, n3*n3)), -qics.vectorize.eye(n3)))
b5 = numpy.zeros((vn3, 1))

A = numpy.vstack((A1, A2, A3, A4, A5))
b = numpy.vstack((b1, b2, b3, b4, b5))

# Define cones to optimize over
cones = [
    qics.cones.PosSemidefinite(n3),
    qics.cones.PosSemidefinite(n3),
    qics.cones.PosSemidefinite(n3)
]

# Initialize model and solver objects
model  = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model, verbose=0)

# Solve problem
info = solver.solve()

print("Solution status:", info["sol_status"])