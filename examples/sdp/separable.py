import numpy as np
import qics
import qics.vectorize as vec
import qics.quantum as qu

n  = 2
n2 = n * n
n3 = n * n * n

vn2 = vec.vec_dim(n2, compact=True)
vn3 = vec.vec_dim(n3, compact=True)

rho_ab = 0.5 * np.array([
    [1., 0., 0., 1.], 
    [0., 0., 0., 0.],
    [0., 0., 0., 0.],
    [1., 0., 0., 1.]   
])

# Define objective function
c = np.zeros((3*n3*n3, 1))

# Build linear constraints
# rho_ab1 = tr_b2(rho_aB)
tr_b2 = vec.lin_to_mat(lambda X : qu.p_tr(X, (n, n, n), 2), (n3, n2))
A1 = np.hstack((tr_b2, np.zeros((vn2, 2*n3*n3))))
b1 = vec.mat_to_vec(rho_ab, compact=True)
# rho_aB = swap_b1,b2(rho_aB)
swap = vec.lin_to_mat(lambda X : qu.swap(X, (n, n, n), 1, 2), (n3, n3))
A2 = np.hstack((swap - vec.eye(n3), np.zeros((vn3, 2*n3*n3))))
b2 = np.zeros((vn3, 1))
# tr[rho_aB] = 1
tr = vec.mat_to_vec(np.eye(n3)).T
A3 = np.hstack((tr, np.zeros((1, 2*n3*n3))))
b3 = np.array([[1.]])
# Y = T_b2(rho_aB)
T_b2 = vec.lin_to_mat(lambda X : qu.partial_transpose(X, (n2, n), 1), (n3, n3))
A4 = np.hstack((T_b2, -vec.eye(n3), np.zeros((vn3, n3*n3))))
b4 = np.zeros((vn3, 1))
# Z = T_b1b2(rho_aB)
T_b1b2 = vec.lin_to_mat(lambda X : qu.partial_transpose(X, (n, n2), 1), (n3, n3))
A5 = np.hstack((T_b1b2, np.zeros((vn3, n3*n3)), -vec.eye(n3)))
b5 = np.zeros((vn3, 1))

A = np.vstack((A1, A2, A3, A4, A5))
b = np.vstack((b1, b2, b3, b4, b5))

# Define cones to optimize over
cones = [
    qics.cones.PosSemidefinite(n3),
    qics.cones.PosSemidefinite(n3),
    qics.cones.PosSemidefinite(n3)
]

# Initialize model and solver objects
model  = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()