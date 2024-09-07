import numpy as np
import qics
import qics.quantum as qu
import qics.vectorize as vec

np.random.seed(1)

n = m = 2

rhoA = qu.random.density_matrix(n, iscomplex=True)
rhoB = qu.random.density_matrix(m, iscomplex=True)

# Generate random objective function
C = np.random.randn(n*m, n*m) + np.random.randn(n*m, n*m)*1j
C = C + C.conj().T
c = vec.mat_to_vec(C)

# Build linear constraints
trA = vec.lin_to_mat(lambda X : qu.p_tr(X, (n, m), 0), (n*m, m), iscomplex=True)
trB = vec.lin_to_mat(lambda X : qu.p_tr(X, (n, m), 1), (n*m, n), iscomplex=True)
A = np.vstack((trA, trB))
b = np.vstack((vec.mat_to_vec(rhoA, compact=True), vec.mat_to_vec(rhoB, compact=True)))

# Define cones to optimize over
cones = [qics.cones.PosSemidefinite(n*m, iscomplex=True)]

# Initialize model and solver objects
model  = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()