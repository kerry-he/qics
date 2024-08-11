import numpy as np
import qics
import qics.utils.symmetric as sym

## Entanglement-assisted channel capacity
#   min  t + s
#   s.t. tr[X] = 1
#        u = 1              X >= 0
#        Y = VXV'           (t, Y) ∈ K_qce
#        Z = trE[VXV']      (s, u, Z) ∈ K_qe
# for the amplitude damping channel

n = 2
N = n*n
gamma = 0.5

V = np.array([[1, 0], [0, np.sqrt(gamma)], [0, np.sqrt(1-gamma)], [0, 0]])

# Define objective functions
# with variables (X, (t, Y), (s, u, Z))
cX = np.zeros((n*n, 1))
ct = np.array([[1./np.log(2)]])
cY = np.zeros((N*N, 1))
cs = np.array([[1./np.log(2)]])
cu = np.array([[0.]])
cZ = np.zeros((n*n, 1))
c = np.vstack((cX, ct, cY, cs, cu, cZ))

# Build linear constraints
vn = sym.vec_dim(n, compact=True)
vN = sym.vec_dim(N, compact=True)
VV = sym.lin_to_mat(lambda X : V @ X @ V.conj().T, (n, n*n))
trE = sym.lin_to_mat(lambda X : sym.p_tr(X, (n, n), 1), (N, n), compact=(True, True))
# tr[X] = 1
A1 = np.hstack((sym.mat_to_vec(np.eye(n)).T, np.zeros((1, 3 + n*n + N*N))))
b1 = np.array([[1.]])
# u = 1
A2 = np.hstack((np.zeros((1, 2 + n*n + N*N)), np.array([[1.]]), np.zeros((1, n*n))))
b2 = np.array([[1.]])
# Y = VXV'
A3 = np.hstack((VV, np.zeros((vN, 1)), -sym.eye(N), np.zeros((vN, 2 + n*n))))
b3 = np.zeros((vN, 1))
# Z = trE[VXV']
A4 = np.hstack((trE @ VV, np.zeros((vn, 3 + N*N)), -sym.eye(n)))
b4 = np.zeros((vn, 1))

A = np.vstack((A1, A2, A3, A4))
b = np.vstack((b1, b2, b3, b4))

# Input into model and solve
cones = [
    qics.cones.PosSemidefinite(n),
    qics.cones.QuantCondEntr((n, n), 0), 
    qics.cones.QuantEntr(n)
]

# Initialize model and solver objects
model  = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()