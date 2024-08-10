import numpy as np
import qics
import qics.utils.symmetric as sym

## Quantum channel capacity
#   min  t + s
#   s.t. tr[X] = 1      X >= 0
#        Y = WN(X)W'    (t, Y) ∈ K_qce
# for the amplitude damping channel

n = 2
N = n*n
gamma = 0.5

V = np.array([[1, 0], [0, np.sqrt(gamma)], [0, np.sqrt(1-gamma)], [0, 0]])
W = np.array([[1, 0], [0, np.sqrt((1-2*gamma)/(1-gamma))], [0, np.sqrt(gamma/(1-gamma))], [0, 0]])

# Define objective functions
# with variables (X, (t, Y))
cX = np.zeros((n*n, 1))
ct = np.array([[1./np.log(2)]])
cY = np.zeros((N*N, 1))
c = np.vstack((cX, ct, cY))

# Build linear constraints
vn = sym.vec_dim(n, compact=True)
vN = sym.vec_dim(N, compact=True)
WNW = sym.lin_to_mat(
    lambda X : W @ sym.p_tr(V @ X @ V.conj().T, (n, n), 1) @ W.conj().T, (n, N)
)
# tr[X] = 1
A1 = np.hstack((sym.mat_to_vec(np.eye(n)).T, np.zeros((1, 1 + N*N))))
b1 = np.array([[1.]])
# Y = WN(X)W'
A2 = np.hstack((WNW, np.zeros((vN, 1)), -sym.eye(N)))
b2 = np.zeros((vN, 1))

A = np.vstack((A1, A2))
b = np.vstack((b1, b2))

# Input into model and solve
cones = [
    qics.cones.PosSemidefinite(n),
    qics.cones.QuantCondEntr((n, n), 1)
]

# Initialize model and solver objects
model  = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()