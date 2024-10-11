import numpy as np

import qics
import qics.vectorize as vec

## Entanglement-assisted channel capacity
#   max  -t + (n - tr[Y])*log(e)
#   s.t. Σ_i zi = k
#        tr[Y] <= k
#        0 <= z <= 1
#        0 <= Y <= I
#        (t, Y, Adiag(z)A' + eY) ∈ K_Plog

n = 10
m = 20
A_dat = 1 / (n**0.25) * np.random.randn(n, m)
k = 5
eps = 1e-6

vn = vec.vec_dim(n, compact=True)

# Define objective function, with x = (t, z, Y) and c=(1, 0, I*log(e))
c1 = np.ones((1, 1))
c2 = np.zeros((m, 1))
c3 = vec.mat_to_vec(np.eye(n), compact=True) * np.log(eps)
c = np.vstack((c1, c2, c3))

# Build linear constraint Σ_i zi = k
A = np.hstack((np.zeros((1, 1)), np.ones((1, m)), np.zeros((1, vn))))
b = np.ones((1, 1)) * k

# Build linear cone constraints
# tr[Y] <= k
G1 = np.hstack(
    (np.zeros((1, 1)), np.zeros((1, m)), vec.mat_to_vec(np.eye(n), compact=True).T)
)
h1 = np.array(([k]))
# 0 <= z
G2 = np.hstack((np.zeros((m, 1)), -np.eye(m), np.zeros((m, vn))))
h2 = np.zeros((m, 1))
# z <= 1
G3 = np.hstack((np.zeros((m, 1)), np.eye(m), np.zeros((m, vn))))
h3 = np.ones((m, 1))
# Y <= I
eye = vec.lin_to_mat(lambda X: X, (n, n), compact=(True, False))
G4 = np.hstack((np.zeros((n * n, 1)), np.zeros((n * n, m)), eye))
h4 = vec.mat_to_vec(np.eye(n), compact=False)
# t_Plog = t
G5 = np.hstack((-np.ones((1, 1)), np.zeros((1, m)), np.zeros((1, vn))))
h5 = np.zeros((1, 1))
# X_Plog = Y
G6 = np.hstack((np.zeros((n * n, 1)), np.zeros((n * n, m)), -eye))
h6 = np.zeros((n * n, 1))
# Y_Plog = Adiag(z)A' + eY
AdiagA = np.hstack(
    [vec.mat_to_vec(A_dat[:, [i]] @ A_dat[:, [i]].T, compact=False) for i in range(m)]
)
G7 = np.hstack((np.zeros((n * n, 1)), -AdiagA, -eye * eps))
h7 = np.zeros((n * n, 1))

G = np.vstack((G1, G2, G3, G4, G5, G6, G7))
h = np.vstack((h1, h2, h3, h4, h5, h6, h7))

# Define cones to optimize over
cones = [
    qics.cones.NonNegOrthant(1 + m + m),
    qics.cones.PosSemidefinite(n),
    qics.cones.OpPerspecTr(n, "log"),
]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones, offset=-n * np.log(eps))
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
