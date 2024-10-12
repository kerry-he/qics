import numpy as np

import qics
from qics.quantum import p_tr
from qics.vectorize import eye, lin_to_mat, vec_dim

n = 2
N = n * n

vn = vec_dim(n)
vN = vec_dim(N)
cn = vec_dim(n, compact=True)

# Define amplitude damping channel
gamma = 0.5
V = np.array([
    [1., 0.              ],
    [0., np.sqrt(1-gamma)],
    [0., np.sqrt(gamma)  ],
    [0., 0.              ]
])  # fmt: skip

# Model problem using primal variables (t1, t2, cvec(X))
# Define objective functions
c = np.block([[1.0], [1.0], [np.zeros((cn, 1))]])

# Build linear constraint tr[X] = 1
trace = lin_to_mat(lambda X: np.trace(X), (n, 1), compact=(True, False))
A = np.block([[0.0, 0.0, trace]])
b = np.array([[1.0]])

# Build conic linear constraints
VV = lin_to_mat(lambda X: V @ X @ V.conj().T, (n, N), compact=(True, False))
trE = lin_to_mat(lambda X: p_tr(X, (n, n), 0), (N, n), compact=(False, False))

G = np.block([
    [-1.0,              0.0,               np.zeros((1, cn))],  # t_qce = t1
    [np.zeros((vN, 1)), np.zeros((vN, 1)), -VV              ],  # X_qce = VXV'
    [0.0,               -1.0,              np.zeros((1, cn))],  # t_qe = t2
    [0.0,               0.0,               np.zeros((1, cn))],  # u_qe = 1
    [np.zeros((vn, 1)), np.zeros((vn, 1)), -trE @ VV        ],  # X_qe = tr_E(VXV')
    [np.zeros((vn, 1)), np.zeros((vn, 1)), -eye(n).T        ]   # X_psd = X
])  # fmt: skip

h = np.block([
    [0.0], 
    [np.zeros((vN, 1))], 
    [0.0], 
    [1.0], 
    [np.zeros((vn, 1))], 
    [np.zeros((vn, 1))],
])  # fmt: skip

# Define cones to optimize over
cones = [
    qics.cones.QuantCondEntr((n, n), 1),  # (t1, VXV') ∈ QCE
    qics.cones.QuantEntr(n),  # (t2, 1, tr_E(XVX')) ∈ QE
    qics.cones.PosSemidefinite(n),  # X ⪰ 0
]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
