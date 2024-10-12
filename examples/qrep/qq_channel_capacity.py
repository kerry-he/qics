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
gamma = 0.25
delta = (1 - 2 * gamma) / (1 - gamma)
V = np.array([
    [1., 0.              ],
    [0., np.sqrt(1-gamma)],
    [0., np.sqrt(gamma)  ],
    [0., 0.              ]
])  # fmt: skip
W = np.array([
    [1., 0.              ],
    [0., np.sqrt(delta)  ],
    [0., np.sqrt(1-delta)],
    [0., 0.              ]
])  # fmt: skip


def W_NX_W(X):
    return W @ p_tr(V @ X @ V.conj().T, (n, n), 1) @ W.conj().T


# Model problem using primal variables (t, cvec(X))
# Define objective functions
c = np.block([[1.0], [np.zeros((cn, 1))]])

# Build linear constraint tr[X] = 1
trace = lin_to_mat(lambda X: np.trace(X), (n, 1), compact=(True, False))
A = np.block([[0.0, trace]])
b = np.array([[1.0]])

# Build conic linear constraints
W_NX_W_mat = lin_to_mat(W_NX_W, (n, N), compact=(True, False))

G = np.block([
    [-1.0,              np.zeros((1, cn))],  # t_qce = t
    [np.zeros((vN, 1)), -W_NX_W_mat      ],  # X_qce = WN(X)W'
    [np.zeros((vn, 1)), -eye(n).T        ]   # X_psd = X
])  # fmt: skip

h = np.block([[0.0], [np.zeros((vN, 1))], [np.zeros((vn, 1))]])

# Define cones to optimize over
cones = [
    qics.cones.QuantCondEntr((n, n), 1),  # (t, WN(X)W') ∈ QCE
    qics.cones.PosSemidefinite(n),  # X ⪰ 0
]

# Initialize model and solver objects
model = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()
