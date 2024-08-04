import numpy as np
import qics
import qics.utils.symmetric as sym
import qics.utils.quantum as qu

## Quantum-quantum channel capacity
#   min  t
#   s.t. tr[X] = 1
#        (t, WN(X)W') ∈ K_qce
#        X ⪰ 0

ni = 8
no = 8
ne = 8
iscomplex = False

V, W = qu.rand_degradable_channel(ni, no, ne, iscomplex=iscomplex)

nei  = ne * ni
sni  = sym.vec_dim(ni, iscomplex=iscomplex, compact=True)
vni  = sym.vec_dim(ni, iscomplex=iscomplex)
vnei = sym.vec_dim(nei, iscomplex=iscomplex)

# Define objective function
c = np.zeros((1 + sni, 1))
c[0] = 1.

# Build linear constraint tr[X] = 1
A = np.hstack((np.zeros((1, 1)), sym.mat_to_vec(np.eye(ni), iscomplex=iscomplex, compact=True).T))
b = np.ones((1, 1))

# Build linear cone constraints
# t_qce = t_qce
G1 = np.hstack((-np.ones((1, 1)), np.zeros((1, sni))))
h1 = np.zeros((1, 1))
# X_qce = WN(X)W'
WNW = sym.lin_to_mat(
    lambda X : W @ sym.p_tr(V @ X @ V.conj().T, (no, ne), 1) @ W.conj().T, 
    (ni, nei), iscomplex=iscomplex, compact=(True, False)
)
G2 = np.hstack((np.zeros((vnei, 1)), -WNW))
h2 = np.zeros((vnei, 1))
# X_psd = X
eye = sym.lin_to_mat(lambda X : X, (ni, ni), iscomplex=iscomplex, compact=(True, False))
G3 = np.hstack((np.zeros((vni, 1)), -eye))
h3 = np.zeros((vni, 1))

G = np.vstack((G1, G2, G3))
h = np.vstack((h1, h2, h3))

# Input into model and solve
cones = [
    qics.cones.QuantCondEntr((ne, ni), 1, iscomplex=iscomplex), 
    qics.cones.PosSemidefinite(ni, iscomplex=iscomplex)
]

# Initialize model and solver objects
model  = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
solver = qics.Solver(model)

# Solve problem
out = solver.solve()