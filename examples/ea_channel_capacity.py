import numpy as np
import qics
import qics.utils.symmetric as sym
import qics.utils.quantum as qu

## Entanglement-assisted channel capacity
#   min  t1 + t2
#   s.t. tr[X] = 1
#        (t1, VXV') ∈ K_qce
#        (t2, 1, Tr_2[VXV']) ∈ K_qe
#        X >= 0

ni = 4
no = 4
ne = 4
iscomplex = False

V = qu.rand_stinespring_operator(ni, no, ne, iscomplex=iscomplex)

noe  = no * ne
sni  = sym.vec_dim(ni, iscomplex=iscomplex)
vni  = sym.vec_dim(ni, iscomplex=iscomplex, compact=False)
vno  = sym.vec_dim(no, iscomplex=iscomplex, compact=False)
vnoe = sym.vec_dim(noe, iscomplex=iscomplex, compact=False)

# Define objective function
c = np.zeros((2 + sni, 1))
c[0:2] = 1.

# Build linear constraint tr[X] = 1
A = np.hstack((np.zeros((1, 2)), sym.mat_to_vec(np.eye(ni), iscomplex=iscomplex).T))
b = np.ones((1, 1))

# Build linear cone constraints
# t_qce = t_qce
G1 = np.hstack((-np.ones((1, 1)), np.zeros((1, 1)), np.zeros((1, sni))))
h1 = np.zeros((1, 1))
# X_qce = VXV'
VV = sym.lin_to_mat(lambda X : V @ X @ V.conj().T, (ni, noe), iscomplex=iscomplex, compact=(True, False))
G2 = np.hstack((np.zeros((vnoe, 2)), -VV))
h2 = np.zeros((vnoe, 1))
# t_qe = t_qe
G3 = np.hstack((np.zeros((1, 1)), -np.ones((1, 1)), np.zeros((1, sni))))
h3 = np.zeros((1, 1))
# u_qe = 1
G4 = np.hstack((np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, sni))))
h4 = np.ones((1, 1))
# X_qe = tr_E[VXV']
trE = sym.lin_to_mat(lambda X : sym.p_tr(X, 1, (no, ne)), (noe, no), iscomplex=iscomplex, compact=(False, False))
G5 = np.hstack((np.zeros((vno, 2)), -trE @ VV))
h5 = np.zeros((vno, 1))
# X_psd = X
eye = sym.lin_to_mat(lambda X : X, (ni, ni), iscomplex=iscomplex, compact=(True, False))
G6 = np.hstack((np.zeros((vni, 2)), -eye))
h6 = np.zeros((vni, 1))

G = np.vstack((G1, G2, G3, G4, G5, G6))
h = np.vstack((h1, h2, h3, h4, h5, h6))

# Input into model and solve
cones = [
    qics.cones.QuantCondEntr((no, ne), 0, iscomplex=iscomplex), 
    qics.cones.QuantEntr(no, iscomplex=iscomplex), 
    qics.cones.PosSemidefinite(ni, iscomplex=iscomplex)
]

# Initialize model and solver objects
model  = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
solver = qics.Solver(model)

# Solve problem
out = solver.solve()