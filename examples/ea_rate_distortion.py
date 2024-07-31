import numpy as np
import qics
import qics.utils.symmetric as sym
import qics.utils.quantum as qu

## Entanglement-assisted rate distortion
#   min  t
#   s.t. Tr_2[X] = rho
#        (t, X) ∈ K_qce
#        <Δ, X> <= D

n = 4                   # Dimension of rho
D = 0.5                 # Maximum allowable distortion
iscomplex = False

rho      = qu.rand_density_matrix(n, iscomplex=iscomplex)
entr_rho = qu.quant_entropy(rho)

N = n * n
sn = sym.vec_dim(n, iscomplex=iscomplex)
vN = sym.vec_dim(N, iscomplex=iscomplex, compact=False)

# Define objective function
c = np.zeros((vN + 2, 1))
c[0] = 1.

# Build linear constraint matrices
# Tr_2[X] = rho
tr2 = sym.lin_to_mat(lambda X : sym.p_tr(X, 1, (n, n)), (N, n), iscomplex=iscomplex)
A1  = np.hstack((np.zeros((sn, 1)), tr2, np.zeros((sn, 1))))
b1  = sym.mat_to_vec(rho, iscomplex=iscomplex)
# <Δ, X> <= D
Delta = sym.mat_to_vec(np.eye(N) - qu.purify(rho), iscomplex=iscomplex, compact=False)
A2    = np.hstack((np.zeros((1, 1)), Delta.T, np.ones((1, 1))))
b2    = np.array([[D]])

A = np.vstack((A1, A2))
b = np.vstack((b1, b2))

# Define cones to optimize over
cones = [qics.cones.QuantCondEntr((n, n), 0, iscomplex=iscomplex), qics.cones.NonNegOrthant(1)]

# Initialize model and solver objects
model  = qics.Model(c=c, A=A, b=b, cones=cones, offset=entr_rho)
solver = qics.Solver(model)

# Solve problem
out = solver.solve()