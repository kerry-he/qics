import numpy as np
import qics
import qics.utils.symmetric as sym

## Ground state energy of Hamiltonian
#   min  <H,X>
#   s.t. Tr_1[X] = Tr_L[X]
#        tr[X] = 1
#        (t, X) ∈ K_qce

L    = 4
dims = [2] * L

N = 2**L
m = 2**(L-1)
vm = sym.vec_dim(m)

# Define objective function
def heisenberg(delta, L):
    sx = np.array([[0.,  1. ], [1. ,  0.]])
    sy = np.array([[0., -1.j], [1.j,  0.]])
    sz = np.array([[1.,  0. ], [0. , -1.]])
    h = -(np.kron(sx,sx) + np.kron(sy,sy) + delta*np.kron(sz,sz))

    return np.kron(h, np.eye(2**(L-2))).real

H = heisenberg(-1, L)
c = np.vstack((np.zeros((1, 1)), H.reshape((-1, 1))))

# Build linear constraint matrices
# Tr_1[X] = Tr_L[X]
tr1 = sym.lin_to_mat(lambda X : sym.p_tr(X,   0, dims), (N, m))
trL = sym.lin_to_mat(lambda X : sym.p_tr(X, L-1, dims), (N, m))
A1 = np.hstack((np.zeros((vm, 1)), tr1 - trL))
b1 = np.zeros((vm, 1))
# tr[X] = 1
A2 = np.hstack((np.zeros((1, 1)), np.eye(N).reshape(-1, 1).T))
b2 = np.ones((1, 1))
# t = 0
A3 = np.hstack((np.ones((1, 1)), np.zeros((1, N*N))))
b3 = np.zeros((1, 1))

A = np.vstack((A1, A2, A3))
b = np.vstack((b1, b2, b3))

# Define cones to optimize over
cones = [qics.cones.QuantCondEntr(dims, 0)]

# Initialize model and solver objects
model  = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
out = solver.solve()