import numpy as np
import qics
import qics.vectorize as vec
import qics.quantum as qu

## Ground state energy of Hamiltonian
#   min  <H,X>
#   s.t. Tr_1[X] = Tr_L[X]
#        tr[X] = 1
#        (t, X) ∈ K_qce

for L in range(2, 8):
    dims = [2] * L

    N = 2**L
    m = 2 ** (L - 1)
    vm = vec.vec_dim(m, compact=True)

    # Define objective function
    def heisenberg(delta, L):
        sx = np.array([[0.0, 1.0], [1.0, 0.0]])
        sy = np.array([[0.0, -1.0j], [1.0j, 0.0]])
        sz = np.array([[1.0, 0.0], [0.0, -1.0]])
        h = -(np.kron(sx, sx) + np.kron(sy, sy) + delta * np.kron(sz, sz))

        return np.kron(h, np.eye(2 ** (L - 2))).real

    H = heisenberg(-1, L)
    c = vec.mat_to_vec(H, compact=True)

    # Build linear constraint matrices
    # Tr_1[X] = Tr_L[X]
    tr1 = vec.lin_to_mat(lambda X: qu.p_tr(X, dims, 0), (N, m), compact=(True, False))
    trL = vec.lin_to_mat(
        lambda X: qu.p_tr(X, dims, L - 1), (N, m), compact=(True, False)
    )
    tr = vec.mat_to_vec(np.eye(N), compact=True).T
    ikr_tr1 = vec.lin_to_mat(
        lambda X: qu.i_kr(qu.p_tr(X, dims, 0), dims, 0), (N, N), compact=(True, False)
    )

    A = np.vstack((tr1 - trL, tr))
    b = np.zeros((A.shape[0], 1))
    b[-1] = 1.0

    G = -np.vstack((np.zeros((1, ikr_tr1.shape[1])), vec.eye(N).T, ikr_tr1))
    h = np.zeros((G.shape[0], 1))

    # Define cones to optimize over
    cones = [qics.cones.QuantRelEntr(N)]

    # Initialize model and solver objects
    model = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
    qics.io.write_cbf(model, "gse_qre_" + str(L) + ".cbf")
