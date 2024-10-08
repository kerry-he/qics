import numpy as np
import qics
import qics.vectorize as vec
import qics.quantum as qu

## Quantum channel capacity
#   min  t + s
#   s.t. tr[X] = 1      X >= 0
#        Y = WN(X)W'    (t, Y) ∈ K_qce
# for the amplitude damping channel

np.random.seed(1)

for n in range(2, 17):
    N = n * n

    vn = vec.vec_dim(n, compact=True, iscomplex=True)
    vN = vec.vec_dim(N, compact=True, iscomplex=True)

    V, W = qu.random.degradable_channel(n, n, n, iscomplex=True)

    # Define objective functions
    # with variables (X, (t, Y))
    c = np.vstack((np.array([[1.0]]), np.zeros((vn, 1))))

    # Build linear constraints
    WNW = vec.lin_to_mat(
        lambda X: W @ qu.p_tr(V @ X @ V.conj().T, (n, n), 1) @ W.conj().T,
        (n, N),
        compact=(True, False),
        iscomplex=True,
    )
    ikr_trB = vec.lin_to_mat(
        lambda X: qu.i_kr(qu.p_tr(X, (n, n), 1), (n, n), 1),
        (N, N),
        compact=(False, False),
        iscomplex=True,
    )
    # tr[X] = 1
    A = np.hstack(
        (
            np.array([[0.0]]),
            vec.mat_to_vec(np.eye(n, dtype=np.complex128), compact=True).T,
        )
    )
    b = np.array([[1.0]])

    G1 = np.hstack((np.array([[1.0]]), np.zeros((1, vn))))
    G2 = np.hstack((np.zeros((2 * N * N, 1)), WNW))
    G3 = np.hstack((np.zeros((2 * N * N, 1)), ikr_trB @ WNW))
    G4 = np.hstack((np.zeros((2 * n * n, 1)), vec.eye(n, iscomplex=True).T))
    G = -np.vstack((G1, G2, G3, G4))

    h = np.zeros((G.shape[0], 1))

    # Input into model and solve
    cones = [
        qics.cones.QuantRelEntr(N, iscomplex=True),
        qics.cones.PosSemidefinite(n, iscomplex=True),
    ]

    # Initialize model and solver objects
    model = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
    qics.io.write_cbf(model, "qqcc_qre_" + str(n) + ".cbf")
