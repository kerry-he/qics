import numpy as np

import qics
import qics.quantum as qu
import qics.vectorize as vec

## Relative entropy of entanglement
#   min  S(X||Y)
#   s.t. tr[Y] = 1
#        T2(Y) >= 0

for n in range(2, 13):
    np.random.seed(1)

    n1 = n
    n2 = n
    N = n1 * n2

    # Generate random (complex) quantum state
    C = qu.random.density_matrix(N, iscomplex=True)

    # Define objective function
    ct = np.array(([[1.0]]))
    cX = np.zeros((2 * N * N, 1))
    cY = np.zeros((2 * N * N, 1))
    cZ = np.zeros((2 * N * N, 1))
    c = np.vstack((ct, cX, cY, cZ))

    # Build linear constraints
    # X = C
    sN = vec.vec_dim(N, iscomplex=True, compact=True)
    A1 = np.hstack(
        (
            np.zeros((sN, 1)),
            vec.eye(N, iscomplex=True),
            np.zeros((sN, 2 * N * N)),
            np.zeros((sN, 2 * N * N)),
        )
    )
    b1 = vec.mat_to_vec(C, compact=True)
    # tr[Y] = 1
    A2 = np.hstack(
        (
            np.zeros((1, 1)),
            np.zeros((1, 2 * N * N)),
            vec.mat_to_vec(np.eye(N, dtype=np.complex128)).T,
            np.zeros((1, 2 * N * N)),
        )
    )
    b2 = np.array([[1.0]])
    # T2(Y) = Z
    p_transpose = vec.lin_to_mat(
        lambda X: qu.partial_transpose(X, (n1, n2), 1), (N, N), iscomplex=True
    )
    A3 = np.hstack(
        (
            np.zeros((sN, 1)),
            np.zeros((sN, 2 * N * N)),
            p_transpose,
            -vec.eye(N, iscomplex=True),
        )
    )
    b3 = np.zeros((sN, 1))

    A = np.vstack((A1, A2, A3))
    b = np.vstack((b1, b2, b3))

    # Input into model and solve
    cones = [
        qics.cones.QuantRelEntr(N, iscomplex=True),
        qics.cones.PosSemidefinite(N, iscomplex=True),
    ]

    # Initialize model and solver objects
    model = qics.Model(c=c, A=A, b=b, cones=cones)
    qics.io.write_cbf(model, "ree_" + str(n1) + "_" + str(n2) + ".cbf")
