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

for n in range(2, 11):
    N = n*n

    vn = vec.vec_dim(n, compact=True, iscomplex=True)
    vN = vec.vec_dim(N, compact=True, iscomplex=True)

    V, W = qu.random.degradable_channel(n, n, n, iscomplex=True)

    # Define objective functions
    # with variables (X, (t, Y))
    cX = np.zeros((2*n*n, 1))
    ct = np.array([[1.]])
    cY = np.zeros((2*N*N, 1))
    c = np.vstack((cX, ct, cY))

    # Build linear constraints
    vn = vec.vec_dim(n, compact=True, iscomplex=True)
    vN = vec.vec_dim(N, compact=True, iscomplex=True)
    WNW = vec.lin_to_mat(
        lambda X : W @ qu.p_tr(V @ X @ V.conj().T, (n, n), 1) @ W.conj().T, (n, N), iscomplex=True
    )
    # tr[X] = 1
    A1 = np.hstack((vec.mat_to_vec(np.eye(n, dtype=np.complex128)).T, np.zeros((1, 1 + 2*N*N))))
    b1 = np.array([[1.]])
    # Y = WN(X)W'
    A2 = np.hstack((WNW, np.zeros((vN, 1)), -vec.eye(N, iscomplex=True)))
    b2 = np.zeros((vN, 1))

    A = np.vstack((A1, A2))
    b = np.vstack((b1, b2))

    # Input into model and solve
    cones = [
        qics.cones.PosSemidefinite(n, iscomplex=True),
        qics.cones.QuantCondEntr((n, n), 1, iscomplex=True)
    ]

    # Initialize model and solver objects
    model  = qics.Model(c=c, A=A, b=b, cones=cones)
    qics.io.write_cbf(model, "qqcc_" + str(n) + ".cbf")

    # solver = qics.Solver(model)

    # # Solve problem
    # info = solver.solve()