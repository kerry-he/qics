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

for n in range(2, 12):
    N = n*n

    vn = vec.vec_dim(n, compact=True, iscomplex=True)
    vN = vec.vec_dim(N, compact=True, iscomplex=True)

    V, W = qu.random.degradable_channel(n, n, n, iscomplex=True)

    # Define objective functions
    # with variables (X, (t, Y), (s, u, Z))
    ct = np.array([[1.]])
    cX = np.zeros((n*n, 1))
    c = np.vstack((ct, cX))

    # Build linear constraints
    # tr[X] = 1
    A = np.hstack((np.zeros((1, 1)), vec.mat_to_vec(np.eye(n, dtype=np.complex128), compact=True).T))
    b = np.array([[1.]])

    # Build linear constraints
    vn = vec.vec_dim(n, compact=True, iscomplex=True)
    vN = vec.vec_dim(N, compact=True, iscomplex=True)
    WNW = vec.lin_to_mat(
        lambda X : W @ qu.p_tr(V @ X @ V.conj().T, (n, n), 1) @ W.conj().T, (n, N), compact=(True, False), iscomplex=True
    )

    G1 = np.hstack((np.array([[1.]]), np.zeros((1, vn))))            # t_cond
    G2 = np.hstack((np.zeros((2*N*N, 1)), WNW))                            # X_cond
    G3 = np.hstack((np.zeros((2*n*n, 1)), vec.eye(n, iscomplex=True).T))  # X_psd

    h1 = np.array([[0.]])     # t_cond  
    h2 = np.zeros((2*N*N, 1))   # X_cond
    h3 = np.zeros((2*n*n, 1))   # X_psd

    G = -np.vstack((G1, G2, G3))
    h =  np.vstack((h1, h2, h3))

    # Input into model and solve
    cones = [
        qics.cones.QuantCondEntr((n, n), 1, iscomplex=True), 
        qics.cones.PosSemidefinite(n, iscomplex=True)
    ]

    # Initialize model and solver objects
    model  = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
    qics.io.write_cbf(model, "ccqq_" + str(n) + ".cbf")

    # solver = qics.Solver(model)

    # # Solve problem
    # info = solver.solve()