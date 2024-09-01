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

    V = qu.random.stinespring_operator(n, n, n, iscomplex=True)
    # V = np.zeros((n*n, n))
    # V[range(0, N, n), range(n)] = 1.    

    # Define objective functions
    # with variables (X, (t, Y), (s, u, Z))
    cX = np.zeros((2*n*n, 1))
    ct = np.array([[1.]])
    cY = np.zeros((2*N*N, 1))
    cs = np.array([[1.]])
    cu = np.array([[0.]])
    cZ = np.zeros((2*n*n, 1))
    c = np.vstack((cX, ct, cY, cs, cu, cZ))

    # Build linear constraints
    vn = vec.vec_dim(n, compact=True, iscomplex=True)
    vN = vec.vec_dim(N, compact=True, iscomplex=True)
    VV = vec.lin_to_mat(lambda X : V @ X @ V.conj().T, (n, n*n), iscomplex=True)
    trE = vec.lin_to_mat(lambda X : qu.p_tr(X, (n, n), 0), (N, n), compact=(True, True), iscomplex=True)
    # tr[X] = 1
    A1 = np.hstack((vec.mat_to_vec(np.eye(n, dtype=np.complex128)).T, np.zeros((1, 3 + 2*n*n + 2*N*N))))
    b1 = np.array([[1.]])
    # u = 1
    A2 = np.hstack((np.zeros((1, 2 + 2*n*n + 2*N*N)), np.array([[1.]]), np.zeros((1, 2*n*n))))
    b2 = np.array([[1.]])
    # Y = VXV'
    A3 = np.hstack((VV, np.zeros((vN, 1)), -vec.eye(N, iscomplex=True), np.zeros((vN, 2 + 2*n*n))))
    b3 = np.zeros((vN, 1))
    # Z = trE[VXV']
    A4 = np.hstack((trE @ VV, np.zeros((vn, 3 + 2*N*N)), -vec.eye(n, iscomplex=True)))
    b4 = np.zeros((vn, 1))

    A = np.vstack((A1, A2, A3, A4))
    b = np.vstack((b1, b2, b3, b4))

    # Input into model and solve
    cones = [
        qics.cones.PosSemidefinite(n, iscomplex=True),
        qics.cones.QuantCondEntr((n, n), 1, iscomplex=True), 
        qics.cones.QuantEntr(n, iscomplex=True)
    ]

    # Initialize model and solver objects
    model  = qics.Model(c=c, A=A, b=b, cones=cones)
    qics.io.write_cbf(model, "ccea_" + str(n) + ".cbf")

    # solver = qics.Solver(model)

    # # Solve problem
    # info = solver.solve()