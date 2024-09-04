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
    ct = np.array([[1.]])
    cs = np.array([[1.]])
    cX = np.zeros((n*n, 1))
    c = np.vstack((ct, cs, cX))

    # Build linear constraints
    # tr[X] = 1
    A = np.hstack((np.zeros((1, 2)), vec.mat_to_vec(np.eye(n, dtype=np.complex128), compact=True).T))
    b = np.array([[1.]])

    vn = vec.vec_dim(n, compact=True, iscomplex=True)
    vN = vec.vec_dim(N, compact=True, iscomplex=True)
    VV = vec.lin_to_mat(lambda X : V @ X @ V.conj().T, (n, n*n), iscomplex=True, compact=(True, False))
    trE = vec.lin_to_mat(lambda X : qu.p_tr(X, (n, n), 0), (N, n), compact=(False, False), iscomplex=True)


    G1 = np.hstack((np.array([[1., 0.]]), np.zeros((1, vn))))           # t_cond
    G2 = np.hstack((np.zeros((2*N*N, 2)), VV))                          # X_cond
    G3 = np.hstack((np.array([[0., 1.]]), np.zeros((1, vn))))           # t_entr
    G4 = np.hstack((np.array([[0., 0.]]), np.zeros((1, vn))))           # y_entr
    G5 = np.hstack((np.zeros((2*n*n, 2)), trE @ VV))                    # X_entr
    G6 = np.hstack((np.zeros((2*n*n, 2)), vec.eye(n, iscomplex=True).T))  # X_psd

    h1 = np.array([[0.]])       # t_cond  
    h2 = np.zeros((2*N*N, 1))   # X_cond
    h3 = np.array([[0.]])       # t_entr
    h4 = np.array([[1.]])       # y_entr
    h5 = np.zeros((2*n*n, 1))   # X_entr
    h6 = np.zeros((2*n*n, 1))   # X_psd

    G = -np.vstack((G1, G2, G3, G4, G5, G6))
    h =  np.vstack((h1, h2, h3, h4, h5, h6))

    # Input into model and solve
    cones = [
        qics.cones.QuantCondEntr((n, n), 1, iscomplex=True), 
        qics.cones.QuantEntr(n, iscomplex=True),
        qics.cones.PosSemidefinite(n, iscomplex=True)
    ]

    # Initialize model and solver objects
    model  = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
    qics.io.write_cbf(model, "ccea_" + str(n) + ".cbf")

    solver = qics.Solver(model)

    # Solve problem
    info = solver.solve()