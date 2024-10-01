import numpy as np
import qics
import qics.vectorize as vec
import qics.quantum as qu

## Entanglement-assisted rate distortion
#   min  t
#   s.t. Tr_2[X] = rho
#        (t, X) ∈ K_qce
#        <Δ, X> <= D

for n in range(2, 11, 2):
    np.random.seed(1)

    # D   = 0.5
    # rho = qu.random.density_matrix(n, iscomplex=True)
    D = 0.0
    rho = qu.random.pure_density_matrix(n, iscomplex=True)

    entr_rho = qu.entropy(rho)

    N = n * n
    sn = vec.vec_dim(n, compact=True, iscomplex=True)
    vN = vec.vec_dim(N, iscomplex=True)

    # Define objective function
    c = np.zeros((vN + 2, 1))
    c[0] = 1.0

    # Build linear constraint matrices
    tr2 = vec.lin_to_mat(lambda X: qu.p_tr(X, (n, n), 1), (N, n), iscomplex=True)
    Delta = vec.mat_to_vec(np.eye(N) - qu.purify(rho))
    # Tr_2[X] = rho
    A1 = np.hstack((np.zeros((sn, 1)), tr2, np.zeros((sn, 1))))
    b1 = vec.mat_to_vec(rho, compact=True)
    # <Δ, X> <= D
    A2 = np.hstack((np.zeros((1, 1)), Delta.T, np.ones((1, 1))))
    b2 = np.array([[D]])

    A = np.vstack((A1, A2))
    b = np.vstack((b1, b2))

    # Define cones to optimize over
    cones = [
        qics.cones.QuantCondEntr((n, n), 0, iscomplex=True),
        qics.cones.NonNegOrthant(1),
    ]

    # Initialize model and solver objects
    model = qics.Model(c=c, A=A, b=b, cones=cones, offset=entr_rho)
    qics.io.write_cbf(model, "qrd_" + str(n) + "_0.cbf")

    solver = qics.Solver(model)

    # Solve problem
    info = solver.solve()
