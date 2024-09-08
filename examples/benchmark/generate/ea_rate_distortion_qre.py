import numpy as np
import qics
import qics.vectorize as vec
import qics.quantum as qu

## Entanglement-assisted rate distortion
#   min  t
#   s.t. Tr_2[X] = rho
#        (t, X) ∈ K_qce
#        <Δ, X> <= D

for n in range(2, 13, 2):
    np.random.seed(1)

    D = 0.5

    rho = qu.random.density_matrix(n, iscomplex=True)
    entr_rho = qu.quant_entropy(rho)

    N = n * n
    sn = vec.vec_dim(n, compact=True, iscomplex=True)
    sN = vec.vec_dim(N, compact=True, iscomplex=True)
    vN = vec.vec_dim(N, iscomplex=True)

    # Define objective function
    c = np.zeros((sN + 1, 1))
    c[0] = 1.0

    # Build linear constraint matrices
    tr2 = vec.lin_to_mat(
        lambda X: qu.p_tr(X, (n, n), 1), (N, n), compact=(True, True), iscomplex=True
    )
    ikr_tr1 = vec.lin_to_mat(
        lambda X: qu.i_kr(qu.p_tr(X, (n, n), 0), (n, n), 0),
        (N, N),
        compact=(True, False),
        iscomplex=True,
    )
    Delta = vec.mat_to_vec(np.eye(N) - qu.purify(rho), compact=True).T

    A = np.hstack((np.zeros((sn, 1)), tr2))
    b = vec.mat_to_vec(rho, compact=True)

    G1 = np.hstack((np.ones((1, 1)), np.zeros((1, sN))))  # t_qre
    G2 = np.hstack((np.zeros((vN, 1)), vec.eye(N, iscomplex=True).T))  # X_qre
    G3 = np.hstack((np.zeros((vN, 1)), ikr_tr1))  # Y_qre
    G4 = np.hstack((np.zeros((1, 1)), -Delta))  # LP
    G = -np.vstack((G1, G2, G3, G4))

    h = np.zeros((1 + vN * 2 + 1, 1))
    h[-1] = D

    # Define cones to optimize over
    cones = [qics.cones.QuantRelEntr(N, iscomplex=True), qics.cones.NonNegOrthant(1)]

    # Initialize model and solver objects
    model = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones, offset=entr_rho)
    qics.io.write_cbf(model, "qrd_qre_" + str(n) + ".cbf")
