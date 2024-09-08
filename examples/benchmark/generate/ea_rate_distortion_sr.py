import numpy as np
import math

import qics
import qics.vectorize as vec
import qics.quantum as qu


def purify(eig):
    n = np.size(eig)

    vec = np.zeros((n * n, 1))
    for i, ii in enumerate(range(0, n * n, n + 1)):
        vec[ii] = math.sqrt(max(0, eig[i]))

    return vec @ vec.T


for n in [2, 4, 8, 16, 32, 64]:
    np.random.seed(1)
    # Define dimensions
    N = n * n
    vn = vec.vec_dim(n, compact=True)
    vN = vec.vec_dim(N, compact=True)

    # Rate-distortion problem data
    rho = qu.random.pure_density_matrix(n, iscomplex=True)
    eig_A = np.linalg.eigvalsh(rho)
    rho_A = np.diag(eig_A)
    rho_AR = purify(eig_A)
    entr_A = qu.entropy(eig_A)

    Delta = np.eye(N) - rho_AR
    Delta_X = vec.mat_to_vec(Delta[:: n + 1, :: n + 1], compact=True)
    D = 0.0

    # Build problem model
    IDX = np.zeros((n, n), "uint64")
    temp = np.arange(n * (n - 1)).reshape((n, n - 1)).T
    IDX[1:, :] += np.tril(temp).astype("uint64")
    IDX[:-1, :] += np.triu(temp, 1).astype("uint64")
    A_y = np.zeros((n, n * (n - 1)))
    A_X = np.zeros((n, vn))
    for i in range(n):
        idx = IDX[i]
        idx = np.delete(idx, i)
        A_y[i, idx] = 1.0

        temp = np.zeros((n, n))
        temp[i, i] = 1.0
        A_X[[i], :] = vec.mat_to_vec(temp, compact=True).T

    A = np.hstack(
        (np.zeros((n, 1)), np.zeros((n, 1)), A_y, A_X)
    )  # Partial trace constraint
    b = eig_A.reshape((-1, 1))

    G3_y = np.zeros((n * (n - 1), n * (n - 1)))
    G3_X = np.zeros((n * (n - 1), vn))
    k = 0
    for i in range(n):
        for j in range(n - 1):
            idx = IDX.T[i]
            idx = np.delete(idx, i)
            G3_y[k, idx] = 1.0

            temp = np.zeros((n, n))
            temp[i, i] = 1.0
            G3_X[[k], :] = vec.mat_to_vec(temp, compact=True).T

            k += 1

    G6_y = np.zeros((n * n, n * (n - 1)))
    G6_X = np.zeros((n * n, vn))
    k = 0
    for j in range(n):
        for i in range(n):
            if i == j:
                idx = IDX.T[j]
                idx = np.delete(idx, j)
                G6_y[k, idx] = 1.0

                temp = np.zeros((n, n))
                temp[j, j] = 1.0
                G6_X[[k], :] = vec.mat_to_vec(temp, compact=True).T

            k += 1

    G1 = -np.hstack(
        (np.ones((1, 1)), np.zeros((1, 1)), np.zeros((1, n * (n - 1) + vn)))
    )  # t
    G2 = -np.hstack(
        (
            np.zeros((n * (n - 1), 1)),
            np.zeros((n * (n - 1), 1)),
            np.eye((n * (n - 1))),
            np.zeros((n * (n - 1), vn)),
        )
    )  # p
    G3 = -np.hstack(
        (np.zeros((n * (n - 1), 1)), np.zeros((n * (n - 1), 1)), G3_y, G3_X)
    )  # q
    G4 = -np.hstack(
        (np.zeros((1, 1)), np.ones((1, 1)), np.zeros((1, n * (n - 1) + vn)))
    )  # t
    G5 = -np.hstack(
        (
            np.zeros((n * n, 1)),
            np.zeros((n * n, 1)),
            np.zeros((n * n, n * (n - 1))),
            vec.eye(n).T,
        )
    )  # X
    G6 = -np.hstack((np.zeros((n * n, 1)), np.zeros((n * n, 1)), G6_y, G6_X))  # Y
    G7 = np.hstack(
        (np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, n * (n - 1))), Delta_X.T)
    )  # Distortion
    G = np.vstack((G1, G2, G3, G4, G5, G6, G7))

    h = np.zeros((1 + 2 * n * (n - 1) + 1 + 2 * n * n + 1, 1))
    h[-1] = D

    c = np.zeros((1 + n * (n - 1) + vn + 1, 1))
    c[0] = 1.0
    c[1] = 1.0

    # Input into model and solve
    cones = [
        qics.cones.ClassRelEntr(n * (n - 1)),
        qics.cones.QuantRelEntr(n),
        qics.cones.NonNegOrthant(1),
    ]
    model = qics.Model(c, A, b, G, h, cones=cones, offset=entr_A)
    qics.io.write_cbf(model, "qrd_sr_" + str(n) + "_0.cbf")
