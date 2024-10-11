import numpy as np

import qics
import qics.quantum as qu
import qics.vectorize as vec

for n in [15, 30, 45, 60, 75]:
    for alpha in [0.25, 0.75, 1.25]:
        np.random.seed(1)

        rho = qu.random.density_matrix(n, iscomplex=True)
        sigma = qu.random.density_matrix(n, iscomplex=True)

        # Define objective function
        if 0 < alpha and alpha < 0.5:
            cT = (1 - alpha) * vec.mat_to_vec(sigma)  # theta
            cX = np.zeros((2 * n * n, 1))  # I
            cY = alpha * vec.mat_to_vec(rho)  # omega

            cones = [qics.cones.OpPerspecEpi(n, alpha / (alpha - 1), iscomplex=True)]
        elif 0.5 <= alpha and alpha < 1:
            cT = alpha * vec.mat_to_vec(rho)
            cX = np.zeros((2 * n * n, 1))
            cY = (1 - alpha) * vec.mat_to_vec(sigma)

            cones = [qics.cones.OpPerspecEpi(n, 1 - 1 / alpha, iscomplex=True)]
        elif 1 < alpha:
            cT = alpha * vec.mat_to_vec(rho)
            cX = np.zeros((2 * n * n, 1))
            cY = -(1 - alpha) * vec.mat_to_vec(sigma)

            cones = [qics.cones.OpPerspecEpi(n, 1 - 1 / alpha, iscomplex=True)]

        c = np.vstack((cT, cX, cY))

        # Build linear constraint matrices
        vn = vec.vec_dim(n, compact=True, iscomplex=True)
        # X = I
        A = np.hstack(
            (
                np.zeros((vn, 2 * n * n)),
                vec.eye(n, iscomplex=True),
                np.zeros((vn, 2 * n * n)),
            )
        )
        b = vec.mat_to_vec(np.eye(n, dtype=np.complex128), compact=True)

        # Initialize model and solver objects
        model = qics.Model(c=c, A=A, b=b, cones=cones)
        qics.io.write_cbf(model, "mre_" + str(n) + "_" + str(alpha) + ".cbf")
        # solver = qics.Solver(model)

        # # Solve problem
        # info = solver.solve()
