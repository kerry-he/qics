import numpy as np

import qics
import qics.vectorize as vec

np.random.seed(1)

# Problem data
for n in range(50, 301, 50):
    vn = vec.vec_dim(n)
    # M = 2 * np.eye(n)
    M = np.random.rand(n, 1)
    M = (M @ M.T)
    M = M / np.max(np.diag(M))

    # Build problem model
    c = np.zeros((n, 1))
    c[0] = 1.

    G1 = np.hstack((np.ones((1, 1)), np.zeros((1, n - 1))))         # QRE t
    G2 = np.hstack((np.zeros((vn, 1)), np.zeros((vn, n - 1))))      # QRE X
    G3 = np.zeros((vn, n))                                          # QRE Y
    for i in range(n - 1):
        H = np.zeros((n, n))
        H[i, i + 1] = np.sqrt(0.5)
        H[i + 1, i] = np.sqrt(0.5)
        G3[:, [1 + i]] = vec.mat_to_vec(H)
    G = -np.vstack((G1, G2, G3))

    h = np.zeros((1 + 2 * vn, 1))
    h[1:vn+1] = vec.mat_to_vec(M)
    h[vn+1:]  = vec.mat_to_vec(np.eye(n))

    # Input into model and solve
    cones = [qics.cones.QuantRelEntr(n)]

    # Initialize model and solver objects
    model  = qics.Model(c=c, G=G, h=h, cones=cones)
    qics.io.write_cbf(model, "nc_tri_" + str(n) + ".cbf")