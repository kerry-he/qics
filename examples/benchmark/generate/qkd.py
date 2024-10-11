import os

import numpy as np
import scipy as sp

import qics
import qics.vectorize as vec

## Quantum key rate
#   min  S( G(X) || Z(G(X)) )
#   s.t. <Ai, X> = bi
#        X >= 0

folder = "./qkd/"
fnames = os.listdir(folder)

for fname in fnames:
    data = sp.io.loadmat(folder + fname)
    gamma = data["gamma"]
    Gamma = list(data["Gamma"].ravel())
    K_list = list(data["Klist"].ravel())
    Z_list = list(data["Zlist"].ravel())

    iscomplex = np.iscomplexobj(Gamma) or np.iscomplexobj(K_list)
    dtype = np.complex128 if iscomplex else np.float64

    no, ni = np.shape(K_list[0])
    nc = np.size(gamma)
    vni = vec.vec_dim(ni, iscomplex=iscomplex, compact=False)

    # Define objective function
    c = np.zeros((1 + vni, 1))
    c[0] = 1.0

    # Build linear constraints
    A = np.zeros((nc, 1 + vni))
    for i in range(nc):
        A[i, 1:] = vec.mat_to_vec(Gamma[i].astype(dtype), compact=False).ravel()
    b = gamma

    # Input into model and solve
    cones = [qics.cones.QuantKeyDist(K_list, Z_list, iscomplex=iscomplex)]

    # Initialize model and solver objects
    model = qics.Model(c=c, A=A, b=b, cones=cones)
    qics.io.write_cbf(model, "qkd_" + fname[:-4] + ".cbf")
