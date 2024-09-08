import numpy as np
import scipy as sp
import qics
import qics.vectorize as vec
import os


## Quantum key rate
#   min  S( G(X) || Z(G(X)) )
#   s.t. <Ai, X> = bi
#        X >= 0

# To be used with .mat files from either:
# - https://www.math.uwaterloo.ca/~hwolkowi/henry/reports/ZGNQKDmainsolverUSEDforPUBLCNJuly31/
# - https://github.com/kerry-he/qrep-structure/tree/main/data
folder = "./examples/benchmark/qkd/"
fnames = os.listdir(folder)


def dds_fr(Klist, Zlist):
    nk = K_list[0].shape[0]

    KK = sum([K @ K.conj().T for K in Klist])
    ZKKZ = sum([Z @ KK @ Z.conj().T for Z in Zlist])
    ZKlist = [Z @ K for Z in Zlist for K in Klist]

    D, U = np.linalg.eigh(ZKKZ)

    KKnzidx = np.where(D > 1e-12)[0]
    nk_fr = np.size(KKnzidx)

    if nk == nk_fr:
        return Klist, ZKlist, nk_fr

    Klist = [U[:, KKnzidx].conj().T @ K for K in Klist]
    ZKlist = [U[:, KKnzidx].conj().T @ ZK for ZK in ZKlist]

    return Klist, ZKlist, nk_fr


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
    vni = vec.vec_dim(ni, iscomplex=iscomplex, compact=True)
    sno = vec.vec_dim(no, iscomplex=iscomplex, compact=False)
    sni = vec.vec_dim(ni, iscomplex=iscomplex, compact=False)

    # Define objective function
    c = np.zeros((1 + vni, 1))
    c[0] = 1.0

    # Build linear constraints
    A = np.zeros((nc, 1 + vni))
    for i in range(nc):
        A[i, 1:] = vec.mat_to_vec(Gamma[i].astype(dtype), compact=True).ravel()
    b = gamma

    # K_list, ZK_list, nf = dds_fr(K_list, Z_list)
    # snf    = vec.vec_dim(nf, iscomplex=iscomplex, compact=False)

    K_mtx = vec.lin_to_mat(
        lambda X: sum([K @ X @ K.conj().T for K in K_list]),
        (ni, no),
        iscomplex=iscomplex,
        compact=(True, False),
    )
    Z_mtx = vec.lin_to_mat(
        lambda X: sum([Z @ X @ Z.conj().T for Z in Z_list]),
        (no, no),
        iscomplex=iscomplex,
        compact=(False, False),
    )

    G = -np.vstack(
        (
            np.hstack((np.array([[1.0]]), np.zeros((1, vni)))),
            np.hstack((np.zeros((sno, 1)), K_mtx)),
            np.hstack((np.zeros((sno, 1)), Z_mtx @ K_mtx)),
            np.hstack((np.zeros((sni, 1)), vec.eye(ni, iscomplex=iscomplex).T)),
        )
    )

    h = np.zeros((G.shape[0], 1))

    # Input into model and solve
    cones = [
        qics.cones.QuantRelEntr(no, iscomplex=iscomplex),
        qics.cones.PosSemidefinite(ni, iscomplex=iscomplex),
    ]

    # Initialize model and solver objects
    model = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
    qics.io.write_cbf(model, "qkd_" + fname[:-4] + ".cbf")
    solver = qics.Solver(model)

    # Solve problem
    info = solver.solve()
