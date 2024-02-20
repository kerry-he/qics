import numpy as np
import scipy as sp
import h5py

from utils import symmetric as sym

np.random.seed(1)

def make_problem(n, X, description=["", ""], offset=0.0, optval=0.0):
    vn = sym.vec_dim(n)

    # Objective function
    c = np.zeros((n, 1))
    c[0] = 1.

    # Build problem model
    A = np.zeros((0, n))
    b = np.zeros((0, 1))

    G1 = np.hstack((np.ones((1, 1)), np.zeros((1, n - 1))))         # QRE t
    G2 = np.hstack((np.zeros((vn, 1)), np.zeros((vn, n - 1))))      # QRE X
    G3 = np.zeros((vn, n))                                          # QRE Y
    for i in range(n - 1):
        H = np.zeros((n, n))
        H[i, i + 1] = np.sqrt(0.5)
        H[i + 1, i] = np.sqrt(0.5)
        G3[:, [1 + i]] = sym.mat_to_vec(H)
    G = -np.vstack((G1, G2, G3))

    h = np.zeros((1 + 2 * vn, 1))
    h[1:vn+1] = sym.mat_to_vec(X)
    h[vn+1:]  = sym.mat_to_vec(np.eye(n))


    # Make A and G sparse
    A_sparse = sp.sparse.coo_array(A)
    G_sparse = sp.sparse.coo_array(G)

    A_vij = np.vstack((A_sparse.data, A_sparse.row, A_sparse.col))
    G_vij = np.vstack((G_sparse.data, G_sparse.row, G_sparse.col))

    # Write problem data to file
    with h5py.File('ncm_tridiag_' + str(n) + "_" + description[1] + '.hdf5', 'w') as f:
        # Auxiliary problem information
        f.attrs['description'] = "Nearest tridiagonal correlation matrix problem, n=" + str(n) + ", " + description[0]
        f.attrs['offset'] = offset
        f.attrs['optval'] = optval

        # Raw problem data
        raw = f.create_group('raw')
        raw.create_dataset('n', data=n)         # Dimension of matrix
        raw.create_dataset('rho', data=X)       # Input matrix
        
        # List of cones
        cones = f.create_group('cones')
        c0 = cones.create_dataset('0', data='qre')
        c0.attrs['complex'] = 0
        c0.attrs['dim'] = 1 + 2*vn
        c0.attrs['n'] = n

        # Objective and constraint matrices
        data = f.create_group('data')
        data.create_dataset('c', data=c)
        data.create_dataset('b', data=b)
        data.create_dataset('h', data=h)
        data_A = data.create_dataset('A', data=A_vij)
        data_A.attrs['sparse'] = 1
        data_A.attrs['shape'] = A.shape
        data_G = data.create_dataset('G', data=G_vij)
        data_G.attrs['sparse'] = 1
        data_G.attrs['shape'] = G.shape


if __name__ == "__main__":
    n = 3
    X = 2 * np.eye(n)

    description = ["rho=2*eye(n)",
                   "eye"]
    optval = 2.0 * np.log(2.0) * n

    make_problem(n, X, description=description, optval=optval)