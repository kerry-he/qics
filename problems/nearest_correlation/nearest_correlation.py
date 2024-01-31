import numpy as np
import scipy as sp
import h5py

from utils import symmetric as sym

def ncm_problem(n, X, description=[None, None], offset=0.0, optval=None):
    vn = sym.vec_dim(n)

    # Objective function
    c = np.zeros((1 + vn, 1))
    c[0] = 1.

    # Linear constraints
    A = np.zeros((n, 1 + vn))
    for i in range(n):
        H = np.zeros((n, n))
        H[i, i] = 1.0
        A[[i], 1:] = sym.mat_to_vec(H).T
    b = np.ones((n, 1))

    # Cone constraints
    G1 = np.hstack((np.ones((1, 1)), np.zeros((1, vn))))
    G2 = np.hstack((np.zeros((vn, 1)), np.zeros((vn, vn))))
    G3 = np.hstack((np.zeros((vn, 1)), np.eye(vn)))
    G = -np.vstack((G1, G2, G3))

    h = np.zeros((1 + 2 * vn, 1))
    h[1:vn+1] = sym.mat_to_vec(X)

    # Make A and G sparse
    A_sparse = sp.sparse.coo_array(A)
    G_sparse = sp.sparse.coo_array(G)

    A_vij = np.vstack((A_sparse.data, A_sparse.row, A_sparse.col))
    G_vij = np.vstack((G_sparse.data, G_sparse.row, G_sparse.col))

    # Write problem data to file
    with h5py.File('ncm_' + str(n) + "_" + description[1] + '.hdf5', 'w') as f:
        # Auxiliary problem information
        f.attrs['description'] = description[0]
        f.attrs['offset'] = offset
        f.attrs['optval'] = optval

        # Raw problem data
        raw = f.create_group('raw')
        raw.create_dataset('n', data=n)         # Dimension of matrix
        raw.create_dataset('rho', data=X)       # Input matrix
        
        # List of cones
        cones = f.create_group('cones')
        c0 = cones.create_dataset('0', data='qre')
        c0.attrs['complex'] = False
        c0.attrs['dim'] = 1 + 2*vn
        c0.attrs['n'] = n

        # Objective and constraint matrices
        data = f.create_group('data')
        data.create_dataset('c', data=c)
        data.create_dataset('b', data=b)
        data.create_dataset('h', data=h)
        data_A = data.create_dataset('A', data=A_vij)
        data_A.attrs['sparse'] = True
        data_A.attrs['shape'] = A.shape
        data_G = data.create_dataset('G', data=G_vij)
        data_G.attrs['sparse'] = True
        data_G.attrs['shape'] = G.shape


if __name__ == "__main__":
    n = 2
    X = 2 * np.eye(n)

    description = ["Nearest correlation matrix problem, N=" + str(n) + ", rho=2*eye(N)",
                   "eye"]
    optval = 2.0 * np.log(2.0) * n

    ncm_problem(n, X, description=description, optval=optval)