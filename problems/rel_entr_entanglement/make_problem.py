import numpy as np
import scipy as sp
import h5py

from cones import *
from utils import symmetric as sym, quantum as quant

def make_problem(n, m, X, description=["", ""], offset=0.0, optval=0.0):
    # Problem data
    N = n * m
    vN = sym.vec_dim(N)

    # Build problem model
    c = np.zeros((1 + vN, 1))
    c[0] = 1.0

    A = np.hstack((np.zeros((1, 1)), sym.mat_to_vec(np.eye(N)).T))
    b = np.ones((1, 1))

    p_transpose = sym.lin_to_mat(lambda x : sym.p_transpose(x, (n, m), 1), n*m, n*m)

    G0 = np.hstack((np.ones((1, 1)), np.zeros((1, vN))))
    G1 = np.hstack((np.zeros((vN, 1)), np.zeros((vN, vN))))
    G2 = np.hstack((np.zeros((vN, 1)), np.eye(vN)))
    G3 = np.hstack((np.zeros((vN, 1)), p_transpose))
    G = -np.vstack((G0, G1, G2, G3))

    h = np.zeros((1 + 3*vN, 1))
    h[1:1+vN] = sym.mat_to_vec(X)

    h = np.zeros((1 + 3*vN, 1))
    h[1:1+vN] = sym.mat_to_vec(X)

    # Make A and G sparse
    A_sparse = sp.sparse.coo_array(A)
    G_sparse = sp.sparse.coo_array(G)

    A_vij = np.vstack((A_sparse.data, A_sparse.row, A_sparse.col))
    G_vij = np.vstack((G_sparse.data, G_sparse.row, G_sparse.col))

    # Write problem data to file
    with h5py.File('ree_' + str(n) + '_' + str(m) + '_' + description[1] + '.hdf5', 'w') as f:
        # Auxiliary problem information
        f.attrs['description'] = 'Relative entropy of entanglement using PPT relaxation, n=' + str(n) + ', m=' + str(m) + ', ' + description[0]
        f.attrs['offset'] = offset
        f.attrs['optval'] = optval

        # Raw problem data
        raw = f.create_group('raw')
        raw.create_dataset('n'  , data=n)       # Dimension of system A
        raw.create_dataset('m'  , data=m)       # Dimension of system B
        raw.create_dataset('rho', data=X)       # Input matrix
        
        # List of cones
        cones = f.create_group('cones')
        c0 = cones.create_dataset('0', data='qre')
        c0.attrs['complex'] = 0
        c0.attrs['dim'] = 1 + 2*vN
        c0.attrs['n'] = N

        c1 = cones.create_dataset('1', data='psd')
        c1.attrs['complex'] = 0
        c1.attrs['dim'] = vN
        c1.attrs['n'] = N


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
    n = 8
    m = 8
    X = quant.randDensityMatrix(n * m)

    description = ["rho=randDensity(n*m)",
                   "rand"]

    make_problem(n, m, X, description=description)