import numpy as np
import scipy as sp
import math
import h5py

from utils import symmetric as vec, quantum as quant

def make_problem(ni, no, rho, Delta, D, description=["", ""], optval=0.0):
    # Define dimensions
    N = no * ni
    vni = vec.vec_dim(ni)
    vN = vec.vec_dim(N)

    # Rate-distortion problem data
    entr_A = quant.quantEntropy(rho)

    # Build problem model
    tr2 = vec.lin_to_mat(lambda x : qu.p_tr(x, (no, ni), 0), no*ni, ni)
    ikr_tr1 = vec.lin_to_mat(lambda x : qu.i_kr(qu.p_tr(x, (no, ni), 1), (no, ni), 1), no*ni, no*ni)

    A = np.hstack((np.zeros((vni, 1)), tr2))
    b = vec.mat_to_vec(rho)

    c = np.zeros((vN + 1, 1))
    c[0] = 1.

    G1 = np.hstack((np.ones((1, 1)), np.zeros((1, vN))))            # t_qre
    G2 = np.hstack((np.zeros((vN, 1)), np.eye(vN)))                 # X_qre
    G3 = np.hstack((np.zeros((vN, 1)), ikr_tr1))                    # Y_qre
    G4 = np.hstack((np.zeros((1, 1)), -vec.mat_to_vec(Delta).T))    # nn
    G = -np.vstack((G1, G2, G3, G4))

    h = np.zeros((1 + vN*2 + 1, 1))
    h[-1] = D

    # Make A and G sparse
    A_sparse = sp.sparse.coo_array(A)
    G_sparse = sp.sparse.coo_array(G)

    A_vij = np.vstack((A_sparse.data, A_sparse.row, A_sparse.col))
    G_vij = np.vstack((G_sparse.data, G_sparse.row, G_sparse.col))

    # Write problem data to file
    with h5py.File('ea-rd_' + str(ni) + "_" + str(no) + "_" + description[1] + '.hdf5', 'w') as f:
        # Auxiliary problem information
        f.attrs['description'] = "Entanglement assisted rate-distortion problem, " + str(ni) + "-dimensional input, " + str(no) + "-dimensional output, " + description[0]
        f.attrs['offset'] = entr_A
        f.attrs['optval'] = optval

        # Raw problem data
        raw = f.create_group('raw')
        raw.create_dataset('ni',    data=ni)      # Dimension of input
        raw.create_dataset('no',    data=no)      # Dimension of output
        raw.create_dataset('rho',   data=rho)     # Input state
        raw.create_dataset('Delta', data=Delta)   # Distortion observable
        raw.create_dataset('D',     data=D)       # Maximum distortion
        
        # List of cones
        cones = f.create_group('cones')
        c0 = cones.create_dataset('0', data='qre')
        c0.attrs['complex'] = 0
        c0.attrs['dim'] = 1 + 2*vN
        c0.attrs['n'] = N

        c1 = cones.create_dataset('1', data='nn')
        c1.attrs['dim'] = 1

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
    n     = 4
    rho   = quant.randDensityMatrix(n)
    Delta = np.eye(n*n) - quant.purify(rho)
    D     = 0.5

    description = ["entanglement fidelity distortion observable",       # Long description
                   "ef"]                                                # Short description

    make_problem(n, n, rho, Delta, D, description=description)