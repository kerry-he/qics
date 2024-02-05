import numpy as np
import scipy as sp
import math
import h5py

from utils import symmetric as sym, quantum as quant

def make_problem(file_name, hermitian, description=["", ""], optval=0.0):
    # Problem is of the type
    # (QKD)     min    f(rho) = D( K(rho) || Z(K(rho)) )  (quantum relative entropy)
    #           s.t.   Gamma(rho) = gamma                 (affine constraint)
    #                  rho >= 0                           (rho positive semidefinite)
    # Obtained from https://math.uwaterloo.ca/~hwolkowi/henry/reports/ZGNQKDmainsolverUSEDforPUBLCNJuly31/

    # Problem data
    data = sp.io.loadmat(file_name)
    gamma = data['gamma'] if (data['gamma'].dtype == 'complex128') else data['gamma'].astype('double')
    Gamma = [G if G.dtype == 'complex128' else G.astype('double') for G in data['Gamma'][:, 0]]
    Klist = [K if K.dtype == 'complex128' else K.astype('double') for K in data['VKVlist'][0, :]]
    ZKlist = [ZK if ZK.dtype == 'complex128' else ZK.astype('double') for ZK in data['VZKVlist'][0, :]]
    
    hermitian = any([g.dtype == 'complex128' for g in gamma])
    hermitian = any([G.dtype == 'complex128' for G in Gamma]) or hermitian
    hermitian = any([K.dtype == 'complex128' for K in Klist]) or hermitian
    hermitian = any([ZK.dtype == 'complex128' for ZK in ZKlist]) or hermitian

    no, ni = Klist[0].shape
    nc = np.size(gamma)

    vni = sym.vec_dim(ni, hermitian=hermitian)
    vno = sym.vec_dim(no, hermitian=hermitian)

    K_op = sym.lin_to_mat(lambda x : sym.apply_kraus(x, Klist), ni, no, hermitian=hermitian)
    ZK_op = sym.lin_to_mat(lambda x : sym.apply_kraus(x, ZKlist), ni, no, hermitian=hermitian)
    Gamma_op = np.array([sym.mat_to_vec(G, hermitian=hermitian).T[0] for G in Gamma])

    # Build problem model
    A = np.hstack((np.zeros((nc, 1)), Gamma_op))
    b = gamma

    c = np.zeros((1 + vni, 1))
    c[0] = 1.

    G1 = np.hstack((np.ones((1, 1)), np.zeros((1, vni))))
    G2 = np.hstack((np.zeros((vno, 1)), K_op))
    G3 = np.hstack((np.zeros((vno, 1)), ZK_op))
    G4 = np.hstack((np.zeros((vni, 1)), np.eye(vni)))
    G = -np.vstack((G1, G2, G3, G4))

    h = np.zeros((1 + 2 * vno + vni, 1))

    # Make A and G sparse
    A_sparse = sp.sparse.coo_array(A)
    G_sparse = sp.sparse.coo_array(G)

    A_vij = np.vstack((A_sparse.data, A_sparse.row, A_sparse.col))
    G_vij = np.vstack((G_sparse.data, G_sparse.row, G_sparse.col))    

    # Write problem data to file
    with h5py.File('qkd' + "_" + description[1] + '.hdf5', 'w') as f:
        # Auxiliary problem information
        f.attrs['description'] = "Quantum key rate problem, " + description[0]
        f.attrs['offset'] = 0.0
        f.attrs['optval'] = optval

        # Raw problem data
        raw = f.create_group('raw')
        raw.create_dataset('gamma', data=gamma)      # Constraint vector
        raw.create_dataset('Gamma', data=Gamma)      # Constraint matrix
        raw.create_dataset('Klist', data=Klist)      # Post-selection channel
        raw.create_dataset('ZKlist', data=ZKlist)    # Key map channel x Post-selection channel
        
        # List of cones
        cones = f.create_group('cones')
        c0 = cones.create_dataset('0', data='qre')
        c0.attrs['complex'] = int(hermitian)
        c0.attrs['dim'] = 1 + 2*vno
        c0.attrs['n'] = no

        c1 = cones.create_dataset('1', data='psd')
        c1.attrs['complex'] = int(hermitian)
        c1.attrs['dim'] = 1 + 2*vni
        c1.attrs['n'] = ni

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
    file_name = 'problems/quant_key_rate/ebBB84.mat'
    hermitian = False

    description = ["ebBB84",       # Long description
                   "ebBB84"]       # Short description

    make_problem(file_name, hermitian, description=description)