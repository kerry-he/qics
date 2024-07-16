import numpy as np
import scipy as sp
import math
import h5py

from utils import symmetric as sym, quantum as quant

def make_problem(file_name, description=["", ""], optval=0.0):
    # Problem is of the type
    # (QKD)     min    f(rho) = D( K(rho) || Z(K(rho)) )  (quantum relative entropy)
    #           s.t.   Gamma(rho) = gamma                 (affine constraint)
    #                  rho >= 0                           (rho positive semidefinite)
    # Obtained from https://math.uwaterloo.ca/~hwolkowi/henry/reports/ZGNQKDmainsolverUSEDforPUBLCNJuly31/

    # Problem data
    data = sp.io.loadmat(file_name)['Data']
    gamma = data['gamma'] if (data['gamma'][0, 0].dtype == 'complex128') else data['gamma'][0, 0].astype('double')
    Gamma = [G if G.dtype == 'complex128' else G.astype('double') for G in data['Gamma'][0, 0][0]]
    Klist = [K if K.dtype == 'complex128' else K.astype('double') for K in data['Klist'][0, 0][0]]
    Zlist = [Z if Z.dtype == 'complex128' else Z.astype('double') for Z in data['Zlist'][0, 0][0]]

    # Facially reduced problem data
    gamma_fr  = data['gamma_fr'] if (data['gamma_fr'][0, 0].dtype == 'complex128') else data['gamma_fr'][0, 0].astype('double')
    Gamma_fr  = [G if G.dtype == 'complex128' else G.astype('double') for G in data['Gamma_fr'][0, 0][0]]
    Klist_fr  = [K if K.dtype == 'complex128' else K.astype('double') for K in data['Klist_fr'][0, 0][0]]
    ZKlist_fr = [Z if Z.dtype == 'complex128' else Z.astype('double') for Z in data['ZKlist_fr'][0, 0][0]]

    optval = data['optval'][0, 0][0, 0]
    
    iscomplex = any([g.dtype == 'complex128' for g in gamma])
    iscomplex = any([G.dtype == 'complex128' for G in Gamma]) or iscomplex
    iscomplex = any([K.dtype == 'complex128' for K in Klist]) or iscomplex
    iscomplex = any([Z.dtype == 'complex128' for Z in Zlist]) or iscomplex

    no, ni = Klist[0].shape
    nc = np.size(gamma)

    vni = sym.vec_dim(ni, iscomplex=iscomplex)
    vno = sym.vec_dim(no, iscomplex=iscomplex)

    K_op = sym.lin_to_mat(lambda x : sym.congr_map(x, Klist), ni, no, iscomplex=iscomplex)
    ZK_op = sym.lin_to_mat(lambda x : sym.congr_map(sym.congr_map(x, Klist), Zlist), ni, no, iscomplex=iscomplex)
    Gamma_op = np.array([sym.mat_to_vec(G, iscomplex=iscomplex).T[0] for G in Gamma])

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
        raw.create_dataset('gamma', data=gamma)    # Constraint vector
        raw.create_dataset('Gamma', data=Gamma)    # Constraint matrix
        raw.create_dataset('Klist', data=Klist)    # Post-selection channel
        raw.create_dataset('Zlist', data=Zlist)    # Key map channel

        raw.create_dataset('gamma_fr',  data=gamma_fr)      # Constraint vector
        raw.create_dataset('Gamma_fr',  data=Gamma_fr)      # Constraint matrix
        raw.create_dataset('Klist_fr',  data=Klist_fr)      # Post-selection channel
        raw.create_dataset('ZKlist_fr', data=ZKlist_fr)    # Key map channel        
        
        # List of cones
        cones = f.create_group('cones')
        c0 = cones.create_dataset('0', data='qre')
        c0.attrs['complex'] = int(iscomplex)
        c0.attrs['dim'] = 1 + 2*vno
        c0.attrs['n'] = no

        c1 = cones.create_dataset('1', data='psd')
        c1.attrs['complex'] = int(iscomplex)
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
    file_name = 'problems/quant_key_rate/DMCV.mat'

    description = ["DMCV",       # Long description
                   "DMCV"]       # Short description

    make_problem(file_name, description=description)