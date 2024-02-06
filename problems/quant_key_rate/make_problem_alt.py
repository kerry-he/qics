import numpy as np
import scipy as sp
import math
import h5py

from utils import symmetric as sym, quantum as quant

def clean_cell(cell):
    out = np.empty((len(cell[0, 0][0]),), dtype=object)
    for (i, X) in enumerate(cell[0, 0][0]):
        out[i] = X if X.dtype == 'complex128' else X.astype('double')
    return out

def make_problem(file_name, description=["", ""], optval=0.0):
    # Problem is of the type
    # (QKD)     min    f(rho) = D( K(rho) || Z(K(rho)) )  (quantum relative entropy)
    #           s.t.   Gamma(rho) = gamma                 (affine constraint)
    #                  rho >= 0                           (rho positive semidefinite)
    # Obtained from https://math.uwaterloo.ca/~hwolkowi/henry/reports/ZGNQKDmainsolverUSEDforPUBLCNJuly31/

    # Problem data
    data = sp.io.loadmat(file_name)['Data']

    # Raw problem data
    gamma = data['gamma'] if (data['gamma'][0, 0].dtype == 'complex128') else data['gamma'][0, 0].astype('double')
    Gamma = clean_cell(data['Gamma'])
    Klist = clean_cell(data['Klist'])
    Zlist = clean_cell(data['Zlist'])

    # Facially reduced problem data
    gamma_fr  = data['gamma_fr'] if (data['gamma_fr'][0, 0].dtype == 'complex128') else data['gamma_fr'][0, 0].astype('double')
    Gamma_fr  = clean_cell(data['Gamma_fr'])
    Klist_fr  = clean_cell(data['Klist_fr'])
    ZKlist_fr = clean_cell(data['ZKlist_fr'])

    optval = data['optval'][0, 0][0, 0]
    
    hermitian = any([g.dtype == 'complex128' for g in gamma])
    hermitian = any([G.dtype == 'complex128' for G in Gamma]) or hermitian
    hermitian = any([K.dtype == 'complex128' for K in Klist]) or hermitian
    hermitian = any([Z.dtype == 'complex128' for Z in Zlist]) or hermitian

    no, ni = Klist[0].shape
    nc     = np.size(gamma)

    vni = sym.vec_dim(ni, hermitian=hermitian)
    vno = sym.vec_dim(no, hermitian=hermitian)

    K_op     = sym.lin_to_mat(lambda x : sym.congr_map(x, Klist), ni, no, hermitian=hermitian)
    ZK_op    = sym.lin_to_mat(lambda x : sym.congr_map(sym.congr_map(x, Klist), Zlist), ni, no, hermitian=hermitian)
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

    raw = {
        'gamma':     gamma,
        'Gamma':     Gamma,
        'Klist':     Klist,
        'Zlist':     Zlist,
        'gamma_fr':  gamma_fr,
        'Gamma_fr':  Gamma_fr,
        'Klist_fr':  Klist_fr,
        'ZKlist_fr': ZKlist_fr
    }

    # List of cones
    cones = [
        {
            'type':    'qre',
            'complex': hermitian,
            'dim':     1 + 2*vno,
            'n':       no
        }, 
        {
            'type':   'psd',
            'complex': hermitian,
            'dim':     1 + 2*vni,
            'n':       ni
        }
    ]

    data = {
        'description': 'Quantum key rate problem, ' + description[0],
        'optval': optval,
        'offset': 0.0,
        'raw': raw,
        'cones': cones,
        'c': c,
        'b': b,
        'h': h,
        'A': A_sparse,
        'G': G_sparse
    }

    sp.io.savemat('qkd' + "_" + description[1] + '.mat', data, do_compression=True)


if __name__ == "__main__":
    problem = 'TFQKD'
    file_name = 'problems/quant_key_rate/' + problem + '.mat'

    description = [problem,       # Long description
                   problem]       # Short description

    make_problem(file_name, description=description)