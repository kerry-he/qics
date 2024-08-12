import numpy as np
import scipy as sp
import math
import h5py

from utils import symmetric as vec, quantum as quant

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
    
    iscomplex = any([g.dtype == 'complex128' for g in gamma])
    iscomplex = any([G.dtype == 'complex128' for G in Gamma]) or iscomplex
    iscomplex = any([K.dtype == 'complex128' for K in Klist]) or iscomplex
    iscomplex = any([Z.dtype == 'complex128' for Z in Zlist]) or iscomplex

    no, ni = Klist[0].shape
    nc     = np.size(gamma_fr)

    vni = vec.vec_dim(ni, iscomplex=iscomplex)
    vno = vec.vec_dim(no, iscomplex=iscomplex)

    # Do our own symmetry reduction (NOTE: Assumes that input has not been symmetry reduced)
    nk = Zlist[0].shape[0]
    KK = sum([K @ K.conj().T for K in Klist])
    ZKKZ = sum([Z @ KK @ Z.conj().T for Z in Zlist])
    
    Dzkkz, Uzkkz = np.linalg.eigh(ZKKZ)
    ZKKZnzidx = np.where(Dzkkz > 1e-12)[0]
    nk_fr = np.size(ZKKZnzidx)
    vnz_fr = vec.vec_dim(nk_fr, iscomplex=iscomplex)

    if nk == nk_fr:
        Q = np.eye(nk)
    else:
        Q = Uzkkz[:, ZKKZnzidx]

    Klist_new  = [Q.conj().T @ K for K in Klist]
    ZKlist_new = [Q.conj().T @ Z @ K for Z in Zlist for K in Klist]

    K_op     = vec.lin_to_mat(lambda x : vec.apply_kraus(x, Klist_new), ni, nk_fr, iscomplex=iscomplex)
    ZK_op    = vec.lin_to_mat(lambda x : vec.apply_kraus(x, ZKlist_new), ni, nk_fr, iscomplex=iscomplex)

    K_op_alt     = lin_to_mat_alt(lambda x : vec.apply_kraus(x, Klist_new), ni, nk_fr, iscomplex=iscomplex)
    ZK_op_alt    = lin_to_mat_alt(lambda x : vec.apply_kraus(x, ZKlist_new), ni, nk_fr, iscomplex=iscomplex)
    eye_alt      = lin_to_mat_alt(lambda x : x, ni, ni, iscomplex=iscomplex)
    vnz_fr_alt   = vec.vec_dim(2*nk_fr, iscomplex=False)
    vni_alt      = vec.vec_dim(2*ni, iscomplex=False)

    Gamma_op = np.array([vec.mat_to_vec(G, iscomplex=iscomplex).T[0] for G in Gamma_fr])

    # Build problem model
    A = np.hstack((np.zeros((nc, 1)), Gamma_op))
    b = gamma_fr

    c = np.zeros((1 + vni, 1))
    c[0] = 1.

    G1 = np.hstack((np.ones((1, 1)), np.zeros((1, vni))))
    G2 = np.hstack((np.zeros((vnz_fr, 1)), K_op))
    G3 = np.hstack((np.zeros((vnz_fr, 1)), ZK_op))
    G4 = np.hstack((np.zeros((vni, 1)), np.eye(vni)))
    G = -np.vstack((G1, G2, G3, G4))

    h = np.zeros((1 + 2 * vnz_fr + vni, 1))

    # Alternative modelling which uses Karimi & Tuncel method for encoding Hermitian matrices as real symmetric matrix
    G1 = np.hstack((np.ones((1, 1)) * 0.5, np.zeros((1, vni))))
    G2 = np.hstack((np.zeros((vnz_fr_alt, 1)), K_op_alt))
    G3 = np.hstack((np.zeros((vnz_fr_alt, 1)), ZK_op_alt))
    G4 = np.hstack((np.zeros((vni_alt, 1)), eye_alt))
    G_alt = -np.vstack((G1, G2, G3, G4))    

    h_alt = np.zeros((1 + 2 * vnz_fr_alt + vni_alt, 1))

    # Make A and G sparse
    A_sparse = sp.sparse.coo_array(A)
    G_sparse = sp.sparse.coo_array(G)
    G_alt_sparse = sp.sparse.coo_array(G_alt)

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
            'type':      'qre',
            'iscomplex': True,
            'dim':       1 + 2*vnz_fr,
            'n':         nk_fr
        }, 
        {
            'type':      'psd',
            'iscomplex': True,
            'dim':       vni,
            'n':         ni
        }
    ]

    cones_alt = [
        {
            'type':      'qre',
            'iscomplex': False,
            'dim':       1 + 2*vnz_fr_alt,
            'n':         2*nk_fr
        }, 
        {
            'type':      'psd',
            'iscomplex': False,
            'dim':       vni_alt,
            'n':         2*ni
        }
    ]    

    data = {
        'description': 'Quantum key rate problem, ' + description[0],
        'optval': optval,
        'offset': 0.0,
        'complex': True,
        'raw': raw,
        'cones': cones,
        'cones_alt': cones_alt,
        'c': c,
        'b': b,
        'h': h,
        'A': A_sparse,
        'G': G_sparse,
        'G_alt': G_alt_sparse,
        'h_alt': h_alt
    }

    sp.io.savemat('qkd' + "_" + description[1] + '.mat', data, do_compression=True)



def lin_to_mat_alt(lin, ni, no, iscomplex):
    # Returns the matrix representation of a linear operator from (ni x ni) symmetric
    # matrices to (no x no) symmetric matrices given as a function handle
    vni = vec.vec_dim(ni, iscomplex=iscomplex)
    vno = vec.vec_dim(2*no, iscomplex=False) if iscomplex else vec.vec_dim(no, iscomplex=False)
    mat = np.zeros((vno, vni))

    rt2  = np.sqrt(2.0)
    irt2 = np.sqrt(0.5)

    for k in range(vni):
        H = np.zeros((vni, 1))
        H[k] = 1.0
        H_mat = vec.vec_to_mat(H, irt2, iscomplex=iscomplex)
        lin_H = lin(H_mat)
        if iscomplex:
            lin_H_real = lin_H.real
            lin_H_imag = lin_H.imag
            lin_H = np.vstack((
                np.hstack((lin_H_real, -lin_H_imag)),
                np.hstack((lin_H_imag,  lin_H_real)),
            ))
        vec_out = vec.mat_to_vec(lin_H, rt2, iscomplex=False)
        mat[:, [k]] = vec_out

    return mat



if __name__ == "__main__":
    problem = 'dprBB84_1_02_15'
    file_name = 'problems/quant_key_rate/' + problem + '.mat'

    description = [problem,       # Long description
                   problem]       # Short description

    make_problem(file_name, description=description)