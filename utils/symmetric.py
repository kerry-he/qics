import math
import numpy as np
import numba as nb

def vec_dim(side, hermitian=False):
    if hermitian:
        return side * side
    else:
        return side * (side + 1) // 2

def mat_dim(len, hermitian=False):
    if hermitian:
        side = math.isqrt(len)
        assert side * side == len
        return side
    else:
        side = math.isqrt(1 + 8 * len) // 2
        assert side * (side + 1) == 2 * len
        return side

def mat_to_vec(mat, rt2=None, hermitian=False):
    if hermitian:
        if mat.shape[0] <= 450:
            return mat_to_vec_complex_single(mat, rt2)
        else:
            return mat_to_vec_complex_parallel(mat, rt2)
    else:
        if mat.shape[0] <= 450:
            return mat_to_vec_single(mat, rt2)
        else:
            return mat_to_vec_parallel(mat, rt2)

@nb.njit
def mat_to_vec_single(mat, rt2):
    rt2 = np.sqrt(2.0) if (rt2 is None) else rt2

    n = mat.shape[0]
    vn = n*(n + 1) // 2
    vec = np.empty((vn, 1))

    k = 0
    for j in range(n):
        for i in range(j):
            vec[k] = mat[i, j] * rt2
            k += 1
        
        vec[k] = mat[j, j]
        k += 1
    
    return vec

@nb.njit(parallel=True)
def mat_to_vec_parallel(mat, rt2, hermitian=False):
    rt2 = np.sqrt(2.0) if (rt2 is None) else rt2

    n = mat.shape[0]
    vn = n*(n + 1) // 2
    vec = np.empty((vn, 1))

    for j in nb.prange(n):
        for i in range(j):
            k = i + (j * (j + 1)) // 2
            vec[k] = mat[i, j] * rt2

        k = j + (j * (j + 1)) // 2
        vec[k] = mat[j, j]
    
    return vec

@nb.njit
def mat_to_vec_complex_single(mat, rt2):
    rt2 = np.sqrt(2.0) if (rt2 is None) else rt2

    n = mat.shape[0]
    vn = n*n
    vec = np.empty((vn, 1))

    k = 0
    for j in range(n):
        for i in range(j):
            vec[k] = mat[i, j].real * rt2
            k += 1
            vec[k] = mat[i, j].imag * rt2
            k += 1
            
        vec[k] = mat[j, j].real
        k += 1

    return vec


@nb.njit(parallel=True)
def mat_to_vec_complex_parallel(mat, rt2):
    rt2 = np.sqrt(2.0) if (rt2 is None) else rt2

    n = mat.shape[0]
    vn = n*n
    vec = np.empty((vn, 1))

    for j in nb.prange(n):
        for i in range(j):
            k = 2*i + j * j
            vec[k]     = mat[i, j].real * rt2
            vec[k + 1] = mat[i, j].imag * rt2
        
        k = 2*j + j * j
        vec[k] = mat[j, j].real
    
    return vec

def vec_to_mat(vec, irt2=None, hermitian=False):
    if hermitian:
        if vec.size <= 1000000:
            return vec_to_mat_complex_single(vec, irt2)
        else:
            return vec_to_mat_complex_parallel(vec, irt2)
    else:        
        if vec.size <= 1280800:
            return vec_to_mat_single(vec, irt2)
        else:
            return vec_to_mat_parallel(vec, irt2)

@nb.njit
def vec_to_mat_single(vec, irt2):
    irt2 = np.sqrt(0.5) if (irt2 is None) else irt2

    vn = vec.size
    n = int(math.sqrt(1 + 8 * vn) // 2)
    mat = np.empty((n, n))

    k = 0
    for j in range(n):
        for i in range(j):
            mat[i, j] = vec[k, 0] * irt2
            mat[j, i] = mat[i, j]
            k += 1
        
        mat[j, j] = vec[k, 0]
        k += 1

    return mat

@nb.njit(parallel=True)
def vec_to_mat_parallel(vec, irt2):
    irt2 = np.sqrt(0.5) if (irt2 is None) else irt2

    vn = vec.size
    n = int(math.sqrt(1 + 8 * vn) // 2)
    mat = np.empty((n, n))

    for j in nb.prange(n):
        for i in range(j):
            k = i + (j * (j + 1)) // 2
            mat[i, j] = vec[k, 0] * irt2
            mat[j, i] = mat[i, j]
        
        k = i + (j * (j + 1)) // 2
        mat[j, j] = vec[k, 0]

    return mat

@nb.njit
def vec_to_mat_complex_single(vec, irt2):
    irt2 = np.sqrt(0.5) if (irt2 is None) else irt2

    vn = vec.size
    n = int(np.sqrt(vn))
    mat = np.zeros((n, n), dtype='complex128')

    for j in range(n):
        for i in range(j):
            k = 2*i + j * j
            mat[i, j] = (vec[k, 0] + vec[k + 1, 0] * 1j) * irt2
            mat[j, i] = mat[i, j].conjugate()

        k = 2*j + j * j
        mat[j, j] = vec[k, 0]

    return mat    

@nb.njit(parallel=True)
def vec_to_mat_complex_parallel(vec, irt2):
    irt2 = np.sqrt(0.5) if (irt2 is None) else irt2

    vn = vec.size
    n = int(np.sqrt(vn))
    mat = np.empty((n, n), dtype='complex128')

    for j in nb.prange(n):
        for i in range(j):
            k = 2*i + j * j
            mat[i, j] = (vec[k, 0] + vec[k + 1, 0] * 1j) * irt2
            mat[j, i] = mat[i, j].conjugate()

        k = 2*j + j * j
        mat[j, j] = vec[k, 0]

    return mat    

def p_tr(mat, sys, dim):
    (n0, n1) = dim
    return np.trace(mat.reshape(n0, n1, n0, n1), axis1=sys, axis2=2+sys)

def p_tr_multi(out, mat, sys, dim):
    (n0, n1) = dim
    np.trace(mat.reshape(-1, n0, n1, n0, n1), axis1=1+sys, axis2=3+sys, out=out)
    return out

def i_kr(mat, sys, dim):
    (n0, n1) = dim
    out = np.zeros((n0*n1, n0*n1), dtype=mat.dtype)
    if sys == 1:
        # Perform (mat kron I)
        r = np.arange(n1)
        out.reshape(n0, n1, n0, n1)[:, r, :, r] = mat
    else:
        # Perform (I kron mat)
        r = np.arange(n0)
        out.reshape(n0, n1, n0, n1)[r, :, r, :] = mat
    return out

def i_kr_multi(out, mat, sys, dim):
    (n0, n1) = dim
    out.fill(0.)
    if sys == 1:
        # Perform (mat kron I)
        r = np.arange(n1)
        out.reshape(-1, n0, n1, n0, n1)[:, :, r, :, r] = mat
    else:
        # Perform (I kron mat)
        r = np.arange(n0)
        out.reshape(-1, n0, n1, n0, n1)[:, r, :, r, :] = mat
    return out

def p_transpose(mat, sys, dim):
    # Partial transpose operation: M_ij,kl -> M_kj,il if sys == 0, or
    #                              M_ij,kl -> M_il,kj if sys == 1
    (n0, n1) = dim
    assert sys == 0 or sys == 1

    temp = mat.reshape(n0, n1, n0, n1)

    if sys == 0:
        temp = temp.transpose(2, 1, 0, 3)
    elif sys == 1:
        temp = temp.transpose(0, 3, 2, 1)

    return temp.reshape(n0*n1, n0*n1)


def lin_to_mat(lin, ni, no, hermitian=False):
    # Returns the matrix representation of a linear operator from (ni x ni) symmetric
    # matrices to (no x no) symmetric matrices given as a function handle
    vni = vec_dim(ni, hermitian=hermitian)
    vno = vec_dim(no, hermitian=hermitian)
    mat = np.zeros((vno, vni))

    rt2  = np.sqrt(2.0)
    irt2 = np.sqrt(0.5)

    for k in range(vni):
        H = np.zeros((vni, 1))
        H[k] = 1.0
        H_mat = vec_to_mat(H, irt2, hermitian=hermitian)
        lin_H = lin(H_mat)
        vec_out = mat_to_vec(lin_H, rt2, hermitian=hermitian)
        mat[:, [k]] = vec_out

    return mat

def congr_map(x, Klist, adjoint=False):
    # Compute congruence map
    if adjoint:
        return sum([K.conj().T @ x @ K for K in Klist])   
    else:
        return sum([K @ x @ K.conj().T for K in Klist])