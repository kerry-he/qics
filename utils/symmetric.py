import math
import numpy as np
import numba as nb

def vec_dim(side):
    return side * (side + 1) // 2

def mat_dim(len):
    side = math.isqrt(1 + 8 * len) // 2
    assert side * (side + 1) == 2 * len
    return side

def mat_to_vec(mat, rt2=None):
    if mat.shape[0] <= 400:
        return mat_to_vec_single(mat, rt2)
    else:
        return mat_to_vec_parallel(mat, rt2)

@nb.njit
def mat_to_vec_single(mat, rt2):
    rt2 = np.sqrt(2.0) if (rt2 is None) else rt2

    n = mat.shape[0]
    vn = n * (n + 1) // 2
    vec = np.empty((vn, 1))

    k = 0
    for j in range(n):
        for i in range(j + 1):
            if i == j:
                vec[k] = mat[i, j]
            else:
                vec[k] = mat[i, j] * rt2
            k += 1
    
    return vec

@nb.njit(parallel=True)
def mat_to_vec_parallel(mat, rt2):
    rt2 = np.sqrt(2.0) if (rt2 is None) else rt2

    n = mat.shape[0]
    vn = n * (n + 1) // 2
    vec = np.empty((vn, 1))

    for j in nb.prange(n):
        for i in range(j + 1):
            k = i + (j * (j + 1)) // 2
            if i == j:
                vec[k] = mat[i, j]
            else:
                vec[k] = mat[i, j] * rt2
    
    return vec

def vec_to_mat(vec, irt2=None):
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
        for i in range(j + 1):
            if i == j:
                mat[i, j] = vec[k, 0]
            else:
                mat[i, j] = vec[k, 0] * irt2
                mat[j, i] = vec[k, 0] * irt2
            k += 1

    return mat

@nb.njit(parallel=True)
def vec_to_mat_parallel(vec, irt2):
    irt2 = np.sqrt(0.5) if (irt2 is None) else irt2

    vn = vec.size
    n = int(math.sqrt(1 + 8 * vn) // 2)
    mat = np.empty((n, n))

    for j in nb.prange(n):
        for i in range(j + 1):
            k = i + (j * (j + 1)) // 2
            if i == j:
                mat[i, j] = vec[k, 0]
            else:
                mat[i, j] = vec[k, 0] * irt2
                mat[j, i] = vec[k, 0] * irt2

    return mat

def p_tr(mat, sys, dim):
    (n0, n1) = dim
    assert n0 * n1 == np.size(mat, 0)
    assert sys == 0 or sys == 1

    if sys == 1:
        out = np.empty((n0, n0))
        for j in range(n0):
            for i in range(j + 1):
                out[i, j] = np.trace( mat[i*n1 : (i+1)*n1, j*n1 : (j+1)*n1] )
                out[j, i] = out[i, j]
    else:
        out = np.zeros((n1, n1))
        for i in range(n0):
            out += mat[i*n1 : (i+1)*n1, i*n1 : (i+1)*n1]

    return out

def i_kr(mat, sys, dim):
    (n0, n1) = dim
    assert sys == 0 or sys == 1

    if sys == 1:
        assert np.size(mat, 0) == n0
        out = np.kron(mat, np.eye(n1))
    else:
        assert np.size(mat, 0) == n1
        out = np.kron(np.eye(n0), mat)

    return out 

def inner(x, y):
    return (x * y).sum()