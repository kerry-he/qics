import math
import numpy as np

def vec_dim(side):
    return side * (side + 1) // 2

def mat_dim(len):
    side = math.isqrt(1 + 8 * len) // 2
    assert side * (side + 1) == 2 * len
    return side

def mat_to_vec(mat, rt2=None):
    rt2 = math.sqrt(2) if (rt2 is None) else rt2
    n = np.size(mat, 0)
    assert n == np.size(mat, 1)

    vn = vec_dim(n)
    vec = np.empty((vn, 1))

    k = 0
    for j in range(n):
        for i in range(j + 1):
            if i == j:
                vec[k] = mat[i, j]
            else:
                vec[k] = mat[i, j] * rt2
            k += 1
    
    assert k == vn
    return vec

def vec_to_mat(vec):
    irt2 = math.sqrt(0.5)
    vn = np.size(vec)

    n = mat_dim(vn)
    mat = np.empty((n, n))

    k = 0
    for j in range(n):
        for i in range(j + 1):
            if i == j:
                mat[i, j] = vec[k]
            else:
                mat[i, j] = vec[k] * irt2
                mat[j, i] = vec[k] * irt2
            k += 1

    assert k == vn
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