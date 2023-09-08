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
    (n, m) = dim
    assert n * m == np.size(mat, 0)
    assert sys == 0 or sys == 1

    if sys == 1:
        out = np.empty((n, n))
        for j in range(n):
            for i in range(j + 1):
                out[i, j] = np.trace( mat[i*m : (i+1)*m, j*m : (j+1)*m] )
                out[j, i] = out[i, j]
    else:
        out = np.zeros((m, m))
        for i in range(n):
            out += mat[i*m : (i+1)*m, i*m : (i+1)*m]

    return out

def inner(x, y):
    return (x * y).sum()