import time
import cProfile
import numpy as np
import numba as nb
import math
import timeit
from utils import symmetric as sym

N = 1600

sN = sym.vec_dim(N)

mat = np.random.rand(N, N)
vec = np.random.rand(sN, 1)

L = 10
rt2 = math.sqrt(2)
irt2 = math.sqrt(0.5)

i_tril = np.tril_indices(N)

rt2 = np.ones((N, N)) * np.sqrt(2.0)
np.fill_diagonal(rt2, 1.)

irt2 = np.ones((N, N)) * np.sqrt(0.5)
np.fill_diagonal(irt2, 1.)

def jit(mat, rt2=None):
    rt2 = np.sqrt(2) if (rt2 is None) else np.sqrt(2.)
    n = mat.shape[0]
    return single(mat, rt2) if (n <= 400) else parallel(mat, rt2)

@nb.njit
def single(vec):
    irt2 = math.sqrt(0.5)
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
def parallel(vec):
    irt2 = math.sqrt(0.5)
    vn = vec.size
    n = int(math.sqrt(1 + 8 * vn) // 2)
    mat = np.empty((n, n))

    k = 0
    for j in nb.prange(n):
        for i in nb.prange(j + 1):
            if i == j:
                mat[i, j] = vec[k, 0]
            else:
                mat[i, j] = vec[k, 0] * irt2
                mat[j, i] = vec[k, 0] * irt2
            k += 1

    return mat


def no_jit(vec, i_tril=None):
    irt2 = math.sqrt(0.5)
    vn = np.size(vec)

    n = sym.mat_dim(vn)

    # Fill in parameters if values are not preallocated
    if irt2 is None:
        irt2 = np.ones((n, n)) * np.sqrt(0.5)
        np.fill_diagonal(irt2, 1.)

    if i_tril is None:
        i_tril = np.tril_indices(n)

    mat = np.zeros((n, n))
    mat[i_tril] = vec[:, 0]                     # Fill out lower triangle
    mat = mat + mat.T - np.diag(np.diag(mat))   # Symmeterize matrix
    mat *= irt2

# def no_jit(mat, rt2, i_tril):
#     n = mat.shape[0]
#     rt2 = np.sqrt(2)
#     # temp = [((mat[i, j] if (i == j) else mat[i, j] * rt2) for i in range(j + 1)) for j in range(n)]
#     temp = [([mat[i, j]] if (i == j) else ([mat[i, j] * rt2])) for j in range(n) for i in range(j + 1)]
#     return np.array(temp)

tic = time.time()
for _ in range(L):
    no_jit(vec, i_tril)
print("no_jit: ", time.time() - tic)

single(vec)
tic = time.time()
for _ in range(L):
    single(vec)
print("single: ", time.time() - tic)

parallel(vec)
tic = time.time()
for _ in range(L):
    parallel(vec)
print("parallel: ", time.time() - tic)