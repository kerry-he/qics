import numpy as np
import scipy as sp
import math
import time

from cones import *
from utils import symmetric as sym, linear as lin
from utils import quantum as quant, mtxgrad as mgrad
from solver import model, solver

np.random.seed(4)
n = 300
vn = sym.vec_dim(n)

X = quant.randDensityMatrix(n)
Y = quant.randDensityMatrix(n)

H = np.random.rand(n, n) - 0.5
H = H + H.T
H_vec = sym.mat_to_vec(H)

Dx, Ux = np.linalg.eigh(X)
Dy, Uy = np.linalg.eigh(Y)

log_Dx = np.log(Dx)
log_Dy = np.log(Dy)

D1y_log = mgrad.D1_log(Dy, log_Dy)
D2y_log = mgrad.D2_log(Dy, D1y_log)

D2_UXU = D2y_log * (Uy.T @ X @ Uy)

# S = mgrad.get_S_matrix(D2_UXU, np.sqrt(2.0))
# tic = time.time()
# S = mgrad.get_S_matrix(D2_UXU, np.sqrt(2.0))
# print("Build S matrix: ", time.time() - tic)

# tic = time.time()
# fact = sp.linalg.cho_factor(-S)
# print("Cho factor: ", time.time() - tic)
# tic = time.time()
# xS = sp.linalg.cho_solve(fact, H_vec)
# print("Cho solve: ", time.time() - tic)


def scnd_frechet_premult_single(x):
    n = int(math.sqrt(x.size))
    X = x.reshape((n, n))

    out = D2_UXU @ X.reshape((n, n, 1))
    out = out.reshape((n, n))

    # out = np.empty((n, n))
    # for k in range(n):
    #     out[k, :] = D2_UXU[k] @ X[k]

    out = out + out.T

    return out.flatten()


def A_func(x):
    n = int(math.sqrt(x.size))

    out = D2_UXU @ x.reshape((n, n, 1))
    out = out.reshape((n, n))
    out = out + out.T

    return out
    

A = sp.sparse.linalg.LinearOperator((n**2, n**2), matvec=scnd_frechet_premult_single)

M_diag = np.zeros(n*n)
for i in range(n):
    for j in range(n):
        M_diag[i*n + j] += D2_UXU[i, j, j]
        M_diag[j*n + i] += D2_UXU[i, j, j]
M = sp.sparse.diags( 1 / M_diag )
M_mat = 1 / M_diag.reshape((n, n))

def M_func(x):
    return x * M_mat

nstep = 0

def callback(xk):
    global nstep
    nstep += 1

nstep = 0
tic = time.time()
x, info = sp.sparse.linalg.cgs(A, H.flatten(), x0=M @ H.flatten(), tol=1e-9, M=M, callback=callback)
print("Solve system CG: ", time.time() - tic)
print("tr steps taken: ", nstep, ";   res: ", np.linalg.norm(A @ x - H.flatten()))

nstep = 0
tic = time.time()
x, info = sp.sparse.linalg.cg(A, H.flatten(), x0=M @ H.flatten(), tol=1e-9, M=M, callback=callback)
print("Solve system CG: ", time.time() - tic)
print("tr steps taken: ", nstep, ";   res: ", np.linalg.norm(A @ x - H.flatten()))


tic = time.time()
x, nstep, res = lin.pcg(A_func, H, M_func, tol=1e-8, max_iter=1000)
print("Solve system CG: ", time.time() - tic)
print("tr steps taken: ", nstep, ";   res: ", res)
print(";check res: ", np.linalg.norm(A @ x.flatten() - H.flatten()))

# print(np.linalg.norm(sym.vec_to_mat(xS) - x.reshape((n, n))))

# np.set_printoptions(threshold=np.inf, linewidth=5000)
# print(A @ np.eye(n**2))
# print(M_diag)

# S_sparse = sp.sparse.csc_matrix(S)

# def callback2(xk):
#     print("res: ", np.linalg.norm(S_sparse @ xk - H_vec))

# tic = time.time()
# x, info = sp.sparse.linalg.cgs(S_sparse, H_vec, M=sp.sparse.diags(1 / np.diag(S)))
# print("Solve system CG: ", time.time() - tic)