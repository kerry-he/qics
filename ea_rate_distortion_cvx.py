import numpy as np, numba as nb
import scipy as sp
import math
import time

import cProfile

from cones import *
from utils import symmetric as sym, linear as lin
from solver import model, solver

from cvxopt import matrix, log, div, spdiag, solvers

def purify(eig):
    n = np.size(eig)

    vec = np.zeros((n*n, 1))
    for (i, ii) in enumerate(range(0, n*n, n + 1)):
        vec[ii] = math.sqrt(eig[i])

    return vec @ vec.T

def get_ikr_tr1(n, sN):
    tr2 = np.zeros((sN, sN))
    N = n * n
    k = -1
    for j in range(N):
        for i in range(j + 1):
            k += 1
        
            H = np.zeros((N, N))
            if i == j:
                H[i, j] = 1
            else:
                H[i, j] = H[j, i] = math.sqrt(0.5)
            
            I_H = sym.i_kr(sym.p_tr(H, 0, (n, n)), 0, (n, n))
            tr2[:, [k]] = sym.mat_to_vec(I_H)

    return tr2

def get_tr2(n, sn, sN):
    tr2 = np.zeros((sn, sN))
    k = -1
    for j in range(n):
        for i in range(j + 1):
            k += 1
        
            H = np.zeros((n, n))
            if i == j:
                H[i, j] = 1
            else:
                H[i, j] = H[j, i] = math.sqrt(0.5)
            
            I_H = sym.i_kr(H, 1, (n, n))
            tr2[[k], :] = sym.mat_to_vec(I_H).T

    return tr2

def get_eye(n, sn):
    eye = np.zeros((n*n, sn))
    k = -1
    for j in range(n):
        for i in range(j + 1):
            k += 1

            H = np.zeros((n, n))
            if i == j:
                H[i, j] = 1
            else:
                H[i, j] = H[j, i] = math.sqrt(0.5)

            eye[:, [k]] = H.reshape((n*n, 1))
    
    return eye


np.random.seed(1)
np.set_printoptions(threshold=np.inf)

# Define dimensions
n = 6
N = n * n
sn = sym.vec_dim(n)
sN = sym.vec_dim(N)

# Rate-distortion problem data
eig_A = np.random.rand(n)
eig_A /= np.sum(eig_A)
rho_A = np.diag(eig_A)
rho_AR = purify(eig_A)

Delta = sym.mat_to_vec(np.eye(N) - rho_AR)
D = 0.5

# Build problem model
tr2 = get_tr2(n, sn, sN)
ikr_tr1 = get_ikr_tr1(n, sN)
eye = get_eye(N, sN)

A1 = np.hstack((tr2, np.zeros((sn, sN))))
A2 = np.hstack((-ikr_tr1.T, np.eye(sN)))
A = np.vstack((A1, A2))

b = np.zeros((sn + sN, 1))
b[:sn] = sym.mat_to_vec(rho_A)

G1 = np.hstack((Delta.T, np.zeros((1, sN))))
G2 = np.hstack((-eye, np.zeros((N*N, sN))))
G3 = np.hstack((np.zeros((N*N, sN)), -eye))
G = np.vstack((G1, G2, G3))

h = np.zeros((1 + N*N + N*N, 1))
h[0] = D

rt2 = math.sqrt(2.0)
irt2 = math.sqrt(0.5)

@nb.njit
def D1_log(D, log_D):
    eps = np.finfo(np.float64).eps
    rteps = np.sqrt(eps)

    n = D.size
    D1 = np.empty((n, n))
    
    for j in range(n):
        for i in range(j):
            d_ij = D[i] - D[j]
            if abs(d_ij) < rteps:
                D1[i, j] = 2 / (D[i] + D[j])
            else:
                D1[i, j] = (log_D[i] - log_D[j]) / d_ij
            D1[j, i] = D1[i, j]

        D1[j, j] = np.reciprocal(D[j])

    return D1

@nb.njit
def D1_log(D, log_D):
    eps = np.finfo(np.float64).eps
    rteps = np.sqrt(eps)

    n = D.size
    D1 = np.empty((n, n))
    
    for j in range(n):
        for i in range(j):
            d_ij = D[i] - D[j]
            if abs(d_ij) < rteps:
                D1[i, j] = 2 / (D[i] + D[j])
            else:
                D1[i, j] = (log_D[i] - log_D[j]) / d_ij
            D1[j, i] = D1[i, j]

        D1[j, j] = np.reciprocal(D[j])

    return D1

@nb.njit
def D2_log(D, D1):
    eps = np.finfo(np.float64).eps
    rteps = np.sqrt(eps)

    n = D.size
    D2 = np.zeros((n, n, n))

    for k in range(n):
        for j in range(k + 1):
            for i in range(j + 1):
                d_jk = D[j] - D[k]
                if abs(d_jk) < rteps:
                    d_ij = D[i] - D[j]
                    if abs(d_ij) < rteps:
                        t = ((3 / (D[i] + D[j] + D[k]))**2) / -2
                    else:
                        t = (D1[i, j] - D1[j, k]) / d_ij
                else:
                    t = (D1[i, j] - D1[i, k]) / d_jk

                D2[i, j, k] = t
                D2[i, k, j] = t
                D2[j, i, k] = t
                D2[j, k, i] = t
                D2[k, i, j] = t
                D2[k, j, i] = t

    return D2

def scnd_frechet(D2, U, UHU, UXU):
    n = U.shape[0]

    D2_UXU = D2 * UXU
    out = D2_UXU @ UHU.reshape((n, n, 1))
    out = out.reshape((n, n))
    out = out + out.T
    out = U @ out @ U.T

    return out

def F(x = None, z = None):
    # Return (m, x0) where m is number of nonlinear constraints, x0 is point in the domain of f
    if x is None:  
        X = sym.mat_to_vec(np.eye(N)) / N
        x0 = np.vstack((X, X))
        return 0, matrix(x0)

    # Compute feasibility
    X = sym.vec_to_mat(np.array(x[:sN]))
    Y = sym.vec_to_mat(np.array(x[sN:]))

    Dx, Ux = np.linalg.eigh(X)
    Dy, Uy = np.linalg.eigh(Y)

    if any(Dx <= 0) or any(Dy <= 0):  return None

    # Compute function value and gradient
    log_Dx = np.log(Dx)
    log_Dy = np.log(Dy)

    log_X = (Ux * log_Dx) @ Ux.T
    log_Y = (Uy * log_Dy) @ Uy.T
    log_XY = log_X - log_Y
    val = lin.inp(X, log_XY)

    D1y_log = D1_log(Dy, log_Dy)
    UyXUy = Uy.T @ X @ Uy
    DPhiX = log_XY + np.eye(N)
    DPhiY = -Uy @ (D1y_log * UyXUy) @ Uy.T
    Df = np.vstack((sym.mat_to_vec(DPhiX), sym.mat_to_vec(DPhiY)))

    if z is None:  return val, matrix(Df.T)

    # Compute Hessian
    D1x_log = D1_log(Dx, log_Dx)
    D2y_log = D2_log(Dy, D1y_log)

    # Hessians of quantum relative entropy
    D2PhiXX = np.empty((sN, sN))
    D2PhiXY = np.empty((sN, sN))
    D2PhiYY = np.empty((sN, sN))

    k = 0
    for j in range(N):
        for i in range(j + 1):
            # D2PhiXX
            UxHUx = np.outer(Ux[i, :], Ux[j, :])
            if i != j:
                UxHUx = UxHUx + UxHUx.T
                UxHUx *= irt2
            temp = Ux @ (D1x_log * UxHUx) @ Ux.T
            D2PhiXX[:, [k]] = sym.mat_to_vec(temp)

            # D2PhiXY
            UyHUy = np.outer(Uy[i, :], Uy[j, :])
            if i != j:
                UyHUy = UyHUy + UyHUy.T
                UyHUy *= irt2
            temp = -Uy @ (D1y_log * UyHUy) @ Uy.T
            D2PhiXY[:, [k]] = sym.mat_to_vec(temp)

            # D2PhiYY
            temp = -scnd_frechet(D2y_log, Uy, UyHUy, UyXUy)
            D2PhiYY[:, [k]] = sym.mat_to_vec(temp)
            k += 1

    H = np.empty((2*sN, 2*sN))
    H[:sN, sN:] = D2PhiXY
    H[sN:, :sN] = D2PhiXY.T
    H[:sN, :sN] = D2PhiXX
    H[sN:, sN:] = D2PhiYY

    return val, matrix(Df.T), z[0] * matrix(H)

G = matrix(G)
h = matrix(h)
A = matrix(A)
b = matrix(b)

dims = {'l': 1, 'q': [], 's':  [N, N]}
tic = time.time()
sol = solvers.cp(F, G, h, dims, A, b)
print("Time elapsed:", time.time() - tic)
print(sol)