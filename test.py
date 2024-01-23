import numpy as np
import scipy as sp
import math
import time

from cones import *
from utils import symmetric as sym, linear as lin
from utils import quantum as quant, mtxgrad as mgrad
from solver import model, solver

np.random.seed(1)
n = 350
vn = sym.vec_dim(n)

zi = 1e2
X  = quant.randDensityMatrix(n)
Y  = quant.randDensityMatrix(n)

H = np.random.rand(n, n) - 0.5
H = H + H.T
H_vec = sym.mat_to_vec(H)

Dx, Ux = np.linalg.eigh(X)
Dy, Uy = np.linalg.eigh(Y)

log_Dx = np.log(Dx)
log_Dy = np.log(Dy)

D1y_log = mgrad.D1_log(Dy, log_Dy)
D2y_log = mgrad.D2_log(Dy, D1y_log)

UXU    =  Uy.T @ X @ Uy
D2_UXU = -(UXU + np.eye(n)/zi) * D2y_log

# Conjugate gradient
M = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        M[i, j] += D2_UXU[i, j, j]
        M[j, i] += D2_UXU[i, j, j]
M = np.reciprocal(M)

tic = time.time()
sol, nstep, res = lin.pcg(lambda x:  mgrad.scnd_frechet(D2_UXU, x), H, 
                          lambda x: x * M, 
                          tol=1e-9, max_iter=2*n)
print("cg steps taken: ", nstep, ";   time elapsed: ", time.time() - tic, ";   res: ", res)

# Douglas Rachford A
gamma = 0.2
alpha = 1.0

M = np.sqrt(M)

MSM = np.array([M[[k]] * D2_UXU[k] * M[[k]].T for k in range(n)])
MSM_I = MSM + gamma * np.eye(n)
MSM_I_fact = [lin.fact(MSM_I_k) for MSM_I_k in MSM_I]

def dr(C, tol, max_iter):
    MC = M * C

    z = M * MC
    x = np.zeros_like(z)
    y = np.zeros_like(z)

    for i in range(max_iter):

        temp = gamma * z + 0.5 * MC
        for k in range(n):
            x[k] = lin.fact_solve(MSM_I_fact[k], temp[k])
        
        temp = gamma * (2*x - z) + 0.5 * MC
        for k in range(n):
            y.T[k] = lin.fact_solve(MSM_I_fact[k], temp.T[k])

        z = z + 2*alpha*(y - x)

        if (i % 5) == 0: 
            Mx = M * x
            res_vec = mgrad.scnd_frechet(D2_UXU, Mx) - C
            res = np.linalg.norm(res_vec)

            if res < tol:
                break

    Mx = (Mx + Mx.T) * 0.5

    return Mx, i, res


tic = time.time()
sol, nstep, res = dr(H, tol=1e-9, max_iter=2*n)
print("dr steps taken: ", nstep, ";   time elapsed: ", time.time() - tic, ";   res: ", res)


# Douglas Rachford B
# Preconditioner
M = np.array( [ np.diag(D2_UXU_k) for D2_UXU_k in D2_UXU ] )
M = np.reciprocal(np.sqrt(M))

# Factor each block 
MSM = np.array([M[[k]] * D2_UXU[k] * M[[k]].T for k in range(n)])

D2_UXU_eigs = np.array( [ np.linalg.eigvalsh(MSM_k) for MSM_k in MSM ] )
gamma = np.sqrt(np.max(D2_UXU_eigs) * np.min(D2_UXU_eigs)) * 2.0
print(gamma)

# gamma = 0.75
alpha = 1.0

MSM_I = 2*MSM + gamma*np.eye(n)
tic = time.time()
MSM_I_fact = [lin.fact(MSM_I_k) for MSM_I_k in MSM_I]
print("fact time: ", time.time() - tic)

temp  = M.T / ((M * M) + (M.T * M.T))
M_P   = M.T * temp
M_P_T = M   * temp

def multi_fact_solve(temp):
    x = np.zeros_like(temp)
    for k in range(n):
        x[k] = lin.fact_solve(MSM_I_fact[k], temp[k])

def dr(C, tol, max_iter):
    MC = M * C

    z = M * MC
    x = np.zeros_like(z)
    y = np.zeros_like(z)

    for i in range(max_iter):

        temp = MC + gamma*z
        # for k in range(n):
        #     x[k] = lin.fact_solve(MSM_I_fact[k], temp[k])
        
        x = np.array( [lin.fact_solve(MSM_I_fact[k], temp[k]) for k in range(n)] )
        
        temp = 2*x - z
        y = M_P * temp + M_P_T * temp.T

        z = z + 2*alpha*(y - x)

        if (i % 5) == 0: 
            Mx = M * x
            res_vec = mgrad.scnd_frechet(D2_UXU, Mx) - C
            res = np.linalg.norm(res_vec)

            if res < tol:
                break

    Mx = (Mx + Mx.T) * 0.5

    return Mx, i, res

tic = time.time()
sol, nstep, res = dr(H, tol=1e-9, max_iter=2*n)
print("dr steps taken: ", nstep, ";   time elapsed: ", time.time() - tic, ";   res: ", res)