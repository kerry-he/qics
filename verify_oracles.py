from cones import *
import numpy as np
from utils import quantum as quant
import time

np.random.seed(1)

def inp(x, y):
    return sum([np.sum(xi * yi.conj()) for (xi, yi) in zip(x, y)]).real

def vec(x):
    return np.hstack([xk.ravel() for xk in x])

hermitian = False
dtype = np.float64 if (not hermitian) else np.complex128
n = 3
K = oplogrelentr.Cone(n, hermitian=hermitian)

while True:
    x0 = [
        quant.randDensityMatrix(n, hermitian=hermitian),
        quant.randDensityMatrix(n, hermitian=hermitian),
        quant.randDensityMatrix(n, hermitian=hermitian)
    ]
    K.set_point(x0, x0)

    if K.get_feas():
        break

H = [
    np.random.randn(n, n),
    np.random.randn(n, n),
    np.random.randn(n, n)
]
H[0] += H[0].conj().T
H[1] += H[1].conj().T
H[2] += H[2].conj().T

eps = 1e-8
f0 = K.get_val()
g0 = [np.zeros((n, n), dtype=dtype), np.zeros((n, n), dtype=dtype), np.zeros((n, n), dtype=dtype)]
K.get_grad(g0)
H0 = [np.zeros((n, n), dtype=dtype), np.zeros((n, n), dtype=dtype), np.zeros((n, n), dtype=dtype)]
K.hess_prod_ip(H0, H)
# T0 = [np.zeros((n, n), dtype=dtype), np.zeros((n, n), dtype=dtype), np.zeros((n, n), dtype=dtype)]
# K.third_dir_deriv_axpy(T0, H)

f1 = [np.zeros((n, n), dtype=dtype), np.zeros((n, n), dtype=dtype), np.zeros((n, n), dtype=dtype)]
for j in range(n):
    for i in range(j + 1):
        x1 = [xk.copy() for xk in x0]
        x1[0][i, j] += eps * 0.5
        x1[0][j, i] += eps * 0.5
        K.set_point(x1, x1)
        K.get_feas()
        f1[0][i, j] = K.get_val()
        f1[0][j, i] = K.get_val()

for j in range(n):
    for i in range(j + 1):
        x1 = [xk.copy() for xk in x0]
        x1[1][i, j] += eps * 0.5
        x1[1][j, i] += eps * 0.5
        K.set_point(x1, x1)
        K.get_feas()
        f1[1][i, j] = K.get_val()
        f1[1][j, i] = K.get_val()

for j in range(n):
    for i in range(j + 1):
        x1 = [xk.copy() for xk in x0]
        x1[2][i, j] += eps * 0.5
        x1[2][j, i] += eps * 0.5
        K.set_point(x1, x1)
        K.get_feas()
        f1[2][i, j] = K.get_val()
        f1[2][j, i] = K.get_val()

x1 = [xk + eps*Hk for (xk, Hk) in zip(x0, H)]
K.set_point(x1, x1)
K.get_feas()
g1 = [np.zeros((n, n), dtype=dtype), np.zeros((n, n), dtype=dtype), np.zeros((n, n), dtype=dtype)]
K.get_grad(g1)
H1 = [np.zeros((n, n), dtype=dtype), np.zeros((n, n), dtype=dtype), np.zeros((n, n), dtype=dtype)]
K.hess_prod_ip(H1, H)
# T1 = [np.zeros((n, n), dtype=dtype), np.zeros((n, n), dtype=dtype), np.zeros((n, n), dtype=dtype)]
# K.third_dir_deriv_axpy(T1, H)

print("Gradient test (FDD=0): ", np.linalg.norm(0.5 * (vec(g0) + vec(g1)) - ((vec(f1) - f0) / eps)))
print("Gradient test (ID=-nu): ", inp(g0, x0))

# work = [np.zeros((n, n), dtype=dtype), np.zeros((n, n), dtype=dtype), np.zeros((n, n), dtype=dtype)]
# K.hess_prod_ip(work, x0)
# work2 = [xk + yk for (xk, yk) in zip(work, g0)]
# print("Hessian test (ID=0): ",  inp(work2, work2))
# print("Hessian test (ID=nu): ",  inp(K.hess_prod_ip(work2, x0), x0))
print("Hessian test (FDD=0): ", np.linalg.norm(0.5 * (vec(H0) + vec(H1)) - ((vec(g1) - vec(g0)) / eps)))

# K.invhess_prod_ip(work, H1)
# print("Inv Hessian test (ID=0): ", np.linalg.norm(vec(H) - vec(work)))
# print("Inv Hessian test (ID=nu): ", inp(K.invhess_prod_ip(work, g0), g0))

# print("TOA test: ", np.linalg.norm(0.5 * (vec(T0) + vec(T1)) - ((vec(H1) - vec(H0)) / eps)))

