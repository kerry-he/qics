from cones import *
import numpy as np
from utils import quantum as quant
import time

n = 4
X = quant.randDensityMatrix(n)*np.random.rand()
K = quantrelentr_Y.QuantRelEntropyY(n, X)
# K = quantrelentr.QuantRelEntropy(n)

while True:
    x0 = np.random.rand(K.dim, 1)
    K.set_point(x0)

    if K.get_feas():
        break

H = np.random.rand(K.dim, 1) - 0.5
eps = 1e-8
f0 = K.get_val()
g0 = K.grad_ip()
H0 = K.hess_prod(H)
T0 = K.third_dir_deriv(H)



f1 = np.zeros((K.dim, 1))
for i in range(K.dim):
    x1 = x0.copy()
    x1[i] += eps
    K.set_point(x1)
    K.get_feas()
    f1[i] = K.get_val()

x1 = x0 + eps * H
K.set_point(x1)
K.get_feas()
g1 = K.grad_ip()
H1 = K.hess_prod(H)
T1 = K.third_dir_deriv(H)

print("Gradient test (FDD=0): ", np.linalg.norm(0.5 * (g0 + g1) - ((f1 - f0) / eps)))
print("Gradient test (ID=nu): ", (-g0.T @ x0)[0, 0])

print("Hessian test (ID=0): ",  np.linalg.norm(K.hess_prod(x0) + g0))
print("Hessian test (ID=nu): ",  (K.hess_prod(x0).T @ x0)[0, 0])
print("Hessian test (FDD=0): ", np.linalg.norm(0.5 * (H0 + H1) - ((g1 - g0) / eps)))




print("Inv Hessian test (ID=0): ", np.linalg.norm(H - K.invhess_prod(H1)))
print("Inv Hessian test (ID=nu): ", (K.invhess_prod(g0).T @ g0)[0, 0])

print("TOA test: ", np.linalg.norm(0.5 * (T0 + T1) - ((H1 - H0) / eps)))