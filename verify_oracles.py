from cones import *
import numpy as np
import scipy as sp
from utils import quantum as quant, symmetric as sym
import time


# np.random.seed(1)
data = sp.io.loadmat('examples/pmBB84.mat')
gamma = data['gamma']
Gamma = data['Gamma'][:, 0]
Klist = data['Klist'][0, :]
Zlist = data['Zlist'][0, :]

no, ni = np.shape(Klist[0])
nc = np.size(gamma)


n = 3
# K = quantrelentr.QuantRelEntropy(n, hermitian=True)
# K = possemidefinite.PosSemiDefinite(n, hermitian=True)
K = quantkeydist.QuantKeyDist(Klist, Zlist, hermitian=True)

while True:
    t = np.random.rand()
    X0 = quant.randDensityMatrix(ni)

    x0 = np.random.rand(K.dim, 1)

    x0[1:] = sym.mat_to_vec(X0, hermitian=True)

    K.set_point(x0)

    if K.get_feas():
        break

H = np.random.rand(K.dim, 1) - 0.5
eps = 1e-8
f0 = K.get_val()
g0 = K.get_grad()
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
g1 = K.get_grad()
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

print(g0)