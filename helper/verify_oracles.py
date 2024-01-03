from cones import *
import numpy as np

n = 3
K = classentr.ClassEntropy(n)

while True:
    x0 = np.random.rand(K.dim, 1)
    K.set_point(x0)

    if K.get_feas():
        break

H = np.random.rand(K.dim, 1) - 0.5
eps = 1e-5

g0 = K.get_grad()
H0 = K.hess_prod(H)
T0 = K.third_dir_deriv(H)

x1 = x0 + eps * H
K.set_point(x1)
K.get_feas()
g1 = K.get_grad()
H1 = K.hess_prod(H)
T1 = K.third_dir_deriv(H)

print("Hessian test: ", np.linalg.norm(H - K.invhess_prod(H1)))
print("TOA test: ", np.linalg.norm(0.5 * (T0 + T1) - ((H1 - H0) / eps)))