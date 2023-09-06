import numpy as np
import scipy as sp

from cones import quantcondentr
from utils import symmetric as sym

np.random.seed(1)

n = 2
m = 2
nm = n*m

K = quantcondentr.QuantCondEntropy(n, m)

A = np.random.rand(nm, nm)
X = A * A.T + np.eye(nm)

point = np.zeros((K.dim, 1))
point[0] = 1
point[1:] = sym.mat_to_vec(X)

K.get_point(point)

print(K.t)
print(K.X)
print(K.Y)

K.get_feas()
K.get_grad()


HX = np.random.rand(nm, nm) - 0.5
HX = HX + HX.T
H = np.zeros((K.dim, 1))
H[0] = np.random.rand() - 0.5
H[1:] = sym.mat_to_vec(HX)

print("Ht: ", H[0])
print(HX)

print(K.hess_prod(H))