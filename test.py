import numpy as np
import scipy as sp

from cones import quantcondentr as qce
from utils import symmetric as sym
from solver import model

np.random.seed(1)

n = 2
N = n**2
sN = sym.vec_dim(N)
dim = sN + 1

tr_A = sym.mat_to_vec(np.eye(N, N))
A = np.hstack((np.array([[0.]]), tr_A.T))
b = np.array([[2.]])
c = np.vstack((np.array([[1.]]), np.zeros((dim - 1, 1))))

cones = [qce.QuantCondEntropy(n, n)]
model = model.Model(c, A, b, cones)