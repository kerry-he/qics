import numpy as np
import scipy as sp
import math

import cProfile

from cones import *
from utils import symmetric as sym
from solver import model, solver

def get_AdiagA(n, m, A_mat):
    AdiagA = np.zeros((n*n, m))
    k = -1
    for i in range(m):
        k += 1
    
        H = np.zeros((m, m))
        H[i, i] = 1
        
        AdiagA[:, k] = (A_mat @ H @ A_mat.T).ravel()
    return AdiagA

def get_eye(n, sn):
    eye = np.zeros((n*n, sn))
    k = -1
    for j in range(n):
        for i in range(j + 1):
            k += 1
        
            H = np.zeros((n, n))
            if i == j:
                H[i, j] = 1.
            else:
                H[i, j] = H[j, i] = math.sqrt(0.5)
            
            eye[:, k] = H.ravel()
    return eye

np.random.seed(1)
np.set_printoptions(threshold=np.inf)

# Define dimensions
n_mat = 40
m_mat = 80
A_mat = 1 / (n_mat**0.25) * np.random.randn(n_mat, m_mat)
k_mat = 2
eps   = 1e-6

vn = sym.vec_dim(n_mat)

eye = get_eye(n_mat, vn)
AdiagA = get_AdiagA(n_mat, m_mat, A_mat)

# Vars: (t \in R, z \in R^m, s \in S^n)

# Build problem model
c = np.vstack((np.ones((1, 1)), np.zeros((m_mat, 1)), np.log(eps) * sym.mat_to_vec(np.eye(n_mat))))

A = np.hstack((np.zeros((1, 1)), np.ones((1, m_mat)), np.zeros((1, vn))))
b = np.ones((1, 1)) * k_mat

G1 = np.hstack((np.zeros((1, 1)), np.zeros((1, m_mat)), sym.mat_to_vec(np.eye(n_mat)).T))
G2 = np.hstack((np.zeros((m_mat, 1)), -np.eye(m_mat), np.zeros((m_mat, vn))))
G3 = np.hstack((np.zeros((m_mat, 1)), np.eye(m_mat), np.zeros((m_mat, vn))))
G4 = np.hstack((np.zeros((n_mat*n_mat, 1)), np.zeros((n_mat*n_mat, m_mat)), eye))
G5 = np.hstack((-np.ones((1, 1)), np.zeros((1, m_mat)), np.zeros((1, vn))))
G6 = np.hstack((np.zeros((n_mat*n_mat, 1)), np.zeros((n_mat*n_mat, m_mat)), -eye))
G7 = np.hstack((np.zeros((n_mat*n_mat, 1)), -AdiagA, -eye * eps))
G = np.vstack((G1, G2, G3, G4, G5, G6, G7))

h = np.vstack((min(k_mat, n_mat) * np.ones((1, 1)), np.zeros((m_mat, 1)), np.ones((m_mat, 1)), np.eye(n_mat).reshape((-1, 1)), np.zeros((1, 1)), np.zeros((n_mat*n_mat, 1)), np.zeros((n_mat*n_mat, 1))))

# Input into model and solve
cones = [nonnegorthant.Cone(1 + m_mat + m_mat), possemidefinite.Cone(n_mat), oplogrelentr.Cone(n_mat)]
mdl = model.Model(c, A, b, G, h, cones=cones, offset=-n_mat*np.log(eps))
slv = solver.Solver(mdl)

profiler = cProfile.Profile()
profiler.enable()

slv.solve()

profiler.disable()
profiler.dump_stats("example.stats")