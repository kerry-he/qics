import math
import numpy as np
import numba as nb
import time

from utils import symmetric as sym, mtxgrad as mgrad
from utils import quantum as quant

np.set_printoptions(suppress=True, edgeitems=30, linewidth=100000, precision=2)

n = 7


X = quant.randDensityMatrix(n, hermitian=True)
Y = quant.randDensityMatrix(n, hermitian=True)

Dx, Ux = np.linalg.eigh(X)
Dy, Uy = np.linalg.eigh(Y)

log_Dx = np.log(Dx)
log_Dy = np.log(Dy)

D1x_log = mgrad.D1_log(Dx, log_Dx)
D1y_log = mgrad.D1_log(Dy, log_Dy)
D2y_log = mgrad.D2_log(Dy, D1y_log)

UyXUy = Uy.conj().T @ X @ Uy
D2y_log_UXU = D2y_log * UyXUy

S0 = sym.lin_to_mat(lambda x : mgrad.scnd_frechet(D2y_log_UXU, x), n, n, hermitian=True)
S1 = mgrad.get_S_matrix_complex(D2y_log_UXU, np.sqrt(2.0))

print(S0)
print(S1)
print(S0 - S1)