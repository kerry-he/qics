import numpy as np
import scipy as sp

def norm_inf(x):
    return max(x.max(initial=0.0), -x.min(initial=0.0))

def inp(x, y):
    # Standard inner product
    x_view = x.view(dtype=np.float64).reshape( 1, -1)
    y_view = y.view(dtype=np.float64).reshape(-1,  1).conj()
    return (x_view @ y_view)[0, 0]

def cho_fact(A):
    # Perform a Cholesky decomposition, while increment diagonals if failed     
    diag_incr = np.finfo(A.dtype).eps
    while True:
        try:
            fact = sp.linalg.cho_factor(A, check_finite=False)
            return fact
        except np.linalg.LinAlgError:
            A.flat[::A.shape[0]+1] += diag_incr
            diag_incr *= 1e1

def cho_solve(fact, x):
    # Factor solve for either Cholesky or LU factorization of A
    return sp.linalg.cho_solve(fact, x, check_finite=False)

def congr(out, A, X, work, B=None):
    if B is None:
        B = A

    # Performs congruence A X_i B' for i = i,...,n
    n, m = A.shape
    np.matmul(A, X, out=work)
    np.matmul(work.reshape((-1, m)), B.conj().T, out=out.reshape((-1, n)))
    return out