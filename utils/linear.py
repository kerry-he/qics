import numpy as np
import scipy as sp
import numba as nb

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

def pcg(A, b, M, tol=1e-8, max_iter=20):

    # Use to compute relative tolerance
    b_max = max(np.max(np.abs(b)), 1.0)

    x_k = M(b)
    r_k = b - A(x_k)

    abs_res = np.linalg.norm(r_k)
    rel_res = abs_res / b_max    
    if abs_res < tol or rel_res < tol:
        return x_k, 0, abs_res

    z_k = M(r_k)
    p_k = z_k
    r_z_k = inp(r_k, z_k)

    for k in range(max_iter):
        A_p_k = A(p_k)
        alpha_k = r_z_k / inp(p_k, A_p_k)
        x_k1 = x_k + alpha_k * p_k
        r_k1 = r_k - alpha_k * A_p_k

        # Check if solved to desired tolerance
        abs_res = np.linalg.norm(r_k1)
        rel_res = abs_res / b_max

        if abs_res < tol or rel_res < tol:
            break

        z_k1 = M(r_k1)
        r_z_k1 = inp(r_k1, z_k1)
        beta_k = r_z_k1 / r_z_k
        p_k1 = z_k1 + beta_k * p_k

        # Step forwards
        x_k = x_k1
        r_k = r_k1
        z_k = z_k1
        p_k = p_k1
        r_z_k = r_z_k1

    return x_k1, (k + 1), abs_res

def kron(a, b):
    # Kroneker product between two (n x n) matrices
    n  = a.shape[0]
    n2 = n*n
    return (a[:, None, :, None] * b[None, :, None, :]).reshape(n2, n2)

def congr(out, A, X, work, B=None):
    if B is None:
        B = A

    # Performs congruence A X_i B' for i = i,...,n
    n, m = A.shape
    np.matmul(A, X, out=work)
    np.matmul(work.reshape((-1, m)), B.conj().T, out=out.reshape((-1, n)))
    return out

if __name__ == "__main__":
    import time
    
    n = 2
    X = np.random.rand(n, n)
    Y = np.random.rand(n, n)
    
    tic = time.time()
    for i in range(10000):
        out = kron(X, Y)
    print("Time elapsed: ", time.time() - tic)

    tic = time.time()
    for i in range(10000):
        out = np.kron(X, Y)
    print("Time elapsed: ", time.time() - tic)