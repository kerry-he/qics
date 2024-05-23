import numpy as np
import scipy as sp
import sksparse.cholmod as cholmod
import numba as nb

def norm_inf(x):
    return max(x.max(initial=0.0), -x.min(initial=0.0))

def inp(x, y):
    # Standard inner product
    if isinstance(x, list) and isinstance(y, list):
        return sum([np.sum(xi * yi.conj()).real for (xi, yi) in zip(x, y)])
    else:
        return np.sum(x * y.conj()).real

def fact(A, fact=None):
    # Perform a Cholesky decomposition, or an LU factorization if Cholesky fails
    if sp.sparse.issparse(A):
        while True:
            try:
                if fact is None:
                    fact = cholmod.cholesky(A)
                    return (fact, "spcho")
                else:
                    fact[0].cholesky_inplace(A)
                    return (fact[0], "spcho")
            except cholmod.CholmodNotPositiveDefiniteError:
                A_diag = A.diagonal()
                A.setdiag(np.max([A_diag, np.ones_like(A_diag) * 1e-12], axis=0) * (1 + 1e-8))
        
    
    while True:
        try:
            fact = sp.linalg.cho_factor(A, check_finite=False)
            return (fact, "cho")
        except np.linalg.LinAlgError:
            diag_idx = np.diag_indices_from(A)
            A[diag_idx] = np.max([A[diag_idx], np.ones_like(A[diag_idx]) * 1e-12], axis=0) * (1 + 1e-8)

    # try:
    #     fact = sp.linalg.cho_factor(A)
    #     return (fact, "cho")
    # except np.linalg.LinAlgError:
    #     fact = sp.linalg.lu_factor(A)
    #     return (fact, "lu")

def fact_solve(A, x):
    # Factor solve for either Cholesky or LU factorization of A
    (fact, type) = A

    if type == "cho":
        return sp.linalg.cho_solve(fact, x, check_finite=False)
    elif type == "lu":
        return sp.linalg.lu_solve(fact, x)
    elif type == "spcho":
        return fact.solve_A(x)

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