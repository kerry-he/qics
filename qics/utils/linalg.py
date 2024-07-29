import numpy as np
import scipy as sp

def norm_inf(x):
    """Computes the infinity norm of a vector, i.e., maximum absolute value.
    
    Parameters
    ----------
    x : ndarray
        Vector to compute infinity norm of.
        
    Returns
    -------
    float
        Infinity norm of x.
    """    
    return max(x.max(initial=0.0), -x.min(initial=0.0))

def inp(x, y):
    """Computes the standard inner product between two vectors.
    
    Parameters
    ----------
    x : ndarray
        First vector to compute inner product with.
    y : ndarray
        Second vector to compute inner product with. Should 
        be the same type and dimension as x.
        
    Returns
    -------
    float
        Standard inner product between x and y.
    """
    x_view = x.view(dtype=np.float64).reshape( 1, -1)
    y_view = y.view(dtype=np.float64).reshape(-1,  1).conj()
    return (x_view @ y_view)[0, 0]

def cho_fact(A):
    """Perform a Cholesky decomposition on a positive definite matrix. 
    Increment diagonals by a small amount if Cholesky decomposition fails
    until the factorization succeeds.
    
    Parameters
    ----------
    A : ndarray
        Symmetric matrix to compute Cholesky decomposition on. It should be
        assumed that A is positive definite (up to some numerical errors).
        
    Returns
    -------
    tuple[ndarray, bool]
        Cholesky decomposition of A from SciPy.
    """
    diag_incr = np.finfo(A.dtype).eps
    while True:
        try:
            fact = sp.linalg.cho_factor(A, check_finite=False)
            return fact
        except np.linalg.LinAlgError:
            A.flat[::A.shape[0]+1] += diag_incr
            diag_incr *= 1e1

def cho_solve(fact, b):
    """Factor solve for Cholesky factorization of A with vectors b
    
    Parameters
    ----------
    fact : tuple[ndarray, bool]
        Cholesky decomposition of a (n, n) matrix A returned from SciPy.
    b : ndarray
        An (n, 1) vector, or (n, k) list of vectors, to solve linear systems with.   
    
    Returns
    -------
    ndarray
        A (n, k) vector containing the solutions to the linear systems.
    """
    return sp.linalg.cho_solve(fact, b, check_finite=False)

def congr_multi(out, A, X, work, B=None):
    """Perform a congruence transform on a list of matrices.

        Xi --> A Xi A'    for    i = 1,...,k    if B = None
        Xi --> A Xi B'    for    i = 1,...,k    otherwise
    
    Parameters
    ----------
    out : ndarray
        Preallocated (k, m, m) array to store the output in.
    A : ndarray
        The (m, n) matrix to left-multiply with.
    X : ndarray
        The (k, n, n) list of matrices to perform a congruence transform on.
    work : ndarray
        Preallocated (k, m, n) array to do work with.
    B : ndarray, optional
        The (m, n) matrix to right-multiply with. Default is A.  
    """    
    if B is None:
        B = A

    # Performs congruence A X_i B' for i = i,...,n
    n, m = A.shape
    np.matmul(A, X, out=work)
    np.matmul(work.reshape((-1, m)), B.conj().T, out=out.reshape((-1, n)))
    return out