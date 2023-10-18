import numpy as np
import scipy as sp

def inp(x, y):
    # Standard inner product
    return np.sum(x * y)

def fact(A):
    # Perform a Cholesky decomposition, or an LU factorization if Cholesky fails
    try:
        fact = sp.linalg.cho_factor(A)
        return (fact, "cho")
    except np.linalg.LinAlgError:
        fact = sp.linalg.lu_factor(A)
        return (fact, "lu")

def fact_solve(A, x):
    # Factor solve for either Cholesky or LU factorization of A
    (fact, type) = A

    if type == "cho":
        return sp.linalg.cho_solve(fact, x)
    elif type == "lu":
        return sp.linalg.lu_solve(fact, x)