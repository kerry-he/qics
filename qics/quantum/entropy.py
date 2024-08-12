import numpy as np

def entropy(x):
    """Computes classical (Shannon) entropy 
    
        H(x) = -Σ_i xi log(xi),
        
    for nonnegative vector x.
    
    Parameters
    ----------
    x : ndarray
        Nonnegative (n, 1) vector to compute classical entropy of.
        
    Returns
    -------
    float
        Classical entropy of x.
    """    
    x = x[x > 0]
    return -sum(x * np.log(x))

def quant_entropy(X):
    """Computes quantum (von Neumann) entropy 
    
        S(X) = -tr[X log(X)],
        
    for positive semidefinite matrix X.
    
    Parameters
    ----------
    X : ndarray
        Positive semidefinite (n, n) matrix to compute quantum entropy of.
        
    Returns
    -------
    float
        Quantum entropy of X.
    """
    eig = np.linalg.eigvalsh(X)
    return entropy(eig)

def purify(X):
    """Returns a purification of a quantum state X. If X has spectral decomposition
    
        X = Σ_i xi (vi vi'),
        
    then the purification is pp' where
    
        p = Σ_i sqrt(xi) (vi ⊗ vi).
    
    Parameters
    ----------
    X : ndarray
        Density matrix of size (n, n).
        
    Returns
    -------
    ndarray
        Purification matrix of X of size (n^2, n^2).
    """    
    n = X.shape[0]
    D, U = np.linalg.eigh(X)

    vec = np.zeros((n*n, 1), dtype=X.dtype)
    for i in range(n):
        vec += np.sqrt(D[i]) * np.kron(U[:, [i]], U[:, [i]])

    return vec @ vec.conj().T