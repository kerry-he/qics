import numpy as np

def rand_density_matrix(n, iscomplex=False):
    """Generate random density matrix on Haar measure,
    i.e., positive semifedinite matrix X satisfying tr[X] = 1.

    Parameters
    ----------
    n : int
        Dimension of random (n, n) matrix
    iscomplex : bool, optional
        Whether the matrix is symmetric (False) or Hermitian (True). Default is False.
        
    Returns
    -------
    ndarray
        Random density matrix of dimension (n, n)        
    """
    if iscomplex:
        X = np.random.normal(size=(n, n)) + np.random.normal(size=(n, n)) * 1j
    else:
        X = np.random.normal(size=(n, n))
    rho = X @ X.conj().T
    return rho / np.trace(rho)

def rand_pure_density_matrix(n, iscomplex=False):
    """Generate random pure density matrix
    i.e., rank 1 positive semifedinite matrix X satisfying tr[X] = 1.
    See: https://sumeetkhatri.com/wp-content/uploads/2020/05/random_pure_states.pdf

    Parameters
    ----------
    n : int
        Dimension of random (n, n) matrix
    iscomplex : bool, optional
        Whether the matrix is symmetric (False) or Hermitian (True). Default is False.
        
    Returns
    -------
    ndarray
        Random density matrix of dimension (n, n)        
    """    
    if iscomplex:
        psi = np.random.normal(size=(n)) + np.random.normal(size=(n)) * 1j
    else:
        psi = np.random.normal(size=(n))
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    rho = (rho + rho.conj().T) * 0.5
    return rho

def rand_unitary(n, iscomplex=False):
    """Generate random unitary uniformly distributed on Haar measure
    i.e., matrix U satisfying U'U = UU' = I.
    See: https://case.edu/artsci/math/esmeckes/Meckes_SAMSI_Lecture2.pdf

    Parameters
    ----------
    n : int
        Dimension of random (n, n) unitary
    iscomplex : bool, optional
        Whether the unitary is real (False) or complex (True). Default is False.
        
    Returns
    -------
    ndarray
        Random unitary of dimension (n, n)        
    """       
    if iscomplex:
        X = np.random.normal(size=(n, n)) + np.random.normal(size=(n, n)) * 1j
    else:
        X = np.random.normal(size=(n, n))
    U, _ = np.linalg.qr(X)
    return U

def rand_stinespring_operator(nin, nout=None, nenv=None, iscomplex=False):
    """Generate random Stinespring operator uniformly distributed on Hilbert-Schmidt measure
    i.e., isometry V corresponding to quantum channel N(X) = tr_E[V X V'].
    See: https://arxiv.org/abs/2011.02994

    Parameters
    ----------
    nin : int
        Dimension of input system.
    nout : int, optional
        Dimension of output system. Default is nin.
    nenv : int, optional
        Dimension of environment system. Default is nout.         
    iscomplex : bool, optional
        Whether the Stinespring is real (False) or complex (True). Default is False.
        
    Returns
    -------
    ndarray
        Random Stinespring operator of dimension (nout*nenv, nin)
    """
    nout = nout if (nout is not None) else nin
    nenv = nenv if (nenv is not None) else nout
    U = rand_unitary(nout * nenv, iscomplex=iscomplex)
    return U[:, :nin]

def rand_degradable_channel(nin, nout, nenv, iscomplex=False):
    """Generate random degradable channel, represented as a Stinespring isometry V such that
    
        N(X)  = Tr_2[V X V']
        Nc(X) = Tr_1[V X V']
        
    Also returns Stinespring isometry W such that
    
        Nc(X) = Tr_2[W N(X) W']
        
    See https://arxiv.org/abs/0802.1360

    Parameters
    ----------
    nin : int
        Dimension of input system.
    nout : int, optional
        Dimension of output system. Default is nin.
    nenv : int, optional
        Dimension of environment system. Default is nout.
    iscomplex : bool, optional
        Whether the Stinespring is real (False) or complex (True). Default is False.
        
    Returns
    -------
    ndarray
        Stinespring operator V of dimension (nout*nenv, nin) corresponding to N(X) = Tr_2[V X V'].
    ndarray
        Stinespring operator W of dimension (nin*nenv, nout) corresponding to Nc(X) = Tr_2[W N(X) W'].
    """
    assert nenv <= nin
    dtype = np.complex128 if iscomplex else np.float64

    V = np.zeros((nout*nenv, nin), dtype=dtype)    # N Stinespring isometry
    W = np.zeros((nin*nenv, nout), dtype=dtype)    # Ξ Stinespring isometry

    U = rand_unitary(nin, iscomplex=iscomplex)
    for k in range(nout):
        # Generate random vector
        if iscomplex:
            v = np.random.normal(size=(nenv, 1)) + np.random.normal(size=(nenv, 1)) * 1j
        else:
            v = np.random.normal(size=(nenv, 1))
        v /= np.linalg.norm(v)

        # Make Kraus operator and insert into N Stinespring isometry
        K = v @ U[[k], :]
        V[k*nenv : (k + 1)*nenv, :] = K

        # Make Kraus operator and insert into Ξ Stinespring isometry
        W[k::nin, [k]] = v

    return V, W

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