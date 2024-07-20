import numpy as np

def randDensityMatrix(n, iscomplex=False):
    # Generate random density matrix on Haar measure
    if iscomplex:
        X = np.random.normal(size=(n, n)) + np.random.normal(size=(n, n)) * 1j
    else:
        X = np.random.normal(size=(n, n))
    rho = X @ X.conj().T
    return rho / np.trace(rho)

def randPureDensityMatrix(n, iscomplex=False):
    # Generate random density matrix on Haar measure
    # https://sumeetkhatri.com/wp-content/uploads/2020/05/random_pure_states.pdf
    if iscomplex:
        psi = np.random.normal(size=(n)) + np.random.normal(size=(n)) * 1j
    else:
        psi = np.random.normal(size=(n))
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    rho = (rho + rho.conj().T) * 0.5
    return rho

def randUnitary(n, iscomplex=False):
    # Random unitary uniformly distributed on Haar measure
    # https://case.edu/artsci/math/esmeckes/Meckes_SAMSI_Lecture2.pdf
    if iscomplex:
        X = np.random.normal(size=(n, n)) + np.random.normal(size=(n, n)) * 1j
    else:
        X = np.random.normal(size=(n, n))
    U, _ = np.linalg.qr(X)
    return U

def randStinespringOperator(nin, nout=None, nenv=None, iscomplex=False):
    # Random Stinespring operator uniformly distributed on Hilbert-Schmidt measure
    # https://arxiv.org/abs/2011.02994
    nout = nout if (nout is not None) else nin
    nenv = nenv if (nenv is not None) else nout
    U = randUnitary(nout * nenv, iscomplex=iscomplex)
    return U[:, :nin]

def randDegradableChannel(nin, nout, nenv, iscomplex=False):
    # Random degradable channel, represented as a Stinespring isometry
    # Returns both Stinespring isometry V such that
    #     N(X)  = Tr_2[VXV']
    #     Nc(X) = Tr_1[VXV']
    # Also returns Stinespring isometry W such that
    #     Nc(X) = Tr_2[WN(X)W']
    # See https://arxiv.org/abs/0802.1360

    assert nenv <= nin
    dtype = np.complex128 if iscomplex else np.float64

    V = np.zeros((nout*nenv, nin), dtype=dtype)    # N Stinespring isometry
    W = np.zeros((nin*nenv, nout), dtype=dtype)    # Ξ Stinespring isometry

    U = randUnitary(nin, iscomplex=iscomplex)
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

def quantEntropy(rho):
    # "Safe" quantum entropy for positive definite matrices
    eig = np.linalg.eigvalsh(rho)
    eig = eig[eig > 0]
    return -sum(eig * np.log(eig))

def entropy(x):
    # "Safe" entropy for positive vectors
    x = x[x > 0]
    return -sum(x * np.log(x))

def purify(rho):
    # Returns a purification of a quantum state
    n = rho.shape[0]
    D, U = np.linalg.eigh(rho)

    vec = np.zeros((n*n, 1), dtype=rho.dtype)
    for i in range(n):
        vec += np.sqrt(D[i]) * np.kron(U[:, [i]], U[:, [i]])

    return vec @ vec.conj().T