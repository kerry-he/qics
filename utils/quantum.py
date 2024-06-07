import numpy as np

def randDensityMatrix(n, hermitian=False):
    # Generate random density matrix on Haar measure
    if hermitian:
        X = np.random.normal(size=(n, n)) + np.random.normal(size=(n, n)) * 1j
    else:
        X = np.random.normal(size=(n, n))
    rho = X @ X.conj().T
    return rho / np.trace(rho)

def randPureDensityMatrix(n, hermitian=False):
    # Generate random density matrix on Haar measure
    # https://sumeetkhatri.com/wp-content/uploads/2020/05/random_pure_states.pdf
    if hermitian:
        psi = np.random.normal(size=(n)) + np.random.normal(size=(n)) * 1j
    else:
        psi = np.random.normal(size=(n))
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    rho = (rho + rho.conj().T) * 0.5
    return rho

def randUnitary(n):
    # Random unitary uniformly distributed on Haar measure
    # https://case.edu/artsci/math/esmeckes/Meckes_SAMSI_Lecture2.pdf
    X = np.random.normal(size=(n, n))
    U, _ = np.linalg.qr(X)
    return U

def randStinespringOperator(nin, nout=None, nenv=None):
    # Random Stinespring operator uniformly distributed on Hilbert-Schmidt measure
    # https://arxiv.org/abs/2011.02994
    nout = nout if (nout is not None) else nin
    nenv = nenv if (nenv is not None) else nout
    U = randUnitary(nout * nenv)
    return U[:, :nin]

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

    vec = np.zeros((n*n, 1))
    for i in range(n):
        vec += np.sqrt(D[i]) * np.kron(U[:, [i]], U[:, [i]])

    return vec @ vec.T