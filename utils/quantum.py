import numpy as np

def randDensityMatrix(n):
    # Generate random density matrix on Haar measure
    x = np.random.normal(size=(n, n))
    rho = x @ x.T
    return rho / np.trace(rho)

def quantEntropy(rho):
    eig = np.linalg.eigvalsh(rho)
    eig = eig[eig > 0]
    return -sum(eig * np.log(eig))