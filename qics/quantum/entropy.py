import numpy as np


def entropy(x):
    r"""Computes classical (Shannon) entropy

    .. math::

        H(x) = -\sum_{i=1}^n x_i \log(x_i),

    for nonnegative vector :math:`x`.

    Parameters
    ----------
    x : ndarray
        Nonnegative ``(n, 1)`` vector to compute classical entropy of.

    Returns
    -------
    float
        Classical entropy of ``x``.
    """
    x = x[x > 0]
    return -sum(x * np.log(x))


def quant_entropy(X):
    r"""Computes quantum (von Neumann) entropy

    .. math::

        S(X) = -\text{tr}[X \log(X)],

    for positive semidefinite matrix :math:`X`.

    Parameters
    ----------
    X : ndarray
        Positive semidefinite ``(n, n)`` matrix to compute quantum entropy of.

    Returns
    -------
    float
        Quantum entropy of ``X``.
    """
    eig = np.linalg.eigvalsh(X)
    return entropy(eig)


def purify(X):
    r"""Returns a purification of a quantum state X. If X has spectral decomposition

    .. math::

        X = \sum_{i=1}^n \lambda_i | v_i \rangle\langle v_i |,

    then the purification is :math:`| \psi \rangle\langle \psi |` where

    .. math::

        | \psi \rangle = \sum_{i=1}^n \sqrt{\lambda_i} (| v_i \rangle \otimes | v_i
        \rangle).

    Parameters
    ----------
    X : ndarray
        Density matrix of size ``(n, n)``.

    Returns
    -------
    ndarray
        Purification matrix of ``X`` of size ``(n*n, n*n)``.
    """
    n = X.shape[0]
    D, U = np.linalg.eigh(X)

    vec = np.zeros((n * n, 1), dtype=X.dtype)
    for i in range(n):
        vec += np.sqrt(max(0.0, D[i])) * np.kron(U[:, [i]], U[:, [i]])

    return vec @ vec.conj().T
