# Copyright (c) 2024, Kerry He, James Saunderson, and Hamza Fawzi

# This Python package QICS is licensed under the MIT license; see LICENSE.md 
# file in the root directory or at https://github.com/kerry-he/qics

import numpy as np


def entropy(x):
    r"""Computes either the classical (Shannon) entropy

    .. math::

        H(x) = -\sum_{i=1}^n x_i \log(x_i),

    for nonnegative vector :math:`x`, or the quantum (von Neumann) entropy

    .. math::

        S(X) = -\text{tr}[X \log(X)],

    for positive semidefinite matrix :math:`X`.

    Parameters
    ----------
    x : :class:`~numpy.ndarray`
        If this is a nonnegative array of size ``(n,)`` or ``(n, 1)``, we
        compute the classical entropy of :math:`x`. If this is a symmetric
        or Hermitian positive semidefinite array of size ``(n, n)``, then
        we compute the quantum entropy of :math:`X`

    Returns
    -------
    :obj:`float`
        Classical entropy of :math:`x` or quantum entropy of :math:`X`.
    """
    if x.size > 1 and len(x.shape) == 2 and x.shape[0] == x.shape[1]:
        eig = np.linalg.eigvalsh(x)
        return entropy(eig)

    x = x[x > 0]
    return -sum(x * np.log(x))

def purify(X):
    r"""Returns a purification of a quantum state :math:`X`. If :math:`X`
    has spectral decomposition

    .. math::

        X = \sum_{i=1}^n \lambda_i | v_i \rangle\langle v_i |,

    then the purification is :math:`| \psi \rangle\langle \psi |` where

    .. math::

        | \psi \rangle = \sum_{i=1}^n \sqrt{\lambda_i} (| v_i \rangle 
        \otimes | v_i \rangle).

    Parameters
    ----------
    X : :class:`~numpy.ndarray`
        Symmetric or Hermitian array of size ``(n, n)`` representing the
        matrix :math:`X` we want to find the purification of.

    Returns
    -------
    :class:`~numpy.ndarray`
        Symmetric or Hermitian array of size ``(n*n, n*n)`` representing the
        purification of :math:`X`.
    """
    n = X.shape[0]
    D, U = np.linalg.eigh(X)

    vec = np.zeros((n * n, 1), dtype=X.dtype)
    for i in range(n):
        vec += np.sqrt(max(0.0, D[i])) * np.kron(U[:, [i]], U[:, [i]])

    return vec @ vec.conj().T
