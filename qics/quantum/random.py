import numpy as np
import scipy as sp
import qics.quantum


def density_matrix(n, iscomplex=False):
    r"""Generate random density matrix on Haar measure, i.e., positive semifedinite 
    matrix :math:`X` satisfying :math:`\text{tr}[X] = 1`.

    Parameters
    ----------
    n : int
        Dimension of random ``(n, n)`` matrix.
    iscomplex : bool, optional
        Whether the matrix is symmetric (``False``) or Hermitian (``True``). Default is 
        ``False``.

    Returns
    -------
    ndarray
        Random density matrix of dimension ``(n, n)``.
    """
    if iscomplex:
        X = np.random.normal(size=(n, n)) + np.random.normal(size=(n, n)) * 1j
    else:
        X = np.random.normal(size=(n, n))
    rho = X @ X.conj().T
    return rho / np.trace(rho)


def pure_density_matrix(n, iscomplex=False):
    r"""Generate random pure density matrix i.e., rank 1 positive semifedinite matrix 
    :math:`X` satisfying :math:`\text{tr}[X] = 1`. See 
    `here <https://sumeetkhatri.com/wp-content/uploads/2020/05/random_pure_states.pdf>`_
    for additional details.

    Parameters
    ----------
    n : int
        Dimension of random ``(n, n)`` matrix.
    iscomplex : bool, optional
        Whether the matrix is symmetric (``False``) or Hermitian (``True``). Default is 
        ``False``.

    Returns
    -------
    ndarray
        Random density matrix of dimension ``(n, n)``.
    """
    if iscomplex:
        psi = np.random.normal(size=(n)) + np.random.normal(size=(n)) * 1j
    else:
        psi = np.random.normal(size=(n))
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    rho = (rho + rho.conj().T) * 0.5
    return rho


def unitary(n, iscomplex=False):
    r"""Generate random unitary uniformly distributed on Haar measure, i.e., matrix 
    :math:`U` satisfying :math:`U^\dagger U = UU^\dagger = \mathbb{I}`. See 
    `here <https://case.edu/artsci/math/esmeckes/Meckes_SAMSI_Lecture2.pdf>`_ for 
    additional details.

    Parameters
    ----------
    n : int
        Dimension of random ``(n, n)`` unitary.
    iscomplex : bool, optional
        Whether the unitary is real (``False``) or complex (``True``). Default is
        ``False``.

    Returns
    -------
    ndarray
        Random unitary of dimension ``(n, n)``.
    """
    if iscomplex:
        X = np.random.normal(size=(n, n)) + np.random.normal(size=(n, n)) * 1j
    else:
        X = np.random.normal(size=(n, n))
    U, _ = np.linalg.qr(X)
    return U


def stinespring_operator(nin, nout=None, nenv=None, iscomplex=False):
    r"""Generate random Stinespring operator uniformly distributed on Hilbert-Schmidt 
    measure, i.e., isometry :math:`V` corresponding to quantum channel 
    :math:`\mathcal{N}(X) = \text{tr}_E[V X V^\dagger]`. See 
    `here <https://arxiv.org/abs/2011.02994>`_ for additional details.

    Parameters
    ----------
    nin : int
        Dimension of input system.
    nout : int, optional
        Dimension of output system. Default is ``nin``.
    nenv : int, optional
        Dimension of environment system. Default is ``nout``.
    iscomplex : bool, optional
        Whether the Stinespring is real (``False``) or complex (``True``). Default is 
        ``False``.

    Returns
    -------
    ndarray
        Random Stinespring operator of dimension ``(nout*nenv, nin)``
    """
    nout = nout if (nout is not None) else nin
    nenv = nenv if (nenv is not None) else nout
    U = unitary(nout * nenv, iscomplex=iscomplex)
    return U[:, :nin]


def degradable_channel(nin, nout, nenv, iscomplex=False):
    r"""Generate random degradable channel, represented as a Stinespring isometry 
    :math:`V` such that

    .. math::

        \mathcal{N}(X)           &= \text{tr}_2[V X V^\dagger]

        \mathcal{N}_\text{c}(X) &= \text{tr}_1[V X V^\dagger]

    Also returns Stinespring isometry W such that

    .. math::

        \mathcal{N}_\text{c}(X) = \text{tr}_2[W \mathcal{N}(X) W^\dagger].

    See `here <https://arxiv.org/abs/0802.1360>`_ for additional details.

    Parameters
    ----------
    nin : int
        Dimension of input system.
    nout : int, optional
        Dimension of output system. Default is ``nin``.
    nenv : int, optional
        Dimension of environment system. Default is ``nout``.
    iscomplex : bool, optional
        Whether the Stinespring is real (``False``) or complex (``True``). Default is 
        ``False``.

    Returns
    -------
    ndarray
        Stinespring operator :math:`V` of dimension ``(nout*nenv, nin)`` corresponding 
        to :math:`\mathcal{N}(X)=\text{tr}_2[V X V^\dagger]`.
    ndarray
        Stinespring operator :math:`W` of dimension ``(nin*nenv, nout)`` corresponding 
        to :math:`\mathcal{N}_\text{c}(X)=\text{tr}_2[W \mathcal{N}(X) W^\dagger]`.
    """
    assert nenv <= nin
    dtype = np.complex128 if iscomplex else np.float64

    V = np.zeros((nout * nenv, nin), dtype=dtype)  # N Stinespring isometry
    W = np.zeros((nin * nenv, nout), dtype=dtype)  # Ξ Stinespring isometry

    U = unitary(nin, iscomplex=iscomplex)
    for k in range(nout):
        # Generate random vector
        if iscomplex:
            v = np.random.normal(size=(nenv, 1)) + np.random.normal(size=(nenv, 1)) * 1j
        else:
            v = np.random.normal(size=(nenv, 1))
        v /= np.linalg.norm(v)

        # Make Kraus operator and insert into N Stinespring isometry
        K = v @ U[[k], :]
        V[k * nenv : (k + 1) * nenv, :] = K

        # Make Kraus operator and insert into Ξ Stinespring isometry
        W[k::nin, [k]] = v

    return V, W


def choi_operator(nin, nout=None, M=None, iscomplex=False):
    r"""Random Choi operator uniformly distributed on Hilbert-Schmidt measure. See 
    `here <https://arxiv.org/abs/2011.02994>`_ for additional details

    Parameters
    ----------
    nin : int
        Dimension of input system.
    nout : int, optional
        Dimension of output system. Default is ``nin``.
    M : int, optional
        Dimension used to generate random Choi operator. Default is ``nout*nin``.
    iscomplex : bool, optional
        Whether the Stinespring is real (``False``) or complex (``True``). Default is 
        ``False``.

    Returns
    -------
    ndarray
        Random Choi operator corresponding to matrix of size ``(nout*nin, nout*nin)``.
    """

    nout = nout if (nout is not None) else nin
    M = M if (M is not None) else nout * nin

    # Sample random Wishart ensemble
    if iscomplex:
        G = (
            np.random.normal(size=(nin * nout, M))
            + np.random.normal(size=(nin * nout, M)) * 1j
        )
        G *= np.sqrt(2.0)
    else:
        G = np.random.normal(size=(nin * nout, M))
    W = G @ G.conj().T

    # Obtain normalization required for trace preserving property
    H = qics.quantum.p_tr(W, (nout, nin), 0)
    I_H_irt2 = qics.quantum.i_kr(sp.linalg.sqrtm(np.linalg.inv(H)), (nout, nin), 0)

    # Return normalized Choi matrix
    J = I_H_irt2 @ W @ I_H_irt2
    return 0.5 * (J + J.conj().T)
