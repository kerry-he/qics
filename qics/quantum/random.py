import numpy as np
import scipy as sp
import qics.quantum


def density_matrix(n, iscomplex=False):
    r"""Generate a random density matrix, i.e., positive semidefinite
    matrix :math:`X\in\mathbb{H}^n` satisfying :math:`\text{tr}[X] = 1`,
    uniformly on the on the Haar measure.

    Parameters
    ----------
    n : :obj:`int`
        Dimension of random density matrix :math:`X`.
    iscomplex : :obj:`bool`, optional
        Whether the matrix is real (``False``) or complex (``True``). 
        The default is ``False``.

    Returns
    -------
    :class:`~numpy.ndarray`
        Random density matrix of dimension ``(n, n)``.
    """
    if iscomplex:
        X = np.random.normal(size=(n, n)) + np.random.normal(size=(n, n)) * 1j
    else:
        X = np.random.normal(size=(n, n))
    rho = X @ X.conj().T
    return rho / np.trace(rho)


def pure_density_matrix(n, iscomplex=False):
    r"""Generate a random pure density matrix i.e., rank 1 positive 
    semidefinite matrix :math:`X\in\mathbb{H}^n` satisfying
    :math:`\text{tr}[X] = 1`.

    Parameters
    ----------
    n : :obj:`int`
        Dimension of random pure density matrix :math:`X`.
    iscomplex : :obj:`bool`, optional
        Whether the matrix is real (``False``) or complex (``True``). 
        The default is ``False``.

    Returns
    -------
    :class:`~numpy.ndarray`
        Random pure density matrix of dimension ``(n, n)``.

    Notes
    -----
    See [1]_ for additional details.

    .. [1] Khatri, S. (2020) "Random Pure States".
           https://sumeetkhatri.com/wp-content/uploads/2020/05/random_pure_states.pdf
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
    r"""Generate a random unitary matrix, i.e., matrix :math:`U` satisfying
    :math:`U^\dagger U = UU^\dagger = \mathbb{I}`, uniformly distributed on
    Haar measure.

    Parameters
    ----------
    n : :obj:`int`
        Dimension of random unitary matrix :math:`U`.
    iscomplex : :obj:`bool`, optional
        Whether the unitary matrix is real (``False``) or complex  
        (``True``). The default is ``False``.

    Returns
    -------
    :class:`~numpy.ndarray`
        Random unitary matrix of dimension ``(n, n)``.

    Notes
    -----
    See [1]_ for additional details.

    .. [1] Meckes, E. (2013) "Random Unitary Matrices and Friends".
           https://case.edu/artsci/math/esmeckes/Meckes_SAMSI_Lecture2.pdf

    """
    if iscomplex:
        X = np.random.normal(size=(n, n)) + np.random.normal(size=(n, n)) * 1j
    else:
        X = np.random.normal(size=(n, n))
    U, _ = np.linalg.qr(X)
    return U


def stinespring_operator(nin, nout=None, nenv=None, iscomplex=False):
    r"""Generate a random Stinespring operator, i.e., isometry :math:`V`
    corresponding to a quantum channel :math:`\mathcal{N}(X)=
    \text{tr}_E[VXV^\dagger]`, uniformly distributed on Hilbert-Schmidt
    measure.

    Parameters
    ----------
    nin : :obj:`int`
        Dimension of the input system.
    nout : :obj:`int`, optional
        Dimension of the output system. The default is ``nin``.
    nenv : :obj:`int`, optional
        Dimension of the environment system. The default is ``nout``.
    iscomplex : :obj:`bool`, optional
        Whether the Stinespring operator is real (``False``) or complex  
        (``True``). The default is ``False``.

    Returns
    -------
    :class:`~numpy.ndarray`
        Random Stinespring operator of dimension ``(nout*nenv, nin)``

    Notes
    -----
    See [1]_ for additional details.

    .. [1] Kukulski, Ryszard, et al. "Generating random quantum channels."
           Journal of Mathematical Physics 62.6 (2021).
    """
    nout = nout if (nout is not None) else nin
    nenv = nenv if (nenv is not None) else nout
    U = unitary(nout * nenv, iscomplex=iscomplex)
    return U[:, :nin]


def degradable_channel(nin, nout=None, nenv=None, iscomplex=False):
    r"""Generate random degradable channel :math:`\mathcal{N}`, represented
    as a Stinespring isometry :math:`V` such that

    .. math::

        \mathcal{N}(X) = \text{tr}_2[V X V^\dagger]

    Also returns Stinespring isometry :math:`W` corresponding to the
    complementary channel :math:`\mathcal{N}_\text{c}` such that

    .. math::

        \mathcal{N}_\text{c}(X) = \text{tr}_1[V X V^\dagger]
          = \text{tr}_2[W \mathcal{N}(X) W^\dagger].

    Parameters
    ----------
    nin : :obj:`int`
        Dimension of the input system.
    nout : :obj:`int`, optional
        Dimension of the output system. The default is ``nin``.
    nenv : :obj:`int`, optional
        Dimension of the environment system. The default is ``nout``.
    iscomplex : :obj:`bool`, optional
        Whether the Stinespring operators are real (``False``) or complex  
        (``True``). The default is ``False``.

    Returns
    -------
    :class:`~numpy.ndarray`
        Stinespring operator :math:`V` of dimension ``(nout*nenv, nin)``
        corresponding to :math:`\mathcal{N}(X)=\text{tr}_2[V X V^\dagger]`.
    :class:`~numpy.ndarray`
        Stinespring operator :math:`W` of dimension ``(nin*nenv, nout)`` 
        corresponding to :math:`\mathcal{N}_\text{c}(X)=
        \text{tr}_2[W \mathcal{N}(X) W^\dagger]`.

    Notes
    -----
    See [1]_ for additional details.

    .. [1] Cubitt, Toby S., Mary Beth Ruskai, and Graeme Smith.
           "The structure of degradable quantum channels."
           Journal of Mathematical Physics 49.10 (2008).    
    """
    nout = nout if (nout is not None) else nin
    nenv = nenv if (nenv is not None) else nout

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
    r"""Generate a random Choi operator :math:`\mathcal{J}_\mathcal{N}`
    representing a quantum channel :math:`\mathcal{N}`, i.e.,

    .. math::

        \mathcal{J}_\mathcal{N} = \sum_{i,j}^n 
        \mathcal{N}(| i \rangle\langle j |) \otimes | i \rangle\langle j |,
     
    uniformly distributed on the Hilbert-Schmidt measure.

    Parameters
    ----------
    nin : :obj:`int`
        Dimension of the input system.
    nout : :obj:`int`, optional
        Dimension of the output system. The default is ``nin``.
    M : :obj:`int`, optional
        Dimension used to determine the rank of the random Choi operator.         
        The default is ``nout*nin``.
    iscomplex : :obj:`bool`, optional
        Whether the Choi operator is real (``False``) or complex  
        (``True``). The default is ``False``.

    Returns
    -------
    :class:`~numpy.ndarray`
        Random Choi operator corresponding to matrix of size 
        ``(nout*nin, nout*nin)``.

    Notes
    -----
    See [1]_ for additional details.

    .. [1] Kukulski, Ryszard, et al. "Generating random quantum channels."
           Journal of Mathematical Physics 62.6 (2021).
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
