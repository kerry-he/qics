import math

import numpy as np


def vec_dim(side, iscomplex=False, compact=False):
    """Computes the dimension of a vectorized matrix.

    Parameters
    ----------
    side : :obj:`int`
        The dimension of the matrix.
    iscomplex : :obj:`bool`, optional
        Whether the matrix is Hermitian (``True``) or symmetric 
        (``False``). The default is ``False``.
    compact : :obj:`bool`, optional
        Whether to assume a compact vector representation or not. The
        default is ``False``.

    Returns
    -------
    :obj:`int`
        The dimension of the vector.
    """
    if compact:
        if iscomplex:
            return side * side
        else:
            return side * (side + 1) // 2
    else:
        if iscomplex:
            return 2 * side * side
        else:
            return side * side


def mat_dim(len, iscomplex=False, compact=False):
    """Computes the dimension of the matrix correpsonding to a vector.

    Parameters
    ----------
    len : :obj:`int`
        The dimension of the vector.
    iscomplex : :obj:`bool`, optional
        Whether the matrix is Hermitian (``True``) or symmetric 
        (``False``). The default is ``False``.
    compact : :obj:`bool`, optional
        Whether to assume a compact vector representation or not. The
        default is ``False``.

    Returns
    -------
    :obj:`int`
        The dimension of the matrix.
    """
    if compact:
        if iscomplex:
            return math.isqrt(len)
        else:
            return math.isqrt(1 + 8 * len) // 2
    else:
        if iscomplex:
            return math.isqrt(len // 2)
        else:
            return math.isqrt(len)


def mat_to_vec(mat, compact=False):
    r"""Reshapes a symmetric or Hermitian matrix into a column vector.

    If ``mat`` is of type :obj:`~numpy.float64` and ``compact=False``, then
    this performs the vectorization
    
    .. math::

        \begin{bmatrix}a & b & d \\ b & c & e \\ d & e & f\end{bmatrix}
        \mapsto
        \begin{bmatrix}a & b & d & b & c & e & d & e & f\end{bmatrix}^\top.

    If ``mat`` is of type :obj:`~numpy.float64` and ``compact=True``, then
    this performs the vectorization

    .. math::

        \begin{bmatrix}a & b & d \\ b & c & e \\ d & e & f\end{bmatrix}
        \mapsto\begin{bmatrix}
            a & \sqrt{2} b & c & \sqrt{2} d & \sqrt{2}e & f
        \end{bmatrix}^{\top}.

    If ``mat`` is of type :obj:`~numpy.complex128` and ``compact=False``,
    then this performs the vectorization

    .. math::

        \begin{bmatrix}
            a & b+ci & e+fi \\ b-ci & d & g+hi \\ e-fi & g-hi & k
        \end{bmatrix}\mapsto
        \begin{bmatrix}
            a & 0 & b & c & e & f & b & -c & d & 0 
            & g & h & e & -f & g & -h & k & 0
        \end{bmatrix}^{\top}.

    If ``mat`` is of type :obj:`~numpy.complex128` and ``compact=True``,
    then this performs the vectorization

    .. math::

        \begin{bmatrix}
            a & b+ci & e+fi \\ b-ci & d & g+hi \\ e-fi & g-hi & k
        \end{bmatrix}\mapsto
        \begin{bmatrix}
            a & \sqrt{2} b & \sqrt{2} c & d & \sqrt{2} e & \sqrt{2} f & 
            \sqrt{2} g & \sqrt{2} h & k
        \end{bmatrix}^{\top}.

    Parameters
    ----------
    mat : :class:`~numpy.ndarray`
        Input matrix to vectorize, either of type :obj:`~numpy.float64`
        or :obj:`~numpy.complex128`.
    compact : :obj:`bool`, optional
        Whether to convert to a compact vector representation or not.
        The default is ``False``.

    Returns
    -------
    :class:`~numpy.ndarray`
        The resulting vectorized matrix.
    """
    assert mat.dtype == np.float64 or mat.dtype == np.complex128
    iscomplex = mat.dtype == np.complex128

    n = mat.shape[0]
    vn = vec_dim(n, iscomplex=iscomplex, compact=compact)

    if compact:
        rt2 = np.sqrt(2.0)
        vec = np.empty((vn, 1))

        k = 0
        for j in range(n):
            for i in range(j):
                vec[k] = mat[i, j].real * rt2
                k += 1
                if iscomplex:
                    vec[k] = mat[i, j].imag * rt2
                    k += 1
            vec[k] = mat[j, j].real
            k += 1

        return vec
    else:
        mat = np.ascontiguousarray(mat)
        return mat.view(dtype=np.float64).reshape(-1, 1).copy()


def vec_to_mat(vec, iscomplex=False, compact=False):
    r"""Reshapes a column vector into a symmetric or Hermitian matrix.

    If ``iscomplex=False`` and ``compact=False``, then this returns the
    matrix
    
    .. math::

        \begin{bmatrix}a & b & d & b & c & e & d & e & f\end{bmatrix}^\top
        \mapsto
        \begin{bmatrix}a & b & d \\ b & c & e \\ d & e & f\end{bmatrix}.

    If ``iscomplex=False`` and ``compact=True``, then this performs the 
    matrix

    .. math::

        \begin{bmatrix}
            a & \sqrt{2} b & c & \sqrt{2} d & \sqrt{2}e & f
        \end{bmatrix}^{\top}\mapsto
        \begin{bmatrix}a & b & d \\ b & c & e \\ d & e & f\end{bmatrix}.

    If ``iscomplex=True`` and ``compact=False``, then this returns the
    matrix

    .. math::

        \begin{bmatrix}
            a & 0 & b & c & e & f & b & -c & d & 0 
            & g & h & e & -f & g & -h & k & 0
        \end{bmatrix}^{\top}\mapsto
        \begin{bmatrix}
            a & b+ci & e+fi \\ b-ci & d & g+hi \\ e-fi & g-hi & k
        \end{bmatrix}.

    If ``iscomplex=True`` and ``compact=True``, then this returns the
    matrix

    .. math::

        \begin{bmatrix}
            a & \sqrt{2} b & \sqrt{2} c & d & \sqrt{2} e & \sqrt{2} f & 
            \sqrt{2} g & \sqrt{2} h & k
        \end{bmatrix}^{\top}\mapsto
        \begin{bmatrix}
            a & b+ci & e+fi \\ b-ci & d & g+hi \\ e-fi & g-hi & k
        \end{bmatrix}.


    Parameters
    ----------
    mat : :class:`~numpy.ndarray`
        Input vector to reshape into a matrix.
    iscomplex : :obj:`bool`, optional
        Whether the resulting matrix is Hermitian (``True``) or symmetric
        (``False``). The default is ``False``.
    compact : :obj:`bool`, optional
        Whether to convert from a compact vector representation or not. 
        The default is ``False``.

    Returns
    -------
    :class:`~numpy.ndarray`
        The resulting vector.
    """
    vn = vec.size
    n = mat_dim(vn, iscomplex=iscomplex, compact=compact)
    dtype = np.complex128 if iscomplex else np.float64

    if compact:
        irt2 = np.sqrt(0.5)
        mat = np.empty((n, n), dtype=dtype)

        k = 0
        for j in range(n):
            for i in range(j):
                if iscomplex:
                    mat[i, j] = (vec[k, 0] + vec[k + 1, 0] * 1j) * irt2
                    k += 2
                else:
                    mat[i, j] = vec[k, 0] * irt2
                    k += 1
                mat[j, i] = mat[i, j].conjugate()

            mat[j, j] = vec[k, 0]
            k += 1

        return mat
    else:
        if iscomplex:
            n = math.isqrt(vn // 2)
            mat = vec.reshape((-1, 2)).view(dtype=np.complex128).reshape(n, n)
            return (mat + mat.conj().T) * 0.5
        else:
            n = math.isqrt(vn)
            mat = vec.reshape((n, n))
            return (mat + mat.T) * 0.5


def lin_to_mat(lin, dims, iscomplex=False, compact=(False, True)):
    """Computes the matrix corresponding to a linear map from vectorized
    symmetric matrices to vectorized symmetric matrices.

    Parameters
    ----------
    lin : :obj:`callable`
        Linear operator sending symmetric matrices to symmetric matrices.
    dims : :obj:`tuple` of :obj:`int` 
        The dimensions ``(ni, no)`` of the input and output matrices of the
        linear operator.
    iscomplex : :obj:`bool`, optional
        Whether the matrix to vectorize is Hermitian (``True``) or 
        symmetric (``False``). Default is ``False``.
    compact : :obj:`tuple` of :obj:`bool`, optional
        Whether to use a compact vector representation or not for the input
        and output matrices, respectively. Default is ``(False, True)``.

    Returns
    -------
    :class:`~numpy.ndarray`
        The matrix representation of the given linear operator.
    """
    vni = vec_dim(dims[0], iscomplex=iscomplex, compact=compact[0])
    vno = vec_dim(dims[1], iscomplex=iscomplex, compact=compact[1])
    mat = np.zeros((vno, vni))

    for k in range(vni):
        H = np.zeros((vni, 1))
        H[k] = 1.0
        H_mat = vec_to_mat(H, iscomplex=iscomplex, compact=compact[0])
        lin_H = lin(H_mat)
        vec_out = mat_to_vec(lin_H, compact=compact[1])
        mat[:, [k]] = vec_out

    return mat


def eye(n, iscomplex=False, compact=(False, True)):
    """Computes the matrix representation of the identity map for
    vectorized symmetric or Hermitian matrices.

    Parameters
    ----------
    n : :obj:`int`
        The dimensions of the ``(n, n)`` matrix the identity is acting on.
    iscomplex : :obj:`bool`, optional
        Whether the matrix to vectorize is Hermitian (``True``) or 
        symmetric (``False``). Default is ``False``.
    compact : :obj:`tuple` of :obj:`bool`, optional
        Whether to use a compact vector representation or not for the input
        and output matrices, respectively. Default is ``(False, True)``.

    Returns
    -------
    :class:`~numpy.ndarray`
        The matrix representation of the identity superoperator.
    """
    return lin_to_mat(lambda X: X, (n, n), iscomplex=iscomplex, compact=compact)
