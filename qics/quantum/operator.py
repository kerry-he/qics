import numpy as np


def p_tr(mat, dims, sys):
    r"""Performs the partial trace on a bipartite matrix, e.g., for
    a bipartite state, this is the unique linear map satisfying

    .. math::

        X \otimes Y \mapsto \text{tr}[X] Y,

    if ``sys=0``, or

    .. math::

        X \otimes Y \mapsto \text{tr}[Y] X,

    if ``sys=1``.

    Parameters
    ----------
    mat : ndarray
        Input ``(n0*n1*...*nk-1, n0*n1*...*nk-1)`` matrix to perform the partial trace 
        on.
    dims : tuple(int)
        The dimensions ``(n0, n1, ..., nk-1)`` of the ``k`` subsystems.
    sys : int or tuple(int)
        Which of the ``k`` subsystems to trace out.

    Returns
    -------
    ndarray
        The resulting matrix after taking the partial trace. Has dimension 
        ``(n0*n1*...*nk-1 / ni, n0*n1*...*nk-1 / ni)`` where ``i`` is the system being 
        traced out.
    """

    if isinstance(sys, int):
        sys = [
            sys,
        ]
    if isinstance(sys, tuple):
        sys = list(sys)
    not_sys = list(set(range(len(dims))) - set(sys))

    # Sort subsystems so the ones we want to partial trace are at the front
    reordered_dims = sys + not_sys
    reordered_dims = reordered_dims + [k + len(dims) for k in reordered_dims]

    tr_dim = np.prod([dims[k] for k in sys], dtype=int)
    new_dim = np.prod([dims[k] for k in not_sys], dtype=int)

    temp = np.transpose(mat.reshape(*dims, *dims), reordered_dims)
    temp = temp.reshape(tr_dim, new_dim, tr_dim, new_dim)

    return np.trace(temp, axis1=0, axis2=2)


def p_tr_multi(out, mat, dims, sys):
    r"""Performs the partial trace on a list of bipartite matrices.

    Parameters
    ----------
    out : ndarray
        Preallocated list of matrices to store the output. Has dimension 
        ``(p, n0*n1*...*nk-1 / ni, n0*n1*...*nk-1 / ni)`` where ``i`` is the system 
        being traced out.
    mat : ndarray
        Input ``(p, n0*n1*...*nk-1, n0*n1*...*nk-1)`` list of matrices to perform the 
        partial trace on.
    dims : tuple[int]
        The dimensions ``(n0, n1, ..., nk)`` of the ``p`` subsystems.
    sys : int or tuple(int)
        Which systems to trace out.
    """

    if isinstance(sys, int):
        sys = [
            sys,
        ]
    if isinstance(sys, tuple):
        sys = list(sys)
    not_sys = list(set(range(len(dims))) - set(sys))

    # Sort subsystems so the ones we want to partial trace are at the front
    reordered_dims = sys + not_sys
    reordered_dims = (
        [0]
        + [k + 1 for k in reordered_dims]
        + [k + 1 + len(dims) for k in reordered_dims]
    )

    tr_dim = np.prod([dims[k] for k in sys], dtype=int)
    new_dim = np.prod([dims[k] for k in not_sys], dtype=int)

    temp = np.transpose(mat.reshape(-1, *dims, *dims), reordered_dims)
    temp = temp.reshape(-1, tr_dim, new_dim, tr_dim, new_dim)

    np.trace(temp, axis1=1, axis2=3, out=out)

    return out


def i_kr(mat, dims, sys):
    r"""Performs Kronecker product between the indentity matrix and a given matrix, 
    e.g., for a bipartite system

    .. math::

        X \mapsto \mathbb{I} \otimes X,

    if ``sys=0``, or

    .. math::

        X \mapsto X \otimes \mathbb{I},

    if ``sys=1``.

    Parameters
    ----------
    mat : ndarray
        Input matrix to perform the partial trace on. Has dimension 
        ``(n0*n1*...*nk-1 / ni, n0*n1*...*nk-1 / ni)`` where ``i`` is the system being 
        traced out.
    dim : tuple[int]
        The dimensions ``(n0, n1, ..., nk)`` of the subsystems.
    sys : int or tuple(int)
        Which system to Kroneker product should act on.

    Returns
    -------
    ndarray
        The resulting ``(n0*n1*...*nk-1, n0*n1*...*nk-1)`` matrix after performing the
        Kronecker product.
    """

    if isinstance(sys, int):
        sys = [
            sys,
        ]
    if isinstance(sys, tuple):
        sys = list(sys)
    not_sys = list(set(range(len(dims))) - set(sys))

    # Sort subsystems so the ones we want to partial trace are at the front
    reordered_dims = (
        sys + [k + len(dims) for k in sys] + not_sys + [k + len(dims) for k in not_sys]
    )

    N = np.prod(dims)
    new_dims = [dims[k] for k in not_sys] if len(not_sys) > 0 else [1]

    out = np.zeros((N, N), dtype=mat.dtype)
    out_view = np.transpose(out.reshape(*dims, *dims), reordered_dims)

    r = np.meshgrid(*[range(dims[k]) for k in sys])
    r = list(np.array(r).reshape(len(sys), -1))
    out_view[*r, *r, ...] = mat.reshape(*new_dims, *new_dims)

    return out


def i_kr_multi(out, mat, dims, sys):
    r"""Performs Kronecker product between the indentity matrix and a given list of 
    matrices.

    Parameters
    ----------
    out : ndarray
        Preallocated ``(p, n0*n1*...*nk-1, n0*n1*...*nk-1)`` list of matrices to store 
        the output.
    mat : ndarray
        Input matrix to perform the partial trace on. Has dimension 
        ``(p, n0*n1*...*nk-1 / ni, n0*n1*...*nk-1 / ni)`` where ``i`` is the system 
        being traced out.
    dim : tuple[int]
        The dimensions ``(n0, n1, ..., nk)`` of the subsystems.
    sys : int or tuple(int)
        Which system to Kroneker product should act on.
    """

    if isinstance(sys, int):
        sys = [
            sys,
        ]
    if isinstance(sys, tuple):
        sys = list(sys)
    not_sys = list(set(range(len(dims))) - set(sys))

    # Sort subsystems so the ones we want to partial trace are at the front
    reordered_dims = [0]
    reordered_dims += [k + 1 for k in sys]
    reordered_dims += [k + 1 + len(dims) for k in sys]
    reordered_dims += [k + 1 for k in not_sys]
    reordered_dims += [k + 1 + len(dims) for k in not_sys]

    new_dims = [dims[k] for k in not_sys]

    out.fill(0.0)
    out_view = np.transpose(out.reshape(-1, *dims, *dims), reordered_dims)

    r = np.meshgrid(*[range(dims[k]) for k in sys])
    r = list(np.array(r).reshape(len(sys), -1))
    out_view[:, *r, *r, ...] = mat.reshape(-1, 1, *new_dims, *new_dims)

    return out


def partial_transpose(mat, dims, sys):
    r"""Performs the partial transpose on a multipartite matrix, e.g., for a bipartite 
    state, the unique linear map satisfying

    .. math::

        X \otimes Y \mapsto X^\top \otimes Y,

    if ``sys=0``, or

    .. math::

        X \otimes Y \mapsto X \otimes Y^\top,

    if ``sys=1``.

    Parameters
    ----------
    mat : ndarray
        Input ``(n0*n1*...*nk-1, n0*n1*...*nk-1)`` matrix to perform the partial 
        transpose on.
    dim : tuple[int]
        The dimensions ``(n0, n1, ..., nk-1)`` of the ``k`` subsystems.
    sys : int
        Which of the ``k`` subsystems to transpose.

    Returns
    -------
    ndarray
        The resulting ``(n0*n1*...*nk-1, n0*n1*...*nk-1)`` matrix after
        performing the partial transpose.
    """
    N = np.prod(dims)

    temp = mat.reshape(*dims, *dims)
    temp = np.swapaxes(temp, sys, sys + len(dims))

    return temp.reshape(N, N)


def swap(mat, dims, sys1, sys2):
    r"""Swaps two systems of a multipartite quantum state, e.g., for a bipartite state, 
    it is the unique linear maps satisfying

    .. math::

        X \otimes Y \mapsto Y \otimes X.

    Parameters
    ----------
    mat : ndarray
        Input ``(n0*n1*...*nk-1, n0*n1*...*nk-1)`` matrix to perform the swap operator 
        on.
    dim : tuple[int]
        The dimensions ``(n0, n1, ..., nk-1)`` of the ``k`` subsystems.
    sys1 : int
        First of the ``k`` subsystems to swap.
    sys2 : int
        Second of the ``k`` subsystems to swap

    Returns
    -------
    ndarray
        The resulting ``(n0*n1*...*nk-1, n0*n1*...*nk-1)`` matrix after performing the 
        swap operator.
    """
    N = np.prod(dims)

    temp = mat.reshape(*dims, *dims)
    temp = np.swapaxes(temp, sys1, sys2)
    temp = np.swapaxes(temp, len(dims) + sys1, len(dims) + sys2)

    return temp.reshape(N, N)
