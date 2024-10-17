# Copyright (c) 2024, Kerry He, James Saunderson, and Hamza Fawzi

# This Python package QICS is licensed under the MIT license; see LICENSE.md
# file in the root directory or at https://github.com/kerry-he/qics

import numpy as np


def p_tr(mat, dims, sys):
    r"""Performs the partial trace on a multipartite state, e.g., for
    a bipartite state with ``dims=(n0, n1)``, this is the unique linear map
    satisfying

    .. math::

        X \otimes Y \mapsto \text{tr}[X] Y,

    if ``sys=0``, or

    .. math::

        X \otimes Y \mapsto \text{tr}[Y] X,

    if ``sys=1``, for all :math:`X,Y\in\mathbb{H}^n`.

    Parameters
    ----------
    mat : :class:`~numpy.ndarray`
        Array of size ``(n0*n1*...*nk-1, n0*n1*...*nk-1)`` represnting a
        matrix defined on :math:`k` subsystems which we want to take the
        partial trace of.
    dims : :obj:`tuple` of :obj:`int`
        The dimensions ``(n0, n1, ..., nk-1)`` of the :math:`k` subsystems.
    sys : :obj:`int` or :obj:`tuple` of :obj:`int`
        Which of the :math:`k` subsystems to trace out. Can define multiple
        subsystems to trace out.

    Returns
    -------
    :class:`~numpy.ndarray`
        The resulting matrix after taking the partial trace. Has dimension
        ``(n0*n1*...*nk-1 / nx, n0*n1*...*nk-1 / nx)`` where ``nx`` is the
        product of the dimensions of the subsystems that have been traced
        out.

    See also
    --------
    i_kr : The Kronecker product with the identity matrix

    Notes
    -----
    This is the adjoint operator of the Kronecker product with the
    identity matrix.
    """

    if isinstance(sys, int):
        sys = [
            sys,
        ]
    if isinstance(sys, tuple) or isinstance(sys, set):
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
    out : :class:`~numpy.ndarray`
        Preallocated list of matrices to store the output. Has dimension
        ``(p, n0*n1*...*nk-1 / ni, n0*n1*...*nk-1 / ni)`` where ``i`` is
        the system being traced out.
    mat : :class:`~numpy.ndarray`
        Input ``(p, n0*n1*...*nk-1, n0*n1*...*nk-1)`` list of matrices to
        perform the partial trace on.
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
    r"""Performs Kronecker product between the identity matrix and a given
    matrix, e.g., if we consider a bipartite setup with ``dims=(n0, n1)``,
    this is the unique linear map satisfying

    .. math::

        X \mapsto \mathbb{I} \otimes X,

    if ``sys=0``, or

    .. math::

        X \mapsto X \otimes \mathbb{I},

    if ``sys=1``, for all :math:`X\in\mathbb{H}^n`.

    Parameters
    ----------
    mat : :class:`~numpy.ndarray`
        Array of size ``(n0*n1*...*nk-1 / nx, n0*n1*...*nk-1 / nx)`` to
        apply the Kronecker product to, where ``nx`` is the product of the
        dimensions of the subsystems specified by ``sys``.
    dims : :obj:`tuple` of :obj:`int`
        The dimensions ``(n0, n1, ..., nk-1)`` of the :math:`k` subsystems
        which the output is defined on.
    sys : :obj:`int` or :obj:`tuple` of :obj:`int`
        Which of the :math:`k` subsystems to apply the Kronecker product
        to. Can define multiple subsystems.

    Returns
    -------
    :class:`~numpy.ndarray`
        The resulting ``(n0*n1*...*nk-1, n0*n1*...*nk-1)`` matrix after
        performing the Kronecker product.

    See also
    --------
    p_tr : The partial trace operator

    Notes
    -----
    This is the adjoint operator of the partial trace.
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
    out_view[tuple(r + r)] = mat.reshape(*new_dims, *new_dims)

    return out


def i_kr_multi(out, mat, dims, sys):
    r"""Performs Kronecker product between the indentity matrix and a given
    list of matrices.

    Parameters
    ----------
    out : :class:`~numpy.ndarray`
        Preallocated ``(p, n0*n1*...*nk-1, n0*n1*...*nk-1)`` list of
        matrices to store the output.
    mat : :class:`~numpy.ndarray`
        Input matrix to perform the partial trace on. Has dimension
        ``(p, n0*n1*...*nk-1 / ni, n0*n1*...*nk-1 / ni)`` where ``i`` is
        the system being traced out.
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
    reordered_dims = [k + 1 for k in sys]
    reordered_dims += [k + 1 + len(dims) for k in sys]
    reordered_dims += [0]
    reordered_dims += [k + 1 for k in not_sys]
    reordered_dims += [k + 1 + len(dims) for k in not_sys]

    new_dims = [dims[k] for k in not_sys]

    out.fill(0.0)
    out_view = np.transpose(out.reshape(-1, *dims, *dims), reordered_dims)

    r = np.meshgrid(*[range(dims[k]) for k in sys])
    r = list(np.array(r).reshape(len(sys), -1))
    out_view[tuple(r + r)] = mat.reshape(-1, *new_dims, *new_dims)

    return out


def partial_transpose(mat, dims, sys):
    r"""Performs the partial transpose on a multipartite matrix, e.g., for
    a bipartite state with ``dims=(n0, n1)``, the unique linear map
    satisfying

    .. math::

        X \otimes Y \mapsto X^\top \otimes Y,

    if ``sys=0``, or

    .. math::

        X \otimes Y \mapsto X \otimes Y^\top,

    if ``sys=1``, for all :math:`X,Y\in\mathbb{C}^{n\times n}`.

    Parameters
    ----------
    mat : :class:`~numpy.ndarray`
        Array of size ``(n0*n1*...*nk-1, n0*n1*...*nk-1)`` represnting a
        matrix defined on :math:`k` subsystems which we want to take the
        partial transpose of.
    dims : :obj:`tuple` of :obj:`int`
        The dimensions ``(n0, n1, ..., nk-1)`` of the :math:`k` subsystems.
    sys : :obj:`int`
        Which of the :math:`k` subsystems to transpose.

    Returns
    -------
    :class:`~numpy.ndarray`
        The resulting ``(n0*n1*...*nk-1, n0*n1*...*nk-1)`` matrix after
        performing the partial transpose.
    """
    N = np.prod(dims)

    temp = mat.reshape(*dims, *dims)
    temp = np.swapaxes(temp, sys, sys + len(dims))

    return temp.reshape(N, N)


def swap(mat, dims, sys1, sys2):
    r"""Swaps two systems of a multipartite quantum state, e.g., for a
    bipartite state with ``dims=(n0, n1)``, it is the unique linear maps
    satisfying

    .. math::

        X \otimes Y \mapsto Y \otimes X,

    for all :math:`X,Y\in\mathbb{H}^{n}`.

    Parameters
    ----------
    mat : :class:`~numpy.ndarray`
        Array of size ``(n0*n1*...*nk-1, n0*n1*...*nk-1)`` represnting a
        matrix defined on :math:`k` subsystems which we want to swap the
        subsystems of.
    dims : :obj:`tuple` of :obj:`int`
        The dimensions ``(n0, n1, ..., nk-1)`` of the :math:`k` subsystems.
    sys1 : :obj:`int`
        First of the :math:`k` subsystems to swap.
    sys2 : :obj:`int`
        Second of the :math:`k` subsystems to swap

    Returns
    -------
    :class:`~numpy.ndarray`
        The resulting ``(n0*n1*...*nk-1, n0*n1*...*nk-1)`` matrix after
        performing the swap operator.
    """
    N = np.prod(dims)

    temp = mat.reshape(*dims, *dims)
    temp = np.swapaxes(temp, sys1, sys2)
    temp = np.swapaxes(temp, len(dims) + sys1, len(dims) + sys2)

    return temp.reshape(N, N)
