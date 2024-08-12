import numpy as np

def p_tr(mat, dims, sys):
    """Performs the partial trace on a bipartite matrix, e.g., for  
    a bipartite state, this is the unique linear map satisfying
    
    .. math::

        X \otimes Y \\mapsto \\text{tr}[X] Y,

    if ``sys=0``, or

    .. math::

        X \otimes Y \\mapsto \\text{tr}[Y] X,

    if ``sys=1``.

    Parameters
    ----------
    mat : ndarray
        Input ``(n0*n1*...*nk-1, n0*n1*...*nk-1)`` matrix to perform the partial trace on.
    dims : tuple[int]
        The dimensions ``(n0, n1, ..., nk-1)`` of the ``k`` subsystems.
    sys : int
        Which of the ``k`` subsystems to trace out.
        
    Returns
    -------
    ndarray
        The resulting matrix after taking the partial trace. Has 
        dimension ``(n0*n1*...*nk-1 / ni, n0*n1*...*nk-1 / ni)`` where 
        ``i`` is the system being traced out.
    """

    N = np.prod(dims) // dims[sys]
    return np.trace(mat.reshape(*dims, *dims), axis1=sys, axis2=len(dims)+sys).reshape(N, N)

def p_tr_multi(out, mat, dims, sys):
    """Performs the partial trace on a list of bipartite matrices.

    Parameters
    ----------
    out : ndarray
        Preallocated list of matrices to store the output. Has 
        dimension ``(p, n0*n1*...*nk-1 / ni, n0*n1*...*nk-1 / ni)`` where 
        ``i`` is the system being traced out.
    mat : ndarray
        Input ``(p, n0*n1*...*nk-1, n0*n1*...*nk-1)`` list of matrices 
        to perform the partial trace on.
    dims : tuple[int]
        The dimensions ``(n0, n1, ..., nk)`` of the ``p`` subsystems.
    sys : int
        Which system to trace out.
    """
    new_dims = [dim for (i, dim) in enumerate(dims) if i != sys]
    np.trace(mat.reshape(-1, *dims, *dims), axis1=1+sys, axis2=1+len(dims)+sys, out=out.reshape(-1, *new_dims, *new_dims))

    return out

def i_kr(mat, dims, sys):
    """Performs Kronecker product between the indentity matrix and 
    a given matrix, e.g., for a bipartite system
    
    .. math::

        X \\mapsto \mathbb{I} \otimes X,

    if ``sys=0``, or

    .. math::

        X \\mapsto X \\otimes \mathbb{I},

    if ``sys=1``.

    Parameters
    ----------
    mat : ndarray
        Input matrix to perform the partial trace on. Has 
        dimension ``(n0*n1*...*nk-1 / ni, n0*n1*...*nk-1 / ni)`` where 
        ``i`` is the system being traced out.
    dim : tuple[int]
        The dimensions ``(n0, n1, ..., nk)`` of the subsystems.
    sys : int
        Which system to Kroneker product should act on.
        
    Returns
    -------
    ndarray
        The resulting ``(n0*n1*...*nk-1, n0*n1*...*nk-1)`` matrix after performing the 
        Kronecker product.
    """
    N = np.prod(dims)
    new_dims = [dim for (i, dim) in enumerate(dims) if i != sys]

    # To reorder systems to shift sys to the front
    swap_idxs = list(range(2 * len(dims)))      
    swap_idxs.insert(0, swap_idxs.pop(sys))
    swap_idxs.insert(1, swap_idxs.pop(len(dims) + sys))

    out = np.zeros((N, N), dtype=mat.dtype)
    out_view = out.reshape(*dims, *dims)
    out_view = np.transpose(out_view, swap_idxs)

    r = np.arange(dims[sys])
    out_view[r, r, ...] = mat.reshape(*new_dims, *new_dims)  
    return out

def i_kr_multi(out, mat, dims, sys):
    """Performs Kronecker product between the indentity matrix and 
    a given list of matrices.

    Parameters
    ----------
    out : ndarray
        Preallocated ``(p, n0*n1*...*nk-1, n0*n1*...*nk-1)`` list of matrices 
        to store the output.    
    mat : ndarray
        Input matrix to perform the partial trace on. Has 
        dimension ``(p, n0*n1*...*nk-1 / ni, n0*n1*...*nk-1 / ni)`` where 
        i is the system being traced out.
    dim : tuple[int]
        The dimensions ``(n0, n1, ..., nk)`` of the subsystems.
    sys : int
        Which system to Kroneker product should act on.
    """
    new_dims = [dim for (i, dim) in enumerate(dims) if i != sys]

    # To reorder systems to shift sys to the front
    swap_idxs = list(range(1 + 2 * len(dims)))      
    swap_idxs.insert(1, swap_idxs.pop(1 + sys))
    swap_idxs.insert(2, swap_idxs.pop(1 + len(dims) + sys))

    out.fill(0.)
    out_view = out.reshape(-1, *dims, *dims)
    out_view = np.transpose(out_view, swap_idxs)

    r = np.arange(dims[sys])
    out_view[:, r, r, ...] = mat.reshape(-1, 1, *new_dims, *new_dims)

    return out

def partial_transpose(mat, dims, sys):
    """Performs the partial transpose on a multipartite matrix, e.g.,
    for a bipartite state, the unique linear map satisfying

    .. math::

        X \otimes Y \\mapsto X^\\top \\otimes Y,

    if ``sys=0``, or

    .. math::

        X \otimes Y \\mapsto X \\otimes Y^\\top,

    if ``sys=1``.

    Parameters
    ----------
    mat : ndarray
        Input ``(n0*n1*...*nk-1, n0*n1*...*nk-1)`` matrix to perform 
        the partial transpose on.
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
    """Swaps two systems of a multipartite quantum state, e.g., for a
    bipartite state, it is the unique linear maps satisfying

    .. math::

        X \otimes Y \mapsto Y \otimes X.

    Parameters
    ----------
    mat : ndarray
        Input ``(n0*n1*...*nk-1, n0*n1*...*nk-1)`` matrix to perform 
        the swap operator on.
    dim : tuple[int]
        The dimensions ``(n0, n1, ..., nk-1)`` of the ``k`` subsystems.
    sys1 : int
        First of the ``k`` subsystems to swap.
    sys2 : int
        Second of the ``k`` subsystems to swap
        
    Returns
    -------
    ndarray
        The resulting ``(n0*n1*...*nk-1, n0*n1*...*nk-1)`` matrix after 
        performing the swap operator.
    """
    N = np.prod(dims)

    temp = mat.reshape(*dims, *dims)
    temp = np.swapaxes(temp, sys1, sys2)
    temp = np.swapaxes(temp, len(dims) + sys1, len(dims) + sys2)

    return temp.reshape(N, N)