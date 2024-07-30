import math
import numpy as np
import numba as nb

def vec_dim(side, iscomplex=False, compact=True):
    """Computes the size of a vectorized matrix.

    Parameters
    ----------
    side : int
        The dimension of the matrix.
    iscomplex : bool, optional
        Whether the matrix is Hermitian (True) or symmetric (False). Default is False.
    compact : bool, optional
        Whether to assume a compact vector representation or not. Default is False.
        
    Returns
    -------
    ndarray
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

def mat_dim(len, iscomplex=False, compact=True):
    """Computes the dimension of the matrix correpsonding to a vector.

    Parameters
    ----------
    len : int
        The dimension of the vector.
    iscomplex : bool, optional
        Whether the matrix is Hermitian (True) or symmetric (False). Default is False.
    compact : bool, optional
        Whether to assume a compact vector representation or not. Default is False.

    Returns
    -------
    ndarray
        The dimension of the matrix.
    """    
    if compact:
        if iscomplex:
            return math.isqrt(len)
        else:
            return math.isqrt(1 + 8 * len) // 2
    else:
        if iscomplex:
            return math.isqrt(len) // 2
        else:
            return math.isqrt(len)

def mat_to_vec(mat, iscomplex=False, compact=True):
    """Reshapes a square matrix into a 1D vector, e.g., the symmetric matrix 
    
        [ a  b  d ]
        [ b  c  e ]
        [ d  e  f ]
        
    is vectorized as the real 1D vector
    
        [a, rt2*b, c, rt2*d, rt2*e, f]    if    compact = False
        [a, b, d, b, c, e, d, e, f]       if    compact = True        
    
    and the Hermitian matrix
    
        [ a     b+cj  e+fj ]
        [ b-cj  d     g+hj ]
        [ e-fj  g-hj  i    ]
        
    is vectorized as the real 1D vector
    
        [a, rt2*b, rt2*c, d, rt2*e, rt2*f, rt2*g, rt2*h, i]          if    compact = False
        [a, 0, b, c, e, f, b, -c, d, 0, g, h, e, -f, g, -h, i, 0]    if    compact = True

    Parameters
    ----------
    mat : ndarray
        Input matrix to vectorize.
    iscomplex : bool, optional
        Whether the matrix to vectorize is Hermitian (True) or symmetric (False). Default is False.
    compact : bool, optional
        Whether to convert to a compact vector representation or not. Default is True.
        
    Returns
    -------
    ndarray
        The resulting vectorized matrix.
    """
    if compact:
        rt2 = np.sqrt(2.0)
        n   = mat.shape[0]
        vn  = vec_dim(n, iscomplex=iscomplex)
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
        return mat.view(dtype=np.float64).reshape(-1, 1).copy()

def vec_to_mat(vec, iscomplex=False, compact=False):
    """Reshapes a 1D vector into a symmetric or Hermitian matrix, e.g., the vectors 
    
        [a, rt2*b, c, rt2*d, rt2*e, f]    if    compact = False
        [a, b, d, b, c, e, d, e, f]       if    compact = True   
        
    are reshaped into the real symmetric matrix     
    
        [ a  b  d ]
        [ b  c  e ]
        [ d  e  f ]
        
    and the vectors
    
        [a, rt2*b, rt2*c, d, rt2*e, rt2*f, rt2*g, rt2*h, i]          if    compact = False
        [a, 0, b, c, e, f, b, -c, d, 0, g, h, e, -f, g, -h, i, 0]    if    compact = True
    
    are reshaped into the complex Hermitian matrix
    
        [ a     b+cj  e+fj ]
        [ b-cj  d     g+hj ]
        [ e-fj  g-hj  i    ]

    Parameters
    ----------
    mat : ndarray
        Input vector to reshape into a matrix.
    iscomplex : bool, optional
        Whether the resulting matrix is Hermitian (True) or symmetric (False). Default is False.
    compact : bool, optional
        Whether to convert from a compact vector representation or not. Default is False.
        
    Returns
    -------
    ndarray
        The resulting matrix.
    """
    vn = vec.size
    
    if compact:
        irt2 = np.sqrt(0.5)
        n    = mat_dim(vn, iscomplex=iscomplex)
        mat  = np.empty((n, n))

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
            n = math.isqrt(vn) // 2
            mat = vec.reshape((-1, 2)).view(dtype=np.complex128).reshape(n, n)
            return (mat + mat.conj().T) * 0.5
        else:
            n = math.isqrt(vn)
            mat = vec.reshape((n, n))
            return (mat + mat.T) * 0.5

def p_tr(mat, sys, dims):
    """Performs the partial trace on a bipartite matrix, e.g., for  
    a bipartite state, this is the unique linear map satisfying
    
        X ⊗ Y --> tr[X] Y    if    sys = 0
        X ⊗ Y --> tr[Y] X    if    sys = 1

    Parameters
    ----------
    mat : ndarray
        Input (n0*n1*...*nk-1, n0*n1*...*nk-1) matrix to perform the partial trace on.
    sys : int
        Which system to trace out.
    dims : tuple[int]
        The dimensions (n0, n1, ..., nk-1) of the k subsystems.
        
    Returns
    -------
    ndarray
        The resulting matrix after taking the partial trace. Has 
        dimension (n0*n1*...*nk-1 / ni, n0*n1*...*nk-1 / ni) where 
        i is the system being traced out.
    """

    N = np.prod(dims) // dims[sys]
    return np.trace(mat.reshape(*dims, *dims), axis1=sys, axis2=len(dims)+sys).reshape(N, N)

def p_tr_multi(out, mat, sys, dims):
    """Performs the partial trace on a list of bipartite matrix, e.g., for  
    a bipartite state, this is the unique linear map satisfying
    
        X ⊗ Y --> tr[X] Y    if    sys = 0
        X ⊗ Y --> tr[Y] X    if    sys = 1

    Parameters
    ----------
    out : ndarray
        Preallocated list of matrices to store the output. Has 
        dimension (p, n0*n1*...*nk-1 / ni, n0*n1*...*nk-1 / ni) where 
        i is the system being traced out.
    mat : ndarray
        Input (p, n0*n1*...*nk-1, n0*n1*...*nk-1) list of matrices 
        to perform the partial trace on.
    sys : int
        Which system to trace out.
    dims : tuple[int]
        The dimensions (n0, n1, ..., nk) of the p subsystems.
    """
    new_dims = [dim for (i, dim) in enumerate(dims) if i != sys]
    np.trace(mat.reshape(-1, *dims, *dims), axis1=1+sys, axis2=1+len(dims)+sys, out=out.reshape(-1, *new_dims, *new_dims))

    return out

def i_kr(mat, sys, dims):
    """Performs Kronecker product between the indentity matrix and 
    a given matrix, e.g., for a bipartite system
    
        X --> I ⊗ X    if    sys = 0
        X --> X ⊗ I    if    sys = 1

    Parameters
    ----------
    mat : ndarray
        Input matrix to perform the partial trace on. Has 
        dimension (n0*n1*...*nk-1 / ni, n0*n1*...*nk-1 / ni) where 
        i is the system being traced out.
    sys : int
        Which system to Kroneker product should act on.
    dim : tuple[int]
        The dimensions (n0, n1, ..., nk) of the subsystems.
        
    Returns
    -------
    ndarray
        The resulting (n0*n1*...*nk-1, n0*n1*...*nk-1) matrix after performing the 
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

def i_kr_multi(out, mat, sys, dims):
    """Performs Kronecker product between the indentity matrix and 
    a given list of matrices, i.e.,
    
        X --> I ⊗ X    if    sys = 0
        X --> X ⊗ I    if    sys = 1

    Parameters
    ----------
    mat : ndarray
        Input matrix to perform the partial trace on. Has 
        dimension (p, n0*n1*...*nk-1 / ni, n0*n1*...*nk-1 / ni) where 
        i is the system being traced out.
    sys : int
        Which system to Kroneker product should act on.
    dim : tuple[int]
        The dimensions (n0, n1, ..., nk) of the subsystems.
        
    Returns
    -------
    ndarray
        The resulting (p, n0*n1*...*nk-1, n0*n1*...*nk-1) matrix after performing the 
        Kronecker product.
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

def p_transpose(mat, sys, dim):
    """Performs the partial transpose on a bipartite matrix, i.e., 
    the unique linear map satisfying
    
        M_ij,kl --> M_kj,il    if    sys == 0
        M_ij,kl --> M_il,kj    if    sys == 1

    Parameters
    ----------
    mat : ndarray
        Input (n0*n1, n0*n1) matrix to perform the partial transpose on.
    sys : int
        Which system to transpose (either 0 or 1).
    dim : tuple[int, int]
        The dimensions (n0, n1) of the first and second subsystems.
        
    Returns
    -------
    ndarray
        The resulting (n0*n1, n0*n1) matrix after performing the partial transpose.
    """    
    (n0, n1) = dim
    assert sys == 0 or sys == 1

    temp = mat.reshape(n0, n1, n0, n1)

    if sys == 0:
        temp = temp.transpose(2, 1, 0, 3)
    elif sys == 1:
        temp = temp.transpose(0, 3, 2, 1)

    return temp.reshape(n0*n1, n0*n1)


def lin_to_mat(lin, dims, iscomplex=False, compact=(False, True)):
    """Computes the matrix corresponding to a linear map from
    vectorized symmetric matrices to symmetric matrices.

    Parameters
    ----------
    lin : callable
        Linear operator sending symmetric matrices to symmetric matrices.
    dims : tuple[int, int]
        The dimensions (ni, no) of the input and output matrices of the 
        linear operator.
    iscomplex : bool, optional
        Whether the matrix to vectorize is Hermitian (True) or symmetric 
        (False). Default is False.
    compact : tuple[bool, bool], optional
        Whether to use a compact vector representation or not for the input 
        and output matrices. Default is (False, True).
        
    Returns
    -------
    ndarray
        The matrix representation of lin.
    """
    vni = vec_dim(dims[0], iscomplex=iscomplex, compact=compact[0])
    vno = vec_dim(dims[1], iscomplex=iscomplex, compact=compact[1])
    mat = np.zeros((vno, vni))

    for k in range(vni):
        H = np.zeros((vni, 1))
        H[k] = 1.0
        H_mat = vec_to_mat(H, iscomplex=iscomplex, compact=compact[0])
        lin_H = lin(H_mat)
        vec_out = mat_to_vec(lin_H, iscomplex=iscomplex, compact=compact[1])
        mat[:, [k]] = vec_out

    return mat