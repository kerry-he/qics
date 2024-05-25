import numpy as np
import scipy as sp
from utils import linear as lin

def scale_axis(A, scale_rows=None, scale_cols=None):
    if sp.sparse.issparse(A):
        A_coo = A.tocoo()
        if scale_rows is not None:
            A_coo.data *= np.take(scale_rows, A_coo.row)
        if scale_cols is not None:
            A_coo.data *= np.take(scale_cols, A_coo.col)
        return A_coo.tocsr()
    else:
        if scale_rows is not None:
            A *= scale_rows.reshape((-1, 1))
        if scale_cols is not None:
            A *= scale_cols.reshape(( 1,-1))
        return A

def abs_max(A, axis):
    if sp.sparse.issparse(A):
        A = A.copy()
        A.data = np.abs(A.data)
        return A.max(axis=axis).toarray().reshape(-1)
    else:
        return np.maximum(A.max(axis=axis), -A.min(axis=axis))