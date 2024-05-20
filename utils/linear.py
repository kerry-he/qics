import numpy as np
import scipy as sp
import sksparse.cholmod as cholmod

from utils import symmetric as sym
# import symmetric as sym


class Vector():
    def __init__(self, cones, vec=None):
        self.dims  = []
        self.types = []
        for cone_k in cones:
            self.dims.append(cone_k.dim)
            self.types.extend(cone_k.type)
        self.dim = sum(self.dims)

        # Initialize vector
        if vec is None:
            self.vec = np.zeros((self.dim, 1))
        else:
            assert vec.size == self.dim
            self.vec = vec

        # Build views of vector
        self.vec_views = []
        self.mat = []
        t = 0
        for (dim_k, type_k) in zip(self.dims, self.types):
            self.vec_views.append(self.vec[t:t+dim_k])

            if type_k == 'r':
                # Real vector
                self.mat.append(self.vec_views[-1])
            elif type_k == 's':
                # Symmetric matrix
                n_k = int(np.sqrt(dim_k))
                self.mat.append(self.vec_views[-1].reshape((n_k, n_k)))
            elif type_k == 'h':
                # Hermitian matrix
                n_k = int(np.sqrt(dim_k // 2))
                self.mat.append(self.vec_views[-1].reshape((-1, 2)).view(dtype=np.complex128).reshape(n_k, n_k))
            t += dim_k

    def __getitem__(self, key):
        return self.mat[key]
    
    def __iadd__(self, other):
        self.vec = sp.linalg.blas.daxpy(other.vec, self.vec, a=1)
        return self
    
    def __isub__(self, other):
        self.vec = sp.linalg.blas.daxpy(other.vec, self.vec, a=-1)
        return self    
    
    def __imul__(self, other):
        self.vec *= other
        return self
    
    def get_vn(self):
        return self.dim

    def inp(self, other):
        return (self.vec.T @ other.vec)[0, 0]
    
    def norm(self, order=None):
        return np.linalg.norm(self.vec, ord=order)
    
    def copy_from(self, other):
        if isinstance(other, np.ndarray):
            np.copyto(self.vec, other)
        else:
            np.copyto(self.vec, other.vec)
        return self 
    
    def axpy(self, a, other):
        self.vec = sp.linalg.blas.daxpy(other.vec, self.vec, a=a)
        return self
    
    def fill(self, a):
        self.vec.fill(a)
        return self

def inp(x, y):
    # Standard inner product
    if isinstance(x, list) and isinstance(y, list):
        return sum([np.sum(xi * yi.conj()).real for (xi, yi) in zip(x, y)])
    else:
        return np.sum(x * y.conj()).real

def norm(x, ord=None):
    # Standard inner product
    if isinstance(x, list):
        return np.linalg.norm(np.array([np.linalg.norm(xi, ord) for xi in x]), ord)
    else:
        return np.linalg.norm(x, ord)
    
def add(x, y):
    # Standard inner product
    if isinstance(x, list):
        return [xi + yi for (xi, yi) in zip(x, y)]
    else:
        return x + y

# @profile
def fact(A, fact=None):
    # Perform a Cholesky decomposition, or an LU factorization if Cholesky fails
    if sp.sparse.issparse(A):
        while True:
            try:
                if fact is None:
                    fact = cholmod.cholesky(A)
                    return (fact, "spcho")
                else:
                    fact[0].cholesky_inplace(A)
                    return (fact[0], "spcho")
            except cholmod.CholmodNotPositiveDefiniteError:
                A_diag = A.diagonal()
                A.setdiag(np.max([A_diag, np.ones_like(A_diag) * 1e-12], axis=0) * (1 + 1e-8))
        
    
    while True:
        try:
            fact = sp.linalg.cho_factor(A, check_finite=False)
            return (fact, "cho")
        except np.linalg.LinAlgError:
            diag_idx = np.diag_indices_from(A)
            A[diag_idx] = np.max([A[diag_idx], np.ones_like(A[diag_idx]) * 1e-12], axis=0) * (1 + 1e-8)

    # try:
    #     fact = sp.linalg.cho_factor(A)
    #     return (fact, "cho")
    # except np.linalg.LinAlgError:
    #     fact = sp.linalg.lu_factor(A)
    #     return (fact, "lu")

def fact_solve(A, x):
    # Factor solve for either Cholesky or LU factorization of A
    (fact, type) = A

    if type == "cho":
        return sp.linalg.cho_solve(fact, x, check_finite=False)
    elif type == "lu":
        return sp.linalg.lu_solve(fact, x)
    elif type == "spcho":
        return fact.solve_A(x)

def pcg(A, b, M, tol=1e-8, max_iter=20):

    # Use to compute relative tolerance
    b_max = max(np.max(np.abs(b)), 1.0)

    x_k = M(b)
    r_k = b - A(x_k)

    abs_res = np.linalg.norm(r_k)
    rel_res = abs_res / b_max    
    if abs_res < tol or rel_res < tol:
        return x_k, 0, abs_res

    z_k = M(r_k)
    p_k = z_k
    r_z_k = inp(r_k, z_k)

    for k in range(max_iter):
        A_p_k = A(p_k)
        alpha_k = r_z_k / inp(p_k, A_p_k)
        x_k1 = x_k + alpha_k * p_k
        r_k1 = r_k - alpha_k * A_p_k

        # Check if solved to desired tolerance
        abs_res = np.linalg.norm(r_k1)
        rel_res = abs_res / b_max

        if abs_res < tol or rel_res < tol:
            break

        z_k1 = M(r_k1)
        r_z_k1 = inp(r_k1, z_k1)
        beta_k = r_z_k1 / r_z_k
        p_k1 = z_k1 + beta_k * p_k

        # Step forwards
        x_k = x_k1
        r_k = r_k1
        z_k = z_k1
        p_k = p_k1
        r_z_k = r_z_k1

    return x_k1, (k + 1), abs_res