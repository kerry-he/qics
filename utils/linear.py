import numpy as np
import scipy as sp
import sksparse.cholmod as cholmod

from utils import symmetric as sym
# import symmetric as sym

class Vector():
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value

    def __add__(self, other):
        return Vector([x + y for (x, y) in zip(self.data, other.data)])
    
    def __iadd__(self, other):
        for (x, y) in zip(self.data, other.data):
            x += y
        return self
    
    def __sub__(self, other):
        return Vector([x - y for (x, y) in zip(self.data, other.data)])    
    
    def __isub__(self, other):
        for (x, y) in zip(self.data, other.data):
            x -= y
        return self    
    
    def __mul__(self, other):
        if np.isscalar(other):
            return Vector([x * other for x in self.data])
        else:
            return Vector([x * y for (x, y) in zip(self.data, other.data)])
    
    def __imul__(self, other):
        if np.isscalar(other):
            for x in self.data:
                x *= other
        else:
            for (x, y) in zip(self.data, other.data):
                x *= y
        return self
    
    __rmul__ = __mul__
    
    def get_vn(self):
        return sum([x.get_vn() for x in self.data])

    def inp(self, other):
        return np.sum([x.inp(y) for (x, y) in zip(self.data, other.data)])
    
    def norm(self, order=None):
        return np.linalg.norm([x.norm(order) for x in self.data])

    def to_vec(self):
        return np.vstack([x.to_vec() for x in self.data])
    
    def from_vec(self, vec):
        i_from = 0
        for x in self.data:
            i_to = i_from + x.get_vn()
            x.from_vec(vec[i_from:i_to])
            i_from = i_to
        return self 
    
    def zeros_like(self):
        return self * 0
    
    def axpy(self, a, other):
        for (xi, yi) in zip(other, self):
            yi.axpy(a, xi)
            
    def copy(self, other):
        for (xi, yi) in zip(other, self):
            yi.copy(xi)
            
    def to_sparse(self):
        for x in self.data:
            x.to_sparse()
        return self
    
    def fill(self, a):
        for x in self.data:
            x.fill(a)
        return self
        

class Real(Vector):
    def __init__(self, data):
        if np.isscalar(data):
            self.data = np.zeros((data, 1))
            self.n = data
            self.vn = data
        else:
            assert data.size == data.shape[0]
            self.data = data
            self.n = data.size
            self.vn = self.n

    def __add__(self, other):
        return Real(self.data + other.data)

    def __iadd__(self, other):
        self.data += other.data
        return self

    def __sub__(self, other):
        return Real(self.data - other.data)
    
    def __isub__(self, other):
        self.data -= other.data
        return self    

    def __mul__(self, other):
        if np.isscalar(other):
            return Real(self.data * other)
        else:
            return Real(self.data * other.data)
        
    def __imul__(self, other):
        self.data *= other
        return self        
            
    __rmul__ = __mul__
    
    def get_vn(self):
        return self.vn    

    def __truediv__(self, other):
        return Real(self.data / other)

    def inp(self, other):
        return np.sum(self.data * other.data)

    def norm(self, order=None):
        return np.linalg.norm(self.data, order)

    def to_vec(self):
        return self.data
    
    def from_vec(self, vec):
        np.copyto(self.data, vec)
        return self
    
    def to_sparse(self):
        self.data = sp.sparse.csr_matrix(self.data)
        return self    
    
    def axpy(self, a, other):
        self.data = sp.linalg.blas.daxpy(other.data, self.data, a=a)
        # self.data += a * other.data
        
    def copy(self, other):
        np.copyto(self.data, other.data)
        
    def fill(self, a):
        self.data.fill(a)

class Symmetric(Vector):
    def __init__(self, data):
        if np.isscalar(data):
            self.data = np.zeros((data, data))
            self.n  = data
            self.vn = self.n * self.n
        else:
            assert data.shape[0] == data.shape[1]
            self.data = (data + data.T) / 2
            self.n  = data.shape[0]
            self.vn = self.n * self.n

    def __add__(self, other):
        return Symmetric(self.data + other.data)

    def __iadd__(self, other):
        self.data += other.data
        return self

    def __sub__(self, other):
        return Symmetric(self.data - other.data)

    def __isub__(self, other):
        self.data -= other.data
        return self   

    def __mul__(self, other):
        if np.isscalar(other):
            return Symmetric(self.data * other)
        else:
            return Symmetric(self.data * other.data)
        
    def __imul__(self, other):
        self.data *= other
        return self          
    
    __rmul__ = __mul__

    def get_vn(self):
        return self.vn

    def __truediv__(self, other):
        return Symmetric(self.data / other)    

    def inp(self, other):
        return np.sum(self.data * other.data)   

    def norm(self, order=None):
        return np.linalg.norm(self.data, order)
    
    def to_vec(self):
        return self.data.reshape((-1, 1)).copy()
        return sym.mat_to_vec(self.data, hermitian=False)
    
    def from_vec(self, vec):
        self.data = vec.reshape((self.n, self.n)).copy()
        # self.data = sym.vec_to_mat(vec, hermitian=False)
        return self
    
    def to_sparse(self):
        self.data = sp.sparse.csr_matrix(self.data)
        return self    
    
    def axpy(self, a, other):
        self.data = sp.linalg.blas.daxpy(other.data.ravel(), self.data.ravel(), a=a).reshape((self.n, self.n))
        # self.data += a * other.data
        
    def copy(self, other):
        np.copyto(self.data, other.data)
        
    def fill(self, a):
        self.data.fill(a)        
        
    
class Hermitian(Vector):
    def __init__(self, data):
        if np.isscalar(data):
            self.data = np.zeros((data, data), dtype=np.complex128)
            self.n  = data
            self.vn = self.n * self.n
        else:
            assert data.shape[0] == data.shape[1]
            assert np.issubdtype(data.dtype, complex)
            self.data = (data + data.conj.T()) / 2
            self.n  = data.shape[0]
            self.vn = self.n * self.n

    def __add__(self, other):
        return Hermitian(self.data + other.data)

    def __iadd__(self, other):
        self.data += other.data
        return self

    def __sub__(self, other):
        return Hermitian(self.data - other.data)

    def __isub__(self, other):
        self.data -= other.data
        return self   

    def __mul__(self, other):
        if np.isscalar(other):
            return Hermitian(self.data * other)
        else:
            return Hermitian(self.data * other.data)
        
    def __imul__(self, other):
        self.data *= other
        return self          
    
    __rmul__ = __mul__

    def get_vn(self):
        return self.vn

    def __truediv__(self, other):
        return Hermitian(self.data / other)        

    def inp(self, other):
        return np.sum(self.data * other.data.conj()).real

    def norm(self, order=None):
        return np.linalg.norm(self.data, order)

    def to_vec(self):
        return sym.mat_to_vec(self.data, hermitian=True)
    
    def from_vec(self, vec):
        self.data = sym.vec_to_mat(vec, hermitian=True)
        return self

    def to_sparse(self):
        self.data = sp.sparse.csr_matrix(self.data)
        return self

    def axpy(self, a, other):
        self.data += a * other.data
        
    def copy(self, other):
        np.copyto(self.data, other.data)
        
    def fill(self, a):
        self.data.fill(a)        
        

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
        if fact is None:
            fact = cholmod.cholesky(A)
        else:
            fact = fact[0]
            fact.cholesky_inplace(A)
        return (fact, "spcho")
    
    while True:
        try:
            fact = sp.linalg.cho_factor(A, check_finite=False)
            return (fact, "cho")
        except np.linalg.LinAlgError:
            diag_idx = np.diag_indices_from(A)
            A[diag_idx] = np.max([A[diag_idx], np.ones_like(A[diag_idx]) * 1e-12], axis=0) * (1 + 1e-8)
        else:
            break

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