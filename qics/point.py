import numpy as np
import scipy as sp


class Vector:
    """Base class for a vector"""

    def __init__(self):
        self.vec = None

    def __iadd__(self, other):
        if self.vec.size > 0:
            self.vec[:] = sp.linalg.blas.daxpy(other.vec, self.vec, a=1)
        return self

    def __isub__(self, other):
        if self.vec.size > 0:
            self.vec[:] = sp.linalg.blas.daxpy(other.vec, self.vec, a=-1)
        return self

    def __imul__(self, a):
        self.vec *= a
        return self

    def axpy(self, a, other):
        if self.vec.size > 0:
            self.vec[:] = sp.linalg.blas.daxpy(other.vec, self.vec, a=a)
        return self

    def copy_from(self, other):
        if isinstance(other, np.ndarray):
            np.copyto(self.vec, other)
        else:
            np.copyto(self.vec, other.vec)
        return self

    def norm(self):
        return np.sqrt((self.vec.T @ self.vec)[0, 0])

    def inp(self, other):
        return (self.vec.T @ other.vec)[0, 0]

    def fill(self, a):
        self.vec.fill(a)
        return self


class Point(Vector):
    r"""A class for a vector containing the variables involved in a homogeneous self-
    dual embedding of a primal-dual conic program :math:`(x, y, z, s, \tau, \kappa)`.
    
    Parameters
    ----------
    model : :class:`qics.Model`
        Model class which specifies the conic program which this class corresponds to.

    Attributes
    ----------
    vec : numpy.ndarray
        Vector representing the full concatenated point :math:`(x, y, z, s, \tau, 
        \kappa)`.
    x : numpy.ndarray
        View of ``vec`` attribute correpsonding to the primal variable :math:`x`.
    y : numpy.ndarray
        View of ``vec`` attribute correpsonding to the dual variable :math:`y`.
    z : :class:`qics.VecProduct`
        View of ``vec`` attribute correpsonding to the dual variable :math:`z`.
    s : :class:`qics.VecProduct`
        View of ``vec`` attribute correpsonding to the dual variable :math:`s`.
    tau : numpy.ndarray
        View of ``vec`` attribute correpsonding to the dual variable :math:`\tau`.
    kap : numpy.ndarray
        View of ``vec`` attribute correpsonding to the dual variable :math:`\kappa`.                
    
    """

    def __init__(self, model):
        (n, p, q) = (model.n, model.p, model.q)

        # Initialize vector
        self.vec = np.zeros((n + p + q + q + 2, 1))

        # Build views of vector
        self.xyz = PointXYZ(model, self.vec[: n + p + q])
        self.x = self.xyz.x
        self.y = self.xyz.y
        self.z = self.xyz.z
        self.s = VecProduct(model.cones, self.vec[n + p + q : n + p + q + q])
        self.tau = self.vec[n + p + q + q : n + p + q + q + 1]
        self.kap = self.vec[n + p + q + q + 1 : n + p + q + q + 2]

        return


class PointXYZ(Vector):
    """A class for a vector containing only the (x,y,z) variables invovled in a
    primal-dual conic program"""

    def __init__(self, model, vec=None):
        (n, p, q) = (model.n, model.p, model.q)

        # Initialize vector
        if vec is not None:
            # If view of vector is already given, use that
            assert vec.size == n + p + q
            self.vec = vec
        else:
            # Otherwise allocate a new vector
            self.vec = np.zeros((n + p + q, 1))

        # Build views of vector
        self.x = self.vec[:n]
        self.y = self.vec[n : n + p]
        self.z = VecProduct(model.cones, self.vec[n + p : n + p + q])

        return


class VecProduct(Vector):
    """A class for a Cartesian product of vector spaces corresponding
    to a list of cones.

    For ``x = VecProduct(cones)``, we use ``x[i][j]`` to obtain the :math:`j`-th 
    variable of the :math:`i`-th cone. For example, if 
    
    >>> cones = [
    ...     qics.cones.NonNegOrthant(2),
    ...     qics.cones.QuantRelEntr(3),
    ...     qics.cones.PosSemidefinite(4, iscomplex=True)
    ... ]
    >>> x = VecProduct(cones)

    then 

        - ``x[0][0]`` is an (2, 1) real vector
        - ``x[1][0]`` is a (1, 1) real vector
        - ``x[1][1]`` and ``x[1][2]`` are (3, 3) real symmetric matrices
        - ``x[2][0]`` is a (4, 4) complex Hermitian matrix 


    Parameters
    ----------
    cones : list(:class:`qics.cones`)
        List of cones defining a Cartesian product of vector spaces.
    vec : :class:`numpy.ndarray`, optional
        If specified, then this class is built as a view of ``vec``. Otherwise, a new
        NumPy is allocated which this class is built on top of. 

    Attributes
    ----------
    vec : numpy.ndarray
        Raw vector representing the full concatenated Cartesian product of vectors.
    """

    def __init__(self, cones, vec=None):
        self.dims = []
        self.types = []
        self.dim = 0
        for cone_k in cones:
            self.dims.append(cone_k.dim)
            self.types.append(cone_k.type)
            self.dim += np.sum(cone_k.dim)

        # Initialize vector
        if vec is not None:
            # If view of vector is already given, use that
            assert vec.size == self.dim
            self.vec = vec
        else:
            # Otherwise allocate a new vector
            self.vec = np.zeros((self.dim, 1))

        # Build views of vector
        def vec_to_mat(vec, dim, type, t):
            if type == "r":
                # Real vector
                return vec[t : t + dim]
            elif type == "s":
                # Symmetric matrix
                n_k = int(np.sqrt(dim))
                return vec[t : t + dim].reshape((n_k, n_k))
            elif type == "h":
                # Hermitian matrix
                n_k = int(np.sqrt(dim // 2))
                return (
                    vec[t : t + dim]
                    .reshape((-1, 2))
                    .view(dtype=np.complex128)
                    .reshape(n_k, n_k)
                )

        self.vecs = []
        self.mats = []
        t = 0
        for dim_k, type_k in zip(self.dims, self.types):
            self.vecs.append(self.vec[t : t + np.sum(dim_k)])
            if isinstance(type_k, list):
                mats_k = []
                for dim_k_j, type_k_j in zip(dim_k, type_k):
                    mats_k.append(vec_to_mat(self.vec, dim_k_j, type_k_j, t))
                    t += dim_k_j
                self.mats.append(mats_k)
            else:
                self.mats.append(vec_to_mat(self.vec, dim_k, type_k, t))
                t += dim_k

    def __getitem__(self, key):
        return self.mats[key]
