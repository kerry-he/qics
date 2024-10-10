# Copyright (c) 2024, Kerry He, James Saunderson, and Hamza Fawzi

# This Python package QICS is licensed under the MIT license; see LICENSE.md
# file in the root directory or at https://github.com/kerry-he/qics

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
    r"""A class for a vector containing the variables involved in a
    homogeneous self-dual embedding of a primal-dual conic program
    :math:`(x, y, z, s, \tau, \kappa)\in\mathbb{R}^n\times\mathbb{R}^p
    \times\mathbb{R}^q\times\mathbb{R}^q\times\mathbb{R}\times\mathbb{R}`.

    Parameters
    ----------
    model : :class:`~qics.Model`
        Model which specifies the conic program which this vector
        corresponds to.

    Attributes
    ----------
    vec : :class:`~numpy.ndarray`
        2D :obj:`~numpy.float64` array of size ``(n+p+2q+2, 1)``
        representing the full concatenated vector :math:`(x, y, z, s, \tau,
        \kappa)`.
    x : :class:`~numpy.ndarray`
        A :obj:`~numpy.ndarray.view` of ``vec`` of size ``(n, 1)``
        correpsonding to the primal variable :math:`x`.
    y : :class:`~numpy.ndarray`
        A :obj:`~numpy.ndarray.view` of ``vec`` of size ``(p, 1)``
        correpsonding to the dual variable :math:`y`.
    z : :class:`~qics.point.VecProduct`
        A :obj:`~numpy.ndarray.view` of ``vec`` of size ``(q, 1)``
        correpsonding to the dual variable :math:`z`.
    s : :class:`~qics.point.VecProduct`
        A :obj:`~numpy.ndarray.view` of ``vec`` of size ``(q, 1)``
        correpsonding to the primal variable :math:`s`.
    tau : :class:`~numpy.ndarray`
        A :obj:`~numpy.ndarray.view` of ``vec`` of size ``(1, 1)``
        correpsonding to the primal homogenizing variable :math:`\tau`.
    kap : :class:`~numpy.ndarray`
        A :obj:`~numpy.ndarray.view` of ``vec`` of size ``(1, 1)``
        correpsonding to the dual homogenizing variable :math:`\kappa`.
    """

    def __init__(self, model):
        (n, p, q) = (model.n, model.p, model.q)

        # Initialize vector
        self.vec = np.zeros((n + p + q + q + 2, 1))

        # Build views of vector
        self._xyz = PointXYZ(model, self.vec[: n + p + q])

        self.x = self._xyz.x
        self.y = self._xyz.y
        self.z = self._xyz.z
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
    r"""A class for a Cartesian product of vectors corresponding to a list
    of cones :math:`\mathcal{K}_i`, i.e., :math:`s\in\mathbb{V}` where

    .. math::

        \mathbb{V} = \mathbb{V}_1 \times \mathbb{V}_2 \times \ldots \times
        \mathbb{V}_k,
    
    and :math:`\mathcal{K}_i \subset \mathbb{V}_i`. Each of these vector
    spaces :math:`\mathbb{V}_i` are themselves a Cartesian product of
    vector spaces

    .. math::

        \mathbb{V}_i = \mathbb{V}_{i,1} \times \mathbb{V}_{i,2} \times
         \ldots \times \mathbb{V}_{i,k_i},
    
    where :math:`\mathbb{V}_{i,j}` are defined as either the set of real 
    vectors :math:`\mathbb{R}^n`, symmetric matrices :math:`\mathbb{S}^n`,
    or Hermitian matrices :math:`\mathbb{H}^n`.

    Parameters
    ----------
    cones : :obj:`list` of :mod:`~qics.cones`
        List of cones defining a Cartesian product of vector spaces.
    vec : :class:`~numpy.ndarray`, optional
        If specified, then this class is initialized as a
        :obj:`~numpy.ndarray.view` of ``vec``. Otherwise, the class is
        initialized using a newly allocated :class:`~numpy.ndarray`.

    Attributes
    ----------
    vec : :class:`~numpy.ndarray`
        2D :obj:`~numpy.float64` array of size ``(q, 1)`` representing the
        full concatenated Cartesian product of vectors.
    mats : :obj:`list` of :obj:`list` of :class:`~numpy.ndarray`
        A nested list of :obj:`~numpy.ndarray.view` of ``vec`` where
        ``mats[i][j]`` returns the array corresponding to the vector space
        :math:`\mathbb{V}_{i,j}`. This attribute can also be called using
        ``__getitem__``, i.e., by directly calling ``self[i][j]``.
    vecs : :obj:`list` of :class:`~numpy.ndarray`
        A list of :obj:`~numpy.ndarray.view` of ``vec`` where ``vecs[i]``
        returns the array corresponding to the vector space
        :math:`\mathbb{V}_{i}` as a vectorized column vector.

    Examples
    --------
    Below we show an example of how to initialize a 
    :class:`~qics.point.VecProduct` and how to access the vectors
    corresponding to each cone and variable.

    >>> import qics
    >>> cones = [                                       \
    ...     qics.cones.PosSemidefinite(3),              \
    ...     qics.cones.QuantRelEntr(2, iscomplex=True)  \
    ... ]
    >>> x = qics.point.VecProduct(cones)
    >>> x[0][0]  # Matrix corresponding to PosSemidefinite cone
    array([[0., 0.],
           [0., 0.]])
    >>> x[1][0]  # Value corresponding to t of QuantRelEntr cone
    array([[0.]])
    >>> x[1][1]  # Matrix corresponding to X of QuantRelEntr cone
    array([[0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j]])
    >>> x[1][2]  # Matrix corresponding to Y of QuantRelEntr cone
    array([[0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j]])
    """

    def __init__(self, cones, vec=None):
        dims = []
        types = []
        dim = 0
        for cone_k in cones:
            dims.append(cone_k.dim)
            types.append(cone_k.type)
            dim += np.sum(cone_k.dim)

        # Initialize vector
        if vec is not None:
            # If view of vector is already given, use that
            assert vec.size == dim
            self.vec = vec
        else:
            # Otherwise allocate a new vector
            self.vec = np.zeros((dim, 1))

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
        for dim_k, type_k in zip(dims, types):
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
