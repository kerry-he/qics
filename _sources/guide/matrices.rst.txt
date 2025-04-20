Representing matrices
=======================

To supply a conic program model to **QICS**, users are reqired to express the
linear constraints :math:`A` and :math:`G` in matrix form. This is
straightforward when variables correspond to real vectors. For variables
corresponding to symmetric or Hermitian matrices, we need to first vectorize 
these matrices. QICS provides functions to perform these operations in the 
:mod:`qics.vectorize` module, which we describe in more detail below.

Symmetric matrices
--------------------

Consider a real symmetric matrix :math:`X \in \mathbb{S}^n`. We can vectorize
the matrices :math:`A_i` by stacking the rows of the matrix side-by-side, i.e.,

.. math::

   \text{vec}(X) = \text{vec}\left(\begin{bmatrix}
         \;\; — & x_{1}^\top & — \;\;  \\
         \;\; — & x_{2}^\top & — \;\;  \\
           & \vdots     &          \\
         \;\; — & x_{n}^\top & — \;\; 
      \end{bmatrix}\right) = \begin{bmatrix}
                                 x_{1} \\
                                 x_{2} \\
                                 \vdots \\
                                 x_{n}
                             \end{bmatrix} \in \mathbb{R}^{n^2}.

Using QICS, this operation can be performed using
:obj:`qics.vectorize.mat_to_vec` on matrices of the type :obj:`numpy.float64`. An
example of this is shown below.

>>> import numpy, qics
>>> X = numpy.array([[0., 1., 2.], [1., 3., 4.], [2., 4., 5.]])
>>> X
array([[0., 1., 2.],
       [1., 3., 4.],
       [2., 4., 5.]])
>>> qics.vectorize.mat_to_vec(X).T
array([[0., 1., 2., 1., 3., 4., 2., 4., 5.]])

This is equivalent to using :obj:`numpy.reshape` to reshape the matrix into a
vector, i.e., ``X.reshape(-1, 1).T``.

Sometimes it is also useful to vectorize matrices in a more compact fashion
which only includes a single copy of elements in the off-diagonals. This is
done in the following fashion

.. math::

   \text{cvec}(X) = \text{cvec}\left(\begin{bmatrix}
      x_{11} & x_{12} & x_{13} & \\
             & x_{22} & x_{23} & \\
             &        & x_{33} & \\
             &        &        & \ddots
   \end{bmatrix}\right) = \begin{bmatrix}
                              x_{11} \\
                              \sqrt{2}x_{12} \\
                              x_{22} \\
                              \sqrt{2}x_{13} \\
                              \sqrt{2}x_{23} \\ 
                              x_{33} \\
                              \vdots 
                           \end{bmatrix} \in \mathbb{R}^{n(n-1)/2}.

Note the scaling of the off-diagonals is to make sure the inner product is
consistent between the two reprsentations. This compact representation is useful
when defining linear constraints to avoid duplicate constraints (see 
:ref:`subsec-superoperator`). This compact vectorization can also be performed
in **QICS** by specifying the optional ``compact`` argument, which is ``False``
by default.

>>> qics.vectorize.mat_to_vec(X, compact=True).T
array([[0.        , 1.41421356, 3.        , 2.82842712, 5.65685425,
        5.        ]])


Hermitian matrices
--------------------

Now instead consider a complex Hermitian matrix :math:`X \in \mathbb{H}^n`. To
vectorize this matrix, we first convert each row :math:`x_i\in\mathbb{C}^n` of 
:math:`X` into a real vector by splitting the real and imaginary components of 
each entry as follows

.. math::

   \text{split}(x_i) = \text{split}\!\left( \begin{bmatrix}
      x_{i1} \\
      x_{i2} \\
      \vdots \\
      x_{in}
   \end{bmatrix} \right) = \begin{bmatrix}
         \text{Re}(x_{i1}) \\
         \text{Im}(x_{i1}) \\
         \text{Re}(x_{i2}) \\
         \text{Im}(x_{i2}) \\
         \vdots            \\
         \text{Re}(x_{in}) \\
         \text{Im}(x_{in}) \\
      \end{bmatrix} \in \mathbb{R}^{2n}.

Using this, Hermitian matrices are vectorized in a similar fashion as symmetric
matrices.

.. math::

   \text{vec}(X) = \text{vec}\!\left(\begin{bmatrix}
      \;\; — & x_{1}^\top & — \;\; \\
      \;\; — & x_{2}^\top & — \;\; \\
        & \vdots     &   \\
      \;\; — & x_{3}^\top & — \;\;
   \end{bmatrix}\right) = \begin{bmatrix}
                              \text{split}(x_{1}) \\
                              \text{split}(x_{2}) \\
                              \vdots \\
                              \text{split}(x_{n})
                           \end{bmatrix} \in \mathbb{R}^{2n^2}.

In practice, we can perform this operation using the same 
:obj:`qics.vectorize.mat_to_vec` function as for the real symmetric case,
except when the argument ``X`` is an array of type ``numpy.complex128``.

>>> X = numpy.array([[0., 1.+1.j, 2.+2.j], [1.-1.j, 3., 4.+4.j], [2.-2.j, 4.-4.j, 5.]])
>>> X
array([[0.+0.j, 1.+1.j, 2.+2.j],
       [1.-1.j, 3.+0.j, 4.+4.j],
       [2.-2.j, 4.-4.j, 5.+0.j]])
>>> qics.vectorize.mat_to_vec(X).T
array([[ 0.,  0.,  1.,  1.,  2.,  2.,  1., -1.,  3.,  0.,  4.,  4.,  2.,
        -2.,  4., -4.,  5.,  0.]])

This is equivalent to taking a :obj:`numpy.float64` view of a 
:obj:`numpy.complex128` array, then using :obj:`numpy.reshape` to reshape the 
matrix into a vector, i.e., ``X.view(numpy.float64).reshape(-1, 1).T``. 

Like the symmetric case, we can also define a compact vectorization for 
Hermitian matrices which only stores a single copy of the real and imaginary
off-diagonal components.

.. math::

   \text{cvec}(X) = \text{cvec}\left(\begin{bmatrix}
      x_{11} & x_{12} & x_{13} & \\
             & x_{22} & x_{23} & \\
             &        & x_{33} & \\
             &        &        & \ddots
   \end{bmatrix}\right) = \begin{bmatrix}
                              x_{11} \\
                              \sqrt{2}\text{Re}(x_{12}) \\
                              \sqrt{2}\text{Im}(x_{12}) \\
                              x_{22} \\
                              \sqrt{2}\text{Re}(x_{13}) \\
                              \sqrt{2}\text{Im}(x_{13}) \\
                              \sqrt{2}\text{Re}(x_{23}) \\ 
                              \sqrt{2}\text{Im}(x_{23}) \\ 
                              x_{33} \\
                              \vdots 
                           \end{bmatrix} \in \mathbb{R}^{n^2},

which can be done in **QICS** as follows 

>>> qics.vectorize.mat_to_vec(X, compact=True).T
array([[0.        , 1.41421356, 1.41421356, 3.        , 2.82842712,
        2.82842712, 5.65685425, 5.65685425, 5.        ]])


Modelling constraints
-------------------------

To see how we use can use these vectorizations to represent linear constraints,
consider the linear constraints

.. math::

   \text{tr}[A_i X] = b_i, \qquad \forall\ i=1,\ldots,p,

where :math:`X \in \mathbb{H}^n` is our matrix variable, and 
:math:`A_i \in \mathbb{H}^n` and :math:`b_i \in \mathbb{R}` encode linear
constraints for :math:`i=1,\ldots,p`. We can represent this constraint as 

.. math::

   A\text{vec}(X) = b, \quad \text{where} \quad
   A =  \begin{bmatrix}
      \;\; — & \text{vec}(A_1)^\top & — \;\; \\
      \;\; — & \text{vec}(A_2)^\top & — \;\; \\
             & \vdots               &        \\
      \;\; — & \text{vec}(A_p)^\top & — \;\;
   \end{bmatrix} \in\mathbb{R}^{p \times n^2}.

Alternatively, if we have linear constraints of the form

.. math::

   \sum_{i=1}^q x_i G_i = H,

where :math:`x \in \mathbb{R}^q` is a variable, and :math:`G_i \in \mathbb{S}^n`
and :math:`H \in \mathbb{H}^n` encode linear constraints for 
:math:`i=1,\ldots,q`, then this is equivalent to 

.. math::

   G x = \text{vec}(H), \quad \text{where} \quad
   G =  \begin{bmatrix}
      \mid & \mid &        & \mid \\
      \text{vec}(G_1)      & \text{vec}(G_2) & \cdots & \text{vec}(G_q) \\
      \mid & \mid &        & \mid
   \end{bmatrix} \in \mathbb{R}^{n^2 \times q}.

.. _subsec-superoperator:

Superoperators
----------------

Often, we need to model linear operators which map matrices to matrices. In 
**QICS**, we will need to find the correct matrix representation for these 
operators. To do this, we simply recognize that each column of the matrix
representation should correspond to the linear operator acting on a 
computational basis element. 

For example, we can represent a superoperator 
:math:`\mathcal{A}:\mathbb{S}^2\rightarrow\mathbb{S}^2` as the matrix

.. math::

   A =  \begin{bmatrix}
      \mid & \mid & \mid & \mid \\
      \text{cvec}(\mathcal{A}(E_{11})) & \text{cvec}(\mathcal{A}(E_{12})) 
      & \text{cvec}(\mathcal{A}(E_{21})) & \text{cvec}(\mathcal{A}(E_{22})) \\
      \mid & \mid & \mid & \mid
   \end{bmatrix}

where

.. math::

    E_{11} = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}, \quad
    E_{12} = E_{21} = 
    \frac{1}{2} \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad
    E_{22} = \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}.

Using this, we can model a linear constraint

.. math::

    \mathcal{A}(X) = B,

as

.. math::

    A \text{vec}(X) = \text{cvec}(B),

Note that we use compact vectorizations for the columns of :math:`A` and for
:math:`B` to avoid redundant equality constraints already enforced by symmetry 
of the matrices.

In **QICS**, we provide the helper function :obj:`qics.vectorize.lin_to_mat`
which does this. Below is an example for showing how a matrix representation for
the identity superoperator on :math:`2\times2` symmetric matrices can be 
generated.

>>> qics.vectorize.lin_to_mat(lambda X : X, (2, 2))
array([[1.        , 0.        , 0.        , 0.        ],
       [0.        , 0.70710678, 0.70710678, 0.        ],
       [0.        , 0.        , 0.        , 1.        ]])

Alternatively, we can use :obj:`qics.vectorize.eye` to directly generate this
matrix.

>>> qics.vectorize.eye(2)
array([[1.        , 0.        , 0.        , 0.        ],
       [0.        , 0.70710678, 0.70710678, 0.        ],
       [0.        , 0.        , 0.        , 1.        ]])

As another example, we show below how to generate the (transposed) matrix
corresponding to the partial trace.

>>> qics.vectorize.lin_to_mat(lambda X : qics.quantum.p_tr(X, (2, 2), 0), (4, 2)).T
array([[1.        , 0.        , 0.        ],
       [0.        , 0.70710678, 0.        ],
       [0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        ],
       [0.        , 0.70710678, 0.        ],
       [0.        , 0.        , 1.        ],
       [0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        ],
       [1.        , 0.        , 0.        ],
       [0.        , 0.70710678, 0.        ],
       [0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        ],
       [0.        , 0.70710678, 0.        ],
       [0.        , 0.        , 1.        ]])

.. warning::

   Many of these functions are not optimized, and can be slow for large
   matrices. Users working with medium to large scale problems should implement
   a custom function for generating these matrix representations of
   superoperators.