.. _Mat to vec:

Representing matrices
=======================

To supply a conic program model to **QICS**, users are reqired
to express the linear constraints :math:`A` and :math:`G` in 
matrix form. This is straightforward when variables correspond to
real vectors. For variables corresponding to symmetric or Hermitian
matrices, we need to first vectorize these matrices. 

Symmetric matrices
--------------------

Consider a real symmetric matrix :math:`X \in \mathbb{S}^n`. We 
can vectorize the matrices :math:`A_i` by stacking the rows of 
the matrix side-by-side, i.e.,

.. math::

   \text{vec}(X) = \text{vec}\left(\begin{bmatrix}
                                    \rule[.5ex]{5ex}{0.5pt} & x_{1}^\top & \rule[.5ex]{5ex}{0.5pt} \\
                                    \rule[.5ex]{5ex}{0.5pt} & x_{2}^\top & \rule[.5ex]{5ex}{0.5pt} \\
                                                            & \vdots      &                         \\
                                    \rule[.5ex]{5ex}{0.5pt} & x_{n}^\top & \rule[.5ex]{5ex}{0.5pt}
                                \end{bmatrix}\right) = \begin{bmatrix}
                                                            x_{1} \\
                                                            x_{2} \\
                                                            \vdots \\
                                                            x_{n}
                                                        \end{bmatrix},

so that :math:`\text{vec}(X) \in \mathbb{R}^{n^2}`. In
Python, this vectorization operation can easily be performed using NumPy's
reshaping functionality.

>>> import numpy as np
>>> X = np.array([[0., 1., 2.], [1., 3., 4.], [2., 4., 5.]])
>>> X
array([[0., 1., 2.],
       [1., 3., 4.],
       [2., 4., 5.]])
>>> X.reshape(-1, 1).T
array([[0., 1., 2., 1., 3., 4., 2., 4., 5.]])

Alternatively, **QICS** supplies the following function to convert matrices 
to vectors.

>>> from qics.vectorize import mat_to_vec
>>> mat_to_vec(X).T
array([[0., 1., 2., 1., 3., 4., 2., 4., 5.]])

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
                                                        \end{bmatrix},

Note the scaling of the off-diagonals is to make sure the inner product is
consistent between the two reprsentations. This compact representation is 
useful when defining linear constraints to avoid duplicate constraints 
(see :ref:`subsec-superoperator`). This compact vectorization
can also be performed in **QICS** by specifying the ``compact`` argument.

>>> mat_to_vec(X, compact=True).T
array([[0.        , 1.41421356, 3.        , 2.82842712, 5.65685425,
        5.        ]])


Hermitian matrices
--------------------

Now instead consider a complex Hermitian matrix :math:`X \in \mathbb{H}^n`.
To vectorize this matrix, we first convert each row of :math:`X` into a 
real vector by splitting the real and imaginary components of each entry
as follows

.. math::

   \text{real}(x_i) = \text{real}\!\left( \begin{bmatrix}
                                          x_{i1} \\
                                          x_{i2} \\
                                          \vdots \\
                                          x_{in}
                                       \end{bmatrix} \right) = \begin{bmatrix}
                                                                  \text{Re}(x_{i1}) \\
                                                                  \text{Im}(x_{i1}) \\
                                                                  \text{Re}(x_{i2}) \\
                                                                  \text{Im}(x_{i2}) \\
                                                                  \vdots \\
                                                                  \text{Re}(x_{in}) \\
                                                                  \text{Im}(x_{in}) \\
                                                               \end{bmatrix}.

which produces a :math:`2n` dimensional real vector.
Using this, Hermitian matrices are vectorized as follows

.. math::

   \text{vec}(X) = \text{vec}\!\left(\begin{bmatrix}
                                    \rule[.5ex]{5ex}{0.5pt} & x_{1}^\top & \rule[.5ex]{5ex}{0.5pt} \\
                                    \rule[.5ex]{5ex}{0.5pt} & x_{2}^\top & \rule[.5ex]{5ex}{0.5pt} \\
                                                            & \vdots      &                        \\
                                    \rule[.5ex]{5ex}{0.5pt} & x_{3}^\top & \rule[.5ex]{5ex}{0.5pt}
                                \end{bmatrix}\right) = \begin{bmatrix}
                                                            \text{real}(x_{1}) \\
                                                            \text{real}(x_{2}) \\
                                                            \vdots \\
                                                            \text{real}(x_{n})
                                                        \end{bmatrix},

so that :math:`\text{vec}(X) \in \mathbb{R}^{2n^2}`. 
Like for the real symmetric case, this vectorization operation can easily 
be performed using NumPy.

>>> X = np.array([[0., 1.+1.j, 2.+2.j], [1.-1.j, 3., 4.+4.j], [2.-2.j, 4.-4.j, 5.]])
>>> X
array([[0.+0.j, 1.+1.j, 2.+2.j],
       [1.-1.j, 3.+0.j, 4.+4.j],
       [2.-2.j, 4.-4.j, 5.+0.j]])
>>> X.view(np.float64).reshape(-1, 1).T
array([[0.,  0.,  1.,  1.,  2.,  2.,  1., -1.,  3.,  0.,  4.,  4.,
        2., -2.,  4., -4.,  5.,  0.]])


Note that we assume the matrix ``X`` is of type ``np.complex128``.
Alternatively, we can use the ``mat_to_vec`` function again to convert Hermitian  
matrices to vectors (the function automatically detects if the matrix is complex).

>>> mat_to_vec(X).T
array([[0.,  0.,  1.,  1.,  2.,  2.,  1., -1.,  3.,  0.,  4.,  4.,
        2., -2.,  4., -4.,  5.,  0.]])

Like the symmetric case, we can also define a compact vectorization for Hermitian
matrices as follows

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
                                                        \end{bmatrix},

which can be done in **QICS** as follows 

>>> mat_to_vec(X, compact=True).T
array([[0.        , 1.41421356, 1.41421356, 3.        , 2.82842712,
        2.82842712, 5.65685425, 5.65685425, 5.        ]])


Modelling constraints
-------------------------

Consider the linear constraints

.. math::

   \text{tr}[A_i X] = b_i, \qquad \forall\ i=1,\ldots,p,

where :math:`X \in \mathbb{S}^n` is our matrix variable, and 
:math:`A_i \in \mathbb{S}^n` and :math:`b_i \in \mathbb{R}` encode 
linear constraints for :math:`i=1,\ldots,p`. We can represent this
constraint as 

.. math::

   A\text{vec}(X) = b,

where :math:`A` is the :math:`p \times n^2` dimensional matrix

.. math::

   A =  \begin{bmatrix}
            \rule[.5ex]{2.5ex}{0.5pt} & \text{vec}(A_1)^\top & \rule[.5ex]{2.5ex}{0.5pt} \\
            \rule[.5ex]{2.5ex}{0.5pt} & \text{vec}(A_2)^\top & \rule[.5ex]{2.5ex}{0.5pt} \\
                                    & \vdots               &                         \\
            \rule[.5ex]{2.5ex}{0.5pt} & \text{vec}(A_p)^\top & \rule[.5ex]{2.5ex}{0.5pt}
        \end{bmatrix}.

Alternatively, if we have linear constraints of the form

.. math::

   \sum_{i=1}^q x_i G_i = H,

where :math:`x \in \mathbb{R}^q` is a variable, and :math:`G_i \in \mathbb{S}^n`
and :math:`H \in \mathbb{S}^n` encode linear constraints for :math:`i=1,\ldots,q`, 
then this is equivalent to 

.. math::

   G x = \text{vec}(H),

where :math:`G` is the :math:`n^2 \times q` dimensional matrix

.. math::

   G =  \begin{bmatrix}
            \rule[-1ex]{0.5pt}{5ex} & \rule[-1ex]{0.5pt}{5ex} &        & \rule[-1ex]{0.5pt}{5ex} \\
            \text{vec}(G_1)         & \text{vec}(G_2)         & \cdots & \text{vec}(G_q) \\
            \rule[-1ex]{0.5pt}{5ex} & \rule[-1ex]{0.5pt}{5ex} &        & \rule[-1ex]{0.5pt}{5ex}
        \end{bmatrix}.        


.. _subsec-superoperator:

Superoperators
----------------

Often, we need to model linear operators which 
map matrices to matrices. In **QICS**, we will need to
find the correct matrix representation for these operators.
To do this, we simply recognize that each column of the matrix
representation should correspond to the linear operator acting on
a computational basis element. 

For example, we can represent a superoperator 
:math:`\mathcal{A}:\mathbb{S}^2\rightarrow\mathbb{S}^2` as the
matrix

.. math::

   A =  \begin{bmatrix}
            \rule[-1ex]{0.5pt}{5ex} & \rule[-1ex]{0.5pt}{5ex} & \rule[-1ex]{0.5pt}{5ex} & \rule[-1ex]{0.5pt}{5ex} \\
            \text{cvec}(\mathcal{A}(E_{11})) & \text{cvec}(\mathcal{A}(E_{12})) & \text{cvec}(\mathcal{A}(E_{21})) & \text{cvec}(\mathcal{A}(E_{22})) \\
            \rule[-1ex]{0.5pt}{5ex} & \rule[-1ex]{0.5pt}{5ex} & \rule[-1ex]{0.5pt}{5ex} & \rule[-1ex]{0.5pt}{5ex}
        \end{bmatrix}

where

.. math::

    E_{11} = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}, \quad
    E_{12} = E_{21} = \frac{1}{2} \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}, \quad
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

In **QICS**, we provide a helper function which does this.
Below is an example for showing how a matrix representation
for the identity superoperator on :math:`2\times2` symmetric 
matrices can be generated.

>>> from qics.vectorize import lin_to_mat
>>> lin_to_mat(lambda X : X, (2, 2))
array([[1.        , 0.        , 0.        , 0.        ],
       [0.        , 0.70710678, 0.70710678, 0.        ],
       [0.        , 0.        , 0.        , 1.        ]])

**QICS** also offers some functions to make linear operators
of common linear operators arising in quanutm information theory,
including

    - Identity operator
    - Partial trace
    - Kronecker product with identity
    - Trace

Note that these functions are not optimized, and can be slow 
for large matrices. Users working with medium to large scale 
problems should implement a custom function for generating
these matrix representations of superoperators.