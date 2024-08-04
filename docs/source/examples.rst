Examples
=============

To supply a conic program model to **QICS**, users are reqired
to express the linear constraints :math:`A` and :math:`G` in 
matrix form. This is straightforward when variables correspond to
real vectors. 

For variables corresponding to symmetric or Hermitian matrices, 
we need to first vectorize the matrices. For example, consider the 
linear constraints

.. math::

   \text{tr}[A_i X] = b_i, \qquad \forall\ i=1,\ldots,p,

where :math:`X \in \mathbb{S}^n` is our matrix variable, and 
:math:`A_i \in \mathbb{S}^n` and :math:`b_i \in \mathbb{R}` encode 
linear constraints for :math:`i=1,\ldots,p`. To express this in the
form required by **QICS**, we will vectorize the matrices :math:`A_i`
by stacking the rows of the matrix side-by-side, i.e., if

.. math::

   \text{vec}(A_i) = \text{vec}\left(\begin{bmatrix}
                                    \rule[.5ex]{5ex}{0.5pt} & a_{i1}^\top & \rule[.5ex]{5ex}{0.5pt} \\
                                    \rule[.5ex]{5ex}{0.5pt} & a_{i2}^\top & \rule[.5ex]{5ex}{0.5pt} \\
                                                            & \vdots      &                         \\
                                    \rule[.5ex]{5ex}{0.5pt} & a_{i3}^\top & \rule[.5ex]{5ex}{0.5pt}
                                \end{bmatrix}\right) = \begin{bmatrix}
                                                            a_{i1} \\
                                                            a_{i2} \\
                                                            \vdots \\
                                                            a_{i3}
                                                        \end{bmatrix},

where :math:`\text{vec}(A_i)` is a column vector with :math:`n^2` elements.
The corresponding constraint matrix :math:`A` can now be represented
by the :math:`p \times n^2` dimensional matrix

.. math::

   A =  \begin{bmatrix}
            \rule[.5ex]{2.5ex}{0.5pt} & \text{vec}(A_1)^\top & \rule[.5ex]{2.5ex}{0.5pt} \\
            \rule[.5ex]{2.5ex}{0.5pt} & \text{vec}(A_2)^\top & \rule[.5ex]{2.5ex}{0.5pt} \\
                                    & \vdots               &                         \\
            \rule[.5ex]{2.5ex}{0.5pt} & \text{vec}(A_p)^\top & \rule[.5ex]{2.5ex}{0.5pt}
        \end{bmatrix},

so that the linear constraints are equivalent to the expression

.. math::

   A\text{vec}(X) = b.

In Python, the this vectorization operation is easily performed using NumPy



If we instead have Hermitian matrices :math:`X \in \mathbb{H}^n` and 
:math:`A_i \in \mathbb{H}^n` for :math:`i=1,\ldots,p`, then we vectorize 