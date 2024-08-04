Examples
=============

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
Python, this vectorization operation can easily be performed using NumPy

.. code-block:: python

   X.reshape(-1, 1)

Alternatively, **QICS** supplies the following function to convert matrices 
to vectors

.. code-block:: python

   qics.utils.symmetric.mat_to_vec(X, compact=False)


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
be performed using NumPy. For a matrix ``X`` of type ``np.complex128``, we can use

.. code-block:: python

   X.view(np.float64).reshape(-1, 1)

Additionally, **QICS** supplies the following function to convert matrices 
to vectors

.. code-block:: python

   qics.utils.symmetric.mat_to_vec(X, iscomplex=True, compact=False)



Modelling constraints
---------------------------

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
