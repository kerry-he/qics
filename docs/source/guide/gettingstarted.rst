Getting started
===============

In this guide, we will use the following quantum relative entropy program

.. math::

    \min_{Y \in \mathbb{S}^2} \quad S( X \| Y ) \quad \text{s.t.} \quad Y_{11} 
    = Y_{22} = 1, \ Y \succeq 0,

where

.. math::

    X = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix},

as a running example to see how we can solve problems in **QICS**. This is a
simple example of the a :ref:`nearest correlation matrix
<examples/qrep/nearest:nearest correlation matrix>` problem. Further examples
showing how QICS can be used to solve semidefinite and quantum relative entropy
programs can be found in :doc:`/examples/index`.

Modelling
---------

First, we need to reformulate the above problem as a standard form conic
program. We can do this by rewriting the problem as

.. math::

    \min_{t, X, Y} \quad & t \\
    \text{s.t.} \quad & \langle A_i, X \rangle = 2, \quad i=1,2,3\\
    & \langle B_j, Y \rangle = 1, \quad j=1,2 \\
    & (t, X, Y) \in \mathcal{QRE}_2,

where

.. math::

    A_1 = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}, \quad
    A_2 = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad
    A_3 = \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}, \quad
    B_1 = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}, \quad
    B_2 = \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}.

In **QICS**, we represent our variables :math:`(t, X, Y)\in\mathbb{R}\times
\mathbb{S}^2\times\mathbb{S}^2` as a vector :math:`x\in\mathbb{R}^9`, with 
lements represented by

.. math::

    x &= \begin{bmatrix} 
           t & \text{vec}(X)^\top & \text{vec}(Y)^\top 
         \end{bmatrix}^\top\\
      &= \begin{bmatrix} 
           t & 
           X_{11} & X_{12} & X_{21} & X_{22} & 
           Y_{11} & Y_{12} & Y_{21} & Y_{22} 
         \end{bmatrix}^\top.

See :doc:`here<matrices>` for additional details about how QICS vectorizes
matrices. We can now represent our linear objective function as 
:math:`c^\top x`, where

.. math::

    c = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \end{bmatrix}^\top.

In Python, we represent this using a :class:`numpy.ndarray` array.

>>> import numpy
>>> c = numpy.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.]]).T

Additionaly, we represent our linear equality constraints using :math:`Ax=b`, 
where

.. math::

    A = \begin{bmatrix} 
        0 & \text{vec}(A_1)^\top & \text{vec}(0)^\top \\
        0 & \text{vec}(A_2)^\top & \text{vec}(0)^\top \\
        0 & \text{vec}(A_3)^\top & \text{vec}(0)^\top \\
        0 & \text{vec}(0)^\top & \text{vec}(B_1)^\top \\
        0 & \text{vec}(0)^\top & \text{vec}(B_2)^\top
    \end{bmatrix} = \begin{bmatrix} 
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
        0 & 0 & 1 & 1 & 0 & 0 & 0 & 0 & 0 \\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
    \end{bmatrix},

and

.. math::

    b = \begin{bmatrix} 2 & 2 & 2 & 1 & 1 \end{bmatrix}^\top,

Again, in Python we represent this using :class:`numpy.ndarray` arrays.

>>> A = numpy.array([                           \
...     [0., 1., 0., 0., 0., 0., 0., 0., 0.],   \
...     [0., 0., 1., 1., 0., 0., 0., 0., 0.],   \
...     [0., 0., 0., 0., 1., 0., 0., 0., 0.],   \
...     [0., 0., 0., 0., 0., 1., 0., 0., 0.],   \
...     [0., 0., 0., 0., 0., 0., 0., 0., 1.]    \
... ])
>>> b = numpy.array([[2., 2., 2., 1., 1.]]).T

Finally, we want to tell **QICS** that :math:`x` must be constrained in the
quantum relative entropy cone :math:`\mathcal{QRE}_2`. We do this by using the 
:class:`qics.cones.QuantRelEntr` class.

>>> import qics
>>> cones = [qics.cones.QuantRelEntr(2)]

.. note::
    We define ``cones`` as a list of cones, as often we solve conic programs
    involving a Cartesian product of cones.

Finally, we initialize a :class:`qics.Model` class to represent our conic
program using the matrices and cones we have defined.

>>> model = qics.Model(c=c, A=A, b=b, cones=cones)

Solving
-------

Now that we have built our model, solving the conic program is fairly
straightforward. First, we initialize a :class:`qics.Solver` class with the
model we have defined.

>>> solver = qics.Solver(model)

Optionally, there are also many solver settings we can specify when initializing
the :class:`qics.Solver`. A list of these options can be found 
:ref:`here<guide/reference:input parameters>`. Once we have initialized our 
:class:`qics.Solver`, we then solve the conic program by calling

.. doctest::
    :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

    >>> info = solver.solve()
    ====================================================================
               QICS v0.2.0 - Quantum Information Conic Solver
                  by K. He, J. Saunderson, H. Fawzi (2024)
    ====================================================================
    Problem summary:
        no. vars:     9                         barr. par:    6
        no. constr:   5                         symmetric:    False
        cone dim:     9                         complex:      False
        no. cones:    1                         sparse:       False
    ...
    Solution summary
        sol. status:  optimal                   num. iter:    7
        exit status:  solved                    solve time:   ...
        primal obj:   2.772588704718e+00        primal feas:  6.28e-09
        dual obj:     2.772588709215e+00        dual feas:    3.14e-09

The solver returns a dictionary ``info`` containing additional information about
the solution. A list of all keys contained in this dictionary can be found
:ref:`here<guide/reference:output parameters>`. For example, we can access the
optimal variable :math:`Y` by using

>>> print(info["s_opt"][0][2])
[[1.  0.5]
 [0.5 1. ]]

.. note::
    The ``info["s_opt"]`` object is a :class:`qics.point.VecProduct`, which
    represents a Cartesian product of real vectors, symmetric matrices, and
    Hermitian matrices. To access arrays corresponding to these vectors, we use
    ``info["s_opt"][i][j]`` to access the :math:`j`-th variable corresponding to
    the :math:`i`-th cone.