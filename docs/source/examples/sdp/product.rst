Block diagonal
==============

**QICS** is designed to solve conic programs involving Cartesian products of
cones. This makes it easy to take advantage of block-diagonal structures arising
in semidefeinite programs. As an example, consider the semidefinite program from
Example 2 of :ref:`[1] <blockdiag_refs>`

.. math::

    \min_{x \in \mathbb{R}^5} &&& 1.1x_1 - 10x_2 + 6.6x_3 + 19x_4 + 4.1x_5

    \text{subj. to} &&& \sum_{i=1}^5 F_ix_i - F_0 \succeq_{\mathbb{S}^7_+} 0,

where

.. math::

    \begin{gather}
        F_0 = \begin{bmatrix} 
                -1.4 & -3.2 &  0.0 &  0.0 &  0.0 &  0.0 &  0.0 \\
                -3.2 &  -28 &  0.0 &  0.0 &  0.0 &  0.0 &  0.0 \\
                 0.0 &  0.0 &   15 &  -12 &  2.1 &  0.0 &  0.0 \\
                 0.0 &  0.0 &  -12 &   16 & -3.8 &  0.0 &  0.0 \\
                 0.0 &  0.0 &  2.1 & -3.8 &   15 &  0.0 &  0.0 \\
                 0.0 &  0.0 &  0.0 &  0.0 &  0.0 &  1.8 &  0.0 \\
                 0.0 &  0.0 &  0.0 &  0.0 &  0.0 &  0.0 & -4.0
            \end{bmatrix}\\ \\
        F_1 = \begin{bmatrix} 
                 0.5 &  5.2 &  0.0 &  0.0 &  0.0 &  0.0 &  0.0 \\
                 5.2 & -5.3 &  0.0 &  0.0 &  0.0 &  0.0 &  0.0 \\
                 0.0 &  0.0 &  7.8 & -2.4 &  6.0 &  0.0 &  0.0 \\
                 0.0 &  0.0 & -2.4 &  4.2 &  6.5 &  0.0 &  0.0 \\
                 0.0 &  0.0 &  6.0 &  6.5 &  2.1 &  0.0 &  0.0 \\
                 0.0 &  0.0 &  0.0 &  0.0 &  0.0 & -4.5 &  0.0 \\
                 0.0 &  0.0 &  0.0 &  0.0 &  0.0 &  0.0 & -3.5
            \end{bmatrix}\\ \\
        \vdots\\ \\
        F_5 = \begin{bmatrix} 
                -6.5 & -5.4 &  0.0 &  0.0 &  0.0 &  0.0 &  0.0 \\
                -5.4 & -6.6 &  0.0 &  0.0 &  0.0 &  0.0 &  0.0 \\
                 0.0 &  0.0 &  6.7 & -7.2 & -3.6 &  0.0 &  0.0 \\
                 0.0 &  0.0 & -7.2 &  7.3 & -3.0 &  0.0 &  0.0 \\
                 0.0 &  0.0 & -3.6 & -3.0 & -1.4 &  0.0 &  0.0 \\
                 0.0 &  0.0 &  0.0 &  0.0 &  0.0 &  6.1 &  0.0 \\
                 0.0 &  0.0 &  0.0 &  0.0 &  0.0 &  0.0 & -1.5
            \end{bmatrix}.
    \end{gather}

It is possible to model this problem using a single :math:`7\times7` dimensional
positive semidefinite cone :math:`\mathbb{S}^7_+`. However, it is
computationally advantageous to take advantage of the block diagonal structure
of the matrices :math:`F_i` and model the problem using the Cartesian product of
cones :math:`\mathbb{S}^2_+\times\mathbb{S}^3_+\times\mathbb{R}^2_+`, i.e., the
semidefinite program is equivalent to

.. math::

    \min_{x \in \mathbb{R}^5} &&& 1.1x_1 - 10x_2 + 6.6x_3 + 19x_4 + 4.1x_5

    \text{subj. to} &&& \sum_{i=1}^5 F_{i0}x_i - F_{00} \succeq_{\mathbb{S}^2_+} 0,

    &&& \sum_{i=1}^5 F_{i1}x_i - F_{01} \succeq_{\mathbb{S}^3_+} 0,

    &&& \sum_{i=1}^5 F_{i2}x_i - F_{02} \geq_{\mathbb{R}^2_+} 0,

where :math:`F_{i0}\in\mathbb{S}^2`, :math:`F_{i1}\in\mathbb{S}^3`, and 
:math:`F_{i2}\in\mathbb{R}^2` represent the first, second, and third blocks of
:math:`F_{i}`, respectively, e.g.,

.. math::

    \begin{gather}
        F_{00} = \begin{bmatrix} 
                -1.4 & -3.2 \\
                -3.2 &  -28
            \end{bmatrix}, \quad 
        F_{01} = \begin{bmatrix} 
                 15 &  -12 &  2.1 \\
                -12 &   16 & -3.8 \\
                2.1 & -3.8 &   15
            \end{bmatrix}, \quad 
        F_{02} = \begin{bmatrix} 
                 1.8 \\
                -4.0
            \end{bmatrix}.
    \end{gather}

We can easily solve this problem involving a Cartesian product of positive
semidefinite cones and nonnegative orthants in **QICS** by defining an
appropriate :class:`list` of :mod:`qics.cones`.

.. code-block:: python

    import numpy as np

    import qics

    # Define objective function
    c = np.array([[1.1, -10, 6.6, 19, 4.1]]).T

    # Define linear constraints. Note that we have defined pre-vectorized 
    # the matrices for convenience
    F = [
        [   # F0
            np.array([[-1.4, -3.2, -3.2, -28]]).T,
            np.array([[15, -12, 2.1, -12, 16, -3.8, 2.1, -3.8, 15]]).T,
            np.array([[1.8, -4.0]]).T
        ],
        [   # F1
            np.array([[0.5, 5.2, 5.2, -5.3]]).T,
            np.array([[7.8, -2.4, 6.0, -2.4, 4.2, 6.5, 6.0, 6.5, 2.1]]).T,
            np.array([[-4.5, -3.5]]).T
        ],
        [   #F2
            np.array([[1.7, 7.0, 7.0, -9.3]]).T,
            np.array([[-1.9, -0.9, -1.3, -0.9, -0.8, -2.1, -1.3, -2.1, 4.0]]).T,
            np.array([[-0.2, -3.7]]).T
        ],
        [   #F3
            np.array([[6.3, -7.5, -7.5, -3.3]]).T,
            np.array([[0.2, 8.8, 5.4, 8.8, 3.4, -0.4, 5.4, -0.4, 7.5]]).T,
            np.array([[-3.3, -4.0]]).T
        ],
        [   #F4
            np.array([[-2.4, -2.5, -2.5, -2.9]]).T,
            np.array([[3.4, -3.2, -4.5, -3.2, 3.0, -4.8, -4.5, -4.8, 3.6]]).T,
            np.array([[4.8, 9.7]]).T
        ],
        [   #F5
            np.array([[-6.5, -5.4, -5.4, -6.6]]).T,
            np.array([[6.7, -7.2, -3.6, -7.2, 7.3, -3.0, -3.6, -3.0, -1.4]]).T,
            np.array([[6.1, -1.5]]).T
        ]
    ]

    h = -np.vstack(F[0])
    G = -np.hstack([np.vstack(Fi) for Fi in F[1:]])

    # Define cones to optimize over
    cones = [
        qics.cones.PosSemidefinite(2),
        qics.cones.PosSemidefinite(3),
        qics.cones.NonNegOrthant(2),
    ]

    # Initialize model and solver objects
    model  = qics.Model(c=c, G=G, h=h, cones=cones)
    solver = qics.Solver(model)

    # Solve problem
    info = solver.solve()

    print("Optimal variable x is: ")
    print(info["x_opt"].ravel())

.. code-block:: none

    Optimal variable x is:
    [1.55164255 0.67096851 0.98149139 1.40657036 0.94216841]

.. _blockdiag_refs:

References
----------

    1. "SDPA (SemiDefinite Programming Algorithm) User’s Manual -- Version 6.2.0.",
       K. Fujisawa, M. Kojima, K. Nakata, and M. Yamashita,
       *Research Reports on Mathematical and Computing Sciences Series B : Operations Research*, 2002.

