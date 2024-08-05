Max cut
=============

Consider the following undirected graph with 5 nodes

.. image:: graph.png
    :align: center
    :alt: Simple directed graph with 5 nodes

The semidefinite approximation of the max cut problem
corresponding to this graph is

.. math::

    \max_{X \in \mathbb{S}^5} &&& \langle C, X \rangle

    \text{s.t.} &&& X_{ii} = 1 \qquad i=1,\ldots,5

    &&& X \succeq 0,

where

.. math::

    C = \begin{bmatrix} 
            2 & -1 & -1 &  0 &  0 \\ 
           -1 &  3 & -1 & -1 &  0 \\
           -1 & -1 &  3 &  0 & -1 \\
            0 & -1 &  0 &  2 & -1 \\
            0 &  0 & -1 & -1 &  2
        \end{bmatrix}.

To solve this semidefinite program using **QICS**, we first 
define the objective function corresponding to the semidefinite
relaxation of the max cut problem.

.. code-block:: python
    
    import numpy as np

    c = -np.array([
        [ 2., -1., -1.,  0.,  0.],
        [-1.,  3., -1., -1.,  0.],
        [-1., -1.,  3.,  0., -1.],
        [ 0., -1.,  0.,  2., -1.],
        [ 0.,  0., -1., -1.,  2.]
    ]).reshape(-1, 1)

.. note::
    We define the objective as :math:`-C` as **QICS** always 
    assumes the linear function is being minimized.

Next, we define the linear constraint matrix corresponding to
:math:`X_{ii}=1` for :math:`i=1,\ldots,5`. To do this, we note
that these constraints are equivalent to

.. math::

   \text{tr}[A_i X] = 1, \qquad \forall\ i=1,\ldots,5,

where :math:`A_i` is the matrix with one in the :math:`(i, i)`-th
element and zeros everywhere else. Then we vectorize these 
matrices into row vectors and stack them on top of each other
(see :ref:`Mat to vec` for more details) to obtain our constraint matrix.

.. code-block:: python

    import scipy as sp

    A = np.zeros((5, 5, 5))
    A[range(5), range(5), range(5)] = 1.
    A = sp.sparse.csr_matrix(A.reshape(5, -1))

    b = np.ones((5, 1))

.. note::
    We represent the constraint matrix ``A`` as a sparse matrix, which
    can lead to significant speed ups for linear and semidefinite programs
    when ``A`` is sufficiently sparse.

We also need to define which cones we are optimizing over. In
this case we are just optimizing over the positive semidefinite cone.

.. code-block:: python

    import qics
    cones = [qics.cones.PosSemidefinite(5)]

Finally, we create a :class:`~qics.Model` to represent the maxcut
semidefinite program, and a :class:`~qics.Solver` to solve this problem.

.. code-block:: python

    # Initialize model and solver objects
    model  = qics.Model(c=c, A=A, b=b, cones=cones)
    solver = qics.Solver(model)

    # Solve problem
    info = solver.solve()

    print("Optimal matrix variable X is: ")
    print(info["s_opt"][0])

.. code-block:: none

    ====================================================================
                QICS v0.0 - Quantum Information Conic Solver
                by K. He, J. Saunderson, H. Fawzi (2024)
    ====================================================================
    Problem summary:
            no. cones:  1                        no. vars:    25
            barr. par:  6                        no. constr:  5
            symmetric:  True                     cone dim:    25
            complex:    False

    ...

    Solution summary
            sol. status:  optimal                num. iter:    7
            exit status:  solved                 solve time:   x.xxx

            primal obj:  -8.741944078919e+00     primal feas:  7.95e-09
            dual obj:    -8.741944049469e+00     dual feas:    3.97e-09
            opt. gap:     3.37e-09

    Optimal matrix variable X is:
    [[ 0.99999999 -0.3668415  -0.3668415   0.12486877  0.12486877]
     [-0.3668415   0.99999999 -0.73085463 -0.96880942  0.87719533]
     [-0.3668415  -0.73085463  0.99999999  0.87719533 -0.96880942]
     [ 0.12486877 -0.96880942  0.87719533  0.99999999 -0.96881558]
     [ 0.12486877  0.87719533 -0.96880942 -0.96881558  0.99999999]]

.. note::
    When ``G`` and ``h`` are not specified when initializing a :class:`~qics.Model`,
    the variables ``x`` and ``s`` represent the same variables, just in different
    formats, i.e., ``x`` is a ``np.ndarray`` vector, while ``s`` is a :class:`~qics.Vector`
    representing a list of real vectors, symmetric matrices, and/or Hermitian matrices.

Complex max cut
--------------------

In signal processing :ref:`[1] <maxcut_refs>`, the following complex 
variation of the semidefinite relaxation of max cut arises

.. math::

    \min_{X \in \mathbb{H}^n} &&& \langle C, X \rangle

    \text{s.t.} &&& X_{ii} = 1 \qquad i=1,\ldots,5

    &&& X \succeq 0,

where

.. math::

    C = \text{diag}(v)(\mathbb{I} - UU^\dagger)\text{diag}(v)

for some complex matrix :math:`U \in \mathbb{C}^{n \times m}` and real 
vector :math:`v \in \mathbb{R}^n`. We can solve this in **QICS** by making
two main changes to the previous code.

- We specify ``iscomplex=True`` when initializing 
  :class:`~qics.cones.PosSemidefinite`.This option is ``False`` by 
  default.

- We vectorize Hermitian matrices by first splitting up real and 
  complex components using ``C.view(np.float64)`` to obtain a real 
  matrix. See :ref:`Mat to vec` for more details.

We supply full example code for the complex max cut below.

.. code-block:: python
    :emphasize-lines: 14, 19, 24

    import numpy as np
    import scipy as sp
    import qics

    np.random.seed(1)

    n = 5
    m = 4

    # Generate random linear objective function
    U = np.random.randn(n, m) + np.random.randn(n, m)*1j
    v = np.random.randn(n)
    C = np.diag(v) @ (np.eye(n) - U @ U.conj().T) @ np.diag(v)
    c = C.view(np.float64).reshape(-1, 1)

    # Build linear constraints A corresponding to Xii=1
    A = np.zeros((n, n, n), dtype=np.complex128)
    A[range(n), range(n), range(n)] = 1.
    A = sp.sparse.csr_matrix(A.view(np.float64).reshape(n, -1))

    b = np.ones((n, 1))

    # Define cones to optimize over
    cones = [qics.cones.PosSemidefinite(n, iscomplex=True)]

    # Initialize model and solver objects
    model  = qics.Model(c=c, A=A, b=b, cones=cones)
    solver = qics.Solver(model)

    # Solve problem
    info = solver.solve()

    np.set_printoptions(precision=2)
    print("Optimal matrix variable X is: ")
    print(info["s_opt"][0])

.. code-block:: none

    ====================================================================
                QICS v0.0 - Quantum Information Conic Solver
                by K. He, J. Saunderson, H. Fawzi (2024)
    ====================================================================
    Problem summary:
            no. cones:  1                        no. vars:    50
            barr. par:  6                        no. constr:  5
            symmetric:  True                     cone dim:    50
            complex:    True

    ...

    Solution summary
            sol. status:  optimal                num. iter:    8
            exit status:  solved                 solve time:   x.xxx

            primal obj:  -5.349376919568e+01     primal feas:  9.86e-09
            dual obj:    -5.349376915221e+01     dual feas:    8.23e-09
            opt. gap:     8.13e-10

    Optimal matrix variable X is:
    [[ 1.  +0.j    0.21-0.98j  0.67-0.74j -0.58+0.81j  0.87-0.5j ]
     [ 0.21+0.98j  1.  +0.j    0.87+0.5j  -0.92-0.4j   0.67+0.74j]
     [ 0.67+0.74j  0.87-0.5j   1.  +0.j   -0.99+0.11j  0.95+0.31j]
     [-0.58-0.81j -0.92+0.4j  -0.99-0.11j  1.  +0.j   -0.91-0.41j]
     [ 0.87+0.5j   0.67-0.74j  0.95-0.31j -0.91+0.41j  1.  +0.j  ]]



.. _maxcut_refs:

References
----------

    1. "Phase recovery, maxcut and complex semidefinite programming",
       I. Waldspurger, A. d'Aspremont, and S. Mallat.
       *Mathematical Programming*, pp. 1-35, 2012.

