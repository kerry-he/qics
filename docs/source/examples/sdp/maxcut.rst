Max cut
=============

Consider the following undirected graph with 5 nodes

.. figure:: graph_light.png
    :figclass: light-only
    :align: center
    :alt: Simple directed graph with 5 nodes

.. figure:: graph_dark.png
    :figclass: dark-only
    :align: center
    :alt: Simple directed graph with 5 nodes

The semidefinite approximation of the max cut problem :ref:`[1] <maxcut_refs>` 
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

We show how we can solve this semidefinite program using QICS below.

.. testcode::
    
    import numpy
    import qics

    # Define objective function
    C = numpy.array([
        [ 2., -1., -1.,  0.,  0.],
        [-1.,  3., -1., -1.,  0.],
        [-1., -1.,  3.,  0., -1.],
        [ 0., -1.,  0.,  2., -1.],
        [ 0.,  0., -1., -1.,  2.]
    ])
    c = -qics.vectorize.mat_to_vec(C)

    # Build linear constraint
    A = numpy.zeros((5, 25))
    A[numpy.arange(5), numpy.arange(0, 25, 6)] = 1.

    b = numpy.ones((5, 1))

    # Define cones to optimize over
    cones = [qics.cones.PosSemidefinite(5)]

    # Initialize model and solver objects
    model  = qics.Model(c=c, A=A, b=b, cones=cones)
    solver = qics.Solver(model, verbose=0)

    # Solve problem
    info = solver.solve()

    print("Optimal matrix variable X is:")
    print(info["s_opt"][0][0])
    print("which has rank:", numpy.linalg.matrix_rank(info["s_opt"][0][0], tol=1e-6))

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    Optimal matrix variable X is:
    [[ 1.         -0.36684149 -0.3668415   0.12486877  0.12486877]
     [-0.36684149  1.         -0.73085463 -0.96880942  0.87719533]
     [-0.3668415  -0.73085463  1.          0.87719533 -0.96880942]
     [ 0.12486877 -0.96880942  0.87719533  1.         -0.96881558]
     [ 0.12486877  0.87719533 -0.96880942 -0.96881558  1.        ]]
    which has rank: 2

Complex max cut
--------------------

In signal processing :ref:`[2] <maxcut_refs>`, the following complex variation 
of the semidefinite relaxation of max cut arises

.. math::

    \min_{X \in \mathbb{H}^n} &&& \langle C, X \rangle

    \text{s.t.} &&& X_{ii} = 1 \qquad i=1,\ldots,5

    &&& X \succeq 0,

where

.. math::

    C = \text{diag}(v)(\mathbb{I} - UU^\dagger)\text{diag}(v)

for some complex matrix :math:`U \in \mathbb{C}^{n \times m}` and real vector
:math:`v \in \mathbb{R}^n`. We can solve this in **QICS** by making a few 
adjustments to the previous code.

.. testcode::

    import numpy
    import qics

    numpy.random.seed(1)

    n = 5
    m = 4

    # Generate random linear objective function
    U = numpy.random.randn(n, m) + numpy.random.randn(n, m)*1j
    v = numpy.random.randn(n)
    C = numpy.diag(v) @ (numpy.eye(n) - U @ U.conj().T) @ numpy.diag(v)
    c = qics.vectorize.mat_to_vec(C)

    # Build linear constraints A corresponding to Xii=1
    A = numpy.zeros((5, 50))
    A[numpy.arange(5), numpy.arange(0, 50, 12)] = 1.

    b = numpy.ones((n, 1))

    # Define cones to optimize over
    cones = [qics.cones.PosSemidefinite(n, iscomplex=True)]

    # Initialize model and solver objects
    model  = qics.Model(c=c, A=A, b=b, cones=cones)
    solver = qics.Solver(model, verbose=0)

    # Solve problem
    info = solver.solve()

    print("Optimal matrix variable X is: ")
    print(numpy.round(info["s_opt"][0][0], 3))
    print("which has rank:", numpy.linalg.matrix_rank(info["s_opt"][0][0], tol=1e-6))

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    Optimal matrix variable X is:
    [[ 1.   +0.j     0.209-0.978j  0.67 -0.743j -0.584+0.812j  0.866-0.499j]
     [ 0.209+0.978j  1.   +0.j     0.866+0.499j -0.916-0.401j  0.67 +0.743j]
     [ 0.67 +0.743j  0.866-0.499j  1.   +0.j    -0.994+0.11j   0.951+0.309j]
     [-0.584-0.812j -0.916+0.401j -0.994-0.11j   1.   +0.j    -0.911-0.412j]
     [ 0.866+0.499j  0.67 -0.743j  0.951-0.309j -0.911+0.412j  1.   +0.j   ]]
    which has rank: 1

.. _maxcut_refs:

References
----------

    1. "Improved approximation algorithms for maximum cut and satisfiability 
       problems using semidefinite programming," M. X. Goemans, and 
       D. P. Williamson, Journal of the ACM (JACM) 42.6 (1995): 1115-1145.

    1. "Phase recovery, maxcut and complex semidefinite programming",
       I. Waldspurger, A. d'Aspremont, and S. Mallat.
       *Mathematical Programming*, pp. 1-35, 2012.

