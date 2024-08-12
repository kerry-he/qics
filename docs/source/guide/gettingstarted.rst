Getting started
====================

In this guide, we will use the following quantum relative entropy program

.. math::

    \min_{Y \in \mathbb{S}^2} \quad S( X \| Y ) \quad \text{s.t.} \quad Y_{11} = Y_{22} = 1, \ Y \succeq 0,

where

.. math::

    X = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix},

as a running example to see how we can solve problems in **QICS**.

Modelling
------------

First, we need to reformulate the above problem as a standard
form conic program. We can do this by rewriting the problem as

.. math::

    \min_{t, X, Y} \quad & t \\
    \text{s.t.} \quad & \langle A_i, X \rangle = 2, \quad i=1,2,3\\
    & \langle B_j, Y \rangle = 1, \quad j=1,2 \\
    & (t, X, Y) \in \mathcal{K}_{\text{qre}},

where

.. math::

    A_1 = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}, \quad
    A_2 = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad
    A_3 = \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix},

and

.. math::

    B_1 = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}, \quad
    B_2 = \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}.

In **QICS**, we represent our variables :math:`(t, X, Y)\in\mathbb{R}\times\mathbb{S}^2\times\mathbb{S}^2`
as a vector :math:`x\in\mathbb{R}^9`, with elements represented by

.. math::

    x &= \begin{bmatrix} t & \text{vec}(X)^\top & \text{vec}(Y)^\top \end{bmatrix}^\top\\
      &= \begin{bmatrix} t & X_{11} & X_{12} & X_{21} & X_{22} & Y_{11} & Y_{12} & Y_{21} & Y_{22} \end{bmatrix}^\top

We can now represent our linear objective function as :math:`c^\top x`, where

.. math::

    c = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \end{bmatrix}^\top.

In Python, we represent this using a NumPy array.

.. code-block:: python

    import numpy as np
    c = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.]]).T

Additionaly, we represent our linear equality constraints using :math:`Ax=b`, where

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

    b = \begin{bmatrix} 2 & 2 & 2 & 1 & 1 \end{bmatrix}^\top

Again, in Python we represent this using NumPy arrays.

.. code-block:: python

    A = np.array([
        [0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1.]
    ])
    b = np.array([[2., 2., 2., 1., 1.]]).T

Finally, we want to tell **QICS** that :math:`x` must be constrained
in the quantum relative entropy cone :math:`\mathcal{K}_{\text{qre}}`.
We do this by using the :class:`~qics.cones.QuantRelEntr` class.

.. code-block:: python

    import qics
    cones = [qics.cones.QuantRelEntr(2)]    # Matrices X and Y are 2x2.

Note we define ``cones`` as a list of cones, as often we solve conic programs
involving a Cartesian product of cones. 

Finally, we initialize a :class:`~qics.Model` class to represent our 
conic program

.. code-block:: python

    model = qics.Model(c=c, A=A, b=b, cones=cones)


Solving
-----------

Now that we have built our model, solving the conic program is 
fairly straightforward. First, we initialize a :class:`~qics.Solver` 
class

.. code-block:: python

    solver = qics.Solver(model)

Optionally, there are also many solver settings we can specify when 
initializing the :class:`~qics.Solver`. These can be found in the API
documentation. 

We then solve the conic program by calling

.. code-block:: python

    info = solver.solve()

The default ``verbose`` level for the solver will give the following output
on the terminal.

.. code-block:: none

    ====================================================================
                QICS v0.0 - Quantum Information Conic Solver
                  by K. He, J. Saunderson, H. Fawzi (2024)
    ====================================================================
    Problem summary:
            no. cones:  1                        no. vars:    9
            barr. par:  6                        no. constr:  5
            symmetric:  False                    cone dim:    9
            complex:    False     

    =================================================================================================
    iter     mu        k/t    |    p_obj       d_obj       gap    |  p_feas    d_feas   |  time (s)
    =================================================================================================
       0   1.0e+00   1.0e+00  |  0.000e+00   0.000e+00   0.0e+00  |  1.3e+00   6.3e-01  |  0.00
       1   3.2e-01   1.4e+00  |  1.107e+00   2.028e+00   8.3e-01  |  6.1e-01   3.0e-01  |  4.33    
       2   6.3e-02   3.1e-01  |  2.374e+00   2.576e+00   8.5e-02  |  1.4e-01   6.8e-02  |  4.34
       3   6.3e-03   2.6e-02  |  2.729e+00   2.743e+00   5.4e-03  |  1.4e-02   7.0e-03  |  4.34
       4   6.4e-05   1.9e-05  |  2.772e+00   2.772e+00   3.3e-05  |  1.4e-04   6.9e-05  |  4.35
       5   5.9e-06   4.1e-05  |  2.773e+00   2.773e+00   1.1e-05  |  1.5e-05   7.4e-06  |  4.35
       6   6.0e-09   1.9e-08  |  2.773e+00   2.773e+00   2.4e-09  |  1.5e-08   7.4e-09  |  4.36
       7   5.9e-11   3.7e-10  |  2.773e+00   2.773e+00   8.9e-11  |  1.5e-10   7.5e-11  |  4.36

    Solution summary
            sol. status:  optimal                num. iter:    7
            exit status:  solved                 solve time:   4.361

            primal obj:   2.772588721774e+00     primal feas:  1.49e-10
            dual obj:     2.772588722021e+00     dual feas:    7.47e-11
            opt. gap:     8.89e-11

The solver returns a dictionary ``info`` containing additional
information about the solution. For example, we can access the 
optimal variables by using

.. code-block:: python

    print("Optimal matrix variable X is: ")
    print(info["s_opt"][0][1])

    print("Optimal matrix variable Y is: ")
    print(info["s_opt"][0][2])

.. code-block:: none

    Optimal matrix variable X is:
    [[2. 1.]
     [1. 2.]]
    Optimal matrix variable Y is:
    [[1.  0.5]
     [0.5 1. ]]

which we can confirm satisfies our desired constraints.

.. note::
    The ``info["s_opt"]`` object is a :class:`~qics.utils.vector.VecProduct`,
    which represent a Cartesian product of real vectors, symmetric matrices, 
    and Hermitian matrices. The first index tells us we are accessing the 
    variables correpsonding to the first cone, i.e., :math:`(t, X, Y)` in the 
    quantum relative entropy cone. The second index tells us which of these 
    three variables we want to access.