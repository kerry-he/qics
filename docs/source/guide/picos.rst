.. _picos-interface:

PICOS interface
===============

The easiest way to use QICS is by using the optimization modelling software 
`PICOS <https://picos-api.gitlab.io/picos/>`_, which provides a high-level 
interface to parse convex optimization problems to a solver. Notably, PICOS 
supports the following functions, which can be used to formulate conic problems
to be solved using QICS.

.. list-table:: **Quantum entropy and noncommutative perspectives supproted by PICOS**
   :widths: 20 20 20 40
   :header-rows: 1
   :align: center

   * - Function
     - PICOS expression
     - Convexity
     - Description
   * - Quantum entropy
     - :func:`picos.quantentr`
     - Concave
     - :math:`S(X) = -\text{tr}[X\log(X)]`
   * - Quantum relative entropy
     - :func:`picos.quantrelentr`
     - Convex
     - :math:`S(X \| Y) = \text{tr}[X\log(X) - X\log(Y)]`
   * - Quantum conditional entropy
     - :func:`picos.quantcondentr`
     - Concave
     - :math:`S(X) - S(\text{tr}_i(X))`
   * - Quantum key distribution
     - :func:`picos.quantkeydist`
     - Convex
     - :math:`-S(\mathcal{G}(X)) + S(\mathcal{Z}(\mathcal{G}(X)))`
   * - Operator relative entropy
     - :func:`picos.oprelentr`
     - Operator convex
     - :math:`P_{\log}(X, Y) = X^{1/2} \log(X^{1/2} Y^{-1} X^{1/2}) X^{1/2}`
   * - Matrix geometric mean
     - :func:`picos.mtxgeomean`
     - Operator convex if :math:`t\in[-1, 0]\cup[1, 2]`
       Operator concave if :math:`t\in[0, 1]`
     - :math:`X\,\#_t\,Y = X^{1/2} (X^{1/2} Y^{-1} X^{1/2})^t X^{1/2}`

Scalar functions (i.e., quantum entropy, quantum relative entropy, quantum 
conditional entropy, and quantum key distribution) can be used by either 
incorporating them in the objective function, e.g.,

.. code-block:: python
    
    P.set_objective("min", picos.quantrelentr(X, Y))

or as an inequality constraint, e.g.,

.. code-block:: python

    P.add_constraint(t > picos.quantrelentr(X, Y))

Matrix-valued functions (i.e., operator relative entropy and matrix geometric 
mean) can be used in a matrix inequality expression, e.g.,

.. code-block:: python
    
    P.add_constraint(T >> picos.oprelentr(X, Y))

or composed with a trace function to represent the corresponding scalar valued function

.. code-block:: python
    
    P.set_objective("min", picos.trace(picos.oprelentr(X, Y)))

Note that these expressions need to define a **convex** optimization problem. Once a 
PICOS problem has been defined, it can be solved using QICS by calling

.. code-block:: python
    
    P.solve(solver="qics")

Example
-------

Below, we show an example of how we can solve the same problem :ref:`nearest 
correlation matrix<Nearest>` problem introcued in :doc:`/guide/gettingstarted`, 
i.e.,

.. math::

    \min_{Y \in \mathbb{S}^2} \quad S( X \| Y ) \quad \text{s.t.} \quad Y_{11} 
    = Y_{22} = 1, \ Y \succeq 0,

where

.. math::

    X = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}.

.. testcode::

    import picos

    # Define the conic program
    P = picos.Problem()
    X = picos.Constant("X", [[2., 1.], [1., 2.]])
    Y = picos.SymmetricVariable("Y", 2)

    P.set_objective("min", picos.quantrelentr(X, Y))
    P.add_constraint(picos.maindiag(Y) == 1)

    print(P)

    # Solve the conic program
    P.solve(solver="qics")

    print("\nOptimal matrix variable Y is:")
    print(Y)

|

.. testoutput::

    Quantum Relative Entropy Program
      minimize S(X‖Y)
      over
        2×2 symmetric variable Y
      subject to
        maindiag(Y) = [1]

    Optimal matrix variable Y is:
    [ 1.00e+00  5.00e-01]
    [ 5.00e-01  1.00e+00]

Further examples for how PICOS can be used with QICS to solve problems arising
in quantum information theory can be found in :doc:`/examples/qrep/index`.