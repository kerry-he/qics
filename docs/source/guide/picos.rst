.. _picos-interface:

PICOS interface
=======================

The easiest way to use QICS is by using the optimization modelling software 
`PICOS <https://picos-api.gitlab.io/picos/>`_, which provides a high-level interface to
parse convex optimization problems to a solver. Below, we show how to initialize a PICOS
problem and variables.

.. code-block:: python
    
    import picos

    P = picos.Problem()
    t = picos.RealVariable("t")
    X = picos.SymmetricVariable("X", 2)
    Y = picos.SymmetricVariable("Y", 2)

Notably, PICOS supports the following functions, which can be used to formulate conic 
problems ot be solved using QICS.

.. list-table::
   :widths: 20 20 20 40
   :header-rows: 1
   :align: center

   * - Function
     - PICOS expression
     - Convexity
     - Description
   * - Quantum entropy
     - :class:`picos.qentr`
     - Concave
     - :math:`S(X) = -\text{tr}[X\log(X)]`
   * - Quantum relative entropy
     - :class:`picos.qrelentr`
     - Convex
     - :math:`S(X \| Y) = \text{tr}[X\log(X) - X\log(Y)]`
   * - Quantum conditional entropy
     - :class:`picos.qcondentr`
     - Concave
     - :math:`S(X) - S(\text{tr}_i(X))`
   * - Quantum key distribution
     - :class:`picos.qkeydist`
     - Convex
     - :math:`-S(\mathcal{G}(X)) + S(\mathcal{Z}(\mathcal{G}(X)))`
   * - Operator relative entropy
     - :class:`picos.oprelentr`
     - Operator convex
     - :math:`P_{\log}(X, Y) = X^{1/2} \log(X^{1/2} Y^{-1} X^{1/2}) X^{1/2}`
   * - Matrix geometric mean
     - :class:`picos.mtxgeomean`
     - Operator convex if :math:`t\in[-1, 0]\cup[1, 2]`
       Operator concave if :math:`t\in[0, 1]`
     - :math:`X\,\#_t\,Y = X^{1/2} (X^{1/2} Y^{-1} X^{1/2})^t X^{1/2}`

Scalar functions (i.e., quantum entropy, quantum relative entropy, quantum conditional
entropy, and quantum key distribution) can be used by either incorporating them in the 
objective function, e.g.,

.. code-block:: python
    
    P.set_objective("min", picos.qrelentr(X, Y))

or as an inequality constraint, e.g.,

.. code-block:: python

    P.add_constraint(t > picos.qrelentr(X, Y))

Matrix-valued functions (i.e., operator relative entropy and matrix geometric mean) can
be used in a matrix inequality expression, e.g.,

.. code-block:: python
    
    P.add_constraint(T >> picos.oprelentr(X, Y))

or composed with a trace function to represent the corresponding scalar valued function

.. code-block:: python
    
    P.set_objective("min", picos.trace(picos.oprelentr(X, Y)))

Note that these expressions need to define a **convex** optimization problem. Once a 
PICOS problem has been defined, it can be solved using QICS by calling

.. code-block:: python
    
    P.solve(solver="qics")
