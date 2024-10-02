Quick reference
===============

This page aims to serve as a reference point for how to initialize and use the
cone, model, and solver classes provided by QICS. Further details about these
classes can be found in the :doc:`/api/index`.


Modelling
---------

**QICS** solves conic programs of the form

.. math::

  \min_{x \in \mathbb{R}^n} \quad c^\top x \quad 
  \text{s.t.} \quad b - Ax = 0, \  h - Gx \in \mathcal{K},

where :math:`c\in\mathbb{R}^n`, :math:`b\in\mathbb{R}^p`, 
:math:`h\in\mathbb{R}^q`, :math:`A\in\mathbb{R}^{p\times n}`, 
:math:`G\in\mathbb{R}^{q\times n}`, and 
:math:`\mathcal{K} \subset \mathbb{R}^{q}` is a Cartesian product of convex 
cones. This is representing using the :class:`qics.Model` class, which is
initialized using the following parameters.

.. list-table:: **Input parameters for** :class:`qics.Model`
   :widths: 15 60 25
   :header-rows: 1

   * - Parameter
     - Description
     - Default
   * - ``c``
     - :class:`numpy.ndarray` of size ``(n, 1)`` representing the linear 
       objective :math:`c`.
     - n/a
   * - ``A``
     - :class:`numpy.ndarray` or :class:`scipy.sparse.sparray` of size 
       ``(p, n)`` representing the left-hand side of the linear equality 
       constraints :math:`A`.
     - ``numpy.zeros((0, 1))``
   * - ``b``
     - :class:`numpy.ndarray` of size ``(p, 1)`` representing the right-hand 
       side of the linear equality constraints :math:`b`.
     - ``numpy.zeros((p, 1))``
   * - ``G``
     - :class:`numpy.ndarray` or :class:`scipy.sparse.sparray` of size 
       ``(q, n)`` representing the left-hand side of the linear conic 
       constraints :math:`G`.
     - ``-numpy.eye(n)``
   * - ``h``
     - :class:`numpy.ndarray` of size ``(q, 1)`` representing the right-hand
       side of the linear equality constraints :math:`h`.
     - ``numpy.zeros((q, 1))``
   * - ``cones``
     - List of :mod:`qics.cones` representing the Cartesian product of cones 
       :math:`\mathcal{K}`.
     - ``[]``
   * - ``offset``
     - Constant offset term to add to the objective function.
     - ``0.0``

.. note::
    When the parameters ``G`` and ``h`` are not specified when initializing a 
    :class:`qics.Model`, QICS instead solves the simplified conic program

    .. math::

      \min_{x \in \mathbb{R}^n} \quad c^\top x \quad 
      \text{s.t.} \quad Ax = b, \  x \in \mathcal{K}.


Cones
-----

Users define the Cartesian product of cones :math:`\mathcal{K}` by defining a
:class:`list` of cone classes from the :mod:`qics.cones` module. We list the 
definitions and interfaces to all of the cones QICS currently support below.

.. list-table:: **Symmetric cones**
   :widths: 25 25 50
   :header-rows: 1
   :align: center

   * - Cone
     - QICS class
     - Description
   * - Nonnegative orthant
     - :class:`qics.cones.NonNegOrthant`
     - :math:`\{ x \in \mathbb{R}^n : x \geq 0 \}`
   * - Positive semidefinite
     - :class:`qics.cones.PosSemidefinite`
     - :math:`\{ X \in \mathbb{H}^n : X \succeq 0 \}`
   * - Second order cone
     - :class:`qics.cones.SecondOrder`
     - :math:`\{(t, x) \in \mathbb{R} \times \mathbb{R}^{n} : t \geq \|x\|_2\}.`


.. list-table:: **Classical entropy cones**
   :widths: 25 25 50
   :header-rows: 1
   :align: center

   * - Cone
     - QICS class
     - Description
   * - Classical entropy
     - :class:`qics.cones.QuantEntr`
     - :math:`\text{cl}\{ (t, u, x) \in \mathbb{R} \times \mathbb{R}_{++} 
       \times \mathbb{R}^n_{++} : t \geq -u H(u^{-1} x) \}`
   * - Classical relative entropy
     - :class:`qics.cones.QuantRelEntr`
     - :math:`\text{cl}\{ (t, x, y) \in \mathbb{R} \times \mathbb{R}^n_{++} 
       \times \mathbb{R}^n_{++} : t \geq H(x \| y) \}`

.. list-table:: **Quantum entropy cones**
   :widths: 25 25 50
   :header-rows: 1
   :align: center

   * - Cone
     - QICS class
     - Description
   * - Quantum entropy
     - :class:`qics.cones.QuantEntr`
     - :math:`\text{cl}\{ (t, u, X) \in \mathbb{R} \times \mathbb{R}_{++} \times
       \mathbb{H}^n_{++} : t \geq -u S(u^{-1} X) \}`
   * - Quantum relative entropy
     - :class:`qics.cones.QuantRelEntr`
     - :math:`\text{cl}\{ (t, X, Y) \in \mathbb{R} \times \mathbb{H}^n_{++} 
       \times \mathbb{H}^n_{++} : t \geq S(X \| Y) \}`
   * - Quantum conditional entropy
     - :class:`qics.cones.QuantCondEntr`
     - :math:`\text{cl}\{ (t, X) \in \mathbb{R} \times \mathbb{H}^{n}_{++} : 
       t \geq -S(X) + S(\text{tr}_i(X)) \}`
   * - Quantum key distribution
     - :class:`qics.cones.QuantKeyDist`
     - :math:`\text{cl}\{ (t, X) \in \mathbb{R} \times \mathbb{H}^n_{++} : 
       t \geq -S(\mathcal{G}(X)) + S(\mathcal{Z}(\mathcal{G}(X))) \}`


.. list-table:: **Noncommutative perspective cones**
   :widths: 25 25 50
   :header-rows: 1
   :align: center

   * - Cone
     - QICS class
     - Description
   * - Operator perspective trace
     - :class:`qics.cones.OpPerspecTr`
     - :math:`\text{cl}\{ (t, X, Y) \in \mathbb{R} \times \mathbb{H}^n_{++}
       \times \mathbb{H}^n_{++} : t \geq \text{tr}[P_g(X, Y)] \}`
   * - Operator perspective epigraph
     - :class:`qics.cones.OpPerspecEpi`
     - :math:`\text{cl}\{ (T, X, Y) \in \mathbb{H}^n \times \mathbb{H}^n_{++}
       \times \mathbb{H}^n_{++} : T \succeq P_g(X, Y) \}`

.. _reference solving:

Solving
-------

Input parameters
~~~~~~~~~~~~~~~~

Once a conic program has been defined by a :class:`qics.Model`, the problem is
solved using a :class:`qics.Solver` class. This can be initialized with the
following settings.

.. list-table:: **Input parameters for** :class:`qics.Solver`
   :widths: 20 65 15
   :header-rows: 1

   * - Parameter
     - Description
     - Default
   * - ``model``
     - :class:`qics.Model` which specifies an instance of a conic program.
     - n/a
   * - ``max_iter``
     - Maximum number of solver iterations before terminating.
     - ``100``
   * - ``max_time``
     - Maximum time elapsed, in seconds, before terminating.
     - ``3600``
   * - ``tol_gap``
     - Stopping tolerance for (relative) optimality gap.
     - ``1e-8``
   * - ``tol_feas``
     - Stopping tolerance for (relative) primal and dual feasibility.
     - ``1e-8``
   * - ``tol_infeas``
     - Tolerance for detecting infeasible problem.
     - ``1e-12``
   * - ``tol_ip``
     - Tolerance for detecting ill-posed problem.
     - ``1e-13``
   * - ``tol_near``
     - Allowable margin for certifying near optimality when solver is stopped
       early.
     - ``1000``
   * - ``verbose``
     - Verbosity level of the solver, where

       - ``0``: No output.
       - ``1``: Only print problem and solution summary.
       - ``2``: Also print summary of the solver at each iteration.
       - ``3``: Also print symmary of the stepper at each iteration.

     - ``2``
   * - ``ir``
     - Whether to use iterative refinement when solving the KKT system.
     - ``True``
   * - ``toa``
     - Whether to use third-order adjustments to improve the stepping 
       directions.
     - ``True``
   * - ``init_pnt``
     - :class:`qics.point.Point` representing where to initialize the 
       interior-point algorithm from. Variables which contain :obj:`numpy.nan`
       are flagged to be intialized using QICS' default initialization.
     - ``None``
   * - ``use_invhess``
     - Whether to avoid using inverse Hessian product oracles by solving a
       modified cone program with :math:`\mathcal{K}'=\{x:-Gx\in\mathcal{K}\}`.
       Requires an initial point :math:`x_0` to be specified such that 
       :math:`-Gx_0\in\text{int}\ \mathcal{K}`. 
     - ``True``

Output parameters
~~~~~~~~~~~~~~~~~

Once a :class:`qics.Solver` has been initialized, the conic program can be 
solved with :meth:`qics.Solver.solve`. This returns a 
dictionary which summarizes the solution of the conic program, and has the 
following keys.

.. list-table:: **Dictionary keys for output of** :meth:`qics.Solver.solve`
   :widths: 22 78
   :header-rows: 1

   * - Parameter
     - Description
   * - ``x_opt``, ``y_opt``, ``z_opt``, ``s_opt``
     - Optimal primal and dual variables :math:`x^*`, :math:`y^*`, :math:`z^*`,
       and :math:`s^*`.
   * - ``sol_status``
     - Solution status. Can either be

       - ``optimal``: Primal-dual optimal solution reached
       - ``pinfeas``: Detected primal infeasibility
       - ``dinfeas``: Detected dual infeasibility
       - ``near_optimal``: Near primal-dual optimal solution
       - ``near_pinfeas``: Near primal infeasibility
       - ``near_dinfeas``: Near dual infeasibiltiy
       - ``illposed``: Problem is ill-posed
       - ``unknown``: Unknown solution status

   * - ``exit_status``
     - Solver exit status. Can either be

       - ``solved``: Terminated at desired tolerance
       - ``max_iter``: Exceeded maximum allowable iterations
       - ``max_time``: Exceeded maximum allowable time
       - ``step_failure``: Unable to take another step
       - ``slow_progress``: Residuals are decreasing too slowly

   * - ``num_iter``
     - Number of solver iterations.
   * - ``solve_time``
     - Total time elapsed by solver (in seconds).
   * - ``p_obj``, ``d_obj``
     - Optimal primal objective :math:`c^\top x^*` and dual objective 
       :math:`-b^\top y^* - h^\top z^*`.
   * - ``opt_gap``
     - Relative optimality gap.
   * - ``p_feas``, ``d_feas``
     - Relative primal feasibility and dual feasiblity.

