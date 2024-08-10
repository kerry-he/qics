Operator perspective
=========================

Measured Renyi relative entropy of states
-------------------------------------------

In :ref:`[1] <opper_refs>`, it was shown that the measured
Renyi relative entropy of states  :math:`Q_\alpha^M(\rho \| \sigma)` 
could be computed using the epigraph of the operator perspective function. 

For example, :math:`\alpha\in(0, 1/2)`, we have

.. math::

    Q_\alpha^M(\rho \| \sigma) \quad = &&\min_{\omega, \theta \in \mathbb{H}^n} &&& \alpha \text{tr}[\omega \rho] + (1 - \alpha) \text{tr}[\theta \sigma]

    &&\text{s.t.} &&& \theta \succeq \omega^{\frac{\alpha}{\alpha-1}}

    &&&&& \omega, \theta \succeq 0.

which we can model using the constraint :math:`(\theta, \mathbb{I}, \omega)\in\mathcal{K}_{\text{op}}^{\frac{\alpha}{\alpha-1}}`

We can solve this in **QICS** using the :class:`~qics.cones.OpPerspecEpi` cone.

.. code-block:: python

    import numpy as np
    import qics
    import qics.utils.symmetric as sym
    import qics.utils.quantum as qu

    np.random.seed(1)

    n = 4
    alpha = 0.25

    rho   = qu.rand_density_matrix(n, iscomplex=True)
    sigma = qu.rand_density_matrix(n, iscomplex=True)

    # Define objective function
    cT = (1 - alpha) * sym.mat_to_vec(sigma)
    cX = np.zeros((2*n*n, 1))
    cY = alpha * sym.mat_to_vec(rho)
    c = np.vstack((cT, cX, cY))

    # Build linear constraint matrices
    vn = sym.vec_dim(n, compact=True, iscomplex=True)
    # X = I
    A = np.hstack((
        np.zeros((vn, 2*n*n)), 
        sym.eye(n, iscomplex=True), 
        np.zeros((vn, 2*n*n))
    ))
    b = sym.mat_to_vec(np.eye(n, dtype=np.complex128), compact=True)

    # Define cones to optimize over
    cones = [qics.cones.OpPerspecEpi(n, alpha/(alpha - 1), iscomplex=True)]

    # Initialize model and solver objects
    model  = qics.Model(c=c, A=A, b=b, cones=cones)
    solver = qics.Solver(model)

    # Solve problem
    info = solver.solve()

.. code-block:: none

    ====================================================================
                QICS v0.0 - Quantum Information Conic Solver
                by K. He, J. Saunderson, H. Fawzi (2024)
    ====================================================================
    Problem summary:
            no. cones:  1                        no. vars:    96
            barr. par:  13                       no. constr:  16
            symmetric:  False                    cone dim:    96
            complex:    True

    ...

    Solution summary
            sol. status:  optimal                num. iter:    14
            exit status:  solved                 solve time:   4.260

            primal obj:   8.299380401228e-01     primal feas:  3.45e-09
            dual obj:     8.299380434717e-01     dual feas:    2.61e-09
            opt. gap:     3.35e-09


.. _opper_refs:

References
----------

    1. Huang, Zixin, and Mark M. Wilde. 
       "Semi-definite optimization of the measured relative 
       entropies of quantum states and channels." 
       arXiv preprint arXiv:2406.19060 (2024).