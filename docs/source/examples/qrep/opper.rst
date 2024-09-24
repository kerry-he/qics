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

.. tabs::

    .. code-tab:: python Native

        import numpy
        import qics

        numpy.random.seed(1)

        n = 4
        alpha = 0.25

        rho   = qics.quantum.random.density_matrix(n, iscomplex=True)
        sigma = qics.quantum.random.density_matrix(n, iscomplex=True)

        # Define objective function
        cT = (1 - alpha) * qics.vectorize.mat_to_vec(sigma)
        cX = numpy.zeros((2*n*n, 1))
        cY = alpha * qics.vectorize.mat_to_vec(rho)
        c = numpy.vstack((cT, cX, cY))

        # Build linear constraint matrices
        vn = qics.vectorize.vec_dim(n, compact=True, iscomplex=True)
        # X = I
        A = numpy.hstack((
            numpy.zeros((vn, 2*n*n)), 
            qics.vectorize.eye(n, iscomplex=True), 
            numpy.zeros((vn, 2*n*n))
        ))
        b = qics.vectorize.mat_to_vec(numpy.eye(n, dtype=numpy.complex128), compact=True)

        # Define cones to optimize over
        cones = [qics.cones.OpPerspecEpi(n, alpha/(alpha - 1), iscomplex=True)]

        # Initialize model and solver objects
        model  = qics.Model(c=c, A=A, b=b, cones=cones)
        solver = qics.Solver(model)

        # Solve problem
        info = solver.solve()

    .. code-tab:: python PICOS

        import numpy
        import picos
        import qics

        numpy.random.seed(1)

        n = 4
        alpha = 0.25

        rho   = qics.quantum.random.density_matrix(n, iscomplex=True)
        sigma = qics.quantum.random.density_matrix(n, iscomplex=True)

        # Define problem
        P = picos.Problem()
        omega = picos.HermitianVariable("omega", n)
        theta = picos.HermitianVariable("theta", n)

        P.set_objective("min", alpha*(omega | rho).real + (1-alpha)*(theta | sigma).real)
        P.add_constraint(theta >> picos.mtxgeomean(picos.I(n), omega, alpha/(alpha-1)))

        # Solve problem
        P.solve(solver="qics", verbosity=2)

.. _opper_refs:

References
----------

    1. Huang, Zixin, and Mark M. Wilde. 
       "Semi-definite optimization of the measured relative 
       entropies of quantum states and channels." 
       arXiv preprint arXiv:2406.19060 (2024).