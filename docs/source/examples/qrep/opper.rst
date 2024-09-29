Noncommutative perspective
==========================

Measured relative entropy
-------------------------

Measured relative entropies are functions used to measure the amount of
dissimilarity between two quantum states which arise in quantum hypothesis 
testing tasks. In :ref:`[1] <opper_refs>`, it was shown that the measured Renyi
relative entropy of states  :math:`Q_\alpha^M(\rho \| \sigma)` could be computed
using the epigraph of the operator perspective function. 

For example, for :math:`\alpha\in(0, 1/2)`, we have

.. math::

    Q_\alpha^M(\rho \| \sigma) \quad = &&\min_{\omega, \theta \in \mathbb{H}^n}
    &&& \alpha\,\text{tr}[\omega \rho] + (1 - \alpha) \text{tr}[\theta \sigma]

    &&\text{s.t.} &&& \theta \succeq \omega^{\frac{\alpha}{\alpha-1}}

    &&&&& \omega, \theta \succeq 0,

which we can model using a constraint of the form 
:math:`(\theta, \mathbb{I}, \omega)\in\mathcal{OPE}`. We can solve this in 
**QICS** using :class:`qics.cones.OpPerspecEpi` below.

.. tabs::

    .. group-tab:: Native

        .. testcode:: native

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
            solver = qics.Solver(model, verbose=0)

            # Solve problem
            info = solver.solve()

            print("Measured Renyi relative entropy of states:", info["p_obj"])

        |

        .. testoutput:: native

            Measured Renyi relative entropy of states: 0.8299380360451464

    .. group-tab:: PICOS

        .. testcode:: picos

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
            P.solve(solver="qics")

            print("Measured Renyi relative entropy of states:", P.value)

        |

        .. testoutput:: picos

            Measured Renyi relative entropy of states: 0.8299380360452833

.. _opper_refs:

References
----------

    1. "Semi-definite optimization of the measured relative entropies of quantum
       states and channels." H. Zixin, and M. M. Wilde. 
       arXiv preprint arXiv:2406.19060 (2024).