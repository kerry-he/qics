Nearest matrix
==============

A common optimization problem that arises is to find the nearest matrix to
another fixed matrix with respect to the quantum relative entropy. We show how
some of these problems can be solved in **QICS** below.

Nearest correlation matrix
---------------------------

We first consider a toy example, in which we are interested in finding the 
closest correlation matrix to a given positive semidefinite matrix 
:math:`C\in\mathbb{S}^n` in the quantum relative entropy sense.

Correlation matrices are characterized by being a real positive semidefinite 
matrices with diagonal entries all equal to one. Therefore, the closest 
correlation matrix to :math:`C` can be found by solving the following problem

.. math::

    \min_{Y \in \mathbb{S}^n} &&& S( C \| Y )

    \text{s.t.} &&& Y_{ii} = 1 \qquad i=1,\ldots,n

    &&& Y \succeq 0.

We show how this problem can be solved using QICS below.

.. tabs::

    .. group-tab:: Native

        .. testcode:: nearest-native

            import numpy
            import qics

            numpy.random.seed(1)

            n = 5
            vn = qics.vectorize.vec_dim(n)
            sn = qics.vectorize.vec_dim(n, compact=True)

            # Generate random matrix C
            C = numpy.random.randn(n, n)
            C = C @ C.T / n

            # Define objective function
            c = numpy.zeros((1 + 2*vn, 1))
            c[0] = 1.

            # Build linear constraints
            # X = C
            A1 = numpy.hstack((
                numpy.zeros((sn, 1)), 
                qics.vectorize.eye(n), 
                numpy.zeros((sn, vn))
            ))
            b1 = qics.vectorize.mat_to_vec(C, compact=True)
            # Yii = 1
            A2 = numpy.zeros((n, 1 + 2*vn))
            A2[numpy.arange(n), numpy.arange(1 + vn, 1 + 2*vn, n+1)] = 1.
            b2 = numpy.ones((n, 1))

            A = numpy.vstack((A1, A2))
            b = numpy.vstack((b1, b2))

            # Define cones to optimize over
            cones = [qics.cones.QuantRelEntr(n)]

            # Initialize model and solver objects
            model = qics.Model(c=c, A=A, b=b, cones=cones)
            solver = qics.Solver(model, verbose=0)

            # Solve problem
            info = solver.solve()

            print("The nearest correlation matrix to\n")
            print(C)
            print("\nis\n")
            print(info["s_opt"][0][2])

        |

        .. testoutput:: nearest-native

            The nearest correlation matrix to

            [[ 1.03838024 -0.9923943   1.03976304 -0.1516761  -0.54476511]
             [-0.9923943   1.81697119 -1.42389728  0.55339008  0.7559633 ]
             [ 1.03976304 -1.42389728  1.58376462 -0.0650662  -0.6859653 ]
             [-0.1516761   0.55339008 -0.0650662   0.47031665  0.1535909 ]
             [-0.54476511  0.7559633  -0.6859653   0.1535909   0.87973255]]

            is

            [[ 1.         -0.68153866  0.77530944 -0.15666279 -0.52667092]
             [-0.68153866  1.         -0.7916631   0.5004458   0.54809203]
             [ 0.77530944 -0.7916631   1.          0.05716203 -0.52919617]
             [-0.15666279  0.5004458   0.05716203  1.          0.16929971]
             [-0.52667092  0.54809203 -0.52919617  0.16929971  1.        ]]

    .. group-tab:: PICOS

        .. testcode:: nearest-picos

            import numpy
            import picos

            numpy.random.seed(1)

            n = 5

            # Generate random matrix C
            C = numpy.random.randn(n, n)
            C = C @ C.T / n

            # Define problem
            P = picos.Problem()
            Y = picos.SymmetricVariable("Y", n)

            P.set_objective("min", picos.quantrelentr(C, Y))
            P.add_constraint(picos.maindiag(Y) == 1)

            # Solve problem
            P.solve(solver="qics")

            print("The nearest correlation matrix to\n")
            print(C)
            print("\nis\n")
            print(Y.np)

        |

        .. testoutput:: nearest-picos

            The nearest correlation matrix to

            [[ 1.03838024 -0.9923943   1.03976304 -0.1516761  -0.54476511]
             [-0.9923943   1.81697119 -1.42389728  0.55339008  0.7559633 ]
             [ 1.03976304 -1.42389728  1.58376462 -0.0650662  -0.6859653 ]
             [-0.1516761   0.55339008 -0.0650662   0.47031665  0.1535909 ]
             [-0.54476511  0.7559633  -0.6859653   0.1535909   0.87973255]]

            is

            [[ 1.         -0.68153866  0.77530944 -0.15666279 -0.52667092]
             [-0.68153866  1.         -0.7916631   0.5004458   0.54809203]
             [ 0.77530944 -0.7916631   1.          0.05716203 -0.52919618]
             [-0.15666279  0.5004458   0.05716203  1.          0.16929971]
             [-0.52667092  0.54809203 -0.52919618  0.16929971  1.        ]]

Relative entropy of entanglement
--------------------------------

Entanglement is an important resource in quantum information theory, and 
therefore it is often useful to characterize the amount of entanglement 
possessed by a quantum state. This can be characterized by the distance (in the
quantum relative entropy sense) between a given bipartite state and the set of
separable states. 

In general, the set of separable states is NP-hard to describe. Therefore, it is
common to estimate the set of separable states using the positive partial 
transpose (PPT) criteria :ref:`[1] <nearest_refs>`, i.e., if a quantum state 
:math:`\rho_{AB} \in \mathbb{H}^{n_An_B}` is separable, then it must be a member of

.. math::

    \mathsf{PPT}=\{\rho_{AB}\in\mathbb{H}^{n_An_B}:T_B(\rho_{AB})\succeq 0\},

where :math:`\mathcal{T}_B` denotes the partial transpose with respect to
subsystem :math:`B`. Note that in general, the PPT crieria is not a sufficient
condition for separability, i.e., there exists entangled quantum states which
also satisfy the PPT criteria. However, it is a sufficient condition when
:math:`n_A=n_B=2`, or :math:`n_A=2, n_B=3`.

Given this, the relative entropy of entanglement of a quantum state 
:math:`\rho_{AB}` is given by

.. math::

    \min_{\sigma_{AB} \in \mathbb{H}^{n_An_B}} &&& S( \rho_{AB} \| \sigma_{AB} )

    \text{s.t.} &&& \text{tr}[\sigma_{AB}] = 1
    
    &&& \mathcal{T}_B(\sigma_{AB}) \succeq 0 

    &&& \sigma_{AB} \succeq 0.

We show how we can solve this problem in QICS below.

.. tabs::

    .. group-tab:: Native

        .. testcode:: ree-native

            import numpy
            import qics

            numpy.random.seed(1)

            n1 = 2
            n2 = 3
            N  = n1 * n2

            # Generate random (complex) quantum state
            C = qics.quantum.random.density_matrix(N, iscomplex=True)

            # Define objective function
            ct = numpy.array(([[1.]]))
            cX = numpy.zeros((2*N*N, 1))
            cY = numpy.zeros((2*N*N, 1))
            cZ = numpy.zeros((2*N*N, 1))
            c  = numpy.vstack((ct, cX, cY, cZ))

            # Build linear constraints
            # X = C
            sN = qics.vectorize.vec_dim(N, iscomplex=True, compact=True)
            A1 = numpy.hstack((
                numpy.zeros((sN, 1)),
                qics.vectorize.eye(N, iscomplex=True),
                numpy.zeros((sN, 2*N*N)),
                numpy.zeros((sN, 2*N*N)),
            ))
            b1 = qics.vectorize.mat_to_vec(C, compact=True)
            # tr[Y] = 1
            A2 = numpy.hstack((
                numpy.zeros((1, 1)),
                numpy.zeros((1, 2*N*N)),
                qics.vectorize.mat_to_vec(numpy.eye(N, dtype=numpy.complex128)).T,
                numpy.zeros((1, 2*N*N))
            ))
            b2 = numpy.array([[1.]])
            # T2(Y) = Z
            p_transpose = qics.vectorize.lin_to_mat(
                lambda X : qics.quantum.partial_transpose(X, (n1, n2), 1),
                (N, N), iscomplex=True
            )
            A3 = numpy.hstack((
                numpy.zeros((sN, 1)),
                numpy.zeros((sN, 2*N*N)),
                p_transpose,
                -qics.vectorize.eye(N, iscomplex=True)
            ))
            b3 = numpy.zeros((sN, 1))

            A = numpy.vstack((A1, A2, A3))
            b = numpy.vstack((b1, b2, b3))

            # Input into model and solve
            cones = [
                qics.cones.QuantRelEntr(N, iscomplex=True),
                qics.cones.PosSemidefinite(N, iscomplex=True)
            ]

            # Initialize model and solver objects
            model = qics.Model(c=c, A=A, b=b, cones=cones)
            solver = qics.Solver(model, verbose=0)

            # Solve problem
            info = solver.solve()

            print("Relative entropy of entanglement:", info["p_obj"])

        |

        .. testoutput:: ree-native

            Relative entropy of entanglement: 0.004838696998726579

    .. group-tab:: PICOS

        .. testcode:: ree-picos

            import numpy
            import picos
            import qics

            numpy.random.seed(1)

            n1 = 2
            n2 = 3
            N  = n1 * n2

            # Generate random (complex) quantum state
            C = qics.quantum.random.density_matrix(N, iscomplex=True)

            # Define problem
            P = picos.Problem()
            Y = picos.HermitianVariable("Y", N)
            
            P.set_objective("min", picos.quantrelentr(C, Y))
            P.add_constraint(picos.trace(Y) == 1.0)
            P.add_constraint(picos.partial_transpose(Y, subsystems=1, dimensions=(n1, n2)) >> 0)

            # Solve problem
            P.solve(solver="qics")

            print("Relative entropy of entanglement:", P.value)

        |

        .. testoutput:: ree-picos

            Relative entropy of entanglement: 0.004838698939471309

Bregman projection
------------------

A Bregman projection is a generalization of a Euclidean projection, which is
commonly used in first-order optimization algorithms called Bregman proximal
methods. As an example, the Bregman projection corresponding to the quantum
relative entropy (see, e.g., :ref:`[2] <nearest_refs>`) of a point 
:math:`Y\in\mathbb{H}^n_{+}` onto the set of density matrices is the solution to

.. math::

    \min_{X \in \mathbb{H}^n} &&& S( X \| Y ) - \text{tr}[X - Y]

    \text{s.t.} &&& \text{tr}[X] = 1

    &&& X \succeq 0.

We can show that the explicit solution to this is given by 
:math:`X=Y/\text{tr}[Y]`, which we can use to validate the solution given by 
QICS.

.. note::

    The Bregman projection problem fixes the second argument of the quantum
    relative entropy, and optimizes over the first argument. This is as opposed
    to the first two examples which fix the first argument and optimize over the
    second. In this case, we can model the problem using
    :class:`qics.cones.QuantEntr`, which allows QICS to solve problems much
    faster than if we modelled the problem using
    :class:`qics.cones.QuantRelEntr`.

.. tabs::

    .. group-tab:: Native

        .. testcode:: bp-native

            import numpy
            import scipy
            import qics

            numpy.random.seed(1)

            n = 5

            # Generate random matrix Y to project
            Y = numpy.random.randn(n, n) + numpy.random.randn(n, n)*1j
            Y = Y @ Y.conj().T
            trY = numpy.trace(Y).real

            # Define objective function
            ct = numpy.array([[1.]])
            cu = numpy.array([[0.]])
            cX = -scipy.linalg.logm(Y) - numpy.eye(n)
            c  = numpy.vstack((ct, cu, qics.vectorize.mat_to_vec(cX)))

            # Build linear constraints
            # u = 1
            A1 = numpy.hstack((numpy.array([[0., 1.]]), numpy.zeros((1, 2*n*n))))
            b1 = numpy.array([[1.]])
            # tr[X] = 1
            A2 = numpy.hstack((
                numpy.array([[0., 0.]]), 
                qics.vectorize.mat_to_vec(numpy.eye(n, dtype=numpy.complex128)).T
            ))
            b2 = numpy.array([[1.]])

            A = numpy.vstack((A1, A2))
            b = numpy.vstack((b1, b2))

            # Define cones to optimize over
            cones = [qics.cones.QuantEntr(n, iscomplex=True)]

            # Initialize model and solver objects
            model = qics.Model(c=c, A=A, b=b, cones=cones, offset=trY)
            solver = qics.Solver(model, verbose=0)

            # Solve problem
            info = solver.solve()

            print("QICS solution:")
            print(numpy.round(info["s_opt"][0][2], 3))
            print("\nAnalytical solution:")
            print(numpy.round(Y / trY, 3))

        |

        .. testoutput:: bp-native

            QICS solution:
            [[ 0.147+0.j    -0.083+0.043j  0.108+0.018j -0.005+0.065j -0.085+0.042j]
             [-0.083-0.043j  0.241+0.j    -0.186+0.029j  0.049+0.022j  0.046-0.03j ]
             [ 0.108-0.018j -0.186-0.029j  0.266+0.j     0.071-0.015j -0.053+0.038j]
             [-0.005-0.065j  0.049-0.022j  0.071+0.015j  0.14 +0.j    -0.013+0.005j]
             [-0.085-0.042j  0.046+0.03j  -0.053-0.038j -0.013-0.005j  0.205+0.j   ]]

            Analytical solution:
            [[ 0.147+0.j    -0.083+0.043j  0.108+0.018j -0.005+0.065j -0.085+0.042j]
             [-0.083-0.043j  0.241+0.j    -0.186+0.029j  0.049+0.022j  0.046-0.03j ]
             [ 0.108-0.018j -0.186-0.029j  0.266+0.j     0.071-0.015j -0.053+0.038j]
             [-0.005-0.065j  0.049-0.022j  0.071+0.015j  0.14 +0.j    -0.013+0.005j]
             [-0.085-0.042j  0.046+0.03j  -0.053-0.038j -0.013-0.005j  0.205+0.j   ]]

    .. group-tab:: PICOS

        .. testcode:: bp-picos

            import numpy
            import scipy
            import picos

            numpy.random.seed(1)

            n = 5

            # Generate random matrix Y to project
            Y = numpy.random.randn(n, n) + numpy.random.randn(n, n)*1j
            Y = Y @ Y.conj().T
            trY = numpy.trace(Y).real
            logY = scipy.linalg.logm(Y)

            # Define problem
            P = picos.Problem()
            X = picos.HermitianVariable("X", n)

            P.set_objective("min", -picos.quantentr(X) - (X | logY + picos.I(n)).real + trY)
            P.add_constraint(picos.trace(X) == 1)

            # Solve problem
            P.solve(solver="qics")

            print("QICS solution:")
            print(numpy.round(X.np, 3))
            print("\nAnalytical solution:")
            print(numpy.round(Y / trY, 3))

        |

        .. testoutput:: bp-picos

            QICS solution:
            [[ 0.147+0.j    -0.083+0.043j  0.108+0.018j -0.005+0.065j -0.085+0.042j]
             [-0.083-0.043j  0.241+0.j    -0.186+0.029j  0.049+0.022j  0.046-0.03j ]
             [ 0.108-0.018j -0.186-0.029j  0.266+0.j     0.071-0.015j -0.053+0.038j]
             [-0.005-0.065j  0.049-0.022j  0.071+0.015j  0.14 +0.j    -0.013+0.005j]
             [-0.085-0.042j  0.046+0.03j  -0.053-0.038j -0.013-0.005j  0.205+0.j   ]]

            Analytical solution:
            [[ 0.147+0.j    -0.083+0.043j  0.108+0.018j -0.005+0.065j -0.085+0.042j]
             [-0.083-0.043j  0.241+0.j    -0.186+0.029j  0.049+0.022j  0.046-0.03j ]
             [ 0.108-0.018j -0.186-0.029j  0.266+0.j     0.071-0.015j -0.053+0.038j]
             [-0.005-0.065j  0.049-0.022j  0.071+0.015j  0.14 +0.j    -0.013+0.005j]
             [-0.085-0.042j  0.046+0.03j  -0.053-0.038j -0.013-0.005j  0.205+0.j   ]]

.. _nearest_refs:

References
----------

    1. “Separability of mixed states: necessary and sufficient conditions,”
       M. Horodecki, P. Horodecki, and R. Horodecki, 
       Physics Letters A, vol. 223, no. 1, pp. 1–8, 1996.

    2. "A Bregman proximal perspective on classical and quantum Blahut-Arimoto 
       algorithms," K. He, J. Saunderson, and H. Fawzi,
       IEEE Transactions on Information Theory, vol. 70, no. 8, pp. 5710-5730, 
       Aug. 2024.