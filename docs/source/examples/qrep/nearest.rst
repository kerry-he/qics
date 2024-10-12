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

    \text{subj. to} &&& Y_{ii} = 1 \qquad i=1,\ldots,n

    &&& Y \succeq 0.

We show how this problem can be solved using QICS below.

.. tabs::

    .. code-tab:: python Native

        import numpy as np

        import qics
        from qics.vectorize import vec_dim, mat_to_vec, eye

        np.random.seed(1)

        n = 5

        vn = vec_dim(n)
        cn = vec_dim(n, compact=True)

        # Generate random positive semidefinite matrix C
        C = np.random.randn(n, n)
        C = C @ C.T / n
        C_cvec = mat_to_vec(C, compact=True)

        # Model problem using primal variables (t, X, Y)
        # Define objective function
        c = np.block([[1.0], [np.zeros((vn, 1))], [np.zeros((vn, 1))]])

        # Build linear constraints
        diag = np.zeros((n, vn))
        diag[np.arange(n), np.arange(0, vn, n + 1)] = 1.

        A = np.block([
            [np.zeros((cn, 1)), eye(n),            np.zeros((cn, vn))],  # X = C
            [np.zeros((n, 1)),  np.zeros((n, vn)), diag              ]   # Yii = 1
        ])  # fmt: skip

        b = np.block([[C_cvec], [np.ones((n, 1))]])

        # Define cones to optimize over
        cones = [qics.cones.QuantRelEntr(n)]

        # Initialize model and solver objects
        model = qics.Model(c=c, A=A, b=b, cones=cones)
        solver = qics.Solver(model)

        # Solve problem
        info = solver.solve()

        Y_opt = info["s_opt"][0][2]
        print("The nearest correlation matrix to")
        print(C)
        print("is")
        print(Y_opt)        

    .. code-tab:: python PICOS

        import numpy as np

        import picos

        np.random.seed(1)

        n = 5

        # Generate random matrix C
        C = np.random.randn(n, n)
        C = C @ C.T / n

        # Define problem
        P = picos.Problem()
        Y = picos.SymmetricVariable("Y", n)

        P.set_objective("min", picos.quantrelentr(C, Y))
        P.add_constraint(picos.maindiag(Y) == 1)

        # Solve problem
        P.solve(solver="qics")

        Y_opt = Y.np
        print("The nearest correlation matrix to")
        print(C)
        print("is")
        print(Y_opt)

.. code-block:: none

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

    \text{subj. to} &&& \text{tr}[\sigma_{AB}] = 1
    
    &&& \mathcal{T}_B(\sigma_{AB}) \succeq 0 

    &&& \sigma_{AB} \succeq 0.

We show how we can solve this problem in QICS below.

.. tabs::

    .. code-tab:: python Native

        import numpy as np

        import qics
        from qics.quantum import partial_transpose
        from qics.quantum.random import density_matrix
        from qics.vectorize import eye, lin_to_mat, mat_to_vec, vec_dim

        np.random.seed(1)

        n1 = 2
        n2 = 3
        N = n1 * n2

        vN = vec_dim(N, iscomplex=True)
        cN = vec_dim(N, iscomplex=True, compact=True)

        # Generate random quantum state
        C = density_matrix(N, iscomplex=True)
        C_cvec = mat_to_vec(C, compact=True)

        # Model problem using primal variables (t, X, Y, Z)
        # Define objective function
        c = np.block([[1.0], [np.zeros((vN, 1))], [np.zeros((vN, 1))], [np.zeros((vN, 1))]])

        # Build linear constraints
        trace = lin_to_mat(lambda X: np.trace(X), (N, 1), True)
        ptranspose = lin_to_mat(lambda X: partial_transpose(X, (n1, n2), 1), (N, N), True)

        A = np.block([
            [np.zeros((cN, 1)), eye(N, True),       np.zeros((cN, vN)), np.zeros((cN, vN))],  # X = C
            [np.zeros((1, 1)),  np.zeros((1, vN)),  trace,              np.zeros((1, vN)) ],  # tr[Y] = 1
            [np.zeros((cN, 1)), np.zeros((cN, vN)), ptranspose,         -eye(N, True)     ]   # T2(Y) = Z
        ])  # fmt: skip

        b = np.block([[C_cvec], [1.0], [np.zeros((cN, 1))]])

        # Input into model and solve
        cones = [
            qics.cones.QuantRelEntr(N, iscomplex=True),     # (t, X, Y) ∈ QRE
            qics.cones.PosSemidefinite(N, iscomplex=True),  # Z = T2(Y) ⪰ 0
        ]

        # Initialize model and solver objects
        model = qics.Model(c=c, A=A, b=b, cones=cones)
        solver = qics.Solver(model)

        # Solve problem
        info = solver.solve()

    .. code-tab:: python PICOS

        import numpy as np

        import picos
        import qics

        np.random.seed(1)

        n1 = 2
        n2 = 3
        N  = n1 * n2

        # Generate random quantum state
        C = qics.quantum.random.density_matrix(N, iscomplex=True)

        # Define problem
        P = picos.Problem()
        Y = picos.HermitianVariable("Y", N)

        P.set_objective("min", picos.quantrelentr(C, Y))
        P.add_constraint(picos.trace(Y) == 1.0)
        P.add_constraint(picos.partial_transpose(Y, subsystems=1, dimensions=(n1, n2)) >> 0)

        # Solve problem
        P.solve(solver="qics", verbosity=1)

Bregman projection
------------------

A Bregman projection is a generalization of a Euclidean projection, which is
commonly used in first-order optimization algorithms called Bregman proximal
methods. As an example, the Bregman projection corresponding to the quantum
relative entropy (see, e.g., :ref:`[2] <nearest_refs>`) of a point 
:math:`Y\in\mathbb{H}^n_{+}` onto the set of density matrices is the solution to

.. math::

    \min_{X \in \mathbb{H}^n} &&& S( X \| Y ) - \text{tr}[X - Y]

    \text{subj. to} &&& \text{tr}[X] = 1

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

    .. code-tab:: python Native

        import numpy as np
        import scipy as sp

        import qics
        from qics.vectorize import lin_to_mat, vec_dim, mat_to_vec

        np.random.seed(1)

        n = 5
        vn = vec_dim(n, iscomplex=True)

        # Generate random positive semidefinite matrix Y to project
        Y = np.random.randn(n, n) + np.random.randn(n, n)*1j
        Y = Y @ Y.conj().T
        trY = np.trace(Y).real

        # Model problem using primal variables (t, u, X)
        # Define objective function
        c = np.block([[1.0], [0.0], [mat_to_vec(-sp.linalg.logm(Y) - np.eye(n))]])

        # Build linear constraints
        trace = lin_to_mat(lambda X: np.trace(X), (n, 1), iscomplex=True)

        A = np.block([
            [0.0, 1.0, np.zeros((1, vn))],  # u = 1
            [0.0, 0.0, trace            ]   # tr[X] = 1
        ])

        b = np.array([[1.0], [1.0]])

        # Define cones to optimize over
        cones = [qics.cones.QuantEntr(n, iscomplex=True)]

        # Initialize model and solver objects
        model = qics.Model(c=c, A=A, b=b, cones=cones, offset=trY)
        solver = qics.Solver(model)

        # Solve problem
        info = solver.solve()

        analytic_X_opt = Y / trY
        numerical_X_opt = info["s_opt"][0][2]

        print("Analytic solution:")
        print(np.round(analytic_X_opt, 3))
        print("Numerical solution:")
        print(np.round(numerical_X_opt, 3))

    .. code-tab:: python PICOS

        import numpy as np

        import picos
        import qics

        np.random.seed(1)

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

        analytic_X_opt = Y / trY
        numerical_X_opt = X.np

        print("Analytic solution:")
        print(np.round(analytic_X_opt, 3))
        print("Numerical solution:")
        print(np.round(numerical_X_opt, 3))

.. code-block:: none

    Analytic solution:
    [[ 0.147+0.j    -0.083+0.043j  0.108+0.018j -0.005+0.065j -0.085+0.042j]
     [-0.083-0.043j  0.241+0.j    -0.186+0.029j  0.049+0.022j  0.046-0.03j ]
     [ 0.108-0.018j -0.186-0.029j  0.266+0.j     0.071-0.015j -0.053+0.038j]
     [-0.005-0.065j  0.049-0.022j  0.071+0.015j  0.14 +0.j    -0.013+0.005j]
     [-0.085-0.042j  0.046+0.03j  -0.053-0.038j -0.013-0.005j  0.205+0.j   ]]
    Numerical solution:
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