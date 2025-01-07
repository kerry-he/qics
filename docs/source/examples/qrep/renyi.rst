Renyi entropy
==========================

Here we provide some examples of how to use **QICS** to solve optimization problems
involving Renyi entropies.

Renyi mutual information
--------------------------------

Let :math:`D_\alpha` denote the :math:`\alpha`-Renyi entropy. The Renyi mutual 
information for a bipartite quantum state :math:`\rho_{AB}` is

.. math::

    I_{\alpha}(A;B)_\rho \quad=\quad \min_{\sigma_B} \quad 
    D_\alpha(\rho_{AB} \| \rho_A \otimes \sigma_B) \quad 
    \text{subj. to} \quad \text{tr}[\sigma_B] = 1.,

In :ref:`[1] <renyi_refs>`, it was shown that the optimal :math:`\sigma_B^*` is given by

.. math::

    \sigma_B^* = \frac{(\text{tr}_A((\rho_A^{1-\alpha}\otimes\mathbb{I})\rho_{AB}^\alpha))^{1/\alpha}}
    {\text{tr}[(\text{tr}_A((\rho_A^{1-\alpha}\otimes\mathbb{I})\rho_{AB}^\alpha))^{1/\alpha}]}.

We can verify this by using **QICS** as follows.

.. tabs::

    .. code-tab:: python Native

        import numpy as np

        import qics
        import qics.quantum
        from qics.quantum.random import density_matrix
        from qics.vectorize import lin_to_mat, vec_dim, mat_to_vec, vec_to_mat

        np.random.seed(1)

        n = 4
        m = 4
        N = n * m

        vN = vec_dim(N, iscomplex=True)
        cm = vec_dim(m, compact=True, iscomplex=True)

        # Define random problem data
        rhoAB = density_matrix(N, iscomplex=True)
        rhoA = qics.quantum.p_tr(rhoAB, (n, m), 1)

        # Model problem using primal variables (t, sigB)
        # Define objective function
        c = np.vstack((np.array([[1.0]]), np.zeros((cm, 1))))

        # Build linear constraint tr[X] = 1
        trace = lin_to_mat(lambda X: np.trace(X), (m, 1), compact=(True, True), iscomplex=True)
        A = np.block([[0.0, trace]])
        b = np.array([[1.0]])

        # Build conic linear constraints
        rhoA_kron = lin_to_mat(lambda X: np.kron(rhoA, X), (m, N), compact=(True, False), 
                               iscomplex=True)

        G = np.block([
            [-1.0,              np.zeros((1, cm)) ],   # t_sre = t
            [np.zeros((vN, 1)), np.zeros((vN, cm))],   # X_sre = rhoAB
            [np.zeros((vN, 1)), -rhoA_kron         ],  # Y_sre = rhoA x sigB
        ])  # fmt: skip

        h = np.block([
            [0.0              ], 
            [mat_to_vec(rhoAB)], 
            [np.zeros((vN, 1))], 
        ])  # fmt: skip

        # Define cones to optimize over
        alpha = 0.5
        cones = [qics.cones.TrRenyiEntr(N, alpha, True)]

        # Initialize model and solver objects
        model = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
        solver = qics.Solver(model, verbose=3)

        # Solve problem
        info = solver.solve()


        # Check if solution satisfies identity
        def mpower(X, p):
            D, U = np.linalg.eigh(X)
            return (U * np.power(D, p)) @ U.conj().T

        sigB_numeric = vec_to_mat(info["x_opt"][1:], iscomplex=True, compact=True)

        temp = np.kron(mpower(rhoA, 1-alpha), np.eye(m)) @ mpower(rhoAB, alpha)
        temp = mpower(qics.quantum.p_tr(temp, (n, m), 0), 1 / alpha)
        sigB_analytic = temp / np.trace(temp)

        print("Analytic solution:")
        print(np.round(sigB_numeric, 3))
        print("Numerical solution:")
        print(np.round(sigB_analytic, 3))

    .. code-tab:: python PICOS

.. code-block:: none

    Analytic solution:
    [[ 0.245+0.j    -0.025-0.059j  0.026-0.01j ]
     [-0.025+0.059j  0.467+0.j    -0.018+0.032j]
     [ 0.026+0.01j  -0.018-0.032j  0.288+0.j   ]]
    Numerical solution:
    [[ 0.245-0.j    -0.025-0.059j  0.026-0.01j ]
     [-0.025+0.059j  0.467-0.j    -0.018+0.032j]
     [ 0.026+0.01j  -0.018-0.032j  0.288+0.j   ]]


Sandwiched Renyi mutual information
-----------------------------------

Similarly, let :math:`\hat{D}_\alpha` denote the sandwiched :math:`\alpha`-Renyi entropy. The sandwiched Renyi mutual 
information for a bipartite quantum state :math:`\rho_{AB}` is

.. math::

    \hat{I}_{\alpha}(A;B)_\rho \quad=\quad \min_{\sigma_B} \quad 
    \hat{D}_\alpha(\rho_{AB} \| \rho_A \otimes \sigma_B) \quad 
    \text{subj. to} \quad \text{tr}[\sigma_B] = 1.,

In :ref:`[1] <renyi_refs>`, it was shown that the optimal :math:`\sigma_B^*` satisfies

.. math::

    \sigma_B^* = \frac{\text{tr}_A(((\rho_A \otimes \sigma_B^*)^{\frac{1-\alpha}{2\alpha}} \rho_{AB} (\rho_A \otimes \sigma_B^*)^{\frac{1-\alpha}{2\alpha}})^{1/\alpha})}
    {\text{tr}[((\rho_A \otimes \sigma_B^*)^{\frac{1-\alpha}{2\alpha}} \rho_{AB} (\rho_A \otimes \sigma_B^*)^{\frac{1-\alpha}{2\alpha}})^{1/\alpha}]}.

We can verify this by using **QICS** as follows.

.. tabs::

    .. code-tab:: python Native

        import numpy as np

        import qics
        import qics.quantum
        from qics.quantum.random import density_matrix
        from qics.vectorize import lin_to_mat, vec_dim, mat_to_vec, vec_to_mat

        np.random.seed(1)

        n = 4
        m = 4
        N = n * m

        vN = vec_dim(N, iscomplex=True)
        cm = vec_dim(m, compact=True, iscomplex=True)

        # Define random problem data
        rhoAB = density_matrix(N, iscomplex=True)
        rhoA = qics.quantum.p_tr(rhoAB, (n, m), 1)

        # Model problem using primal variables (t, sigB)
        # Define objective function
        c = np.vstack((np.array([[1.0]]), np.zeros((cm, 1))))

        # Build linear constraint tr[X] = 1
        trace = lin_to_mat(lambda X: np.trace(X), (m, 1), compact=(True, True), iscomplex=True)
        A = np.block([[0.0, trace]])
        b = np.array([[1.0]])

        # Build conic linear constraints
        rhoA_kron = lin_to_mat(lambda X: np.kron(rhoA, X), (m, N), compact=(True, False), 
                               iscomplex=True)

        G = np.block([
            [-1.0,              np.zeros((1, cm)) ],   # t_sre = t
            [np.zeros((vN, 1)), np.zeros((vN, cm))],   # X_sre = rhoAB
            [np.zeros((vN, 1)), -rhoA_kron         ],  # Y_sre = rhoA x sigB
        ])  # fmt: skip

        h = np.block([
            [0.0              ], 
            [mat_to_vec(rhoAB)], 
            [np.zeros((vN, 1))], 
        ])  # fmt: skip

        # Define cones to optimize over
        alpha = 0.5
        cones = [qics.cones.TrSandRenyiEntr(N, alpha, True)]

        # Initialize model and solver objects
        model = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
        solver = qics.Solver(model, verbose=3)

        # Solve problem
        info = solver.solve()


        # Check if solution satisfies identity
        def mpower(X, p):
            D, U = np.linalg.eigh(X)
            return (U * np.power(D, p)) @ U.conj().T

        LHS = vec_to_mat(info["x_opt"][1:], iscomplex=True, compact=True)

        temp = mpower(np.kron(rhoA, LHS), (1-alpha)/(2*alpha))
        temp = mpower(temp @ rhoAB @ temp, alpha)
        RHS = qics.quantum.p_tr(temp, (n, m), 0) / np.trace(temp)

        print("LHS of identity is:")
        print(np.round(LHS, 3))
        print("RHS of identity is:")
        print(np.round(RHS, 3))

    .. code-tab:: python PICOS

.. code-block:: none

    LHS of identity is:
    [[ 0.234+0.j    -0.031-0.076j  0.03 -0.006j]
     [-0.031+0.076j  0.497+0.j    -0.022+0.032j]
     [ 0.03 +0.006j -0.022-0.032j  0.269+0.j   ]]
    RHS of identity is:
    [[ 0.234-0.j    -0.031-0.076j  0.03 -0.006j]
     [-0.031+0.076j  0.497+0.j    -0.022+0.032j]
     [ 0.03 +0.006j -0.022-0.032j  0.269-0.j   ]]

.. _renyi_refs:

References
----------

    1. M. Hayashi and M. Tomamichel, “Correlation detection and an operational 
    interpretation of the Renyi mutual information,” Journal of Mathematical Physics, 
    vol. 57, no. 10, 2016.
