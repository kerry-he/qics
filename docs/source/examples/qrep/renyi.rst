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
        alpha = 0.5

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
            [-1.0,              np.zeros((1, cm)) ],  # t_re = t
            [0.0,               np.zeros((1, cm)) ],  # u_re = 1 
            [np.zeros((vN, 1)), np.zeros((vN, cm))],  # X_re = rhoAB
            [np.zeros((vN, 1)), -rhoA_kron        ],  # Y_re = rhoA x sigB
        ])

        h = np.block([
            [0.0              ],
            [1.0              ],
            [mat_to_vec(rhoAB)], 
            [np.zeros((vN, 1))], 
        ])

        # Define cones to optimize over
        cones = [qics.cones.RenyiEntr(N, alpha, True)]

        # Initialize model and solver objects
        model = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
        solver = qics.Solver(model)

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

        import numpy as np

        import picos
        import qics

        np.random.seed(1)

        # Generate problem data
        n = 4
        m = 4
        N = n * m        
        alpha = 0.75

        rhoAB = qics.quantum.random.density_matrix(N, iscomplex=True)
        rhoA = qics.quantum.p_tr(rhoAB, (n, m), 1)

        # Define problem
        P = picos.Problem()
        sigB = picos.HermitianVariable("sigB", m)

        obj = picos.renyientr(rhoAB, rhoA @ sigB, alpha)
        P.set_objective("min", obj)
        P.add_constraint(picos.trace(sigB) == 1)

        # Solve problem
        P.solve(solver="qics")


        # Check if solution satisfies identity
        def mpower(X, p):
            D, U = np.linalg.eigh(X)
            return (U * np.power(D, p)) @ U.conj().T

        sigB_numeric = sigB

        temp = np.kron(mpower(rhoA, 1-alpha), np.eye(m)) @ mpower(rhoAB, alpha)
        temp = mpower(qics.quantum.p_tr(temp, (n, m), 0), 1 / alpha)
        sigB_analytic = temp / np.trace(temp)

        print("Analytic solution:")
        print(np.round(sigB_numeric, 3))
        print("Numerical solution:")
        print(np.round(sigB_analytic, 3))

.. code-block:: none

    Analytic solution:
    [[ 0.257+0.j     0.016+0.021j -0.007-0.031j -0.008+0.026j]
     [ 0.016-0.021j  0.247+0.j    -0.02 +0.05j  -0.033+0.025j]
     [-0.007+0.031j -0.02 -0.05j   0.251+0.j    -0.008+0.017j]
     [-0.008-0.026j -0.033-0.025j -0.008-0.017j  0.245+0.j   ]]
    Numerical solution:
    [[ 0.257+0.j     0.016+0.021j -0.007-0.031j -0.008+0.026j]
     [ 0.016-0.021j  0.247-0.j    -0.02 +0.05j  -0.033+0.025j]
     [-0.007+0.031j -0.02 -0.05j   0.251+0.j    -0.008+0.017j]
     [-0.008-0.026j -0.033-0.025j -0.008-0.017j  0.245+0.j   ]]


Due to monotonicity of the logarithm, when :math:`\alpha\in[0, 1)`, we can equivalently
compute the Renyi mutual information as

.. math::

    I_{\alpha}(A;B)_\rho \quad=\quad \frac{1}{\alpha - 1}
    \log\left( \max_{\text{tr}[\sigma_B] = 1} \Psi_\alpha(\rho_{AB}, 
    \rho_A\otimes\sigma_B) \right),

where 

.. math::

    \Psi_\alpha(X, Y) = \text{tr}[ X^\alpha Y^{1-\alpha} ],

see :class:`~qics.cones.QuasiEntr`. We can check the same identity by minimizing
the trace function :math:`\Psi_\alpha` instead of directly minimizing the Renyi entropy
by using nearly the same code.

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
        alpha = 0.5

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
            [-1.0,              np.zeros((1, cm)) ],  # t_tre = t
            [np.zeros((vN, 1)), np.zeros((vN, cm))],  # X_tre = rhoAB
            [np.zeros((vN, 1)), -rhoA_kron        ],  # Y_tre = rhoA x sigB
        ])

        h = np.block([
            [0.0              ],
            [mat_to_vec(rhoAB)], 
            [np.zeros((vN, 1))], 
        ])

        # Define cones to optimize over
        cones = [qics.cones.QuasiEntr(N, alpha, True)]

        # Initialize model and solver objects
        model = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
        solver = qics.Solver(model)

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

        import numpy as np

        import picos
        import qics

        np.random.seed(1)

        # Generate problem data
        n = 4
        m = 4
        N = n * m        
        alpha = 0.75

        rhoAB = qics.quantum.random.density_matrix(N, iscomplex=True)
        rhoA = qics.quantum.p_tr(rhoAB, (n, m), 1)

        # Define problem
        P = picos.Problem()
        sigB = picos.HermitianVariable("sigB", m)

        obj = picos.quasientr(rhoAB, rhoA @ sigB, alpha)
        P.set_objective("max", obj)
        P.add_constraint(picos.trace(sigB) == 1)

        # Solve problem
        P.solve(solver="qics")


        # Check if solution satisfies identity
        def mpower(X, p):
            D, U = np.linalg.eigh(X)
            return (U * np.power(D, p)) @ U.conj().T

        sigB_numeric = sigB

        temp = np.kron(mpower(rhoA, 1-alpha), np.eye(m)) @ mpower(rhoAB, alpha)
        temp = mpower(qics.quantum.p_tr(temp, (n, m), 0), 1 / alpha)
        sigB_analytic = temp / np.trace(temp)

        print("Analytic solution:")
        print(np.round(sigB_numeric, 3))
        print("Numerical solution:")
        print(np.round(sigB_analytic, 3))

.. code-block:: none

    Analytic solution:
    [[ 0.257+0.j     0.016+0.021j -0.007-0.031j -0.008+0.026j]
     [ 0.016-0.021j  0.247+0.j    -0.02 +0.05j  -0.033+0.025j]
     [-0.007+0.031j -0.02 -0.05j   0.251+0.j    -0.008+0.017j]
     [-0.008-0.026j -0.033-0.025j -0.008-0.017j  0.245+0.j   ]]
    Numerical solution:
    [[ 0.257+0.j     0.016+0.021j -0.007-0.031j -0.008+0.026j]
     [ 0.016-0.021j  0.247-0.j    -0.02 +0.05j  -0.033+0.025j]
     [-0.007+0.031j -0.02 -0.05j   0.251+0.j    -0.008+0.017j]
     [-0.008-0.026j -0.033-0.025j -0.008-0.017j  0.245+0.j   ]]

The advantage of optimizing the trace function is that :math:`\Psi_\alpha` is 
jointly concave for :math:`\alpha\in[1/2, 1]`, and jointly convex for 
:math:`\alpha\in[-1, 0] \cup [1, 2]`, whereas :math:`D_\alpha` is jointly convex for
:math:`\alpha\in[0, 1)`, but is neither convex nor concave for
:math:`\alpha\in[-1, 0) \cup (1, 2]`. Therefore, QICS supports :math:`\Psi_\alpha`
for a wider range of :math:`\alpha`. However, sometimes it is not straightforward to 
reformulate an optimization problem involving the Renyi entropy into an equivalent
convex problem involving the trace function.


Sandwiched Renyi mutual information
-----------------------------------

Similarly, let :math:`\hat{D}_\alpha` denote the sandwiched :math:`\alpha`-Renyi entropy. The sandwiched Renyi mutual 
information for a bipartite quantum state :math:`\rho_{AB}` is

.. math::

    \hat{I}_{\alpha}(A;B)_\rho \quad=\quad \min_{\sigma_B} \quad 
    \hat{D}_\alpha(\rho_{AB} \| \rho_A \otimes \sigma_B) \quad 
    \text{subj. to} \quad \text{tr}[\sigma_B] = 1.

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
        alpha = 0.75

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
        ])

        h = np.block([
            [0.0              ], 
            [mat_to_vec(rhoAB)], 
            [np.zeros((vN, 1))], 
        ])

        # Define cones to optimize over
        cones = [qics.cones.SandQuasiEntr(N, alpha, True)]

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

        import numpy as np

        import picos
        import qics

        np.random.seed(1)

        # Generate problem data
        n = 4
        m = 4
        N = n * m        
        alpha = 0.75

        rhoAB = qics.quantum.random.density_matrix(N, iscomplex=True)
        rhoA = qics.quantum.p_tr(rhoAB, (n, m), 1)

        # Define problem
        P = picos.Problem()
        sigB = picos.HermitianVariable("sigB", m)

        obj = picos.renyientr(rhoAB, rhoA @ sigB, alpha)
        P.set_objective("min", obj)
        P.add_constraint(picos.trace(sigB) == 1)

        # Solve problem
        P.solve(solver="qics")


        # Check if solution satisfies identity
        def mpower(X, p):
            D, U = np.linalg.eigh(X)
            return (U * np.power(D, p)) @ U.conj().T

        LHS = sigB

        temp = mpower(np.kron(rhoA, LHS), (1-alpha)/(2*alpha))
        temp = mpower(temp @ rhoAB @ temp, alpha)
        RHS = qics.quantum.p_tr(temp, (n, m), 0) / np.trace(temp)

        print("LHS of identity is:")
        print(np.round(LHS, 3))
        print("RHS of identity is:")
        print(np.round(RHS, 3))

.. code-block:: none

    LHS of identity is:
    [[ 0.258+0.j     0.01 +0.019j -0.009-0.026j -0.007+0.023j]
     [ 0.01 -0.019j  0.247+0.j    -0.018+0.046j -0.031+0.025j]
     [-0.009+0.026j -0.018-0.046j  0.246+0.j    -0.008+0.013j]
     [-0.007-0.023j -0.031-0.025j -0.008-0.013j  0.249+0.j   ]]
    RHS of identity is:
    [[ 0.258+0.j     0.01 +0.019j -0.009-0.026j -0.007+0.023j]
     [ 0.01 -0.019j  0.247-0.j    -0.018+0.046j -0.031+0.025j]
     [-0.009+0.026j -0.018-0.046j  0.246+0.j    -0.008+0.013j]
     [-0.007-0.023j -0.031-0.025j -0.008-0.013j  0.249+0.j   ]]

Just as for the Renyi entropy, due to monotonicity of the logarithm, when
:math:`\alpha\in[1/2, 1)`, we can equivalently compute the sandwiched Renyi mutual 
information as

.. math::

    \hat{I}_{\alpha}(A;B)_\rho \quad=\quad \frac{1}{\alpha - 1}
    \log\left( \max_{\text{tr}[\sigma_B] = 1} \hat{\Psi}_\alpha(\rho_{AB}, 
    \rho_A\otimes\sigma_B) \right),

where 

.. math::

    \hat{\Psi}_\alpha(X, Y) = \text{tr}\!\left[ \left(Y^\frac{1-\alpha}{2\alpha} X
    Y^\frac{1-\alpha}{2\alpha} \right)^\alpha \right].

see :class:`~qics.cones.SandQuasiEntr`. We can check the same identity by minimizing
the trace function :math:`\\hat{Psi}_\alpha` instead of directly minimizing the 
sandwiched Renyi entropy by using nearly the same code.

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
        alpha = 0.75

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
            [-1.0,              np.zeros((1, cm)) ],  # t_tre = t
            [np.zeros((vN, 1)), np.zeros((vN, cm))],  # X_tre = rhoAB
            [np.zeros((vN, 1)), -rhoA_kron        ],  # Y_tre = rhoA x sigB
        ])

        h = np.block([
            [0.0              ],
            [mat_to_vec(rhoAB)], 
            [np.zeros((vN, 1))], 
        ])

        # Define cones to optimize over
        cones = [qics.cones.SandQuasiEntr(N, alpha, True)]

        # Initialize model and solver objects
        model = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
        solver = qics.Solver(model)

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

        import numpy as np

        import picos
        import qics

        np.random.seed(1)

        # Generate problem data
        n = 4
        m = 4
        N = n * m        
        alpha = 0.75

        rhoAB = qics.quantum.random.density_matrix(N, iscomplex=True)
        rhoA = qics.quantum.p_tr(rhoAB, (n, m), 1)

        # Define problem
        P = picos.Problem()
        sigB = picos.HermitianVariable("sigB", m)

        obj = picos.sandquasientr(rhoAB, rhoA @ sigB, alpha)
        P.set_objective("max", obj)
        P.add_constraint(picos.trace(sigB) == 1)

        # Solve problem
        P.solve(solver="qics")


        # Check if solution satisfies identity
        def mpower(X, p):
            D, U = np.linalg.eigh(X)
            return (U * np.power(D, p)) @ U.conj().T

        LHS = sigB

        temp = mpower(np.kron(rhoA, LHS), (1-alpha)/(2*alpha))
        temp = mpower(temp @ rhoAB @ temp, alpha)
        RHS = qics.quantum.p_tr(temp, (n, m), 0) / np.trace(temp)

        print("LHS of identity is:")
        print(np.round(LHS, 3))
        print("RHS of identity is:")
        print(np.round(RHS, 3))

.. code-block:: none

    LHS of identity is:
    [[ 0.258+0.j     0.01 +0.019j -0.009-0.026j -0.007+0.023j]
     [ 0.01 -0.019j  0.247+0.j    -0.018+0.046j -0.031+0.025j]
     [-0.009+0.026j -0.018-0.046j  0.246+0.j    -0.008+0.013j]
     [-0.007-0.023j -0.031-0.025j -0.008-0.013j  0.249+0.j   ]]
    RHS of identity is:
    [[ 0.258+0.j     0.01 +0.019j -0.009-0.026j -0.007+0.023j]
     [ 0.01 -0.019j  0.247-0.j    -0.018+0.046j -0.031+0.025j]
     [-0.009+0.026j -0.018-0.046j  0.246+0.j    -0.008+0.013j]
     [-0.007-0.023j -0.031-0.025j -0.008-0.013j  0.249+0.j   ]]

The advantage of optimizing the trace function is that :math:`\hat{\Psi}_\alpha` is jointly
concave for :math:`\alpha\in[1/2, 1]`, and jointly convex for :math:`\alpha\in[1, 2]`,
whereas :math:`D_\alpha` is jointly convex for :math:`\alpha\in[1/2, 1)`, but is neither
convex nor concave for :math:`\alpha\in(1, 2]`. Therefore, QICS supports 
:math:`\hat{\Psi}_\alpha` for a wider range of :math:`\alpha`. However, sometimes it is 
not straightforward to reformulate an optimization problem involving the sandwiched  
Renyi entropy into an equivalent convex problem involving the trace function.


.. _renyi_refs:

References
----------

    1. M. Hayashi and M. Tomamichel, “Correlation detection and an operational 
    interpretation of the Renyi mutual information,” Journal of Mathematical Physics, 
    vol. 57, no. 10, 2016.
