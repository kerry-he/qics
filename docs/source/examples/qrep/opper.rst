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

    &&\text{subj. to} &&& \theta \succeq \omega^{\frac{\alpha}{\alpha-1}}

    &&&&& \omega, \theta \succeq 0,

which we can model using a constraint of the form 
:math:`(\theta, \mathbb{I}, \omega)\in\mathcal{OPE}`. We can solve this in 
**QICS** using :class:`qics.cones.OpPerspecEpi` below.

.. tabs::

    .. code-tab:: python Native

        import numpy as np

        import qics
        from qics.quantum.random import density_matrix
        from qics.vectorize import eye, mat_to_vec, vec_dim

        np.random.seed(1)

        n = 4
        vn = vec_dim(n, iscomplex=True)
        cn = vec_dim(n, iscomplex=True, compact=True)

        # Model problem using primal variables (T, X, Y)
        # Define random problem data
        alpha = 0.25
        rho = density_matrix(n, iscomplex=True)
        sigma = density_matrix(n, iscomplex=True)

        # Define objective function
        c_T = (1 - alpha) * mat_to_vec(sigma)
        c_X = np.zeros((vn, 1))
        c_Y = alpha * mat_to_vec(rho)
        c = np.block([[c_T], [c_X], [c_Y]])

        # Build linear constraint X = I
        A = np.block([[np.zeros((cn, vn)), eye(n, iscomplex=True), np.zeros((cn, vn))]])
        b = mat_to_vec(np.eye(n, dtype=np.complex128), compact=True)

        # Define cones to optimize over
        cones = [qics.cones.OpPerspecEpi(n, alpha / (alpha - 1), iscomplex=True)]

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
        P.solve(solver="qics", verbosity=1)

D-optimal design
-------------------------

In :ref:`[2] <opper_refs>`, a matrix perspective reformulation technique is proposed to
obtain a relaxation of the D-optimal design problem by solving the following conic
optimization problem

.. math::

    \max_{z\in\mathbb{R}^m, Y\in\mathbb{H}^n} &&& 
    \text{tr}[P_{\log}(Y, A \text{diag}(z) A^\top + \varepsilon Y)] 
    + (n - \text{tr})\log(\varepsilon)

    \text{subj. to} &&& \sum_{i=1}^m z_i = k

    &&& \text{tr}[Y] \leq k

    &&& 0 \leq z \leq 1

    &&& 0 \preceq Y \preceq \mathbb{I}.

where :math:`A\in\mathbb{R}^{n\times m}` is a matrix of linear measurements, and 
:math:`k` is the number of experiments we select. This was shown to achieve empirically
tighter bounds compared to other relaxation strategies. We show how we can solve this
in QICS below.

.. tabs::

    .. code-tab:: python Native

        import numpy as np

        import qics
        from qics.vectorize import eye, lin_to_mat, mat_to_vec, vec_dim

        np.random.seed(1)

        n = 5
        m = 10
        k = 2
        eps = 1e-6

        vn = vec_dim(n)
        cn = vec_dim(n, compact=True)

        # Define random problem data
        A_dat = 1 / (n**0.25) * np.random.randn(n, m)

        # Model problem using primal variables (t, z, cvec(Y))
        # Define objective function
        c_t = np.ones((1, 1))
        c_z = np.zeros((m, 1))
        c_Y = mat_to_vec(np.eye(n), compact=True) * np.log(eps)
        c = np.block([[c_t], [c_z], [c_Y]])

        # Build linear constraint Σ_i zi = k
        A = np.block([0.0, np.ones((1, m)), np.zeros((1, cn))])
        b = np.array([[k]], dtype=np.float64)

        # Build linear cone constraints
        trace = lin_to_mat(lambda X: np.trace(X), (n, 1), compact=(True, False))
        AdiagA = np.hstack([mat_to_vec(A_dat[:, [i]] @ A_dat[:, [i]].T) for i in range(m)])

        G = np.block([
            [0.0,               np.zeros((1, m)),  trace            ],  # x_nn1 = k - tr[Y]
            [np.zeros((m, 1)),  -np.eye(m),        np.zeros((m, cn))],  # x_nn2 = z
            [np.zeros((m, 1)),  np.eye(m),         np.zeros((m, cn))],  # x_nn3 = 1 - z
            [np.zeros((vn, 1)), np.zeros((vn, m)), eye(n).T         ],  # X_psd = I - Y
            [-1.0,              np.zeros((1, m)),  np.zeros((1, cn))],  # t_ore = t
            [np.zeros((vn, 1)), np.zeros((vn, m)), -eye(n).T        ],  # X_ore = Y
            [np.zeros((vn, 1)), -AdiagA,           -eye(n).T * eps  ]   # Y_ore = Adiag(z)A' + eY
        ])  # fmt: skip

        h = np.block([
            [k],
            [np.zeros((m, 1))],
            [np.ones((m, 1))],
            [mat_to_vec(np.eye(n))],
            [0.0],
            [np.zeros((vn, 1))],
            [np.zeros((vn, 1))]
        ])  # fmt: skip

        # Define cones to optimize over
        cones = [
            qics.cones.NonNegOrthant(1 + m + m),  # (k - tr[Y], -z, z - 1) >= 0
            qics.cones.PosSemidefinite(n),  # I - Y ⪰ 0
            qics.cones.OpPerspecTr(n, "log"),  # (t, Y, Adiag(z)A' + eY) ∈ ORE
        ]

        # Initialize model and solver objects
        model = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones, offset=-n * np.log(eps))
        solver = qics.Solver(model)

        # Solve problem
        info = solver.solve()

    .. code-tab:: python PICOS

        import numpy as np

        import picos

        np.random.seed(1)

        n = 5
        m = 10
        k = 2
        eps = 1e-6

        # Define random problem data
        A = 1 / (n**0.25) * np.random.randn(n, m)

        # Define problem
        P = picos.Problem()
        z = picos.RealVariable("z", m)
        Y = picos.SymmetricVariable("Y", n)

        AzA = A * picos.diag(z) * A.T + eps * picos.I(n)

        obj1 = -picos.trace(picos.oprelentr(Y, AzA))
        obj2 = (n - picos.trace(Y)) * np.log(eps)

        P.set_objective("max", obj1 + obj2)
        P.add_constraint(picos.sum(z) == k)
        P.add_constraint(picos.trace(Y) < k)
        P.add_constraint(0 < z)
        P.add_constraint(z < 1)
        P.add_constraint(Y << np.eye(n))

        # Solve problem
        P.solve(solver="qics", verbosity=1)


.. _opper_refs:

References
----------

    1. M. Berta, O. Fawzi, and M. Tomamichel, “On variational expressions for quantum
       relative entropies,” Letters in Mathematical Physics, vol. 107, pp. 2239–2265,
       2017.

    2. D. Bertsimas, R. Cory-Wright, and J. Pauphilet, “A new perspective on low-rank
       optimization,” Mathematical Programming, vol. 202, no. 1, pp. 47–92, 2023.