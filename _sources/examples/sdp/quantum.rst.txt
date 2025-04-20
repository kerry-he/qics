Quantum information
===================

Semidefinite programs commonly arise in quantum information as quantum states
are defined using positive semidefinite matrices. We survey some of these
applications below.


Quantum state discrimination
-------------------------------

Consider a setup where a quantum state :math:`\rho_i` is sent with probability
:math:`p_i`. We want to design a set of POVMs :math:`\{ E_i \}_{i=1}^p` which
maximizes the probability of success of correctly identifying the quantum 
states. This can be done by solving the following semidefinite program.

.. math::

    \min_{E_i \in \mathbb{H}^n} &&& \sum_{i=1}^p p_i \text{tr}[ E_i \rho_i ]

    \text{subj. to} &&& \sum_{i=1}^p E_i = \mathbb{I}

    &&& E_i \succeq 0, \quad \forall i=1,\ldots,p.

Consider the concrete example from :ref:`[1] <quantum_refs>` where we have
states

.. math::

    \rho_1 = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}, \quad
    \rho_2 = \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}.

For any probabilities :math:`p_1` and :math:`p_2=1-p_1`, it is known that
the solution to the semidefinite program is :math:`E_1=\rho_1` and 
:math:`E_2=\rho_2`. We can confirm this using **QICS** as shown below.

.. code-block:: python

    import numpy as np

    import qics

    n = 2
    p1 = p2 = 0.5

    eye_mtx = qics.vectorize.eye(n)
    rho_1 = qics.vectorize.mat_to_vec(np.array([[1.0, 0.0], [0.0, 0.0]]))
    rho_2 = qics.vectorize.mat_to_vec(np.array([[0.0, 0.0], [0.0, 1.0]]))

    # Model problem using primal variables (E1, E2)
    # Define objective function
    c = -np.block([[p1 * rho_1], [p2 * rho_2]])

    # Build linear constraint E1 + E2 = I
    A = np.block([eye_mtx, eye_mtx])
    b = qics.vectorize.mat_to_vec(np.eye(n), compact=True)

    # Define cones to optimize over
    cones = [qics.cones.PosSemidefinite(n), qics.cones.PosSemidefinite(n)]

    # Initialize model and solver objects
    model = qics.Model(c=c, A=A, b=b, cones=cones)
    solver = qics.Solver(model)

    # Solve problem
    info = solver.solve()

    E1_opt, E2_opt = info["s_opt"][0][0], info["s_opt"][1][0]
    print("Optimal POVMs are:")
    print(np.round(E1_opt, 4))
    print("and:")
    print(np.round(E2_opt, 4))

.. code-block:: none

    Optimal POVMs are:
    [[1. 0.]
     [0. 0.]]
    and:
    [[0. 0.]
     [0. 1.]]

Quantum state fidelity
----------------------

The quantum state fidelity is a measure of dissimilarity between two quantum
states, and is defined by 

.. math::

    F(\rho, \sigma) = \| \sqrt{\rho} \sqrt{\sigma} \|_1.

In :ref:`[2] <quantum_refs>`, it was shown that the quantumn state fidelity
could also be represented using the following semidefinite program

.. math::

    \max_{X \in \mathbb{C}^{n\times n}} \quad \frac{1}{2} 
    \text{tr}[X + X^\dagger] \quad \text{subj. to} \quad \begin{bmatrix} \rho & X
    \\ X^\dagger & \sigma \end{bmatrix} \succeq 0.

We show how this can be solved using QICS below, which we verify against the
analytic equation.

.. code-block:: python

    import numpy as np
    import scipy as sp

    import qics
    from qics.quantum.random import density_matrix
    from qics.vectorize import lin_to_mat, mat_to_vec

    np.random.seed(1)

    n = 2

    # Generate random problem data
    rho = density_matrix(n, iscomplex=True)
    sig = density_matrix(n, iscomplex=True)

    rho_cvec = mat_to_vec(rho, compact=True)
    sig_cvec = mat_to_vec(sig, compact=True)

    # Model problem using primal variable M
    # Define objective function
    eye_n = np.eye(n, dtype=np.complex128)
    zero_n = np.zeros((n, n), dtype=np.complex128)
    C = np.block([[zero_n, eye_n], [eye_n, zero_n]])
    c = -0.5 * qics.vectorize.mat_to_vec(C)

    # Build linear constraints M11 = rho and M22 = sig
    submat_11 = lin_to_mat(lambda X: X[:n, :n], (2 * n, n), iscomplex=True)
    submat_22 = lin_to_mat(lambda X: X[n:, n:], (2 * n, n), iscomplex=True)

    A = np.block([[submat_11], [submat_22]])
    b = np.block([[rho_cvec], [sig_cvec]])

    # Define cones to optimize over
    cones = [qics.cones.PosSemidefinite(2 * n, iscomplex=True)]

    # Initialize model and solver objects
    model = qics.Model(c=c, A=A, b=b, cones=cones)
    solver = qics.Solver(model)

    # Solve problem
    info = solver.solve()

    rt_rho = sp.linalg.sqrtm(rho)
    rt_sig = sp.linalg.sqrtm(sig)
    analytic_fidelity = np.linalg.norm(rt_rho @ rt_sig, "nuc")
    numerical_fidelity = -info["p_obj"]

    print("Analytic fidelity: ", analytic_fidelity)
    print("Numerical fidelity:", numerical_fidelity)

.. code-block:: none

    Analytic fidelity:  0.7536085481796011
    Numerical fidelity: 0.7536085578284577

Diamond norm
--------------

The diamond norm is used to measure the dissimilarity between two quantum 
channels. Formally, it is given as

.. math::

    \| \mathcal{N} \|_\diamond = \max_{\rho_{AA}\in\mathbb{H}^{n^2}} \| 
    \mathcal{N}\otimes\mathbb{I} (\rho_{AA}) \|_1 \quad \text{subj. to} \quad 
    \| \rho_{AA} \|_1 \leq 1

If we associate the quantum channel :math:`\mathcal{N}` with a Choi-Jamiolkowski 
representation :math:`J`, then in :ref:`[2] <quantum_refs>`
it was shown that the diamond norm could be computed using the semidefinite
program

.. math::

    \max_{\rho,\sigma,Z} \quad & \frac{1}{2} (\langle J, Z \rangle + \langle J^\dagger, Z^\dagger \rangle)\\
    \text{subj. to} \quad &\begin{bmatrix}I\otimes\rho & Z \\\ Z^\dagger & I\otimes\sigma\end{bmatrix} \succeq 0\\
    & \text{tr}[\rho] = \text{tr}[\sigma] = 1\\
    & \rho,\sigma\succeq 0.

We show how this can be computed in QICS below.

.. code-block:: python

    import numpy as np

    import qics
    from qics.quantum import i_kr
    from qics.quantum.random import choi_operator
    from qics.vectorize import lin_to_mat, mat_to_vec, vec_dim

    np.random.seed(1)

    n = 4
    N = n * n
    vn = vec_dim(n, iscomplex=True)
    vN = vec_dim(N, iscomplex=True)
    cN = vec_dim(N, iscomplex=True, compact=True)

    # Generate random problem data
    J1 = choi_operator(n, iscomplex=True)
    J2 = choi_operator(n, iscomplex=True)
    J = J1 - J2

    # Model problem using primal variables (M, rho, sig)
    # Define objective function
    C_M = np.block([[np.zeros((N, N)), J], [J.conj().T, np.zeros((N, N))]])

    c_M = -0.5 * mat_to_vec(C_M)
    c_rho = np.zeros((vn, 1))
    c_sig = np.zeros((vn, 1))
    c = np.block([[c_M], [c_rho], [c_sig]])

    # Build linear constraints
    trace = lin_to_mat(lambda X: np.trace(X), (n, 1), iscomplex=True)
    ikr_1 = lin_to_mat(lambda X: i_kr(X, (n, n), 0), (n, N), iscomplex=True)
    submat_11 = lin_to_mat(lambda X: X[:N, :N], (2 * N, N), iscomplex=True)
    submat_22 = lin_to_mat(lambda X: X[N:, N:], (2 * N, N), iscomplex=True)

    A = np.block([
        [submat_11,             -ikr_1,             np.zeros((cN, vn))],  # M11 = I ⊗ rho
        [submat_22,             np.zeros((cN, vn)), -ikr_1            ],  # M22 = I ⊗ sig
        [np.zeros((1, 4 * vN)), trace,              np.zeros((1, vn)) ],  # tr[rho] = 1
        [np.zeros((1, 4 * vN)), np.zeros((1, vn)),  trace             ]   # tr[sig] = 1
    ])

    b = np.block([[np.zeros((cN, 1))], [np.zeros((cN, 1))], [1.0], [1.0]])

    # Define cones to optimize over
    cones = [
        qics.cones.PosSemidefinite(2 * N, iscomplex=True),  # M ⪰ 0
        qics.cones.PosSemidefinite(n, iscomplex=True),      # rho ⪰ 0
        qics.cones.PosSemidefinite(n, iscomplex=True),      # sig ⪰ 0
    ]

    # Initialize model and solver objects
    model = qics.Model(c=c, A=A, b=b, cones=cones)
    solver = qics.Solver(model)

    # Solve problem
    info = solver.solve()

Quantum optimal transport
---------------------------

The classical optimal transport is involved with minimizing a joint probability
distribution represented by a matrix :math:`X` over a linear function, subject
to the distribution satisfying given marginal distributions. The quantum analog
of this problem can be defined as follows :ref:`[3] <quantum_refs>`

.. math::

    \max_{X\in\mathbb{H}^{nm}} \quad & \langle C, X \rangle \\ 
    \text{subj. to} \quad & \text{tr}_A(X) = \rho_B\\
    & \text{tr}_B(X) = \rho_A\\
    & X\succeq 0,

where partial traces are used analogously to marginal distributions. We show how
this problem can be solved in QICS below.

.. code-block:: python

    import numpy as np

    import qics
    from qics.quantum import p_tr
    from qics.quantum.random import density_matrix
    from qics.vectorize import lin_to_mat, mat_to_vec

    np.random.seed(1)

    n = m = 2

    # Generate random problem data
    rho_A = density_matrix(n, iscomplex=True)
    rho_B = density_matrix(m, iscomplex=True)

    rho_A_cvec = mat_to_vec(rho_A, compact=True)
    rho_B_cvec = mat_to_vec(rho_B, compact=True)

    # Model problem using primal variable X
    # Generate random objective function
    C = np.random.randn(n * m, n * m) + np.random.randn(n * m, n * m) * 1j
    c = mat_to_vec(C + C.conj().T)

    # Build linear constraints tr_A(X) = rho_A and tr_B(X) = rho_B
    ptr_A = lin_to_mat(lambda X: p_tr(X, (n, m), 1), (n * m, n), iscomplex=True)
    ptr_B = lin_to_mat(lambda X: p_tr(X, (n, m), 0), (n * m, m), iscomplex=True)

    A = np.block([[ptr_A], [ptr_B]])
    b = np.block([[rho_A_cvec], [rho_B_cvec]])

    # Define cones to optimize over
    cones = [qics.cones.PosSemidefinite(n * m, iscomplex=True)]

    # Initialize model and solver objects
    model = qics.Model(c=c, A=A, b=b, cones=cones)
    solver = qics.Solver(model)

    # Solve problem
    info = solver.solve()

Detecting entanglement
----------------------

A quantum state :math:`\rho_{ab}`, defined on the bipartite system 
:math:`\mathcal{H}_a\otimes\mathcal{H}_b`, is separable if we can express it in
the form

.. math::

    \rho_{ab} = \sum_{i} p_i \rho_a^i \otimes \rho_b^i,

for some probability distribution :math:`p` and density matrices
:math:`\rho_a^i` and :math:`\rho_b^i`. A state that is not separable is called
entangled.

One way to detect if a quantum state is entangled or separable is to use the
Doherty-Parrilo-Spedalieri hierarchy :ref:`[4] <quantum_refs>`, which is a 
heirarchy of semidefinite representable criteria that must be satisfied by 
separable states. For the :math:`k=2` level heirarchy, the corresponding 
feasibiltiy problem is (see :ref:`[1] <quantum_refs>`)

.. math::

    \text{find} \quad \rho_{aB} \quad \text{subj. to} 
    \quad & \text{tr}_{b_2}(\rho_{aB}) \\
    & \rho_{aB} = \Pi_{b_1,b_2} \rho_{aB} \Pi_{b_1,b_2} \\
    & \text{tr}[\rho_{aB}] = 1 \\
    & \rho_{aB} \succeq 0 \\
    & \mathcal{T}_{b_2}(\rho_{aB}) \succeq 0 \\
    & \mathcal{T}_{b_1b_2}(\rho_{aB}) \succeq 0.

where :math:`\mathcal{H}_B=\mathcal{H}_{b_1}\otimes\mathcal{H}_{b_2}`, 
:math:`\mathcal{T}_X` denotes the partial transpose with respect to subsystem
:math:`X`, and :math:`\Pi_{b_1,b_2}` is the swap operator that exchanges the
positions of the subsystems :math:`b_1` and :math:`b_2`. 

We show how we can solve this feasibility problem for the entangled quantum 
state

.. math::

    \rho_{ab} = \frac{1}{2} \begin{bmatrix} 
        1 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 1 
    \end{bmatrix}

in **QICS** below.

.. code-block:: python

    import numpy as np

    import qics
    from qics.quantum import p_tr, partial_transpose, swap
    from qics.vectorize import eye, lin_to_mat, mat_to_vec, vec_dim

    n = 2
    n2 = n * n
    n3 = n * n * n

    vn3 = vec_dim(n3)
    cn2 = vec_dim(n2, compact=True)
    cn3 = vec_dim(n3, compact=True)

    # Define an entangled quantum state
    rho = 0.5 * np.array([[1.0, 0.0, 0.0, 1.0],
                          [0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0, 1.0]])  # fmt: skip

    # Model problem using primal variables (rho_aB, sigma_aB, omega_aB)
    # Define objective function
    c = np.zeros((3 * vn3, 1))

    # Build linear constraints
    trace = lin_to_mat(lambda X: np.trace(X), (n3, 1))
    ptr_b2 = lin_to_mat(lambda X: p_tr(X, (n, n, n), 2), (n3, n2))
    swap_b1b2 = lin_to_mat(lambda X: swap(X, (n, n, n), 1, 2), (n3, n3))
    T_b2 = lin_to_mat(lambda X: partial_transpose(X, (n2, n), 1), (n3, n3))
    T_b1b2 = lin_to_mat(lambda X: partial_transpose(X, (n, n2), 1), (n3, n3))

    A = np.block([
        [ptr_b2,              np.zeros((cn2, vn3)), np.zeros((cn2, vn3))],  # tr_b2(rho_aB) = rho
        [swap_b1b2 - eye(n3), np.zeros((cn3, vn3)), np.zeros((cn3, vn3))],  # swap_b1b2(rho_aB) = rho_aB
        [trace,               np.zeros((1, vn3)),   np.zeros((1, vn3))  ],  # tr[rho_aB] = 1
        [T_b2,                -eye(n3),             np.zeros((cn3, vn3))],  # sigma_aB = T_b2(rho_aB)
        [T_b1b2,              np.zeros((cn3, vn3)), -eye(n3)            ]   # omega_aB = T_b1b2(rho_aB)
    ])

    b = np.block([
        [mat_to_vec(rho, compact=True)], 
        [np.zeros((cn3, 1))], 
        [1.0], 
        [np.zeros((cn3, 1))], 
        [np.zeros((cn3, 1))]
    ])

    # Define cones to optimize over
    cones = [
        qics.cones.PosSemidefinite(n3),  # rho_aB ⪰ 0
        qics.cones.PosSemidefinite(n3),  # sigma_aB = T_b2(rho_aB) ⪰ 0
        qics.cones.PosSemidefinite(n3),  # omega_aB = T_b1b2(rho_aB) ⪰ 0
    ]

    # Initialize model and solver objects
    model = qics.Model(c=c, A=A, b=b, cones=cones)
    solver = qics.Solver(model)

    # Solve problem
    info = solver.solve()

QICS returns a solution summary which should look like

.. code-block:: none

    Solution summary
            sol. status:  pinfeas                   num. iter:    8
            exit status:  solved                    solve time:   0.430
            primal obj:   0.000000000000e+00        primal feas:  4.79e-01
            dual obj:     4.165819513474e+13        dual feas:    4.81e-01

As the semidefinite program is infeasible, then :math:`\rho_{ab}` must be
entangled, which we know is true for this quantum state.



.. _quantum_refs:

References
----------

    1. Siddhu, V. and Tayur, S.
       "Five starter pieces: Quantum Information Science via semidefinite programs", 
       Tutorials in Operations Research: Emerging and Impactful Topics in Operations, pp. 59–92. 2022.

    2. J. Watrous, “Simpler semidefinite programs for completely bounded norms,” 
       arXiv preprint arXiv:1207.5726, 2012.

    3. Cole, S. et al. (2023) "On Quantum Optimal Transport", Mathematical Physics, Analysis and Geometry, 26(2).

    4. Andrew C. Doherty, Pablo A. Parrilo, and Federico M. Spedalieri. Complete family of
       separability criteria. Physical Review A, 69(2), Feb 2004.