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

    \text{s.t.} &&& \sum_{i=1}^p E_i = \mathbb{I}

    &&& E_i \succeq 0, \quad \forall i=1,\ldots,p.

Consider the concrete example from :ref:`[1] <quantum_refs>` where we have
states

.. math::

    \rho_1 = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}, \quad
    \rho_2 = \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}.

For any probabilities :math:`p_1` and :math:`p_2=1-p_1`, it is known that
the solution to the semidefinite program is :math:`E_1=\rho_1` and 
:math:`E_2=\rho_2`. We can confirm this using **QICS** as shown below.

.. testcode::

    import numpy
    import qics

    n = 2
    p1 = p2 = 0.5

    # Define standard basis for symmetric matrices
    E11 = qics.vectorize.mat_to_vec(numpy.array([[1., 0.], [0., 0.]]))
    E12 = qics.vectorize.mat_to_vec(numpy.array([[0., .5], [.5, 0.]]))
    E22 = qics.vectorize.mat_to_vec(numpy.array([[0., 0.], [0., 1.]]))

    # Define objective function
    c = -numpy.vstack((p1 * E11, p2 * E22))

    # Build linear constraints
    A = numpy.vstack((
        numpy.hstack((E11.T, E11.T)),
        numpy.hstack((E12.T, E12.T)),
        numpy.hstack((E22.T, E22.T))
    ))
    b = numpy.array([[1.], [0.], [1.]])

    # Define cones to optimize over
    cones = [
        qics.cones.PosSemidefinite(n), 
        qics.cones.PosSemidefinite(n)
    ]

    # Initialize model and solver objects
    model  = qics.Model(c=c, A=A, b=b, cones=cones)
    solver = qics.Solver(model, verbose=0)

    # Solve problem
    info = solver.solve()

    print("Optimal POVMs are E1:")
    print(info["s_opt"][0][0])
    print("and E2:")
    print(info["s_opt"][1][0])

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    Optimal POVMs are E1:
    [[9.99999999e-01 0.00000000e+00]
    [0.00000000e+00 2.71999999e-09]]
    and E2:
    [[2.71999999e-09 0.00000000e+00]
    [0.00000000e+00 9.99999999e-01]]


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
    \text{tr}[X + X^\dagger] \quad \text{s.t.} \quad \begin{bmatrix} \rho & X
    \\ X^\dagger & \sigma \end{bmatrix} \succeq 0.

We show how this can be solved using QICS below, which we verify against the
analytic equation.

.. testcode::

    import numpy
    import scipy
    import qics

    numpy.random.seed(1)

    n = 2

    rho = qics.quantum.random.density_matrix(n, iscomplex=True)
    sig = qics.quantum.random.density_matrix(n, iscomplex=True)

    # Define objective function
    c = -0.5 * qics.vectorize.mat_to_vec(numpy.block([
        [numpy.zeros((n, n)), numpy.eye(n)],
        [numpy.eye(n), numpy.zeros((n, n))]
    ]).astype(numpy.complex128))

    # Build linear constraints
    A = numpy.vstack((
        qics.vectorize.lin_to_mat(lambda X : X[:n, :n], (2*n, n), iscomplex=True),
        qics.vectorize.lin_to_mat(lambda X : X[n:, n:], (2*n, n), iscomplex=True)
    ))

    b = numpy.vstack((
        qics.vectorize.mat_to_vec(rho, compact=True),
        qics.vectorize.mat_to_vec(sig, compact=True)
    ))

    # Define cones to optimize over
    cones = [qics.cones.PosSemidefinite(2*n, iscomplex=True)]

    # Initialize model and solver objects
    model  = qics.Model(c=c, A=A, b=b, cones=cones)
    solver = qics.Solver(model, verbose=0)

    # Solve problem
    info = solver.solve()

    rt_rho = scipy.linalg.sqrtm(rho)
    rt_sig = scipy.linalg.sqrtm(sig)
    analytic = numpy.linalg.norm(rt_rho @ rt_sig, "nuc")

    print("QICS fidelity:    ", -info["p_obj"])
    print("Analytic fidelity:", analytic)

.. testoutput::

    QICS fidelity:     0.7536085578284577
    Analytic fidelity: 0.7536085481796011

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

.. testcode::

    import numpy
    import qics

    numpy.random.seed(1)

    n = 2
    N = n*n

    J1 = qics.quantum.random.choi_operator(n, iscomplex=True)
    J2 = qics.quantum.random.choi_operator(n, iscomplex=True)
    J = J1 - J2

    # Define objective function
    c1 = -0.5 * qics.vectorize.mat_to_vec(numpy.block([
        [numpy.zeros((N, N)), J],
        [J.conj().T, numpy.zeros((N, N))]
    ]))
    c2 = numpy.zeros((2*n*n, 1))
    c3 = numpy.zeros((2*n*n, 1))
    c = numpy.vstack((c1, c2, c3))

    # Build linear constraints
    vN = qics.vectorize.vec_dim(N, iscomplex=True, compact=True)
    submtx_11 = qics.vectorize.lin_to_mat(lambda X : X[:N, :N], (2*N, N), iscomplex=True)
    submtx_22 = qics.vectorize.lin_to_mat(lambda X : X[N:, N:], (2*N, N), iscomplex=True)
    i_kr = qics.vectorize.lin_to_mat(
        lambda X : qics.quantum.i_kr(X, (n, n), 0), (n, N), iscomplex=True)
    tr = qics.vectorize.mat_to_vec(numpy.eye(n, dtype=numpy.complex128)).T
    # I ⊗ rho block
    A1 = numpy.hstack((submtx_11, -i_kr, numpy.zeros((vN, 2*n*n))))
    b1 = numpy.zeros((vN, 1))
    # I ⊗ sig block
    A2 = numpy.hstack((submtx_22, numpy.zeros((vN, 2*n*n)), -i_kr))
    b2 = numpy.zeros((vN, 1))
    # tr[rho] = 1
    A3 = numpy.hstack((numpy.zeros((1, 8*N*N)), tr, numpy.zeros((1, 2*n*n))))
    b3 = numpy.array([[1.]])
    # tr[sig] = 1
    A4 = numpy.hstack((numpy.zeros((1, 8*N*N)), numpy.zeros((1, 2*n*n)), tr))
    b4 = numpy.array([[1.]])

    A = numpy.vstack((A1, A2, A3, A4))
    b = numpy.vstack((b1, b2, b3, b4))

    # Define cones to optimize over
    cones = [
        qics.cones.PosSemidefinite(2*n*n, iscomplex=True),
        qics.cones.PosSemidefinite(n, iscomplex=True),
        qics.cones.PosSemidefinite(n, iscomplex=True),
    ]

    # Initialize model and solver objects
    model  = qics.Model(c=c, A=A, b=b, cones=cones)
    solver = qics.Solver(model, verbose=0)

    # Solve problem
    info = solver.solve()

    print("Diamond norm:", -info["p_obj"])

.. testoutput::

    Diamond norm: 1.0697369635368625


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

.. testcode::

    import numpy
    import qics

    numpy.random.seed(1)

    n = m = 2

    rhoA = qics.quantum.random.density_matrix(n, iscomplex=True)
    rhoB = qics.quantum.random.density_matrix(m, iscomplex=True)

    # Generate random objective function
    C = numpy.random.randn(n*m, n*m) + numpy.random.randn(n*m, n*m)*1j
    C = C + C.conj().T
    c = qics.vectorize.mat_to_vec(C)

    # Build linear constraints
    trA = qics.vectorize.lin_to_mat(
        lambda X : qics.quantum.p_tr(X, (n, m), 0), (n*m, m), iscomplex=True)
    trB = qics.vectorize.lin_to_mat(
        lambda X : qics.quantum.p_tr(X, (n, m), 1), (n*m, n), iscomplex=True)

    A = numpy.vstack((trA, trB))
    b = numpy.vstack((
        qics.vectorize.mat_to_vec(rhoA, compact=True), 
        qics.vectorize.mat_to_vec(rhoB, compact=True)
    ))

    # Define cones to optimize over
    cones = [qics.cones.PosSemidefinite(n*m, iscomplex=True)]

    # Initialize model and solver objects
    model  = qics.Model(c=c, A=A, b=b, cones=cones)
    solver = qics.Solver(model, verbose=0)

    # Solve problem
    info = solver.solve()

    print("Optimal value:", -info["p_obj"])

.. testoutput::

    Optimal value: 1.9485265803931466

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

    \text{find} \quad \rho_{aB} \quad \text{s.t.} 
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

.. testcode::

    import numpy
    import qics

    n  = 2
    n2 = n * n
    n3 = n * n * n

    vn2 = qics.vectorize.vec_dim(n2, compact=True)
    vn3 = qics.vectorize.vec_dim(n3, compact=True)

    rho_ab = 0.5 * numpy.array([
        [1., 0., 0., 1.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [1., 0., 0., 1.]
    ])

    # Define objective function
    c = numpy.zeros((3*n3*n3, 1))

    # Build linear constraints
    # rho_ab1 = tr_b2(rho_aB)
    tr_b2 = qics.vectorize.lin_to_mat(
        lambda X : qics.quantum.p_tr(X, (n, n, n), 2), (n3, n2))
    A1 = numpy.hstack((tr_b2, numpy.zeros((vn2, 2*n3*n3))))
    b1 = qics.vectorize.mat_to_vec(rho_ab, compact=True)
    # rho_aB = swap_b1,b2(rho_aB)
    swap = qics.vectorize.lin_to_mat(
        lambda X : qics.quantum.swap(X, (n, n, n), 1, 2), (n3, n3))
    A2 = numpy.hstack((swap - qics.vectorize.eye(n3), numpy.zeros((vn3, 2*n3*n3))))
    b2 = numpy.zeros((vn3, 1))
    # tr[rho_aB] = 1
    tr = qics.vectorize.mat_to_vec(numpy.eye(n3)).T
    A3 = numpy.hstack((tr, numpy.zeros((1, 2*n3*n3))))
    b3 = numpy.array([[1.]])
    # Y = T_b2(rho_aB)
    T_b2 = qics.vectorize.lin_to_mat(
        lambda X : qics.quantum.partial_transpose(X, (n2, n), 1), (n3, n3))
    A4 = numpy.hstack((T_b2, -qics.vectorize.eye(n3), numpy.zeros((vn3, n3*n3))))
    b4 = numpy.zeros((vn3, 1))
    # Z = T_b1b2(rho_aB)
    T_b1b2 = qics.vectorize.lin_to_mat(
        lambda X : qics.quantum.partial_transpose(X, (n, n2), 1), (n3, n3))
    A5 = numpy.hstack((T_b1b2, numpy.zeros((vn3, n3*n3)), -qics.vectorize.eye(n3)))
    b5 = numpy.zeros((vn3, 1))

    A = numpy.vstack((A1, A2, A3, A4, A5))
    b = numpy.vstack((b1, b2, b3, b4, b5))

    # Define cones to optimize over
    cones = [
        qics.cones.PosSemidefinite(n3),
        qics.cones.PosSemidefinite(n3),
        qics.cones.PosSemidefinite(n3)
    ]

    # Initialize model and solver objects
    model  = qics.Model(c=c, A=A, b=b, cones=cones)
    solver = qics.Solver(model, verbose=0)

    # Solve problem
    info = solver.solve()

    print("Solution status:", info["sol_status"])

.. testoutput::

    Solution status: pinfeas

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