Quantum SDPs
==============

Semidefinite programs commonly arise in quantum information
as quantum states are defined using positive semidefinite 
matrices. We survey some of these applications below.


Quantum state discrimination
-------------------------------

Consider a setup where a quantum state :math:`\rho_i` is sent
with probability :math:`p_i`. We want to design a set of POVMs
:math:`\{ E_i \}_{i=1}^n` which maximizes the probability of success of
correctly identifying the quantum states. This can be done
by solving the following semidefinite program.

.. math::

    \min_{E_i \in \mathbb{H}^n} &&& \sum_{i=1}^p p_i \text{tr}[ E_i \rho_i ]

    \text{s.t.} &&& \sum_{i=1}^p E_i = \mathbb{I}

    &&& E_i \succeq 0, \quad \forall i=1,\ldots,p.

Consider the concrete example from :ref:`[1] <quantum_refs>` where we have states

.. math::

    \sigma_1 = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}, \quad
    \sigma_2 = \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}.

For any probabilities :math:`p_1` and :math:`p_2=1-p_1`, it is known that
the solution to the semidefinite program is :math:`E_1=\sigma_1` and :math:`E_2=\sigma_2`.
We can confirm this using **QICS**.

.. code-block:: python

    import numpy as np
    import qics

    n = 2
    p1 = p2 = 0.5

    E11 = np.array([[1., 0.], [0., 0.]])
    E12 = np.array([[0., .5], [.5, 0.]])
    E22 = np.array([[0., 0.], [0., 1.]])

    # Define objective function
    c = -np.vstack((p1 * E11.reshape(-1, 1), p2 * E22.reshape(-1, 1)))

    # Build linear constraints
    A = np.vstack((
        np.hstack((E11.reshape(1, -1), E11.reshape(1, -1))),
        np.hstack((E12.reshape(1, -1), E12.reshape(1, -1))),
        np.hstack((E22.reshape(1, -1), E22.reshape(1, -1)))
    ))
    b = np.array([[1.], [0.], [1.]])

    # Define cones to optimize over
    cones = [
        qics.cones.PosSemidefinite(n), 
        qics.cones.PosSemidefinite(n)
    ]

    # Initialize model and solver objects
    model  = qics.Model(c=c, A=A, b=b, cones=cones)
    solver = qics.Solver(model)

    # Solve problem
    info = solver.solve()

    print("Optimal POVMs are")
    print(info["s_opt"][0])
    print("and")
    print(info["s_opt"][1])

.. code-block:: none

    ====================================================================
                QICS v0.0 - Quantum Information Conic Solver
                by K. He, J. Saunderson, H. Fawzi (2024)
    ====================================================================
    Problem summary:
            no. cones:  2                        no. vars:    8
            barr. par:  5                        no. constr:  3
            symmetric:  True                     cone dim:    8
            complex:    False

    ...

    Solution summary
            sol. status:  optimal                num. iter:    5
            exit status:  solved                 solve time:   0.009

            primal obj:  -9.999999972800e-01     primal feas:  1.46e-09
            dual obj:    -9.999999963390e-01     dual feas:    1.46e-09
            opt. gap:     9.41e-10

    Optimal POVMs are E1=
    [[9.99999999e-01 0.00000000e+00]
     [0.00000000e+00 2.71999999e-09]]
    and E2=
    [[2.71999999e-09 0.00000000e+00]
     [0.00000000e+00 9.99999999e-01]]


Quantum state fidelity
-------------------------

The quantum state fidelity is a measure of dissimilarity between
two quantum states, and is defined by 

.. math::

    F(\rho, \sigma) = \| \sqrt{\rho} \sqrt{\sigma} \|_1.

In :ref:`[2] <quantum_refs>`, it was shown that the quantumn state
fidelity could also be represented using the following semidefinite program

.. math::

    \min_{X \in \mathbb{C}^{n\times n}} \quad \frac{1}{2} \text{tr}[X + X^\dagger] \quad 
    \text{s.t.} \quad \begin{bmatrix} \rho & X \\ X^\dagger & \sigma \end{bmatrix} \succeq 0.

.. code-block:: python

    import numpy as np
    import qics
    import qics.vectorize as vec
    import qics.quantum as qu

    np.random.seed(1)

    n = 2

    rho = qu.random.density_matrix(n, iscomplex=True)
    sig = qu.random.density_matrix(n, iscomplex=True)

    # Define objective function
    c = -0.5 * vec.mat_to_vec(np.block([
        [np.zeros((n, n)), np.eye(n)],
        [np.eye(n), np.zeros((n, n))]
    ]).astype(np.complex128))

    # Build linear constraints
    A = np.vstack((
        vec.lin_to_mat(lambda X : X[:n, :n], (2*n, n), iscomplex=True),
        vec.lin_to_mat(lambda X : X[n:, n:], (2*n, n), iscomplex=True)
    ))

    b = np.vstack((
        vec.mat_to_vec(rho, compact=True),
        vec.mat_to_vec(sig, compact=True)
    ))

    # Define cones to optimize over
    cones = [qics.cones.PosSemidefinite(2*n, iscomplex=True)]

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
            no. cones:  1                        no. vars:    32
            barr. par:  5                        no. constr:  8
            symmetric:  True                     cone dim:    32
            complex:    True

    ...

    Solution summary
            sol. status:  optimal                num. iter:    12
            exit status:  solved                 solve time:   0.020

            primal obj:  -7.536085578285e-01     primal feas:  8.90e-09
            dual obj:    -7.536085557553e-01     dual feas:    5.93e-09
            opt. gap:     2.07e-09

We can verify the solution using the original definition of the fidelity.

>>> import scipy as sp
>>> rt2_rho = sp.linalg.sqrtm(rho)
>>> rt2_sig = sp.linalg.sqrtm(sig)
>>> sp.linalg.norm(rt2_rho @ rt2_sig, "nuc")
0.7536085481796011


Diamond norm
--------------

The diamond norm is used to measure the dissimilarity between two quantum 
channels. Formally, it is given as

.. math::

    \| \mathcal{N} \|_\diamond = \max_{\rho_{AA}\in\mathbb{H}^{n^2}} \| \mathcal{N}\otimes\mathbb{I} (\rho_{AA}) \|_1 \quad \text{subj. to} \quad \| \rho_{AA} \|_1 \leq 1

If we associate the quantum channel :math:`\mathcal{N}` with a Choi-Jamiolkowski 
representation :math:`J`, then in :ref:`[2] <quantum_refs>`
it was shown that the diamond norm could be computed using the semidefinite
program

.. math::

    \max_{\rho,\sigma,Z} \quad & \frac{1}{2} (\langle J, Z \rangle + \langle J^\dagger, Z^\dagger \rangle)\\
    \text{subj. to} \quad &\begin{bmatrix}I\otimes\rho & Z \\\ Z^\dagger & I\otimes\sigma\end{bmatrix} \succeq 0\\
    & \text{tr}[\rho] = \text{tr}[\sigma] = 1\\
    & \rho,\sigma\succeq 0


.. code-block:: python

    import numpy as np
    import qics
    import qics.vectorize as vec
    import qics.quantum as qu

    np.random.seed(1)

    n = 2
    N = n*n

    J1 = qu.random.choi_operator(n, iscomplex=True)
    J2 = qu.random.choi_operator(n, iscomplex=True)
    J = J1 - J2

    # Define objective function
    c1 = -0.5 * vec.mat_to_vec(np.block([
        [np.zeros((N, N)), J],
        [J.conj().T, np.zeros((N, N))]
    ]))
    c2 = np.zeros((2*n*n, 1))
    c3 = np.zeros((2*n*n, 1))
    c = np.vstack((c1, c2, c3))

    # Build linear constraints
    vN = vec.vec_dim(N, iscomplex=True, compact=True)
    submtx_11 = vec.lin_to_mat(lambda X : X[:N, :N], (2*N, N), iscomplex=True)
    submtx_22 = vec.lin_to_mat(lambda X : X[N:, N:], (2*N, N), iscomplex=True)
    i_kr = vec.lin_to_mat(lambda X : qu.i_kr(X, (n, n), 0), (n, N), iscomplex=True)
    tr = vec.mat_to_vec(np.eye(n, dtype=np.complex128)).T
    # I ⊗ rho block
    A1 = np.hstack((submtx_11, -i_kr, np.zeros((vN, 2*n*n))))
    b1 = np.zeros((vN, 1))
    # I ⊗ sig block
    A2 = np.hstack((submtx_22, np.zeros((vN, 2*n*n)), -i_kr))
    b2 = np.zeros((vN, 1))
    # tr[rho] = 1
    A3 = np.hstack((np.zeros((1, 8*N*N)), tr, np.zeros((1, 2*n*n))))
    b3 = np.array([[1.]])
    # tr[sig] = 1
    A4 = np.hstack((np.zeros((1, 8*N*N)), np.zeros((1, 2*n*n)), tr))
    b4 = np.array([[1.]])

    A = np.vstack((A1, A2, A3, A4))
    b = np.vstack((b1, b2, b3, b4))

    # Define cones to optimize over
    cones = [
        qics.cones.PosSemidefinite(2*n*n, iscomplex=True),
        qics.cones.PosSemidefinite(n, iscomplex=True),
        qics.cones.PosSemidefinite(n, iscomplex=True),
    ]

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
            no. cones:  3                        no. vars:    144
            barr. par:  13                       no. constr:  34
            symmetric:  True                     cone dim:    144
            complex:    True

    ...

    Solution summary
            sol. status:  optimal                num. iter:    11
            exit status:  solved                 solve time:   0.025

            primal obj:  -1.069736963537e+00     primal feas:  8.82e-09
            dual obj:    -1.069736964946e+00     dual feas:    7.45e-09
            opt. gap:     1.32e-09


Quantum optimal transport
---------------------------

The classical optimal transport is involved with minimizing
a joint probability distribution represented by a matrix :math:`X` 
over a linear function, subject to the distribution satisfying
given marginal distributions. The quantum analog of this problem
can be defined as follows :ref:`[3] <quantum_refs>`

.. math::

    \max_{X\in\mathbb{H}^{nm}} \quad & \langle C, X \rangle \\ 
    \text{subj. to} \quad & \text{tr}_A[X] = \rho_B\\
    & \text{tr}_B[X] = \rho_A\\
    & X\succeq 0

.. code-block:: python

    import numpy as np
    import qics
    import qics.quantum as qu
    import qics.vectorize as vec

    np.random.seed(1)

    n = m = 2

    rhoA = qu.random.density_matrix(n, iscomplex=True)
    rhoB = qu.random.density_matrix(m, iscomplex=True)

    # Generate random objective function
    C = np.random.randn(n*m, n*m) + np.random.randn(n*m, n*m)*1j
    C = C + C.conj().T
    c = vec.mat_to_vec(C)

    # Build linear constraints
    trA = vec.lin_to_mat(lambda X : qu.p_tr(X, (n, m), 0), (n*m, m), iscomplex=True)
    trB = vec.lin_to_mat(lambda X : qu.p_tr(X, (n, m), 1), (n*m, n), iscomplex=True)
    A = np.vstack((trA, trB))
    b = np.vstack((vec.mat_to_vec(rhoA, compact=True), vec.mat_to_vec(rhoB, compact=True)))

    # Define cones to optimize over
    cones = [qics.cones.PosSemidefinite(n*m, iscomplex=True)]

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
            no. cones:  1                        no. vars:    32
            barr. par:  5                        no. constr:  8
            symmetric:  True                     cone dim:    32
            complex:    True

    ...

    Solution summary
            sol. status:  optimal                num. iter:    10
            exit status:  solved                 solve time:   0.019

            primal obj:  -1.948526580393e+00     primal feas:  7.88e-10
            dual obj:    -1.948526575690e+00     dual feas:    5.45e-10
            opt. gap:     2.41e-09



Detecting entanglement
------------------------

A quantum state :math:`\rho_{ab}`, defined on the bipartite 
system :math:`\mathcal{H}_a\otimes\mathcal{H}_b`, is separable
if we can express it in the form

.. math::

    \rho_{ab} = \sum_{i} p_i \rho_a^i \otimes \rho_b^i,

for some probability distribution :math:`p` and density matrices
:math:`\rho_a^i` and :math:`\rho_b^i`. A state that is not separable
is called entangled.

There are several ways to detect if a state is separable or entangled.
One of these methods is the PPT symmetric extension criterion by
:ref:`[4] <quantum_refs>`, which provides a heirarchy of semidefinite
representable criteria that must be satisfied by separable states.
For the :math:`k=2` level heirarchy, the corresponding feasibiltiy
problem is (see :ref:`[1] <quantum_refs>`)

.. math::

    \text{find} \quad \rho_{aB} \quad \text{s.t.} \quad & \text{tr}_{b2}(\rho_{aB}) \\
    & \rho_{aB} = \Pi_{b1b2} \rho_{aB} \Pi_{b1b2} \\
    & \text{tr}[\rho_{aB}] = 1 \\
    & \rho_{aB} \succeq 0 \\
    & \mathcal{T}_{b2}(\rho_{aB}) \succeq 0 \\
    & \mathcal{T}_{b1b2}(\rho_{aB}) \succeq 0.

We solve this feasibility problem for the entangled quantum state

.. math::

    \rho_{ab} = \frac{1}{2} \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 1 \end{bmatrix}

in **QICS** below.

.. code-block:: python

    import numpy as np
    import qics
    import qics.vectorize as vec
    import qics.quantum as qu

    n  = 2
    n2 = n * n
    n3 = n * n * n

    vn2 = vec.vec_dim(n2, compact=True)
    vn3 = vec.vec_dim(n3, compact=True)

    rho_ab = 0.5 * np.array([
        [1., 0., 0., 1.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [1., 0., 0., 1.]
    ])

    # Define objective function
    c = np.zeros((3*n3*n3, 1))

    # Build linear constraints
    # rho_ab1 = tr_b2(rho_aB)
    tr_b2 = vec.lin_to_mat(lambda X : qu.p_tr(X, (n, n, n), 2), (n3, n2))
    A1 = np.hstack((tr_b2, np.zeros((vn2, 2*n3*n3))))
    b1 = vec.mat_to_vec(rho_ab, compact=True)
    # rho_aB = swap_b1,b2(rho_aB)
    swap = vec.lin_to_mat(lambda X : qu.swap(X, (n, n, n), 1, 2), (n3, n3))
    A2 = np.hstack((swap - vec.eye(n3), np.zeros((vn3, 2*n3*n3))))
    b2 = np.zeros((vn3, 1))
    # tr[rho_aB] = 1
    tr = vec.mat_to_vec(np.eye(n3)).T
    A3 = np.hstack((tr, np.zeros((1, 2*n3*n3))))
    b3 = np.array([[1.]])
    # Y = T_b2(rho_aB)
    T_b2 = vec.lin_to_mat(lambda X : qu.partial_transpose(X, (n2, n), 1), (n3, n3))
    A4 = np.hstack((T_b2, -vec.eye(n3), np.zeros((vn3, n3*n3))))
    b4 = np.zeros((vn3, 1))
    # Z = T_b1b2(rho_aB)
    T_b1b2 = vec.lin_to_mat(lambda X : qu.partial_transpose(X, (n, n2), 1), (n3, n3))
    A5 = np.hstack((T_b1b2, np.zeros((vn3, n3*n3)), -vec.eye(n3)))
    b5 = np.zeros((vn3, 1))

    A = np.vstack((A1, A2, A3, A4, A5))
    b = np.vstack((b1, b2, b3, b4, b5))

    # Define cones to optimize over
    cones = [
        qics.cones.PosSemidefinite(n3),
        qics.cones.PosSemidefinite(n3),
        qics.cones.PosSemidefinite(n3)
    ]

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
            no. cones:  3                        no. vars:    192
            barr. par:  25                       no. constr:  119
            symmetric:  True                     cone dim:    192
            complex:    False

    ...

    Solution summary
            sol. status:  pinfeas                num. iter:    8
            exit status:  solved                 solve time:   0.030

            primal obj:   0.000000000000e+00     primal feas:  4.79e-01
            dual obj:     4.165819513474e+13     dual feas:    5.39e-01
            opt. gap:     4.17e+13    

As the semidefinite program is infeasible, then :math:`\rho_{ab}` must be
entangled, which we know is true for this quantum state.



.. _quantum_refs:

References
----------

    1. Siddhu, V. and Tayur, S.
       "Five starter pieces: Quantum Information Science via semidefinite programs", 
       Tutorials in Operations Research: Emerging and Impactful Topics in Operations, pp. 59–92. 2022.

    2. J. Watrous, “Simpler semidefinite programs for completely bounded norms,” arXiv preprint arXiv:1207.5726, 2012.

    3. Cole, S. et al. (2023) "On Quantum Optimal Transport", Mathematical Physics, Analysis and Geometry, 26(2).

    4. Andrew C. Doherty, Pablo A. Parrilo, and Federico M. Spedalieri. Complete family of
       separability criteria. Physical Review A, 69(2), Feb 2004.