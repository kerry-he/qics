Quantum channels
================

Shannon's seminal work for classical channels introduced two important 
fundamental limits of sending information through channels, and provided
mathematical descriptions for these quantities :ref:`[1]<channel_refs>`.

- **Channel capacity**: The maximum amount of information we can transmit 
  through a noisy quantum channel.
- **Rate-distortion**: The minimum amount of information required to compress
  and transmit a given message without exceeding a given distortion threshold.

We explore several of these settings below in the quantum setting.


Classical-quantum channel capacity
----------------------------------

Consider a discrete input alphabet :math:`\mathcal{X}=\{ x_1, \ldots, x_m \}`
where each letter is sent according to a probability distribution 
:math:`p\in\mathbb{R}^m`. To send messages using this input alphabet through a 
quantum channel :math:`\mathcal{N}`, an encoder needs to first map these 
classical states to quantum states :math:`x_j\mapsto\rho_j`. After the state has
been sent through the channel, a decoder performs a quantum measurement on the 
state to recover the original message.

Given this setup, for a given quantum channel :math:`\mathcal{N}` which is 
characterized as a completely positive trace preserving linear map, the 
classical-quantum channel capacity is given by the 
Holevo-Schumacher–Westmoreland theorem :ref:`[2,3] <channel_refs>`

.. math::

    \max_{p \in \mathbb{R}^m} &&& S\biggl(\mathcal{N}\biggl(\sum_{i=1}^m 
    p_i\rho_i\biggr)\biggr) - \sum_{i=1}^m p_iS(\mathcal{N}(\rho_i))

    \text{subj. to} &&& \sum_{i=1}^n p_i = 1

    &&& p \geq 0.

We can solve for this channel capacity using the :class:`qics.cones.QuantEntr`
cone as follows.

.. tabs::

    .. code-tab:: python Native

        import numpy as np

        import qics
        from qics.quantum import entropy
        from qics.quantum.random import density_matrix
        from qics.vectorize import mat_to_vec, vec_dim

        np.random.seed(1)

        n = m = 16
        vn = vec_dim(n, iscomplex=True)

        # Generate random problem data
        rhos = [density_matrix(n, iscomplex=True) for _ in range(m)]
        rho_vecs = np.hstack(([mat_to_vec(rho) for rho in rhos]))

        # Model problem using primal variables (p, t)
        # Define objective function
        c_p = np.array([[-entropy(rho)] for rho in rhos])
        c_t = 1.0
        c = np.block([[c_p], [c_t]])

        # Build linear constraint Σ_i pi = 1
        A = np.block([np.ones((1, m)), 0.0])
        b = np.array([[1.0]])

        # Build linear cone constraints
        G = np.block([
            [-np.eye(m),       np.zeros((m, 1)) ],  # x_nn = p
            [np.zeros((1, m)), -np.ones((1, 1)) ],  # t_qe = t
            [np.zeros((1, m)), np.zeros((1, 1)) ],  # u_qe = 1
            [-rho_vecs,        np.zeros((vn, 1))]   # X_qe = Σ_i pi N(Xi)
        ])  

        h = np.block([[np.zeros((m, 1))], [0.0], [1.0], [np.zeros((vn, 1))]])

        # Define cones to optimize over
        cones = [
            qics.cones.NonNegOrthant(n),  # p >= 0
            qics.cones.QuantEntr(n, iscomplex=True),  # (t, 1, Σ_i pi N(Xi)) ∈ QE
        ]

        # Initialize model and solver objects
        model = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
        solver = qics.Solver(model)

        # Solve problem
        info = solver.solve()

    .. code-tab:: python PICOS

        import numpy as np

        import picos
        import qics

        np.random.seed(1)

        n = m = 16

        rhos = [qics.quantum.random.density_matrix(n, iscomplex=True) for i in range(m)]
        entr_rhos = np.array([[qics.quantum.entropy(rho)] for rho in rhos])

        # Define problem
        P = picos.Problem()
        p = picos.RealVariable("p", m)
        average_rho = picos.sum([p[i]*rhos[i] for i in range(m)])

        P.set_objective("max", picos.quantentr(average_rho) + (p | entr_rhos))
        P.add_constraint(picos.sum(p) == 1)
        P.add_constraint(p > 0)

        # Solve problem
        P.solve(solver="qics", verbosity=1)

Entanglement-assisted channel capacity
----------------------------------------

Consider the same alphabet and channel setup as the classical-quantum channel 
capacity. However, the sender and receiver share an unlimited number of 
entangled states prior to sending messages through the channel. Like before, 
the sender chooses a classical message to send, but now encodes their part of
the entangled state to represent this message, then sends this through the 
quantum channel. The receiver combines this state with their own part of the 
entangled state, then jointly performs a measurement on them to recover the 
original message.

Given this setup, for a given quantum channel :math:`\mathcal{N}` with 
Stinespring representation :math:`\mathcal{N}(\rho)=\text{tr}_E(V\rho
V^\dagger)`, the entanglement-assisted channel capacity is given by the 
Bennet-Shor-Smolin-Thapliyal theorem :ref:`[4] <channel_refs>`

.. math::

    \max_{\rho \in \mathbb{H}^n} &&& S(\rho) - S(\text{tr}_B(V \rho V^\dagger)) 
    + S(\text{tr}_E(V \rho V^\dagger))

    \text{subj. to} &&& \text{tr}[\rho] = 1

    &&& \rho \succeq 0.

The objective function is known as the quantum mutual information, and by 
recognizing that :math:`S(V\rho V^\dagger)=S(\rho)`, we can model this function
by using a combination of :class:`qics.cones.QuantCondEntr` and 
:class:`qics.cones.QuantEntr`. As a concrete example, consider the amplitude
damping channel defined by the isometry

.. math::

    V = \begin{bmatrix} 
        1 & 0 \\ 0 & \sqrt{\gamma} \\ 0 & \sqrt{1-\gamma} \\ 0 & 0 
    \end{bmatrix}

and some parameter :math:`\gamma\in[0, 1]`. We can solve this in **QICS** as 
follows.

.. tabs::

    .. code-tab:: python Native

        import numpy as np

        import qics
        from qics.quantum import p_tr
        from qics.vectorize import eye, lin_to_mat, vec_dim

        n = 2
        N = n * n

        vn = vec_dim(n)
        vN = vec_dim(N)
        cn = vec_dim(n, compact=True)

        # Define amplitude damping channel
        gamma = 0.5
        V = np.array([
            [1., 0.              ],
            [0., np.sqrt(1-gamma)],
            [0., np.sqrt(gamma)  ],
            [0., 0.              ]
        ])  

        # Model problem using primal variables (t1, t2, cvec(X))
        # Define objective functions
        c = np.block([[1.0], [1.0], [np.zeros((cn, 1))]])

        # Build linear constraint tr[X] = 1
        trace = lin_to_mat(lambda X: np.trace(X), (n, 1), compact=(True, False))
        A = np.block([[0.0, 0.0, trace]])
        b = np.array([[1.0]])

        # Build conic linear constraints
        VV = lin_to_mat(lambda X: V @ X @ V.conj().T, (n, N), compact=(True, False))
        trE = lin_to_mat(lambda X: p_tr(X, (n, n), 0), (N, n), compact=(False, False))

        G = np.block([
            [-1.0,              0.0,               np.zeros((1, cn))],  # t_qce = t1
            [np.zeros((vN, 1)), np.zeros((vN, 1)), -VV              ],  # X_qce = VXV'
            [0.0,               -1.0,              np.zeros((1, cn))],  # t_qe = t2
            [0.0,               0.0,               np.zeros((1, cn))],  # u_qe = 1
            [np.zeros((vn, 1)), np.zeros((vn, 1)), -trE @ VV        ],  # X_qe = tr_E(VXV')
            [np.zeros((vn, 1)), np.zeros((vn, 1)), -eye(n).T        ]   # X_psd = X
        ])  

        h = np.block([
            [0.0], 
            [np.zeros((vN, 1))], 
            [0.0], 
            [1.0], 
            [np.zeros((vn, 1))], 
            [np.zeros((vn, 1))],
        ])  

        # Define cones to optimize over
        cones = [
            qics.cones.QuantCondEntr((n, n), 1),  # (t1, VXV') ∈ QCE
            qics.cones.QuantEntr(n),  # (t2, 1, tr_E(XVX')) ∈ QE
            qics.cones.PosSemidefinite(n),  # X ⪰ 0
        ]

        # Initialize model and solver objects
        model = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
        solver = qics.Solver(model)

        # Solve problem
        info = solver.solve()

    .. code-tab:: python PICOS

        import numpy as np

        import picos

        gamma = 0.5

        V = np.array([
            [1., 0.              ],
            [0., np.sqrt(1-gamma)],
            [0., np.sqrt(gamma)  ],
            [0., 0.              ]
        ])

        # Define problem
        P = picos.Problem()
        X = picos.SymmetricVariable("X", 2)

        P.set_objective("max", picos.quantcondentr(V*X*V.T, 1)
                        + picos.quantentr(picos.partial_trace(V*X*V.T, 0)))
        P.add_constraint(picos.trace(X) == 1)
        P.add_constraint(X >> 0)

        # Solve problem
        P.solve(solver="qics", verbosity=1)

Quantum channel capacity of degradable channels
-------------------------------------------------

Now we turn our attention to the scenario where we want to send quantum 
information through a quantum channel. Instead of a classical alphabet, 
the sender has a quantum alphabet will be encoded, transmitted, and decoded 
by the receiver. 

In general, the quantum channel capacity is given by a non-convex optimization
problem. However, when a channel :math:`\mathcal{N}` is degradable, meaning
its complementary channel :math:`\mathcal{N}_\text{c}` can be expressed as 
:math:`\mathcal{N}_\text{c}=\Xi\circ\mathcal{N}` for some quantum channel 
:math:`\Xi`, then the quantum channel capacity is given by 
:ref:`[5] <channel_refs>`

.. math::

    \max_{\rho \in \mathbb{H}^n} &&& S(\mathcal{N}(\rho)) -
    S(\text{tr}_F(W \mathcal{N}(\rho) W^\dagger))

    \text{subj. to} &&& \text{tr}[\rho] = 1

    &&& \rho \succeq 0,

where :math:`W` is the Stinespring isometry associated with :math:`\Xi`. Like
the entanglement-assisted channel capacity example, we can model this using
:class:`qics.cones.QuantCondEntr`. As a concrete example, again consider the 
amplitude damping channel, which has Stinespring isometry for :math:`\Xi` given 
by

.. math::

    W = \begin{bmatrix} 
        1 & 0 \\ 0 & \sqrt{\delta} \\ 0 & \sqrt{1-\delta} \\ 0 & 0 
    \end{bmatrix}

where :math:`\delta=(1-2\gamma) / (1-\gamma)`. We show how QICS can solve this
problem below.

.. tabs::

    .. code-tab:: python Native

        import numpy as np

        import qics
        from qics.quantum import p_tr
        from qics.vectorize import eye, lin_to_mat, vec_dim

        n = 2
        N = n * n

        vn = vec_dim(n)
        vN = vec_dim(N)
        cn = vec_dim(n, compact=True)

        # Define amplitude damping channel
        gamma = 0.25
        delta = (1 - 2 * gamma) / (1 - gamma)
        V = np.array([
            [1., 0.              ],
            [0., np.sqrt(1-gamma)],
            [0., np.sqrt(gamma)  ],
            [0., 0.              ]
        ])  
        W = np.array([
            [1., 0.              ],
            [0., np.sqrt(delta)  ],
            [0., np.sqrt(1-delta)],
            [0., 0.              ]
        ])  


        def W_NX_W(X):
            return W @ p_tr(V @ X @ V.conj().T, (n, n), 1) @ W.conj().T


        # Model problem using primal variables (t, cvec(X))
        # Define objective functions
        c = np.block([[1.0], [np.zeros((cn, 1))]])

        # Build linear constraint tr[X] = 1
        trace = lin_to_mat(lambda X: np.trace(X), (n, 1), compact=(True, False))
        A = np.block([[0.0, trace]])
        b = np.array([[1.0]])

        # Build conic linear constraints
        W_NX_W_mat = lin_to_mat(W_NX_W, (n, N), compact=(True, False))

        G = np.block([
            [-1.0,              np.zeros((1, cn))],  # t_qce = t
            [np.zeros((vN, 1)), -W_NX_W_mat      ],  # X_qce = WN(X)W'
            [np.zeros((vn, 1)), -eye(n).T        ]   # X_psd = X
        ])  

        h = np.block([[0.0], [np.zeros((vN, 1))], [np.zeros((vn, 1))]])

        # Define cones to optimize over
        cones = [
            qics.cones.QuantCondEntr((n, n), 1),  # (t, WN(X)W') ∈ QCE
            qics.cones.PosSemidefinite(n),  # X ⪰ 0
        ]

        # Initialize model and solver objects
        model = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
        solver = qics.Solver(model)

        # Solve problem
        info = solver.solve()

    .. code-tab:: python PICOS

        import numpy as np

        import picos

        gamma = 0.25
        delta = (1-2*gamma) / (1-gamma)

        V = np.array([
            [1., 0.              ],
            [0., np.sqrt(1-gamma)],
            [0., np.sqrt(gamma)  ],
            [0., 0.              ]
        ])

        W = np.array([
            [1., 0.              ],
            [0., np.sqrt(delta)  ],
            [0., np.sqrt(1-delta)],
            [0., 0.              ]
        ])

        # Define problem
        P = picos.Problem()
        X = picos.SymmetricVariable("X", 2)
        W_Nx_W = W * picos.partial_trace(V*X*V.T, 1) * W.T

        P.set_objective("max", picos.quantcondentr(W_Nx_W, 1))
        P.add_constraint(picos.trace(X) == 1)
        P.add_constraint(X >> 0)

        # Solve problem
        P.solve(solver="qics", verbosity=1)

Entanglement-assisted rate-distortion
----------------------------------------

Whereas channel capacities quantify the maximum rate of information we can
trasmit in a lossless manner, the rate-distortion function quantifies the
maximum amount we can compress information in a lossy manner to transmit over a
channel.

Consider a quantum state :math:`\sigma_A` which we want to transmit in a lossy
manner, without exceeding a distortion threshold :math:`D`. The minimum amount
of information required to do this is given by the entanglement-assisted
rate-distortion function :ref:`[6,7] <channel_refs>`, which involves solving

.. math::

    \min_{\rho_{AB} \in \mathbb{H}^{n^2}} &&& -S(\rho_{AB}) 
    + S(\text{tr}_A(\rho_{AB})) + S(\sigma_A)

    \text{subj. to} &&& \text{tr}_B(\rho_{AB}) = \sigma_A

    &&& 1 - \langle \psi | \rho_{AB} | \psi \rangle \leq D

    &&& \rho_{AB} \succeq 0,

where :math:`| \psi \rangle` is the purification of :math:`\sigma_A`. We can
model this problem using :class:`qics.cones.QuantCondEntr`, which we demonstrate
below.

.. tabs::

    .. code-tab:: python Native

        import numpy as np

        import qics
        from qics.quantum import p_tr, purify, entropy
        from qics.quantum.random import density_matrix
        from qics.vectorize import lin_to_mat, vec_dim, mat_to_vec

        np.random.seed(1)

        n = 4
        N = n * n
        vN = vec_dim(N, iscomplex=True)
        cn = vec_dim(n, iscomplex=True, compact=True)

        # Define random problem data
        rho = density_matrix(n, iscomplex=True)
        entr_rho = entropy(rho)
        rho_cvec = mat_to_vec(rho, compact=True)

        D = 0.25
        Delta = np.eye(N) - purify(rho)
        Delta_vec = mat_to_vec(Delta)

        # Model problem using primal variables (t, X, d)
        # Define objective function
        c = np.block([[1.0], [np.zeros((vN, 1))], [0.0]])

        # Build linear constraint matrices
        tr_B = lin_to_mat(lambda X : p_tr(X, (n, n), 1), (N, n), iscomplex=True)

        A = np.block([
            [np.zeros((cn, 1)), tr_B,        np.zeros((cn, 1))],  # tr_B[X] = rho
            [0.0,               Delta_vec.T, 1.0              ]   # d = D - <Delta, X>
        ])

        b = np.block([[rho_cvec], [D]])

        # Define cones to optimize over
        cones = [
            qics.cones.QuantCondEntr((n, n), 0, iscomplex=True),  # (t, X) ∈ QCE
            qics.cones.NonNegOrthant(1)  # d = D - <Delta, X> >= 0
        ]

        # Initialize model and solver objects
        model  = qics.Model(c=c, A=A, b=b, cones=cones, offset=entr_rho)
        solver = qics.Solver(model)

        # Solve problem
        info = solver.solve()

    .. code-tab:: python PICOS

        import numpy as np

        import picos
        import qics

        np.random.seed(1)

        n = 4
        D = 0.25

        rho = qics.quantum.random.density_matrix(n)
        entr_rho = qics.quantum.entropy(rho)
        distortion_observable = picos.I(n*n) - qics.quantum.purify(rho)

        # Define problem
        P = picos.Problem()
        X = picos.SymmetricVariable("X", n*n)

        P.set_objective("min", -picos.quantcondentr(X, 0, (n, n)) + entr_rho)
        P.add_constraint(picos.partial_trace(X, 1, (n, n)) == rho)
        P.add_constraint((X | distortion_observable) < D)

        # Solve problem
        P.solve(solver="qics", verbosity=1)

.. _channel_refs:

References
----------

    1. C. E. Shannon, “A mathematical theory of communication,” The Bell
       system technical journal, vol. 27, no. 3, pp. 379–423, 1948.

    2. B. Schumacher and M. D. Westmoreland, “Sending classical information
       via noisy quantum channels,” Physical Review A, vol. 56, no. 1, p. 131,
       1997.

    3. A. S. Holevo, “The capacity of the quantum channel with general signal
       states,” IEEE Transactions on Information Theory, vol. 44, no. 1, pp. 269–
       273, 1998.

    4. C. H. Bennett, P. W. Shor, J. A. Smolin, and A. V. Thapliyal,
       “Entanglement-assisted capacity of a quantum channel and the reverse
       shannon theorem,” IEEE transactions on Information Theory, vol. 48,
       no. 10, pp. 2637–2655, 2002.

    5. I. Devetak and P. W. Shor, “The capacity of a quantum channel for simultaneous transmission
       of classical and quantum information,” Communications in Mathematical Physics, vol. 256, pp.
       287–303, 2005.

    6. N. Datta, M.-H. Hsieh, and M. M. Wilde, “Quantum rate distortion, reverse Shannon theorems, and
       source-channel separation,” IEEE Transactions on Information Theory, vol. 59, no. 1, pp. 615–630,
       2012.

    7. M. M. Wilde, N. Datta, M.-H. Hsieh, and A. Winter, “Quantum rate-distortion coding with auxiliary
       resources,” IEEE Transactions on Information Theory, vol. 59, no. 10, pp. 6755–6773, 2013.