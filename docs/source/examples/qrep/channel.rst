Quantum channels
==================

Shannon's seminal work for classical channels introduced
two important fundamental limits of transmitting information
through channels, and provided mathematical descriptions for 
these quantities :ref:`[1] <channel_refs>`.

    - **Channel capacity**: The maximum amount of information
      we can transmit through a noisy quantum channel.
    - **Rate-distortion**: The smallets amount of information
      required to compress and transmit a given message without
      exceeding a given distortion threshold.

We explore several of these settings below in the quantum setting.


Classical-quantum channel capacity
------------------------------------

Consider a discrete input alphabet :math:`\mathcal{X}=\{ x_1, \ldots, x_m \}`
where each letter is sent according to a probability distribution :math:`p\in\mathbb{R}^m`.
To send messages using this input alphabet through a quantum channel 
:math:`\mathcal{N}`, an encoder needs to first map these classical states 
to quantum states :math:`x_j\mapsto\rho_j`. After the state has been sent 
through the channel, a decoder performs a quantum measurement on the 
state to recover the original message.

For a given quantum channel :math:`\mathcal{N}`, the classical-quantum channel 
capacity is given by the Holevo-Schumacher–Westmoreland theorem :ref:`[2,3] <channel_refs>`

.. math::

    \max_{p \in \mathbb{R}^m} &&& S\biggl(\mathcal{N}\biggl(\sum_{i=1}^m p_i\rho_i\biggr)\biggr) - \sum_{i=1}^m p_iS(\mathcal{N}(\rho_i))

    \text{s.t.} &&& \sum_{i=1}^n p_i = 1

    &&& p \geq 0.

We can solve for this channel capacity using the :class:`~qics.cones.QuantEntr` cone
as follows

.. tabs::

    .. code-tab:: python Native

        import numpy
        import qics

        numpy.random.seed(1)

        n = m = 16

        rhos = [qics.quantum.random.density_matrix(n, iscomplex=True) for i in range(m)]

        # Define objective function
        # where x = ({pi}, t) and c = ({-S(N(Xi))}, 1)
        c1 = numpy.array([[-qics.quantum.quant_entropy(rho)] for rho in rhos])
        c2 = numpy.array([[1.0]])
        c  = numpy.vstack((c1, c2))

        # Build linear constraint Σ_i pi = 1
        A = numpy.hstack((numpy.ones((1, m)), numpy.zeros((1, 1))))
        b = numpy.ones((1, 1))

        # Build linear cone constraints
        # x_nn = p
        G1 = numpy.hstack((-numpy.eye(m), numpy.zeros((m, 1))))
        h1 = numpy.zeros((m, 1))
        # t_qe = t
        G2 = numpy.hstack((numpy.zeros((1, m)), -numpy.ones((1, 1))))
        h2 = numpy.zeros((1, 1))
        # u_qe = 1
        G3 = numpy.hstack((numpy.zeros((1, m)), numpy.zeros((1, 1))))
        h3 = numpy.ones((1, 1))
        # X_qe = Σ_i pi N(Xi)
        rhos_vec = numpy.hstack(([qics.vectorize.mat_to_vec(rho) for rho in rhos]))
        G4 = numpy.hstack((-rhos_vec, numpy.zeros((2*n*n, 1))))
        h4 = numpy.zeros((2*n*n, 1))

        G = numpy.vstack((G1, G2, G3, G4))
        h = numpy.vstack((h1, h2, h3, h4))

        # Input into model and solve
        cones = [
            qics.cones.NonNegOrthant(n), 
            qics.cones.QuantEntr(n, iscomplex=True)
        ]

        # Initialize model and solver objects
        model = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
        solver = qics.Solver(model)

        # Solve problem
        info = solver.solve()

    .. code-tab:: python PICOS

        import numpy
        import picos
        import qics

        numpy.random.seed(1)

        n = m = 16

        rhos = [qics.quantum.random.density_matrix(n, iscomplex=True) for i in range(m)]
        entr_rhos = numpy.array([[qics.quantum.quant_entropy(rho)] for rho in rhos])

        # Define problem
        P = picos.Problem()
        p = picos.RealVariable("p", m)
        average_rho = picos.sum([p[i]*rhos[i] for i in range(m)])

        P.set_objective("max", picos.qentr(average_rho) + (p | entr_rhos))
        P.add_constraint(picos.sum(p) == 1)
        P.add_constraint(p > 0)

        # Solve problem
        P.solve(solver="qics", verbosity=2)


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

For a given quantum channel :math:`\mathcal{N}`, with Stinespring representation
:math:`\mathcal{N}(X)=\text{tr}_E(V X V^\dagger)`, the entanglement-assisted channel 
capacity is given by the Bennet-Shor-Smolin-Thapliyal theorem :ref:`[4] <channel_refs>`

.. math::

    \max_{X \in \mathbb{H}^n} &&& -S( V X V^\dagger \| \mathbb{I} \otimes \text{tr}_B(V X V^\dagger) ) + S(\text{tr}_E(V X V^\dagger))

    \text{s.t.} &&& \text{tr}[X] = 1

    &&& X \succeq 0.

As a concrete example, consider the amplitude damping channel defined by the isometry

.. math::

    V = \begin{bmatrix} 1 & 0 \\ 0 & \sqrt{\gamma} \\ 0 & \sqrt{1-\gamma} \\ 0 & 0 \end{bmatrix}

and some parameter :math:`\gamma\in[0, 1]`. We can solve this in **QICS** as follows.

.. tabs::

    .. code-tab:: python Native

        import numpy
        import qics

        n = 2
        N = n * n
        gamma = 0.5

        V = numpy.array([
            [1., 0.                 ], 
            [0., numpy.sqrt(1-gamma)], 
            [0., numpy.sqrt(gamma)  ], 
            [0., 0.                 ]
        ])

        # Define objective functions
        # with variables (X, (t, Y), (s, u, Z))
        cX = numpy.zeros((n * n, 1))
        ct = numpy.array([[1.0 / numpy.log(2)]])
        cY = numpy.zeros((N * N, 1))
        cs = numpy.array([[1.0 / numpy.log(2)]])
        cu = numpy.array([[0.0]])
        cZ = numpy.zeros((n * n, 1))
        c = numpy.vstack((cX, ct, cY, cs, cu, cZ))

        # Build linear constraints
        vn = qics.vectorize.vec_dim(n, compact=True)
        vN = qics.vectorize.vec_dim(N, compact=True)
        VV = qics.vectorize.lin_to_mat(lambda X: V @ X @ V.T, (n, n * n))
        trE = qics.vectorize.lin_to_mat(
            lambda X: qics.quantum.p_tr(X, (n, n), 1), (N, n), compact=(True, True)
        )
        # tr[X] = 1
        A1 = numpy.hstack(
            (qics.vectorize.mat_to_vec(numpy.eye(n)).T, numpy.zeros((1, 3 + n * n + N * N)))
        )
        b1 = numpy.array([[1.0]])
        # u = 1
        A2 = numpy.hstack(
            (numpy.zeros((1, 2 + n * n + N * N)), numpy.array([[1.0]]), numpy.zeros((1, n * n)))
        )
        b2 = numpy.array([[1.0]])
        # Y = VXV'
        A3 = numpy.hstack(
            (VV, numpy.zeros((vN, 1)), -qics.vectorize.eye(N), numpy.zeros((vN, 2 + n * n)))
        )
        b3 = numpy.zeros((vN, 1))
        # Z = trE[VXV']
        A4 = numpy.hstack((trE @ VV, numpy.zeros((vn, 3 + N * N)), -qics.vectorize.eye(n)))
        b4 = numpy.zeros((vn, 1))

        A = numpy.vstack((A1, A2, A3, A4))
        b = numpy.vstack((b1, b2, b3, b4))

        # Input into model and solve
        cones = [
            qics.cones.PosSemidefinite(n),
            qics.cones.QuantCondEntr((n, n), 0),
            qics.cones.QuantEntr(n),
        ]

        # Initialize model and solver objects
        model = qics.Model(c=c, A=A, b=b, cones=cones)
        solver = qics.Solver(model)

        # Solve problem
        info = solver.solve()

    .. code-tab:: python PICOS

        import numpy
        import picos

        gamma = 0.5

        V = numpy.array([
            [1., 0.                 ], 
            [0., numpy.sqrt(1-gamma)], 
            [0., numpy.sqrt(gamma)  ], 
            [0., 0.                 ]
        ])

        # Define problem
        P = picos.Problem()
        X = picos.SymmetricVariable("X", 2)

        P.set_objective("max", (picos.qcondentr(V*X*V.T, 1) 
                        + picos.qentr(picos.partial_trace(V*X*V.T, 0))) / numpy.log(2))
        P.add_constraint(picos.trace(X) == 1)
        P.add_constraint(X >> 0)

        # Solve problem
        P.solve(solver="qics", verbosity=2)

Quantum channel capacity of degradable channels
-------------------------------------------------

Now we turn our attention to the scenario where we want to send quantum 
information through a quantum channel. Instead of a classical alphabet, 
the sender has a quantum alphabet will be encoded, transmitted, and decoded 
by the receiver. 

In general, the quantum channel capacity is given by a non-convex optimization
problem. However, when a channel :math:`\mathcal{N}` is degradable, meaning
its complementary channel :math:`\mathcal{N}_\text{c}` can be expressed as 
:math:`\mathcal{N}_\text{c}=\Xi\circ\mathcal{N}` for some quantum channel :math:`\Xi`,
then the quantum channel capacity is given by :ref:`[5] <channel_refs>`

.. math::

    \max_{X \in \mathbb{H}^n} &&& -S( W \mathcal{N}(X) W^\dagger \| \mathbb{I} \otimes \text{tr}_F(W \mathcal{N}(X) W^\dagger) )

    \text{s.t.} &&& \text{tr}[X] = 1

    &&& X \succeq 0,

where :math:`W` is the Stinespring isometry associated with :math:`\Xi`.

As a concrete example, again consider the amplitude damping channel, which
has Stinespring isometry for :math:`\Xi` given by

.. math::

    W = \begin{bmatrix} 1 & 0 \\ 0 & \sqrt{\delta} \\ 0 & \sqrt{1-\delta} \\ 0 & 0 \end{bmatrix}

where :math:`\delta=(1-2\gamma) / (1-\gamma)`.

.. tabs::

    .. code-tab:: python Native

        import numpy
        import qics

        n = 2
        N = n*n
        gamma = 0.5
        delta = (1-2*gamma) / (1-gamma)

        V = numpy.array([
            [1., 0.                 ], 
            [0., numpy.sqrt(1-gamma)], 
            [0., numpy.sqrt(gamma)  ], 
            [0., 0.                 ]
        ])

        W = numpy.array([
            [1., 0.                 ], 
            [0., numpy.sqrt(delta)], 
            [0., numpy.sqrt(1-delta)  ], 
            [0., 0.                 ]
        ])

        # Define objective functions
        # with variables (X, (t, Y))
        cX = numpy.zeros((n*n, 1))
        ct = numpy.array([[1./numpy.log(2)]])
        cY = numpy.zeros((N*N, 1))
        c = numpy.vstack((cX, ct, cY))

        # Build linear constraints
        vn = qics.vectorize.vec_dim(n, compact=True)
        vN = qics.vectorize.vec_dim(N, compact=True)
        WNW = qics.vectorize.lin_to_mat(
            lambda X : W @ qics.quantum.p_tr(V @ X @ V.T, (n, n), 1) @ W.T, 
            (n, N)
        )
        # tr[X] = 1
        A1 = numpy.hstack((qics.vectorize.mat_to_vec(numpy.eye(n)).T, numpy.zeros((1, 1 + N*N))))
        b1 = numpy.array([[1.]])
        # Y = WN(X)W'
        A2 = numpy.hstack((WNW, numpy.zeros((vN, 1)), -qics.vectorize.eye(N)))
        b2 = numpy.zeros((vN, 1))

        A = numpy.vstack((A1, A2))
        b = numpy.vstack((b1, b2))

        # Input into model and solve
        cones = [
            qics.cones.PosSemidefinite(n),
            qics.cones.QuantCondEntr((n, n), 1)
        ]

        # Initialize model and solver objects
        model  = qics.Model(c=c, A=A, b=b, cones=cones)
        solver = qics.Solver(model)

        # Solve problem
        info = solver.solve()

    .. code-tab:: python PICOS

        import numpy
        import picos

        gamma = 0.5
        delta = (1-2*gamma) / (1-gamma)

        V = numpy.array([
            [1., 0.                 ], 
            [0., numpy.sqrt(1-gamma)], 
            [0., numpy.sqrt(gamma)  ], 
            [0., 0.                 ]
        ])

        W = numpy.array([
            [1., 0.                 ], 
            [0., numpy.sqrt(delta)], 
            [0., numpy.sqrt(1-delta)  ], 
            [0., 0.                 ]
        ])

        # Define problem
        P = picos.Problem()
        X = picos.SymmetricVariable("X", 2)
        W_Nx_W = W * picos.partial_trace(V*X*V.T, 1) * W.T

        P.set_objective("max", picos.qcondentr(W_Nx_W, 1) / numpy.log(2))
        P.add_constraint(picos.trace(X) == 1)
        P.add_constraint(X >> 0)

        # Solve problem
        P.solve(solver="qics", verbosity=2)


Entanglement-assisted rate-distortion
----------------------------------------

Whereas channel capacities are interested in characterising the 
maximum rate of information we can trasmit in a lossless manner, 
the rate-distortion function is interested in the maximum amount 
we can compress information in a lossy manner to transmit over a
channel.

Consider a quantum state :math:`\sigma` which we want to compress.
The entanglement-assisted rate-distortion function is given by :ref:`[6,7] <channel_refs>`

.. math::

    R(D) \quad = &&\min_{\rho \in \mathbb{H}^{n^2}} &&& S( \rho \| \mathbb{I} \otimes \rho ) + S(\sigma)

    &&\text{s.t.} &&& \text{tr}[\rho] = 1

    &&&&& 1 - \langle \psi | \rho | \psi \rangle \leq D

    &&&&& \rho \succeq 0,

where :math:`| \psi \rangle` is the purification of :math:`\sigma`.

.. tabs::

    .. code-tab:: python Native

        import numpy
        import qics

        numpy.random.seed(1)

        n = 4
        D = 0.25

        rho = qics.quantum.random.density_matrix(n)
        entr_rho = qics.quantum.quant_entropy(rho)

        N = n * n
        sn = qics.vectorize.vec_dim(n, compact=True)
        vN = qics.vectorize.vec_dim(N)

        # Define objective function
        c = numpy.zeros((vN + 2, 1))
        c[0] = 1.

        # Build linear constraint matrices
        tr2 = qics.vectorize.lin_to_mat(lambda X : qics.quantum.p_tr(X, (n, n), 1), (N, n))
        purification = qics.vectorize.mat_to_vec(qics.quantum.purify(rho))
        # Tr_2[X] = rho
        A1 = numpy.hstack((numpy.zeros((sn, 1)), tr2, numpy.zeros((sn, 1))))
        b1 = qics.vectorize.mat_to_vec(rho, compact=True)
        # 1 - tr[Psi X] <= D
        A2 = numpy.hstack((numpy.zeros((1, 1)), -purification.T, numpy.ones((1, 1))))
        b2 = numpy.array([[D - 1]])

        A = numpy.vstack((A1, A2))
        b = numpy.vstack((b1, b2))

        # Define cones to optimize over
        cones = [
            qics.cones.QuantCondEntr((n, n), 0), 
            qics.cones.NonNegOrthant(1)
        ]

        # Initialize model and solver objects
        model  = qics.Model(c=c, A=A, b=b, cones=cones, offset=entr_rho)
        solver = qics.Solver(model)

        # Solve problem
        info = solver.solve()

    .. code-tab:: python PICOS

        import numpy
        import picos
        import qics

        numpy.random.seed(1)

        n = 4
        D = 0.25

        rho = qics.quantum.random.density_matrix(n)
        entr_rho = qics.quantum.quant_entropy(rho)
        distortion_observable = picos.I(n*n) - qics.quantum.purify(rho)

        # Define problem
        P = picos.Problem()
        X = picos.SymmetricVariable("X", n*n)

        P.set_objective("min", -picos.qcondentr(X, 0, (n, n)) + entr_rho)
        P.add_constraint(picos.partial_trace(X, 1, (n, n)) == rho)
        P.add_constraint((X | distortion_observable) < D)

        # Solve problem
        P.solve(solver="qics", verbosity=2)

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