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

.. code-block:: python

    import numpy as np
    import qics
    import qics.vectorize as vec
    import qics.quantum as qu

    np.random.seed(1)

    n = m = 16

    rhos = [qu.random.density_matrix(n, iscomplex=True) for i in range(m)]

    # Define objective function
    # where x = ({pi}, t) and c = ({-S(N(Xi))}, 1)
    c1 = np.array([[-qu.quant_entropy(rho)] for rho in rhos])
    c2 = np.array([[1.0]])
    c  = np.vstack((c1, c2))

    # Build linear constraint Σ_i pi = 1
    A = np.hstack((np.ones((1, m)), np.zeros((1, 1))))
    b = np.ones((1, 1))

    # Build linear cone constraints
    # x_nn = p
    G1 = np.hstack((-np.eye(m), np.zeros((m, 1))))
    h1 = np.zeros((m, 1))
    # t_qe = t
    G2 = np.hstack((np.zeros((1, m)), -np.ones((1, 1))))
    h2 = np.zeros((1, 1))
    # u_qe = 1
    G3 = np.hstack((np.zeros((1, m)), np.zeros((1, 1))))
    h3 = np.ones((1, 1))
    # X_qe = Σ_i pi N(Xi)
    rhos_vec = np.hstack(([vec.mat_to_vec(rho) for rho in rhos]))
    G4 = np.hstack((-rhos_vec, np.zeros((2*n*n, 1))))
    h4 = np.zeros((2*n*n, 1))

    G = np.vstack((G1, G2, G3, G4))
    h = np.vstack((h1, h2, h3, h4))

    # Input into model and solve
    cones = [
        qics.cones.NonNegOrthant(n), 
        qics.cones.QuantEntr(n, iscomplex=True)
    ]

    # Initialize model and solver objects
    model  = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones)
    solver = qics.Solver(model)

    # Solve problem
    info = solver.solve()

.. code-block:: none

    ====================================================================
                QICS v0.0 - Quantum Information Conic Solver
                by K. He, J. Saunderson, H. Fawzi (2024)
    ====================================================================
    Problem summary:
            no. cones:  2                        no. vars:    17
            barr. par:  35                       no. constr:  1
            symmetric:  False                    cone dim:    530
            complex:    True

    ...

    Solution summary
            sol. status:  optimal                num. iter:    20
            exit status:  solved                 solve time:   1.246

            primal obj:  -5.030868958255e+00     primal feas:  1.80e-09
            dual obj:    -5.030868964121e+00     dual feas:    4.03e-09
            opt. gap:     1.17e-09


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

.. code-block:: python

    import numpy as np
    import qics
    import qics.vectorize as vec

    n = 2
    N = n*n
    gamma = 0.5

    V = np.array([[1, 0], [0, np.sqrt(gamma)], [0, np.sqrt{1-gamma}], [0, 0]])

    # Define objective functions
    # with variables (X, (t, Y), (s, u, Z))
    cX = np.zeros((n*n, 1))
    ct = np.array([[1./np.log(2)]])
    cY = np.zeros((N*N, 1))
    cs = np.array([[1./np.log(2)]])
    cu = np.array([[0.]])
    cZ = np.zeros((n*n, 1))
    c = np.vstack((cX, ct, cY, cs, cu, cZ))

    # Build linear constraints
    vn = vec.vec_dim(n, compact=True)
    vN = vec.vec_dim(N, compact=True)
    VV = vec.lin_to_mat(lambda X : V @ X @ V.conj().T, (n, n*n))
    trE = vec.lin_to_mat(lambda X : qu.p_tr(X, (n, n), 1), (N, n), compact=(True, True))
    # tr[X] = 1
    A1 = np.hstack((vec.mat_to_vec(np.eye(n)).T, np.zeros((1, 3 + n*n + N*N))))
    b1 = np.array([[1.]])
    # u = 1
    A2 = np.hstack((np.zeros((1, 2 + n*n + N*N)), np.array([[1.]]), np.zeros((1, n*n))))
    b2 = np.array([[1.]])
    # Y = VXV'
    A3 = np.hstack((VV, np.zeros((vN, 1)), -vec.eye(N), np.zeros((vN, 2 + n*n))))
    b3 = np.zeros((vN, 1))
    # Z = trE[VXV']
    A4 = np.hstack((trE @ VV, np.zeros((vn, 3 + N*N)), -vec.eye(n)))
    b4 = np.zeros((vn, 1))

    A = np.vstack((A1, A2, A3, A4))
    b = np.vstack((b1, b2, b3, b4))

    # Input into model and solve
    cones = [
        qics.cones.PosSemidefinite(n),
        qics.cones.QuantCondEntr((n, n), 0), 
        qics.cones.QuantEntr(n)
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
            no. cones:  3                        no. vars:    27
            barr. par:  12                       no. constr:  15
            symmetric:  False                    cone dim:    27
            complex:    False

    ...

    Solution summary
            sol. status:  optimal                num. iter:    15
            exit status:  solved                 solve time:   1.473

            primal obj:  -1.000000044516e+00     primal feas:  1.16e-09
            dual obj:    -1.000000037504e+00     dual feas:    5.26e-10
            opt. gap:     7.01e-09



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

.. code-block:: python

    import numpy as np
    import qics
    import qics.vectorize as vec

    n = 2
    N = n*n
    gamma = 0.5
    delta = (1-2*gamma) / (1-gamma)

    V = np.array([[1, 0], [0, np.sqrt(gamma)], [0, np.sqrt(1-gamma)], [0, 0]])
    W = np.array([[1, 0], [0, np.sqrt(delta)], [0, np.sqrt(1-delta)], [0, 0]])

    # Define objective functions
    # with variables (X, (t, Y))
    cX = np.zeros((n*n, 1))
    ct = np.array([[1./np.log(2)]])
    cY = np.zeros((N*N, 1))
    c = np.vstack((cX, ct, cY))

    # Build linear constraints
    vn = vec.vec_dim(n, compact=True)
    vN = vec.vec_dim(N, compact=True)
    WNW = vec.lin_to_mat(
        lambda X : W @ qu.p_tr(V @ X @ V.conj().T, (n, n), 1) @ W.conj().T, 
        (n, N)
    )
    # tr[X] = 1
    A1 = np.hstack((vec.mat_to_vec(np.eye(n)).T, np.zeros((1, 1 + N*N))))
    b1 = np.array([[1.]])
    # Y = WN(X)W'
    A2 = np.hstack((WNW, np.zeros((vN, 1)), -vec.eye(N)))
    b2 = np.zeros((vN, 1))

    A = np.vstack((A1, A2))
    b = np.vstack((b1, b2))

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

.. code-block:: none

    ====================================================================
                QICS v0.0 - Quantum Information Conic Solver
                by K. He, J. Saunderson, H. Fawzi (2024)
    ====================================================================
    Problem summary:
            no. cones:  2                        no. vars:    21
            barr. par:  8                        no. constr:  11
            symmetric:  False                    cone dim:    21
            complex:    False

    ...

    Solution summary
            sol. status:  optimal                num. iter:    15
            exit status:  solved                 solve time:   1.442

            primal obj:  -1.729304935610e-08     primal feas:  2.97e-10
            dual obj:    -1.126936521182e-08     dual feas:    1.35e-10
            opt. gap:     6.02e-09


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

.. code-block:: python

    import numpy as np
    import qics
    import qics.vectorize as vec
    import qics.quantum as qu

    np.random.seed(1)

    n = 4
    D = 0.25

    rho      = qu.random.density_matrix(n)
    entr_rho = qu.quant_entropy(rho)

    N = n * n
    sn = vec.vec_dim(n, compact=True)
    vN = vec.vec_dim(N)

    # Define objective function
    c = np.zeros((vN + 2, 1))
    c[0] = 1.

    # Build linear constraint matrices
    tr2 = vec.lin_to_mat(lambda X : qu.p_tr(X, (n, n), 1), (N, n))
    purification = vec.mat_to_vec(qu.purify(rho))
    # Tr_2[X] = rho
    A1 = np.hstack((np.zeros((sn, 1)), tr2, np.zeros((sn, 1))))
    b1 = vec.mat_to_vec(rho, compact=True)
    # 1 - tr[Psi X] <= D
    A2 = np.hstack((np.zeros((1, 1)), -purification.T, np.ones((1, 1))))
    b2 = np.array([[D - 1]])

    A = np.vstack((A1, A2))
    b = np.vstack((b1, b2))

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

.. code-block:: none

    ====================================================================
                QICS v0.0 - Quantum Information Conic Solver
                by K. He, J. Saunderson, H. Fawzi (2024)
    ====================================================================
    Problem summary:
            no. cones:  2                        no. vars:    258
            barr. par:  19                       no. constr:  11
            symmetric:  False                    cone dim:    258
            complex:    False

    ...

    Solution summary
            sol. status:  optimal                num. iter:    21
            exit status:  solved                 solve time:   1.489

            primal obj:   5.121637612238e-01     primal feas:  2.73e-09
            dual obj:     5.121637686593e-01     dual feas:    1.36e-09
            opt. gap:     7.44e-09


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