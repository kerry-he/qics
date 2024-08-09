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

We can solve for this channel capacity using the :class:`~qics.QuantEntr` cone
as follows

.. code-block:: python

    import numpy as np
    import qics
    import qics.utils.symmetric as sym
    import qics.utils.quantum as qu

    np.random.seed(1)

    n = m = 16

    rhos = [qu.rand_density_matrix(n, iscomplex=True) for i in range(m)]

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
    rhos_vec = np.hstack(([sym.mat_to_vec(rho) for rho in rhos]))
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

.. code-block: none

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

    \max_{X \in \mathbb{H}^n} &&& -S( V X V^\dagger \| \mathbb{I} \otimes \tr_E(V X V^\dagger) ) + S(\tr_B(V X V^\dagger))

    \text{s.t.} &&& \tr[X] = 1

    &&& X \succeq 0.

As a concrete example, consider the amplitude damping channel defined by the isometry

.. math::

    V = \begin{bmatrix} 1 & 0 \\ 0 & \sqrt{\gamma} \\ 0 & \sqrt(1-\gamma) \\ 0 & 0 \end{bmatrix}

and some parameter :math:`\gamma\in[0, 1]`. We can solve this in **QICS** as follows.



Entanglement-assisted rate-distortion
----------------------------------------






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