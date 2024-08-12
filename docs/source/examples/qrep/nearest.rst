Nearest matrix
==================

A common optimization problem that arises is to find the 
nearest matrix to another fixed matrix with respect to the 
quantum relative entropy. We show how some of these problems
can be solved in **QICS** below.


Bregman projection
--------------------

For a given convex function :math:`\varphi:\text{dom}\ \varphi\rightarrow\mathbb{R}`,
the associated Bregman divergence :math:`D_\varphi : \text{dom}\ \varphi\times\text{int}\ \text{dom}\ \varphi\rightarrow\mathbb{R}`

.. math::

    D_\varphi( x \| y ) = \varphi(x) - \varphi(y) - \langle \nabla\varphi(y), x - y \rangle .  

A Bregman projection of a point :math:`\mathcal{y}` onto a set 
:math:`\mathcal{C}\subset\text{dom}\ \varphi` is given by

.. math::

    \min_{x \in \mathcal{C}} \quad D_\varphi( x \| y ).

When :math:`\varphi(X)=-S(X)` is the negative quantum entropy, 
the Bregman divergence is the (normalized) quantum relative entropy
:math:`D_\varphi( X \| Y ) = S( X \| Y ) - \text{tr}[X - Y]`.

The Bregman projection for this kernel of a matrix :math:`Y\in\mathbb{H}^n` 
onto the intersection of the positive semidefinite cone and a polytope is 

.. math::

    \min_{X \in \mathbb{H}^n} &&& S( X \| Y ) - \text{tr}[X - Y]

    \text{s.t.} &&& \langle A_i, X \rangle = b_i \qquad i=1,\ldots,p

    &&& X \succeq 0,

For linear constraints encoded by :math:`A_i\in\mathbb{H}^n` and 
:math:`b_i\in\mathbb{R}` for :math:`i=1,\ldots,p`. As the second argument
of the quantum relative entropy is fixed, we can model the problem
using just the quantum entropy cone.

.. math::

    \min_{t,u\in\mathbb{R},\  X \in \mathbb{H}^n} &&& t - \langle \log(Y)+\mathbb{I}, X \rangle + \text{tr}[Y]

    \text{s.t.} \quad\; &&& u = 1

    &&& \langle A_i, X \rangle = b_i \qquad i=1,\ldots,p

    &&& (t, u, X) \in \mathcal{K}_{\text{qe}}^n,

.. code-block:: python

    import numpy as np
    import scipy as sp
    import qics

    from qics.vectorize import mat_to_vec

    np.random.seed(1)

    n = 5
    p = 2

    # Generate random matrix Y to project
    Y = np.random.randn(n, n) + np.random.randn(n, n)*1j
    Y = Y @ Y.conj().T
    tr_Y = np.trace(Y).real

    # Define objective function
    ct = np.array([[1.]])
    cu = np.array([[0.]])
    cX = -sp.linalg.logm(Y) - np.eye(n)
    c  = np.vstack((ct, cu, mat_to_vec(cX)))

    # Build linear constraints
    # u = 1
    A1 = np.hstack((np.array([[0., 1.]]), np.zeros((1, 2*n*n))))
    b1 = np.array([[1.]])
    # <X, Ai> = bi for randomly generated Ai, bi
    A2 = np.zeros((p, 2 + 2*n*n))
    for i in range(p):
        Ai = np.random.randn(n, n) + np.random.rand(n, n)*1j
        A2[[i], 2:] = mat_to_vec(Ai + Ai.conj().T).T
    b2 = np.random.randn(p, 1)

    A = np.vstack((A1, A2))
    b = np.vstack((b1, b2))

    # Define cones to optimize over
    cones = [qics.cones.QuantEntr(n, iscomplex=True)]

    # Initialize model and solver objects
    model  = qics.Model(c=c, A=A, b=b, cones=cones, offset=tr_Y)
    solver = qics.Solver(model)

    # Solve problem
    info = solver.solve()

.. code-block:: none

    ====================================================================
                QICS v0.0 - Quantum Information Conic Solver
                by K. He, J. Saunderson, H. Fawzi (2024)
    ====================================================================
    Problem summary:
            no. cones:  1                        no. vars:    52
            barr. par:  8                        no. constr:  3
            symmetric:  False                    cone dim:    52
            complex:    True

    ...

    Solution summary
            sol. status:  optimal                num. iter:    13
            exit status:  solved                 solve time:   1.338

            primal obj:   3.411085972738e+00     primal feas:  1.06e-10
            dual obj:     3.411085973116e+00     dual feas:    5.38e-11
            opt. gap:     1.22e-11

Nearest correlation matrix
---------------------------

Correlation matrices are characterized by being a real positive 
semidefinite matrices with diagonal entries all equal to one.
Therefore, the closest correlation matrix to a given matrix 
:math:`C\in\mathbb{S}^n`, can be found by solving the following
problem

.. math::

    \min_{Y \in \mathbb{S}^n} &&& S( C \| Y )

    \text{s.t.} &&& Y_{ii} = 1 \qquad i=1,\ldots,n

    &&& Y \succeq 0.

To write this in the form accepted by **QICS**, we will represent
the problem in standard form

.. math::

    \min_{t \in\mathbb{R}, \ X,Y \in \mathbb{S}^n} &&& t

    \text{s.t.} \quad\; &&& X = C
    
    &&& Y_{ii} = 1 \qquad i=1,\ldots,n

    &&& (t, X, Y) \in \mathcal{K}_{\text{qre}}^n.

.. code-block:: python

    import numpy as np
    import qics
    import qics.vectorize as vec

    np.random.seed(1)

    n = 5

    # Generate random matrix C
    C = np.random.randn(n, n)
    C = C @ C.T

    # Define objective function
    ct = np.array(([[1.]]))
    cX = np.zeros((n*n, 1))
    cY = np.zeros((n*n, 1))
    c  = np.vstack((ct, cX, cY))

    # Build linear constraints
    # X = C
    sn = vec.vec_dim(n, compact=True)
    A1 = np.hstack((np.zeros((sn, 1)), vec.eye(n), np.zeros((sn, n*n))))
    b1 = vec.mat_to_vec(C, compact=True)
    # Yii = 1
    A2 = np.zeros((n, 1 + 2*n*n))
    A2[range(n), range(1 + n*n, 1 + 2*n*n, n+1)] = 1.
    b2 = np.ones((n, 1))

    A = np.vstack((A1, A2))
    b = np.vstack((b1, b2))

    # Define cones to optimize over
    cones = [qics.cones.QuantRelEntr(n)]

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
            no. cones:  1                        no. vars:    51
            barr. par:  12                       no. constr:  20
            symmetric:  False                    cone dim:    51
            complex:    False

    ...

    Solution summary
            sol. status:  optimal                num. iter:    12
            exit status:  solved                 solve time:   4.268

            primal obj:   5.450544797212e+01     primal feas:  7.76e-09
            dual obj:     5.450544802133e+01     dual feas:    3.88e-09
            opt. gap:     9.03e-10


Relative entropy of entanglement
----------------------------------

Entanglement is an important resource in quantum information
theory, and therefore it is often useful to characterize the
amount of entanglement possessed by a quantum state. This can
be characterized by the distance (in the quantum relative 
entropy sense) between a given bipartite state and the set of
separable states. 

In general, the set of separable states is NP-hard to describe.
Therefore, it is common to estimate the set of separable states 
using the positive partial transpose (PPT) criteria, i.e., if a 
quantum state :math:`X \in \mathbb{H}^{n_1n_2}` is separable, then
it must be a member of

.. math::

    \mathsf{PPT} = \{ X \in \mathbb{H}^{n_1n_2} : T_2(X) \succeq 0 \},

where :math:`T_1:\mathbb{S}^{n_1n_2}\rightarrow\mathbb{S}^{n_1n_2}`
denotes the partial transpose operator with respect to the second
subsystem. Note that in general, the PPT crieria is not a sufficient 
condition for separability, i.e., there exists entangled quantum 
states which also satisfy the PPT criteria. However, it is a sufficient
condition when :math:`n_0=n_1=2`, or :math:`n_0=2, n_1=3`.

Given this, the relative entropy of entagnlement of a quantum state 
:math:`C \in \mathbb{H}^{n_1n_2}` is given by

.. math::

    \min_{Y \in \mathbb{H}^{n_1n_2}} &&& S( C \| Y )

    \text{s.t.} &&& \text{tr}[Y] = 1
    
    &&& T_2(Y) \succeq 0 

    &&& Y \succeq 0.

We can model this in the standard form accepted by **QICS** as

.. math::

    \min_{t \in\mathbb{R}, \ X,Y,Z \in \mathbb{H}^{n_1n_2}} &&& t

    \text{s.t.} \quad\quad &&& X = C

    &&& \text{tr}[Y] = 1
    
    &&& T_2(Y) - Z = 0

    &&& (t, X, Y, Z) \in \mathcal{K}_{\text{qre}}^{n_1n_2} \times \mathbb{H}^{n_1n_2}_+.

.. code-block:: python

    import numpy as np
    import qics
    import qics.vectorize as vec
    import qics.quantum as qu

    np.random.seed(1)

    n1 = 2
    n2 = 3
    N  = n1 * n2

    # Generate random (complex) quantum state
    C = qu.random.density_matrix(N, iscomplex=True)

    # Define objective function
    ct = np.array(([[1.]]))
    cX = np.zeros((2*N*N, 1))
    cY = np.zeros((2*N*N, 1))
    cZ = np.zeros((2*N*N, 1))
    c  = np.vstack((ct, cX, cY, cZ))

    # Build linear constraints
    # X = C
    sN = vec.vec_dim(N, iscomplex=True, compact=True)
    A1 = np.hstack((
        np.zeros((sN, 1)),
        vec.eye(N, iscomplex=True), 
        np.zeros((sN, 2*N*N)),
        np.zeros((sN, 2*N*N)),
    ))
    b1 = vec.mat_to_vec(C, compact=True)
    # tr[Y] = 1
    A2 = np.hstack((
        np.zeros((1, 1)), 
        np.zeros((1, 2*N*N)), 
        vec.mat_to_vec(np.eye(N, dtype=np.complex128)).T, 
        np.zeros((1, 2*N*N))
    ))
    b2 = np.array([[1.]])
    # T2(Y) = Z
    p_transpose = vec.lin_to_mat(
        lambda X : qu.p_transpose(X, (n1, n2), 1), 
        (N, N), iscomplex=True
    )
    A3 = np.hstack((
        np.zeros((1, 1)), 
        np.zeros((1, 2*N*N)),
        p_transpose, 
        -vec.eye(N, iscomplex=True)
    ))
    b3 = np.zeros((sN, 1))

    A = np.vstack((A1, A2, A3))
    b = np.vstack((b1, b2, b3))

    # Input into model and solve
    cones = [
        qics.cones.QuantRelEntr(N, iscomplex=True), 
        qics.cones.PosSemidefinite(N, iscomplex=True)
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
            no. cones:  2                        no. vars:    217
            barr. par:  20                       no. constr:  73
            symmetric:  False                    cone dim:    217
            complex:    True

    ...

    Solution summary
            sol. status:  optimal                num. iter:    10
            exit status:  solved                 solve time:   5.030

            primal obj:   4.838694958245e-03     primal feas:  2.07e-09
            dual obj:     4.838693850761e-03     dual feas:    1.03e-09
            opt. gap:     1.11e-09