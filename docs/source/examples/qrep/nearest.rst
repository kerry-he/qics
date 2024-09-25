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
is defined as

.. math::

    D_\varphi( x \| y ) = \varphi(x) - \varphi(y) - \langle \nabla\varphi(y), x - y \rangle .  

A Bregman projection of a point :math:`\mathcal{y}` onto a set 
:math:`\mathcal{C}\subset\text{dom}\ \varphi` is given by

.. math::

    \min_{x \in \mathcal{C}} \quad D_\varphi( x \| y ).

When :math:`\varphi(X)=-S(X)` is the negative quantum entropy, the Bregman divergence is
the (normalized) quantum relative entropy :math:`D_\varphi( X \| Y ) = S( X \| Y ) - \text{tr}[X - Y]`.

The Bregman projection for this kernel of a matrix :math:`Y\in\mathbb{H}^n` onto the
set of density matrices if given by

.. math::

    \min_{X \in \mathbb{H}^n} &&& S( X \| Y ) - \text{tr}[X - Y]

    \text{s.t.} &&& \text{tr}[X] = 1

    &&& X \succeq 0,

As the second argument of the quantum relative entropy is fixed, we can model the 
problem using just the quantum entropy cone.

.. tabs::

    .. code-tab:: python Native

        import numpy
        import scipy
        import qics

        numpy.random.seed(1)

        n = 5

        # Generate random matrix Y to project
        Y = numpy.random.randn(n, n) + numpy.random.randn(n, n)*1j
        Y = Y @ Y.conj().T
        tr_Y = numpy.trace(Y).real

        # Define objective function
        ct = numpy.array([[1.]])
        cu = numpy.array([[0.]])
        cX = -scipy.linalg.logm(Y) - numpy.eye(n)
        c  = numpy.vstack((ct, cu, qics.vectorize.mat_to_vec(cX)))

        # Build linear constraints
        # u = 1
        A1 = numpy.hstack((numpy.array([[0., 1.]]), numpy.zeros((1, 2*n*n))))
        b1 = numpy.array([[1.]])
        # tr[X] = 1
        A2 = numpy.hstack((
            numpy.array([[0., 0.]]), 
            qics.vectorize.mat_to_vec(numpy.eye(n, dtype=numpy.complex128)).T
        ))
        b2 = numpy.array([[1.]])

        A = numpy.vstack((A1, A2))
        b = numpy.vstack((b1, b2))

        # Define cones to optimize over
        cones = [qics.cones.QuantEntr(n, iscomplex=True)]

        # Initialize model and solver objects
        model = qics.Model(c=c, A=A, b=b, cones=cones, offset=tr_Y)
        solver = qics.Solver(model)

        # Solve problem
        info = solver.solve()

    .. code-tab:: python PICOS

        import numpy
        import scipy
        import picos

        numpy.random.seed(1)

        n = 5

        # Generate random matrix Y to project
        Y = numpy.random.randn(n, n) + numpy.random.randn(n, n)*1j
        Y = Y @ Y.conj().T
        trY = numpy.trace(Y).real
        logY = scipy.linalg.logm(Y)

        # Define problem
        P = picos.Problem()
        X = picos.HermitianVariable("X", n)

        P.set_objective("min", -picos.quantentr(X) - (X | logY + picos.I(n)).real + trY)
        P.add_constraint(picos.trace(X) == 1)

        # Solve problem
        P.solve(solver="qics", verbosity=1)        

.. _Nearest:

Nearest correlation matrix
---------------------------

Correlation matrices are characterized by being a real positive semidefinite matrices 
with diagonal entries all equal to one. Therefore, the closest correlation matrix to a 
given matrix  :math:`C\in\mathbb{S}^n`, can be found by solving the following problem

.. math::

    \min_{Y \in \mathbb{S}^n} &&& S( C \| Y )

    \text{s.t.} &&& Y_{ii} = 1 \qquad i=1,\ldots,n

    &&& Y \succeq 0.

.. tabs::

    .. code-tab:: python Native

        import numpy
        import qics

        numpy.random.seed(1)

        n = 5

        # Generate random matrix C
        C = numpy.random.randn(n, n)
        C = C @ C.T

        # Define objective function
        ct = numpy.array(([[1.]]))
        cX = numpy.zeros((n*n, 1))
        cY = numpy.zeros((n*n, 1))
        c  = numpy.vstack((ct, cX, cY))

        # Build linear constraints
        # X = C
        sn = qics.vectorize.vec_dim(n, compact=True)
        A1 = numpy.hstack((numpy.zeros((sn, 1)), qics.vectorize.eye(n), numpy.zeros((sn, n*n))))
        b1 = qics.vectorize.mat_to_vec(C, compact=True)
        # Yii = 1
        A2 = numpy.zeros((n, 1 + 2*n*n))
        A2[range(n), range(1 + n*n, 1 + 2*n*n, n+1)] = 1.
        b2 = numpy.ones((n, 1))

        A = numpy.vstack((A1, A2))
        b = numpy.vstack((b1, b2))

        # Define cones to optimize over
        cones = [qics.cones.QuantRelEntr(n)]

        # Initialize model and solver objects
        model = qics.Model(c=c, A=A, b=b, cones=cones)
        solver = qics.Solver(model)

        # Solve problem
        info = solver.solve()

    .. code-tab:: python PICOS

        import numpy
        import picos

        numpy.random.seed(1)

        n = 5

        # Generate random matrix C
        C = numpy.random.randn(n, n)
        C = C @ C.T

        # Define problem
        P = picos.Problem()
        Y = picos.SymmetricVariable("Y", n)

        P.set_objective("min", picos.quantrelentr(C, Y))
        P.add_constraint(picos.maindiag(Y) == 1)

        # Solve problem
        P.solve(solver="qics", verbosity=1)

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

where :math:`T_2:\mathbb{S}^{n_1n_2}\rightarrow\mathbb{S}^{n_1n_2}`
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

.. tabs::

    .. code-tab:: python Native

        import numpy
        import qics

        numpy.random.seed(1)

        n1 = 2
        n2 = 3
        N  = n1 * n2

        # Generate random (complex) quantum state
        C = qics.quantum.random.density_matrix(N, iscomplex=True)

        # Define objective function
        ct = numpy.array(([[1.]]))
        cX = numpy.zeros((2*N*N, 1))
        cY = numpy.zeros((2*N*N, 1))
        cZ = numpy.zeros((2*N*N, 1))
        c  = numpy.vstack((ct, cX, cY, cZ))

        # Build linear constraints
        # X = C
        sN = qics.vectorize.vec_dim(N, iscomplex=True, compact=True)
        A1 = numpy.hstack((
            numpy.zeros((sN, 1)),
            qics.vectorize.eye(N, iscomplex=True),
            numpy.zeros((sN, 2*N*N)),
            numpy.zeros((sN, 2*N*N)),
        ))
        b1 = qics.vectorize.mat_to_vec(C, compact=True)
        # tr[Y] = 1
        A2 = numpy.hstack((
            numpy.zeros((1, 1)),
            numpy.zeros((1, 2*N*N)),
            qics.vectorize.mat_to_vec(numpy.eye(N, dtype=numpy.complex128)).T,
            numpy.zeros((1, 2*N*N))
        ))
        b2 = numpy.array([[1.]])
        # T2(Y) = Z
        p_transpose = qics.vectorize.lin_to_mat(
            lambda X : qics.quantum.partial_transpose(X, (n1, n2), 1),
            (N, N), iscomplex=True
        )
        A3 = numpy.hstack((
            numpy.zeros((sN, 1)),
            numpy.zeros((sN, 2*N*N)),
            p_transpose,
            -qics.vectorize.eye(N, iscomplex=True)
        ))
        b3 = numpy.zeros((sN, 1))

        A = numpy.vstack((A1, A2, A3))
        b = numpy.vstack((b1, b2, b3))

        # Input into model and solve
        cones = [
            qics.cones.QuantRelEntr(N, iscomplex=True),
            qics.cones.PosSemidefinite(N, iscomplex=True)
        ]

        # Initialize model and solver objects
        model = qics.Model(c=c, A=A, b=b, cones=cones)
        solver = qics.Solver(model)

        # Solve problem
        info = solver.solve()

    .. code-tab:: python PICOS

        import numpy
        import picos
        import qics

        numpy.random.seed(1)

        n1 = 2
        n2 = 3
        N  = n1 * n2

        # Generate random (complex) quantum state
        C = qics.quantum.random.density_matrix(N, iscomplex=True)

        # Define problem
        P = picos.Problem()
        Y = picos.HermitianVariable("Y", N)
        
        P.set_objective("min", picos.quantrelentr(C, Y))
        P.add_constraint(picos.trace(Y) == 1.0)
        P.add_constraint(picos.partial_transpose(Y, subsystems=1, dimensions=(n1, n2)) >> 0)

        # Solve problem
        P.solve(solver="qics", verbosity=1)