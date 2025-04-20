Advanced tips
=============

How efficiently QICS can solve a given problem can sometimes depend quite significantly
on how the problem is formulated and input into the solver. Here, we outline some best
practices to solve problems in QICS as efficiently as possible.


Modelling quantum relative entropy programs
-------------------------------------------

It is well known that some cones can be used to represent other cones. For example, the
positive semidefinite cone generalizes the nonnegative orthant in the following way

.. math::

    \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n_+
    \qquad \Longleftrightarrow \qquad 
    \begin{bmatrix} x_1 & & & \\ & x_2 & & \\ & & \ddots \\ & & & x_n \end{bmatrix} 
    \in \mathbb{H}^n_+,

which allows us to solve linear programs as semidefinite programs. Similarly, we can 
model the hypograph of quantum entropy using the epigraph of quantum relative entropy as

.. math::

    t \leq S(X) \qquad \Longleftrightarrow \qquad -t \geq S(X \| \,\mathbb{I}\,),

which means we can model the quantum entropy cone using the quantum relative entropy
cone. 

However, although some cones may have stronger generalizability, it is usually better to
model problems using the simplest appropriate cones as possible, i.e., use the 
nonnegative orthant to model linear programs instead of the semidefeinite cone, and use
the quantum entropy cone to model the quantum entropy hypograph instead of the quantum
relative entropy cone. The main reason for this that we are able to implement more 
efficient cone oracles (i.e., gradient, Hessian product, inverse Hessian product, and 
third order derivative oracles) for simpler cones. For example, QICS' inverse Hessian 
product oracle for the quantum relative entropy cone has a complexity of :math:`O(n^6)` 
flops, whereas QICS' inverse Hessian product oracle for the quantum entropy cone has a 
complexity of :math:`O(n^3)`. 

In QICS, we provide efficient implementations of certain slices of the quantum relative
entropy cone which commonly arise in practice. Although it is possible to model these
sets using the quantum relative entropy cone, it is recommended to use these specialized
cones instead, which implement significantly more efficient cone oracles.

.. list-table:: **Slices of the quantum relative entropy cone**
   :header-rows: 1
   :align: center

   * - Function
     - Slice of quantum relative entropy
     - Simplified expression
     - QICS cone
   * - Classical entropy
     - :math:`-S(\text{diag}(x)\|\,\mathbb{I}\,)`
     - :math:`H(x)`
     - :class:`qics.cones.ClassEntr`
   * - Classical relative entropy
     - :math:`S(\text{diag}(x)\|\text{diag}(y))`
     - :math:`H(x\|y)`
     - :class:`qics.cones.ClassRelEntr`
   * - Quantum entropy
     - :math:`-S(X \| \,\mathbb{I}\,)`
     - :math:`S(X)`
     - :class:`qics.cones.QuantEntr`
   * - Quantum conditional entropy
     - :math:`-S(X \| \mathbb{I} \otimes \text{tr}_1(X))`
     - :math:`S(X) - S(\text{tr}_1(X))`
     - :class:`qics.cones.QuantCondEntr`
   * - Quantum key distribution
     - :math:`S(\mathcal{G}(X) \| \mathcal{Z}(\mathcal{G}(X)))`
     - :math:`-S(\mathcal{G}(X))+S(\mathcal{Z}(\mathcal{G}(X)))`
     - :class:`qics.cones.QuantKeyDist`

To showcase the impact that modelling a problem using the "correct" cone can make, we
solve the :ref:`examples/qrep/nearest:bregman projection` problem using the quantum
entropy cone

.. code-block:: python

    import numpy as np
    import scipy as sp

    import qics
    from qics.vectorize import lin_to_mat, vec_dim, mat_to_vec

    np.random.seed(1)

    n = 50
    vn = vec_dim(n)

    # Generate random positive semidefinite matrix Y to project
    Y = np.random.randn(n, n)
    Y = Y @ Y.T
    trY = np.trace(Y).real

    # Model problem using primal variables (t, u, X)
    # Define objective function
    c = np.block([[1.0], [0.0], [mat_to_vec(-sp.linalg.logm(Y) - np.eye(n))]])

    # Build linear constraints
    trace = lin_to_mat(lambda X: np.trace(X), (n, 1))

    A = np.block([
        [0.0, 1.0, np.zeros((1, vn))],  # u = 1
        [0.0, 0.0, trace            ]   # tr[X] = 1
    ])

    b = np.array([[1.0], [1.0]])

    # Define cones to optimize over
    cones = [qics.cones.QuantEntr(n)]

    # Initialize model and solver objects
    model = qics.Model(c=c, A=A, b=b, cones=cones, offset=trY)
    solver = qics.Solver(model)

    # Solve problem
    info = solver.solve()

.. code-block:: none

    Solution summary
            sol. status:  optimal                   num. iter:    29
            exit status:  solved                    solve time:   0.514
            primal obj:   2.513696087465e+03        primal feas:  2.90e-09
            dual obj:     2.513696087470e+03        dual feas:    1.42e-09

Now, we solve the same problem using the quantum relative entropy cone.

.. code-block:: python

    import numpy as np

    import qics
    from qics.vectorize import eye, lin_to_mat, vec_dim, mat_to_vec

    np.random.seed(1)

    n = 50
    vn = vec_dim(n)
    cn = vec_dim(n, compact=True)

    # Generate random positive semidefinite matrix Y to project
    Y = np.random.randn(n, n)
    Y = Y @ Y.T

    # Model problem using primal variables (t, X, Y)
    # Define objective function
    c = np.block([[1.0], [-mat_to_vec(np.eye(n))], [mat_to_vec(np.eye(n))]])

    # Build linear constraints
    trace = lin_to_mat(lambda X: np.trace(X), (n, 1))

    A = np.block([
        [np.zeros((cn, 1)), np.zeros((cn, vn)), eye(n)           ],  # Y = Y
        [0.0,               trace,              np.zeros((1, vn))]   # tr[X] = 1
    ])

    b = np.block([[mat_to_vec(Y, compact=True)], [1.0]])

    # Define cones to optimize over
    cones = [qics.cones.QuantRelEntr(n)]

    # Initialize model and solver objects
    model = qics.Model(c=c, A=A, b=b, cones=cones)
    solver = qics.Solver(model)

    # Solve problem
    info = solver.solve()

.. code-block:: none

    Solution summary
            sol. status:  optimal                   num. iter:    28
            exit status:  solved                    solve time:   13.701
            primal obj:   2.513696077734e+03        primal feas:  3.80e-09
            dual obj:     2.513696077837e+03        dual feas:    3.44e-09

We see that solving the same problem using the quantum relative entropy cone takes
over **25** times longer! This discrepency only gets more significant the larger the
problem dimensions we solve. We refer to :ref:`[1] <advanced_refs>` for further 
numerical results showcasing the computational savings we can obtain by using the
special slices of the quantum relative entropy cone which QICS supports.


Modelling linear constraints
----------------------------

When modelling a conic problem, a decision we often need to make is whether to model
constraints in kernel form, i.e., our primal variable :math:`x\in\mathcal{K}` has to
satisfy :math:`Ax=b`, or in image form, i.e., our primal variable :math:`x` has to 
satisfy :math:`h-Gx\in\mathcal{K}`.

QICS accepts both forms of constraints. However, how we choose to model our linear
constraints can drastically impact how quickly QICS can solve a problem. To understand
this, we recognize that to solve the Newton system required in every iteration of the
interior-point algorithm,

- if :math:`G=-\mathbb{I}`, we need to build and Cholesky factor the matrix 
  :math:`AH^{-1}A^\top`,

- otherwise, we need to build and Cholesky factor the matrix :math:`G^\top HG`,

where :math:`H` is the Hessian corresponding to the barrier function of 
:math:`\mathcal{K}`. Therefore, we want to model our problem so that :math:`A` or 
:math:`G` are as small as possible to make this step as efficient as possible.

Example 1
~~~~~~~~~

To illustrate this with an example, first consider an instance of the 
:ref:`examples/qrep/nearest:nearest correlation matrix` problem, except we further 
constrain the matrix :math:`Y` to be tridiagonal. This is a highly constrained problem,
as the only free variables are the :math:`1`-diagonal of the matrix :math:`Y`, and so we
expect that it will be better to model our problem in image form.

To demonstrate this, we first model the problem by pushing all constraints into the 
:math:`A` matrix as follows

.. math::

    \min_{(t,X,Y) \in \mathcal{QRE}_n} \quad t \quad \text{subj. to} \quad X = C \quad
    \text{and} \quad Y = \begin{bmatrix} 1 & \cdot & 0 & 0 \\ \cdot & 1 & \cdot & 0 \\ 
    0 & \cdot & 1 & \cdot \\ 0 & 0 & \cdot & 1 \end{bmatrix},

where we use :math:`\cdot` to denote that the corresponding entry of :math:`Y` is 
unconstrained. The size of the matrix :math:`A` used to model this problem has a 
dimension of :math:`(n-1)^2\times (1+2n^2)`. We can model and solve this in QICS as
follows.
 
.. code-block:: python

    import numpy as np

    import qics
    from qics.vectorize import vec_dim, mat_to_vec, eye

    np.random.seed(1)

    n = 50

    vn = vec_dim(n)
    cn = vec_dim(n, compact=True)

    # Generate random positive semidefinite matrix C
    C = np.random.randn(n, n)
    C = C @ C.T / n
    C_cvec = mat_to_vec(C, compact=True)

    # Model problem using primal variables (t, X, Y)
    # Define objective function
    c = np.block([[1.0], [np.zeros((vn, 1))], [np.zeros((vn, 1))]])

    # Build linear constraints
    diag = np.zeros((n, vn))
    diag[np.arange(n), np.arange(0, vn, n + 1)] = 1.

    m = cn - 2 * n + 1
    off_tridiag = np.zeros((m, n, n))
    t = 0
    for j in range(n):
        for i in range(j - 1):
            off_tridiag[t, i, j] = off_tridiag[t, j, i] = 1.0
            t += 1
    off_tridiag = off_tridiag.reshape(-1, vn)

    A = np.block([
        [np.zeros((cn, 1)), eye(n),            np.zeros((cn, vn))],  # X = C
        [np.zeros((n, 1)),  np.zeros((n, vn)), diag              ],  # Yii = 1
        [np.zeros((m, 1)),  np.zeros((m, vn)), off_tridiag       ],  # Yij = 0 for off-tridiagonal ij
    ])

    b = np.block([[C_cvec], [np.ones((n, 1))], [np.zeros((m, 1))]])

    # Define cones to optimize over
    cones = [qics.cones.QuantRelEntr(n)]

    # Initialize model and solver objects
    model = qics.Model(c=c, A=A, b=b, cones=cones)
    solver = qics.Solver(model)

    # Solve problem
    info = solver.solve()

.. code-block:: none

    Solution summary
            sol. status:  optimal                   num. iter:    16
            exit status:  solved                    solve time:   14.507
            primal obj:   2.598505231622e+01        primal feas:  5.17e-09
            dual obj:     2.598505231892e+01        dual feas:    2.59e-09

Alternatively, we can push all linear constraints into the :math:`G` matrix as follows

.. math::

    \min_{t,z} \quad t \quad \text{subj. to} \quad \left(t, C, 
    \begin{bmatrix} 1 & z_1 & 0 & 0 \\ z_1 & 1 & z_2 & 0 \\ 0 & z_2 & 1 & z_3 
    \\ 0 & 0 & z_3 & 1 \end{bmatrix}\right)\in\mathcal{QRE}_n,

The size of the matrix :math:`G` has a dimension of :math:`(1+2n^2)\times n`, which has
significantly fewer columns than the number of rows :math:`A` had in the previous 
formulation of the problem. We can model and solve this new formulation in QICS as
follows.

.. code-block:: python

    import numpy as np

    import qics
    from qics.vectorize import vec_dim, mat_to_vec

    np.random.seed(1)

    n = 50

    vn = vec_dim(n)
    cn = vec_dim(n, compact=True)

    # Generate random positive semidefinite matrix C
    C = np.random.randn(n, n)
    C = C @ C.T / n
    C_vec = mat_to_vec(C)

    # Model problem using primal variables (t, z)
    # Define objective function
    c = np.block([[1.0], [np.zeros((n - 1, 1))]])

    # Build linear constraints
    tridiag = np.zeros((n - 1, n, n))
    for i in range(n - 1):
        tridiag[i, i, i + 1] = tridiag[i, i + 1, i] = 1.0
    tridiag = tridiag.reshape(-1, vn).T

    G = np.block([
        [-1.0,              np.zeros((1, n - 1)) ],  # t_qre = t
        [np.zeros((vn, 1)), np.zeros((vn, n - 1))],  # X_qre = C
        [np.zeros((vn, 1)), -tridiag             ]   # Y_qre = tridiag(z)
    ])  # fmt: skip

    h = np.block([[0.0], [C_vec], [mat_to_vec(np.eye(n))]])

    # Define cones to optimize over
    cones = [qics.cones.QuantRelEntr(n)]

    # Initialize model and solver objects
    model = qics.Model(c=c, G=G, h=h, cones=cones)
    solver = qics.Solver(model)

    # Solve problem
    info = solver.solve()

.. code-block:: none

    Solution summary
            sol. status:  optimal                   num. iter:    16
            exit status:  solved                    solve time:   1.145
            primal obj:   2.598505247414e+01        primal feas:  2.95e-09
            dual obj:     2.598505247569e+01        dual feas:    1.41e-09

Overall, we have reduced the solve time by over a factor of **10** just by modelling our
problem differently!

Example 2
~~~~~~~~~

As another example which illustrates when it can be more beneficial to model the problem
by pushing constraints into :math:`A`, consider the 
:ref:`examples/qrep/channel:entanglement-assisted rate-distortion` problem. To handle
the inequality constraint, we present two potential options.

First, we can introduce a scalar dummy variable :math:`d` and model the problem as 
follows

.. math::

    \min_{d,t,\rho_{AB}} &&& t + S(\sigma_A)

    \text{subj. to} &&& \text{tr}_B(\rho_{AB}) = \sigma_A

    &&& d = D - 1 + \langle \psi | \rho_{AB} | \psi \rangle

    &&& (t, \rho_{AB}) \in \mathcal{QCE}_{\{n, n\}, A}
    
    &&& d \geq 0.

In this formulation, :math:`G=-\mathbb{I}`, and we can solve this in QICS as follows.

.. code-block:: python

    import numpy as np

    import qics
    from qics.quantum import p_tr, purify, entropy
    from qics.quantum.random import density_matrix
    from qics.vectorize import lin_to_mat, vec_dim, mat_to_vec

    np.random.seed(1)

    n = 8
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

.. code-block:: none

    Solution summary
            sol. status:  optimal                   num. iter:    33
            exit status:  solved                    solve time:   1.320
            primal obj:   1.621013114292e+00        primal feas:  1.22e-09
            dual obj:     1.621013117881e+00        dual feas:    6.12e-10

Alternatively, we can avoid introducing any dummy variables by directly encoding
the inequality constraint in the matrix :math:`G`, i.e., by directly formulating the
problem as

.. math::

    \min_{\rho_{AB} \in \mathbb{H}^{n^2}} &&& -S(\rho_{AB}) 
    + S(\text{tr}_A(\rho_{AB})) + S(\sigma_A)

    \text{subj. to} &&& \text{tr}_B(\rho_{AB}) = \sigma_A

    &&& 1 - \langle \psi | \rho_{AB} | \psi \rangle \leq D

    &&& \rho_{AB} \succeq 0.

Note that :math:`G` is no longer the negative identity matrix, but is essentially the
negative identity matrix with an extra row. We can formulate this in QICS as follows.

.. code-block:: python

    import numpy as np

    import qics
    from qics.quantum import p_tr, purify, entropy
    from qics.quantum.random import density_matrix
    from qics.vectorize import lin_to_mat, vec_dim, mat_to_vec, eye

    np.random.seed(1)

    n = 8
    N = n * n
    vN = vec_dim(N)
    cN = vec_dim(N, compact=True)
    cn = vec_dim(n, compact=True)

    # Define random problem data
    rho = density_matrix(n)
    entr_rho = entropy(rho)

    D = 0.25
    Delta = np.eye(N) - purify(rho)
    Delta_cvec = mat_to_vec(Delta, compact=True)

    # Model problem using primal variables (t, X)
    # Define objective function
    c = np.block([[1.0], [np.zeros((cN, 1))]])

    # Build linear constraint matrix tr_B[X] = rho
    tr_B = lin_to_mat(lambda X : p_tr(X, (n, n), 1), (N, n), compact=(True, True))

    A = np.block([[np.zeros((cn, 1)), tr_B]])
    b = mat_to_vec(rho, compact=True)

    # Build linear cone constraints
    G = np.block([
        [-1.0,              np.zeros((1, cN))],  # t_qce = t
        [np.zeros((vN, 1)), -eye(N).T        ],  # X_qce = X
        [0.0,               Delta_cvec.T     ],  # x_nn = D - <Delta, X>
    ])

    h = np.block([[0.0], [np.zeros((vN, 1))], [D]])

    # Define cones to optimize over
    cones = [
        qics.cones.QuantCondEntr((n, n), 0),  # (t, X) ∈ QCE
        qics.cones.NonNegOrthant(1)  # d = D - <Delta, X> >= 0
    ]

    # Initialize model and solver objects
    model  = qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones, offset=entr_rho)
    solver = qics.Solver(model)

    # Solve problem
    info = solver.solve()

.. code-block:: none

    Solution summary
        sol. status:  optimal                   num. iter:    33
        exit status:  solved                    solve time:   11.320
        primal obj:   1.621013114294e+00        primal feas:  9.80e-10
        dual obj:     1.621013117578e+00        dual feas:    6.04e-10

We see that although we avoided introducing a dummy variable, the problem solves almost
**10** times slower now, as we are no longer exploiting the fact that :math:`G` is
almost the identity matrix. Another way to see this is that we now need to solve linear  
systems with :math:`G^\top HG`, which does not exploit the fact that we have an 
efficient oracle for inverse Hessian products for the quantum conditional entropy cone.


.. _advanced_refs:

References
----------

    1. K. He, J. Saunderson, and H. Fawzi, “Exploiting structure in quantum relative 
       entropy programs,” arXiv preprint arXiv:2407.00241, 2024.
