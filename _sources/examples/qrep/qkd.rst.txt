Quantum key distribution
==========================

Quantum key distribution is an important application in quantum cryptography, in
which a private key is securely generated and communicated between two parties
using a quantum protocol. The quantum key rate is a quantity which characterizes
the security of a given quantum protocol.

The quantum key rate can be computed using using the following
quantum relative entropy program from :ref:`[1,2] <qkd_refs>`

.. math::

    \min_{X\in\mathbb{H}^n} &&& S(\mathcal{G}(X) \| \mathcal{Z}(\mathcal{G}(X)))

    \text{s.t.} &&& \langle A_i, X \rangle = b_i, \quad \forall i,\ldots,p

    &&& X \succeq 0,

where :math:`\mathcal{G}:\mathbb{H}^n\rightarrow\mathbb{H}^{mr}` is a positive
linear map related to the quantum protocol, and is usually described using 
Kraus operators :math:`K_i\in\mathbb{C}^{mr\times n}`

.. math::

    \mathcal{G}(X) = \sum_{i=1}^l K_i X K_i^\dagger,

and :math:`\mathcal{Z}:\mathbb{H}^{mr}\rightarrow\mathbb{H}^{mr}` is
the pinching map which zeros off-diagonal blocks of a block matrix

.. math::

    \mathcal{Z}(X) = \sum_{i=1}^r (| i \rangle \langle i | \otimes \mathbb{I}_m)
    X (| i \rangle \langle i | \otimes \mathbb{I}_m).

**QICS** provides the cone :class:`qics.cones.QuantKeyDist` to represent this
slice of the quantum relative entropy.

.. note::

    We provide several ways to define the linear maps :math:`\mathcal{G}` and 
    :math:`\mathcal{Z}` when initializing a :class:`qics.cones.QuantKeyDist`.
    See the API documentation for further details.


Entanglement based BB84
--------------------------------

As a concrete example, we consider the entanglement based BB84 protocol
described in :ref:`[3] <qkd_refs>` where only the Z basis is used to 
measure the key. In this case, we have :math:`\mathcal{G}(X) = X`, dimensions 
:math:`n=m=r=2`, and we have the three linear constraints defined by

.. math::

    A_1 &= \mathbb{I}\\ \\
    A_2 &= \frac{1}{4} \left(\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} 
    \otimes \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix} 
    + \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix} \otimes 
    \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}  \right) \\ \\
    A_3 &= \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix} \otimes 
    \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix} 
    + \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix} \otimes 
    \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}  ,

and :math:`b = (1, q_x, q_z)`. The closed form solution for this quantum key
rate is

.. math::

    \log(2) + q_x \log(q_x) + (1 - q_x) \log(1 - q_x)

which we use to confirm that **QICS** gives the correct solution.

.. tabs::

    .. code-tab:: python Native

        import numpy as np

        import qics
        from qics.vectorize import mat_to_vec

        qx = 0.25
        qz = 0.75

        X0 = np.array([[0.5, 0.5], [0.5, 0.5]])
        X1 = np.array([[0.5, -0.5], [-0.5, 0.5]])
        Z0 = np.array([[1.0, 0.0], [0.0, 0.0]])
        Z1 = np.array([[0.0, 0.0], [0.0, 1.0]])

        # Model problem using primal variables (t, X)
        # Define objective function
        c = np.vstack((np.array([[1.0]]), np.zeros((16, 1))))

        # Build linear constraints <Ai, X> = bi for all i
        Ax = np.kron(X0, X1) + np.kron(X1, X0)
        Az = np.kron(Z0, Z1) + np.kron(Z1, Z0)
        A_mats = [np.eye(4), Ax, Az]

        A = np.block([[0., mat_to_vec(Ak).T] for Ak in A_mats])
        b = np.array([[1.0], [qx], [qz]])

        # Input into model and solve
        cones = [qics.cones.QuantKeyDist(4, 2)]

        # Initialize model and solver objects
        model = qics.Model(c=c, A=A, b=b, cones=cones)
        solver = qics.Solver(model)

        # Solve problem
        info = solver.solve()

        analytic_rate = np.log(2) + (qx*np.log(qx) + qz*np.log(qz))
        numerical_rate = info["opt_val"]

        print("Analytic key rate: ", analytic_rate)
        print("Numerical key rate:", numerical_rate)

    .. code-tab:: python PICOS

        import numpy as np

        import picos

        qx = 0.25
        qz = 0.75

        X0 = np.array([[.5,  .5], [ .5, .5]])
        X1 = np.array([[.5, -.5], [-.5, .5]])
        Z0 = np.array([[1.,  0.], [ 0., 0.]])
        Z1 = np.array([[0.,  0.], [ 0., 1.]])

        Ax = np.kron(X0, X1) + np.kron(X1, X0)
        Az = np.kron(Z0, Z1) + np.kron(Z1, Z0)

        # Define problem
        P = picos.Problem()
        X = picos.SymmetricVariable("X", 4) 
        
        P.set_objective("min", picos.quantkeydist(X))
        P.add_constraint(picos.trace(X) == 1)
        P.add_constraint((X | Ax) == qx)
        P.add_constraint((X | Az) == qz)        

        # Solve problem
        P.solve(solver="qics")

        analytic_rate = np.log(2) + (qx*np.log(qx) + qz*np.log(qz))
        numerical_rate = P.value

        print("Analytic key rate: ", analytic_rate)
        print("Numerical key rate:", numerical_rate)

.. code-block:: none

    Analytic key rate:  0.130812035941137
    Numerical key rate: 0.1308120333864307

Reading protocols from files
--------------------------------

It is also fairly straightforward to solve quantum key rates from
``.mat`` files from, e.g., `here <https://www.math.uwaterloo.ca/~hwolkowi/henry/reports/ZGNQKDmainsolverUSEDforPUBLCNJuly31/>`__ or 
`here <https://github.com/kerry-he/qrep-structure/tree/main/data>`__.
We supply some sample code for how to do this below.

.. code-block:: python
    :caption: read_qkd_file.py

    import numpy as np
    import scipy as sp

    import qics

    # Read file
    data   = sp.io.loadmat('filename.mat')
    gamma  = data['gamma']
    Gamma  = list(data['Gamma'].ravel())
    K_list = list(data['Klist'].ravel())
    Z_list = list(data['Zlist'].ravel())

    iscomplex = np.iscomplexobj(Gamma) or np.iscomplexobj(K_list)
    dtype = np.complex128 if iscomplex else np.float64

    no, ni = np.shape(K_list[0])
    nc     = np.size(gamma)
    vni    = qics.vectorize.vec_dim(ni, iscomplex=iscomplex)

    # Define objective function
    c = np.vstack((np.array([[1.]]), np.zeros((vni, 1))))

    # Build linear constraints
    A = np.zeros((nc, 1 + vni))
    for i in range(nc):
        A[i, 1:] = qics.vectorize.mat_to_vec(Gamma[i].astype(dtype)).ravel()
    b = gamma

    # Input into model and solve
    cones = [qics.cones.QuantKeyDist(K_list, Z_list, iscomplex=iscomplex)]

    # Initialize model and solver objects
    model = qics.Model(c=c, A=A, b=b, cones=cones)
    solver = qics.Solver(model)

    # Solve problem
    info = solver.solve()


.. _qkd_refs:

References
----------

    1. “Reliable numerical key rates for quantum key distribution”, 
       A. Winick, N. L ̈utkenhaus, and P. J. Coles.
       *Quantum*, vol. 2, p. 77, 2018.

    2. “Numerical approach for unstructured quantum key distribution”,
       P. J. Coles, E. M. Metodiev, and N. L ̈utkenhaus.
       *Nature Communications*, vol. 7, no. 1, p. 11712, 2016

    3. "Quantum key distribution rates from non-symmetric conic optimization",
       L. A. González, et al. *arXiv preprint* arXiv:2407.00152, 2024.

