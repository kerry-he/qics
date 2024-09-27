Quantum key distribution
==========================

Quantum key distribution is an important application in
quantum cryptography, in which a private key is securely 
generated and communicated between two parties using a 
qautnum protocol. The quantum key rate is a quantity 
which characterizes the security of a given quantum protocol.

The quantum key rate can be computed using using the following
quantum relative entropy from :ref:`[1,2] <qkd_refs>`

.. math::

    \max_{X \in \mathbb{H}^n} &&& S( \mathcal{G}(X) \| \mathcal{Z}(\mathcal{G}(X)) )

    \text{s.t.} &&& \langle A_i, X \rangle = b_i, \quad \forall i,\ldots,p

    &&& X \succeq 0,

where :math:`\mathcal{G}:\mathbb{H}^n\rightarrow\mathbb{H}^{mr}` is
related to the quantum protocol, and is usually described using 
Kraus operators

.. math::

    \mathcal{G}(X) = \sum_{i=1}^l K_i X K_i^\dagger,

for :math:`K_i:\mathbb{C}^n\rightarrow\mathbb{C}^{mr}`, and 
:math:`\mathcal{Z}:\mathbb{H}^{mr}\rightarrow\mathbb{H}^{mr}` is
the pinching map which zeros off-diagonal blocks of a block matrix

.. math::

    \mathcal{Z}(X) = \sum_{i=1}^l (| i \rangle \langle i | \otimes \mathbb{I}_m) X (| i \rangle \langle i | \otimes \mathbb{I}_m).


Entanglement based BB84
--------------------------------

As a concrete example, we consider the entanglement based BB84 protocol
described in :ref:`[3] <qkd_refs>` where only the Z basis is used to 
measure the key. In this case, we have :math:`\mathcal{G}(X) = X`, :math:`l=2`,
and we have the three linear constraints defined by

.. math::

    A_1 &= \mathbb{I}\\ \\
    A_2 &= \frac{1}{4} \left(\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \otimes \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix} 
    + \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix} \otimes \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}  \right) \\ \\
    A_3 &= \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix} \otimes \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix} 
    + \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix} \otimes \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}  ,

and :math:`b = (1, q_x, q_z)`. We can solve this in **QICS** using the
:class:`~qics.cones.QuantKeyDist` cone.

.. tabs::

    .. group-tab:: Native

        .. testcode::

            import numpy
            import qics

            qx = 0.25
            qz = 0.75

            # Define objective function
            c = numpy.vstack((numpy.array([[1.]]), numpy.zeros((16, 1))))

            # Build linear constraints
            X0 = numpy.array([[.5,  .5], [ .5, .5]])
            X1 = numpy.array([[.5, -.5], [-.5, .5]])
            Z0 = numpy.array([[1.,  0.], [ 0., 0.]])
            Z1 = numpy.array([[0.,  0.], [ 0., 1.]])

            Ax = numpy.kron(X0, X1) + numpy.kron(X1, X0)
            Az = numpy.kron(Z0, Z1) + numpy.kron(Z1, Z0)

            A = numpy.vstack((
                numpy.hstack((numpy.array([[0.]]), numpy.eye(4).reshape(1, -1))),
                numpy.hstack((numpy.array([[0.]]), Ax.reshape(1, -1))),
                numpy.hstack((numpy.array([[0.]]), Az.reshape(1, -1)))
            ))

            b = numpy.array([[1., qx, qz]]).T

            # Input into model and solve
            cones = [qics.cones.QuantKeyDist(4, 2)]

            # Initialize model and solver objects
            model  = qics.Model(c=c, A=A, b=b, cones=cones)
            solver = qics.Solver(model, verbose=0)

            # Solve problem
            info = solver.solve()
            print("Optimal value is:", numpy.round(info['p_obj'], 4))
        |
        .. testoutput::

            Optimal value is: 0.1308

    .. group-tab:: PICOS

        .. testcode::

            import numpy
            import picos

            qx = 0.25
            qz = 0.75

            X0 = numpy.array([[.5,  .5], [ .5, .5]])
            X1 = numpy.array([[.5, -.5], [-.5, .5]])
            Z0 = numpy.array([[1.,  0.], [ 0., 0.]])
            Z1 = numpy.array([[0.,  0.], [ 0., 1.]])

            Ax = numpy.kron(X0, X1) + numpy.kron(X1, X0)
            Az = numpy.kron(Z0, Z1) + numpy.kron(Z1, Z0)

            # Define problem
            P = picos.Problem()
            X = picos.SymmetricVariable("X", 4) 
            
            P.set_objective("min", picos.quantkeydist(X))
            P.add_constraint(picos.trace(X) == 1)
            P.add_constraint((X | Ax) == qx)
            P.add_constraint((X | Az) == qz)        

            # Solve problem
            P.solve(solver="qics")
            print("Optimal value is:", round(P, 4))
        |
        .. testoutput::

            Optimal value is: 0.1308

The closed form solution for this quantum key rate is

.. math::

    \log(2) + q_x \log(q_x) + (1 - q_x) \log(1 - q_x)

which we use to confirm that **QICS** gives the correct solution.

>>> import numpy
>>> qx = 0.25
>>> numpy.log(2) + ( qx*numpy.log(qx) + (1-qx)*numpy.log(1-qx) )
0.130812035941137


Reading protocols from files
--------------------------------

It is also fairly straightforward to solve quantum key rates from
``.mat`` files from, e.g., `here <https://www.math.uwaterloo.ca/~hwolkowi/henry/reports/ZGNQKDmainsolverUSEDforPUBLCNJuly31/>`_ or 
`here <https://github.com/kerry-he/qrep-structure/tree/main/data>`_.
We supply some sample code for how to do this below.

.. code-block:: python

    import numpy
    import scipy
    import qics

    # Read file
    data   = scipy.io.loadmat('filename.mat')
    gamma  = data['gamma']
    Gamma  = list(data['Gamma'].ravel())
    K_list = list(data['Klist'].ravel())
    Z_list = list(data['Zlist'].ravel())

    iscomplex = numpy.iscomplexobj(Gamma) or numpy.iscomplexobj(K_list)
    dtype = numpy.complex128 if iscomplex else numpy.float64

    no, ni = numpy.shape(K_list[0])
    nc     = numpy.size(gamma)
    vni    = qics.vectorize.vec_dim(ni, iscomplex=iscomplex)

    # Define objective function
    c = numpy.vstack((numpy.array([[1.]]), numpy.zeros((vni, 1))))

    # Build linear constraints
    A = numpy.zeros((nc, 1 + vni))
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

