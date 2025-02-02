Quantum Information Conic Solver
================================

.. image:: https://readthedocs.org/projects/qics/badge/?version=latest
  :target: https://qics.readthedocs.io/en/latest/?badge=latest
.. image:: http://github.com/kerry-he/qics/actions/workflows/ci.yml/badge.svg?event=push
  :target: http://github.com/kerry-he/qics/actions/workflows/ci.yml
.. image:: http://img.shields.io/pypi/v/qics.svg
  :target: https://pypi.python.org/pypi/qics/
.. image:: https://img.shields.io/aur/version/python-qics
  :target: https://aur.archlinux.org/packages/python-qics

**QICS** is a primal-dual interior point solver fully implemented in Python, and
is specialized towards problems arising in quantum information theory. **QICS**
solves conic programs of the form

.. math::

  \min_{x \in \mathbb{R}^n} \quad c^\top x \quad 
  \text{s.t.} \quad b - Ax = 0, \  h - Gx \in \mathcal{K},

where :math:`c\in\mathbb{R}^n`, :math:`b\in\mathbb{R}^p`, 
:math:`h\in\mathbb{R}^q`, :math:`A\in\mathbb{R}^{p\times n}`, 
:math:`G\in\mathbb{R}^{q\times n}`, and :math:`\mathcal{K}\subset\mathbb{R}^{q}`
is a Cartesian product of convex cones. Some notable cones that **QICS** 
supports include:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1
   :align: center

   * - Cone
     - QICS class
     - Description
   * - Positive semidefinite
     - :class:`qics.cones.PosSemidefinite`
     - :math:`\{ X \in \mathbb{H}^n : X \succeq 0 \}`
   * - Quantum entropy
     - :class:`qics.cones.QuantEntr`
     - :math:`\text{cl}\{ (t, u, X) \in \mathbb{R} \times \mathbb{R}_{++} \times 
       \mathbb{H}^n_{++} : t \geq -u S(u^{-1} X) \}`
   * - Quantum relative entropy
     - :class:`qics.cones.QuantRelEntr`
     - :math:`\text{cl}\{ (t, X, Y) \in \mathbb{R} \times \mathbb{H}^n_{++}
       \times \mathbb{H}^n_{++} : t \geq S(X \| Y) \}`
   * - Quantum conditional entropy
     - :class:`qics.cones.QuantCondEntr`
     - :math:`\text{cl}\{ (t, X) \in \mathbb{R}\times\mathbb{H}^{\Pi_in_i}_{++}:
       t \geq -S(X) + S(\text{tr}_i(X)) \}`
   * - Quantum key distribution
     - :class:`qics.cones.QuantKeyDist`
     - :math:`\text{cl}\{ (t, X) \in \mathbb{R} \times \mathbb{H}^n_{++} :
       t \geq -S(\mathcal{G}(X)) + S(\mathcal{Z}(\mathcal{G}(X))) \}`
   * - Operator perspective epigraph
     - :class:`qics.cones.OpPerspecEpi`
     - :math:`\text{cl}\{ (T, X, Y) \in \mathbb{H}^n \times \mathbb{H}^n_{++}
       \times \mathbb{H}^n_{++} : T \succeq P_g(X, Y) \}`
   * - :math:`\alpha`-Renyi entropy, for :math:`\alpha\in[0, 1)`
     - :class:`qics.cones.RenyiEntr`
     - :math:`\text{cl} \{ (t, u, X, Y) \in \mathbb{R} \times \mathbb{R}_{++} \times 
       \mathbb{H}^n_{++} \times \mathbb{H}^n_{++} : t \geq u D_\alpha(u^{-1}X \| u^{-1}Y) \}`
   * - Sandwiched :math:`\alpha`-Renyi entropy, for :math:`\alpha\in[1/2, 1)`
     - :class:`qics.cones.SandRenyiEntr`
     - :math:`\text{cl} \{ (t, u, X, Y) \in \mathbb{R} \times \mathbb{R}_{++} \times 
       \mathbb{H}^n_{++} \times \mathbb{H}^n_{++} : t \geq u \hat{D}_\alpha(u^{-1}X \| u^{-1}Y) \}`
   * - :math:`\alpha`-Quasi-relative entropy, for :math:`\alpha\in[-1, 2]`
     - :class:`qics.cones.QuasiEntr`
     - :math:`\text{cl} \{ (t, X, Y) \in \mathbb{R} \times \mathbb{H}^n_{++} \times
       \mathbb{H}^n_{++} : t \geq \pm \text{tr}[ X^\alpha Y^{1-\alpha} ] \}`
   * - Sandwiched :math:`\alpha`-quasi-relative entropy, for :math:`\alpha\in[1/2, 2]`
     - :class:`qics.cones.SandQuasiEntr`
     - :math:`\text{cl} \{ (t, X, Y) \in \mathbb{R} \times \mathbb{H}^n_{++} \times
       \mathbb{H}^n_{++} : t \geq \pm \text{tr}[ ( Y^{\frac{1-\alpha}{2\alpha}} X 
       Y^{\frac{1-\alpha}{2\alpha}} )^\alpha ] \}`

where we define the following functions

- Quantum entropy: :math:`S(X)=-\text{tr}[X\log(X)]`
- Quantum relative entropy: :math:`S(X \| Y)=\text{tr}[X\log(X) - X\log(Y)]`
- Noncommutative perspective: :math:`P_g(X, Y)=X^{1/2} g(X^{-1/2} Y X^{-1/2}) X^{1/2}`
- :math:`\alpha`-Renyi entropy: :math:`D_\alpha(X\|Y)=\frac{1}{1-\alpha} \log(\text{tr}[X^\alpha Y^{1-\alpha}])`
- Sandwiched :math:`\alpha`-Renyi entropy: :math:`\hat{D}_\alpha(X \| Y) = \frac{1}{1-\alpha} \log(\text{tr}[ (Y^{\frac{1-\alpha}{2\alpha}} X Y^{\frac{1-\alpha}{2\alpha}})^\alpha ])`

The full list of supported cones can be found 
:ref:`here<guide/reference:cones>`.


Features
--------------------

- **Efficient quantum relative entropy programming**

  We support optimizing over the quantum relative entropy cone, as well as 
  related cones including the quantum conditional entropy cone, as well as 
  slices of the quantum relative entropy cone that arise when solving quantum 
  key rates of quantum cryptographic protocols. Numerical results show that 
  **QICS** solves problems much faster than existing quantum relative entropy 
  programming solvers, such as 
  `Hypatia <https://github.com/jump-dev/Hypatia.jl>`_, `DDS
  <https://github.com/mehdi-karimi-math/DDS>`_, and `CVXQUAD
  <https://github.com/hfawzi/cvxquad>`_.

- **Efficient semidefinite programming**

  We implement an efficient semidefinite programming solver which utilizes 
  state-of-the-art techniques for symmetric cone programming, including using 
  Nesterov-Todd scalings and exploiting sparsity in the problem structure. 
  Numerical results show that **QICS** has comparable performance to 
  state-of-the-art semidefinite programming software, such as 
  `MOSEK <https://www.mosek.com/>`_, 
  `SDPA <https://sdpa.sourceforge.net/index.html>`_, `SDPT3 
  <https://www.math.cmu.edu/~reha/sdpt3.html>`_ and `SeDuMi 
  <https://sedumi.ie.lehigh.edu/>`_.

- **Complex-valued matrices**

  Users can specify whether cones involving symmetric matrices, such as the
  positive semidefinite cone or quantum relative entropy cone, are real-valued
  or complex-valued (i.e., Hermitian). Support for Hermitian matrices is 
  embedded directly in the definition of the cone, which can be more 
  computationally efficient than `lifting into the real-valued symmetric cone 
  <https://docs.mosek.com/modeling-cookbook/sdo.html#hermitian-matrices>`_.


Installation
------------

**QICS** is currently supported for Python 3.8 or later, and can be directly
installed from `pip <https://pypi.org/project/qics/>`_ by calling

.. code-block:: bash

    pip install qics

Note that the performance of QICS is highly dependent on the version of BLAS and
LAPACK that `Numpy <https://numpy.org/devdocs/building/blas_lapack.html>`_ and 
`SciPy <https://docs.scipy.org/doc/scipy/building/blas_lapack.html>`_ are linked to.


PICOS interface
---------------

The easiest way to use **QICS** is through the Python optimization modelling 
interface `PICOS <https://picos-api.gitlab.io/picos/>`_, which can be installed using

.. code-block:: bash

    pip install picos

Below, we show how a simple :ref:`nearest correlation matrix<examples/qrep/nearest:nearest 
correlation matrix>` problem can be solved.

.. code-block:: python

   import picos

   # Define the conic program
   P = picos.Problem()
   X = picos.Constant("X", [[2., 1.], [1., 2.]])
   Y = picos.SymmetricVariable("Y", 2)
   
   P.set_objective("min", picos.quantrelentr(X, Y))
   P.add_constraint(picos.maindiag(Y) == 1)

   # Solve the conic program
   P.solve(solver="qics")

Some additional details about how to use QICS with PICOS can be found 
:doc:`here<guide/picos>`.


Native interface
----------------

Alternatively, advanced users can use the QICS' native interface, which provides 
additional flexibilty in how the problem is parsed to the solver. Below, we show
how the same nearest correlation matrix problem can be solved using QICS' native
interface. 

.. code-block:: python

   import numpy
   import qics

   # Define the conic program
   c = numpy.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.]]).T
   A = numpy.array([
      [0., 1., 0., 0., 0., 0., 0., 0., 0.],
      [0., 0., 1., 1., 0., 0., 0., 0., 0.],
      [0., 0., 0., 0., 1., 0., 0., 0., 0.],
      [0., 0., 0., 0., 0., 1., 0., 0., 0.],
      [0., 0., 0., 0., 0., 0., 0., 0., 1.]
   ])
   b = numpy.array([[2., 2., 2., 1., 1.]]).T
   cones = [qics.cones.QuantRelEntr(2)]
   model = qics.Model(c=c, A=A, b=b, cones=cones)

   # Solve the conic program
   solver = qics.Solver(model)
   info = solver.solve()

Additional details explaining this example can be found 
:doc:`here<guide/gettingstarted>`.

Citing QICS
-----------

If you find our work useful, please cite our `paper <http://arxiv.org/abs/2410.17803>`_
using:

.. code-block:: bibtex

    @article{he2024qics,
      title={{QICS}: {Q}uantum Information Conic Solver},
      author={He, Kerry and Saunderson, James and Fawzi, Hamza},
      journal={arXiv preprint arXiv:2410.17803},
      year={2024}
    }


.. toctree::
   :hidden:
   :maxdepth: 3

   Introduction<self>
   guide/index.rst
   examples/index.rst
   api/index.rst