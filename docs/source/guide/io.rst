.. |_| unicode:: 0xA0 0xA0 0xA0 0xA0 0xA0 0xA0 0xA0 0xA0
   :trim:

Reading and writing
===================

We provide some helper functions in the :mod:`qics.io` module to read and write
conic problems. We currently support the SDPA sparse and Conic Benchmark
Library formats, which we have modified to account support problems with complex
Hermitian variables and the additional cones which **QICS** supports.


SDPA sparse
-----------

The SDPA sparse file format ``*.dat-s`` is used to encode sparse semidefinite
programs in standard form

.. math::

    \min_{x \in \mathbb{R}^p} &&& c^\top x

    \text{s.t.} &&& \sum_{i=1}^p F_i x_i - F_0 \succeq 0.

We refer to :ref:`[1] <io_refs>` for additional details about this file 
format. To account for complex semidefinite programs, we introduce the complex
SDPA sparse file format ``*.dat-c``, which is a straightforward modification
of the original format which allows values corresponding to :math:`F_i` to be
complex. 

For example, the problem

.. math::

    \min_{x \in \mathbb{R}^3} \quad & 48x_1 - 8x_2 + 20x_3 \\
    \text{s.t.} \quad & \begin{bmatrix} 10 & 4i \\ -4i & 0 \end{bmatrix} x_1
    + \begin{bmatrix} 0 & 0 \\ 0 & -8 \end{bmatrix} x_2
    + \begin{bmatrix} 0 & -8-2i \\ -8+2i & 2 \end{bmatrix} x_3
    - \begin{bmatrix} -11 & 23 \\ 23 & 0 \end{bmatrix} \succeq 0,

can be stored as the following ``*.dat-c`` file.

.. code-block:: text
    :caption: example.dat-c

    3 = mDIM
    1 = nBLOCK
    2 = bLOCKsTRUCTURE
    48.0 -8.0 20.0 
    0 1 1 1 -11-0j
    0 1 1 2 23-0j
    1 1 1 1 10+0j
    1 1 1 2 4j
    2 1 2 2 -8+0j
    3 1 1 2 -8-2j
    3 1 2 2 2+0j

To read this file, we can use the :func:`qics.io.read_sdpa` function as follows.

.. code-block::

    import qics
    model = qics.io.read_sdpa("example.dat-c")
    solver = qics.Solver(model)
    solver.solve()

Similarly, we can write a semidefinite program represented by a 
:class:`qics.Model` to a file by calling

.. code-block::

    qics.io.write_sdpa(model, "example.dat-c")

which writes a semidefinite program in the SDPA sparse format into the file 
``example.dat-c``.

.. note::

    The SDPA sparse format only supports standard form semidefinite programs.
    Therefore, :func:`qics.io.write_sdpa` must be used with a 
    :class:`qics.Model` which is initialized like

    - ``qics.Model(c, A=A, b=b, cones=cones)``
    - ``qics.Model(c, G=G, h=h, cones=cones)``

    i.e., in standard primal or dual form.



Conic Benchmark Library
-----------------------

The Conic Benchmark Library (CBF) file format ``*.cbf`` is used to encode 
conic programs of a general standardized form which encompasses the standard
form conic program used by **QICS**

.. math::

  \min_{x \in \mathbb{R}^n} &&& c^\top x
  
  \text{s.t.} &&& b - Ax = 0
  
              &&& h - Gx \in \mathcal{K}.

We refer to :ref:`[2] <io_refs>` for details about this format. The advantage of
the CBF format is that it provides flexibility in how the cone
:math:`\mathcal{K}` is defined by using keywords to define the type and 
structure of a Cartesian product of cones. Notable examples of cones currently
supported by CBF are listed below.

.. list-table:: **Existing cones**
   :widths: 30 15 55
   :header-rows: 1
   :align: center

   * - Name
     - CBF name
     - |_| |_| |_| |_| |_| |_| Description |_| |_| |_| |_| |_| |_|
   * - Positive orthant
     - ``L+``
     - :math:`\{ x \in \mathbb{R}^n : x \geq 0 \}`
   * - Quadratic cone
     - ``Q``
     - :math:`\{(t, x)\in\mathbb{R}\times\mathbb{R}^{n}:t\geq\|x\|_2\}.`
   * - Semidefinite cone
     - ``SVECPSD``
     - :math:`\{ X \in \mathbb{S}^n : X \succeq 0 \}`

We note that the ``SVECPSD`` cone is a symmetric vector form of the positive
semidefinite cone which represents matrices using the compact vectorization
described in :doc:`matrices`, i.e., the ``SVECPSD`` cone is actually

.. math::

    \{ \text{cvec}(X) : X\in\mathbb{S}^n_{++} \}.

Following this convention, we define new cones in the CBF format for all cones
supported by **QICS**.

.. list-table:: **New non-parametric cones**
   :widths: 30 15 55
   :header-rows: 1
   :align: center

   * - Name
     - CBF name
     - Description
   * - Complex semidefinite cone
     - ``HVECPSD``
     - :math:`\{ X \in \mathbb{H}^n : X \succeq 0 \}`
   * - Classical entropy cone
     - ``CE``
     - :math:`\text{cl}\{ (t, u, x) \in \mathbb{R} \times \mathbb{R}_{++} \times
       \mathbb{R}^n_{++} : t \geq -u H(x / u) \}`
   * - Classical relative entropy cone
     - ``CRE``
     - :math:`\text{cl}\{ (t, x, y) \in \mathbb{R} \times \mathbb{R}^n_{++}
       \times \mathbb{R}^n_{++} : t \geq H(x \| y) \}`
   * - Quantum entropy cone
     - ``SVECQE``
     - :math:`\text{cl}\{ (t, u, X) \in \mathbb{R} \times \mathbb{R}_{++} \times
       \mathbb{S}^n_{++} : t \geq -u S(X / u) \}`
   * - Complex quantum entropy cone
     - ``HVECQE``
     - :math:`\text{cl}\{ (t, u, X) \in \mathbb{R} \times \mathbb{R}_{++} \times
       \mathbb{H}^n_{++} : t \geq -u S(X / u) \}`
   * - Quantum relative entropy cone
     - ``SVECQRE``
     - :math:`\text{cl}\{ (t, X, Y) \in \mathbb{R} \times \mathbb{S}^n_{++}
       \times \mathbb{S}^n_{++} : t \geq S(X \| Y) \}`
   * - Complex quantum relative entropy cone
     - ``HVECQRE``
     - :math:`\text{cl}\{ (t, X, Y) \in \mathbb{R} \times \mathbb{H}^n_{++}
       \times \mathbb{H}^n_{++} : t \geq S(X \| Y) \}`
   * - Operator relative entropy cone
     - ``SVECORE``
     - :math:`\text{cl}\{ (T, X, Y) \in \mathbb{S}^n \times \mathbb{S}^n_{++}
       \times \mathbb{S}^n_{++} : T \succeq -P_{\log}(X, Y) \}`
   * - Complex operator relative entropy cone
     - ``HVECORE``
     - :math:`\text{cl}\{ (T, X, Y) \in \mathbb{H}^n \times \mathbb{H}^n_{++}
       \times \mathbb{H}^n_{++} : T \succeq -P_{\log}(X, Y) \}`
   * - Trace operator relative entropy cone
     - ``SVECTRE``
     - :math:`\text{cl}\{ (t, X, Y) \in \mathbb{R} \times \mathbb{S}^n_{++}
       \times \mathbb{S}^n_{++} : t \geq -\text{tr}[P_{\log}(X, Y)] \}`
   * - Complex trace operator relative entropy cone
     - ``HVECTRE``
     - :math:`\text{cl}\{ (t, X, Y) \in \mathbb{R} \times \mathbb{H}^n_{++}
       \times \mathbb{H}^n_{++} : t \geq -\text{tr}[P_{\log}(X, Y)] \}`


.. list-table:: **New parametric cones**
   :widths: 30 15 55
   :header-rows: 1
   :align: center

   * - Name
     - CBF name
     - Description
   * - Quantum conditional entropy cone
     - ``SVECQCE``
     - :math:`\text{cl}\{ (t, X) \in \mathbb{R} \times \mathbb{S}^{n}_{++} :
       t \geq -S(X) + S(\text{tr}_i(X)) \}`
   * - Complex quantum conditional entropy cone
     - ``HVECQCE``
     - :math:`\text{cl}\{ (t, X) \in \mathbb{R} \times \mathbb{H}^{n}_{++} :
       t \geq -S(X) + S(\text{tr}_i(X)) \}`
   * - Quantum key distribution cone
     - ``SVECQKD``
     - :math:`\text{cl}\{ (t, X) \in \mathbb{R} \times \mathbb{S}^n_{++} :
       t \geq -S(\mathcal{G}(X)) + S(\mathcal{Z}(\mathcal{G}(X))) \}`
   * - Complex quantum key distribution cone
     - ``HVECQKD``
     - :math:`\text{cl}\{ (t, X) \in \mathbb{R} \times \mathbb{H}^n_{++} :
       t \geq -S(\mathcal{G}(X)) + S(\mathcal{Z}(\mathcal{G}(X))) \}`
   * - Matrix geometric mean cone
     - ``SVECMGM``
     - :math:`\text{cl}\{ (T, X, Y) \in \mathbb{S}^n \times \mathbb{S}^n_{++}
       \times \mathbb{S}^n_{++} : T \succeq P_{\alpha}(X, Y) \}`
   * - Complex matrix geometric mean cone
     - ``HVECMGM``
     - :math:`\text{cl}\{ (T, X, Y) \in \mathbb{H}^n \times \mathbb{H}^n_{++}
       \times \mathbb{H}^n_{++} : T \succeq P_{\alpha}(X, Y) \}`
   * - Trace matrix geometric mean cone
     - ``SVECTGM``
     - :math:`\text{cl}\{ (t, X, Y) \in \mathbb{R} \times \mathbb{S}^n_{++}
       \times \mathbb{S}^n_{++} : t \geq \text{tr}[P_{\alpha}(X, Y)] \}`
   * - Complex trace matrix geometric mean cone
     - ``HVECTGM``
     - :math:`\text{cl}\{ (t, X, Y) \in \mathbb{R} \times \mathbb{H}^n_{++}
       \times \mathbb{H}^n_{++} : t \geq \text{tr}[P_{\alpha}(X, Y)] \}`

We also introduce the following new keywords to describe the above parametric
cones.

.. list-table:: **New keywords for parametric cones**
   :widths: 10 90
   :header-rows: 1

   * - Keyword
     - Description
   * - ``QCECONES``
     - Defines a lookup table for quantum conditional entropy cones.

       - ``HEADER``: One line formatted as ``INT INT``. The first number is the 
         number of cones to be specified. The second number is the combined
         length of their parameter vectors.
       - ``BODY``: A list of chunks specifying parameter vectors of a quantum
         conditional entropy cone.

         - ``CHUNKHEADER``: One line formatted as ``INT`` representing paramter
           vector length.
         - ``CHUNKBODY``: Consists of two lines. The first line is formatted as
           ``INT INT...`` representing the dimensions of the subsystems. The
           second line is formatted as ``INT INT...`` representing which
           subsystems are being traced out. 

         The specified cone at index :math:`k` (counted from 0) is registered 
         under the CBF name ``@k:SVECQCE`` or ``@k:HVECQCE``. The first and 
         second number stated in the header should match the number of chunks 
         and the sum of chunk header values, respectively.
   * - ``QKDCONES``
     - Defines a lookup table for quantum key distribution cones.

       - ``HEADER``: One line formatted as ``INT INT``. The first number is the 
         number of cones to be specified. The second number is the combined
         length of their parameter vectors.
       - ``BODY``: A list of chunks specifying parameter vectors of a quantum
         key distribution cone.

         - ``CHUNKHEADER``: One line formatted as ``INT`` representing paramter
           vector length.
         - ``CHUNKBODY``: Contains two subchunks corresponding to the Kraus
           operators of the linear maps :math:`\mathcal{G}` and 
           :math:`\mathcal{Z}`.

           - ``GCHUNKHEADER``: One line formatted as ``INT INT INT INT BOOL``,
             representing the total number of nonzeros, the number of Kraus 
             operators, the dimensions of the Kraus operators, and whether
             the Kraus operators are real or complex, respectively.
           - ``GCHUNKBODY``: A list of lines formatted as ``INT INT INT FLOAT
             (FLOAT)``, representing the index of the Kraus operator, the row
             index, column index, and real and complex components of the 
             coefficient value. The number of lines should match the number
             stated in the subchunk header.
           - ``ZCHUNKHEADER``: One line formatted as ``INT INT INT INT BOOL``,
             representing the total number of nonzeros, the number of Kraus 
             operators, the dimensions of the Kraus operators, and whether
             the Kraus operators are real or complex, respectively.
           - ``ZCHUNKBODY``: A list of lines formatted as ``INT INT INT FLOAT
             (FLOAT)``, representing the index of the Kraus operator, the row
             index, column index, and real and complex components of the 
             coefficient value. The number of lines should match the number
             stated in the subchunk header.

         The specified cone at index :math:`k` (counted from 0) is registered 
         under the CBF name ``@k:SVECQKD`` or ``@k:HVECQKD``. The first and 
         second number stated in the header should match the number of chunks
         and the sum of chunk header values, respectively.
   * - ``MGMCONES``
     - Defines a lookup table for matrix geometric mean cones.

       - ``HEADER``: One line formatted as ``INT INT``. The first number is the 
         number of cones to be specified. The second number is the combined
         length of their parameter vectors.
       - ``BODY``: A list of chunks specifying parameter vectors of a matrix
         geometric mean cones.

         - ``CHUNKHEADER``: One line formatted as ``INT`` representing paramter
           vector length.
         - ``CHUNKBODY``: One line formatted as ``FLOAT`` representing the 
           power of the weighted matrix geometric mean. 

         The specified cone at index :math:`k` (counted from 0) is registered 
         under the CBF name ``@k:SVECMGM``, ``@k:HVECMGM``, ``@k:SVECTGM`` or 
         ``@k:HVECTGM``. The first and second number stated in the header should
         match the number of chunks and the sum of chunk header values,
         respectively.

To read a file in the CBF format, we can use the :func:`qics.io.read_cbf`
function as follows.

.. code-block::

    import qics
    model = qics.io.read_sdpa("example.cbf")
    solver = qics.Solver(model)
    solver.solve()

Similarly, we can write a semidefinite program represented by a 
:class:`qics.Model` to a file by calling

.. code-block::

    qics.io.write_cbf(model, "example.cbf")

which writes a semidefinite program in the SDPA sparse format into the file 
``my_arch0.cbf``.

.. warning::

    The support for the CBF format by **QICS** is currently quite limited, and 
    it is recommended that reading and writing using file format is restricted
    to problems generated by **QICS**.

.. _io_refs:

References
----------

    1. "SDPA (SemiDefinite Programming Algorithm) User’s Manual -- 
       Version 6.2.0.", K. Fujisawa, M. Kojima, K. Nakata, and M. Yamashita,
       *Research Reports on Mathematical and Computing Sciences Series B : 
       Operations Research*, 2002.

    2. "CBLIB 2014: a benchmark library for conic mixed-integer and continuous 
       optimization," H. A. Friberg, Mathematical Programming Computation 
       8 (2016): 191-214.