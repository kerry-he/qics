.. py:module:: qics.cones

qics.cones
===============

QICS requires constraints and objectives to be represented using cones. Here, we
categorize and list all of the cones which are currently supported by QICS.

**Symmetric cones**

.. list-table::
   :widths: 25 25 50
   :header-rows: 1
   :align: center

   * - Cone
     - QICS class
     - Description
   * - Nonnegative orthant
     - :class:`qics.cones.NonNegOrthant`
     - :math:`\{ x \in \mathbb{R}^n : x \geq 0 \}`
   * - Positive semidefinite
     - :class:`qics.cones.PosSemidefinite`
     - :math:`\{ X \in \mathbb{H}^n : X \succeq 0 \}`
   * - Second order cone
     - :class:`qics.cones.SecondOrder`
     - :math:`\{ (t, x) \in \mathbb{R} \times \mathbb{R}^{n} : t \geq \| x \|_2 \}.`


**Classical entropy (exponential) cones**

.. list-table::
   :widths: 25 25 50
   :header-rows: 1
   :align: center

   * - Cone
     - QICS class
     - Description
   * - Classical entropy
     - :class:`qics.cones.QuantEntr`
     - :math:`\text{cl}\{ (t, u, x) \in \mathbb{R} \times \mathbb{R}_{++} \times 
       \mathbb{R}^n_{++} : t \geq -u S(u^{-1} x) \}`
   * - Classical relative entropy
     - :class:`qics.cones.QuantRelEntr`
     - :math:`\text{cl}\{ (t, x, y) \in \mathbb{R} \times \mathbb{R}^n_{++} \times 
       \mathbb{R}^n_{++} : t \geq H(x \| y) \}`

**Quantum entropy cones**

.. list-table::
   :widths: 25 25 50
   :header-rows: 1
   :align: center

   * - Cone
     - QICS class
     - Description
   * - Quantum entropy
     - :class:`qics.cones.QuantEntr`
     - :math:`\text{cl}\{ (t, u, X) \in \mathbb{R} \times \mathbb{R}_{++} \times 
       \mathbb{H}^n_{++} : t \geq -u S(u^{-1} X) \}`
   * - Quantum relative entropy
     - :class:`qics.cones.QuantRelEntr`
     - :math:`\text{cl}\{ (t, X, Y) \in \mathbb{R} \times \mathbb{H}^n_{++} \times 
       \mathbb{H}^n_{++} : t \geq S(X \| Y) \}`
   * - Quantum conditional entropy
     - :class:`qics.cones.QuantCondEntr`
     - :math:`\text{cl}\{ (t, X) \in \mathbb{R} \times \mathbb{H}^{n}_{++} : 
       t \geq -S(X) + S(\text{tr}_i(X)) \}`
   * - Quantum key distribution
     - :class:`qics.cones.QuantKeyDist`
     - :math:`\text{cl}\{ (t, X) \in \mathbb{R} \times \mathbb{H}^n_{++} : 
       t \geq -S(\mathcal{G}(X)) + S(\mathcal{Z}(\mathcal{G}(X))) \}`

**Noncommutative perspective cones**

.. list-table::
   :widths: 25 25 50
   :header-rows: 1
   :align: center

   * - Cone
     - QICS class
     - Description
   * - Operator perspective trace
     - :class:`qics.cones.OpPerspecTr`
     - :math:`\text{cl}\{ (t, X, Y) \in \mathbb{R} \times \mathbb{H}^n_{++} \times 
       \mathbb{H}^n_{++} : t \geq \text{tr}[P_g(X, Y)] \}`
   * - Operator perspective epigraph
     - :class:`qics.cones.OpPerspecEpi`
     - :math:`\text{cl}\{ (T, X, Y) \in \mathbb{H}^n \times \mathbb{H}^n_{++} \times 
       \mathbb{H}^n_{++} : T \succeq P_g(X, Y) \}`

Symmetric cones
--------------------

.. autoclass:: qics.cones.NonNegOrthant

.. autoclass:: qics.cones.PosSemidefinite

.. autoclass:: qics.cones.SecondOrder


Classical entropy cones
---------------------------

.. autoclass:: qics.cones.ClassEntr

.. autoclass:: qics.cones.ClassRelEntr


Quantum entropy cones
---------------------------

.. autoclass:: qics.cones.QuantEntr

.. autoclass:: qics.cones.QuantRelEntr

.. autoclass:: qics.cones.QuantCondEntr

.. autoclass:: qics.cones.QuantKeyDist


Operator perspective cones
---------------------------

.. autoclass:: qics.cones.OpPerspecTr

.. autoclass:: qics.cones.OpPerspecEpi