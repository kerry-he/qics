.. py:module:: qics.cones

Cone oracles (:mod:`qics.cones`)
================================

This module contains classes representing conic sets which users use to define 
the Cartesian product of cones :math:`\mathcal{K}` the conic program is defined
over. These classes contain feasibility, gradient, Hessian product, inverse
Hessian product, and third order derivative oracles which are required by the
interior-point algorithm. Symmetric cones contain additional oracles for 
computing the Nesterov-Todd scalings and other related functions.

Symmetric cones
--------------------

.. list-table::
   :widths: 50 50

   * - :class:`~qics.cones.NonNegOrthant`\ (n)
     - Nonnegative orthant
   * - :class:`~qics.cones.PosSemidefinite`\ (n[, iscomplex])
     - Positive semidefinite cone
   * - :class:`~qics.cones.SecondOrder`\ (n)
     - Second order cone


Classical entropy cones
---------------------------

.. list-table::
   :widths: 50 50

   * - :class:`~qics.cones.ClassEntr`\ (n)
     - Classical entropy cone
   * - :class:`~qics.cones.ClassRelEntr`\ (n)
     - Classical relative entropy cone


Quantum entropy cones
---------------------------

.. list-table::
   :widths: 50 50

   * - :class:`~qics.cones.QuantEntr`\ (n[, iscomplex])
     - Quantum entropy cone
   * - :class:`~qics.cones.QuantRelEntr`\ (n[, iscomplex])
     - Quantum relative entropy cone
   * - :class:`~qics.cones.QuantCondEntr`\ (sys, dims[, iscomplex])
     - Quantum conditional entropy cone
   * - :class:`~qics.cones.QuantKeyDist`\ (G_info, K_info[, iscomplex])
     - Quantum key distribution cone

Operator perspective cones
---------------------------

.. list-table::
   :widths: 50 50

   * - :class:`~qics.cones.OpPerspecTr`\ (n, func[, iscomplex])
     - Trace operator perspective cone
   * - :class:`~qics.cones.OpPerspecEpi`\ (n, func[, iscomplex])
     - Operator perspective epigraph


.. toctree::
   :hidden:
   :maxdepth: 0

   cones/NonNegOrthant
   cones/PosSemidefinite
   cones/SecondOrder
   cones/ClassEntr
   cones/ClassRelEntr
   cones/QuantEntr
   cones/QuantRelEntr
   cones/QuantCondEntr
   cones/QuantKeyDist
   cones/OpPerspecTr
   cones/OpPerspecEpi