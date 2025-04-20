.. py:module:: qics.point

Products of vectors (:mod:`qics.point`)
=======================================

This module contains classes which are used to represent vectors which lie in
Cartesian products of vector spaces, including

- Real vectors :math:`\mathbb{R}^n`
- Symmetric matrices :math:`\mathbb{S}^n`
- Hermitian matrices :math:`\mathbb{H}^n`

These classes contain a ``vec`` attribute which is the column vector 
representation of variable (see :doc:`/guide/matrices`), and we use
:obj:`numpy.ndarray.view` to link different parts of ``vec`` to different
variables. This allows us to perform efficient vectorized NumPy operations on
our vectors, while easily being able to access the different variables 
contained within the vector.

.. list-table::
   :widths: 30 70

   * - :class:`~qics.point.Point`\ (model)
     - Vector containing the variables involved in a homogeneous self-dual
       embedding of a primal-dual conic program
   * - :class:`~qics.point.VecProduct`\ (cones[, vec])
     - Cartesian product of vectors corresponding to a list of cones 


.. toctree::
   :hidden:
   :maxdepth: 0

   point/Point
   point/VecProduct