API reference
=============

The main ``qics`` namespace only contains two classes, which are the primary
interfaces used to construct and solve a conic program in QICS.

.. list-table::
   :widths: 20 80

   * - :mod:`~qics.Model`
     - Representation of a conic program
   * - :mod:`~qics.Solver`
     - Solves a given conic program

Submodules
----------

QICS also provides the following submodules to help construct conic programs
and provide some additional quality-of-life functionalities.

.. list-table::
   :widths: 20 80

   * - :mod:`~qics.cones`
     - Cone oracles
   * - :mod:`~qics.io`
     - Reading and writing conic programs to files
   * - :mod:`~qics.point`
     - Cartesian products of vector spaces
   * - :mod:`~qics.quantum`
     - Useful functions for quantum information theory
   * - :mod:`~qics.vectorize`
     - Converting symmetric and Hermitian matrices to column vectors

.. toctree::
   :hidden:
   :maxdepth: 2

   qics.rst
   cones.rst
   io.rst
   point.rst
   quantum.rst
   vectorize.rst