.. py:module:: qics.quantum

Quantum information theory (:mod:`qics.quantum`)
================================================

This module contains several helper functions to perform operations which
commonly arise in quantum information theory. This module also contains the
submodule :mod:`~qics.quantum.random` which contains functions used to
generate random quantum states, channels, or other operators.

.. list-table::
   :widths: 40 60

   * - :obj:`~qics.quantum.entropy`\ (x)
     - Computes classical or quantum entropy 
   * - :obj:`~qics.quantum.i_kr`\ (mat, dims, sys)
     - Kronecker product with the identity matrix
   * - :obj:`~qics.quantum.p_tr`\ (mat, dims, sys)
     - Partial trace of a multipartite state
   * - :obj:`~qics.quantum.partial_transpose`\ (mat, dims, sys)
     - Partial transpose of a multipartite state
   * - :obj:`~qics.quantum.purify`\ (X)
     - Computes the purification of a quantum state
   * - :obj:`~qics.quantum.swap`\ (mat, dims, sys1, sys2)
     - Swaps two subsystems of a multipartite state

.. toctree::
   :hidden:
   :maxdepth: 0

   quantum/entropy
   quantum/i_kr
   quantum/p_tr
   quantum/partial_transpose
   quantum/purify
   quantum/swap


.. py:module:: qics.quantum.random

Random quantum objects (:mod:`qics.quantum.random`)
---------------------------------------------------

.. list-table::
   :widths: 40 60

   * - :obj:`~qics.quantum.random.choi_operator`\ 
       (nin[, nout, M, iscomplex])
     - Random Choi operator corresponding to a quantum channel
   * - :obj:`~qics.quantum.random.degradable_channel`\ 
       (nin[, nout, nenv, iscomplex])
     - Random isometry matrix corresponding to a degradable channel
   * - :obj:`~qics.quantum.random.density_matrix`\ (n[, iscomplex])
     - Random density matrix
   * - :obj:`~qics.quantum.random.pure_density_matrix`\ (n[, iscomplex])
     - Random rank one density matrix
   * - :obj:`~qics.quantum.random.stinespring_operator`\ 
       (nin[, nout, nenv, iscomplex])
     - Random Stinespring isometry corresponding to a degradable channel
   * - :obj:`~qics.quantum.random.unitary`\ (n[, iscomplex])
     - Random unitary matrix

.. toctree::
   :hidden:
   :maxdepth: 0

   quantum/random/choi_operator
   quantum/random/degradable_channel
   quantum/random/density_matrix
   quantum/random/pure_density_matrix
   quantum/random/stinespring_operator
   quantum/random/unitary