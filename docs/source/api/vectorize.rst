.. py:module:: qics.vectorize

Representing matrices as vectors (:mod:`qics.vectorize`)
========================================================

This module provides functionalities for obtaining column vector representations
of symmetric and Hermitian matrices, and also to convert these vector 
representations back into their matrix forms. Most of these functions share the
following two parameters which lead to different vectorizations of a matrix.

- ``iscomplex``: Whether to assume the matrix we are vectorizing to or from is
  a complex Hermitian matrix or just a real symmetric matrix.
- ``compact``: Whether to use a compact vectorization which discards duplicate
  or redundant entries from the symmetric or Hermitian matrix.

Further details about these vectorizations can be found in the
:doc:`user guide</guide/matrices>`.

.. list-table::
   :widths: 40 60

   * - :obj:`~qics.vectorize.eye`\ (n[, iscomplex , compact])
     - Matrix representation of the identity superoperator
   * - :obj:`~qics.vectorize.lin_to_mat`\ 
       (lin, dims[, iscomplex , compact])
     - Matrix representation of a given linear superoperator
   * - :obj:`~qics.vectorize.mat_dim`\ (len[, iscomplex , compact])
     - Dimension of the matrix corresponding to a vector representation
   * - :obj:`~qics.vectorize.mat_to_vec`\ (mat[, compact])
     - Converts a symmetric or Hermitian matrix into a column vector
   * - :obj:`~qics.vectorize.vec_dim`\ (side[, iscomplex , compact])
     - Dimension of the vector representation of a matrix
   * - :obj:`~qics.vectorize.vec_to_mat`\ (vec[, iscomplex , compact])
     - Converts a column vector into a symmetric or Hermitian matrix

.. toctree::
   :hidden:
   :maxdepth: 0

   vectorize/eye
   vectorize/lin_to_mat
   vectorize/mat_dim
   vectorize/mat_to_vec
   vectorize/vec_dim
   vectorize/vec_to_mat