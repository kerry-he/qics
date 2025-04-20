.. py:module:: qics.io

Reading and writing (:mod:`qics.io`)
====================================

This module contains functions used to read and write conic programs 
represented by a :class:`~qics.Model` to a file of a specified format.
Currently, we support the following file formats

- ``*.dat-s``: SDPA sparse format
- ``*.dat-c``: Complex SDPA sparse format
- ``*.cbf``: Conic Benchmark Format

Further details about these file formats can be found in the
:doc:`user guide</guide/io>`.

.. list-table::
   :widths: 40 60

   * - :obj:`~qics.io.read_file`\ (filename)
     - Reads a conic program from a file of a specified format.
   * - :obj:`~qics.io.write_file`\ (model, filename)
     - Write a conic program to a file of a specified format

SDPA sparse format
------------------

.. list-table::
   :widths: 40 60

   * - :obj:`~qics.io.read_sdpa`\ (filename)
     - Reads a semidefinite program in the SDPA sparse format.
   * - :obj:`~qics.io.write_sdpa`\ (model, filename)
     - Write a semidefinite program to a SDPA sparse file.

Conic Benchmark Format
----------------------

.. list-table::
   :widths: 40 60

   * - :obj:`~qics.io.read_cbf`\ (filename)
     - Reads a conic program in the CBF format.
   * - :obj:`~qics.io.write_cbf`\ (model, filename)
     - Write a conic program to a CBF file.


.. toctree::
   :hidden:
   :maxdepth: 0

   io/read_file
   io/write_file
   io/read_sdpa
   io/write_sdpa
   io/read_cbf
   io/write_cbf
