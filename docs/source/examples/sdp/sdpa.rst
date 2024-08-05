SDPA files
================

**QICS** also supports reading (and writing) sparse semidefinite
programs from files in the SDPA sparse file format :ref:`[1] <sdpa_refs>`.
To do this, we use the functions available from the :py:mod:`~qics.io`
submodule.

A popular library for semidefinite programs in the SDPA sparse format is
the SDPLIB library, which can be found `here <https://github.com/vsdp/SDPLIB>`_
and cloned by calling

.. code-block:: console

    $ git clone https://github.com/vsdp/SDPLIB.git

from the command line in the desired directory. In Python, we can now solve
the example semidefinite programs as follows.

.. code-block:: pycon

    >>> import qics
    >>> c, b, A, cones = qics.io.read_sdpa("SDPLIB/data/arch0.dat-s")
    >>> model = qics.Model(c=c, A=A, b=b, cones=cones)
    >>> solver = qics.Solver(model)
    >>> info = solver.solve()
    ====================================================================
                QICS v0.0 - Quantum Information Conic Solver
                by K. He, J. Saunderson, H. Fawzi (2024)
    ====================================================================
    Problem summary:
            no. cones:  2                        no. vars:    26095
            barr. par:  336                      no. constr:  174
            symmetric:  True                     cone dim:    26095
            complex:    False

    ...

    Solution summary
            sol. status:  optimal                num. iter:    23
            exit status:  solved                 solve time:   x.xxx

            primal obj:  -5.665176736338e-01     primal feas:  5.64e-10
            dual obj:    -5.665176683196e-01     dual feas:    5.56e-10
            opt. gap:     5.31e-09

We can also write a semidefinite program represented by a :class:`~qics.Model`
to a file by calling

.. code-block:: pycon

    >>> qics.io.write_sdpa(model, "my_arch0.dat-s")

which writes a semidefinite program in the SDPA sparse format into
the file ``my_arch0.dat-s``.
    
.. _sdpa_refs:

References
----------

    1. "SDPA (SemiDefinite Programming Algorithm) User’s Manual -- Version 6.2.0.",
       K. Fujisawa, M. Kojima, K. Nakata, and M. Yamashita,
       *Research Reports on Mathematical and Computing Sciences Series B : Operations Research*, 2002.

