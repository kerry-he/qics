User guide
==========

This section is serve as an introductory guide and reference for how to use 
QICS, and contains the following pages.

.. grid:: 1 1 2 3
    :gutter: 2
    :padding: 0
    :class-container: surface

    .. grid-item-card:: :octicon:`zap` Quick reference
        :link: reference.html

        Serves as a reference point for how to initialize and use the cone, 
        model, and solver classes provided by QICS.

    .. grid-item-card:: :octicon:`book` Getting started
        :link: gettingstarted.html

        Walks through an example of how to solve a simple quantum relative
        entropy program using QICS.

    .. grid-item-card:: :octicon:`info` Representing matrices
        :link: matrices.html

        Explains how symmetric and Hermitian matrix variables are represented 
        as column vectors in QICS.

    .. grid-item-card:: :octicon:`mortar-board` Advanced tips
        :link: advanced.html

        Tips for advanced users to obtain potentially significant speedups
        when solving problems.

    .. grid-item-card:: :octicon:`file` Reading and writing
        :link: io.html

        Reading and writing semidefinite and conic programs using the SDPA
        sparse and CBF formats.

    .. grid-item-card:: :octicon:`command-palette` PICOS interface
        :link: picos.html

        Briefly introduces how QICS can be used with the PICOS
        optimization modelling interface. 


.. toctree::
   :hidden:
   :maxdepth: 3

   reference.rst
   gettingstarted.rst
   matrices.rst
   advanced.rst
   io.rst
   picos.rst