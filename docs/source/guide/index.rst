User guide
==========

This section is serve as an introductory guide and reference for how to use 
QICS, and contains the following pages.

.. grid:: 1 1 3 2
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

        Tips for advanced users for best practices on how to input problems to
        QICS to obtain potentially significant speedups.

    .. grid-item-card:: :octicon:`command-palette` PICOS interface
        :link: picos.html

        Briefly introduces how QICS can be used with the PICOS
        interface, a Python interface for modelling optimization problems. 


.. toctree::
   :hidden:
   :maxdepth: 3

   reference.rst
   gettingstarted.rst
   matrices.rst
   advanced.rst
   picos.rst