Semidefinite programming
===========================

Here, we outline how various examples of semidefinite programs can be solved 
using QICS, which we have organized into the following categories.

.. grid:: 1 1 2 2
    :gutter: 2
    :padding: 0
    :class-container: surface

    .. grid-item-card:: Max cut
        :link: maxcut.html

        Semidefinite relaxation of the max cut problem and its complex
        counterpart.

    .. grid-item-card:: Block diagonal
        :link: nearest.html

        How block-diagonal structure in a semidefinite program can be exploited.

    .. grid-item-card:: Quantum information
        :link: quantum.html

        Suite of semidefinite programs arising in quantum information theory.

    .. grid-item-card:: SDPA files
        :link: sdpa.html

        Reading and writing semidefinite programs in the SDPLIB sparse file
        format.

.. toctree::
   :hidden:
   :maxdepth: 1

   maxcut.rst
   product.rst
   quantum.rst
   sdpa.rst