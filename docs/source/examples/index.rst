Examples
==================

Here, we provide examples demonstrating how **QICS** can be used to solve a
variety of semidefinite and quantum relative entropy programs, with an emphasis
on applications arising in quantum informaiton theory. 

It is recommended that beginners first read :doc:`/guide/gettingstarted` and 
:doc:`/guide/matrices` to better understand the example code provided in these
pages.

**Semidefinite programming**

.. grid:: 1 1 2 2
    :gutter: 2
    :padding: 0
    :class-container: surface

    .. grid-item-card:: Max cut
        :link: sdp/maxcut.html

        Semidefinite relaxation of the max cut problem and its complex
        counterpart.

    .. grid-item-card:: Block diagonal
        :link: sdp/nearest.html

        How block-diagonal structure in a semidefinite program can be exploited.

    .. grid-item-card:: Quantum information
        :link: sdp/quantum.html

        Suite of semidefinite programs arising in quantum information theory.

    .. grid-item-card:: SDPA files
        :link: sdp/sdpa.html

        Reading and writing semidefinite programs in the SDPLIB sparse file
        format.

**Quantum relative entropy programming**

.. grid:: 1 1 2 2
    :gutter: 2
    :padding: 0
    :class-container: surface

    .. grid-item-card:: Quantum key distribution
        :link: qrep/qkd.html

        How to can compute the quantum key rate for a given quantum
        cryptographic protocol. 

    .. grid-item-card:: Nearest matrix
        :link: qrep/nearest.html

        Finding the closest matrix to a given matrix in the quantum relative
        entropy sense.

    .. grid-item-card:: Quantum channels
        :link: qrep/channel.html

        Explores the fundamental limits of communication with quantum
        resources. 

    .. grid-item-card:: Noncommutative perspective
        :link: qrep/opper.html

        Some examples involving operator relative entropies and weighted 
        geometric means.


.. toctree::
   :hidden:
   :maxdepth: 2

   sdp/index.rst
   qrep/index.rst