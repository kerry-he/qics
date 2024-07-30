Introduction
============

**QICS** (Quantum Information Conic Solver) is a primal-dual interior point 
solver fully implemented in Python, and is specialized towards problems arising 
in quantum information theory. **QICS** solves conic programs of the form

.. math::

   (\text{P}) &&\min_{x \in \mathbb{R}^n} &&& c^\top x  &&&  (\text{D}) &&\max_{y \in \mathbb{R}^p, z \in \mathbb{R}^q} &&& -b^\top y - h^\top z

    &&\text{s.t.} &&& b - Ax = 0                &&&  &&\text{s.t.}\;\;\ &&& c + A^\top y + G^\top z = 0

    &&&&& h - Gx \in \mathcal{K}                &&&  &&&&& z \in \mathcal{K}_*

where :math:`c \in \mathbb{R}^n`, :math:`b \in \mathbb{R}^p`, 
:math:`h \in \mathbb{R}^q`, :math:`A \in \mathbb{R}^{p \times n}`, 
:math:`G \in \mathbb{R}^{q \times n}`, and :math:`\mathcal{K} \subset \mathbb{R}^{q}` 
is a Cartesian product of convex cones. Some notable cones that **QICS**
supports include:

- Complex-valued (i.e., Hermitian) positive semidefinite cone
- Quantum relative entropy
- Quantum conditional entropy
- Quantum key distribution
- Operator perspective

The full list of supported cones can be found :doc:`here</cones>`.


Features
--------------------

Features of **QICS** include:

- **Efficient SDP solver**

  NT scaling algorithm is used whenever :math:`\mathcal{K}`
  is a symmetric cone (e.g., for LP and SDP). Standard techniques used by SDP software to 
  exploit sparsity in :math:`A` or :math:`G` are implemented, which can significantly speed 
  up the solver.

- **Complex-valued matrices**

  Users can specify whether cones involving symmetric matrices, 
  such as the positive semidefinite cone or quantum relative entropy cone, are real-valued
  or complex-valued (i.e., Hermitian). Support for Hermitian matrices is embedded directly in
  the definition of the cone, which is more computationally efficient than `lifting into the real-valued 
  symmetric cone <https://docs.mosek.com/modeling-cookbook/sdo.html#hermitian-matrices>`_.

- **Cones for quantum information**

  Efficient implementations of epigraphs of important functions arising in
  quantum information theory, including the quantum relative entropy, 
  quantum conditional entropy, and operator perspective. **QICS** also
  supports an efficient implementation of a slice of the quantum relative 
  entropy cone used to solve for quantum key rates.