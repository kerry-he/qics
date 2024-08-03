# QICS: Quantum Information Conic Solver

**QICS** is a primal-dual interior point solver fully implemented in Python, and is specialized towards problems arising in quantum information theory. **QICS** solves conic programs of the form

$$
\begin{align} 
   (\text{P}) &&\min_{x \in \mathbb{R}^n} &&& c^\top x  &&&  (\text{D}) &&\max_{y \in \mathbb{R}^p, z \in \mathbb{R}^q} &&& -b^\top y - h^\top z \\
    &&\text{s.t.} &&& b - Ax = 0                &&&  &&\text{s.t.}\\;\\;\\ &&& c + A^\top y + G^\top z = 0 \\
    &&&&& h - Gx \in \mathcal{K}                &&&  &&&&& z \in \mathcal{K}_*
\end{align}
$$

where $c \in \mathbb{R}^n$, $b \in \mathbb{R}^p$, $h \in \mathbb{R}^q$, $A \in \mathbb{R}^{p \times n}$, $G \in \mathbb{R}^{q \times n}$, and $\mathcal{K} \subset \mathbb{R}^{q}$ is a Cartesian product of convex cones. Some notable cones that **QICS** supports include:

- Complex-valued (i.e., Hermitian) positive semidefinite cone
- Quantum relative entropy
- Quantum conditional entropy
- Quantum key distribution
- Operator perspective

The full list of supported cones can be found <INSERT LINK HERE>.


## Features

Features of **QICS** include:

- **Cones for quantum information**

  Efficient implementations of epigraphs of important functions arising inquantum information theory, including the quantum relative entropy, quantum conditional entropy, and operator perspective. **QICS** also
  supports an efficient implementation of a slice of the quantum relative entropy cone used to solve for quantum key rates.

- **Efficient SDP solver**

  NT scaling algorithm is used whenever $\mathcal{K}$ is a symmetric cone (e.g., for LP and SDP). Standard techniques used by SDP software to exploit sparsity in $A$ or $G$ are implemented, which can significantly speed up the solver.

- **Complex-valued matrices**

  Users can specify whether cones involving symmetric matrices, such as the positive semidefinite cone or quantum relative entropy cone, are real-valuedor complex-valued (i.e., Hermitian). Support for Hermitian matrices is embedded directly in the definition of the cone, which is more computationally efficient than [lifting into the real-valued symmetric cone](https://docs.mosek.com/modeling-cookbook/sdo.html#hermitian-matrices).
