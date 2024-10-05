# QICS: Quantum Information Conic Solver

[![Documentation Status](https://readthedocs.org/projects/qics/badge/?version=latest)](https://qics.readthedocs.io/en/latest/?badge=latest)
[![Build Status](http://github.com/kerry-he/qics/actions/workflows/ci.yml/badge.svg?event=push)](http://github.com/kerry-he/qics/actions/workflows/ci.yml)
[![PyPi Status](http://img.shields.io/pypi/v/qics.svg)](https://pypi.python.org/pypi/qics/)

**QICS** is a primal-dual interior point solver fully implemented in Python, and is specialized towards problems arising in quantum information theory. **QICS** solves conic programs of the form

$$
\min_{x \in \mathbb{R}^n} \quad c^\top x \quad \text{s.t.} \quad b - Ax = 0, \  h - Gx \in \mathcal{K},
$$

where $c \in \mathbb{R}^n$, $b \in \mathbb{R}^p$, $h \in \mathbb{R}^q$, $A \in \mathbb{R}^{p \times n}$, $G \in \mathbb{R}^{q \times n}$, and $\mathcal{K} \subset \mathbb{R}^{q}$ is a Cartesian product of convex cones. Some notable cones that QICS supports include

| Cone           |  QICS class  |  Description  |
|----------------|:---------------------:|:---------------:|
| Positive semidefinite |  [`qics.cones.PosSemidefinite`](https://qics.readthedocs.io/en/stable/api/cones.html#qics.cones.PosSemidefinite)  | $\\{ X \in \mathbb{H}^n : X \succeq 0 \\}$ |
| Quantum entropy |  [`qics.cones.QuantEntr`](https://qics.readthedocs.io/en/stable/api/cones.html#qics.cones.QuantEntr)  | $\text{cl}\\{ (t, u, X) \in \mathbb{R} \times \mathbb{R}_{++} \times \mathbb{H}^n\_{++} : t \geq -u S(u^{-1} X) \\}$ |
| Quantum relative entropy |  [`qics.cones.QuantRelEntr`](https://qics.readthedocs.io/en/stable/api/cones.html#qics.cones.QuantRelEntr)  | $\text{cl}\{ (t, X, Y) \in \mathbb{R} \times \mathbb{H}^n_{++} \times \mathbb{H}^n_{++} : t \geq S(X \\| Y) \}$ |
| Quantum conditional entropy |  [`qics.cones.QuantCondEntr`](https://qics.readthedocs.io/en/stable/api/cones.html#qics.cones.QuantCondEntr)  | $\text{cl}\\{ (t, X) \in \mathbb{R} \times \mathbb{H}^{n}_{++} : t \geq -S(X) + S(\text{tr}_i(X)) \\}$ |
| Quantum key distribution |  [`qics.cones.QuantKeyDist`](https://qics.readthedocs.io/en/stable/api/cones.html#qics.cones.QuantKeyDist)  | $\text{cl}\\{ (t, X) \in \mathbb{R} \times \mathbb{H}^n_{++} : t \geq -S(\mathcal{G}(X)) + S(\mathcal{Z}(\mathcal{G}(X))) \\}$ |
| Operator perspective epigraph |  [`qics.cones.OpPerspecEpi`](https://qics.readthedocs.io/en/stable/api/cones.html#qics.cones.OpPerspecEpi)  | $\text{cl}\\{ (T, X, Y) \in \mathbb{H}^n \times \mathbb{H}^n_{++} \times \mathbb{H}^n_{++} : T \succeq P_g(X, Y) \\}$ |

where $S(X)=-\text{tr}[X\log(X)]$ is the quantum entropy, $S(X \\| Y)=\text{tr}[X\log(X) - X\log(Y)]$ is the quantum relative entropy, and $P_g(X, Y)=X^{1/2} g(X^{-1/2} Y X^{-1/2}) X^{1/2}$ is the non-commutative or operator perspective.

A full list of cones which we support can be found in our [documentation](https://qics.readthedocs.io/en/stable/api/cones.html).

## Features

- **Efficient quantum relative entropy programming**

  We support optimizing over the quantum relative entropy cone, as well as related cones including the quantum conditional entropy cone, as well as slices of the quantum relative entropy cone that arise when solving quantum key rates of quantum cryptographic protocols. Numerical results show that **QICS** solves problems much faster than existing quantum relative entropy programming solvers, such as [Hypatia](https://github.com/jump-dev/Hypatia.jl), [DDS](https://github.com/mehdi-karimi-math/DDS), and [CVXQUAD](https://github.com/hfawzi/cvxquad).

- **Efficient semidefinite programming**

  We implement an efficient semidefinite programming solver which utilizes state-of-the-art techniques for symmetric cone programming, including using Nesterov-Todd scalings and exploiting sparsity in the problem structure. Numerical results show that **QICS** has comparable performance to state-of-the-art semidefinite programming software, such as [MOSEK](https://www.mosek.com/), [SDPA](https://sdpa.sourceforge.net/index.html), [SDPT3](https://www.math.cmu.edu/~reha/sdpt3.html) and [SeDuMi](https://sedumi.ie.lehigh.edu/).

- **Complex-valued matrices**

  Users can specify whether cones involving variables which are symmetric matrices, such as the positive semidefinite cone or quantum relative entropy cone, involve real-valued or complex-valued (i.e., Hermitian) matrix variables. Support for Hermitian matrices is embedded directly in the definition of the cone, which can be more computationally efficient than [lifting into the real-valued symmetric cone](https://docs.mosek.com/modeling-cookbook/sdo.html#hermitian-matrices).

## Installation

**QICS** is currently supported for Python 3.8 or later, and can be directly installed from [pip](https://pypi.org/project/qics/) by calling

```bash
pip install qics
```

## Documentation

The full documentation of the code can be found [here](https://qics.readthedocs.io/en/stable/). Technical details about our implementation can be found in our paper.

## PICOS interface (coming soon)

The easiest way to use **QICS** is through the Python optimization modelling interface [PICOS](https://picos-api.gitlab.io/picos/). Below, we show how a simple [nearest 
correlation matrix](https://qics.readthedocs.io/en/stable/examples/qrep/nearest.html#nearest-correlation-matrix) problem can be solved. 

```python
  import numpy
  import picos

  # Define the conic program
  P = picos.Problem()
  X = numpy.array([[2., 1.], [1., 2.]])
  Y = picos.SymmetricVariable("Y", 2)
  
  P.set_objective("min", picos.qrelentr(X, Y))
  P.add_constraint(picos.maindiag(Y) == 1)

  # Solve the conic program
  P.solve(solver="qics")
```

Some additional details about how to use QICS with PICOS can be found [here](https://qics.readthedocs.io/en/stable/guide/gettingstarted.html).

## Native interface

Alternatively, advanced users can use the QICS' native interface, which provides additional flexibilty in how the problem is parsed to the solver. Below, we show how the same nearest correlation matrix problem can be solved using QICS' native interface.

```python
 import numpy
 import qics

 # Define the conic program
 c = numpy.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.]]).T
 A = numpy.array([
     [0., 1., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 1., 1., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 1., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 1., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 0., 0., 1.]
 ])
 b = numpy.array([[2., 2., 2., 1., 1.]]).T
 cones = [qics.cones.QuantRelEntr(2)]
 model = qics.Model(c=c, A=A, b=b, cones=cones)

 # Solve the conic program
 solver = qics.Solver(model)
 info = solver.solve()
```

Additional details describing this example can be found [here](https://qics.readthedocs.io/en/stable/guide/gettingstarted.html).
