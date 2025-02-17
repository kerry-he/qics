# QICS: Quantum Information Conic Solver

[![Documentation Status](https://readthedocs.org/projects/qics/badge/?version=latest)](https://qics.readthedocs.io/en/latest/?badge=latest)
[![Build Status](http://github.com/kerry-he/qics/actions/workflows/ci.yml/badge.svg?event=push)](http://github.com/kerry-he/qics/actions/workflows/ci.yml)
[![PyPi Status](http://img.shields.io/pypi/v/qics.svg)](https://pypi.python.org/pypi/qics/)
[![AUR Status](https://img.shields.io/aur/version/python-qics)](https://aur.archlinux.org/packages/python-qics)

**QICS** is a primal-dual interior point solver fully implemented in Python, and
is specialized towards problems arising in quantum information theory. **QICS**
solves conic programs of the form

$$
\min_{x \in \mathbb{R}^n} \quad c^\top x \quad \text{s.t.} \quad b - Ax = 0, \  h - Gx \in \mathcal{K},
$$

where $c \in \mathbb{R}^n$, $b \in \mathbb{R}^p$, $h \in \mathbb{R}^q$,
$A \in \mathbb{R}^{p \times n}$, $G \in \mathbb{R}^{q \times n}$, and 
$\mathcal{K} \subset \mathbb{R}^{q}$ is a Cartesian product of convex cones.
Some notable cones that QICS supports include

| Cone           |  QICS class  |  Description  |
|----------------|:---------------------:|:---------------:|
| Positive semidefinite |  [`qics.cones.PosSemidefinite`](https://qics.readthedocs.io/en/stable/api/cones/PosSemidefinite.html#qics.cones.PosSemidefinite)  | $\\{ X \in \mathbb{H}^n : X \succeq 0 \\}$ |
| Quantum entropy |  [`qics.cones.QuantEntr`](https://qics.readthedocs.io/en/stable/api/cones/QuantEntr.html#qics.cones.QuantEntr)  | $\text{cl}\\{ (t, u, X) \in \mathbb{R} \times \mathbb{R}_{++} \times \mathbb{H}^n\_{++} : t \geq -u S(u^{-1} X) \\}$ |
| Quantum relative entropy |  [`qics.cones.QuantRelEntr`](https://qics.readthedocs.io/en/stable/api/cones/QuantRelEntr.html#qics.cones.QuantRelEntr)  | $\text{cl}\\{ (t, X, Y) \in \mathbb{R} \times \mathbb{H}^n_{++} \times \mathbb{H}^n_{++} : t \geq S(X \\| Y) \\}$ |
| Quantum conditional entropy |  [`qics.cones.QuantCondEntr`](https://qics.readthedocs.io/en/stable/api/cones/QuantCondEntr.html#qics.cones.QuantCondEntr)  | $\text{cl}\\{ (t, X) \in \mathbb{R} \times \mathbb{H}^{\Pi_in_i}_{++} : t \geq -S(X) + S(\text{tr}_i(X)) \\}$ |
| Quantum key distribution |  [`qics.cones.QuantKeyDist`](https://qics.readthedocs.io/en/stable/api/cones/QuantKeyDist.html#qics.cones.QuantKeyDist)  | $\text{cl}\\{ (t, X) \in \mathbb{R} \times \mathbb{H}^n_{++} : t \geq -S(\mathcal{G}(X)) + S(\mathcal{Z}(\mathcal{G}(X))) \\}$ |
| Operator perspective epigraph |  [`qics.cones.OpPerspecEpi`](https://qics.readthedocs.io/en/stable/api/cones/OpPerspecEpi.html#qics.cones.OpPerspecEpi)  | $\text{cl}\\{ (T, X, Y) \in \mathbb{H}^n \times \mathbb{H}^n_{++} \times \mathbb{H}^n_{++} : T \succeq P_g(X, Y) \\}$ |
| $\alpha$-Renyi entropy, for $\alpha\in[0,1)$ |  [`qics.cones.RenyiEntr`](https://qics.readthedocs.io/en/stable/api/cones/RenyiEntr.html#qics.cones.RenyiEntr)  | $\text{cl}\\{ (t, u, X, Y) \in \mathbb{R} \times \mathbb{R}\_{++} \times \mathbb{H}^n_{++} \times \mathbb{H}^n_{++} : t \geq u D_\alpha(u^{-1}X \| u^{-1}Y) \\}$ |
| Sandwiched $\alpha$-Renyi entropy, for $\alpha\in\[\frac{1}{2},1)$ |  [`qics.cones.SandRenyiEntr`](https://qics.readthedocs.io/en/stable/api/cones/SandRenyiEntr.html#qics.cones.SandRenyiEntr)  | $\text{cl}\\{ (t, u, X, Y) \in \mathbb{R} \times \mathbb{R}\_{++} \times\mathbb{H}^n_{++} \times \mathbb{H}^n_{++} : t \geq u \hat{D}_\alpha(u^{-1}X \| u^{-1}Y) \\}$ |
| $\alpha$-Quasi-relative entropy, for $\alpha\in[-1,2]$ |  [`qics.cones.QuasiEntr`](https://qics.readthedocs.io/en/stable/api/cones/QuasiEntr.html#qics.cones.QuasiEntr)  | $\text{cl} \\{ (t, X, Y) \in \mathbb{R} \times \mathbb{H}^n_{++} \times \mathbb{H}^n_{++} : t \geq \pm \text{tr}[ X^\alpha Y^{1-\alpha} ] \\}$ |
| Sandwiched $\alpha$-quasi-relative entropy, for $\alpha\in[\frac{1}{2},2]$ |  [`qics.cones.SandQuasiEntr`](https://qics.readthedocs.io/en/stable/api/cones/SandQuasiEntr.html#qics.cones.SandQuasiEntr)  | $\text{cl} \\{ (t, X, Y) \in \mathbb{R} \times \mathbb{H}^n_{++} \times \mathbb{H}^n_{++} : t \geq \pm \text{tr}[ ( Y^{\frac{1-\alpha}{2\alpha}} X Y^{\frac{1-\alpha}{2\alpha}} )^\alpha ] \\}$ |

where we define the following functions

  - Quantum entropy: $S(X)=-\text{tr}[X\log(X)]$
  - Quantum relative entropy: $S(X \| Y)=\text{tr}[X\log(X) - X\log(Y)]$
  - Noncommutative perspective: $P_g(X, Y)=X^{1/2} g(X^{-1/2} Y X^{-1/2}) X^{1/2}$
  - $\alpha$-Renyi entropy: $D_\alpha(X\|Y)=\frac{1}{1-\alpha} \log(\text{tr}[X^\alpha Y^{1-\alpha}])$
  - Sandwiched $\alpha$-Renyi entropy: $\hat{D}_\alpha(X \| Y) = \frac{1}{1-\alpha} \log(\text{tr}[ (Y^{\frac{1-\alpha}{2\alpha}} X Y^{\frac{1-\alpha}{2\alpha}})^\alpha ])$

A full list of cones which we support can be found in our
[documentation](https://qics.readthedocs.io/en/stable/guide/reference.html#cones).

## Features

- **Efficient quantum relative entropy programming**

  We support optimizing over the quantum relative entropy cone, as well as
  related cones including the quantum conditional entropy cone, and slices of
  the quantum relative entropy cone that arise when solving quantum key rates of
  quantum cryptographic protocols. Numerical results show that **QICS** solves
  problems much faster than existing quantum relative entropy programming
  solvers, such as [Hypatia](https://github.com/jump-dev/Hypatia.jl),
  [DDS](https://github.com/mehdi-karimi-math/DDS), and
  [CVXQUAD](https://github.com/hfawzi/cvxquad).

- **Efficient semidefinite programming**

  We implement an efficient semidefinite programming solver which utilizes
  state-of-the-art techniques for symmetric cone programming, including using
  Nesterov-Todd scalings and exploiting sparsity in the problem structure.
  Numerical results show that **QICS** has comparable performance to 
  state-of-the-art semidefinite programming software, such as 
  [MOSEK](https://www.mosek.com/), 
  [SDPA](https://sdpa.sourceforge.net/index.html), 
  [SDPT3](https://www.math.cmu.edu/~reha/sdpt3.html) and
  [SeDuMi](https://sedumi.ie.lehigh.edu/).

- **Complex-valued matrices**

  Users can specify whether cones involving variables which are symmetric
  matrices, such as the positive semidefinite cone or quantum relative entropy
  cone, involve real-valued or complex-valued (i.e., Hermitian) matrix
  variables. Support for Hermitian matrices is embedded directly in the
  definition of the cone, which can be more computationally efficient than
  [lifting into the real-valued symmetric cone](https://docs.mosek.com/modeling-cookbook/sdo.html#hermitian-matrices).

## Installation

**QICS** is currently supported for Python 3.8 or later, and can be directly
installed from [pip](https://pypi.org/project/qics/) by calling

```bash
pip install qics
```

Note that the performance of QICS is highly dependent on the version of BLAS and
LAPACK that [NumPy](https://numpy.org/doc/stable/building/blas_lapack.html) and 
[SciPy](https://docs.scipy.org/doc/scipy/building/blas_lapack.html) are linked to.

## Documentation

The full documentation of the code can be found
[here](https://qics.readthedocs.io/en/stable/). Technical details about our
implementation can be found in our [paper](http://arxiv.org/abs/2410.17803).

## PICOS interface

The easiest way to use **QICS** is through the Python optimization modelling
interface [PICOS](https://picos-api.gitlab.io/picos/), and can be installed using

```bash
pip install picos
```

Below, we show how a simple [nearest correlation matrix](https://qics.readthedocs.io/en/stable/examples/qrep/nearest.html#nearest-correlation-matrix) 
problem can be solved. 

```python
import picos

# Define the conic program
P = picos.Problem()
X = picos.Constant("X", [[2., 1.], [1., 2.]])
Y = picos.SymmetricVariable("Y", 2)

P.set_objective("min", picos.quantrelentr(X, Y))
P.add_constraint(picos.maindiag(Y) == 1)

# Solve the conic program
P.solve(solver="qics")
```

Some additional details about how to use QICS with PICOS can be found
[here](https://qics.readthedocs.io/en/stable/guide/picos.html).

## Native interface

Alternatively, advanced users can use the QICS' native interface, which provides
additional flexibilty in how the problem is parsed to the solver. Below, we
show how the same nearest correlation matrix problem can be solved using QICS'
native interface.

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

Additional details describing this example can be found
[here](https://qics.readthedocs.io/en/stable/guide/gettingstarted.html).

## Citing QICS

If you find our work useful, please cite our [paper](http://arxiv.org/abs/2410.17803)
using:

    @misc{he2024qics,
      title={{QICS}: {Q}uantum Information Conic Solver}, 
      author={Kerry He and James Saunderson and Hamza Fawzi},
      year={2024},
      eprint={2410.17803},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2410.17803}, 
    }

If you found our sandwiched Renyi and quasi-relative entropy cones useful, please cite
out [paper](https://www.arxiv.org/abs/2502.05627) using:

    @misc{he2025operator,
      title={Operator convexity along lines, self-concordance, and sandwiched {R}\'enyi entropies}, 
      author={Kerry He and James Saunderson and Hamza Fawzi},
      year={2025},
      eprint={2502.05627},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2502.05627}, 
    }
