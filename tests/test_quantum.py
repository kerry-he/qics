# Copyright (c) 2024, Kerry He, James Saunderson, and Hamza Fawzi
# Based on test_examples.py from CVXOPT by M. Andersen and L. Vandenberghe.

# This Python package QICS is licensed under the MIT license; see LICENSE.md
# file in the root directory or at https://github.com/kerry-he/qics

def test_ptr_ikr():
    # Tests that p_tr and i_kr satisfy the adjoint relationship
    #     <p_tr(X), Y> = <X, i_kr(Y)>
    import numpy as np

    from qics.quantum import i_kr, p_tr

    np.random.seed(1)

    dims = [2, 4, 3, 5]

    x_dim = np.prod(dims)
    X = np.random.randn(x_dim, x_dim) + np.random.randn(x_dim, x_dim) * 1j
    X = X + X.conj().T

    sys_tests = [
        0,
        1,
        2,
        3,
        (0, 1),
        (2, 0),
        (0, 3),
        (2, 1),
        (1, 3),
        (3, 2),
        (0, 1, 2),
        (3, 0, 1),
        (0, 3, 2),
        (3, 2, 1),
        (0, 1, 2, 3),
    ]

    for sys in sys_tests:
        dim = dims[sys] if isinstance(sys, int) else np.prod([dims[k] for k in sys])
        y_dim = x_dim // dim
        Y = np.random.randn(y_dim, y_dim) + np.random.randn(y_dim, y_dim) * 1j
        Y = Y + Y.conj().T

        assert np.allclose(
            np.trace(p_tr(X, dims, sys) @ Y), np.trace(X @ i_kr(Y, dims, sys))
        ), "qics.quantum.p_tr and qics.quantum.i_kr are not adjoint linear operators"


def test_ptr_ikr_multi():
    # Tests that p_tr_multi and i_kr_multi satisfy the adjoint relationship
    #     <p_tr_multi(Xs), Ys> = <Xs, i_kr_multi(Ys)>
    import numpy as np

    from qics.quantum import i_kr_multi, p_tr_multi

    np.random.seed(1)

    dims = [2, 4, 3, 5]

    x_dim = np.prod(dims)
    Xs = np.random.randn(5, x_dim, x_dim) + np.random.randn(5, x_dim, x_dim) * 1j
    Xs = Xs + Xs.conj().transpose(0, 2, 1)

    sys_tests = [
        0,
        1,
        2,
        3,
        (0, 1),
        (2, 0),
        (0, 3),
        (2, 1),
        (1, 3),
        (3, 2),
        (0, 1, 2),
        (3, 0, 1),
        (0, 3, 2),
        (3, 2, 1),
        (0, 1, 2, 3),
    ]

    for sys in sys_tests:
        dim = dims[sys] if isinstance(sys, int) else np.prod([dims[k] for k in sys])
        y_dim = x_dim // dim
        Ys = np.random.randn(5, y_dim, y_dim) + np.random.randn(5, y_dim, y_dim) * 1j
        Ys = Ys + Ys.conj().transpose(0, 2, 1)

        assert np.allclose(
            np.trace(
                p_tr_multi(np.ones_like(Ys), Xs, dims, sys) @ Ys, axis1=1, axis2=2
            ),
            np.trace(
                Xs @ i_kr_multi(np.ones_like(Xs), Ys, dims, sys), axis1=1, axis2=2
            ),
        ), "qics.quantum.p_tr_multi and qics.quantum.i_kr_multi are not adjoint linear operators"
