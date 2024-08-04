def test_ptr_ikr():
    # Tests that p_tr and i_kr satisfy the adjoint relationship
    #     <p_tr(X), Y> = <X, i_kr(Y)>
    import numpy as np
    from qics.utils.symmetric import p_tr, i_kr

    np.random.seed(1)

    dims = [2, 4, 3, 5]
    
    x_dim = np.prod(dims)
    X = np.random.randn(x_dim, x_dim) + np.random.randn(x_dim, x_dim)*1j
    X = X + X.conj().T

    for (sys, dim) in enumerate(dims):
        y_dim = x_dim // dim
        Y = np.random.randn(y_dim, y_dim) + np.random.randn(y_dim, y_dim)*1j
        Y = Y + Y.conj().T

        assert np.allclose(
            np.trace(p_tr(X, dims, sys) @ Y), 
            np.trace(X @ i_kr(Y, dims, sys))
        ), "qics.utils.symmetric.p_tr and qics.utils.symmetric.i_kr are not adjoint linear operators"
        
def test_ptr_ikr_multi():
    # Tests that p_tr_multi and i_kr_multi satisfy the adjoint relationship
    #     <p_tr_multi(Xs), Ys> = <Xs, i_kr_multi(Ys)>    
    import numpy as np
    from qics.utils.symmetric import p_tr_multi, i_kr_multi

    np.random.seed(1)

    dims = [2, 4, 3, 5]

    x_dim = np.prod(dims)
    Xs = np.random.randn(5, x_dim, x_dim) + np.random.randn(5, x_dim, x_dim)*1j
    Xs = Xs + Xs.conj().transpose(0, 2, 1)

    for (sys, dim) in enumerate(dims):
        y_dim = x_dim // dim
        Ys = np.random.randn(5, y_dim, y_dim) + np.random.randn(5, y_dim, y_dim)*1j
        Ys = Ys + Ys.conj().transpose(0, 2, 1)

        assert np.allclose(
            np.trace(p_tr_multi(np.zeros_like(Ys), Xs, dims, sys) @ Ys, axis1=1, axis2=2), 
            np.trace(Xs @ i_kr_multi(np.zeros_like(Xs), Ys, dims, sys), axis1=1, axis2=2)
        ), "qics.utils.symmetric.p_tr_multi and qics.utils.symmetric.i_kr_multi are not adjoint linear operators"