import qics
import numpy as np
import math

def inp(x, y):
    return sum([np.sum(xi * yi.conj()) for (xi, yi) in zip(x, y)]).real

def vec(x):
    return np.hstack([xk.view(np.float64).ravel() for xk in x])

def rand(dims, types):
    def rand_single(dim, type):
        if type == 'r':
            return np.random.randn(dim, 1)
        elif type == 's':
            n = math.isqrt(dim)
            X = np.random.randn(n, n)
            return X + X.T
        elif type == 'h':
            n = math.isqrt(dim // 2)
            X = np.random.randn(n, n) + np.random.randn(n, n)*1j
            return X + X.conj().T
    return [rand_single(dim, type) for (dim, type) in zip(dims, types)]

def pos(dims, types):
    def rand_single(dim, type):
        if type == 'r':
            return np.abs(np.random.randn(dim, 1))
        elif type == 's':
            n = math.isqrt(dim)
            X = np.random.randn(n, n)
            return X @ X.T
        elif type == 'h':
            n = math.isqrt(dim // 2)
            X = np.random.randn(n, n) + np.random.randn(n, n)*1j
            return X @ X.conj().T
    return [rand_single(dim, type) for (dim, type) in zip(dims, types)]
    
def zeros(dims, types):
    def zeros_single(dim, type):
        if type == 'r':
            return np.zeros((dim, 1))
        elif type == 's':
            n = math.isqrt(dim)
            return np.zeros((n, n))
        elif type == 'h':
            n = math.isqrt(dim // 2)
            return np.zeros((n, n), dtype=np.complex128)
    return [zeros_single(dim, type) for (dim, type) in zip(dims, types)]

def isclose(x, y, tol=1e-3):
    if isinstance(x, np.ndarray):
        return np.linalg.norm(x - y, ord=np.inf) / (1 + np.linalg.norm(x, ord=np.inf)) <= tol
    else:
        return np.abs(x - y) / (1 + np.abs(x)) <= tol


def _test_cone(K):
    # Initialize at random feasible point
    while True:
        x0 = pos(K.dim, K.type)
        K.set_point(x0, x0)

        if K.get_feas():
            break

    # Get random direction
    H = rand(K.dim, K.type)

    # Compute gradient, Hessian, and TOA at x0
    eps = 1e-8
    f0 = K.get_val()
    g0 = zeros(K.dim, K.type)
    K.grad_ip(g0)
    H0 = zeros(K.dim, K.type)
    K.hess_prod_ip(H0, H)
    T0 = zeros(K.dim, K.type)
    K.third_dir_deriv_axpy(T0, H)

    # Get finite differenced gradient
    f1   = zeros(K.dim, K.type)
    g_fd = zeros(K.dim, K.type)
    for k in range(len(f1)):
        if K.type[k] == 'r':
            for i in range(K.dim[k]):
                x1 = [xk.copy() for xk in x0]
                x1[k][i] += eps
                K.set_point(x1, x1)
                K.get_feas()
                f1[k][i] = K.get_val()
        elif K.type[k] == 's' or K.type[k] == 'h':
            n = math.isqrt(K.dim[k]) if K.type[k] == 's' else math.isqrt(K.dim[k] // 2)
            for j in range(n):
                for i in range(j + 1):
                    x1 = [xk.copy() for xk in x0]
                    x1[k][i, j] += eps * 0.5
                    x1[k][j, i] += eps * 0.5
                    K.set_point(x1, x1)
                    K.get_feas()
                    f1[k][i, j] = K.get_val()
                    f1[k][j, i] = K.get_val()

                    if K.type[k] == 'h':
                        x1 = [xk.copy() for xk in x0]
                        x1[k][i, j] += eps * 0.5j
                        x1[k][j, i] -= eps * 0.5j
                        K.set_point(x1, x1)
                        K.get_feas()
                        f1[k][i, j] += K.get_val() * 1j
                        f1[k][j, i] -= K.get_val() * 1j
        
        if K.type[k] == 'r' or K.type[k] == 's':
            g_fd[k] = (f1[k] - f0) / eps
        elif K.type[k] == 'h':
            mask = np.ones_like(g_fd[k])
            mask[np.triu_indices(mask.shape[0], k=1)] += 1j
            mask[np.tril_indices(mask.shape[0], k=-1)] -= 1j
            g_fd[k] = (f1[k] - mask * f0) / eps

    # Compute gradient, Hessian, and TOA at x0
    x1 = [xk + eps*Hk for (xk, Hk) in zip(x0, H)]
    K.set_point(x1, x1)
    K.get_feas()
    g1 = zeros(K.dim, K.type)
    K.grad_ip(g1)
    H1 = zeros(K.dim, K.type)
    K.hess_prod_ip(H1, H)
    T1 = zeros(K.dim, K.type)
    K.third_dir_deriv_axpy(T1, H)

    # Gradient checks
    assert isclose(
        (vec(g0) + vec(g1)) / 2, vec(g_fd)
    ), f"{K} gradient oracle does not match finite differencing approximation"
    assert isclose(
        inp(g0, x0), -K.nu
    ), f"{K} gradient oracle does not satisfy <g(x), x> = -nu identity"    

    # Hessian checks
    work = zeros(K.dim, K.type)
    K.hess_prod_ip(work, x0)
    assert isclose(
        vec(work), -vec(g0)
    ), f"{K} Hessian oracle does not satisfy H(x)x = -g(x) identity"
    assert isclose(
        inp(K.hess_prod_ip(work, x0), x0), K.nu
    ), f"{K} Hessian oracle does not satisfy <H(x)x, x> = nu identity"    
    assert isclose(
        (vec(H0) + vec(H1)) / 2, (vec(g1) - vec(g0)) / eps
    ), f"{K} Hessian oracle does not match finite differencing approximation"        

    # Inverse Hessian checks
    K.invhess_prod_ip(work, H1)
    assert isclose(
        vec(H), vec(work)
    ), f"{K} inverse Hessian oracle does not satisfy H^-1(H(h)) = h identity"
    assert isclose(
        inp(K.invhess_prod_ip(work, g0), g0), K.nu
    ), f"{K} inverse Hessian oracle does not satisfy <H^-1(g(x)), g(x)> = nu identity"

    # Third order derivative checks
    assert isclose(
        (vec(T0) + vec(T1)) / 2, (vec(H1) - vec(H0)) / eps
    ), f"{K} third order derivative oracle does not match finite differencing approximation"

def test_cone_oracles():
    np.random.seed(1)

    _test_cone(qics.cones.NonNegOrthant(3))
    _test_cone(qics.cones.SecondOrder(3))
    _test_cone(qics.cones.PosSemidefinite(3))
    _test_cone(qics.cones.PosSemidefinite(3, True))

    _test_cone(qics.cones.ClassEntr(3))
    _test_cone(qics.cones.ClassRelEntr(3))

    _test_cone(qics.cones.QuantEntr(3))
    _test_cone(qics.cones.QuantEntr(3, True))

    _test_cone(qics.cones.QuantRelEntr(3))
    _test_cone(qics.cones.QuantRelEntr(3, True))

    _test_cone(qics.cones.QuantCondEntr((2, 2, 2, 2), (1, 3)))
    _test_cone(qics.cones.QuantCondEntr((2, 2, 2, 2), (1, 3), True))

    K_list = [np.random.randn(4, 2), np.random.randn(4, 2)]
    _test_cone(qics.cones.QuantKeyDist(K_list, 2))
    _test_cone(qics.cones.QuantKeyDist(K_list, 2, True))    

    _test_cone(qics.cones.OpPerspecTr(3, "log"))
    _test_cone(qics.cones.OpPerspecTr(3, "log", True))
    _test_cone(qics.cones.OpPerspecTr(3, 0.3))
    _test_cone(qics.cones.OpPerspecTr(3, 0.3, True))

    _test_cone(qics.cones.OpPerspecEpi(3, "log"))
    _test_cone(qics.cones.OpPerspecEpi(3, "log", True))
    _test_cone(qics.cones.OpPerspecEpi(3, 0.3))
    _test_cone(qics.cones.OpPerspecEpi(3, 0.3, True))