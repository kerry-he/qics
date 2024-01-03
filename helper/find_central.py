from cones import *
import numpy as np

n = 3
K = classentr.ClassEntropy(n)

x = K.set_init_point()

while True:
    assert K.get_feas()
    g = K.get_grad() + x
    H = K.hess_prod(np.eye(K.dim)) + np.eye(K.dim)

    delta_x = -np.linalg.solve(H, g)
    decrement = -delta_x.T @ g

    print("Decrement: ", decrement[0, 0])

    if decrement / 2. <= 1e-12:
        break

    x += delta_x
    K.set_point(x)

np.printoptions(precision=15, suppress=True)
print("%.15f" % (x[0, 0]))
print("%.15f" % (x[1, 0]))