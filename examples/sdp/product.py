import numpy as np

import qics

# Define objective function
c = np.array([[1.1, -10, 6.6, 19, 4.1]]).T

# Define linear constraints. Note that we have defined pre-vectorized 
# the matrices for convenience
F = [
    [   # F0
        np.array([[-1.4, -3.2, -3.2, -28]]).T,
        np.array([[15, -12, 2.1, -12, 16, -3.8, 2.1, -3.8, 15]]).T,
        np.array([[1.8, -4.0]]).T
    ],
    [   # F1
        np.array([[0.5, 5.2, 5.2, -5.3]]).T,
        np.array([[7.8, -2.4, 6.0, -2.4, 4.2, 6.5, 6.0, 6.5, 2.1]]).T,
        np.array([[-4.5, -3.5]]).T
    ],
    [   #F2
        np.array([[1.7, 7.0, 7.0, -9.3]]).T,
        np.array([[-1.9, -0.9, -1.3, -0.9, -0.8, -2.1, -1.3, -2.1, 4.0]]).T,
        np.array([[-0.2, -3.7]]).T
    ],
    [   #F3
        np.array([[6.3, -7.5, -7.5, -3.3]]).T,
        np.array([[0.2, 8.8, 5.4, 8.8, 3.4, -0.4, 5.4, -0.4, 7.5]]).T,
        np.array([[-3.3, -4.0]]).T
    ],
    [   #F4
        np.array([[-2.4, -2.5, -2.5, -2.9]]).T,
        np.array([[3.4, -3.2, -4.5, -3.2, 3.0, -4.8, -4.5, -4.8, 3.6]]).T,
        np.array([[4.8, 9.7]]).T
    ],
    [   #F5
        np.array([[-6.5, -5.4, -5.4, -6.6]]).T,
        np.array([[6.7, -7.2, -3.6, -7.2, 7.3, -3.0, -3.6, -3.0, -1.4]]).T,
        np.array([[6.1, -1.5]]).T
    ]
]

h = -np.vstack(F[0])
G = -np.hstack([np.vstack(Fi) for Fi in F[1:]])

# Define cones to optimize over
cones = [
    qics.cones.PosSemidefinite(2),
    qics.cones.PosSemidefinite(3),
    qics.cones.NonNegOrthant(2),
]

# Initialize model and solver objects
model  = qics.Model(c=c, G=G, h=h, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()