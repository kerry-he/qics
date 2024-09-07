import numpy as np
import qics

n = 2
p1 = p2 = 0.5

E11 = np.array([[1., 0.], [0., 0.]])
E12 = np.array([[0., .5], [.5, 0.]])
E22 = np.array([[0., 0.], [0., 1.]])

# Define objective function
c = -np.vstack((p1 * E11.reshape(-1, 1), p2 * E22.reshape(-1, 1)))

# Build linear constraints
A = np.vstack((
    np.hstack((E11.reshape(1, -1), E11.reshape(1, -1))),
    np.hstack((E12.reshape(1, -1), E12.reshape(1, -1))),
    np.hstack((E22.reshape(1, -1), E22.reshape(1, -1)))
))
b = np.array([[1.], [0.], [1.]])

# Define cones to optimize over
cones = [
    qics.cones.PosSemidefinite(n), 
    qics.cones.PosSemidefinite(n)
]

# Initialize model and solver objects
model  = qics.Model(c=c, A=A, b=b, cones=cones)
solver = qics.Solver(model)

# Solve problem
info = solver.solve()

print("Optimal POVMs are")
print(info["s_opt"][0])
print("and")
print(info["s_opt"][1])