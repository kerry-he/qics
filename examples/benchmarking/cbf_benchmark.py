import os
import qics.io

model = qics.io.read_cbf("test.cbf")

# Initialize solver objects
solver = qics.Solver(model)

# Solve problem
info = solver.solve()