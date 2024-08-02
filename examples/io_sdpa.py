import os
import qics.utils.io

# To be used with SDP problems stored in the SDPA sparse format
# e.g., https://github.com/vsdp/SDPLIB
folder = "./problems/sdplib/"
fnames = os.listdir(folder)

for fname in fnames:
    c, b, A, cones = qics.utils.io.read_sdpa(folder + fname)

    # Initialize model and solver objects
    model  = qics.Model(c=c, A=A, b=b, cones=cones)
    solver = qics.Solver(model)

    # Solve problem
    out = solver.solve()