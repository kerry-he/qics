import os
import qics.io

# To be used with SDP problems stored in the SDPA sparse format
# e.g., https://github.com/vsdp/SDPLIB
folder = "./problems/sdplib/"
fnames = os.listdir(folder)

for fname in fnames:
    # Read file
    model = qics.io.read_sdpa(folder + fname)

    # Initialize solver objects
    solver = qics.Solver(model)

    # Solve problem
    info = solver.solve()