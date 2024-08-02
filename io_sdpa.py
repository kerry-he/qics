import numpy as np
import qics.utils.io


if __name__ == "__main__":
    import os

    folder = "./problems/sdplib/"
    fnames = os.listdir(folder)

    for fname in fnames:
        c, b, A, cones = qics.utils.io.read_sdpa(folder + fname)

        # Initialize model and solver objects
        model  = qics.Model(c=c, A=A, b=b, cones=cones)
        solver = qics.Solver(model)

        # Solve problem
        out = solver.solve()