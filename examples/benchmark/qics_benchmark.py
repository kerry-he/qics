import csv
import os

import qics

folder = "./qreps/"
fnames = os.listdir(folder)
# fnames = [fnames[0]] + fnames
# fnames = ["qrd_sr_64_5.cbf"]

fout_name = "data_qics.csv"
with open(fout_name, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "problem",
            "solver",
            "status",
            "optval",
            "time",
            "iter",
            "gap",
            "pfeas",
            "dfeas",
        ]
    )

for fname in fnames:
    try:
        model = qics.io.read_cbf(folder + fname)

        # Initialize solver objects
        solver = qics.Solver(model, max_iter=500)

        # Solve problem
        info = solver.solve()
        with open(fout_name, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    fname,
                    "ours",
                    info["sol_status"],
                    info["p_obj"],
                    info["solve_time"],
                    info["num_iter"],
                    info["opt_gap"],
                    info["p_feas"],
                    info["d_feas"],
                ]
            )

    except Exception as e:
        with open(fout_name, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([fname, "ours", e, None, None, None, None, None, None])
