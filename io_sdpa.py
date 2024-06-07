import cProfile

import numpy as np
import scipy as sp
from utils import symmetric as sym
from cones import *
from solver import model, solver

from utils.other_solvers import cvxopt_solve_sdp, mosek_solve_sdp, clarabel_solve_sdp
from utils import other_solvers

import sdpap


if __name__ == "__main__":
    import os, csv

    folder = "./problems/sdplib/"
    # Run all instances in folder
    fnames = os.listdir(folder)
    # Run single instance
    # fnames = ["qsd_50_10.dat-c"]

    fout_name = 'data.csv'
    with open(fout_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["problem", "solver", "status", "optval", "time", "iter", "gap", "pfeas", "dfeas"])

    for fname in fnames:
        # ==============================================================
        # Read problem data
        # ==============================================================
        C_sdpa, b_sdpa, A_sdpa, blockStruct = other_solvers.read_sdpa(folder + fname)
        
        # Vectorize C
        dims = []
        cones = []
        for bi in blockStruct:
            if bi >= 0:
                if fname[-1] == 'c':
                    cones.append(possemidefinite.Cone(bi, hermitian=True))
                    dims.append(2 * bi * bi)
                else:
                    cones.append(possemidefinite.Cone(bi))
                    dims.append(bi * bi)
            else:
                cones.append(nonnegorthant.Cone(-bi))
                dims.append(-bi)
        
        n = sum(dims)
        
        c = np.zeros((n, 1))
        b = b_sdpa.reshape((-1, 1))
        A = A_sdpa
        
        t = 0
        for (i, Ci) in enumerate(C_sdpa):
            if blockStruct[i] >= 0:
                c[t : t+dims[i]] = Ci.view(dtype=np.float64).reshape(-1, 1)
            else:
                c[t : t+dims[i], 0] = Ci
            t += dims[i]
        c *= -1

        # ==============================================================
        # Our algorithm
        # ==============================================================
        try:
        #     mdl = model.Model(c=-b, G=A.T, h=c, cones=cones)
            mdl = model.Model(c=c, A=A, b=b, cones=cones)
            slv = solver.Solver(mdl, sym=True, ir=True)

            profiler = cProfile.Profile()
            profiler.enable()
            slv.solve()
            profiler.disable()
            profiler.dump_stats("example.stats")

            with open(fout_name, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([fname, "ours", slv.solution_status, slv.p_obj, slv.solve_time, slv.iter, slv.gap, max(slv.y_feas, slv.z_feas), slv.x_feas])        
        except Exception as e:
            with open(fout_name, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([fname, "ours", e, None, None, None, None, None, None])

        # ==============================================================
        # CVXOPT
        # ==============================================================
        try:
            sol = cvxopt_solve_sdp(C_sdpa, b, A, blockStruct)

            with open(fout_name, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([fname, "cvxopt", sol['status'], sol['obj'], sol['time'], sol['iter'], sol['gap'], sol['pfeas'], sol['dfeas']])        

            print("optval: ", sol['gap']) 
            print("time:   ", sol['time'])   
        except Exception as e:
            with open(fout_name, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([fname, "cvxopt", e, None, None, None, None, None, None])        

        # ==============================================================
        # MOSEK
        # ==============================================================
        try:
            sol = mosek_solve_sdp(C_sdpa, b, A, blockStruct)

            with open(fout_name, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([fname, "mosek", sol['status'], sol['obj'], sol['time'], sol['iter'], sol['gap'], sol['pfeas'], sol['dfeas']])
        except Exception as e:
            with open(fout_name, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([fname, "mosek", e, None, None, None, None, None, None])

        # ==============================================================
        # SDPA
        # ==============================================================
        try:
            A, b, c, K, J = sdpap.fromsdpa(folder + fname)
            sdpap_options = {}
            sdpap_options['epsilonStar'] = 1e-8
            # sdpap_options['numThreads'] = 1
            x, y, timeinfo, sdpainfo = other_solvers.solve_sdpap(-A,-b,-c,K,J,)

            with open(fout_name, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    fname, 
                    "sdpa", 
                    sdpainfo['phasevalue'], 
                    sdpainfo['primalObj'], 
                    sdpainfo['sdpaTime'], 
                    sdpainfo['iteration'], 
                    sdpainfo['dualityGap'] / (1 + abs(sdpainfo['primalObj'])), 
                    sdpainfo['primalError'], 
                    sdpainfo['dualError']
                ])
        except Exception as e:
            with open(fout_name, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([fname, "cvxopt", e, None, None, None, None, None, None])