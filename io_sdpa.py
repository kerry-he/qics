import cProfile

import numpy as np
import scipy as sp
from utils import symmetric as sym
from cones import *
from solver import model, solver
import os

from utils.other_solvers import cvxopt_solve_sdp, mosek_solve_sdp, clarabel_solve_sdp

def read_sdpa(filename):
    fp = open(filename, "r")
    line = fp.readline()
    # Skip comments
    while line[0] == '*' or line[0] == '"':
        line = fp.readline()
        
    # Read mDim
    mDim = int(line.strip().split(' ')[0])

    # Read nBlock
    line = fp.readline()
    nBlock = int(line.strip().split(' ')[0])
    
    # Read blockStruct
    line = fp.readline()
    blockStruct = [int(i) for i in line.strip().split(' ')]
    
    # Read b
    line = fp.readline()
    line = line.strip()
    line = line.strip('{}()')
    if ',' in line:
        b_str = line.strip().split(',')
    else:
        b_str = line.strip().split()
    while b_str.count('') > 0:
        b_str.remove('')
    b = np.array([float(bi) for bi in b_str])
    
    # Read C and A
    C = []
    for bi in blockStruct:
        if bi >= 0:
            C.append(np.zeros((bi, bi)))
        else:
            C.append(np.zeros(-bi))
            
    totDim = 0
    idxs = [0]
    totDim_compressed = 0
    idxs_compressed = [0]
    for n in blockStruct:
        if n >= 0:
            totDim += n*n
            idxs.append(idxs[-1] + n*n)

            totDim_compressed += n*(n+1)//2
            idxs_compressed.append(idxs_compressed[-1] + n*(n+1)//2)
        else:
            totDim -= n
            idxs.append(idxs[-1] - n)

            totDim_compressed -= n
            idxs_compressed.append(idxs_compressed[-1] - n)

    C_compresed = np.zeros(totDim_compressed)

    Acols = []
    Arows = []
    Avals = []

    Acols_compressed = []
    Arows_compressed = []
    Avals_compressed = []

    lineList = fp.readlines()
    for line in lineList:
        row, block, colI, colJ, val = line.split()[0:5]
        row = int(row.strip(',')) - 1
        block = int(block.strip(',')) - 1
        colI = int(colI.strip(',')) - 1
        colJ = int(colJ.strip(',')) - 1
        val = float(val.strip(','))
        
        if val == 0:
            continue
        
        if row == -1:
            if blockStruct[block] >= 0:
                C[block][colI, colJ] = val
                C[block][colJ, colI] = val
            else:
                assert colI == colJ
                C[block][colI] = val
        else:
            if blockStruct[block] >= 0:
                Acols.append(idxs[block] + colI + colJ*blockStruct[block])
                Arows.append(row)
                Avals.append(val)

                if colJ != colI:
                    Acols.append(idxs[block] + colJ + colI*blockStruct[block])
                    Arows.append(row)
                    Avals.append(val)
            else:
                assert colI == colJ
                Acols.append(idxs[block] + colI)
                Arows.append(row)
                Avals.append(val)

    A = sp.sparse.csr_matrix((Avals, (Arows, Acols)), shape=(mDim, totDim))
            
    return C, b, A, blockStruct


if __name__ == "__main__":
    # fnames = os.listdir("./problems/sdp/")
    fnames = ["control10.dat-s"]

    for fname in fnames:
        C_sdpa, b_sdpa, A_sdpa, blockStruct = read_sdpa("./problems/sdp/" + fname)
        
        # Vectorize C
        dims = []
        cones = []
        for bi in blockStruct:
            if bi >= 0:
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
                c[t : t+dims[i]] = Ci.reshape((-1, 1))
            else:
                c[t : t+dims[i], 0] = Ci
            t += dims[i]
        c *= -1
                
        # mdl = model.Model(c=-b, G=A.T, h=c, cones=cones)
        mdl = model.Model(c=c, A=A, b=b, cones=cones)
        slv = solver.Solver(mdl, sym=True, ir=True)

        # profiler = cProfile.Profile()
        # profiler.enable()

        slv.solve()

        # profiler.disable()
        # profiler.dump_stats("example.stats")    

        # profiler = cProfile.Profile()
        # profiler.enable()

        # sol = cvxopt_solve_sdp(C_sdpa, b, A, blockStruct)

        # profiler.disable()
        # profiler.dump_stats("example1.stats")        

        # print("optval: ", sol['primal']) 
        # print("time:   ", sol['time'])   

        # sol = mosek_solve_sdp(C_sdpa, b, A, blockStruct)

        # sol = clarabel_solve_sdp(C_sdpa, b, A, blockStruct)