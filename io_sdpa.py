import cProfile

import numpy as np
import scipy as sp
from utils import symmetric as sym
from cones import *
from solver import model, solver

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
    
    # Read C
    C = []
    for bi in blockStruct:
        if bi >= 0:
            C.append(np.zeros((bi, bi)))
        else:
            C.append(np.zeros(-bi))
            
    A = [[] for ri in range(mDim)]
    for Ai in A:
        for bi in blockStruct:
            if bi >= 0:
                Ai.append(np.zeros((bi, bi)))
            else:
                Ai.append(np.zeros(-bi))
    
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
                A[row][block][colI, colJ] = val
                A[row][block][colJ, colI] = val
            else:
                assert colI == colJ
                A[row][block][colI] = val
            
    return C, b, A, blockStruct


if __name__ == "__main__":
    C_sdpa, b_sdpa, A_sdpa, blockStruct = read_sdpa("/home/kerry/qce-ipm/problems/sdp/arch0.dat-s")
    
    # Vectorize C
    dims = []
    cones = []
    for bi in blockStruct:
        if bi >= 0:
            cones.append(possemidefinite.Cone(bi))
            dims.append(bi * bi)
            # dims.append(bi * (bi + 1) // 2)
        else:
            cones.append(nonnegorthant.Cone(-bi))
            dims.append(-bi)
            
    n = sum(dims)
    p = len(A_sdpa)
    
    c = np.zeros((n, 1))
    b = b_sdpa.reshape((-1, 1))
    A = np.zeros((p, n))
    
    t = 0
    for (i, Ci) in enumerate(C_sdpa):
        if blockStruct[i] >= 0:
            c[t : t+dims[i]] = Ci.reshape((-1, 1))
            # c[t : t+dims[i]] = sym.mat_to_vec(Ci)
        else:
            c[t : t+dims[i], 0] = Ci
        t += dims[i]
    c *= -1
            
    for (j, Aj) in enumerate(A_sdpa):
        t = 0
        for (i, Aji) in enumerate(Aj):
            if blockStruct[i] >= 0:
                A[j, t : t+dims[i]] = Aji.flat
                # A[[j], t : t+dims[i]] = sym.mat_to_vec(Aji).T
                # print("Rank: ", np.linalg.matrix_rank(Aji), "  nnz: ", np.count_nonzero(Aji))
            else:
                A[j, t : t+dims[i]] = Aji
            t += dims[i]
    A = sp.sparse.coo_array(A)
            
    model = model.Model(c, A, b, cones=cones)
    solver = solver.Solver(model, sym=True, ir=True)

    profiler = cProfile.Profile()
    profiler.enable()

    solver.solve()

    profiler.disable()
    profiler.dump_stats("example.stats")