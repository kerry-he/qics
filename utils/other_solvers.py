"""
    Various routines to solve primal-dual pair of SDPs:
        min  c'*x   s.t. A*x    == b,  x >= 0
        max  b'*y   s.t. A'*y+s == c,  s >= 0
    * cvxopt_solve_sdp: using CVXOPT with a custom kktsolver
    * mosek_solve_sdp: using Mosek
    * myipm: using basic implementation of PDIPM
"""

import mosek
from mosek.fusion import Model, Matrix, Domain, Expr, ObjectiveSense, ProblemStatus

import numpy as np

import scipy as sp
from scipy import sparse
from scipy.sparse import csc_matrix, coo_matrix
from scipy.linalg import cho_factor, cho_solve

import cvxopt
import time
import sys


def cvxopt_solve_sdp(C, b, A, blockStruct):
    # Solve problem
    #   min c'x s.t. Ax == b, x >= 0
    #   max b'y s.t. A'y <= c, y \in R^n
    # using CVXOPT
    # Two options for kktsolver: either "default" or "elim"
    #
    # Note: CVXOPT takes problems in the following general form (see https://cvxopt.org/userguide/coneprog.html#semidefinite-programming):
    #   min c'x s.t. A x == b, G1 x + s == h1, s >= 0
    #   max -h1'z s.t. G1'z + A'y + c == 0, z1 >= 0
    # with data (A,b,c,G1,h1)
    #
    # There are two ways to fit the standard SDP form data above (A0,b0,c0) to the CVXOPT form either by setting
    #   (A,b,c,G1,h1) = (0,0,-b0,A0',c0)  [in which case (x,y,z,s) <-> (y0,.,x0,.)]
    # or
    #   (A,b,c,G1,h1) = (A,b,c,-I,0) [in which case (x,y,z,s) <-> (x0,y0,.,.)]
    #
    # I am using the *first* one of these.

    A = sp.sparse.csr_matrix(A)
    p = b.size

    # Split A into SDP and LP components
    t = 0
    Gs = []
    hs = []
    Gl = None
    hl = None
    for (k, n) in enumerate(blockStruct):
        if n > 0:
            # SDP
            A_k = A[:, t:t+n*n].tocoo()
            Gs.append(cvxopt.spmatrix(A_k.data, A_k.col, A_k.row, (n*n, p), 'd'))
            hs.append(cvxopt.matrix(-C[k]))
            t += n*n
        else:
            # LP
            A_k = A[:, t:t-n].tocoo()
            Gl = cvxopt.spmatrix(A_k.data, A_k.col, A_k.row, (-n, p), 'd')
            hl = cvxopt.matrix(-C[k])

    # Solve with cvxopt
    # Fit the form max -<h1,z> s.t. G1'*z + c == 0, z >= 0
    t0_cvxopt = time.time()
    cvxoptsol = cvxopt.solvers.sdp(
                    c  = cvxopt.matrix(-b),
                    Gl = Gl,
                    hl = hl,
                    Gs = Gs,
                    hs = hs)
    cvxopt_time = time.time() - t0_cvxopt

    return {'primal': -cvxoptsol['dual objective'],
            'dual': -cvxoptsol['primal objective'],
            'x': np.array(cvxoptsol['zs'][0]).ravel(),
            'y': np.array(cvxoptsol['x']).ravel(),
            'status': cvxoptsol['status'],
            'time': cvxopt_time}


def mosek_solve_sdp(C, b, A, blockStruct):
    # Solve problem
    #   min c'x s.t. Ax == b, x >= 0
    #   max b'y s.t. A'y <= c, y \in R^n
    # using Mosek

    A = sp.sparse.csr_matrix(A)
    p = b.size

    # Split A into SDP and LP components
    t = 0
    msk_M = Model("mosek")

    msk_X  = []
    msk_A  = []
    msk_AX = []
    msk_CX = []

    for (k, n) in enumerate(blockStruct):
        if n > 0:
            # SDP
            A_k = A[:, t:t+n*n].tocoo()

            msk_X.append(msk_M.variable(Domain.inPSDCone(n)))
            msk_A.append(Matrix.sparse(p, n*n, list(A_k.row), list(A_k.col), list(A_k.data)))
            msk_AX.append(Expr.mul(msk_A[k], msk_X[k].reshape(n*n)))
            msk_CX.append(Expr.dot(C[k], msk_X[k]))

            t += n*n
        else:
            # LP
            A_k = A[:, t:t-n].tocoo()

            msk_X.append(msk_M.variable(Domain.greaterThan(0.0, -n)))
            msk_A.append(Matrix.sparse(p, -n, list(A_k.row), list(A_k.col), list(A_k.data)))
            msk_AX.append(Expr.mul(msk_A[k], msk_X[k].reshape(-n)))
            msk_CX.append(Expr.dot(C[k], msk_X[k].reshape(-n)))

    # Mosek MODEL
    msk_M.constraint(Expr.add(msk_AX), Domain.equalsTo(list(b.ravel())))
    msk_M.objective(ObjectiveSense.Maximize, Expr.add(msk_CX))
    msk_M.setSolverParam("numThreads", 1)
    msk_M.setLogHandler(sys.stdout)
    msk_t0 = time.perf_counter()
    msk_M.solve()
    msk_time = time.perf_counter() - msk_t0
    msk_status = msk_M.getProblemStatus()

    return {'primal': msk_M.primalObjValue(), 'dual': msk_M.dualObjValue(), 'X': msk_X.level(), 'time': msk_time, 'status': msk_status}