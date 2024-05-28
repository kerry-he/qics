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

import cvxpy
import clarabel

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

    return {
        'obj': -cvxoptsol['dual objective'],
        'status': cvxoptsol['status'],
        'time': cvxopt_time,
        'iter': cvxoptsol['iterations'],
        'gap': cvxoptsol['relative gap'],
        'dfeas': cvxoptsol['dual infeasibility'],
        'pfeas': cvxoptsol['primal infeasibility']
    }


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
    msk_M.solve()
    msk_status = msk_M.getProblemStatus()

    return {
        'obj': msk_M.getSolverDoubleInfo("intpntPrimalObj"), 
        'time': msk_M.getSolverDoubleInfo("intpntTime"), 
        'status': msk_status,
        'iter': msk_M.getSolverIntInfo("intpntIter"),
        'gap': msk_M.getSolverDoubleInfo("intpntPrimalObj") - msk_M.getSolverDoubleInfo("intpntDualObj"),
        'dfeas': msk_M.getSolverDoubleInfo("intpntDualFeas"),
        'pfeas': msk_M.getSolverDoubleInfo("intpntPrimalFeas")
    }

def clarabel_solve_sdp(c, b, A, blockStruct):
    import cvxpy as cp

    (p, n) = A.shape

    # Create optimization variables
    xs = []
    constraints = []
    c_x = []
    A_x = []
    t = 0
    for (k, bi) in enumerate(blockStruct):
        if bi >= 0:
            x_k = cp.Variable((bi, bi), symmetric=True)
            xs.append(cp.vec(x_k))
            constraints += [x_k >> 0]

            c_x.append(cp.sum(cp.multiply(c[k], x_k)))

            A_k = A[:, t:t+bi*bi].tocoo()
            A_x.append(A_k @ xs[k])

            t += bi*bi
        else:
            x_k = cp.Variable(-bi)
            xs.append(x_k)
            constraints += [x_k >= 0]

            c_x.append(cp.sum(cp.multiply(c[k], x_k)))

            A_k = A[:, t:t-bi].tocoo()
            A_x.append(A_k @ xs[k])

    constraints += [sum(A_x) == b.ravel()]     

    # Form objective.
    obj = cp.Maximize(sum(c_x))

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)
    prob.solve(solver='CLARABEL', verbose=True)


    # # Split A into SDP and LP components
    # (p, n) = A.shape

    # # Clarabel constraint data
    # P = sp.sparse.csc_matrix((n, n))

    # A = sparse.vstack([A, sp.sparse.csr_matrix(np.eye(n))]).tocsc()
    # b = np.concatenate([b.reshape(-1), np.zeros(n)])

    # cones = [clarabel.ZeroConeT(p)]
    # for bi in blockStruct:
    #     if bi >= 0:
    #         cones.append(clarabel.PSDTriangleConeT(bi))
    #     else:
    #         cones.append(clarabel.NonnegativeConeT(-bi))

    # settings = clarabel.DefaultSettings()

    # solver = clarabel.DefaultSolver(P, c, A, b, cones, settings)

    # solution = solver.solve()
    # solution.x  # primal solution
    # solution.z  # dual solution
    # solution.s  # primal slacks

    # return 



