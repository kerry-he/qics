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
    Gl = None
    for (k, n) in enumerate(blockStruct):
        if n > 0:
            # SDP
            A_k = A[:, t:t+n*n].tocoo()
            Gs.append(cvxopt.spmatrix(A_k.data, A_k.col, A_k.row, (n*n, p), 'd'))
            hs.append(cvxopt.matrix(C[k]))
            t += n*n
        else:
            # LP
            A_k = A[:, t:t-n].tocoo()
            Gl = cvxopt.spmatrix(A_k.data, A_k.col, A_k.row, (-n, p), 'd')
            hl = cvxopt.matrix(C[k])

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


def mosek_solve_sdp(A,b,c):
    # Solve problem
    #   min c'x s.t. Ax == b, x >= 0
    #   max b'y s.t. A'y <= c, y \in R^n
    # using Mosek

    assert type(A) == coo_matrix

    k = int(np.sqrt(A.shape[1]))
    nvars,totDim = A.shape
    rowsA = A.row
    colsA = A.col
    dataA = A.data

    # Mosek MODEL
    msk_M = Model("testrandom")
    msk_A = Matrix.sparse(nvars,totDim,list(rowsA),list(colsA),list(dataA))
    msk_X = msk_M.variable(Domain.inPSDCone(k))
    msk_x = msk_X.reshape(k**2)
    msk_M.constraint(Expr.mul(msk_A,msk_x),Domain.equalsTo(list(b)))
    msk_M.objective(ObjectiveSense.Minimize,Expr.dot(c,msk_x))
    msk_M.setSolverParam("numThreads", 1)
    msk_M.setLogHandler(sys.stdout)
    msk_t0 = time.perf_counter()
    msk_M.solve()
    msk_time = time.perf_counter() - msk_t0
    msk_status = msk_M.getProblemStatus()

    return {'primal': msk_M.primalObjValue(), 'dual': msk_M.dualObjValue(), 'X': msk_X.level(), 'time': msk_time, 'status': msk_status}

def line_search(X,dX):
    """
    Finds largest alpha \in [0,1] s.t. X+alpha*dX > 0. Assumes X > 0 (posdef)
    Uses bisection with calls to chol to detect positivity
    """
    lb, ub = 0.0, 1.0
    while ub-lb > .1:
        alpha = (lb+ub)/2
        try:
            np.linalg.cholesky(X+alpha*dX)
            lb = alpha
        except:
            ub = alpha
    return lb

def sdp_ipm(k,A1,b1,c1,x0=None,y0=None,s0=None):

    # Solve a primal-dual pair of SDPs of the form
    #    min  c'*x   s.t. A*x    == b,  x >= 0
    #    max  b'*y   s.t. A'*y+s == c,  s >= 0
    # using interior-point method
    # 
    # k is the size of the psd cone, let N=k^2
    # A is a nxN sparse matrix in csr format
    # b is a (dense) vector of length n
    # c is a (dense) vector of length N
    #
    # NOTE: A kxk symmetric matrix is represented as an array of length k^2
    # We assume that the rows of A, as well as the vector c, are valid vectorizations of symmetric matrices

    # Compute n and N
    N = k**2
    n = A1.shape[0]

    assert A1.shape == (n,N) and b1.shape == (n,) and c1.shape == (N,)

    A = A1.copy()
    b = b1.copy()
    c = c1.copy()

    scm_form = scm_gen(A)

    print("Constraints: ", n)
    print("Scalarized:  ", N)

    AT = A.transpose()

    def vec2mat(v): return np.reshape(v,(k,k))
    def mat2vec(V): return np.ravel(V)
    def symmetrize(H): return .5*(H+H.T)

    # Check that c and rows of A represent valid symmetric matrices
    C = vec2mat(c)
    assert np.max(np.abs(C.T-C)) <= 1e-12, "c doesn't represent a symmetric matrix"
    # TODO: check that the rows of A correspond to symmetric matrices

    # Initial point
    if x0 is None: x = mat2vec(np.eye(k))
    else: x = x0.copy()
    if y0 is None: y = np.zeros(n)
    else: y = y0.copy()
    if s0 is None: s = mat2vec(np.eye(k))
    else: s = s0.copy()

    # KKT system to solve in (x,y,s)
    #   x >= 0, s >= 0
    #   Ax    = b
    #   A'y+s = c
    #   xs = 0
    MAX_IP_ITER = 100
    ipiter = 0

    # This is fixed (heuristic) -- sigma=~by how much we reduce mu at each step
    sigma = .5

    # Tolerances
    feastol = 1e-7
    abstol = 1e-7 # absolute gap
    reltol = 1e-6 # relative gap

    t0 = time.perf_counter()

    while ipiter < MAX_IP_ITER:

        # Form residuals
        rp = b - A@x
        rd = c - (AT@y+s)
        mu = np.sum(x*s)/k
        primal = np.dot(c,x)
        dual = np.dot(b,y)

        pres = np.linalg.norm(rp)/max(1,np.linalg.norm(b))
        dres = np.linalg.norm(rd)/max(1,np.linalg.norm(c))

        # Convergence criterion from cvxopt
        # See https://cvxopt.org/userguide/coneprog.html#algorithm-parameters
        if pres < feastol and dres < feastol and (k*mu < abstol or (min(primal,-dual) < 0 and k*mu/(-min(primal,-dual)) < reltol)):
            print("Converged")
            break

        ipiter += 1

        # Put x and s in matrix form
        X = vec2mat(x)
        S = vec2mat(s)
        
        # Form linear system and solve it
        # System to solve is (where rp=b-A(X) and rd=C-(A'y+S))
        #   dS + A' dy  = rd
        #   A(dX)       = rp
        #   dX + .5*( X*dS*S^{-1} + S^{-1}*dS*X ) = sigma*mu*S^{-1} - X
        # We eliminate dX using the last equation to get
        #   dS + A' dy = rd
        #   A(LdS) = A(sigma*mu*S^{-1}-X)-rp
        # where L is the linear map LdS = .5*(X*dS*Sinv + Sinv*dS*X)
        # We eliminate dS using the first equation to get an equation in dy only:
        #   ALA' dy = rp + A(Lrd) + A(X-sigma*mu*S^{-1})) 
        # Call G = ALA'. Then the matrix entries of G are
        #     G[i,j] = <A_i, L(A_j)>
        # where A_i are the rows of A
        Sinv = np.linalg.inv(S)
        sinv = mat2vec(Sinv)
        def L(H): return symmetrize(X@H@Sinv)
        Lrd = mat2vec(L(vec2mat(rd)))
        rhs = rp+A@(x+Lrd-sigma*mu*sinv)
        G = scm_form(L)
        dy = np.linalg.solve(G,rhs)
        ds = rd-AT @ dy
        dS = vec2mat(ds)
        dX = sigma*mu*Sinv - X - L(dS)
        dx = mat2vec(dX)

        # Line search
        alpha_p = min(1.0, 0.98 * line_search(X, dX))
        alpha_d = min(1.0, 0.98 * line_search(S, dS))

        # Log
        print("iter={:d}, pres={:.1e}, dres={:.1e}, mu={:.1e}, primal={:.4e}, dual={:.4e}, alpha_p={:.2f}, alpha_d={:.2f}, time={:.2f}".format(ipiter, pres, dres, mu, primal, dual, alpha_p, alpha_d, time.perf_counter()-t0))
        
        # Update
        x = x + alpha_p*dx
        y = y + alpha_d*dy
        s = s + alpha_d*ds

    primal = np.dot(c,x)
    dual = np.dot(b,y)

    sol = {'primal': np.dot(c,x), 'dual': np.dot(b,y), 'X': vec2mat(x), 'y': y, 'S': vec2mat(s), 'ip_iters': ipiter}

    return sol