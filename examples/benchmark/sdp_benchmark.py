import numpy as np
import scipy as sp

import time
import sys


def read_sdpa(filename):
    # Determine if this is a complex or real SDP file
    dtype = np.complex128 if (filename[-1] == "c") else np.float64

    fp = open(filename, "r")
    line = fp.readline()
    # Skip comments
    while line[0] == "*" or line[0] == '"':
        line = fp.readline()

    # Read mDim
    mDim = int(line.strip().split(" ")[0])

    # Read nBlock
    line = fp.readline()
    # nBlock = int(line.strip().split(" ")[0])

    # Read blockStruct
    line = fp.readline()
    blockStruct = [int(i) for i in line.strip().split(" ")]

    # Read b
    line = fp.readline()
    line = line.strip()
    line = line.strip("{}()")
    if "," in line:
        b_str = line.strip().split(",")
    else:
        b_str = line.strip().split()
    while b_str.count("") > 0:
        b_str.remove("")
    b = np.array([float(bi) for bi in b_str])

    # Read C and A
    C = []
    for bi in blockStruct:
        if bi >= 0:
            C.append(np.zeros((bi, bi), dtype=dtype))
        else:
            C.append(np.zeros(-bi))

    totDim = 0
    idxs = [0]
    for n in blockStruct:
        if n >= 0:
            if dtype == np.complex128:
                totDim += 2 * n * n
                idxs.append(idxs[-1] + 2 * n * n)
            else:
                totDim += n * n
                idxs.append(idxs[-1] + n * n)
        else:
            totDim -= n
            idxs.append(idxs[-1] - n)

    Acols = []
    Arows = []
    Avals = []

    lineList = fp.readlines()
    for line in lineList:
        row, block, colI, colJ, val = line.split()[0:5]
        row = int(row.strip(",")) - 1
        block = int(block.strip(",")) - 1
        colI = int(colI.strip(",")) - 1
        colJ = int(colJ.strip(",")) - 1
        val = (
            complex(val.strip(","))
            if (dtype == np.complex128 and blockStruct[block] >= 0)
            else float(val.strip(","))
        )

        if val == 0:
            continue

        if row == -1:
            if blockStruct[block] >= 0:
                C[block][colI, colJ] = val
                C[block][colJ, colI] = np.conj(val)
            else:
                assert colI == colJ
                C[block][colI] = val
        else:
            if blockStruct[block] >= 0:
                if dtype == np.complex128:
                    if val.real != 0.0:
                        Acols.append(
                            idxs[block] + (colI + colJ * blockStruct[block]) * 2
                        )
                        Arows.append(row)
                        Avals.append(val.real)

                        if colJ != colI:
                            Acols.append(
                                idxs[block] + (colJ + colI * blockStruct[block]) * 2
                            )
                            Arows.append(row)
                            Avals.append(val.real)

                    if val.imag != 0.0:
                        Acols.append(
                            idxs[block] + (colI + colJ * blockStruct[block]) * 2 + 1
                        )
                        Arows.append(row)
                        Avals.append(val.imag)

                        Acols.append(
                            idxs[block] + (colJ + colI * blockStruct[block]) * 2 + 1
                        )
                        Arows.append(row)
                        Avals.append(-val.imag)
                else:
                    Acols.append(idxs[block] + colI + colJ * blockStruct[block])
                    Arows.append(row)
                    Avals.append(val)

                    if colJ != colI:
                        Acols.append(idxs[block] + colJ + colI * blockStruct[block])
                        Arows.append(row)
                        Avals.append(val)
            else:
                assert colI == colJ
                Acols.append(idxs[block] + colI)
                Arows.append(row)
                Avals.append(val)

    A = sp.sparse.csr_matrix((Avals, (Arows, Acols)), shape=(mDim, totDim))

    return C, b, A, blockStruct


def cvxopt_solve_sdp(C, b, A, blockStruct):
    import cvxopt

    A = sp.sparse.csr_matrix(A)
    p = b.size

    # Split A into SDP and LP components
    t = 0
    Gs = []
    hs = []
    Gl = None
    hl = None
    for k, n in enumerate(blockStruct):
        if n > 0:
            # SDP
            A_k = A[:, t : t + n * n].tocoo()
            Gs.append(cvxopt.spmatrix(A_k.data, A_k.col, A_k.row, (n * n, p), "d"))
            hs.append(cvxopt.matrix(-C[k]))
            t += n * n
        else:
            # LP
            A_k = A[:, t : t - n].tocoo()
            Gl = cvxopt.spmatrix(A_k.data, A_k.col, A_k.row, (-n, p), "d")
            hl = cvxopt.matrix(-C[k])

    # Solve with cvxopt
    # Fit the form max -<h1,z> s.t. G1'*z + c == 0, z >= 0
    t0_cvxopt = time.time()
    opts = {"maxiters": 50, "abstol": 1e-8, "reltol": 1e-8, "feastol": 1e-8}
    cvxoptsol = cvxopt.solvers.sdp(
        c=cvxopt.matrix(-b), Gl=Gl, hl=hl, Gs=Gs, hs=hs, options=opts
    )
    cvxopt_time = time.time() - t0_cvxopt

    return {
        "obj": cvxoptsol["dual objective"],
        "status": cvxoptsol["status"],
        "time": cvxopt_time,
        "iter": cvxoptsol["iterations"],
        "gap": min(cvxoptsol["relative gap"], cvxoptsol["gap"]),
        "dfeas": cvxoptsol["dual infeasibility"],
        "pfeas": cvxoptsol["primal infeasibility"],
    }


def mosek_solve_sdp(C, b, A, blockStruct):
    from mosek.fusion import Model, Matrix, Domain, Expr, ObjectiveSense

    # Solve problem
    #   min c'x s.t. Ax == b, x >= 0
    #   max b'y s.t. A'y <= c, y \in R^n
    # using Mosek

    A = sp.sparse.csr_matrix(A)
    p = b.size

    # Split A into SDP and LP components
    t = 0
    msk_M = Model("mosek")

    msk_X = []
    msk_A = []
    msk_AX = []
    msk_CX = []

    for k, n in enumerate(blockStruct):
        if n > 0:
            # SDP
            A_k = A[:, t : t + n * n].tocoo()

            msk_X.append(msk_M.variable(Domain.inPSDCone(n)))
            msk_A.append(
                Matrix.sparse(p, n * n, list(A_k.row), list(A_k.col), list(A_k.data))
            )
            msk_AX.append(Expr.mul(msk_A[k], msk_X[k].reshape(n * n)))
            msk_CX.append(Expr.dot(C[k], msk_X[k]))

            t += n * n
        else:
            # LP
            A_k = A[:, t : t - n].tocoo()

            msk_X.append(msk_M.variable(Domain.greaterThan(0.0, -n)))
            msk_A.append(
                Matrix.sparse(p, -n, list(A_k.row), list(A_k.col), list(A_k.data))
            )
            msk_AX.append(Expr.mul(msk_A[k], msk_X[k].reshape(-n)))
            msk_CX.append(Expr.dot(C[k], msk_X[k].reshape(-n)))

    # Mosek MODEL
    msk_M.constraint(Expr.add(msk_AX), Domain.equalsTo(list(b.ravel())))
    msk_M.objective(ObjectiveSense.Maximize, Expr.add(msk_CX))
    msk_M.setLogHandler(sys.stdout)
    msk_M.setSolverParam("intpntCoTolNearRel", 1.0)
    msk_M.solve()
    msk_status = msk_M.getProblemStatus()

    return {
        "obj": msk_M.getSolverDoubleInfo("intpntPrimalObj"),
        "time": msk_M.getSolverDoubleInfo("intpntTime"),
        "status": msk_status,
        "iter": msk_M.getSolverIntInfo("intpntIter"),
        "gap": (
            msk_M.getSolverDoubleInfo("intpntPrimalObj")
            - msk_M.getSolverDoubleInfo("intpntDualObj")
        )
        / max(1, abs(msk_M.getSolverDoubleInfo("intpntPrimalObj"))),
        "dfeas": msk_M.getSolverDoubleInfo("intpntDualFeas"),
        "pfeas": msk_M.getSolverDoubleInfo("intpntPrimalFeas"),
    }


def solve_sdpap(A, b, c, K, J, option={}):
    """Solve CLP by SDPA

    If J.l or J.q or J.s > 0, clp_toLMI() or clp_toEQ() is called before solve.

    Args:
      A, b, c: Scipy matrices to denote the CLP.
      K, J: Symcone object to denote the CLP.
      option: Parameters. If None, default parameters is used.

    Returns:
      A tuple (x, y, sdpapinfo, timeinfo, sdpainfo).
      x, y: Primal and Dual solutions
      sdpapinfo, timeinfo, sdpainfo: Result information
    """

    from sdpap import convert
    from sdpap import fileio
    from sdpap.param import param
    from sdpap.sdpacall import sdpacall
    from sdpap.spcolo import spcolo
    from sdpap.fvelim import fvelim
    from scipy import sparse
    import numpy as np
    import copy
    import time

    timeinfo = dict()
    timeinfo["total"] = time.time()

    if "print" not in option:
        option["print"] = "display"
    verbose = len(option["print"]) != 0 and option["print"] != "no"
    maybe_print = print if verbose else lambda *a, **k: None

    # --------------------------------------------------
    # Set parameter
    # --------------------------------------------------
    backend_info = sdpacall.get_backend_info()
    option = param(option, backend_info["gmp"])
    maybe_print("---------- SDPAP Start ----------")

    # Write to output file
    if option["resultFile"]:
        fpout = open(option["resultFile"], "w")
        fileio.write_version(fpout)
        fileio.write_parameter(fpout, option)

    # --------------------------------------------------
    # Check validity
    # --------------------------------------------------
    if not K.check_validity() or not J.check_validity():
        return None

    if not isinstance(b, np.ndarray) and not sparse.issparse(b):
        raise TypeError("sdpap.solve(): b must be a np.ndarray or a sparse matrix.")
    if not isinstance(c, np.ndarray) and not sparse.issparse(c):
        raise TypeError("sdpap.solve(): c must be a np.ndarray or a sparse matrix.")
    if not isinstance(A, np.ndarray) and not sparse.issparse(A):
        raise TypeError("sdpap.solve(): A must be a np.ndarray or a sparse matrix.")

    if isinstance(b, np.ndarray) and len(b.shape) > 2:
        raise ValueError("sdpap.solve(): Expected 1D or 2D ndarray for b")
    if isinstance(c, np.ndarray) and len(c.shape) > 2:
        raise ValueError("sdpap.solve(): Expected 1D or 2D ndarray for c")
    if isinstance(A, np.ndarray) and len(A.shape) != 2:
        raise ValueError("sdpap.solve(): Expected 2D ndarray for A")

    if not sparse.isspmatrix_csc(b):
        b = sparse.csc_matrix(b)
    if b.shape[1] != 1:
        b = (b.T).tocsc()

    if not sparse.isspmatrix_csc(c):
        c = sparse.csc_matrix(c)
    if c.shape[1] != 1:
        c = (c.T).tocsc()

    if not sparse.isspmatrix_csc(A):
        A = sparse.csc_matrix(A)

    size_row = max(b.shape)
    size_col = max(c.shape)
    mA, nA = A.shape

    totalSize_n = K.f + K.l + sum(K.q) + sum(z**2 for z in K.s)
    totalSize_m = J.f + J.l + sum(J.q) + sum(z**2 for z in J.s)
    if size_row != mA or size_col != nA:
        maybe_print(
            "Size A[m = %d, n = %d], b[m = %d], c[n = %d] ::"
            % (mA, nA, size_row, size_col)
        )
        maybe_print("nnz(A) = %d, nnz(c) = %d" % (A.nnz, c.nnz))
        raise ValueError("Inconsistent Size")
    if size_col != totalSize_n:
        maybe_print(
            "Size A[m = %d, n = %d], b[m = %d], c[n = %d] ::"
            % (mA, nA, size_row, size_col)
        )
        maybe_print("nnz(A) = %d, nnz(c) = %d" % (A.nnz, c.nnz))
        raise ValueError("Inconsistent Size c[n = %d], K[%d]" % (size_col, totalSize_n))
    if size_row != totalSize_m:
        maybe_print(
            "Size A[m = %d, n = %d], b[m = %d], c[n = %d] ::"
            % (mA, nA, size_row, size_col)
        )
        maybe_print("nnz(A) = %d, nnz(c) = %d" % (A.nnz, c.nnz))
        raise ValueError("Inconsistent Size b[n = %d], J[%d]" % (size_row, totalSize_m))

    if option["resultFile"]:
        fpout.write("----- Input Problem -----\n")
        fileio.write_symcone(fpout, K, J)

    # --------------------------------------------------
    # Exploiting sparsity conversion
    # --------------------------------------------------
    timeinfo["conv_domain"] = time.time()

    # Convert domain space sparsity
    if len(K.s) > 0:
        if option["domainMethod"] == "clique":
            maybe_print(
                "Applying the d-space conversion method " "using clique trees..."
            )
            dom_A, dom_b, dom_c, dom_K, dom_J, cliqueD = spcolo.dconv_cliquetree(
                A, b, c, K, J
            )
            ############################################################
            # Under construction
            ############################################################
            return
        elif option["domainMethod"] == "basis":
            maybe_print(
                "Applying the d-space conversion method "
                "using basis representation..."
            )
            dom_A, dom_b, dom_c, dom_K, dom_J, cliqueD = spcolo.dconv_basisrep(
                A, b, c, K, J
            )
        else:
            dom_A = copy.deepcopy(A)
            dom_b = copy.deepcopy(b)
            dom_c = copy.deepcopy(c)
            dom_K = copy.deepcopy(K)
            dom_J = copy.deepcopy(J)
    else:
        dom_A = copy.deepcopy(A)
        dom_b = copy.deepcopy(b)
        dom_c = copy.deepcopy(c)
        dom_K = copy.deepcopy(K)
        dom_J = copy.deepcopy(J)

    timeinfo["conv_domain"] = time.time() - timeinfo["conv_domain"]

    if option["resultFile"] and option["domainMethod"] != "none":
        fpout.write("----- Domain Space Sparsity Converted Problem-----\n")
        fileio.write_symcone(fpout, dom_K, dom_J)

    # Convert range space sparsity
    timeinfo["conv_range"] = time.time()
    if len(dom_J.s) > 0:
        if option["rangeMethod"] == "clique":
            maybe_print(
                "Applying the r-space conversion method " "using clique trees..."
            )
            ran_A, ran_b, ran_c, ran_K, ran_J, cliqueR = spcolo.rconv_cliquetree(
                dom_A, dom_b, dom_c, dom_K, dom_J
            )
            ############################################################
            # Under construction
            ############################################################
        elif option["rangeMethod"] == "decomp":
            maybe_print(
                "Applying the r-space conversion method "
                "using matrix decomposition..."
            )
            ran_A, ran_b, ran_c, ran_K, ran_J, cliqueR = spcolo.rconv_matdecomp(
                dom_A, dom_b, dom_c, dom_K, dom_J
            )
        else:
            ran_A, ran_b, ran_c = dom_A, dom_b, dom_c
            ran_K = copy.deepcopy(dom_K)
            ran_J = copy.deepcopy(dom_J)
    else:
        ran_A, ran_b, ran_c = dom_A, dom_b, dom_c
        ran_K = copy.deepcopy(dom_K)
        ran_J = copy.deepcopy(dom_J)

    timeinfo["conv_range"] = time.time() - timeinfo["conv_range"]

    if option["resultFile"] and option["rangeMethod"] != "none":
        fpout.write("----- Range Space Sparsity Converted Problem-----\n")
        fileio.write_symcone(fpout, ran_K, ran_J)

    # --------------------------------------------------
    # Convert to SeDuMi standard form
    # --------------------------------------------------
    timeinfo["conv_std"] = time.time()

    useConvert = False
    if ran_J.l > 0 or len(ran_J.q) > 0 or len(ran_J.s) > 0:
        useConvert = True
        if option["convMethod"] == "LMI":
            maybe_print("Converting CLP format to LMI standard form...")
            A2, b2, c2, K2, J2, map_sdpIndex = convert.clp_toLMI(
                ran_A, ran_b, ran_c, ran_K, ran_J
            )
        elif option["convMethod"] == "EQ":
            maybe_print("Converting CLP format to EQ standard form.")
            ##################################################
            # This method is under construction
            ##################################################
            A2, b2, c2, K2, J2 = convert.clp_toEQ(ran_A, ran_b, ran_c, ran_K, ran_J)
        else:
            raise ValueError("convMethod must be 'LMI' or 'EQ'")
    else:
        A2, b2, c2 = ran_A, ran_b, ran_c
        K2 = copy.deepcopy(ran_K)
        # J2 = copy.deepcopy(ran_J)

    timeinfo["conv_std"] = time.time() - timeinfo["conv_std"]

    if option["resultFile"] and (ran_J.l > 0 or len(ran_J.q) > 0 or len(ran_J.s) > 0):
        fpout.write("----- SeDuMi format Converted Problem-----\n")
        fileio.write_symcone(fpout, K2)

    # --------------------------------------------------
    # Eliminate free variables
    # --------------------------------------------------
    timeinfo["conv_fv"] = time.time()

    if K2.f > 0:
        if option["frvMethod"] == "split":
            maybe_print("Eliminating free variables with split method...")
            A3, b3, c3, K3 = fvelim.split(A2, b2, c2, K2, option["rho"])
        elif option["frvMethod"] == "elimination":
            maybe_print("Eliminationg free variables with elimination method...")
            (A3, b3, c3, K3, LiP, U, Q, LPA_B, LPb_B, cfQU, gamma, rank_Af) = (
                fvelim.eliminate(A2, b2, c2, K2, option["rho"], option["zeroPoint"])
            )
        else:
            raise ValueError("frvMethod must be 'split' or 'elimination'")
    else:
        A3, b3, c3, K3 = A2, b2, c2, K2

    timeinfo["conv_fv"] = time.time() - timeinfo["conv_fv"]

    if option["resultFile"] and K2.f > 0:
        fpout.write("----- Free Variables Eliminated Problem -----\n")
        fileio.write_symcone(fpout, K3)

    # --------------------------------------------------
    # Solve by SDPA
    # --------------------------------------------------
    timeinfo["sdpa"] = time.time()
    x3, y3, s3, sdpainfo = sdpacall.solve_sdpa(A3, b3, c3, K3, option)
    timeinfo["sdpa"] = time.time() - timeinfo["sdpa"]

    # --------------------------------------------------
    # Get Result
    # --------------------------------------------------
    maybe_print("Start: getCLPresult")

    # Retrieve result of fvelim
    timeinfo["ret_fv"] = time.time()
    if K2.f > 0:
        if option["frvMethod"] == "split":
            maybe_print("Retrieving result with split method...")
            x2, y2, s2 = fvelim.result_split(x3, y3, s3, K2)
        elif option["frvMethod"] == "elimination":
            maybe_print("Retrieving result with elimination method...")
            x2, y2, s2 = fvelim.result_elimination(
                x3, y3, s3, K2, LiP, U, Q, LPA_B, LPb_B, cfQU, rank_Af
            )
        else:
            raise ValueError("frvMethod must be 'split' or 'elimination'")
    else:
        x2, y2 = x3, y3

    timeinfo["ret_fv"] = time.time() - timeinfo["ret_fv"]

    # Retrieve result from LMI or EQ
    timeinfo["ret_std"] = time.time()
    if useConvert:
        if option["convMethod"] == "LMI":
            maybe_print("Retrieving result from LMI standard form...")
            x, y = convert.result_fromLMI(x2, y2, ran_K, ran_J, map_sdpIndex)
            tmp = -sdpainfo["primalObj"]
            sdpainfo["primalObj"] = -sdpainfo["dualObj"]
            sdpainfo["dualObj"] = tmp
        elif option["convMethod"] == "EQ":
            maybe_print("Retrieving result from EQ standard form...")
            ##################################################
            # This method is under construction
            ##################################################
            # x, y = result_fromEQ(x2, y2, ran_K, ran_J)
        else:
            raise ValueError("Something wrong about option['convMethod']")
    else:
        x, y = x2, y2

    timeinfo["ret_std"] = time.time() - timeinfo["ret_std"]

    # Retrieve an optiomal solution from range space sparsity converted problem
    timeinfo["ret_range"] = time.time()
    if option["rangeMethod"] != "none" and len(J.s) > 0:
        if option["rangeMethod"] == "clique":
            maybe_print(
                "Retrieving result with r-space conversion method "
                "using clique trees..."
            )
            ############################################################
            # Under construction
            ############################################################
            x, y = spcolo.rconv_cliqueresult(x, y, dom_K, dom_J, ran_K, cliqueR)
        elif option["rangeMethod"] == "decomp":
            maybe_print(
                "Retrieving result with r-space conversion method "
                "using matrix decomposition..."
            )
            x, y = spcolo.rconv_decompresult(x, y, dom_K, dom_J, ran_J, cliqueR)

    timeinfo["ret_range"] = time.time() - timeinfo["ret_range"]

    # Retrieve an optiomal solution from domain space sparsity converted problem
    timeinfo["ret_domain"] = time.time()
    if option["domainMethod"] != "none" and len(K.s) > 0:
        if option["domainMethod"] == "clique":
            maybe_print(
                "Retrieving result with d-space conversion method "
                "using clique trees..."
            )
            ############################################################
            # Under construction
            ############################################################
            x, y = spcolo.dconv_cliqueresult(x, y, K, J, dom_J, cliqueD)
        elif option["domainMethod"] == "basis":
            maybe_print(
                "Retrieving result with d-space conversion method "
                "using basis representation..."
            )
            x, y = spcolo.dconv_basisresult(x, y, K, J, dom_K, cliqueD)

    timeinfo["ret_domain"] = time.time() - timeinfo["ret_domain"]
    timeinfo["total"] = time.time() - timeinfo["total"]

    # --------------------------------------------------
    # Make dictionary 'info'
    # --------------------------------------------------
    timeinfo["convert"] = (
        timeinfo["conv_domain"]
        + timeinfo["conv_range"]
        + timeinfo["conv_std"]
        + timeinfo["conv_fv"]
    )
    timeinfo["retrieve"] = (
        timeinfo["ret_fv"]
        + timeinfo["ret_std"]
        + timeinfo["ret_range"]
        + timeinfo["ret_domain"]
    )

    if ran_K.f > 0 and option["frvMethod"] == "elimination":
        sdpainfo["primalObj"] += gamma
        sdpainfo["dualObj"] += gamma

    return x, y, timeinfo, sdpainfo


if __name__ == "__main__":
    import os
    import csv

    folder = "./sdps/"
    # Run all instances in folder
    fnames = os.listdir(folder)
    # Run single instance
    # fnames = ["qsd_50_10.dat-c"]

    fout_name = "data_sdp.csv"
    # with open(fout_name, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["problem", "solver", "status", "optval", "time", "iter", "gap", "pfeas", "dfeas"])

    for fname in fnames:
        # ==============================================================
        # Read problem data
        # ==============================================================
        try:
            C_sdpa, b_sdpa, A_sdpa, blockStruct = read_sdpa(folder + fname)
            b = b_sdpa.reshape((-1, 1))
            A = A_sdpa
        except Exception as e:
            print(e)
            continue

        # ==============================================================
        # CVXOPT
        # ==============================================================
        try:
            sol = cvxopt_solve_sdp(C_sdpa, b, A, blockStruct)

            with open(fout_name, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        fname,
                        "cvxopt",
                        sol["status"],
                        sol["obj"],
                        sol["time"],
                        sol["iter"],
                        sol["gap"],
                        sol["pfeas"],
                        sol["dfeas"],
                    ]
                )

            print("optval: ", sol["gap"])
            print("time:   ", sol["time"])
        except Exception as e:
            with open(fout_name, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [fname, "cvxopt", e, None, None, None, None, None, None]
                )

        # # ==============================================================
        # # MOSEK
        # # ==============================================================
        # # try:
        # sol = mosek_solve_sdp(C_sdpa, b, A, blockStruct)

        # with open(fout_name, 'a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([fname, "mosek", sol['status'], sol['obj'], sol['time'], sol['iter'], sol['gap'], sol['pfeas'], sol['dfeas']])
        # # except Exception as e:
        # #     with open(fout_name, 'a', newline='') as file:
        # #         writer = csv.writer(file)
        # #         writer.writerow([fname, "mosek", e, None, None, None, None, None, None])

        # # ==============================================================
        # # SDPA
        # # ==============================================================
        # try:
        #     A, b, c, K, J = sdpap.fromsdpa(folder + fname)
        #     sdpap_options = {}
        #     sdpap_options['epsilonStar'] = 1e-8
        #     x, y, timeinfo, sdpainfo = solve_sdpap(-A,-b,-c,K,J,sdpap_options)

        #     with open(fout_name, 'a', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow([
        #             fname,
        #             "sdpa",
        #             sdpainfo['phasevalue'],
        #             sdpainfo['primalObj'],
        #             sdpainfo['sdpaTime'],
        #             sdpainfo['iteration'],
        #             abs(sdpainfo['primalObj'] - sdpainfo['dualObj']) / max(1., (abs(sdpainfo['primalObj']) + abs(sdpainfo['dualObj'])) / 2),
        #             sdpainfo['primalError'],
        #             sdpainfo['dualError']
        #         ])
        # except Exception as e:
        #     with open(fout_name, 'a', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow([fname, "cvxopt", e, None, None, None, None, None, None])
