import os

import numpy as np
import scipy as sp

import qics
from qics.vectorize import mat_dim, vec_to_mat


def read_file(filename):
    """Reads a file representing a conic program, and return a
    :class:`~qics.Model` representing this problem. Currently supports
    ``*.dat-s``, ``*.dat-c``, and ``*.cbf`` file formats.

    Parameters
    ----------
    filename : :obj:`string`
        Name of the file and file format we want to read.

    Returns
    -------
    :class:`~qics.Model`
        Model representing the conic program from the specified file.

    See Also
    --------
    write_file : Write a conic program to a file of a specified format.
    read_sdpa : Read file in the SDPA sparse format.
    read_cbf : Read file in the Conic Benchmark Format.
    """
    file_extension = os.path.splitext(filename)[1]
    if file_extension == ".dat-s" or file_extension == ".dat-c":
        return read_sdpa(filename)
    if file_extension == ".cbf":
        return read_cbf(filename)
    raise Exception("Unsupported file extension.")


def write_file(model, filename):
    """Writes a conic program represented by a :class:`~qics.Model`
    to a specified file and format. Currently supports ``*.dat-s``,
    ``*.dat-c``, and ``*.cbf`` formats.

    Parameters
    ----------
    model : :class:`~qics.Model`
        Model representing the conic program we want to write to a file.
    filename : :obj:`string`
        Name of the file and file format we want to write to.

    See Also
    --------
    read_file : Read conic program from a file of a specified format.
    write_sdpa : Write file in the SDPA sparse format.
    write_cbf : Write file in the Conic Benchmark Format.
    """
    file_extension = os.path.splitext(filename)[1]
    if file_extension == ".dat-s" or file_extension == ".dat-c":
        return write_sdpa(model, filename)
    if file_extension == ".cbf":
        return write_cbf(model, filename)
    raise Exception("Unsupported file extension.")


def read_sdpa(filename):
    r"""Reads a file in the SDPA sparse format, and returns a
    :class:`~qics.Model` represnting the standard form semidefinite program

    .. math::

        \min_{x \in \mathbb{R}^p} &&& c^\top x

        \text{s.t.} &&& \sum_{i=1}^p F_i x_i - F_0 \succeq 0

    Two types of SDPA sparse file formats are supported.

    - ``*.dat-s``: Standard SDPA sparse file, where 
      :math:`F_i\in\mathbb{S}^n` for :math:`i=0,\ldots,p`.
    - ``*.dat-c``: Complex-valued SDPA sparse file, where 
      :math:`F_i\in\mathbb{H}^n` for :math:`i=0,\ldots,p`.

    Parameters
    ----------
    filename : :obj:`string`
        Name of the SPDA sparse file we want to read.

    Returns
    -------
    :class:`~qics.Model`
        Model representing the semidefinite program from the specified 
        file.

    See Also
    --------
    write_sdpa : Write file in the SDPA sparse format.
    """

    # Determine if this is a complex or real SDP file
    file_extension = os.path.splitext(filename)[1]
    assert file_extension == ".dat-s" or file_extension == ".dat-c"
    iscomplex = file_extension[-1] == "c"

    f = open(filename, "r")
    line = f.readline()

    ##############################################################
    # Skip comments
    ##############################################################
    # From user manual:
    #   On top of the input data file, we can write a single or
    #   multiple lines of Title and Comment. Each line of Title
    #   and Comment must begin with " or * and consist of no more
    #   than 75 letters;
    while line[0] == "*" or line[0] == '"':
        line = f.readline()

    ##############################################################
    # Read mDim (number of linear constraints b)
    ##############################################################
    mDim = int(line.strip().split(" ")[0])

    ##############################################################
    # Read nBlock (number of blocks in X and Z)
    ##############################################################
    line = f.readline()
    # nBlock = int(line.strip().split(' ')[0])

    ##############################################################
    # Read blockStruct (structure of blocks in X and Z; negative
    # integer represents a diagonal block)
    ##############################################################
    line = f.readline()
    blockStruct = [int(i) for i in line.strip().split(" ")]

    ##############################################################
    # Read b
    ##############################################################
    line = f.readline()
    line = line.strip()
    line = line.strip("{}()")
    if "," in line:
        b_str = line.strip().split(",")
    else:
        b_str = line.strip().split()
    while b_str.count("") > 0:
        b_str.remove("")
    b = np.array([[float(bi)] for bi in b_str])

    ##############################################################
    # Read c and A
    ##############################################################
    # Some useful dimension information
    step = 2 if iscomplex else 1
    dims = [step * n * n if n >= 0 else -n for n in blockStruct]
    idxs = np.insert(np.cumsum(dims), 0, 0)
    totDim = sum(dims)

    # Preallocate c
    c = np.zeros((totDim, 1))
    C = []
    for i, ni in enumerate(blockStruct):
        if ni >= 0:
            if iscomplex:
                C.append(
                    c[idxs[i] : idxs[i + 1]]
                    .reshape((-1, 2))
                    .view(dtype=np.complex128)
                    .reshape(ni, ni)
                )
            else:
                C.append(c[idxs[i] : idxs[i + 1]].reshape((ni, ni)))
        else:
            # Real vector
            C.append(c[idxs[i] : idxs[i + 1]])

    # Preallocate A (we will build a sparse matrix)
    Acols = []
    Arows = []
    Avals = []

    lineList = f.readlines()
    for line in lineList:
        row, block, colI, colJ, val = line.split()[0:5]
        row = int(row.strip(",")) - 1
        block = int(block.strip(",")) - 1
        colI = int(colI.strip(",")) - 1
        colJ = int(colJ.strip(",")) - 1

        ni = blockStruct[block]
        val = (
            complex(val.strip(","))
            if (iscomplex and ni >= 0)
            else float(val.strip(","))
        )

        if val == 0:
            continue

        if row == -1:
            # First row corresponds to data for c
            if ni >= 0:
                # Symmetric or Hermitian matrix
                C[block][colI, colJ] = val
                C[block][colJ, colI] = np.conj(val)
            else:
                # Real vector
                assert colI == colJ
                C[block][colI] = val
        else:
            # All other rows correspond to data for A
            if ni >= 0:
                # Symmetric or Hermitian matrix
                if val.real != 0.0:
                    Acols.append(idxs[block] + (colI + colJ * ni) * step)
                    Arows.append(row)
                    Avals.append(val.real)

                    if colJ != colI:
                        Acols.append(idxs[block] + (colJ + colI * ni) * step)
                        Arows.append(row)
                        Avals.append(val.real)

                if val.imag != 0.0:
                    # Hermitian matrices should have real diagonal entries
                    assert colI != colJ
                    assert iscomplex
                    Acols.append(idxs[block] + (colI + colJ * ni) * step + 1)
                    Arows.append(row)
                    Avals.append(-val.imag)

                    Acols.append(idxs[block] + (colJ + colI * ni) * step + 1)
                    Arows.append(row)
                    Avals.append(val.imag)
            else:
                # Real vector
                assert colI == colJ
                Acols.append(idxs[block] + colI)
                Arows.append(row)
                Avals.append(val)

    c *= -1  # SDPA format maximizes c
    A = sp.sparse.csr_matrix((Avals, (Arows, Acols)), shape=(mDim, totDim))

    ##############################################################
    # Get cones
    ##############################################################
    cones = []
    for bi in blockStruct:
        if bi >= 0:
            cones.append(qics.cones.PosSemidefinite(bi, iscomplex=iscomplex))
        else:
            cones.append(qics.cones.NonNegOrthant(-bi))

    return qics.Model(c=c, A=A, b=b, cones=cones)


def write_sdpa(model, filename):
    r"""Writes a standard form semidefinite program, i.e.,

    .. math::

        \min_{x \in \mathbb{R}^p} &&& c^\top x

        \text{s.t.} &&& \sum_{i=1}^p F_i x_i - F_0 \succeq 0,

    or

    .. math::

        \min_{X \in \mathbb{S}^n} &&& \langle F_0, X \rangle

        \text{s.t.} &&& \langle F_i, X \rangle = b_i, \quad i=1,\ldots,p

        &&& X \succeq 0,

    represented by a :class:`~qics.Model` to a file in the SDPA sparse
    format ``*.dat-s``. If any of the matrices :math:`F_i` are complex
    for :math:`i=0,\ldots,p`, then we use the complex SDPA sparse format
    ``*.dat-c``.

    Parameters
    ----------
    model : :class:`~qics.Model`
        Model representing the standard form semidefinite program we want
        to write to a file.
    filename : :obj:`string`
        Name of the SDPA sparse file we want to write to.

    See Also
    --------
    write_sdpa : Read file in the SDPA sparse format.
    """

    # Get SDP data from model
    assert (model.use_A) != (model.use_G)
    c = model.c_raw if model.use_A else model.h_raw
    A = model.A_raw if model.use_A else model.G_raw.T
    b = model.b_raw if model.use_A else -model.c_raw
    cones = model.cones
    iscomplex = model.iscomplex
    if not sp.sparse.issparse(A):
        A = sp.sparse.coo_matrix(A)
    assert iscomplex == (filename[-1] == "c")

    f = open(filename, "w")

    # Write mDim (length of A)
    mDim = b.size
    f.write(str(mDim) + "\n")

    # Write nBlock (number of blocks in the X variable)
    nBlock = len(cones)
    f.write(str(nBlock) + "\n")

    # Write blockStruct (structure of blocks in X variable)
    blockStruct = []
    for cone_k in cones:
        if isinstance(cone_k, qics.cones.PosSemidefinite):
            blockStruct.append(cone_k.n)
        elif isinstance(cone_k, qics.cones.NonNegOrthant):
            blockStruct.append(-cone_k.n)
        else:
            raise Exception("Invalid SDP: model contains unsupported cones.")
    f.write(" ".join(str(ni) for ni in blockStruct) + "\n")

    # Write b
    b_str = ""
    for bi in b.ravel():
        b_str += str(bi) + " "
    f.write(b_str + "\n")

    # Some useful dimension information
    dims = [cone_k.dim for cone_k in cones]
    idxs = np.insert(np.cumsum(dims), 0, 0)

    # Write C
    # Write in format (k, blk, i, j, v), where k=0, blk=block, (i, j)=index, v=value
    for blk in range(nBlock):
        # Turn vectorised Cl into matrix (make diagonal if corresponds to LP)
        Cl = c[idxs[blk] : idxs[blk + 1]]
        if isinstance(cones[blk], qics.cones.PosSemidefinite):
            Cl = vec_to_mat(Cl, iscomplex=cones[blk].get_iscomplex(), compact=False)
        elif isinstance(cones[blk], qics.cones.NonNegOrthant):
            Cl = np.diag(Cl.ravel())
        Cl = sp.sparse.coo_matrix(Cl)

        # Write upper triangular component of matrix
        for i, j, v in zip(Cl.row, Cl.col, Cl.data):
            if i <= j:
                v_str = str(-v).replace("(", "").replace(")", "")
                f.write(
                    "0 "
                    + str(blk + 1)
                    + " "
                    + str(i + 1)
                    + " "
                    + str(j + 1)
                    + " "
                    + v_str
                    + "\n"
                )

    # Write A
    # Write in format (k, blk, i, j, v), where k=0, blk=block, (i, j)=index, v=value
    A = A.tocsr()
    for k in range(mDim):
        for blk in range(nBlock):
            Akl = A[k, idxs[blk] : idxs[blk + 1]]

            if cones[blk].get_iscomplex():
                Akl = Akl[:, ::2] + Akl[:, 1::2] * 1j

            for idx, v in zip(Akl.indices, Akl.data):
                if isinstance(cones[blk], qics.cones.PosSemidefinite):
                    (i, j) = (idx // cones[blk].n, idx % cones[blk].n)
                else:
                    (i, j) = (idx, idx)

                if i <= j:
                    v_str = str(v).replace("(", "").replace(")", "")
                    f.write(
                        str(k + 1)
                        + " "
                        + str(blk + 1)
                        + " "
                        + str(i + 1)
                        + " "
                        + str(j + 1)
                        + " "
                        + v_str
                        + "\n"
                    )

    f.close()

    return


def read_cbf(filename):
    r"""Reads a file in the Conic Benchmark Format ``*.cbf``, and returns a
    :class:`~qics.Model` representing a conic program of the form

    .. math::

        \min_{x \in \mathbb{R}^n} &&& c^\top x

        \text{s.t.} &&& b - Ax = 0

         &&& h - Gx \in \mathcal{K}.

    .. warning::

        This function is quite limited in the types of ``*.cbf`` files that
        can be read. It is recommended to only use this function together
        with files written using :func:`~qics.io.write_cbf`.

    Parameters
    ----------
    filename : :obj:`string`
        Name of the CBF file we want to read.

    Returns
    -------
    :class:`~qics.Model`
        Model representing the conic program from the specified file.

    See Also
    --------
    write_cbf : Write file in the Conic Benchmark Format.
    """
    # Determine if this is a complex or real SDP file
    file_extension = os.path.splitext(filename)[1]
    assert file_extension == ".cbf"

    f = open(filename, "r")
    cones = []
    offset = 0.0
    lookup = {"qce": [], "qkd": [], "mgm": []}

    def _read_cones(cone_type, cone_dim, lookup):
        if cone_type == "L+":
            return qics.cones.NonNegOrthant(cone_dim)
        elif cone_type == "Q":
            n = 1 - cone_dim
            return qics.cones.SecondOrder(n)
        elif cone_type == "SVECPSD":
            n = mat_dim(cone_dim, compact=True)
            return qics.cones.PosSemidefinite(n)
        elif cone_type == "HVECPSD":
            n = mat_dim(cone_dim, iscomplex=True, compact=True)
            return qics.cones.PosSemidefinite(n, iscomplex=True)
        elif cone_type == "CE":
            n = cone_dim - 2
            return qics.cones.ClassEntr(n)
        elif cone_type == "CRE":
            n = (cone_dim - 1) // 2
            return qics.cones.ClassRelEntr(n)
        elif cone_type == "SVECQE":
            n = mat_dim(cone_dim - 2, compact=True)
            return qics.cones.QuantEntr(n)
        elif cone_type == "HVECQE":
            n = mat_dim(cone_dim - 2, iscomplex=True, compact=True)
            return qics.cones.QuantEntr(n, iscomplex=True)
        elif cone_type == "SVECQRE":
            n = mat_dim((cone_dim - 1) // 2, compact=True)
            return qics.cones.QuantRelEntr(n)
        elif cone_type == "HVECQRE":
            n = mat_dim((cone_dim - 1) // 2, iscomplex=True, compact=True)
            return qics.cones.QuantRelEntr(n, iscomplex=True)
        elif "SVECQCE" in cone_type:
            lookup_id = int(cone_type[1])
            dims = lookup["qce"][lookup_id][0]
            sys = lookup["qce"][lookup_id][1]
            return qics.cones.QuantCondEntr(dims, sys)
        elif "HVECQCE" in cone_type:
            lookup_id = int(cone_type[1])
            dims = lookup["qce"][lookup_id][0]
            sys = lookup["qce"][lookup_id][1]
            return qics.cones.QuantCondEntr(dims, sys, iscomplex=True)
        elif "SVECQKD" in cone_type:
            lookup_id = int(cone_type[1])
            G_info = lookup["qkd"][lookup_id][0]
            Z_info = lookup["qkd"][lookup_id][1]
            return qics.cones.QuantKeyDist(G_info, Z_info)
        elif "HVECQKD" in cone_type:
            lookup_id = int(cone_type[1])
            G_info = lookup["qkd"][lookup_id][0]
            Z_info = lookup["qkd"][lookup_id][1]
            return qics.cones.QuantKeyDist(G_info, Z_info, iscomplex=True)
        elif cone_type == "SVECORE":
            n = mat_dim(cone_dim // 3, compact=True)
            return qics.cones.OpPerspecEpi(n, "log")
        elif cone_type == "HVECORE":
            n = mat_dim(cone_dim // 3, iscomplex=True, compact=True)
            return qics.cones.OpPerspecEpi(n, "log", iscomplex=True)
        elif "SVECMGM" in cone_type:
            lookup_id = int(cone_type[1])
            power = lookup["mgm"][lookup_id]
            n = mat_dim(cone_dim // 3, compact=True)
            return qics.cones.OpPerspecEpi(n, power)
        elif "HVECMGM" in cone_type:
            lookup_id = int(cone_type[1])
            power = lookup["mgm"][lookup_id]
            n = mat_dim(cone_dim // 3, iscomplex=True, compact=True)
            return qics.cones.OpPerspecEpi(n, power, iscomplex=True)
        elif cone_type == "SVECTRE":
            n = mat_dim((cone_dim - 1) // 2, compact=True)
            return qics.cones.OpPerspecTr(n, "log")
        elif cone_type == "HVECTRE":
            n = mat_dim((cone_dim - 1) // 2, iscomplex=True, compact=True)
            return qics.cones.OpPerspecTr(n, "log", iscomplex=True)
        elif "SVECTGM" in cone_type:
            lookup_id = int(cone_type[1])
            power = lookup["mgm"][lookup_id]
            n = mat_dim((cone_dim - 1) // 2, compact=True)
            return qics.cones.OpPerspecTr(n, power)
        elif "HVECTGM" in cone_type:
            lookup_id = int(cone_type[1])
            power = lookup["mgm"][lookup_id]
            n = mat_dim((cone_dim - 1) // 2, iscomplex=True, compact=True)
            return qics.cones.OpPerspecTr(n, power, iscomplex=True)

    while True:
        line = f.readline()
        if not line:
            break
        keyword = line.strip()
        if keyword == "" or keyword[0] == "#":
            continue

        ########################
        ## File information
        ########################
        if keyword == "VER":
            ver = int(f.readline().strip())
            if ver != 4:
                print("Warning: Version of .cbf file not supported.")

        ########################
        ## Model structure
        ########################
        if keyword == "OBJSENSE":
            line = f.readline().strip()
            if line == "MIN":
                objsense = 1
            elif line == "MAX":
                objsense = -1
            else:
                raise Exception(
                    "Invalid OBJSENSE read from .cbf file (must be MIN or MAX)."
                )

        if keyword == "QCECONES":
            line = f.readline().strip()
            (ncones, _) = [int(i) for i in line.strip().split(" ")]
            for i in range(ncones):
                _ = int(f.readline().strip())
                line = f.readline()
                dims_k = [int(i) for i in line.strip().split(" ")]
                sys_k = int(f.readline().strip())
                lookup["qce"] += [(dims_k, sys_k)]

        if keyword == "QKDCONES":
            line = f.readline().strip()
            (ncones, totalparam) = [int(i) for i in line.strip().split(" ")]
            for k in range(ncones):
                line = f.readline().strip()
                (nnz_k, Klen_k, Kdim0_k, Kdim1_k, iscomplex_k) = [
                    int(i) for i in line.strip().split(" ")
                ]
                dtype = np.complex128 if iscomplex_k else np.float64
                K_list = [
                    np.zeros((Kdim0_k, Kdim1_k), dtype=dtype) for _ in range(Klen_k)
                ]
                for _ in range(nnz_k):
                    line = f.readline().strip().split(" ")
                    if iscomplex_k:
                        K_list[int(line[0])][int(line[1]), int(line[2])] = complex(
                            float(line[3]), float(line[4])
                        )
                    else:
                        K_list[int(line[0])][int(line[1]), int(line[2])] = float(
                            line[3]
                        )

                line = f.readline().strip()
                (nnz_k, Zlen_k, Zdim0_k, Zdim1_k, _) = [
                    int(i) for i in line.strip().split(" ")
                ]
                Z_list = [np.zeros((Zdim0_k, Zdim1_k)) for _ in range(Zlen_k)]
                for _ in range(nnz_k):
                    line = f.readline().strip().split(" ")
                    Z_list[int(line[0])][int(line[1]), int(line[2])] = float(line[3])

                lookup["qkd"] += [(K_list, Z_list)]

        if keyword == "MGMCONES":
            line = f.readline().strip()
            (ncones, _) = [int(i) for i in line.strip().split(" ")]
            for i in range(ncones):
                _ = int(f.readline().strip())
                power_k = float(f.readline().strip())
                lookup["mgm"] += [power_k]

        if keyword == "VAR":
            # Number and domain of variables
            # i.e., variables of the form x \in K
            line = f.readline()
            (nx, ncones) = [int(i) for i in line.strip().split(" ")]
            for i in range(ncones):
                line = f.readline().strip().split(" ")
                (cone_type, cone_dim) = (line[0], int(line[1]))
                if cone_type == "F":
                    # Corresponds to use_G = True
                    assert ncones == 1
                    assert nx == cone_dim
                    use_G = True
                else:
                    cones += [_read_cones(cone_type, cone_dim, lookup)]
                    use_G = False

        if keyword == "CON":
            # Number and domain of affine constrained variables
            # i.e., variables of the form Ax-b \in K
            line = f.readline()
            total_cone_dim = 0
            A_idxs = np.arange(0)
            (ng, ncones) = [int(i) for i in line.strip().split(" ")]
            for i in range(ncones):
                line = f.readline().strip().split(" ")
                (cone_type, cone_dim) = (line[0], int(line[1]))
                if cone_type == "L=":
                    A_idxs = np.arange(total_cone_dim, total_cone_dim + cone_dim)
                else:
                    cones += [_read_cones(cone_type, cone_dim, lookup)]
                total_cone_dim += cone_dim

        ########################
        ## Problem data
        ########################
        if keyword == "OBJACOORD":
            # Sparse objective
            c = np.zeros((nx, 1))
            nnz = int(f.readline().strip())
            for i in range(nnz):
                line = f.readline().strip().split(" ")
                c[int(line[0])] = float(line[1])

        if keyword == "OBJBCOORD":
            # Objective offset
            offset = float(f.readline().strip())

        if keyword == "ACOORD":
            # Sparse constraint matrix A
            A = np.zeros((ng, nx))
            nnz = int(f.readline().strip())
            for k in range(nnz):
                line = f.readline().strip().split(" ")
                A[int(line[0]), int(line[1])] = float(line[2])

        if keyword == "BCOORD":
            # Sparse constraint vector b
            b = np.zeros((ng, 1))
            nnz = int(f.readline().strip())
            for i in range(nnz):
                line = f.readline().strip().split(" ")
                b[int(line[0])] = float(line[1])

    ########################
    ## Process problem data
    ########################
    if use_G:
        T = _get_expand_compact_matrices(cones)

        # Need to split A into [-G; -A] and b into [-h; -b]
        # and uncompact G, h
        c *= objsense
        G_idxs = np.delete(np.arange(A.shape[0]), A_idxs)
        G = -T.T @ A[G_idxs]
        h = -T.T @ b[G_idxs]
        A = A[A_idxs]
        b = b[A_idxs]
    else:
        T = _get_expand_compact_matrices(cones)

        # No G, just need to uncompact c and A
        c = T.T @ c * objsense
        A = A @ T
        G = None
        h = None

    return qics.Model(c=c, A=A, b=b, G=G, h=h, cones=cones, offset=offset)


def write_cbf(model, filename):
    r"""Writes a conic program

    .. math::

        \min_{x \in \mathbb{R}^n} &&& c^\top x

        \text{s.t.} &&& b - Ax = 0

         &&& h - Gx \in \mathcal{K}

    represented by a :class:`~qics.Model` to a ``.cbf`` file using the
    Conic Benchmark Format.

    Parameters
    ----------
    model : :class:`~qics.Model`
        Model representing the conic program we want to write to a CBF 
        file.
    filename : :obj:`string`
        Name of the CBF file we want to write to.

    See Also
    --------
    write_cbf : Read file in the Conic Benchmark Format.
    """
    if model.use_G:
        T = _get_expand_compact_matrices(model.cones)

        c = model.c_raw.copy()
        A = model.A_raw.copy()
        b = model.b_raw.copy()
        G = T @ model.G_raw
        h = T @ model.h_raw
    else:
        T = _get_expand_compact_matrices(model.cones)

        c = T @ model.c_raw
        A = model.A_raw @ T.T
        b = model.b_raw.copy()
    cones = model.cones

    f = open(filename, "w")

    ########################
    ## File information
    ########################
    # Version number
    f.write("VER" + "\n")
    f.write(str(4) + "\n\n")

    ########################
    ## Model structure
    ########################
    # Objective sense
    f.write("OBJSENSE" + "\n")
    f.write("MIN" + "\n\n")

    # Constraints
    q = 0
    cones_string = ""
    lookup = {"qce": [], "qkd": [], "mgm": []}
    for cone_k in cones:
        if isinstance(cone_k, qics.cones.NonNegOrthant):
            (cone_name, cone_size) = ("L+", cone_k.n)
        if isinstance(cone_k, qics.cones.SecondOrder):
            (cone_name, cone_size) = ("Q", 1 + cone_k.n)
        if isinstance(cone_k, qics.cones.PosSemidefinite):
            if cone_k.get_iscomplex():
                (cone_name, cone_size) = ("HVECPSD", cone_k.n * cone_k.n)
            else:
                (cone_name, cone_size) = ("SVECPSD", cone_k.n * (cone_k.n + 1) // 2)
        if isinstance(cone_k, qics.cones.ClassEntr):
            (cone_name, cone_size) = ("CE", 2 + cone_k.n)
        if isinstance(cone_k, qics.cones.ClassRelEntr):
            (cone_name, cone_size) = ("CRE", 1 + 2 * cone_k.n)
        if isinstance(cone_k, qics.cones.QuantEntr):
            if cone_k.get_iscomplex():
                (cone_name, cone_size) = ("HVECQE", 2 + cone_k.n * cone_k.n)
            else:
                (cone_name, cone_size) = ("SVECQE", 2 + cone_k.n * (cone_k.n + 1) // 2)
        if isinstance(cone_k, qics.cones.QuantRelEntr):
            if cone_k.get_iscomplex():
                (cone_name, cone_size) = ("HVECQRE", 1 + 2 * cone_k.n * cone_k.n)
            else:
                (cone_name, cone_size) = ("SVECQRE", 1 + cone_k.n * (cone_k.n + 1))
        if isinstance(cone_k, qics.cones.QuantCondEntr):
            if cone_k.get_iscomplex():
                (cone_name, cone_size) = (
                    "@" + str(len(lookup["qce"])) + ":HVECQCE",
                    1 + cone_k.N * cone_k.N,
                )
            else:
                (cone_name, cone_size) = (
                    "@" + str(len(lookup["qce"])) + ":SVECQCE",
                    1 + cone_k.N * (cone_k.N + 1) // 2,
                )
            lookup["qce"] += [(cone_k.dims, cone_k.sys)]
        if isinstance(cone_k, qics.cones.QuantKeyDist):
            if cone_k.get_iscomplex():
                cone_name = "@" + str(len(lookup["qkd"])) + ":HVECQKD"
                cone_size = 1 + cone_k.n * cone_k.n
            else:
                cone_name = "@" + str(len(lookup["qkd"])) + ":SVECQKD"
                cone_size = 1 + cone_k.n * (cone_k.n + 1) // 2
            lookup["qkd"] += [(cone_k.K_list_raw, cone_k.Z_list_raw)]
        if isinstance(cone_k, qics.cones.OpPerspecTr):
            if cone_k.get_iscomplex():
                if cone_k.func == "log":
                    cone_name = "HVECTRE"
                else:
                    cone_name = "@" + str(len(lookup["mgm"])) + ":HVECTGM"
                    lookup["mgm"] += [cone_k.func]
                cone_size = 1 + 2 * cone_k.n * cone_k.n
            else:
                if cone_k.func == "log":
                    cone_name = "SVECTRE"
                else:
                    cone_name = "@" + str(len(lookup["mgm"])) + ":SVECTGM"
                    lookup["mgm"] += [cone_k.func]
                cone_size = 1 + 2 * cone_k.n * (cone_k.n + 1) // 2
        if isinstance(cone_k, qics.cones.OpPerspecEpi):
            if cone_k.get_iscomplex():
                if cone_k.func == "log":
                    cone_name = "HVECORE"
                else:
                    cone_name = "@" + str(len(lookup["mgm"])) + ":HVECMGM"
                    lookup["mgm"] += [cone_k.func]
                cone_size = 3 * cone_k.n * cone_k.n
            else:
                if cone_k.func == "log":
                    cone_name = "SVECORE"
                else:
                    cone_name = "@" + str(len(lookup["mgm"])) + ":SVECMGM"
                    lookup["mgm"] += [cone_k.func]
                cone_size = 3 * cone_k.n * (cone_k.n + 1) // 2

        q += cone_size
        cones_string += cone_name + " " + str(cone_size) + "\n"

    # Lookup table
    if len(lookup["qce"]) > 0:
        f.write("QCECONES" + "\n")
        f.write(str(len(lookup["qce"])) + " " + str(2 * len(lookup["qce"])) + "\n")
        for k, qce_k in enumerate(lookup["qce"]):
            f.write(str(2) + "\n")
            f.write(" ".join(str(ni) for ni in qce_k[0]) + "\n")
            f.write(str(qce_k[1]) + "\n")
        f.write("\n")

    if len(lookup["qkd"]) > 0:
        f.write("QKDCONES" + "\n")
        total_nnz = sum(
            [
                sum([np.count_nonzero(Ki) for Ki in qkd_k[0]])
                + sum([np.count_nonzero(Zi) for Zi in qkd_k[1]])
                for qkd_k in lookup["qkd"]
            ]
        )
        f.write(str(len(lookup["qkd"])) + " " + str(total_nnz) + "\n")
        for k, qkd_k in enumerate(lookup["qkd"]):
            # Total nnz, how many Klist, and size of Klist, and if complex or not
            totalnnz_k = sum([np.count_nonzero(Ki) for Ki in qkd_k[0]])
            iscomplex_k = any([np.iscomplexobj(Ki) for Ki in qkd_k[0]])
            dtype = np.complex128 if iscomplex_k else np.float64
            f.write(str(totalnnz_k) + " " + str(len(qkd_k[0])) + " "
                + str(qkd_k[0][0].shape[0]) + " " + str(qkd_k[0][0].shape[1])
                + " " + str(int(iscomplex_k)) + "\n")
            for t, Kt in enumerate(qkd_k[0]):
                # Write Ki
                Kt = sp.sparse.coo_matrix(Kt, dtype=dtype)
                for it, jt, vt in zip(Kt.row, Kt.col, Kt.data):
                    if iscomplex_k:
                        f.write(str(t) + " " + str(it) + " " + str(jt) + " "
                            + str(vt.real) + " " + str(vt.imag) + "\n")
                    else:
                        f.write(str(t) + " " + str(it) + " " + str(jt) + " "
                            + str(vt) + "\n")

            # How many Zlist, and size of Zlist, and if complex or not
            totalnnz_k = sum([np.count_nonzero(Zi) for Zi in qkd_k[1]])
            f.write(
                str(totalnnz_k)
                + " "
                + str(len(qkd_k[1]))
                + " "
                + str(qkd_k[1][0].shape[0])
                + " "
                + str(qkd_k[1][0].shape[1])
                + " 0\n"
            )
            for t, Zt in enumerate(qkd_k[1]):
                # Write Zi
                Zt = sp.sparse.coo_matrix(Zt, dtype=np.float64)
                for it, jt, vt in zip(Zt.row, Zt.col, Zt.data):
                    f.write(
                        str(t) + " " + str(it) + " " + str(jt) + " " + str(vt) + "\n"
                    )
        f.write("\n")

    if len(lookup["mgm"]) > 0:
        f.write("MGMCONES" + "\n")
        f.write(str(len(lookup["mgm"])) + " " + str(len(lookup["mgm"])) + "\n")
        for mgm_k in lookup["mgm"]:
            f.write(str(1) + "\n")
            f.write(str(mgm_k) + "\n")
        f.write("\n")

    if model.use_G:
        f.write("VAR" + "\n")
        f.write(str(model.n) + " " + str(1) + "\n")
        f.write("F" + " " + str(model.n) + "\n\n")

        f.write("CON" + "\n")
        f.write(str(q + model.p) + " " + str(len(cones) + model.use_A) + "\n")
    else:
        f.write("VAR" + "\n")
        f.write(str(q) + " " + str(len(cones)) + "\n")

    f.write(cones_string)

    if model.use_A:
        if not model.use_G:
            f.write("\n" + "CON" + "\n")
            f.write(str(model.p) + " " + str(1) + "\n")

        f.write("L=" + " " + str(model.p) + "\n")

    ########################
    ## Problem data
    ########################
    c = sp.sparse.csc_matrix(c)
    f.write("\n" + "OBJACOORD" + "\n")
    f.write(str(c.nnz) + "\n")
    for ik, vk in zip(c.indices, c.data):
        f.write(str(ik) + " " + str(vk) + "\n")

    if model.offset != 0.0:
        f.write("\n" + "OBJBCOORD" + "\n")
        f.write(str(model.offset) + "\n")

    if model.use_G:
        if not sp.sparse.issparse(G):
            G = sp.sparse.coo_matrix(G)
        if not sp.sparse.issparse(A):
            A = sp.sparse.coo_matrix(A)        
        A = sp.sparse.coo_matrix(sp.sparse.vstack([-G, A]))
    else:
        A = sp.sparse.coo_matrix(A)
    f.write("\n" + "ACOORD" + "\n")
    f.write(str(A.nnz) + "\n")
    for ik, jk, vk in zip(A.row, A.col, A.data):
        f.write(str(ik) + " " + str(jk) + " " + str(vk) + "\n")

    if model.use_G:
        b = sp.sparse.csc_matrix(np.vstack((-h, b)))
    else:
        b = sp.sparse.csc_matrix(b)
    f.write("\n" + "BCOORD" + "\n")
    f.write(str(b.nnz) + "\n")
    for ik, vk in zip(b.indices, b.data):
        f.write(str(ik) + " " + str(vk) + "\n")

    f.close()

    return


def _get_compact_to_full_op(n, iscomplex=False):
    import scipy

    dim_compact = n * n if iscomplex else n * (n + 1) // 2
    dim_full = 2 * n * n if iscomplex else n * n

    rows = np.zeros(dim_full)
    cols = np.zeros(dim_full)
    vals = np.zeros(dim_full)

    irt2 = np.sqrt(0.5)

    row = 0
    k = 0
    for j in range(n):
        for i in range(j):
            rows[k : k + 2] = row
            cols[k : k + 2] = (
                [2 * (i + j * n), 2 * (j + i * n)]
                if iscomplex
                else [i + j * n, j + i * n]
            )
            vals[k : k + 2] = irt2
            k += 2
            row += 1

            if iscomplex:
                rows[k : k + 2] = row
                cols[k : k + 2] = [2 * (i + j * n) + 1, 2 * (j + i * n) + 1]
                vals[k : k + 2] = [-irt2, irt2]
                k += 2
                row += 1

        rows[k] = row
        cols[k] = 2 * j * (n + 1) if iscomplex else j * (n + 1)
        vals[k] = 1.0
        k += 1
        row += 1

    return scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(dim_compact, dim_full))


def _get_expand_compact_matrices(cones):
    # Split A into columns correpsonding to each variable
    compact_to_full_op = []
    for cone_k in cones:
        for type_k, dim_k in zip(cone_k.type, cone_k.dim):
            if type_k == "s":
                n = int(np.sqrt(dim_k))
                compact_to_full_op += [_get_compact_to_full_op(n)]
            elif type_k == "h":
                n = int(np.sqrt(dim_k // 2))
                compact_to_full_op += [_get_compact_to_full_op(n, True)]
            else:
                compact_to_full_op += [sp.sparse.eye(dim_k)]

    return sp.sparse.block_diag(compact_to_full_op, format="csc")