import numpy as np
import scipy as sp
import os
import qics

def read_sdpa(filename):
    """Reads a file in the SDPA sparse format, and returns the data
    matrices :math:`C\\in\\mathbb{S}^n`, :math:`A_i\\in\\mathbb{S}^n`, 
    and :math:`b\\in\\mathbb{R}^p` representing the primal

    .. math::

        \\max_{X\\in\\mathbb{S}^n} &&& \\langle C, X \\rangle

        \\text{s.t.} &&& \\langle A_i, X \\rangle = b_i, \\quad i=1,\\ldots,p

         &&& X \\succeq 0

    and dual

    .. math::

        \\min_{y \\in \\mathbb{R}^p} &&& -b^\\top y

        \\text{s.t.} &&& \\sum_{i=1}^p A_i y_i - C \succeq 0 

    pair of semidefinite programs. Accepts either file extensions
    
        - ``.dat-s``: Standard SDPA sparse file.
        - ``.dat-c``: Complex-valued SDPA sparse file, in which case the
          data matrices :math:`C\\in\\mathbb{H}^n`, :math:`A_i\\in\\mathbb{H}^n`
          are assumed to all be Hermitian.

    Data is returned in a form that can directly be used to initialize
    a :class:`~qics.Model`.
    
    Code adapted from: https://sdpa-python.github.io/

    Parameters
    ----------
    filename : string
        Filename of file in SDPA sparse format to read.
        
    Returns
    -------
    ndarray (n, 1)
        Float array representing the linear objective ``c`` as 
        the vectorized matrix :math:`-C`.
    ndarray (p, n)
        Float array representing the matrix representation ``A`` 
        for the linear equality constraint data :math:`A_i`.
    ndarray (p, 1)
        Float array representing linear equality constraints ``b``
        for the data :math:`b`.
    list
        List of cone classes representing the Cartesian product of 
        cones :math:`\\mathcal{K}`.
    """

    # Determine if this is a complex or real SDP file
    file_extension = os.path.splitext(filename)[1]
    assert file_extension == ".dat-s" or file_extension == ".dat-c"
    iscomplex = (file_extension[-1] == 'c')
    dtype     = np.complex128 if iscomplex else np.float64

    fp = open(filename, "r")
    line = fp.readline()

    ##############################################################
    # Skip comments
    ##############################################################
    # From user manual: 
    #   On top of the input data file, we can write a single or 
    #   multiple lines of Title and Comment. Each line of Title 
    #   and Comment must begin with " or * and consist of no more 
    #   than 75 letters;
    while line[0] == '*' or line[0] == '"':
        line = fp.readline()
        
    ##############################################################
    # Read mDim (number of linear constraints b)
    ##############################################################
    mDim = int(line.strip().split(' ')[0])

    ##############################################################
    # Read nBlock (number of blocks in X and Z)
    ##############################################################
    line = fp.readline()
    nBlock = int(line.strip().split(' ')[0])
    
    ##############################################################
    # Read blockStruct (structure of blocks in X and Z; negative 
    # integer represents a diagonal block)
    ##############################################################
    line = fp.readline()
    blockStruct = [int(i) for i in line.strip().split(' ')]
    
    ##############################################################
    # Read b
    ##############################################################
    line = fp.readline()
    line = line.strip()
    line = line.strip('{}()')
    if ',' in line:
        b_str = line.strip().split(',')
    else:
        b_str = line.strip().split()
    while b_str.count('') > 0:
        b_str.remove('')
    b = np.array([[float(bi)] for bi in b_str])

    ##############################################################
    # Read c and A
    ##############################################################
    # Some useful dimension information
    step   = 2 if iscomplex else 1
    dims   = [step*n*n if n >= 0 else -n for n in blockStruct]
    idxs   = np.insert(np.cumsum(dims), 0, 0)
    totDim = sum(dims)

    # Preallocate c
    c = np.zeros((totDim, 1))
    C = []
    for (i, ni) in enumerate(blockStruct):
        if ni >= 0:
            if iscomplex:
                C.append(c[idxs[i]:idxs[i + 1]].reshape((-1, 2)).view(dtype=np.complex128).reshape(ni, ni))
            else:
                C.append(c[idxs[i]:idxs[i + 1]].reshape((ni, ni)))
        else:
            # Real vector
            C.append(c[idxs[i]:idxs[i + 1]])

    # Preallocate A (we will build a sparse matrix)
    Acols = []
    Arows = []
    Avals = []

    lineList = fp.readlines()
    for line in lineList:
        row, block, colI, colJ, val = line.split()[0:5]
        row   = int(row.strip(','))   - 1
        block = int(block.strip(',')) - 1
        colI  = int(colI.strip(','))  - 1
        colJ  = int(colJ.strip(','))  - 1

        ni  = blockStruct[block]
        val = complex(val.strip(',')) if (iscomplex and ni>=0) else float(val.strip(','))
        
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
                if val.real != 0.:
                    Acols.append(idxs[block] + (colI + colJ*ni)*step)
                    Arows.append(row)
                    Avals.append(val.real)

                    if colJ != colI:
                        Acols.append(idxs[block] + (colJ + colI*ni)*step)
                        Arows.append(row)
                        Avals.append(val.real)

                if val.imag != 0.:
                    # Hermitian matrices should have real diagonal entries
                    assert colI != colJ
                    assert iscomplex
                    Acols.append(idxs[block] + (colI + colJ*ni)*step + 1)
                    Arows.append(row)
                    Avals.append(-val.imag)

                    Acols.append(idxs[block] + (colJ + colI*ni)*step + 1)
                    Arows.append(row)
                    Avals.append(val.imag)
            else:
                # Real vector
                assert colI == colJ
                Acols.append(idxs[block] + colI)
                Arows.append(row)
                Avals.append(val)

    c *= -1     # SDPA format maximizes c
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
            
    return c, b, A, cones

def write_sdpa(model, filename):
    """Writes a semidefinite program 

    .. math::

        \\max_{X\\in\\mathbb{S}^n} &&& \\langle C, X \\rangle

        \\text{s.t.} &&& \\langle A_i, X \\rangle = b_i, \\quad i=1,\\ldots,p

         &&& X \\succeq 0

    with dual

    .. math::

        \\min_{y \\in \\mathbb{R}^p} &&& -b^\\top y

        \\text{s.t.} &&& \\sum_{i=1}^p A_i y_i - C \succeq 0   
    
    represented by a :class:`~qics.Model` to a ``.dat-s`` (or 
    ``.dat-c`` for complex-valued SDPs) file using the sparse SDPA 
    file format.

    Parameters
    ----------
    model : qics.Model
        Instance of model containing model of the semidefinite program.
    filename : string
        Filename of file in SDPA sparse format to save in.
    """

    # Get SDP data from model
    assert (model.use_A) != (model.use_G)
    c         = model.c_raw if model.use_A else  model.h_raw
    A         = model.A_raw if model.use_A else  model.G_raw.T
    b         = model.b_raw if model.use_A else -model.c_raw
    cones     = model.cones
    iscomplex = model.iscomplex
    assert sp.sparse.issparse(A)
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
    f.write(' '.join(str(ni) for ni in blockStruct) + "\n")
    
    # Write b
    b_str = ""
    for bi in b.ravel():
        b_str += str(bi) + " "
    f.write(b_str + "\n")
    

    # Some useful dimension information
    dims = [cone_k.dim for cone_k in cones]
    idxs = np.insert(np.cumsum(dims), 0, 0)

    # Write C
    # Write in format (k, l, i, j, v), where k=0, l=block, (i, j)=index, v=value
    for l in range(nBlock):
        # Turn vectorised Cl into matrix (make diagonal if corresponds to LP)
        Cl = c[idxs[l] : idxs[l + 1]]
        if isinstance(cones[l], qics.cones.PosSemidefinite):
            Cl = qics.vectorize.vec_to_mat(Cl, iscomplex=cones[l].get_iscomplex(), compact=False)
        elif isinstance(cones[l], qics.cones.NonNegOrthant):
            Cl = np.diag(Cl.ravel())
        Cl = sp.sparse.coo_matrix(Cl)

        # Write upper triangular component of matrix
        for (i, j, v) in zip(Cl.row, Cl.col, Cl.data):
            if i <= j:
                v_str = str(-v).replace('(', '').replace(')', '')
                f.write("0 " + str(l+1) + " " + str(i+1) + " " + str(j+1) + " " + v_str + "\n")

    # Write A
    # Write in format (k, l, i, j, v), where k=0, l=block, (i, j)=index, v=value
    A = A.tocsr()
    for k in range(mDim):
        for l in range(nBlock):
            Akl = A[k, idxs[l] : idxs[l + 1]]

            if cones[l].get_iscomplex():
                Akl = Akl[:, ::2] + Akl[:, 1::2]*1j

            for (idx, v) in zip(Akl.indices, Akl.data):
                if isinstance(cones[l], qics.cones.PosSemidefinite):
                    (i, j) = (idx // cones[l].n, idx % cones[l].n)
                else:
                    (i, j) = (idx, idx)

                if i <= j:
                    v_str = str(v).replace('(', '').replace(')', '')
                    f.write(str(k + 1) + " " + str(l+1) + " " + str(i+1) + " " + str(j+1) + " " + v_str + "\n")

    f.close()
            
    return

def read_cbf(filename):
    # Determine if this is a complex or real SDP file
    file_extension = os.path.splitext(filename)[1]
    assert file_extension == ".cbf"

    fp = open(filename, "r")
    cones = []

    while True:
        ########################
        ## File information
        ########################
        keyword = fp.readline().strip()
        if keyword == "VER":
            ver = int(fp.readline().strip())
            if ver != 1 and ver != 2 and ver != 3:
                print("Warning: Version of .cbf file not supported.")
        
        ########################
        ## Model structure
        ########################
        if keyword == "OBJSENSE":
            line = fp.readline().strip()
            if line == "MIN":
                objsense = 1
            elif line == "MAX":
                objsense = -1
            else:
                raise Exception("Invalid OBJSENSE read from .cbf file (must be MIN or MAX).")
            
        if keyword == "VAR":
            # Number and domain of variables
            # i.e., variables of the form x \in K
            line = fp.readline()
            (nx, ncones) = [int(i) for i in line.strip().split(' ')]
            for i in range(ncones):
                line = fp.readline().strip().split(' ')
                (cone_type, cone_dim) = (line[0], int(line[1]))
                if cone_type == "F":
                    pass
                elif cone_type == "L+":
                    pass
                elif cone_type == "L-":
                    pass
                elif cone_type == "L=":
                    pass
                elif cone_type == "Q":
                    pass
                elif cone_type == "EXP":
                    pass
                else:
                    raise Exception("Cone type ", cone_type, " is not supported.") 

        
        if keyword == "INT":
            raise Exception("INT keyword not supported.")
        
        if keyword == "CON":
            # Number and domain of affine constrained variables
            # i.e., variables of the form Ax-b \in K
            line = fp.readline()
            (ng, ncones) = [int(i) for i in line.strip().split(' ')]
            for i in range(ncones):
                line = fp.readline().strip().split(' ')
                (cone_type, cone_dim) = (line[0], int(line[1]))
                if cone_type == "F":
                    # x is unconstrained
                    pass
                elif cone_type == "L+":
                    # x >= 0
                    pass
                elif cone_type == "L-":
                    # x <= 0
                    pass
                elif cone_type == "L=":
                    # x == 0
                    pass
                elif cone_type == "Q":
                    # Quadratic cone
                    pass
                elif cone_type == "EXP":
                    # Exponential cone
                    pass
                else:
                    raise Exception("Cone type ", cone_type, " is not supported.") 

        ########################
        ## Problem data
        ########################
        if keyword == "OBJACOORD":
            # Sparse objective 
            c = np.zeros((nx, 1))
            nnz = int(fp.readline().strip())
            for i in range(nnz):
                line = fp.readline().strip().split(' ')
                c[int(line[0])] = float(line[1])
        
        if keyword == "ACOORD":
            # Sparse constraint matrix A
            nnz = int(fp.readline().strip())
            i_list = np.zeros(nnz, dtype=int)
            j_list = np.zeros(nnz, dtype=int)
            v_list = np.zeros(nnz)
            for k in range(nnz):
                line = fp.readline().strip().split(' ')
                i_list[k] = int(line[0])
                j_list[k] = int(line[1])
                v_list[k] = float(line[2])
            A = sp.sparse.csr_matrix((i_list, j_list), v_list)
        
        if keyword == "BCOORD":
            # Sparse constraint vector b
            b = np.zeros((nx, 1))
            nnz = int(fp.readline().strip())
            for i in range(nnz):
                line = fp.readline().strip().split(' ')
                b[int(line[0])] = float(line[1])



def write_cbf(model, filename):

    c         = model.c_raw if model.use_A else  model.h_raw
    A         = model.A_raw if model.use_A else  model.G_raw.T
    b         = model.b_raw if model.use_A else -model.c_raw
    cones     = model.cones
    iscomplex = model.iscomplex
    
    f = open(filename, "w")
        
    ########################
    ## File information
    ########################
    # Version number
    f.write(str(3) + "\n")

    ########################
    ## Model structure
    ########################
    if not model.use_G:
        f.write("VAR" + "\n")
        f.write(str(model.n) + " " + str(len(cones)) + "\n")
        for cone_k in cones:
            if isinstance(cone_k, qics.cones.NonNegOrthant):
                f.write("L+" + " " + str(cone_k.dim) + "\n")