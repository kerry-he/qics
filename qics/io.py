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
    while line[0] == '*' or line[0] == '"':
        line = f.readline()
        
    ##############################################################
    # Read mDim (number of linear constraints b)
    ##############################################################
    mDim = int(line.strip().split(' ')[0])

    ##############################################################
    # Read nBlock (number of blocks in X and Z)
    ##############################################################
    line = f.readline()
    nBlock = int(line.strip().split(' ')[0])
    
    ##############################################################
    # Read blockStruct (structure of blocks in X and Z; negative 
    # integer represents a diagonal block)
    ##############################################################
    line = f.readline()
    blockStruct = [int(i) for i in line.strip().split(' ')]
    
    ##############################################################
    # Read b
    ##############################################################
    line = f.readline()
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

    lineList = f.readlines()
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

    f = open(filename, "r")
    cones = []

    while True:
        ########################
        ## File information
        ########################
        keyword = f.readline().strip()
        if keyword == "VER":
            ver = int(f.readline().strip())
            if ver != 1 and ver != 2 and ver != 3:
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
                raise Exception("Invalid OBJSENSE read from .cbf file (must be MIN or MAX).")
            
        if keyword == "VAR":
            # Number and domain of variables
            # i.e., variables of the form x \in K
            line = f.readline()
            (nx, ncones) = [int(i) for i in line.strip().split(' ')]
            for i in range(ncones):
                line = f.readline().strip().split(' ')
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
            line = f.readline()
            (ng, ncones) = [int(i) for i in line.strip().split(' ')]
            for i in range(ncones):
                line = f.readline().strip().split(' ')
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
            nnz = int(f.readline().strip())
            for i in range(nnz):
                line = f.readline().strip().split(' ')
                c[int(line[0])] = float(line[1])
        
        if keyword == "ACOORD":
            # Sparse constraint matrix A
            nnz = int(f.readline().strip())
            i_list = np.zeros(nnz, dtype=int)
            j_list = np.zeros(nnz, dtype=int)
            v_list = np.zeros(nnz)
            for k in range(nnz):
                line = f.readline().strip().split(' ')
                i_list[k] = int(line[0])
                j_list[k] = int(line[1])
                v_list[k] = float(line[2])
            A = sp.sparse.csr_matrix((i_list, j_list), v_list)
        
        if keyword == "BCOORD":
            # Sparse constraint vector b
            b = np.zeros((nx, 1))
            nnz = int(f.readline().strip())
            for i in range(nnz):
                line = f.readline().strip().split(' ')
                b[int(line[0])] = float(line[1])



def write_cbf(model, filename):

    c, A, b, G, h = _compact_model(model)
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
    for cone_k in cones:
        if isinstance(cone_k, qics.cones.NonNegOrthant):
            (cone_name, cone_size) = ("L+", cone_k.n)
        if isinstance(cone_k, qics.cones.SecondOrder):
            (cone_name, cone_size) = ("Q", 1+cone_k.n)
        if isinstance(cone_k, qics.cones.PosSemidefinite):
            if cone_k.get_iscomplex():
                (cone_name, cone_size) = ("HVECPSD", cone_k.n*cone_k.n)
            else:
                (cone_name, cone_size) = ("SVECPSD", cone_k.n*(cone_k.n+1)//2)
        if isinstance(cone_k, qics.cones.ClassEntr):
            (cone_name, cone_size) = ("CE", 2 + cone_k.n)
        if isinstance(cone_k, qics.cones.ClassRelEntr):
            (cone_name, cone_size) = ("CRE", 1 + 2*cone_k.n)
        if isinstance(cone_k, qics.cones.QuantEntr):
            if cone_k.get_iscomplex():
                (cone_name, cone_size) = ("HVECQE", 2+cone_k.n*cone_k.n)
            else:
                (cone_name, cone_size) = ("SVECQE", 2+cone_k.n*(cone_k.n+1)//2)
        if isinstance(cone_k, qics.cones.QuantRelEntr):
            if cone_k.get_iscomplex():
                (cone_name, cone_size) = ("HVECQRE", 1+2*cone_k.n*cone_k.n)
            else:
                (cone_name, cone_size) = ("SVECQRE", 1+cone_k.n*(cone_k.n+1))
        if isinstance(cone_k, qics.cones.QuantCondEntr):
            if cone_k.get_iscomplex():
                (cone_name, cone_size) = ("HVECQCE", 1+cone_k.n*cone_k.n)
            else:
                (cone_name, cone_size) = ("SVECQCE", 1+cone_k.n*(cone_k.n+1)//2)
        if isinstance(cone_k, qics.cones.QuantKeyDist):
            if cone_k.get_iscomplex():
                (cone_name, cone_size) = ("HVECQKD", 1+cone_k.n*cone_k.n)
            else:
                (cone_name, cone_size) = ("SVECQKD", 1+cone_k.n*(cone_k.n+1)//2)
        if isinstance(cone_k, qics.cones.OpPerspecTr):
            if cone_k.get_iscomplex():
                (cone_name, cone_size) = ("HVECOPT", 1+2*cone_k.n*cone_k.n)
            else:
                (cone_name, cone_size) = ("SVECOPT", 1+2*cone_k.n*(cone_k.n+1)//2)
        if isinstance(cone_k, qics.cones.OpPerspecEpi):
            if cone_k.get_iscomplex():
                (cone_name, cone_size) = ("HVECOPE", 3*cone_k.n*cone_k.n)
            else:
                (cone_name, cone_size) = ("SVECOPE", 3*cone_k.n*(cone_k.n+1)//2)

        q += cone_size
        cones_string += cone_name + " " + str(cone_size) + "\n"

    if model.use_G:
        f.write("VAR" + "\n")
        f.write(str(model.n) + " " + str(1) + "\n")
        f.write("F" + " " + str(model.n) + "\n\n")

        f.write("CON" +"\n")
        f.write(str(q + model.p) + " " + str(len(cones) + 1) + "\n")
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
    for (ik, vk) in zip(c.indices, c.data):
        f.write(str(ik) + " " + str(vk) + "\n")

    if model.offset != 0.0:
        f.write("\n" + "OBJBCOORD" + "\n")
        f.write(str(model.offset) + "\n")

    if model.use_G:
        A = sp.sparse.coo_matrix(np.vstack((A, G)))
    else:
        A = sp.sparse.coo_matrix(A)
    f.write("\n" + "ACOORD" + "\n")
    f.write(str(A.nnz) + "\n")
    for (ik, jk, vk) in zip(A.row, A.col, A.data):
        f.write(str(ik) + " " + str(jk) + " " + str(-vk) + "\n")

    if model.use_G:
        b = sp.sparse.csc_matrix(np.vstack((b, h)))
    else:
        b = sp.sparse.csc_matrix(A)
    f.write("\n" + "BCOORD" + "\n")
    f.write(str(b.nnz) + "\n")
    for (ik, vk) in zip(b.indices, b.data):
        f.write(str(ik) + " " + str(vk) + "\n")

    f.close()
            
    return

from qics.vectorize import mat_to_vec, vec_to_mat

def _compact_model(model):
    if model.use_G:
        # Just need to compact columns of G
        cc = model.c_raw.copy()
        Ac = model.A_raw.copy()
        bc = model.b_raw.copy()
        Gc = _compact_matrix(model.G_raw.copy(), model)
        hc = _compact_matrix(model.h_raw.copy(), model)
    else:
        cc = _compact_matrix(model.c_raw.copy(), model)
        Ac = _compact_matrix(model.A_raw.copy().T, model).T
        bc = model.b_raw.copy()
        Gc = model.G_raw.copy()
        hc = model.h_raw.copy()

    return cc, Ac, bc, Gc, hc

def _compact_matrix(G, model):
    # Loop through columns
    Gc = []
    for i in range(G.shape[1]):
        # Loop through cones
        Gc_i = []
        for (j, cone_j) in enumerate(model.cones):
            G_ij = G[model.cone_idxs[j], [i]]
            # Loop through subvectors (if necessary)
            if isinstance(cone_j.dim, list):
                Gc_ij = []
                idxs = np.insert(np.cumsum(cone_j.dim), 0, 0)
                for k in range(len(cone_j.dim)):
                    Gc_ijk = G_ij[idxs[k]:idxs[k+1]]
                    if cone_j.type[k] == "s":
                        Gc_ijk = mat_to_vec(vec_to_mat(Gc_ijk), compact=True)
                    elif cone_j.type[k] == "h":
                        Gc_ijk = mat_to_vec(vec_to_mat(Gc_ijk, iscomplex=True), compact=True)
                    Gc_ij += [Gc_ijk] 

                Gc_ij = np.vstack(Gc_ij)

            else:
                Gc_ij = G_ij
                if cone_j.type == "s":
                    Gc_ij = mat_to_vec(vec_to_mat(Gc_ij), compact=True)
                elif cone_j.type == "h":
                    Gc_ij = mat_to_vec(vec_to_mat(Gc_ij, iscomplex=True), compact=True)
            
            Gc_i += [Gc_ij]
        Gc += [np.vstack(Gc_i)]
    Gc = np.hstack(Gc)

    return Gc