import numpy as np
import numba as nb

from utils.linear import congr

@nb.njit
def D1_log(D, log_D):
    eps = np.finfo(np.float64).eps
    rteps = np.sqrt(eps)

    n = D.size
    D1 = np.empty((n, n))
    
    for j in range(n):
        for i in range(j):
            d_ij = D[i] - D[j]
            if abs(d_ij) < rteps:
                D1[i, j] = 2.0 / (D[i] + D[j])
            elif (D[i] < D[j]/2) or (D[j] < D[i]/2):
                D1[i, j] = (log_D[i] - log_D[j]) / d_ij
            else:
                z = d_ij / (D[i] + D[j])
                D1[i, j] = 2.0 * np.arctanh(z) / d_ij
            D1[j, i] = D1[i, j]

        D1[j, j] = np.reciprocal(D[j])

    return D1

@nb.njit
def D1_entr(D, log_D, entr_D):
    eps = np.finfo(np.float64).eps
    rteps = np.sqrt(eps)

    n = D.size
    D1 = np.empty((n, n))
    
    for j in range(n):
        for i in range(j):
            d_ij = D[i] - D[j]
            if abs(d_ij) < rteps:
                D1[i, j] = 0.5 * (log_D[i] + log_D[j]) + 1.
            else:
                D1[i, j] = (entr_D[i] - entr_D[j]) / d_ij
            D1[j, i] = D1[i, j]

        D1[j, j] = log_D[j] + 1.

    return D1

@nb.njit
def D2_log(D, D1):
    eps = np.finfo(np.float64).eps
    rteps = np.sqrt(eps)

    n = D.size
    D2 = np.zeros((n, n, n))

    for k in range(n):
        for j in range(k + 1):
            for i in range(j + 1):
                d_jk = D[j] - D[k]
                if abs(d_jk) < rteps:
                    d_ij = D[i] - D[j]
                    if abs(d_ij) < rteps:
                        t = ((3 / (D[i] + D[j] + D[k]))**2) / -2
                    else:
                        t = (D1[i, j] - D1[j, k]) / d_ij
                else:
                    t = (D1[i, j] - D1[i, k]) / d_jk

                D2[i, j, k] = t
                D2[i, k, j] = t
                D2[j, i, k] = t
                D2[j, k, i] = t
                D2[k, i, j] = t
                D2[k, j, i] = t

    return D2

@nb.njit
def D2_entr(D, D1):
    eps = np.finfo(np.float64).eps
    rteps = np.sqrt(eps)

    n = D.size
    D2 = np.zeros((n, n, n))

    for k in range(n):
        for j in range(k + 1):
            for i in range(j + 1):
                d_jk = D[j] - D[k]
                if abs(d_jk) < rteps:
                    d_ij = D[i] - D[j]
                    if abs(d_ij) < rteps:
                        t = (3 / (D[i] + D[j] + D[k])) / 2
                    else:
                        t = (D1[i, j] - D1[j, k]) / d_ij
                else:
                    t = (D1[i, j] - D1[i, k]) / d_jk

                D2[i, j, k] = t
                D2[i, k, j] = t
                D2[j, i, k] = t
                D2[j, k, i] = t
                D2[k, i, j] = t
                D2[k, j, i] = t

    return D2

@nb.njit
def D3_log_ij(i, j, D3, D):
    eps = np.finfo(np.float64).eps
    rteps = np.sqrt(eps)
    n = D.size
    D_i = D[i]
    D_j = D[j]

    D3_ij = np.zeros((n, n))

    for l in range(n):
        for k in range(l + 1):
            D_k = D[k]
            D_l = D[l]
            D_ij = D_i - D_j
            D_ik = D_i - D_k
            D_il = D_i - D_l
            B_ik = (abs(D_ik) < rteps)
            B_il = (abs(D_il) < rteps)
    
            if (abs(D_ij) < rteps) and B_ik and B_il:
                t = D_i**-3 / 3
            elif B_ik and B_il:
                t = (D3[i, i, i] - D3[i, i, j]) / D_ij
            elif B_il:
                t = (D3[i, i, j] - D3[i, j, k]) / D_ik
            else:
                t = (D3[i, j, k] - D3[j, k, l]) / D_il
    
            D3_ij[k, l] = t
            D3_ij[l, k] = t
    
    return D3_ij    

def scnd_frechet(D2, UHU, UXU=None, U=None):
    n = D2.shape[0]

    D2_UXU = (D2 * UXU) if (UXU is not None) else (D2)
    out = UHU.reshape((n, 1, n)) @ D2_UXU
    out = out.reshape((n, n))
    out = out + out.conj().T
    out = (U @ out @ U.conj().T) if (U is not None) else (out)

    return out

def scnd_frechet_multi(out, D2, UHU, UXU=None, U=None, work1=None, work2=None, work3=None):
    D2_UXU = (D2 * UXU) if (UXU is not None) else (D2)
    np.matmul(D2_UXU, UHU.conj().transpose(1, 2, 0), out=work3)

    if U is not None:
        np.add(work3.transpose(2, 1, 0), work3.conj().transpose(2, 0, 1), out=work1)
        congr(out, U, work1, work2)
    else:
        np.add(work3.transpose(2, 1, 0), work3.conj().transpose(2, 0, 1), out=out)

    return out

def thrd_frechet(D2, D, U, H1, H2, H3):
    if (H1.dtype == 'complex128') or (H2.dtype == 'complex128') or (H3.dtype == 'complex128'):
        return thrd_frechet_complex(D2, D, U, H1, H2, H3)
    else:
        return thrd_frechet_real(D2, D, U, H1, H2, H3)

@nb.njit
def thrd_frechet_real(D2, D, U, H1, H2, H3):
    n = D.size
    out = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            D3_ij = D3_log_ij(i, j, D2, D)

            for k in range(n):
                for l in range(n):
                    temp  = H1[i, k] * H2[k, l] * H3[l, j] 
                    temp += H1[i, k] * H3[k, l] * H2[l, j] 
                    temp += H2[i, k] * H1[k, l] * H3[l, j] 
                    temp += H2[i, k] * H3[k, l] * H1[l, j] 
                    temp += H3[i, k] * H1[k, l] * H2[l, j] 
                    temp += H3[i, k] * H2[k, l] * H1[l, j]
                    out[i, j] = out[i, j] + D3_ij[k, l] * temp
            
            out[j, i] = out[i, j]

    return U @ out @ U.conj().T

@nb.njit
def thrd_frechet_complex(D2, D, U, H1, H2, H3):
    n = D.size
    out = np.zeros((n, n), 'complex128')

    for i in range(n):
        for j in range(i + 1):
            D3_ij = D3_log_ij(i, j, D2, D)

            for k in range(n):
                for l in range(n):
                    temp  = H1[i, k] * H2[k, l] * H3[l, j] 
                    temp += H1[i, k] * H3[k, l] * H2[l, j] 
                    temp += H2[i, k] * H1[k, l] * H3[l, j] 
                    temp += H2[i, k] * H3[k, l] * H1[l, j] 
                    temp += H3[i, k] * H1[k, l] * H2[l, j] 
                    temp += H3[i, k] * H2[k, l] * H1[l, j]
                    out[i, j] = out[i, j] + D3_ij[k, l] * temp
            
            out[j, i] = np.conj(out[i, j])

    return U @ out @ U.conj().T

def get_S_matrix(D2_UXU, rt2, hermitian=False):
    if hermitian:
        return get_S_matrix_hermitian(D2_UXU, rt2)
    else:
        return get_S_matrix_symmetric(D2_UXU, rt2)

@nb.njit
def get_S_matrix_symmetric(D2_UXU, rt2):
    n = D2_UXU.shape[0]
    vn = n * (n + 1) // 2
    S = np.zeros((vn, vn))
    col = 0

    for j in range(n):

        for i in range(j):
            # Column corresponding to unit vector (Hij + Hji) / sqrt(2)
            # Increment rows
            for k in range(j):
                row = k + (j * (j + 1)) // 2
                S[row, col] = D2_UXU[j, k, i]
            row = j + (j * (j + 1)) // 2    
            S[row, col] = D2_UXU[j, j, i] * rt2
            for k in range(j + 1, n):
                row = j + (k * (k + 1)) // 2
                S[row, col] = D2_UXU[j, k, i]

            # Increment columns
            for k in range(i):
                row = k + (i * (i + 1)) // 2
                S[row, col] += D2_UXU[i, k, j]
            row = i + (i * (i + 1)) // 2    
            S[row, col] = D2_UXU[i, j, i] * rt2
            for k in range(i + 1, n):
                row = i + (k * (k + 1)) // 2
                S[row, col] += D2_UXU[i, k, j]

            col += 1

        # Column corresponding to unit vector Hjj
        for k in range(j):
            row = k + (j * (j + 1)) // 2
            S[row, col] = D2_UXU[j, j, k] * rt2
        row = j + (j * (j + 1)) // 2    
        S[row, col] = 2 * D2_UXU[j, j, j]
        for k in range(j + 1, n):
            row = j + (k * (k + 1)) // 2
            S[row, col] = D2_UXU[j, j, k] * rt2

        col += 1

    return S

@nb.njit
def get_S_matrix_hermitian(D2_UXU, rt2):
    n = D2_UXU.shape[0]
    vn = n * n
    S = np.zeros((vn, vn))
    col = 0

    for j in range(n):

        for i in range(j):
            # Column corresponding to unit vector (Hij + Hji) / sqrt(2)
            # Increment rows
            for k in range(j):
                row = 2*k + j*j
                S[row    , col] = D2_UXU[j, k, i].real
                S[row + 1, col] = D2_UXU[j, k, i].imag
            row = 2*j + j*j
            S[row, col] = D2_UXU[j, j, i].real * rt2
            for k in range(j + 1, n):
                row = 2*j + k*k
                S[row    , col] =  D2_UXU[j, k, i].real
                S[row + 1, col] = -D2_UXU[j, k, i].imag

            # Increment columns
            for k in range(i):
                row = 2*k + i*i
                S[row    , col] += D2_UXU[i, k, j].real
                S[row + 1, col] += D2_UXU[i, k, j].imag
            row = 2*i + i*i
            S[row, col] = D2_UXU[i, j, i].real * rt2
            for k in range(i + 1, n):
                row = 2*i + k*k
                S[row    , col] +=  D2_UXU[i, k, j].real
                S[row + 1, col] += -D2_UXU[i, k, j].imag

            col += 1

            # Increment rows
            for k in range(j):
                row = 2*k + j*j
                S[row    , col] = -D2_UXU[j, k, i].imag
                S[row + 1, col] =  D2_UXU[j, k, i].real
            row = 2*j + j*j
            S[row, col] = -D2_UXU[j, j, i].imag * rt2
            for k in range(j + 1, n):
                row = 2*j + k*k
                S[row    , col] = -D2_UXU[j, k, i].imag
                S[row + 1, col] = -D2_UXU[j, k, i].real

            # Increment columns
            for k in range(i):
                row = 2*k + i*i
                S[row    , col] +=  D2_UXU[i, k, j].imag
                S[row + 1, col] += -D2_UXU[i, k, j].real
            row = 2*i + i*i
            S[row, col] = -D2_UXU[i, j, i].imag * rt2
            for k in range(i + 1, n):
                row = 2*i + k*k
                S[row    , col] += D2_UXU[i, k, j].imag
                S[row + 1, col] += D2_UXU[i, k, j].real

            col += 1

        # Column corresponding to unit vector Hjj
        for k in range(j):
            row = 2*k + j*j
            S[row    , col] =  D2_UXU[j, j, k].real * rt2
            S[row + 1, col] = -D2_UXU[j, j, k].imag * rt2
        row = 2*j + j*j
        S[row, col] = 2 * D2_UXU[j, j, j].real
        for k in range(j + 1, n):
            row = 2*j + k*k
            S[row    , col] = D2_UXU[j, j, k].real * rt2
            S[row + 1, col] = D2_UXU[j, j, k].imag * rt2

        col += 1

    return S