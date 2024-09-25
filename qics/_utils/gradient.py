import numpy as np
import numba as nb

from qics._utils.linalg import congr_multi


@nb.njit
def D1_f(D, f_D, df_D):
    eps = np.finfo(np.float64).eps
    rteps = np.sqrt(eps)

    n = D.size
    D1 = np.empty((n, n))

    for j in range(n):
        for i in range(j):
            d_ij = D[i] - D[j]
            if abs(d_ij) < rteps:
                D1[i, j] = 0.5 * (df_D[i] + df_D[j])
            else:
                D1[i, j] = (f_D[i] - f_D[j]) / d_ij
            D1[j, i] = D1[i, j]

        D1[j, j] = df_D[j]

    return D1


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
            elif (D[i] < D[j] / 2) or (D[j] < D[i] / 2):
                D1[i, j] = (log_D[i] - log_D[j]) / d_ij
            else:
                z = d_ij / (D[i] + D[j])
                D1[i, j] = 2.0 * np.arctanh(z) / d_ij
            D1[j, i] = D1[i, j]

        D1[j, j] = np.reciprocal(D[j])

    return D1


@nb.njit
def D2_f(D, D1, d2f_D):
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
                        t = (d2f_D[i] + d2f_D[j] + d2f_D[k]) / 6
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
                        t = ((3 / (D[i] + D[j] + D[k])) ** 2) / -2
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
def D3_f_ij(i, j, D, D2, d3f_D):
    eps = np.finfo(np.float64).eps
    rteps = np.sqrt(eps)
    n = D.size
    D_i = D[i]
    D_j = D[j]

    D3_ij = np.zeros((n, n))

    for a in range(n):
        for b in range(a + 1):
            D_b = D[b]
            D_a = D[a]
            D_ij = D_i - D_j
            D_ib = D_i - D_b
            D_ia = D_i - D_a
            B_ib = abs(D_ib) < rteps
            B_ia = abs(D_ia) < rteps

            if (abs(D_ij) < rteps) and B_ib and B_ia:
                t = d3f_D[i] / 6
            elif B_ib and B_ia:
                t = (D2[i, i, i] - D2[i, i, j]) / D_ij
            elif B_ia:
                t = (D2[i, i, j] - D2[i, j, b]) / D_ib
            else:
                t = (D2[i, j, b] - D2[j, b, a]) / D_ia

            D3_ij[b, a] = t
            D3_ij[a, b] = t

    return D3_ij


@nb.njit
def D3_log_ij(i, j, D2, D, f):
    eps = np.finfo(np.float64).eps
    rteps = np.sqrt(eps)
    n = D.size
    D_i = D[i]
    D_j = D[j]

    D3_ij = np.zeros((n, n))

    for a in range(n):
        for b in range(a + 1):
            D_b = D[b]
            D_a = D[a]
            D_ij = D_i - D_j
            D_ib = D_i - D_b
            D_ia = D_i - D_a
            B_ib = abs(D_ib) < rteps
            B_ia = abs(D_ia) < rteps

            if (abs(D_ij) < rteps) and B_ib and B_ia:
                t = D_i**-3 / 3 if (f == "log") else -(D_i**-2) / 6
            elif B_ib and B_ia:
                t = (D2[i, i, i] - D2[i, i, j]) / D_ij
            elif B_ia:
                t = (D2[i, i, j] - D2[i, j, b]) / D_ib
            else:
                t = (D2[i, j, b] - D2[j, b, a]) / D_ia

            D3_ij[b, a] = t
            D3_ij[a, b] = t

    return D3_ij


def scnd_frechet(D2, UHU, UXU=None, U=None):
    n = D2.shape[0]

    D2_UXU = (D2 * UXU) if (UXU is not None) else (D2)
    out = UHU.reshape((n, 1, n)) @ D2_UXU
    out = out.reshape((n, n))
    out = out + out.conj().T
    out = (U @ out @ U.conj().T) if (U is not None) else (out)

    return out


def scnd_frechet_multi(
    out, D2, UHU, UXU=None, U=None, work1=None, work2=None, work3=None
):
    D2_UXU = (D2 * UXU) if (UXU is not None) else (D2)
    np.matmul(D2_UXU, UHU.conj().transpose(1, 2, 0), out=work3)

    if U is not None:
        np.add(work3.transpose(2, 1, 0), work3.conj().transpose(2, 0, 1), out=work1)
        congr_multi(out, U, work1, work2)
    else:
        np.add(work3.transpose(2, 1, 0), work3.conj().transpose(2, 0, 1), out=out)

    return out


@nb.njit(parallel=True)
def thrd_frechet(D, D2, d3f_D, U, H1, H2, H3=None):
    n = D.size
    out = np.zeros_like(H1)

    # If H3 is None, then assume H2=H3
    if H3 is None:
        for i in nb.prange(n):
            for j in range(i + 1):
                D3_ij = D3_f_ij(i, j, D, D2, d3f_D)

                for b in range(n):
                    for a in range(n):
                        temp  = H1[i, b] * H2[b, a] * H2[a, j]
                        temp += H2[i, b] * (H1[b, a] * H2[a, j] + H2[b, a] * H1[a, j])
                        out[i, j] = out[i, j] + D3_ij[b, a] * temp

                out[j, i] = np.conj(out[i, j])

        out *= 2

    else:
        for i in nb.prange(n):
            for j in range(i + 1):
                D3_ij = D3_f_ij(i, j, D, D2, d3f_D)

                for b in range(n):
                    for a in range(n):
                        work = H2[b, a] * H3[a, j]
                        work += H3[b, a] * H2[a, j]
                        temp  = H1[i, b] * work

                        work = H1[b, a] * H3[a, j]
                        work += H3[b, a] * H1[a, j]
                        temp += H2[i, b] * work

                        work = H1[b, a] * H2[a, j]
                        work += H2[b, a] * H1[a, j]
                        temp += H3[i, b] * work
                        out[i, j] = out[i, j] + D3_ij[b, a] * temp

                out[j, i] = np.conj(out[i, j])

    return U @ out @ U.conj().T


def get_S_matrix(D2_UXU, rt2, iscomplex=False):
    if iscomplex:
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
                row = 2 * k + j * j
                S[row, col] = D2_UXU[j, k, i].real
                S[row + 1, col] = D2_UXU[j, k, i].imag
            row = 2 * j + j * j
            S[row, col] = D2_UXU[j, j, i].real * rt2
            for k in range(j + 1, n):
                row = 2 * j + k * k
                S[row, col] = D2_UXU[j, k, i].real
                S[row + 1, col] = -D2_UXU[j, k, i].imag

            # Increment columns
            for k in range(i):
                row = 2 * k + i * i
                S[row, col] += D2_UXU[i, k, j].real
                S[row + 1, col] += D2_UXU[i, k, j].imag
            row = 2 * i + i * i
            S[row, col] = D2_UXU[i, j, i].real * rt2
            for k in range(i + 1, n):
                row = 2 * i + k * k
                S[row, col] += D2_UXU[i, k, j].real
                S[row + 1, col] += -D2_UXU[i, k, j].imag

            col += 1

            # Increment rows
            for k in range(j):
                row = 2 * k + j * j
                S[row, col] = -D2_UXU[j, k, i].imag
                S[row + 1, col] = D2_UXU[j, k, i].real
            row = 2 * j + j * j
            S[row, col] = -D2_UXU[j, j, i].imag * rt2
            for k in range(j + 1, n):
                row = 2 * j + k * k
                S[row, col] = -D2_UXU[j, k, i].imag
                S[row + 1, col] = -D2_UXU[j, k, i].real

            # Increment columns
            for k in range(i):
                row = 2 * k + i * i
                S[row, col] += D2_UXU[i, k, j].imag
                S[row + 1, col] += -D2_UXU[i, k, j].real
            row = 2 * i + i * i
            S[row, col] = -D2_UXU[i, j, i].imag * rt2
            for k in range(i + 1, n):
                row = 2 * i + k * k
                S[row, col] += D2_UXU[i, k, j].imag
                S[row + 1, col] += D2_UXU[i, k, j].real

            col += 1

        # Column corresponding to unit vector Hjj
        for k in range(j):
            row = 2 * k + j * j
            S[row, col] = D2_UXU[j, j, k].real * rt2
            S[row + 1, col] = -D2_UXU[j, j, k].imag * rt2
        row = 2 * j + j * j
        S[row, col] = 2 * D2_UXU[j, j, j].real
        for k in range(j + 1, n):
            row = 2 * j + k * k
            S[row, col] = D2_UXU[j, j, k].real * rt2
            S[row + 1, col] = D2_UXU[j, j, k].imag * rt2

        col += 1

    return S
