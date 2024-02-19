import numpy as np
import scipy as sp
import numba as nb
import math
from utils import symmetric as sym
from utils import linear    as lin
from utils import mtxgrad   as mgrad

class QuantKeyDist():
    def __init__(self, Klist, ZKlist, Klist_raw, Zlist_raw, hermitian=False):
        # Dimension properties
        self.nk0, self.ni = np.shape(Klist[0])  # Get input and output dimension
        self.nzk0, self.ni = np.shape(ZKlist[0])  # Get input and output dimension
        self.hermitian = hermitian
        
        self.vni = sym.vec_dim(self.ni, self.hermitian)     # Get input vector dimension

        self.dim = 1 + self.vni             # Total dimension of cone
        self.use_sqrt = False

        # Reduce systems
        KK = np.zeros((self.nk0, self.nk0), 'complex128') if hermitian else np.zeros((self.nk0, self.nk0))
        for K in Klist:
            KK += K @ K.conj().T
        ZKKZ = np.zeros((self.nzk0, self.nzk0), 'complex128') if hermitian else np.zeros((self.nzk0, self.nzk0))
        for ZK in ZKlist:
            ZKKZ += ZK @ ZK.conj().T

        Dkk, Ukk     = np.linalg.eigh(KK)
        Dzkkz, Uzkkz = np.linalg.eigh(ZKKZ)

        KKnzidx   = np.where(Dkk > 1e-12)[0]
        ZKKZnzidx = np.where(Dzkkz > 1e-12)[0]

        self.nk   = np.size(KKnzidx)
        self.nzk  = np.size(ZKKZnzidx)
        self.vnk  = sym.vec_dim(self.nk, self.hermitian)
        self.vnzk = sym.vec_dim(self.nzk, self.hermitian)

        self.Qkk   = np.eye(self.nk) if (self.nk == self.nk0) else Ukk[:, KKnzidx]
        self.Qzkkz = np.eye(self.nzk) if (self.nzk == self.nzk0) else Uzkkz[:, ZKKZnzidx]

        # Get reduced Kraus operators
        self.K_list = [self.Qkk.conj().T @ K for K in Klist]
        self.ZK_list = [self.Qzkkz.conj().T @ ZK for ZK in ZKlist]

        self.K = sym.lin_to_mat(lambda x : sym.congr_map(x, self.K_list), self.ni, self.nk, hermitian=self.hermitian)
        self.ZK = sym.lin_to_mat(lambda x : sym.congr_map(x, self.ZK_list), self.ni, self.nzk, hermitian=self.hermitian)

        # Stuff for dprBB84
        self.Klist_raw  = Klist_raw
        self.Zlist_raw  = Zlist_raw
        self.ZKlist_raw = [Z @ K for K in Klist_raw for Z in Zlist_raw]

        # Stuff for DMCV 
        # self.ZKlist_reduced = [ZK[i::4, :] for (i, ZK) in enumerate(self.ZK_list)]

        # Update flags
        self.feas_updated        = False
        self.grad_updated        = False
        self.hess_aux_updated    = False
        self.invhess_aux_updated = False
        self.dder3_aux_updated   = False

        return
        
    def get_nu(self):
        return 1 + self.ni
    
    def set_init_point(self):
        point = np.empty((self.dim, 1))
        point[0] = 1.
        point[1:] = sym.mat_to_vec(np.eye(self.ni), hermitian=self.hermitian)

        self.set_point(point)

        return point
    
    def set_point(self, point):
        assert np.size(point) == self.dim
        self.point = point

        self.t   = point[0]
        self.X   = sym.vec_to_mat(point[1:], hermitian=self.hermitian)
        self.KX  = sym.congr_map(self.X, self.K_list)
        self.ZKX = sym.congr_map(self.X, self.ZK_list)

        self.feas_updated        = False
        self.grad_updated        = False
        self.hess_aux_updated    = False
        self.invhess_aux_updated = False
        self.dder3_aux_updated   = False

        return
    
    def get_feas(self):
        if self.feas_updated:
            return self.feas
        
        self.feas_updated = True

        self.Dx, self.Ux     = np.linalg.eigh(self.X)
        if any(self.Dx <= 0):
            self.feas = False
            return self.feas

        self.Dkx, self.Ukx   = np.linalg.eigh(self.KX)
        self.Dzkx, self.Uzkx = np.linalg.eigh(self.ZKX)

        # for K in self.ZKlist_reduced:
        #     KX  = K @ self.X @ K.conj().T
        #     Dkx, Ukx   = np.linalg.eigh(KX)

        #     if any(Dkx <= 0):
        #         self.feas = False
        #         return self.feas
        
        # Shouldn't be necessary as K and Z are both positive, but just to be safe against numerical imprecisions
        if any(self.Dkx <= 0) or any(self.Dzkx <= 0):
            self.feas = False
            return self.feas

        self.log_Dkx  = np.log(self.Dkx)
        self.log_Dzkx = np.log(self.Dzkx)

        entr_KX  = lin.inp(self.Dkx, self.log_Dkx)
        entr_ZKX = lin.inp(self.Dzkx, self.log_Dzkx)

        self.z = self.t - (entr_KX - entr_ZKX)

        self.feas = (self.z > 0)
        return self.feas

    def get_val(self):
        assert self.feas_updated

        return -np.log(self.z) - np.sum(np.log(self.Dx))
    
    def get_grad(self):
        assert self.feas_updated

        if self.grad_updated:
            return self.grad
        
        log_KX  = (self.Ukx * self.log_Dkx) @ self.Ukx.conj().T
        log_ZKX = (self.Uzkx * self.log_Dzkx) @ self.Uzkx.conj().T

        self.K_log_KX = sym.congr_map(log_KX + np.eye(self.nk), self.K_list, adjoint=True)
        self.ZK_log_ZKX = sym.congr_map(log_ZKX + np.eye(self.nzk), self.ZK_list, adjoint=True)

        self.inv_Dx = np.reciprocal(self.Dx)
        self.inv_X  = (self.Ux * self.inv_Dx) @ self.Ux.conj().T

        self.zi   = np.reciprocal(self.z)
        self.DPhi = sym.mat_to_vec(self.K_log_KX - self.ZK_log_ZKX, hermitian=self.hermitian)

        self.grad     =  np.empty((self.dim, 1))
        self.grad[0]  = -self.zi
        self.grad[1:] =  self.zi * self.DPhi - sym.mat_to_vec(self.inv_X, hermitian=self.hermitian)

        self.grad_updated = True
        return self.grad
    
    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        irt2 = math.sqrt(0.5)

        self.D1kx_log  = mgrad.D1_log(self.Dkx, self.log_Dkx)
        self.D1zkx_log = mgrad.D1_log(self.Dzkx, self.log_Dzkx)

        # Hessians of quantum relative entropy
        invXX  = np.empty((self.vni, self.vni))
        k = 0
        for j in range(self.ni):
            for i in range(j + 1):
                # invXX
                temp = self.inv_X[:, [i]] @ self.inv_X[[j], :]
                if i != j:
                    temp *= irt2
                    invXX[:, [k]] = sym.mat_to_vec(temp + temp.conj().T, hermitian=self.hermitian)
                    k += 1
                    if self.hermitian:
                        temp *= 1j
                        invXX[:, [k]] = sym.mat_to_vec(temp + temp.conj().T, hermitian=self.hermitian)                        
                        k += 1
                else:
                    invXX[:, [k]] = sym.mat_to_vec(temp, hermitian=self.hermitian)
                    k += 1

        if self.hermitian:
            real_complex_factor = np.ones(self.nk) + (np.ones(self.nk) - np.eye(self.nk)) * 1j          # Ugly factor to get the correct diagonal matrix
        else:
            real_complex_factor = np.ones(self.nk)
        sqrt_D1kx_log = np.sqrt(sym.mat_to_vec(self.D1kx_log * real_complex_factor, 1.0, hermitian=self.hermitian))
        UnUnK = np.empty((self.vni, self.vnk))
        for k in range(self.vni):
            K = sym.vec_to_mat(self.K[:, [k]], hermitian=self.hermitian)
            UnKUn = self.Ukx.conj().T @ K @ self.Ukx
            UnUnK[[k], :] = (sym.mat_to_vec(UnKUn, hermitian=self.hermitian) * sqrt_D1kx_log).T
        D2PhiKX = UnUnK @ UnUnK.T

        if self.hermitian:
            real_complex_factor = np.ones(self.nzk) + (np.ones(self.nzk) - np.eye(self.nzk)) * 1j          # Ugly factor to get the correct diagonal matrix
        else:
            real_complex_factor = np.ones(self.nzk)        
        sqrt_D1zkx_log = np.sqrt(sym.mat_to_vec(self.D1zkx_log * real_complex_factor, 1.0, hermitian=self.hermitian))
        UxkUxkZK = np.empty((self.vni, self.vnzk))
        for k in range(self.vni):
            ZK = sym.vec_to_mat(self.ZK[:, [k]], hermitian=self.hermitian)
            UzkZKUzk = self.Uzkx.conj().T @ ZK @ self.Uzkx
            UxkUxkZK[[k], :] = (sym.mat_to_vec(UzkZKUzk, hermitian=self.hermitian) * sqrt_D1zkx_log).T
        D2PhiZKX = UxkUxkZK @ UxkUxkZK.T 

        # Preparing other required variables
        self.zi2 = self.zi * self.zi
        D2Phi = D2PhiKX - D2PhiZKX

        self.hess = self.zi * D2Phi + invXX

        # self.hess = np.empty((self.dim, self.dim))
        # self.hess[0, 0] = self.zi2
        # self.hess[1:, [0]] = -self.zi2 * self.DPhi
        # self.hess[[0], 1:] = self.hess[1:, [0]].T
        # self.hess[1:, 1:] = self.zi2 * np.outer(self.DPhi, self.DPhi) + self.zi * D2Phi + invXX

        self.hess_aux_updated = True

        return
    
    def update_hessprod_aux_dprBB84(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        irt2 = math.sqrt(0.5)

        self.zi2 = self.zi * self.zi
        self.D1kx_log  = mgrad.D1_log(self.Dkx, self.log_Dkx)
        self.D1zkx_log = mgrad.D1_log(self.Dzkx, self.log_Dzkx)

        # Hessians of quantum relative entropy
        self.hess = np.empty((self.vni, self.vni))
        k = 0
        for j in range(self.ni):
            for i in range(j + 1):
                # invXX
                temp = self.inv_X[:, [i]] @ self.inv_X[[j], :]
                if i != j:
                    temp *= irt2
                    self.hess[:, [k]] = sym.mat_to_vec(temp + temp.conj().T, hermitian=self.hermitian)
                    k += 1
                    if self.hermitian:
                        temp *= 1j
                        self.hess[:, [k]] = sym.mat_to_vec(temp + temp.conj().T, hermitian=self.hermitian)                        
                        k += 1
                else:
                    self.hess[:, [k]] = sym.mat_to_vec(temp, hermitian=self.hermitian)
                    k += 1
                

        def make_KHK(Klist, c, out):
            for K in Klist:
                (_, Kj, _) = sp.sparse.find(sp.sparse.coo_matrix(K))
                Kj = np.sort(Kj)
                nk = Kj.size
                Kx = self.X[np.ix_(Kj, Kj)] / 2
                Dkx, Ukx = np.linalg.eigh(Kx)
                D1kx_log = mgrad.D1_log(Dkx, np.log(Dkx))

                # Build matrix
                Hkx = build_frechet(Ukx, D1kx_log) / 2 / 2

                # Get indices
                k = 0
                K = np.zeros((nk * nk,), dtype='uint32')
                for j in range(nk):
                    for i in range(j):
                        (I, J) = (Kj[i], Kj[j]) # Indices of full matrix
                        K[k]     = 2*I + J*J
                        K[k + 1] = 2*I + J*J + 1
                        k += 2
                    
                    J = Kj[j]
                    K[k] = 2*J + J*J
                    k += 1

                out[np.ix_(K, K)] += Hkx * c
            
            return out
        
        def build_frechet(U, D1, hermitian=True):
            n = U.shape[0]
            rt2 = np.sqrt(2.0)

            Hess = np.zeros((n*n, n*n))

            k = 0
            for j in range(n):
                for i in range(j):
                    UHU = U.conj().T[:, [i]] @ U[[j], :] / rt2
                    D_H = U @ (D1 * UHU) @ U.conj().T
                    
                    Hess[:, [k]]     = sym.mat_to_vec(D_H + D_H.conj().T, rt2, hermitian)
                    D_H             *= 1j
                    Hess[:, [k + 1]] = sym.mat_to_vec(D_H + D_H.conj().T, rt2, hermitian)
                    k += 2

                UHU          = U.conj().T[:, [j]] @ U[[j], :]
                D_H          = U @ (D1 * UHU) @ U.conj().T
                Hess[:, [k]] = sym.mat_to_vec(D_H, rt2, hermitian)
                k += 1

            return Hess

        make_KHK(self.Klist_raw, self.zi, self.hess)
        make_KHK(self.ZKlist_raw, -self.zi, self.hess)

        # Preparing other required variables
        self.hess_aux_updated = True

        return
    
    def update_hessprod_aux(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        irt2 = math.sqrt(0.5)

        self.zi2 = self.zi * self.zi
        self.D1kx_log  = mgrad.D1_log(self.Dkx, self.log_Dkx)
        self.D1zkx_log = mgrad.D1_log(self.Dzkx, self.log_Dzkx)


        # Get indices for projection mappings
        def small_to_big_idx(K_small):
            # Get indices
            k = 0
            nk = K_small.size
            K_big = np.zeros((nk * nk,), dtype='uint32')
            for j in range(nk):
                for i in range(j):
                    (I, J) = (K_small[i], K_small[j]) # Indices of full matrix
                    K_big[k]     = 2*I + J*J
                    K_big[k + 1] = 2*I + J*J + 1
                    k += 2
                
                J = K_small[j]
                K_big[k] = 2*J + J*J
                k += 1

            return K_big
        
        K_small = [np.sort(sp.sparse.find(sp.sparse.coo_matrix(K))[1]) for K in self.Klist_raw]
        K_big   = [small_to_big_idx(K) for K in K_small]
        self.K_big_all = np.array(K_big).ravel()
        nK = K_big[0].size


        # Hessians of quantum relative entropy
        X00 = self.X[np.ix_(K_small[0], K_small[0])]
        X01 = self.X[np.ix_(K_small[1], K_small[0])]
        X11 = self.X[np.ix_(K_small[1], K_small[1])]
        
        def make_XX(X, out):
            k = 0
            for j in range(X.shape[0]):
                for i in range(j + 1):
                    # invXX
                    temp = X[:, [i]] @ X.conj().T[[j], :]
                    if i != j:
                        temp *= irt2
                        out[:, [k]] = sym.mat_to_vec(temp + temp.conj().T, hermitian=True)
                        k += 1
                        if self.hermitian:
                            temp *= 1j
                            out[:, [k]] = sym.mat_to_vec(temp + temp.conj().T, hermitian=True)                        
                            k += 1
                    else:
                        out[:, [k]] = sym.mat_to_vec(temp, hermitian=True)
                        k += 1

            return out    
        
        small_XX = np.empty((nK * 2, nK * 2))
        make_XX(X00, small_XX[:nK, :nK])
        make_XX(X11, small_XX[nK:, nK:])
        make_XX(X01, small_XX[nK:, :nK])
        small_XX[:nK, nK:] = small_XX[nK:, :nK].T
                

        def make_KHK(Klist, c, out):
            for K in Klist:
                (_, Kj, _) = sp.sparse.find(sp.sparse.coo_matrix(K))
                Kj = np.sort(Kj)
                nk = Kj.size
                Kx = self.X[np.ix_(Kj, Kj)] / 2
                Dkx, Ukx = np.linalg.eigh(Kx)
                D1kx_log = mgrad.D1_log(Dkx, np.log(Dkx))

                # Build matrix
                Hkx = build_frechet(Ukx, D1kx_log) / 2 / 2

                # Get indices
                k = 0
                K = np.zeros((nk * nk,), dtype='uint32')
                for j in range(nk):
                    for i in range(j):
                        (I, J) = (Kj[i], Kj[j]) # Indices of full matrix
                        K[k]     = 2*I + J*J
                        K[k + 1] = 2*I + J*J + 1
                        k += 2
                    
                    J = Kj[j]
                    K[k] = 2*J + J*J
                    k += 1

                out[np.ix_(K, K)] += Hkx * c
            
            return out
        
        def build_frechet(U, D1, hermitian=True):
            n = U.shape[0]
            rt2 = np.sqrt(2.0)

            Hess = np.zeros((n*n, n*n))

            k = 0
            for j in range(n):
                for i in range(j):
                    UHU = U.conj().T[:, [i]] @ U[[j], :] / rt2
                    D_H = U @ (D1 * UHU) @ U.conj().T
                    
                    Hess[:, [k]]     = sym.mat_to_vec(D_H + D_H.conj().T, rt2, hermitian)
                    D_H             *= 1j
                    Hess[:, [k + 1]] = sym.mat_to_vec(D_H + D_H.conj().T, rt2, hermitian)
                    k += 2

                UHU          = U.conj().T[:, [j]] @ U[[j], :]
                D_H          = U @ (D1 * UHU) @ U.conj().T
                Hess[:, [k]] = sym.mat_to_vec(D_H, rt2, hermitian)
                k += 1

            return Hess

        temp = np.zeros((self.vni, self.vni))

        make_KHK(self.Klist_raw, self.zi, temp)
        make_KHK(self.ZKlist_raw, -self.zi, temp)

        temp = temp[np.ix_(self.K_big_all, self.K_big_all)]

        M1 = temp[:nK, :nK]
        M2 = temp[nK:, nK:]

        D1, U1 = np.linalg.eigh(M1)
        # cutoff = np.max(D1) * 1e-14
        idx = np.where(D1 <= 1e-8)[0]
        D1 = np.delete(D1, idx)
        U1 = np.delete(U1, idx, 1)

        D2, U2 = np.linalg.eigh(M2)
        # cutoff = np.max(D2) * 1e-14
        idx = np.where(D2 <= 1e-8)[0]
        D2 = np.delete(D2, idx)
        U2 = np.delete(U2, idx, 1)

        # print()
        # print(np.min(D1), np.max(D1))
        # print(np.min(D2), np.max(D2))
        # print()

        self.U12 = sp.linalg.block_diag(U1, U2)
        D12_inv = np.hstack((np.reciprocal(D1), np.reciprocal(D2)))

        self.schur = self.U12.T @ small_XX @ self.U12
        self.schur[np.diag_indices_from(self.schur)] += D12_inv

        self.schur_fact = lin.fact(self.schur)
        
        # Preparing other required variables
        self.hess_aux_updated = True

        return    

    def update_hessprod_aux_DMCV(self):
        assert not self.hess_aux_updated
        assert self.grad_updated

        irt2 = math.sqrt(0.5)

        self.zi2 = self.zi * self.zi
        self.D1kx_log  = mgrad.D1_log(self.Dkx, self.log_Dkx)
        self.D1zkx_log = mgrad.D1_log(self.Dzkx, self.log_Dzkx)

        # Hessians of quantum relative entropy
        self.hess = np.empty((self.vni, self.vni))
        k = 0
        for j in range(self.ni):
            for i in range(j + 1):
                # invXX
                temp = self.inv_X[:, [i]] @ self.inv_X[[j], :]
                if i != j:
                    temp *= irt2
                    self.hess[:, [k]] = sym.mat_to_vec(temp + temp.conj().T, hermitian=self.hermitian)
                    k += 1
                    if self.hermitian:
                        temp *= 1j
                        self.hess[:, [k]] = sym.mat_to_vec(temp + temp.conj().T, hermitian=self.hermitian)                        
                        k += 1
                else:
                    self.hess[:, [k]] = sym.mat_to_vec(temp, hermitian=self.hermitian)
                    k += 1
                

        def make_KHK(Klist, c, out):
            for K in Klist:
                Kx = K @ self.X @ K.conj().T
                Dkx, Ukx = np.linalg.eigh(Kx)
                D1kx_log = mgrad.D1_log(Dkx, np.log(Dkx))

                # Build matrix
                Hkx = build_frechet(Ukx, D1kx_log, K)

                out += Hkx * c
            
            return out
        
        def build_frechet(U, D1, K, hermitian=True):
            n = K.shape[1]
            rt2 = np.sqrt(2.0)

            Hess = np.zeros((n*n, n*n))

            KU = K.conj().T @ U

            k = 0
            for j in range(n):
                for i in range(j):
                    UHU = KU.conj().T[:, [i]] @ KU[[j], :] / rt2
                    D_H = KU @ (D1 * UHU) @ KU.conj().T
                    
                    Hess[:, [k]]     = sym.mat_to_vec(D_H + D_H.conj().T, rt2, hermitian)
                    D_H             *= 1j
                    Hess[:, [k + 1]] = sym.mat_to_vec(D_H + D_H.conj().T, rt2, hermitian)
                    k += 2

                UHU          = KU.conj().T[:, [j]] @ KU[[j], :]
                D_H          = KU @ (D1 * UHU) @ KU.conj().T
                Hess[:, [k]] = sym.mat_to_vec(D_H, rt2, hermitian)
                k += 1

            return Hess

        make_KHK(self.K_list, self.zi, self.hess)
        make_KHK(self.ZKlist_reduced, -self.zi, self.hess)

        # Preparing other required variables
        self.hess_aux_updated = True

        return

    def hess_prod(self, dirs):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        for k in range(p):
            Ht = dirs[0, k]
            Hx = sym.vec_to_mat(dirs[1:, [k]], hermitian=self.hermitian)
            Hx_vec = dirs[1:, [k]]

            K_H = sym.congr_map(Hx, self.K_list)
            ZK_H = sym.congr_map(Hx, self.ZK_list)

            UkKHUk    = self.Ukx.conj().T @ K_H @ self.Ukx
            UkzZKHUkz = self.Uzkx.conj().T @ ZK_H @ self.Uzkx

            # Hessian product of conditional entropy
            D2PhiH  = sym.congr_map(self.Ukx @ (self.D1kx_log * UkKHUk) @ self.Ukx.conj().T, self.K_list, adjoint=True)
            D2PhiH -= sym.congr_map(self.Uzkx @ (self.D1zkx_log * UkzZKHUkz) @ self.Uzkx.conj().T, self.ZK_list, adjoint=True)
            
            # Hessian product of barrier function
            out[0, k] = (Ht - lin.inp(self.DPhi, Hx_vec)) * self.zi2

            out[1:, [k]]  = -out[0, k] * self.DPhi
            out[1:, [k]] +=  sym.mat_to_vec(self.zi * D2PhiH + self.inv_X @ Hx @ self.inv_X, hermitian=self.hermitian)

        return out

        return self.hess @ dirs
    
    def update_invhessprod_aux(self):
        assert not self.invhess_aux_updated
        assert self.grad_updated
        assert self.hess_aux_updated

        self.hess_fact = lin.fact(self.hess)

        self.invhess_aux_updated = True

        return

    def invhess_prod(self, dirs):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.invhess_aux_updated:
            self.update_invhessprod_aux()

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        Ht = dirs[[0], :]
        Hx = dirs[1:, :]

        out[1:, :] = lin.fact_solve(self.hess_fact, Hx + Ht * self.DPhi)
        out[[0], :] = self.z * self.z * Ht + np.sum(out[1:, :] * self.DPhi, axis=0)

        return out
    
    def invhess_prod(self, dirs):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        # if not self.invhess_aux_updated:
        #     self.update_invhessprod_aux()

        p = np.size(dirs, 1)
        out = np.empty((self.dim, p))

        Ht = dirs[[0], :]
        Hx = dirs[1:, :]
        Wx = Hx + Ht * self.DPhi

        temp_vec = np.zeros((self.K_big_all.size, p))

        for k in range(p):
            Wx_k = sym.vec_to_mat(Wx[:, [k]], hermitian=self.hermitian)

            temp = self.X @ Wx_k @ self.X
            temp = sym.mat_to_vec(temp, hermitian=self.hermitian)
            temp_vec[:, [k]] = temp[self.K_big_all]

        temp_vec = self.U12.T @ temp_vec
        temp_vec = lin.fact_solve(self.schur_fact, temp_vec)
        temp_vec = self.U12 @ temp_vec

        Wx[self.K_big_all] -= temp_vec

        for k in range(p):
            Wx_k = sym.vec_to_mat(Wx[:, [k]], hermitian=self.hermitian)

            temp = self.X @ Wx_k @ self.X

            outX = sym.mat_to_vec(temp, hermitian=self.hermitian)
            outt = self.z * self.z * Ht[0, k] + self.DPhi.T @ outX

            out[0, k] = outt
            out[1:, [k]] = outX

        return out

    def update_dder3_aux(self):
        assert not self.dder3_aux_updated
        assert self.hess_aux_updated

        self.D2kx_log   = mgrad.D2_log(self.Dkx, self.D1kx_log)
        self.D2zkx_log  = mgrad.D2_log(self.Dzkx, self.D1zkx_log)

        self.dder3_aux_updated = True

        return

    def third_dir_deriv(self, dirs):
        assert self.grad_updated
        if not self.hess_aux_updated:
            self.update_hessprod_aux()
        if not self.dder3_aux_updated:
            self.update_dder3_aux()

        Ht   = dirs[0]
        Hx   = sym.vec_to_mat(dirs[1:, [0]], hermitian=self.hermitian)
        Hkx  = sym.congr_map(Hx, self.K_list)
        Hzkx = sym.congr_map(Hx, self.ZK_list)        
        Hx_vec = dirs[1:, [0]]

        UkxHkxUkx    = self.Ukx.conj().T @ Hkx @ self.Ukx
        UzkxHzkxUzkx = self.Uzkx.conj().T @ Hzkx @ self.Uzkx

        # Quantum conditional entropy oracles
        D2PhiH  = sym.congr_map(self.Ukx @ (self.D1kx_log * UkxHkxUkx) @ self.Ukx.conj().T, self.K_list, adjoint=True)
        D2PhiH -= sym.congr_map(self.Uzkx @ (self.D1zkx_log * UzkxHzkxUzkx) @ self.Uzkx.conj().T, self.ZK_list, adjoint=True)

        D3PhiHH  = sym.congr_map(mgrad.scnd_frechet(self.D2kx_log * UkxHkxUkx, UkxHkxUkx, U=self.Ukx), self.K_list, adjoint=True)
        D3PhiHH -= sym.congr_map(mgrad.scnd_frechet(self.D2zkx_log * UzkxHzkxUzkx, UzkxHzkxUzkx, U=self.Uzkx), self.ZK_list, adjoint=True)

        # Third derivative of barrier
        DPhiH = lin.inp(self.DPhi, Hx_vec)
        D2PhiHH = lin.inp(D2PhiH, Hx)
        chi = Ht - DPhiH

        dder3 = np.zeros((self.dim, 1))
        dder3[0] = -2 * (self.zi**3) * (chi**2) - (self.zi**2) * D2PhiHH

        dder3[1:] = -dder3[0] * self.DPhi
        temp  = -2 * (self.zi**2) * chi * D2PhiH
        temp += self.zi * D3PhiHH
        temp -= 2 * self.inv_X @ Hx @ self.inv_X @ Hx @ self.inv_X
        dder3[1:] += sym.mat_to_vec(temp, hermitian=self.hermitian)

        return dder3
    
    def norm_invhess(self, x):
        return 0.0