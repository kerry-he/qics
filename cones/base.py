import numpy as np
from utils import linear as lin

class BaseCone():
    def __init__(self):
        pass

    def issym(self):
        return False
    
    def zeros(self):
        out = []
        for (dim_k, type_k) in zip(self.dim, self.type):
            if type_k == 'r':
                out += [np.zeros((dim_k, 1))]
            elif type_k == 's':
                n_k = int(np.sqrt(dim_k))
                out += [np.zeros((n_k, n_k))]
            elif type_k == 'h':
                n_k = int(np.sqrt(dim_k // 2))
                out += [np.zeros((n_k, n_k), dtype=np.complex128)]
        return out

    def prox(self):
        assert self.feas_updated
        if not self.grad_updated:
            self.update_grad()

        # Proximity measure is given by <psi, H^-1 psi>
        # where psi = z/mu + g(s)
        psi  = [dual_k + grad_k for (dual_k, grad_k) in zip(self.dual, self.grad)]

        H_psi = self.zeros()
        self.invhess_prod_ip(H_psi, psi)

        return sum([lin.inp(H_psi_k, psi_k) for (H_psi_k, psi_k) in zip(H_psi, psi)])

    def set_point(self, primal, dual, a=True):
        self.primal = [primal_k * a for primal_k in primal]
        self.dual   = [dual_k   * a for dual_k   in dual]

        self.feas_updated        = False
        self.grad_updated        = False
        self.hess_aux_updated    = False
        self.invhess_aux_updated = False
        self.dder3_aux_updated   = False

    def grad_ip(self, out):
        assert self.feas_updated
        if not self.grad_updated:
            self.update_grad()

        for (out_k, grad_k) in zip(out, self.grad):
            out_k[:] = grad_k
        
        return out
    
    def precompute_mat_vec(self):
        # Indices to convert to and from compact vectors and matrices
        if self.hermitian:
            self.diag_idxs = np.append(0, np.cumsum([i for i in range(3, 2*self.n+1, 2)]))
            self.triu_idxs = np.empty(self.n*self.n, dtype=int)
            self.scale        = np.empty(self.n*self.n)
            k = 0
            for j in range(self.n):
                for i in range(j):
                    self.triu_idxs[k]     = 2 * (j + i*self.n)
                    self.triu_idxs[k + 1] = 2 * (j + i*self.n) + 1
                    self.scale[k:k+2]        = np.sqrt(2.)
                    k += 2
                self.triu_idxs[k] = 2 * (j + j*self.n)
                self.scale[k]        = 1.
                k += 1
        else:
            self.diag_idxs = np.append(0, np.cumsum([i for i in range(2, self.n+1, 1)]))
            self.triu_idxs = np.array([j + i*self.n for j in range(self.n) for i in range(j + 1)])
            self.scale = np.array([1 if i==j else np.sqrt(2.) for j in range(self.n) for i in range(j + 1)])

        # Computational basis for symmetric/Hermitian matrices
        rt2 = np.sqrt(0.5)
        self.E = np.zeros((self.vn, self.n, self.n), dtype=self.dtype)
        k = 0
        for j in range(self.n):
            for i in range(j):
                self.E[k, i, j] = rt2
                self.E[k, j, i] = rt2
                k += 1
                if self.hermitian:
                    self.E[k, i, j] = rt2 *  1j
                    self.E[k, j, i] = rt2 * -1j
                    k += 1
            self.E[k, j, j] = 1.
            k += 1

    # Functions that the child class has to implement
    def get_init_point(self, out):
        pass
    
    def get_feas(self):
        pass
    
    def update_grad(self):
        pass

    def hess_prod_ip(self, out, H):
        pass

    def hess_congr(self, A):
        pass

    def invhess_prod_ip(self, out, H):
        pass

    def invhess_congr(self, A):
        pass

    def third_dir_deriv_axpy(self, out, H, a=True):
        pass

class SymCone(BaseCone):
    def issym(self):
        return True

    # Functions that the child class has to implement
    def nt_prod_ip(self, out, H):
        pass

    def nt_congr(self, A):
        pass
    
    def invnt_prod_ip(self, out, H):
        pass

    def invnt_congr(self, A):
        pass

    def comb_dir(self, out, ds, dz, sigma_mu):
        # Compute the residual for rs where rs is given as the lhs of
        #     Lambda o (W dz + W^-T ds) = -Lambda o Lambda - (W^-T ds_a) o (W dz_a) 
        #                                 + sigma * mu * I
        # which is rearranged into the form H ds + dz = rs, i.e.,
        #     rs := W^-1 [ Lambda \ (-Lambda o Lambda - (W^-T ds_a) o (W dz_a) + sigma*mu I) ]
        # See: [Section 5.4]https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf
        pass

    def step_to_boundary(self, ds, dz):
        # Compute the maximum step alpha in [0, 1] we can take such that 
        #     s + alpha ds >= 0
        #     z + alpha dz >= 0  
        # See: [Section 8.3]https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf
        pass