from utils import linear as lin

class BaseCone():
    def __init__(self):
        pass

    def issym(self):
        return False
    
    def zeros(self):
        pass

    def prox(self):
        assert self.feas_updated
        if not self.grad_updated:
            self.update_grad()

        psi  = [dual_k + grad_k for (dual_k, grad_k) in zip(self.dual, self.grad)]
        temp = self.zeros()
        self.invhess_prod_ip(temp, psi)
        return lin.inp(temp[0], psi[0]) + lin.inp(temp[1], psi[1]) 

    # Functions that the child class has to implement
    def get_init_point(self, out):
        pass
    
    def set_point(self, point, dual, a=True):
        pass
    
    def get_feas(self):
        pass
    
    def update_grad(self):
        pass

    def get_grad(self, out):
        pass

    def hess_prod_ip(self, out, H):
        pass

    def hess_congr(self, A):
        pass

    def hess_mtx(self):
        pass

    def invhess_prod_ip(self, out, H):
        pass

    def invhess_congr(self, A):
        pass

    def invhess_mtx(self):
        pass

    def third_dir_deriv_axpy(self, out, dir, a=True):
        pass

class SymCone(BaseCone):
    def issym(self):
        return True

    # Functions that the child class has to implement
    def nt_prod_ip(self, out, H):
        pass

    def nt_congr(self, A):
        pass
     
    def nt_mtx(self):
        pass
    
    def invnt_prod_ip(self, out, H):
        pass

    def invnt_congr(self, A):
        pass
    
    def invnt_mtx(self):
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