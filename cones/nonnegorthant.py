import numpy as np

class NonNegOrthant():
    def __init__(self, dim):
        # Dimension properties
        self.dim = dim
        self.use_sqrt = True
        return
        
    def get_nu(self):
        return self.dim
    
    def set_init_point(self):
        point = np.ones((self.dim, 1))
        self.set_point(point)
        return point
    
    def set_point(self, point):
        assert np.size(point) == self.dim
        self.point = point
        return
    
    def get_feas(self):
        return all(self.point > 0)
    
    def get_val(self):
        return -sum(np.log(self.point))    
    
    def get_grad(self):
        return -1 / self.point

    def hess_prod(self, dirs):
        return dirs / (self.point**2)

    def sqrt_hess_prod(self, dirs):
        return dirs / (self.point)        

    def invhess_prod(self, dirs):
        return dirs * (self.point**2)

    def sqrt_invhess_prod(self, dirs):
        return dirs * (self.point)        

    def third_dir_deriv(self, dirs):
        return -2 * (dirs**2) / (self.point**3)

    def norm_invhess(self, x):
        return 0.0