import numpy as np
import scipy as sp
import math
import time

from utils import linear as lin
from utils import vector as vec
from solver.stepper.basic import BasicStepper
from solver.stepper.aggressive import AggressiveStepper
from solver.stepper.comb import CombinedStepper
from solver.stepper.cvxopt import CVXOPTStepper
from solver.syssolver import SysSolver

from utils import symmetric as sym

class Solver():
    def __init__(
        self, 
        model, 
        max_iter = 1000, 
        max_time = np.inf,
        tol_gap = 1e-8,
        tol_feas = 1e-8,
        tol_infeas = 1e-12,
        tol_ip = 1e-13,
        verbose  = True,
        stepper = None,
        ir = True,
        sym = False
    ):
        self.max_iter = max_iter
        self.max_time = max_time        
        self.verbose = verbose

        self.tol_gap = tol_gap
        self.tol_feas = tol_feas
        self.tol_infeas = tol_infeas
        self.tol_ip = tol_ip

        self.model = model
        syssolver = SysSolver(model, ir=ir, sym=sym)
        self.stepper = CombinedStepper(syssolver, model) if (stepper is None) else stepper

        return
    
    def solve(self):
        self.setup_solver()
        # self.rescale_model()
        self.setup_point()

        while True:
            if self.step_and_check():
                break

        # Get solve data
        self.get_gap_feas()

        if self.verbose:
            if not self.status == "step_failure":
                print("%5d" % (self.num_iters), " %8.1e" % (self.mu), " %8.1e" % (self.point.tau), " %8.1e" % (self.point.kap),
                    " | %10.3e" % (self.p_obj), " %10.3e" % (self.d_obj), " %10.3e" % (self.gap), 
                    " | %10.3e" % (self.x_feas), " %10.3e" % (self.y_feas), " %10.3e" % (self.z_feas), end="")

            print()
            print("Opt value:  %.10f" % (self.p_obj))
            print("Tolerance:  %.10e" % (self.gap))
            print("Solve time: %.10f" % (time.time() - self.solve_time), " seconds")
            print()

        return
    
    def step_and_check(self):
        # Check termination
        if self.num_iters >= self.max_iter:
            self.status = "max_iter"
            if self.verbose:
                print("Maximum iteration limit reached")
            return True
        
        if time.time() - self.solve_time >= self.max_time:
            self.status = "max_time"
            if self.verbose:
                print("Maximum time limit reached")
            return True
        
        # Update barrier parameter mu
        self.calc_mu()

        # Get solve data
        self.get_gap_feas()      

        # Check optimality
        if self.gap <= self.tol_gap:
            # Check feasibility
            if self.x_feas <= self.tol_feas and self.y_feas <= self.tol_feas and self.z_feas <= self.tol_feas:
                self.status = "solved"
                if self.verbose:
                    print("Solved to desired tolerance")
                return True

        # Check infeasibility
        if self.x_infeas <= self.tol_infeas:
            self.status = "primal_infeas"
            if self.verbose:
                print("Detected primal infeasibility")
            return True       

        if self.y_infeas <= self.tol_infeas and self.z_infeas <= self.tol_infeas:
            self.status = "dual_infeas"
            if self.verbose:
                print("Detected dual infeasibility")
            return True            
            
        # Check ill-posedness
        if self.mu <= self.tol_ip and self.point.tau <= self.tol_ip * min([1., self.point.kap]):
            self.status = "ill_posed"
            if self.verbose:
                print("Detected ill-posed problem")     
            return True       

        if self.verbose:
            if self.num_iters % 20 == 0:
                print("===================================================================================================================================================================")
                print("%5s" % "iter", " %8s" % "mu", " %8s" % "tau", " %8s" % "kap",
                    " | %10s" % "p_obj", " %10s" % "d_obj", " %10s" % "gap", 
                    " | %10s" % "x_feas", " %10s" % "y_feas", " %10s" % "z_feas",
                    " | %6s" % "step", "%10s" % "dir_tol", "%10s" % "prox", " %5s" % "alpha")
                print("===================================================================================================================================================================")                
            
            print("%5d" % (self.num_iters), " %8.1e" % (self.mu), " %8.1e" % (self.point.tau), " %8.1e" % (self.point.kap),
                  " | %10.3e" % (self.p_obj), " %10.3e" % (self.d_obj), " %10.3e" % (self.gap),
                  " | %10.3e" % (self.x_feas), " %10.3e" % (self.y_feas), " %10.3e" % (self.z_feas), end="")
            
        # Step
        self.point, success = self.stepper.step(self.model, self.point, (self.x_res, self.y_res, self.z_res, self.tau_res), self.mu, self.verbose)

        if not success:
            if self.verbose:
                self.status = "step_failure"
                print("\nFailed to step")
                return True            

        self.num_iters += 1

        return False

    def setup_solver(self):
        self.num_iters = 0
        self.solve_time = time.time()

        model = self.model

        self.c_max = lin.norm_inf(model.c)
        self.b_max = lin.norm_inf(model.b)
        self.h_max = lin.norm_inf(model.h)

        return

    def setup_point(self):
        model = self.model
        self.point = vec.Point(model)

        self.point.tau[:] = 1.
        self.point.kap[:] = 1.

        # # Precompute
        # if model.use_G:
        #     GG = model.G_T @ model.G
        #     if sp.sparse.issparse(GG):
        #         GG = sp.sparse.csc_matrix(GG)
        #     GG_fact = lin.fact(GG)

        #     # Initialize primal variables x and s
        #     x = lin.fact_solve(GG_fact, model.G_T @ model.h)
        #     s = model.h - model.G @ x

        #     self.point.s.from_vec(s)
        #     eig = min(np.linalg.eigvalsh(self.point.s[0].data))
        #     if eig < 0:
        #         self.point.s[0].data[np.diag_indices_from(self.point.s[0].data)] += -eig + 1            
        #     self.point.x = x
            

        #     # Initialize dual variables y and z
        #     x = lin.fact_solve(GG_fact, -model.c)
        #     z = model.G @ x

        #     self.point.z.from_vec(z)
        #     eig = min(np.linalg.eigvalsh(self.point.z[0].data))
        #     if eig < 0:
        #         self.point.z[0].data[np.diag_indices_from(self.point.z[0].data)] += -eig + 1
        # else:
        #     AA = model.A @ model.A_T
        #     if sp.sparse.issparse(AA):
        #         AA = sp.sparse.csc_matrix(AA)
        #     AA_fact = lin.fact(AA)

        #     # Initialize primal variables x and s
        #     y = lin.fact_solve(AA_fact, -model.A @ model.h - model.b)
        #     s = -model.A.T @ y
        #     x = s - model.h

        #     self.point.s.from_vec(s)
        #     self.point.x = x
            

        #     # Initialize dual variables y and z
        #     y = lin.fact_solve(AA_fact, -model.A @ model.c)
        #     z = model.A.T @ y

        #     self.point.z.from_vec(z)
        #     eig = min(np.linalg.eigvalsh(self.point.z[0].data))
        #     if eig < 0:
        #         self.point.z[0].data[np.diag_indices_from(self.point.z[0].data)] += -eig + 1
        #     self.point.y = y

        
        # for (k, cone_k) in enumerate(model.cones):
        #     cone_k.set_point(self.point.s[k], self.point.z[k], 1.0)
        #     assert cone_k.get_feas()
        #     cone_k.get_grad()


        for (k, cone_k) in enumerate(model.cones):
            np.copyto(self.point.s[k], cone_k.set_init_point())
            assert cone_k.get_feas()
            np.copyto(self.point.z[k], -cone_k.get_grad())

        # if model.use_G:
        #     self.point.x[:] = np.linalg.pinv(np.vstack((model.A, model.G))) @ np.vstack((model.b, model.h - self.point.s.vec))
        #     self.point.y[:] = np.linalg.pinv(model.A.T) @ (-model.G.T @ self.point.z.vec - model.c)
        # else:
        #     self.point.x[:] = -(model.h - self.point.s.vec)
        #     self.point.x[:] = np.linalg.pinv(np.vstack((model.A, model.G.toarray()))) @ np.vstack((model.b, model.h - self.point.s.vec))
        #     self.point.y[:] = np.linalg.pinv(model.A.T) @ (self.point.z.vec - model.c)

        self.calc_mu()
        if not math.isclose(self.mu, 1.):
            print(f"Initial mu is {self.mu} but should be 1")

        return

    def calc_mu(self):
        self.mu = (self.point.s.inp(self.point.z) + self.point.tau[0, 0]*self.point.kap[0, 0]) / self.model.nu
        return self.mu

    def get_gap_feas(self):
        model = self.model
        c = self.model.c
        b = self.model.b
        h = self.model.h
        A = self.model.A
        G = self.model.G

        x   = self.point.x
        y   = self.point.y
        z   = self.point.z.vec
        s   = self.point.s.vec
        tau = self.point.tau[0, 0]
        kap = self.point.kap[0, 0]

        # Get primal and dual objectives and optimality gap
        p_obj_tau =  c.T @ x
        d_obj_tau = -b.T @ y - h.T @ z

        self.p_obj = p_obj_tau / tau + self.model.offset
        self.d_obj = d_obj_tau / tau + self.model.offset
        self.gap   = min([self.point.z.inp(self.point.s) / tau, abs(p_obj_tau - d_obj_tau)]) / max([tau, min([abs(p_obj_tau), abs(d_obj_tau)])])

        # Get primal and dual infeasibilities
        self.x_res  = model.A_T @ y
        self.x_res += model.G_T @ z
        self.x_res *= -1

        self.y_res  = model.A @ x
        self.z_res  = model.G @ x

        self.z_res += s

        self.x_infeas =  lin.norm_inf(self.x_res) / d_obj_tau if (d_obj_tau > 0) else np.inf
        self.y_infeas = -lin.norm_inf(self.y_res) / p_obj_tau if (p_obj_tau < 0) else np.inf
        self.z_infeas = -lin.norm_inf(self.z_res) / p_obj_tau if (p_obj_tau < 0) else np.inf

        # Get primal and dual feasibilities
        self.x_res   = sp.linalg.blas.daxpy(c, self.x_res, a=-tau)
        self.y_res   = sp.linalg.blas.daxpy(b, self.y_res, a=-tau) if model.use_A else self.y_res
        self.z_res   = sp.linalg.blas.daxpy(h, self.z_res, a=-tau)
        self.tau_res = p_obj_tau - d_obj_tau + kap

        self.x_feas = lin.norm_inf(self.x_res) / (1. + self.c_max) / tau
        self.y_feas = lin.norm_inf(self.y_res) / (1. + self.b_max) / tau
        self.z_feas = lin.norm_inf(self.z_res) / (1. + self.h_max) / tau

        return

    def rescale_model(self):
        model = self.model

        self.c_scale = np.zeros_like(model.c)
        self.b_scale = np.zeros_like(model.b)
        
        # Rescale c
        for i in range(model.n):
            self.c_scale[i] = np.sqrt(max(abs(model.c[i, 0]), np.max(np.abs(model.A[:, i])), np.max(np.abs(model.G[:, i]))))

        # Rescale b
        for i in range(model.p):
            self.b_scale[i] = np.sqrt(max(abs(model.b[i, 0]), np.max(np.abs(model.A[i, :]))))

        model.c[:, :] /= self.c_scale
        model.A[:, :] /= self.c_scale.T
        model.G[:, :] /= self.c_scale.T

        model.A[:, :] /= self.b_scale
        model.b[:, :] /= self.b_scale

        return