import numpy as np
import scipy as sp
import math
import time

from utils import linear as lin
from utils import vector as vec
from solver.stepper.nonsym import NonSymStepper
from solver.stepper.sym import SymStepper
from solver.syssolver import SysSolver

class Solver():
    def __init__(
        self, 
        model, 
        max_iter = 100, 
        max_time = np.inf,
        tol_gap = 1e-8,
        tol_feas = 1e-8,
        tol_infeas = 1e-12,
        tol_ip = 1e-13,
        tol_near = 1e3,
        verbose  = True,
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
        self.tol_near = tol_near

        self.point = vec.Point(model)
        self.point_best = vec.Point(model)

        self.small_step_tol = 0.005
        self.consecutive_small_step_limit = 1
        self.consecutive_small_steps = 0
        
        self.solution_status = None
        self.exit_status = None

        self.model = model
        syssolver = SysSolver(model, ir=ir, sym=sym)
        self.stepper = NonSymStepper(syssolver, model) if (sym is False) else SymStepper(syssolver, model)

        return
    
    def solve(self):
        # Setup solver
        self.setup_solver()
        self.setup_point()
        self.calc_mu()
        self.get_gap_feas()

        # ==============================================================
        # Print iteration status
        # ==============================================================
        if self.verbose:
            if self.iter % 20 == 0:
                print("===================================================================================================================================================================")
                print("%5s" % "iter", " %8s" % "mu", " %8s" % "tau", " %8s" % "kap",
                    " | %10s" % "p_obj", " %10s" % "d_obj", " %10s" % "gap", 
                    " | %10s" % "x_feas", " %10s" % "y_feas", " %10s" % "z_feas",
                    " | %6s" % "step", "%10s" % "dir_tol", "%10s" % "prox", " %5s" % "alpha")
                print("===================================================================================================================================================================")                
            
            print("%5d" % (self.iter), " %8.1e" % (self.mu), " %8.1e" % (self.point.tau), " %8.1e" % (self.point.kap),
                  " | %10.3e" % (self.p_obj), " %10.3e" % (self.d_obj), " %10.3e" % (self.gap),
                  " | %10.3e" % (self.x_feas), " %10.3e" % (self.y_feas), " %10.3e" % (self.z_feas), end="")        

        while True:
            if self.step_and_check():
                break

        self.solve_time = time.time() - self.solve_time

        # If we didn't reach a solution, check if we are close to optimal
        if self.exit_status != "solved":
            self.copy_solver_data()
            self.retrieve_best_data()

            self.solution_status = "unknown"

            # Check near optimality
            if self.gap <= self.tol_gap * self.tol_near:
                if self.x_feas <= self.tol_feas * self.tol_near:
                    if self.y_feas <= self.tol_feas * self.tol_near: 
                        if self.z_feas <= self.tol_feas * self.tol_near:
                            self.solution_status = "near_optimal"

            # Check near infeasibility
            if self.x_infeas <= self.tol_infeas * self.tol_near:
                self.solution_status = "near_pinfeas"

            if self.y_infeas <= self.tol_infeas * self.tol_near:
                if self.z_infeas <= self.tol_infeas * self.tol_near:
                    self.solution_status = "near_dinfeas"


        if self.verbose:
            print()
            print("Solution status: %s" % (self.solution_status))
            print("Exit status:     %s" % (self.exit_status))
            print("Opt value:       %.10f" % (self.p_obj))
            print("Tolerance:       %.10e" % (self.gap))
            print("Solve time:      %.10f" % (self.solve_time), " seconds")
            print()

        return
    
    def step_and_check(self): 
        # ==============================================================
        # Step
        # ==============================================================
        # Make a copy of current point before taking a step
        self.copy_solver_data()

        # Take a step
        self.point, success, alpha = self.stepper.step(self.model, self.point, (self.x_res, self.y_res, self.z_res, self.tau_res), self.mu, self.verbose)
        self.iter += 1

        # Compute barrier parameter and residuals
        self.calc_mu()
        self.get_gap_feas()

        # ==============================================================
        # Print iteration status
        # ==============================================================
        if self.verbose:
            if self.iter % 20 == 0:
                print("===================================================================================================================================================================")
                print("%5s" % "iter", " %8s" % "mu", " %8s" % "tau", " %8s" % "kap",
                    " | %10s" % "p_obj", " %10s" % "d_obj", " %10s" % "gap", 
                    " | %10s" % "x_feas", " %10s" % "y_feas", " %10s" % "z_feas",
                    " | %6s" % "step", "%10s" % "dir_tol", "%10s" % "prox", " %5s" % "alpha")
                print("===================================================================================================================================================================")                
            
            print("%5d" % (self.iter), " %8.1e" % (self.mu), " %8.1e" % (self.point.tau), " %8.1e" % (self.point.kap),
                  " | %10.3e" % (self.p_obj), " %10.3e" % (self.d_obj), " %10.3e" % (self.gap),
                  " | %10.3e" % (self.x_feas), " %10.3e" % (self.y_feas), " %10.3e" % (self.z_feas), end="")        

        # ==============================================================
        # Check termination criteria
        # ==============================================================
        # 1) Check optimality
        if self.gap <= self.tol_gap:
            if self.x_feas <= self.tol_feas and self.y_feas <= self.tol_feas and self.z_feas <= self.tol_feas:
                self.solution_status = "optimal"
                self.exit_status = "solved"
                return True

        # 2) Check primal and dual infeasibility
        if self.x_infeas <= self.tol_infeas:
            self.solution_status = "pinfeas"
            self.exit_status = "solved"
            return True       

        if self.y_infeas <= self.tol_infeas and self.z_infeas <= self.tol_infeas:
            self.solution_status = "dinfeas"
            self.exit_status = "solved"
            return True            
            
        # 3) Check ill-posedness
        if self.illposed_res <= self.tol_ip:
            self.solution_status = "illposed"
            self.exit_status = "solved"
            return True
        
        # 4) Check if maximum iterations is exceeded
        if self.iter >= self.max_iter:
            self.exit_status = "max_iter"
            return True
        
        # 5) Check if maximum time is exceeded
        if time.time() - self.solve_time >= self.max_time:
            self.exit_status = "max_time"
            return True 
        
        # 6) Did the step fail or not
        if not success:
            self.exit_status = "step_failure"
            return True
            
        # 7) Check if progress is slow or degrading at high tolerance
        if alpha <= self.small_step_tol:
            self.consecutive_small_steps += 1
            if self.consecutive_small_steps >= self.consecutive_small_step_limit:
                self.exit_status = "slow_progress"
                return True
        else:
            self.consecutive_small_steps = 0

        return False

    def setup_solver(self):
        self.iter = 0
        self.solve_time = time.time()

        model = self.model

        self.c_max = lin.norm_inf(model.c)
        self.b_max = lin.norm_inf(model.b)
        self.h_max = lin.norm_inf(model.h)

        return

    def setup_point(self):
        model = self.model

        self.point.tau[:] = 1.
        self.point.kap[:] = 1.

        for (k, cone_k) in enumerate(model.cones):
            np.copyto(self.point.s[k], cone_k.set_init_point())
            assert cone_k.get_feas()
            np.copyto(self.point.z[k], -cone_k.get_grad())

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
        p_obj_tau =  (c.T @ x)[0, 0]
        d_obj_tau = -(b.T @ y + h.T @ z)[0, 0]

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

        norm_x_res = lin.norm_inf(self.x_res)
        norm_y_res = lin.norm_inf(self.x_res)
        norm_z_res = lin.norm_inf(self.x_res)

        self.x_infeas =  norm_x_res / d_obj_tau if (d_obj_tau > 0) else np.inf
        self.y_infeas = -norm_y_res / p_obj_tau if (p_obj_tau < 0) else np.inf
        self.z_infeas = -norm_z_res / p_obj_tau if (p_obj_tau < 0) else np.inf

        # Get ill posedness certificates
        norm_xyzs = max(lin.norm_inf(x), lin.norm_inf(y), lin.norm_inf(z), lin.norm_inf(s))
        self.illposed_res = max(norm_x_res, norm_y_res, norm_z_res) / norm_xyzs if (norm_xyzs > 0) else np.inf

        # Get primal and dual feasibilities
        self.x_res   = sp.linalg.blas.daxpy(c, self.x_res, a=-tau)
        self.y_res   = sp.linalg.blas.daxpy(b, self.y_res, a=-tau) if model.use_A else self.y_res
        self.z_res   = sp.linalg.blas.daxpy(h, self.z_res, a=-tau)
        self.tau_res = p_obj_tau - d_obj_tau + kap

        self.x_feas = lin.norm_inf(self.x_res) / (1. + self.c_max) / tau
        self.y_feas = lin.norm_inf(self.y_res) / (1. + self.b_max) / tau
        self.z_feas = lin.norm_inf(self.z_res) / (1. + self.h_max) / tau

        return
    
    def copy_solver_data(self):
        if (self.iter == 0) or max(
            self.x_feas_best, self.y_feas_best, 
            self.z_feas_best, self.gap_best
        ) > max(
            self.x_feas, self.y_feas, 
            self.z_feas, self.gap
        ):
            self.point_best.vec[:] = self.point.vec
            self.best_iter = self.iter
            
            self.p_obj_best = self.p_obj
            self.d_obj_best = self.d_obj
            self.gap_best   = self.gap

            self.x_feas_best = self.x_feas
            self.y_feas_best = self.y_feas
            self.z_feas_best = self.z_feas
            self.tau_res_best = self.tau_res

            self.x_infeas_best = self.x_infeas
            self.y_infeas_best = self.y_infeas
            self.z_infeas_best = self.z_infeas

    def retrieve_best_data(self):
        if self.best_iter != self.iter:
            if self.verbose:
                print("\nRetrieving data from iteration ", self.best_iter)

            self.point.vec = self.point_best.vec[:]
            
            self.p_obj = self.p_obj_best
            self.d_obj = self.d_obj_best
            self.gap = self.gap_best

            self.x_feas = self.x_feas_best
            self.y_feas = self.y_feas_best
            self.z_feas = self.z_feas_best
            self.tau_res = self.tau_res_best

            self.x_infeas = self.x_infeas_best
            self.y_infeas = self.y_infeas_best
            self.z_infeas = self.z_infeas_best