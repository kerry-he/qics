import itertools
import math
import sys
import time

import numpy as np
import scipy as sp

import qics._utils.linalg as lin
import qics.point as vec
from qics import __version__
from qics._stepper import KKTSolver, NonSymStepper, SymStepper
import qics.cones

spinner = itertools.cycle(["-", "/", "|", "\\"])


class Solver:
    r"""A class representing an instance of a solver, and is parameterized
    by a conic program model and adjustable solver settings.

    Parameters
    ----------
    model : :class:`~qics.Model`
        Representation of a given conic program.
    max_iter : :obj:`int`, optional
        Maximum number of solver iterations before terminating. The default
        is ``100``.
    max_time : :obj:`float`, optional
        Maximum time elapsed, in seconds, before terminating. The default
        is ``3600``.
    tol_gap : :obj:`float`, optional
        Stopping tolerance for (relative) optimality gap. The default is
        ``1e-8``.
    tol_feas : :obj:`float`, optional
        Stopping tolerance for (relative) primal and dual feasibility. The
        default is ``1e-8``.
    tol_infeas : :obj:`float`, optional
        Tolerance for detecting infeasible problem. The default is 
        ``1e-12``.
    tol_ip : :obj:`float`, optional
        Tolerance for detecting ill-posed problem. The default is 
        ``1e-13``.
    tol_near : :obj:`float`, optional
        Allowable margin for certifying near optimality when solver is
        stopped early. The default is ``1e3``.
    verbose : {``0``, ``1``, ``2``, ``3``}, optional
        Verbosity level of the solver, where

        - ``0`` : No output.
        - ``1`` : Only print problem and solution summary.
        - ``2`` : Also print summary of the solver at each iteration.
        - ``3`` : Also print summary of the stepper at each iteration.

        The default is ``2``.
    ir : :obj:`bool`, optional
        Whether to use iterative refinement when solving the KKT system.
        The default is ``True``.
    toa : :obj:`bool`, optional
        Whether to use third-order adjustments to improve the stepping
        directions. The default is ``True``.
    init_pnt : :class:`~qics.point.Point`, optional
        Where to initialize the interior-point algorithm from. Variables
        which contain :obj:`~numpy.nan` are flagged to be intialized using
        the default initialization method. The default is ``None``, which
        intializes all variables using the default method.
    use_invhess : :obj:`bool`, optional
        Whether or not to avoid using inverse Hessian product oracles by
        solving a modified cone program with
        :math:`-G^{-1}(\mathcal{K})=\{ x : -Gx \in \mathcal{K} \}`. The default
        is ``True``.
    """

    def __init__(
        self,
        model,
        max_iter=100,
        max_time=3600,
        tol_gap=1e-8,
        tol_feas=1e-8,
        tol_infeas=1e-12,
        tol_ip=1e-13,
        tol_near=1e3,
        verbose=2,
        ir=True,
        toa=True,
        init_pnt=None,
        use_invhess=None,
    ):
        # Basic solver options
        self.max_iter = max_iter
        self.max_time = max_time
        self.verbose = verbose

        self.tol_gap = tol_gap
        self.tol_feas = tol_feas
        self.tol_infeas = tol_infeas
        self.tol_ip = tol_ip
        self.tol_near = tol_near

        # Slow progress options
        self.small_step_tol = 0.005
        self.consecutive_small_step_limit = 2
        self.consecutive_small_steps = 0

        # Avoiding inverse Hessian options
        cones = model.cones
        if use_invhess is None:
            # Default decision tree for whether to avoid inverse Hessian oracles
            SLOW_CONES = (qics.cones.QuantRelEntr, qics.cones.OpPerspecEpi, 
                          qics.cones.OpPerspecTr)
            
            use_invhess = model.issymmetric or not model.use_G \
                or all([not isinstance(cone, SLOW_CONES) for cone in cones]) \
                or lin.is_full_col_rank(model.G)
        elif not use_invhess:
            assert not model.use_G, "Avoiding inverse Hessian oracles is" \
                "not supported nor recommended if G is easily invertible."
            assert lin.is_full_col_rank(model.G), "Avoiding inverse Hessian" \
                "oracles is not supported when G is not full column rank."
        self.use_invhess = use_invhess
        
        # Preprocess model
        self.model = model
        model._preprocess(use_invhess, init_pnt)

        # Initialize steppers
        kktsolver = KKTSolver(model, ir=ir, use_invhess=use_invhess)
        if model.issymmetric:
            self.stepper = SymStepper(kktsolver, model, toa=toa)
        else:
            self.stepper = NonSymStepper(kktsolver, model, toa=toa)

        # Initialize points and process initial point
        self.point = vec.Point(model)
        self.point_best = vec.Point(model)

        self.point.vec.fill(np.nan)
        if init_pnt is not None:
            (n_orig, p_orig) = (model.n_orig, model.p_orig)
            self.point.x[:n_orig] = init_pnt.x * model.c_scale[:n_orig]
            self.point.y[:p_orig] = init_pnt.y * model.b_scale[:p_orig]
            self.point.z.vec[:] = init_pnt.z.vec * model.h_scale
            self.point.s.vec[:] = init_pnt.s.vec * model.h_scale
            self.point.tau[:] = init_pnt.tau
            self.point.kap[:] = init_pnt.kap

        return

    def solve(self):
        r"""Run the primal-dual interior point solver for a given problem 
        model.

        Returns
        -------
        :obj:`dict`
            Dictionary which summarizes the solution of a conic program. 
            Contains the following keys.

            - x_opt (:obj:`~numpy.ndarray`) : Optimal primal variable 
              :math:`x^*`.
            - y_opt (:obj:`~numpy.ndarray`) : Optimal dual variable 
              :math:`y^*`.
            - z_opt (:obj:`~qics.point.VecProduct`) : Optimal dual variable
              :math:`z^*`.
            - s_opt (:obj:`~qics.point.VecProduct`) : Optimal primal
              variable :math:`s^*=h-Gx^*`.
            - sol_status (:obj:`string`) : Solution status. Can either be:

                - ``optimal``      : Primal-dual optimal solution reached
                - ``pinfeas``      : Detected primal infeasibility
                - ``dinfeas``      : Detected dual infeasibility
                - ``near_optimal`` : Near primal-dual optimal solution
                - ``near_pinfeas`` : Near primal infeasibility
                - ``near_dinfeas`` : Near dual infeasibiltiy
                - ``illposed``     : Problem is ill-posed
                - ``unknown``      : Unknown solution status

            - exit_status (:obj:`string`) : Solver exit status. Can either
              be:

                - ``solved``       : Terminated at desired tolerance
                - ``max_iter``     : Exceeded maximum allowable iterations
                - ``max_time``     : Exceeded maximum allowable time
                - ``step_failure`` : Unable to take another step
                - ``slow_progress``: Residuals are decreasing too slowly

            - num_iter (:obj:`int`) : Number of solver iterations.
            - solve_time (:obj:`float`) : Total time elapsed (in seconds).
            - p_obj (:obj:`float`) : Optimal primal objective 
              :math:`c^\top x^*`.
            - d_obj (:obj:`float`) : Optimal dual objective
              :math:`-b^\top y^* - h^\top z^*`.
            - opt_gap (:obj:`float`) : Relative optimality gap.
            - p_feas (:obj:`float`) : Relative primal feasibility.
            - d_feas (:obj:`float`) : Relative dual feasibiltiy.
        """
        # Setup solver
        self.setup_solver()
        self.setup_point()
        self.calc_mu()
        self.get_gap_feas()

        # Print header
        if self.verbose:
            self.print_title()
            if self.verbose >= 2:
                self.print_iter_heading()
                self.print_iter()
            else:
                print()
                sys.stdout.write("Running...")

        # Run main solver loop
        while True:
            if self.step_and_check():
                break
        
        # Solver has termianted, perform final clean up
        self.solve_time = time.time() - self.start_time

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
            self.print_solution()

        # Scale variables back
        x_opt = self.point.x / self.model.c_scale / self.point.tau
        y_opt = self.point.y / self.model.b_scale / self.point.tau
        if not self.use_invhess:
            x_opt = (x_opt + self.model.x_offset)[:self.model.n_orig]
            y_opt = y_opt[:self.model.p_orig]
        z_opt = self.point.z
        z_opt.vec[:] = z_opt.vec / self.model.h_scale / self.point.tau
        s_opt = self.point.s
        s_opt.vec[:] = s_opt.vec / self.model.h_scale / self.point.tau

        return {
            "x_opt": x_opt,
            "y_opt": y_opt,
            "z_opt": z_opt,
            "s_opt": s_opt,
            "sol_status": self.solution_status,
            "exit_status": self.exit_status,
            "num_iter": self.iter,
            "solve_time": self.solve_time,
            "p_obj": self.p_obj,
            "d_obj": self.d_obj,
            "opt_gap": self.gap,
            "p_feas": max(self.y_feas, self.z_feas),
            "d_feas": self.x_feas,
        }

    def step_and_check(self):
        # ==============================================================
        # Step
        # ==============================================================
        # Make a copy of current point before taking a step
        self.copy_solver_data()

        # Take a step
        self.point, success, alpha = self.stepper.step(self.model, self.point,
            self.res, self.mu, self.verbose)
        self.iter += 1
        self.elapsed_time = time.time() - self.start_time

        # Compute barrier parameter and residuals
        self.calc_mu()
        self.get_gap_feas()

        # ==============================================================
        # Print iteration status
        # ==============================================================
        if self.verbose >= 2:
            self.print_iter()
        elif self.verbose:
            sys.stdout.write(next(spinner))
            sys.stdout.flush()
            sys.stdout.write("\b")

        # ==============================================================
        # Check termination criteria
        # ==============================================================
        # 1) Check optimality
        if self.gap <= self.tol_gap:
            if (
                self.x_feas <= self.tol_feas
                and self.y_feas <= self.tol_feas
                and self.z_feas <= self.tol_feas
            ):
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
        if self.point.tau < self.tol_infeas and self.point.kap < self.tol_infeas:
            if self.illposed_res <= self.tol_ip:
                self.solution_status = "illposed"
                self.exit_status = "solved"
                return True

        # 4) Check if maximum iterations is exceeded
        if self.iter >= self.max_iter:
            self.exit_status = "max_iter"
            return True

        # 5) Check if maximum time is exceeded
        if self.elapsed_time >= self.max_time:
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
        self.start_time = time.time()
        self.elapsed_time = 0.0

        model = self.model

        self.c_max = lin.norm_inf(model.c)
        self.b_max = lin.norm_inf(model.b)
        self.h_max = lin.norm_inf(model.h)

        if self.verbose == 3:
            if self.model.issymmetric:
                self.printbar_size = 125
            else:
                self.printbar_size = 136
        else:
            self.printbar_size = 97

        return

    def setup_point(self):
        model = self.model

        if any(np.isnan(self.point.x)):
            self.point.x.fill(0.0)
        if any(np.isnan(self.point.y)):
            self.point.y.fill(0.0)

        for k, cone_k in enumerate(model.cones):
            if any(np.isnan(self.point.s.vecs[k])):
                cone_k.get_init_point(self.point.s[k])
            else:
                cone_k.set_point(self.point.s[k])
            assert cone_k.get_feas(), "Initial primal variable s is infeasible."

            if any(np.isnan(self.point.z.vecs[k])):
                cone_k.grad_ip(self.point.z[k])
                self.point.z.vecs[k] *= -1
            cone_k.set_dual(self.point.z[k])
            assert cone_k.get_dual_feas(), "Initial dual variable z is" \
                "infeasible."

        if any(np.isnan(self.point.tau)):
            self.point.tau.fill(1.0)
        if any(np.isnan(self.point.kap)):
            self.point.kap.fill(1.0)

        self.calc_mu()
        if not math.isclose(self.mu, 1.0):
            print(f"Initial mu is {self.mu} but should be 1")

        return

    def calc_mu(self):
        s_inp_z = self.point.s.inp(self.point.z)
        kap_inp_tau = self.point.tau[0, 0] * self.point.kap[0, 0]
        self.mu = (s_inp_z + kap_inp_tau) / self.model.nu
        return self.mu

    def get_gap_feas(self):
        model = self.model
        c = self.model.c
        b = self.model.b
        h = self.model.h

        x = self.point.x
        y = self.point.y
        z = self.point.z.vec
        s = self.point.s.vec
        tau = self.point.tau[0, 0]
        kap = self.point.kap[0, 0]

        self.kap_tau = kap / tau

        # Get primal and dual objectives and optimality gap
        p_obj_tau = (c.T @ x)[0, 0]
        d_obj_tau = -(b.T @ y + h.T @ z)[0, 0]

        self.p_obj = p_obj_tau / tau + self.model.offset
        self.d_obj = d_obj_tau / tau + self.model.offset
        
        gap1 = self.point.z.inp(self.point.s) / tau
        gap2 = abs(p_obj_tau - d_obj_tau)
        self.gap = min(gap1, gap2)
        self.gap /= max(tau, min([abs(p_obj_tau), abs(d_obj_tau)]))

        # Get primal and dual infeasibilities
        self.x_res = model.A_T @ y
        self.x_res += model.G_T @ z
        self.x_res *= -1
        self.y_res = model.A @ x
        self.z_res = model.G @ x
        self.z_res += s

        norm_x_res = lin.norm_inf(self.x_res)
        norm_y_res = lin.norm_inf(self.y_res)
        norm_z_res = lin.norm_inf(self.z_res)

        self.x_infeas = norm_x_res / d_obj_tau if (d_obj_tau > 0) else np.inf
        self.y_infeas = -norm_y_res / p_obj_tau if (p_obj_tau < 0) else np.inf
        self.z_infeas = -norm_z_res / p_obj_tau if (p_obj_tau < 0) else np.inf

        # Get ill posedness certificates
        norm_xyzs = max(lin.norm_inf(x), lin.norm_inf(y), lin.norm_inf(z), 
                        lin.norm_inf(s))
        if norm_xyzs > 0:
            norm_xyz_res = max(norm_x_res, norm_y_res, norm_z_res)
            self.illposed_res = norm_xyz_res / norm_xyzs
        else:
            self.illposed_res = np.inf

        # Get primal and dual feasibilities
        self.x_res = sp.linalg.blas.daxpy(c, self.x_res, a=-tau)
        if model.p > 0:
            self.y_res = sp.linalg.blas.daxpy(b, self.y_res, a=-tau)
        if model.q > 0:
            self.z_res = sp.linalg.blas.daxpy(h, self.z_res, a=-tau)
        self.tau_res = p_obj_tau - d_obj_tau + kap

        self.x_feas = lin.norm_inf(self.x_res) / (1.0 + self.c_max) / tau
        self.y_feas = lin.norm_inf(self.y_res) / (1.0 + self.b_max) / tau
        self.z_feas = lin.norm_inf(self.z_res) / (1.0 + self.h_max) / tau

        self.res = {"x": self.x_res, "y": self.y_res, "z": self.z_res, 
                    "tau": self.tau_res,}

        return

    def copy_solver_data(self):
        gap_feas = max(self.x_feas, self.y_feas, self.z_feas, self.gap)
        if self.iter == 0:
            # Best point hasn't been initialized yet
            gap_feas_best = np.inf
        else:
            gap_feas_best = max(self.x_feas_best, self.y_feas_best,
                                self.z_feas_best, self.gap_best)
        
        if gap_feas_best > gap_feas:
            self.point_best.vec[:] = self.point.vec
            self.best_iter = self.iter

            self.p_obj_best = self.p_obj
            self.d_obj_best = self.d_obj
            self.gap_best = self.gap

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

    def print_title(self):
        filler = "="
        print(f"{'':{filler}^{68}}")
        print(f"{'QICS v' + __version__ + ' - Quantum Information Conic Solver':^68}")
        print(f"{'by K. He, J. Saunderson, H. Fawzi (2024)':^68}")
        print(f"{'':{filler}^{68}}")

        print("Problem summary:")
        print(f"\tno. vars:     {self.model.n:<10}", end="")
        print(f"\t\tbarr. par:    {self.model.nu:<10}")
        print(f"\tno. constr:   {self.model.p:<10}", end="")
        print(f"\t\tsymmetric:    {self.model.issymmetric!r:<10}")
        print(f"\tcone dim:     {self.model.q:<10}", end="")
        print(f"\t\tcomplex:      {self.model.iscomplex!r:<10}")
        print(f"\tno. cones:    {len(self.model.cones):<10}", end="")
        print(f"\t\tsparse:       {self.model.issparse!r:<10}")

    def print_iter_heading(self):
        filler = "="
        print(f"\n{'':{filler}^{self.printbar_size}}", end="")
        print(f"\n {'iter':^4}   {'mu':^7}   {'k/t':^7}  ", end="")
        print(f"|  {'p_obj':^10}  {'d_obj':^10}  {'gap':^7}  ", end="")
        print(f"|  {'p_feas':^7}   {'d_feas':^7}  ", end="")
        print(f"|  {'time (s)':^8}  ", end="")
        if self.verbose == 3:
            if self.model.issymmetric:
                print(f"|  {'dir_tol':^7}   {'sigma':^5}   {'alpha':^5}", end="")
            else:
                print(f"|  {'step':^6}   {'dir_tol':^7}   ", end="")
                print(f"{'prox':^7}   {'alpha':^5}", end="")
        print(f"\n{'':{filler}^{self.printbar_size}}", end="")

    def print_iter(self):
        yz_feas = max(self.y_feas, self.z_feas)
        print(f"\n {self.iter:>4}   {self.mu:>7.1e}   {self.kap_tau:>7.1e}  ", end="")
        print(f"| {self.p_obj:>10.3e}  {self.d_obj:>10.3e}  {self.gap:>8.1e}  ", end="")
        print(f"|  {yz_feas:>7.1e}   {self.x_feas:>7.1e}  ", end="")
        print(f"|  {self.elapsed_time:<8.2f}", end="")

    def print_solution(self):
        print("\n\nSolution summary")
        print(f"\tsol. status:  {self.solution_status:<19}", end="")
        print(f"\tnum. iter:    {self.iter:<10}")
        print(f"\texit status:  {self.exit_status:<19}", end="")
        print(f"\tsolve time:   {self.solve_time:<10.3f}")
        print(f"\tprimal obj:  {self.p_obj:>19.12e}", end="")
        print(f"\tprimal feas:  {max(self.y_feas, self.z_feas):<.2e}")
        print(f"\tdual obj:    {self.d_obj:>19.12e}", end="")
        print(f"\tdual feas:    {self.x_feas:<.2e}")
        print()
