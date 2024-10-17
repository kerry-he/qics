# Copyright (c) 2024, Kerry He, James Saunderson, and Hamza Fawzi

# This Python package QICS is licensed under the MIT license; see LICENSE.md
# file in the root directory or at https://github.com/kerry-he/qics

import numpy as np

from qics._stepper.kktsolver import blk_hess_congruence
from qics._utils.linalg import cho_fact, cho_solve
from qics.point import Point, VecProduct

alpha_sched = [0.9999, 0.999, 0.99, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0.01, 0.001]


class NonSymStepper:
    def __init__(self, kktsolver, model, toa=True):
        self.toa = toa

        self.kktsolver = kktsolver
        self.prox = 0.0

        self.rhs = Point(model)
        self.dir_c = Point(model)
        self.dir_c_toa = Point(model)
        self.dir_p = Point(model)
        self.dir_p_toa = Point(model)
        self.next_point = Point(model)

        self.grad = VecProduct(model.cones)

        return

    def step(self, model, point, res, mu, verbose):
        self.kktsolver.update_lhs(model, point, mu)

        # Get prediction direction
        self.update_rhs_pred(model, point, res)
        res_norm = self.kktsolver.solve_sys(self.dir_p, self.rhs)

        # Get TOA prediction direction
        if self.toa:
            self.update_rhs_pred_toa(model, point, mu, self.dir_p)
            temp_res_norm = self.kktsolver.solve_sys(self.dir_p_toa, self.rhs)
            res_norm = max(temp_res_norm, res_norm)

        # Get centering direction
        self.update_rhs_cent(model, point, mu)
        temp_res_norm = self.kktsolver.solve_sys(self.dir_c, self.rhs)
        res_norm = max(temp_res_norm, res_norm)

        # Get TOA centering direction
        if self.toa:
            self.update_rhs_cent_toa(model, point, mu, self.dir_c)
            temp_res_norm = self.kktsolver.solve_sys(self.dir_c_toa, self.rhs)
            res_norm = max(temp_res_norm, res_norm)

        success = False

        if self.toa:
            step_mode = "co_toa"
            point, alpha, success = self.line_search(model, point, step_mode)

        if not success:
            step_mode = "comb"
            point, alpha, success = self.line_search(model, point, step_mode)

        if not success and self.toa:
            step_mode = "ce_toa"
            point, alpha, success = self.line_search(model, point, step_mode)

        if not success:
            step_mode = "cent"
            point, alpha, success = self.line_search(model, point, step_mode)

        if verbose == 3:
            print(
                f"  |  {step_mode:>6}   {res_norm:>7.1e}   {self.prox:>7.1e}   {alpha:>5.3f}",
                end="",
            )

        return point, success, alpha

    def line_search(self, model, point, mode="co_toa"):
        alpha = alpha_sched[0]
        alpha_iter = -1
        eta = 0.99

        dir_p = self.dir_p
        dir_c = self.dir_c
        dir_p_toa = self.dir_p_toa
        dir_c_toa = self.dir_c_toa
        next_point = self.next_point

        while True:
            alpha_iter += 1

            if alpha_iter >= len(alpha_sched):
                return point, alpha, False

            alpha = alpha_sched[alpha_iter]

            # Step point in direction and step size
            next_point.vec[:] = point.vec
            if mode == "co_toa":
                # step := alpha * (dir_p + dir_p_toa * alpha) + (dir_c + dir_c_toa * (1.0 - alpha)) * (1.0 - alpha)
                next_point.axpy(alpha, dir_p)
                next_point.axpy(alpha**2, dir_p_toa)
                next_point.axpy((1.0 - alpha), dir_c)
                next_point.axpy((1.0 - alpha) ** 2, dir_c_toa)
            elif mode == "comb":
                # step := dir_p * alpha + dir_c * (1.0 - alpha)
                next_point.axpy(alpha, dir_p)
                next_point.axpy((1.0 - alpha), dir_c)
            elif mode == "ce_toa":
                # step := (dir_p + dir_p_toa * alpha) * alpha
                next_point.axpy(alpha, dir_c)
                next_point.axpy(alpha**2, dir_c_toa)
            elif mode == "cent":
                # step := dir_c * alpha
                next_point.axpy(alpha, dir_c)

            # Check that tau, kap, mu are well defined
            # Make sure tau, kap are positive
            taukap = next_point.tau * next_point.kap
            if next_point.tau <= 0 or next_point.kap <= 0 or taukap <= 0:
                continue
            # Compute barrier parameter mu, ensure it is positive
            sz = next_point.s.inp(next_point.z)
            mu = (sz + taukap) / model.nu
            if sz <= 0 or mu <= 0:
                continue

            # Check cheap proximity conditions first
            szs = [
                s_k.T @ z_k for (s_k, z_k) in zip(next_point.s.vecs, next_point.z.vecs)
            ]
            nus = [cone_k.nu for cone_k in model.cones]
            rho = [
                np.abs(sz_k / mu - nu_k) / np.sqrt(nu_k)
                for (sz_k, nu_k) in zip(szs, nus)
            ]
            if any(np.array(rho) > eta):
                continue

            rtmu = np.sqrt(mu)
            irtmu = np.reciprocal(rtmu)
            self.prox = 0.0

            # Check feasibility
            in_prox = True
            for k, cone_k in enumerate(model.cones):
                cone_k.set_point(next_point.s[k], next_point.z[k], irtmu)

                # Check if feasible
                if not cone_k.get_feas():
                    in_prox = False
                    break

                if self.kktsolver.use_invhess:
                    # Check if close enough to central path
                    prox_k = cone_k.prox()
                    self.prox = max(self.prox, prox_k)
                    in_prox = self.prox < eta
                    if not in_prox:
                        break
                else:
                    cone_k.grad_ip(self.grad[k])

            if in_prox and not self.kktsolver.use_invhess:
                GHG = blk_hess_congruence(model.G_T_views, model)
                self.kktsolver.GHG_fact = cho_fact(GHG)
                psi = model.G_T @ (next_point.z.vec / rtmu + self.grad.vec)

                self.prox = (psi.T @ cho_solve(self.kktsolver.GHG_fact, psi))[0, 0]
                in_prox = self.prox < eta

            # If feasible, return point
            if in_prox:
                point.vec[:] = next_point.vec
                return point, alpha, True

    def update_rhs_cent(self, model, point, mu):
        self.rhs.x.fill(0.0)
        self.rhs.y.fill(0.0)
        self.rhs.z.fill(0.0)

        # rs := -z - mu*g(s)
        rtmu = np.sqrt(mu)
        for k, cone_k in enumerate(model.cones):
            cone_k.grad_ip(self.rhs.s[k])
        self.rhs.s.vec *= -rtmu
        self.rhs.s.vec -= point.z.vec

        self.rhs.tau[:] = 0.0
        self.rhs.kap[:] = -point.kap + mu / point.tau

        return self.rhs

    def update_rhs_cent_toa(self, model, point, mu, dir_c):
        self.rhs.x.fill(0.0)
        self.rhs.y.fill(0.0)
        self.rhs.z.fill(0.0)

        rtmu = np.sqrt(mu)
        self.rhs.s.fill(0.0)
        for k, cone_k in enumerate(model.cones):
            cone_k.third_dir_deriv_axpy(self.rhs.s[k], dir_c.s[k])
        self.rhs.s.vec *= -0.5 / rtmu

        self.rhs.tau[:] = 0.0
        self.rhs.kap[:] = 0.0

        return self.rhs

    def update_rhs_pred(self, model, point, res):
        self.rhs.x[:] = res["x"]
        self.rhs.y[:] = res["y"]
        self.rhs.z.vec[:] = res["z"]

        self.rhs.s.vec[:] = -point.z.vec

        self.rhs.tau[:] = res["tau"]
        self.rhs.kap[:] = -point.kap

        return self.rhs

    def update_rhs_pred_toa(self, model, point, mu, dir_p):
        self.rhs.x.fill(0.0)
        self.rhs.y.fill(0.0)
        self.rhs.z.fill(0.0)

        rtmu = np.sqrt(mu)
        for k, cone_k in enumerate(model.cones):
            cone_k.hess_prod_ip(self.rhs.s[k], dir_p.s[k])
            cone_k.third_dir_deriv_axpy(self.rhs.s[k], dir_p.s[k], a=-0.5 / rtmu)

        self.rhs.tau[:] = 0.0
        self.rhs.kap[:] = -dir_p.tau * dir_p.kap / point.tau

        return self.rhs
