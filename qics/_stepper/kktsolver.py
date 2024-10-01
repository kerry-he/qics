import numpy as np
import scipy as sp
import qics._utils.linalg as lin
import qics.point


class KKTSolver:
    """A class which is used to solve the KKT system

        [ rx ]    [      A'  G'  c ] [ dx ]   [    ]
        [ ry ] := [ -A           b ] [ dy ] - [    ]
        [ rz ]    [ -G           h ] [ dz ]   [ ds ]
        [rtau]    [ -c' -b' -h'    ] [dtau]   [dkap]

    and

        rs   :=  mu H(s) ds + dz
        rkap :=  (mu / tau^2) dtau + dkap

    or, if NT scaling is used,

        rs   :=  H(w) ds + dz
        rkap :=  (kap / tau) dtau + dkap

    for (dx, dy, dz, dtau, dkap) given right-hand residuals (rx, ry, rz, rtau, rkap) by
    using block elimination and Cholesky factorization of the Schur complement matrix.
    """

    def __init__(self, model, ir=True, use_invhess=True):
        self.model = model

        # Iterative refinement settings
        self.ir = ir  # Use iterative refinement or not
        self.ir_settings = {
            "maxiter": 1,  # Maximum IR iterations
            "tol": 1e-8,  # Tolerance for when IR is used
            "improv_ratio": 5.0,  # Expected reduction in tolerance, else slow progress
        }

        self.use_invhess = use_invhess

        # Preallocate vectors do to computations with
        self.cbh = qics.point.PointXYZ(model)
        self.cbh.x[:] = model.c
        self.cbh.y[:] = model.b
        self.cbh.z.vec[:] = model.h

        self.ir_pnt = qics.point.Point(model)
        self.res_pnt = qics.point.Point(model)

        self.ir_xyz = qics.point.PointXYZ(model)
        self.res_xyz = qics.point.PointXYZ(model)
        self.c_xyz = qics.point.PointXYZ(model)
        self.v_xyz = qics.point.PointXYZ(model)

        self.work1 = qics.point.VecProduct(model.cones)
        self.work2 = qics.point.VecProduct(model.cones)

        self.GHG_fact = None

        return

    def update_lhs(self, model, point, mu):
        self.mu = mu
        self.pnt = point

        # Precompute and factor Schur complement matrix
        if model.use_G:
            if self.GHG_fact is None or self.use_invhess:
                GHG = blk_hess_congruence(model.G_T_views, model)
                self.GHG_fact = lin.cho_fact(GHG, increment_diag=(not model.use_A))

            if model.use_A:
                self.GHG_issingular = (self.GHG_fact is None)
                if self.GHG_issingular:
                    # GHG is singular, Cholesky factor GHG + AA instead
                    GHG += model.A.T @ model.A
                    self.GHG_fact = lin.cho_fact(GHG)

                GHGA = lin.cho_solve(self.GHG_fact, model.A_T_dense)
                AGHGA = lin.dense_dot_x(GHGA.T, model.A_coo.T).T
                self.AGHGA_fact = lin.cho_fact(AGHGA)

        elif model.use_A:
            AHA = blk_invhess_congruence(model.A_invG_views, model)
            self.AHA_fact = lin.cho_fact(AHA)

        # Solve constant 3x3 subsystem
        self.solve_sys_3(self.c_xyz, self.cbh)
        if self.ir and self.use_invhess:
            self.solve_sys_3_ir(self.c_xyz, self.cbh)

        return

    def solve_sys(self, dir, rhs, ir=True):
        # Solve system
        self.solve_sys_6(dir, rhs)
        if self.ir and ir:
            res_norm = self.solve_sys_6_ir(dir, rhs)
        else:
            res_norm = 0.0

        return res_norm

    def solve_sys_6(self, d, r):
        # Compute (dx, dy, dz, ds, dtau, dkap) by solving the
        # 6x6 block system
        #     [ rx ]    [      A'  G'  c ]  [ dx ]   [    ]
        #     [ ry ] := [ -A           b ]  [ dy ] - [    ]
        #     [ rz ]    [ -G           h ]  [ dz ]   [ ds ]
        #     [rtau]    [ -c' -b' -h'    ]  [dtau]   [dkap]
        # and
        #       rs   :=  mu H(s) ds + dz
        #      rkap  :=  (mu / tau^2) dtau + dkap
        # or, if NT scaling is used,
        #       rs   :=  H(w) ds + dz
        #      rkap  :=  (kap / tau) dtau + dkap
        # for a given (rx, ry, rz, rs, rtau, rkap).

        model = self.model
        pnt = self.pnt

        # First, solve the two reduced 3x3 subsystems
        #     [ rx ]    [      A'  G' ]  [ dx ]
        #     [ ry ] := [ -A          ]  [ dy ]
        #     [ rz ]    [ -G     H^-1 ]  [ dz ]
        #               \____ = M ____/
        # for
        #     1) (cx, cy, cz) := M \ (c, b, h)
        #     2) (vx, vy, vz) := M \ (rx, ry, rz + H \ rs)
        # (the first one has been precomputed)
        self.solve_sys_3(self.v_xyz, r._xyz, r.s)

        # Second, backsubstitute to obtain solutions for the full 6x6 system
        #    dtau := (rtau + rkap + c' vx + b' vy + h' vz) / (T + c' cx + b' cy + h' cz)
        #     dx  := vx - dtau cx
        #     dy  := vy - dtau cy
        #     dz  := vz - dtau cz
        #     ds  := -G dx + dtau * h - rz
        #    dkap := rkap - T * dtau
        # where T := mu/tau^2, or T := kap/tau if NT scaling is used

        # taunum := rtau + rkap + c' vx + b' vy + h' vz
        tau_num = r.tau + r.kap + self.cbh.inp(self.v_xyz)
        # tauden := kap / tau + c' cx + b' cy + h' cz
        tau_den = pnt.kap / pnt.tau + self.cbh.inp(self.c_xyz)
        # dtau := taunum / tauden
        d.tau[:] = tau_num / tau_den

        # (dx, dy, dz) := (vx, vy, vz) - dtau * (cx, cy, cz)
        d._xyz.vec[:] = sp.linalg.blas.daxpy(
            self.c_xyz.vec, self.v_xyz.vec, a=-d.tau[0, 0]
        )

        # ds := -G dx + dtau * h - rz
        np.multiply(model.h, d.tau[0, 0], out=d.s.vec)
        d.s.vec -= model.G @ d.x
        d.s.vec -= r.z.vec

        # dkap := rkap - (kap/tau) * dtau
        d.kap[:] = r.kap - (pnt.kap / pnt.tau) * d.tau

        return d

    def solve_sys_3(self, d, r, rs=None):
        # Compute (dx, dy, dz) by solving the 3x3 block system
        #     [ rx ]   [    ]    [      A'  G' ]  [ dx ]
        #     [ ry ] + [    ] := [ -A          ]  [ dy ]
        #     [ rz ]   [H\rs]    [ -G     H^-1 ]  [ dz ]
        # where H = mu H(s), or H = H(w) if NT scaling is used,
        # for a given (rx, ry, rz, rs).

        model = self.model

        if model.use_A and model.use_G:
            # In the general case
            #     dy := (A (G'HG)^-1 A') \ [ry + A (G'HG) \ [rx - G' (H rz + rs)]]
            #     dx := (G'HG) \ [rx - G' (H rz + rs) - A' dy]
            #     dz := H (rz + G dx) + rs

            # dy := (A (G'HG)^-1 A') \ [ry + A (G'HG) \ [rx - G' (H rz + rs)]]
            blk_hess_prod_ip(self.work1, r.z, model)
            if rs is not None:
                self.work1 += rs
            temp = r.x - model.G_T @ self.work1.vec
            if self.GHG_issingular:
                temp -= model.A_T @ r.y
            temp = r.y + model.A @ lin.cho_solve(self.GHG_fact, temp)
            d.y[:] = lin.cho_solve(self.AGHGA_fact, temp)

            # dx := (G'HG) \ [rx - G' (H rz + rs) - A' dy]
            temp = r.x - model.G_T @ self.work1.vec - model.A_T @ d.y
            if self.GHG_issingular:
                temp -= model.A_T @ r.y
            d.x[:] = lin.cho_solve(self.GHG_fact, temp)

            # dz := H (rz + G dx) + rs
            d.z.vec[:] = self.work1.vec
            self.work2.vec[:] = model.G @ d.x
            blk_hess_prod_ip(self.work1, self.work2, model)
            d.z.vec += self.work1.vec

        elif model.use_A and not model.use_G:
            # If G = -I (or some easily invertible square diagonal scaling), then
            #     [ rx ]   [    ]    [      A' -I  ]  [ dx ]
            #     [ ry ] + [    ] := [ -A          ]  [ dy ]
            #     [ rz ]   [H\rs]    [  I     H^-1 ]  [ dz ]
            # dy := (AG^-1 H^-1 (AG^-1)') \ [ry + A (G^-1 (H \ [G^-1 rx - rs] - rz)]
            # dz := G^-1 (rx - A' dy)
            # dx := G^-1 (H \ [G^-1 (rx - A' dy) - rs] - rz)

            # dy := (AG^-1 H^-1 (AG^-1)') \ [ry + A (G^-1 (H \ [G^-1 rx - rs] - rz)]
            np.multiply(r.x, model.G_inv, out=self.work1.vec)
            if rs is not None:
                self.work1.vec -= rs.vec
            blk_invhess_prod_ip(self.work2, self.work1, model)
            self.work2.vec -= r.z.vec
            np.multiply(self.work2.vec, model.G_inv, out=d.x)
            temp = model.A @ d.x
            temp += r.y
            d.y[:] = lin.cho_solve(self.AHA_fact, temp)

            # dz := G^-1 (rx - A' dy)
            A_T_dy = model.A_T @ d.y
            np.subtract(r.x, A_T_dy, out=d.z.vec)
            d.z.vec *= model.G_inv

            # dx := G^-1 (H \ [G^-1 rx - rs] - rz - H \ [G^-1 A' dy])
            np.multiply(A_T_dy, model.G_inv, out=self.work1.vec)
            blk_invhess_prod_ip(self.work2, self.work1, model)
            self.work2.vec *= model.G_inv
            d.x -= self.work2.vec

        elif not model.use_A and model.use_G:
            # If A = [] (i.e, no primal linear constraints), then
            #     [ rx ] + [    ] := [       G' ]  [ dx ]
            #     [ rz ]   [H\rs]    [ -G  H^-1 ]  [ dz ]
            # dx := G'HG \ [rx - G' (H rz + rs)]
            # dz := H (rz + G dx) + rs

            # dx := GHG \ [rx - G' (H rz + rs)]
            blk_hess_prod_ip(self.work1, r.z, model)
            if rs is not None:
                self.work1 += rs
            temp = r.x - model.G_T @ self.work1.vec
            d.x[:] = lin.cho_solve(self.GHG_fact, temp)

            # dz := H (rz + G dx) + rs
            self.work1.copy_from(model.G @ d.x)
            self.work1 += r.z
            blk_hess_prod_ip(d.z, self.work1, model)
            if rs is not None:
                d.z += rs

        elif not model.use_A and not model.use_G:
            # If both A = [] and G = -I, then
            #     [ rx ] + [    ] := [       G  ]  [ dx ]
            #     [ rz ]   [H\rs]    [ -G  H^-1 ]  [ dz ]
            # dz := G \ rx
            # dx := -G \ [rz + H \ (rs - dz)]

            # dz := G \ rx
            np.multiply(r.x, model.G_inv, out=d.z.vec)

            # dx := rz + H \ [rx - dz]
            self.work1.vec[:] = -d.z.vec
            if rs is not None:
                self.work1 += rs
            blk_invhess_prod_ip(self.work2, self.work1, model)
            d.x[:] = self.work2.vec
            d.x += r.z.vec
            d.x *= -model.G_inv

        return d

    def apply_sys_6(self, r, d):
        # Compute (rx, ry, rz, rs, rtau, rkap) as a forwards pass
        # of the 6x6 block system
        #     [ rx ]    [      A'  G'  c ]  [ dx ]   [    ]
        #     [ ry ] := [ -A           b ]  [ dy ] - [    ]
        #     [ rz ]    [ -G           h ]  [ dz ]   [ ds ]
        #     [rtau]    [ -c' -b' -h'    ]  [dtau]   [dkap]
        # and
        #       rs   :=  mu H(s) ds + dz
        #      rkap  :=  (mu / tau^2) dtau + dkap
        # or, if NT scaling is used,
        #       rs   :=  H(w) ds + dz
        #      rkap  :=  (kap / tau) dtau + dkap
        # for a given (dx, dy, dz, ds, dtau, dkap).

        model = self.model
        pnt = self.pnt

        # rx := A' dy + G' dz + c dtau
        np.multiply(model.c, d.tau[0, 0], out=r.x)
        r.x += model.A_T @ d.y
        r.x += model.G_T @ d.z.vec

        # ry := -A dx + b dtau
        np.multiply(model.b, d.tau[0, 0], out=r.y)
        r.y -= model.A @ d.x

        # rz := -G dx + h dtau - ds
        np.multiply(model.h, d.tau[0, 0], out=r.z.vec)
        r.z.vec -= model.G @ d.x
        r.z.vec -= d.s.vec

        # rs := mu H ds + dz
        blk_hess_prod_ip(r.s, d.s, model)
        r.s.vec += d.z.vec

        # rtau := -c' dx - b' dy - h' dz - dkap
        r.tau[:] = (
            -(model.c.T @ d.x) - (model.b.T @ d.y) - (model.h.T @ d.z.vec) - d.kap[0, 0]
        )

        # rkap := (kap / tau) dtau + dkap
        r.kap[:] = (pnt.kap / pnt.tau) * d.tau[0, 0] + d.kap[0, 0]

        return r

    def apply_sys_3(self, r, d):
        # Compute (rx, ry, rz) as a forwards pass
        # of the 3x3 block system
        #     [ rx ]    [      A'  G'  ]  [ dx ]
        #     [ ry ] := [ -A           ]  [ dy ]
        #     [ rz ]    [ -G      H^-1 ]  [ dz ]
        # where H = mu H(s), or H = H(w) if NT scaling is used,
        # for a given (rx, ry, rz).

        model = self.model

        # rx := A' dy + G' dz
        r.x[:] = model.A_T @ d.y
        r.x += model.G_T @ d.z.vec

        # ry := -A dx
        r.y[:] = model.A @ d.x
        r.y *= -1

        # pz := -G dx + H \ dz
        blk_invhess_prod_ip(r.z, d.z, model)
        r.z.vec -= model.G @ d.x

        return r

    def solve_sys_6_ir(self, d, r):
        return solve_sys_ir(
            d,
            r,
            self.apply_sys_6,
            self.solve_sys_6,
            self.res_pnt,
            self.ir_pnt,
            self.ir_settings,
        )

    def solve_sys_3_ir(self, d, r):
        return solve_sys_ir(
            d,
            r,
            self.apply_sys_3,
            self.solve_sys_3,
            self.res_xyz,
            self.ir_xyz,
            self.ir_settings,
        )


def solve_sys_ir(x, b, A, A_inv, res, cor, settings):
    # Perform iterative refinement on a solution x of a linear system
    #     A x = b

    ir_maxiter = settings["maxiter"]
    ir_tol = settings["tol"]
    ir_improv_ratio = settings["improv_ratio"]

    r_norm = b.norm()

    # Compute residuals:
    # res := b - A x
    A(res, x)
    res -= b
    res_norm = res.norm() / (1 + r_norm)

    for i in range(ir_maxiter):
        # If solution is accurate enough, exit iterative refinement
        if res_norm < ir_tol:
            break
        prev_res_norm = res_norm

        # Perform iterative refinement
        # cor := A \ res
        A_inv(cor, res)
        cor *= -1
        # x := x + cor
        cor += x

        # Check residuals
        #     res := b - A x
        A(res, cor)
        res -= b
        res_norm = res.norm() / (1 + r_norm)

        # Exit if iterative refinement made things worse
        improv_ratio = prev_res_norm / (res_norm + 1e-15)
        if improv_ratio < 1:
            return prev_res_norm

        # Otherwise, update solution
        x.vec[:] = cor.vec

        # Exit if iterative refinement is slow
        if improv_ratio < ir_improv_ratio:
            break

    return res_norm


def blk_hess_prod_ip(out, dirs, model):
    if model.issymmetric:
        for k, cone_k in enumerate(model.cones):
            cone_k.nt_prod_ip(out[k], dirs[k])
    else:
        for k, cone_k in enumerate(model.cones):
            cone_k.hess_prod_ip(out[k], dirs[k])
    return out


def blk_invhess_prod_ip(out, dirs, model):
    if model.issymmetric:
        for k, cone_k in enumerate(model.cones):
            cone_k.invnt_prod_ip(out[k], dirs[k])
    else:
        for k, cone_k in enumerate(model.cones):
            cone_k.invhess_prod_ip(out[k], dirs[k])
    return out


def blk_hess_congruence(dirs, model):
    n = model.n
    out = np.zeros((n, n))

    if model.issymmetric:
        for k, cone_k in enumerate(model.cones):
            out += cone_k.nt_congr(dirs[k])
    else:
        for k, cone_k in enumerate(model.cones):
            out += cone_k.hess_congr(dirs[k])

    return out


def blk_invhess_congruence(dirs, model):
    p = model.p
    out = np.zeros((p, p))

    if model.issymmetric:
        for k, cone_k in enumerate(model.cones):
            out += cone_k.invnt_congr(dirs[k])
    else:
        for k, cone_k in enumerate(model.cones):
            out += cone_k.invhess_congr(dirs[k])

    return out
