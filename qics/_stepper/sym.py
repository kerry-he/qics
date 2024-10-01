from qics.point import Point, VecProduct


class SymStepper:
    def __init__(self, kktsolver, model, toa=True):
        self.toa = toa
        if not toa:
            self.dir_dummy = VecProduct(model.cones)

        self.kktsolver = kktsolver 

        self.rhs = Point(model)
        self.dir_a = Point(model)
        self.dir_comb = Point(model)
        self.next_point = Point(model)

        return

    def step(self, model, point, res, mu, verbose):
        # Step 1: Pre-build and -factor Schur complement matrix
        self.kktsolver.update_lhs(model, point, mu)

        # Step 2: Get affine direction
        self.update_rhs_affine(model, point, res)
        res_norm = self.kktsolver.solve_sys(self.dir_a, self.rhs, ir=False)

        # Step 3: Step size and centering parameter
        alpha = self.taukap_step_to_boundary(self.dir_a, point)
        for k, cone_k in enumerate(model.cones):
            alpha = min(
                alpha, cone_k.step_to_boundary(self.dir_a.s[k], self.dir_a.z[k])
            )
        alpha = min(alpha, 1.0)
        sigma = (1 - alpha) ** 3

        # Step 4: Combined direction
        self.update_rhs_comb(model, point, mu, self.dir_a, sigma, res)
        temp_res_norm = self.kktsolver.solve_sys(self.dir_comb, self.rhs, ir=True)
        res_norm = max(temp_res_norm, res_norm)

        # Step 5: Line search
        alpha = self.taukap_step_to_boundary(self.dir_comb, point)
        for k, cone_k in enumerate(model.cones):
            alpha = min(
                alpha, cone_k.step_to_boundary(self.dir_comb.s[k], self.dir_comb.z[k])
            )
        alpha = min(alpha, 1.0) * 0.99
        if alpha == 0:
            point, False

        if verbose == 3:
            print(f"  |  {res_norm:>7.1e}   {sigma:>5.3f}   {alpha:>5.3f}", end="")

        # Take step
        point.vec += alpha * self.dir_comb.vec
        for k, cone_k in enumerate(model.cones):
            cone_k.set_point(point.s[k], point.z[k])
            if not cone_k.get_feas():
                return point, False, alpha

        return point, True, alpha

    def taukap_step_to_boundary(self, dir, point):
        tau_alpha = dir.tau[0, 0] / point.tau[0, 0]
        kap_alpha = dir.kap[0, 0] / point.kap[0, 0]
        if tau_alpha >= 0 and kap_alpha >= 0:
            # Can take maximum step
            return 1.0
        else:
            return 1.0 / max(-tau_alpha, -kap_alpha)

    def update_rhs_affine(self, model, point, res):
        self.rhs.x[:] = res["x"]
        self.rhs.y[:] = res["y"]
        self.rhs.z.vec[:] = res["z"]

        self.rhs.s.vec[:] = -1 * point.z.vec

        self.rhs.tau[:] = res["tau"]
        self.rhs.kap[:] = -point.kap

        return self.rhs

    def update_rhs_comb(self, model, point, mu, dir_a, sigma, res):
        self.rhs.x[:] = res["x"] * (1 - sigma)
        self.rhs.y[:] = res["y"] * (1 - sigma)
        self.rhs.z.vec[:] = res["z"] * (1 - sigma)

        if self.toa:
            for k, cone_k in enumerate(model.cones):
                cone_k.comb_dir(self.rhs.s[k], dir_a.s[k], dir_a.z[k], sigma * mu)
        else:
            for k, cone_k in enumerate(model.cones):
                cone_k.comb_dir(self.rhs.s[k], self.dir_dummy[k], self.dir_dummy[k], sigma * mu)

        self.rhs.tau[:] = res["tau"] * (1 - sigma)
        self.rhs.kap[:] = -point.kap + (-dir_a.kap * dir_a.tau + sigma * mu) / point.tau

        return self.rhs
