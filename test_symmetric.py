# Copyright (c) 2024, Kerry He, James Saunderson, and Hamza Fawzi
# Based on test_examples.py from CVXOPT by M. Andersen and L. Vandenberghe.

# This Python package QICS is licensed under the MIT license; see LICENSE.md
# file in the root directory or at https://github.com/kerry-he/qics

import os
import unittest


class TestExamples(unittest.TestCase):
    def setUp(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.expath = os.path.normpath(dir_path + "/examples")

    def exec_example(self, example):
        fname = os.path.join(self.expath, example)
        gdict = dict()
        with open(fname) as f:
            code = compile(f.read(), fname, "exec")
            exec(code, gdict)
        return gdict

    def assertAlmostEqualLists(self, L1, L2, places=7):
        self.assertEqual(len(L1), len(L2))
        for u, v in zip(L1, L2):
            self.assertAlmostEqual(u, v, places)

    ## SDPS
    def test_sdp_max_cut(self):
        gdict = self.exec_example("sdp/max_cut.py")
        self.assertEqual(gdict["info"]["sol_status"], "optimal")
        self.assertEqual(gdict["info"]["exit_status"], "solved")
        self.assertAlmostEqual(gdict["info"]["p_obj"], -53.4937692, places=6)

    def test_sdp_diamond_norm(self):
        gdict = self.exec_example("sdp/diamond_norm.py")
        self.assertEqual(gdict["info"]["sol_status"], "optimal")
        self.assertEqual(gdict["info"]["exit_status"], "solved")
        self.assertAlmostEqual(gdict["info"]["p_obj"], -1.1439575, places=6)

    def test_sdp_state_discrimination(self):
        gdict = self.exec_example("sdp/state_discrimination.py")
        self.assertEqual(gdict["info"]["sol_status"], "optimal")
        self.assertEqual(gdict["info"]["exit_status"], "solved")
        self.assertAlmostEqual(gdict["info"]["p_obj"], -1.0000000, places=6)

    def test_sdp_state_fidelity(self):
        gdict = self.exec_example("sdp/state_fidelity.py")
        self.assertEqual(gdict["info"]["sol_status"], "optimal")
        self.assertEqual(gdict["info"]["exit_status"], "solved")
        self.assertAlmostEqual(gdict["info"]["p_obj"], -0.7536086, places=6)

    def test_sdp_optimal_transport(self):
        gdict = self.exec_example("sdp/optimal_transport.py")
        self.assertEqual(gdict["info"]["sol_status"], "optimal")
        self.assertEqual(gdict["info"]["exit_status"], "solved")
        self.assertAlmostEqual(gdict["info"]["p_obj"], 0.3777573, places=6)


if __name__ == "__main__":
    unittest.main()
