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

    # Semidefinite programs
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

    def test_sdp_dps_heirarchy(self):
        gdict = self.exec_example("sdp/dps_heirarchy.py")
        self.assertEqual(gdict["info"]["sol_status"], "pinfeas")
        self.assertEqual(gdict["info"]["exit_status"], "solved")

    # Quantum relative entropy programs
    def test_qrep_qkd_ebBB84(self):
        gdict = self.exec_example("qrep/qkd_ebBB84.py")
        self.assertEqual(gdict["info"]["sol_status"], "optimal")
        self.assertEqual(gdict["info"]["exit_status"], "solved")
        self.assertAlmostEqual(gdict["info"]["p_obj"], 0.1308120, places=6)

    def test_qrep_nearest_correlation(self):
        gdict = self.exec_example("qrep/nearest_correlation.py")
        self.assertEqual(gdict["info"]["sol_status"], "optimal")
        self.assertEqual(gdict["info"]["exit_status"], "solved")
        self.assertAlmostEqual(gdict["info"]["p_obj"], 1.5837876, places=6)

    def test_qrep_rel_entr_entanglement(self):
        gdict = self.exec_example("qrep/rel_entr_entanglement.py")
        self.assertEqual(gdict["info"]["sol_status"], "optimal")
        self.assertEqual(gdict["info"]["exit_status"], "solved")
        self.assertAlmostEqual(gdict["info"]["p_obj"], 0.0048387, places=6)

    def test_qrep_bregman_projection(self):
        gdict = self.exec_example("qrep/bregman_projection.py")
        self.assertEqual(gdict["info"]["sol_status"], "optimal")
        self.assertEqual(gdict["info"]["exit_status"], "solved")
        self.assertAlmostEqual(gdict["info"]["p_obj"], 47.0377192, places=6)

if __name__ == "__main__":
    unittest.main()
