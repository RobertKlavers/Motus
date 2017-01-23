from unittest import TestCase

import math
import numpy as np

from motus.cda import Ellipse
from motus.trn import TRN


class TestEllipse(TestCase):
    def test_eccentricity_equal(self):
        e = Ellipse(0, [0., 0.], 10., 10.)
        self.assertAlmostEqual(e.eccentricity(), 0.)

    def test_eccentricity_large(self):
        e = Ellipse(0, [0., 0.], 1., 100000.)
        self.assertAlmostEqual(e.eccentricity(), 1.)

    def test_matrix_form(self):
        e = Ellipse(0, [0., 0.], 8., 4.)
        matrix_form = e.quadratic_form()
        manual_form = [16., 0., 64., 0., 0., -1024.]
        for entry in zip(matrix_form, manual_form):
            self.assertAlmostEqual(entry[0], entry[1])

    def test_matrix_form_2(self):
        e = Ellipse(0, [2., 4.], 8., 4., math.pi / 2)
        matrix_form = e.quadratic_form()
        manual_form = [64.0, 0., 16.0, -256., -128.0, -512.0]
        for entry in zip(matrix_form, manual_form):
            self.assertAlmostEqual(entry[0], entry[1])

    def test_angle_invariants(self):
        e_a = Ellipse(0, [2., 2.], 2., 2.)
        e_b = Ellipse(0, [0., 0.], 2., 2.)
        invs = e_a.invariant(e_b)
        self.assertAlmostEqual(invs[0], math.pi / 4)
        self.assertAlmostEqual(invs[1], -3 * math.pi / 4)
