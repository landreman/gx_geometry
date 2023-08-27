#!/usr/bin/env python

import unittest
import os
import logging

import numpy as np

from gx_geometry import differentiation_matrix

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)


class DifferentiationMatrixTests(unittest.TestCase):
    def test_row_sums(self):
        N = 17
        D = differentiation_matrix(N, 1.0)
        np.testing.assert_allclose(np.sum(D, axis=1), np.zeros(N))
        
    def test_linear(self):
        """Test that the matrix is exactly correct for a linear function."""
        N = 15
        xmin = -2.3
        xmax = 4.1
        x = np.linspace(xmin, xmax, N)
        ddx = differentiation_matrix(N, xmax - xmin)
        a = -0.7
        b = -0.9
        np.testing.assert_allclose(ddx @ (a * x + b), a, atol=1e-14, rtol=1e-14)
        
    def test_quadratic(self):
        """Test that the matrix is correct for a quadratic function."""
        N = 15
        xmin = -2.3
        xmax = 4.1
        x = np.linspace(xmin, xmax, N)
        ddx = differentiation_matrix(N, xmax - xmin)
        a = -0.7
        b = -0.9
        c = 0.3
        np.testing.assert_allclose(ddx @ (a * x * x + b * x + c), 2 * a * x + b, atol=1e-14, rtol=1e-14)


if __name__ == "__main__":
    unittest.main()
