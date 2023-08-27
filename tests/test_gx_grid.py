#!/usr/bin/env python

import unittest
import os
import logging

import numpy as np
import matplotlib

from gx_geometry.util import Struct
from gx_geometry import uniform_arclength, Vmec, vmec_fieldlines

from . import TEST_DIR

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)


class Tests(unittest.TestCase):
    def test_unchanged(self):
        """If you start with a uniform arclength grid, nothing should change."""
        fl1 = Struct()
        ns = 2
        nalpha = 3
        nl = 15
        fl1.ns = ns
        fl1.nalpha = nalpha
        fl1.nl = nl
        fl1.modB = np.random.rand(ns, nalpha, nl)
        fl1.theta_pest = np.zeros_like(fl1.modB)
        fl1.gradpar_theta_pest = np.zeros_like(fl1.modB)
        theta1d = np.linspace(-1, 1, nl)
        for js in range(ns):
            for jalpha in range(nalpha):
                fl1.theta_pest[js, jalpha, :] = theta1d
                fl1.gradpar_theta_pest[js, jalpha, :] = np.ones(nl) * (np.random.rand() - 0.5) * 2

        fl2 = uniform_arclength(fl1)
        np.testing.assert_allclose(fl1.modB, fl2.modB)

    def test_sanity(self):
        """Run a bunch of checks for a real equlibrium."""
        filename = "wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc"
        vmec = Vmec(os.path.join(TEST_DIR, filename))
        #s = [0.5, 1.0]
        s = [0.5]
        ns = len(s)
        nalpha = 1
        alpha = np.linspace(0, 2 * np.pi, nalpha, endpoint=False)
        nl = 201
        theta1d = np.linspace(-2.9, 2.9, nl)
        fl1 = vmec_fieldlines(vmec, s, alpha, theta1d=theta1d)
        fl2 = uniform_arclength(fl1)

        # First and last elements of z should be +/- pi:
        np.testing.assert_allclose(fl1.z[:, :, 0], -np.pi)
        np.testing.assert_allclose(fl1.z[:, :, -1], np.pi)
        np.testing.assert_allclose(fl2.z[:, :, 0], -np.pi)
        np.testing.assert_allclose(fl2.z[:, :, -1], np.pi)

        for varname in dir(fl1):
            var1 = fl1.__getattribute__(varname)
            if isinstance(var1, np.ndarray) and var1.ndim == 3 and varname != "z":
                var2 = fl2.__getattribute__(varname)
                # Leftmost and rightmost points should match:
                np.testing.assert_allclose(var1[:, :, 0], var2[:, :, 0], atol=1e-12)
                np.testing.assert_allclose(var1[:, :, -1], var2[:, :, -1], atol=1e-12)
                # Max and min should match:
                np.testing.assert_allclose(np.max(var1, axis=2), np.max(var2, axis=2), rtol=0.03, atol=1e-12)
                np.testing.assert_allclose(np.min(var1, axis=2), np.min(var2, axis=2), rtol=0.03, atol=1e-12)
        
        # Compute arclength another way
        for js in range(fl1.ns):
            for jalpha in range(fl1.nalpha):
                L = np.trapz(1 / fl1.gradpar_theta_pest[js, jalpha, :], fl1.theta_pest[js, jalpha, :])
                np.testing.assert_allclose(L, fl2.arclength[js, jalpha, -1])
    
        """
        # Check that linear solve was correct:
        for js in range(fl1.ns):
            for jalpha in range(fl1.nalpha):
                # Compute physical length of flux tube:
                L = np.trapz(1 / fl1.gradpar_theta_pest[js, jalpha, :], fl1.theta_pest[js, jalpha, :])
                gradpar2 = 2 * np.pi / L
                rhs = gradpar2 / fl1.gradpar_theta_pest[js, jalpha, :]
                print("post D:")
                print(fl2.D)
                print("post rhs:", rhs)
                print("post fl1.z:", fl1.z[js, jalpha, :])
                np.testing.assert_allclose(fl2.D @ (fl1.z[js, jalpha, :] +
                np.pi), rhs)
        """

if __name__ == "__main__":
    unittest.main()
