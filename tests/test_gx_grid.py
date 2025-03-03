#!/usr/bin/env python

import unittest
import os
import logging

import numpy as np
from numpy.lib import NumpyVersion
if NumpyVersion(np.__version__) < '2.0.0':
    from numpy import trapz as trapezoid
else:
    from numpy import trapezoid as trapezoid

from gx_geometry.util import Struct
from gx_geometry import (
    uniform_arclength,
    Vmec,
    vmec_fieldline,
    resample,
    vmec_fieldline_from_center,
)

from . import TEST_DIR

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


class UniformArclengthTests(unittest.TestCase):
    def test_unchanged(self):
        """If you start with a uniform arclength grid, nothing should change."""
        fl1 = Struct()
        nl = 15
        fl1.nl = nl
        fl1.modB = np.random.rand(nl)
        fl1.theta_pest = np.zeros_like(fl1.modB)
        fl1.gradpar_theta_pest = np.zeros_like(fl1.modB)
        theta1d = np.linspace(-1, 1, nl)
        fl1.theta_pest = theta1d
        fl1.gradpar_theta_pest = np.ones(nl) * (np.random.rand() - 0.5) * 2

        fl2 = uniform_arclength(fl1)
        np.testing.assert_allclose(fl1.modB, fl2.modB)

    def test_sanity(self):
        """Run a bunch of checks for a real equlibrium."""
        filename = "wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc"
        vmec = Vmec(os.path.join(TEST_DIR, filename))
        s = 0.5
        alpha = 0.0
        nl = 201
        theta1d = np.linspace(-2.9, 2.9, nl)
        fl1 = vmec_fieldline(vmec, s, alpha, theta1d=theta1d)
        fl2 = uniform_arclength(fl1)

        # First and last elements of z should be +/- pi:
        np.testing.assert_allclose(fl1.z[0], -np.pi)
        np.testing.assert_allclose(fl1.z[-1], np.pi)
        np.testing.assert_allclose(fl2.z[0], -np.pi)
        np.testing.assert_allclose(fl2.z[-1], np.pi)

        for varname in dir(fl1):
            var1 = fl1.__getattribute__(varname)
            if isinstance(var1, np.ndarray) and var1.ndim == 3 and varname != "z":
                var2 = fl2.__getattribute__(varname)
                # Leftmost and rightmost points should match:
                np.testing.assert_allclose(var1[0], var2[0], atol=1e-12)
                np.testing.assert_allclose(var1[-1], var2[-1], atol=1e-12)
                # Max and min should match:
                np.testing.assert_allclose(
                    np.max(var1, axis=2), np.max(var2, axis=2), rtol=0.03, atol=1e-12
                )
                np.testing.assert_allclose(
                    np.min(var1, axis=2), np.min(var2, axis=2), rtol=0.03, atol=1e-12
                )

        # Compute arclength another way
        L = trapezoid(1 / fl1.gradpar_theta_pest, fl1.theta_pest)
        np.testing.assert_allclose(L, fl2.arclength[-1])

        """
        # Check that linear solve was correct:
        for js in range(fl1.ns):
            for jalpha in range(fl1.nalpha):
                # Compute physical length of flux tube:
                L = trapezoid(1 / fl1.gradpar_theta_pest[js, jalpha, :], fl1.theta_pest[js, jalpha, :])
                gradpar2 = 2 * np.pi / L
                rhs = gradpar2 / fl1.gradpar_theta_pest[js, jalpha, :]
                print("post D:")
                print(fl2.D)
                print("post rhs:", rhs)
                print("post fl1.z:", fl1.z[js, jalpha, :])
                np.testing.assert_allclose(fl2.D @ (fl1.z[js, jalpha, :] +
                np.pi), rhs)
        """


class ResampleTests(unittest.TestCase):
    def test_2_ways(self):
        """
        If you make a field line at nz1 and resample to nz2, it should
        match a field line created originally at nz2.
        """

        nz1 = 400
        nz2 = 251

        filename_base = "wout_w7x_from_gx_repository.nc"
        filename = os.path.join(TEST_DIR, filename_base)
        vmec = Vmec(filename)
        s = 0.9
        theta0 = np.pi / 3
        zeta0 = 0.8 * np.pi / 5
        poloidal_turns = 0.7
        types_to_compare = (int, float, np.ndarray)

        # Lower the resolution:
        fl1 = vmec_fieldline_from_center(vmec, s, theta0, zeta0, poloidal_turns, nz1)
        fl1 = uniform_arclength(fl1)
        fl2 = resample(fl1, nz2)
        fl3 = vmec_fieldline_from_center(vmec, s, theta0, zeta0, poloidal_turns, nz2)
        fl3 = uniform_arclength(fl3)
        varnames = set(dir(fl2) + dir(fl3))
        for varname in varnames:
            var1 = fl2.__getattribute__(varname)
            var2 = fl3.__getattribute__(varname)
            if isinstance(var1, types_to_compare):
                np.testing.assert_allclose(var1, var2, rtol=3e-4, atol=3e-4)

        # Raise the resolution:
        fl4 = resample(fl3, nz1)
        varnames = set(dir(fl1) + dir(fl4))
        for varname in varnames:
            var1 = fl1.__getattribute__(varname)
            var2 = fl4.__getattribute__(varname)
            if isinstance(var1, types_to_compare):
                np.testing.assert_allclose(var1, var2, rtol=3e-4, atol=3e-4)


if __name__ == "__main__":
    unittest.main()
