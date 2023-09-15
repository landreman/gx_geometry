#!/usr/bin/env python

import unittest
import os
import logging

import numpy as np
from desc.vmec import VMECIO

from gx_geometry.desc import desc_fieldline
from gx_geometry.vmec import Vmec
from gx_geometry.vmec_diagnostics import vmec_fieldlines

from . import TEST_DIR

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)

class Tests(unittest.TestCase):
    def test_wout_as_desc(self):
        filename_base = "wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc"
        filename = os.path.join(TEST_DIR, filename_base)
        eq = VMECIO.load(filename)

        s = 0.5
        alpha = 0.0
        ntheta = 49
        theta1d = np.linspace(-np.pi, np.pi, ntheta)
        fl1 = desc_fieldline(eq, s, alpha, theta1d=theta1d)
        vmec = Vmec(filename)
        fl2 = vmec_fieldlines(vmec, s, alpha, theta1d=theta1d)

        # Jacobian of (rho, theta, phi) is negative for vmec, positive for desc,
        # as theta_desc = -theta_vmec. Therefore iota flips sign.
        np.testing.assert_allclose(fl1.iota, -fl2.iota)
        np.testing.assert_allclose(fl1.shat, fl2.shat, rtol=0.0002)
        np.testing.assert_allclose(fl1.theta_pest, fl2.theta_pest)
        np.testing.assert_allclose(fl1.theta_desc, fl2.theta_vmec)
        np.testing.assert_allclose(fl1.zeta, -fl2.phi)


if __name__ == "__main__":
    unittest.main()
