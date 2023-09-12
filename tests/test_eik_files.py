#!/usr/bin/env python

import unittest
import os
import logging

import numpy as np

from gx_geometry import uniform_arclength, Vmec, vmec_fieldlines, add_gx_definitions, write_eik, read_eik

from . import TEST_DIR

logger = logging.getLogger(__name__)


class Tests(unittest.TestCase):
    def test_write_read_eik(self):
        filename = "wout_w7x_from_gx_repository.nc"
        vmec = Vmec(os.path.join(TEST_DIR, filename))
        s = 0.64
        alpha = 0
        nl = 49
        theta1d = np.linspace(-np.pi, np.pi, nl)
        fl1 = vmec_fieldlines(vmec, s, alpha, theta1d=theta1d)
        fl2 = uniform_arclength(fl1)
        kxfac = -1.0
        add_gx_definitions(fl2, kxfac)
        eik_filename = "eik.out"
        write_eik(fl2, eik_filename)

        # Now read in the eik file we wrote:
        fl3 = read_eik(eik_filename)

        variables = [
            "ns", "nl", "nalpha", "shat", "iota", "kxfac", 
            "z", "bmag", "gds2", "gds21", "gds22", "cvdrift", "gbdrift", "cvdrift0", "gbdrift0", "grho", "gradpar",
        ]
        for variable in variables:
            np.testing.assert_allclose(fl2.__getattribute__(variable), fl3.__getattribute__(variable))


if __name__ == "__main__":
    unittest.main()
