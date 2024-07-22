import unittest
import os
import logging

import numpy as np
import matplotlib.pyplot as plt

from gx_geometry import (
    create_eik_from_vmec,
    create_eik_from_desc,
    create_eik_from_desc_given_length,
)

from . import TEST_DIR

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


def assert_symmetric(arr, sign):
    """Check that the array is symmetric about the center."""
    np.testing.assert_allclose(arr, sign * arr[::-1], atol=1e-14)


def check_field_line_symmetry(fl):
    assert_symmetric(fl.modB, 1)
    assert_symmetric(fl.gbdrift, 1)
    assert_symmetric(fl.gbdrift0, -1)
    assert_symmetric(fl.cvdrift, 1)
    assert_symmetric(fl.cvdrift0, -1)
    assert_symmetric(fl.gds2, 1)
    assert_symmetric(fl.gds21, -1)
    assert_symmetric(fl.gds22, 1)
    np.testing.assert_allclose(fl.z, np.linspace(-np.pi, np.pi, fl.nl), atol=1e-14)


class Tests(unittest.TestCase):
    def test_vmec_driver(self):
        filename_base = "wout_w7x_from_gx_repository.nc"
        filename = os.path.join(TEST_DIR, filename_base)
        s = 0.64
        create_eik_from_vmec(filename, s=s)
        create_eik_from_vmec(filename, s=s, eik_filename="eik.nc")

    def test_desc_driver(self):
        filename_base = "w7x_from_gx_repository_LMN8.h5"
        filename = os.path.join(TEST_DIR, filename_base)
        s = 0.9
        create_eik_from_desc(filename, s=s)
        create_eik_from_desc(filename, s=s, eik_filename="eik.nc")

    def test_desc_driver_from_length(self):
        filename_base = "w7x_from_gx_repository_LMN8.h5"
        filename = os.path.join(TEST_DIR, filename_base)
        s = 1.0
        Rmajor = 5.5
        length = 2 * np.pi * Rmajor * 2
        create_eik_from_desc_given_length(filename, s=s, length=length)
        create_eik_from_desc_given_length(
            filename, s=s, length=length, eik_filename="eik.nc"
        )

    def test_vmec_driver_symmetric(self):
        """GX geometry inputs should be stellarator symmetric when expected."""
        filename_base = "wout_w7x_from_gx_repository.nc"
        filename = os.path.join(TEST_DIR, filename_base)
        nfp = 5
        s = 0.5
        for theta0 in [0, np.pi]:
            for zeta0 in [0, np.pi / nfp]:
                print("theta0:", theta0, " zeta0:", zeta0)
                fl = create_eik_from_vmec(filename, s=s, theta0=theta0, zeta0=zeta0)
                check_field_line_symmetry(fl)

    def test_vmec_driver_symmetric_cut(self):
        """GX geometry inputs should be stellarator symmetric when expected."""
        filename_base = "wout_w7x_from_gx_repository.nc"
        filename = os.path.join(TEST_DIR, filename_base)
        nfp = 5
        s = 0.5
        params = {"Geometry": {"npol_min": 2}, "Domain": {"boundary": "fix aspect", "jtwist": 1}}
        for theta0 in [0, np.pi]:
            for zeta0 in [0, np.pi / nfp]:
                print("theta0:", theta0, " zeta0:", zeta0)
                fl = create_eik_from_vmec(filename, s=s, theta0=theta0, zeta0=zeta0, **params)
                check_field_line_symmetry(fl)

    def test_desc_driver_symmetric(self):
        """GX geometry inputs should be stellarator symmetric when expected."""
        filename_base = "w7x_from_gx_repository_LMN8.h5"
        filename = os.path.join(TEST_DIR, filename_base)
        nfp = 5
        s = 0.25
        for theta0 in [0, np.pi]:
            for zeta0 in [0, np.pi / nfp]:
                print("theta0:", theta0, " zeta0:", zeta0)
                fl = create_eik_from_desc(filename, s=s, theta0=theta0, zeta0=zeta0)
                check_field_line_symmetry(fl)

    def test_desc_driver_symmetric_cut(self):
        """GX geometry inputs should be stellarator symmetric when expected."""
        filename_base = "w7x_from_gx_repository_LMN8.h5"
        filename = os.path.join(TEST_DIR, filename_base)
        nfp = 5
        s = 0.25
        params = {"Geometry": {"npol_min": 2}, "Domain": {"boundary": "fix aspect", "jtwist": 1}}
        for theta0 in [0, np.pi]:
            for zeta0 in [0, np.pi / nfp]:
                print("theta0:", theta0, " zeta0:", zeta0)
                fl = create_eik_from_desc(filename, s=s, theta0=theta0, zeta0=zeta0, **params)
                check_field_line_symmetry(fl)


if __name__ == "__main__":
    unittest.main()
