import unittest
import os
import logging

import numpy as np
import matplotlib.pyplot as plt

from gx_geometry import create_eik_from_vmec, create_eik_from_desc

from . import TEST_DIR

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


class Tests(unittest.TestCase):
    def test_vmec_driver(self):
        filename_base = "wout_w7x_from_gx_repository.nc"
        filename = os.path.join(TEST_DIR, filename_base)
        create_eik_from_vmec(filename)
        create_eik_from_vmec(filename, eik_filename="eik.nc")

    def test_desc_driver(self):
        filename_base = "w7x_from_gx_repository_LMN8.h5"
        filename = os.path.join(TEST_DIR, filename_base)
        create_eik_from_desc(filename)
        create_eik_from_desc(filename, eik_filename="eik.nc")


if __name__ == "__main__":
    unittest.main()
