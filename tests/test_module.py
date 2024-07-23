import unittest
import os

from gx_geometry.module import run_module

from . import TEST_DIR


class Tests(unittest.TestCase):
    def test_run_module(self):
        cwd = os.getcwd()
        os.chdir(TEST_DIR)

        print(os.path.join(TEST_DIR, "w7x_adiabatic_electrons_vmec.in"))
        run_module([0, "w7x_adiabatic_electrons_vmec.in"])
        run_module([0, "w7x_adiabatic_electrons_desc.in"])
        run_module([0, "w7x_adiabatic_electrons_vmec.in", "w7x.eik.nc"])

        os.chdir(cwd)
