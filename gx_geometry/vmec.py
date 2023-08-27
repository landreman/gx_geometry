"""
This module provides a class that handles the VMEC equilibrium code.
"""

import logging
import os.path

import numpy as np
from scipy.io import netcdf_file

from .util import Struct

logger = logging.getLogger(__name__)

__all__ = ["Vmec"]

class Vmec():
    r"""
    This class represents the VMEC equilibrium code.

    Args:
        filename: Name of a VMEC ``wout_<extension>.nc`` output file.
    """

    def __init__(self, filename):

        self.output_file = filename
        basename = os.path.basename(filename)
        if basename[:4] == 'wout':
            logger.info(f"Initializing a VMEC object from wout file: {filename}")
        else:
            raise ValueError('Invalid filename - it must be a wout file.')

        self.wout = Struct()
        self.load_wout()

    def load_wout(self):
        """
        Read in the most recent ``wout`` file created, and store all the
        data in a ``wout`` attribute of this Vmec object.
        """
        ierr = 0
        logger.info(f"Attempting to read file {self.output_file}")

        with netcdf_file(self.output_file, mmap=False) as f:
            for key, val in f.variables.items():
                # 2D arrays need to be transposed.
                val2 = val[()]  # Convert to numpy array
                val3 = val2.T if len(val2.shape) == 2 else val2
                self.wout.__setattr__(key, val3)

            if self.wout.ier_flag != 0:
                logger.info("VMEC did not succeed!")

            # Shorthand for a long variable name:
            self.wout.lasym = f.variables['lasym__logical__'][()]
            self.wout.volume = self.wout.volume_p

        self.s_full_grid = np.linspace(0, 1, self.wout.ns)
        self.ds = self.s_full_grid[1] - self.s_full_grid[0]
        self.s_half_grid = self.s_full_grid[1:] - 0.5 * self.ds

        return ierr

    def run(self):
        """For compatibility with simsopt routines that expect this function to exist."""
        pass