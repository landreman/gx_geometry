"""
This module contains small utility functions.
"""

import numpy as np

__all__ = ["mu_0", "differentiation_matrix"]

mu_0 = 4 * np.pi * (1.0e-7)

class Struct:
    """
    This class is just a dummy mutable object to which we can add attributes.
    """

def differentiation_matrix(N, length):
    """Return a differentiation matrix for a uniform grid.
    
    Args:
        N: Number of grid points.
        length: distance between the maximum and minimum grid points.
    """
    dx = length / (N - 1)
    D = np.diag(0.5 * np.ones(N - 1), 1) - np.diag(0.5 * np.ones(N - 1), -1)
    D[0, 0] = -1.5
    D[0, 1] = 2
    D[0, 2] = -0.5

    D[-1, -1] = 1.5
    D[-1, -2] = -2
    D[-1, -3] = 0.5

    return D / dx
