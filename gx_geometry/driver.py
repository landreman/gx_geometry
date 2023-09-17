import numpy as np
from .gx_grid import uniform_arclength, add_gx_definitions
from .eik_files import write_eik
from .vmec import Vmec
from .vmec_diagnostics import vmec_fieldline

__all__ = ["create_eik"]

def create_eik(filename, sigma_Bxy=1.0, s=0.64, alpha=0, nl=49):
    vmec = Vmec(filename)
    theta1d = np.linspace(-np.pi, np.pi, nl)
    fl1 = vmec_fieldline(vmec, s, alpha, theta1d=theta1d)
    fl2 = uniform_arclength(fl1)
    add_gx_definitions(fl2, sigma_Bxy)
    eik_filename = "eik.out"
    write_eik(fl2, eik_filename)
