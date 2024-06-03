import numpy as np

try:
    import desc.io
    from desc.equilibrium import Equilibrium
    from .desc import desc_fieldline
except ImportError:
    pass

from .gx_grid import uniform_arclength, add_gx_definitions
from .eik_files import write_eik
from .vmec import Vmec
from .vmec_diagnostics import vmec_fieldline

__all__ = ["create_eik_from_vmec", "create_eik_from_desc"]


def create_eik_from_vmec(
    filename,
    s=0.64,
    theta0=0,
    zeta0=0,
    nz=49,
    poloidal_turns=1,
    sigma_Bxy=-1.0,
    eik_filename="eik.out",
):
    r"""Driver to create an eik file that GX can read from a vmec wout file.

    Args:
        filename: Name of a vmec wout file to read.
        s: Normalized toroidal flux for the field line.
        theta0, zeta0: Center of the field line.
        nz: Number of grid points parallel to the field.
        poloidal_turns: Number of poloidal turns to cover in the parallel direction.
        sigma_Bxy: (1 / |B|^2) \vec{B} \cdot \nabla x \times \nabla y, usually -1.
        eik_filename: Name of the eik file to save.

    Returns:
        fl: A Struct containing the field line data.
    """
    vmec = Vmec(filename)
    theta1d = np.linspace(
        theta0 - np.pi * poloidal_turns, theta0 + np.pi * poloidal_turns, nz
    )
    alpha = theta0
    fl1 = vmec_fieldline(vmec, s, alpha, theta1d=theta1d, phi_center=zeta0)
    fl2 = uniform_arclength(fl1)
    add_gx_definitions(fl2, sigma_Bxy)
    write_eik(fl2, eik_filename)

    return fl2


def create_eik_from_desc(
    eq,
    s=0.64,
    theta0=0,
    zeta0=0,
    nz=49,
    poloidal_turns=1,
    sigma_Bxy=-1.0,
    eik_filename="eik.out",
):
    r"""Driver to create an eik file that GX can read from a desc .h5 output file.

    Args:
        eq: DESC Equilibrium object, or name of a desc .h5 output file to read.
        s: Normalized toroidal flux for the field line.
        theta0, zeta0: Center of the field line.
        nz: Number of grid points parallel to the field.
        poloidal_turns: Number of poloidal turns to cover in the parallel direction.
        sigma_Bxy: (1 / |B|^2) \vec{B} \cdot \nabla x \times \nabla y, usually -1.
        eik_filename: Name of the eik file to save.

    Returns:
        fl: A Struct containing the field line data.
    """
    if not isinstance(eq, Equilibrium):
        eq = desc.io.load(eq)
        try:
            # If an EquilibriumFamily, choose the final equilibrium:
            eq = eq[-1]
        except:
            pass

    theta1d = np.linspace(
        theta0 - np.pi * poloidal_turns, theta0 + np.pi * poloidal_turns, nz
    )
    alpha = theta0
    fl1 = desc_fieldline(eq, s, alpha, theta1d=theta1d, zeta0=zeta0)
    fl2 = uniform_arclength(fl1)
    add_gx_definitions(fl2, sigma_Bxy)
    write_eik(fl2, eik_filename)

    return fl2
