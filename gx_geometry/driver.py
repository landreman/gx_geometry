import numpy as np

try:
    import desc.io
    from desc.equilibrium import Equilibrium
    from .desc import desc_fieldline_from_center, desc_fieldline_specified_length
except ImportError:
    pass

from .gx_grid import uniform_arclength, add_gx_definitions, cut_field_line, resample
from .eik_files import write_eik
from .vmec import Vmec
from .vmec_diagnostics import vmec_fieldline_from_center

__all__ = [
    "create_eik_from_vmec",
    "create_eik_from_desc",
    "create_eik_from_desc_given_length",
]


def create_eik_from_vmec(
    filename,
    s,
    theta0=0,
    zeta0=0,
    nz=49,
    poloidal_turns=1,
    sigma_Bxy=-1.0,
    eik_filename="eik.nc",
    **params,
):
    r"""Driver to create an eik file that GX can read from a vmec wout file.

    Args:
        filename: Name of a vmec wout file to read.
        s: Normalized toroidal flux for the field line.
        theta0, zeta0: Center of the field line.
        nz: Number of grid points parallel to the field.
        poloidal_turns: Number of poloidal turns to cover in the parallel direction.
        sigma_Bxy: (1 / |B|^2) \vec{B} \cdot \nabla x \times \nabla y, usually -1.
        eik_filename: Name of the eik file to save. If name ends in '.nc' file will be written in NetCDF format, otherwise it will be written in plain-text format
        params: (optional) dictionary of parameters from GX input file, for specifying where to cut the field line in z.

    Returns:
        fl: A Struct containing the field line data.
    """
    npol_min = params.get('Geometry', {}).get("npol_min", None)
    if npol_min is not None:
        poloidal_turns = 3*npol_min
    nz_big = max(1001, nz * 3)
    vmec = Vmec(filename)
    fl1 = vmec_fieldline_from_center(vmec, s, theta0, zeta0, poloidal_turns, nz_big)
    add_gx_definitions(fl1, sigma_Bxy)
    fl2 = cut_field_line(fl1, **params)
    fl3 = uniform_arclength(fl2)
    fl4 = resample(fl3, nz)
    write_eik(fl4, eik_filename)

    return fl4


def create_eik_from_desc(
    eq,
    s,
    theta0=0,
    zeta0=0,
    nz=49,
    poloidal_turns=1,
    sigma_Bxy=-1.0,
    eik_filename="eik.out",
    **params,
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
        params: (optional) dictionary of parameters from GX input file, for specifying where to cut the field line in z.

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

    npol_min = params.get('Geometry', {}).get("npol_min", None)
    if npol_min is not None:
        poloidal_turns = 3*npol_min
    nz_big = max(1001, nz * 3)
    fl1 = desc_fieldline_from_center(eq, s, theta0, zeta0, poloidal_turns, nz_big)
    add_gx_definitions(fl1, sigma_Bxy)
    fl2 = cut_field_line(fl1, **params)
    fl3 = uniform_arclength(fl2)
    fl4 = resample(fl3, nz)
    write_eik(fl4, eik_filename)

    return fl4


def create_eik_from_desc_given_length(
    eq,
    s,
    theta0=0,
    zeta0=0,
    nz=49,
    length=None,
    sigma_Bxy=-1.0,
    eik_filename="eik.out",
):
    r"""Driver to create an eik file that GX can read from a desc .h5 output file.

    Args:
        eq: DESC Equilibrium object, or name of a desc .h5 output file to read.
        s: Normalized toroidal flux for the field line.
        theta0, zeta0: Center of the field line.
        nz: Number of grid points parallel to the field.
        length: Desired length of the field line, divided by L_reference.
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

    nz_big = max(1001, nz * 3)
    fl1 = desc_fieldline_specified_length(eq, s, theta0, zeta0, nz_big, length)
    fl2 = uniform_arclength(fl1)
    add_gx_definitions(fl2, sigma_Bxy)
    fl3 = resample(fl2, nz)
    write_eik(fl3, eik_filename)

    return fl3
