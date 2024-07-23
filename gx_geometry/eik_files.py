import numpy as np
from netCDF4 import Dataset
from .util import Struct

__all__ = ["read_eik", "read_eik_text", "read_eik_netcdf", "write_eik"]


def read_eik(filename):
    """Read an eik file in either plain-text or netcdf format and store the results in a data structure.

    Args:
        filename: Name of the file to read

    Returns:
        Data structure with all the geometric quantities as attributes.
    """

    if filename.endswith(".nc"):
        return read_eik_netcdf(filename)
    else:
        return read_eik_text(filename)


def read_eik_text(filename):
    """Read an eik file in plain-text format and store the results in a data structure.

    Args:
        filename: Name of the file to read

    Returns:
        Data structure with all the geometric quantities as attributes.
    """
    fl = Struct()
    fl.nalpha = 1
    fl.ns = 1
    with open(filename, "r") as f:
        f.readline()  # Names of the scalars
        line = f.readline().split(" ")
        fl.nl = int(line[2]) + 1
        fl.shat = float(line[5])
        fl.kxfac = float(line[6])
        fl.sigma_Bxy = fl.kxfac
        fl.iota = 1 / float(line[7])

    data = np.loadtxt(filename, skiprows=3, max_rows=fl.nl)
    fl.gbdrift = data[:, 0]
    fl.gradpar = data[:, 1]
    fl.grho = data[:, 2]
    fl.z = data[:, 3]

    data = np.loadtxt(filename, skiprows=4 + fl.nl, max_rows=fl.nl)
    fl.cvdrift = data[:, 0]
    fl.gds2 = data[:, 1]
    fl.bmag = data[:, 2]

    data = np.loadtxt(filename, skiprows=5 + 2 * fl.nl, max_rows=fl.nl)
    fl.gds21 = data[:, 0]
    fl.gds22 = data[:, 1]

    data = np.loadtxt(filename, skiprows=6 + 3 * fl.nl, max_rows=fl.nl)
    fl.cvdrift0 = data[:, 0]
    fl.gbdrift0 = data[:, 1]

    return fl


def read_eik_netcdf(filename):
    """Read an eik file in netcdf format and store the results in a data structure.

    Args:
        filename: Name of the file to read

    Returns:
        Data structure with all the geometric quantities as attributes.
    """
    fl = Struct()
    fl.nalpha = 1
    fl.ns = 1

    with Dataset(filename) as f:
        fl.shat = f.variables["shat"][()]
        fl.kxfac = f.variables["kxfac"][()]
        fl.sigma_Bxy = fl.kxfac
        fl.iota = 1 / f.variables["q"][()]
        fl.gbdrift = f.variables["gbdrift"][()]
        fl.gradpar = f.variables["gradpar"][()]
        fl.grho = f.variables["grho"][()]
        fl.z = f.variables["theta"][()]
        fl.cvdrift = f.variables["cvdrift"][()]
        fl.gds2 = f.variables["gds2"][()]
        fl.bmag = f.variables["bmag"][()]
        fl.gds21 = f.variables["gds21"][()]
        fl.gds22 = f.variables["gds22"][()]
        fl.cvdrift0 = f.variables["cvdrift0"][()]
        fl.gbdrift0 = f.variables["gbdrift0"][()]
        fl.nl = len(fl.bmag)
        fl.Rmajor_p = f.variables["Rmaj"][()]
        fl.alpha = f.variables["alpha"][()]
        fl.phi_center = f.variables["zeta_center"][()]
        fl.nfp = f.variables["nfp"][()]

    return fl


def write_eik(fl, filename):
    """
    Write an eik file that GX can read, in either plain-text or NetCDF format.

    Args:
        fl: A structure with data on a field line.
        filename: Name of the eik file to write.
    """
    if filename.endswith(".nc"):
        write_eik_netcdf(fl, filename)
    else:
        write_eik_text(fl, filename)


def write_eik_text(fl, filename):
    """
    Write an eik file that GX can read, in plain-text format.

    Args:
        fl: A structure with data on a field line.
        filename: Name of the eik file to write.
    """
    nperiod = 1
    assert fl.nl % 2 == 1
    ntheta = fl.nl - 1
    ntgrid = int(ntheta / 2)
    kxfac = fl.kxfac
    drhodpsi = 1.0
    rmaj = fl.Rmajor_p
    shat = fl.shat
    q = 1 / fl.iota
    scale = 1.0
    with open(filename, "w") as f:
        f.write(f"ntgrid nperiod ntheta drhodpsi rmaj shat kxfac q scale\n")
        f.write(
            f"{ntgrid} {nperiod} {ntheta} {drhodpsi} {rmaj} {shat} {kxfac} {q} {scale}\n"
        )
        f.write(f"gbdrift  gradpar         grho    tgrid\n")
        for j in range(fl.nl):
            f.write(
                f"{fl.gbdrift[j]:23} {fl.gradpar[j]:23} {fl.grho[j]:23} {fl.z[j]:23}\n"
            )
        f.write(f"cvdrift  gds2    bmag    tgrid\n")
        for j in range(fl.nl):
            f.write(
                f"{fl.cvdrift[j]:23} {fl.gds2[j]:23} {fl.bmag[j]:23} {fl.z[j]:23}\n"
            )
        f.write(f"gds21    gds22   tgrid\n")
        for j in range(fl.nl):
            f.write(f"{fl.gds21[j]:23} {fl.gds22[j]:23} {fl.z[j]:23}\n")
        f.write(f"cvdrift0         gbdrift0        tgrid\n")
        for j in range(fl.nl):
            f.write(f"{fl.cvdrift0[j]:23} {fl.gbdrift0[j]:23} {fl.z[j]:23}\n")


def write_eik_netcdf(fl, filename):
    """
    Write an eik file that GX can read, in NetCDF format.

    Args:
        fl: A structure with data on a field line.
        filename: Name of the eik file to write.
    """
    assert fl.nl % 2 == 1
    file = Dataset(filename, mode="w", format="NETCDF3_64BIT_OFFSET")
    file.createDimension("z", fl.nl)

    # Scalars:

    var = file.createVariable("sigma_Bxy", np.float64)
    var[:] = fl.sigma_Bxy

    var = file.createVariable("drhodpsi", np.float64)
    var[:] = 1.0

    var = file.createVariable("kxfac", np.float64)
    var[:] = fl.kxfac

    var = file.createVariable("Rmaj", np.float64)
    var[:] = fl.Rmajor_p

    var = file.createVariable("q", np.float64)
    var[:] = 1 / fl.iota

    var = file.createVariable("shat", np.float64)
    var[:] = fl.shat

    var = file.createVariable("scale", np.float64)
    var[:] = 1.0

    var = file.createVariable("alpha", np.float64)
    var[:] = fl.alpha

    var = file.createVariable("zeta_center", np.float64)
    var[:] = fl.phi_center

    var = file.createVariable("nfp", np.int32)
    var[:] = fl.nfp

    # 1D arrays

    var = file.createVariable("theta", np.float64, ("z",))
    var[:] = fl.z

    vars = [
        "bmag",
        "gradpar",
        "grho",
        "gds2",
        "gds21",
        "gds22",
        "gbdrift",
        "gbdrift0",
        "cvdrift",
        "cvdrift0",
    ]
    for varname in vars:
        var = file.createVariable(varname, np.float64, ("z",))
        var[:] = fl.__getattribute__(varname)

    vars = ["jacob", "Rplot", "Zplot"]
    for varname in vars:
        var = file.createVariable(varname, np.float64, ("z",))
        var[:] = 0.0
