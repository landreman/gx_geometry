import numpy as np
from .util import Struct

__all__ = ["read_eik", "write_eik"]

def read_eik(filename):
    """Read an eik file and store the results in a data structure.
    
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
        fl.iota = 1 / float(line[7])

    data = np.loadtxt(filename, skiprows=3, max_rows=fl.nl)
    fl.gbdrift = data[:, 0].reshape((1, 1, fl.nl))
    fl.gradpar = data[:, 1].reshape((1, 1, fl.nl))
    fl.grho = data[:, 2].reshape((1, 1, fl.nl))
    fl.z = data[:, 3].reshape((1, 1, fl.nl))

    data = np.loadtxt(filename, skiprows=4 + fl.nl, max_rows=fl.nl)
    fl.cvdrift = data[:, 0].reshape((1, 1, fl.nl))
    fl.gds2 = data[:, 1].reshape((1, 1, fl.nl))
    fl.bmag = data[:, 2].reshape((1, 1, fl.nl))

    data = np.loadtxt(filename, skiprows=5 + 2 * fl.nl, max_rows=fl.nl)
    fl.gds21 = data[:, 0].reshape((1, 1, fl.nl))
    fl.gds22 = data[:, 1].reshape((1, 1, fl.nl))

    data = np.loadtxt(filename, skiprows=6 + 3 * fl.nl, max_rows=fl.nl)
    fl.cvdrift0 = data[:, 0].reshape((1, 1, fl.nl))
    fl.gbdrift0 = data[:, 1].reshape((1, 1, fl.nl))

    return fl


def write_eik(fl, filename):
    """
    Write an eik file that GX can read.
    
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
    shat = fl.shat[0]
    q = 1 / fl.iota[0]
    scale = 1.0
    with open(filename, "w") as f:
        f.write(f"ntgrid nperiod ntheta drhodpsi rmaj shat kxfac q scale\n")
        f.write(f"{ntgrid} {nperiod} {ntheta} {drhodpsi} {rmaj} {shat} {kxfac} {q} {scale}\n")
        f.write(f"gbdrift  gradpar         grho    tgrid\n")
        for j in range(fl.nl):
            f.write(f"{fl.gbdrift[0, 0, j]:23} {fl.gradpar[0, 0, j]:23} {fl.grho[0, 0, j]:23} {fl.z[0, 0, j]:23}\n")
        f.write(f"cvdrift  gds2    bmag    tgrid\n")
        for j in range(fl.nl):
            f.write(f"{fl.cvdrift[0, 0, j]:23} {fl.gds2[0, 0, j]:23} {fl.bmag[0, 0, j]:23} {fl.z[0, 0, j]:23}\n")
        f.write(f"gds21    gds22   tgrid\n")
        for j in range(fl.nl):
            f.write(f"{fl.gds21[0, 0, j]:23} {fl.gds22[0, 0, j]:23} {fl.z[0, 0, j]:23}\n")
        f.write(f"cvdrift0         gbdrift0        tgrid\n")
        for j in range(fl.nl):
            f.write(f"{fl.cvdrift0[0, 0, j]:23} {fl.gbdrift0[0, 0, j]:23} {fl.z[0, 0, j]:23}\n")
