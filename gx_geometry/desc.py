import numpy as np
from desc.grid import Grid, LinearGrid
from .util import Struct

def desc_fieldline(eq, s, alpha, theta1d):
    rho = np.sqrt(s)
    flux_function_keys = ["iota", "iota_r", "p_r", "psi", "a"]
    linear_grid = LinearGrid(rho=rho, M=64, N=65, NFP=eq.NFP)
    flux_functions = eq.compute(flux_function_keys, grid=linear_grid)

    iota = linear_grid.compress(flux_functions["iota"])
    iota_r = linear_grid.compress(flux_functions["iota_r"])
    shat = -(rho / iota) * iota_r
    print(flux_functions)

    nl = len(theta1d)
    theta = theta1d
    theta_pest = theta
    # alpha = theta - iota * zeta
    # zeta = (theta - alpha) / iota
    zeta = (theta - alpha) / iota
    rhoa = rho * np.ones(nl)
    c = np.vstack([rhoa, theta, zeta]).T
    coords = eq.compute_theta_coords(c, tol=1e-10, maxiter=50)
    print("coords:")
    print(coords)
    theta_desc = coords[:, 1]
    grid = Grid(coords)

    field_line_keys = [
        "|B|",
    ]
    data = eq.compute(field_line_keys, grid=grid)

    psi = flux_functions['psi'][-1]
    #normalizations       
    L_reference = flux_functions['a']
    B_reference = 2 * np.abs(psi) / (L_reference**2)
    #calculate bmag
    modB = data['|B|']
    bmag = modB / B_reference


    fl = Struct()
    variables = [
        "s", "rho", "nl", "theta_pest", "theta_desc", "zeta",
        "iota", "shat",
        "bmag"
    ]
    for v in variables:
        fl.__setattr__(v, eval(v))
    return fl
