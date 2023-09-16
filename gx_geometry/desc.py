import numpy as np
from desc.grid import Grid, LinearGrid
from desc.compute.utils import cross, dot
from .util import Struct

def desc_fieldline(eq, s, alpha, theta1d):
    psi = eq.Psi / (2 * np.pi)
    rho = np.sqrt(s)

    # Compute flux functions on the surface of interest:
    flux_function_keys = ["iota", "iota_r", "p_r", "a"]
    linear_grid = LinearGrid(rho=rho, M=34, N=35, NFP=eq.NFP)
    flux_functions = eq.compute(flux_function_keys, grid=linear_grid)

    iota = linear_grid.compress(flux_functions["iota"])
    iota_r = linear_grid.compress(flux_functions["iota_r"])
    shat = -(rho / iota) * iota_r

    nl = len(theta1d)
    theta = theta1d
    theta_pest = theta
    # alpha = theta - iota * zeta
    # zeta = (theta - alpha) / iota
    zeta = (theta - alpha) / iota
    phi = zeta
    rhoa = rho * np.ones(nl)
    c = np.vstack([rhoa, theta, zeta]).T
    coords = eq.compute_theta_coords(c, tol=1e-10, maxiter=50)
    theta_desc = coords[:, 1]
    theta_vmec = theta_desc
    grid = Grid(coords)

    field_line_keys = [
        "|B|", "|grad(psi)|^2", "grad(|B|)", "grad(alpha)", "grad(psi)",
        "B", "grad(|B|)", "kappa", "B^theta", "B^zeta", "lambda_t", "lambda_z",
    ]
    data = eq.compute(field_line_keys, grid=grid)

    #normalizations       
    L_reference = flux_functions['a']
    B_reference = 2 * np.abs(psi) / (L_reference**2)

    modB = data['|B|']
    gradpar_theta_pest = L_reference * (data["B^theta"] * (1 + data["lambda_t"]) + data["B^zeta"] * data["lambda_z"]) / data["|B|"]

    grad_psi_dot_grad_psi = data["|grad(psi)|^2"]
    grad_alpha_dot_grad_psi = dot(data["grad(alpha)"], data["grad(psi)"])
    grad_alpha_dot_grad_alpha = dot(data["grad(alpha)"], data["grad(alpha)"])

    B_cross_grad_B_dot_grad_psi = dot(cross(data["B"], data["grad(|B|)"]), data["grad(psi)"])
    B_cross_grad_B_dot_grad_alpha = dot(cross(data["B"], data["grad(|B|)"]), data["grad(alpha)"])
    B_cross_kappa_dot_grad_psi = dot(cross(data["B"], data["kappa"]), data["grad(psi)"])
    B_cross_kappa_dot_grad_alpha = dot(cross(data["B"], data["kappa"]), data["grad(alpha)"])

    fl = Struct()
    variables = [
        "s", "rho", "nl", "theta_pest", "theta_desc", "theta_vmec", "zeta", "phi",
        "iota", "shat", "B_reference", "L_reference",
        "modB", "gradpar_theta_pest",
        "grad_psi_dot_grad_psi", "grad_alpha_dot_grad_psi", "grad_alpha_dot_grad_alpha",
        "B_cross_grad_B_dot_grad_psi", "B_cross_grad_B_dot_grad_alpha",
        "B_cross_kappa_dot_grad_psi", "B_cross_kappa_dot_grad_alpha",
    ]
    for v in variables:
        fl.__setattr__(v, eval(v))
    return fl
