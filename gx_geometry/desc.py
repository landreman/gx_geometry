import numpy as np
from desc.grid import Grid, LinearGrid
from desc.compute.utils import cross, dot
from .util import Struct

__all__ = ["desc_fieldline"]

def desc_fieldline(eq, s, alpha, theta1d):
    psi = float(eq.Psi / (2 * np.pi))
    toroidal_flux_sign = np.sign(psi)
    rho = np.sqrt(s)
    sqrt_s = rho
    edge_toroidal_flux_over_2pi = psi

    global_quantities = eq.compute(["a", "R0"])

    # Compute flux functions on the surface of interest:
    flux_function_keys = ["iota", "iota_r", "p_r"]
    linear_grid = LinearGrid(rho=rho, M=34, N=35, NFP=eq.NFP)
    flux_functions = eq.compute(flux_function_keys, grid=linear_grid)

    iota = float(linear_grid.compress(flux_functions["iota"])[0])
    iota_r = float(linear_grid.compress(flux_functions["iota_r"])[0])
    shat = -(rho / iota) * iota_r
    d_pressure_d_s = float(linear_grid.compress(flux_functions["p_r"])[0] / (2 * rho))

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
    theta_desc = np.array(coords[:, 1])

    # desc's compute_theta_coords forces the resulting theta to lie in [0, 2pi].
    # Undo this by assuming that |theta - theta_desc| is < pi.
    multiples_of_2pi_to_shift = np.round((theta - theta_desc) / (2 * np.pi))
    theta_desc += 2 * np.pi * multiples_of_2pi_to_shift

    theta_vmec = theta_desc
    # In next line, if we don't set sort=False, desc flips the direction of the
    # grid when iota < 0!
    grid = Grid(coords, sort=False)

    field_line_keys = [
        "|B|", "|grad(psi)|^2", "grad(|B|)", "grad(alpha)", "grad(psi)",
        "B", "grad(|B|)", "kappa", "B^theta", "B^zeta", "lambda_t", "lambda_z",
    ]
    data = eq.compute(field_line_keys, grid=grid)

    L_reference = float(global_quantities['a'])
    B_reference = float(2 * np.abs(psi) / (L_reference**2))
    Rmajor_p = float(global_quantities['R0'])

    # Convert jax arrays to numpy arrays
    modB = np.array(data['|B|'])
    bmag = modB / B_reference
    gradpar_theta_pest = np.array(L_reference * (data["B^theta"] * (1 + data["lambda_t"]) + data["B^zeta"] * data["lambda_z"]) / data["|B|"])

    grad_psi_dot_grad_psi = np.array(data["|grad(psi)|^2"])
    grad_alpha_dot_grad_psi = np.array(dot(data["grad(alpha)"], data["grad(psi)"]))
    grad_alpha_dot_grad_alpha = np.array(dot(data["grad(alpha)"], data["grad(alpha)"]))

    grho = np.sqrt(grad_psi_dot_grad_psi / (L_reference * L_reference * B_reference * B_reference * s))
    gds2 = grad_alpha_dot_grad_alpha * L_reference * L_reference * s
    gds22 = grad_psi_dot_grad_psi * shat * shat / (L_reference * L_reference * B_reference * B_reference * s)

    B_cross_grad_B_dot_grad_psi = np.array(dot(cross(data["B"], data["grad(|B|)"]), data["grad(psi)"]))
    B_cross_grad_B_dot_grad_alpha = np.array(dot(cross(data["B"], data["grad(|B|)"]), data["grad(alpha)"]))
    B_cross_kappa_dot_grad_psi = np.array(dot(cross(data["B"], data["kappa"]), data["grad(psi)"]))
    B_cross_kappa_dot_grad_alpha = np.array(dot(cross(data["B"], data["kappa"]), data["grad(alpha)"]))

    fl = Struct()
    variables = [
        "s", "rho", "nl", "theta_pest", "theta_desc", "theta_vmec", "zeta", "phi",
        "edge_toroidal_flux_over_2pi", "Rmajor_p",
        "iota", "shat", "B_reference", "L_reference", "toroidal_flux_sign", "sqrt_s",
        "modB", "bmag", "gradpar_theta_pest", "d_pressure_d_s",
        "grad_psi_dot_grad_psi", "grad_alpha_dot_grad_psi", "grad_alpha_dot_grad_alpha",
        "grho", "gds2", "gds22",
        "B_cross_grad_B_dot_grad_psi", "B_cross_grad_B_dot_grad_alpha",
        "B_cross_kappa_dot_grad_psi", "B_cross_kappa_dot_grad_alpha",
    ]
    for v in variables:
        val = eval(v)
        #print("  ", v, "has type", type(val))
        if not isinstance(val, (int, float, np.ndarray)):
            raise RuntimeError(f"Variable {v} may still have a jax type: {type(val)}")
        fl.__setattr__(v, val)
    return fl
