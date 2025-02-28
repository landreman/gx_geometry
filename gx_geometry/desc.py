import numpy as np
from scipy.optimize import root_scalar

try:
    from desc.grid import Grid, LinearGrid
    from desc.utils import cross, dot
except ImportError:
    pass
from .util import Struct

__all__ = [
    "desc_fieldline",
    "desc_fieldline_from_center",
    "desc_fieldline_specified_length",
]


def desc_fieldline(eq, s, alpha, theta1d, zeta0=0.0):
    phi_center = zeta0
    nfp = int(eq.NFP)
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
    # alpha = theta - iota * (zeta - zeta0)
    # zeta = zeta0 + (theta - alpha) / iota
    zeta = zeta0 + (theta - alpha) / iota
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
        "|B|",
        "|grad(psi)|^2",
        "grad(|B|)",
        "grad(alpha)",
        "grad(psi)",
        "B",
        "grad(|B|)",
        "kappa",
        "B^theta",
        "B^zeta",
        "lambda_t",
        "lambda_z",
        "iota_r",
        "e^rho",
    ]
    data = eq.compute(field_line_keys, grid=grid)

    L_reference = float(global_quantities["a"])
    B_reference = float(2 * np.abs(psi) / (L_reference**2))
    Rmajor_p = float(global_quantities["R0"])

    # Desc's grad alpha does not include zeta0, so shift it appropriately
    grad_alpha_shifted = (
        data["grad(alpha)"] + zeta0 * data["iota_r"][:, None] * data["e^rho"]
    )

    # Convert jax arrays to numpy arrays
    modB = np.array(data["|B|"])
    bmag = modB / B_reference
    gradpar_theta_pest = np.array(
        L_reference
        * (data["B^theta"] * (1 + data["lambda_t"]) + data["B^zeta"] * data["lambda_z"])
        / data["|B|"]
    )
    length = float(np.abs(np.trapz(1 / gradpar_theta_pest, theta_pest)))

    grad_psi_dot_grad_psi = np.array(data["|grad(psi)|^2"])
    grad_alpha_dot_grad_psi = np.array(dot(grad_alpha_shifted, data["grad(psi)"]))
    grad_alpha_dot_grad_alpha = np.array(dot(grad_alpha_shifted, grad_alpha_shifted))

    grho = np.sqrt(
        grad_psi_dot_grad_psi
        / (L_reference * L_reference * B_reference * B_reference * s)
    )
    gds2 = grad_alpha_dot_grad_alpha * L_reference * L_reference * s
    gds22 = (
        grad_psi_dot_grad_psi
        * shat
        * shat
        / (L_reference * L_reference * B_reference * B_reference * s)
    )

    B_cross_grad_B_dot_grad_psi = np.array(
        dot(cross(data["B"], data["grad(|B|)"]), data["grad(psi)"])
    )
    B_cross_grad_B_dot_grad_alpha = np.array(
        dot(cross(data["B"], data["grad(|B|)"]), grad_alpha_shifted)
    )
    B_cross_kappa_dot_grad_psi = np.array(
        dot(cross(data["B"], data["kappa"]), data["grad(psi)"])
    )
    B_cross_kappa_dot_grad_alpha = np.array(
        dot(cross(data["B"], data["kappa"]), grad_alpha_shifted)
    )

    fl = Struct()
    variables = [
        "s",
        "rho",
        "alpha",
        "nl",
        "theta_pest",
        "theta_desc",
        "theta_vmec",
        "zeta",
        "phi",
        "phi_center",
        "edge_toroidal_flux_over_2pi",
        "Rmajor_p",
        "iota",
        "shat",
        "B_reference",
        "L_reference",
        "toroidal_flux_sign",
        "sqrt_s",
        "modB",
        "bmag",
        "gradpar_theta_pest",
        "length",
        "d_pressure_d_s",
        "grad_psi_dot_grad_psi",
        "grad_alpha_dot_grad_psi",
        "grad_alpha_dot_grad_alpha",
        "grho",
        "gds2",
        "gds22",
        "B_cross_grad_B_dot_grad_psi",
        "B_cross_grad_B_dot_grad_alpha",
        "B_cross_kappa_dot_grad_psi",
        "B_cross_kappa_dot_grad_alpha",
        "nfp",
    ]
    for v in variables:
        val = eval(v)
        # print("  ", v, "has type", type(val))
        if not isinstance(val, (int, float, np.ndarray)):
            raise RuntimeError(f"Variable {v} may still have a jax type: {type(val)}")
        fl.__setattr__(v, val)
    return fl


def desc_fieldline_from_center(eq, s, theta0, zeta0, poloidal_turns, nl):
    """Create a field line similar to desc_fieldline, but taking the arguments
    in a different form.
    """
    theta1d = np.linspace(
        theta0 - np.pi * poloidal_turns, theta0 + np.pi * poloidal_turns, nl
    )
    alpha = theta0
    fl = desc_fieldline(eq, s, alpha, theta1d=theta1d, zeta0=zeta0)
    fl.theta0 = theta0
    fl.poloidal_turns = poloidal_turns
    return fl


def _length_residual(
    poloidal_turns, target_length_over_L_ref, eq, s, theta0, zeta0, nl, verbose
):
    fl = desc_fieldline_from_center(eq, s, theta0, zeta0, poloidal_turns, nl)
    if verbose:
        print("poloidal_turns:", poloidal_turns, "length:", fl.length)
    return fl.length - target_length_over_L_ref


def desc_fieldline_specified_length(eq, s, theta0, zeta0, nl, length, verbose=True):
    """Solve for the number of poloidal turns to match a specified length.

    The length argument is (physical length) / L_reference.
    """
    soln = root_scalar(
        _length_residual,
        args=(length, eq, s, theta0, zeta0, nl, verbose),
        bracket=[0.05, 10],
    )
    poloidal_turns = soln.root
    fl = desc_fieldline_from_center(eq, s, theta0, zeta0, poloidal_turns, nl)
    return fl
