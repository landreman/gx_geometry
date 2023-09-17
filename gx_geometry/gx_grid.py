import copy
import numpy as np
from scipy.interpolate import interp1d
from .util import mu_0

__all__ = ["uniform_arclength", "add_gx_definitions"]

def uniform_arclength(fl1):
    """
    Given data along fieldlines, interpolate the data onto a parallel coordinate
    in which the differential arclength is uniform.

    The new parallel coordinate runs from -pi to pi.

    Args:
        fl1: Input field line structure, as computed by ``vmec_fieldline``
    """
    fl1.z = np.zeros_like(fl1.modB)
    fl2 = copy.deepcopy(fl1)
    fl2.gradpar = np.zeros_like(fl1.modB)
    fl2.arclength = np.zeros_like(fl1.modB)
    # We will over-write the functions of z, but the above command is a
    # convenint way to initialize arrays of the correct shape.


    """
    # Differentiation matrix for a uniform grid:
    D = differentiation_matrix(fl1.nl, fl1.theta_pest[0, 0, -1] - fl1.theta_pest[0, 0, 0])
    print("condition number:", np.linalg.cond(D))
    # Add a constraint that z=0 at first grid point:
    #D[0, 0] = D[0, 0] + 1
    D = D + 1
    if True:    
        np.set_printoptions(linewidth=400)
        print("pre D:")
        print(D)
        #print("inv(D):")
        #print(np.linalg.inv(D))
    #exit(0)

    LU, pivot = lu_factor(D)

    # Save this info in the return structure:
    fl2.D = D
    """

    # Interpolate to "half grid":
    inv_gradpar1 = 0.5 * (1 / fl1.gradpar_theta_pest[1:] + 1 / fl1.gradpar_theta_pest[:-1])
    # Compute physical arclength along flux tube:
    dz1 = fl1.theta_pest[1] - fl1.theta_pest[0]
    fl2.arclength = dz1 * np.concatenate(([0], np.cumsum(inv_gradpar1)))
    L = fl2.arclength[-1]
    gradpar2 = 2 * np.pi / L
    fl2.gradpar = np.full(fl2.nl, gradpar2)
    """
    rhs = gradpar2 / fl1.gradpar_theta_pest
    print(" pre rhs:", rhs)
    #fl1.z = lu_solve((LU, pivot), rhs) - np.pi
    fl1.z = np.linalg.solve(D, rhs) - np.pi
    print("soln:", np.linalg.solve(D, rhs))
    print(" pre fl1.z:", fl1.z)
    np.testing.assert_allclose(fl2.D @ (fl1.z + np.pi),
    rhs)
    """
    fl1.z = fl2.arclength * (2 * np.pi / L) - np.pi
    uniform_z_grid = np.linspace(-np.pi, np.pi, fl1.nl)
    fl2.z = uniform_z_grid
    for varname in dir(fl1):
        var = fl1.__getattribute__(varname)
        if isinstance(var, np.ndarray) and varname != "z":
            interpolator = interp1d(
                fl1.z, 
                var,
                kind="cubic",
                fill_value="extrapolate",
            )
            fl2.__getattribute__(varname)[:] = interpolator(uniform_z_grid)                    

    return fl2

def add_gx_definitions(fl, kxfac):
    """
    Add the quantities gds21, gbdrift, gbdrift0, cvdrift, and cvdrift0 to a
    field line structure

    Args:
        fl: A structure with data on a field line.
    """
    fl.kxfac = kxfac

    fl.gds21 = kxfac * fl.grad_alpha_dot_grad_psi * fl.shat / fl.B_reference

    fl.gbdrift = (
        2 * kxfac * fl.toroidal_flux_sign * fl.B_reference * fl.L_reference * fl.L_reference 
        * fl.sqrt_s * fl.B_cross_grad_B_dot_grad_alpha 
        / (fl.modB ** 3)
    )

    fl.cvdrift = (
        fl.gbdrift 
        + 2 * mu_0 * kxfac * fl.toroidal_flux_sign * fl.B_reference * fl.L_reference * fl.L_reference 
        * fl.sqrt_s * fl.d_pressure_d_s
        / (fl.edge_toroidal_flux_over_2pi * fl.modB * fl.modB)
    )

    fl.gbdrift0 = (
        2 * fl.toroidal_flux_sign * fl.shat * fl.B_cross_grad_B_dot_grad_psi
        / (fl.modB * fl.modB * fl.modB * fl.sqrt_s)
    )

    fl.cvdrift0 = fl.gbdrift0

    return fl
