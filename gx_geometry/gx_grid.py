import copy
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
from .util import mu_0

__all__ = ["uniform_arclength", "add_gx_definitions", "resample"]


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
    inv_gradpar1 = 0.5 * (
        1 / fl1.gradpar_theta_pest[1:] + 1 / fl1.gradpar_theta_pest[:-1]
    )
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
    fl2.domain_scaling_factor = -(fl1.theta_pest[0]-fl1.theta_pest[-1])/(2*np.pi)
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

    print(f"Final (unscaled) theta grid goes from [{fl1.theta_pest[0]}, {fl1.theta_pest[-1]}]")
    print(f"domain_scaling_factor = {fl2.domain_scaling_factor} so that scaled theta grid is [-pi, pi]")


    return fl2


def add_gx_definitions(fl, sigma_Bxy):
    r"""
    Add the quantities gds21, gbdrift, gbdrift0, cvdrift, and cvdrift0 to a
    field line structure

    Args:
        fl: A structure with data on a field line.
        sigma_Bxy: (1 / |B|^2) \vec{B} \cdot \nabla x \times \nabla y
    """
    fl.sigma_Bxy = sigma_Bxy
    fl.kxfac = sigma_Bxy

    fl.gds21 = sigma_Bxy * fl.grad_alpha_dot_grad_psi * fl.shat / fl.B_reference

    fl.gbdrift = (
        2
        * sigma_Bxy
        * fl.toroidal_flux_sign
        * fl.B_reference
        * fl.L_reference
        * fl.L_reference
        * fl.sqrt_s
        * fl.B_cross_grad_B_dot_grad_alpha
        / (fl.modB**3)
    )

    fl.cvdrift = (
        fl.gbdrift
        + 2
        * mu_0
        * sigma_Bxy
        * fl.toroidal_flux_sign
        * fl.B_reference
        * fl.L_reference
        * fl.L_reference
        * fl.sqrt_s
        * fl.d_pressure_d_s
        / (fl.edge_toroidal_flux_over_2pi * fl.modB * fl.modB)
    )

    fl.gbdrift0 = (
        2
        * fl.toroidal_flux_sign
        * fl.shat
        * fl.B_cross_grad_B_dot_grad_psi
        / (fl.modB * fl.modB * fl.modB * fl.sqrt_s)
    )

    fl.cvdrift0 = fl.gbdrift0

    fl.twist_shift_geo_fac = 2.*fl.shat*fl.gds21/fl.gds22
    #fl.jtwist = (twist_shift_geo_fac)/y0*x0

    return fl

def cut_field_line(fl, **params):
    geometry_params = params.get('Geometry', {})
    domain_params = params.get('Domain', {})

    boundary = domain_params.get("boundary", "linked")
    y0 = domain_params.get("y0", 10.0)
    x0 = domain_params.get("x0", y0)
    
    jtwist_in = domain_params.get("jtwist", None)
    jtwist_max = domain_params.get("jtwist_max", None)

    if boundary == "exact periodic":
        flux_tube_cut = "gds21"
    elif boundary == "continuous drifts":
        flux_tube_cut = "gbdrift0"
    elif boundary == "fix aspect":
        flux_tube_cut = "aspect"
    else:
        flux_tube_cut = "none"

    npol_min = geometry_params.get("npol_min", None)
    default = -1
    if npol_min is not None:
        default = 0
    which_crossing = geometry_params.get("which_crossing", default)

    if flux_tube_cut == "gds21":
        print("***************************************************************************")
        print("You have chosen to cut the flux tube to enforce exact periodicity (gds21=0)")
        print("***************************************************************************")
    
        from scipy.interpolate import splrep, PPoly
        tck = splrep(fl.theta_pest, fl.gds21, s=0)
        ppoly = PPoly.from_spline(tck)
        gds21_roots = ppoly.roots(extrapolate=False)
    
        if npol_min is not None:
            gds21_roots = gds21_roots[gds21_roots > npol_min*np.pi + fl.theta0]
    
        # determine theta cut
        cut = gds21_roots[which_crossing]
    elif flux_tube_cut == "gbdrift0":
        print("***************************************************************************************")
        print("You have chosen to cut the flux tube to enforce continuous magnetic drifts (gbdrift0=0)")
        print("***************************************************************************************")
        
        from scipy.interpolate import splrep, PPoly
        tck = splrep(fl.theta_pest, fl.gbdrift0, s=0)
        ppoly = PPoly.from_spline(tck)
        gbdrift0_roots = ppoly.roots(extrapolate=False)
        
        if npol_min is not None:
            gbdrift0_roots = gbdrift0_roots[gbdrift0_roots > npol_min*np.pi + fl.theta0]
        
        # determine theta cut
        cut = gbdrift0_roots[which_crossing]
    if flux_tube_cut == "aspect":
        print("*************************************************************************")
        print("You have chosen to cut the flux tube to enforce y0/x0 = ", y0/x0)
        print("*************************************************************************")
    
        jtwist = fl.twist_shift_geo_fac/y0*x0
        jtwist_spl = CubicSpline(fl.theta_pest, jtwist)
    
        # find locations where jtwist_spl is integer valued. we'll check jtwist = [-30, 30] unless jtwist_max is set
        if jtwist_in is not None:
            vals = np.array([-jtwist_in, jtwist_in])
        elif jtwist_max is not None:
            vals =  np.arange(-jtwist_max,jtwist_max)
        else:
            vals =  np.arange(-30,30)
        vals = vals[(vals < -0.1) | (vals > 0.1)] # omit jtwist = 0
        crossings = [jtwist_spl.solve(i, extrapolate=False) for i in vals]
        crossings = np.concatenate(crossings)
        crossings.sort()
    
        if npol_min is not None:
            crossings = crossings[crossings > npol_min*np.pi + fl.theta0]

            if len(crossings) == 0:
                vals =  np.arange(-30,30)
                vals = vals[(vals < -0.1) | (vals > 0.1)] # omit jtwist = 0
                crossings = [jtwist_spl.solve(i, extrapolate=False) for i in vals]
                crossings = np.concatenate(crossings)
                crossings.sort()
                crossings = crossings[crossings > npol_min*np.pi + fl.theta0]
                print(f"Warning: could not satisfy jtwist constraint with npol_min = {npol_min}. Final jtwist = {abs(int(jtwist_spl(crossings[which_crossing])))}.")

        # determine theta cut
        cut = crossings[which_crossing]

    fl2 = copy.deepcopy(fl)

    if flux_tube_cut != "none":
        # new truncated theta array
        theta_cut = np.linspace(fl.theta0 - (cut - fl.theta0), cut, fl.nl)

        fl2.theta_pest = theta_cut
        for varname in dir(fl):
            var = fl.__getattribute__(varname)
            if isinstance(var, np.ndarray) and varname != "theta_pest":
                interpolator = interp1d(
                    fl.theta_pest,
                    var,
                    kind="cubic",
                    fill_value="extrapolate",
                )
                fl2.__getattribute__(varname)[:] = interpolator(theta_cut)

    return fl2

def resample(fl, nz):
    """Resample to a different number of z grid points."""
    fl2 = copy.deepcopy(fl)
    z_new = np.linspace(-np.pi, np.pi, nz)
    for varname in dir(fl):
        var = fl.__getattribute__(varname)
        if isinstance(var, np.ndarray) and varname != "z":
            interpolator = interp1d(
                fl.z,
                var,
                kind="cubic",
                fill_value="extrapolate",
            )
            fl2.__setattr__(varname, interpolator(z_new))

    fl2.z = z_new
    fl2.nl = nz
    fl2.nphi = nz
    # It seems the interpolation can introduce machine-precision-scale variation in gradpar which causes gx to reject the geometry file. The next line fixes this:
    fl2.gradpar[:] = np.mean(fl2.gradpar)

    return fl2
