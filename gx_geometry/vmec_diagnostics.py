# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module contains functions that can postprocess VMEC output.
"""

import logging
#from typing import Union

import numpy as np
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.optimize import newton

from .vmec import Vmec
from .util import Struct

logger = logging.getLogger(__name__)

__all__ = ['vmec_splines', 'vmec_compute_geometry', 'vmec_fieldline']


def vmec_splines(vmec):
    """
    Initialize radial splines for a VMEC equilibrium.

    Args:
        vmec: An instance of :obj:`simsopt.mhd.vmec.Vmec`.

    Returns:
        A structure with the splines as attributes.
    """
    vmec.run()
    results = Struct()
    if vmec.wout.lasym:
        raise ValueError("vmec_splines is not yet set up for non-stellarator-symmetric cases.")

    rmnc = []
    zmns = []
    lmns = []
    d_rmnc_d_s = []
    d_zmns_d_s = []
    d_lmns_d_s = []
    for jmn in range(vmec.wout.mnmax):
        rmnc.append(InterpolatedUnivariateSpline(vmec.s_full_grid, vmec.wout.rmnc[jmn, :]))
        zmns.append(InterpolatedUnivariateSpline(vmec.s_full_grid, vmec.wout.zmns[jmn, :]))
        lmns.append(InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.lmns[jmn, 1:]))
        d_rmnc_d_s.append(rmnc[-1].derivative())
        d_zmns_d_s.append(zmns[-1].derivative())
        d_lmns_d_s.append(lmns[-1].derivative())

    gmnc = []
    bmnc = []
    bsupumnc = []
    bsupvmnc = []
    bsubsmns = []
    bsubumnc = []
    bsubvmnc = []
    d_bmnc_d_s = []
    d_bsupumnc_d_s = []
    d_bsupvmnc_d_s = []
    for jmn in range(vmec.wout.mnmax_nyq):
        gmnc.append(InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.gmnc[jmn, 1:]))
        bmnc.append(InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.bmnc[jmn, 1:]))
        bsupumnc.append(InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.bsupumnc[jmn, 1:]))
        bsupvmnc.append(InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.bsupvmnc[jmn, 1:]))
        # Note that bsubsmns is on the full mesh, unlike the other components:
        bsubsmns.append(InterpolatedUnivariateSpline(vmec.s_full_grid, vmec.wout.bsubsmns[jmn, :]))
        bsubumnc.append(InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.bsubumnc[jmn, 1:]))
        bsubvmnc.append(InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.bsubvmnc[jmn, 1:]))
        d_bmnc_d_s.append(bmnc[-1].derivative())
        d_bsupumnc_d_s.append(bsupumnc[-1].derivative())
        d_bsupvmnc_d_s.append(bsupvmnc[-1].derivative())

    # Handle 1d profiles:
    results.pressure = InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.pres[1:])
    results.d_pressure_d_s = results.pressure.derivative()
    results.iota = InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.iotas[1:])
    results.d_iota_d_s = results.iota.derivative()

    # Save other useful quantities:
    results.phiedge = vmec.wout.phi[-1]
    variables = ['Aminor_p', 'Rmajor_p', 'mnmax', 'xm', 'xn', 'mnmax_nyq', 'xm_nyq', 'xn_nyq', 'nfp']
    for v in variables:
        results.__setattr__(v, eval('vmec.wout.' + v))

    variables = ['rmnc', 'zmns', 'lmns', 'd_rmnc_d_s', 'd_zmns_d_s', 'd_lmns_d_s',
                 'gmnc', 'bmnc', 'd_bmnc_d_s', 'bsupumnc', 'bsupvmnc', 'd_bsupumnc_d_s', 'd_bsupvmnc_d_s',
                 'bsubsmns', 'bsubumnc', 'bsubvmnc']
    for v in variables:
        results.__setattr__(v, eval(v))

    return results


def vmec_compute_geometry(vs, s, theta, phi, phi_center=0):
    r"""
    Compute many geometric quantities of interest from a vmec configuration.

    Some of the quantities computed by this function refer to
    ``alpha``, a field line label coordinate defined by

    .. math::

        \alpha = \theta_{pest} - \iota (\phi - \phi_{center}).

    Here, :math:`\phi_{center}` is a constant, usually 0, which can be
    set to a nonzero value if desired so the magnetic shear
    contribution to :math:`\nabla\alpha` vanishes at a toroidal angle
    different than 0.  Also, wherever the term ``psi`` appears in
    variable names in this function and the returned arrays, it means
    :math:`\psi =` the toroidal flux divided by :math:`2\pi`, so

    .. math::

        \vec{B} = \nabla\psi\times\nabla\theta_{pest} + \iota\nabla\phi\times\nabla\psi = \nabla\psi\times\nabla\alpha.

    Most of the arrays that are returned by this function have shape
    ``(ns, ntheta, nphi)``, where ``ns`` is the number of flux
    surfaces, ``ntheta`` is the number of grid points in VMEC's
    poloidal angle, and ``nphi`` is the number of grid points in the
    standard toroidal angle. For the arguments ``theta`` and ``phi``,
    you can either provide 1D arrays, in which case a tensor product
    grid is used, or you can provide 3D arrays of shape ``(ns, ntheta,
    nphi)``. In this latter case, the grids are not necessarily
    tensor-product grids.  Note that all angles in this function have
    period :math:`2\pi`, not period 1.

    The output arrays are returned as attributes of the
    returned object. Many intermediate quantities are included, such
    as the Cartesian components of the covariant and contravariant
    basis vectors. Some of the most useful of these output arrays are (all with SI units):

    - ``phi``: The standard toroidal angle :math:`\phi`.
    - ``theta_vmec``: VMEC's poloidal angle :math:`\theta_{vmec}`.
    - ``theta_pest``: The straight-field-line angle :math:`\theta_{pest}` associated with :math:`\phi`.
    - ``modB``: The magnetic field magnitude :math:`|B|`.
    - ``B_sup_theta_vmec``: :math:`\vec{B}\cdot\nabla\theta_{vmec}`.
    - ``B_sup_phi``: :math:`\vec{B}\cdot\nabla\phi`.
    - ``B_cross_grad_B_dot_grad_alpha``: :math:`\vec{B}\times\nabla|B|\cdot\nabla\alpha`.
    - ``B_cross_grad_B_dot_grad_psi``: :math:`\vec{B}\times\nabla|B|\cdot\nabla\psi`.
    - ``B_cross_kappa_dot_grad_alpha``: :math:`\vec{B}\times\vec{\kappa}\cdot\nabla\alpha`,
      where :math:`\vec{\kappa}=\vec{b}\cdot\nabla\vec{b}` is the curvature and :math:`\vec{b}=|B|^{-1}\vec{B}`.
    - ``B_cross_kappa_dot_grad_psi``: :math:`\vec{B}\times\vec{\kappa}\cdot\nabla\psi`.
    - ``grad_alpha_dot_grad_alpha``: :math:`|\nabla\alpha|^2 = \nabla\alpha\cdot\nabla\alpha`.
    - ``grad_alpha_dot_grad_psi``: :math:`\nabla\alpha\cdot\nabla\psi`.
    - ``grad_psi_dot_grad_psi``: :math:`|\nabla\psi|^2 = \nabla\psi\cdot\nabla\psi`.
    - ``iota``: The rotational transform :math:`\iota`. This array has shape ``(ns,)``.
    - ``shat``: The magnetic shear :math:`\hat s= (x/q) (d q / d x)` where 
      :math:`x = \mathrm{Aminor_p} \, \sqrt{s}` and :math:`q=1/\iota`. This array has shape ``(ns,)``.

    The following normalized versions of these quantities used in the
    gyrokinetic codes ``stella``, ``gs2``, and ``GX`` are also
    returned: ``bmag``, ``gbdrift``, ``gbdrift0``, ``cvdrift``,
    ``cvdrift0``, ``gds2``, ``gds21``, and ``gds22``, along with
    ``L_reference`` and ``B_reference``.  Instead of ``gradpar``, two
    variants are returned, ``gradpar_theta_pest`` and ``gradpar_phi``,
    corresponding to choosing either :math:`\theta_{pest}` or
    :math:`\phi` as the parallel coordinate.

    The value(s) of ``s`` provided as input need not coincide with the
    full grid or half grid in VMEC, as spline interpolation will be
    used radially.

    The implementation in this routine is similar to the one in the
    gyrokinetic code ``stella``.

    Example usage::

        import numpy as np
        from simsopt.mhd import Vmec, vmec_compute_geometry

        v = Vmec("wout_li383_1.4m.nc")
        s = 1
        theta = np.linspace(0, 2 * np.pi, 50)
        phi = np.linspace(0, 2 * np.pi / 3, 60)
        data = vmec_compute_geometry(v, s, theta, phi)
        print(data.grad_s_dot_grad_s)

    Args:
        vs: Either an instance of :obj:`simsopt.mhd.vmec.Vmec`
          or the structure returned by :func:`vmec_splines`.
        s: Values of normalized toroidal flux on which to construct the field lines.
          You can give a single number, or a list or numpy array.
        theta: Values of vmec's poloidal angle. You can provide a float, a 1d array of size
          ``(ntheta,)``, or a 3d array of size ``(ns, ntheta, nphi)``.
        phi: Values of the standard toroidal angle. You can provide a float, a 1d array of size
          ``(nphi,)`` or a 3d array of size ``(ns, ntheta, nphi)``.
        phi_center: :math:`\phi_{center}`, an optional shift to the toroidal angle
          in the definition of :math:`\alpha`.
    """
    # If given a Vmec object, convert it to vmec_splines:
    if isinstance(vs, Vmec):
        vs = vmec_splines(vs)

    # Make sure s is an array:
    try:
        ns = len(s)
    except:
        s = [s]
    s = np.array(s)
    ns = len(s)

    # Handle theta
    try:
        ntheta = len(theta)
    except:
        theta = [theta]
    theta_vmec = np.array(theta)
    if theta_vmec.ndim == 1:
        ntheta = len(theta_vmec)
    elif theta_vmec.ndim == 3:
        ntheta = theta_vmec.shape[1]
    else:
        raise ValueError("theta argument must be a float, 1d array, or 3d array.")

    # Handle phi
    try:
        nphi = len(phi)
    except:
        phi = [phi]
    phi = np.array(phi)
    if phi.ndim == 1:
        nphi = len(phi)
    elif phi.ndim == 3:
        nphi = phi.shape[2]
    else:
        raise ValueError("phi argument must be a float, 1d array, or 3d array.")

    # If theta and phi are not already 3D, make them 3D:
    if theta_vmec.ndim == 1:
        theta_vmec = np.kron(np.ones((ns, 1, nphi)), theta_vmec.reshape(1, ntheta, 1))
    if phi.ndim == 1:
        phi = np.kron(np.ones((ns, ntheta, 1)), phi.reshape(1, 1, nphi))

    # Shorthand:
    mnmax = vs.mnmax
    xm = vs.xm
    xn = vs.xn
    mnmax_nyq = vs.mnmax_nyq
    xm_nyq = vs.xm_nyq
    xn_nyq = vs.xn_nyq

    # Now that we have an s grid, evaluate everything on that grid:
    d_pressure_d_s = vs.d_pressure_d_s(s)
    iota = vs.iota(s)
    d_iota_d_s = vs.d_iota_d_s(s)
    # shat = (r/q)(dq/dr) where r = a sqrt(s)
    #      = - (r/iota) (d iota / d r) = -2 (s/iota) (d iota / d s)
    shat = (-2 * s / iota) * d_iota_d_s

    rmnc = np.zeros((ns, mnmax))
    zmns = np.zeros((ns, mnmax))
    lmns = np.zeros((ns, mnmax))
    d_rmnc_d_s = np.zeros((ns, mnmax))
    d_zmns_d_s = np.zeros((ns, mnmax))
    d_lmns_d_s = np.zeros((ns, mnmax))
    for jmn in range(mnmax):
        rmnc[:, jmn] = vs.rmnc[jmn](s)
        zmns[:, jmn] = vs.zmns[jmn](s)
        lmns[:, jmn] = vs.lmns[jmn](s)
        d_rmnc_d_s[:, jmn] = vs.d_rmnc_d_s[jmn](s)
        d_zmns_d_s[:, jmn] = vs.d_zmns_d_s[jmn](s)
        d_lmns_d_s[:, jmn] = vs.d_lmns_d_s[jmn](s)

    gmnc = np.zeros((ns, mnmax_nyq))
    bmnc = np.zeros((ns, mnmax_nyq))
    d_bmnc_d_s = np.zeros((ns, mnmax_nyq))
    bsupumnc = np.zeros((ns, mnmax_nyq))
    bsupvmnc = np.zeros((ns, mnmax_nyq))
    bsubsmns = np.zeros((ns, mnmax_nyq))
    bsubumnc = np.zeros((ns, mnmax_nyq))
    bsubvmnc = np.zeros((ns, mnmax_nyq))
    for jmn in range(mnmax_nyq):
        gmnc[:, jmn] = vs.gmnc[jmn](s)
        bmnc[:, jmn] = vs.bmnc[jmn](s)
        d_bmnc_d_s[:, jmn] = vs.d_bmnc_d_s[jmn](s)
        bsupumnc[:, jmn] = vs.bsupumnc[jmn](s)
        bsupvmnc[:, jmn] = vs.bsupvmnc[jmn](s)
        bsubsmns[:, jmn] = vs.bsubsmns[jmn](s)
        bsubumnc[:, jmn] = vs.bsubumnc[jmn](s)
        bsubvmnc[:, jmn] = vs.bsubvmnc[jmn](s)

    # Now that we know theta_vmec, compute all the geometric quantities
    angle = xm[:, None, None, None] * theta_vmec[None, :, :, :] - xn[:, None, None, None] * phi[None, :, :, :]
    cosangle = np.cos(angle)
    sinangle = np.sin(angle)
    mcosangle = xm[:, None, None, None] * cosangle
    ncosangle = xn[:, None, None, None] * cosangle
    msinangle = xm[:, None, None, None] * sinangle
    nsinangle = xn[:, None, None, None] * sinangle
    # Order of indices in cosangle and sinangle: mn, s, theta, phi
    # Order of indices in rmnc, bmnc, etc: s, mn
    R = np.einsum('ij,jikl->ikl', rmnc, cosangle)
    d_R_d_s = np.einsum('ij,jikl->ikl', d_rmnc_d_s, cosangle)
    d_R_d_theta_vmec = -np.einsum('ij,jikl->ikl', rmnc, msinangle)
    d_R_d_phi = np.einsum('ij,jikl->ikl', rmnc, nsinangle)

    Z = np.einsum('ij,jikl->ikl', zmns, sinangle)
    d_Z_d_s = np.einsum('ij,jikl->ikl', d_zmns_d_s, sinangle)
    d_Z_d_theta_vmec = np.einsum('ij,jikl->ikl', zmns, mcosangle)
    d_Z_d_phi = -np.einsum('ij,jikl->ikl', zmns, ncosangle)

    lambd = np.einsum('ij,jikl->ikl', lmns, sinangle)
    d_lambda_d_s = np.einsum('ij,jikl->ikl', d_lmns_d_s, sinangle)
    d_lambda_d_theta_vmec = np.einsum('ij,jikl->ikl', lmns, mcosangle)
    d_lambda_d_phi = -np.einsum('ij,jikl->ikl', lmns, ncosangle)
    theta_pest = theta_vmec + lambd

    # Now handle the Nyquist quantities:
    angle = xm_nyq[:, None, None, None] * theta_vmec[None, :, :, :] - xn_nyq[:, None, None, None] * phi[None, :, :, :]
    cosangle = np.cos(angle)
    sinangle = np.sin(angle)
    mcosangle = xm_nyq[:, None, None, None] * cosangle
    ncosangle = xn_nyq[:, None, None, None] * cosangle
    msinangle = xm_nyq[:, None, None, None] * sinangle
    nsinangle = xn_nyq[:, None, None, None] * sinangle

    sqrt_g_vmec = np.einsum('ij,jikl->ikl', gmnc, cosangle)
    modB = np.einsum('ij,jikl->ikl', bmnc, cosangle)
    d_B_d_s = np.einsum('ij,jikl->ikl', d_bmnc_d_s, cosangle)
    d_B_d_theta_vmec = -np.einsum('ij,jikl->ikl', bmnc, msinangle)
    d_B_d_phi = np.einsum('ij,jikl->ikl', bmnc, nsinangle)

    B_sup_theta_vmec = np.einsum('ij,jikl->ikl', bsupumnc, cosangle)
    B_sup_phi = np.einsum('ij,jikl->ikl', bsupvmnc, cosangle)
    B_sub_s = np.einsum('ij,jikl->ikl', bsubsmns, sinangle)
    B_sub_theta_vmec = np.einsum('ij,jikl->ikl', bsubumnc, cosangle)
    B_sub_phi = np.einsum('ij,jikl->ikl', bsubvmnc, cosangle)
    B_sup_theta_pest = iota[:, None, None] * B_sup_phi

    sqrt_g_vmec_alt = R * (d_Z_d_s * d_R_d_theta_vmec - d_R_d_s * d_Z_d_theta_vmec)

    # Note the minus sign. psi in the straight-field-line relation seems to have opposite sign to vmec's phi array.
    edge_toroidal_flux_over_2pi = -vs.phiedge / (2 * np.pi)

    # *********************************************************************
    # Using R(theta,phi) and Z(theta,phi), compute the Cartesian
    # components of the gradient basis vectors using the dual relations:
    # *********************************************************************
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)
    X = R * cosphi
    d_X_d_theta_vmec = d_R_d_theta_vmec * cosphi
    d_X_d_phi = d_R_d_phi * cosphi - R * sinphi
    d_X_d_s = d_R_d_s * cosphi
    Y = R * sinphi
    d_Y_d_theta_vmec = d_R_d_theta_vmec * sinphi
    d_Y_d_phi = d_R_d_phi * sinphi + R * cosphi
    d_Y_d_s = d_R_d_s * sinphi

    # Now use the dual relations to get the Cartesian components of grad s, grad theta_vmec, and grad phi:
    grad_s_X = (d_Y_d_theta_vmec * d_Z_d_phi - d_Z_d_theta_vmec * d_Y_d_phi) / sqrt_g_vmec
    grad_s_Y = (d_Z_d_theta_vmec * d_X_d_phi - d_X_d_theta_vmec * d_Z_d_phi) / sqrt_g_vmec
    grad_s_Z = (d_X_d_theta_vmec * d_Y_d_phi - d_Y_d_theta_vmec * d_X_d_phi) / sqrt_g_vmec

    grad_theta_vmec_X = (d_Y_d_phi * d_Z_d_s - d_Z_d_phi * d_Y_d_s) / sqrt_g_vmec
    grad_theta_vmec_Y = (d_Z_d_phi * d_X_d_s - d_X_d_phi * d_Z_d_s) / sqrt_g_vmec
    grad_theta_vmec_Z = (d_X_d_phi * d_Y_d_s - d_Y_d_phi * d_X_d_s) / sqrt_g_vmec

    grad_phi_X = (d_Y_d_s * d_Z_d_theta_vmec - d_Z_d_s * d_Y_d_theta_vmec) / sqrt_g_vmec
    grad_phi_Y = (d_Z_d_s * d_X_d_theta_vmec - d_X_d_s * d_Z_d_theta_vmec) / sqrt_g_vmec
    grad_phi_Z = (d_X_d_s * d_Y_d_theta_vmec - d_Y_d_s * d_X_d_theta_vmec) / sqrt_g_vmec
    # End of dual relations.

    # *********************************************************************
    # Compute the Cartesian components of other quantities we need:
    # *********************************************************************

    grad_psi_X = grad_s_X * edge_toroidal_flux_over_2pi
    grad_psi_Y = grad_s_Y * edge_toroidal_flux_over_2pi
    grad_psi_Z = grad_s_Z * edge_toroidal_flux_over_2pi

    # Form grad alpha = grad (theta_vmec + lambda - iota * phi)
    grad_alpha_X = (d_lambda_d_s - (phi - phi_center) * d_iota_d_s[:, None, None]) * grad_s_X
    grad_alpha_Y = (d_lambda_d_s - (phi - phi_center) * d_iota_d_s[:, None, None]) * grad_s_Y
    grad_alpha_Z = (d_lambda_d_s - (phi - phi_center) * d_iota_d_s[:, None, None]) * grad_s_Z

    grad_alpha_X += (1 + d_lambda_d_theta_vmec) * grad_theta_vmec_X + (-iota[:, None, None] + d_lambda_d_phi) * grad_phi_X
    grad_alpha_Y += (1 + d_lambda_d_theta_vmec) * grad_theta_vmec_Y + (-iota[:, None, None] + d_lambda_d_phi) * grad_phi_Y
    grad_alpha_Z += (1 + d_lambda_d_theta_vmec) * grad_theta_vmec_Z + (-iota[:, None, None] + d_lambda_d_phi) * grad_phi_Z

    grad_B_X = d_B_d_s * grad_s_X + d_B_d_theta_vmec * grad_theta_vmec_X + d_B_d_phi * grad_phi_X
    grad_B_Y = d_B_d_s * grad_s_Y + d_B_d_theta_vmec * grad_theta_vmec_Y + d_B_d_phi * grad_phi_Y
    grad_B_Z = d_B_d_s * grad_s_Z + d_B_d_theta_vmec * grad_theta_vmec_Z + d_B_d_phi * grad_phi_Z

    B_X = edge_toroidal_flux_over_2pi * ((1 + d_lambda_d_theta_vmec) * d_X_d_phi + (iota[:, None, None] - d_lambda_d_phi) * d_X_d_theta_vmec) / sqrt_g_vmec
    B_Y = edge_toroidal_flux_over_2pi * ((1 + d_lambda_d_theta_vmec) * d_Y_d_phi + (iota[:, None, None] - d_lambda_d_phi) * d_Y_d_theta_vmec) / sqrt_g_vmec
    B_Z = edge_toroidal_flux_over_2pi * ((1 + d_lambda_d_theta_vmec) * d_Z_d_phi + (iota[:, None, None] - d_lambda_d_phi) * d_Z_d_theta_vmec) / sqrt_g_vmec

    # *********************************************************************
    # For gbdrift, we need \vect{B} cross grad |B| dot grad alpha.
    # For cvdrift, we also need \vect{B} cross grad s dot grad alpha.
    # Let us compute both of these quantities 2 ways, and make sure the two
    # approaches give the same answer (within some tolerance).
    # *********************************************************************

    B_cross_grad_s_dot_grad_alpha = (B_sub_phi * (1 + d_lambda_d_theta_vmec) \
                                     - B_sub_theta_vmec * (d_lambda_d_phi - iota[:, None, None])) / sqrt_g_vmec

    B_cross_grad_s_dot_grad_alpha_alternate = 0 \
        + B_X * grad_s_Y * grad_alpha_Z \
        + B_Y * grad_s_Z * grad_alpha_X \
        + B_Z * grad_s_X * grad_alpha_Y \
        - B_Z * grad_s_Y * grad_alpha_X \
        - B_X * grad_s_Z * grad_alpha_Y \
        - B_Y * grad_s_X * grad_alpha_Z

    B_cross_grad_B_dot_grad_alpha = 0 \
        + (B_sub_s * d_B_d_theta_vmec * (d_lambda_d_phi - iota[:, None, None]) \
           + B_sub_theta_vmec * d_B_d_phi * (d_lambda_d_s - (phi - phi_center) * d_iota_d_s[:, None, None]) \
           + B_sub_phi * d_B_d_s * (1 + d_lambda_d_theta_vmec) \
           - B_sub_phi * d_B_d_theta_vmec * (d_lambda_d_s - (phi - phi_center) * d_iota_d_s[:, None, None]) \
           - B_sub_theta_vmec * d_B_d_s * (d_lambda_d_phi - iota[:, None, None]) \
           - B_sub_s * d_B_d_phi * (1 + d_lambda_d_theta_vmec)) / sqrt_g_vmec

    B_cross_grad_B_dot_grad_alpha_alternate = 0 \
        + B_X * grad_B_Y * grad_alpha_Z \
        + B_Y * grad_B_Z * grad_alpha_X \
        + B_Z * grad_B_X * grad_alpha_Y \
        - B_Z * grad_B_Y * grad_alpha_X \
        - B_X * grad_B_Z * grad_alpha_Y \
        - B_Y * grad_B_X * grad_alpha_Z

    grad_alpha_dot_grad_alpha = grad_alpha_X * grad_alpha_X + grad_alpha_Y * grad_alpha_Y + grad_alpha_Z * grad_alpha_Z

    grad_alpha_dot_grad_psi = grad_alpha_X * grad_psi_X + grad_alpha_Y * grad_psi_Y + grad_alpha_Z * grad_psi_Z

    grad_psi_dot_grad_psi = grad_psi_X * grad_psi_X + grad_psi_Y * grad_psi_Y + grad_psi_Z * grad_psi_Z

    grad_s_dot_grad_s = grad_s_X * grad_s_X + grad_s_Y * grad_s_Y + grad_s_Z * grad_s_Z

    B_cross_grad_B_dot_grad_psi = (B_sub_theta_vmec * d_B_d_phi - B_sub_phi * d_B_d_theta_vmec) / sqrt_g_vmec * edge_toroidal_flux_over_2pi

    B_cross_kappa_dot_grad_psi = B_cross_grad_B_dot_grad_psi / modB

    mu_0 = 4 * np.pi * (1.0e-7)
    B_cross_kappa_dot_grad_alpha = B_cross_grad_B_dot_grad_alpha / modB + mu_0 * d_pressure_d_s[:, None, None] / edge_toroidal_flux_over_2pi

    # stella / gs2 / gx quantities:

    L_reference = vs.Aminor_p
    B_reference = 2 * abs(edge_toroidal_flux_over_2pi) / (L_reference * L_reference)
    Rmajor_p = vs.Rmajor_p
    toroidal_flux_sign = np.sign(edge_toroidal_flux_over_2pi)
    sqrt_s = np.sqrt(s)

    bmag = modB / B_reference

    gradpar_theta_pest = L_reference * B_sup_theta_pest / modB

    gradpar_phi = L_reference * B_sup_phi / modB

    gds2 = grad_alpha_dot_grad_alpha * L_reference * L_reference * s[:, None, None]

    gds22 = grad_psi_dot_grad_psi * shat[:, None, None] * shat[:, None, None] / (L_reference * L_reference * B_reference * B_reference * s[:, None, None])

    grho = np.sqrt(grad_psi_dot_grad_psi / (L_reference * L_reference * B_reference * B_reference * s[:, None, None]))

    # For the following quantities, the sign is controvertial, so I'll comment
    # out the lines from simsopt:
    """
    gds21 = grad_alpha_dot_grad_psi * shat[:, None, None] / B_reference

    # See issue #238 and the discussion therein
    gbdrift = -1 * 2 * B_reference * L_reference * L_reference * sqrt_s[:, None, None] * B_cross_grad_B_dot_grad_alpha / (modB * modB * modB) * toroidal_flux_sign

    gbdrift0 = B_cross_grad_B_dot_grad_psi * 2 * shat[:, None, None] / (modB * modB * modB * sqrt_s[:, None, None]) * toroidal_flux_sign

    # See issue #238 and the discussion therein
    cvdrift = gbdrift - 2 * B_reference * L_reference * L_reference * sqrt_s[:, None, None] * mu_0 * d_pressure_d_s[:, None, None] * toroidal_flux_sign / (edge_toroidal_flux_over_2pi * modB * modB)

    cvdrift0 = gbdrift0
    """


    # Package results into a structure to return:
    results = Struct()
    variables = ['ns', 'ntheta', 'nphi', 's', 'iota', 'd_iota_d_s', 'd_pressure_d_s', 'shat',
                 'theta_vmec', 'phi', 'theta_pest',
                 'd_lambda_d_s', 'd_lambda_d_theta_vmec', 'd_lambda_d_phi', 'sqrt_g_vmec', 'sqrt_g_vmec_alt',
                 'modB', 'd_B_d_s', 'd_B_d_theta_vmec', 'd_B_d_phi', 'B_sup_theta_vmec', 'B_sup_theta_pest', 'B_sup_phi',
                 'B_sub_s', 'B_sub_theta_vmec', 'B_sub_phi', 'edge_toroidal_flux_over_2pi', 'sinphi', 'cosphi',
                 'R', 'd_R_d_s', 'd_R_d_theta_vmec', 'd_R_d_phi', 'X', 'Y', 'Z', 'd_Z_d_s', 'd_Z_d_theta_vmec', 'd_Z_d_phi',
                 'd_X_d_theta_vmec', 'd_X_d_phi', 'd_X_d_s', 'd_Y_d_theta_vmec', 'd_Y_d_phi', 'd_Y_d_s',
                 'grad_s_X', 'grad_s_Y', 'grad_s_Z', 'grad_theta_vmec_X', 'grad_theta_vmec_Y', 'grad_theta_vmec_Z',
                 'grad_phi_X', 'grad_phi_Y', 'grad_phi_Z', 'grad_psi_X', 'grad_psi_Y', 'grad_psi_Z',
                 'grad_alpha_X', 'grad_alpha_Y', 'grad_alpha_Z', 'grad_B_X', 'grad_B_Y', 'grad_B_Z',
                 'B_X', 'B_Y', 'B_Z', "grad_s_dot_grad_s",
                 'B_cross_grad_s_dot_grad_alpha', 'B_cross_grad_s_dot_grad_alpha_alternate',
                 'B_cross_grad_B_dot_grad_alpha', 'B_cross_grad_B_dot_grad_alpha_alternate',
                 'B_cross_grad_B_dot_grad_psi', 'B_cross_kappa_dot_grad_psi', 'B_cross_kappa_dot_grad_alpha',
                 'grad_alpha_dot_grad_alpha', 'grad_alpha_dot_grad_psi', 'grad_psi_dot_grad_psi',
                 'L_reference', 'B_reference', 'toroidal_flux_sign',
                 'bmag', 'gradpar_theta_pest', 'gradpar_phi', 'gds2', 'gds22', 'sqrt_s', 'grho', 'Rmajor_p']
    #             'bmag', 'gradpar_theta_pest', 'gradpar_phi', 'gds2', 'gds21', 'gds22', 'gbdrift', 'gbdrift0', 'cvdrift', 'cvdrift0']
    for v in variables:
        results.__setattr__(v, eval(v))

    return results


def vmec_fieldline(vs, s, alpha, theta1d=None, phi1d=None, phi_center=0, plot=False, show=True):
    r"""
    Compute field lines in a vmec configuration, and compute many
    geometric quantities of interest along the field lines. In
    particular, this routine computes the geometric quantities that
    enter the gyrokinetic equation.

    One task performed by this function is to convert between
    the poloidal angles :math:`\theta_{vmec}` and
    :math:`\theta_{pest}`. The latter is the angle in which the field
    lines are straight when used in combination with the standard
    toroidal angle :math:`\phi`. Note that all angles in this function
    have period :math:`2\pi`, not period 1.

    To specify the parallel extent of the field lines, you can provide
    either a grid of :math:`\theta_{pest}` values or a grid of
    :math:`\phi` values. If you specify both or neither, ``ValueError``
    will be raised.

    The geometric quanties computed by this function are the same as for
    :func:`vmec_compute_geometry()`. See the documentation of that
    function for details.

    Most of the arrays that are returned by this function have shape
    ``(nl,)``, where  ``nl`` is the number of grid points along each field line. In
    other words, ``nl`` is
    the size of the input ``theta1d`` or ``phi1d`` array. The output
    arrays are returned as attributes of the returned object.

    The value of ``s`` provided as input need not coincide with the
    full grid or half grid in VMEC, as spline interpolation will be
    used radially.

    Example usage::

        import numpy as np
        from simsopt.mhd import Vmec, vmec_fieldline

        v = Vmec('wout_li383_1.4m.nc')
        theta = np.linspace(-np.pi, np.pi, 50)
        fl = vmec_fieldline(v, 0.5, 0, theta1d=theta)
        print(fl.B_cross_grad_B_dot_grad_alpha)

    Args:
        vs: Either an instance of :obj:`simsopt.mhd.vmec.Vmec`
          or the structure returned by :func:`vmec_splines`.
        s: Value of normalized toroidal flux on which to construct the field line.
        alpha: Value of the field line label :math:`\alpha` on which to construct the field line.
        theta1d: 1D array of :math:`\theta_{pest}` values, setting the grid points
          along the field line and the parallel extent of the field line.
        phi1d: 1D array of :math:`\phi` values, setting the grid points along the
          field line and the parallel extent of the field line.
        phi_center: :math:`\phi_{center}`, an optional shift to the toroidal angle
          in the definition of :math:`\alpha`.
        plot: Whether to create a plot of the main geometric quantities. Only one field line will
          be plotted, corresponding to the leading elements of ``s`` and ``alpha``.
        show: Only matters if ``plot==True``. Whether to call matplotlib's ``show()`` function
          after creating the plot.
    """
    # If given a Vmec object, convert it to vmec_splines:
    if isinstance(vs, Vmec):
        vs = vmec_splines(vs)

    # Make sure s is an array:
    s = [s]
    s = np.array(s)
    ns = len(s)

    # Make sure alpha is an array
    alpha = [alpha]
    alpha = np.array(alpha)
    nalpha = len(alpha)

    if (theta1d is not None) and (phi1d is not None):
        raise ValueError('You cannot specify both theta and phi')
    if (theta1d is None) and (phi1d is None):
        raise ValueError('You must specify either theta or phi')
    if theta1d is None:
        nl = len(phi1d)
    else:
        nl = len(theta1d)

    # Shorthand:
    mnmax = vs.mnmax
    xm = vs.xm
    xn = vs.xn
    mnmax_nyq = vs.mnmax_nyq
    xm_nyq = vs.xm_nyq
    xn_nyq = vs.xn_nyq

    # Now that we have an s grid, evaluate everything on that grid:
    iota = vs.iota(s)
    lmns = np.zeros((ns, mnmax))
    for jmn in range(mnmax):
        lmns[:, jmn] = vs.lmns[jmn](s)

    theta_pest = np.zeros((ns, nalpha, nl))
    phi = np.zeros((ns, nalpha, nl))

    if theta1d is None:
        # We are given phi. Compute theta_pest:
        for js in range(ns):
            phi[js, :, :] = phi1d[None, :]
            theta_pest[js, :, :] = alpha[:, None] + iota[js] * (phi1d[None, :] - phi_center)
    else:
        # We are given theta_pest. Compute phi:
        for js in range(ns):
            theta_pest[js, :, :] = theta1d[None, :]
            phi[js, :, :] = phi_center + (theta1d[None, :] - alpha[:, None]) / iota[js]

    def residual(theta_v, phi0, theta_p_target, jradius):
        """
        This function is used for computing an array of values of theta_vmec that
        give a desired theta_pest array.
        """
        return theta_p_target - (theta_v + np.sum(lmns[jradius, :, None] * np.sin(xm[:, None] * theta_v - xn[:, None] * phi0), axis=0))

    theta_vmec = np.zeros((ns, nalpha, nl))
    for js in range(ns):
        for jalpha in range(nalpha):
            theta_guess = theta_pest[js, jalpha, :]
            solution = newton(
                residual,
                x0=theta_guess,
                x1=theta_guess + 0.1,
                args=(phi[js, jalpha, :], theta_pest[js, jalpha, :], js),
            )
            theta_vmec[js, jalpha, :] = solution

    # Now that we have theta_vmec, compute all the geometric quantities:
    results = vmec_compute_geometry(vs, s, theta_vmec, phi, phi_center)

    # Add a few more quantities to the results:
    variables = ["nalpha", "nl", "alpha", "theta1d", "phi1d", "phi"]
    for v in variables:
        results.__setattr__(v, eval(v))

    # Convert all 1D arrays to scalars, and 3D quantities to 1D:
    for attr in dir(results):
        if attr[0] == "_":
            continue
        var = results.__getattribute__(attr)
        if isinstance(var, np.ndarray):
            if var.ndim == 1 and attr not in ["theta1d", "phi1d"]:
                results.__setattr__(attr, var[0])
            if var.ndim == 3:
                results.__setattr__(attr, var.flatten())

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(13, 7))
        nrows = 4
        ncols = 5
        variables = ['modB', 'B_sup_theta_pest', 'B_sup_phi', 'B_cross_grad_B_dot_grad_alpha', 'B_cross_grad_B_dot_grad_psi',
                     'B_cross_kappa_dot_grad_alpha', 'B_cross_kappa_dot_grad_psi',
                     'grad_alpha_dot_grad_alpha', 'grad_alpha_dot_grad_psi', 'grad_psi_dot_grad_psi',
                     'bmag', 'gradpar_theta_pest', 'gradpar_phi', 'gds2', 'gds22']
        #             'bmag', 'gradpar_theta_pest', 'gradpar_phi', 'gbdrift', 'gbdrift0', 'cvdrift', 'cvdrift0', 'gds2', 'gds21', 'gds22']
        for j, variable in enumerate(variables):
            plt.subplot(nrows, ncols, j + 1)
            plt.plot(results.phi, eval("results." + variable))
            plt.xlabel('Standard toroidal angle $\phi$')
            plt.title(variable)

        plt.figtext(0.5, 0.995, f's={s}, alpha={alpha}', ha='center', va='top')
        plt.tight_layout()
        if show:
            plt.show()

    return results
