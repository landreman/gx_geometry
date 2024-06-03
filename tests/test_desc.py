#!/usr/bin/env python

import unittest
import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from desc.vmec import VMECIO
import desc.io
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile

from gx_geometry import (
    desc_fieldline,
    Vmec,
    vmec_fieldline,
    uniform_arclength,
    add_gx_definitions,
)

from . import TEST_DIR

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


class Tests(unittest.TestCase):
    def test_tokamak(self):
        """
        Compare to analytic formulas for a tokamak from section 5.7 of
        20230917-01 Signs in GX geometry.lyx
        """
        aminor = 2.0
        aspect = 100.0
        Rmajor = aspect * aminor
        surface = FourierRZToroidalSurface(
            R_lmn=[Rmajor, aminor],
            modes_R=[[0, 0], [1, 0]],  # modes given as [m, n] for each coefficient
            Z_lmn=[0.0, -aminor],
            modes_Z=[[0, 0], [-1, 0]],
        )
        # Try all 4 possible directions of the magnetic field:
        for sign_psi in [1, -1]:
            for sign_iota in [1, -1]:
                iota_coeffs = sign_iota * np.array([0.9, -0.65])
                iota_profile = PowerSeriesProfile(params=iota_coeffs, modes=[0, 2])
                iota_edge = sum(iota_coeffs)
                abs_Psi = np.pi * aminor**2 * 5  # So |B| ~ 5 T.
                eq = Equilibrium(
                    surface=surface,
                    iota=iota_profile,
                    Psi=sign_psi * abs_Psi,
                    NFP=1,
                    L=6,
                    M=6,
                    N=0,
                    L_grid=12,
                    M_grid=12,
                    N_grid=0,
                    sym=True,
                )
                eq.solve()
                # eq.save("axisymm.h5")
                # eq = desc.io.load("axisymm.h5")

                theta = np.linspace(-3 * np.pi, 3 * np.pi, 200)
                fl = desc_fieldline(eq, s=1, alpha=0, theta1d=theta)
                B0 = abs_Psi / (np.pi * aminor**2)
                eps = 1 / aspect
                safety_factor_q = 1 / iota_edge
                phi = theta / iota_edge
                d_iota_d_s = iota_coeffs[1]  # From differentiating c0 + c1 * s
                d_iota_d_r = d_iota_d_s * 2 / aminor
                # print('sign of psi in grad psi cross grad theta + iota grad phi cross grad psi:', fl.toroidal_flux_sign)

                # See Matt Landreman's note "20220315-02 Geometry arrays for gyrokinetics in a circular tokamak.docx"
                # for the analytic results below
                np.testing.assert_allclose(
                    fl.modB, B0 * (1 - eps * np.cos(theta)), rtol=0.0002
                )
                # np.testing.assert_allclose(fl.gradpar_theta_pest, -fl.toroidal_flux_sign * B0 / (safety_factor_q * Rmajor), rtol=0.0006)
                np.testing.assert_allclose(
                    fl.B_cross_grad_B_dot_grad_psi,
                    -(B0**3) * eps * np.sin(theta),
                    rtol=0.03,
                    atol=1e-10,
                )
                np.testing.assert_allclose(
                    fl.B_cross_kappa_dot_grad_psi,
                    -(B0**2) * eps * np.sin(theta),
                    rtol=0.02,
                    atol=1e-10,
                )
                np.testing.assert_allclose(
                    fl.grad_psi_dot_grad_psi, B0 * B0 * aminor * aminor, rtol=0.03
                )
                np.testing.assert_allclose(
                    fl.grad_alpha_dot_grad_psi,
                    -fl.toroidal_flux_sign * phi * d_iota_d_r * aminor * B0,
                    rtol=0.02,
                )
                np.testing.assert_allclose(
                    fl.grad_alpha_dot_grad_alpha,
                    1 / (aminor * aminor) + (phi * phi * d_iota_d_r * d_iota_d_r),
                    rtol=0.04,
                )
                np.testing.assert_allclose(
                    fl.B_cross_grad_B_dot_grad_alpha,
                    fl.toroidal_flux_sign
                    * (B0 * B0 / aminor)
                    * (
                        -np.cos(theta) / Rmajor + phi * d_iota_d_r * eps * np.sin(theta)
                    ),
                    atol=0.015,
                )
                np.testing.assert_allclose(
                    fl.B_cross_kappa_dot_grad_alpha,
                    fl.toroidal_flux_sign
                    * (B0 / aminor)
                    * (
                        -np.cos(theta) / Rmajor + phi * d_iota_d_r * eps * np.sin(theta)
                    ),
                    atol=0.0007,
                )

    def test_wout_as_desc(self):
        """
        Make sure vmec and desc geometry calculations agree for a field line
        with alpha=0 and an equilibrium with beta=0.
        """
        filename_base = "wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc"
        filename = os.path.join(TEST_DIR, filename_base)
        eq = VMECIO.load(filename)

        s = 0.5
        alpha = 0.0
        ntheta = 49
        theta1d = np.linspace(-np.pi, np.pi, ntheta)
        fl1 = desc_fieldline(eq, s, alpha, theta1d=theta1d)
        vmec = Vmec(filename)
        fl2 = vmec_fieldline(vmec, s, alpha, theta1d=theta1d)

        plot_all_quantities = False
        if plot_all_quantities:
            plt.figure(figsize=(14, 7.5))
            nrows = 3
            ncols = 4
            jplot = 1

            def compare_vmec_desc(field, signflip=1):
                nonlocal jplot
                plt.subplot(nrows, ncols, jplot)
                jplot = jplot + 1
                plt.plot(fl1.theta_pest, fl1.__getattribute__(field), label="desc")
                plt.plot(
                    fl2.theta_pest,
                    np.flip(fl2.__getattribute__(field)) * signflip,
                    label="vmec",
                )
                plt.xlabel("theta_pest")
                plt.title(field)

            compare_vmec_desc("modB", signflip=1)
            compare_vmec_desc("theta_vmec", signflip=1)
            compare_vmec_desc("phi", signflip=-1)
            compare_vmec_desc("gradpar_theta_pest", signflip=-1)
            compare_vmec_desc("grad_psi_dot_grad_psi", signflip=1)
            compare_vmec_desc("grad_alpha_dot_grad_psi", signflip=-1)
            compare_vmec_desc("grad_alpha_dot_grad_alpha", signflip=1)
            compare_vmec_desc("B_cross_grad_B_dot_grad_psi", signflip=1)
            compare_vmec_desc("B_cross_kappa_dot_grad_psi", signflip=1)
            compare_vmec_desc("B_cross_grad_B_dot_grad_alpha", signflip=-1)
            compare_vmec_desc("B_cross_kappa_dot_grad_alpha", signflip=-1)
            plt.tight_layout()
            plt.legend(loc=0, fontsize=7)
            plt.figtext(
                0.5,
                0.995,
                "Before uniform arclength",
                ha="center",
                va="top",
                fontsize=9,
            )
            # plt.show()

        np.set_printoptions(linewidth=400)

        def show_diffs(field1, field2):
            print("abs diff:", np.abs(field1 - field2))
            print("rel diff:", np.abs(field1 - field2) / (0.5 * (field1 + field2)))
            plt.figure()
            plt.semilogy(np.abs(field1 - field2), label="abs")
            plt.plot(np.abs((field1 - field2) / (0.5 * (field1 + field2))), label="rel")
            plt.legend(loc=0, fontsize=6)
            plt.show()

        # Jacobian of (rho, theta, phi) is negative for vmec, positive for desc,
        # as theta_desc = -theta_vmec. Therefore iota flips sign. To understand
        # all the signs below, see section 6 of
        # 20230910-01 Signs in GX geometry.lyx

        # Scalars:
        np.testing.assert_allclose(fl1.iota, -fl2.iota)
        np.testing.assert_allclose(fl1.shat, fl2.shat, rtol=0.0002)
        np.testing.assert_allclose(fl1.L_reference, fl2.L_reference)
        np.testing.assert_allclose(fl1.B_reference, fl2.B_reference)

        # Quantities that vary along a field line:
        np.testing.assert_allclose(fl1.theta_pest, -np.flip(fl2.theta_pest))
        np.testing.assert_allclose(fl1.theta_desc, -np.flip(fl2.theta_vmec), rtol=0.006)
        np.testing.assert_allclose(fl1.zeta, np.flip(fl2.phi))
        np.testing.assert_allclose(fl1.modB, np.flip(fl2.modB), rtol=0.008)
        # show_diffs(fl1.gradpar_theta_pest, np.flip(fl2.gradpar_theta_pest))
        np.testing.assert_allclose(
            fl1.gradpar_theta_pest, -np.flip(fl2.gradpar_theta_pest), rtol=0.001
        )
        np.testing.assert_allclose(fl1.grho, np.flip(fl2.grho), rtol=0.01)
        np.testing.assert_allclose(
            fl1.grad_psi_dot_grad_psi, np.flip(fl2.grad_psi_dot_grad_psi), rtol=0.01
        )
        np.testing.assert_allclose(
            fl1.grad_alpha_dot_grad_psi, np.flip(fl2.grad_alpha_dot_grad_psi), atol=0.2
        )
        np.testing.assert_allclose(
            fl1.grad_alpha_dot_grad_alpha,
            np.flip(fl2.grad_alpha_dot_grad_alpha),
            atol=0.06,
        )
        np.testing.assert_allclose(
            fl1.B_cross_grad_B_dot_grad_psi,
            -np.flip(fl2.B_cross_grad_B_dot_grad_psi),
            rtol=0.1,
            atol=1.3,
        )
        np.testing.assert_allclose(
            fl1.B_cross_kappa_dot_grad_psi,
            -np.flip(fl2.B_cross_kappa_dot_grad_psi),
            atol=0.04,
        )
        np.testing.assert_allclose(
            fl1.B_cross_grad_B_dot_grad_alpha,
            -np.flip(fl2.B_cross_grad_B_dot_grad_alpha),
            atol=0.75,
        )
        np.testing.assert_allclose(
            fl1.B_cross_kappa_dot_grad_alpha,
            -np.flip(fl2.B_cross_kappa_dot_grad_alpha),
            atol=0.006,
        )

        # Now convert to gx grid and quantities:
        for sigma_Bxy in [-1, 1]:
            fl3 = uniform_arclength(fl1)
            fl4 = uniform_arclength(fl2)
            add_gx_definitions(fl3, sigma_Bxy)
            add_gx_definitions(fl4, sigma_Bxy)

            if plot_all_quantities:
                plt.figure(figsize=(14, 7.5))
                nrows = 3
                ncols = 4
                jplot = 1

                def compare_vmec_desc(field, signflip=1):
                    nonlocal jplot
                    plt.subplot(nrows, ncols, jplot)
                    jplot = jplot + 1
                    # plt.plot(fl3.theta_pest, fl3.__getattribute__(field), label="desc")
                    # plt.plot(fl4.theta_pest, np.flip(fl4.__getattribute__(field)) * signflip, label="vmec")
                    plt.plot(fl3.__getattribute__(field), label="desc")
                    plt.plot(
                        np.flip(fl4.__getattribute__(field)) * signflip, label="vmec"
                    )
                    plt.xlabel("theta_pest")
                    plt.title(field)

                compare_vmec_desc("modB", signflip=1)
                compare_vmec_desc("theta_vmec", signflip=-1)
                compare_vmec_desc("phi", signflip=1)
                compare_vmec_desc("gradpar_theta_pest", signflip=-1)
                compare_vmec_desc("grad_psi_dot_grad_psi", signflip=1)
                compare_vmec_desc("grad_alpha_dot_grad_psi", signflip=1)
                compare_vmec_desc("grad_alpha_dot_grad_alpha", signflip=1)
                compare_vmec_desc("B_cross_grad_B_dot_grad_psi", signflip=-1)
                compare_vmec_desc("B_cross_kappa_dot_grad_psi", signflip=-1)
                compare_vmec_desc("B_cross_grad_B_dot_grad_alpha", signflip=-1)
                compare_vmec_desc("B_cross_kappa_dot_grad_alpha", signflip=-1)
                plt.tight_layout()
                plt.legend(loc=0, fontsize=7)
                plt.figtext(
                    0.5,
                    0.995,
                    "After uniform arclength",
                    ha="center",
                    va="top",
                    fontsize=9,
                )
                # plt.show()

            # Scalars:
            np.testing.assert_allclose(fl3.iota, -fl4.iota)
            np.testing.assert_allclose(fl3.shat, fl4.shat, rtol=0.0002)
            np.testing.assert_allclose(fl3.L_reference, fl4.L_reference)
            np.testing.assert_allclose(fl3.B_reference, fl4.B_reference)

            # Quantities that vary along a field line:
            # show_diffs(fl3.theta_pest, -np.flip(fl4.theta_pest))
            np.testing.assert_allclose(
                fl3.theta_pest, -np.flip(fl4.theta_pest), atol=0.0002
            )
            np.testing.assert_allclose(
                fl3.theta_desc, -np.flip(fl4.theta_vmec), rtol=0.006, atol=0.003
            )
            np.testing.assert_allclose(fl3.zeta, np.flip(fl4.phi), atol=0.0002)
            np.testing.assert_allclose(fl3.modB, np.flip(fl4.modB), rtol=0.008)
            np.testing.assert_allclose(
                fl3.gradpar_theta_pest, -np.flip(fl4.gradpar_theta_pest), rtol=0.001
            )
            np.testing.assert_allclose(fl3.grho, np.flip(fl4.grho), rtol=0.01)
            np.testing.assert_allclose(
                fl3.grad_psi_dot_grad_psi, np.flip(fl4.grad_psi_dot_grad_psi), rtol=0.01
            )
            np.testing.assert_allclose(
                fl3.grad_alpha_dot_grad_psi,
                np.flip(fl4.grad_alpha_dot_grad_psi),
                atol=0.2,
            )
            np.testing.assert_allclose(
                fl3.grad_alpha_dot_grad_alpha,
                np.flip(fl4.grad_alpha_dot_grad_alpha),
                atol=0.06,
            )
            np.testing.assert_allclose(
                fl3.B_cross_grad_B_dot_grad_psi,
                -np.flip(fl4.B_cross_grad_B_dot_grad_psi),
                rtol=0.1,
                atol=1.3,
            )
            np.testing.assert_allclose(
                fl3.B_cross_kappa_dot_grad_psi,
                -np.flip(fl4.B_cross_kappa_dot_grad_psi),
                atol=0.04,
            )
            # show_diffs(fl3.B_cross_grad_B_dot_grad_alpha, -np.flip(fl4.B_cross_grad_B_dot_grad_alpha))
            np.testing.assert_allclose(
                fl3.B_cross_grad_B_dot_grad_alpha,
                -np.flip(fl4.B_cross_grad_B_dot_grad_alpha),
                atol=0.82,
            )
            np.testing.assert_allclose(
                fl3.B_cross_kappa_dot_grad_alpha,
                -np.flip(fl4.B_cross_kappa_dot_grad_alpha),
                atol=0.006,
            )

            # Now test GX quantities
            if False:
                plt.figure(figsize=(14, 7.5))
                nrows = 3
                ncols = 3
                jplot = 1

                def compare_vmec_desc(field, signflip=1):
                    nonlocal jplot
                    plt.subplot(nrows, ncols, jplot)
                    jplot = jplot + 1
                    plt.plot(fl3.__getattribute__(field), label="desc")
                    plt.plot(
                        np.flip(fl4.__getattribute__(field)) * signflip, label="vmec"
                    )
                    plt.title(field)

                compare_vmec_desc("z", signflip=-1)
                compare_vmec_desc("gradpar", signflip=-1)
                compare_vmec_desc("gds2", signflip=1)
                compare_vmec_desc("gds21", signflip=1)
                compare_vmec_desc("gds22", signflip=1)
                compare_vmec_desc("gbdrift", signflip=1)
                compare_vmec_desc("cvdrift", signflip=1)
                compare_vmec_desc("gbdrift0", signflip=1)
                compare_vmec_desc("cvdrift0", signflip=1)
                plt.tight_layout()
                plt.legend(loc=0, fontsize=7)
                plt.show()

            np.testing.assert_allclose(fl3.z, -np.flip(fl4.z))
            np.testing.assert_allclose(fl3.gradpar, -np.flip(fl4.gradpar), atol=1e-5)
            np.testing.assert_allclose(fl3.gds2, np.flip(fl4.gds2), rtol=0.04)
            np.testing.assert_allclose(fl3.gds22, np.flip(fl4.gds22), rtol=0.01)
            np.testing.assert_allclose(fl3.gds21, np.flip(fl4.gds21), atol=1e-4)
            np.testing.assert_allclose(fl3.gbdrift, np.flip(fl4.gbdrift), atol=0.07)
            np.testing.assert_allclose(fl3.cvdrift, np.flip(fl4.cvdrift), atol=0.07)
            np.testing.assert_allclose(fl3.gbdrift0, np.flip(fl4.gbdrift0), atol=7e-5)
            np.testing.assert_allclose(fl3.cvdrift0, np.flip(fl4.cvdrift0), atol=7e-5)

    def test_vmec_desc_benchmark_nonzero_alpha(self):
        """
        This test covers lots of items: a non-symmetric flux tube (nonzero
        alpha), nonzero beta, and >1 poloidal turn.
        """
        filename_base = "wout_w7x_from_gx_repository.nc"
        filename = os.path.join(TEST_DIR, filename_base)
        vmec = Vmec(filename)

        # For this configuration, VMECIO.load-ing the wout file was not accurate
        # enough - I needed to call eq.solve(), after lowering the resolution.
        filename_base = "w7x_from_gx_repository_LMN8.h5"
        filename = os.path.join(TEST_DIR, filename_base)
        eq = desc.io.load(filename)

        s = 0.9
        alpha_vmec = 1.5
        alpha_desc = -alpha_vmec
        ntheta = 201
        theta1d = np.linspace(-2 * np.pi, 2 * np.pi, ntheta)
        fl1 = desc_fieldline(eq, s, alpha_desc, theta1d=theta1d)
        fl2 = vmec_fieldline(vmec, s, alpha_vmec, theta1d=theta1d)

        plot_all_quantities = False
        if plot_all_quantities:
            plt.figure(figsize=(14, 7.5))
            nrows = 3
            ncols = 4
            jplot = 1

            def compare_vmec_desc(field, signflip=1):
                nonlocal jplot
                plt.subplot(nrows, ncols, jplot)
                jplot = jplot + 1
                # plt.plot(fl1.theta_pest, fl1.__getattribute__(field), label="desc")
                # plt.plot(fl2.theta_pest, np.flip(fl2.__getattribute__(field)) * signflip, label="vmec")
                plt.plot(fl1.__getattribute__(field), label="desc")
                plt.plot(np.flip(fl2.__getattribute__(field)) * signflip, label="vmec")
                # plt.xlabel("theta_pest")
                plt.title(field)

            compare_vmec_desc("modB", signflip=1)
            compare_vmec_desc("theta_vmec", signflip=-1)
            compare_vmec_desc("phi", signflip=1)
            compare_vmec_desc("gradpar_theta_pest", signflip=-1)
            compare_vmec_desc("grad_psi_dot_grad_psi", signflip=1)
            compare_vmec_desc("grad_alpha_dot_grad_psi", signflip=1)
            compare_vmec_desc("grad_alpha_dot_grad_alpha", signflip=1)
            compare_vmec_desc("B_cross_grad_B_dot_grad_psi", signflip=-1)
            compare_vmec_desc("B_cross_kappa_dot_grad_psi", signflip=-1)
            compare_vmec_desc("B_cross_grad_B_dot_grad_alpha", signflip=-1)
            compare_vmec_desc("B_cross_kappa_dot_grad_alpha", signflip=-1)
            plt.tight_layout()
            plt.legend(loc=0, fontsize=7)
            plt.figtext(
                0.5,
                0.995,
                "Before uniform arclength",
                ha="center",
                va="top",
                fontsize=9,
            )
            plt.show()

        np.set_printoptions(linewidth=400)

        def show_diffs(field1, field2):
            print("abs diff:", np.abs(field1 - field2))
            print("rel diff:", np.abs(field1 - field2) / (0.5 * (field1 + field2)))
            plt.figure()
            plt.semilogy(np.abs(field1 - field2), label="abs")
            plt.plot(np.abs((field1 - field2) / (0.5 * (field1 + field2))), label="rel")
            plt.legend(loc=0, fontsize=6)
            plt.show()

        # To understand all the signs below, see section 6 of
        # 20230910-01 Signs in GX geometry.lyx

        # Scalars:
        np.testing.assert_allclose(fl1.iota, -fl2.iota, rtol=1e-6)
        np.testing.assert_allclose(fl1.shat, fl2.shat, rtol=0.0002)
        np.testing.assert_allclose(fl1.L_reference, fl2.L_reference)
        np.testing.assert_allclose(fl1.B_reference, fl2.B_reference)

        # Quantities that vary along a field line:
        np.testing.assert_allclose(fl1.theta_pest, -np.flip(fl2.theta_pest), atol=1e-13)
        np.testing.assert_allclose(fl1.theta_desc, -np.flip(fl2.theta_vmec), rtol=0.06)
        np.testing.assert_allclose(fl1.zeta, np.flip(fl2.phi), atol=6e-6)
        np.testing.assert_allclose(fl1.modB, np.flip(fl2.modB), atol=0.003)
        # show_diffs(fl1.gradpar_theta_pest, np.flip(fl2.gradpar_theta_pest))
        np.testing.assert_allclose(
            fl1.gradpar_theta_pest, -np.flip(fl2.gradpar_theta_pest), rtol=0.001
        )
        np.testing.assert_allclose(fl1.grho, np.flip(fl2.grho), atol=0.08)
        np.testing.assert_allclose(
            fl1.grad_psi_dot_grad_psi, np.flip(fl2.grad_psi_dot_grad_psi), rtol=0.07
        )
        np.testing.assert_allclose(
            fl1.grad_alpha_dot_grad_psi, np.flip(fl2.grad_alpha_dot_grad_psi), atol=1.2
        )
        np.testing.assert_allclose(
            fl1.grad_alpha_dot_grad_alpha,
            np.flip(fl2.grad_alpha_dot_grad_alpha),
            atol=1.96,
        )
        np.testing.assert_allclose(
            fl1.B_cross_grad_B_dot_grad_psi,
            -np.flip(fl2.B_cross_grad_B_dot_grad_psi),
            atol=0.12,
        )
        np.testing.assert_allclose(
            fl1.B_cross_kappa_dot_grad_psi,
            -np.flip(fl2.B_cross_kappa_dot_grad_psi),
            atol=0.038,
        )
        np.testing.assert_allclose(
            fl1.B_cross_grad_B_dot_grad_alpha,
            -np.flip(fl2.B_cross_grad_B_dot_grad_alpha),
            atol=0.51,
        )
        np.testing.assert_allclose(
            fl1.B_cross_kappa_dot_grad_alpha,
            -np.flip(fl2.B_cross_kappa_dot_grad_alpha),
            atol=0.18,
        )

        # Now convert to gx grid and quantities:
        for sigma_Bxy in [-1, 1]:
            fl3 = uniform_arclength(fl1)
            fl4 = uniform_arclength(fl2)
            add_gx_definitions(fl3, sigma_Bxy)
            add_gx_definitions(fl4, sigma_Bxy)

            if False:
                plt.figure(figsize=(14, 7.5))
                nrows = 3
                ncols = 3
                jplot = 1

                def compare_vmec_desc(field, signflip=1):
                    nonlocal jplot
                    plt.subplot(nrows, ncols, jplot)
                    jplot = jplot + 1
                    plt.plot(fl3.__getattribute__(field), label="desc")
                    plt.plot(
                        np.flip(fl4.__getattribute__(field)) * signflip, label="vmec"
                    )
                    plt.title(field)

                compare_vmec_desc("z", signflip=-1)
                compare_vmec_desc("gradpar", signflip=-1)
                compare_vmec_desc("gds2", signflip=1)
                compare_vmec_desc("gds21", signflip=1)
                compare_vmec_desc("gds22", signflip=1)
                compare_vmec_desc("gbdrift", signflip=1)
                compare_vmec_desc("cvdrift", signflip=1)
                compare_vmec_desc("gbdrift0", signflip=1)
                compare_vmec_desc("cvdrift0", signflip=1)
                plt.tight_layout()
                plt.legend(loc=0, fontsize=7)
                plt.show()

            # GX quantities
            np.testing.assert_allclose(fl3.z, -np.flip(fl4.z), atol=1e-13)
            np.testing.assert_allclose(fl3.gradpar, -np.flip(fl4.gradpar), atol=2e-6)
            np.testing.assert_allclose(fl3.gds2, np.flip(fl4.gds2), atol=0.43)
            np.testing.assert_allclose(fl3.gds22, np.flip(fl4.gds22), atol=0.01)
            np.testing.assert_allclose(fl3.gds21, np.flip(fl4.gds21), atol=0.08)
            np.testing.assert_allclose(fl3.gbdrift, np.flip(fl4.gbdrift), atol=0.028)
            np.testing.assert_allclose(fl3.cvdrift, np.flip(fl4.cvdrift), atol=0.028)
            np.testing.assert_allclose(fl3.gbdrift0, np.flip(fl4.gbdrift0), atol=0.0017)
            np.testing.assert_allclose(fl3.cvdrift0, np.flip(fl4.cvdrift0), atol=0.0017)

    def test_vmec_desc_benchmark_nonzero_zeta0(self):
        """
        This test covers lots of items: a non-symmetric flux tube (nonzero
        alpha), nonzero beta, >1 poloidal turn, nonzero theta0, and nonzero zeta0.
        """
        filename_base = "wout_w7x_from_gx_repository.nc"
        filename = os.path.join(TEST_DIR, filename_base)
        vmec = Vmec(filename)

        # For this configuration, VMECIO.load-ing the wout file was not accurate
        # enough - I needed to call eq.solve(), after lowering the resolution.
        filename_base = "w7x_from_gx_repository_LMN8.h5"
        filename = os.path.join(TEST_DIR, filename_base)
        eq = desc.io.load(filename)

        s = 0.9
        theta0_vmec = np.pi / 3
        zeta0 = 0.8 * np.pi / 5
        theta0_desc = -theta0_vmec
        alpha_vmec = theta0_vmec
        alpha_desc = -alpha_vmec
        ntheta = 201
        poloidal_turns = 1.4
        theta1d_vmec = np.linspace(
            theta0_vmec - poloidal_turns * np.pi,
            theta0_vmec + poloidal_turns * np.pi,
            ntheta,
        )
        theta1d_desc = np.linspace(
            theta0_desc - poloidal_turns * np.pi,
            theta0_desc + poloidal_turns * np.pi,
            ntheta,
        )
        fl1 = desc_fieldline(eq, s, alpha_desc, theta1d=theta1d_desc, zeta0=zeta0)
        fl2 = vmec_fieldline(
            vmec, s, alpha_vmec, theta1d=theta1d_vmec, phi_center=zeta0
        )

        plot_all_quantities = False
        if plot_all_quantities:
            plt.figure(figsize=(14, 7.5))
            nrows = 3
            ncols = 4
            jplot = 1

            def compare_vmec_desc(field, signflip=1):
                nonlocal jplot
                plt.subplot(nrows, ncols, jplot)
                jplot = jplot + 1
                # plt.plot(fl1.theta_pest, fl1.__getattribute__(field), label="desc")
                # plt.plot(fl2.theta_pest, np.flip(fl2.__getattribute__(field)) * signflip, label="vmec")
                plt.plot(fl1.__getattribute__(field), label="desc")
                plt.plot(np.flip(fl2.__getattribute__(field)) * signflip, label="vmec")
                # plt.xlabel("theta_pest")
                plt.title(field)

            compare_vmec_desc("modB", signflip=1)
            compare_vmec_desc("theta_vmec", signflip=-1)
            compare_vmec_desc("phi", signflip=1)
            compare_vmec_desc("gradpar_theta_pest", signflip=-1)
            compare_vmec_desc("grad_psi_dot_grad_psi", signflip=1)
            compare_vmec_desc("grad_alpha_dot_grad_psi", signflip=1)
            compare_vmec_desc("grad_alpha_dot_grad_alpha", signflip=1)
            compare_vmec_desc("B_cross_grad_B_dot_grad_psi", signflip=-1)
            compare_vmec_desc("B_cross_kappa_dot_grad_psi", signflip=-1)
            compare_vmec_desc("B_cross_grad_B_dot_grad_alpha", signflip=-1)
            compare_vmec_desc("B_cross_kappa_dot_grad_alpha", signflip=-1)
            plt.tight_layout()
            plt.legend(loc=0, fontsize=7)
            plt.figtext(
                0.5,
                0.995,
                "Before uniform arclength",
                ha="center",
                va="top",
                fontsize=9,
            )
            plt.show()

        np.set_printoptions(linewidth=400)

        def show_diffs(field1, field2):
            print("abs diff:", np.abs(field1 - field2))
            print("rel diff:", np.abs(field1 - field2) / (0.5 * (field1 + field2)))
            plt.figure()
            plt.semilogy(np.abs(field1 - field2), label="abs")
            plt.plot(np.abs((field1 - field2) / (0.5 * (field1 + field2))), label="rel")
            plt.legend(loc=0, fontsize=6)
            plt.show()

        # To understand all the signs below, see section 6 of
        # 20230910-01 Signs in GX geometry.lyx

        # Scalars:
        np.testing.assert_allclose(fl1.iota, -fl2.iota, rtol=1e-6)
        np.testing.assert_allclose(fl1.shat, fl2.shat, rtol=0.0002)
        np.testing.assert_allclose(fl1.L_reference, fl2.L_reference)
        np.testing.assert_allclose(fl1.B_reference, fl2.B_reference)

        # Quantities that vary along a field line:
        np.testing.assert_allclose(fl1.theta_pest, -np.flip(fl2.theta_pest), atol=1e-13)
        np.testing.assert_allclose(fl1.theta_desc, -np.flip(fl2.theta_vmec), rtol=0.06)
        np.testing.assert_allclose(fl1.zeta, np.flip(fl2.phi), atol=6e-6)
        np.testing.assert_allclose(fl1.modB, np.flip(fl2.modB), atol=0.003)
        # show_diffs(fl1.gradpar_theta_pest, np.flip(fl2.gradpar_theta_pest))
        np.testing.assert_allclose(
            fl1.gradpar_theta_pest, -np.flip(fl2.gradpar_theta_pest), rtol=0.001
        )
        np.testing.assert_allclose(fl1.grho, np.flip(fl2.grho), atol=0.08)
        np.testing.assert_allclose(
            fl1.grad_psi_dot_grad_psi, np.flip(fl2.grad_psi_dot_grad_psi), rtol=0.07
        )
        np.testing.assert_allclose(
            fl1.grad_alpha_dot_grad_psi, np.flip(fl2.grad_alpha_dot_grad_psi), atol=1.2
        )
        np.testing.assert_allclose(
            fl1.grad_alpha_dot_grad_alpha,
            np.flip(fl2.grad_alpha_dot_grad_alpha),
            atol=1.7,
        )
        np.testing.assert_allclose(
            fl1.B_cross_grad_B_dot_grad_psi,
            -np.flip(fl2.B_cross_grad_B_dot_grad_psi),
            atol=0.12,
        )
        np.testing.assert_allclose(
            fl1.B_cross_kappa_dot_grad_psi,
            -np.flip(fl2.B_cross_kappa_dot_grad_psi),
            atol=0.038,
        )
        np.testing.assert_allclose(
            fl1.B_cross_grad_B_dot_grad_alpha,
            -np.flip(fl2.B_cross_grad_B_dot_grad_alpha),
            atol=0.51,
        )
        np.testing.assert_allclose(
            fl1.B_cross_kappa_dot_grad_alpha,
            -np.flip(fl2.B_cross_kappa_dot_grad_alpha),
            atol=0.18,
        )

        # Now convert to gx grid and quantities:
        for sigma_Bxy in [-1, 1]:
            fl3 = uniform_arclength(fl1)
            fl4 = uniform_arclength(fl2)
            add_gx_definitions(fl3, sigma_Bxy)
            add_gx_definitions(fl4, sigma_Bxy)

            if False:
                plt.figure(figsize=(14, 7.5))
                nrows = 3
                ncols = 3
                jplot = 1

                def compare_vmec_desc(field, signflip=1):
                    nonlocal jplot
                    plt.subplot(nrows, ncols, jplot)
                    jplot = jplot + 1
                    plt.plot(fl3.__getattribute__(field), label="desc")
                    plt.plot(
                        np.flip(fl4.__getattribute__(field)) * signflip, label="vmec"
                    )
                    plt.title(field)

                compare_vmec_desc("z", signflip=-1)
                compare_vmec_desc("gradpar", signflip=-1)
                compare_vmec_desc("gds2", signflip=1)
                compare_vmec_desc("gds21", signflip=1)
                compare_vmec_desc("gds22", signflip=1)
                compare_vmec_desc("gbdrift", signflip=1)
                compare_vmec_desc("cvdrift", signflip=1)
                compare_vmec_desc("gbdrift0", signflip=1)
                compare_vmec_desc("cvdrift0", signflip=1)
                plt.tight_layout()
                plt.legend(loc=0, fontsize=7)
                plt.show()

            # GX quantities
            np.testing.assert_allclose(fl3.z, -np.flip(fl4.z), atol=1e-13)
            np.testing.assert_allclose(fl3.gradpar, -np.flip(fl4.gradpar), atol=4e-6)
            np.testing.assert_allclose(fl3.gds2, np.flip(fl4.gds2), atol=0.35)
            np.testing.assert_allclose(fl3.gds22, np.flip(fl4.gds22), atol=0.01)
            np.testing.assert_allclose(fl3.gds21, np.flip(fl4.gds21), atol=0.08)
            np.testing.assert_allclose(fl3.gbdrift, np.flip(fl4.gbdrift), atol=0.028)
            np.testing.assert_allclose(fl3.cvdrift, np.flip(fl4.cvdrift), atol=0.028)
            np.testing.assert_allclose(fl3.gbdrift0, np.flip(fl4.gbdrift0), atol=0.0017)
            np.testing.assert_allclose(fl3.cvdrift0, np.flip(fl4.cvdrift0), atol=0.0017)


if __name__ == "__main__":
    unittest.main()
