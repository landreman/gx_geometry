#!/usr/bin/env python

import unittest
import os
import logging

import numpy as np
import matplotlib

from gx_geometry import Vmec, vmec_splines, vmec_compute_geometry, vmec_fieldline, mu_0

from . import TEST_DIR

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)


class VmecComputeGeometryTests(unittest.TestCase):
    def test_1d_matches_3d(self):
        """
        If we call the function with 1d arrays for theta and phi,
        we should get the same results as if we call the routine with
        equivalent 3d arrays for theta and phi.
        """
        vmec = Vmec(os.path.join(TEST_DIR, 'wout_li383_low_res_reference.nc'))
        splines = vmec_splines(vmec)
        s = 0.5
        theta = [0.2, 0.7, -10.2]
        phi = [-9.9, -4.4, 0, 3.3, 7.7]
        ntheta = len(theta)
        nphi = len(phi)

        theta3d = np.zeros((1, ntheta, nphi))
        phi3d = np.zeros_like(theta3d)
        for jtheta in range(ntheta):
            theta3d[:, jtheta, :] = theta[jtheta]
        for jphi in range(nphi):
            phi3d[:, :, jphi] = phi[jphi]

        results1 = vmec_compute_geometry(splines, s, theta, phi)
        results2 = vmec_compute_geometry(vmec, np.array([s]), theta3d, phi3d)

        variables = ["theta_pest", "grad_psi_dot_grad_psi", "B_cross_kappa_dot_grad_psi"]
        for v in variables:
            np.testing.assert_allclose(eval("results1." + v), eval("results2." + v))

    def test_compare_to_desc(self):
        """
        Compare some values to an independent calculation in desc.
        """
        vmec = Vmec(os.path.join(TEST_DIR, "wout_LandremanPaul2021_QA_lowres.nc"))

        s = [0.25, 1.0]
        ntheta = 4
        nphi = 5
        theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        phi = np.linspace(0, 2 * np.pi / vmec.wout.nfp, nphi, endpoint=False)

        simsopt_data = vmec_compute_geometry(vmec, s, theta, phi)

        desc_data = np.zeros((len(s), ntheta, nphi))
        desc_data[0, :, :] = np.array([[0.0232505427, 0.0136928264, 0.0045250425, 0.0045250425, 0.0136928264],
                                       [0.001595767, 0.006868957, 0.0126580432, 0.0124698027, 0.0069361438],
                                       [0.0418344846, 0.0234485798, 0.0058187257, 0.0058187257, 0.0234485798],
                                       [0.001595767, 0.0069361438, 0.0124698027, 0.0126580432, 0.006868957]])
        desc_data[1, :, :] = np.array([[0.0682776505, 0.0419440941, 0.0159952307, 0.0159952307, 0.0419440941],
                                       [0.006650641, 0.0301276863, 0.0552814479, 0.0525678846, 0.0244553647],
                                       [0.2151059496, 0.1238328858, 0.0297237057, 0.0297237057, 0.1238328858],
                                       [0.006650641, 0.0244553647, 0.0525678846, 0.0552814479, 0.0301276863]])

        np.testing.assert_allclose(simsopt_data.grad_psi_dot_grad_psi, desc_data, rtol=0.005, atol=0.0005)


class VmecFieldlinesTests(unittest.TestCase):
    def test_fieldline_grids(self):
        """
        Check the grids in theta and phi created by vmec_fieldline().
        """
        vmec = Vmec(os.path.join(TEST_DIR, 'wout_li383_low_res_reference.nc'))

        # Try a case in which theta is specified:
        ss = [1.0e-15, 0.5, 1]
        alphas = [0, np.pi]
        theta = np.linspace(-np.pi, np.pi, 5)
        for s in ss:
            for alpha in alphas:
                fl = vmec_fieldline(vmec, s, alpha, theta1d=theta)
                np.testing.assert_allclose(fl.theta_pest, theta, atol=1e-15)
                np.testing.assert_allclose(fl.theta_pest - fl.iota * fl.phi, alpha,
                                           atol=1e-15)

        # Try a case in which phi is specified:
        s = 1
        alpha = -np.pi
        phi = np.linspace(-np.pi, np.pi, 6)
        fl = vmec_fieldline(vmec, s, alpha, phi1d=phi)
        np.testing.assert_allclose(fl.phi, phi)
        np.testing.assert_allclose(fl.theta_pest - fl.iota * fl.phi, alpha)

        # Try specifying both theta and phi:
        with self.assertRaises(ValueError):
            fl = vmec_fieldline(vmec, s, alpha, phi1d=phi, theta1d=theta)
        # Try specifying neither theta nor phi:
        with self.assertRaises(ValueError):
            fl = vmec_fieldline(vmec, s, alpha)

    def test_consistency(self):
        """
        Check internal consistency of the results of vmec_fieldline().
        """
        filenames = ['wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc',
                     'wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc',
                     'wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc']
        for filename in filenames:
            for j_theta_phi_in in range(2):
                logger.info(f'Testing vmec_fieldline for file {filename}, j_theta_phi_in={j_theta_phi_in}')
                vmec = Vmec(os.path.join(TEST_DIR, filename))
                for s in [0.25, 0.75]:
                    for alpha in np.linspace(0, 2 * np.pi, 3, endpoint=False):
                        z_grid = np.linspace(-np.pi / 2, np.pi / 2, 7)
                        if j_theta_phi_in == 0:
                            fl = vmec_fieldline(vmec, s=s, alpha=alpha, phi1d=z_grid)
                        else:
                            fl = vmec_fieldline(vmec, s=s, alpha=alpha, theta1d=z_grid)

                        np.testing.assert_allclose(fl.sqrt_g_vmec, fl.sqrt_g_vmec_alt, rtol=1e-4, atol=1e-4)

                        # Verify that (B dot grad theta_pest) / (B dot grad phi) = iota
                        should_be_iota = (fl.B_sup_theta_vmec * (1 + fl.d_lambda_d_theta_vmec) + fl.B_sup_phi * fl.d_lambda_d_phi) / fl.B_sup_phi
                        np.testing.assert_allclose(fl.iota, should_be_iota, rtol=1e-4, atol=1e-4)

                        # Compare 2 methods of computing B_sup_theta_pest:
                        np.testing.assert_allclose(fl.B_sup_theta_vmec * (1 + fl.d_lambda_d_theta_vmec) + fl.B_sup_phi * fl.d_lambda_d_phi,
                                                fl.B_sup_theta_pest, rtol=1e-4)

                        # grad_phi_X should be -sin(phi) / R:
                        np.testing.assert_allclose(fl.grad_phi_X, -fl.sinphi / fl.R, rtol=1e-4)
                        # grad_phi_Y should be cos(phi) / R:
                        np.testing.assert_allclose(fl.grad_phi_Y, fl.cosphi / fl.R, rtol=1e-4)
                        # grad_phi_Z should be 0:
                        np.testing.assert_allclose(fl.grad_phi_Z, 0, atol=1e-16)

                        # Verify that the Jacobian equals the appropriate cross
                        # product of the basis vectors.
                        test_arr = 0 \
                            + fl.d_X_d_s * fl.d_Y_d_theta_vmec * fl.d_Z_d_phi \
                            + fl.d_Y_d_s * fl.d_Z_d_theta_vmec * fl.d_X_d_phi \
                            + fl.d_Z_d_s * fl.d_X_d_theta_vmec * fl.d_Y_d_phi \
                            - fl.d_Z_d_s * fl.d_Y_d_theta_vmec * fl.d_X_d_phi \
                            - fl.d_X_d_s * fl.d_Z_d_theta_vmec * fl.d_Y_d_phi \
                            - fl.d_Y_d_s * fl.d_X_d_theta_vmec * fl.d_Z_d_phi
                        np.testing.assert_allclose(test_arr, fl.sqrt_g_vmec, rtol=1e-4)

                        test_arr = 0 \
                            + fl.grad_s_X * fl.grad_theta_vmec_Y * fl.grad_phi_Z \
                            + fl.grad_s_Y * fl.grad_theta_vmec_Z * fl.grad_phi_X \
                            + fl.grad_s_Z * fl.grad_theta_vmec_X * fl.grad_phi_Y \
                            - fl.grad_s_Z * fl.grad_theta_vmec_Y * fl.grad_phi_X \
                            - fl.grad_s_X * fl.grad_theta_vmec_Z * fl.grad_phi_Y \
                            - fl.grad_s_Y * fl.grad_theta_vmec_X * fl.grad_phi_Z
                        np.testing.assert_allclose(test_arr, 1 / fl.sqrt_g_vmec, rtol=2e-4)

                        # Verify that \vec{B} dot (each of the covariant and
                        # contravariant basis vectors) matches the corresponding term
                        # from VMEC.
                        test_arr = fl.B_X * fl.d_X_d_theta_vmec + fl.B_Y * fl.d_Y_d_theta_vmec + fl.B_Z * fl.d_Z_d_theta_vmec
                        np.testing.assert_allclose(test_arr, fl.B_sub_theta_vmec, rtol=0.01, atol=0.01)

                        test_arr = fl.B_X * fl.d_X_d_s + fl.B_Y * fl.d_Y_d_s + fl.B_Z * fl.d_Z_d_s
                        np.testing.assert_allclose(test_arr, fl.B_sub_s, rtol=2e-3, atol=0.005)

                        test_arr = fl.B_X * fl.d_X_d_phi + fl.B_Y * fl.d_Y_d_phi + fl.B_Z * fl.d_Z_d_phi
                        np.testing.assert_allclose(test_arr, fl.B_sub_phi, rtol=1e-3)

                        test_arr = fl.B_X * fl.grad_s_X + fl.B_Y * fl.grad_s_Y + fl.B_Z * fl.grad_s_Z
                        np.testing.assert_allclose(test_arr, 0, atol=1e-14)

                        test_arr = fl.B_X * fl.grad_phi_X + fl.B_Y * fl.grad_phi_Y + fl.B_Z * fl.grad_phi_Z
                        np.testing.assert_allclose(test_arr, fl.B_sup_phi, rtol=1e-4)

                        test_arr = fl.B_X * fl.grad_theta_vmec_X + fl.B_Y * fl.grad_theta_vmec_Y + fl.B_Z * fl.grad_theta_vmec_Z
                        np.testing.assert_allclose(test_arr, fl.B_sup_theta_vmec, rtol=2e-4)

                        # Check 2 ways of computing B_cross_grad_s_dot_grad_alpha:
                        np.testing.assert_allclose(fl.B_cross_grad_s_dot_grad_alpha, fl.B_cross_grad_s_dot_grad_alpha_alternate, rtol=1e-3)

                        # Check 2 ways of computing B_cross_grad_B_dot_grad_alpha:
                        np.testing.assert_allclose(fl.B_cross_grad_B_dot_grad_alpha, fl.B_cross_grad_B_dot_grad_alpha_alternate, atol=0.02)

                        # Check 2 ways of computing cvdrift:
                        # For this comparison, I'll use the definitions of
                        # gbdrift/cvdrift presently in simsospt, which may differ
                        # in sign from the definitions we want in gx.
                        fl.gbdrift = -1 * 2 * fl.B_reference * fl.L_reference * fl.L_reference * fl.sqrt_s * fl.B_cross_grad_B_dot_grad_alpha / (fl.modB * fl.modB * fl.modB) * fl.toroidal_flux_sign
                        fl.cvdrift = fl.gbdrift - 2 * fl.B_reference * fl.L_reference * fl.L_reference * fl.sqrt_s * mu_0 * fl.d_pressure_d_s * fl.toroidal_flux_sign / (fl.edge_toroidal_flux_over_2pi *fl.modB * fl.modB)
                        cvdrift_alt = -1 * 2 * fl.B_reference * fl.L_reference * fl.L_reference \
                            * np.sqrt(fl.s) * fl.B_cross_kappa_dot_grad_alpha \
                            / (fl.modB * fl.modB) * fl.toroidal_flux_sign
                        np.testing.assert_allclose(fl.cvdrift, cvdrift_alt)

    def test_stella_regression(self):
        """
        Test vmec_fieldline() by comparing to calculations with the
        geometry interface in the gyrokinetic code stella.
        """
        vmec = Vmec(os.path.join(TEST_DIR, 'wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc'))
        phi = np.linspace(-np.pi / 2, np.pi / 2, 7)
        for jalpha, alpha in enumerate(np.linspace(0, 2 * np.pi, 3, endpoint=False)):
            s = 0.5
            fl = vmec_fieldline(vmec, s=s, alpha=alpha, phi1d=phi)

            theta_vmec_reference = np.array([[-0.58175490233450466, -0.47234459364571935, -0.27187109234173445, 0.0000000000000000, 0.27187109234173445, 0.47234459364571935, 0.58175490233450466],
                                            [1.3270163720562491, 1.5362773754015560, 1.7610074217338225, 1.9573739260757410, 2.1337336171762495, 2.3746560860701522, 2.7346254905898566],
                                            [3.5485598165897296, 3.9085292211094336, 4.1494516900033354, 4.3258113811038443, 4.5221778855704500, 4.7469079317780292, 4.9561689351233360]])
            np.testing.assert_allclose(fl.theta_vmec, theta_vmec_reference[jalpha, :], rtol=3e-10, atol=3e-10)

            modB_reference = np.array([[5.5314557915824105, 5.4914726973937169, 5.4623705859501106, 5.4515507758815724, 5.4623705859501106, 5.4914726973937169, 5.5314557915824105],
                                    [5.8059964497774104, 5.8955040231659970, 6.0055216679618466, 6.1258762025742488, 6.2243609712000030, 6.2905766503767877, 6.3322981435019834],
                                    [6.3322981435019843, 6.2905766503767877, 6.2243609712000039, 6.1258762025742506, 6.0055216679119008, 5.8955040231659970, 5.8059964497774113]])
            np.testing.assert_allclose(fl.modB, modB_reference[jalpha, :], rtol=1e-10, atol=1e-10)

            gradpar_reference = np.array([[0.16966605523903308, 0.16128460473382750, 0.13832030925488656, 0.12650509948655936, 0.13832030925488656, 0.16128460473382750, 0.16966605523903308],
                                        [0.18395371643088332, 0.16267463440217919, 0.13722808138129616, 0.13475541851920048, 0.16332776641155652, 0.20528460628581335, 0.22219989371670806],
                                        [0.22219989371670806, 0.20528460628581333, 0.16332776641155652, 0.13475541851920050, 0.13722808138029491, 0.16267463440217922, 0.18395371643088337]])
            L_reference = vmec.wout.Aminor_p
            np.testing.assert_allclose(L_reference * fl.B_sup_phi / fl.modB,
                                    gradpar_reference[jalpha, :], rtol=1e-11, atol=1e-11)

            s = 1.0

            fl = vmec_fieldline(vmec, s=s, alpha=alpha, phi1d=phi)
            theta_vmec_reference = np.array([[-0.54696449126914626, -0.48175613245664178, -0.27486119402681097, 0.0000000000000000, 0.27486119402681097, 0.48175613245664178, 0.54696449126914626],
                                            [1.2860600505790374, 1.4762629621552081, 1.7205057726038357, 1.8975933573125818, 2.0499492968214290, 2.3142882369220339, 2.7218102365172787],
                                            [3.5613750706623071, 3.9688970702575519, 4.2332360103581568, 4.3855919498670044, 4.5626795345757500, 4.8069223450243772, 4.9971252566005484]])
            # Tolerances for s=1 are a bit looser since we extrapolate lambda off the half grid:
            np.testing.assert_allclose(fl.theta_vmec, theta_vmec_reference[jalpha, :], rtol=3e-5, atol=3e-5)


        # Compare to an output file from the stella geometry interface.        
        vmec = Vmec(os.path.join(TEST_DIR, 'wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc'))
        s = 0.5
        nalpha = 3
        alpha = np.linspace(0, 2 * np.pi, nalpha, endpoint=False)
        phi = np.linspace(-np.pi / 5, np.pi / 5, 7)
        with open(os.path.join(TEST_DIR, 'geometry_W7-X_without_coil_ripple_beta0p05_d23p4_tm.dat')) as f:
            lines = f.readlines()
        np.testing.assert_allclose(alpha, np.fromstring(lines[4], sep=' '))
        phi_stella = np.fromstring(lines[6], sep=' ')

        for j in range(nalpha):
            fl = vmec_fieldline(vmec, s=s, alpha=alpha[j], phi1d=phi)

            # For this comparison, I'll use the definitions of
            # gds21/gbdrift/gbdrift0/cvdrift presently in simsospt, which may differ
            # in sign from the definitions we want in gx.

            fl.gds21 = fl.grad_alpha_dot_grad_psi * fl.shat / fl.B_reference

            # See issue #238 and the discussion therein
            fl.gbdrift = -1 * 2 * fl.B_reference * fl.L_reference * fl.L_reference * fl.sqrt_s * fl.B_cross_grad_B_dot_grad_alpha / (fl.modB * fl.modB * fl.modB) * fl.toroidal_flux_sign

            fl.gbdrift0 = fl.B_cross_grad_B_dot_grad_psi * 2 * fl.shat / (fl.modB * fl.modB * fl.modB * fl.sqrt_s) * fl.toroidal_flux_sign

            # See issue #238 and the discussion therein
            fl.cvdrift = fl.gbdrift - 2 * fl.B_reference * fl.L_reference * fl.L_reference * fl.sqrt_s * mu_0 * fl.d_pressure_d_s * fl.toroidal_flux_sign / (fl.edge_toroidal_flux_over_2pi *fl.modB * fl.modB)

            fl.cvdrift0 = fl.gbdrift0

            np.testing.assert_allclose(fl.phi, phi_stella)
            np.testing.assert_allclose(fl.bmag, np.fromstring(lines[8 + j], sep=' '), rtol=1e-6)
            np.testing.assert_allclose(fl.gradpar_phi, np.fromstring(lines[12 + j], sep=' '), rtol=1e-6)
            np.testing.assert_allclose(fl.gds2, np.fromstring(lines[16 + j], sep=' '), rtol=2e-4)
            np.testing.assert_allclose(fl.gds21, np.fromstring(lines[20 + j], sep=' '), atol=2e-4)
            np.testing.assert_allclose(fl.gds22, np.fromstring(lines[24 + j], sep=' '), atol=2e-4)
            np.testing.assert_allclose(-1 * fl.gbdrift, np.fromstring(lines[28 + j], sep=' '), atol=2e-4)
            np.testing.assert_allclose(fl.gbdrift0, np.fromstring(lines[32 + j], sep=' '), atol=1e-4)
            np.testing.assert_allclose(-1 * fl.cvdrift, np.fromstring(lines[36 + j], sep=' '), atol=2e-4)
            np.testing.assert_allclose(fl.cvdrift0, np.fromstring(lines[40 + j], sep=' '), atol=1e-4)

    def test_axisymm(self):
        """
        Test that vmec_fieldline() gives sensible results for axisymmetry.
        """
        filenames = [
            "wout_circular_tokamak_aspect_100_phiedgePositive_reference.nc",
            "wout_circular_tokamak_aspect_100_phiedgeNegative_reference.nc",
        ]
        for filename in filenames:
            vmec = Vmec(os.path.join(TEST_DIR, filename))
            theta = np.linspace(-3 * np.pi, 3 * np.pi, 200)
            fl = vmec_fieldline(vmec, s=1, alpha=0, theta1d=theta, plot=0)
            B0 = vmec.wout.volavgB
            eps = 1 / vmec.wout.aspect
            safety_factor_q = 1 / vmec.wout.iotaf[-1]
            phi = theta / vmec.wout.iotaf[-1]
            R = vmec.wout.Rmajor_p
            Aminor = vmec.wout.Aminor_p
            d_iota_d_s = (vmec.wout.iotas[-1] - vmec.wout.iotas[-2]) / vmec.ds
            d_iota_d_r = d_iota_d_s * 2 / Aminor
            #print('sign of psi in grad psi cross grad theta + iota grad phi cross grad psi:', fl.toroidal_flux_sign)

            # See Matt Landreman's note "20220315-02 Geometry arrays for gyrokinetics in a circular tokamak.docx"
            # for the analytic results below
            np.testing.assert_allclose(fl.modB, B0 * (1 - eps * np.cos(theta)), rtol=0.0002)
            np.testing.assert_allclose(fl.B_sup_theta_vmec, -fl.toroidal_flux_sign * B0 / (safety_factor_q * R), rtol=0.0006)
            np.testing.assert_allclose(fl.B_cross_grad_B_dot_grad_psi, -(B0 ** 3) * eps * np.sin(theta), rtol=0.03)
            np.testing.assert_allclose(fl.B_cross_kappa_dot_grad_psi, -(B0 ** 2) * eps * np.sin(theta), rtol=0.02)
            np.testing.assert_allclose(fl.grad_psi_dot_grad_psi, B0 * B0 * Aminor * Aminor, rtol=0.03)
            np.testing.assert_allclose(fl.grad_alpha_dot_grad_psi, -fl.toroidal_flux_sign * phi * d_iota_d_r * Aminor * B0, rtol=0.02)
            np.testing.assert_allclose(fl.grad_alpha_dot_grad_alpha, 1 / (Aminor * Aminor) + (phi * phi * d_iota_d_r * d_iota_d_r), rtol=0.04)
            np.testing.assert_allclose(fl.B_cross_grad_B_dot_grad_alpha,
                                    fl.toroidal_flux_sign * (B0 * B0 / Aminor) * (-np.cos(theta) / R + phi * d_iota_d_r * eps * np.sin(theta)),
                                    atol=0.006)
            np.testing.assert_allclose(fl.B_cross_kappa_dot_grad_alpha,
                                    fl.toroidal_flux_sign * (B0 / Aminor) * (-np.cos(theta) / R + phi * d_iota_d_r * eps * np.sin(theta)),
                                    atol=0.006)

    def test_plot(self):
        """
        Test the plotting function of vmec_fieldline()
        """
        vmec = Vmec(os.path.join(TEST_DIR, 'wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc'))

        phi = np.linspace(-np.pi / 5, np.pi / 5, 7)
        fl = vmec_fieldline(vmec, s=1, alpha=0, phi1d=phi, plot=True, show=False)

        theta = np.linspace(-np.pi, np.pi, 100)
        fl = vmec_fieldline(vmec, s=0.5, alpha=np.pi, theta1d=theta, plot=True, show=False)


if __name__ == "__main__":
    unittest.main()
