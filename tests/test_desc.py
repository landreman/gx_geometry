#!/usr/bin/env python

import unittest
import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from desc.vmec import VMECIO

from gx_geometry import desc_fieldline, Vmec, vmec_fieldline, uniform_arclength, add_gx_definitions

from . import TEST_DIR

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)

class Tests(unittest.TestCase):
    def test_wout_as_desc(self):
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
                plt.plot(fl2.theta_pest, np.flip(fl2.__getattribute__(field)) * signflip, label="vmec")
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
            plt.figtext(0.5, 0.995, "Before uniform arclength", ha="center", va="top", fontsize=9)
            #plt.show()

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
        #show_diffs(fl1.gradpar_theta_pest, np.flip(fl2.gradpar_theta_pest))
        np.testing.assert_allclose(fl1.gradpar_theta_pest, -np.flip(fl2.gradpar_theta_pest), rtol=0.001)
        np.testing.assert_allclose(fl1.grho, np.flip(fl2.grho), rtol=0.01)
        np.testing.assert_allclose(fl1.grad_psi_dot_grad_psi, np.flip(fl2.grad_psi_dot_grad_psi), rtol=0.01)
        np.testing.assert_allclose(fl1.grad_alpha_dot_grad_psi, np.flip(fl2.grad_alpha_dot_grad_psi), atol=0.2)
        np.testing.assert_allclose(fl1.grad_alpha_dot_grad_alpha, np.flip(fl2.grad_alpha_dot_grad_alpha), atol=0.06)
        np.testing.assert_allclose(fl1.B_cross_grad_B_dot_grad_psi, -np.flip(fl2.B_cross_grad_B_dot_grad_psi), rtol=0.1, atol=1.3)
        np.testing.assert_allclose(fl1.B_cross_kappa_dot_grad_psi, -np.flip(fl2.B_cross_kappa_dot_grad_psi), atol=0.04)
        np.testing.assert_allclose(fl1.B_cross_grad_B_dot_grad_alpha, -np.flip(fl2.B_cross_grad_B_dot_grad_alpha), atol=0.75)
        np.testing.assert_allclose(fl1.B_cross_kappa_dot_grad_alpha, -np.flip(fl2.B_cross_kappa_dot_grad_alpha), atol=0.006)

        # Now convert to gx grid and quantities:
        fl3 = uniform_arclength(fl1)
        fl4 = uniform_arclength(fl2)
        kxfac = -1
        add_gx_definitions(fl3, kxfac)
        add_gx_definitions(fl4, kxfac)

        if plot_all_quantities:
            plt.figure(figsize=(14, 7.5))
            nrows = 3
            ncols = 4
            jplot = 1
            def compare_vmec_desc(field, signflip=1):
                nonlocal jplot
                plt.subplot(nrows, ncols, jplot)
                jplot = jplot + 1
                #plt.plot(fl3.theta_pest, fl3.__getattribute__(field), label="desc")
                #plt.plot(fl4.theta_pest, np.flip(fl4.__getattribute__(field)) * signflip, label="vmec")
                plt.plot(fl3.__getattribute__(field), label="desc")
                plt.plot(np.flip(fl4.__getattribute__(field)) * signflip, label="vmec")
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
            plt.figtext(0.5, 0.995, "After uniform arclength", ha="center", va="top", fontsize=9)
            #plt.show()

        # Scalars:
        np.testing.assert_allclose(fl3.iota, -fl4.iota)
        np.testing.assert_allclose(fl3.shat, fl4.shat, rtol=0.0002)
        np.testing.assert_allclose(fl3.L_reference, fl4.L_reference)
        np.testing.assert_allclose(fl3.B_reference, fl4.B_reference)

        # Quantities that vary along a field line:
        #show_diffs(fl3.theta_pest, -np.flip(fl4.theta_pest))
        np.testing.assert_allclose(fl3.theta_pest, -np.flip(fl4.theta_pest), atol=0.0002)
        np.testing.assert_allclose(fl3.theta_desc, -np.flip(fl4.theta_vmec), rtol=0.006, atol=0.003)
        np.testing.assert_allclose(fl3.zeta, np.flip(fl4.phi), atol=0.0002)
        np.testing.assert_allclose(fl3.modB, np.flip(fl4.modB), rtol=0.008)
        np.testing.assert_allclose(fl3.gradpar_theta_pest, -np.flip(fl4.gradpar_theta_pest), rtol=0.001)
        np.testing.assert_allclose(fl3.grho, np.flip(fl4.grho), rtol=0.01)
        np.testing.assert_allclose(fl3.grad_psi_dot_grad_psi, np.flip(fl4.grad_psi_dot_grad_psi), rtol=0.01)
        np.testing.assert_allclose(fl3.grad_alpha_dot_grad_psi, np.flip(fl4.grad_alpha_dot_grad_psi), atol=0.2)
        np.testing.assert_allclose(fl3.grad_alpha_dot_grad_alpha, np.flip(fl4.grad_alpha_dot_grad_alpha), atol=0.06)
        np.testing.assert_allclose(fl3.B_cross_grad_B_dot_grad_psi, -np.flip(fl4.B_cross_grad_B_dot_grad_psi), rtol=0.1, atol=1.3)
        np.testing.assert_allclose(fl3.B_cross_kappa_dot_grad_psi, -np.flip(fl4.B_cross_kappa_dot_grad_psi), atol=0.04)
        #show_diffs(fl3.B_cross_grad_B_dot_grad_alpha, -np.flip(fl4.B_cross_grad_B_dot_grad_alpha))
        np.testing.assert_allclose(fl3.B_cross_grad_B_dot_grad_alpha, -np.flip(fl4.B_cross_grad_B_dot_grad_alpha), atol=0.82)
        np.testing.assert_allclose(fl3.B_cross_kappa_dot_grad_alpha, -np.flip(fl4.B_cross_kappa_dot_grad_alpha), atol=0.006)

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
                plt.plot(np.flip(fl4.__getattribute__(field)) * signflip, label="vmec")
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


if __name__ == "__main__":
    unittest.main()
