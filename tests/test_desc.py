#!/usr/bin/env python

import unittest
import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from desc.vmec import VMECIO

from gx_geometry.desc import desc_fieldline
from gx_geometry.vmec import Vmec
from gx_geometry.vmec_diagnostics import vmec_fieldlines

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
        fl2 = vmec_fieldlines(vmec, s, alpha, theta1d=theta1d)

        if False:
            plt.figure(figsize=(14, 7.5))
            nrows = 3
            ncols = 4
            jplot = 1
            def compare_vmec_desc(field, signflip=1):
                nonlocal jplot
                plt.subplot(nrows, ncols, jplot)
                jplot = jplot + 1
                plt.plot(fl1.theta_pest, fl1.__getattribute__(field), label="desc")
                plt.plot(fl2.theta_pest, fl2.__getattribute__(field) * signflip, label="vmec")
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

        # Jacobian of (rho, theta, phi) is negative for vmec, positive for desc,
        # as theta_desc = -theta_vmec. Therefore iota flips sign.

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
        np.testing.assert_allclose(fl1.grad_psi_dot_grad_psi, np.flip(fl2.grad_psi_dot_grad_psi), rtol=0.01)
        np.testing.assert_allclose(fl1.grad_alpha_dot_grad_psi, np.flip(fl2.grad_alpha_dot_grad_psi), atol=0.2)
        np.testing.assert_allclose(fl1.grad_alpha_dot_grad_alpha, np.flip(fl2.grad_alpha_dot_grad_alpha), atol=0.06)
        np.testing.assert_allclose(fl1.B_cross_grad_B_dot_grad_psi, -np.flip(fl2.B_cross_grad_B_dot_grad_psi), rtol=0.1, atol=1.3)
        np.testing.assert_allclose(fl1.B_cross_kappa_dot_grad_psi, -np.flip(fl2.B_cross_kappa_dot_grad_psi), atol=0.04)
        np.testing.assert_allclose(fl1.B_cross_grad_B_dot_grad_alpha, -np.flip(fl2.B_cross_grad_B_dot_grad_alpha), atol=0.75)
        np.testing.assert_allclose(fl1.B_cross_kappa_dot_grad_alpha, -np.flip(fl2.B_cross_kappa_dot_grad_alpha), atol=0.006)


if __name__ == "__main__":
    unittest.main()
