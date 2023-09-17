# gx_geometry

This repository is for making sure we completely understand the signs of the
geometric quantities in GX. It is associated with the note
https://www.overleaf.com/read/rsnfjjdyvnpt and contains implementations of the
formulas there.
The tests in this repository make sure that the geometric quantities are
internally consistent, that they correctly match expected values for a tokamak, and
that results computed from vmec and desc agree with each other, in addition to
other checks.

This package is not in PyPI, but you can install it as follows:
~~~~
pip install -e .
~~~~

There are two main driver routines, for creating GX eik files from vmec or desc
equilibria:
~~~~py
import gx_geometry
gx_geometry.create_eik_from_vmec("wout_extension.nc")
gx_geometry.create_eik_from_desc("desc_output_file.h5")
~~~~
In these driver routines you can also specify the flux surface, field line label, output file name, etc.
