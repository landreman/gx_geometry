#!/usr/bin/env python

from desc.vmec import VMECIO

filename = "wout_w7x_from_gx_repository.nc"

eq = VMECIO.load(filename)
LMN = 8
eq.change_resolution(
    L=LMN,
    M=LMN,
    N=LMN,
    L_grid=LMN*2,
    M_grid=LMN*2,
    N_grid=LMN*2,
)
eq.solve()
eq.save("w7x_from_gx_repository.h5")
