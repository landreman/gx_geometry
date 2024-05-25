#!/usr/bin/env python

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import gx_geometry as gx

print("Usage:", __file__, "<1 or more eik files>")

fls = []
files = []
nfiles = len(sys.argv) - 1
for jfile in range(nfiles):
    filename = sys.argv[jfile + 1]
    files.append(filename)
    print("Reading file", filename)
    fls.append(gx.read_eik(filename))

plt.figure(figsize=(14.5, 7.5))
nrows = 3
ncols = 3

fields = [
    "bmag",
    "gbdrift",
    "gbdrift0",
    "cvdrift",
    "cvdrift0",
    "gds2",
    "gds21",
    "gds22",
    "grho",
]

for jfield in range(len(fields)):
    plt.subplot(nrows, ncols, jfield + 1)
    field = fields[jfield]
    for jfile in range(nfiles):
        plt.plot(
            fls[jfile].z.flatten(),
            fls[jfile].__getattribute__(field).flatten(),
            label=files[jfile],
        )
    plt.xlabel("z")
    plt.title(field)

plt.figtext(0.5, 0.005, os.path.abspath(__file__), ha="center", va="bottom", fontsize=6)
plt.tight_layout()
plt.legend(loc="upper right", fontsize=7)
plt.show()
