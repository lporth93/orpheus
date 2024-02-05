# Debugging for orpheus

import os
import sys
import numpy as np
from pathlib import Path
from time import time
import pickle
import logging
from astropy.table import Table
import orpheus
from matplotlib import pyplot as plt

dotomo=False
fthin = 1
data = Table.read("/cosma7/data/dp004/dc-port3/Data/SLICS/KiDS450/GalCatalog_LOS1.fits")
pos1 = data["x_arcmin"].data
pos2 = data["y_arcmin"].data
gamma1 = data["shear1"].data
gamma2 = data["shear2"].data
e1 = data["eps_obs1"].data
e2 = data["eps_obs2"].data
zspec = data["z_spectroscopic"].data
zphot = data["z_photometric"].data
zbin = np.zeros(len(pos1),dtype=np.int32)
if dotomo:
    zbin[zphot<=0.25] = 0 
    zbin[(zphot>0.25)*(zphot<=0.5)] = 0
    zbin[(zphot>0.5)*(zphot<=0.75)] = 1
    zbin[(zphot>0.75)*(zphot<=1.)] = 1
    zbin[zphot>1.] = 1
    
slicsfullreso = orpheus.SpinTracerCatalog(spin=2, pos1=pos1[::fthin], pos2=pos2[::fthin], 
                                          tracer_1=gamma1[::fthin], tracer_2=-gamma2[::fthin], zbins=zbin[::fthin])
print("We have %i galaxies in the mock catalog"%slicsfullreso.ngal)

n_cfs = 4
min_sep = 0.5
max_sep_disc = 5.
max_sep = 200.
rmin_pixsize=20
dpixs = [0,0.25,0.5,1.0,2.0,4.0]
binsize=.15
nthreads = 16

print("Doint disc")
#threepcf_disc = orpheus.GGGCorrelation(n_cfs=n_cfs, min_sep=min_sep, max_sep=5., binsize=binsize, method="Discrete")
#threepcf_disc.process(slicsfullreso, nthreads=16, dotomo=dotomo)

print("Doing dtree")
threepcf_dtree = orpheus.GGGCorrelation(n_cfs=n_cfs, min_sep=min_sep, max_sep=max_sep, tree_resos=dpixs, multicountcorr=False,
                                       binsize=binsize, rmin_pixsize=rmin_pixsize,  method="DoubleTree")
threepcf_dtree.process(slicsfullreso, nthreads=nthreads, dotomo=dotomo)
print(threepcf_dtree.bin_edges)
print(threepcf_dtree.bin_centers)