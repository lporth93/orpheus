import os
import sys
import numpy as np
from pathlib import Path
from time import time
import pickle
import logging
from astropy.table import Table
import orpheus

# Parameters for 3pcf
path_to_slics = "/cosma7/data/dp004/dc-port3/Data/SLICS/KiDS450/GalCatalog_LOS1.fits"
base_save = "/cosma/home/dp004/dc-port3/Projects/ToGithub/Orpheus/data/"
min_sep = 0.25
max_sep = 256.
rbins_per_double = 6
nbinsr = int(10*rbins_per_double)
nmax = 100
nthreads = 16

# Helper functions
def pickle_save(data, fname):
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def save_data(inst, t_elapsed, fname):
    results = {}
    results["Time"] = t_elapsed
    results["Pars"] = {}
    results["Pars"]["catalog"] = path_to_slics
    results["Pars"]["min_sep"] = inst.min_sep
    results["Pars"]["max_sep"] = inst.max_sep
    results["Pars"]["bin_edges"] = inst.bin_edges
    results["Pars"]["bin_centers"] = inst.bin_centers
    results["Pars"]["nbinsphi"] = inst.nbinsphi
    results["Pars"]["nmaxs"] = inst.nmaxs
    results["Pars"]["method"] = inst.method
    results["Pars"]["multicountcorr"] = inst.multicountcorr
    results["Pars"]["tree_resos"] = inst.tree_resos
    results["Pars"]["tree_nresos"] = inst.tree_nresos
    results["Pars"]["tree_redges"] = inst.tree_redges
    results["Pars"]["rmin_pixsize"] = inst.rmin_pixsize
    results["Pars"]["tree_resosatr"] = inst.tree_resosatr
    results["Pars"]["nthreads"] = nthreads    
    results["Upsilon"] = inst.npcf_multipoles
    results["Norm"] = inst.npcf_multipoles_norm
    results["NPCF"] = inst.npcf
    results["Ntriangles"] = inst.npcf_norm
    pickle_save(results, fname)    

# Initialize full catalog
data = Table.read(path_to_slics)
pos1 = data["x_arcmin"].data
pos2 = data["y_arcmin"].data
gamma1 = data["shear1"].data
gamma2 = data["shear2"].data
e1 = data["eps_obs1"].data
e2 = data["eps_obs2"].data
zspec = data["z_spectroscopic"].data
zphot = data["z_photometric"].data
slicsfullreso = orpheus.SpinTracerCatalog(spin=2, pos1=pos1, pos2=pos2, tracer_1=gamma1, tracer_2=gamma2, zbins=None)


# 1) Do fully discrete computation with reduced catalogs
reduced_resos = [8.0,4.0,2.0,1.0,0.5,0.25]
max_seps = [max_sep,max_sep,max_sep,max_sep,max_sep/4.]
nbinsrs = [nbinsr,nbinsr,nbinsr,nbinsr,nbinsr-2*rbins_per_double]
str_resos = ["8p0", "4p0", "2p0", "1p0", "0p5", "0p25"]
red_shuffle=True
for elreso, reduced_reso in enumerate(reduced_resos[1:4]):
    fname_save = "ResoTest_SLICSNoiseless_discrete_reduced_shuffled%s"%str_resos[elreso]
    t1 = time()
    slics_reduced = slicsfullreso.reduce(reduced_reso, shuffle=red_shuffle, ret_inst=True)
    threepcf_reduced = orpheus.GGGCorrelation(n_cfs=4, min_sep=min_sep, max_sep=max_seps[elreso], nbinsr=nbinsrs[elreso], 
                                              nmaxs=nmax, method="Discrete")
    threepcf_reduced.process(slics_reduced, nthreads=nthreads)
    t2 = time()
    threepcf_reduced.multipoles2npcf()
    save_data(threepcf_reduced, t2-t1, base_save+fname_save)

"""
# 2) Do tree-based computation for different values of `rmin_pixsize'
rmin_pixsizes = [5,10,20,30,40]
for elrmin, rmin_pixsize in enumerate(rmin_pixsizes):
    fname_save = "ResoTest_SLICSNoiseless_tree_rminpixsizes%i"%rmin_pixsize
    t1 = time()
    threepcf_fulltree = orpheus.GGGCorrelation(n_cfs=4,min_sep=min_sep, max_sep=max_sep, nbinsr=nbinsr, 
                                               tree_resos=[0,0.25,0.5,1.,2.,4.], nmaxs=nmax, rmin_pixsize=rmin_pixsize, method="Tree")
    print(threepcf_fulltree.tree_redges)
    threepcf_fulltree.process(slicsfullreso, nthreads=nthreads)
    t2 = time()
    threepcf_fulltree.multipoles2npcf()
    save_data(threepcf_fulltree, t2-t1, base_save+fname_save)

# 3) Do doubletree-based computation for different values of `rmin_pixsize'
    
# 4) Do full discrete computation
fname_save = "ResoTest_SLICSNoiseless_discrete_fullcat"
t1 = time()
threepcf_full = orpheus.GGGCorrelation(n_cfs=4, min_sep=min_sep, max_sep=max_sep/4., nbinsr=nbinsr-2*rbins_per_double, 
                                       nmaxs=nmax, method="Discrete")
threepcf_full.process(slicsfullreso, nthreads=nthreads)
t2 = time()
threepcf_full.multipoles2npcf()
save_data(threepcf_full, t2-t1, base_save+fname_save)
"""