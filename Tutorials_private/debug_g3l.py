import os
import sys
import numpy as np
from pathlib import Path
from time import time
import pickle
import logging
import time
from astropy.table import Table
import pandas as pd
import orpheus

#Load Millenium
x_sources = np.array([])
y_sources = np.array([])
w_sources = np.array([])
z_sources = np.array([])
e1_sources = np.array([])
e2_sources = np.array([])

x_lenses = np.array([])
y_lenses = np.array([])
w_lenses = np.array([])
z_lenses = np.array([])

nbar_sources = 8.
eltile = 6
basemillenium = "/vol/euclidraid4/data/lporth/HigherOrderLensing/Mocks/millenium_catalogs/"
nextsources = pd.read_csv(basemillenium + "tile%i_LOS_%i.sources.dat"%(eltile,800), header=None, sep=' ')
nextlenses  = pd.read_csv(basemillenium + "tile%i_LOS_%i.objects.dat"%(eltile,800), header=None, sep=' ')

source_inds = np.random.choice(nextsources.shape[0], size=int(nbar_sources*60*60), replace=False)
x_lenses = np.append(x_lenses, np.array(nextlenses[0]))
y_lenses = np.append(y_lenses, np.array(nextlenses[1]))
z_lenses = np.append(z_lenses, np.array(nextlenses[4]))
w_lenses = np.append(w_lenses, np.array(nextlenses[5]))
x_sources = np.append(x_sources, np.array(nextsources[0])[source_inds])
y_sources = np.append(y_sources, np.array(nextsources[1])[source_inds])
e1_sources = np.append(e1_sources, np.array(nextsources[2])[source_inds])
e2_sources = np.append(e2_sources, np.array(nextsources[3])[source_inds])
z_sources = np.append(z_sources, np.array(nextsources[4])[source_inds])
w_sources = np.append(w_sources, np.array(nextsources[5])[source_inds])
zbin_lenses = np.zeros(len(w_lenses), dtype=int)
zbin_sources = np.zeros(len(w_sources), dtype=int)

n_cfs = 1
min_sep = 0.5
max_sep = 40.
nbinsr = 30
nmax = 30
dpixs = [0.,0.25,0.5,1.0]
rmin_pixsize = 20
method = "DoubleTree"
shuffle_pix = 1
resoshift_leafs = 0
nthreads = 16
dotomo_lens=False
dotomo_source=False

# Some adhoc zbins
if dotomo_lens:
    zedges = [0,0.2,0.3,2.0]
    for elz in range(len(zedges)-1):
        selz = np.logical_and(z_lenses>zedges[elz], z_lenses<=zedges[elz+1])
        zbin_lenses[selz] = elz
if dotomo_source:
    zbin_sources[::3] = 1
    
sources = orpheus.SpinTracerCatalog(spin=2, pos1=x_sources, pos2=y_sources, tracer_1=e1_sources, tracer_2=e2_sources, zbins=zbin_sources)
lenses = orpheus.ScalarTracerCatalog(pos1=x_lenses, pos2=y_lenses, tracer=w_lenses, zbins=zbin_lenses)
g3l = orpheus.GNNCorrelation(min_sep=min_sep, max_sep=max_sep, nmaxs=nmax, tree_resos=dpixs, multicountcorr=True,
                                        rmin_pixsize=rmin_pixsize,  method=method, shuffle_pix=shuffle_pix, nbinsr=nbinsr, 
                                        resoshift_leafs=resoshift_leafs, nthreads=nthreads)
print(sources.ngal, lenses.ngal)
t1 = time.time()
g3l.process(sources, lenses, nthreads=16, dotomo_lens=dotomo_lens, dotomo_source=dotomo_source)
t2 = time.time()
print("Needed %.2f seconds"%(t2-t1))
NNMap_disc = g3l.computeNNM(np.geomspace(1,10,10))
print(NNMap_disc.shape)
print(NNMap_disc.real)
print(NNMap_disc.imag)