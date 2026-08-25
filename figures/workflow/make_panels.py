# In this script we run the analysis and create the small figures
# that are used for the main workflow figure in the README
# To recreate this you only need to download the corresponding lensplane
# URL to suite: http://cosmo.phys.hirosaki-u.ac.jp/takahasi/allsky_raytracing/
# Lensplane used here: allskymap_nres13r000.zs16.mag.dat

import os

import numpy as np
import healpy as hp
from scipy.ndimage import gaussian_filter

import orpheus

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


##############
# PARAMETERS #
##############
# File paths
BASE_ORPHEUS = "/vol/euclidraid4/data/lporth/HigherOrderLensing/Estimator/orpheus/"
BASE_PANELS = BASE_ORPHEUS+"figures/workflow/"
PATH_T17 = BASE_ORPHEUS+"docs/source/notebooks/allskymap_nres13r000.zs16.mag.dat"
PATH_MEAS = BASE_ORPHEUS+"figures/workflow/data/t17_ggg_full.npz"
PATH_SMALLMEAS = BASE_ORPHEUS+"figures/workflow/data/t17_ggg_forfig.npz"
DO_MEAS = False

# Seeds and geometry for footprint generation and associated plots
SEED_MASK, SEED_TRIANGLES = 42, 44
NSIDE_VIEW, NSIDE_MASK = 1024, 512
NHOLES, R_HOLE_MEAN, R_HOLE_STD = 420, 1.1, .35 # degrees
PATCH_CENTRE, PATCH_RESO, PATCH_SPAN = (40., -12.), .1, 300. # deg, deg, arcmin, arcmin
THETA1, THETA2 = 10.5, 15.8 # arcmin, arcmin
NGAL_SHOWN = 400 # galaxies drawn in the patch
NTR_SHOWN_ROW = 4 # sqrt of triangles shown in patch
SHOW_ALL_ORDERS = True # If false only show triangles else show 2/3/4pt multiplets

# Measurement setup
NBAR, NSIDE_HP = .5, 8192
MIN_SEP, MAX_SEP, BINSIZE = .25, 240., .1
NBINSPHI, NMAXS, RMIN_PIXSIZE = 100, 30, 20
NPATCHES, NTHREADS = 125, 48
APRADII = np.geomspace(1., 32., 25)

################
# HELPER FUNCS #
################

# Read T17 plane; 
def read_t17(path):
    """kappa, gamma1, gamma2 from a T17 all-sky map, in the layout the T17 site documents."""
    skip = [0, 536870908, 1073741818, 1610612728, 2147483638, 2684354547, 3221225457]
    blocks = [skip[i + 1] - skip[i] for i in range(6)]
    out = []
    with open(path, "rb") as f:
        np.fromfile(f, dtype="uint32", count=1)
        np.fromfile(f, dtype="int32", count=1)
        npix = np.fromfile(f, dtype="int64", count=1)[0]
        np.fromfile(f, dtype="uint32", count=2)
        for _ in range(3): # kappa, gamma1, gamma2
            got, left = [], npix
            for i, l in enumerate(blocks):
                n = min(l, left)
                got.append(np.fromfile(f, dtype="float32", count=n))
                np.fromfile(f, dtype="uint32", count=2)
                left -= n
                if left == 0:
                    break
                if i == len(blocks) - 1:
                    got.append(np.fromfile(f, dtype="float32", count=left))
                    np.fromfile(f, dtype="uint32", count=2)
            out.append(np.concatenate(got))
    return out

# Put poisson distributed on points on sphere and cut restrict them to available area
def masked_catalog(kappa, g1, g2):
    npoints = int(41_252.96*3600*NBAR)
    rng = np.random.default_rng(SEED_MASK)
    ra = rng.uniform(0., 2.*np.pi, npoints)
    dec = np.arcsin(rng.uniform(-1., 1., npoints))

    big = (ra < 2.*np.pi/3.) & ((ra + 2.*dec) < .7*np.pi)
    big &= ~(((ra - .4*dec) > np.radians(70.)) & ((ra - .4*dec) < np.radians(80.)))
    rad = np.abs(rng.normal(R_HOLE_MEAN, R_HOLE_STD, NHOLES))
    idx = rng.integers(hp.nside2npix(NSIDE_MASK), size=NHOLES)
    holes = set()
    for i in range(NHOLES):
        holes |= set(hp.query_disc(NSIDE_MASK, hp.pix2vec(NSIDE_MASK, idx[i]),
                                   np.radians(rad[i])))
    hole_map = np.ones(hp.nside2npix(NSIDE_MASK))
    hole_map[list(holes)] = 0.
    _, on = orpheus.cat2hpx(lon=np.degrees(ra[big]), lat=np.degrees(dec[big]),
                            nside=NSIDE_MASK, return_indices=True)
    keep = (on*hole_map[on]).astype(bool)
    ra, dec = np.degrees(ra[big][keep]), np.degrees(dec[big][keep])
    _, pix = orpheus.cat2hpx(lon=ra, lat=dec, nside=NSIDE_HP, return_indices=True)
    print("catalog: %d galaxies on the footprint" % len(ra), flush=True)
    return orpheus.SpinTracerCatalog(spin=2, pos1=ra, pos2=dec,
                                     tracer_1=-g1[pix], tracer_2=-g2[pix],
                                     units_pos1="deg", units_pos2="deg",
                                     geometry="spherical")

# Build & plot mock survey foorprint in the same way as in the advanced GG/GGG tutorial
def footprint(kappa):
    
    # Build footprint
    theta, phi = hp.pix2ang(NSIDE_VIEW, np.arange(hp.nside2npix(NSIDE_VIEW)))
    ra, dec = phi, np.pi/2. - theta
    keep = (ra < 2.*np.pi/3.) & ((ra + 2.*dec) < .7*np.pi)
    keep &= ~(((ra - .4*dec) > np.radians(70.)) & ((ra - .4*dec) < np.radians(80.)))
    rng = np.random.default_rng(SEED_MASK)
    rad = np.abs(rng.normal(R_HOLE_MEAN, R_HOLE_STD, NHOLES))
    idx = rng.integers(hp.nside2npix(NSIDE_MASK), size=NHOLES)
    holes = set()
    for i in range(NHOLES):
        holes |= set(hp.query_disc(NSIDE_MASK, hp.pix2vec(NSIDE_MASK, idx[i]),
                                   np.radians(rad[i])))
    hole_map = np.ones(hp.nside2npix(NSIDE_MASK))
    hole_map[list(holes)] = 0.
    keep &= hole_map[hp.ang2pix(NSIDE_MASK, theta, phi)].astype(bool)
    # Prepare and plot masked kappa map
    k = hp.smoothing(hp.ud_grade(kappa.astype(np.float64), NSIDE_VIEW), fwhm=np.radians(.10))
    shown = np.full(hp.nside2npix(NSIDE_VIEW), hp.UNSEEN)
    shown[keep] = k[keep]
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("0.72")
    v = np.std(k[keep])
    fig = plt.figure(figsize=(6.4, 6.4))
    _rot = (PATCH_CENTRE[0], PATCH_CENTRE[1], 0.) # Keep like this to avoid more transformations :D
    hp.orthview(shown, rot=_rot, half_sky=True, fig=fig.number, cmap=cmap,
                min=-2.2*v, max=3.6*v, cbar=False, title="", notext=True,
                badcolor="0.72", bgcolor="none")
    hp.graticule(dpar=30., dmer=30., color="0.55", lw=1., alpha=.8)
    # Get and plot rough outline of the patch that we zoom in
    span = PATCH_SPAN / 60.0
    lon0, lat0 = PATCH_CENTRE
    corners = np.array([[lon0-span/2, lat0-span/2],[lon0+span/2, lat0-span/2],
                        [lon0+span/2, lat0+span/2],[lon0-span/2, lat0+span/2]])
    for i in range(4):
        j = (i + 1) % 4
        lon = np.linspace(corners[i, 0], corners[j, 0], 20)
        lat = np.linspace(corners[i, 1], corners[j, 1], 20)
        hp.projplot(np.radians(90. - lat), np.radians(lon), "-", color="#5ee6dc", lw=3)

    out = BASE_PANELS+"footprint.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", transparent=True, pad_inches=0)
    plt.close(fig)
    print("wrote", out, "-- %.0f deg^2" % (keep.sum()*hp.nside2pixarea(NSIDE_VIEW, degrees=True)))

# Prepare bottom left panel showing close-up of full-sky regions with 
# highlighted shape and multiplets
def triangles(kappa, g1, g2):

    # Project full-sky map to patch and get local map of the lensing fields
    xs = int(2*PATCH_SPAN/PATCH_RESO)
    pr = lambda m: np.asarray(hp.gnomview(m, rot=PATCH_CENTRE, reso=PATCH_RESO, xsize=xs,
                                          return_projected_map=True, no_plot=True))
    K, G1, G2 = gaussian_filter(pr(kappa), 1.5), pr(g1), pr(g2)
    at = lambda q, M: M[int(np.clip((q[1] + PATCH_SPAN)/PATCH_RESO, 0, xs - 1)),
                        int(np.clip((q[0] + PATCH_SPAN)/PATCH_RESO, 0, xs - 1))]

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    fig.patch.set_alpha(0.)
    v = np.std(K)
    # Plot kappa as background
    ax.imshow(K, origin="lower", cmap="magma", vmin=-2.2*v, vmax=3.4*v, alpha=.75,
              extent=(-PATCH_SPAN, PATCH_SPAN, -PATCH_SPAN, PATCH_SPAN),
              interpolation="bilinear")
    
    # Plot a few selected galaxies, each with their true interpolated shape
    rng = np.random.default_rng(SEED_TRIANGLES)
    gx, gy = rng.uniform(-.95*PATCH_SPAN, .95*PATCH_SPAN, (2, NGAL_SHOWN))
    for x, y in zip(gx, gy):
        a = .5*np.arctan2(at((x, y), G2), at((x, y), G1))
        L = .030*PATCH_SPAN
        ax.plot([x - L*np.cos(a), x + L*np.cos(a)], [y - L*np.sin(a), y + L*np.sin(a)],
                "-", color="w", lw=.8, zorder=3, solid_capstyle="round")

    # Plot a few selected multiplet configurations where we highlight the vertices
    # We do stratified sampling in subregion of patch with edge=0.1
    _pixsize=1.8*PATCH_SPAN/NTR_SHOWN_ROW
    _pixlowers = -0.9*PATCH_SPAN + _pixsize*np.arange(NTR_SHOWN_ROW)
    _ntrshown = NTR_SHOWN_ROW*NTR_SHOWN_ROW
    shiftsx, shiftsy = rng.uniform(0, _pixsize, (2, NTR_SHOWN_ROW, NTR_SHOWN_ROW))
    x0s, y0s = np.meshgrid(_pixlowers,_pixlowers)
    x0s = (x0s+shiftsx).flatten()
    y0s = (y0s+shiftsy).flatten()
    thetai = rng.uniform(.1*PATCH_SPAN, .3*PATCH_SPAN, 3*_ntrshown)
    phi0i = rng.uniform(0., 2.*np.pi, 3*_ntrshown)
    _olo, _ohi = (2, 5) if SHOW_ALL_ORDERS else (3, 4)
    orders = rng.integers(_olo,_ohi,_ntrshown) # Just do third-order
    for (order,x0,y0,t1,t2,t3,ph1,ph2,ph3) in zip(orders,x0s, y0s, 
         thetai[:_ntrshown], thetai[_ntrshown:2*_ntrshown], thetai[2*_ntrshown:], 
         phi0i[:_ntrshown], phi0i[_ntrshown:2*_ntrshown], phi0i[2*_ntrshown:]):
        x1 = x0+t1*np.cos(ph1); y1=y0+t1*np.sin(ph1)
        x2 = x0+t2*np.cos(ph2); y2=y0+t2*np.sin(ph2)
        x3 = x0+t3*np.cos(ph3); y3=y0+t3*np.sin(ph3)
        a0 = .5*np.arctan2(at((x0, y0), G2), at((x0, y0), G1))
        a1 = .5*np.arctan2(at((x1, y1), G2), at((x1, y1), G1))
        a2 = .5*np.arctan2(at((x2, y2), G2), at((x2, y2), G1))
        a3 = .5*np.arctan2(at((x3, y3), G2), at((x3, y3), G1))
        L = .030*PATCH_SPAN
        qx0 = [x0-L*np.cos(a0), x0+L*np.cos(a0)]; qy0 = [y0-L*np.sin(a0), y0+L*np.sin(a0)]
        qx1 = [x1-L*np.cos(a1), x1+L*np.cos(a1)]; qy1 = [y1-L*np.sin(a1), y1+L*np.sin(a1)]
        qx2 = [x2-L*np.cos(a2), x2+L*np.cos(a2)]; qy2 = [y2-L*np.sin(a2), y2+L*np.sin(a2)]
        qx3 = [x3-L*np.cos(a3), x3+L*np.cos(a3)]; qy3 = [y3-L*np.sin(a3), y3+L*np.sin(a3)]
        ax.plot(qx0, qy0, "-", color="#5ee6dc", lw=2, zorder=8, solid_capstyle="round")
        ax.plot(qx1, qy1, "-", color="#5ee6dc", lw=2, zorder=8, solid_capstyle="round")
        ax.plot([x0,x1], [y0,y1], color="#5ee6dc", lw=1.5, zorder=8)
        if order>2:
            ax.plot([x0,x2], [y0,y2], color="#5ee6dc", lw=1.5, zorder=8)
            ax.plot(qx2, qy2, "-", color="#5ee6dc", lw=2, zorder=8, solid_capstyle="round")
        if order>3:
            ax.plot([x0,x3], [y0,y3], color="#5ee6dc", lw=1.5, zorder=8)
            ax.plot(qx3, qy3, "-", color="#5ee6dc", lw=2, zorder=8, solid_capstyle="round")

    ax.set(xlim=(-PATCH_SPAN, PATCH_SPAN), ylim=(-PATCH_SPAN, PATCH_SPAN),
           xticks=[], yticks=[])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_color("0.3")
        s.set_linewidth(1.2)
    out = BASE_PANELS+"triangles.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("wrote", out, "-- %d triangles at (%g', %g'), %d galaxies shown"
          % (_ntrshown, THETA1, THETA2, NGAL_SHOWN))

def perform_measurement(kappa, g1, g2):
    cat = masked_catalog(kappa, g1, g2)
    cat.topatches(npatches=NPATCHES, patchextend_deg=MAX_SEP/60.)
    ggg = orpheus.GGGCorrelation(n_cfs=4, min_sep=MIN_SEP, max_sep=MAX_SEP,
                                binsize=BINSIZE, nbinsphi=NBINSPHI, nmaxs=NMAXS,
                                rmin_pixsize=RMIN_PIXSIZE, sep_units="arcmin",
                                verbosity=1, nthreads=NTHREADS)
    ggg.autoset_tree(cat)
    ggg.process(cat, keep_patchres=True)
    ggg.multipoles2npcf()
    np.savez_compressed(PATH_MEAS, npcf=np.asarray(ggg.npcf), phi=np.asarray(ggg.phi),
                        bin_centers=np.asarray(ggg.bin_centers),
                        multipoles=np.asarray(ggg.npcf_multipoles),
                        apradii=APRADII,
                        map3=np.asarray(ggg.computeMap3(radii=APRADII)))

def reduce_measdata():
    full = np.load(PATH_MEAS)
    r = full["bin_centers"].ravel()
    ia = int(np.argmin(np.abs(r - THETA1)))
    ib = int(np.argmin(np.abs(r - THETA2)))
    np.savez_compressed(PATH_SMALLMEAS,
                        phi=full["phi"], gamma0=full["npcf"][0, 0, ia, ib],
                        theta1_target=THETA1, theta2_target=THETA2,
                        theta1=r[ia], theta2=r[ib],
                        apradii=full["apradii"], map3=full["map3"])
    print("wrote t17_ggg_forfig.npz  (theta1=%.2f', theta2=%.2f')" % (r[ia], r[ib]))

    

if __name__ == "__main__":
    print("Loading data")
    kappa, g1, g2 = read_t17(PATH_T17)
    print("Generating footprint panel")
    footprint(kappa)
    print("Generating triangle panel ")
    triangles(kappa, g1, g2)
    if DO_MEAS:
        print("Perform measurement")
        perform_measurement(kappa, g1, g2)
        print("Extracting data")
        reduce_measdata()
    else:
        try:
            reduced = np.load(PATH_SMALLMEAS)
            if reduced['theta1_target']!=THETA1 or reduced['theta2_target']!=THETA2:
                try:
                    full = np.load(PATH_MEAS)
                    reduce_measdata()
                    print("Warning: No fitting reduced measurment file found. Updated reduction.")
                except:
                    raise FileNotFoundError("Neither full measurement file or a fitting reduction of it found.")
        except:
            try:
                full = np.load(PATH_MEAS)
                reduce_measdata()
                print("Warning: No fitting reduced measurement file found. Updated reduction.")
            except:
                raise FileNotFoundError("Neither full measurement file or a reduction of it found.")