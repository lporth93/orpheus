# Here we run the scaling tests and create the scaling figure for the README
# * To not have the scaling influenced by edge-effects we perform all measurements on a
#   padded catalog where the buffer region extends by at least the largest separation.
#   In orpheus we do this setting isinner correspondingly while for treecorr we pass 
#   cat1=inner, cat2=full and set ordered to True to mirror orpheus' setup.
# * Note that there are two effects that can make the scaling deviate from the expected theory
#   1) Between machines different setups exhaust the L3 cache at different settings
#   2) We always use autoset_tree before processing which means that for each setup the 
#      chosen tree might differ a bit which can be more or less efficient
# * You can create such a figure measurement by 
#   1) Adjusting the parameters to your preferred setup
#   2) Run python benchmarks/scaling.py run to perform the measurement
#   3) Run python benchmarks/scaling.py plot to create the figure

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import time

import numpy as np

import orpheus

###################
# PARAMETER SETUP #
###################

# Paths for data/figure; whether to plot the scaling wrt wall or cpu time
# Both carry the hostname: timings are only comparable within a machine, and a sweep
# started on a second machine must not read, or write, the first one's cache.
SAVEPATH_DATA = "benchmarks/scaling_%s.npz" % platform.node()
SAVEPATH_FIG = "benchmarks/orpheus_scaling_%s.png" % platform.node()
METRIC = "cpu" # "wall" or "cpu"

# Size of full box and inner region; which classes of orpheus estimators to run
BOX = 1024.
INNER = 512.
METHODS = ["Discrete", "Tree", "DoubleTree"]

# Shared pars for measurement setup
MIN_SEP = .25
NMAX = 10
NBINSPHI = 50
BINSIZE = .15
TREE_MAXCELLSIZE = 4.
NTHREADS = 48

# Pars for the plot for scaling with theta_max
SEP_NBAR = 10.
SEP_MAXSEPS = [2.**n for n in range(9)]
SEP_CAP = {"Discrete": 64., "Tree": 256., "DoubleTree": 256.}
SEP_CAP_TC = {"tc-real-def": 16., "tc-real-exact":  8.}

# Pars for the plot for scaling with density
DEN_MAXSEP = 128.
DEN_NBARS = [.125, .25, .5, 1., 2., 4., 8., 16., 32.]
DEN_CAP = {"Discrete": 4., "Tree": 16., "DoubleTree": 32.}

# Pars for the plot for scaling with nthreads.
THR_NBAR = 1.
THR_NBARS = {"Discrete": THR_NBAR, "Tree": 2., "DoubleTree": 16.}
THR_MAXSEP = 128.
THR_ALLOWEDNTHREADS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


#############################
# FIXED/INFERRED PARAMETERS #
#############################

# Get the actual number of physical cores on the machine and limit cap the allowed 
# threads to that. If the number of physical cores sits between use this as last point.
_SMT = int(re.search(r"^Thread\(s\) per core:\s*(\d+)$",
                     subprocess.run(["lscpu"], capture_output=True, text=True).stdout,
                     re.M).group(1))
_NCORES = len(os.sched_getaffinity(0))//_SMT
THR_THREADS = sorted({t for t in THR_ALLOWEDNTHREADS if t <= _NCORES} | {_NCORES})

# Treecorr setup used to measure real-space equivalents
TC_B = .1 # Standard choice of binslop: binsize*binslop<=TC_B
TC_VARIANTS = {"tc-real-def": ("triangle", None),
               "tc-real-exact": ("triangle", 0.)}
TC_STYLE = {"tc-real-exact": ("x", 7., ".45", "real space: triplet sum", True),
            "tc-real-def": ("*", 9.5, "k", "real space: Tree", False)}

# Theoretical scaling of various estimators
THEORY_SLOPE_MULTIPOLES = {"Discrete": 2.}
THEORY_SLOPE_REALSPACE = {"sep": 4., "density": 3.}

# Plot setup
TITLE_PAD = 28
LEGEND_SIZE = 8.
COLORS = {"Discrete": "#4477aa", "Tree": "#ee6677", "DoubleTree": "#228833"}
MARKERS = {"Discrete": "o", "Tree": "^", "DoubleTree": "s"}

METRIC_COL = 1 if METRIC == "cpu" else 0


############################
# HELPERS FOR MACHINE INFO #
############################

# Crude estimate of the currently available memory bandwidth of this machine
def triad_bandwidth(nbytes=1 << 28):
    n = nbytes//8
    a, b, c = np.ones(n), np.ones(n), np.ones(n)
    best = np.inf
    for _ in range(3):
        t0 = time.perf_counter()
        a[:] = b + 3.*c
        best = min(best, time.perf_counter() - t0)
    return 3.*n*8/best/1e9

# Get some basic info on the machine that we are running on
def machine_info():
    info = {"host": platform.node()}
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        for key, field in (("Model name", "cpu"), ("Socket\\(s\\)", "sockets"),
                           ("Core\\(s\\) per socket", "cores_per_socket"),
                           ("Thread\\(s\\) per core", "threads_per_core"),
                           ("L3 cache", "l3"), ("NUMA node\\(s\\)", "numa_nodes")):
            m = re.search(r"^%s:\s*(.+)$" % key, out, re.M)
            if m:
                info[field] = m.group(1).strip()
    except OSError:
        pass
    info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE")*os.sysconf("SC_PHYS_PAGES")/2**30)
    info["triad_gbs"] = round(triad_bandwidth(), 1)
    return info


###############
# MEASUREMENT #
###############

# Create the catalog
def catalog(nbar, box=BOX, inner=INNER, seed=42):
    ngal = int(nbar*box**2)
    rng = np.random.default_rng(seed)
    pos1, pos2 = rng.uniform(0., box, ngal), rng.uniform(0., box, ngal)
    lo, hi = (box-inner)/2., (box+inner)/2.
    isinner = ((pos1>lo) & (pos1<hi) & (pos2>lo) & (pos2<hi)).astype(np.float64)
    return orpheus.SpinTracerCatalog(spin=2, pos1=pos1, pos2=pos2, isinner=isinner,
                                     tracer_1=rng.normal(0., .3, ngal),
                                     tracer_2=rng.normal(0., .3, ngal))

# Create the orpheus GGG correlator
def build(method, max_sep, nthreads):
    return orpheus.GGGCorrelation(n_cfs=4, min_sep=MIN_SEP, max_sep=max_sep, binsize=BINSIZE,
                                  nbinsphi=NBINSPHI, nmaxs=NMAX, method=method,
                                  nthreads=nthreads, tree_maxcellsize=TREE_MAXCELLSIZE)

# Number of neighbours and effective tree cells inside max_sep for current setup
def counts(inst, nbar):
    area = np.pi*(inst.tree_redges[1:]**2 - inst.tree_redges[:-1]**2)
    percell = np.divide(1., inst.tree_resos**2, where=inst.tree_resos > 0,
                        out=np.full_like(inst.tree_resos, np.inf, dtype=float))
    return nbar*area.sum(), np.sum(area*np.minimum(nbar, percell))

# Time an orpheus .process call
def timeit(cat, nbar, method, max_sep, nthreads):
    inst = build(method, max_sep, nthreads)
    inst.autoset_tree(cat)
    w0, c0 = time.perf_counter(), time.process_time()
    inst.process(cat)
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    nnb, nc = counts(inst, nbar)
    return wall, cpu, nnb, nc

# Time a treecorr .process call
def treecorr_timeit(cat, variant, max_sep, nthreads):
    import treecorr
    # Define binning of GGG
    algo, opening = TC_VARIANTS[variant]
    if opening is None:
        slop = {} # Default
    elif opening == "orpheus":
        slop = dict(bin_slop=1./build("DoubleTree", max_sep, 1).rmin_pixsize/BINSIZE) # Orpheus-matched setup
    else:
        slop = dict(bin_slop=opening, angle_slop=opening) # Custom setup
    ggg = treecorr.GGGCorrelation(min_sep=MIN_SEP, max_sep=max_sep, bin_size=BINSIZE,
                                      nphi_bins=NBINSPHI, num_threads=nthreads, verbose=0, **slop)
    assert opening is not None or abs(ggg.bin_slop*ggg.bin_size - TC_B) < 1e-9, "treecorr b moved" # Make sure default stays default

    # Create catalogs and process
    sel_inner = cat.isinner>.5
    tccat_inner = treecorr.Catalog(x=cat.pos1[sel_inner], y=cat.pos2[sel_inner],
                                   g1=cat.tracer_1[sel_inner], g2=cat.tracer_2[sel_inner])
    tccat_full = treecorr.Catalog(x=cat.pos1, y=cat.pos2, g1=cat.tracer_1, g2=cat.tracer_2)
    w0, c0 = time.perf_counter(), time.process_time()
    ggg.process(cat1=tccat_inner, cat2=tccat_full, num_threads=nthreads, algo=algo,
                ordered=True)

    return (time.perf_counter() - w0, time.process_time() - c0, float(ggg.ntri.sum()), 0.)

# Create a hash corresponding to the measurement setup. As this does not contain the point
# new points can easily be added under the same hash
def gridkey():
    spec = (MIN_SEP, BINSIZE, TREE_MAXCELLSIZE, NMAX, NBINSPHI, BOX, INNER, SEP_NBAR,
            DEN_MAXSEP, THR_NBAR, THR_MAXSEP)
    return hashlib.md5(repr(spec).encode()).hexdigest()[:8]

# Print key identifying measurement together with more specific parameters
def cachekey(host, leg, method, param, nthreads):
    return "%s|%s|%s|%s|%g|%d" % (host, gridkey(), leg, method, param, nthreads)

# Run the measurement
def run(legs=("sep", "density", "threads")):

    host = platform.node()

    # Load previous results.
    if os.path.exists(SAVEPATH_DATA):
        store = dict(np.load(SAVEPATH_DATA))
        results = json.loads(str(store["results"])) if "results" in store else {}
        machines = json.loads(str(store["machines"])) if "machines" in store else {}
    else:
        results, machines = {}, {}

    # Record machine/software information.
    machines.setdefault(host, machine_info())
    try:
        import treecorr
        machines[host]["treecorr"] = treecorr.__version__
        has_treecorr = True
    except ImportError:
        has_treecorr = False
    print("machine", json.dumps(machines[host]), flush=True)
    print("grid     %s: min_sep=%g binsize=%g tree_maxcellsize=%g box=%g inner=%g "
          "nthreads=%d" % (gridkey(), MIN_SEP, BINSIZE, TREE_MAXCELLSIZE, BOX, INNER,
                           NTHREADS), flush=True)

    def save():
        np.savez(SAVEPATH_DATA, results=json.dumps(results), machines=json.dumps(machines))

    def pending(leg, method, param, nthreads):
        return cachekey(host, leg, method, param, nthreads) not in results

    def measure(leg, method, param, nthreads, measure_func):
        k = cachekey(host, leg, method, param, nthreads)
        # Don't remeasure if result already computed
        if k in results:
            print("  cached   %-8s %-11s %-8g threads=%-3d  wall %9.2f s" % (
                leg[:8], method, param, nthreads, results[k][0]), flush=True)
            return
        # Add result to measurement  if not yet in data
        t0 = time.perf_counter()
        wall, cpu, nnb, nc = measure_func()
        results[k] = [wall, cpu, nnb, nc]
        save()
        print("  measured %-8s %-11s %-8g threads=%-3d  wall %9.2f s  cpu %10.1f s"
              "  (Nnb %.3g, Nc %.3g, took %.0f s)" % (
                  leg[:8], method, param, nthreads, wall, cpu, nnb, nc,
                  time.perf_counter() - t0), flush=True)

    # Run the scaling with separations. Make sure to only include not-yet measured points
    # and adhere to the bonds on which each method is computed
    if "sep" in legs:
        print("== separation leg: nbar=%g, max_sep %g..%g arcmin" % (
            SEP_NBAR, SEP_MAXSEPS[0], SEP_MAXSEPS[-1]), flush=True)
        tcs = [v for v in SEP_CAP_TC] if has_treecorr else []
        caps = dict(SEP_CAP, **{v: SEP_CAP_TC[v] for v in tcs})
        todo = [(s, m) for s in SEP_MAXSEPS for m in METHODS + tcs
                if s <= caps[m] and pending("sep", m, s, NTHREADS)]
        cat = catalog(SEP_NBAR) if todo else None
        for max_sep in SEP_MAXSEPS:
            for method in METHODS:
                if max_sep <= SEP_CAP[method]:
                    measure("sep", method, max_sep, NTHREADS,
                            lambda: timeit(cat, SEP_NBAR, method, max_sep, NTHREADS))
            for variant in tcs:
                if max_sep <= caps[variant]:
                    measure("sep", variant, max_sep, NTHREADS,
                            lambda: treecorr_timeit(cat, variant, max_sep, NTHREADS))
        cat = None
    # Run the scaling with number density. 
    if "density" in legs:
        print("== density leg: max_sep=%g, nbar %g..%g" % (
            DEN_MAXSEP, DEN_NBARS[0], DEN_NBARS[-1]), flush=True)
        for nbar in DEN_NBARS:
            methods_todo = [m for m in METHODS if nbar <= DEN_CAP[m]]
            if any(pending("density", m, nbar, NTHREADS) for m in methods_todo):
                cat = catalog(nbar)
            for method in methods_todo:
                measure("density", method, nbar, NTHREADS,
                        lambda: timeit(cat, nbar, method, DEN_MAXSEP, NTHREADS))
            cat = None
    # Run the scaling with number of threads
    if "threads" in legs:
        print("== thread leg: max_sep=%g, threads %s, nbar %s" % (
            THR_MAXSEP, THR_THREADS,
            ", ".join("%s=%g" % (m, THR_NBARS[m]) for m in METHODS)), flush=True)
        for method in METHODS:
            nbar = THR_NBARS[method]
            leg = "threads%g"%nbar
            if any(pending(leg, method, t, t) for t in THR_THREADS):
                cat = catalog(nbar)
                # Make sure to always built the spatial hash on the same scale for this test
                cat.build_spatialhash(dpix=np.sqrt(10./nbar))
            for nthreads in THR_THREADS:
                measure(leg, method, nthreads, nthreads,
                        lambda: timeit(cat, nbar, method, THR_MAXSEP, nthreads))
            cat = None

    save()
    print("wrote", SAVEPATH_DATA, "(legs: %s)" % ", ".join(legs))


#############
# PLOTTING  #
#############

# One method's curve over xs, nan where it was not measured. Column 0/1 --> wall/CPU time
def series(results, host, leg, xs, nthreads, method, column=0):
    out = np.full(len(xs), np.nan) # Sets to nan for non measured points
    for i, x in enumerate(xs):
        k = cachekey(host, leg, method, x, nthreads if nthreads else x)
        if k in results:
            out[i] = results[k][column]
    return out

# Prints the proportionality of a symbol. A cost model gives a whole number, a fit a decimal
def _exponent(sym, slope, exact):
    return r"$\propto %s^{%s}$" % (sym, ("%d" if exact else "%.1f") % slope)

# Fit a few points to a powerlaw
def _fit(x, y):
    ok = np.isfinite(y)
    return np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0] if ok.sum() >= 2 else np.nan

# Extend a capped curve as a power law. Without a slope it is fitted to the measured range,
# with one it is that exponent anchored on the last point measured
def extend(x, y, slope=None):
    ok = np.isfinite(y)
    if ok.all() or ok.sum() < 2:
        return y.copy()
    if slope is None:
        slope, const = np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)
    else:
        last = int(np.max(np.flatnonzero(ok)))
        const = np.log(y[last]) - slope*np.log(x[last])
    full = y.copy()
    full[~ok] = np.exp(const)*x[~ok]**slope
    return full

# Load the measurements file
def load():
    if not os.path.exists(SAVEPATH_DATA):
        raise SystemExit("%s does not exist; run the sweep first, from the repository "
                         "root." % SAVEPATH_DATA)
    dat = np.load(SAVEPATH_DATA)
    results = json.loads(str(dat["results"]))
    machines = json.loads(str(dat["machines"]))
    host = platform.node()
    prefix = "%s|%s|" % (host, gridkey())
    if not any(k.startswith(prefix) for k in results):
        have = sorted({tuple(k.split("|")[:2]) for k in results})
        raise SystemExit("no measurements for %s under grid %s; file holds %s."%(
            host, gridkey(), have))
    return results, machines.get(host, {"host": host}), host


# Plot the three orpheus schemes on one panel. For capped ones extend with power law
# Returns the completed curves, their on-curve labels and the legend handles
def method_curves(ax, results, host, leg, xs, sym, name_slopes):
    full, notes, entries = {}, [], []
    for method in METHODS:
        y = series(results, host, leg, xs, NTHREADS, method, column=METRIC_COL)
        if not np.isfinite(y).any():
            continue
        slope = THEORY_SLOPE_MULTIPOLES.get(method)
        full[method] = extend(xs, y, slope)
        exact = slope is not None
        note = _exponent(sym, slope if exact else _fit(xs, full[method]), exact)
        line, = ax.loglog(xs, y, marker=MARKERS[method], color=COLORS[method])
        entries.append((line, "orpheus: %s  %s" % (method, note) if exact or name_slopes
                        else "orpheus: %s" % method))
        notes.append((COLORS[method], full[method], "%s  %s" % (method, note),
                      "below" if method == "DoubleTree" else "above"))
        capped = ~np.isfinite(y)
        if capped.any():
            # from the last measured point on, so the continuation joins the curve
            first = max(int(np.argmax(capped)) - 1, 0)
            seg = np.full(len(xs), np.nan)
            seg[first:] = full[method][first:]
            ax.loglog(xs, seg, ls=":", lw=1.2, color=COLORS[method])
    return full, notes, entries


# Plot panel probing scaling against the largest probed separation at fixed density
def sep_panel(ax, results, host):
    xs, sym = np.array(SEP_MAXSEPS), r"\theta"

    # Get curves for orpheus and realspace with their extensions
    full, _, orph_meas = method_curves(ax, results, host, "sep", xs, sym, False)
    realspace_meas = []
    for variant, (marker, ms, color, label, carry) in TC_STYLE.items():
        m = series(results, host, "sep", xs, NTHREADS, variant, column=METRIC_COL)
        if not np.isfinite(m).any():
            continue
        if carry:
            last = int(np.max(np.flatnonzero(np.isfinite(m))))
            cont = np.full(len(xs), np.nan)
            cont[last:] = m[last]*(xs[last:]/xs[last])**THEORY_SLOPE_REALSPACE["sep"]
            ax.loglog(xs, cont, ls="--", lw=1., color=color)
            label = "%s  %s" % (label, _exponent(sym, THEORY_SLOPE_REALSPACE["sep"], True))
        line, = ax.loglog(xs, m, ls="-", lw=1., marker=marker, ms=ms, mew=1.6, color=color)
        realspace_meas.append((line, label))

    # Cut panel to focus on orpheus measurements
    ax.set_ylim(top=10.*np.nanmax(full["Discrete"]))
    ax.set_xlabel(r"$\theta_{\rm max}$ [arcmin]")
    ax.set_title(r"Larger separations at $\bar{n} = %g\,$arcmin$^{-2}$" % SEP_NBAR,
                 fontsize=11, pad=TITLE_PAD)
    entries = realspace_meas + orph_meas
    ax.legend([h for h, _ in entries], [t for _, t in entries], fontsize=LEGEND_SIZE,
              frameon=False, labelspacing=.4, handlelength=1.8, loc="upper left")
    ax.set_xticks(xs, ["%g" % v for v in xs])
    ax.set_xticks([], minor=True)


# Plot panel probing scaling against tracer density at constant thetamax
def density_panel(ax, results, host, anchor):
    xs, sym = np.array(DEN_NBARS), r"\bar{n}"

    # Get orpheus points and optionally estimate of realspace by using measurements from sep_panel  
    full, notes, _ = method_curves(ax, results, host, "density", xs, sym, True)
    if anchor is not None:
        y = anchor*(xs/SEP_NBAR)**THEORY_SLOPE_REALSPACE["density"]
        note = _exponent(sym, THEORY_SLOPE_REALSPACE["density"], True)
        ax.loglog(xs, y, ls="--", lw=1., color=".45", alpha=.8)
        notes.append((".45", y, "triplet sum  %s" % note, "middle"))

    ax.set_ylim(top=10.*np.nanmax(full["Discrete"]))
    ax.set_xlabel(r"$\bar{n}$ [arcmin$^{-2}$]")
    ax.set_title(r"Deeper survey at $\theta_{\rm max} = %g^{\prime}$" % DEN_MAXSEP,
                 fontsize=11, pad=TITLE_PAD)
    ax.set_xticks(xs, ["%g" % v for v in xs])
    ax.set_xticks([], minor=True)
    return xs, notes


# Plot panel probing strong scaling
def thread_panel(ax, results, host):
    legs = {"threads%g"%THR_NBARS[m] for m in METHODS}
    threads = np.array(sorted({float(k.split("|")[4]) for k in results
                               if k.split("|")[2] in legs}), dtype=float)
    if not len(threads):
        return
    for method in METHODS:
        # Get curves for wall/cpu times
        y = series(results, host, "threads%g"%THR_NBARS[method], threads, None, method)
        c = series(results, host, "threads%g"%THR_NBARS[method], threads, None, method, column=1)
        if not np.isfinite(y).any():
            continue
        # Get at largest thread count speedup for wall and mean occupancy for CPU
        speedup = y[0]/y[-1]
        occupied = threads[-1]/threads[0]*c[0]/c[-1]
        # Named on the first row only; the rows below it inherit the reading.
        tags = ("Wall: ", "CPU: ") if method == METHODS[0] else ("", "")
        ax.plot(threads, y[0]/y, marker=MARKERS[method], color=COLORS[method],
                label=r"orpheus: %s at $\bar{n} = %g$  (%s$%.0f\times$ / %s$%.0f\times$)" % (
                    method, THR_NBARS[method], tags[0], speedup, tags[1], occupied))
    ax.plot(threads, threads/threads[0], ls=":", color="0.6", lw=1., label="ideal")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("threads")
    ax.set_ylabel(r"speedup over %d thread%s" % (threads[0], "" if threads[0] == 1 else "s"))
    ax.set_title(r"Strong scaling at $\theta_{\rm max} = %g^{\prime}$" % THR_MAXSEP,
                 fontsize=11, pad=TITLE_PAD)
    ax.legend(fontsize=LEGEND_SIZE, frameon=False, loc="upper left")
    ax.set_xticks(threads, ["%g" % t for t in threads])
    ax.set_xticks([], minor=True)
    yt = 2.**np.arange(0., np.ceil(np.log2(threads[-1])) + 1.)
    ax.set_yticks(yt, ["%g" % v for v in yt])
    ax.set_yticks([], minor=True)


# Catalog size along the top of the density panel: number density is what the estimators see,
# but galaxies in the field is what a reader has a feel for
def galaxy_axis(ax):
    nbars = [.125, .5, 2., 8., 32.]
    ngals = [n*INNER**2 for n in nbars]
    str_ngals = ["%.0fk" % (n/1e3) if n < 1e6 else "%.1fM" % (n/1e6) for n in ngals]
    top = ax.twiny()
    top.set_xscale("log")
    top.set_xlim(ax.get_xlim())
    # ticks sit at the densities the panel below is drawn on; only the labels are counts
    top.set_xticks(nbars, str_ngals)
    top.set_xticks([], minor=True)
    top.set_xlabel(r"galaxies in the field", fontsize=9, labelpad=4)
    top.tick_params(which="both", direction="in", labelsize=8)
    ax.tick_params(top=False)
    ax.set_title(ax.get_title(), fontsize=11, pad=TITLE_PAD)


# Name each curve on the curve rather than in a legend, turned to its slope. Reads display
# coordinates, so it wants a settled layout --> call it after tight_layout
def label_curves(ax, xs, notes):
    xlo, xhi = np.log(ax.get_xlim())
    ybot, ytop = np.log(ax.get_ylim())
    for color, y, note, place in notes:
        ok = np.isfinite(y)
        if not ok.any():
            continue
        lx, ly = np.log(xs[ok]), np.log(y[ok])
        if place == "middle":
            # the reference leaves the top of the panel, so it is labelled halfway
            # along the stretch of it that is actually drawn
            g = np.linspace(max(xlo, lx[0]), min(xhi, lx[-1]), 200)
            on = np.flatnonzero((np.interp(g, lx, ly) < ytop)
                                & (np.interp(g, lx, ly) > ybot))
            if not len(on):
                continue
            at, ha = .5*(g[on[0]] + g[on[-1]]), "center"
        else:
            # the rest end their labels at the same place, four fifths of the way
            # across, so they line up down the panel
            at, ha = np.clip(xlo + .8*(xhi - xlo), lx[0], lx[-1]), "right"
        # the slope where it sits, in display units, so the text lies along the curve
        d = .01*(xhi - xlo)
        (x0, y0), (x1, y1) = ax.transData.transform(
            [(np.exp(at - d), np.exp(np.interp(at - d, lx, ly))),
             (np.exp(at + d), np.exp(np.interp(at + d, lx, ly)))])
        dy, va = (-5, "top") if place == "below" else (5, "bottom")
        ax.annotate(note, (np.exp(at), np.exp(np.interp(at, lx, ly))),
                    textcoords="offset points", xytext=(0, dy), ha=ha, va=va,
                    rotation=np.degrees(np.arctan2(y1 - y0, x1 - x0)),
                    rotation_mode="anchor", fontsize=LEGEND_SIZE, color=color)


# Produce the figure
def plot():
    from matplotlib import pyplot as plt

    # Computer Modern instead of LaTeX so that figure renders also when no Tex is installed
    plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"],
                         "mathtext.fontset": "cm", "axes.unicode_minus": False,
                         "axes.formatter.use_mathtext": True})

    results, _, host = load()
    fig, axes = plt.subplots(1, 3, figsize=(15., 4.6))

    # Generate anchor for real-space density panel using measurements from sep panel
    tc = series(results, host, "sep", np.array(SEP_MAXSEPS), NTHREADS, "tc-real-exact",
                column=METRIC_COL)
    measured = np.flatnonzero(np.isfinite(tc))
    anchor = (tc[measured[-1]]*(DEN_MAXSEP/SEP_MAXSEPS[measured[-1]])
              **THEORY_SLOPE_REALSPACE["sep"] if len(measured) else None)

    # Plot the three panels
    sep_panel(axes[0], results, host)
    den_xs, den_notes = density_panel(axes[1], results, host, anchor)
    thread_panel(axes[2], results, host)
    # Axis cosmetics
    axes[0].set_ylabel("%s time [s]" % ("CPU" if METRIC == "cpu" else "wall"))
    for ax in axes:
        ax.tick_params(which="both", direction="in", top=True, right=True)
    galaxy_axis(axes[1])

    fig.text(.5, .05, "Shear three-point correlation function in a field of about %.0f deg$^2$. "
             "Shared parameters: $\\theta_{\\rm min} = %g^{\\prime}$, binsize $= %g$, "
             "tree cells capped at $%g^{\\prime}$."
             % ((INNER/60.)**2, MIN_SEP, BINSIZE, TREE_MAXCELLSIZE),
             ha="center", fontsize=8.5, color=".3")
    fig.tight_layout(rect=(0., .09, 1., 1.))
    label_curves(axes[1], den_xs, den_notes) # Put after tight_layout as this reads display coords

    fig.savefig(SAVEPATH_FIG, dpi=150)
    print("wrote", SAVEPATH_FIG)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["run", "plot"])
    p.add_argument("--legs", default="sep,density,threads",
                   help="Which legs to sweep. Comma-separated subset of sep,density,threads")
    args = p.parse_args()
    run(tuple(args.legs.split(","))) if args.mode == "run" else plot()
