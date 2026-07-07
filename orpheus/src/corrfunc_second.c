#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <complex.h>

#include "utils.h"
#include "spatialhash.h"
#include "assign.h"
#include "corrfunc_second.h"
#include "healpix_utils.h"
#include "multires_structs.h"

#define mymin(x,y) ((x) <= (y)) ? (x) : (y)
#define mymax(x,y) ((x) >= (y)) ? (x) : (y)
#define M_PI      3.14159265358979323846
#define INV_2PI   0.15915494309189534561

////////////////////////////////////////////////
/// SECOND-ORDER SHEAR CORRELATION FUNCTIONS ///
////////////////////////////////////////////////

////////////////////////////////////////////////
/// SECOND-ORDER (NN) PAIR COUNTS             ///
////////////////////////////////////////////////

// Shared radial bin lookup + accumulation: dist -> log-bin via the linear helper
// array, then scatter into the per-thread tomographic count arrays. Geometry-
// and metric-agnostic; both _nn_flat and _nn_spherical call this.
static inline void nn_bin_accumulate(
    double dist, double w1, double w2, int z1, int z2,
    double rmin, double *binedges, int *linarr_bins, double dbin_lin_inv,
    int nbinsz, int nbinsr, int thread,
    int *tmpnpair, double *tmpwcount, double *tmpwnorm){
    int tmplogbin = (int) ((dist-rmin)*dbin_lin_inv);
    int rbin = linarr_bins[tmplogbin];
    rbin += (dist > binedges[rbin+1]) ? 1 : 0;
    int ind = thread*nbinsz*nbinsz*nbinsr + z1*nbinsz*nbinsr + z2*nbinsr + rbin;
    tmpnpair[ind] += 1;
    tmpwcount[ind] += w1*w2*dist;
    tmpwnorm[ind] += w1*w2;
}

// Build the shared, region/thread-independent radial-binning helper arrays
// once. Previously recomputed inside every OpenMP thread on the flat path;
// hoisted here since none of it depends on thread, region, or galaxy data --
// only on (rmin, rmax, nbinsr, reso_redges). Caller frees the three outputs.
static void build_radial_helpers(
    const TreeResoParams *tree, const BinningParams *bin,
    double **out_binedges, int **out_linarr_bins, int **out_reso_rindedges,
    double *out_drbin, double *out_dbin_lin_inv){

    int nresos = tree->nresos, nbinsr = bin->nbinsr;
    double rmin = bin->rmin, rmax = bin->rmax;
    double drbin = (log(rmax)-log(rmin))/nbinsr;

    int *reso_rindedges = calloc(nresos+1, sizeof(int));
    double *binedges = calloc(nbinsr+2, sizeof(double));
    int tmpreso = 0;
    double thisredge = 0, tmpr = rmin;
    for (int elr=0; elr<nbinsr; elr++){
        binedges[elr] = tmpr;
        tmpr *= exp(drbin);
        thisredge = tree->reso_redges[mymin(nresos,tmpreso+1)];
        if (thisredge < tmpr){
            reso_rindedges[mymin(nresos,tmpreso+1)] = elr;
            if ((tmpr-thisredge) < (thisredge - (tmpr/exp(drbin)))){ reso_rindedges[mymin(nresos,tmpreso+1)] += 1; }
            tmpreso += 1;
        }
    }
    binedges[nbinsr] = tmpr;
    binedges[nbinsr+1] = tmpr*exp(drbin);
    reso_rindedges[nresos] = nbinsr;

    double dbin_lin = 0.9*rmin*(exp(drbin)-1);
    double dbin_lin_inv = 1./dbin_lin;
    int nbinsr_lin = (int) ceil(binedges[nbinsr]/dbin_lin);
    int *linarr_bins = calloc(nbinsr_lin+1, sizeof(int));
    int tmplogbin = 0;
    tmpr = rmin;
    for (int elr=0; elr<=nbinsr_lin; elr++){
        if (tmpr > binedges[tmplogbin+1]){ tmplogbin += 1; }
        linarr_bins[elr] = tmplogbin;
        tmpr += dbin_lin;
        if (tmpr >= binedges[nbinsr]){ break; }
    }

    *out_binedges = binedges;
    *out_linarr_bins = linarr_bins;
    *out_reso_rindedges = reso_rindedges;
    *out_drbin = drbin;
    *out_dbin_lin_inv = dbin_lin_inv;
}

// Build the per-reso shift arrays for the flat grid hash once. Previously
// recomputed inside every OpenMP thread; depends only on ngal_resos and the
// hash grid size, never on region or galaxy data.
static void build_flat_rshifts(
    const MultiresoCatalog *cat, const NavHash *nav,
    int **out_rshift_index_matcher, int **out_rshift_pixs_galind_bounds,
    int **out_rshift_pix_gals){

    int nresos = cat->nresos;
    int npix_hash = nav->pix1_n * nav->pix2_n;
    int *rshift_index_matcher = calloc(nresos, sizeof(int));
    int *rshift_pixs_galind_bounds = calloc(nresos, sizeof(int));
    int *rshift_pix_gals = calloc(nresos, sizeof(int));
    for (int elreso=1; elreso<nresos; elreso++){
        rshift_index_matcher[elreso] = rshift_index_matcher[elreso-1] + npix_hash;
        rshift_pixs_galind_bounds[elreso] = rshift_pixs_galind_bounds[elreso-1] + cat->ngal_resos[elreso-1]+1;
        rshift_pix_gals[elreso] = rshift_pix_gals[elreso-1] + cat->ngal_resos[elreso-1];
    }
    *out_rshift_index_matcher = rshift_index_matcher;
    *out_rshift_pixs_galind_bounds = rshift_pixs_galind_bounds;
    *out_rshift_pix_gals = rshift_pix_gals;
}

// ---------------------------------------------------------------------------
// Flat-sky NN pair counts. Patch/region decomposition + pixel-box navigation.
// Numerically identical to the previous METRIC_FLAT branch of
// alloc_nn_doubletree; only the per-thread-redundant setup has moved out.
// ---------------------------------------------------------------------------
static void _nn_flat(const MultiresoCatalog *cat, const NavHash *nav,
                      const TreeResoParams *tree, const BinningParams *bin,
                      int nthreads, int verbose, NPCFOutput *out){

    int nbinsz = cat->nbinsz, nbinsr = bin->nbinsr, nresos = tree->nresos;
    double *totcount = calloc(nbinsz*nbinsz*nbinsr, sizeof(double));
    int *tmpnpair = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(int));
    double *tmpwcount = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
    double *tmpwnorm = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));

    // Hoisted out of the thread loop (see build_radial_helpers / build_flat_rshifts
    // docstrings above) -- read-only inside the parallel region below.
    double drbin, dbin_lin_inv;
    double *binedges; int *linarr_bins; int *reso_rindedges_base;
    build_radial_helpers(tree, bin, &binedges, &linarr_bins, &reso_rindedges_base, &drbin, &dbin_lin_inv);
    int *rshift_index_matcher, *rshift_pixs_galind_bounds, *rshift_pix_gals;
    build_flat_rshifts(cat, nav, &rshift_index_matcher, &rshift_pixs_galind_bounds, &rshift_pix_gals);

    // filledregions is now an input (NavHash), matching the convention GGG
    // already used -- the "every region is filled" placeholder, if desired,
    // is the caller's responsibility to construct, not duplicated here.
    int nfilledregions = nav->nfilledregions;
    int *filledregions = nav->filledregions;
    int nregionsdone = 0;
    int progress_step = nfilledregions/100;
    if (progress_step <= 0) progress_step = 1;

    #pragma omp parallel num_threads(nthreads)
    {
        int elthread = omp_get_thread_num();

        #pragma omp for schedule(dynamic, 64)
        for (int _elregion=0; _elregion<nfilledregions; _elregion++){
            int elregion = filledregions[_elregion];

            int ind_pix1, ind_pix2, ind_inpix1, ind_inpix2, ind_red, ind_gal1, ind_gal2, z_gal1, z_gal2;
            int pix1_lower, pix2_lower, pix1_upper, pix2_upper;
            int lower1, upper1, lower2, upper2;
            double innergal;
            int rbin, nbinsz2r, nbinszr, ind_rbin;
            double pos1_gal1, pos2_gal1, pos1_gal2, pos2_gal2, w_gal1, w_gal2;
            double rel1, rel2, dist, dist_sq;
            double rmin_reso, rmax_reso, rmin_reso_sq, rmax_reso_sq;
            int elreso_leaf, rbinmin, rbinmax;
            nbinsz2r = nbinsz*nbinsz*nbinsr;
            nbinszr = nbinsz*nbinsr;

            for (int elreso=0; elreso<nresos; elreso++){
                elreso_leaf = mymin(mymax(tree->minresoind_leaf, elreso+tree->resoshift_leafs), tree->maxresoind_leaf);
                rbinmin = reso_rindedges_base[elreso];
                rbinmax = reso_rindedges_base[elreso+1];
                rmin_reso = bin->rmin*exp(rbinmin*drbin);
                rmax_reso = bin->rmin*exp(rbinmax*drbin);
                rmin_reso_sq = rmin_reso*rmin_reso;
                rmax_reso_sq = rmax_reso*rmax_reso;
                lower1 = nav->pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion];
                upper1 = nav->pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion+1];

                for (ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                    ind_gal1 = rshift_pix_gals[elreso] + nav->pix_gals[rshift_pix_gals[elreso]+ind_inpix1];
                    innergal = cat->isinner_resos[ind_gal1];
                    if (innergal<1e-5){ continue; }
                    z_gal1 = cat->zbin_resos[ind_gal1];
                    pos1_gal1 = cat->pos1_resos[ind_gal1];
                    pos2_gal1 = cat->pos2_resos[ind_gal1];
                    w_gal1 = innergal*cat->weight_resos[ind_gal1];

                    pix1_lower = mymax(0, (int) floor((pos1_gal1 - (rmax_reso+nav->pix1_d) - nav->pix1_start)/nav->pix1_d));
                    pix2_lower = mymax(0, (int) floor((pos2_gal1 - (rmax_reso+nav->pix2_d) - nav->pix2_start)/nav->pix2_d));
                    pix1_upper = mymin(nav->pix1_n-1, (int) floor((pos1_gal1 + (rmax_reso+nav->pix1_d) - nav->pix1_start)/nav->pix1_d));
                    pix2_upper = mymin(nav->pix2_n-1, (int) floor((pos2_gal1 + (rmax_reso+nav->pix2_d) - nav->pix2_start)/nav->pix2_d));

                    for (ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                        for (ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                            ind_red = nav->index_matcher[rshift_index_matcher[elreso_leaf] + ind_pix2*nav->pix1_n + ind_pix1];
                            if (ind_red==-1){ continue; }
                            lower2 = nav->pixs_galind_bounds[rshift_pixs_galind_bounds[elreso_leaf]+ind_red];
                            upper2 = nav->pixs_galind_bounds[rshift_pixs_galind_bounds[elreso_leaf]+ind_red+1];
                            for (ind_inpix2=lower2; ind_inpix2<upper2; ind_inpix2++){
                                ind_gal2 = rshift_pix_gals[elreso_leaf] + nav->pix_gals[rshift_pix_gals[elreso_leaf]+ind_inpix2];
                                pos1_gal2 = cat->pos1_resos[ind_gal2];
                                pos2_gal2 = cat->pos2_resos[ind_gal2];
                                rel1 = pos1_gal2 - pos1_gal1;
                                rel2 = pos2_gal2 - pos2_gal1;
                                dist_sq = rel1*rel1 + rel2*rel2;
                                if (rel1<0 && bin->do_dc==0){ continue; }
                                if (dist_sq < rmin_reso_sq || dist_sq >= rmax_reso_sq){ continue; }
                                dist = sqrt(dist_sq);
                                w_gal2 = cat->weight_resos[ind_gal2];
                                z_gal2 = cat->zbin_resos[ind_gal2];

                                nn_bin_accumulate(dist, w_gal1, w_gal2, z_gal1, z_gal2,
                                                  bin->rmin, binedges, linarr_bins, dbin_lin_inv,
                                                  nbinsz, nbinsr, elthread, tmpnpair, tmpwcount, tmpwnorm);
                            }
                        }
                    }
                }
            }
            #pragma omp atomic
            nregionsdone += 1;
            if ((verbose>0) && (nregionsdone%progress_step==0)) {
                #pragma omp critical
                { printf("."); }
            }
        }
    }

    free(binedges); free(linarr_bins); free(reso_rindedges_base);
    free(rshift_index_matcher); free(rshift_pixs_galind_bounds); free(rshift_pix_gals);

    // --- accumulate (identical to the previous shared tail) ---
    for (int elbinr=0; elbinr<nbinsr; elbinr++){
        for (int elbinz1=0; elbinz1<nbinsz; elbinz1++){
            for (int elbinz2=0; elbinz2<nbinsz; elbinz2++){
                int tmpind = elbinz1*nbinsz*nbinsr + elbinz2*nbinsr + elbinr;
                for (int thisthread=0; thisthread<nthreads; thisthread++){
                    int tshift = thisthread*nbinsz*nbinsz*nbinsr;
                    totcount[tmpind] += tmpwcount[tshift+tmpind];
                    out->npair_cell[tmpind] += tmpnpair[tshift+tmpind];
                    out->norm[tmpind] += tmpwnorm[tshift+tmpind];
                }
            }
        }
    }
    for (int elbinz1=0; elbinz1<nbinsz; elbinz1++){
        for (int elbinz2=0; elbinz2<nbinsz; elbinz2++){
            for (int elbinr=0; elbinr<nbinsr; elbinr++){
                int tmpind = elbinz1*nbinsz*nbinsr + elbinz2*nbinsr + elbinr;
                if (out->norm[tmpind] != 0){
                    out->bin_centers[tmpind] = totcount[tmpind]/out->norm[tmpind];
                }
            }
        }
    }
    free(totcount); free(tmpwcount); free(tmpnpair); free(tmpwnorm);
}

// ---------------------------------------------------------------------------
// Curved-sky NN pair counts. Per radial band, gal1 = that band's legs; a live
// nested-HEALPix query_disc at the leaf band's nside returns candidate pixels,
// whose legs (via the bucket-hash CSR) are the gal2 partners. Numerically
// identical to the previous METRIC_SPHERICAL branch.
// ---------------------------------------------------------------------------
static void _nn_spherical(const MultiresoCatalog *cat, const NavHash *nav,
                           const TreeResoParams *tree, const BinningParams *bin,
                           int nthreads, int verbose, NPCFOutput *out){

    int nbinsz = cat->nbinsz, nbinsr = bin->nbinsr, nresos = tree->nresos;
    double *totcount = calloc(nbinsz*nbinsz*nbinsr, sizeof(double));
    int *tmpnpair = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(int));
    double *tmpwcount = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
    double *tmpwnorm = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));

    double drbin, dbin_lin_inv;
    double *binedges; int *linarr_bins; int *reso_rindedges;
    build_radial_helpers(tree, bin, &binedges, &linarr_bins, &reso_rindedges, &drbin, &dbin_lin_inv);

    for (int elreso=0; elreso<nresos; elreso++){
        int elreso_leaf = mymin(mymax(tree->minresoind_leaf, elreso+tree->resoshift_leafs), tree->maxresoind_leaf);
        int rbinmin = reso_rindedges[elreso];
        int rbinmax = reso_rindedges[elreso+1];
        if (rbinmax <= rbinmin){ continue; }
        double rmin_reso = bin->rmin*exp(rbinmin*drbin);
        double rmax_reso = bin->rmin*exp(rbinmax*drbin);
        long ns_leaf = nav->nside_nav[elreso_leaf];
        int n1 = cat->ngal_resos[elreso];
        long red1_off = nav->rshift_red[elreso];
        long redleaf_off = nav->rshift_red[elreso_leaf];
        const long *cellpix_leaf = nav->cell_pix + nav->rshift_cellpix[elreso_leaf];
        const int  *bounds_leaf  = nav->cell_redbounds + nav->rshift_cellbounds[elreso_leaf];
        int ncells_leaf = nav->ncells_resos[elreso_leaf];

        #pragma omp parallel num_threads(nthreads)
        {
            int thread = omp_get_thread_num();
            long cap = 2048;
            long *ranges = malloc(2*cap*sizeof(long));
            #pragma omp for schedule(dynamic, 64)
            for (int i1=0; i1<n1; i1++){
                long g1 = red1_off + i1;
                if (cat->isinner_resos[g1] < 1e-5){ continue; }
                double cx = cat->vx_resos[g1], cy = cat->vy_resos[g1], cz = cat->vz_resos[g1];
                double w1 = cat->isinner_resos[g1]*cat->weight_resos[g1];
                int z1 = cat->zbin_resos[g1];
                double v1[3] = {cx, cy, cz};
                long nr = hpx_query_disc_nest_ranges(ns_leaf, v1, rmax_reso, ranges, cap);
                if (nr > cap){ cap = nr; ranges = realloc(ranges, 2*cap*sizeof(long));
                               nr = hpx_query_disc_nest_ranges(ns_leaf, v1, rmax_reso, ranges, cap); }
                int ci = 0;
                for (long r=0; r<nr; r++){
                    long plo = ranges[2*r], phi = ranges[2*r+1];
                    int loi = ci, hii = ncells_leaf;
                    while (loi < hii){ int m = (loi+hii)>>1;
                        if (cellpix_leaf[m] < plo){ loi = m+1; } else { hii = m; } }
                    ci = loi;
                    while (ci < ncells_leaf && cellpix_leaf[ci] < phi){
                        int lo = bounds_leaf[ci], hi = bounds_leaf[ci+1];
                        for (int j=lo; j<hi; j++){
                            long g2 = redleaf_off + j;
                            // do_dc=0: count each unordered pair once. query_disc is symmetric,
                            // so when leaf reso == base reso (the default resoshift_leafs=0) both
                            // orderings are enumerated; keep g2>g1. Halves the geodesic work; the
                            // Python layer restores norm/npair via its x2 do_dc rescale. Guarded to
                            // same-reso bands, where the swap symmetry and this index compare hold.
                            if (bin->do_dc==0 && elreso_leaf==elreso && g2 <= g1){ continue; }
                            double dist = sphere_dist(cx, cy, cz, cat->vx_resos[g2], cat->vy_resos[g2], cat->vz_resos[g2]);
                            if (dist < rmin_reso || dist >= rmax_reso){ continue; }
                            nn_bin_accumulate(dist, w1, cat->weight_resos[g2], z1, cat->zbin_resos[g2],
                                              bin->rmin, binedges, linarr_bins, dbin_lin_inv,
                                              nbinsz, nbinsr, thread, tmpnpair, tmpwcount, tmpwnorm);
                        }
                        ci++;
                    }
                }
            }
            free(ranges);
        }
        if (verbose>0){ printf("."); }
    }
    free(binedges); free(linarr_bins); free(reso_rindedges);

    for (int elbinr=0; elbinr<nbinsr; elbinr++){
        for (int elbinz1=0; elbinz1<nbinsz; elbinz1++){
            for (int elbinz2=0; elbinz2<nbinsz; elbinz2++){
                int tmpind = elbinz1*nbinsz*nbinsr + elbinz2*nbinsr + elbinr;
                for (int thisthread=0; thisthread<nthreads; thisthread++){
                    int tshift = thisthread*nbinsz*nbinsz*nbinsr;
                    totcount[tmpind] += tmpwcount[tshift+tmpind];
                    out->npair_cell[tmpind] += tmpnpair[tshift+tmpind];
                    out->norm[tmpind] += tmpwnorm[tshift+tmpind];
                }
            }
        }
    }
    for (int elbinz1=0; elbinz1<nbinsz; elbinz1++){
        for (int elbinz2=0; elbinz2<nbinsz; elbinz2++){
            for (int elbinr=0; elbinr<nbinsr; elbinr++){
                int tmpind = elbinz1*nbinsz*nbinsr + elbinz2*nbinsr + elbinr;
                if (out->norm[tmpind] != 0){
                    out->bin_centers[tmpind] = totcount[tmpind]/out->norm[tmpind];
                }
            }
        }
    }
    free(totcount); free(tmpwcount); free(tmpnpair); free(tmpwnorm);
}

// ---------------------------------------------------------------------------
// Public entry point: a thin metric dispatch. This is the only function the
// Python ctypes binding calls. GGG's analogous entry point
// (alloc_ggg_doubletree, not implemented here) would have the identical shape:
// dispatch on cat->metric to _ggg_flat / _ggg_spherical, sharing these same
// four struct types as input.
// ---------------------------------------------------------------------------
void alloc_nn_doubletree(const MultiresoCatalog *cat, const NavHash *nav,
                          const TreeResoParams *tree, const BinningParams *bin,
                          int nthreads, int verbose, NPCFOutput *out){
    switch (cat->metric) {
        case METRIC_SPHERICAL:
            _nn_spherical(cat, nav, tree, bin, nthreads, verbose, out);
            break;
        case METRIC_FLAT:
        default:
            _nn_flat(cat, nav, tree, bin, nthreads, verbose, out);
            break;
    }
    if (verbose>0){ printf("\n"); }
}

////////////////////////////////////////////////
/// SECOND-ORDER (GG) SHEAR 2PCF              ///
////////////////////////////////////////////////

// Shared radial bin lookup + spin-2 accumulation. Like nn_bin_accumulate, but
// also scatters the two natural-component contributions the caller has already
// projected onto the pair geodesic: xip_c -> tmpggstar (xi_plus, no net phase),
// xim_c -> tmpgg (xi_minus). Geometry-agnostic; both _gg_flat and _gg_spherical
// call it.
static inline void gg_bin_accumulate(
    double dist, double w1, double w2, int z1, int z2,
    double complex xip_c, double complex xim_c,
    double rmin, double *binedges, int *linarr_bins, double dbin_lin_inv,
    int nbinsz, int nbinsr, int thread,
    int *tmpnpair, double *tmpwcount, double *tmpwnorm,
    double complex *tmpgg, double complex *tmpggstar){
    int tmplogbin = (int) ((dist-rmin)*dbin_lin_inv);
    int rbin = linarr_bins[tmplogbin];
    rbin += (dist > binedges[rbin+1]) ? 1 : 0;
    int ind = thread*nbinsz*nbinsz*nbinsr + z1*nbinsz*nbinsr + z2*nbinsr + rbin;
    tmpnpair[ind] += 1;
    tmpwcount[ind] += w1*w2*dist;
    tmpwnorm[ind] += w1*w2;
    tmpggstar[ind] += xip_c;
    tmpgg[ind] += xim_c;
}

// Reduce the per-thread scatter arrays into the unified NPCFOutput and normalise. Shared
// by _gg_flat and _gg_spherical (their accumulation tails are identical).
static void gg_reduce(int nbinsz, int nbinsr, int nthreads,
                      double *totcount, int *tmpnpair, double *tmpwcount, double *tmpwnorm,
                      double complex *tmpgg, double complex *tmpggstar, NPCFOutput *out){
    for (int elbinr=0; elbinr<nbinsr; elbinr++){
        for (int elbinz1=0; elbinz1<nbinsz; elbinz1++){
            for (int elbinz2=0; elbinz2<nbinsz; elbinz2++){
                int tmpind = elbinz1*nbinsz*nbinsr + elbinz2*nbinsr + elbinr;
                for (int thisthread=0; thisthread<nthreads; thisthread++){
                    int tshift = thisthread*nbinsz*nbinsz*nbinsr;
                    totcount[tmpind] += tmpwcount[tshift+tmpind];
                    out->npair[tmpind] += tmpnpair[tshift+tmpind];
                    out->norm[tmpind] += tmpwnorm[tshift+tmpind];
                    out->npcf[tmpind] += tmpggstar[tshift+tmpind];
                    out->npcf[nbinsz*nbinsz*nbinsr + tmpind] += tmpgg[tshift+tmpind];
                }
            }
        }
    }
    for (int elbinz1=0; elbinz1<nbinsz; elbinz1++){
        for (int elbinz2=0; elbinz2<nbinsz; elbinz2++){
            for (int elbinr=0; elbinr<nbinsr; elbinr++){
                int tmpind = elbinz1*nbinsz*nbinsr + elbinz2*nbinsr + elbinr;
                if (out->norm[tmpind] != 0){
                    out->bin_centers[tmpind] = totcount[tmpind]/out->norm[tmpind];
                    out->npcf[tmpind] /= out->norm[tmpind];
                    out->npcf[nbinsz*nbinsz*nbinsr + tmpind] /= out->norm[tmpind];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Flat-sky shear 2PCF. Same patch/region decomposition + pixel-box navigation
// as _nn_flat; the spin-2 phase is the flat quadrant form phirotc_sq=e^{-2i*phi}
// with phi=atan2(rel2,rel1). Numerically identical to the previous
// alloc_xipm_doubletree flat kernel (the shear accumulation expressions are
// unchanged; only the per-thread-redundant setup has moved out).
// ---------------------------------------------------------------------------
static void _gg_flat(const MultiresoCatalog *cat, const NavHash *nav,
                     const TreeResoParams *tree, const BinningParams *bin,
                     int nthreads, int verbose, NPCFOutput *out){

    int nbinsz = cat->nbinsz, nbinsr = bin->nbinsr, nresos = tree->nresos;
    double *totcount = calloc(nbinsz*nbinsz*nbinsr, sizeof(double));
    int *tmpnpair = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(int));
    double *tmpwcount = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
    double *tmpwnorm = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
    double complex *tmpgg = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double complex));
    double complex *tmpggstar = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double complex));

    double drbin, dbin_lin_inv;
    double *binedges; int *linarr_bins; int *reso_rindedges_base;
    build_radial_helpers(tree, bin, &binedges, &linarr_bins, &reso_rindedges_base, &drbin, &dbin_lin_inv);
    int *rshift_index_matcher, *rshift_pixs_galind_bounds, *rshift_pix_gals;
    build_flat_rshifts(cat, nav, &rshift_index_matcher, &rshift_pixs_galind_bounds, &rshift_pix_gals);

    int nfilledregions = nav->nfilledregions;
    int *filledregions = nav->filledregions;
    int nregionsdone = 0;
    int progress_step = nfilledregions/100;
    if (progress_step <= 0) progress_step = 1;

    #pragma omp parallel num_threads(nthreads)
    {
        int elthread = omp_get_thread_num();

        #pragma omp for schedule(dynamic, 64)
        for (int _elregion=0; _elregion<nfilledregions; _elregion++){
            int elregion = filledregions[_elregion];

            int ind_pix1, ind_pix2, ind_inpix1, ind_inpix2, ind_red, ind_gal1, ind_gal2, z_gal1, z_gal2;
            int pix1_lower, pix2_lower, pix1_upper, pix2_upper;
            int lower1, upper1, lower2, upper2;
            double innergal;
            double pos1_gal1, pos2_gal1, pos1_gal2, pos2_gal2, w_gal1, w_gal2;
            double rel1, rel2, dist, dist_sq;
            double complex wshape_gal1, wshape_gal2, phirotc_sq;
            double rmin_reso, rmax_reso, rmin_reso_sq, rmax_reso_sq;
            int elreso_leaf, rbinmin, rbinmax;

            for (int elreso=0; elreso<nresos; elreso++){
                elreso_leaf = mymin(mymax(tree->minresoind_leaf, elreso+tree->resoshift_leafs), tree->maxresoind_leaf);
                rbinmin = reso_rindedges_base[elreso];
                rbinmax = reso_rindedges_base[elreso+1];
                rmin_reso = bin->rmin*exp(rbinmin*drbin);
                rmax_reso = bin->rmin*exp(rbinmax*drbin);
                rmin_reso_sq = rmin_reso*rmin_reso;
                rmax_reso_sq = rmax_reso*rmax_reso;
                lower1 = nav->pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion];
                upper1 = nav->pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion+1];

                for (ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                    ind_gal1 = rshift_pix_gals[elreso] + nav->pix_gals[rshift_pix_gals[elreso]+ind_inpix1];
                    innergal = cat->isinner_resos[ind_gal1];
                    if (innergal<1e-5){ continue; }
                    z_gal1 = cat->zbin_resos[ind_gal1];
                    pos1_gal1 = cat->pos1_resos[ind_gal1];
                    pos2_gal1 = cat->pos2_resos[ind_gal1];
                    w_gal1 = innergal*cat->weight_resos[ind_gal1];
                    wshape_gal1 = (double complex) w_gal1 * (cat->e1_resos[ind_gal1]+I*cat->e2_resos[ind_gal1]);

                    pix1_lower = mymax(0, (int) floor((pos1_gal1 - (rmax_reso+nav->pix1_d) - nav->pix1_start)/nav->pix1_d));
                    pix2_lower = mymax(0, (int) floor((pos2_gal1 - (rmax_reso+nav->pix2_d) - nav->pix2_start)/nav->pix2_d));
                    pix1_upper = mymin(nav->pix1_n-1, (int) floor((pos1_gal1 + (rmax_reso+nav->pix1_d) - nav->pix1_start)/nav->pix1_d));
                    pix2_upper = mymin(nav->pix2_n-1, (int) floor((pos2_gal1 + (rmax_reso+nav->pix2_d) - nav->pix2_start)/nav->pix2_d));

                    for (ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                        for (ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                            ind_red = nav->index_matcher[rshift_index_matcher[elreso_leaf] + ind_pix2*nav->pix1_n + ind_pix1];
                            if (ind_red==-1){ continue; }
                            lower2 = nav->pixs_galind_bounds[rshift_pixs_galind_bounds[elreso_leaf]+ind_red];
                            upper2 = nav->pixs_galind_bounds[rshift_pixs_galind_bounds[elreso_leaf]+ind_red+1];
                            for (ind_inpix2=lower2; ind_inpix2<upper2; ind_inpix2++){
                                ind_gal2 = rshift_pix_gals[elreso_leaf] + nav->pix_gals[rshift_pix_gals[elreso_leaf]+ind_inpix2];
                                pos1_gal2 = cat->pos1_resos[ind_gal2];
                                pos2_gal2 = cat->pos2_resos[ind_gal2];
                                rel1 = pos1_gal2 - pos1_gal1;
                                rel2 = pos2_gal2 - pos2_gal1;
                                dist_sq = rel1*rel1 + rel2*rel2;
                                if (rel1<0 && bin->do_dc==0){ continue; }
                                if (dist_sq < rmin_reso_sq || dist_sq >= rmax_reso_sq){ continue; }
                                dist = sqrt(dist_sq);
                                w_gal2 = cat->weight_resos[ind_gal2];
                                z_gal2 = cat->zbin_resos[ind_gal2];
                                wshape_gal2 = (double complex) w_gal2 * (cat->e1_resos[ind_gal2]+I*cat->e2_resos[ind_gal2]);
                                phirotc_sq = (rel1*rel1-rel2*rel2-2*I*rel1*rel2)/dist_sq;
                                gg_bin_accumulate(dist, w_gal1, w_gal2, z_gal1, z_gal2,
                                                  wshape_gal1*conj(wshape_gal2),
                                                  wshape_gal1*wshape_gal2*phirotc_sq*phirotc_sq,
                                                  bin->rmin, binedges, linarr_bins, dbin_lin_inv,
                                                  nbinsz, nbinsr, elthread,
                                                  tmpnpair, tmpwcount, tmpwnorm, tmpgg, tmpggstar);
                            }
                        }
                    }
                }
            }
            #pragma omp atomic
            nregionsdone += 1;
            if ((verbose>0) && (nregionsdone%progress_step==0)) {
                #pragma omp critical
                { printf("."); }
            }
        }
    }

    free(binedges); free(linarr_bins); free(reso_rindedges_base);
    free(rshift_index_matcher); free(rshift_pixs_galind_bounds); free(rshift_pix_gals);

    gg_reduce(nbinsz, nbinsr, nthreads, totcount, tmpnpair, tmpwcount, tmpwnorm, tmpgg, tmpggstar, out);
    free(totcount); free(tmpwcount); free(tmpnpair); free(tmpwnorm); free(tmpgg); free(tmpggstar);
}

// ---------------------------------------------------------------------------
// Curved-sky shear 2PCF. Same nested-HEALPix navigation as _nn_spherical, but
// the spin-2 shear is projected onto the connecting geodesic at BOTH ends: each
// galaxy's shape is rotated by e^{-2i*bearing} in its own east-north tangent
// frame (the sphere back-bearing is not the forward bearing + pi, so a single
// angle no longer suffices). With proj = w*(e1+i*e2)*e^{-2i*bearing_to_partner},
//   xi_plus  = proj1 * conj(proj2)   (reduces to w1 g1 conj(w2 g2), no phase)
//   xi_minus = proj1 * proj2         (reduces to w1 g1 w2 g2 e^{-4i*phi}).
// This is exactly the flat kernel when the two bearings differ by pi; see
// Tutorials_private/fullsky_covariance_notes.md sections 1.2-1.3.
// ---------------------------------------------------------------------------
static void _gg_spherical(const MultiresoCatalog *cat, const NavHash *nav,
                          const TreeResoParams *tree, const BinningParams *bin,
                          int nthreads, int verbose, NPCFOutput *out){

    int nbinsz = cat->nbinsz, nbinsr = bin->nbinsr, nresos = tree->nresos;
    double *totcount = calloc(nbinsz*nbinsz*nbinsr, sizeof(double));
    int *tmpnpair = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(int));
    double *tmpwcount = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
    double *tmpwnorm = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
    double complex *tmpgg = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double complex));
    double complex *tmpggstar = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double complex));

    double drbin, dbin_lin_inv;
    double *binedges; int *linarr_bins; int *reso_rindedges;
    build_radial_helpers(tree, bin, &binedges, &linarr_bins, &reso_rindedges, &drbin, &dbin_lin_inv);

    for (int elreso=0; elreso<nresos; elreso++){
        int elreso_leaf = mymin(mymax(tree->minresoind_leaf, elreso+tree->resoshift_leafs), tree->maxresoind_leaf);
        int rbinmin = reso_rindedges[elreso];
        int rbinmax = reso_rindedges[elreso+1];
        if (rbinmax <= rbinmin){ continue; }
        double rmin_reso = bin->rmin*exp(rbinmin*drbin);
        double rmax_reso = bin->rmin*exp(rbinmax*drbin);
        long ns_leaf = nav->nside_nav[elreso_leaf];
        int n1 = cat->ngal_resos[elreso];
        long red1_off = nav->rshift_red[elreso];
        long redleaf_off = nav->rshift_red[elreso_leaf];
        const long *cellpix_leaf = nav->cell_pix + nav->rshift_cellpix[elreso_leaf];
        const int  *bounds_leaf  = nav->cell_redbounds + nav->rshift_cellbounds[elreso_leaf];
        int ncells_leaf = nav->ncells_resos[elreso_leaf];

        #pragma omp parallel num_threads(nthreads)
        {
            int thread = omp_get_thread_num();
            long cap = 2048;
            long *ranges = malloc(2*cap*sizeof(long));
            #pragma omp for schedule(dynamic, 64)
            for (int i1=0; i1<n1; i1++){
                long g1 = red1_off + i1;
                if (cat->isinner_resos[g1] < 1e-5){ continue; }
                double cx = cat->vx_resos[g1], cy = cat->vy_resos[g1], cz = cat->vz_resos[g1];
                double w1 = cat->isinner_resos[g1]*cat->weight_resos[g1];
                int z1 = cat->zbin_resos[g1];
                double sd1 = cat->sindec_resos[g1], cd1 = cat->cosdec_resos[g1];
                double complex wshape1 = (double complex) w1 * (cat->e1_resos[g1]+I*cat->e2_resos[g1]);
                double v1[3] = {cx, cy, cz};
                long nr = hpx_query_disc_nest_ranges(ns_leaf, v1, rmax_reso, ranges, cap);
                if (nr > cap){ cap = nr; ranges = realloc(ranges, 2*cap*sizeof(long));
                               nr = hpx_query_disc_nest_ranges(ns_leaf, v1, rmax_reso, ranges, cap); }
                int ci = 0;
                for (long r=0; r<nr; r++){
                    long plo = ranges[2*r], phi = ranges[2*r+1];
                    int loi = ci, hii = ncells_leaf;
                    while (loi < hii){ int m = (loi+hii)>>1;
                        if (cellpix_leaf[m] < plo){ loi = m+1; } else { hii = m; } }
                    ci = loi;
                    while (ci < ncells_leaf && cellpix_leaf[ci] < phi){
                        int lo = bounds_leaf[ci], hi = bounds_leaf[ci+1];
                        for (int j=lo; j<hi; j++){
                            long g2 = redleaf_off + j;
                            // do_dc=0: count each unordered pair once (see _nn_spherical). query_disc
                            // is symmetric, so with leaf reso == base reso both orderings appear; keep
                            // g2>g1. Halves the geodesic + spin-2 projection work; Python restores
                            // norm/npair via its x2 do_dc rescale. Guarded to same-reso bands.
                            if (bin->do_dc==0 && elreso_leaf==elreso && g2 <= g1){ continue; }
                            double dist = sphere_dist(cx, cy, cz, cat->vx_resos[g2], cat->vy_resos[g2], cat->vz_resos[g2]);
                            if (dist < rmin_reso || dist >= rmax_reso){ continue; }
                            double sd2 = cat->sindec_resos[g2], cd2 = cat->cosdec_resos[g2];
                            double vx2 = cat->vx_resos[g2], vy2 = cat->vy_resos[g2];
                            double complex wshape2 = (double complex) cat->weight_resos[g2] * (cat->e1_resos[g2]+I*cat->e2_resos[g2]);
                            // Spin-2 geodesic projection phase e^{-2i*bearing} at each end, built
                            // directly from the tangent-frame east/north bearing components (E,N)
                            // -- no atan2/cexp in the pair loop. sphere_bearing gives phi=atan2(n,e)
                            // so e^{-2i*phi} = (e^2-n^2-2i*e*n)/(e^2+n^2); here (E,N)=cosdec_a*(e,n)
                            // (a positive scale, so the phase is unchanged) is formed from the
                            // position-vector cross/dot, reusing vx/vy already loaded for the
                            // distance. The back bearing shares E (up to sign) and the equatorial
                            // dot P. Curved-sky analogue of the flat phirotc_sq.
                            double P = cx*vx2 + cy*vy2;
                            double E12 = cx*vy2 - cy*vx2;
                            double N12 = cd1*cd1*sd2 - sd1*P;
                            double N21 = cd2*cd2*sd1 - sd2*P;
                            double complex rc1 = (E12*E12 - N12*N12 - 2.*I*E12*N12)/(E12*E12 + N12*N12);
                            double complex rc2 = (E12*E12 - N21*N21 + 2.*I*E12*N21)/(E12*E12 + N21*N21);
                            double complex proj1 = wshape1 * rc1;
                            double complex proj2 = wshape2 * rc2;
                            gg_bin_accumulate(dist, w1, cat->weight_resos[g2], z1, cat->zbin_resos[g2],
                                              proj1*conj(proj2), proj1*proj2,
                                              bin->rmin, binedges, linarr_bins, dbin_lin_inv,
                                              nbinsz, nbinsr, thread,
                                              tmpnpair, tmpwcount, tmpwnorm, tmpgg, tmpggstar);
                        }
                        ci++;
                    }
                }
            }
            free(ranges);
        }
        if (verbose>0){ printf("."); }
    }
    free(binedges); free(linarr_bins); free(reso_rindedges);

    gg_reduce(nbinsz, nbinsr, nthreads, totcount, tmpnpair, tmpwcount, tmpwnorm, tmpgg, tmpggstar, out);
    free(totcount); free(tmpwcount); free(tmpnpair); free(tmpwnorm); free(tmpgg); free(tmpggstar);
}

// ---------------------------------------------------------------------------
// Public entry point: a thin metric dispatch, mirroring alloc_nn_doubletree.
// ---------------------------------------------------------------------------
void alloc_gg_doubletree(const MultiresoCatalog *cat, const NavHash *nav,
                         const TreeResoParams *tree, const BinningParams *bin,
                         int nthreads, int verbose, NPCFOutput *out){
    switch (cat->metric) {
        case METRIC_SPHERICAL:
            _gg_spherical(cat, nav, tree, bin, nthreads, verbose, out);
            break;
        case METRIC_FLAT:
        default:
            _gg_flat(cat, nav, tree, bin, nthreads, verbose, out);
            break;
    }
    if (verbose>0){ printf("\n"); }
}


///////////////////////////////
// Position-shape (NG) 2PCF //
///////////////////////////////
// Scatter one lens-source pair into the NG accumulators. Layout mirrors
// gg_bin_accumulate but rectangular in (z_lens, z_source) and with a single
// natural component xi = w_l * w_s(e1+i*e2) * e^{-2i*phi}.
static inline void ng_bin_accumulate(
    double dist, double w1, double w2, int z1, int z2, double complex xi_c,
    double rmin, double *binedges, int *linarr_bins, double dbin_lin_inv,
    int nbinsz_lens, int nbinsz_source, int nbinsr, int thread,
    int *tmpnpair, double *tmpwcount, double *tmpwnorm, double complex *tmpxi){
    int tmplogbin = (int) ((dist-rmin)*dbin_lin_inv);
    int rbin = linarr_bins[tmplogbin];
    rbin += (dist > binedges[rbin+1]) ? 1 : 0;
    int ind = thread*nbinsz_lens*nbinsz_source*nbinsr + z1*nbinsz_source*nbinsr + z2*nbinsr + rbin;
    tmpnpair[ind] += 1;
    tmpwcount[ind] += w1*w2*dist;
    tmpwnorm[ind] += w1*w2;
    tmpxi[ind] += xi_c;
}

// Reduce the per-thread NG scatter arrays into the unified NPCFOutput and normalise.
static void ng_reduce(int nbinsz_lens, int nbinsz_source, int nbinsr, int nthreads,
                      double *totcount, int *tmpnpair, double *tmpwcount, double *tmpwnorm,
                      double complex *tmpxi, NPCFOutput *out){
    int nzzr = nbinsz_lens*nbinsz_source*nbinsr;
    for (int z1=0; z1<nbinsz_lens; z1++){
        for (int z2=0; z2<nbinsz_source; z2++){
            for (int elbinr=0; elbinr<nbinsr; elbinr++){
                int tmpind = z1*nbinsz_source*nbinsr + z2*nbinsr + elbinr;
                for (int thisthread=0; thisthread<nthreads; thisthread++){
                    int tshift = thisthread*nzzr;
                    totcount[tmpind] += tmpwcount[tshift+tmpind];
                    out->npair[tmpind] += tmpnpair[tshift+tmpind];
                    out->norm[tmpind] += tmpwnorm[tshift+tmpind];
                    out->npcf[tmpind] += tmpxi[tshift+tmpind];
                }
            }
        }
    }
    for (int i=0; i<nzzr; i++){
        if (out->norm[i] != 0){
            out->bin_centers[i] = totcount[i]/out->norm[i];
            out->npcf[i] /= out->norm[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Flat-sky position-shape (galaxy-galaxy lensing) 2PCF. A scalar lens catalog
// (cat_lens/nav_lens, central gal1) cross-correlated with a spin-2 source
// catalog (cat_source/nav_source, field gal2). Both hashes share the same flat
// grid (built on a joint extent), so a lens-position search box indexes the
// source grid directly. Single-sided projection phirotc_sq=e^{-2i*phi}:
//   xi = <w_l w_s (e1+i*e2)_s e^{-2i*phi}> / <w_l w_s>,  Re xi = -gamma_t.
// No do_dc: a cross counts every lens-source pair once.
// ---------------------------------------------------------------------------
static void _ng_flat(const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                     const MultiresoCatalog *cat_source, const NavHash *nav_source,
                     const TreeResoParams *tree, const BinningParams *bin,
                     int nthreads, int verbose, NPCFOutput *out){

    int nbinsz_l = cat_lens->nbinsz, nbinsz_s = cat_source->nbinsz;
    int nbinsr = bin->nbinsr, nresos = tree->nresos;
    int nzzr = nbinsz_l*nbinsz_s*nbinsr;
    double *totcount = calloc(nzzr, sizeof(double));
    int *tmpnpair = calloc(nthreads*nzzr, sizeof(int));
    double *tmpwcount = calloc(nthreads*nzzr, sizeof(double));
    double *tmpwnorm = calloc(nthreads*nzzr, sizeof(double));
    double complex *tmpxi = calloc(nthreads*nzzr, sizeof(double complex));

    double drbin, dbin_lin_inv;
    double *binedges; int *linarr_bins; int *reso_rindedges_base;
    build_radial_helpers(tree, bin, &binedges, &linarr_bins, &reso_rindedges_base, &drbin, &dbin_lin_inv);
    int *rsim_l, *rspgb_l, *rspg_l;
    build_flat_rshifts(cat_lens, nav_lens, &rsim_l, &rspgb_l, &rspg_l);
    int *rsim_s, *rspgb_s, *rspg_s;
    build_flat_rshifts(cat_source, nav_source, &rsim_s, &rspgb_s, &rspg_s);

    int nfilledregions = nav_lens->nfilledregions;
    int *filledregions = nav_lens->filledregions;

    #pragma omp parallel num_threads(nthreads)
    {
        int elthread = omp_get_thread_num();

        #pragma omp for schedule(dynamic, 64)
        for (int _elregion=0; _elregion<nfilledregions; _elregion++){
            int elregion = filledregions[_elregion];

            int ind_pix1, ind_pix2, ind_inpix1, ind_inpix2, ind_red, ind_gal1, ind_gal2, z_gal1, z_gal2;
            int pix1_lower, pix2_lower, pix1_upper, pix2_upper;
            int lower1, upper1, lower2, upper2;
            double innergal;
            double pos1_gal1, pos2_gal1, pos1_gal2, pos2_gal2, w_gal1, w_gal2;
            double rel1, rel2, dist, dist_sq;
            double complex wshape_gal2, phirotc_sq;
            double rmin_reso, rmax_reso, rmin_reso_sq, rmax_reso_sq;
            int elreso_leaf, rbinmin, rbinmax;

            for (int elreso=0; elreso<nresos; elreso++){
                elreso_leaf = mymin(mymax(tree->minresoind_leaf, elreso+tree->resoshift_leafs), tree->maxresoind_leaf);
                rbinmin = reso_rindedges_base[elreso];
                rbinmax = reso_rindedges_base[elreso+1];
                rmin_reso = bin->rmin*exp(rbinmin*drbin);
                rmax_reso = bin->rmin*exp(rbinmax*drbin);
                rmin_reso_sq = rmin_reso*rmin_reso;
                rmax_reso_sq = rmax_reso*rmax_reso;
                lower1 = nav_lens->pixs_galind_bounds[rspgb_l[elreso]+elregion];
                upper1 = nav_lens->pixs_galind_bounds[rspgb_l[elreso]+elregion+1];

                for (ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                    ind_gal1 = rspg_l[elreso] + nav_lens->pix_gals[rspg_l[elreso]+ind_inpix1];
                    innergal = cat_lens->isinner_resos[ind_gal1];
                    if (innergal<1e-5){ continue; }
                    z_gal1 = cat_lens->zbin_resos[ind_gal1];
                    pos1_gal1 = cat_lens->pos1_resos[ind_gal1];
                    pos2_gal1 = cat_lens->pos2_resos[ind_gal1];
                    w_gal1 = innergal*cat_lens->weight_resos[ind_gal1];

                    pix1_lower = mymax(0, (int) floor((pos1_gal1 - (rmax_reso+nav_source->pix1_d) - nav_source->pix1_start)/nav_source->pix1_d));
                    pix2_lower = mymax(0, (int) floor((pos2_gal1 - (rmax_reso+nav_source->pix2_d) - nav_source->pix2_start)/nav_source->pix2_d));
                    pix1_upper = mymin(nav_source->pix1_n-1, (int) floor((pos1_gal1 + (rmax_reso+nav_source->pix1_d) - nav_source->pix1_start)/nav_source->pix1_d));
                    pix2_upper = mymin(nav_source->pix2_n-1, (int) floor((pos2_gal1 + (rmax_reso+nav_source->pix2_d) - nav_source->pix2_start)/nav_source->pix2_d));

                    for (ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                        for (ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                            ind_red = nav_source->index_matcher[rsim_s[elreso_leaf] + ind_pix2*nav_source->pix1_n + ind_pix1];
                            if (ind_red==-1){ continue; }
                            lower2 = nav_source->pixs_galind_bounds[rspgb_s[elreso_leaf]+ind_red];
                            upper2 = nav_source->pixs_galind_bounds[rspgb_s[elreso_leaf]+ind_red+1];
                            for (ind_inpix2=lower2; ind_inpix2<upper2; ind_inpix2++){
                                ind_gal2 = rspg_s[elreso_leaf] + nav_source->pix_gals[rspg_s[elreso_leaf]+ind_inpix2];
                                pos1_gal2 = cat_source->pos1_resos[ind_gal2];
                                pos2_gal2 = cat_source->pos2_resos[ind_gal2];
                                rel1 = pos1_gal2 - pos1_gal1;
                                rel2 = pos2_gal2 - pos2_gal1;
                                dist_sq = rel1*rel1 + rel2*rel2;
                                if (dist_sq < rmin_reso_sq || dist_sq >= rmax_reso_sq){ continue; }
                                dist = sqrt(dist_sq);
                                w_gal2 = cat_source->weight_resos[ind_gal2];
                                z_gal2 = cat_source->zbin_resos[ind_gal2];
                                wshape_gal2 = (double complex) w_gal2 * (cat_source->e1_resos[ind_gal2]+I*cat_source->e2_resos[ind_gal2]);
                                phirotc_sq = (rel1*rel1-rel2*rel2-2*I*rel1*rel2)/dist_sq;
                                ng_bin_accumulate(dist, w_gal1, w_gal2, z_gal1, z_gal2,
                                                  w_gal1*wshape_gal2*phirotc_sq,
                                                  bin->rmin, binedges, linarr_bins, dbin_lin_inv,
                                                  nbinsz_l, nbinsz_s, nbinsr, elthread,
                                                  tmpnpair, tmpwcount, tmpwnorm, tmpxi);
                            }
                        }
                    }
                }
            }
        }
    }

    free(binedges); free(linarr_bins); free(reso_rindedges_base);
    free(rsim_l); free(rspgb_l); free(rspg_l);
    free(rsim_s); free(rspgb_s); free(rspg_s);

    ng_reduce(nbinsz_l, nbinsz_s, nbinsr, nthreads, totcount, tmpnpair, tmpwcount, tmpwnorm, tmpxi, out);
    free(totcount); free(tmpwcount); free(tmpnpair); free(tmpwnorm); free(tmpxi);
}

// Public entry point for the position-shape 2PCF. Flat-sky only for now; a
// spherical variant would mirror _gg_spherical with the single-sided projection.
void alloc_ng_doubletree(const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                         const MultiresoCatalog *cat_source, const NavHash *nav_source,
                         const TreeResoParams *tree, const BinningParams *bin,
                         int nthreads, int verbose, NPCFOutput *out){
    _ng_flat(cat_lens, nav_lens, cat_source, nav_source, tree, bin, nthreads, verbose, out);
    if (verbose>0){ printf("\n"); }
}


/////////////////////////
// Slab-hashed GN pairs //
/////////////////////////
// Discrete estimator kernel for the 2-pt position-shape (NI) correlator and its
// RR pair-count normalization, in a '3dbox' geometry (projected NI / w_{g+};
// Vedder et al. 2026, arXiv:2601.17914 Eq. 15). Correlates a query (position)
// catalog against a hashed catalog whose 2D spatial hash is split into
// line-of-sight slabs of width dpix_z: for each query it visits only the slabs
// overlapping [z-Pi, z+Pi], runs the transverse search box in each, and for
// pairs with |dz|<Pi and r_perp in [rmin,rmax) accumulates the spin-2 sum
// w_q w_h eps_h e^{-2i phi} (has_shapes=1) and/or the weighted pair count
// (has_shapes=0). Outputs are indexed [z_query, z_hashed, r_perp bin]. This is
// just the discrete 2-pt correlator with a line-of-sight-window metric; the
// third-order slab kernels live in corrfunc_third.c.
void ng_slab(
    double *q_pos1, double *q_pos2, double *q_pos3, double *q_w, int *q_zbin,
    int q_ngal, int nbinsz_q,
    double *h_pos1, double *h_pos2, double *h_pos3, double *h_w, int *h_zbin,
    double *h_e1, double *h_e2, int nbinsz_h,
    int nslabs, double z0, double dpix_z,
    double pix1_start, double pix1_d, int pix1_n,
    double pix2_start, double pix2_d, int pix2_n,
    int *slab_offsets, int *index_matcher, int *pixs_galind_bounds,
    int *rshift_bounds, int *pix_gals,
    double rmin, double rmax, int nbinsr, double Pi,
    int self_pairs, int has_shapes, int nthreads,
    double *out_xs_re, double *out_xs_im, double *out_wnorm,
    double *out_rsum, long *out_npairs){

    int npix = pix1_n * pix2_n;
    int nbinszz = nbinsz_q * nbinsz_h;
    int nout = nbinszz * nbinsr;
    double rmin2 = rmin * rmin;
    double rmax2 = rmax * rmax;
    double dlnr_inv = nbinsr / log(rmax/rmin);

    // Per-thread accumulators to avoid contention; reduced at the end.
    double *tmp_xs_re = calloc((size_t)nthreads*nout, sizeof(double));
    double *tmp_xs_im = calloc((size_t)nthreads*nout, sizeof(double));
    double *tmp_wnorm = calloc((size_t)nthreads*nout, sizeof(double));
    double *tmp_rsum  = calloc((size_t)nthreads*nout, sizeof(double));
    long   *tmp_npair = calloc((size_t)nthreads*nout, sizeof(long));

    #pragma omp parallel num_threads(nthreads)
    {
        int elthread = omp_get_thread_num();
        double *l_xs_re = tmp_xs_re + (size_t)elthread*nout;
        double *l_xs_im = tmp_xs_im + (size_t)elthread*nout;
        double *l_wnorm = tmp_wnorm + (size_t)elthread*nout;
        double *l_rsum  = tmp_rsum  + (size_t)elthread*nout;
        long   *l_npair = tmp_npair + (size_t)elthread*nout;

        #pragma omp for schedule(dynamic, 256)
        for (int i=0; i<q_ngal; i++){
            double p1 = q_pos1[i];
            double p2 = q_pos2[i];
            double zq = q_pos3[i];
            double wq = q_w[i];
            int zbin_q = q_zbin[i];

            // Slabs overlapping [zq-Pi, zq+Pi].
            int s_lo = (int) floor((zq - Pi - z0)/dpix_z);
            int s_hi = (int) floor((zq + Pi - z0)/dpix_z);
            if (s_lo < 0){ s_lo = 0; }
            if (s_hi > nslabs-1){ s_hi = nslabs-1; }

            // Transverse search box enclosing all neighbours within rmax.
            int pix1_lo = (int) floor((p1 - (rmax + pix1_d) - pix1_start)/pix1_d);
            int pix1_hi = (int) floor((p1 + (rmax + pix1_d) - pix1_start)/pix1_d);
            int pix2_lo = (int) floor((p2 - (rmax + pix2_d) - pix2_start)/pix2_d);
            int pix2_hi = (int) floor((p2 + (rmax + pix2_d) - pix2_start)/pix2_d);
            pix1_lo = mymax(0, pix1_lo); pix1_hi = mymin(pix1_n-1, pix1_hi);
            pix2_lo = mymax(0, pix2_lo); pix2_hi = mymin(pix2_n-1, pix2_hi);

            for (int s=s_lo; s<=s_hi; s++){
                int matcher_shift = s*npix;
                int bounds_shift = rshift_bounds[s];
                int gals_shift = slab_offsets[s];
                for (int ip1=pix1_lo; ip1<=pix1_hi; ip1++){
                    for (int ip2=pix2_lo; ip2<=pix2_hi; ip2++){
                        int ind_raw = ip2*pix1_n + ip1;
                        int ind_red = index_matcher[matcher_shift + ind_raw];
                        if (ind_red == -1){ continue; }
                        int lower = pixs_galind_bounds[bounds_shift + ind_red];
                        int upper = pixs_galind_bounds[bounds_shift + ind_red + 1];
                        for (int k=lower; k<upper; k++){
                            int j = pix_gals[gals_shift + k];
                            if (self_pairs && j==i){ continue; }
                            double rel1 = h_pos1[j] - p1;
                            double rel2 = h_pos2[j] - p2;
                            double d2 = rel1*rel1 + rel2*rel2;
                            if (d2 < rmin2 || d2 >= rmax2){ continue; }
                            double dz = h_pos3[j] - zq;
                            if (dz < 0){ dz = -dz; }
                            if (dz >= Pi){ continue; }

                            double r = sqrt(d2);
                            int rbin = (int) floor(log(r/rmin)*dlnr_inv);
                            if (rbin < 0 || rbin >= nbinsr){ continue; }
                            double w = wq * h_w[j];
                            int outind = (zbin_q*nbinsz_h + h_zbin[j])*nbinsr + rbin;

                            l_wnorm[outind] += w;
                            l_rsum[outind]  += w*r;
                            l_npair[outind] += 1;
                            if (has_shapes){
                                // e^{-2i phi}: Re=(rel1^2-rel2^2)/d2, Im=-2 rel1 rel2/d2
                                double pr = (rel1*rel1 - rel2*rel2)/d2;
                                double pi = -2.*rel1*rel2/d2;
                                double e1 = h_e1[j];
                                double e2 = h_e2[j];
                                l_xs_re[outind] += w*(e1*pr - e2*pi);
                                l_xs_im[outind] += w*(e1*pi + e2*pr);
                            }
                        }
                    }
                }
            }
        }
    }

    // Reduce per-thread accumulators into the output arrays.
    for (int o=0; o<nout; o++){
        for (int t=0; t<nthreads; t++){
            size_t ind = (size_t)t*nout + o;
            out_xs_re[o]  += tmp_xs_re[ind];
            out_xs_im[o]  += tmp_xs_im[ind];
            out_wnorm[o]  += tmp_wnorm[ind];
            out_rsum[o]   += tmp_rsum[ind];
            out_npairs[o] += tmp_npair[ind];
        }
    }

    free(tmp_xs_re); free(tmp_xs_im); free(tmp_wnorm); free(tmp_rsum); free(tmp_npair);
}
