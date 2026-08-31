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

#define M_PI      3.14159265358979323846
#define INV_2PI   0.15915494309189534561


///////////////////////
/// General helpers ///
///////////////////////

// Build the shared, radial-binning helper arrays
static void build_radial_helpers(
    const TreeResoParams *tree, const BinningParams *bin,
    double **out_binedges, int **out_linarr_bins, int **out_reso_rindedges,
    double *out_drbin, double *out_dbin_lin_inv){

    int nresos = tree->nresos, nbinsr = bin->nbinsr;
    double rmin = bin->rmin, rmax = bin->rmax;
    double drbin = (log(rmax)-log(rmin))/nbinsr;

    int *reso_rindedges = orpheus_calloc(nresos+1, sizeof(int));
    double *binedges = orpheus_calloc(nbinsr+2, sizeof(double));
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
    int *linarr_bins = orpheus_calloc(nbinsr_lin+1, sizeof(int));
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

// Build the per-reso shift arrays for the flat grid hash.
static void build_flat_rshifts(
    const MultiresoCatalog *cat, const NavHash *nav,
    int **out_rshift_index_matcher, int **out_rshift_pixs_galind_bounds,
    int **out_rshift_pix_gals){

    int nresos = cat->nresos;
    int npix_hash = nav->pix1_n * nav->pix2_n;
    int *rshift_index_matcher = orpheus_calloc(nresos, sizeof(int));
    int *rshift_pixs_galind_bounds = orpheus_calloc(nresos, sizeof(int));
    int *rshift_pix_gals = orpheus_calloc(nresos, sizeof(int));
    for (int elreso=1; elreso<nresos; elreso++){
        rshift_index_matcher[elreso] = rshift_index_matcher[elreso-1] + npix_hash;
        rshift_pixs_galind_bounds[elreso] = rshift_pixs_galind_bounds[elreso-1] + cat->ngal_resos[elreso-1]+1;
        rshift_pix_gals[elreso] = rshift_pix_gals[elreso-1] + cat->ngal_resos[elreso-1];
    }
    *out_rshift_index_matcher = rshift_index_matcher;
    *out_rshift_pixs_galind_bounds = rshift_pixs_galind_bounds;
    *out_rshift_pix_gals = rshift_pix_gals;
}


//////////////////////////////////
/// CORRELATOR SPECIFIC HELPERS //
//////////////////////////////////

// Shared bin update for every 2nd-order flat/spherical kernel
// the number of cfs to allocate is given by ncomp (0 for NN, 1 for NG, 2 for GG) 
static inline void bin_accumulate(
    double dist, double w1, double w2, int z1, int z2,
    double rmin, double *binedges, int *linarr_bins, double dbin_lin_inv,
    int nbinsz_lens, int nbinsz_source, int nbinsr, int thread, int nthreads,
    int *tmpnpair, double *tmpwcount, double *tmpwnorm,
    int ncomp, const double complex *comps, double complex *tmpcomp){
    int tmplogbin = (int) ((dist-rmin)*dbin_lin_inv);
    int rbin = linarr_bins[tmplogbin];
    rbin += (dist > binedges[rbin+1]) ? 1 : 0;
    int nzzr = nbinsz_lens*nbinsz_source*nbinsr;
    int ind = thread*nzzr + z1*nbinsz_source*nbinsr + z2*nbinsr + rbin;
    tmpnpair[ind] += 1;
    tmpwcount[ind] += w1*w2*dist;
    tmpwnorm[ind] += w1*w2;
    for (int c=0; c<ncomp; c++){ tmpcomp[c*nthreads*nzzr + ind] += comps[c]; }
}

// Reduce the per-thread arrays into the unified NPCFOutput for every 2nd-order flat/spherical kernel.
// the number of cfs to allocate is given by ncomp (0 for NN, 1 for NG, 2 for GG) 
static void bin_reduce(int nbinsz_lens, int nbinsz_source, int nbinsr, int nthreads,
                       double *totcount, int *tmpnpair, double *tmpwcount, double *tmpwnorm,
                       int ncomp, double complex *tmpcomp, NPCFOutput *out){
    int nzzr = nbinsz_lens*nbinsz_source*nbinsr;
    long long int *npair_out = (out->npair != NULL) ? out->npair : out->npair_cell;
    for (int binind=0; binind<nzzr; binind++){
        for (int t=0; t<nthreads; t++){
            int tind = t*nzzr + binind;
            totcount[binind]  += tmpwcount[tind];
            npair_out[binind] += tmpnpair[tind];
            out->norm[binind] += tmpwnorm[tind];
            for (int c=0; c<ncomp; c++){ out->npcf[c*nzzr + binind] += tmpcomp[c*nthreads*nzzr + tind]; }
        }
    }
    for (int binind=0; binind<nzzr; binind++){
        if (out->norm[binind] != 0){
            out->bin_centers[binind] = totcount[binind]/out->norm[binind];
            for (int c=0; c<ncomp; c++){ out->npcf[c*nzzr + binind] /= out->norm[binind]; }
        }
    }
}


///////////////////////////
// NN CORRELATOR CLASSES //
///////////////////////////

// Flat-sky DoubleTree estimtor of 2pt pair counts
static void _nn_flat(const MultiresoCatalog *cat, const NavHash *nav,
                      const TreeResoParams *tree, const BinningParams *bin,
                      int nthreads, int verbose, NPCFOutput *out){

    int nbinsz = cat->nbinsz, nbinsr = bin->nbinsr, nresos = tree->nresos;
    double *totcount = orpheus_calloc(nbinsz*nbinsz*nbinsr, sizeof(double));
    int *tmpnpair = orpheus_calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(int));
    double *tmpwcount = orpheus_calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
    double *tmpwnorm = orpheus_calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));

    // Setup binning and shift vectors
    double drbin, dbin_lin_inv;
    double *binedges; int *linarr_bins; int *reso_rindedges_base;
    build_radial_helpers(tree, bin, &binedges, &linarr_bins, &reso_rindedges_base, &drbin, &dbin_lin_inv);
    int *rshift_index_matcher, *rshift_pixs_galind_bounds, *rshift_pix_gals;
    build_flat_rshifts(cat, nav, &rshift_index_matcher, &rshift_pixs_galind_bounds, &rshift_pix_gals);
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){
        free(totcount); free(tmpnpair); free(tmpwcount); free(tmpwnorm); free(binedges);
        free(linarr_bins); free(reso_rindedges_base); free(rshift_index_matcher);
        free(rshift_pixs_galind_bounds); free(rshift_pix_gals);
        return;
    }

    // Prepare vars for parallel region
    int nfilledregions = nav->nfilledregions;
    int *filledregions = nav->filledregions;
    int nregionsdone = 0;
    reset_progress();

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

                                bin_accumulate(dist, w_gal1, w_gal2, z_gal1, z_gal2,
                                               bin->rmin, binedges, linarr_bins, dbin_lin_inv,
                                               nbinsz, nbinsz, nbinsr, elthread, nthreads,
                                               tmpnpair, tmpwcount, tmpwnorm, 0, NULL, NULL);
                            }
                        }
                    }
                }
            }
            #pragma omp atomic
            nregionsdone += 1;
            print_progress(nregionsdone, nfilledregions, verbose);
        }
    }

    free(binedges); free(linarr_bins); free(reso_rindedges_base);
    free(rshift_index_matcher); free(rshift_pixs_galind_bounds); free(rshift_pix_gals);

    bin_reduce(nbinsz, nbinsz, nbinsr, nthreads, totcount, tmpnpair, tmpwcount, tmpwnorm, 0, NULL, out);
    if (verbose>0){ printf("\n"); }

    free(totcount); free(tmpwcount); free(tmpnpair); free(tmpwnorm);
}

// Full-sky DoubleTree estimtor of 2pt pair counts
static void _nn_spherical(const MultiresoCatalog *cat, const NavHash *nav,
                           const TreeResoParams *tree, const BinningParams *bin,
                           int nthreads, int verbose, NPCFOutput *out){

    int nbinsz = cat->nbinsz, nbinsr = bin->nbinsr, nresos = tree->nresos;
    double *totcount = orpheus_calloc(nbinsz*nbinsz*nbinsr, sizeof(double));
    int *tmpnpair = orpheus_calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(int));
    double *tmpwcount = orpheus_calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
    double *tmpwnorm = orpheus_calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));

    double drbin, dbin_lin_inv;
    double *binedges; int *linarr_bins; int *reso_rindedges;
    build_radial_helpers(tree, bin, &binedges, &linarr_bins, &reso_rindedges, &drbin, &dbin_lin_inv);
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){
        free(totcount); free(tmpnpair); free(tmpwcount); free(tmpwnorm); free(binedges);
        free(linarr_bins); free(reso_rindedges);
        return;
    }

    // Progress is tracked per central galaxy across all processed resolutions
    int nregionsdone = 0, progtot = 0;
    for (int elreso=0; elreso<nresos; elreso++){
        if (reso_rindedges[elreso+1] > reso_rindedges[elreso]){ progtot += cat->ngal_resos[elreso]; }
    }
    if (progtot <= 0){ progtot = 1; }
    reset_progress();

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
            long *ranges = orpheus_malloc(2*cap*sizeof(long));
            #pragma omp for schedule(dynamic, 64)
            for (int i1=0; i1<n1; i1++){
                #pragma omp atomic
                nregionsdone += 1;
                print_progress(nregionsdone, progtot, verbose);
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
                            // When we do not double count we use that query_disc is symmetric; so for
                            // the same reso when keeping only the ordering with g2>g1 eliminates dc.
                            if (bin->do_dc==0 && elreso_leaf==elreso && g2 <= g1){ continue; }
                            double dist = sphere_dist(cx, cy, cz, cat->vx_resos[g2], cat->vy_resos[g2], cat->vz_resos[g2]);
                            if (dist < rmin_reso || dist >= rmax_reso){ continue; }
                            bin_accumulate(dist, w1, cat->weight_resos[g2], z1, cat->zbin_resos[g2],
                                           bin->rmin, binedges, linarr_bins, dbin_lin_inv,
                                           nbinsz, nbinsz, nbinsr, thread, nthreads,
                                           tmpnpair, tmpwcount, tmpwnorm, 0, NULL, NULL);
                        }
                        ci++;
                    }
                }
            }
            free(ranges);
        }
    }
    free(binedges); free(linarr_bins); free(reso_rindedges);

    bin_reduce(nbinsz, nbinsz, nbinsr, nthreads, totcount, tmpnpair, tmpwcount, tmpwnorm, 0, NULL, out);
    if (verbose>0){ printf("\n"); }
    free(totcount); free(tmpwcount); free(tmpnpair); free(tmpwnorm);
}


// Public entry point: Choose function based on passed metric.
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
}

///////////////////////////
// GG CORRELATOR CLASSES //
///////////////////////////


// Flat-sky DoubleTree estimator of the shear 2PCFs in the xipm-basis
static void _gg_flat(const MultiresoCatalog *cat, const NavHash *nav,
                     const TreeResoParams *tree, const BinningParams *bin,
                     int nthreads, int verbose, NPCFOutput *out){

    int nbinsz = cat->nbinsz, nbinsr = bin->nbinsr, nresos = tree->nresos;
    double *totcount = orpheus_calloc(nbinsz*nbinsz*nbinsr, sizeof(double));
    int *tmpnpair = orpheus_calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(int));
    double *tmpwcount = orpheus_calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
    double *tmpwnorm = orpheus_calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
    double complex *tmpcomp = orpheus_calloc(nthreads*2*nbinsz*nbinsz*nbinsr, sizeof(double complex));

    double drbin, dbin_lin_inv;
    double *binedges; int *linarr_bins; int *reso_rindedges_base;
    build_radial_helpers(tree, bin, &binedges, &linarr_bins, &reso_rindedges_base, &drbin, &dbin_lin_inv);
    int *rshift_index_matcher, *rshift_pixs_galind_bounds, *rshift_pix_gals;
    build_flat_rshifts(cat, nav, &rshift_index_matcher, &rshift_pixs_galind_bounds, &rshift_pix_gals);
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){
        free(totcount); free(tmpnpair); free(tmpwcount); free(tmpwnorm); free(tmpcomp);
        free(binedges); free(linarr_bins); free(reso_rindedges_base); free(rshift_index_matcher);
        free(rshift_pixs_galind_bounds); free(rshift_pix_gals);
        return;
    }

    int nfilledregions = nav->nfilledregions;
    int *filledregions = nav->filledregions;
    int nregionsdone = 0;
    reset_progress();

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
                                double complex comps[2] = {wshape_gal1*conj(wshape_gal2),
                                                           wshape_gal1*wshape_gal2*phirotc_sq*phirotc_sq};
                                bin_accumulate(dist, w_gal1, w_gal2, z_gal1, z_gal2,
                                               bin->rmin, binedges, linarr_bins, dbin_lin_inv,
                                               nbinsz, nbinsz, nbinsr, elthread, nthreads,
                                               tmpnpair, tmpwcount, tmpwnorm, 2, comps, tmpcomp);
                            }
                        }
                    }
                }
            }
            #pragma omp atomic
            nregionsdone += 1;
            print_progress(nregionsdone, nfilledregions, verbose);
        }
    }

    free(binedges); free(linarr_bins); free(reso_rindedges_base);
    free(rshift_index_matcher); free(rshift_pixs_galind_bounds); free(rshift_pix_gals);

    bin_reduce(nbinsz, nbinsz, nbinsr, nthreads, totcount, tmpnpair, tmpwcount, tmpwnorm, 2, tmpcomp, out);
    if (verbose>0){ printf("\n"); }
    free(totcount); free(tmpwcount); free(tmpnpair); free(tmpwnorm); free(tmpcomp);
}

// Full-sky DoubleTree estimator of the shear 2PCFs in the xipm-basis
static void _gg_spherical(const MultiresoCatalog *cat, const NavHash *nav,
                          const TreeResoParams *tree, const BinningParams *bin,
                          int nthreads, int verbose, NPCFOutput *out){

    int nbinsz = cat->nbinsz, nbinsr = bin->nbinsr, nresos = tree->nresos;
    double *totcount = orpheus_calloc(nbinsz*nbinsz*nbinsr, sizeof(double));
    int *tmpnpair = orpheus_calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(int));
    double *tmpwcount = orpheus_calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
    double *tmpwnorm = orpheus_calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
    double complex *tmpcomp = orpheus_calloc(nthreads*2*nbinsz*nbinsz*nbinsr, sizeof(double complex));

    double drbin, dbin_lin_inv;
    double *binedges; int *linarr_bins; int *reso_rindedges;
    build_radial_helpers(tree, bin, &binedges, &linarr_bins, &reso_rindedges, &drbin, &dbin_lin_inv);
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){
        free(totcount); free(tmpnpair); free(tmpwcount); free(tmpwnorm); free(tmpcomp);
        free(binedges); free(linarr_bins); free(reso_rindedges);
        return;
    }

    // Progress is tracked per central galaxy across all processed resolutions
    int nregionsdone = 0, progtot = 0;
    for (int elreso=0; elreso<nresos; elreso++){
        if (reso_rindedges[elreso+1] > reso_rindedges[elreso]){ progtot += cat->ngal_resos[elreso]; }
    }
    if (progtot <= 0){ progtot = 1; }
    reset_progress();

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
            long *ranges = orpheus_malloc(2*cap*sizeof(long));
            #pragma omp for schedule(dynamic, 64)
            for (int i1=0; i1<n1; i1++){
                #pragma omp atomic
                nregionsdone += 1;
                print_progress(nregionsdone, progtot, verbose);
                long g1 = red1_off + i1;
                if (cat->isinner_resos[g1] < 1e-5){ continue; }
                double cx = cat->vx_resos[g1], cy = cat->vy_resos[g1], cz = cat->vz_resos[g1];
                double w1 = cat->isinner_resos[g1]*cat->weight_resos[g1];
                int z1 = cat->zbin_resos[g1];
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
                            // When we do not double count we use that query_disc is symmetric; so for
                            // the same reso when keeping only the ordering with g2>g1 eliminates dc.
                            if (bin->do_dc==0 && elreso_leaf==elreso && g2 <= g1){ continue; }
                            double dist = sphere_dist(cx, cy, cz, cat->vx_resos[g2], cat->vy_resos[g2], cat->vz_resos[g2]);
                            if (dist < rmin_reso || dist >= rmax_reso){ continue; }
                            double vx2 = cat->vx_resos[g2], vy2 = cat->vy_resos[g2], vz2 = cat->vz_resos[g2];
                            double complex wshape2 = (double complex) cat->weight_resos[g2] * (cat->e1_resos[g2]+I*cat->e2_resos[g2]);
                            BearingAB g12 = bearing_AB_cart(cx,cy,cz, vx2,vy2,vz2);
                            BearingAB g21 = bearing_AB_cart(vx2,vy2,vz2, cx,cy,cz);
                            double complex proj1 = wshape1 * bearing_rc(g12);
                            double complex proj2 = wshape2 * bearing_rc(g21);
                            double complex comps[2] = {proj1*conj(proj2), proj1*proj2};
                            bin_accumulate(dist, w1, cat->weight_resos[g2], z1, cat->zbin_resos[g2],
                                           bin->rmin, binedges, linarr_bins, dbin_lin_inv,
                                           nbinsz, nbinsz, nbinsr, thread, nthreads,
                                           tmpnpair, tmpwcount, tmpwnorm, 2, comps, tmpcomp);
                        }
                        ci++;
                    }
                }
            }
            free(ranges);
        }
    }
    free(binedges); free(linarr_bins); free(reso_rindedges);

    bin_reduce(nbinsz, nbinsz, nbinsr, nthreads, totcount, tmpnpair, tmpwcount, tmpwnorm, 2, tmpcomp, out);
    if (verbose>0){ printf("\n"); }
    free(totcount); free(tmpwcount); free(tmpnpair); free(tmpwnorm); free(tmpcomp);
}

// Public entry point: Choose function based on passed metric.
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
}


///////////////////////////
// NG CORRELATOR CLASSES //
///////////////////////////


// Flat-sky DoubleTree estimator of the NG correlators in the -(gamma_t, gamma_x) basis.
static void _ng_flat(const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                     const MultiresoCatalog *cat_source, const NavHash *nav_source,
                     const TreeResoParams *tree, const BinningParams *bin,
                     int nthreads, int verbose, NPCFOutput *out){

    int nbinsz_l = cat_lens->nbinsz, nbinsz_s = cat_source->nbinsz;
    int nbinsr = bin->nbinsr, nresos = tree->nresos;
    int nzzr = nbinsz_l*nbinsz_s*nbinsr;
    double *totcount = orpheus_calloc(nzzr, sizeof(double));
    int *tmpnpair = orpheus_calloc(nthreads*nzzr, sizeof(int));
    double *tmpwcount = orpheus_calloc(nthreads*nzzr, sizeof(double));
    double *tmpwnorm = orpheus_calloc(nthreads*nzzr, sizeof(double));
    double complex *tmpcomp = orpheus_calloc(nthreads*1*nzzr, sizeof(double complex));

    double drbin, dbin_lin_inv;
    double *binedges; int *linarr_bins; int *reso_rindedges_base;
    build_radial_helpers(tree, bin, &binedges, &linarr_bins, &reso_rindedges_base, &drbin, &dbin_lin_inv);
    int *rsim_l, *rspgb_l, *rspg_l;
    build_flat_rshifts(cat_lens, nav_lens, &rsim_l, &rspgb_l, &rspg_l);
    int *rsim_s, *rspgb_s, *rspg_s;
    build_flat_rshifts(cat_source, nav_source, &rsim_s, &rspgb_s, &rspg_s);
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){
        free(totcount); free(tmpnpair); free(tmpwcount); free(tmpwnorm); free(tmpcomp);
        free(binedges); free(linarr_bins); free(reso_rindedges_base); free(rsim_l); free(rspgb_l);
        free(rspg_l); free(rsim_s); free(rspgb_s); free(rspg_s);
        return;
    }

    int nfilledregions = nav_lens->nfilledregions;
    int *filledregions = nav_lens->filledregions;
    int nregionsdone = 0;
    reset_progress();

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
                                double complex comps[1] = {w_gal1*wshape_gal2*phirotc_sq};
                                bin_accumulate(dist, w_gal1, w_gal2, z_gal1, z_gal2,
                                               bin->rmin, binedges, linarr_bins, dbin_lin_inv,
                                               nbinsz_l, nbinsz_s, nbinsr, elthread, nthreads,
                                               tmpnpair, tmpwcount, tmpwnorm, 1, comps, tmpcomp);
                            }
                        }
                    }
                }
            }
            #pragma omp atomic
            nregionsdone += 1;
            print_progress(nregionsdone, nfilledregions, verbose);
        }
    }

    free(binedges); free(linarr_bins); free(reso_rindedges_base);
    free(rsim_l); free(rspgb_l); free(rspg_l);
    free(rsim_s); free(rspgb_s); free(rspg_s);

    bin_reduce(nbinsz_l, nbinsz_s, nbinsr, nthreads, totcount, tmpnpair, tmpwcount, tmpwnorm, 1, tmpcomp, out);
    if (verbose>0){ printf("\n"); }
    free(totcount); free(tmpwcount); free(tmpnpair); free(tmpwnorm); free(tmpcomp);
}

// Public entry point for the position-shape 2PCF. Flat-sky only for now...
void alloc_ng_doubletree(const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                         const MultiresoCatalog *cat_source, const NavHash *nav_source,
                         const TreeResoParams *tree, const BinningParams *bin,
                         int nthreads, int verbose, NPCFOutput *out){
    _ng_flat(cat_lens, nav_lens, cat_source, nav_source, tree, bin, nthreads, verbose, out);
}


/////////////////////////
// Slab-hashed GN pairs //
/////////////////////////
// Discrete estimator of a NG correlator in slabs of a 3dbox geometry
void ng_slab(const MultiresoCatalog *cat_lens, const MultiresoCatalog *cat_source,
             const NavHash *nav_hash, const BinningParams *bin,
             int self_pairs, int has_shapes, int nthreads, int verbose, NPCFOutput *out)
{
    // Dereference input args
    double *scalar_pos1 = cat_lens->pos1_resos, *scalar_pos2 = cat_lens->pos2_resos, *scalar_pos3 = cat_lens->pos3_resos;
    double *scalar_w = cat_lens->weight_resos; int *scalar_zbin = cat_lens->zbin_resos;
    int scalar_ngal = cat_lens->ngal_resos[0], nbinsz_scalar = cat_lens->nbinsz;
    double *polar_pos1 = cat_source->pos1_resos, *polar_pos2 = cat_source->pos2_resos, *polar_pos3 = cat_source->pos3_resos;
    double *polar_w = cat_source->weight_resos; int *polar_zbin = cat_source->zbin_resos;
    double *polar_e1 = cat_source->e1_resos, *polar_e2 = cat_source->e2_resos; int nbinsz_polar = cat_source->nbinsz;
    int nslabs = nav_hash->nslabs; double z0 = nav_hash->z0, dpix_z = nav_hash->dpix_z;
    double pix1_start = nav_hash->pix1_start, pix1_d = nav_hash->pix1_d; int pix1_n = nav_hash->pix1_n;
    double pix2_start = nav_hash->pix2_start, pix2_d = nav_hash->pix2_d; int pix2_n = nav_hash->pix2_n;
    int *slab_offsets = nav_hash->slab_offsets, *index_matcher = nav_hash->index_matcher;
    int *pixs_galind_bounds = nav_hash->pixs_galind_bounds, *rshift_bounds = nav_hash->rshift_bounds, *pix_gals = nav_hash->pix_gals;
    double rmin = bin->rmin, rmax = bin->rmax, Pi = bin->Pi; int nbinsr = bin->nbinsr;
    double complex *out_xs = out->npcf;
    double *out_wnorm = out->norm, *out_rsum = out->bin_centers;
    long long int *out_npairs = out->npair;

    int npix = pix1_n * pix2_n;
    int nbinszz = nbinsz_scalar * nbinsz_polar;
    int nout = nbinszz * nbinsr;

    double rmin2 = rmin * rmin;
    double rmax2 = rmax * rmax;
    double dlnr_inv = nbinsr / log(rmax / rmin);

    // Per-thread accumulators.
    double *tmp_xs_re = orpheus_calloc((size_t)nthreads * nout, sizeof(double));
    double *tmp_xs_im = orpheus_calloc((size_t)nthreads * nout, sizeof(double));
    double *tmp_wnorm = orpheus_calloc((size_t)nthreads * nout, sizeof(double));
    double *tmp_rsum  = orpheus_calloc((size_t)nthreads * nout, sizeof(double));
    long   *tmp_npair = orpheus_calloc((size_t)nthreads * nout, sizeof(long));
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){
        free(tmp_xs_re); free(tmp_xs_im); free(tmp_wnorm); free(tmp_rsum); free(tmp_npair);
        return;
    }

    int nregionsdone = 0;
    reset_progress();

    #pragma omp parallel num_threads(nthreads)
    {
        int thread = omp_get_thread_num();
        size_t base = (size_t)thread * nout;

        #pragma omp for schedule(dynamic, 256)
        for (int i=0; i<scalar_ngal; i++) {
            #pragma omp atomic
            nregionsdone += 1;
            print_progress(nregionsdone, scalar_ngal, verbose);

            double p1 = scalar_pos1[i];
            double p2 = scalar_pos2[i];
            double z_scalar = scalar_pos3[i];
            double w_scalar = scalar_w[i];
            int zbin_scalar = scalar_zbin[i];

            // Slabs overlapping [z_scalar-Pi, z_scalar+Pi].
            int s_lo = (int)floor((z_scalar - Pi - z0) / dpix_z);
            int s_hi = (int)floor((z_scalar + Pi - z0) / dpix_z);
            if (s_lo < 0) s_lo = 0;
            if (s_hi > nslabs - 1) s_hi = nslabs - 1;

            // Transverse search box enclosing all neighbours within rmax.
            int pix1_lo = (int)floor((p1 - (rmax + pix1_d) - pix1_start) / pix1_d);
            int pix1_hi = (int)floor((p1 + (rmax + pix1_d) - pix1_start) / pix1_d);
            int pix2_lo = (int)floor((p2 - (rmax + pix2_d) - pix2_start) / pix2_d);
            int pix2_hi = (int)floor((p2 + (rmax + pix2_d) - pix2_start) / pix2_d);
            pix1_lo = mymax(0, pix1_lo); pix1_hi = mymin(pix1_n - 1, pix1_hi);
            pix2_lo = mymax(0, pix2_lo); pix2_hi = mymin(pix2_n - 1, pix2_hi);

            for (int s=s_lo; s<=s_hi; s++) {
                int matcher_shift = s * npix;
                int bounds_shift = rshift_bounds[s];
                int gals_shift = slab_offsets[s];
                for (int ip1=pix1_lo; ip1<=pix1_hi; ip1++) {
                    for (int ip2 = pix2_lo; ip2 <= pix2_hi; ip2++) {
                        int ind_raw = ip2*pix1_n + ip1;
                        int ind_red = index_matcher[matcher_shift + ind_raw];
                        if (ind_red == -1){continue;}
                        int lower = pixs_galind_bounds[bounds_shift + ind_red];
                        int upper = pixs_galind_bounds[bounds_shift + ind_red + 1];
                        for (int k=lower; k<upper; k++) {
                            int j = pix_gals[gals_shift + k];
                            if (self_pairs && j==i){continue;}
                            double rel1 = polar_pos1[j]-p1;
                            double rel2 = polar_pos2[j]-p2;
                            double d2 = rel1*rel1 + rel2*rel2;
                            if (d2<rmin2 || d2>=rmax2){continue;}
                            double dz=fabs(polar_pos3[j]-z_scalar);
                            if (dz >= Pi){continue;}
                            double r = sqrt(d2);
                            int rbin = (int)floor(log(r/rmin) * dlnr_inv);
                            if (rbin<0 || rbin>=nbinsr){continue;}
                            double w = w_scalar * polar_w[j];
                            int outind = (zbin_scalar*nbinsz_polar + polar_zbin[j])*nbinsr + rbin;
                            size_t ind = base + (size_t)outind;
                            tmp_wnorm[ind] += w;
                            tmp_rsum[ind] += w * r;
                            tmp_npair[ind] += 1;
                            if (has_shapes) {
                                double phi_re = (rel1*rel1 - rel2*rel2) / d2;
                                double phi_im = -2.0*rel1*rel2/d2;
                                double e1 = polar_e1[j];
                                double e2 = polar_e2[j];
                                tmp_xs_re[ind] += w * (e1*phi_re - e2 *phi_im);
                                tmp_xs_im[ind] += w * (e1*phi_im + e2 *phi_re);
                            }
                        }
                    }
                }
            }
        }
    }
    // Reduce per-thread accumulators into the output arrays.
    for (int o=0; o<nout; o++) {
        for (int t=0; t<nthreads; t++) {
            size_t ind = (size_t)t * nout + o;
            out_xs[o]     += tmp_xs_re[ind] + I*tmp_xs_im[ind];
            out_wnorm[o]  += tmp_wnorm[ind];
            out_rsum[o]   += tmp_rsum[ind];
            out_npairs[o] += tmp_npair[ind];
        }
    }
    if (verbose>0){ printf("\n"); }
    free(tmp_xs_re);
    free(tmp_xs_im);
    free(tmp_wnorm);
    free(tmp_rsum);
    free(tmp_npair);
}