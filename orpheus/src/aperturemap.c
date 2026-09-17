#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <complex.h>

#include "utils.h"
#include "healpix_utils.h"
#include "multires_structs.h"
#include "directestimator.h"
#include "aperturemap.h"


///////////////////////////////////
// Curved-sky aperture mass maps //
///////////////////////////////////

// Nested pixel ranges of a disc
// * We grow the buffer only on demand
// * exact keeps only the pixels centred inside the disc, else all pixels overlapping it
static long disc_ranges(long nside, const double *v, double radius, int exact,
                        long **ranges, long *cap){
    long n_ranges = exact ? hpx_query_disc_nest_ranges_exact(nside, v, radius, *ranges, *cap)
                    : hpx_query_disc_nest_ranges(nside, v, radius, *ranges, *cap);
    if (n_ranges > *cap){
        *cap = n_ranges; *ranges = realloc(*ranges, 2*(*cap)*sizeof(long));
        n_ranges = exact ? hpx_query_disc_nest_ranges_exact(nside, v, radius, *ranges, *cap)
                   : hpx_query_disc_nest_ranges(nside, v, radius, *ranges, *cap);
    }
    return n_ranges;
}

// Integer floor log2 of an integer
static int log2long(long n){
    int k=0;
    while (n>1){n >>= 1; k++;}
    return k;
}

// Get aperture mass map on the sphere.
void aperturemassmap_spherical(
    const MultiresoCatalog *cat, const NavHash *nav, const TreeResoParams *tree,
    double R_ap, int ind_filter,
    const double *cvx, const double *cvy, const double *cvz, long ncenters,
    const double *mask, long nside_mask,
    int nthreads, int verbose,
    double complex *out_Map, double *out_norm, double *out_normQ, double *out_var,
    double *out_cov){

    int nbinsz = cat->nbinsz, nresos = tree->nresos;
    double supp = getFilterSupp(ind_filter);
    double supp2 = supp*supp;
    double rmax_ap = supp*R_ap;
    double R2_ap = R_ap*R_ap;
    int has_mask = (nside_mask > 0) && (mask != NULL);

    // Some preparations in case a mask is present
    double Qtot_ap = 0.;
    float *mask_ring = NULL;
    if (has_mask){
        // Get Q-weight of a fully unmasked aperture on the mask grid,
        // normalised as (pi R^2/A_pix) * int_0^supp^2 Q(x) dx.
        int nq = 4096;
        double dx = supp2/nq;
        for (int i=0; i<nq; i++){ Qtot_ap += getFilterQ(ind_filter, (i+0.5)*dx); }
        Qtot_ap *= dx*3.*(double)nside_mask*(double)nside_mask*R2_ap;
        // Transform mask from nest to ring scheme. This makes the coverage pass
        // of the mask much faster
        long npix_mask = 12*nside_mask*nside_mask, chunk = 1<<16;
        mask_ring = orpheus_malloc(npix_mask*sizeof(float));
        if (mask_ring != NULL){
            #pragma omp parallel num_threads(nthreads)
            {
                long *ring = malloc(chunk*sizeof(long));
                #pragma omp for schedule(static)
                for (long p0=0; p0<npix_mask; p0+=chunk){
                    long n = mymin(chunk, npix_mask-p0);
                    hpx_nest2ring_range(nside_mask, p0, n, ring);
                    for (long i=0; i<n; i++){ mask_ring[ring[i]] = (float) mask[p0+i]; }
                }
                free(ring);
            }
        }
        has_mask = (mask_ring != NULL);
    }

    int nregionsdone = 0, progtot = (int) ncenters;
    if (progtot <= 0){ progtot = 1; }
    reset_progress();

    #pragma omp parallel num_threads(nthreads)
    {
        long cap = 2048, cap_excl = 2048;
        long *ranges = orpheus_malloc(2*cap*sizeof(long));
        long *excl = orpheus_malloc(2*cap_excl*sizeof(long));
        #pragma omp for schedule(dynamic, 64)
        for (long ic=0; ic<ncenters; ic++){
            #pragma omp atomic
            nregionsdone += 1;
            print_progress(nregionsdone, progtot, verbose);
            if (orpheus_get_error()){ continue; }
            // Preparations
            double cx = cvx[ic], cy = cvy[ic], cz = cvz[ic];
            double v1[3] = {cx, cy, cz};
            for (int z=0; z<nbinsz; z++){
                long ind = (long)z*ncenters + ic;
                out_Map[ind] = 0.; out_norm[ind] = 0.; out_normQ[ind] = 0.; out_var[ind] = 0.;
            }
            out_cov[ic] = 0.; out_cov[ncenters+ic] = 0.;

            // Get masked area fraction of this aperture, raw and Q-weighted.
            // We use the ring scheme and only need to do the distance calc for masked pixels
            if (has_mask){
                double npix_m = 0., npix_t = 0., Qpix_m = 0.;
                long n_ranges = hpx_query_disc_ring_ranges_exact(nside_mask, v1, rmax_ap, ranges, cap);
                if (n_ranges > cap){ cap = n_ranges; ranges = realloc(ranges, 2*cap*sizeof(long));
                               n_ranges = hpx_query_disc_ring_ranges_exact(nside_mask, v1, rmax_ap, ranges, cap); }
                for (long r=0; r<n_ranges; r++){
                    npix_t += (double)(ranges[2*r+1] - ranges[2*r]);
                    for (long p=ranges[2*r]; p<ranges[2*r+1]; p++){
                        if (mask_ring[p] <= 0.f){ continue; }
                        double pv[3];
                        hpx_pix2vec_ring(nside_mask, p, pv);
                        double dist = sphere_dist(cx, cy, cz, pv[0], pv[1], pv[2]);
                        npix_m += mask_ring[p];
                        Qpix_m += mask_ring[p]*getFilterQ(ind_filter, dist*dist/R2_ap);
                    }
                }
                if (npix_t > 0.){ out_cov[ic] = npix_m/npix_t; }
                if (Qtot_ap > 0.){ out_cov[ncenters+ic] = mymin(Qpix_m/Qtot_ap, 1.); }
            }

            // Get aperture mass and normalisation per resolution band
            for (int elreso=0; elreso<nresos; elreso++){
                // Band setup
                double rmin_reso = tree->reso_redges[elreso];
                double rmax_reso = mymin(tree->reso_redges[elreso+1], rmax_ap);
                if (rmax_reso <= rmin_reso){ continue; }
                long nside_band = nav->nside_nav[elreso];
                long red_off = nav->rshift_red[elreso];
                const long *cellpix = nav->cell_pix + nav->rshift_cellpix[elreso];
                const int  *bounds  = nav->cell_redbounds + nav->rshift_cellbounds[elreso];
                int ncells = nav->ncells_resos[elreso];
                int last = (elreso == nresos-1);

                // Get candidate cells per band.
                // To avoid double counting/skipping gals we do the following
                // * For all but the last band we check whether the cell centre of the next-coarser
                //   reso lies inside the band's outer edge; we only keep the children of the ones
                //   where this is true
                // * Obviously this doesnt work for the last band so there we do an inclusive search
                //   and cut the tracers at rmax_ap by distance
                long n_ranges;
                if (last){ n_ranges = disc_ranges(nside_band, v1, rmax_reso, 0, &ranges, &cap); }
                else {
                    long nside_parent = nav->nside_nav[elreso+1];
                    int shift = 2*(log2long(nside_band) - log2long(nside_parent)); // doublings between the grids
                    n_ranges = disc_ranges(nside_parent, v1, rmax_reso, 1, &ranges, &cap);
                    for (long r=0; r<2*n_ranges; r++){ ranges[r] <<= shift; }
                }
                // Get cells centred inside the inner edge that went to the band below
                long n_excl = 0, ind_excl = 0;
                if (elreso > 0){ n_excl = disc_ranges(nside_band, v1, rmin_reso, 1, &excl, &cap_excl); }

                // Loop over the candidate ranges, the cells in each range, and the tracers of
                // each cell that is not excluded
                int cellind = 0;
                for (long r=0; r<n_ranges; r++){
                    // Get range and index of the first cell inside it
                    long pixlo = ranges[2*r], pixhi = ranges[2*r+1];
                    int ind_lo = cellind, ind_hi = ncells;
                    while (ind_lo<ind_hi){ int m = (ind_lo+ind_hi)>>1;
                        if (cellpix[m] < pixlo){ ind_lo = m+1; } else { ind_hi = m; } }
                    cellind = ind_lo;
                    while (cellind<ncells && cellpix[cellind]<pixhi){
                        // Skip cells inside the current exclusion range. Pixel ids only grow
                        // along the walk, so the exclusion pointer only moves forward
                        long pix = cellpix[cellind];
                        while (ind_excl < n_excl && excl[2*ind_excl+1] <= pix){ ind_excl++; }
                        if (ind_excl < n_excl && excl[2*ind_excl] <= pix){ cellind++; continue; }

                        // Accumulate tracers within the cell; aperture boundary is only distance cut
                        int lo = bounds[cellind], hi = bounds[cellind+1];
                        for (int j=lo; j<hi; j++){
                            // Get relevant quantities of tracer
                            long g = red_off + j;
                            double gx = cat->vx_resos[g], gy = cat->vy_resos[g], gz = cat->vz_resos[g];
                            double dist = sphere_dist(cx, cy, cz, gx, gy, gz);
                            if (dist >= rmax_ap){ continue; }
                            if (dist < 1e-8*R_ap){ continue; }
                            double w = cat->weight_resos[g];
                            int z = cat->zbin_resos[g];
                            double Q = getFilterQ(ind_filter, dist*dist/R2_ap);
                            double wQ = w*Q;
                            // e_t + i*e_x = -(e1 + i*e2) * exp(-2i*phi_{galaxy->center})
                            double e1 = cat->e1_resos[g], e2 = cat->e2_resos[g];
                            double complex etx = -(e1 + I*e2) *
                                                 bearing_rc(bearing_AB_cart(gx,gy,gz, cx,cy,cz));
                            // Update all the maps
                            long ind = (long)z*ncenters + ic;
                            out_Map[ind] += wQ*etx;
                            out_norm[ind] += w;
                            out_normQ[ind] += wQ;
                            out_var[ind] += wQ*wQ*(e1*e1 + e2*e2);
                        }
                        cellind++;
                    }
                }
            }

            // Normalisation, eqns (22),(23) of arXiv:2106.04594 at first order. Variance
            // only contains local shape noise contribution
            for (int z=0; z<nbinsz; z++){
                long ind = (long)z*ncenters + ic;
                if (out_norm[ind] > 0.){
                    out_Map[ind] *= supp2/out_norm[ind];
                    out_var[ind] *= 0.5*supp2*supp2/(out_norm[ind]*out_norm[ind]);
                }
                else { out_Map[ind] = 0.; out_var[ind] = 0.; }
            }
        }
        free(ranges); free(excl);
    }
    free(mask_ring);
    if (verbose>0){ printf("\n"); }
}
