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


////////////////////////////////
// Curved-sky aperture masses //
////////////////////////////////

// Aperture mass map on a fixed set of curved-sky aperture centers.
void aperturemassmap_spherical(
    const MultiresoCatalog *cat, const NavHash *nav, const TreeResoParams *tree,
    double R_ap, int ind_filter,
    const double *cvx, const double *cvy, const double *cvz, long ncenters,
    const double *mask, long nside_mask,
    int nthreads, int verbose,
    double complex *out_Map, double *out_norm, double *out_normQ, double *out_cov){

    int nbinsz = cat->nbinsz, nresos = tree->nresos;
    double supp = getFilterSupp(ind_filter);
    double supp2 = supp*supp;
    double rmax_ap = supp*R_ap;
    double R2_ap = R_ap*R_ap;
    int has_mask = (nside_mask > 0) && (mask != NULL);

    int nregionsdone = 0, progtot = (int) ncenters;
    if (progtot <= 0){ progtot = 1; }
    reset_progress();

    #pragma omp parallel num_threads(nthreads)
    {
        long cap = 2048;
        long *ranges = orpheus_malloc(2*cap*sizeof(long));
        #pragma omp for schedule(dynamic, 64)
        for (long ic=0; ic<ncenters; ic++){
            #pragma omp atomic
            nregionsdone += 1;
            print_progress(nregionsdone, progtot, verbose);
            if (orpheus_get_error()){ continue; }
            double cx = cvx[ic], cy = cvy[ic], cz = cvz[ic];
            double v1[3] = {cx, cy, cz};
            for (int z=0; z<nbinsz; z++){
                long ind = (long)z*ncenters + ic;
                out_Map[ind] = 0.; out_norm[ind] = 0.; out_normQ[ind] = 0.;
            }
            out_cov[ic] = 0.; out_cov[ncenters+ic] = 0.;

            // Get masked area fraction of this aperture, raw and Q-weighted
            if (has_mask){
                double npix_m = 0., npix_t = 0., Qpix_m = 0., Qpix_t = 0.;
                long nr = hpx_query_disc_nest_ranges(nside_mask, v1, rmax_ap, ranges, cap);
                if (nr > cap){ cap = nr; ranges = realloc(ranges, 2*cap*sizeof(long));
                               nr = hpx_query_disc_nest_ranges(nside_mask, v1, rmax_ap, ranges, cap); }
                for (long r=0; r<nr; r++){
                    for (long p=ranges[2*r]; p<ranges[2*r+1]; p++){
                        double pv[3];
                        hpx_pix2vec_nest(nside_mask, p, pv);
                        double dist = sphere_dist(cx, cy, cz, pv[0], pv[1], pv[2]);
                        // query_disc is inclusive, so the exact geodesic cut is still needed
                        if (dist >= rmax_ap){ continue; }
                        double Qpix = getFilterQ(ind_filter, dist*dist/R2_ap);
                        npix_m += mask[p]; npix_t += 1.;
                        Qpix_m += mask[p]*Qpix; Qpix_t += Qpix;
                    }
                }
                if (npix_t > 0.){ out_cov[ic] = npix_m/npix_t; }
                if (Qpix_t != 0.){ out_cov[ncenters+ic] = Qpix_m/Qpix_t; }
            }

            // Get aperture mass and normalisation per resolution band
            for (int elreso=0; elreso<nresos; elreso++){
                double rmin_reso = tree->reso_redges[elreso];
                double rmax_reso = mymin(tree->reso_redges[elreso+1], rmax_ap);
                if (rmax_reso <= rmin_reso){ continue; }
                long ns = nav->nside_nav[elreso];
                long red_off = nav->rshift_red[elreso];
                const long *cellpix = nav->cell_pix + nav->rshift_cellpix[elreso];
                const int  *bounds  = nav->cell_redbounds + nav->rshift_cellbounds[elreso];
                int ncells = nav->ncells_resos[elreso];

                long nr = hpx_query_disc_nest_ranges(ns, v1, rmax_reso, ranges, cap);
                if (nr > cap){ cap = nr; ranges = realloc(ranges, 2*cap*sizeof(long));
                               nr = hpx_query_disc_nest_ranges(ns, v1, rmax_reso, ranges, cap); }
                int ci = 0;
                for (long r=0; r<nr; r++){
                    long plo = ranges[2*r], phi = ranges[2*r+1];
                    int loi = ci, hii = ncells;
                    while (loi < hii){ int m = (loi+hii)>>1;
                        if (cellpix[m] < plo){ loi = m+1; } else { hii = m; } }
                    ci = loi;
                    while (ci < ncells && cellpix[ci] < phi){
                        int lo = bounds[ci], hi = bounds[ci+1];
                        for (int j=lo; j<hi; j++){
                            long g = red_off + j;
                            double gx = cat->vx_resos[g], gy = cat->vy_resos[g], gz = cat->vz_resos[g];
                            double dist = sphere_dist(cx, cy, cz, gx, gy, gz);
                            if (dist < rmin_reso || dist >= rmax_reso){ continue; }
                            // Avoid zero-divisions
                            if (dist < 1e-8*R_ap){ continue; }
                            double w = cat->weight_resos[g];
                            int z = cat->zbin_resos[g];
                            double Q = getFilterQ(ind_filter, dist*dist/R2_ap);
                            // e_t + i*e_x = -(e1 + i*e2) * exp(-2i*phi_{galaxy->center})
                            double complex etx = -(cat->e1_resos[g] + I*cat->e2_resos[g]) *
                                                 bearing_rc(bearing_AB_cart(gx,gy,gz, cx,cy,cz));
                            long ind = (long)z*ncenters + ic;
                            out_Map[ind] += w*Q*etx;
                            out_norm[ind] += w;
                            out_normQ[ind] += w*Q;
                        }
                        ci++;
                    }
                }
            }

            // Normalisation, eqns (22),(23) of arXiv:2106.04594 at first order.
            for (int z=0; z<nbinsz; z++){
                long ind = (long)z*ncenters + ic;
                if (out_norm[ind] > 0.){ out_Map[ind] *= supp2/out_norm[ind]; }
                else { out_Map[ind] = 0.; }
            }
        }
        free(ranges);
    }
    if (verbose>0){ printf("\n"); }
}
