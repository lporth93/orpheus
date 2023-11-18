#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <complex.h>










void alloc_Gammans_discrete_ggg(
    int* isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int *zbins, int nbinsz, int ngal, 
    int nmin, int nmax, double rmin, double rmax, double *rbins, int nbinsr, int dccorr,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, double *bin_centers, double complex *Gammans, double complex *Gammans_norm);



////////////////////////
/// UNUSED FUNCTIONS ///
////////////////////////
typedef struct {
    int* index_matcher;
    int* pixs_galind_bounds;
    int* pix_gals;
    double pixstarts[2];
    double pixsize[2];
    int gridsize[2];
    int ngal;
    int npix;
} SpatialHash;

void alloc_Gammans_combined(
    int nmin, int nmax, int nthreads, 
    double *weight, double *pos1, double *pos2, double *e1, double *e2, int *zbins, int nbinsz, int ngal,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    double rmin_disc, double rmax_disc, double *rbins_disc, int nbinsr_disc,
    int *gal2stripe, int *gal2redpix, 
    double *weights_pix_stripes, double *e1_pix_stripes, double *e2_pix_stripes,
    int nreso, int nbinsr_grid, int *nbinsr_reso, int *nfieldpix_z_reso_stripe,
    double *Gns_grid_re, double *Gns_grid_im, double *Gnnorm_grid_re, double *Gnnorm_grid_im, 
    int *nGnpix_reso_stripe, int *Gnindices_z_reso_stripe,
    double *bin_centers, double complex *Gammans, double complex *Gammans_norm);

void alloc_Gns_discrete_basic(
    double *weight, double *pos1, double *pos2, double *e1, double *e2, int *zbins, int nbinsz, int ngal, 
    int nmin, int nmax, double rmin, double rmax, int nbinsr,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, double *bin_centers, int *counts, double complex *Gns, double complex *Gns_norm);



/** @brief Computes multipoles of nth multipole of polar 3pcf
 *         for a given set of precomputed Gns. Allocates the
 *         Gns using the discrete methodology and then transforms
 *         them to the blocks of Gammans
 *
 *  The length of the output functions are set in xxx.py
 *
 *  @param weight The weights associated with the galaxies.
 *  @param inner_region Whether galaxy is in interior of patch.
 *         Only relevant for domain decomposed catalogs.
 *  @param pos1 The x-coordinates of the galaxies.
 *  @param pos2 The y-coordinates of the galaxies.
 *  @param e1 The first ellipticity component of the galaxies.
 *  @param e2 The second ellipticity component of the galaxies.
 *  @param zbins The redshift bin the galaxy resides in.
 *  @param nbinsz The number of tomographic redshift bins.
 *  @param ngal The number of galaxies within the patch.
 *  @param nmin The absolute value of the smallest multipole considered.
 *  @param nmax The absolute value of the largest multipole considered.
 *  @param rmin The smallest radial separation considered.
 *  @param rmax The largest radial separation considered.
 *  @param nbinsr The total number of radial separations considered.
 *  @param index_matcher Part of spatial hash (see spatialhash.h).
 *  @param pixs_galind_bounds Part of spatial hash (see spatialhash.h).
 *  @param pix_gals Part of spatial hash (see spatialhash.h).
 *  @param pix1_start Part of spatial hash (see spatialhash.h).
 *  @param pix1_d Part of spatial hash (see spatialhash.h).
 *  @param pix1_n Part of spatial hash (see spatialhash.h).
 *  @param pix2_start Part of spatial hash (see spatialhash.h).
 *  @param pix2_d Part of spatial hash (see spatialhash.h).
 *  @param pix2_n Part of spatial hash (see spatialhash.h).
 *  @param inner_xmin Smallest x value of inner part of patch.
 *  @param inner_xmax Largest x value of inner part of patch.
 *  @param gridmatcher_inds The pixel indices of the 2D grid
 *         used by the FFT corresponding to the selected pixels in the
 *         mass assignment scheme. Only defined on inner domain.
 *         length: npix_scheme*ngal_inner
 *  @param gridmatcher_weights The pixel weights of the 2D grid
 *         used by the FFT corresponding to the selected pixels in the
 *         mass assignment scheme. Only defined on inner domain.
 *         length: npix_scheme*ngal_inner
 *  @param reducedgridmatcher Maps pixel indices of 2D grid to indices of
 *         nonempty pixels of that grid when all galaxies are conidered.
 *         length: npix_2dgrid (empty pixels get assigned index -1).
 *  @param mas_scheme Mass assignment used for data grid : 0:NGP, 1:CIC, 2:TSC
 *  @param npix_inner The number of nonempty pixels within the inner patch domain.
 *  @param nthreads Number of parallel threads used. Only defined on inner domain.
 *  @param bin_centers Center of the radial bins.
 *         length: nbinsz*nbinsr
 *  @param counts Pair counts in the Gn multipoles.
 *         length: nbinsz*nbinsr
 *  @param Gns Discrete Gn multipoles mapped to reduced grid used by FFT.
 *         length: nnvals*nrbins*nzbins^2*npix_reducedgrid (40*20*25*1e6 ~ 2e10 KiDS; 40*20*100*1e6 ~ 8e10 Stage IV)
 *  @param Gns_norm Discrete norm of Gn multipoles mapped to reduced grid
 *         used by FFT.
 *         length: (nmax-nmin+1)*nrbins*nzbins^2*npix_reducedgrid
 *  @param Gammans Discrete multipoles of 3pcf.
 *         length: (nmax-nmin+1)*4*nrbins^2*nzbins^2 (40*4*20^2*5^3 ~ 8e6 KiDS; 40*4*20^2*10^3 ~ 6e7 Stage IV)
 *  @param Gammans_norm Norm of discrete multipoles of 3pcf.
 *     
 */
void alloc_Gnsingle_discrete_basic(
    double *weight, double *pos1, double *pos2, double *e1, double *e2, int *zbins, int nbinsz, int ngal, 
    int n, double rmin, double rmax, int nbinsr,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, double *bin_centers, int *counts, double complex *Gns, double complex *Gns_norm);

void alloc_Gammansingle_discretemixed_basic(
    int *isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int *zbins, int nbinsz, int ngal, 
    int n, double rmin, double rmax, int nbinsr_disc,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    double stripes_start, double stripes_d,  
    int nreso, double *pix1start_reso, double *pix2start_reso, double *pixd_reso, int *nbinsr_reso, 
    int *pix1n_reso, int *pix2n_reso, int *npixred_reso, int *npixbare_reso, int *red_indices,
    double *weight_grid, double *e1_grid, double *e2_grid, double complex *Gns_grid, int nbinsr_grid,
    double *bin_centers_disc, int *counts_disc, double complex *Gamman, double complex *Gamman_norm,
    int nthreads);
    

void alloc_Gns_discrete_basic2(
    double *weight, double *pos1, double *pos2, double *e1, double *e2, int *zbins, int nbinsz, int ngal, 
    int nmin, int nmax, double rmin, double rmax, int nbinsr,
    SpatialHash *hash,
    int nthreads, double *bin_centers, int *counts, double complex *Gns, double complex *Gns_norm);

/** @brief Computes multipoles Gamman using discrete 
 *         methodology and maps the relevant Gns of the
 *         discrete methodology to the reduced grid that
 *         is used by the FFT method.
 *
 *  The length of the output functions are set in xxx.py
 *
 *  @param weight The weights associated with the galaxies.
 *  @param inner_region Whether galaxy is in interior of patch.
 *         Only relevant for domain decomposed catalogs.
 *  @param pos1 The x-coordinates of the galaxies.
 *  @param pos2 The y-coordinates of the galaxies.
 *  @param e1 The first ellipticity component of the galaxies.
 *  @param e2 The second ellipticity component of the galaxies.
 *  @param zbins The redshift bin the galaxy resides in.
 *  @param nbinsz The number of tomographic redshift bins.
 *  @param ngal The number of galaxies within the patch.
 *  @param nmin The absolute value of the smallest multipole considered.
 *  @param nmax The absolute value of the largest multipole considered.
 *  @param rmin The smallest radial separation considered.
 *  @param rmax The largest radial separation considered.
 *  @param nbinsr The total number of radial separations considered.
 *  @param index_matcher Part of spatial hash (see spatialhash.h).
 *  @param pixs_galind_bounds Part of spatial hash (see spatialhash.h).
 *  @param pix_gals Part of spatial hash (see spatialhash.h).
 *  @param pix1_start Part of spatial hash (see spatialhash.h).
 *  @param pix1_d Part of spatial hash (see spatialhash.h).
 *  @param pix1_n Part of spatial hash (see spatialhash.h).
 *  @param pix2_start Part of spatial hash (see spatialhash.h).
 *  @param pix2_d Part of spatial hash (see spatialhash.h).
 *  @param pix2_n Part of spatial hash (see spatialhash.h).
 *  @param inner_xmin Smallest x value of inner part of patch.
 *  @param inner_xmax Largest x value of inner part of patch.
 *  @param gridmatcher_inds The pixel indices of the 2D grid
 *         used by the FFT corresponding to the selected pixels in the
 *         mass assignment scheme. Only defined on inner domain.
 *         length: npix_scheme*ngal_inner
 *  @param gridmatcher_weights The pixel weights of the 2D grid
 *         used by the FFT corresponding to the selected pixels in the
 *         mass assignment scheme. Only defined on inner domain.
 *         length: npix_scheme*ngal_inner
 *  @param reducedgridmatcher Maps pixel indices of 2D grid to indices of
 *         nonempty pixels of that grid when all galaxies are conidered.
 *         length: npix_2dgrid (empty pixels get assigned index -1).
 *  @param mas_scheme Mass assignment used for data grid : 0:NGP, 1:CIC, 2:TSC
 *  @param npix_inner The number of nonempty pixels within the inner patch domain.
 *  @param nthreads Number of parallel threads used. Only defined on inner domain.
 *  @param bin_centers Center of the radial bins.
 *         length: nbinsz*nbinsr
 *  @param counts Pair counts in the Gn multipoles.
 *         length: nbinsz*nbinsr
 *  @param Gns Discrete Gn multipoles mapped to reduced grid used by FFT.
 *         length: nnvals*nrbins*nzbins^2*npix_reducedgrid (40*20*25*1e6 ~ 2e10 KiDS; 40*20*100*1e6 ~ 8e10 Stage IV)
 *  @param Gns_norm Discrete norm of Gn multipoles mapped to reduced grid
 *         used by FFT.
 *         length: (nmax-nmin+1)*nrbins*nzbins^2*npix_reducedgrid
 *  @param Gammans Discrete multipoles of 3pcf.
 *         length: (nmax-nmin+1)*4*nrbins^2*nzbins^2 (40*4*20^2*5^3 ~ 8e6 KiDS; 40*4*20^2*10^3 ~ 6e7 Stage IV)
 *  @param Gammans_norm Norm of discrete multipoles of 3pcf.
 *         length: (nmax-nmin+1)*nrbins^2*nzbins^2
 */
void alloc_GnsGammans_discrete_basic(
    double *weight, int *inner_region, double *pos1, double *pos2, double *e1, double *e2, int *zbins, int nbinsz, int ngal,
    int nmin, int nmax, double rmin, double rmax, int nbinsr,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    double inner_xmin, double inner_xmax,
    int *gridmatcher_inds, double *gridmatcher_weights, int *reducedgridmatcher, int mas_scheme, int npix_inner,
    int nthreads, double *bin_centers, int *counts, 
    double complex *Gns, double complex *Gns_norm, double complex *Gammans, double complex *Gammans_norm);

void update_Gamman_discrete_worker(
    double complex *Gns, double complex *Gnnorms,
    int nmin, int nmax, int n, int nbinsr, int nbinsz, int nworker, 
    double weight, double e1, double e2, int zbin1,
    double complex *threepcf, double complex *threepcf_n);

void update_Gammansingle_discmixed_worker( 
    double complex *Gns_disc, double complex *Gnnorms_disc, 
    double complex *Gns_grid, double complex *Gnnorms_grid,
    int nbinsr_disc, int nbinsr_grid, int nbinsz, int nreso,
    int nworker, int zbin1, double weight_disc, double e1_disc, double e2_disc,
    double *weight_grid, double *e1_grid, double *e2_grid, int *cumnbinsr_grid,
    double complex *Gamman_dd, double complex *Gammannorm_dd,
    double complex *Gamman_dg, double complex *Gammannorm_dg,
    double complex *Gamman_gd, double complex *Gammannorm_gd);

void alloc_Gn_discrete_basic(
    double *weight, double *pos1, double *pos2, double *e1, double *e2, int *zbins, int nbinsz, int ngal, 
    int n, double rmin, double rmax, int nbinsr,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, double *bin_centers, int *counts, double complex *Gns, double complex *Gn_norm);

void alloc_Gamman_discrete_basic(
    complex *Gns, complex *norms,
    double *weight, double *e1, double *e2, int *zbins, int nbinsz, int ngal, 
    int n, int nbinsr, double complex *threepcf, double complex *threepcf_n);

void test_rsearch(double *pos1, double *pos2, int ngals, double center1, double center2, double *radii, int nradii,
                  int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
                  double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
                  int *indices_radii, int *rshells, int ninradii);

void test_inshell(double *pos1, double *pos2, int ngals, double rmin, double rmax,
                  int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
                  double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
                  int *countshell);

void test_G01_discrete(
    double *weight, double *pos1, double *pos2, double *e1, double *e2, int ngal, 
    double *bin_edges, int nbinedges,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, complex *G01s, double complex *G01s_norm);

void alloc_Gammans_discrete_G3L(
    double *w_source, double *pos1_source, double *pos2_source, double *e1, double *e2, int ngal_source,
    double *w_lens, double *pos1_lens, double *pos2_lens, int ngal_lens, 
    int nmax, double rmin, double rmax, int nbinsr,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, double *rbin_means, double complex *Gammans, double complex *Gammans_norm);

void alloc_Gammans_discrete_SSL(
    double *w_source, double *pos1_source, double *pos2_source, double *e1, double *e2, int ngal_source,
    double *w_lens, double *pos1_lens, double *pos2_lens, int ngal_lens, 
    int nmax, double rmin, double rmax, int nbinsr,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, double *rbin_means, double complex *Gammans, double complex *Gammans_norm);