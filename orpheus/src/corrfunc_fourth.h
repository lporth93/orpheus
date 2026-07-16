#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <complex.h>

////////////////////////
// Shear 4PCF related //
////////////////////////
void alloc_notomoGammans_discrete_gggg(
    double *isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int ngal, 
    int nmax, double rmin, double rmax, double *rbins, int nbinsr, int dccorr,
    int *index_matcher_hash, int *pixs_galind_bounds, int *pix_gals, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, int verbose, double *bin_centers, double complex *Upsilon_n, double complex *N_n);

void alloc_notomoGammans_tree_gggg(
    double *isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int ngal, 
    int nmax, double rmin, double rmax, int nbinsr, int nthetacombis, int dccorr,
    int *nindices, int len_nindices, 
    int nresos, double *reso_redges, int *ngal_resos, 
    double *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos, 
    double *e1_resos, double *e2_resos,
    int *index_matcher_hash, int *pixs_galind_bounds, int *pix_gals, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, int verbose, double *bin_centers, double complex *Upsilon_n, double complex *N_n);

void alloc_notomoMap4_disc_gggg(
    double *isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int ngal, 
    int nmax, double rmin, double rmax, int nbinsr, int dccorr, double *phibins, double *dbinsphi, int nbinsphi,
    int *index_matcher_hash, int *pixs_galind_bounds, int *pix_gals, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, int verbose, int projection, double *mapradii, int nmapradii, double complex *M4correlators,
    int alloc_4pcfmultipoles, int alloc_4pcfreal,
    double *bin_centers, double complex *Upsilon_n, double complex *N_n, double complex *Gammas, double complex *Norms);

void alloc_notomoMap4_tree_gggg(
    double *isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int ngal, 
    int nmax, double rmin, double rmax, int nbinsr, int dccorr, 
    int *nindices, int len_nindices, double *phibins, double *dbinsphi, int nbinsphi,
    int nresos, double *reso_redges, int *ngal_resos, 
    double *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos, 
    double *e1_resos, double *e2_resos,
    int *index_matcher_hash, int *pixs_galind_bounds, int *pix_gals, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, int verbose, int projection, double *mapradii, int nmapradii, double complex *M4correlators, 
    int alloc_4pcfmultipoles, int alloc_4pcfreal,
    double *bin_centers, double complex *Upsilon_n, double complex *N_n, double complex *Gammas, double complex *Norms);

void alloc_notomoGammans_discrete_gnnn(
    double *isinner_source, double *weight_source, double *pos1_source, double *pos2_source, double *e1_source, double *e2_source, int ngal_source, 
    double *weight_lens, double *pos1_lens, double *pos2_lens, int ngal_lens, 
    int *index_matcher_lens, int *pixs_galind_bounds_lens, int *pix_gals_lens, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int nmax, double rmin, double rmax, int nbinsr, int dccorr,
    int nthreads, double *bin_centers, double complex *Gtilde_n, double complex *N_n);

void alloc_notomoGammans_tree_gnnn(
    int nresos, double *reso_redges,
    double *isinner_source, double *weight_source, double *pos1_source, double *pos2_source, double *e1_source, double *e2_source, int ngal_source, 
    double *weight_lens_resos, double *pos1_lens_resos, double *pos2_lens_resos, int *ngal_lens_resos, 
    int *index_matcher_source, int *pixs_galind_bounds_source, int *pix_gals_source, 
    int *index_matcher_lens, int *pixs_galind_bounds_lens, int *pix_gals_lens, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int nmax, double rmin, double rmax, int nbinsr, int dccorr, int nthetacombis, 
    int *nindices, int len_nindices, 
    int nthreads, int verbose, double *bin_centers, double complex *Gtilde_n, double complex *N_n);

void alloc_notomoMapNap3_tree_gnnn(
    int nresos, double *reso_redges,
    double *isinner_source, double *weight_source, double *pos1_source, double *pos2_source, double *e1_source, double *e2_source, int ngal_source, 
    double *weight_lens_resos, double *pos1_lens_resos, double *pos2_lens_resos, int *ngal_lens_resos, 
    int *index_matcher_source, int *pixs_galind_bounds_source, int *pix_gals_source, 
    int *index_matcher_lens, int *pixs_galind_bounds_lens, int *pix_gals_lens, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int nmax, double rmin, double rmax, int nbinsr, int dccorr,
    int *nindices, int len_nindices, double *phibins, double *dbinsphi, int nbinsphi,
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, double *apradii, int napradii, double complex *NM3correlator, 
    int alloc_4pcfmultipoles, int alloc_4pcfreal,
    double *bin_centers, double complex *Gtilde_n, double complex *N_n, double complex *Gtilde, double complex *Norms);

void alloc_notomoGammans_tree_nnnn(
    double *isinner, double *weight, double *pos1, double *pos2, int ngal, 
    int nmax, double rmin, double rmax, int nbinsr, int nthetacombis, int dccorr, 
    int *nindices, int len_nindices, 
    int nresos, double *reso_redges, int *ngal_resos, 
    double *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos, 
    int *index_matcher_hash, int *pixs_galind_bounds, int *pix_gals, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, int verbose,
    double *bin_centers, double complex *N_n);

void alloc_notomoNap4_tree_nnnn(
    double *isinner, double *weight, double *pos1, double *pos2, int ngal, 
    int nmax, double rmin, double rmax, int nbinsr, int dccorr, 
    int *nindices, int len_nindices, double *phibins, double *dbinsphi, int nbinsphi,
    int nresos, double *reso_redges, int *ngal_resos, 
    double *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos, 
    int *index_matcher_hash, int *pixs_galind_bounds, int *pix_gals, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, double *napradii, int nnapradii, double complex *N4correlators, 
    int alloc_4pcfmultipoles, int alloc_4pcfreal,
    double *bin_centers, double complex *N_n, double complex *Counts);

void alloc_notomoNap4_doubletree_nnnn(
    double *isinner, double *weight, double *pos1, double *pos2, int ngal,
    int nmax, double rmin, double rmax, int nbinsr, int dccorr,
    int *nindices, int len_nindices, double *phibins, double *dbinsphi, int nbinsphi,
    int nresos, double *reso_redges, int *ngal_resos,
    double *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos,
    int *index_matcher_hash, int *pixs_galind_bounds, int *pix_gals, int nregions,
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, double *napradii, int nnapradii, double complex *N4correlators,
    int alloc_4pcfmultipoles, int alloc_4pcfreal,
    double *bin_centers, double complex *N_n, double complex *Counts);

void alloc_nnnn_tree(
    double *isinner, double *weight, double *pos1, double *pos2, int ngal,
    int nmax, double rmin, double rmax, int nbinsr, int dccorr,
    int *nindices, int len_nindices,
    int nresos, double *reso_redges, int *ngal_resos,
    double *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos,
    int *index_matcher_hash, int *pixs_galind_bounds, int *pix_gals, int nregions,
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, double memory_bound, double *bin_centers, double complex *N_n);

void alloc_nnnn_tree_spherical(
    double *cen_isinner, double *cen_w,
    double *cen_vx, double *cen_vy, double *cen_vz,
    double *cen_ra, double *cen_sindec, double *cen_cosdec, int ngal,
    int nmax, double rmin, double rmax, int nbinsr, int dccorr,
    int *nindices, int len_nindices,
    int nresos, double *reso_redges, int *ngal_resos, int *ncells_resos,
    long *nside_nav,
    double *red_w, double *red_vx, double *red_vy, double *red_vz,
    double *red_ra, double *red_sindec, double *red_cosdec, int *rshift_red,
    long *cell_pix, int *cell_redbounds, int *rshift_cellpix, int *rshift_cellbounds,
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, double memory_bound, double *bin_centers, double complex *N_n);

void alloc_nnnn_doubletree(
    int nresos, int nresos_grid, double *dpix1_resos, double *dpix2_resos, double *reso_redges,
    int resoshift_leafs, int minresoind_leaf, int maxresoind_leaf,
    double *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos, int *ngal_resos,
    int nmax, double rmin, double rmax, int nbinsr, int dccorr,
    int *nindices, int len_nindices,
    int *index_matcher_hash, int *index_matcher_full, int *pixs_galind_bounds, int *pix_gals,
    int *filledregions, int nfilledregions, int nregions,
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, double memory_bound, int verbose, double *bin_centers, double complex *N_n);
