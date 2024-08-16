#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <complex.h>

/////////////////////////////
// Xipm covariance related //
/////////////////////////////
void alloc_triplets_tree_xipxipcov(
    int *isinner, double *weight, double *pos1, double *pos2, int *zbins, int nbinsz, int ngal, 
    int nresos, double *reso_redges, int *ngal_resos, 
    double *weight_resos, double *pos1_resos, double *pos2_resos, int *zbin_resos,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int nmin, int nmax, double rmin, double rmax, double *rbins, int nbinsr, int dccorr, 
    int nthreads, double *bin_centers, 
    double *wwcounts, double *w2wcounts, double complex *w2wwcounts, double complex *wwwcounts);

void alloc_triplets_doubletree_xipxipcov(
    int nresos, int nresos_grid, double *dpix1_resos,  double *dpix2_resos, double *reso_redges, int *ngal_resos, int nbinsz,
    int *isinner_resos, double *weight_resos, double *weight_sq_resos, double *pos1_resos, double *pos2_resos, int *zbin_resos,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, int nregions, int *index_matcher_hash,
    int nmax, double rmin, double rmax, int nbinsr, int dccorr, 
    int nthreads, double *bin_centers, 
    double *wwcounts, double *w2wcounts, double complex *w2wwcounts, double complex *wwwcounts);

void alloc_triplets_basetree_xipxipcov(
    int nresos, int nresos_grid, double *dpix1_resos,  double *dpix2_resos, double *reso_redges, int *ngal_resos, int nbinsz,
    int *isinner_resos, double *weight_resos, double *weight_sq_resos, double *pos1_resos, double *pos2_resos, int *zbin_resos,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, int nregions, int *index_matcher_hash,
    int nmax, double rmin, double rmax, int nbinsr, int dccorr, 
    int nthreads, double *bin_centers, 
    double *wwcounts, double *w2wcounts, double complex *w2wwcounts, double complex *wwwcounts);