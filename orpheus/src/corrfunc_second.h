#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <complex.h>

/////////////////////////
/// Shear 2PCF related //
/////////////////////////
// Does not do cross-correlations
void alloc_NNcounts_doubletree(
    int nresos, int nresos_grid, double *dpix1_resos,  double *dpix2_resos, double *reso_redges, 
    int resoshift_leafs, int minresoind_leaf, int maxresoind_leaf,
    int *ngal_resos, int nbinsz, int *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos, 
    double *scalar_tracer, int *zbin_resos, 
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int nregions, int *index_matcher_hash,
    double rmin, double rmax, int nbinsr, int do_dc,
    int nthreads, double *bin_centers, double *counts, long long int *npair);

void alloc_xipm_doubletree(
    int nresos, int nresos_grid, double *dpix1_resos,  double *dpix2_resos, double *reso_redges, 
    int resoshift_leafs, int minresoind_leaf, int maxresoind_leaf,
    int *ngal_resos, int nbinsz, int *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos, 
    double *e1_resos, double *e2_resos, int *zbin_resos, 
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int nregions, int *index_matcher_hash,
    double rmin, double rmax, int nbinsr, int do_dc,
    int nthreads, double *bin_centers, double complex *xip, double complex *xim, double *norm, long long int *npair);


