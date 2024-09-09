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
    int *isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int ngal, 
    int nmax, double rmin, double rmax, double *rbins, int nbinsr, int dccorr,
    int *index_matcher_hash, int *pixs_galind_bounds, int *pix_gals, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, double *bin_centers, double complex *Upsilon_n, double complex *N_n);

void alloc_notomoMap4_disc_gggg(
    int *isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int ngal, 
    int nmax, double rmin, double rmax, int nbinsr, int dccorr, double *phibins, double *dbinsphi, int nbinsphi,
    int *index_matcher_hash, int *pixs_galind_bounds, int *pix_gals, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, int projection, double *mapradii, int nmapradii, double complex *M4correlators,
    int alloc_4pcfmultipoles, int alloc_4pcfreal,
    double *bin_centers, double complex *Upsilon_n, double complex *N_n, double complex *Gammas, double complex *Norms);

void alloc_notomoMap4_tree_gggg(
    int *isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int ngal, 
    int nmax, double rmin, double rmax, int nbinsr, int dccorr, 
    int *nindices, int len_nindices, double *phibins, double *dbinsphi, int nbinsphi,
    int nresos, double *reso_redges, int *ngal_resos, 
    int *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos, 
    double *e1_resos, double *e2_resos,
    int *index_matcher_hash, int *pixs_galind_bounds, int *pix_gals, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, int projection, double *mapradii, int nmapradii, double complex *M4correlators, 
    int alloc_4pcfmultipoles, int alloc_4pcfreal,
    double *bin_centers, double complex *Upsilon_n, double complex *N_n, double complex *Gammas, double complex *Norms);