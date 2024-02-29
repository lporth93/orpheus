#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <complex.h>

//#define int int32_t // Should be changed :D

void alloc_Gammans_discrete_ggg(
    int* isinner, double *weight, double *pos1, double *pos2, double *e1, 
    double *e2, int *zbins, int nbinsz, int ngal, 
    int nmin, int nmax, double rmin, double rmax, double *rbins, int nbinsr, int dccorr,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, double *bin_centers, double complex *Gammans, double complex *Gammans_norm);

/**
 * @brief Calculates Multipoles for source-lens-lens correlation function. 
 * @todo Currently only works for same lens population!
 * @todo Currently no tomgraphy!
 * 
 * @param isinner_source 
 * @param w_source 
 * @param pos1_source 
 * @param pos2_source 
 * @param e1 
 * @param e2 
 * @param ngal_source 
 * @param w_lens 
 * @param pos1_lens 
 * @param pos2_lens 
 * @param ngal_lens 
 * @param nmax 
 * @param rmin 
 * @param rmax 
 * @param nbinsr 
 * @param dccorr 
 * @param index_matcher 
 * @param pixs_galind_bounds 
 * @param pix_gals 
 * @param pix1_start 
 * @param pix1_d 
 * @param pix1_n 
 * @param pix2_start 
 * @param pix2_d 
 * @param pix2_n 
 * @param nthreads 
 * @param rbin_means 
 * @param Gammans Multipoles 
 * @param Gammans_norm Normalization Multipoles
 */
void alloc_Gammans_discrete_gnn(
    int *isinner_source, double *w_source, double *pos1_source, double *pos2_source, double *e1, double *e2, int ngal_source,
    double *w_lens, double *pos1_lens, double *pos2_lens, int ngal_lens, 
    int nmax, double rmin, double rmax, int nbinsr, int dccorr,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, double *rbin_means, double complex *Gammans, double complex *Gammans_norm);


void alloc_Gammans_discrete_ggn(
    double *w_source, double *pos1_source, double *pos2_source, double *e1, double *e2, int ngal_source,
    double *w_lens, double *pos1_lens, double *pos2_lens, int ngal_lens, 
    int nmax, double rmin, double rmax, int nbinsr, int dccorr,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, double *rbin_means, double complex *Gammans, double complex *Gammans_norm);

// /**
//  * @brief Calculates Multipoles for source-source-lens correlation function. 
//  * 
//  * @param w_source 
//  * @param pos1_source 
//  * @param pos2_source 
//  * @param e1 
//  * @param e2 
//  * @param ngal_source 
//  * @param w_lens 
//  * @param pos1_lens 
//  * @param pos2_lens 
//  * @param ngal_lens 
//  * @param nmax 
//  * @param rmin 
//  * @param rmax 
//  * @param nbinsr 
//  * @param dccorr 
//  * @param index_matcher 
//  * @param pixs_galind_bounds 
//  * @param pix_gals 
//  * @param pix1_start 
//  * @param pix1_d 
//  * @param pix1_n 
//  * @param pix2_start 
//  * @param pix2_d 
//  * @param pix2_n 
//  * @param nthreads 
//  * @param rbin_means 
//  * @param Gammans 
//  * @param Gammans_norm 
//  */
// void alloc_Gammans_discrete_ggn(
//     double *w_source, double *pos1_source, double *pos2_source, double *e1, double *e2, int ngal_source,
//     double *w_lens, double *pos1_lens, double *pos2_lens, int ngal_lens, 
//     int nmax, double rmin, double rmax, int nbinsr, int dccorr,
//     int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
//     double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
//     int nthreads, double *rbin_means, double complex *Gammans, double complex *Gammans_norm);



void alloc_Gammans_tree_ggg(
    int *isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int *zbins, int nbinsz, int ngal, 
    int nresos, double *reso_redges, int *ngal_resos, 
    double *weight_resos, double *pos1_resos, double *pos2_resos, 
    double *e1_resos, double *e2_resos, int *zbin_resos, double *weightsq_resos,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int nmin, int nmax, double rmin, double rmax, double *rbins, int nbinsr, int dccorr, 
    int nthreads, double *bin_centers, double complex *Gammans, double complex *Gammans_norm);

void alloc_Gammans_doubletree_ggg(
    int nresos, int nresos_grid, double *dpix1_resos,  double *dpix2_resos, double *reso_redges, 
    int resoshift_leafs, int minresoind_leaf, int maxresoind_leaf,
    int *ngal_resos, int nbinsz, int *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos, 
    double *e1_resos, double *e2_resos, int *zbin_resos, double *weightsq_resos,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, int nregions, int *index_matcher_hash,
    int nmax, double rmin, double rmax, int nbinsr, int dccorr, 
    int nthreads, double *bin_centers, double complex *Gammans, double complex *Gammans_norm);

void alloc_Gammans_basetree_ggg(
    int nresos, int nresos_grid, double *dpix1_resos,  double *dpix2_resos, double *reso_redges, int *ngal_resos, int nbinsz,
    int *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos, 
    double *e1_resos, double *e2_resos, int *zbin_resos, double *weightsq_resos,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, int nregions, int *index_matcher_hash,
    int nmax, double rmin, double rmax, int nbinsr, int dccorr, 
    int nthreads, double *bin_centers, double complex *Gammans, double complex *Gammans_norm);

void alloc_Gammans_discrete_GNN(
    int *isinner_source, double *w_source, double *pos1_source, double *pos2_source, double *e1, double *e2, int *zbin_source, int nbinsz_source, int ngal_source,
    double *w_lens, double *pos1_lens, double *pos2_lens, int *zbin_lens, int nbinsz_lens, int ngal_lens, 
    int nmax, double rmin, double rmax, int nbinsr, int dccorr,
    int *index_matcher_source, int *pixs_galind_bounds_source, int *pix_gals_source, 
    int *index_matcher_lens, int *pixs_galind_bounds_lens, int *pix_gals_lens, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int nthreads, double *bin_centers, double complex *Upsilon_n, double complex *Norm_n);

void alloc_Gammans_doubletree_GNN(
    int nresos, int nresos_grid, double *dpix1_resos,  double *dpix2_resos, double *reso_redges, 
    int resoshift_leafs, int minresoind_leaf, int maxresoind_leaf,
    int *isinner_source_resos, double *w_source_resos, double *pos1_source_resos, double *pos2_source_resos, 
    double *e1_source_resos, double *e2_source_resos, int *zbin_source_resos, int *ngal_source_resos, int nbinsz_source, 
    int *isinner_lens_resos, double *w_lens_resos, double *pos1_lens_resos, double *pos2_lens_resos, 
    int *zbin_lens_resos, int *ngal_lens_resos, int nbinsz_lens, 
    int nmax, double rmin, double rmax, int nbinsr, int dccorr,
    int *index_matcher_source, int *pixs_galind_bounds_source, int *pix_gals_source, 
    int *index_matcher_lens, int *pixs_galind_bounds_lens, int *pix_gals_lens, int *index_matcher_hash, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int nthreads, double *bin_centers, double complex *Upsilon_n, double complex *Norm_n);

void alloc_Gammans_discrete_NGG(
    double *w_source, double *pos1_source, double *pos2_source, double *e1_source, double *e2_source, int *zbin_source, int nbinsz_source, int ngal_source,
    int *isinner_lens, double *w_lens, double *pos1_lens, double *pos2_lens, int *zbin_lens, int nbinsz_lens, int ngal_lens, 
    int nmax, double rmin, double rmax, int nbinsr, int dccorr,
    int *index_matcher_source, int *pixs_galind_bounds_source, int *pix_gals_source, 
    int *index_matcher_lens, int *pixs_galind_bounds_lens, int *pix_gals_lens, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int nthreads, double *bin_centers, double complex *Upsilon_n, double complex *Norm_n);

void alloc_triplets_tree_xipxipcov(
    int *isinner, double *weight, double *pos1, double *pos2, int *zbins, int nbinsz, int ngal, 
    int nresos, double *reso_redges, int *ngal_resos, 
    double *weight_resos, double *pos1_resos, double *pos2_resos, int *zbin_resos,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int nmin, int nmax, double rmin, double rmax, double *rbins, int nbinsr, int dccorr, 
    int nthreads, double *bin_centers, double *wwcounts, double *w2wcounts, double complex *w2wwcounts, double complex *wwwcounts);

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