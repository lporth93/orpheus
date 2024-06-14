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

/////////////////////////
/// Shear 3PCF related //
/////////////////////////
void alloc_Gammans_discrete_ggg(
    int* isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int *zbins, int nbinsz, int ngal, 
    int nmin, int nmax, double rmin, double rmax, double *rbins, int nbinsr, int dccorr,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, double *bin_centers, double complex *Gammans, double complex *Gammans_norm);

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

///////////////////
/// G3L related ///
///////////////////
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

//////////////////////////////
// Lens-Shear-Shear related //
//////////////////////////////
void alloc_Gammans_discrete_NGG(
    double *w_source, double *pos1_source, double *pos2_source, double *e1_source, double *e2_source, int *zbin_source, int nbinsz_source, int ngal_source,
    int *isinner_lens, double *w_lens, double *pos1_lens, double *pos2_lens, int *zbin_lens, int nbinsz_lens, int ngal_lens, 
    int nmax, double rmin, double rmax, int nbinsr, int dccorr,
    int *index_matcher_source, int *pixs_galind_bounds_source, int *pix_gals_source, 
    int *index_matcher_lens, int *pixs_galind_bounds_lens, int *pix_gals_lens, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int nthreads, double *bin_centers, double complex *Upsilon_n, double complex *Norm_n);

void alloc_Gammans_tree_NGG(
    int nresos, double *reso_redges, 
    double *w_source_resos, double *pos1_source_resos, double *pos2_source_resos,
    double *e1_source_resos, double *e2_source_resos, int *zbin_source_resos, int nbinsz_source, int *ngal_source_resos,
    int *isinner_lens, double *w_lens, double *pos1_lens, double *pos2_lens, int *zbin_lens, int nbinsz_lens, int ngal_lens, 
    int *index_matcher_source, int *pixs_galind_bounds_source, int *pix_gals_source, 
    int *index_matcher_lens, int *pixs_galind_bounds_lens, int *pix_gals_lens, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int nmax, double rmin, double rmax, int nbinsr, int dccorr,
    int nthreads, double *bin_centers, double complex *Upsilon_n, double complex *Norm_n);

void alloc_Gammans_doubletree_NGG(
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

void multipoles2npcf_gggg(double complex *upsilon_n, double complex *N_n, double *rcenters, int projection,
                          int n_cfs, int nbinsr, int nmax, double *phis12, int nbinsphi12, double *phis13, int nbinsphi13,
                          int nthreads, double complex *npcf, double complex *npcf_norm);

void multipoles2npcf_gggg_rtriple(double complex *Upsilon_n, double complex *N_n, int n1max, int n2max,
                                  double *theta1, double *theta2, double *theta3, int nthetas,
                                  double *phis12, double *phis13, int nbinsphi12, int nbinsphi13,
                                  int projection, double complex *npcf, double complex *npcf_norm);

void multipoles2npcf_gggg_singletheta(double complex *Upsilon_n, double complex *N_n, int n1max, int n2max,
                                      double theta1, double theta2, double theta3,
                                      double *phis12, double *phis13, int nbinsphi12, int nbinsphi13,
                                      int projection, double complex *npcf, double complex *npcf_norm);

void fourpcf2M4correlators(int nzcombis,
                           double y1, double y2, double y3, double dy1, double dy2, double dy3,
                           double *phis1, double *phis2, double *dphis1, double *dphis2, int nbinsphi1, int nbinsphi2,
                           double complex *fourpcf, double complex *m4corr);

void fourpcf2M4correlators_parallel(int nzcombis,
                           double y1, double y2, double y3, double dy1, double dy2, double dy3,
                           double *phis1, double *phis2, double *dphis1, double *dphis2, int nbinsphi1, int nbinsphi2,
                           int nthreads, double complex *fourpcf, double complex *m4corr);

void alloc_notomoMap4_analytic(
    double rmin, double rmax, int nbinsr, double *phibins, double *dbinsphi, int nbinsphi,
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, double *mapradii, int nmapradii, 
    double *xip, double *xim, double thetamin_xi, double thetamax_xi, int nthetabins_xi, int nsubsample_filter,
    double complex *M4correlators);

void gauss4pcf_analytic(double theta1, double theta2, double theta3, double *phis, int nphis,
                        double *xip, double *xim, double thetamin_xis, double thetamax_xis, double dtheta_xis,
                        double complex *gaussfourpcf);

void filter_Map4(double y1, double y2, double y3, double phi1, double phi2, double complex *output);

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