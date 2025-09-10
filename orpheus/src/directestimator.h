#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <complex.h>


double getFilterQ(int type_filter, double reff2);
double getFilterU(int type_filter, double reff2);
double getFilterSupp(int type_filter);
double getFilterSuppU(int type_filter);


void ApertureMassMap_Equal(
    double R_ap, double *centers_1, double *centers_2, int ncenters_1, int ncenters_2,
    int max_order, int ind_filter, int weight_method, double weight_outer, double weight_inpainted, 
    double *weight, double *insurvey, double *pos1, double *pos2, double complex *g, int *zbins, int nbinsz, int ngal, 
    double *mask, 
    double mask1_start, double mask2_start, double mask1_d, double mask2_d, int mask1_n, int mask2_n,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    int nthreads, 
    double *out_counts, double *out_covs, double *out_Msn, double *out_Sn, double *out_Mapn, double *out_Mapn_var);

void ApertureCountsMap_Equal(
    double R_ap, double *centers_1, double *centers_2, int ncenters_1, int ncenters_2,
    int max_order, int ind_filter, int do_subtractions, int Nbar_policy, double weight_outer, double weight_inpainted, 
    double *weight, double *insurvey, double *pos1, double *pos2, double *tracer, int *zbins, int nbinsz, int ngal, 
    double *mask, 
    double mask1_start, double mask2_start, double mask1_d, double mask2_d, int mask1_n, int mask2_n,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    int nthreads,
    double *out_counts, double *out_covs, double *out_Msn, double *out_Sn, double *out_Napn, double *out_Napn_norm);

void MapnSingleEonlyDisc(
    double R_ap, double *centers_1, double *centers_2, int ncenters,
    int max_order, int ind_filter, int weight_method, int do_subtractions, double weight_outer, double weight_inpainted, 
    double *weight, double *insurvey, double *pos1, double *pos2, double complex *g, int *zbins, int nbinsz, int ngal, 
    double *mask, double *fraccov_cuts, int nfrac_cuts, int fraccov_method,
    double mask1_start, double mask2_start, double mask1_d, double mask2_d, int mask1_n, int mask2_n,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    int nthreads, double *Mapn, double *wtot_Mapn);

void singleAp_MapnSingleEonlyDisc(
    double R_ap, double center_1, double center_2, 
    int max_order, int ind_filter, double weight_outer, double weight_inpainted, 
    double *weight, double *insurvey, double *pos1, double *pos2, double complex *g, int *zbins, int nbinsz, int ngal, 
    double *mask,
    double mask1_start, double mask2_start, double mask1_d, double mask2_d, int mask1_n, int mask2_n,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double *counts, double *covs, double *Msn, double *Sn, double *Sn_w, double *S2n_w);

void NapnSingleDisc(
    double R_ap, double *centers_1, double *centers_2, int ncenters,
    int max_order, int ind_filter, int do_subtractions, int Nbar_policy, double weight_outer, double weight_inpainted, 
    double *weight, double *insurvey, double *pos1, double *pos2, double *tracer, int *zbins, int nbinsz, int ngal, 
    double *mask, double *fraccov_cuts, int nfrac_cuts, int fraccov_method,
    double mask1_start, double mask2_start, double mask1_d, double mask2_d, int mask1_n, int mask2_n,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    int nthreads, double *Napn, double *wtot_Napn);
    
void singleAp_NapnSingleDisc(
    double R_ap, double center_1, double center_2, 
    int max_order, int ind_filter, int Nbar_policy, double weight_outer, double weight_inpainted, 
    double *weight, double *insurvey, double *pos1, double *pos2, double *tracer, int *zbins, int nbinsz, int ngal, 
    double *mask,
    double mask1_start, double mask2_start, double mask1_d, double mask2_d, int mask1_n, int mask2_n,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double *counts, double *covs, double *Msn, double *Sn);