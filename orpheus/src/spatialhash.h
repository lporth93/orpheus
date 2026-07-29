#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Setup function that does two things
// 1) Remove outliers from the galaxies that are not covered in the mask file
// 2) Mapping between pixels and galaxies therein
// !!! WE ASSUME THAT THE MASK IS EVENLY SPACED !!!
// !!! WE ASSUME THAT THE UNITS OF mask_d1/2 AND pos_1/2 ARE THE SAME !!!
void build_spatialhash(
    double *pos_1, double *pos_2, int ngal,
    double mask_d1, double mask_d2, double mask_min1, double mask_min2, int mask_n1, int mask_n2,
    int *result);

// Puts galaxy catalogs on pixel grid
// We use the weights to average over shear/positions
void _gen_pixmeans(double *pos_1, double *pos_2, double *e1, double *e2, double *w, double *wc, int ngal,
    double mask_d1, double mask_d2, double mask_min1, double mask_min2, int mask_n1, int mask_n2,
    double *result);

// Pixelizes catalog onto regular grid
// Although as an intermediate step, the spatial hash is beig allocated, we still
// Have a smaller memory footprint in case the number of pixels exceeds the number
// of galaxies.
void reducecat(double *isinner, double *w, double *pos_1, double *pos_2, double *scalarquants, int ngal, int nscalarquants,
               int normed,
               double mask_d1, double mask_d2, double mask_min1, double mask_min2, int mask_n1, int mask_n2, int shuffle,
               double *isinner_red, double *w_red, double *pos1_red, double *pos2_red, double *scalarquants_red, int ngal_red);

// Tomographic + parallel version of reducecat.
void reducecat_tomo(double *isinner, double *w, double *pos_1, double *pos_2, double *scalarquants,
               int *zbins, int ngal, int nscalarquants, int nbinsz, int normed,
               double mask_d1, double mask_d2, double mask_min1, double mask_min2, int mask_n1, int mask_n2, int shuffle,
               int nthreads,
               double *isinner_red, double *w_red, double *pos1_red, double *pos2_red, int *zbins_red, double *scalarquants_red);

// FOREACH-style macro walking the occupied flat-grid pixels within a search
// radius of (pos1, pos2): mirrors the
// `for(ip1=lo1;ip1<hi1;ip1++){ for(ip2=lo2;ip2<hi2;ip2++){ ... } }` pixel-box +
// index_matcher/pixs_galind_bounds lookup duplicated across the flat
// (MultiresoCatalog) gal2-finders. 
// Do not nest two invocations in the same block (internal _fc_* names collide).
#define FLATCELL_FOREACH(INDEX_MATCHER, RSHIFT_M, BOUNDS, RSHIFT_B, POS1, POS2, RSEARCH, \
    PIX1_START, PIX1_D, PIX1_N, PIX2_START, PIX2_D, PIX2_N, LOWERV, UPPERV) \
    for (int _fc_ip1 = mymax(0, (int) floor(((POS1)-((RSEARCH)+(PIX1_D))-(PIX1_START))/(PIX1_D))), \
             _fc_hi1 = mymin((PIX1_N), (int) floor(((POS1)+((RSEARCH)+(PIX1_D))-(PIX1_START))/(PIX1_D))), \
             _fc_lo2 = mymax(0, (int) floor(((POS2)-((RSEARCH)+(PIX2_D))-(PIX2_START))/(PIX2_D))), \
             _fc_hi2 = mymin((PIX2_N), (int) floor(((POS2)+((RSEARCH)+(PIX2_D))-(PIX2_START))/(PIX2_D))); \
         _fc_ip1 < _fc_hi1; _fc_ip1++) \
      for (int _fc_ip2 = _fc_lo2, _fc_ind_red = -2; \
           _fc_ip2 < _fc_hi2 && (_fc_ind_red = (INDEX_MATCHER)[(RSHIFT_M) + _fc_ip2*(PIX1_N) + _fc_ip1], 1); \
           _fc_ip2++) \
        if (_fc_ind_red != -1 && (((LOWERV) = (BOUNDS)[(RSHIFT_B)+_fc_ind_red]), \
                                   ((UPPERV) = (BOUNDS)[(RSHIFT_B)+_fc_ind_red+1]), 1))

