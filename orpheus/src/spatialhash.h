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
// Notes:
// * We do not assume the `averaged' galaxy to be in the center of the pixel,
//   but use the mean value of the galaxy within that pixel
// * We allow for arbitrary (double-valued) scalar quantities that are mapped
//   onto the pixelgrid, without normalization of any sort.
// * Returns a new reduced catalog that formally has the same length as the 
//   input catalog, however only the first `ngal_red' components are allocated.
// We prefer this method as it `might' be more stable for small pixel sizes.
// Although as an intermediate step, the spatial hash is beig allocated, we still
// Have a smaller memory footprint in case the number of pixels exceeds the number
// of galaxies.
void reducecat(double *isinner, double *w, double *pos_1, double *pos_2, double *scalarquants, int ngal, int nscalarquants,
               int normed,
               double mask_d1, double mask_d2, double mask_min1, double mask_min2, int mask_n1, int mask_n2, int shuffle,
               double *isinner_red, double *w_red, double *pos1_red, double *pos2_red, double *scalarquants_red, int ngal_red);

// Tomographic + parallel version of reducecat.
// Builds a single spatial hash over all galaxies, then reduces every tomographic
// bin in one call (the zbin loop is internal, not in Python). Galaxies in the same
// pixel but different zbins yield separate super-galaxies. The pixel loop is
// OpenMP-parallel: a prefix sum over the per-pixel (distinct-zbin) counts assigns
// each occupied pixel a disjoint, contiguous output slice, so threads never race.
// Output arrays have length ngal (isinner/w/pos/zbins_red) resp. nscalarquants*ngal;
// only the first (returned) entries are filled, the caller drops trailing w_red==0.
void reducecat_tomo(double *isinner, double *w, double *pos_1, double *pos_2, double *scalarquants,
               int *zbins, int ngal, int nscalarquants, int nbinsz, int normed,
               double mask_d1, double mask_d2, double mask_min1, double mask_min2, int mask_n1, int mask_n2, int shuffle,
               int nthreads,
               double *isinner_red, double *w_red, double *pos1_red, double *pos2_red, int *zbins_red, double *scalarquants_red);

// FOREACH-style macro walking the occupied flat-grid pixels within a search
// radius of (pos1, pos2): mirrors the
// `for(ip1=lo1;ip1<hi1;ip1++){ for(ip2=lo2;ip2<hi2;ip2++){ ... } }` pixel-box +
// index_matcher/pixs_galind_bounds lookup duplicated across the flat
// (MultiresoCatalog) gal2-finders. Bounds are strictly `<` (not `<=`), matching
// every adopting kernel's existing convention exactly. Pure textual expansion
// (no struct, no call boundary) so the compiler sees the same flat nested loop
// it would from hand-written code -- a struct/function-based iterator was
// tried first and measured a reproducible few-percent slowdown (its state has
// to survive across the inlined call boundary instead of living in registers
// like flat loop counters do). rshift_m/rshift_b are the (possibly 0)
// per-resolution offsets into index_matcher / pixs_galind_bounds; pass 0 for
// single-resolution (discrete) kernels. lower_v/upper_v must already be
// declared int locals; the caller supplies its own
// `for(k=lower_v;k<upper_v;k++){ ... }` galaxy loop as the trailing block:
//   FLATCELL_FOREACH(index_matcher, rshift_m, bounds, rshift_b, pos1, pos2, rsearch,
//                     pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n,
//                     lower_v, upper_v){
//       for (k = lower_v; k < upper_v; k++){ ... }
//   }
// Do not nest two invocations in the same block (internal _fc_* names collide).
#define FLATCELL_FOREACH(INDEX_MATCHER, RSHIFT_M, BOUNDS, RSHIFT_B, POS1, POS2, RSEARCH, \
    PIX1_START, PIX1_D, PIX1_N, PIX2_START, PIX2_D, PIX2_N, LOWERV, UPPERV) \
    for (int _fc_ip1 = mymax(0, (int) floor(((POS1)-((RSEARCH)+(PIX1_D))-(PIX1_START))/(PIX1_D))), \
             _fc_hi1 = mymin((PIX1_N)-1, (int) floor(((POS1)+((RSEARCH)+(PIX1_D))-(PIX1_START))/(PIX1_D))), \
             _fc_lo2 = mymax(0, (int) floor(((POS2)-((RSEARCH)+(PIX2_D))-(PIX2_START))/(PIX2_D))), \
             _fc_hi2 = mymin((PIX2_N)-1, (int) floor(((POS2)+((RSEARCH)+(PIX2_D))-(PIX2_START))/(PIX2_D))); \
         _fc_ip1 < _fc_hi1; _fc_ip1++) \
      for (int _fc_ip2 = _fc_lo2, _fc_ind_red = -2; \
           _fc_ip2 < _fc_hi2 && (_fc_ind_red = (INDEX_MATCHER)[(RSHIFT_M) + _fc_ip2*(PIX1_N) + _fc_ip1], 1); \
           _fc_ip2++) \
        if (_fc_ind_red != -1 && (((LOWERV) = (BOUNDS)[(RSHIFT_B)+_fc_ind_red]), \
                                   ((UPPERV) = (BOUNDS)[(RSHIFT_B)+_fc_ind_red+1]), 1))

