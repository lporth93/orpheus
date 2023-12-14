#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
void reducecat(double *w, double *pos_1, double *pos_2, double *scalarquants, int ngal, int nscalarquants,
               double mask_d1, double mask_d2, double mask_min1, double mask_min2, int mask_n1, int mask_n2, int shuffle,
               double *w_red, double *pos1_red, double *pos2_red, double *scalarquants_red, int ngal_red);
    
// Pixelizes catalog onto regular grid
// Note that we do not assume the `averaged' galaxy to be in the center of the pixel, 
// but use the mean value of the galaxy within that pixel.
// Furthermore, weights and shapes are added together and not normalized by the pixel area
// Returns the galaxies on the grid, where empty pixels have zero weight - this means that
// the this function has the same memory footprint of a spatial hash
void reducecat2(double *pos_1, double *pos_2, double *e1, double *e2, double *w, int ngal,
               double mask_d1, double mask_d2, double mask_min1, double mask_min2, int mask_n1, int mask_n2,
               double *pos1_red, double *pos2_red, double *e1_red, double *e2_red, double *w_red);