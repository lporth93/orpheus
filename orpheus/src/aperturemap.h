#ifndef ORPHEUS_APERTUREMAP_H
#define ORPHEUS_APERTUREMAP_H

#include <complex.h>

#include "multires_structs.h"

void aperturemassmap_spherical(
    const MultiresoCatalog *cat, const NavHash *nav, const TreeResoParams *tree,
    double R_ap, int ind_filter,
    const double *cvx, const double *cvy, const double *cvz, long ncenters,
    const double *mask, long nside_mask,
    int nthreads, int verbose,
    double complex *out_Map, double *out_norm, double *out_normQ, double *out_cov);

#endif // ORPHEUS_APERTUREMAP_H
