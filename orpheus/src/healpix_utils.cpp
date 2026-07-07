// extern "C" shim over healpix_cxx -- see healpix_utils.h. The only C++ TU in
// orpheus; it exists solely so the C estimators can call query_disc without
// reimplementing the nested-HEALPix geometry.

#include <cmath>
#include "healpix_base.h"
#include "rangeset.h"
#include "pointing.h"
#include "vec3.h"
#include "datatypes.h"
#include "healpix_utils.h"

extern "C" long hpx_query_disc_nest(long nside, const double *vec, double radius,
                                    long *out, long max_out){
    T_Healpix_Base<int64> base((int64)nside, NEST, SET_NSIDE);
    pointing ptg(vec3(vec[0], vec[1], vec[2]));
    rangeset<int64> pixset;
    // fact=4: balances false positives vs. work; never yields false negatives.
    base.query_disc_inclusive(ptg, radius, pixset, 4);

    long n = 0;
    tsize nr = pixset.nranges();
    for (tsize i=0; i<nr; ++i){
        int64 a = pixset.ivbegin(i);
        int64 b = pixset.ivend(i);
        for (int64 p=a; p<b; ++p){
            if (n < max_out){ out[n] = (long)p; }
            ++n;
        }
    }
    return n;
}

extern "C" long hpx_query_disc_nest_ranges(long nside, const double *vec, double radius,
                                           long *out_lohi, long max_pairs){
    T_Healpix_Base<int64> base((int64)nside, NEST, SET_NSIDE);
    pointing ptg(vec3(vec[0], vec[1], vec[2]));
    rangeset<int64> pixset;
    // fact=1: loosest inclusive refinement (~5% more candidate pixels than fact=4
    // but a cheaper query); the exact geodesic filter removes the extras anyway.
    base.query_disc_inclusive(ptg, radius, pixset, 1);
    long nr = pixset.nranges();
    for (long i=0; i<nr && i<max_pairs; ++i){
        out_lohi[2*i]   = (long)pixset.ivbegin(i);
        out_lohi[2*i+1] = (long)pixset.ivend(i);
    }
    return nr;
}

extern "C" double hpx_nside2resol(long nside){
    long npix = 12L*nside*nside;
    return std::sqrt(4.0*M_PI/(double)npix);
}

extern "C" long hpx_ang2pix_nest(long nside, const double *vec){
    T_Healpix_Base<int64> base((int64)nside, NEST, SET_NSIDE);
    pointing ptg(vec3(vec[0], vec[1], vec[2]));
    return (long)base.ang2pix(ptg);
}
