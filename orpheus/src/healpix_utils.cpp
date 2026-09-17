// extern "C" shim over healpix_cxx. We only need this such that the C estimators
// can call query_disc without reimplementing the HEALPix geometry.

#include <cmath>
#include "healpix_base.h"
#include "rangeset.h"
#include "pointing.h"
#include "vec3.h"
#include "datatypes.h"
#include "healpix_utils.h"


extern "C" double hpx_nside2resol(long nside){
    long npix = 12L*nside*nside;
    return std::sqrt(4.0*M_PI/(double)npix);
}

extern "C" long hpx_ang2pix_nest(long nside, const double *vec){
    T_Healpix_Base<int64> base((int64)nside, NEST, SET_NSIDE);
    pointing ptg(vec3(vec[0], vec[1], vec[2]));
    return (long)base.ang2pix(ptg);
}

// Get ring indices of the nested pixels p0,...,p0+n-1
extern "C" void hpx_nest2ring_range(long nside, long p0, long n, long *out){
    T_Healpix_Base<int64> base((int64)nside, NEST, SET_NSIDE);
    for (long i=0; i<n; i++){out[i] = (long)base.nest2ring((int64)(p0+i));}
}

extern "C" void hpx_pix2vec_nest(long nside, long ipix, double *vec){
    T_Healpix_Base<int64> base((int64)nside, NEST, SET_NSIDE);
    vec3 v = base.pix2vec((int64)ipix);
    vec[0] = v.x; vec[1] = v.y; vec[2] = v.z;
}

extern "C" void hpx_pix2vec_ring(long nside, long ipix, double *vec){
    T_Healpix_Base<int64> base((int64)nside, RING, SET_NSIDE);
    vec3 v = base.pix2vec((int64)ipix);
    vec[0] = v.x; vec[1] = v.y; vec[2] = v.z;
}


extern "C" long hpx_query_disc_nest(long nside, const double *vec, double radius,
                                    long *out, long max_out){
    T_Healpix_Base<int64> base((int64)nside, NEST, SET_NSIDE);
    pointing ptg(vec3(vec[0], vec[1], vec[2]));
    rangeset<int64> pixset;
    // fact=4: balance false positives vs. work; never yields false negatives.
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
    // fact=1: loosest inclusive refinement at cheaper query. 
    base.query_disc_inclusive(ptg, radius, pixset, 1);
    long nr = pixset.nranges();
    for (long i=0; i<nr && i<max_pairs; ++i){
        out_lohi[2*i]   = (long)pixset.ivbegin(i);
        out_lohi[2*i+1] = (long)pixset.ivend(i);
    }
    return nr;
}

// Range search in nest scheme; this is slower than in ring scheme
extern "C" long hpx_query_disc_nest_ranges_exact(long nside, const double *vec, double radius,
                                                 long *out_lohi, long max_pairs){
    T_Healpix_Base<int64> base((int64)nside, NEST, SET_NSIDE);
    pointing ptg(vec3(vec[0], vec[1], vec[2]));
    rangeset<int64> pixset;
    base.query_disc(ptg, radius, pixset);
    long nr = pixset.nranges();
    for (long i=0; i<nr && i<max_pairs; ++i){
        out_lohi[2*i]   = (long)pixset.ivbegin(i);
        out_lohi[2*i+1] = (long)pixset.ivend(i);
    }
    return nr;
}

// Range search in ring scheme; this is faster than in nest scheme
extern "C" long hpx_query_disc_ring_ranges_exact(long nside, const double *vec, double radius,
                                                 long *out_lohi, long max_pairs){
    T_Healpix_Base<int64> base((int64)nside, RING, SET_NSIDE);
    pointing ptg(vec3(vec[0], vec[1], vec[2]));
    rangeset<int64> pixset;
    base.query_disc(ptg, radius, pixset);
    long nr = pixset.nranges();
    for (long i=0; i<nr && i<max_pairs; ++i){
        out_lohi[2*i]   = (long)pixset.ivbegin(i);
        out_lohi[2*i+1] = (long)pixset.ivend(i);
    }
    return nr;
}


