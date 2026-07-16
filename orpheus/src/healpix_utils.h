#ifndef ORPHEUS_HEALPIX_UTILS_H
#define ORPHEUS_HEALPIX_UTILS_H


#ifdef __cplusplus
extern "C" {
#endif

long hpx_query_disc_nest(long nside, const double *vec, double radius,
                         long *out, long max_out);
long hpx_query_disc_nest_ranges(long nside, const double *vec, double radius,
                                long *out_lohi, long max_pairs);
double hpx_nside2resol(long nside);
long hpx_ang2pix_nest(long nside, const double *vec);

#ifdef __cplusplus
}
#endif

#endif
