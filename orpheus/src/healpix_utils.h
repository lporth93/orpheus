#ifndef ORPHEUS_HEALPIX_UTILS_H
#define ORPHEUS_HEALPIX_UTILS_H


#ifdef __cplusplus
extern "C" {
#endif

double hpx_nside2resol(long nside);
long hpx_ang2pix_nest(long nside, const double *vec);
void hpx_nest2ring_range(long nside, long p0, long n, long *out);
void hpx_pix2vec_nest(long nside, long ipix, double *vec);
void hpx_pix2vec_ring(long nside, long ipix, double *vec);

long hpx_query_disc_nest(long nside, const double *vec, double radius,
                         long *out, long max_out);
long hpx_query_disc_nest_ranges(long nside, const double *vec, double radius,
                                long *out_lohi, long max_pairs);
long hpx_query_disc_nest_ranges_exact(long nside, const double *vec, double radius,
                                      long *out_lohi, long max_pairs);
long hpx_query_disc_ring_ranges_exact(long nside, const double *vec, double radius,
                                      long *out_lohi, long max_pairs);


#ifdef __cplusplus
}
#endif

#endif
