#ifndef ORPHEUS_HEALPIX_UTILS_H
#define ORPHEUS_HEALPIX_UTILS_H

// C-callable shim around healpix_cxx, so the (C, OpenMP) estimators can run the
// nested-HEALPix disc query live in their hot loop -- the curved-sky replacement
// for the flat pixel-box neighbour enumeration. The disc query itself is the
// tested healpix_cxx query_disc_inclusive; we only expose an extern "C" surface.

#ifdef __cplusplus
extern "C" {
#endif

// Nested-scheme disc query. Writes, into caller-allocated `out` (capacity
// `max_out`), the NESTED indices of every pixel overlapping a disc of angular
// radius `radius` (radians) centred on unit vector `vec` (length 3), and returns
// the pixel count. "Inclusive" => no pixel overlapping the disc is ever missed,
// so a leg within `radius` of the centre is always in a returned pixel (exact
// separation is then filtered in the estimator). If the return value exceeds
// `max_out`, `out` holds only the first `max_out` ids: the caller must grow the
// buffer to >= the returned size and call again.
long hpx_query_disc_nest(long nside, const double *vec, double radius,
                         long *out, long max_out);

// Same inclusive disc, returned as its native sorted, disjoint NESTED-id ranges
// (the rangeset query_disc already produces) rather than expanded pixels: writes
// [lo0,hi0, lo1,hi1, ...) half-open pairs into `out_lohi` (capacity `max_pairs`
// pairs) and returns the number of ranges. This lets the caller merge-join the
// disc against a sorted occupied-cell list instead of binary-searching every
// pixel. If the return exceeds `max_pairs`, grow the buffer and call again.
long hpx_query_disc_nest_ranges(long nside, const double *vec, double radius,
                                long *out_lohi, long max_pairs);

// Mean angular pixel size sqrt(4*pi/npix) in radians (matches healpy nside2resol).
double hpx_nside2resol(long nside);

// Nested pixel id (nside) containing unit vector `vec` (length 3). Used by the
// GGG doubletree to map a central to its coarse super-galaxy cell for the
// cross-resolution Gn cache (the curved-sky analogue of the flat pix2redpix).
long hpx_ang2pix_nest(long nside, const double *vec);

#ifdef __cplusplus
}
#endif

#endif
