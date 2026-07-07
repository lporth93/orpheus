#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <complex.h>

#include <multires_structs.h>

/////////////////////////
/// Shear 2PCF related //
/////////////////////////

// alloc_nn_doubletree (scalar pair counts) now takes the shared multi-resolution
// structs; its declaration lives in multires_structs.h.

// Public entry point: a thin metric dispatch (see corrfunc_second.c). This is
// the only NN symbol the Python ctypes binding calls. GGG's analogous entry
// (alloc_ggg_doubletree) would have the identical shape: dispatch on cat->metric
// to _ggg_flat / _ggg_spherical, sharing these same four input struct types.
void alloc_nn_doubletree(const MultiresoCatalog *cat, const NavHash *nav,
                         const TreeResoParams *tree, const BinningParams *bin,
                         int nthreads, int verbose, NPCFOutput *out);

// alloc_gg_doubletree (shear 2PCF natural components) mirrors alloc_nn_doubletree:
// a thin metric dispatch on cat->metric to _gg_flat / _gg_spherical (corrfunc_second.c),
// sharing the same four input struct types. The struct types come from the
// multires_structs.h include above.
void alloc_gg_doubletree(const MultiresoCatalog *cat, const NavHash *nav,
                         const TreeResoParams *tree, const BinningParams *bin,
                         int nthreads, int verbose, NPCFOutput *out);

// alloc_ng_doubletree (position-shape 2PCF / galaxy-galaxy lensing) cross-
// correlates a scalar lens catalog (cat_lens/nav_lens, the central) with a
// spin-2 source catalog (cat_source/nav_source, the field). Both hashes share
// the same flat grid (built on a joint extent). Flat-sky only for now.
void alloc_ng_doubletree(const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                         const MultiresoCatalog *cat_source, const NavHash *nav_source,
                         const TreeResoParams *tree, const BinningParams *bin,
                         int nthreads, int verbose, NPCFOutput *out);

// Discrete, slab-hashed position-shape (NI / w_{g+}) estimator for the '3dbox'
// geometry (Vedder et al. 2026, arXiv:2601.17914 Eq. 15). Query = positions;
// the hashed catalog carries the shapes (has_shapes=1) or supplies random pair
// counts (has_shapes=0). Only the line-of-sight-window metric distinguishes it
// from the flat discrete correlator, so it lives here beside the other 2nd-order
// kernels rather than in a separate translation unit.
void ng_slab(
    double *q_pos1, double *q_pos2, double *q_pos3, double *q_w, int *q_zbin,
    int q_ngal, int nbinsz_q,
    double *h_pos1, double *h_pos2, double *h_pos3, double *h_w, int *h_zbin,
    double *h_e1, double *h_e2, int nbinsz_h,
    int nslabs, double z0, double dpix_z,
    double pix1_start, double pix1_d, int pix1_n,
    double pix2_start, double pix2_d, int pix2_n,
    int *slab_offsets, int *index_matcher, int *pixs_galind_bounds,
    int *rshift_bounds, int *pix_gals,
    double rmin, double rmax, int nbinsr, double Pi,
    int self_pairs, int has_shapes, int nthreads,
    double *out_xs_re, double *out_xs_im, double *out_wnorm,
    double *out_rsum, long *out_npairs);