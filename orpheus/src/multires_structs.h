#ifndef MULTIRES_STRUCTS_H
#define MULTIRES_STRUCTS_H

#include <stdint.h>
#include <complex.h>

// Geometry tag, shared across every correlator order. The flat path keeps the
// validated pixel-box navigation; the spherical path swaps in the nested-HEALPix
// bucket hash + a live query_disc (no global frame, geodesic distance). Adding a
// geometry to one correlator order means adding a metric branch in *its* dispatch
// function only -- these structs and their field layout don't change.
#define METRIC_FLAT      0
#define METRIC_SPHERICAL 1
#define METRIC_3DBOX     2   // flat transverse hash + line-of-sight slabs (Vedder IA)

// ---------------------------------------------------------------------------
// Per-galaxy multi-resolution data. Geometry- and order-agnostic: unused
// fields are NULL. Core fields (isinner/weight/zbin) are used by every order;
// pos1/pos2 vs vx/vy/vz are geometry-exclusive; e1/e2/weightsq are shear-only
// (NULL for NN, populated for GGG/GGL).
// ---------------------------------------------------------------------------
typedef struct {
    int metric;
    int nresos;
    int *ngal_resos;            // [nresos]
    int nbinsz;

    double *isinner_resos;      // [sum ngal_resos], order/geometry-agnostic
    double *weight_resos;
    int    *zbin_resos;

    // --- flat / 3dbox transverse ---
    double *pos1_resos;
    double *pos2_resos;
    // --- 3dbox line-of-sight coordinate (NULL except METRIC_3DBOX) ---
    double *pos3_resos;

    // --- spherical-only (unit-vector reduced galaxies; cf. multihash_spherical's red_vx/vy/vz) ---
    double *vx_resos;
    double *vy_resos;
    double *vz_resos;
    // (ra, sin dec, cos dec) reduced galaxies; the spin-2 kernels feed these to
    // sphere_bearing for the geodesic projection. NULL for flat and for scalar NN.
    double *ra_resos;
    double *sindec_resos;
    double *cosdec_resos;

    // --- shear-only (NULL for NN) ---
    double *e1_resos;
    double *e2_resos;
    double *weightsq_resos;     // GGG-only today (sum of w^2 per super-galaxy)
} MultiresoCatalog;

// ---------------------------------------------------------------------------
// Spatial navigation. Geometry-exclusive fields; metric tag picks the active
// half. pixs_galind_bounds / pix_gals / index_matcher are identical in spirit
// (and, for the flat case, in literal field name) to the per-level structure
// in Catalog._multihash; the spherical case is the bucket-hash CSR from
// Catalog.multihash_spherical.
// ---------------------------------------------------------------------------
typedef struct {
    int metric;

    // --- flat ---
    int *index_matcher;         // dense grid -> occupied-cell-slot lookup, per reso
    int *pixs_galind_bounds;    // CSR bounds into pix_gals, per reso
    int *pix_gals;              // galaxy indices in cell order, per reso
    double pix1_start, pix1_d; int pix1_n;
    double pix2_start, pix2_d; int pix2_n;
    int nregions;
    int *index_matcher_hash;
    int *filledregions;         // GGG-only today; belongs here, not in the call signature
    int nfilledregions;

    // --- 3dbox slab (extends the flat transverse hash above with line-of-sight
    // slabs of width dpix_z starting at z0; pix_gals is ordered slab-major, with
    // rshift_bounds[slab] the offset into pixs_galind_bounds for each slab) ---
    int *slab_offsets;
    int *rshift_bounds;
    int nslabs;
    double z0, dpix_z;

    // --- spherical ---
    int  *ncells_resos;         // [nresos]
    long *nside_nav;            // [nresos]
    long *cell_pix;             // sorted nested ids, concatenated over resos
    int  *cell_redbounds;       // CSR bounds into the reduced arrays, concatenated
    int  *rshift_red;           // [nresos+1] offsets into red_*
    int  *rshift_cellpix;       // [nresos+1] offsets into cell_pix
    int  *rshift_cellbounds;    // [nresos+1] offsets into cell_redbounds
} NavHash;

// ---------------------------------------------------------------------------
// Shared multi-resolution tree parameters -- identical between NN and GGG
// today (reso_redges, leaf-resolution shift/clamp params, per-reso true pixel
// sizes for the flat grid hierarchy). Geometry-agnostic.
// ---------------------------------------------------------------------------
typedef struct {
    int nresos;
    int nresos_grid;
    double *dpix1_resos;        // flat-grid true pixel sizes per level (GGG uses
    double *dpix2_resos;        // these for the reduced-pixel search box; NN's
                                 // flat path doesn't need them but carries them
                                 // for signature uniformity across orders)
    double *reso_redges;        // [nresos+1], radians for spherical, native unit for flat
    int resoshift_leafs;
    int minresoind_leaf;
    int maxresoind_leaf;
} TreeResoParams;

typedef struct {
    double rmin, rmax;
    int nbinsr;
    int do_dc;
    int nmax;    // multipole order (GGG and higher); NN/GG ignore it
    int nmin;    // lowest multipole order (GGG discrete); default 0
    int dccorr;  // multi-count correction toggle (GGG); NN/GG ignore it
    double Pi;   // line-of-sight window half-width (METRIC_3DBOX); ignored otherwise
    double *rbins;  // explicit log-r bin edges (GGG discrete); NULL -> recomputed
} BinningParams;

// ---------------------------------------------------------------------------
// Unified NPCF output, runtime-sized. Replaces NN/GG/NG/GGGOutput: every order
// returns bin_centers plus some subset of {complex npcf, real pair-weight norm,
// complex multipole norm, integer pair/triplet counts}. Unused fields are NULL.
// Per-order layout (ncomp = number of natural components stacked in npcf):
//   NN      npcf=NULL;                norm=<weighted pair count>; npair_cell set
//   GG      npcf=[xip,xim] (ncomp=2); norm=<pair weight>;         npair set
//   NG      npcf=xi        (ncomp=1); norm=<pair weight>;         npair set
//   GGG     npcf=Gammans   (ncomp=4); norm_mp=Gammans_norm
//   GNN/NGG npcf=Upsilon_n (ncomp=1); norm_mp=Norm_n
// ---------------------------------------------------------------------------
typedef struct {
    double *bin_centers;
    double complex *npcf;
    double *norm;
    double complex *norm_mp;
    long long int *npair;
    long long int *npair_cell;
    int ncomp;
    int nmax;
} NPCFOutput;

// Public entry point: a thin metric dispatch (see corrfunc_second.c). This is
// the only NN symbol the Python ctypes binding calls. GGG's analogous entry
// (alloc_ggg_doubletree) would have the identical shape: dispatch on cat->metric
// to _ggg_flat / _ggg_spherical, sharing these same four input struct types.
//void alloc_nn_doubletree(const MultiresoCatalog *cat, const NavHash *nav,
//                         const TreeResoParams *tree, const BinningParams *bin,
//                         int nthreads, int verbose, NPCFOutput *out);

#endif // MULTIRES_STRUCTS_H
