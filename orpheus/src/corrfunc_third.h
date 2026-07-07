#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <complex.h>

#include <multires_structs.h>

//////////////////////////
/// Counts 3PCF related //
//////////////////////////
// Scalar NNN triplet counts, struct interface (single multireso catalog like
// basetree/doubletree GGG). Currently has no Python binding (kept for parity).
void alloc_Gammans_doubletree_nnn(const MultiresoCatalog *cat, const NavHash *nav,
                                  const TreeResoParams *tree, const BinningParams *bin,
                                  int nthreads, int verbose, NPCFOutput *out);

/////////////////////////
/// Shear 3PCF related //
/////////////////////////
// GGG shear 3PCF Discrete/Tree: struct interface (hoist-to-locals shim, validated
// inner loops unchanged). Discrete takes one nresos=1 catalog; Tree adds a reduced
// per-reso field catalog (cat_field) alongside the base query catalog (cat).
void alloc_Gammans_discrete_ggg(const MultiresoCatalog *cat, const NavHash *nav,
                                const BinningParams *bin, int nthreads, int verbose,
                                NPCFOutput *out);

void alloc_Gammans_tree_ggg(const MultiresoCatalog *cat, const MultiresoCatalog *cat_field,
                            const NavHash *nav, const TreeResoParams *tree,
                            const BinningParams *bin, int nthreads, int verbose,
                            NPCFOutput *out);

// GGG shear 3PCF DoubleTree: struct interface + metric dispatch on cat->metric
// to _ggg_flat / _ggg_spherical (corrfunc_third.c). Replaces the retired
// positional alloc_Gammans_doubletree_ggg; mirrors alloc_gg_doubletree.
void alloc_ggg_doubletree(const MultiresoCatalog *cat, const NavHash *nav,
                          const TreeResoParams *tree, const BinningParams *bin,
                          int nthreads, int verbose, NPCFOutput *out);

// GGG BaseTree: same single multi-resolution catalog as doubletree (base = reso 0,
// with isinner_resos + the occupied-region list in nav), struct interface.
void alloc_Gammans_basetree_ggg(const MultiresoCatalog *cat, const NavHash *nav,
                                const TreeResoParams *tree, const BinningParams *bin,
                                int nthreads, int verbose, NPCFOutput *out);

// Slab-hashed polar-polar-polar (GGG) cross-correlator in the projected '3dbox'
// geometry (line-of-sight window |dz| < Pi): three polar (shape) vertices. Emits
// the raw, f-free 4-component SSS numerator (out->npcf) and the shared random RRR
// count (out->norm_mp); the Python layer applies f = W_S/W_R and forms the III
// estimator S.S.S / RRR (Vedder et al. 2026, arXiv:2601.17914 Eq.17). The shape
// catalog is looped (numerator central) and hashed (nav_polar, G-legs); the shape
// random is looped (RRR central) and hashed (nav_R, count legs). The Gamma_3 and
// RRR r-index layout matches multipoles2npcf_ggg (transposed in the two r bins).
void alloc_Gammans_slab_GGG(const MultiresoCatalog *cat_polar, const NavHash *nav_polar,
                            const MultiresoCatalog *cat_R, const NavHash *nav_R,
                            const BinningParams *bin, int nthreads, int verbose,
                            NPCFOutput *out);

///////////////////
/// G3L related ///
///////////////////
// Shape (source) central + two scalar lens legs. Struct interface (hoist-to-
// locals shim). Discrete: two nresos=1 catalogs. DoubleTree: two multireso
// catalogs + full tree params. Source nav carries the occupied-region list.
void alloc_Gammans_discrete_GNN(const MultiresoCatalog *cat_source, const NavHash *nav_source,
                                const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                                const BinningParams *bin, int nthreads, int verbose,
                                NPCFOutput *out);

void alloc_Gammans_doubletree_GNN(const MultiresoCatalog *cat_source, const NavHash *nav_source,
                                  const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                                  const TreeResoParams *tree, const BinningParams *bin,
                                  int nthreads, int verbose, NPCFOutput *out);

// Slab-hashed polar-scalar-scalar (GNN) cross-correlator in the projected
// '3dbox' geometry (line-of-sight window |dz| < Pi): polar central + two scalar
// legs (data D + random R). Emits the four raw, f-free numerator sub-correlators
// S.(D/R).(D/R) (out->npcf, component axis of length 4) and the shared random RRR
// count (out->norm_mp); the Python layer applies f = W_D/W_R and forms the ggI
// estimator S.D~.D~ / RRR (Vedder et al. 2026, arXiv:2601.17914 Eq.17). Struct
// interface; the polar central is looped directly, D and R are slab-hashed on the
// shared grid, R is also looped as the RRR central.
void alloc_Gammans_slab_GNN(const MultiresoCatalog *cat_polar, const MultiresoCatalog *cat_D,
                            const NavHash *nav_D, const MultiresoCatalog *cat_R,
                            const NavHash *nav_R, const BinningParams *bin,
                            int nthreads, int verbose, NPCFOutput *out);

//////////////////////////////
// Lens-Shear-Shear related //
//////////////////////////////
// Scalar-position central lens + two shear (source) legs. Struct interface
// (hoist-to-locals shim). Discrete: two nresos=1 catalogs. Tree: reduced source
// field + base lens. DoubleTree: two multireso catalogs + full tree params. The
// lens (central) nav carries the occupied-region list.
void alloc_Gammans_discrete_NGG(const MultiresoCatalog *cat_source, const NavHash *nav_source,
                                const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                                const BinningParams *bin, int nthreads, int verbose,
                                NPCFOutput *out);

void alloc_Gammans_tree_NGG(const MultiresoCatalog *cat_source, const NavHash *nav_source,
                            const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                            const TreeResoParams *tree, const BinningParams *bin,
                            int nthreads, int verbose, NPCFOutput *out);

void alloc_Gammans_doubletree_NGG(const MultiresoCatalog *cat_source, const NavHash *nav_source,
                                  const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                                  const TreeResoParams *tree, const BinningParams *bin,
                                  int nthreads, int verbose, NPCFOutput *out);

// Slab-hashed scalar-polar-polar (NGG) cross-correlator in the projected '3dbox'
// geometry (line-of-sight window |dz| < Pi): scalar (density) central + two polar
// legs. Emits the two raw, f-free numerator sub-correlators D.S.S / R.S.S (each 2
// natural components; out->npcf, component axis of length 2) and the shared random
// RRR count (out->norm_mp); the Python layer applies f = W_D/W_R and forms the gII
// estimator D~.S.S / RRR (Vedder et al. 2026, arXiv:2601.17914 Eq.17). The polar
// legs use the shape-data catalog (nav_shapeD); the single random (lens random) is
// looped as the R central and hashed (nav_lensR) for the RRR count legs.
void alloc_Gammans_slab_NGG(const MultiresoCatalog *cat_lensD, const MultiresoCatalog *cat_lensR,
                            const MultiresoCatalog *cat_shapeD, const NavHash *nav_shapeD,
                            const NavHash *nav_lensR, const BinningParams *bin,
                            int nthreads, int verbose, NPCFOutput *out);
