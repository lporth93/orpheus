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
// Scalar NNN correlation in DoubleTree approximation
void alloc_nnn_doubletree(const MultiresoCatalog *cat, const NavHash *nav,
                          const TreeResoParams *tree, const BinningParams *bin,
                          int nthreads, int verbose, NPCFOutput *out);

/////////////////////////
/// Shear 3PCF related //
/////////////////////////
// Polar GGG correlation in brute force
void alloc_Gammans_discrete_ggg(const MultiresoCatalog *cat, const NavHash *nav,
                                const BinningParams *bin, int nthreads, int verbose,
                                NPCFOutput *out);

// Polar GGG correlation in Tree approximation
void alloc_Gammans_tree_ggg(const MultiresoCatalog *cat, const MultiresoCatalog *cat_field,
                            const NavHash *nav, const TreeResoParams *tree,
                            const BinningParams *bin, int nthreads, int verbose,
                            NPCFOutput *out);

// Polar GGG correlation in DoubleTree approximation
void alloc_ggg_doubletree(const MultiresoCatalog *cat, const NavHash *nav,
                          const TreeResoParams *tree, const BinningParams *bin,
                          int nthreads, int verbose, NPCFOutput *out);

// Polar GGG correlation in BaseTree approximation
void alloc_Gammans_basetree_ggg(const MultiresoCatalog *cat, const NavHash *nav,
                                const TreeResoParams *tree, const BinningParams *bin,
                                int nthreads, int verbose, NPCFOutput *out);

// Discrete GGG correlators used for slab computation
void alloc_Gammans_slab_GGG(const MultiresoCatalog *cat_polar, const NavHash *nav_polar,
                            const MultiresoCatalog *cat_R, const NavHash *nav_R,
                            const BinningParams *bin, int nthreads, int verbose,
                            NPCFOutput *out);

///////////////////
/// G3L related ///
///////////////////
// Mixed scalar-polar GNN correlation in brute force
void alloc_Gammans_discrete_GNN(const MultiresoCatalog *cat_source, const NavHash *nav_source,
                                const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                                const BinningParams *bin, int nthreads, int verbose,
                                NPCFOutput *out);

// Mixed scalar-polar GNN correlation in DoubleTree approximation
void alloc_Gammans_doubletree_GNN(const MultiresoCatalog *cat_source, const NavHash *nav_source,
                                  const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                                  const TreeResoParams *tree, const BinningParams *bin,
                                  int nthreads, int verbose, NPCFOutput *out);

// Discrete GNN correlators used for slab computation
void alloc_Gammans_slab_GNN(const MultiresoCatalog *cat_polar, const MultiresoCatalog *cat_D,
                            const NavHash *nav_D, const MultiresoCatalog *cat_R,
                            const NavHash *nav_R, const BinningParams *bin,
                            int nthreads, int verbose, NPCFOutput *out);

//////////////////////////////
// Lens-Shear-Shear related //
//////////////////////////////
// Mixed scalar-polar NGG correlation in brute force
void alloc_Gammans_discrete_NGG(const MultiresoCatalog *cat_source, const NavHash *nav_source,
                                const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                                const BinningParams *bin, int nthreads, int verbose,
                                NPCFOutput *out);

// Mixed scalar-polar NGG correlation in Tree approximation
void alloc_Gammans_tree_NGG(const MultiresoCatalog *cat_source, const NavHash *nav_source,
                            const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                            const TreeResoParams *tree, const BinningParams *bin,
                            int nthreads, int verbose, NPCFOutput *out);

// Mixed scalar-polar NGG correlation in DoubleTree approximation
void alloc_Gammans_doubletree_NGG(const MultiresoCatalog *cat_source, const NavHash *nav_source,
                                  const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                                  const TreeResoParams *tree, const BinningParams *bin,
                                  int nthreads, int verbose, NPCFOutput *out);

// Discrete NGG correlators used for slab computation
void alloc_Gammans_slab_NGG(const MultiresoCatalog *cat_lensD, const MultiresoCatalog *cat_lensR,
                            const MultiresoCatalog *cat_shapeD, const NavHash *nav_shapeD,
                            const NavHash *nav_lensR, const BinningParams *bin,
                            int nthreads, int verbose, NPCFOutput *out);
