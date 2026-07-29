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


void alloc_nn_doubletree(const MultiresoCatalog *cat, const NavHash *nav,
                         const TreeResoParams *tree, const BinningParams *bin,
                         int nthreads, int verbose, NPCFOutput *out);

void alloc_gg_doubletree(const MultiresoCatalog *cat, const NavHash *nav,
                         const TreeResoParams *tree, const BinningParams *bin,
                         int nthreads, int verbose, NPCFOutput *out);

void alloc_ng_doubletree(const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                         const MultiresoCatalog *cat_source, const NavHash *nav_source,
                         const TreeResoParams *tree, const BinningParams *bin,
                         int nthreads, int verbose, NPCFOutput *out);

void ng_slab(const MultiresoCatalog *cat_query, const MultiresoCatalog *cat_hash,
             const NavHash *nav_hash, const BinningParams *bin,
             int self_pairs, int has_shapes, int nthreads, int verbose, NPCFOutput *out);