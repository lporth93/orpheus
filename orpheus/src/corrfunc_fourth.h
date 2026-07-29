#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <complex.h>

#include "multires_structs.h"

/////////////////////////
// Counts 4PCF related //
/////////////////////////

void alloc_notomoNap4_tree_nnnn(const MultiresoCatalog *cat_base, const MultiresoCatalog *cat_leaf,
    const NavHash *nav, const TreeResoParams *tree, const BinningParams *bin, const FourthParams *fourth,
    double *napradii, int nnapradii, double complex *N4correlators, int alloc_4pcfmultipoles, int alloc_4pcfreal,
    int nthreads, int verbose,
    double *bin_centers, double complex *N_n, double complex *Counts);

void alloc_notomoNap4_tree_nnnn_highmem(const MultiresoCatalog *cat_base, const MultiresoCatalog *cat_leaf,
    const NavHash *nav, const TreeResoParams *tree, const BinningParams *bin, const FourthParams *fourth,
    double *napradii, int nnapradii, double complex *N4correlators, int alloc_4pcfmultipoles, int alloc_4pcfreal,
    int nthreads, int verbose,
    double *bin_centers, double complex *N_n, double complex *Counts);

void alloc_nnnn_tree(const MultiresoCatalog *cat_base, const MultiresoCatalog *cat_leaf,
    const NavHash *nav, const TreeResoParams *tree, const BinningParams *bin, const FourthParams *fourth,
    double memory_bound, int nthreads, int verbose, NPCFOutput *out);

void alloc_nnnn_tree_spherical(const MultiresoCatalog *cat_base, const MultiresoCatalog *cat_leaf,
    const NavHash *nav, const TreeResoParams *tree, const BinningParams *bin, const FourthParams *fourth,
    double memory_bound, int nthreads, int verbose, NPCFOutput *out);

void alloc_nnnn_doubletree(const MultiresoCatalog *cat_leaf, const NavHash *nav,
    const TreeResoParams *tree, const BinningParams *bin, const FourthParams *fourth,
    double memory_bound, int nthreads, int verbose, NPCFOutput *out);

////////////////////////
// Shear 4PCF related //
////////////////////////
void alloc_notomoGammans_discrete_gggg(const MultiresoCatalog *cat, const NavHash *nav,
                                       const BinningParams *bin, const FourthParams *fourth,
                                       int nthreads, int verbose, NPCFOutput *out);

void alloc_notomoGammans_tree_gggg(const MultiresoCatalog *cat_base, const MultiresoCatalog *cat_leaf,
                                   const NavHash *nav, const TreeResoParams *tree,
                                   const BinningParams *bin, const FourthParams *fourth,
                                   int nthreads, int verbose, NPCFOutput *out);

void alloc_notomoMap4_disc_gggg(const MultiresoCatalog *cat, const NavHash *nav,
    const BinningParams *bin, const FourthParams *fourth,
    int projection, double *mapradii, int nmapradii, double complex *M4correlators, int alloc_4pcfmultipoles, int alloc_4pcfreal,
    int nthreads, int verbose,
    double *bin_centers, double complex *Upsilon_n, double complex *N_n, double complex *Gammas, double complex *Norms);

void alloc_notomoMap4_tree_gggg(const MultiresoCatalog *cat_base, const MultiresoCatalog *cat_leaf,
    const NavHash *nav, const TreeResoParams *tree,
    const BinningParams *bin, const FourthParams *fourth,
    int projection, double *mapradii, int nmapradii, double complex *M4correlators, int alloc_4pcfmultipoles, int alloc_4pcfreal,
    int nthreads, int verbose,
    double *bin_centers, double complex *Upsilon_n, double complex *N_n, double complex *Gammas, double complex *Norms);

/////////////////
// G4L related //
/////////////////
void alloc_notomoGammans_discrete_gnnn(const MultiresoCatalog *cat_source, const NavHash *nav_source,
                                       const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                                       const BinningParams *bin, const FourthParams *fourth,
                                       int nthreads, int verbose, NPCFOutput *out);

void alloc_notomoGammans_tree_gnnn(const MultiresoCatalog *cat_source, const NavHash *nav_source,
                                   const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                                   const TreeResoParams *tree, const BinningParams *bin,
                                   const FourthParams *fourth, int nthreads, int verbose, NPCFOutput *out);

void alloc_notomoMapNap3_tree_gnnn(const MultiresoCatalog *cat_source, const NavHash *nav_source,
    const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
    const TreeResoParams *tree, const BinningParams *bin, const FourthParams *fourth,
    const ClustCorr *clustcorr,
    double *apradii, int napradii, int alloc_4pcfmultipoles, int alloc_4pcfreal,
    int nthreads, int verbose,
    double *bin_centers, double complex *Gtilde_n, double complex *N_n,
    double complex *Gtilde, double complex *Norms, double complex *NM3correlator);
