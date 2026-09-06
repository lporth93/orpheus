#ifndef MULTIRES_STRUCTS_H
#define MULTIRES_STRUCTS_H

#include <stdint.h>
#include <complex.h>

// Geometry tag, shared across all correlators.
#define METRIC_FLAT      0   // 2D flat geometry
#define METRIC_SPHERICAL 1   // spherical geometry
#define METRIC_3DBOX     2   // 3D box + line-of-sight slabs

// Structure related to tracer catalogs and their reductions. Unused fields are set to NULL
typedef struct {
    // General content
    int metric;
    int nresos;
    int *ngal_resos;
    int nbinsz;
    double *isinner_resos;
    double *weight_resos;
    int    *zbin_resos;
    // Cartesian position coordinate (used for 2d and 3d flat methods)
    double *pos1_resos;
    double *pos2_resos;
    double *pos3_resos;
    // 3D-vectors used for spherical ra/dec coordinates (used for sphierical geometries)
    double *vx_resos;
    double *vy_resos;
    double *vz_resos;
    // Helper quantities for the spin-2 geodesic bearing (used for sphierical geometries)
    double *ra_resos;
    double *sindec_resos;
    double *cosdec_resos;
    // Spin-2 tracer quantities (Used for SpinTracerCatalogs)
    double *e1_resos;
    double *e2_resos;
    double *weightsq_resos;
} MultiresoCatalog;

// Quantities neccessary to navigate through a multihash nested spatial hashing structur
typedef struct {
    int metric;
    // Relevant for flat2d metric
    int *index_matcher;
    int *pixs_galind_bounds;
    int *pix_gals;
    double pix1_start, pix1_d; int pix1_n;
    double pix2_start, pix2_d; int pix2_n;
    int nregions;
    int *index_matcher_hash;
    int *filledregions;
    int nfilledregions;
    // Offset parameters that are additionally relevant for 3dbox ensemble of multihashes
    int *slab_offsets;
    int *rshift_bounds;
    int nslabs;
    double z0, dpix_z;
    // Relevant for spherical metric
    int  *ncells_resos;
    long *nside_nav;
    long *cell_pix;
    int  *cell_redbounds;
    int  *rshift_red;
    int  *rshift_cellpix;
    int  *rshift_cellbounds; 
} NavHash;

// Quantities neccessary to describe the resolution levels of the multihashes
typedef struct {
    int nresos;
    int nresos_grid;
    double *dpix1_resos;
    double *dpix2_resos; 
    double *reso_redges;
    int resoshift_leafs;
    int minresoind_leaf;
    int maxresoind_leaf;
    int batch_membudget_mb;
} TreeResoParams;

// NPCF binning in real space (N=2) and multipole space (N=3).
typedef struct {
    double rmin, rmax;
    int nbinsr;
    int do_dc;
    int nmax;
    int nmin;
    int dccorr;
    double Pi;
    double *rbins;
} BinningParams;

// All quantities that are are returned by the npcf estimator functions
// See multires_structs.py for a more detailed explanation
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

// NPCF binning parameters used for the various fourth-order correlators
typedef struct {
    int nbinsphi1, nbinsphi2;
    double *phibins1, *phibins2, *dbinsphi1, *dbinsphi2;
    int *nindices; int len_nindices;
    int nthetacombis, nthetbatches;
    int *thetacombis_batches, *nthetacombis_batches, *cumthetacombis_batches;
    double count_floor;
} FourthParams;

// Pars related to the clustering-correction of GNL-correlators
typedef struct {
    double count_floor;
    double *xi_nn; double thetamin_xi, thetamax_xi, dtheta_xi; int has_xi;
    double *zeta, *zeta_rbins; int zeta_nr; double *zeta_phis; int zeta_nphi; int has_zeta;
} ClustCorr;

#endif // MULTIRES_STRUCTS_H