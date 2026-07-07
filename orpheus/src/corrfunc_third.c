#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <complex.h>

#include "spatialhash.h"
#include "assign.h"
#include "corrfunc_third.h"
#include "utils.h"
#include "healpix_utils.h"
#include "multires_structs.h"

#define mymin(x,y) ((x) <= (y)) ? (x) : (y)
#define mymax(x,y) ((x) >= (y)) ? (x) : (y)
#define M_PI      3.14159265358979323846
#define INV_2PI   0.15915494309189534561

// ---------------------------------------------------------------------------
// Shared multi-resolution region setup (flat basetree/doubletree kernels).
//
// Every flat multi-reso kernel repeats the same cold per-region setup: the
// radial-bin -> resolution edge map, cumulative offsets into the stacked
// per-reso hash arrays, per-region galaxy counts + reduced-grid offsets, the
// pixel -> reduced-pixel matcher, and the region cache-slot layout. The
// helpers below hold the single copy of each block; the per-pair hot loops
// stay in the kernels.
// ---------------------------------------------------------------------------

// Radial bin -> resolution edge indices. Depends only on the binning and band
// edges; identical for flat and spherical (edges in the same unit as rmin/rmax).
static void build_reso_rindedges(int nresos, const double *reso_redges,
    double rmin, double rmax, int nbinsr, int *reso_rindedges){
    double logrmin = log(rmin);
    double drbin = (log(rmax)-logrmin)/(nbinsr);
    int tmpreso = 0;
    double thisredge = 0;
    double tmpr = rmin;
    for (int elr=0;elr<nbinsr;elr++){
        tmpr *= exp(drbin);
        thisredge = reso_redges[mymin(nresos,tmpreso+1)];
        if (thisredge<tmpr){
            reso_rindedges[mymin(nresos,tmpreso+1)] = elr;
            if ((tmpr-thisredge)<(thisredge - (tmpr/exp(drbin)))){reso_rindedges[mymin(nresos,tmpreso+1)]+=1;}
            tmpreso+=1;
        }
    }
    reso_rindedges[nresos] = nbinsr;
}

// Cumulative per-reso offsets into the stacked multi-resolution hash arrays
// (index_matcher / pixs_galind_bounds / pix_gals). Output arrays calloc'd by
// the caller so entry 0 stays 0.
static void build_rshift_offsets(int nresos, int npix_hash, const int *ngal_resos,
    int *rshift_index_matcher, int *rshift_pixs_galind_bounds, int *rshift_pix_gals){
    for (int elreso=1;elreso<nresos;elreso++){
        rshift_index_matcher[elreso] = rshift_index_matcher[elreso-1] + npix_hash;
        rshift_pixs_galind_bounds[elreso] = rshift_pixs_galind_bounds[elreso-1] + ngal_resos[elreso-1]+1;
        rshift_pix_gals[elreso] = rshift_pix_gals[elreso-1] + ngal_resos[elreso-1];
    }
}

// Per-region galaxy counts per (zbin, reso) of the hashed central catalog and
// cumulative pixel offsets of the reduced grids; returns len_matcher. Output
// arrays calloc'd by the caller.
static int build_region_galinpix(int nresos, int nresos_grid, int hasdiscrete,
    int elregion, const int *pixs_galind_bounds, const int *rshift_pixs_galind_bounds,
    const int *pix_gals, const int *rshift_pix_gals, const int *zbin_resos,
    int *matchers_resoshift, int *ngal_in_pix){
    for (int elreso=0;elreso<nresos;elreso++){
        int elreso_grid = elreso - hasdiscrete;
        int lower = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion];
        int upper = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion+1];
        for (int ind_inpix=lower; ind_inpix<upper; ind_inpix++){
            int ind_gal = rshift_pix_gals[elreso] + pix_gals[rshift_pix_gals[elreso]+ind_inpix];
            ngal_in_pix[zbin_resos[ind_gal]*nresos+elreso] += 1;
        }
        if (elreso_grid>=0){
            int npix_side = 1 << (nresos_grid-elreso_grid-1);
            matchers_resoshift[elreso_grid+1] = matchers_resoshift[elreso_grid] + npix_side*npix_side;
        }
    }
    return matchers_resoshift[nresos_grid];
}

// Dense per-zbin index of each hashed galaxy's reduced pixel at every grid
// resolution, plus the region's hash-pixel origin (needed later to key a
// central into pix2redpix). pix2redpix calloc'd by the caller.
static void build_region_pix2redpix(int nresos_grid, int hasdiscrete, int elregion,
    int nbinsz, const int *index_matcher_hash,
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d,
    const int *pixs_galind_bounds, const int *rshift_pixs_galind_bounds,
    const int *pix_gals, const int *rshift_pix_gals, const int *zbin_resos,
    const double *pos1_resos, const double *pos2_resos,
    const double *dpix1_resos, const double *dpix2_resos,
    const int *matchers_resoshift, int len_matcher,
    double *hashpix_start1, double *hashpix_start2, int *pix2redpix){
    int elregion_fullhash = index_matcher_hash[elregion];
    double hstart1 = pix1_start + (elregion_fullhash%pix1_n)*pix1_d;
    double hstart2 = pix2_start + (elregion_fullhash/pix1_n)*pix2_d;
    for (int elreso=0;elreso<nresos_grid;elreso++){
        int thisreso = elreso + hasdiscrete;
        int lower = pixs_galind_bounds[rshift_pixs_galind_bounds[thisreso]+elregion];
        int upper = pixs_galind_bounds[rshift_pixs_galind_bounds[thisreso]+elregion+1];
        int npix_side = 1 << (nresos_grid-elreso-1);
        int *tmpcounts = calloc(nbinsz, sizeof(int));
        for (int ind_inpix=lower; ind_inpix<upper; ind_inpix++){
            int ind_gal = rshift_pix_gals[thisreso] + pix_gals[rshift_pix_gals[thisreso]+ind_inpix];
            int zbin_gal = zbin_resos[ind_gal];
            int elhashpix_1 = (int) floor((pos1_resos[ind_gal] - hstart1)/dpix1_resos[elreso]);
            int elhashpix_2 = (int) floor((pos2_resos[ind_gal] - hstart2)/dpix2_resos[elreso]);
            int elhashpix = elhashpix_2*npix_side + elhashpix_1;
            pix2redpix[zbin_gal*len_matcher+matchers_resoshift[elreso]+elhashpix] = tmpcounts[zbin_gal];
            tmpcounts[zbin_gal] += 1;
        }
        free(tmpcounts);
    }
    *hashpix_start1 = hstart1;
    *hashpix_start2 = hstart2;
}

// Region cache-slot layout (cumresoshift_z / thetashifts_z / zbinshifts and the
// derived zbin2shift / nshift) from the central catalog's per-(zbin, reso)
// counts. The discrete band (reso 0 when hasdiscrete) shares the reso-1 grid
// slots, so its own count is skipped. nshift spans the partner catalog's zbins
// (equal to the central's for single-catalog kernels). Shift arrays are
// assumed zeroed on entry (calloc'd per region).
static void setup_region_shifts(int nbinsz_central, int nbinsz_partner, int nresos,
    int hasdiscrete, int nbinsr, const int *ngal_in_pix,
    int *cumresoshift_z, int *thetashifts_z, int *zbinshifts,
    int *zbin2shift, int *nshift){
    for (int elz=0; elz<nbinsz_central; elz++){
        for (int elreso=0; elreso<nresos; elreso++){
            if (hasdiscrete==1 && elreso==0){
                cumresoshift_z[elz*(nresos+1) + elreso+1] = ngal_in_pix[elz*nresos + elreso+1];
            } else {
                cumresoshift_z[elz*(nresos+1) + elreso+1] =
                    cumresoshift_z[elz*(nresos+1) + elreso] + ngal_in_pix[elz*nresos + elreso];
            }
        }
        thetashifts_z[elz] = cumresoshift_z[elz*(nresos+1) + nresos];
        zbinshifts[elz+1] = zbinshifts[elz] + nbinsr*thetashifts_z[elz];
    }
    *zbin2shift = zbinshifts[nbinsz_central];
    *nshift = nbinsz_partner*(*zbin2shift);
}


void alloc_Gammans_doubletree_nnn(const MultiresoCatalog *cat, const NavHash *nav,
                                  const TreeResoParams *tree, const BinningParams *bin,
                                  int nthreads, int verbose, NPCFOutput *out){
    // Scalar NNN triplet counts, single multi-resolution catalog (base = reso 0),
    // struct interface (hoist-to-locals shim). Triplets_n lives in out->npcf.
    int nresos = tree->nresos, nresos_grid = tree->nresos_grid;
    double *dpix1_resos = tree->dpix1_resos, *dpix2_resos = tree->dpix2_resos, *reso_redges = tree->reso_redges;
    int resoshift_leafs = tree->resoshift_leafs, minresoind_leaf = tree->minresoind_leaf, maxresoind_leaf = tree->maxresoind_leaf;
    int *ngal_resos = cat->ngal_resos, nbinsz = cat->nbinsz, *zbin_resos = cat->zbin_resos;
    double *isinner_resos = cat->isinner_resos, *weight_resos = cat->weight_resos;
    double *pos1_resos = cat->pos1_resos, *pos2_resos = cat->pos2_resos, *weightsq_resos = cat->weightsq_resos;
    int *index_matcher = nav->index_matcher, *pixs_galind_bounds = nav->pixs_galind_bounds, *pix_gals = nav->pix_gals;
    double pix1_start = nav->pix1_start, pix1_d = nav->pix1_d; int pix1_n = nav->pix1_n;
    double pix2_start = nav->pix2_start, pix2_d = nav->pix2_d; int pix2_n = nav->pix2_n;
    int *index_matcher_hash = nav->index_matcher_hash, nregions = nav->nregions;
    int *filledregions = nav->filledregions, nfilledregions = nav->nfilledregions;
    int nmax = bin->nmax, nbinsr = bin->nbinsr, dccorr = bin->dccorr;
    double rmin = bin->rmin, rmax = bin->rmax;
    double *bin_centers = out->bin_centers;
    double complex *Triplets_n = out->npcf;

    // Index shift for the Gamman
    int _gamma_zshift = nbinsr*nbinsr;
    int _gamma_nshift = _gamma_zshift*nbinsz*nbinsz*nbinsz;
    int _gamma_compshift = (nmax+1)*_gamma_nshift;
    
    // Helper array that checks how many regions have been already computed
    int *regionsdone = calloc(nfilledregions, sizeof(int));
    int nregionsdone = 0;
    
    double *totcounts = calloc(nbinsz*nbinsr, sizeof(double));
    double *totnorms = calloc(nbinsz*nbinsr, sizeof(double));
    
    // Temporary arrays that are allocated in parallel and later reduced
    double *tmpwcounts = calloc(nthreads*nbinsz*nbinsr, sizeof(double));
    double *tmpwnorms = calloc(nthreads*nbinsz*nbinsr, sizeof(double));
    double complex *tmpTriplets_n = calloc(nthreads*_gamma_compshift, sizeof(double complex));
    
    #pragma omp parallel for num_threads(nthreads)
    for(int elthread=0;elthread<nthreads;elthread++){
        int nregions_per_thread = nfilledregions/nthreads;
        int hasdiscrete = nresos-nresos_grid;
        int nnvals_Nn = nmax+1;
        
        // Compute how large the caches have to be at most for this thread
        // Largest possible nshift: each zbin does completely fill out the lowest reso grid.
        // The remaining grids then have 1/4 + 1/16 + ... --> 0.33.... times the data of the largest grid. 
        // Now allocate the caches
        int size_max_nshift = (int) ((1+hasdiscrete+0.34)*nbinsz*nbinsz*nbinsr*pow(4,nresos_grid-1));
        double complex *Nncache = calloc(nnvals_Nn*size_max_nshift, sizeof(double complex));
        double complex *wNncache = calloc(nnvals_Nn*size_max_nshift, sizeof(double complex));
        int *Nncache_updates = calloc(size_max_nshift, sizeof(int));
        for (int _elregion=0; _elregion<2*nfilledregions; _elregion++){
            int region_debug=-99999;
            
            // Check if this thread needs to allocate the region. In the first pass we split the work evenly 
            // while in the second pass we just work on the next best region, s.t. the 'fast' threads will
            // steal work from the 'slow' threads.
            int elregion;
            int wasdone = 0;
            if (_elregion<nfilledregions){
                int nthread_target = mymin(_elregion/nregions_per_thread, nthreads-1);
                if (nthread_target!=elthread){continue;}
            }
            elregion = filledregions[_elregion%nfilledregions];
            #pragma omp critical
            {   
                if (regionsdone[_elregion%nfilledregions]==1){wasdone = 1;}
                else{
                    regionsdone[_elregion%nfilledregions]=1;
                    nregionsdone+=1; 
                }
            }
            if (wasdone==1){continue;}
            bool printregdbg = (verbose>1) && (elregion==region_debug);

            // Check which sets of radii are evaluated for each resolution
            int *reso_rindedges = calloc(nresos+1, sizeof(int));
            double logrmin = log(rmin);
            double drbin = (log(rmax)-logrmin)/(nbinsr);
            build_reso_rindedges(nresos, reso_redges, rmin, rmax, nbinsr, reso_rindedges);
                        
            // Shift variables for 3pcf quantities
            int gamma_zshift = nbinsr*nbinsr;
            int gamma_nshift = gamma_zshift*nbinsz*nbinsz*nbinsz;
            int gamma_compshift = (nmax+1)*gamma_nshift;
            
            // Shift variables for spatial hash
            int npix_hash = pix1_n*pix2_n;
            int *rshift_index_matcher = calloc(nresos, sizeof(int));
            int *rshift_pixs_galind_bounds = calloc(nresos, sizeof(int));
            int *rshift_pix_gals = calloc(nresos, sizeof(int));
            build_rshift_offsets(nresos, npix_hash, ngal_resos,
                rshift_index_matcher, rshift_pixs_galind_bounds, rshift_pix_gals);
            
            // Shift variables for the matching between the pixel grids
            int lower1, upper1, lower2, upper2;
            int *matchers_resoshift = calloc(nresos_grid+1, sizeof(int));
            int *ngal_in_pix = calloc(nresos*nbinsz, sizeof(int));
            int len_matcher = build_region_galinpix(nresos, nresos_grid, hasdiscrete,
                elregion, pixs_galind_bounds, rshift_pixs_galind_bounds,
                pix_gals, rshift_pix_gals, zbin_resos, matchers_resoshift, ngal_in_pix);

            // Build the matcher from pixels to reduced pixels in the region
            double hashpix_start1, hashpix_start2;
            int *pix2redpix = calloc(nbinsz*len_matcher, sizeof(int)); // For each z matches pixel in unreduced grid to index in reduced grid
            build_region_pix2redpix(nresos_grid, hasdiscrete, elregion, nbinsz,
                index_matcher_hash, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d,
                pixs_galind_bounds, rshift_pixs_galind_bounds, pix_gals, rshift_pix_gals,
                zbin_resos, pos1_resos, pos2_resos, dpix1_resos, dpix2_resos,
                matchers_resoshift, len_matcher, &hashpix_start1, &hashpix_start2, pix2redpix);

            // Setup all shift variables for the Gncache in the region
            // Gncache has structure
            // n --> zbin2 --> zbin1 --> radius
            //   --> [ [0]*ngal_zbin1_reso1 | [0]*ngal_zbin1_reso1/2 | ... | [0]*ngal_zbin1_reson ]
            int *cumresoshift_z = calloc(nbinsz*(nresos+1), sizeof(int));
            int *thetashifts_z = calloc(nbinsz, sizeof(int));
            int *zbinshifts = calloc(nbinsz+1, sizeof(int));
            int zbin2shift, nshift;
            setup_region_shifts(nbinsz, nbinsz, nresos, hasdiscrete, nbinsr, ngal_in_pix,
                cumresoshift_z, thetashifts_z, zbinshifts, &zbin2shift, &nshift);
            // Set all the cache indices that are updated in this region to zero
            if (printregdbg){printf("zbin2shift=%d: nshift=%d: \n", zbin2shift,  nshift);}
            for (int _i=0; _i<nnvals_Nn*nshift; _i++){ Nncache[_i] = 0; wNncache[_i] = 0;}
            for (int _i=0; _i<nshift; _i++){ Nncache_updates[_i] = 0;}
            int Nncache_totupdates=0;
            
            // Now, for each resolution, loop over all the galaxies in the region and
            // allocate the Gn & Nn, as well as their caches  for the corresponding 
            // set of radii
            // For elreso in resos
            //.  for gal in reso 
            //.    allocate Gn for allowed radii
            //.    allocate the Gncaches
            //.    compute the Gamman for all combinations of the same resolution
            int ind_pix1, ind_pix2, ind_inpix1, ind_inpix2, ind_red, ind_gal1, ind_gal2, z_gal1, z_gal2;
            int ind_Gn, ind_Gnnorm, ind_Gncacheshift, ind_Nncacheshift;
            int rbin, nextn, nextnshift, nbinszr, nbinszr_reso, zrshift, ind_rbin;
            double innergal, pos1_gal1, pos2_gal1, pos1_gal2, pos2_gal2, w_gal1, w_gal2, wsq_gal2;
            double rel1, rel2, dist;
            double complex _wwphic, _wwphi;
            double complex nphirot, twophirotc, nphirotc, phirot, phirotc;
            double rmin_reso, rmax_reso;
            int elreso_leaf, rbinmin, rbinmax, rbinmin1, rbinmax1, rbinmin2, rbinmax2;
            int nzero = nmax+3;
            nbinszr =  nbinsz*nbinsr;
            for (int elreso=0;elreso<nresos;elreso++){
                //elreso_leaf = mymin(mymax(minresoind_leaf,elreso+resoshift_leafs),maxresoind_leaf);
                elreso_leaf = elreso;
                rbinmin = reso_rindedges[elreso];
                rbinmax = reso_rindedges[elreso+1];
                rmin_reso = rmin*exp(rbinmin*drbin);
                rmax_reso = rmin*exp(rbinmax*drbin);
                int nbinsr_reso = rbinmax-rbinmin;
                nbinszr_reso = nbinsz*nbinsr_reso;
                lower1 = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion];
                upper1 = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion+1];
                double complex *nextWns =  calloc(nnvals_Nn*nbinszr_reso, sizeof(double complex));
                double complex *nextW2ns =  calloc(nbinszr_reso, sizeof(double complex));
                double complex *nextW2ndiscs =  calloc(nbinszr_reso, sizeof(double complex));
                int *nextncounts = calloc(nbinszr_reso, sizeof(int));
                int *allowedrinds = calloc(nbinszr_reso, sizeof(int));
                int *allowedzinds = calloc(nbinszr_reso, sizeof(int));
                if (printregdbg){printf("rbinmin=%d, rbinmax%d\n",rbinmin,rbinmax);}
                for (ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                    ind_gal1 = rshift_pix_gals[elreso] + pix_gals[rshift_pix_gals[elreso]+ind_inpix1];
                    innergal = isinner_resos[ind_gal1];
                    if (innergal<1e-5){continue;}
                    z_gal1 = zbin_resos[ind_gal1];
                    pos1_gal1 = pos1_resos[ind_gal1];
                    pos2_gal1 = pos2_resos[ind_gal1];
                    w_gal1 = innergal*weight_resos[ind_gal1];
                    
                    int pix1_lower = mymax(0, (int) floor((pos1_gal1 - (rmax_reso+pix1_d) - pix1_start)/pix1_d));
                    int pix2_lower = mymax(0, (int) floor((pos2_gal1 - (rmax_reso+pix2_d) - pix2_start)/pix2_d));
                    int pix1_upper = mymin(pix1_n-1, (int) floor((pos1_gal1 + (rmax_reso+pix1_d) - pix1_start)/pix1_d));
                    int pix2_upper = mymin(pix2_n-1, (int) floor((pos2_gal1 + (rmax_reso+pix2_d) - pix2_start)/pix2_d));
                    
                    for (ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                        for (ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                            ind_red = index_matcher[rshift_index_matcher[elreso_leaf] + ind_pix2*pix1_n + ind_pix1];
                            if (ind_red==-1){continue;}
                            lower2 = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso_leaf]+ind_red];
                            upper2 = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso_leaf]+ind_red+1];
                            for (ind_inpix2=lower2; ind_inpix2<upper2; ind_inpix2++){
                                ind_gal2 = rshift_pix_gals[elreso_leaf] + pix_gals[rshift_pix_gals[elreso_leaf]+ind_inpix2];
                                pos1_gal2 = pos1_resos[ind_gal2];
                                pos2_gal2 = pos2_resos[ind_gal2];
                                w_gal2 = weight_resos[ind_gal2];
                                wsq_gal2 = weightsq_resos[ind_gal2];
                                z_gal2 = zbin_resos[ind_gal2];
                                
                                rel1 = pos1_gal2 - pos1_gal1;
                                rel2 = pos2_gal2 - pos2_gal1;
                                dist = sqrt(rel1*rel1 + rel2*rel2);
                                if(dist < rmin_reso || dist >= rmax_reso) continue;
                                rbin = (int) floor((log(dist)-logrmin)/drbin) - rbinmin;
                                
                                phirot = (rel1+I*rel2)/dist;// * fabs(rel1)/rel1;
                                phirotc = conj(phirot);
                                twophirotc = phirotc*phirotc;
                                zrshift = z_gal2*nbinsr_reso + rbin;
                                ind_rbin = elthread*nbinszr + z_gal2*nbinsr + rbin+rbinmin;
                                
                                // nmin=0 
                                //   -> Wn axis: [0,...,nmax]
                                ind_Gnnorm = zrshift;
                                nphirot = 1+I*0;
                                
                                // n = 0
                                nextncounts[zrshift] += 1;
                                tmpwcounts[ind_rbin] += w_gal1*w_gal2*dist; 
                                tmpwnorms[ind_rbin] += w_gal1*w_gal2; 
                                nextWns[ind_Gnnorm] += w_gal2*nphirot;  
                                nextW2ns[zrshift] += w_gal2*w_gal2;
                                nextW2ndiscs[zrshift] += wsq_gal2;
                                nphirot *= phirot;
                                
                                // n in [1, ..., nmax-1] x {+1,-1}
                                nextnshift = 0;
                                for (nextn=1;nextn<=nmax;nextn++){
                                    nextnshift = nextn*nbinszr_reso;
                                    nextWns[ind_Gnnorm+nextnshift] += w_gal2*nphirot;  
                                    nphirot *= phirot;
                                }
                            }
                        }
                    }
                    
                    // Update the Gncache and Gnnormcache
                    int red_reso2, npix_side_reso2, elhashpix_1_reso2, elhashpix_2_reso2, elhashpix_reso2, redpix_reso2;
                    double complex thisWn;
                    int _tmpindcache, _tmpindWn;
                    for (int elreso2=elreso; elreso2<nresos; elreso2++){
                        red_reso2 = elreso2 - hasdiscrete;
                        if (hasdiscrete==1 && elreso==0 && elreso2==0){red_reso2 += hasdiscrete;}
                        npix_side_reso2 = 1 << (nresos_grid-red_reso2-1);
                        elhashpix_1_reso2 = (int) floor((pos1_gal1 - hashpix_start1)/dpix1_resos[red_reso2]);
                        elhashpix_2_reso2 = (int) floor((pos2_gal1 - hashpix_start2)/dpix2_resos[red_reso2]);
                        elhashpix_reso2 = elhashpix_2_reso2*npix_side_reso2 + elhashpix_1_reso2;
                        redpix_reso2 = pix2redpix[z_gal1*len_matcher+matchers_resoshift[red_reso2]+elhashpix_reso2];
                        for (int zbin2=0; zbin2<nbinsz; zbin2++){
                            if (printregdbg){
                                printf("Gnupdates for reso1=%d reso2=%d red_reso2=%d, galindex=%d, z1=%d, z2=%d:%d radial updates; shiftstart %d = %d+%d+%d+%d+%d \n"
                                       ,elreso,elreso2,red_reso2,ind_gal1,z_gal1,zbin2,rbinmax-rbinmin,
                                       zbin2*zbin2shift + zbinshifts[z_gal1] + rbinmin*thetashifts_z[z_gal1] + 
                                       cumresoshift_z[z_gal1*(nresos+1) + elreso2] + redpix_reso2,
                                       zbin2*zbin2shift, zbinshifts[z_gal1], rbinmin*thetashifts_z[z_gal1],
                                       cumresoshift_z[z_gal1*(nresos+1) + elreso2], redpix_reso2);
                            }
                            for (int thisrbin=rbinmin; thisrbin<rbinmax; thisrbin++){
                                zrshift = zbin2*nbinsr_reso + thisrbin-rbinmin;
                                if (cabs(nextWns[zrshift])<1e-10){continue;}
                                ind_Gncacheshift = zbin2*zbin2shift + zbinshifts[z_gal1] + thisrbin*thetashifts_z[z_gal1] + 
                                    cumresoshift_z[z_gal1*(nresos+1) + elreso2] + redpix_reso2;
                                _tmpindWn = zrshift;
                                _tmpindcache = ind_Gncacheshift;
                                for(int thisn=0; thisn<nnvals_Nn; thisn++){
                                    thisWn = nextWns[_tmpindWn];
                                    Nncache[_tmpindcache] += thisWn;
                                    wNncache[_tmpindcache] += w_gal1*thisWn;
                                    _tmpindWn += nbinszr_reso;
                                    _tmpindcache += nshift;
                                }
                                Nncache_updates[ind_Gncacheshift] += 1;
                                Nncache_totupdates += 1;
                            }
                        } 
                    }
                    
                    // Allocate same reso Gammas
                    // First check for zero count bins (most likely only in discrete-discrete bit)
                    int nallowedcounts = 0;
                    for (int zbin1=0; zbin1<nbinsz; zbin1++){
                        for (int elb1=0; elb1<nbinsr_reso; elb1++){
                            zrshift = zbin1*nbinsr_reso + elb1;
                            if (nextncounts[zbin1*nbinsr_reso + elb1] != 0){
                                allowedrinds[nallowedcounts] = elb1;
                                allowedzinds[nallowedcounts] = zbin1;
                                nallowedcounts += 1;
                            }
                        }
                    }
                    // Now update the Gammans
                    // tmpGammas have shape (nthreads, nmax+1, nzcombis3, r*r, 4)
                    // Gns have shape (nnvals, nbinsz, nbinsr)
                    double complex h0, h1, h2, h3, w0, Gmnm3;
                    int thisnshift;
                    int _gammashift1, gammashift1, gammashift;
                    int ind_mnm3, ind_mnm1, ind_nm3, ind_nm1, ind_norm;
                    int _zcombi, zcombi, elb1_full, elb2_full;
                    for (int thisn=0; thisn<nmax+1; thisn++){
                        ind_norm = thisn*nbinszr_reso;
                        thisnshift = elthread*gamma_compshift + thisn*gamma_nshift;
                        int elb1, zbin2;
                        for (int zrcombis1=0; zrcombis1<nallowedcounts; zrcombis1++){
                            elb1 = allowedrinds[zrcombis1];
                            zbin2 = allowedzinds[zrcombis1];
                            elb1_full = elb1 + rbinmin;
                            zrshift = zbin2*nbinsr_reso + elb1;
                            // Double counting correction
                            if (dccorr==1){
                                zcombi = z_gal1*nbinsz*nbinsz + zbin2*nbinsz + zbin2;
                                gammashift1 = thisnshift + zcombi*gamma_zshift + elb1_full*nbinsr;
                                tmpTriplets_n[gammashift1 + elb1_full] -=  w_gal1*nextW2ns[zrshift];
                            }
                            w0 = w_gal1 * nextWns[ind_norm + zrshift];
                            _zcombi = z_gal1*nbinsz*nbinsz+zbin2*nbinsz;
                            _gammashift1 = thisnshift + elb1_full*nbinsr;
                            for (int zrcombis2=0; zrcombis2<nallowedcounts; zrcombis2++){
                                zcombi = _zcombi+allowedzinds[zrcombis2];
                                gammashift1 = _gammashift1 + zcombi*gamma_zshift; 
                                elb2_full = allowedrinds[zrcombis2] + rbinmin;
                                zrshift = allowedzinds[zrcombis2]*nbinsr_reso + allowedrinds[zrcombis2];
                                tmpTriplets_n[gammashift1 + elb2_full] += w0*conj(nextWns[ind_norm + zrshift]);
                            }
                        }
                    }
                    
                    for (int _i=0;_i<nnvals_Nn*nbinszr_reso;_i++){nextWns[_i]=0;}
                    for (int _i=0;_i<nbinszr_reso;_i++){nextW2ns[_i]=0; nextW2ndiscs[_i]=0; 
                                                        nextncounts[_i]=0; allowedrinds[_i]=0; allowedzinds[_i]=0;}
                }
                free(nextWns);
                free(nextW2ns);
                free(nextW2ndiscs);
                free(nextncounts);
                free(allowedrinds);
                free(allowedzinds);
            }
            
            // Allocate the Gamman for different grid resolutions from all the cached arrays 
            //
            // Note that for different configurations of the resolutions we do the Gamman
            // allocation as follows - see eq. (32) in 2309.08601 for the reasoning:
            // * Gamma0 = wshape * G_nm3 * G_mnm3
            //          --> (wG_nm3) * G_mnm3 if reso1 < reso2
            //          --> G_nm3 * wG_mnm3   if reso1 > reso2
            // * Gamma1 = conj(wshape) * G_nm1 * G_mnm1
            //          --> cwG_nm1 * G_mnm1 if reso1 < reso2
            //          --> G_nm1 * cwG_mnm1 if reso1 > reso2
            // * Gamma2 = wshape * conj(G_mnm1) * G_mnm3
            //          --> conj(cwG_mnm1) * G_mnm3 if reso1 < reso2
            //          --> conj(G_mnm1) * wG_mnm3  if reso1 > reso2
            // * Gamma3 = wshape * G_nm3 * conj(G_nm1)
            //          --> wG_nm3 * conj(G_nm1)  if reso1 < reso2
            //          --> G_nm3 * conj(cwG_nm1) if reso1 > reso2
            // where wG_xxx := wshape*G_xxx and cwG_xxx := conj(wshape)*G_xxx
            double complex w0;
            int thisnshift;
            int gammashift1, gammashift;
            int zcombi;
            for (int thisn=0; thisn<nmax+1; thisn++){
                thisnshift = elthread*gamma_compshift + thisn*gamma_nshift;
                
                for (int zbin1=0; zbin1<nbinsz; zbin1++){
                    for (int zbin2=0; zbin2<nbinsz; zbin2++){
                        for (int zbin3=0; zbin3<nbinsz; zbin3++){
                            zcombi = zbin1*nbinsz*nbinsz + zbin2*nbinsz + zbin3;
                            int _in;
                            int _thetashift_z = thetashifts_z[zbin1];
                            //if (zcombis_allowed[zcombi]==0){continue;}
                            
                            // Case max(reso1, reso2) = reso2
                            for (int thisreso1=0; thisreso1<nresos; thisreso1++){
                                //rbinmin1 = (int) floor((log(reso_redges[thisreso1])-logrmin)/drbin);
                                //rbinmax1= mymin((int) floor((log(reso_redges[thisreso1+1])-logrmin)/drbin), nbinsr-1);
                                rbinmin1 = reso_rindedges[thisreso1];
                                rbinmax1 = reso_rindedges[thisreso1+1];
                                for (int thisreso2=thisreso1+1; thisreso2<nresos; thisreso2++){
                                    //rbinmin2 = (int) floor((log(reso_redges[thisreso2])-logrmin)/drbin);
                                    //rbinmax2= mymin((int) floor((log(reso_redges[thisreso2+1])-logrmin)/drbin), nbinsr-1);
                                    rbinmin2 = reso_rindedges[thisreso2];
                                    rbinmax2 = reso_rindedges[thisreso2+1];
                                    for (int elgal=0; elgal<ngal_in_pix[zbin1*nresos+thisreso2]; elgal++){
                                        for (int elb1=rbinmin1; elb1<rbinmax1; elb1++){
                                            gammashift1 = thisnshift + zcombi*gamma_zshift + elb1*nbinsr;
                                            // n --> zbin2 --> zbin1 --> radius --> [ [0]*ngal_zbin1_reso1 | ... | [0]*ngal_zbin1_reson ]
                                            ind_Nncacheshift = zbin2*zbin2shift + zbinshifts[zbin1] + elb1*thetashifts_z[zbin1] +
                                                cumresoshift_z[zbin1*(nresos+1) + thisreso2] + elgal;
                                            w0 = wNncache[thisn*nshift + ind_Nncacheshift];
                                            ind_Nncacheshift = zbin3*zbin2shift + zbinshifts[zbin1] + rbinmin2*thetashifts_z[zbin1] +
                                                    cumresoshift_z[zbin1*(nresos+1) + thisreso2] + elgal;
                                            _in = thisn*nshift + ind_Nncacheshift;
                                            for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                                tmpTriplets_n[gammashift1 + elb2] += w0*conj(Nncache[_in]);
                                                ind_Nncacheshift += _thetashift_z;
                                                _in += _thetashift_z;
                                            }
                                        }
                                    }
                                }
                            }
                            
                            // Case max(reso1, reso2) = reso1
                            for (int thisreso2=0; thisreso2<nresos; thisreso2++){
                                //rbinmin2 = (int) floor((log(reso_redges[thisreso2])-logrmin)/drbin);
                                //rbinmax2= mymin((int) floor((log(reso_redges[thisreso2+1])-logrmin)/drbin), nbinsr-1);
                                rbinmin2 = reso_rindedges[thisreso2];
                                rbinmax2 = reso_rindedges[thisreso2+1];
                                for (int thisreso1=thisreso2+1; thisreso1<nresos; thisreso1++){
                                    //rbinmin1 = (int) floor((log(reso_redges[thisreso1])-logrmin)/drbin);
                                    //rbinmax1= mymin((int) floor((log(reso_redges[thisreso1+1])-logrmin)/drbin), nbinsr-1);
                                    rbinmin1 = reso_rindedges[thisreso1];
                                    rbinmax1 = reso_rindedges[thisreso1+1];
                                    for (int elgal=0; elgal<ngal_in_pix[zbin1*nresos+thisreso1]; elgal++){
                                        for (int elb1=rbinmin1; elb1<rbinmax1; elb1++){
                                            gammashift1 = thisnshift + zcombi*gamma_zshift + elb1*nbinsr;
                                            ind_Nncacheshift = zbin2*zbin2shift + zbinshifts[zbin1] + elb1*thetashifts_z[zbin1] +
                                                cumresoshift_z[zbin1*(nresos+1) + thisreso1] + elgal;                                            
                                            w0 = Nncache[thisn*nshift + ind_Nncacheshift];
                                            ind_Nncacheshift = zbin3*zbin2shift + zbinshifts[zbin1] + rbinmin2*thetashifts_z[zbin1] +
                                                    cumresoshift_z[zbin1*(nresos+1) + thisreso1] + elgal;
                                            _in = thisn*nshift + ind_Nncacheshift;
                                            for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                                tmpTriplets_n[gammashift1 + elb2] += w0*conj(wNncache[_in]);
                                                ind_Nncacheshift += _thetashift_z;
                                                _in += _thetashift_z;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }            
            free(reso_rindedges);
            free(rshift_index_matcher);
            free(rshift_pixs_galind_bounds);
            free(rshift_pix_gals);
            free(matchers_resoshift);
            free(ngal_in_pix);
            free(pix2redpix);  
            free(cumresoshift_z);
            free(thetashifts_z);
            free(zbinshifts);
            
            // Update progress bar
            print_progress(nregionsdone, nfilledregions, verbose);

        }
        free(Nncache);
        free(wNncache);
        free(Nncache_updates);
    }
    
    // Accumulate the Gamman
    #pragma omp parallel for num_threads(nthreads)
    for (int thisn=0; thisn<nmax+1; thisn++){
        int itmpGamma, iGamma;
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            for (int zcombi=0; zcombi<nbinsz*nbinsz*nbinsz; zcombi++){
                for (int elb1=0; elb1<nbinsr; elb1++){
                    for (int elb2=0; elb2<nbinsr; elb2++){
                        iGamma = thisn*_gamma_nshift + zcombi*_gamma_zshift + elb1*nbinsr + elb2;
                        itmpGamma = iGamma + thisthread*_gamma_compshift;
                        Triplets_n[iGamma] += tmpTriplets_n[itmpGamma];
                    }
                }
            }
        }
    }
    
    // Accumulate the bin distances and weights
    for (int elbinz=0; elbinz<nbinsz; elbinz++){
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            int tmpind = elbinz*nbinsr + elbinr;
            for (int thisthread=0; thisthread<nthreads; thisthread++){
                int tshift = thisthread*nbinsz*nbinsr; 
                totcounts[tmpind] += tmpwcounts[tshift+tmpind];
                totnorms[tmpind] += tmpwnorms[tshift+tmpind];
            }
        }
    }
    
    // Get bin centers
    for (int elbinz=0; elbinz<nbinsz; elbinz++){
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            int tmpind = elbinz*nbinsr + elbinr;
            if (totnorms[tmpind] != 0){
                bin_centers[tmpind] = totcounts[tmpind]/totnorms[tmpind];
            }
        }
    } 
    
    if (verbose>0){printf("\n");} 

    free(tmpwcounts);
    free(tmpwnorms);
    free(tmpTriplets_n);
    free(totcounts);
    free(totnorms);
    free(regionsdone);
}

///////////////////////////////////////////////
/// THIRD-ORDER SHEAR CORRELATION FUNCTIONS ///
///////////////////////////////////////////////
// Allocates multipoles of shape catalog via discrete estimator
void alloc_Gammans_discrete_ggg(const MultiresoCatalog *cat, const NavHash *nav,
                                const BinningParams *bin, int nthreads, int verbose,
                                NPCFOutput *out){
    // --- unpack the shared structs into the locals the validated body uses (nresos=1) ---
    double *isinner = cat->isinner_resos, *weight = cat->weight_resos;
    double *pos1 = cat->pos1_resos, *pos2 = cat->pos2_resos;
    double *e1 = cat->e1_resos, *e2 = cat->e2_resos;
    int *zbins = cat->zbin_resos, nbinsz = cat->nbinsz, ngal = cat->ngal_resos[0];
    int nmin = bin->nmin, nmax = bin->nmax, nbinsr = bin->nbinsr, dccorr = bin->dccorr;
    double rmin = bin->rmin, rmax = bin->rmax, *rbins = bin->rbins;
    int *index_matcher = nav->index_matcher, *pixs_galind_bounds = nav->pixs_galind_bounds, *pix_gals = nav->pix_gals;
    double pix1_start = nav->pix1_start, pix1_d = nav->pix1_d; int pix1_n = nav->pix1_n;
    double pix2_start = nav->pix2_start, pix2_d = nav->pix2_d; int pix2_n = nav->pix2_n;
    double *bin_centers = out->bin_centers;
    double complex *Gammans = out->npcf, *Gammans_norm = out->norm_mp;

    // Index shift for the Gamman
    int _gamma_zshift = nbinsr*nbinsr;
    int _gamma_nshift = _gamma_zshift*nbinsz*nbinsz*nbinsz;
    int _gamma_compshift = (nmax-nmin+1)*_gamma_nshift;
    
    double *totcounts = calloc(nbinsz*nbinsr, sizeof(double));
    double *totnorms = calloc(nbinsz*nbinsr, sizeof(double));
           
    // Allocate Gns
    // We do this in parallel as follows:
    // * Split survey in 2*nthreads equal area stripes along x-axis
    // * Do two parallelized iterations over galaxies
    //   - In first one only consider galaxies within stripes of even number
    //   - In second one only consider galaxies within stripes of odd number
    // --> We avoid race conditions in calling the spatial hash arrays. - This
    //    is explicitly made sure by (re)setting nthreads in the python layer.
    for (int odd=0; odd<2; odd++){
        
        // Temporary arrays that are allocated in parallel and later reduced
        double *tmpwcounts = calloc(nthreads*nbinsz*nbinsr, sizeof(double));
        double *tmpwnorms = calloc(nthreads*nbinsz*nbinsr, sizeof(double));
        double complex *tmpGammans = calloc(nthreads*4*_gamma_compshift, sizeof(double complex));
        double complex *tmpGammans_norm = calloc(nthreads*_gamma_compshift, sizeof(double complex));
        #pragma omp parallel for num_threads(nthreads)
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            int gamma_zshift = nbinsr*nbinsr;
            int gamma_nshift = _gamma_zshift*nbinsz*nbinsz*nbinsz;
            int gamma_compshift = (nmax-nmin+1)*_gamma_nshift;
            for (int ind_gal=0; ind_gal<ngal; ind_gal++){
                // Check if galaxy falls in stripe used in this process
                double p11, p12, w1, e11, e12;
                int zbin1;
                double innergal;
                #pragma omp critical
                {p11 = pos1[ind_gal];
                p12 = pos2[ind_gal];
                w1 = weight[ind_gal];
                zbin1 = zbins[ind_gal];
                e11 = e1[ind_gal];
                e12 = e2[ind_gal];
                innergal = isinner[ind_gal];}
                if (innergal<1e-5){continue;}
                w1 *= innergal;
                int thisstripe = 2*thisthread + odd;
                int galstripe = (int) floor((p11-pix1_start)/pix1_d * (2*nthreads)/pix1_n);
                if (thisstripe != galstripe){continue;}
                
                int ind_pix1, ind_pix2, ind_inpix, ind_gal2;
                int ind_red, lower, upper; 
                double  p21, p22, w2, z2, e21, e22;
                double rel1, rel2, dist;
                double complex wshape;
                int nnvals, nnvals_norm, nextn, nzero;
                double complex nphirot, twophirotc, nphirotc, phirot, phirotc, phirotm, phirotp, phirotn;
                
                // Check how many ns we need for Gn
                // Gns have shape (nnvals, nbinsz, nbinsr)
                // where the ns are ordered as 
                //  * [-nmax-3, ..., nmax-1] / [0, ..., nmax]
                if (nmin<4){nmin=0;}
                if (nmin==0){nnvals=2*nmax+3;nnvals_norm=nmax+1;}
                else{nnvals=2*(nmax-nmin+3);nnvals_norm=nmax-nmin+1;}
                double complex *nextGns =  calloc(nnvals*nbinsr*nbinsz, sizeof(double complex));
                double complex *nextGns_norm =  calloc(nnvals_norm*nbinsr*nbinsz, sizeof(double complex));
                double complex *nextG2ns =  calloc(4*nbinsz*nbinsr, sizeof(double complex));
                double complex *nextG2ns_norm =  calloc(nbinsz*nbinsr, sizeof(double complex));

                int ind_rbin, rbin;
                int ind_Gn, ind_Gnnorm, zrshift, nextnshift;
                int nbinszr = nbinsz*nbinsr;
                double drbin = (log(rmax)-log(rmin))/(nbinsr);
                /*if (ind_gal%10000==0){
                    printf("%d %d %d %d %d \n",nmin,nmax,nnvals,nbinsr,nbinsz);
                }*/
                int pix1_lower = mymax(0, (int) floor((p11 - (rmax+pix1_d) - pix1_start)/pix1_d));
                int pix2_lower = mymax(0, (int) floor((p12 - (rmax+pix2_d) - pix2_start)/pix2_d));
                int pix1_upper = mymin(pix1_n-1, (int) floor((p11 + (rmax+pix1_d) - pix1_start)/pix1_d));
                int pix2_upper = mymin(pix2_n-1, (int) floor((p12 + (rmax+pix2_d) - pix2_start)/pix2_d));

                for (ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                    for (ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                        ind_red = index_matcher[ind_pix2*pix1_n + ind_pix1];
                        if (ind_red==-1){continue;}
                        lower = pixs_galind_bounds[ind_red];
                        upper = pixs_galind_bounds[ind_red+1];
                        for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                            ind_gal2 = pix_gals[ind_inpix];
                            p21 = pos1[ind_gal2];
                            p22 = pos2[ind_gal2];
                            w2 = weight[ind_gal2];
                            z2 = zbins[ind_gal2];
                            e21 = e1[ind_gal2];
                            e22 = e2[ind_gal2];

                            rel1 = p21 - p11;
                            rel2 = p22 - p12;
                            dist = sqrt(rel1*rel1 + rel2*rel2);
                            if(dist < rmin || dist >= rmax) continue;
                            if (rbins[0] < 0){
                                rbin = (int) floor((log(dist)-log(rmin))/drbin);
                            }
                            else{
                                rbin=0;
                                while(rbins[rbin+1] < dist){rbin+=1;}
                            }
                            wshape = (double complex) w2 * (e21+I*e22);
                            //phirot = csqrt((rel1+I*rel2)/(rel1-I*rel2));
                            //if (rel1<0){phirot*=-1;}
                            //phirotc = conj(phirot);
                            //twophirotc = phirotc*phirotc;
                            double dphi = atan2(rel2,rel1);
                            phirot = cexp(I*dphi);
                            phirotc = conj(phirot);
                            twophirotc = phirotc*phirotc;
                            
                            zrshift = z2*nbinsr + rbin;
                            ind_rbin = thisthread*nbinszr + zrshift;
                            // nmin=0 -
                            //   -> Gns axis: [-nmax-3, ..., -nmin-1, nmin-3, nmax-1]
                            //   -> Gn_norm axis: [0,...,nmax]
                            if (nmin==0){
                                nzero = nmax+3;
                                ind_Gn = nzero*nbinszr + zrshift;
                                ind_Gnnorm = zrshift;
                                nphirot = 1+I*0;
                                nphirotc = 1+I*0;
                                
                                // n = 0
                                tmpwcounts[ind_rbin] += w1*w2*dist; 
                                tmpwnorms[ind_rbin] += w1*w2; 
                                nextGns[ind_Gn] += wshape*nphirot;
                                nextGns_norm[ind_Gnnorm] += w2*nphirot;  
                                nextG2ns[zrshift] += wshape*wshape*twophirotc*twophirotc*twophirotc;
                                nextG2ns[nbinszr+zrshift] += wshape*wshape*twophirotc;
                                nextG2ns[2*nbinszr+zrshift] += wshape*conj(wshape)*twophirotc;
                                nextG2ns[3*nbinszr+zrshift] += wshape*conj(wshape)*twophirotc;
                                nextG2ns_norm[zrshift] += w2*w2;
                                nphirot *= phirot;
                                nphirotc *= phirotc; 
                                // n in [1, ..., nmax-1] x {+1,-1}
                                nextnshift=0;
                                for (nextn=1;nextn<nmax;nextn++){
                                    nextnshift = nextn*nbinszr;
                                    nextGns[ind_Gn+nextnshift] += wshape*nphirot;
                                    nextGns[ind_Gn-nextnshift] += wshape*nphirotc;
                                    nextGns_norm[ind_Gnnorm+nextnshift] += w2*nphirot;  
                                    nphirot *= phirot;
                                    nphirotc *= phirotc; 
                                }
                                // n in [nmax, -nmax, -nmax-1, -nmax-2, -nmax-3]
                                nextGns_norm[ind_Gnnorm+nextnshift+nbinszr] += w2*nphirot;  
                                nextGns[zrshift+3*nbinszr] += wshape*nphirotc;
                                nphirotc *= phirotc; 
                                nextGns[zrshift+2*nbinszr] += wshape*nphirotc;
                                nphirotc *= phirotc; 
                                nextGns[zrshift+nbinszr] += wshape*nphirotc;
                                nphirotc *= phirotc; 
                                nextGns[zrshift] += wshape*nphirotc;
                            }
                            
                            // nmin>3 
                            //   --> Gns axis: [-nmax-3, ..., -nmin+1, nmin-3, ..., nmax+1]
                            //   --> Gn_norm axis: [nmin, ..., nmax]
                            else{
                                phirotm = cpow(phirotc,nmax+3);
                                phirotp = cpow(phirot,nmin-3);
                                phirotn = phirotp*phirot*phirot*phirot;
                                int pshift = (nmax-nmin+3)*nbinszr;
                                nextnshift = zrshift;
                                // n in [-nmax-3, ..., -nmin-3] + [nmin-3, ..., nmax-3]
                                for (nextn=0;nextn<nmax-nmin+1;nextn++){
                                    nextGns[nextnshift] += wshape*phirotm;
                                    nextGns[pshift+nextnshift] += wshape*phirotp;
                                    nextGns_norm[nextnshift] += w2*phirotn;
                                    phirotm *= phirot;
                                    phirotp *= phirot;
                                    phirotn *= phirot;
                                    nextnshift += nbinszr;
                                }
                                // n in [-nmin-2, -nmin-1] + [nmax-2, nmax-1]
                                nextGns[nextnshift] += wshape*phirotm;
                                nextGns[pshift+nextnshift] += wshape*phirotp;
                                phirotm *= phirot;
                                phirotp *= phirot;
                                nextnshift += nbinszr;
                                nextGns[nextnshift] += wshape*phirotm;
                                nextGns[pshift+nextnshift] += wshape*phirotp;
                            } 
                        }
                    }
                }
                
                // Now update the Gammans
                // tmpGammas have shape (nthreads, nmax+1, nzcombis3, r*r, 4)
                // Gns have shape (nnvals, nbinsz, nbinsr)
                //int nonzero_tmpGammas = 0;
                double complex h0, h1, h2, h3, w0;
                int thisnshift, r12shift;
                int gammashift1, gammashiftt1, gammashift, gammashiftt;
                int ind_mnm3, ind_mnm1, ind_nm3, ind_nm1, ind_norm;
                int elb2, zbin3, zcombi;
                wshape = w1 * (e11+I*e12);
                for (int thisn=0; thisn<nmax-nmin+1; thisn++){
                    if (nmin==0){
                        nzero = nmax+3;
                        ind_mnm3 = (nzero-thisn-3)*nbinszr;
                        ind_mnm1 = (nzero-thisn-1)*nbinszr;
                        ind_nm3 = (nzero+thisn-3)*nbinszr;
                        ind_nm1 = (nzero+thisn-1)*nbinszr;
                        ind_norm = thisn*nbinszr;
                    }
                    else{
                        ind_mnm3 = (nmax-nmin-thisn)*nbinszr;
                        ind_mnm1 = (nmax-nmin+2-thisn)*nbinszr;
                        ind_nm3 = (nmax-nmin+3+thisn)*nbinszr;
                        ind_nm1 = (nmax-nmin+5+thisn)*nbinszr;
                        ind_norm = thisn*nbinszr;
                    }
                    thisnshift = thisthread*gamma_compshift + thisn*gamma_nshift;
                    for (int zbin2=0; zbin2<nbinsz; zbin2++){
                        for (int elb1=0; elb1<nbinsr; elb1++){
                            zrshift = zbin2*nbinsr + elb1;
                            h0 = -wshape * nextGns[ind_nm3 + zrshift];
                            h1 = -conj(wshape) * nextGns[ind_nm1 + zrshift];
                            h2 = -wshape * conj(nextGns[ind_mnm1 + zrshift]);
                            h3 = -wshape * conj(nextGns[ind_nm1 + zrshift]);
                            w0 = w1 * conj(nextGns_norm[ind_norm + zrshift]);
                            for (zbin3=0; zbin3<nbinsz; zbin3++){
                                zcombi = zbin1*nbinsz*nbinsz+zbin2*nbinsz+zbin3;
                                gammashift1 = thisnshift + zcombi*gamma_zshift + elb1*nbinsr;
                                gammashiftt1 = thisnshift + zcombi*gamma_zshift;
                                // Double counting correction
                                if (zbin1==zbin2 && zbin1==zbin3 && dccorr==1){
                                    zrshift = zbin2*nbinsr + elb1;
                                    r12shift = elb1*nbinsr+elb1;
                                    gammashift = 4*(gammashift1 + elb1);
                                    gammashiftt = gammashiftt1 + r12shift;
                                    //phirotm = wshape*nextG2ns[zrshift];
                                    tmpGammans[gammashift] += wshape*nextG2ns[zrshift];
                                    tmpGammans[gammashift+1] += conj(wshape)*nextG2ns[nbinszr+zrshift];
                                    tmpGammans[gammashift+2] += wshape*nextG2ns[2*nbinszr+zrshift];
                                    tmpGammans[4*gammashiftt+3] += wshape*nextG2ns[3*nbinszr+zrshift];
                                    tmpGammans_norm[gammashiftt] -= w1*nextG2ns_norm[zrshift];
                                }
                                // Nominal allocation
                                for (elb2=0; elb2<nbinsr; elb2++){
                                    zrshift = zbin3*nbinsr + elb2;
                                    r12shift = elb2*nbinsr+elb1;
                                    gammashift = 4*(gammashift1 + elb2);
                                    gammashiftt = gammashiftt1 + r12shift;
                                    //phirotm = h0*nextGns[ind_mnm3 + zrshift];
                                    tmpGammans[gammashift] += h0*nextGns[ind_mnm3 + zrshift];
                                    tmpGammans[gammashift+1] += h1*nextGns[ind_mnm1 + zrshift];
                                    tmpGammans[gammashift+2] += h2*nextGns[ind_mnm3 + zrshift];
                                    tmpGammans[4*gammashiftt+3] += h3*nextGns[ind_nm3 + zrshift];
                                    tmpGammans_norm[gammashiftt] += w0*nextGns_norm[ind_norm + zrshift];
                                    //if(thisthread==0 && ind_gal%1000==0){
                                    //    if (cabs(tmpGammans[gammashift] )>1e-5){nonzero_tmpGammas += 1;}
                                    //}
                                }
                            }
                        }
                    }
                }
                
                free(nextGns);
                free(nextGns_norm);
                free(nextG2ns);
                free(nextG2ns_norm);
                nextGns = NULL;
                nextGns_norm = NULL;
                nextG2ns = NULL;
                nextG2ns_norm = NULL;
            }
        }
        
        // Accumulate the Gamman
        #pragma omp parallel for num_threads(nthreads)
        for (int thisn=0; thisn<nmax-nmin+1; thisn++){
            int itmpGamma, iGamma;
            for (int thisthread=0; thisthread<nthreads; thisthread++){
                for (int zcombi=0; zcombi<nbinsz*nbinsz*nbinsz; zcombi++){
                    for (int elb1=0; elb1<nbinsr; elb1++){
                        for (int elb2=0; elb2<nbinsr; elb2++){
                            iGamma = thisn*_gamma_nshift + zcombi*_gamma_zshift + elb1*nbinsr + elb2;
                            itmpGamma = iGamma + thisthread*_gamma_compshift;
                            for (int elcomp=0; elcomp<4; elcomp++){
                                Gammans[elcomp*_gamma_compshift+iGamma] += tmpGammans[4*itmpGamma+elcomp];
                            }
                            Gammans_norm[iGamma] += tmpGammans_norm[itmpGamma];
                        }
                    }
                }
            }
        }
        
        // Update the bin distances and weights
        for (int elbinz=0; elbinz<nbinsz; elbinz++){
            for (int elbinr=0; elbinr<nbinsr; elbinr++){
                int tmpind = elbinz*nbinsr + elbinr;
                for (int thisthread=0; thisthread<nthreads; thisthread++){
                    int tshift = thisthread*nbinsz*nbinsr; 
                    totcounts[tmpind] += tmpwcounts[tshift+tmpind];
                    totnorms[tmpind] += tmpwnorms[tshift+tmpind];
                }
            }
        }
        free(tmpwcounts);
        free(tmpwnorms);
        free(tmpGammans);
        free(tmpGammans_norm); 
        tmpwcounts = NULL;
        tmpwnorms = NULL;
        tmpGammans = NULL;
        tmpGammans_norm = NULL;
    }
    
    // Get bin centers
    for (int elbinz=0; elbinz<nbinsz; elbinz++){
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            int tmpind = elbinz*nbinsr + elbinr;
            if (totnorms[tmpind] != 0){
                bin_centers[tmpind] = totcounts[tmpind]/totnorms[tmpind];
            }
            
        }
    } 
    free(totcounts);
    free(totnorms);
    totcounts = NULL;
    totnorms = NULL;
}        
  

void alloc_Gammans_tree_ggg(const MultiresoCatalog *cat, const MultiresoCatalog *cat_field,
                            const NavHash *nav, const TreeResoParams *tree,
                            const BinningParams *bin, int nthreads, int verbose,
                            NPCFOutput *out){
    // --- base (full-resolution query) catalog, nresos=1 ---
    double *isinner = cat->isinner_resos, *weight = cat->weight_resos;
    double *pos1 = cat->pos1_resos, *pos2 = cat->pos2_resos;
    double *e1 = cat->e1_resos, *e2 = cat->e2_resos;
    int *zbins = cat->zbin_resos, nbinsz = cat->nbinsz, ngal = cat->ngal_resos[0];
    // --- reduced per-reso field super-galaxies ---
    int nresos = tree->nresos; double *reso_redges = tree->reso_redges;
    int *ngal_resos = cat_field->ngal_resos, *zbin_resos = cat_field->zbin_resos;
    double *weight_resos = cat_field->weight_resos, *pos1_resos = cat_field->pos1_resos, *pos2_resos = cat_field->pos2_resos;
    double *e1_resos = cat_field->e1_resos, *e2_resos = cat_field->e2_resos, *weightsq_resos = cat_field->weightsq_resos;
    // --- navigation + binning + output ---
    int *index_matcher = nav->index_matcher, *pixs_galind_bounds = nav->pixs_galind_bounds, *pix_gals = nav->pix_gals;
    double pix1_start = nav->pix1_start, pix1_d = nav->pix1_d; int pix1_n = nav->pix1_n;
    double pix2_start = nav->pix2_start, pix2_d = nav->pix2_d; int pix2_n = nav->pix2_n;
    int nmin = bin->nmin, nmax = bin->nmax, nbinsr = bin->nbinsr, dccorr = bin->dccorr;
    double rmin = bin->rmin, rmax = bin->rmax, *rbins = bin->rbins;
    double *bin_centers = out->bin_centers;
    double complex *Gammans = out->npcf, *Gammans_norm = out->norm_mp;

    // Index shift for the Gamman
    int _gamma_zshift = nbinsr*nbinsr;
    int _gamma_nshift = _gamma_zshift*nbinsz*nbinsz*nbinsz;
    int _gamma_compshift = (nmax-nmin+1)*_gamma_nshift;
    
    double *totcounts = calloc(nbinsz*nbinsr, sizeof(double));
    double *totnorms = calloc(nbinsz*nbinsr, sizeof(double));
    
           
    // Allocate Gns
    // We do this in parallel as follows:
    // * Split survey in 2*nthreads equal area stripes along x-axis
    // * Do two parallelized iterations over galaxies
    //   - In first one only consider galaxies within stripes of even number
    //   - In second one only consider galaxies within stripes of odd number
    // --> We avoid race conditions in calling the spatial hash arrays. - This
    //    is explicitly made sure by (re)setting nthreads in the python layer.
    for (int odd=0; odd<2; odd++){
        
        
        // Temporary arrays that are allocated in parallel and later reduced
        double *tmpwcounts = calloc(nthreads*nbinsz*nbinsr, sizeof(double));
        double *tmpwnorms = calloc(nthreads*nbinsz*nbinsr, sizeof(double));
        double complex *tmpGammans = calloc(nthreads*4*_gamma_compshift, sizeof(double complex));
        double complex *tmpGammans_norm = calloc(nthreads*_gamma_compshift, sizeof(double complex));
        #pragma omp parallel for num_threads(nthreads)
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            int gamma_zshift = nbinsr*nbinsr;
            int gamma_nshift = _gamma_zshift*nbinsz*nbinsz*nbinsz;
            int gamma_compshift = (nmax-nmin+1)*_gamma_nshift;
            int ngalproc = 0;
            int npix_hash = pix1_n*pix2_n;
            int *rshift_index_matcher = calloc(nresos, sizeof(int));
            int *rshift_pixs_galind_bounds = calloc(nresos, sizeof(int));
            int *rshift_pix_gals = calloc(nresos, sizeof(int));
            build_rshift_offsets(nresos, npix_hash, ngal_resos,
                rshift_index_matcher, rshift_pixs_galind_bounds, rshift_pix_gals);
                
            for (int ind_gal=0; ind_gal<ngal; ind_gal++){
                // Check if galaxy falls in stripe used in this process
                double p11, p12, w1, e11, e12;
                int zbin1;
                double innergal;
                #pragma omp critical
                {p11 = pos1[ind_gal];
                p12 = pos2[ind_gal];
                w1 = weight[ind_gal];
                zbin1 = zbins[ind_gal];
                e11 = e1[ind_gal];
                e12 = e2[ind_gal];
                innergal = isinner[ind_gal];}
                if (innergal<1e-5){continue;}
                w1 *= innergal;
                int thisstripe = 2*thisthread + odd;
                int galstripe = (int) floor((p11-pix1_start)/pix1_d * (2*nthreads)/pix1_n);
                if (thisstripe != galstripe){continue;}
                
                if ((verbose>0) && (thisthread==nthreads/2)){
                    printf("\rDone %.2f per cent",50*odd+50*((double) 2*nthreads*ngalproc/ngal));
                    ngalproc += 1;
                }
                
                
                int ind_pix1, ind_pix2, ind_inpix, ind_gal2;
                int ind_red, lower, upper; 
                double  p21, p22, w2, z2, e21, e22;
                double rel1, rel2, dist;
                double complex wshape;
                int nnvals, nnvals_norm, nextn, nzero;
                double complex nphirot, twophirotc, nphirotc, phirot, phirotc, phirotm, phirotp, phirotn;
                
                // Check how many ns we need for Gn
                // Gns have shape (nnvals, nbinsz, nbinsr)
                // where the ns are ordered as 
                //  * [-nmax-3, ..., nmax-1] / [0, ..., nmax]
                if (nmin<4){nmin=0;}
                if (nmin==0){nnvals=2*nmax+3;nnvals_norm=nmax+1;}
                else{nnvals=2*(nmax-nmin+3);nnvals_norm=nmax-nmin+1;}
                double complex *nextGns =  calloc(nnvals*nbinsr*nbinsz, sizeof(double complex));
                double complex *nextGns_norm =  calloc(nnvals_norm*nbinsr*nbinsz, sizeof(double complex));
                double complex *nextG2ns =  calloc(4*nbinsz*nbinsr, sizeof(double complex));
                double complex *nextG2ns_norm =  calloc(nbinsz*nbinsr, sizeof(double complex));

                int ind_rbin, rbin;
                int ind_Gn, ind_Gnnorm, zrshift, nextnshift;
                int nbinszr = nbinsz*nbinsr;
                double drbin = (log(rmax)-log(rmin))/(nbinsr);
                /*if (ind_gal%10000==0){
                    printf("%d %d %d %d %d \n",nmin,nmax,nnvals,nbinsr,nbinsz);
                }*/
                
                for (int elreso=0;elreso<nresos;elreso++){
                    int pix1_lower = mymax(0, (int) floor((p11 - (reso_redges[elreso+1]+pix1_d) - pix1_start)/pix1_d));
                    int pix2_lower = mymax(0, (int) floor((p12 - (reso_redges[elreso+1]+pix2_d) - pix2_start)/pix2_d));
                    int pix1_upper = mymin(pix1_n-1, (int) floor((p11 + (reso_redges[elreso+1]+pix1_d) - pix1_start)/pix1_d));
                    int pix2_upper = mymin(pix2_n-1, (int) floor((p12 + (reso_redges[elreso+1]+pix2_d) - pix2_start)/pix2_d));

                    for (ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                        for (ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                            ind_red = index_matcher[rshift_index_matcher[elreso] + ind_pix2*pix1_n + ind_pix1];
                            if (ind_red==-1){continue;}
                            lower = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+ind_red];
                            upper = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+ind_red+1];
                            for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                                ind_gal2 = rshift_pix_gals[elreso] + pix_gals[rshift_pix_gals[elreso]+ind_inpix];
                                p21 = pos1_resos[ind_gal2];
                                p22 = pos2_resos[ind_gal2];
                                w2 = weight_resos[ind_gal2];
                                z2 = zbin_resos[ind_gal2];
                                e21 = e1_resos[ind_gal2];
                                e22 = e2_resos[ind_gal2];

                                rel1 = p21 - p11;
                                rel2 = p22 - p12;
                                dist = sqrt(rel1*rel1 + rel2*rel2);
                                if(dist < reso_redges[elreso] || dist >= reso_redges[elreso+1]) continue;
                                if (rbins[0] < 0){
                                    rbin = (int) floor((log(dist)-log(rmin))/drbin);
                                }
                                else{
                                    rbin=0;
                                    while(rbins[rbin+1] < dist){rbin+=1;}
                                }
                                wshape = (double complex) w2 * (e21+I*e22);
                                //phirot = csqrt((rel1+I*rel2)/(rel1-I*rel2));
                                //if (rel1<0){phirot*=-1;}
                                //phirotc = conj(phirot);
                                //twophirotc = phirotc*phirotc;
                                double dphi = atan2(rel2,rel1);
                                phirot = cexp(I*dphi);
                                phirotc = conj(phirot);
                                twophirotc = phirotc*phirotc;

                                zrshift = z2*nbinsr + rbin;
                                ind_rbin = thisthread*nbinszr + zrshift;
                                // nmin=0 -
                                //   -> Gns axis: [-nmax-3, ..., -nmin-1, nmin-3, nmax-1]
                                //   -> Gn_norm axis: [0,...,nmax]
                                if (nmin==0){
                                    nzero = nmax+3;
                                    ind_Gn = nzero*nbinszr + zrshift;
                                    ind_Gnnorm = zrshift;
                                    nphirot = 1+I*0;
                                    nphirotc = 1+I*0;

                                    // n = 0
                                    tmpwcounts[ind_rbin] += w1*w2*dist; 
                                    tmpwnorms[ind_rbin] += w1*w2; 
                                    nextGns[ind_Gn] += wshape*nphirot;
                                    nextGns_norm[ind_Gnnorm] += w2*nphirot;  
                                    nextG2ns[zrshift] += wshape*wshape*twophirotc*twophirotc*twophirotc;
                                    nextG2ns[nbinszr+zrshift] += wshape*wshape*twophirotc;
                                    nextG2ns[2*nbinszr+zrshift] += wshape*conj(wshape)*twophirotc;
                                    nextG2ns[3*nbinszr+zrshift] += wshape*conj(wshape)*twophirotc;
                                    nextG2ns_norm[zrshift] += w2*w2;
                                    nphirot *= phirot;
                                    nphirotc *= phirotc; 
                                    // n in [1, ..., nmax-1] x {+1,-1}
                                    nextnshift = 0;
                                    for (nextn=1;nextn<nmax;nextn++){
                                        nextnshift = nextn*nbinszr;
                                        nextGns[ind_Gn+nextnshift] += wshape*nphirot;
                                        nextGns[ind_Gn-nextnshift] += wshape*nphirotc;
                                        nextGns_norm[ind_Gnnorm+nextnshift] += w2*nphirot;  
                                        nphirot *= phirot;
                                        nphirotc *= phirotc; 
                                    }
                                    // n in [nmax, -nmax, -nmax-1, -nmax-2, -nmax-3]
                                    nextGns_norm[ind_Gnnorm+nextnshift+nbinszr] += w2*nphirot;  
                                    nextGns[zrshift+3*nbinszr] += wshape*nphirotc;
                                    nphirotc *= phirotc; 
                                    nextGns[zrshift+2*nbinszr] += wshape*nphirotc;
                                    nphirotc *= phirotc; 
                                    nextGns[zrshift+nbinszr] += wshape*nphirotc;
                                    nphirotc *= phirotc; 
                                    nextGns[zrshift] += wshape*nphirotc;
                                }

                                // nmin>3 
                                //   --> Gns axis: [-nmax-3, ..., -nmin+1, nmin-3, ..., nmax+1]
                                //   --> Gn_norm axis: [nmin, ..., nmax]
                                else{
                                    phirotm = cpow(phirotc,nmax+3);
                                    phirotp = cpow(phirot,nmin-3);
                                    phirotn = phirotp*phirot*phirot*phirot;
                                    int pshift = (nmax-nmin+3)*nbinszr;
                                    nextnshift = zrshift;
                                    // n in [-nmax-3, ..., -nmin-3] + [nmin-3, ..., nmax-3]
                                    for (nextn=0;nextn<nmax-nmin+1;nextn++){
                                        nextGns[nextnshift] += wshape*phirotm;
                                        nextGns[pshift+nextnshift] += wshape*phirotp;
                                        nextGns_norm[nextnshift] += w2*phirotn;
                                        phirotm *= phirot;
                                        phirotp *= phirot;
                                        phirotn *= phirot;
                                        nextnshift += nbinszr;
                                    }
                                    // n in [-nmin-2, -nmin-1] + [nmax-2, nmax-1]
                                    nextGns[nextnshift] += wshape*phirotm;
                                    nextGns[pshift+nextnshift] += wshape*phirotp;
                                    phirotm *= phirot;
                                    phirotp *= phirot;
                                    nextnshift += nbinszr;
                                    nextGns[nextnshift] += wshape*phirotm;
                                    nextGns[pshift+nextnshift] += wshape*phirotp;
                                } 
                            }
                        }
                    }
                }
                
                // Now update the Gammans
                // tmpGammas have shape (nthreads, nmax+1, nzcombis3, r*r, 4)
                // Gns have shape (nnvals, nbinsz, nbinsr)
                //int nonzero_tmpGammas = 0;
                double complex h0, h1, h2, h3, w0;
                int thisnshift, r12shift;
                int gammashift1, gammashiftt1, gammashift, gammashiftt;
                int ind_mnm3, ind_mnm1, ind_nm3, ind_nm1, ind_norm;
                int elb2, zbin3, zcombi;
                wshape = w1 * (e11+I*e12);
                for (int thisn=0; thisn<nmax-nmin+1; thisn++){
                    if (nmin==0){
                        nzero = nmax+3;
                        ind_mnm3 = (nzero-thisn-3)*nbinszr;
                        ind_mnm1 = (nzero-thisn-1)*nbinszr;
                        ind_nm3 = (nzero+thisn-3)*nbinszr;
                        ind_nm1 = (nzero+thisn-1)*nbinszr;
                        ind_norm = thisn*nbinszr;
                    }
                    else{
                        ind_mnm3 = (nmax-nmin-thisn)*nbinszr;
                        ind_mnm1 = (nmax-nmin+2-thisn)*nbinszr;
                        ind_nm3 = (nmax-nmin+3+thisn)*nbinszr;
                        ind_nm1 = (nmax-nmin+5+thisn)*nbinszr;
                        ind_norm = thisn*nbinszr;
                    }
                    thisnshift = thisthread*gamma_compshift + thisn*gamma_nshift;
                    for (int zbin2=0; zbin2<nbinsz; zbin2++){
                        for (int elb1=0; elb1<nbinsr; elb1++){
                            zrshift = zbin2*nbinsr + elb1;
                            h0 = -wshape * nextGns[ind_nm3 + zrshift];
                            h1 = -conj(wshape) * nextGns[ind_nm1 + zrshift];
                            h2 = -wshape * conj(nextGns[ind_mnm1 + zrshift]);
                            h3 = -wshape * conj(nextGns[ind_nm1 + zrshift]);
                            w0 = w1 * conj(nextGns_norm[ind_norm + zrshift]);
                            for (zbin3=0; zbin3<nbinsz; zbin3++){
                                zcombi = zbin1*nbinsz*nbinsz+zbin2*nbinsz+zbin3;
                                gammashift1 = thisnshift + zcombi*gamma_zshift + elb1*nbinsr;
                                gammashiftt1 = thisnshift + zcombi*gamma_zshift;
                                // Double counting correction
                                if (zbin1==zbin2 && zbin1==zbin3 && dccorr==1){
                                    r12shift = elb1*nbinsr+elb1;
                                    gammashift = 4*(gammashift1 + elb1);
                                    gammashiftt = gammashiftt1 + r12shift;
                                    tmpGammans[gammashift] += wshape*nextG2ns[zrshift];
                                    tmpGammans[gammashift+1] += conj(wshape)*nextG2ns[nbinszr+zrshift];
                                    tmpGammans[gammashift+2] += wshape*nextG2ns[2*nbinszr+zrshift];
                                    tmpGammans[4*gammashiftt+3] += wshape*nextG2ns[3*nbinszr+zrshift];
                                    tmpGammans_norm[gammashiftt] -= w1*nextG2ns_norm[zrshift];
                                }
                                // Nominal allocation
                                for (elb2=0; elb2<nbinsr; elb2++){
                                    zrshift = zbin3*nbinsr + elb2;
                                    r12shift = elb2*nbinsr+elb1;
                                    gammashift = 4*(gammashift1 + elb2);
                                    gammashiftt = gammashiftt1 + r12shift;
                                    //phirotm = h0*nextGns[ind_mnm3 + zrshift];
                                    tmpGammans[gammashift] += h0*nextGns[ind_mnm3 + zrshift];
                                    tmpGammans[gammashift+1] += h1*nextGns[ind_mnm1 + zrshift];
                                    tmpGammans[gammashift+2] += h2*nextGns[ind_mnm3 + zrshift];
                                    tmpGammans[4*gammashiftt+3] += h3*nextGns[ind_nm3 + zrshift];
                                    tmpGammans_norm[gammashiftt] += w0*nextGns_norm[ind_norm + zrshift];
                                    //if(thisthread==0 && ind_gal%1000==0){
                                    //    if (cabs(tmpGammans[gammashift] )>1e-5){nonzero_tmpGammas += 1;}
                                    //}
                                }
                            }
                        }
                    }
                }
                
                free(nextGns);
                free(nextGns_norm);
                free(nextG2ns);
                free(nextG2ns_norm);
                nextGns = NULL;
                nextGns_norm = NULL;
                nextG2ns = NULL;
                nextG2ns_norm = NULL;
            }
            
            free(rshift_index_matcher);
            free(rshift_pixs_galind_bounds);
            free(rshift_pix_gals);
        }
        
        // Accumulate the Gamman
        #pragma omp parallel for num_threads(nthreads)
        for (int thisn=0; thisn<nmax-nmin+1; thisn++){
            int itmpGamma, iGamma;
            for (int thisthread=0; thisthread<nthreads; thisthread++){
                for (int zcombi=0; zcombi<nbinsz*nbinsz*nbinsz; zcombi++){
                    for (int elb1=0; elb1<nbinsr; elb1++){
                        for (int elb2=0; elb2<nbinsr; elb2++){
                            iGamma = thisn*_gamma_nshift + zcombi*_gamma_zshift + elb1*nbinsr + elb2;
                            itmpGamma = iGamma + thisthread*_gamma_compshift;
                            for (int elcomp=0; elcomp<4; elcomp++){
                                Gammans[elcomp*_gamma_compshift+iGamma] += tmpGammans[4*itmpGamma+elcomp];
                            }
                            Gammans_norm[iGamma] += tmpGammans_norm[itmpGamma];
                        }
                    }
                }
            }
        }
        
        // Update the bin distances and weights
        for (int elbinz=0; elbinz<nbinsz; elbinz++){
            for (int elbinr=0; elbinr<nbinsr; elbinr++){
                int tmpind = elbinz*nbinsr + elbinr;
                for (int thisthread=0; thisthread<nthreads; thisthread++){
                    int tshift = thisthread*nbinsz*nbinsr; 
                    totcounts[tmpind] += tmpwcounts[tshift+tmpind];
                    totnorms[tmpind] += tmpwnorms[tshift+tmpind];
                }
            }
        }
        free(tmpwcounts);
        free(tmpwnorms);
        free(tmpGammans);
        free(tmpGammans_norm); 
        tmpwcounts = NULL;
        tmpwnorms = NULL;
        tmpGammans = NULL;
        tmpGammans_norm = NULL;
    }
    
    // Get bin centers
    for (int elbinz=0; elbinz<nbinsz; elbinz++){
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            int tmpind = elbinz*nbinsr + elbinr;
            if (totnorms[tmpind] != 0){
                bin_centers[tmpind] = totcounts[tmpind]/totnorms[tmpind];
            }
            
        }
    } 
    free(totcounts);
    free(totnorms);
    totcounts = NULL;
    totnorms = NULL;
}

// ---------------------------------------------------------------------------
// Shear 3PCF (GGG) DoubleTree -- struct interface + metric dispatch.
//
// The DoubleTree body splits into a small geometry-specific part (neighbour
// search + building the per-central Gn multipoles) and a large geometry-agnostic
// part (the Gn-cache update, the same-reso and cross-reso Upsilon_n allocation,
// and the thread reduction). Only the first part differs between the flat and
// curved-sky metrics, so _ggg_flat / _ggg_spherical share the rest through the
// helpers below (mirroring how _gg_flat/_gg_spherical share gg_reduce).
//
// Multipole storage. The four natural-component arms read the ABSOLUTE Gn slots
//   {nmax+n, nmax+n+2, nmax-n+2, nmax-n}
// (see ggg_accum_samereso / ggg_accum_crossreso). Flat fills these with the
// UNPROJECTED Gn using the n-3 index trick (nzero=nmax+3): slot s holds
// G_{s-nmax-3}. Curved-sky fills them with the PROJECTED Gn^P (partner shear
// projected onto its geodesic back-bearing) using n-1 (nzero=nmax+1): slot s
// holds G^P_{s-nmax-1}. Both make slot nmax+n hold the physically correct arm,
// so the consuming helpers are metric-agnostic; only the fill differs.
// ---------------------------------------------------------------------------

// Region-scoped context shared by the GGG DoubleTree helpers. Populated per
// thread (dimensions, caches, global accumulators) and per region (shift arrays).
typedef struct {
    int nbinsz, nbinsr, nmax, nresos;
    int nnvals_Gn, nnvals_Nn;                 // 2*nmax+3, nmax+1
    int gamma_zshift, gamma_nshift, gamma_compshift;
    int dccorr;
    int elthread;
    // region-scoped shift arrays
    int *reso_rindedges;                       // [nresos+1]
    int *ngal_in_pix;                          // [nresos*nbinsz], index z*nresos+reso
    int *cumresoshift_z;                       // [nbinsz*(nresos+1)]
    int *thetashifts_z;                        // [nbinsz]
    int *zbinshifts;                           // [nbinsz+1]
    int zbin2shift, nshift;
    // per-thread Gn caches (region-scoped, zeroed per region)
    double complex *Gncache, *wGncache, *cwGncache, *Nncache, *wNncache;
    // global per-thread Gamma accumulators
    double complex *tmpGamma0s, *tmpGamma1s, *tmpGamma2s, *tmpGamma3s, *tmpGammans_norm;
} GggCtx;

// Per-region cache-slot layout for the GGG caches (see setup_region_shifts).
static void ggg_setup_shifts(GggCtx *c, int hasdiscrete){
    setup_region_shifts(c->nbinsz, c->nbinsz, c->nresos, hasdiscrete, c->nbinsr,
        c->ngal_in_pix, c->cumresoshift_z, c->thetashifts_z, c->zbinshifts,
        &c->zbin2shift, &c->nshift);
}

static void ggg_zero_caches(GggCtx *c){
    for (int _i=0; _i<c->nnvals_Gn*c->nshift; _i++){c->Gncache[_i]=0; c->wGncache[_i]=0; c->cwGncache[_i]=0;}
    for (int _i=0; _i<c->nnvals_Nn*c->nshift; _i++){c->Nncache[_i]=0; c->wNncache[_i]=0;}
}

// Scatter a central's Gn (nextGns / nextGns_norm) into the region caches, keyed
// by the coarse super-galaxy (redpix_by_reso2[elreso2]) that contains the central
// at each resolution elreso2 >= elreso. Metric-agnostic: slots are copied
// verbatim, so whatever multipole convention filled nextGns is preserved.
static void ggg_update_gncache(GggCtx *c, int elreso, int rbinmin, int rbinmax,
    int nbinsr_reso, int z_gal1, double w_gal1, double complex wshape_gal1,
    const int *redpix_by_reso2,
    const double complex *nextGns, const double complex *nextGns_norm){
    int nbinszr_reso = c->nbinsz*nbinsr_reso;
    for (int elreso2=elreso; elreso2<c->nresos; elreso2++){
        int redpix_reso2 = redpix_by_reso2[elreso2];
        for (int zbin2=0; zbin2<c->nbinsz; zbin2++){
            for (int thisrbin=rbinmin; thisrbin<rbinmax; thisrbin++){
                int zrshift = zbin2*nbinsr_reso + thisrbin-rbinmin;
                if (cabs(nextGns_norm[zrshift])<1e-10){continue;}
                int ind_Gncacheshift = zbin2*c->zbin2shift + c->zbinshifts[z_gal1] +
                    thisrbin*c->thetashifts_z[z_gal1] +
                    c->cumresoshift_z[z_gal1*(c->nresos+1) + elreso2] + redpix_reso2;
                int _tmpindGn = zrshift;
                int _tmpindcache = ind_Gncacheshift;
                for(int thisn=0; thisn<c->nnvals_Gn; thisn++){
                    double complex thisGn = nextGns[_tmpindGn];
                    c->Gncache[_tmpindcache] += thisGn;
                    c->wGncache[_tmpindcache] += wshape_gal1*thisGn;
                    c->cwGncache[_tmpindcache] += conj(wshape_gal1)*thisGn;
                    _tmpindGn += nbinszr_reso;
                    _tmpindcache += c->nshift;
                }
                _tmpindGn = zrshift;
                _tmpindcache = ind_Gncacheshift;
                for(int thisn=0; thisn<c->nnvals_Nn; thisn++){
                    double complex thisGnnorm = nextGns_norm[_tmpindGn];
                    c->Nncache[_tmpindcache] += thisGnnorm;
                    c->wNncache[_tmpindcache] += w_gal1*thisGnnorm;
                    _tmpindGn += nbinszr_reso;
                    _tmpindcache += c->nshift;
                }
            }
        }
    }
}

// Same-resolution Upsilon_n allocation: both partners of the triangle come from
// the current central's nextGns (same band). nextG2ns carries the g2==g3
// self-terms removed when dccorr==1.
static void ggg_accum_samereso(GggCtx *c, int rbinmin, int nbinsr_reso,
    int z_gal1, double w_gal1, double complex wshape_gal1,
    const double complex *nextGns, const double complex *nextGns_norm,
    const double complex *nextG2ns, const double complex *nextG2ns_norm,
    const int *nextncounts, int *allowedrinds, int *allowedzinds){
    int nbinsz=c->nbinsz, nbinsr=c->nbinsr, nmax=c->nmax;
    int nbinszr_reso = nbinsz*nbinsr_reso;
    int nzero = nmax+3;
    int nallowedcounts = 0;
    for (int zbin1=0; zbin1<nbinsz; zbin1++){
        for (int elb1=0; elb1<nbinsr_reso; elb1++){
            if (nextncounts[zbin1*nbinsr_reso + elb1] != 0){
                allowedrinds[nallowedcounts] = elb1;
                allowedzinds[nallowedcounts] = zbin1;
                nallowedcounts += 1;
            }
        }
    }
    for (int thisn=0; thisn<nmax+1; thisn++){
        int ind_mnm3 = (nzero-thisn-3)*nbinszr_reso;
        int ind_mnm1 = (nzero-thisn-1)*nbinszr_reso;
        int ind_nm3 = (nzero+thisn-3)*nbinszr_reso;
        int ind_nm1 = (nzero+thisn-1)*nbinszr_reso;
        int ind_norm = thisn*nbinszr_reso;
        int thisnshift = c->elthread*c->gamma_compshift + thisn*c->gamma_nshift;
        for (int zrcombis1=0; zrcombis1<nallowedcounts; zrcombis1++){
            int elb1 = allowedrinds[zrcombis1];
            int zbin2 = allowedzinds[zrcombis1];
            int elb1_full = elb1 + rbinmin;
            int zrshift = zbin2*nbinsr_reso + elb1;
            if (c->dccorr==1){
                int zcombi = z_gal1*nbinsz*nbinsz + zbin2*nbinsz + zbin2;
                int gammashift1 = thisnshift + zcombi*c->gamma_zshift + elb1_full*nbinsr;
                int gammashift = gammashift1 + elb1_full;
                c->tmpGamma0s[gammashift] += wshape_gal1*nextG2ns[0*nbinszr_reso + zrshift];
                c->tmpGamma1s[gammashift] += conj(wshape_gal1)*nextG2ns[1*nbinszr_reso + zrshift];
                c->tmpGamma2s[gammashift] += wshape_gal1*nextG2ns[2*nbinszr_reso + zrshift];
                c->tmpGamma3s[gammashift] += wshape_gal1*nextG2ns[3*nbinszr_reso + zrshift];
                c->tmpGammans_norm[gammashift1 + elb1_full] -= w_gal1*nextG2ns_norm[zrshift];
            }
            double complex h0 = -wshape_gal1 * nextGns[ind_nm3 + zrshift];
            double complex h1 = -conj(wshape_gal1) * nextGns[ind_nm1 + zrshift];
            double complex h2 = -wshape_gal1 * conj(nextGns[ind_mnm1 + zrshift]);
            double complex h3 = -wshape_gal1 * nextGns[ind_nm3 + zrshift];
            double complex w0 = w_gal1 * nextGns_norm[ind_norm + zrshift];
            int _zcombi = z_gal1*nbinsz*nbinsz+zbin2*nbinsz;
            int _gammashift1 = thisnshift + elb1_full*nbinsr;
            for (int zrcombis2=0; zrcombis2<nallowedcounts; zrcombis2++){
                int zcombi = _zcombi+allowedzinds[zrcombis2];
                int gammashift1 = _gammashift1 + zcombi*c->gamma_zshift;
                int elb2_full = allowedrinds[zrcombis2] + rbinmin;
                int zrshift2 = allowedzinds[zrcombis2]*nbinsr_reso + allowedrinds[zrcombis2];
                int gammashift = gammashift1 + elb2_full;
                double complex Gmnm3 = nextGns[ind_mnm3 + zrshift2];
                c->tmpGamma0s[gammashift] += h0*Gmnm3;
                c->tmpGamma1s[gammashift] += h1*nextGns[ind_mnm1 + zrshift2];
                c->tmpGamma2s[gammashift] += h2*Gmnm3;
                c->tmpGamma3s[gammashift] += h3*conj(nextGns[ind_nm1 + zrshift2]);
                c->tmpGammans_norm[gammashift1 + elb2_full] += w0*conj(nextGns_norm[ind_norm + zrshift2]);
            }
        }
    }
}

// Cross-resolution Upsilon_n allocation from the region Gn caches: the two
// partners sit in different bands (reso1 != reso2). Metric-agnostic.
static void ggg_accum_crossreso(GggCtx *c){
    int nbinsz=c->nbinsz, nbinsr=c->nbinsr, nmax=c->nmax, nresos=c->nresos, nshift=c->nshift;
    for (int thisn=0; thisn<nmax+1; thisn++){
        int thisnshift = c->elthread*c->gamma_compshift + thisn*c->gamma_nshift;
        for (int zbin1=0; zbin1<nbinsz; zbin1++){
            for (int zbin2=0; zbin2<nbinsz; zbin2++){
                for (int zbin3=0; zbin3<nbinsz; zbin3++){
                    int zcombi = zbin1*nbinsz*nbinsz + zbin2*nbinsz + zbin3;
                    int _thetashift_z = c->thetashifts_z[zbin1];
                    // Case max(reso1, reso2) = reso2
                    for (int thisreso1=0; thisreso1<nresos; thisreso1++){
                        int rbinmin1 = c->reso_rindedges[thisreso1];
                        int rbinmax1 = c->reso_rindedges[thisreso1+1];
                        for (int thisreso2=thisreso1+1; thisreso2<nresos; thisreso2++){
                            int rbinmin2 = c->reso_rindedges[thisreso2];
                            int rbinmax2 = c->reso_rindedges[thisreso2+1];
                            for (int elgal=0; elgal<c->ngal_in_pix[zbin1*nresos+thisreso2]; elgal++){
                                for (int elb1=rbinmin1; elb1<rbinmax1; elb1++){
                                    int gammashift1 = thisnshift + zcombi*c->gamma_zshift + elb1*nbinsr;
                                    int ind_Nncacheshift = zbin2*c->zbin2shift + c->zbinshifts[zbin1] + elb1*c->thetashifts_z[zbin1] +
                                        c->cumresoshift_z[zbin1*(nresos+1) + thisreso2] + elgal;
                                    int ind_Gncacheshift = (nmax+3)*nshift + ind_Nncacheshift;
                                    double complex h0 = -c->wGncache[(thisn-3)*nshift + ind_Gncacheshift];
                                    double complex h1 = -c->cwGncache[(thisn-1)*nshift + ind_Gncacheshift];
                                    double complex h2 = -conj(c->cwGncache[(-thisn-1)*nshift + ind_Gncacheshift]);
                                    double complex h3 = -c->wGncache[(thisn-3)*nshift + ind_Gncacheshift];
                                    double complex w0 = c->wNncache[thisn*nshift + ind_Nncacheshift];
                                    ind_Nncacheshift = zbin3*c->zbin2shift + c->zbinshifts[zbin1] + rbinmin2*c->thetashifts_z[zbin1] +
                                            c->cumresoshift_z[zbin1*(nresos+1) + thisreso2] + elgal;
                                    ind_Gncacheshift = (nmax+3)*nshift + ind_Nncacheshift;
                                    int _imnm3 = (-thisn-3)*nshift + ind_Gncacheshift;
                                    int _imnm1 = (-thisn-1)*nshift + ind_Gncacheshift;
                                    int _inm1 = (thisn-1)*nshift + ind_Gncacheshift;
                                    int _in = thisn*nshift + ind_Nncacheshift;
                                    for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                        int gammashift = gammashift1 + elb2;
                                        c->tmpGamma0s[gammashift] += h0*c->Gncache[_imnm3];
                                        c->tmpGamma1s[gammashift] += h1*c->Gncache[_imnm1];
                                        c->tmpGamma2s[gammashift] += h2*c->Gncache[_imnm3];
                                        c->tmpGamma3s[gammashift] += h3*conj(c->Gncache[_inm1]);
                                        c->tmpGammans_norm[gammashift1 + elb2] += w0*conj(c->Nncache[_in]);
                                        ind_Nncacheshift += _thetashift_z; ind_Gncacheshift += _thetashift_z;
                                        _imnm3 += _thetashift_z; _imnm1 += _thetashift_z; _inm1 += _thetashift_z; _in += _thetashift_z;
                                    }
                                }
                            }
                        }
                    }
                    // Case max(reso1, reso2) = reso1
                    for (int thisreso2=0; thisreso2<nresos; thisreso2++){
                        int rbinmin2 = c->reso_rindedges[thisreso2];
                        int rbinmax2 = c->reso_rindedges[thisreso2+1];
                        for (int thisreso1=thisreso2+1; thisreso1<nresos; thisreso1++){
                            int rbinmin1 = c->reso_rindedges[thisreso1];
                            int rbinmax1 = c->reso_rindedges[thisreso1+1];
                            for (int elgal=0; elgal<c->ngal_in_pix[zbin1*nresos+thisreso1]; elgal++){
                                for (int elb1=rbinmin1; elb1<rbinmax1; elb1++){
                                    int gammashift1 = thisnshift + zcombi*c->gamma_zshift + elb1*nbinsr;
                                    int ind_Nncacheshift = zbin2*c->zbin2shift + c->zbinshifts[zbin1] + elb1*c->thetashifts_z[zbin1] +
                                        c->cumresoshift_z[zbin1*(nresos+1) + thisreso1] + elgal;
                                    int ind_Gncacheshift = (nmax+3)*nshift + ind_Nncacheshift;
                                    double complex h0 = -c->Gncache[(thisn-3)*nshift + ind_Gncacheshift];
                                    double complex h1 = -c->Gncache[(thisn-1)*nshift + ind_Gncacheshift];
                                    double complex h2 = -conj(c->Gncache[(-thisn-1)*nshift + ind_Gncacheshift]);
                                    double complex h3 = -c->Gncache[(thisn-3)*nshift + ind_Gncacheshift];
                                    double complex w0 = c->Nncache[thisn*nshift + ind_Nncacheshift];
                                    ind_Nncacheshift = zbin3*c->zbin2shift + c->zbinshifts[zbin1] + rbinmin2*c->thetashifts_z[zbin1] +
                                            c->cumresoshift_z[zbin1*(nresos+1) + thisreso1] + elgal;
                                    ind_Gncacheshift = (nmax+3)*nshift + ind_Nncacheshift;
                                    int _imnm3 = (-thisn-3)*nshift + ind_Gncacheshift;
                                    int _imnm1 = (-thisn-1)*nshift + ind_Gncacheshift;
                                    int _inm1 = (thisn-1)*nshift + ind_Gncacheshift;
                                    int _in = thisn*nshift + ind_Nncacheshift;
                                    for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                        int gammashift = gammashift1 + elb2;
                                        c->tmpGamma0s[gammashift] += h0*c->wGncache[_imnm3];
                                        c->tmpGamma1s[gammashift] += h1*c->cwGncache[_imnm1];
                                        c->tmpGamma2s[gammashift] += h2*c->wGncache[_imnm3];
                                        c->tmpGamma3s[gammashift] += h3*conj(c->cwGncache[_inm1]);
                                        c->tmpGammans_norm[gammashift1 + elb2] += w0*conj(c->wNncache[_in]);
                                        ind_Nncacheshift += _thetashift_z; ind_Gncacheshift += _thetashift_z;
                                        _imnm3 += _thetashift_z; _imnm1 += _thetashift_z; _inm1 += _thetashift_z; _in += _thetashift_z;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Reduce the per-thread Gamma accumulators into the unified NPCFOutput and fill the
// tomographic bin_centers. Shared by both metrics.
static void ggg_reduce(int nbinsz, int nbinsr, int nmax, int nthreads,
    const double complex *tmpGamma0s, const double complex *tmpGamma1s,
    const double complex *tmpGamma2s, const double complex *tmpGamma3s,
    const double complex *tmpGammans_norm,
    const double *tmpwcounts, const double *tmpwnorms, NPCFOutput *out){
    int _gamma_zshift = nbinsr*nbinsr;
    int _gamma_nshift = _gamma_zshift*nbinsz*nbinsz*nbinsz;
    int _gamma_compshift = (nmax+1)*_gamma_nshift;
    double *totcounts = calloc(nbinsz*nbinsr, sizeof(double));
    double *totnorms = calloc(nbinsz*nbinsr, sizeof(double));
    #pragma omp parallel for num_threads(nthreads)
    for (int thisn=0; thisn<nmax+1; thisn++){
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            for (int zcombi=0; zcombi<nbinsz*nbinsz*nbinsz; zcombi++){
                for (int elb1=0; elb1<nbinsr; elb1++){
                    for (int elb2=0; elb2<nbinsr; elb2++){
                        int iGamma = thisn*_gamma_nshift + zcombi*_gamma_zshift + elb1*nbinsr + elb2;
                        int itmpGamma = iGamma + thisthread*_gamma_compshift;
                        out->npcf[0*_gamma_compshift+iGamma] += tmpGamma0s[itmpGamma];
                        out->npcf[1*_gamma_compshift+iGamma] += tmpGamma1s[itmpGamma];
                        out->npcf[2*_gamma_compshift+iGamma] += tmpGamma2s[itmpGamma];
                        out->npcf[3*_gamma_compshift+iGamma] += tmpGamma3s[itmpGamma];
                        out->norm_mp[iGamma] += tmpGammans_norm[itmpGamma];
                    }
                }
            }
        }
    }
    for (int elbinz=0; elbinz<nbinsz; elbinz++){
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            int tmpind = elbinz*nbinsr + elbinr;
            for (int thisthread=0; thisthread<nthreads; thisthread++){
                int tshift = thisthread*nbinsz*nbinsr;
                totcounts[tmpind] += tmpwcounts[tshift+tmpind];
                totnorms[tmpind] += tmpwnorms[tshift+tmpind];
            }
        }
    }
    for (int elbinz=0; elbinz<nbinsz; elbinz++){
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            int tmpind = elbinz*nbinsr + elbinr;
            if (totnorms[tmpind] != 0){ out->bin_centers[tmpind] = totcounts[tmpind]/totnorms[tmpind]; }
        }
    }
    free(totcounts); free(totnorms);
}

// ---------------------------------------------------------------------------
// Flat-sky GGG DoubleTree. Pixel-box navigation + the n-3 index trick; the
// geometry-specific part is the neighbour loop that fills nextGns and the
// pix2redpix lookup for the cross-reso cache. Numerically identical to the
// retired positional alloc_Gammans_doubletree_ggg.
// ---------------------------------------------------------------------------
static void _ggg_flat(const MultiresoCatalog *cat, const NavHash *nav,
                      const TreeResoParams *tree, const BinningParams *bin,
                      int nthreads, int verbose, NPCFOutput *out){
    int nresos = tree->nresos, nresos_grid = tree->nresos_grid;
    double *dpix1_resos = tree->dpix1_resos, *dpix2_resos = tree->dpix2_resos;
    double *reso_redges = tree->reso_redges;
    int resoshift_leafs = tree->resoshift_leafs;
    int minresoind_leaf = tree->minresoind_leaf, maxresoind_leaf = tree->maxresoind_leaf;
    int *ngal_resos = cat->ngal_resos, nbinsz = cat->nbinsz;
    double *isinner_resos = cat->isinner_resos, *weight_resos = cat->weight_resos;
    double *pos1_resos = cat->pos1_resos, *pos2_resos = cat->pos2_resos;
    double *e1_resos = cat->e1_resos, *e2_resos = cat->e2_resos;
    int *zbin_resos = cat->zbin_resos;
    int *index_matcher = nav->index_matcher;
    int *pixs_galind_bounds = nav->pixs_galind_bounds, *pix_gals = nav->pix_gals;
    double pix1_start = nav->pix1_start, pix1_d = nav->pix1_d; int pix1_n = nav->pix1_n;
    double pix2_start = nav->pix2_start, pix2_d = nav->pix2_d; int pix2_n = nav->pix2_n;
    int *index_matcher_hash = nav->index_matcher_hash;
    int *filledregions = nav->filledregions, nfilledregions = nav->nfilledregions;
    int nmax = bin->nmax, nbinsr = bin->nbinsr, dccorr = bin->dccorr;
    double rmin = bin->rmin, rmax = bin->rmax;

    int _gamma_zshift = nbinsr*nbinsr;
    int _gamma_nshift = _gamma_zshift*nbinsz*nbinsz*nbinsz;
    int _gamma_compshift = (nmax+1)*_gamma_nshift;

    int nregionsdone = 0;
    double *tmpwcounts = calloc(nthreads*nbinsz*nbinsr, sizeof(double));
    double *tmpwnorms = calloc(nthreads*nbinsz*nbinsr, sizeof(double));
    double complex *tmpGamma0s = calloc(nthreads*_gamma_compshift, sizeof(double complex));
    double complex *tmpGamma1s = calloc(nthreads*_gamma_compshift, sizeof(double complex));
    double complex *tmpGamma2s = calloc(nthreads*_gamma_compshift, sizeof(double complex));
    double complex *tmpGamma3s = calloc(nthreads*_gamma_compshift, sizeof(double complex));
    double complex *tmpGammans_norm = calloc(nthreads*_gamma_compshift, sizeof(double complex));

    #pragma omp parallel num_threads(nthreads)
    {
        int elthread = omp_get_thread_num();
        int hasdiscrete = nresos-nresos_grid;
        int nnvals_Gn = 2*nmax+3;
        int nnvals_Nn = nmax+1;
        int size_max_nshift = (int) ((1+hasdiscrete+0.34)*nbinsz*nbinsz*nbinsr*pow(4,nresos_grid-1));
        double complex *Gncache = calloc(nnvals_Gn*size_max_nshift, sizeof(double complex));
        double complex *wGncache = calloc(nnvals_Gn*size_max_nshift, sizeof(double complex));
        double complex *cwGncache = calloc(nnvals_Gn*size_max_nshift, sizeof(double complex));
        double complex *Nncache = calloc(nnvals_Nn*size_max_nshift, sizeof(double complex));
        double complex *wNncache = calloc(nnvals_Nn*size_max_nshift, sizeof(double complex));

        GggCtx ctx;
        ctx.nbinsz=nbinsz; ctx.nbinsr=nbinsr; ctx.nmax=nmax; ctx.nresos=nresos;
        ctx.nnvals_Gn=nnvals_Gn; ctx.nnvals_Nn=nnvals_Nn;
        ctx.gamma_zshift=_gamma_zshift; ctx.gamma_nshift=_gamma_nshift; ctx.gamma_compshift=_gamma_compshift;
        ctx.dccorr=dccorr; ctx.elthread=elthread;
        ctx.Gncache=Gncache; ctx.wGncache=wGncache; ctx.cwGncache=cwGncache;
        ctx.Nncache=Nncache; ctx.wNncache=wNncache;
        ctx.tmpGamma0s=tmpGamma0s; ctx.tmpGamma1s=tmpGamma1s; ctx.tmpGamma2s=tmpGamma2s;
        ctx.tmpGamma3s=tmpGamma3s; ctx.tmpGammans_norm=tmpGammans_norm;

        #pragma omp for schedule(dynamic, 8)
        for (int _elregion=0; _elregion<nfilledregions; _elregion++){
            int elregion = filledregions[_elregion];

            double logrmin = log(rmin);
            double drbin = (log(rmax)-logrmin)/(nbinsr);
            int *reso_rindedges = calloc(nresos+1, sizeof(int));
            build_reso_rindedges(nresos, reso_redges, rmin, rmax, nbinsr, reso_rindedges);
            ctx.reso_rindedges = reso_rindedges;

            int npix_hash = pix1_n*pix2_n;
            int *rshift_index_matcher = calloc(nresos, sizeof(int));
            int *rshift_pixs_galind_bounds = calloc(nresos, sizeof(int));
            int *rshift_pix_gals = calloc(nresos, sizeof(int));
            build_rshift_offsets(nresos, npix_hash, ngal_resos,
                rshift_index_matcher, rshift_pixs_galind_bounds, rshift_pix_gals);

            int *matchers_resoshift = calloc(nresos_grid+1, sizeof(int));
            int *ngal_in_pix = calloc(nresos*nbinsz, sizeof(int));
            int len_matcher = build_region_galinpix(nresos, nresos_grid, hasdiscrete,
                elregion, pixs_galind_bounds, rshift_pixs_galind_bounds,
                pix_gals, rshift_pix_gals, zbin_resos, matchers_resoshift, ngal_in_pix);
            ctx.ngal_in_pix = ngal_in_pix;

            double hashpix_start1, hashpix_start2;
            int *pix2redpix = calloc(nbinsz*len_matcher, sizeof(int));
            build_region_pix2redpix(nresos_grid, hasdiscrete, elregion, nbinsz,
                index_matcher_hash, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d,
                pixs_galind_bounds, rshift_pixs_galind_bounds, pix_gals, rshift_pix_gals,
                zbin_resos, pos1_resos, pos2_resos, dpix1_resos, dpix2_resos,
                matchers_resoshift, len_matcher, &hashpix_start1, &hashpix_start2, pix2redpix);

            int *cumresoshift_z = calloc(nbinsz*(nresos+1), sizeof(int));
            int *thetashifts_z = calloc(nbinsz, sizeof(int));
            int *zbinshifts = calloc(nbinsz+1, sizeof(int));
            ctx.cumresoshift_z = cumresoshift_z; ctx.thetashifts_z = thetashifts_z; ctx.zbinshifts = zbinshifts;
            ggg_setup_shifts(&ctx, hasdiscrete);
            ggg_zero_caches(&ctx);

            int *redpix_by_reso2 = calloc(nresos, sizeof(int));
            for (int elreso=0;elreso<nresos;elreso++){
                int rbinmin = reso_rindedges[elreso];
                int rbinmax = reso_rindedges[elreso+1];
                double rmin_reso = rmin*exp(rbinmin*drbin);
                double rmax_reso = rmin*exp(rbinmax*drbin);
                int nbinsr_reso = rbinmax-rbinmin;
                int nbinszr_reso = nbinsz*nbinsr_reso;
                int lower1 = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion];
                int upper1 = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion+1];
                double complex *nextGns = calloc(nnvals_Gn*nbinszr_reso, sizeof(double complex));
                double complex *nextGns_norm = calloc(nnvals_Nn*nbinszr_reso, sizeof(double complex));
                double complex *nextG2ns = calloc(4*nbinszr_reso, sizeof(double complex));
                double complex *nextG2ns_norm = calloc(nbinszr_reso, sizeof(double complex));
                int *nextncounts = calloc(nbinszr_reso, sizeof(int));
                int *allowedrinds = calloc(nbinszr_reso, sizeof(int));
                int *allowedzinds = calloc(nbinszr_reso, sizeof(int));
                int _leaf_lo, _leaf_hi;
                if (resoshift_leafs < 0) {
                    _leaf_lo = mymax(minresoind_leaf, elreso + resoshift_leafs);
                    _leaf_hi = mymin(elreso, maxresoind_leaf);
                    _leaf_lo = mymin(_leaf_lo, _leaf_hi);
                } else {
                    _leaf_lo = _leaf_hi = mymin(mymax(minresoind_leaf, elreso + resoshift_leafs), maxresoind_leaf);
                }
                double _dpix_ratio = dpix1_resos[nresos_grid-1] / dpix1_resos[nresos_grid-2];

                for (int ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                    int ind_gal1 = rshift_pix_gals[elreso] + pix_gals[rshift_pix_gals[elreso]+ind_inpix1];
                    double innergal = isinner_resos[ind_gal1];
                    if (innergal<1e-5){continue;}
                    int z_gal1 = zbin_resos[ind_gal1];
                    double pos1_gal1 = pos1_resos[ind_gal1];
                    double pos2_gal1 = pos2_resos[ind_gal1];
                    double w_gal1 = innergal*weight_resos[ind_gal1];
                    double e1_gal1 = e1_resos[ind_gal1];
                    double e2_gal1 = e2_resos[ind_gal1];
                    double complex wshape_gal1 = (double complex) w_gal1 * (e1_gal1+I*e2_gal1);
                    int nzero = nmax+3;

                    for (int elreso_leaf = _leaf_lo; elreso_leaf <= _leaf_hi; elreso_leaf++) {
                        double _rmin_sub2, _rmax_sub2;
                        double _rmin_sub = rmin_reso, _rmax_sub = rmax_reso;
                        if (resoshift_leafs < 0) {
                            int k = elreso_leaf - _leaf_lo;
                            _rmin_sub = rmin_reso * pow(_dpix_ratio, (double)k);
                            _rmax_sub = (elreso_leaf < _leaf_hi) ? rmin_reso * pow(_dpix_ratio, (double)(k+1)) : rmax_reso;
                            _rmin_sub = fmax(_rmin_sub, rmin_reso);
                            _rmax_sub = fmin(_rmax_sub, rmax_reso);
                            if (_rmin_sub >= _rmax_sub) continue;
                        }
                        _rmin_sub2=_rmin_sub*_rmin_sub;
                        _rmax_sub2=_rmax_sub*_rmax_sub;
                        int pix1_lower = mymax(0, (int) floor((pos1_gal1 - (_rmax_sub+pix1_d) - pix1_start)/pix1_d));
                        int pix2_lower = mymax(0, (int) floor((pos2_gal1 - (_rmax_sub+pix2_d) - pix2_start)/pix2_d));
                        int pix1_upper = mymin(pix1_n-1, (int) floor((pos1_gal1 + (_rmax_sub+pix1_d) - pix1_start)/pix1_d));
                        int pix2_upper = mymin(pix2_n-1, (int) floor((pos2_gal1 + (_rmax_sub+pix2_d) - pix2_start)/pix2_d));
                        for (int ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                            for (int ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                                int ind_red = index_matcher[rshift_index_matcher[elreso_leaf] + ind_pix2*pix1_n + ind_pix1];
                                if (ind_red==-1){continue;}
                                int lower2 = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso_leaf]+ind_red];
                                int upper2 = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso_leaf]+ind_red+1];
                                for (int ind_inpix2=lower2; ind_inpix2<upper2; ind_inpix2++){
                                    int ind_gal2 = rshift_pix_gals[elreso_leaf] + pix_gals[rshift_pix_gals[elreso_leaf]+ind_inpix2];
                                    double pos1_gal2 = pos1_resos[ind_gal2];
                                    double pos2_gal2 = pos2_resos[ind_gal2];
                                    double rel1 = pos1_gal2 - pos1_gal1;
                                    double rel2 = pos2_gal2 - pos2_gal1;
                                    double dist2 = rel1*rel1 + rel2*rel2;
                                    if (dist2 < _rmin_sub2 || dist2 >= _rmax_sub2) continue;
                                    double dist = sqrt(dist2);
                                    int rbin = (int) floor((log(dist)-logrmin)/drbin) - rbinmin;
                                    double w_gal2 = weight_resos[ind_gal2];
                                    int z_gal2 = zbin_resos[ind_gal2];
                                    double e1_gal2 = e1_resos[ind_gal2];
                                    double e2_gal2 = e2_resos[ind_gal2];
                                    double complex wshape_gal2 = (double complex) w_gal2 * (e1_gal2+I*e2_gal2);
                                    double complex phirot = (rel1+I*rel2)/dist;
                                    double complex phirotc = conj(phirot);
                                    double complex twophirotc = phirotc*phirotc;
                                    int zrshift = z_gal2*nbinsr_reso + rbin;
                                    int ind_rbin = elthread*nbinsz*nbinsr + z_gal2*nbinsr + rbin+rbinmin;
                                    int ind_Gn = nzero*nbinszr_reso + zrshift;
                                    int ind_Gnnorm = zrshift;
                                    double complex nphirot = 1+I*0;
                                    double complex nphirotc = 1+I*0;
                                    nextncounts[zrshift] += 1;
                                    tmpwcounts[ind_rbin] += w_gal1*w_gal2*dist;
                                    tmpwnorms[ind_rbin] += w_gal1*w_gal2;
                                    nextGns[ind_Gn] += wshape_gal2*nphirot;
                                    nextGns_norm[ind_Gnnorm] += w_gal2*nphirot;
                                    double complex _wwphi = wshape_gal2*wshape_gal2*twophirotc;
                                    double complex _wwphic = wshape_gal2*conj(wshape_gal2)*twophirotc;
                                    nextG2ns[0*nbinszr_reso+zrshift] += _wwphi*twophirotc*twophirotc;
                                    nextG2ns[1*nbinszr_reso+zrshift] += _wwphi;
                                    nextG2ns[2*nbinszr_reso+zrshift] += _wwphic;
                                    nextG2ns[3*nbinszr_reso+zrshift] += _wwphic;
                                    nextG2ns_norm[zrshift] += w_gal2*w_gal2;
                                    nphirot *= phirot;
                                    nphirotc *= phirotc;
                                    int nextnshift = 0;
                                    for (int nextn=1;nextn<nmax;nextn++){
                                        nextnshift = nextn*nbinszr_reso;
                                        nextGns[ind_Gn+nextnshift] += wshape_gal2*nphirot;
                                        nextGns[ind_Gn-nextnshift] += wshape_gal2*nphirotc;
                                        nextGns_norm[ind_Gnnorm+nextnshift] += w_gal2*nphirot;
                                        nphirot *= phirot;
                                        nphirotc *= phirotc;
                                    }
                                    nextGns_norm[ind_Gnnorm+nextnshift+nbinszr_reso] += w_gal2*nphirot;
                                    nextGns[zrshift+3*nbinszr_reso] += wshape_gal2*nphirotc;
                                    nphirotc *= phirotc;
                                    nextGns[zrshift+2*nbinszr_reso] += wshape_gal2*nphirotc;
                                    nphirotc *= phirotc;
                                    nextGns[zrshift+nbinszr_reso] += wshape_gal2*nphirotc;
                                    nphirotc *= phirotc;
                                    nextGns[zrshift] += wshape_gal2*nphirotc;
                                }
                            }
                        }
                    }

                    for (int elreso2=elreso; elreso2<nresos; elreso2++){
                        int red_reso2 = elreso2 - hasdiscrete;
                        if (hasdiscrete==1 && elreso==0 && elreso2==0){red_reso2 += hasdiscrete;}
                        int npix_side_reso2 = 1 << (nresos_grid-red_reso2-1);
                        int elhashpix_1_reso2 = (int) floor((pos1_gal1 - hashpix_start1)/dpix1_resos[red_reso2]);
                        int elhashpix_2_reso2 = (int) floor((pos2_gal1 - hashpix_start2)/dpix2_resos[red_reso2]);
                        int elhashpix_reso2 = elhashpix_2_reso2*npix_side_reso2 + elhashpix_1_reso2;
                        redpix_by_reso2[elreso2] = pix2redpix[z_gal1*len_matcher+matchers_resoshift[red_reso2]+elhashpix_reso2];
                    }
                    ggg_update_gncache(&ctx, elreso, rbinmin, rbinmax, nbinsr_reso, z_gal1, w_gal1, wshape_gal1,
                                       redpix_by_reso2, nextGns, nextGns_norm);
                    ggg_accum_samereso(&ctx, rbinmin, nbinsr_reso, z_gal1, w_gal1, wshape_gal1,
                                       nextGns, nextGns_norm, nextG2ns, nextG2ns_norm,
                                       nextncounts, allowedrinds, allowedzinds);

                    for (int _i=0;_i<nnvals_Gn*nbinszr_reso;_i++){nextGns[_i]=0;}
                    for (int _i=0;_i<nnvals_Nn*nbinszr_reso;_i++){nextGns_norm[_i]=0;}
                    for (int _i=0;_i<4*nbinszr_reso;_i++){nextG2ns[_i]=0;}
                    for (int _i=0;_i<nbinszr_reso;_i++){nextG2ns_norm[_i]=0; nextncounts[_i]=0; allowedrinds[_i]=0; allowedzinds[_i]=0;}
                }
                free(nextGns); free(nextGns_norm); free(nextG2ns); free(nextG2ns_norm);
                free(nextncounts); free(allowedrinds); free(allowedzinds);
            }

            ggg_accum_crossreso(&ctx);

            free(reso_rindedges);
            free(rshift_index_matcher); free(rshift_pixs_galind_bounds); free(rshift_pix_gals);
            free(matchers_resoshift); free(ngal_in_pix); free(pix2redpix);
            free(cumresoshift_z); free(thetashifts_z); free(zbinshifts);
            free(redpix_by_reso2);
            #pragma omp atomic
            nregionsdone += 1;
            print_progress(nregionsdone, nfilledregions, verbose);
        }
        free(Gncache); free(wGncache); free(cwGncache); free(Nncache); free(wNncache);
    }

    ggg_reduce(nbinsz, nbinsr, nmax, nthreads, tmpGamma0s, tmpGamma1s, tmpGamma2s, tmpGamma3s,
               tmpGammans_norm, tmpwcounts, tmpwnorms, out);
    if (verbose>0){printf("\n");}
    free(tmpwcounts); free(tmpwnorms);
    free(tmpGamma0s); free(tmpGamma1s); free(tmpGamma2s); free(tmpGamma3s); free(tmpGammans_norm);
}

// Number of trailing zero bits of a power-of-two nside == log2(nside).
static inline int ggg_nside_level(long nside){ return (nside>0) ? __builtin_ctzl((unsigned long)nside) : 0; }

// Lower bound: first index i in cp[0..n) with cp[i] >= key.
static inline int ggg_lower_bound_long(const long *cp, int n, long key){
    int lo=0, hi=n;
    while (lo<hi){ int m=(lo+hi)>>1; if (cp[m]<key){ lo=m+1; } else { hi=m; } }
    return lo;
}

// ---------------------------------------------------------------------------
// Curved-sky GGG DoubleTree. Regions are coarse nested-HEALPix cells (analogue
// of the flat filledregions); neighbours are found with query_disc; the Gn are
// projected onto their geodesic (partner shear -> Gn^P), and the cross-reso
// cache is indexed by nested-HEALPix coarsening (ang2pix at the coarser nside)
// -- the curved-sky analogue of pix2redpix. All non-navigation work reuses the
// shared helpers, so this shares the same same-reso / cross-reso / cache /
// reduce paths as _ggg_flat.
// ---------------------------------------------------------------------------
static void _ggg_spherical(const MultiresoCatalog *cat, const NavHash *nav,
                           const TreeResoParams *tree, const BinningParams *bin,
                           int nthreads, int verbose, NPCFOutput *out){
    int nresos = tree->nresos, nresos_grid = tree->nresos_grid;
    double *reso_redges = tree->reso_redges;
    int resoshift_leafs = tree->resoshift_leafs;
    int minresoind_leaf = tree->minresoind_leaf, maxresoind_leaf = tree->maxresoind_leaf;
    int nbinsz = cat->nbinsz;
    double *isinner = cat->isinner_resos, *weight = cat->weight_resos;
    double *vx = cat->vx_resos, *vy = cat->vy_resos, *vz = cat->vz_resos;
    double *sindec = cat->sindec_resos, *cosdec = cat->cosdec_resos;
    double *e1 = cat->e1_resos, *e2 = cat->e2_resos;
    int *zbin = cat->zbin_resos;
    int *ncells_resos = nav->ncells_resos;
    long *nside_nav = nav->nside_nav;
    long *cell_pix = nav->cell_pix;
    int *cell_redbounds = nav->cell_redbounds;
    int *rshift_red = nav->rshift_red, *rshift_cellpix = nav->rshift_cellpix, *rshift_cellbounds = nav->rshift_cellbounds;
    int nmax = bin->nmax, nbinsr = bin->nbinsr, dccorr = bin->dccorr;
    double rmin = bin->rmin, rmax = bin->rmax;
    int hasdiscrete = nresos - nresos_grid;

    int _gamma_zshift = nbinsr*nbinsr;
    int _gamma_nshift = _gamma_zshift*nbinsz*nbinsz*nbinsz;
    int _gamma_compshift = (nmax+1)*_gamma_nshift;

    double *tmpwcounts = calloc(nthreads*nbinsz*nbinsr, sizeof(double));
    double *tmpwnorms = calloc(nthreads*nbinsz*nbinsr, sizeof(double));
    double complex *tmpGamma0s = calloc(nthreads*_gamma_compshift, sizeof(double complex));
    double complex *tmpGamma1s = calloc(nthreads*_gamma_compshift, sizeof(double complex));
    double complex *tmpGamma2s = calloc(nthreads*_gamma_compshift, sizeof(double complex));
    double complex *tmpGamma3s = calloc(nthreads*_gamma_compshift, sizeof(double complex));
    double complex *tmpGammans_norm = calloc(nthreads*_gamma_compshift, sizeof(double complex));

    // Per-reso nested levels; regions = cells of the coarsest band (smallest nside).
    int *level = calloc(nresos, sizeof(int));
    int r_region = 0;
    for (int r=0;r<nresos;r++){ level[r] = ggg_nside_level(nside_nav[r]); if (level[r] < level[r_region]) r_region = r; }
    int l_region = level[r_region];
    int nregions = ncells_resos[r_region];
    const long *region_cellpix = cell_pix + rshift_cellpix[r_region];

    int nregionsdone = 0;
    double logrmin = log(rmin);
    double drbin = (log(rmax)-logrmin)/(nbinsr);

    #pragma omp parallel num_threads(nthreads)
    {
        int elthread = omp_get_thread_num();
        int nnvals_Gn = 2*nmax+3;
        int nnvals_Nn = nmax+1;

        // Gn caches grown on demand to the region's nshift.
        long cache_cap = 0;
        double complex *Gncache=NULL, *wGncache=NULL, *cwGncache=NULL, *Nncache=NULL, *wNncache=NULL;

        int *reso_rindedges = calloc(nresos+1, sizeof(int));
        build_reso_rindedges(nresos, reso_redges, rmin, rmax, nbinsr, reso_rindedges);

        int *ngal_in_pix = calloc(nresos*nbinsz, sizeof(int));
        int *cumresoshift_z = calloc(nbinsz*(nresos+1), sizeof(int));
        int *thetashifts_z = calloc(nbinsz, sizeof(int));
        int *zbinshifts = calloc(nbinsz+1, sizeof(int));
        // Per-reso region slice bounds (into that reso's cell list) and reduced-galaxy range.
        int *slice_clo = calloc(nresos, sizeof(int));
        int *slice_chi = calloc(nresos, sizeof(int));
        // Per-reso, per-cell-in-slice dense super-galaxy index per zbin (cellzidx[r]).
        int **cellzidx = calloc(nresos, sizeof(int*));

        GggCtx ctx;
        ctx.nbinsz=nbinsz; ctx.nbinsr=nbinsr; ctx.nmax=nmax; ctx.nresos=nresos;
        ctx.nnvals_Gn=nnvals_Gn; ctx.nnvals_Nn=nnvals_Nn;
        ctx.gamma_zshift=_gamma_zshift; ctx.gamma_nshift=_gamma_nshift; ctx.gamma_compshift=_gamma_compshift;
        ctx.dccorr=dccorr; ctx.elthread=elthread;
        ctx.tmpGamma0s=tmpGamma0s; ctx.tmpGamma1s=tmpGamma1s; ctx.tmpGamma2s=tmpGamma2s;
        ctx.tmpGamma3s=tmpGamma3s; ctx.tmpGammans_norm=tmpGammans_norm;
        ctx.reso_rindedges = reso_rindedges;
        ctx.ngal_in_pix = ngal_in_pix;
        ctx.cumresoshift_z = cumresoshift_z; ctx.thetashifts_z = thetashifts_z; ctx.zbinshifts = zbinshifts;

        long qcap = 2048;
        long *ranges = malloc(2*qcap*sizeof(long));
        int *redpix_by_reso2 = calloc(nresos, sizeof(int));

        #pragma omp for schedule(dynamic, 8)
        for (int elregion=0; elregion<nregions; elregion++){
            long region_id = region_cellpix[elregion];

            // Per-reso region slice + reduced-galaxy enumeration -> ngal_in_pix + cellzidx.
            for (int _i=0;_i<nresos*nbinsz;_i++){ ngal_in_pix[_i]=0; }
            for (int _i=0;_i<nbinsz*(nresos+1);_i++){ cumresoshift_z[_i]=0; }
            for (int _i=0;_i<nbinsz;_i++){ thetashifts_z[_i]=0; }
            for (int _i=0;_i<=nbinsz;_i++){ zbinshifts[_i]=0; }
            int has_inner = 0;
            for (int r=0;r<nresos;r++){
                const long *cp = cell_pix + rshift_cellpix[r];
                const int *cb = cell_redbounds + rshift_cellbounds[r];
                int nc = ncells_resos[r];
                int shift = 2*(level[r] - l_region);
                long lo_id = region_id << shift;
                long hi_id = (region_id+1) << shift;
                int clo = ggg_lower_bound_long(cp, nc, lo_id);
                int chi = ggg_lower_bound_long(cp, nc, hi_id);
                slice_clo[r] = clo; slice_chi[r] = chi;
                int ncslice = chi - clo;
                cellzidx[r] = calloc((ncslice>0?ncslice:1)*nbinsz, sizeof(int));
                int *running = calloc(nbinsz, sizeof(int));
                for (int cc=0; cc<ncslice; cc++){
                    int c = clo + cc;
                    for (int j=cb[c]; j<cb[c+1]; j++){
                        long g = rshift_red[r] + j;
                        int z = zbin[g];
                        cellzidx[r][cc*nbinsz + z] = running[z];
                        running[z] += 1;
                        ngal_in_pix[z*nresos + r] += 1;
                        if (isinner[g] >= 1e-5) has_inner = 1;
                    }
                }
                free(running);
            }
            if (!has_inner){
                for (int r=0;r<nresos;r++){ free(cellzidx[r]); }
                continue;
            }

            ggg_setup_shifts(&ctx, hasdiscrete);
            long need = (long)nnvals_Gn * ctx.nshift;
            if (need > cache_cap){
                cache_cap = need;
                Gncache = realloc(Gncache, cache_cap*sizeof(double complex));
                wGncache = realloc(wGncache, cache_cap*sizeof(double complex));
                cwGncache = realloc(cwGncache, cache_cap*sizeof(double complex));
                Nncache = realloc(Nncache, cache_cap*sizeof(double complex));
                wNncache = realloc(wNncache, cache_cap*sizeof(double complex));
            }
            ctx.Gncache=Gncache; ctx.wGncache=wGncache; ctx.cwGncache=cwGncache;
            ctx.Nncache=Nncache; ctx.wNncache=wNncache;
            ggg_zero_caches(&ctx);

            for (int elreso=0; elreso<nresos; elreso++){
                int rbinmin = reso_rindedges[elreso];
                int rbinmax = reso_rindedges[elreso+1];
                if (rbinmax <= rbinmin){ continue; }
                double rmin_reso = rmin*exp(rbinmin*drbin);
                double rmax_reso = rmin*exp(rbinmax*drbin);
                int nbinsr_reso = rbinmax-rbinmin;
                int nbinszr_reso = nbinsz*nbinsr_reso;
                int nzero = nmax+1;
                int elreso_leaf = mymin(mymax(minresoind_leaf, elreso+resoshift_leafs), maxresoind_leaf);
                long ns_leaf = nside_nav[elreso_leaf];
                long redleaf_off = rshift_red[elreso_leaf];
                const long *cellpix_leaf = cell_pix + rshift_cellpix[elreso_leaf];
                const int *bounds_leaf = cell_redbounds + rshift_cellbounds[elreso_leaf];
                int ncells_leaf = ncells_resos[elreso_leaf];

                double complex *nextGns = calloc(nnvals_Gn*nbinszr_reso, sizeof(double complex));
                double complex *nextGns_norm = calloc(nnvals_Nn*nbinszr_reso, sizeof(double complex));
                double complex *nextG2ns = calloc(4*nbinszr_reso, sizeof(double complex));
                double complex *nextG2ns_norm = calloc(nbinszr_reso, sizeof(double complex));
                int *nextncounts = calloc(nbinszr_reso, sizeof(int));
                int *allowedrinds = calloc(nbinszr_reso, sizeof(int));
                int *allowedzinds = calloc(nbinszr_reso, sizeof(int));

                // Central galaxies of this band in the region = its cell slice.
                const int *cb1 = cell_redbounds + rshift_cellbounds[elreso];
                const long *cp1 = cell_pix + rshift_cellpix[elreso];
                for (int cc=slice_clo[elreso]; cc<slice_chi[elreso]; cc++){
                    long cell_id1 = cp1[cc];
                    for (int j1=cb1[cc]; j1<cb1[cc+1]; j1++){
                        long g1 = rshift_red[elreso] + j1;
                        double innergal = isinner[g1];
                        if (innergal<1e-5){continue;}
                        int z_gal1 = zbin[g1];
                        double cx = vx[g1], cy = vy[g1], cz = vz[g1];
                        double cd1 = cosdec[g1], sd1 = sindec[g1];
                        double w_gal1 = innergal*weight[g1];
                        double complex wshape_gal1 = (double complex) w_gal1 * (e1[g1]+I*e2[g1]);

                        // Neighbours via query_disc at the leaf reso (merge-join vs cell_pix).
                        double v1[3] = {cx, cy, cz};
                        long nr = hpx_query_disc_nest_ranges(ns_leaf, v1, rmax_reso, ranges, qcap);
                        if (nr > qcap){ qcap = nr; ranges = realloc(ranges, 2*qcap*sizeof(long));
                                        nr = hpx_query_disc_nest_ranges(ns_leaf, v1, rmax_reso, ranges, qcap); }
                        int ci = 0;
                        for (long rr=0; rr<nr; rr++){
                            long plo = ranges[2*rr], phi = ranges[2*rr+1];
                            int loi = ci, hii = ncells_leaf;
                            while (loi < hii){ int m=(loi+hii)>>1; if (cellpix_leaf[m] < plo){ loi=m+1; } else { hii=m; } }
                            ci = loi;
                            while (ci < ncells_leaf && cellpix_leaf[ci] < phi){
                                int lo = bounds_leaf[ci], hi = bounds_leaf[ci+1];
                                for (int j=lo; j<hi; j++){
                                    long g2 = redleaf_off + j;
                                    double dist = sphere_dist(cx, cy, cz, vx[g2], vy[g2], vz[g2]);
                                    if (dist < rmin_reso || dist >= rmax_reso){ continue; }
                                    int rbin = (int) floor((log(dist)-logrmin)/drbin) - rbinmin;
                                    double vx2 = vx[g2], vy2 = vy[g2];
                                    double sd2 = sindec[g2], cd2 = cosdec[g2];
                                    double w_gal2 = weight[g2];
                                    int z_gal2 = zbin[g2];
                                    // Efficient form of tangent-frame bearing on the sphere
                                    // * Define spherical coords: r(ra,dec) = (cos(dec)cos(ra), cos(dec)sin(ra),sin(dec)) = (x,y,z)
                                    // * Tangent basis is obtained by differentiation
                                    //   East: ∂r/∂ra = cos(dec) (-sin(ra),cos(ra),0) --> e_E = (∂r/∂ra)/|∂r/∂ra| = (-sin(ra),cos(ra),0)
                                    //   North: ∂r/∂dec = (-sin(dec)cos(ra),-sin(dec)sin(ra),cos(dec)) --> e_N = ∂r/∂dec
                                    // * Now define center galaxy, c = (cx, cy, z1) and second galaxy, g2 = (vx, vy, z2)
                                    // * Tangent direction from c to g2 is obtained by projecting g2 onto e_E:
                                    //   E = e_E·g2 = (-y1/cd1)*vx + (x1/cd1)*vy = (x1 vy - y1 vx)/cd1 == E12/cd1
                                    // * Similarly, projecting g2 onto e_N gives
                                    //   N = e_N·g2 = -(z1/cd1)*(x1*vx + y1*vy) + cd1*z2 = [cd1^2*z2 - z1*(x1*vx+y1*vy)]/cd1 == N12/cd1
                                    // * Now define the xy-plane scalar product P = x1*vx + y1*vy and apply it to N12
                                    //   --> N12 = cd1^2*sd2 - sd1*P 
                                    // * This finally gives an efficient expression for the complex exponential of the bearing angle beta=-atan2(N,E):
                                    //   such that  e^{-2i*phi} = ((E12 + i N12)/sqrt(E12² + N12²) )^2 = (E12*E12 - N21*N21 + 2.*I*E12*N21)/(E12*E12 + N21*N21);
                                    //
                                    // Side note: This should be equivalent to what treecorr is doing, but here we explicitly normalise and choose bearing at g2 instead of c
                                    double P = cx*vx2 + cy*vy2;
                                    double E12 = cx*vy2 - cy*vx2;
                                    double N12 = cd1*cd1*sd2 - sd1*P;
                                    double N21 = cd2*cd2*sd1 - sd2*P;
                                    double hyp12 = sqrt(E12*E12 + N12*N12);
                                    double complex phirot = (E12 + I*N12)/hyp12;
                                    double complex rc2 = (E12*E12 - N21*N21 + 2.*I*E12*N21)/(E12*E12 + N21*N21);
                                    double complex wshape_gal2 = ((double complex) w_gal2 * (e1[g2]+I*e2[g2])) * rc2;
                                    double complex phirotc = conj(phirot);
                                    double complex twophirotc = phirotc*phirotc;
                                    int zrshift = z_gal2*nbinsr_reso + rbin;
                                    int ind_rbin = elthread*nbinsz*nbinsr + z_gal2*nbinsr + rbin+rbinmin;
                                    int ind_Gn = nzero*nbinszr_reso + zrshift;

                                    nextncounts[zrshift] += 1;
                                    tmpwcounts[ind_rbin] += w_gal1*w_gal2*dist;
                                    tmpwnorms[ind_rbin] += w_gal1*w_gal2;
                                    // Projected Gn^P, symmetric layout m in [-(nmax+1), nmax+1].
                                    nextGns[ind_Gn] += wshape_gal2;
                                    nextGns_norm[zrshift] += w_gal2;
                                    nextG2ns[0*nbinszr_reso+zrshift] += wshape_gal2*wshape_gal2*twophirotc;
                                    nextG2ns[1*nbinszr_reso+zrshift] += wshape_gal2*wshape_gal2*conj(twophirotc);
                                    nextG2ns[2*nbinszr_reso+zrshift] += wshape_gal2*conj(wshape_gal2)*twophirotc;
                                    nextG2ns[3*nbinszr_reso+zrshift] += wshape_gal2*conj(wshape_gal2)*twophirotc;
                                    nextG2ns_norm[zrshift] += w_gal2*w_gal2;
                                    double complex nphirot = phirot;
                                    double complex nphirotc = phirotc;
                                    for (int m=1; m<=nmax+1; m++){
                                        nextGns[ind_Gn + m*nbinszr_reso] += wshape_gal2*nphirot;
                                        nextGns[ind_Gn - m*nbinszr_reso] += wshape_gal2*nphirotc;
                                        if (m <= nmax){ nextGns_norm[zrshift + m*nbinszr_reso] += w_gal2*nphirot; }
                                        nphirot *= phirot;
                                        nphirotc *= phirotc;
                                    }
                                }
                                ci++;
                            }
                        }

                        // Cross-reso cache index: map the central to its coarse super-galaxy
                        // cell at each reso2 (ang2pix), then its dense per-zbin slot.
                        for (int elreso2=elreso; elreso2<nresos; elreso2++){
                            int grid_reso = elreso2 - hasdiscrete;
                            if (hasdiscrete==1 && elreso==0 && elreso2==0){ grid_reso += hasdiscrete; }
                            int map_reso = grid_reso + hasdiscrete;
                            int redpix = 0;
                            if (map_reso >= nresos){ redpix_by_reso2[elreso2] = 0; continue; }  // degenerate single-band guard
                            if (map_reso == elreso){
                                // Same band: the central is its own super-galaxy.
                                redpix = cellzidx[elreso][(cc - slice_clo[elreso])*nbinsz + z_gal1];
                            } else {
                                long P2 = hpx_ang2pix_nest(nside_nav[map_reso], v1);
                                const long *cp2 = cell_pix + rshift_cellpix[map_reso];
                                int c2 = ggg_lower_bound_long(cp2, ncells_resos[map_reso], P2);
                                if (c2 >= slice_clo[map_reso] && c2 < slice_chi[map_reso] && cp2[c2] == P2){
                                    redpix = cellzidx[map_reso][(c2 - slice_clo[map_reso])*nbinsz + z_gal1];
                                }
                            }
                            redpix_by_reso2[elreso2] = redpix;
                        }
                        ggg_update_gncache(&ctx, elreso, rbinmin, rbinmax, nbinsr_reso, z_gal1, w_gal1, wshape_gal1,
                                           redpix_by_reso2, nextGns, nextGns_norm);
                        ggg_accum_samereso(&ctx, rbinmin, nbinsr_reso, z_gal1, w_gal1, wshape_gal1,
                                           nextGns, nextGns_norm, nextG2ns, nextG2ns_norm,
                                           nextncounts, allowedrinds, allowedzinds);

                        for (int _i=0;_i<nnvals_Gn*nbinszr_reso;_i++){nextGns[_i]=0;}
                        for (int _i=0;_i<nnvals_Nn*nbinszr_reso;_i++){nextGns_norm[_i]=0;}
                        for (int _i=0;_i<4*nbinszr_reso;_i++){nextG2ns[_i]=0;}
                        for (int _i=0;_i<nbinszr_reso;_i++){nextG2ns_norm[_i]=0; nextncounts[_i]=0; allowedrinds[_i]=0; allowedzinds[_i]=0;}
                    }
                }
                free(nextGns); free(nextGns_norm); free(nextG2ns); free(nextG2ns_norm);
                free(nextncounts); free(allowedrinds); free(allowedzinds);
            }

            ggg_accum_crossreso(&ctx);

            for (int r=0;r<nresos;r++){ free(cellzidx[r]); }
            #pragma omp atomic
            nregionsdone += 1;
            if (verbose>0 && (nregionsdone % (nregions/100 + 1) == 0)){
                #pragma omp critical
                { printf("."); fflush(stdout); }
            }
        }
        free(ranges); free(redpix_by_reso2);
        free(Gncache); free(wGncache); free(cwGncache); free(Nncache); free(wNncache);
        free(reso_rindedges); free(ngal_in_pix);
        free(cumresoshift_z); free(thetashifts_z); free(zbinshifts);
        free(slice_clo); free(slice_chi); free(cellzidx);
    }
    free(level);

    ggg_reduce(nbinsz, nbinsr, nmax, nthreads, tmpGamma0s, tmpGamma1s, tmpGamma2s, tmpGamma3s,
               tmpGammans_norm, tmpwcounts, tmpwnorms, out);
    if (verbose>0){printf("\n");}
    free(tmpwcounts); free(tmpwnorms);
    free(tmpGamma0s); free(tmpGamma1s); free(tmpGamma2s); free(tmpGamma3s); free(tmpGammans_norm);
}

// Public entry point: a thin metric dispatch, mirroring alloc_gg_doubletree.
void alloc_ggg_doubletree(const MultiresoCatalog *cat, const NavHash *nav,
                          const TreeResoParams *tree, const BinningParams *bin,
                          int nthreads, int verbose, NPCFOutput *out){
    switch (cat->metric) {
        case METRIC_SPHERICAL:
            _ggg_spherical(cat, nav, tree, bin, nthreads, verbose, out);
            break;
        case METRIC_FLAT:
        default:
            _ggg_flat(cat, nav, tree, bin, nthreads, verbose, out);
            break;
    }
}

// Exactly the same as doubletree, but here we bruteforce the calculation of the Gn
// --> Same speed as tree and accurate on the diagonals!
void alloc_Gammans_basetree_ggg(const MultiresoCatalog *cat, const NavHash *nav,
                                const TreeResoParams *tree, const BinningParams *bin,
                                int nthreads, int verbose, NPCFOutput *out){
    // --- multi-resolution catalog (base = reso 0, WITH isinner_resos), like doubletree ---
    int nresos = tree->nresos, nresos_grid = tree->nresos_grid;
    double *dpix1_resos = tree->dpix1_resos, *dpix2_resos = tree->dpix2_resos, *reso_redges = tree->reso_redges;
    int *ngal_resos = cat->ngal_resos, nbinsz = cat->nbinsz, *zbin_resos = cat->zbin_resos;
    double *isinner_resos = cat->isinner_resos, *weight_resos = cat->weight_resos;
    double *pos1_resos = cat->pos1_resos, *pos2_resos = cat->pos2_resos;
    double *e1_resos = cat->e1_resos, *e2_resos = cat->e2_resos, *weightsq_resos = cat->weightsq_resos;
    // --- navigation (flat hash + occupied-region list) ---
    int *index_matcher = nav->index_matcher, *pixs_galind_bounds = nav->pixs_galind_bounds, *pix_gals = nav->pix_gals;
    double pix1_start = nav->pix1_start, pix1_d = nav->pix1_d; int pix1_n = nav->pix1_n;
    double pix2_start = nav->pix2_start, pix2_d = nav->pix2_d; int pix2_n = nav->pix2_n;
    int *index_matcher_hash = nav->index_matcher_hash, nregions = nav->nregions;
    int *filledregions = nav->filledregions, nfilledregions = nav->nfilledregions;
    // --- binning + output ---
    int nmax = bin->nmax, nbinsr = bin->nbinsr, dccorr = bin->dccorr;
    double rmin = bin->rmin, rmax = bin->rmax;
    double *bin_centers = out->bin_centers;
    double complex *Gammans = out->npcf, *Gammans_norm = out->norm_mp;

    // Index shift for the Gamman
    int _gamma_zshift = nbinsr*nbinsr;
    int _gamma_nshift = _gamma_zshift*nbinsz*nbinsz*nbinsz;
    int _gamma_compshift = (nmax+1)*_gamma_nshift;
    
    double *totcounts = calloc(nbinsz*nbinsr, sizeof(double));
    double *totnorms = calloc(nbinsz*nbinsr, sizeof(double));
    
    // Temporary arrays that are allocated in parallel and later reduced
    double *tmpwcounts = calloc(nthreads*nbinsz*nbinsr, sizeof(double));
    double *tmpwnorms = calloc(nthreads*nbinsz*nbinsr, sizeof(double));
    double complex *tmpGammans = calloc(nthreads*4*_gamma_compshift, sizeof(double complex));
    double complex *tmpGammans_norm = calloc(nthreads*_gamma_compshift, sizeof(double complex));
    
    #pragma omp parallel for num_threads(nthreads)
    for(int elthread=0;elthread<nthreads;elthread++){
        int nregions_per_thread = nregions/nthreads;
        int hasdiscrete = nresos-nresos_grid;
        int nnvals_Gn = 2*nmax+3;
        int nnvals_Nn = nmax+1;
        
        // Largest possible nshift: each zbin does completely fill out the lowest reso grid.
        // The remaining grids then have 1/4 + 1/16 + ... --> 0.33.... times the data of the largest grid. 
        // Now allocate the caches
        int size_max_nshift = (int) ((1+hasdiscrete+0.34)*nbinsz*nbinsz*nbinsr*pow(4,nresos_grid-1));
        double complex *Gncache = calloc(nnvals_Gn*size_max_nshift, sizeof(double complex));
        double complex *wGncache = calloc(nnvals_Gn*size_max_nshift, sizeof(double complex));
        double complex *cwGncache = calloc(nnvals_Gn*size_max_nshift, sizeof(double complex));
        double complex *Nncache = calloc(nnvals_Nn*size_max_nshift, sizeof(double complex));
        double complex *wNncache = calloc(nnvals_Nn*size_max_nshift, sizeof(double complex));
        int *Nncache_updates = calloc(size_max_nshift, sizeof(int));
        
        for (int elregion=0; elregion<nregions; elregion++){
            int region_debug=99999;
            bool printregdbg = (verbose>0) && (elregion==region_debug);
            // Check if this thread is responsible for the region
            int nthread_target = mymin(elregion/nregions_per_thread, nthreads-1);
            if (nthread_target!=elthread){continue;}
            // printf("Region %d is in thread %d\n",elregion,elthread);
            if (printregdbg){printf("Region %d is in thread %d\n",elregion,elthread);}
            if ((verbose>0) && (elthread==nthreads/2)){
                printf("\rDone %.2f per cent",100*((double) elregion-nregions_per_thread*(int)(nthreads/2))/nregions_per_thread);
            }
            
            
            // Check which sets of radii are evaluated for each resolution
            int *reso_rindedges = calloc(nresos+1, sizeof(int));
            double logrmin = log(rmin);
            double drbin = (log(rmax)-logrmin)/(nbinsr);
            build_reso_rindedges(nresos, reso_redges, rmin, rmax, nbinsr, reso_rindedges);
                        
            // Shift variables for 3pcf quantities
            int gamma_zshift = nbinsr*nbinsr;
            int gamma_nshift = gamma_zshift*nbinsz*nbinsz*nbinsz;
            int gamma_compshift = (nmax+1)*gamma_nshift;
            
            // Shift variables for spatial hash
            int npix_hash = pix1_n*pix2_n;
            int *rshift_index_matcher = calloc(nresos, sizeof(int));
            int *rshift_pixs_galind_bounds = calloc(nresos, sizeof(int));
            int *rshift_pix_gals = calloc(nresos, sizeof(int));
            build_rshift_offsets(nresos, npix_hash, ngal_resos,
                rshift_index_matcher, rshift_pixs_galind_bounds, rshift_pix_gals);
            
            // Shift variables for the matching between the pixel grids
            int lower1, upper1, lower2, upper2;
            int *matchers_resoshift = calloc(nresos_grid+1, sizeof(int));
            int *ngal_in_pix = calloc(nresos*nbinsz, sizeof(int));
            int len_matcher = build_region_galinpix(nresos, nresos_grid, hasdiscrete,
                elregion, pixs_galind_bounds, rshift_pixs_galind_bounds,
                pix_gals, rshift_pix_gals, zbin_resos, matchers_resoshift, ngal_in_pix);

            // Build the matcher from pixels to reduced pixels in the region
            double hashpix_start1, hashpix_start2;
            int *pix2redpix = calloc(nbinsz*len_matcher, sizeof(int)); // For each z matches pixel in unreduced grid to index in reduced grid
            build_region_pix2redpix(nresos_grid, hasdiscrete, elregion, nbinsz,
                index_matcher_hash, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d,
                pixs_galind_bounds, rshift_pixs_galind_bounds, pix_gals, rshift_pix_gals,
                zbin_resos, pos1_resos, pos2_resos, dpix1_resos, dpix2_resos,
                matchers_resoshift, len_matcher, &hashpix_start1, &hashpix_start2, pix2redpix);

            // Setup all shift variables for the Gncache in the region
            // Gncache has structure
            // n --> zbin2 --> zbin1 --> radius
            //   --> [ [0]*ngal_zbin1_reso1 | [0]*ngal_zbin1_reso1/2 | ... | [0]*ngal_zbin1_reson ]
            int *cumresoshift_z = calloc(nbinsz*(nresos+1), sizeof(int));
            int *thetashifts_z = calloc(nbinsz, sizeof(int));
            int *zbinshifts = calloc(nbinsz+1, sizeof(int));
            int zbin2shift, nshift;
            setup_region_shifts(nbinsz, nbinsz, nresos, hasdiscrete, nbinsr, ngal_in_pix,
                cumresoshift_z, thetashifts_z, zbinshifts, &zbin2shift, &nshift);
            // Set all the cache indeces that are updated in this region to zero
            if (printregdbg){printf("zbin2shift=%d: nshift=%d: \n", zbin2shift,  nshift);}
            for (int _i=0; _i<nnvals_Gn*nshift; _i++){Gncache[_i] = 0; wGncache[_i] = 0; cwGncache[_i] = 0;}
            for (int _i=0; _i<nnvals_Nn*nshift; _i++){ Nncache[_i] = 0; wNncache[_i] = 0;}
            for (int _i=0; _i<nshift; _i++){ Nncache_updates[_i] = 0;}
            int Nncache_totupdates=0;
            
            // Now, for each resolution, loop over all the galaxies in the region and
            // allocate the Gn & Nn, as well as their caches  for the corresponding 
            // set of radii
            // For elreso in resos
            //.  for gal in reso 
            //.    allocate Gn for allowed radii
            //.    allocate the Gncaches
            //.    compute the Gamman for all combinations of the same resolution
            int ind_pix1, ind_pix2, ind_inpix1, ind_inpix2, ind_red, ind_gal1, ind_gal2, z_gal1, z_gal2;
            int ind_Gn, ind_Gnnorm, ind_Gncacheshift, ind_Nncacheshift;
            int rbin, nextn, nextnshift, nbinszr, nbinszr_reso, zrshift, ind_rbin;
            double innergal, pos1_gal1, pos2_gal1, pos1_gal2, pos2_gal2, w_gal1, w_gal2, e1_gal1, e2_gal1, e1_gal2, e2_gal2;
            double rel1, rel2, dist;
            double complex wshape_gal1, wshape_gal2;
            double complex _wwphic, _wwphi;
            double complex nphirot, twophirotc, nphirotc, phirot, phirotc;
            double rmin_reso, rmax_reso, rmin_reso2, rmax_reso2;
            int rbinmin, rbinmax, rbinmin1, rbinmax1, rbinmin2, rbinmax2;
            int nzero = nmax+3;
            nbinszr =  nbinsz*nbinsr;
            int elreso_leaf = 0;
            for (int elreso=0;elreso<nresos;elreso++){
                rbinmin = reso_rindedges[elreso];
                rbinmax = reso_rindedges[elreso+1];
                rmin_reso = rmin*exp(rbinmin*drbin);
                rmax_reso = rmin*exp(rbinmax*drbin);
                rmin_reso2 = rmin_reso*rmin_reso;
                rmax_reso2 = rmax_reso*rmax_reso;
                int nbinsr_reso = rbinmax-rbinmin;
                nbinszr_reso = nbinsz*nbinsr_reso;
                lower1 = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion];
                upper1 = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion+1];
                double complex *nextGns =  calloc(nnvals_Gn*nbinszr_reso, sizeof(double complex));
                double complex *nextGns_norm =  calloc(nnvals_Nn*nbinszr_reso, sizeof(double complex));
                double complex *nextG2ns =  calloc(4*nbinszr_reso, sizeof(double complex));
                double complex *nextG2ns_norm =  calloc(nbinszr_reso, sizeof(double complex));
                int *nextncounts = calloc(nbinszr_reso, sizeof(int));
                int *allowedrinds = calloc(nbinszr_reso, sizeof(int));
                int *allowedzinds = calloc(nbinszr_reso, sizeof(int));
                if (printregdbg){printf("rbinmin=%d, rbinmax%d\n",rbinmin,rbinmax);}
                for (ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                    ind_gal1 = rshift_pix_gals[elreso] + pix_gals[rshift_pix_gals[elreso]+ind_inpix1];
                    innergal = isinner_resos[ind_gal1];
                    if (innergal<1e-5){continue;}
                    z_gal1 = zbin_resos[ind_gal1];
                    pos1_gal1 = pos1_resos[ind_gal1];
                    pos2_gal1 = pos2_resos[ind_gal1];
                    w_gal1 = innergal*weight_resos[ind_gal1];
                    e1_gal1 = e1_resos[ind_gal1];
                    e2_gal1 = e2_resos[ind_gal1];
                    wshape_gal1 = (double complex) w_gal1 * (e1_gal1+I*e2_gal1);
                    
                    int pix1_lower = mymax(0, (int) floor((pos1_gal1 - (rmax_reso+pix1_d) - pix1_start)/pix1_d));
                    int pix2_lower = mymax(0, (int) floor((pos2_gal1 - (rmax_reso+pix2_d) - pix2_start)/pix2_d));
                    int pix1_upper = mymin(pix1_n-1, (int) floor((pos1_gal1 + (rmax_reso+pix1_d) - pix1_start)/pix1_d));
                    int pix2_upper = mymin(pix2_n-1, (int) floor((pos2_gal1 + (rmax_reso+pix2_d) - pix2_start)/pix2_d));
                    
                    for (ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                        for (ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                            ind_red = index_matcher[rshift_index_matcher[elreso_leaf] + ind_pix2*pix1_n + ind_pix1];
                            if (ind_red==-1){continue;}
                            lower2 = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso_leaf]+ind_red];
                            upper2 = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso_leaf]+ind_red+1];
                            for (ind_inpix2=lower2; ind_inpix2<upper2; ind_inpix2++){
                                ind_gal2 = rshift_pix_gals[elreso_leaf] + pix_gals[rshift_pix_gals[elreso_leaf]+ind_inpix2];
                                pos1_gal2 = pos1_resos[ind_gal2];
                                pos2_gal2 = pos2_resos[ind_gal2];
                                rel1 = pos1_gal2 - pos1_gal1;
                                rel2 = pos2_gal2 - pos2_gal1;
                                dist = rel1*rel1 + rel2*rel2;
                                if(dist < rmin_reso2 || dist >= rmax_reso2) continue;
                                w_gal2 = weight_resos[ind_gal2];
                                z_gal2 = zbin_resos[ind_gal2];
                                e1_gal2 = e1_resos[ind_gal2];
                                e2_gal2 = e2_resos[ind_gal2];
                                wshape_gal2 = (double complex) w_gal2 * (e1_gal2+I*e2_gal2);
                                
                                // This bit is super inefficient...
                                dist = sqrt(dist);
                                rbin = (int) floor((log(dist)-logrmin)/drbin) - rbinmin;
                                if (rbin<0 || rbin>=rbinmax){continue;}
                                //rbin = mymax(mymin(rbin, rbinmax-1), rbinmin)-rbinmin;
                                
                                phirot = (rel1+I*rel2)/dist;// * fabs(rel1)/rel1;
                                phirotc = conj(phirot);
                                twophirotc = phirotc*phirotc;
                                zrshift = z_gal2*nbinsr_reso + rbin;
                                ind_rbin = elthread*nbinszr + z_gal2*nbinsr + rbin+rbinmin;

                                // nmin=0 
                                //   -> Gns axis: [-nmax-3, ..., -nmin-1, nmin-3, nmax-1]
                                //   -> Gn_norm axis: [0,...,nmax]
                                ind_Gn = nzero*nbinszr_reso + zrshift;
                                ind_Gnnorm = zrshift;
                                nphirot = 1+I*0;
                                nphirotc = 1+I*0;

                                // n = 0
                                nextncounts[zrshift] += 1;
                                tmpwcounts[ind_rbin] += w_gal1*w_gal2*dist; 
                                tmpwnorms[ind_rbin] += w_gal1*w_gal2; 
                                nextGns[ind_Gn] += wshape_gal2*nphirot;
                                nextGns_norm[ind_Gnnorm] += w_gal2*nphirot;  
                                _wwphi = wshape_gal2*wshape_gal2*twophirotc;
                                _wwphic = wshape_gal2*conj(wshape_gal2)*twophirotc;
                                nextG2ns[0*nbinszr_reso+zrshift] += _wwphi*twophirotc*twophirotc;
                                nextG2ns[1*nbinszr_reso+zrshift] += _wwphi;
                                nextG2ns[2*nbinszr_reso+zrshift] += _wwphic;
                                nextG2ns[3*nbinszr_reso+zrshift] += _wwphic;
                                nextG2ns_norm[zrshift] += w_gal2*w_gal2;
                                nphirot *= phirot;
                                nphirotc *= phirotc; 

                                // n in [1, ..., nmax-1] x {+1,-1}
                                nextnshift = 0;
                                for (nextn=1;nextn<nmax;nextn++){
                                    nextnshift = nextn*nbinszr_reso;
                                    nextGns[ind_Gn+nextnshift] += wshape_gal2*nphirot;
                                    nextGns[ind_Gn-nextnshift] += wshape_gal2*nphirotc;
                                    nextGns_norm[ind_Gnnorm+nextnshift] += w_gal2*nphirot;  
                                    nphirot *= phirot;
                                    nphirotc *= phirotc; 
                                }
                                // n in [nmax, -nmax, -nmax-1, -nmax-2, -nmax-3]
                                nextGns_norm[ind_Gnnorm+nextnshift+nbinszr_reso] += w_gal2*nphirot;  
                                nextGns[zrshift+3*nbinszr_reso] += wshape_gal2*nphirotc;
                                nphirotc *= phirotc; 
                                nextGns[zrshift+2*nbinszr_reso] += wshape_gal2*nphirotc;
                                nphirotc *= phirotc; 
                                nextGns[zrshift+nbinszr_reso] += wshape_gal2*nphirotc;
                                nphirotc *= phirotc; 
                                nextGns[zrshift] += wshape_gal2*nphirotc;
                            }
                        }
                    }
                    // Update the Gncache and Gnnormcache
                    int red_reso2, npix_side_reso2, elhashpix_1_reso2, elhashpix_2_reso2, elhashpix_reso2, redpix_reso2;
                    double complex thisGn, thisGnnorm;
                    int _tmpindcache, _tmpindGn;
                    for (int elreso2=elreso; elreso2<nresos; elreso2++){
                        red_reso2 = elreso2 - hasdiscrete;
                        if (hasdiscrete==1 && elreso==0 && elreso2==0){red_reso2 += hasdiscrete;}
                        npix_side_reso2 = 1 << (nresos_grid-red_reso2-1);
                        elhashpix_1_reso2 = (int) floor((pos1_gal1 - hashpix_start1)/dpix1_resos[red_reso2]);
                        elhashpix_2_reso2 = (int) floor((pos2_gal1 - hashpix_start2)/dpix2_resos[red_reso2]);
                        elhashpix_reso2 = elhashpix_2_reso2*npix_side_reso2 + elhashpix_1_reso2;
                        redpix_reso2 = pix2redpix[z_gal1*len_matcher+matchers_resoshift[red_reso2]+elhashpix_reso2];
                        for (int zbin2=0; zbin2<nbinsz; zbin2++){
                            if (printregdbg){
                                printf("Gnupdates for reso1=%d reso2=%d red_reso2=%d, galindex=%d, z1=%d, z2=%d:%d radial updates; shiftstart %d = %d+%d+%d+%d+%d \n"
                                       ,elreso,elreso2,red_reso2,ind_gal1,z_gal1,zbin2,rbinmax-rbinmin,
                                       zbin2*zbin2shift + zbinshifts[z_gal1] + rbinmin*thetashifts_z[z_gal1] + 
                                       cumresoshift_z[z_gal1*(nresos+1) + elreso2] + redpix_reso2,
                                       zbin2*zbin2shift, zbinshifts[z_gal1], rbinmin*thetashifts_z[z_gal1],
                                       cumresoshift_z[z_gal1*(nresos+1) + elreso2], redpix_reso2);
                            }
                            for (int thisrbin=rbinmin; thisrbin<rbinmax; thisrbin++){
                                zrshift = zbin2*nbinsr_reso + thisrbin-rbinmin;
                                if (cabs(nextGns_norm[zrshift])<1e-10){continue;}
                                ind_Gncacheshift = zbin2*zbin2shift + zbinshifts[z_gal1] + thisrbin*thetashifts_z[z_gal1] + 
                                    cumresoshift_z[z_gal1*(nresos+1) + elreso2] + redpix_reso2;
                                _tmpindGn = zrshift;
                                _tmpindcache = ind_Gncacheshift;
                                for(int thisn=0; thisn<nnvals_Gn; thisn++){
                                    thisGn = nextGns[_tmpindGn];
                                    Gncache[_tmpindcache] += thisGn;
                                    wGncache[_tmpindcache] += wshape_gal1*thisGn;
                                    cwGncache[_tmpindcache] += conj(wshape_gal1)*thisGn;
                                    _tmpindGn += nbinszr_reso;
                                    _tmpindcache += nshift;
                                }
                                _tmpindGn = zrshift;
                                _tmpindcache = ind_Gncacheshift;
                                for(int thisn=0; thisn<nnvals_Nn; thisn++){
                                    thisGnnorm = nextGns_norm[_tmpindGn];
                                    Nncache[_tmpindcache] += thisGnnorm;
                                    wNncache[_tmpindcache] += w_gal1*thisGnnorm;
                                    _tmpindGn += nbinszr_reso;
                                    _tmpindcache += nshift;
                                }
                                Nncache_updates[ind_Gncacheshift] += 1;
                                Nncache_totupdates += 1;
                            }
                            
                        } 
                    }                    
                    // Allocate same reso Gammas
                    // First check for zero count bins (most likely only in discrete-discrete bit)
                    int nallowedcounts = 0;
                    for (int zbin1=0; zbin1<nbinsz; zbin1++){
                        for (int elb1=0; elb1<nbinsr_reso; elb1++){
                            zrshift = zbin1*nbinsr_reso + elb1;
                            if (nextncounts[zbin1*nbinsr_reso + elb1] != 0){
                                allowedrinds[nallowedcounts] = elb1;
                                allowedzinds[nallowedcounts] = zbin1;
                                nallowedcounts += 1;
                            }
                        }
                    }
                    // Now update the Gammans
                    // tmpGammas have shape (nthreads, nmax+1, nzcombis3, r*r, 4)
                    // Gns have shape (nnvals, nbinsz, nbinsr)
                    double complex h0, h1, h2, h3, w0, Gmnm3;
                    int thisnshift;
                    int _gammashift1, gammashift1, gammashift;
                    int ind_mnm3, ind_mnm1, ind_nm3, ind_nm1, ind_norm;
                    int _zcombi, zcombi, elb1_full, elb2_full;
                    for (int thisn=0; thisn<nmax+1; thisn++){
                        ind_mnm3 = (nzero-thisn-3)*nbinszr_reso;
                        ind_mnm1 = (nzero-thisn-1)*nbinszr_reso;
                        ind_nm3 = (nzero+thisn-3)*nbinszr_reso;
                        ind_nm1 = (nzero+thisn-1)*nbinszr_reso;
                        ind_norm = thisn*nbinszr_reso;
                        thisnshift = elthread*gamma_compshift + thisn*gamma_nshift;
                        int elb1, zbin2;
                        for (int zrcombis1=0; zrcombis1<nallowedcounts; zrcombis1++){
                            elb1 = allowedrinds[zrcombis1];
                            zbin2 = allowedzinds[zrcombis1];
                            elb1_full = elb1 + rbinmin;
                            zrshift = zbin2*nbinsr_reso + elb1;
                            // Double counting correction
                            if (dccorr==1){
                                zcombi = z_gal1*nbinsz*nbinsz + zbin2*nbinsz + zbin2;
                                gammashift1 = thisnshift + zcombi*gamma_zshift + elb1_full*nbinsr;
                                gammashift = 4*(gammashift1 + elb1_full);
                                //phirotm = wshape_gal1*nextG2ns[zrshift];
                                tmpGammans[gammashift] += wshape_gal1*nextG2ns[0*nbinszr_reso + zrshift];
                                tmpGammans[gammashift+1] += conj(wshape_gal1)*nextG2ns[1*nbinszr_reso + zrshift];
                                tmpGammans[gammashift+2] += wshape_gal1*nextG2ns[2*nbinszr_reso + zrshift];
                                tmpGammans[gammashift+3] += wshape_gal1*nextG2ns[3*nbinszr_reso + zrshift];
                                tmpGammans_norm[gammashift1 + elb1_full] -=  w_gal1*nextG2ns_norm[zrshift];
                            }
                            h0 = -wshape_gal1 * nextGns[ind_nm3 + zrshift];
                            h1 = -conj(wshape_gal1) * nextGns[ind_nm1 + zrshift];
                            h2 = -wshape_gal1 * conj(nextGns[ind_mnm1 + zrshift]);
                            h3 = -wshape_gal1 * nextGns[ind_nm3 + zrshift];
                            w0 = w_gal1 * nextGns_norm[ind_norm + zrshift];
                            _zcombi = z_gal1*nbinsz*nbinsz+zbin2*nbinsz;
                            _gammashift1 = thisnshift + elb1_full*nbinsr;
                            for (int zrcombis2=0; zrcombis2<nallowedcounts; zrcombis2++){
                                zcombi = _zcombi+allowedzinds[zrcombis2];
                                gammashift1 = _gammashift1 + zcombi*gamma_zshift; 
                                elb2_full = allowedrinds[zrcombis2] + rbinmin;
                                zrshift = allowedzinds[zrcombis2]*nbinsr_reso + allowedrinds[zrcombis2];
                                gammashift = 4*(gammashift1 + elb2_full);
                                Gmnm3 = nextGns[ind_mnm3 + zrshift];
                                tmpGammans[gammashift] += h0*Gmnm3;
                                tmpGammans[gammashift+1] += h1*nextGns[ind_mnm1 + zrshift];
                                tmpGammans[gammashift+2] += h2*Gmnm3;
                                tmpGammans[gammashift+3] += h3*conj(nextGns[ind_nm1 + zrshift]);
                                tmpGammans_norm[gammashift1 + elb2_full] += w0*conj(nextGns_norm[ind_norm + zrshift]);
                            }
                        }
                    }
                    for (int _i=0;_i<nnvals_Gn*nbinszr_reso;_i++){nextGns[_i]=0;}
                    for (int _i=0;_i<nnvals_Nn*nbinszr_reso;_i++){nextGns_norm[_i]=0;}
                    for (int _i=0;_i<4*nbinszr_reso;_i++){nextG2ns[_i]=0;}
                    for (int _i=0;_i<nbinszr_reso;_i++){nextG2ns_norm[_i]=0; 
                                                        nextncounts[_i]=0; allowedrinds[_i]=0; allowedzinds[_i]=0;}
                }
                free(nextGns);
                free(nextGns_norm);
                free(nextG2ns);
                free(nextG2ns_norm);
                free(nextncounts);
                free(allowedrinds);
                free(allowedzinds);
            }            
            
            // Allocate the Gamman for different grid resolutions from all the cached arrays 
            //
            // Note that for different configurations of the resolutions we do the Gamman
            // allocation as follows - see eq. (32) in 2309.08601 for the reasoning:
            // * Gamma0 = wshape * G_nm3 * G_mnm3
            //          --> (wG_nm3) * G_mnm3 if reso1 < reso2
            //          --> G_nm3 * wG_mnm3   if reso1 > reso2
            // * Gamma1 = conj(wshape) * G_nm1 * G_mnm1
            //          --> cwG_nm1 * G_mnm1 if reso1 < reso2
            //          --> G_nm1 * cwG_mnm1 if reso1 > reso2
            // * Gamma2 = wshape * conj(G_mnm1) * G_mnm3
            //          --> conj(cwG_mnm1) * G_mnm3 if reso1 < reso2
            //          --> conj(G_mnm1) * wG_mnm3  if reso1 > reso2
            // * Gamma3 = wshape * G_nm3 * conj(G_nm1)
            //          --> wG_nm3 * conj(G_nm1)  if reso1 < reso2
            //          --> G_nm3 * conj(cwG_nm1) if reso1 > reso2
            // where wG_xxx := wshape*G_xxx and cwG_xxx := conj(wshape)*G_xxx
            double complex h0, h1, h2, h3, w0;
            int thisnshift;
            int gammashift1, gammashift;
            int  zcombi;
            for (int thisn=0; thisn<nmax+1; thisn++){
                thisnshift = elthread*gamma_compshift + thisn*gamma_nshift;
                
                for (int zbin1=0; zbin1<nbinsz; zbin1++){
                    for (int zbin2=0; zbin2<nbinsz; zbin2++){
                        for (int zbin3=0; zbin3<nbinsz; zbin3++){
                            zcombi = zbin1*nbinsz*nbinsz + zbin2*nbinsz + zbin3;
                            int _imnm3, _imnm1, _inm1, _in;
                            int _thetashift_z = thetashifts_z[zbin1];
                            //if (zcombis_allowed[zcombi]==0){continue;}
                            
                            // Case max(reso1, reso2) = reso2
                            for (int thisreso1=0; thisreso1<nresos; thisreso1++){
                                rbinmin1 = reso_rindedges[thisreso1];
                                rbinmax1 = reso_rindedges[thisreso1+1];
                                for (int thisreso2=thisreso1+1; thisreso2<nresos; thisreso2++){
                                    rbinmin2 = reso_rindedges[thisreso2];
                                    rbinmax2 = reso_rindedges[thisreso2+1];
                                    for (int elgal=0; elgal<ngal_in_pix[zbin1*nresos+thisreso2]; elgal++){
                                        for (int elb1=rbinmin1; elb1<rbinmax1; elb1++){
                                            gammashift1 = thisnshift + zcombi*gamma_zshift + elb1*nbinsr;
                                            // n --> zbin2 --> zbin1 --> radius --> [ [0]*ngal_zbin1_reso1 | ... | [0]*ngal_zbin1_reson ]
                                            ind_Nncacheshift = zbin2*zbin2shift + zbinshifts[zbin1] + elb1*thetashifts_z[zbin1] +
                                                cumresoshift_z[zbin1*(nresos+1) + thisreso2] + elgal;
                                            ind_Gncacheshift = (nmax+3)*nshift + ind_Nncacheshift;
                                            h0 = -wGncache[(thisn-3)*nshift + ind_Gncacheshift];
                                            h1 = -cwGncache[(thisn-1)*nshift + ind_Gncacheshift];
                                            h2 = -conj(cwGncache[(-thisn-1)*nshift + ind_Gncacheshift]);
                                            h3 = -wGncache[(thisn-3)*nshift + ind_Gncacheshift];
                                            w0 = wNncache[thisn*nshift + ind_Nncacheshift];
                                            
                                            ind_Nncacheshift = zbin3*zbin2shift + zbinshifts[zbin1] + rbinmin2*thetashifts_z[zbin1] +
                                                    cumresoshift_z[zbin1*(nresos+1) + thisreso2] + elgal;
                                            ind_Gncacheshift = (nmax+3)*nshift + ind_Nncacheshift;
                                            _imnm3 = (-thisn-3)*nshift + ind_Gncacheshift;
                                            _imnm1 = (-thisn-1)*nshift + ind_Gncacheshift;
                                            _inm1 = (thisn-1)*nshift + ind_Gncacheshift;
                                            _in = thisn*nshift + ind_Nncacheshift;
                                            for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                                //ind_Nncacheshift = zbin3*zbin2shift + zbinshifts[zbin1] + elb2*thetashifts_z[zbin1] +
                                                //    cumresoshift_z[zbin1*(nresos+1) + thisreso2] + elgal;
                                                //ind_Gncacheshift = (nmax+3)*nshift + ind_Nncacheshift;
                                                gammashift = 4*(gammashift1 + elb2);
                                                tmpGammans[gammashift] += h0*Gncache[_imnm3];
                                                tmpGammans[gammashift+1] += h1*Gncache[_imnm1];
                                                tmpGammans[gammashift+2] += h2*Gncache[_imnm3];
                                                tmpGammans[gammashift+3] += h3*conj(Gncache[_inm1]);
                                                tmpGammans_norm[gammashift1 + elb2] += w0*conj(Nncache[_in]);
                                                ind_Nncacheshift += _thetashift_z;
                                                ind_Gncacheshift += _thetashift_z;
                                                _imnm3 += _thetashift_z;
                                                _imnm1 += _thetashift_z;
                                                _inm1 += _thetashift_z;
                                                _in += _thetashift_z;
                                            }
                                        }
                                    }
                                }
                            }
                            
                            // Case max(reso1, reso2) = reso1
                            for (int thisreso2=0; thisreso2<nresos; thisreso2++){
                                //rbinmin2 = (int) floor((log(reso_redges[thisreso2])-logrmin)/drbin);
                                //rbinmax2= mymin((int) floor((log(reso_redges[thisreso2+1])-logrmin)/drbin), nbinsr-1);
                                rbinmin2 = reso_rindedges[thisreso2];
                                rbinmax2 = reso_rindedges[thisreso2+1];
                                for (int thisreso1=thisreso2+1; thisreso1<nresos; thisreso1++){
                                    //rbinmin1 = (int) floor((log(reso_redges[thisreso1])-logrmin)/drbin);
                                    //rbinmax1= mymin((int) floor((log(reso_redges[thisreso1+1])-logrmin)/drbin), nbinsr-1);
                                    rbinmin1 = reso_rindedges[thisreso1];
                                    rbinmax1 = reso_rindedges[thisreso1+1];
                                    for (int elgal=0; elgal<ngal_in_pix[zbin1*nresos+thisreso1]; elgal++){
                                        for (int elb1=rbinmin1; elb1<rbinmax1; elb1++){
                                            gammashift1 = thisnshift + zcombi*gamma_zshift + elb1*nbinsr;
                                            ind_Nncacheshift = zbin2*zbin2shift + zbinshifts[zbin1] + elb1*thetashifts_z[zbin1] +
                                                cumresoshift_z[zbin1*(nresos+1) + thisreso1] + elgal;
                                            ind_Gncacheshift = (nmax+3)*nshift + ind_Nncacheshift;
                                            h0 = -Gncache[(thisn-3)*nshift + ind_Gncacheshift];
                                            h1 = -Gncache[(thisn-1)*nshift + ind_Gncacheshift];
                                            h2 = -conj(Gncache[(-thisn-1)*nshift + ind_Gncacheshift]);
                                            h3 = -Gncache[(thisn-3)*nshift + ind_Gncacheshift];
                                            w0 = Nncache[thisn*nshift + ind_Nncacheshift];
                                            ind_Nncacheshift = zbin3*zbin2shift + zbinshifts[zbin1] + rbinmin2*thetashifts_z[zbin1] +
                                                    cumresoshift_z[zbin1*(nresos+1) + thisreso1] + elgal;
                                            ind_Gncacheshift = (nmax+3)*nshift + ind_Nncacheshift;
                                            _imnm3 = (-thisn-3)*nshift + ind_Gncacheshift;
                                            _imnm1 = (-thisn-1)*nshift + ind_Gncacheshift;
                                            _inm1 = (thisn-1)*nshift + ind_Gncacheshift;
                                            _in = thisn*nshift + ind_Nncacheshift;
                                            for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                                gammashift = 4*(gammashift1 + elb2);
                                                tmpGammans[gammashift] += h0*wGncache[_imnm3];
                                                tmpGammans[gammashift+1] += h1*cwGncache[_imnm1];
                                                tmpGammans[gammashift+2] += h2*wGncache[_imnm3];
                                                tmpGammans[gammashift+3] += h3*conj(cwGncache[_inm1]);
                                                tmpGammans_norm[gammashift1 + elb2] += w0*conj(wNncache[_in]);
                                                ind_Nncacheshift += _thetashift_z;
                                                ind_Gncacheshift += _thetashift_z;
                                                _imnm3 += _thetashift_z;
                                                _imnm1 += _thetashift_z;
                                                _inm1 += _thetashift_z;
                                                _in += _thetashift_z;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }            
            
            free(reso_rindedges);
            free(rshift_index_matcher);
            free(rshift_pixs_galind_bounds);
            free(rshift_pix_gals);
            free(matchers_resoshift);
            free(ngal_in_pix);
            free(pix2redpix);  
            free(cumresoshift_z);
            free(thetashifts_z);
            free(zbinshifts);
        }
        free(Gncache);
        free(wGncache);
        free(cwGncache);
        free(Nncache);
        free(wNncache);
        free(Nncache_updates);
    }
    
    // Accumulate the Gamman
    #pragma omp parallel for num_threads(nthreads)
    for (int thisn=0; thisn<nmax+1; thisn++){
        int itmpGamma, iGamma;
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            for (int zcombi=0; zcombi<nbinsz*nbinsz*nbinsz; zcombi++){
                for (int elb1=0; elb1<nbinsr; elb1++){
                    for (int elb2=0; elb2<nbinsr; elb2++){
                        iGamma = thisn*_gamma_nshift + zcombi*_gamma_zshift + elb1*nbinsr + elb2;
                        itmpGamma = iGamma + thisthread*_gamma_compshift;
                        for (int elcomp=0; elcomp<4; elcomp++){
                            Gammans[elcomp*_gamma_compshift+iGamma] += tmpGammans[4*itmpGamma+elcomp];
                        }
                        Gammans_norm[iGamma] += tmpGammans_norm[itmpGamma];
                    }
                }
            }
        }
    }
    
    // Accumulate the bin distances and weights
    for (int elbinz=0; elbinz<nbinsz; elbinz++){
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            int tmpind = elbinz*nbinsr + elbinr;
            for (int thisthread=0; thisthread<nthreads; thisthread++){
                int tshift = thisthread*nbinsz*nbinsr; 
                totcounts[tmpind] += tmpwcounts[tshift+tmpind];
                totnorms[tmpind] += tmpwnorms[tshift+tmpind];
            }
        }
    }
    
    // Get bin centers
    for (int elbinz=0; elbinz<nbinsz; elbinz++){
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            int tmpind = elbinz*nbinsr + elbinr;
            if (totnorms[tmpind] != 0){
                bin_centers[tmpind] = totcounts[tmpind]/totnorms[tmpind];
            }
        }
    } 
    
    free(tmpwcounts);
    free(tmpwnorms);
    free(tmpGammans);
    free(tmpGammans_norm);
    free(totcounts);
    free(totnorms);
}


///////////////////////////////////////////////
/// THIRD-ORDER MIXED CORRELATION FUNCTIONS ///
///     (IE SOMETHING LIKE NGG AND GNN)     ///
///////////////////////////////////////////////

// Discrete estimtor of Source-Lens-Lens (G3L) Correlator
void alloc_Gammans_discrete_GNN(const MultiresoCatalog *cat_source, const NavHash *nav_source,
                                const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                                const BinningParams *bin, int nthreads, int verbose,
                                NPCFOutput *out){
    // --- shape (source) central catalog + scalar lens legs, both nresos=1 ---
    double *isinner_source = cat_source->isinner_resos, *w_source = cat_source->weight_resos;
    double *pos1_source = cat_source->pos1_resos, *pos2_source = cat_source->pos2_resos;
    double *e1_source = cat_source->e1_resos, *e2_source = cat_source->e2_resos;
    int *zbin_source = cat_source->zbin_resos, nbinsz_source = cat_source->nbinsz, ngal_source = cat_source->ngal_resos[0];
    double *w_lens = cat_lens->weight_resos, *pos1_lens = cat_lens->pos1_resos, *pos2_lens = cat_lens->pos2_resos;
    int *zbin_lens = cat_lens->zbin_resos, nbinsz_lens = cat_lens->nbinsz, ngal_lens = cat_lens->ngal_resos[0];
    int nmax = bin->nmax, nbinsr = bin->nbinsr, dccorr = bin->dccorr;
    double rmin = bin->rmin, rmax = bin->rmax;
    int *index_matcher_source = nav_source->index_matcher, *pixs_galind_bounds_source = nav_source->pixs_galind_bounds, *pix_gals_source = nav_source->pix_gals;
    int *index_matcher_lens = nav_lens->index_matcher, *pixs_galind_bounds_lens = nav_lens->pixs_galind_bounds, *pix_gals_lens = nav_lens->pix_gals;
    int nregions = nav_source->nregions;
    double pix1_start = nav_lens->pix1_start, pix1_d = nav_lens->pix1_d; int pix1_n = nav_lens->pix1_n;
    double pix2_start = nav_lens->pix2_start, pix2_d = nav_lens->pix2_d; int pix2_n = nav_lens->pix2_n;
    double *bin_centers = out->bin_centers;
    double complex *Upsilon_n = out->npcf, *Norm_n = out->norm_mp;

    int _upsilonzshift = nbinsr*nbinsr;
    int _nzcombis = nbinsz_source*nbinsz_lens*nbinsz_lens;
    int _upsilonnshift = _upsilonzshift*_nzcombis;
    int _upsilonthreadshift = (nmax+1)*_upsilonnshift;
    
    double *tmpwcounts = calloc(nthreads*nbinsz_source*nbinsz_lens*nbinsr, sizeof(double));
    double *tmpwnorms  = calloc(nthreads*nbinsz_source*nbinsz_lens*nbinsr, sizeof(double));
    double *totcounts = calloc(nbinsz_source*nbinsz_lens*nbinsr, sizeof(double));
    double *totnorms  = calloc(nbinsz_source*nbinsz_lens*nbinsr, sizeof(double));
    // Temporary arrays that are allocated in parallel and later reduced
    // Shape of tmpUpsilon ~ (nthreads, nnvals, nz_source, nz_lens, nz_lens, nbinsr, nbinsr)
    double complex *tmpUpsilon = calloc(nthreads*_upsilonthreadshift, sizeof(double complex));
    double complex *tmpNorm = calloc(nthreads*_upsilonthreadshift, sizeof(double complex));
    #pragma omp parallel for num_threads(nthreads)
    for(int elthread=0;elthread<nthreads;elthread++){
        int nregions_per_thread = nregions/nthreads;
        int nnvals_Gn = nmax+3; // Need [-1, ..., nmax+1]
        //int nnvals_Wn = nmax+1; // Need [0, ..., nmax]
        int nnvals_Ups = nmax+1;
        int nzcombis = nbinsz_source*nbinsz_lens*nbinsz_lens;
        int upsilon_zshift = nbinsr*nbinsr;
        int upsilon_nshift = upsilon_zshift*nzcombis;
        int upsilon_threadshift = nnvals_Ups*upsilon_nshift;
        int threadshift_counts = elthread*nbinsz_source*nbinsz_lens*nbinsr;
        int nbinszr_Gn = nbinsz_lens*nbinsr;
        double rmin_sq = rmin*rmin;
        double rmax_sq = rmax*rmax;
        double drbin = log(rmax/rmin)/nbinsr;
        
        for (int elregion=0; elregion<nregions; elregion++){
            int region_debug=99999;
            // Check if this thread is responsible for the region
            int nthread_target = mymin(elregion/nregions_per_thread, nthreads-1);
            if (nthread_target!=elthread){continue;}
            bool printregdbg = (verbose>0) && (elregion==region_debug);
            // printf("Region %d is in thread %d\n",elregion,elthread);
            if (printregdbg){printf("Region %d is in thread %d\n",elregion,elthread);}
            if ((verbose>0) && (elthread==nthreads/2)){
                int elreg_inthread = elregion-nregions_per_thread*(nthreads/2);
                printf("\rDone %.2f per cent",100*((double) elreg_inthread/nregions_per_thread));
            }
            
            int zbin_gal1, zbin_gal2;
            double isinner_gal1, pos1_gal1, pos2_gal1, w_gal1, e1_gal1, e2_gal1;
            double pos1_gal2, pos2_gal2, w_gal2;
            double complex wshape_gal1;
            int ind_red, ind_gal1, ind_gal2, lower1, upper1, lower2, upper2;
            int pix1_lower, pix2_lower, pix1_upper, pix2_upper;
            lower1 = pixs_galind_bounds_source[elregion];
            upper1 = pixs_galind_bounds_source[elregion+1];
            for (int ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                
                // Load source galaxy info
                ind_gal1 = pix_gals_source[ind_inpix1];
                isinner_gal1 = isinner_source[ind_gal1];
                if(isinner_gal1<1e-5){continue;}
                pos1_gal1 = pos1_source[ind_gal1];
                pos2_gal1 = pos2_source[ind_gal1];
                w_gal1 = isinner_gal1*w_source[ind_gal1];
                zbin_gal1 = zbin_source[ind_gal1];
                e1_gal1 = e1_source[ind_gal1];
                e2_gal1 = e2_source[ind_gal1];
                zbin_gal1 = zbin_source[ind_gal1];
                wshape_gal1 = w_gal1*(e1_gal1+I*e2_gal1);
                
                // Allocate the G_n and W_n coefficients + Double-counting correction factors
                double complex phirot, phirotc, nphirot;
                double rel1, rel2, dist;
                int ind_Wn, ind_counts, z1shift, z2rshift, rbin;
                double complex *thisWns = calloc(nnvals_Gn*nbinszr_Gn, sizeof(double complex)); // Here we do not need Gns!
                double complex *thisG2ns = calloc(nbinszr_Gn, sizeof(double complex));
                double complex *thisW2ns = calloc(nbinszr_Gn, sizeof(double complex));
                int *thisncounts = calloc(nbinszr_Gn, sizeof(int));
                int *allowedrinds = calloc(nbinszr_Gn, sizeof(int));
                int *allowedzinds = calloc(nbinszr_Gn, sizeof(int));
                z1shift = zbin_gal1*nbinsz_lens*nbinsr;
                pix1_lower = mymax(0, (int) floor((pos1_gal1 - (rmax+pix1_d) - pix1_start)/pix1_d));
                pix2_lower = mymax(0, (int) floor((pos2_gal1 - (rmax+pix2_d) - pix2_start)/pix2_d));
                pix1_upper = mymin(pix1_n-1, (int) floor((pos1_gal1 + (rmax+pix1_d) - pix1_start)/pix1_d));
                pix2_upper = mymin(pix2_n-1, (int) floor((pos2_gal1 + (rmax+pix2_d) - pix2_start)/pix2_d));
                for (int ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                    for (int ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                        ind_red = index_matcher_lens[ind_pix2*pix1_n + ind_pix1];
                        if (ind_red==-1){continue;}
                        lower2 = pixs_galind_bounds_lens[ind_red];
                        upper2 = pixs_galind_bounds_lens[ind_red+1];
                        for (int ind_inpix=lower2; ind_inpix<upper2; ind_inpix++){
                            ind_gal2 = pix_gals_lens[ind_inpix];
                            pos1_gal2 = pos1_lens[ind_gal2];
                            pos2_gal2 = pos2_lens[ind_gal2];
                            w_gal2 = w_lens[ind_gal2];
                            zbin_gal2 = zbin_lens[ind_gal2];
                            rel1 = pos1_gal2 - pos1_gal1;
                            rel2 = pos2_gal2 - pos2_gal1;
                            dist = rel1*rel1 + rel2*rel2;
                            if(dist < rmin_sq || dist >= rmax_sq) continue;
                            dist = sqrt(dist);
                            rbin = (int) floor(log(dist/rmin)/drbin);
                            if (rbin<0 || rbin>=nbinsr){
                                printf("%.2f %d",dist,rbin);
                                continue;
                            }
                            
                            z2rshift = zbin_gal2*nbinsr + rbin;
                            ind_counts = threadshift_counts + z1shift + z2rshift;
                            
                            phirot = (rel1+I*rel2)/dist;
                            phirotc = conj(phirot);
                            thisncounts[z2rshift] += 1;
                            tmpwcounts[ind_counts] += w_gal1*w_gal2*dist; 
                            tmpwnorms[ind_counts] += w_gal1*w_gal2; 
                            thisG2ns[z2rshift] += wshape_gal1*w_gal2*w_gal2*phirotc*phirotc;
                            thisW2ns[z2rshift] += w_gal1*w_gal2*w_gal2;
                            
                            ind_Wn = z2rshift;
                            nphirot = phirotc;
                            for (int nextn=-1;nextn<=nmax+1;nextn++){
                                thisWns[ind_Wn] += w_gal2*nphirot;
                                nphirot *= phirot; 
                                ind_Wn += nbinszr_Gn;
                            }
                        }
                    }
                }
                
                // Update the Upsilon_n & N_n for this galaxy
                // shape (nthreads, nmax+1, nbinsz_source, nbinsz_lens, nbinsz_lens, nbinsr, nbinsr)
                // First check for zero count bins
                // Note: Expected number of tracers in tomobin: <N> ~ 2*pi*nbar*drbin*<rbin>
                //   --> If we put sources (with nbar<~1/arcmin^2) in tomo bins, most 3pcf bins will be empty...
                int nallowedcounts = 0;
                for (int zbin1=0; zbin1<nbinsz_lens; zbin1++){
                    for (int elb1=0; elb1<nbinsr; elb1++){
                        z2rshift = zbin1*nbinsr + elb1;
                        if (thisncounts[z2rshift] != 0){
                            allowedrinds[nallowedcounts] = elb1;
                            allowedzinds[nallowedcounts] = zbin1;
                            nallowedcounts += 1;
                        }
                    }
                }
                // Now allocate only nonzero bins
                // Upsilon(thet1, thet2) ~ - we * W_{n-1}(thet1) * conj(W_{n+1})(thet2) + delta^K_{thet1,thet2} * (we * w*w*exp(-2phi))
                // Norm(thet1, thet2)    ~   w  * W_{n}(thet1)   * conj(W_{n})(thet2)   - delta^K_{thet1,thet2} * (w  * w*w)
                for (int thisn=0; thisn<nmax+1; thisn++){
                    int thisnshift = elthread*upsilon_threadshift + thisn*upsilon_nshift;
                    int _wind, _gammashift, zrshift, _zcombi, zcombi, gammashift, elb1, zbin2, elb2, zbin3;
                    double complex nextUps, nextN;
                    for (int zrcombis1=0; zrcombis1<nallowedcounts; zrcombis1++){
                        elb1 = allowedrinds[zrcombis1];
                        zbin2 = allowedzinds[zrcombis1];
                        zrshift = zbin2*nbinsr + elb1;
                        // Double counting correction
                        if (dccorr==1){
                            zcombi = zbin_gal1*nbinsz_lens*nbinsz_lens + zbin2*nbinsz_lens + zbin2;
                            gammashift = thisnshift + zcombi*upsilon_zshift + elb1*nbinsr+elb1;
                            tmpUpsilon[gammashift] += thisG2ns[zrshift];
                            tmpNorm[gammashift] -= thisW2ns[zrshift];
                        }
                        _zcombi = zbin_gal1*nbinsz_lens*nbinsz_lens + zbin2*nbinsz_lens;
                        _wind = (thisn+1)*nbinszr_Gn+zrshift;
                        _gammashift = thisnshift + elb1*nbinsr;
                        //nextUps = -wshape_gal1*thisWns[_wind+nbinszr_Gn]; //LP
                        nextUps = -wshape_gal1*thisWns[_wind-nbinszr_Gn]; //LL
                        nextN = w_gal1*thisWns[_wind];
                        for (int zrcombis2=0; zrcombis2<nallowedcounts; zrcombis2++){
                            elb2 = allowedrinds[zrcombis2];
                            zbin3 = allowedzinds[zrcombis2];
                            _wind = (thisn+1)*nbinszr_Gn + zbin3*nbinsr + elb2;
                            zcombi = _zcombi + zbin3;
                            gammashift = _gammashift + zcombi*upsilon_zshift + elb2;
                            tmpUpsilon[gammashift] += nextUps*conj(thisWns[_wind+nbinszr_Gn]);//LL
                            //tmpUpsilon[gammashift] += nextUps*conj(thisWns[_wind-nbinszr_Gn]);//LP
                            tmpNorm[gammashift] += nextN*conj(thisWns[_wind]);
                        }
                    }
                }
                free(thisWns);
                free(thisG2ns);
                free(thisW2ns);
                free(thisncounts);
                free(allowedrinds);
                free(allowedzinds);
            }
        }
    }
    
    // Accumulate the Upsilon_n / N_n
    #pragma omp parallel for num_threads(nthreads)
    for (int thisn=0; thisn<nmax+1; thisn++){
        int iUps;
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            int thisthreadshift = thisthread*_upsilonthreadshift;
            for (int zcombi=0; zcombi<_nzcombis; zcombi++){
                for (int elb1=0; elb1<nbinsr; elb1++){
                    for (int elb2=0; elb2<nbinsr; elb2++){
                        iUps = thisn*_upsilonnshift + zcombi*_upsilonzshift + elb1*nbinsr + elb2;
                        Upsilon_n[iUps] += tmpUpsilon[thisthreadshift+iUps];
                        Norm_n[iUps] += tmpNorm[thisthreadshift+iUps];
                    }
                }
            }
        }
    }
    
    // Accumulate the bin distances and weights
    for (int thisthread=0; thisthread<nthreads; thisthread++){
        int tmpind;
        int thisthreadshift = thisthread*nbinsz_source*nbinsz_lens*nbinsr; 
        for (int elbinz=0; elbinz<nbinsz_source*nbinsz_lens; elbinz++){
            for (int elbinr=0; elbinr<nbinsr; elbinr++){
                tmpind = elbinz*nbinsr + elbinr;
                totcounts[tmpind] += tmpwcounts[thisthreadshift+tmpind];
                totnorms[tmpind] += tmpwnorms[thisthreadshift+tmpind];
            }
        }
    }
    
    // Get bin centers
    for (int elbinz=0; elbinz<nbinsz_source*nbinsz_lens; elbinz++){
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            int tmpind = elbinz*nbinsr + elbinr;
            if (totnorms[tmpind] != 0){
                bin_centers[tmpind] = totcounts[tmpind]/totnorms[tmpind];
            }
        }
    } 
    free(tmpUpsilon);
    free(tmpNorm);
    free(tmpwcounts);
    free(tmpwnorms);
    free(totcounts);
    free(totnorms);
}

///////////////////////////////////////
// Slab-hashed third-order IA (ggI)   //
///////////////////////////////////////
// Third-order intrinsic-alignment estimators of Vedder et al. 2026
// (arXiv:2601.17914 Eq. 17) in a '3dbox' geometry: the discrete GNN multipole
// estimator (alloc_Gammans_discrete_GNN above) with a line-of-sight-window
// metric -- in direct analogy to how ng_slab (corrfunc_second.c) lifts the 2-pt
// correlator to the box. Only ggI = S.D~.D~ / RRR is wired up here (shape at the
// central vertex, two galaxy-position legs, D~ = D - f.R with f = W_D/W_R; the
// RRR normalization uses random legs). gII and III would reuse the same leg-
// multipole helper below with a different vertex combination.

// Shared slab-hashed leg-multipole helper for the projected scalar/polar family:
// for one central galaxy and one slab-hashed leg catalog, accumulate per
// (zbin_leg, r_perp bin) the leg count multipoles Wn[m] = sum_leg w e^{i m phi}
// (block b <-> m = m0_W+b, nW blocks; pass Wn=NULL to skip) and the leg polar
// multipoles Gn[m] = sum_leg w(e1+i e2) e^{i m phi} (block b <-> m = m0_G+b, nG
// blocks; pass Gn=NULL / no e1,e2 to skip -- scalar legs). The multipole ranges
// are parameterized because GNN, NGG and GGG need different windows. Optional
// double-counting self-terms (each computed only when its pointer is non-NULL):
// sumG2 = sum w^2 e^{-2i phi} and sumW2 = sum w^2 (count legs); sumG4 =
// sum (w(e1+ie2))^2 e^{-4i phi} and sumGabs = sum |w(e1+ie2)|^2 (NGG polar legs);
// sumG6 = sum (w(e1+ie2))^2 e^{-6i phi}, sumG2p = sum (w(e1+ie2))^2 e^{-2i phi} and
// sumGabsp = sum |w(e1+ie2)|^2 e^{-2i phi} (the GGG polar-leg self-terms).
// Only neighbours with r_perp in [rmin,rmax) and |dz| < Pi contribute. When
// wcounts/wnorms are non-NULL it also accumulates the weighted radial sums for
// the bin centers (pass the base for this thread + central tomo-bin; indexed by
// zbin_leg*nbinsr + rbin).
static void ia_slab_legmultipoles(
    double c_pos1, double c_pos2, double c_pos3, double c_w,
    const double *h_pos1, const double *h_pos2, const double *h_pos3,
    const double *h_w, const int *h_zbin, const double *h_e1, const double *h_e2,
    int nbinsz_leg, int nslabs, double z0, double dpix_z,
    double pix1_start, double pix1_d, int pix1_n,
    double pix2_start, double pix2_d, int pix2_n,
    const int *slab_offsets, const int *index_matcher, const int *pixs_galind_bounds,
    const int *rshift_bounds, const int *pix_gals,
    double rmin, double rmax, int nbinsr, double Pi,
    int m0_W, int nW, int m0_G, int nG,
    double complex *Wn, double complex *Gn, double complex *sumG2, double *sumW2,
    double complex *sumG4, double complex *sumGabs,
    double complex *sumG6, double complex *sumG2p, double complex *sumGabsp,
    int *ncounts, double *wcounts, double *wnorms){

    int npix = pix1_n*pix2_n;
    int nbinszr_leg = nbinsz_leg*nbinsr;
    double rmin2 = rmin*rmin, rmax2 = rmax*rmax;
    double dlnr_inv = nbinsr/log(rmax/rmin);

    // Slabs overlapping [c_pos3-Pi, c_pos3+Pi] and the transverse search box.
    int s_lo = (int) floor((c_pos3 - Pi - z0)/dpix_z);
    int s_hi = (int) floor((c_pos3 + Pi - z0)/dpix_z);
    if (s_lo < 0){ s_lo = 0; }
    if (s_hi > nslabs-1){ s_hi = nslabs-1; }
    int pix1_lo = mymax(0, (int) floor((c_pos1 - (rmax+pix1_d) - pix1_start)/pix1_d));
    int pix1_hi = mymin(pix1_n-1, (int) floor((c_pos1 + (rmax+pix1_d) - pix1_start)/pix1_d));
    int pix2_lo = mymax(0, (int) floor((c_pos2 - (rmax+pix2_d) - pix2_start)/pix2_d));
    int pix2_hi = mymin(pix2_n-1, (int) floor((c_pos2 + (rmax+pix2_d) - pix2_start)/pix2_d));

    for (int s=s_lo; s<=s_hi; s++){
        int matcher_shift = s*npix;
        int bounds_shift = rshift_bounds[s];
        int gals_shift = slab_offsets[s];
        for (int ip1=pix1_lo; ip1<=pix1_hi; ip1++){
            for (int ip2=pix2_lo; ip2<=pix2_hi; ip2++){
                int ind_red = index_matcher[matcher_shift + ip2*pix1_n + ip1];
                if (ind_red == -1){ continue; }
                int lower = pixs_galind_bounds[bounds_shift + ind_red];
                int upper = pixs_galind_bounds[bounds_shift + ind_red + 1];
                for (int k=lower; k<upper; k++){
                    int j = pix_gals[gals_shift + k];
                    double rel1 = h_pos1[j] - c_pos1;
                    double rel2 = h_pos2[j] - c_pos2;
                    double d2 = rel1*rel1 + rel2*rel2;
                    if (d2 < rmin2 || d2 >= rmax2){ continue; }
                    double dz = h_pos3[j] - c_pos3;
                    if (dz < 0){ dz = -dz; }
                    if (dz >= Pi){ continue; }
                    double dist = sqrt(d2);
                    int rbin = (int) floor(log(dist/rmin)*dlnr_inv);
                    if (rbin < 0 || rbin >= nbinsr){ continue; }

                    int z2rshift = h_zbin[j]*nbinsr + rbin;
                    double w = h_w[j];
                    double complex phirot = (rel1 + I*rel2)/dist;
                    double complex phirotc = conj(phirot);

                    ncounts[z2rshift] += 1;
                    if (sumG2){ sumG2[z2rshift] += w*w*phirotc*phirotc; }   // sum w^2 e^{-2i phi}
                    if (sumW2){ sumW2[z2rshift] += w*w; }
                    // Wn[m] = sum w e^{i m phi}, block b <-> m = m0_W + b.
                    if (Wn){
                        double complex nphirot = 1.;
                        if (m0_W >= 0){ for (int q=0; q<m0_W; q++){ nphirot *= phirot; } }
                        else          { for (int q=0; q<-m0_W; q++){ nphirot *= phirotc; } }
                        int ind = z2rshift;
                        for (int b=0; b<nW; b++){
                            Wn[ind] += w*nphirot;
                            nphirot *= phirot;
                            ind += nbinszr_leg;
                        }
                    }
                    // Gn[m] = sum w(e1+ie2) e^{i m phi}, block b <-> m = m0_G + b.
                    if (Gn){
                        double complex wshape = w*(h_e1[j] + I*h_e2[j]);
                        double complex tpc = phirotc*phirotc;   // e^{-2i phi}
                        if (sumG4){ sumG4[z2rshift] += wshape*wshape*tpc*tpc; }
                        if (sumGabs){ sumGabs[z2rshift] += wshape*conj(wshape); }
                        if (sumG6){ sumG6[z2rshift] += wshape*wshape*tpc*tpc*tpc; }
                        if (sumG2p){ sumG2p[z2rshift] += wshape*wshape*tpc; }
                        if (sumGabsp){ sumGabsp[z2rshift] += wshape*conj(wshape)*tpc; }
                        double complex nphirot = 1.;
                        if (m0_G >= 0){ for (int q=0; q<m0_G; q++){ nphirot *= phirot; } }
                        else          { for (int q=0; q<-m0_G; q++){ nphirot *= phirotc; } }
                        int ind = z2rshift;
                        for (int b=0; b<nG; b++){
                            Gn[ind] += wshape*nphirot;
                            nphirot *= phirot;
                            ind += nbinszr_leg;
                        }
                    }
                    if (wcounts){
                        wcounts[z2rshift] += c_w*w*dist;
                        wnorms[z2rshift]  += c_w*w;
                    }
                }
            }
        }
    }
}

// Slab-hashed polar-scalar-scalar (GNN) cross-correlator in the projected
// '3dbox' geometry. The polar (spin-2) catalog is the central vertex, looped
// directly (like ng_slab's query, not hashed); the two scalar legs come from a
// data (D) and a random (R) catalog, slab-hashed on the same shared transverse+
// LOS grid. The intrinsic-alignment ggI estimator (Vedder et al. 2026 Eq.17,
// S.D~.D~ / RRR) is the motivating application, but rather than form D~ = D - f.R
// in C this kernel emits the four raw, f-free numerator sub-correlators and the
// shared random RRR count; the Python layer applies f = W_D/W_R and combines.
//   S{ab}_n(t1,t2) = -wpolar * W^a_{n-1}(t1) * conj(W^b_{n+1}(t2)) [+ dc if a==b],
//                    with (a,b) = (D,D),(D,R),(R,D),(R,R) -> components 0..3,
//   RRR_n(t1,t2)   =  w_R    * W^R_n(t1)      * conj(W^R_n(t2))    [- dc].
// Only same-catalog leg pairs (DD, RR) carry a diagonal double-counting term.
void alloc_Gammans_slab_GNN(const MultiresoCatalog *cat_polar, const MultiresoCatalog *cat_D,
                            const NavHash *nav_D, const MultiresoCatalog *cat_R,
                            const NavHash *nav_R, const BinningParams *bin,
                            int nthreads, int verbose, NPCFOutput *out){
    // --- central polar (spin-2) catalog, looped directly (nresos=1, no nav) ---
    double *pos1_shape = cat_polar->pos1_resos, *pos2_shape = cat_polar->pos2_resos, *pos3_shape = cat_polar->pos3_resos;
    double *w_shape = cat_polar->weight_resos, *e1_shape = cat_polar->e1_resos, *e2_shape = cat_polar->e2_resos;
    int *zbin_shape = cat_polar->zbin_resos, nbinsz_shape = cat_polar->nbinsz, ngal_shape = cat_polar->ngal_resos[0];
    // --- scalar legs: data (D) and random (R), each slab-hashed ---
    double *pos1_D = cat_D->pos1_resos, *pos2_D = cat_D->pos2_resos, *pos3_D = cat_D->pos3_resos, *w_D = cat_D->weight_resos;
    int *zbin_D = cat_D->zbin_resos, nbinsz_pos = cat_D->nbinsz;
    int *slab_offsets_D = nav_D->slab_offsets, *index_matcher_D = nav_D->index_matcher, *pixs_galind_bounds_D = nav_D->pixs_galind_bounds;
    int *rshift_bounds_D = nav_D->rshift_bounds, *pix_gals_D = nav_D->pix_gals;
    double *pos1_R = cat_R->pos1_resos, *pos2_R = cat_R->pos2_resos, *pos3_R = cat_R->pos3_resos, *w_R = cat_R->weight_resos;
    int *zbin_R = cat_R->zbin_resos, ngal_R = cat_R->ngal_resos[0];
    int *slab_offsets_R = nav_R->slab_offsets, *index_matcher_R = nav_R->index_matcher, *pixs_galind_bounds_R = nav_R->pixs_galind_bounds;
    int *rshift_bounds_R = nav_R->rshift_bounds, *pix_gals_R = nav_R->pix_gals;
    // --- shared slab grid (D and R share it) + binning ---
    int nslabs = nav_D->nslabs; double z0 = nav_D->z0, dpix_z = nav_D->dpix_z;
    double pix1_start = nav_D->pix1_start, pix1_d = nav_D->pix1_d; int pix1_n = nav_D->pix1_n;
    double pix2_start = nav_D->pix2_start, pix2_d = nav_D->pix2_d; int pix2_n = nav_D->pix2_n;
    int nmax = bin->nmax, nbinsr = bin->nbinsr, dccorr = bin->dccorr;
    double rmin = bin->rmin, rmax = bin->rmax, Pi = bin->Pi;
    double *bin_centers = out->bin_centers;
    double complex *Comp_n = out->npcf, *RRR_n = out->norm_mp;

    int nnvals = nmax+3;                        // W_n blocks: m in [-1, nmax+1]
    int ncomp = 4;                              // SDD, SDR, SRD, SRR
    int nbinszr_leg = nbinsz_pos*nbinsr;
    int nzcombis = nbinsz_shape*nbinsz_pos*nbinsz_pos;
    int comp_zshift = nbinsr*nbinsr;
    int comp_nshift = comp_zshift*nzcombis;
    int comp_size = (nmax+1)*comp_nshift;       // one estimator component (all n)
    int ups_threadshift = ncomp*comp_size;
    int counts_threadshift = nbinsz_shape*nbinsz_pos*nbinsr;

    double complex *tmpComp = calloc((size_t)nthreads*ups_threadshift, sizeof(double complex));
    double complex *tmpRRR  = calloc((size_t)nthreads*comp_size, sizeof(double complex));
    double *tmpwcounts = calloc((size_t)nthreads*counts_threadshift, sizeof(double));
    double *tmpwnorms  = calloc((size_t)nthreads*counts_threadshift, sizeof(double));

    // (A) polar central -> four raw numerator components S.(D/R).(D/R) + bin centers.
    #pragma omp parallel num_threads(nthreads)
    {
        int elthread = omp_get_thread_num();
        double complex *thComp = tmpComp + (size_t)elthread*ups_threadshift;
        double complex *SDD = thComp, *SDR = thComp+comp_size, *SRD = thComp+2*comp_size, *SRR = thComp+3*comp_size;
        double *thwcounts = tmpwcounts + (size_t)elthread*counts_threadshift;
        double *thwnorms  = tmpwnorms  + (size_t)elthread*counts_threadshift;

        #pragma omp for schedule(dynamic, 256)
        for (int ig=0; ig<ngal_shape; ig++){
            double c_pos1 = pos1_shape[ig];
            double c_pos2 = pos2_shape[ig];
            double c_pos3 = pos3_shape[ig];
            double c_w = w_shape[ig];
            int zbin_c = zbin_shape[ig];
            double complex wshape = c_w*(e1_shape[ig] + I*e2_shape[ig]);

            double complex *Wn_D = calloc(nnvals*nbinszr_leg, sizeof(double complex));
            double complex *Wn_R = calloc(nnvals*nbinszr_leg, sizeof(double complex));
            double complex *sumG2_D = calloc(nbinszr_leg, sizeof(double complex));
            double complex *sumG2_R = calloc(nbinszr_leg, sizeof(double complex));
            int *ncounts = calloc(nbinszr_leg, sizeof(int));
            int *allowedr = calloc(nbinszr_leg, sizeof(int));
            int *allowedz = calloc(nbinszr_leg, sizeof(int));

            double *wc_base = thwcounts + zbin_c*nbinsz_pos*nbinsr;
            double *wn_base = thwnorms  + zbin_c*nbinsz_pos*nbinsr;

            // D legs (also feed the bin-center weighted sums). Scalar legs:
            // count multipoles Wn over m in [-1, nmax+1], no polar terms.
            ia_slab_legmultipoles(c_pos1, c_pos2, c_pos3, c_w,
                pos1_D, pos2_D, pos3_D, w_D, zbin_D, NULL, NULL,
                nbinsz_pos, nslabs, z0, dpix_z, pix1_start, pix1_d, pix1_n,
                pix2_start, pix2_d, pix2_n, slab_offsets_D, index_matcher_D,
                pixs_galind_bounds_D, rshift_bounds_D, pix_gals_D,
                rmin, rmax, nbinsr, Pi, -1, nnvals, 0, 0,
                Wn_D, NULL, sumG2_D, NULL, NULL, NULL, NULL, NULL, NULL, ncounts, wc_base, wn_base);
            // R legs (no bin-center accumulation).
            ia_slab_legmultipoles(c_pos1, c_pos2, c_pos3, c_w,
                pos1_R, pos2_R, pos3_R, w_R, zbin_R, NULL, NULL,
                nbinsz_pos, nslabs, z0, dpix_z, pix1_start, pix1_d, pix1_n,
                pix2_start, pix2_d, pix2_n, slab_offsets_R, index_matcher_R,
                pixs_galind_bounds_R, rshift_bounds_R, pix_gals_R,
                rmin, rmax, nbinsr, Pi, -1, nnvals, 0, 0,
                Wn_R, NULL, sumG2_R, NULL, NULL, NULL, NULL, NULL, NULL, ncounts, NULL, NULL);

            // Nonzero (zbin_leg, r) bins (union of D and R occupancy).
            int nallowed = 0;
            for (int z=0; z<nbinsz_pos; z++){
                for (int r=0; r<nbinsr; r++){
                    if (ncounts[z*nbinsr + r] != 0){
                        allowedr[nallowed] = r; allowedz[nallowed] = z; nallowed++;
                    }
                }
            }

            for (int thisn=0; thisn<nmax+1; thisn++){
                int nshift = thisn*comp_nshift;
                for (int a1=0; a1<nallowed; a1++){
                    int elb1 = allowedr[a1], zbin2 = allowedz[a1];
                    int zr1 = zbin2*nbinsr + elb1;
                    double complex D1 = Wn_D[thisn*nbinszr_leg + zr1];
                    double complex R1 = Wn_R[thisn*nbinszr_leg + zr1];
                    if (dccorr==1){
                        int zc = zbin_c*nbinsz_pos*nbinsz_pos + zbin2*nbinsz_pos + zbin2;
                        int gd = nshift + zc*comp_zshift + elb1*nbinsr + elb1;
                        SDD[gd] += wshape*sumG2_D[zr1];
                        SRR[gd] += wshape*sumG2_R[zr1];
                    }
                    for (int a2=0; a2<nallowed; a2++){
                        int elb2 = allowedr[a2], zbin3 = allowedz[a2];
                        int zr2 = zbin3*nbinsr + elb2;
                        int zc = zbin_c*nbinsz_pos*nbinsz_pos + zbin2*nbinsz_pos + zbin3;
                        int gs = nshift + zc*comp_zshift + elb1*nbinsr + elb2;
                        double complex cD2 = conj(Wn_D[(thisn+2)*nbinszr_leg + zr2]);
                        double complex cR2 = conj(Wn_R[(thisn+2)*nbinszr_leg + zr2]);
                        SDD[gs] += -wshape * D1 * cD2;
                        SDR[gs] += -wshape * D1 * cR2;
                        SRD[gs] += -wshape * R1 * cD2;
                        SRR[gs] += -wshape * R1 * cR2;
                    }
                }
            }

            free(Wn_D); free(Wn_R); free(sumG2_D); free(sumG2_R);
            free(ncounts); free(allowedr); free(allowedz);
        }
    }

    // (B) random central -> shared random RRR count (f-free).
    #pragma omp parallel num_threads(nthreads)
    {
        int elthread = omp_get_thread_num();
        double complex *thRRR = tmpRRR + (size_t)elthread*comp_size;

        #pragma omp for schedule(dynamic, 256)
        for (int ig=0; ig<ngal_R; ig++){
            double c_pos1 = pos1_R[ig], c_pos2 = pos2_R[ig], c_pos3 = pos3_R[ig], c_w = w_R[ig];
            int zbin_c = zbin_R[ig];
            double complex *Wn = calloc(nnvals*nbinszr_leg, sizeof(double complex));
            double *sumW2 = calloc(nbinszr_leg, sizeof(double));
            int *ncounts = calloc(nbinszr_leg, sizeof(int));
            int *allowedr = calloc(nbinszr_leg, sizeof(int));
            int *allowedz = calloc(nbinszr_leg, sizeof(int));

            ia_slab_legmultipoles(c_pos1, c_pos2, c_pos3, c_w,
                pos1_R, pos2_R, pos3_R, w_R, zbin_R, NULL, NULL,
                nbinsz_pos, nslabs, z0, dpix_z, pix1_start, pix1_d, pix1_n,
                pix2_start, pix2_d, pix2_n, slab_offsets_R, index_matcher_R,
                pixs_galind_bounds_R, rshift_bounds_R, pix_gals_R,
                rmin, rmax, nbinsr, Pi, -1, nnvals, 0, 0,
                Wn, NULL, NULL, sumW2, NULL, NULL, NULL, NULL, NULL, ncounts, NULL, NULL);

            int nallowed = 0;
            for (int z=0; z<nbinsz_pos; z++){
                for (int r=0; r<nbinsr; r++){
                    if (ncounts[z*nbinsr + r] != 0){
                        allowedr[nallowed] = r; allowedz[nallowed] = z; nallowed++;
                    }
                }
            }

            for (int thisn=0; thisn<nmax+1; thisn++){
                int nshift = thisn*comp_nshift;
                for (int a1=0; a1<nallowed; a1++){
                    int elb1 = allowedr[a1], zbin2 = allowedz[a1];
                    int zr1 = zbin2*nbinsr + elb1;
                    double complex Rn1 = Wn[(thisn+1)*nbinszr_leg + zr1];
                    if (dccorr==1){
                        int zc = zbin_c*nbinsz_pos*nbinsz_pos + zbin2*nbinsz_pos + zbin2;
                        int gd = nshift + zc*comp_zshift + elb1*nbinsr + elb1;
                        thRRR[gd] -= c_w*sumW2[zr1];
                    }
                    for (int a2=0; a2<nallowed; a2++){
                        int elb2 = allowedr[a2], zbin3 = allowedz[a2];
                        int zr2 = zbin3*nbinsr + elb2;
                        int zc = zbin_c*nbinsz_pos*nbinsz_pos + zbin2*nbinsz_pos + zbin3;
                        int gs = nshift + zc*comp_zshift + elb1*nbinsr + elb2;
                        thRRR[gs] += c_w * Rn1 * conj(Wn[(thisn+1)*nbinszr_leg + zr2]);
                    }
                }
            }

            free(Wn); free(sumW2); free(ncounts); free(allowedr); free(allowedz);
        }
    }

    // Reduce the components and the RRR count across threads.
    #pragma omp parallel for num_threads(nthreads)
    for (int i=0; i<comp_size; i++){
        for (int t=0; t<nthreads; t++){
            for (int c=0; c<ncomp; c++){
                Comp_n[c*comp_size + i] += tmpComp[(size_t)t*ups_threadshift + c*comp_size + i];
            }
            RRR_n[i] += tmpRRR[(size_t)t*comp_size + i];
        }
    }

    // Reduce the bin-center weighted sums and finalize the centers.
    double *totcounts = calloc(counts_threadshift, sizeof(double));
    double *totnorms  = calloc(counts_threadshift, sizeof(double));
    for (int t=0; t<nthreads; t++){
        size_t tshift = (size_t)t*counts_threadshift;
        for (int i=0; i<counts_threadshift; i++){
            totcounts[i] += tmpwcounts[tshift + i];
            totnorms[i]  += tmpwnorms[tshift + i];
        }
    }
    for (int i=0; i<counts_threadshift; i++){
        if (totnorms[i] != 0){ bin_centers[i] = totcounts[i]/totnorms[i]; }
    }

    free(tmpComp); free(tmpRRR); free(tmpwcounts); free(tmpwnorms);
    free(totcounts); free(totnorms);
}

///////////////////////////////////////////
// Slab-hashed scalar-polar-polar (NGG)   //
///////////////////////////////////////////
// Accumulate one central's contribution to the two NGG numerator natural
// components from its polar-data leg multipoles Gn (block b <-> m = b-nmax-2, so
// G_{n-2} lives at block nmax+n) and the diagonal self-terms sumG4 = sum (we)^2
// e^{-4i phi}, sumGabs = sum |we|^2. fac_c = central weight (+w); the D~ = D - f.R
// combination is done in Python across the raw DSS/RSS component blocks (thUps is
// the base of the target block). Iterates only the occupied (zbin_leg, r) bins.
//   Ups_-(t1,t2) += fac_c [G_{n-2}(t1) G_{-n-2}(t2) - dc],  (comp 0)
//   Ups_+(t1,t2) += fac_c [G_{n-2}(t1) conj(G_{n-2}(t2)) - dc].  (comp 1)
static void ngg_accum_upsilon(
    double complex *thUps, double fac_c, int zc,
    const double complex *Gn, const double complex *sumG4, const double complex *sumGabs,
    const int *allowedr, const int *allowedz, int nallowed,
    int nmax, int nbinsr, int nbinsz_lens, int nbinsz_polar, int dccorr){

    int nbinszr = nbinsz_polar*nbinsr;
    int nmp = 2*nmax+1;
    int nzcombis = nbinsz_lens*nbinsz_polar*nbinsz_polar;
    int ups_zshift = nbinsr*nbinsr;
    int ups_nshift = ups_zshift*nzcombis;
    int ups_compshift = nmp*ups_nshift;
    int zc_base = zc*nbinsz_polar*nbinsz_polar;

    for (int thisn=-nmax; thisn<=nmax; thisn++){
        int nshift = (thisn+nmax)*ups_nshift;
        int blk1 = (nmax+thisn)*nbinszr;   // G_{thisn-2}
        int blk2 = (nmax-thisn)*nbinszr;   // G_{-thisn-2}
        for (int a1=0; a1<nallowed; a1++){
            int elb1 = allowedr[a1], z2 = allowedz[a1];
            int zr1 = z2*nbinsr + elb1;
            double complex u = fac_c*Gn[blk1 + zr1];
            if (dccorr){
                int zc3 = zc_base + z2*nbinsz_polar + z2;
                int gd = nshift + zc3*ups_zshift + elb1*nbinsr + elb1;
                thUps[gd]               -= fac_c*sumG4[zr1];
                thUps[gd+ups_compshift] -= fac_c*sumGabs[zr1];
            }
            for (int a2=0; a2<nallowed; a2++){
                int elb2 = allowedr[a2], z3 = allowedz[a2];
                int zr2 = z3*nbinsr + elb2;
                int zc3 = zc_base + z2*nbinsz_polar + z3;
                int gs = nshift + zc3*ups_zshift + elb1*nbinsr + elb2;
                thUps[gs]               += u*Gn[blk2 + zr2];
                thUps[gs+ups_compshift] += u*conj(Gn[blk1 + zr2]);
            }
        }
    }
}

// Accumulate one random-central's contribution to the shared RRR count from its
// random leg count multipoles Wn (block b <-> m = b-nmax) and the diagonal self-
// term sumW2 = sum w^2 (f-free; Python applies the f^3 rescaling).
//   RRR(t1,t2) += w_c [W_n(t1) W_{-n}(t2) - dc].
static void ngg_accum_norm(
    double complex *thNorm, double w_c, int zc,
    const double complex *Wn, const double *sumW2,
    const int *allowedr, const int *allowedz, int nallowed,
    int nmax, int nbinsr, int nbinsz_lens, int nbinsz_polar, int dccorr){

    int nbinszr = nbinsz_polar*nbinsr;
    int nmp = 2*nmax+1;
    int nzcombis = nbinsz_lens*nbinsz_polar*nbinsz_polar;
    int norm_zshift = nbinsr*nbinsr;
    int norm_nshift = norm_zshift*nzcombis;
    int zc_base = zc*nbinsz_polar*nbinsz_polar;

    for (int thisn=-nmax; thisn<=nmax; thisn++){
        int nshift = (thisn+nmax)*norm_nshift;
        int blk1 = (nmax+thisn)*nbinszr;   // W_{thisn}
        int blk2 = (nmax-thisn)*nbinszr;   // W_{-thisn}
        for (int a1=0; a1<nallowed; a1++){
            int elb1 = allowedr[a1], z2 = allowedz[a1];
            int zr1 = z2*nbinsr + elb1;
            double complex nN = w_c*Wn[blk1 + zr1];
            if (dccorr){
                int zc3 = zc_base + z2*nbinsz_polar + z2;
                int gd = nshift + zc3*norm_zshift + elb1*nbinsr + elb1;
                thNorm[gd] -= w_c*sumW2[zr1];
            }
            for (int a2=0; a2<nallowed; a2++){
                int elb2 = allowedr[a2], z3 = allowedz[a2];
                int zr2 = z3*nbinsr + elb2;
                int zc3 = zc_base + z2*nbinsz_polar + z3;
                int gs = nshift + zc3*norm_zshift + elb1*nbinsr + elb2;
                thNorm[gs] += nN*Wn[blk2 + zr2];
            }
        }
    }
}

// Slab-hashed scalar-polar-polar (NGG) cross-correlator in the projected '3dbox'
// geometry (line-of-sight window |dz| < Pi). Scalar (density) central + two polar
// legs, normalized by RRR (Vedder et al. 2026 Eq.17, D~.S.S / RRR). Rather than
// form D~ = D - f.R in C, the kernel emits the two raw, f-free numerator sub-
// correlators D.S.S (data-lens central) and R.S.S (random-lens central), each with
// 2 natural components, plus the shared random RRR count; the Python layer applies
// f = W_D/W_R and combines D~.S.S = DSS - f.RSS over RRR. The polar legs use the
// shape-data catalog (G-multipoles); the single random (lens random) is looped as
// the R central AND hashed (nav_lensR) for the RRR count legs. Struct interface;
// both lens catalogs are looped directly (lensR also hashed via nav_lensR).
void alloc_Gammans_slab_NGG(const MultiresoCatalog *cat_lensD, const MultiresoCatalog *cat_lensR,
                            const MultiresoCatalog *cat_shapeD, const NavHash *nav_shapeD,
                            const NavHash *nav_lensR, const BinningParams *bin,
                            int nthreads, int verbose, NPCFOutput *out){
    // --- scalar (density) central: data D and random R, looped directly ---
    double *pos1_D = cat_lensD->pos1_resos, *pos2_D = cat_lensD->pos2_resos, *pos3_D = cat_lensD->pos3_resos;
    double *w_D = cat_lensD->weight_resos; int *zbin_D = cat_lensD->zbin_resos;
    int ngal_D = cat_lensD->ngal_resos[0], nbinsz_lens = cat_lensD->nbinsz;
    double *pos1_Rl = cat_lensR->pos1_resos, *pos2_Rl = cat_lensR->pos2_resos, *pos3_Rl = cat_lensR->pos3_resos;
    double *w_Rl = cat_lensR->weight_resos; int *zbin_Rl = cat_lensR->zbin_resos;
    int ngal_Rl = cat_lensR->ngal_resos[0];
    // --- polar legs: shape-data catalog (signal G), slab-hashed on nav_shapeD ---
    double *pos1_sD = cat_shapeD->pos1_resos, *pos2_sD = cat_shapeD->pos2_resos, *pos3_sD = cat_shapeD->pos3_resos;
    double *w_sD = cat_shapeD->weight_resos, *e1_sD = cat_shapeD->e1_resos, *e2_sD = cat_shapeD->e2_resos;
    int *zbin_sD = cat_shapeD->zbin_resos, nbinsz_polar = cat_shapeD->nbinsz;
    int *im_sD = nav_shapeD->index_matcher, *pgb_sD = nav_shapeD->pixs_galind_bounds, *pg_sD = nav_shapeD->pix_gals;
    int *so_sD = nav_shapeD->slab_offsets, *rsb_sD = nav_shapeD->rshift_bounds;
    int nslabs_sD = nav_shapeD->nslabs; double z0_sD = nav_shapeD->z0, dpixz_sD = nav_shapeD->dpix_z;
    double p1s_sD = nav_shapeD->pix1_start, p1d_sD = nav_shapeD->pix1_d; int p1n_sD = nav_shapeD->pix1_n;
    double p2s_sD = nav_shapeD->pix2_start, p2d_sD = nav_shapeD->pix2_d; int p2n_sD = nav_shapeD->pix2_n;
    // --- RRR count legs: the single random (lens random), slab-hashed on nav_lensR ---
    int *im_Rl = nav_lensR->index_matcher, *pgb_Rl = nav_lensR->pixs_galind_bounds, *pg_Rl = nav_lensR->pix_gals;
    int *so_Rl = nav_lensR->slab_offsets, *rsb_Rl = nav_lensR->rshift_bounds;
    int nslabs_Rl = nav_lensR->nslabs; double z0_Rl = nav_lensR->z0, dpixz_Rl = nav_lensR->dpix_z;
    double p1s_Rl = nav_lensR->pix1_start, p1d_Rl = nav_lensR->pix1_d; int p1n_Rl = nav_lensR->pix1_n;
    double p2s_Rl = nav_lensR->pix2_start, p2d_Rl = nav_lensR->pix2_d; int p2n_Rl = nav_lensR->pix2_n;
    // --- binning + output ---
    int nmax = bin->nmax, nbinsr = bin->nbinsr, dccorr = bin->dccorr;
    double rmin = bin->rmin, rmax = bin->rmax, Pi = bin->Pi;
    double *bin_centers = out->bin_centers;
    double complex *Comp_n = out->npcf, *RRR_n = out->norm_mp;

    int nmp = 2*nmax+1;
    int nnvals_Gn = 2*nmax+5;                  // G blocks, m in [-nmax-2, nmax+2]
    int nnvals_Wn = 2*nmax+1;                  // W blocks, m in [-nmax, nmax]
    int nbinszr_leg = nbinsz_polar*nbinsr;
    int nzcombis = nbinsz_lens*nbinsz_polar*nbinsz_polar;
    int ups_zshift = nbinsr*nbinsr;
    int ups_nshift = ups_zshift*nzcombis;
    int ups_compshift = nmp*ups_nshift;        // one natural component (all n)
    int ncomp_est = 2;                         // DSS, RSS
    int ups_threadshift = ncomp_est*2*ups_compshift;  // 2 est x 2 natural
    int norm_threadshift = nmp*ups_nshift;
    int counts_threadshift = nbinsz_lens*nbinsz_polar*nbinsr;

    double complex *tmpComp = calloc((size_t)nthreads*ups_threadshift, sizeof(double complex));
    double complex *tmpNorm = calloc((size_t)nthreads*norm_threadshift, sizeof(double complex));
    double *tmpwcounts = calloc((size_t)nthreads*counts_threadshift, sizeof(double));
    double *tmpwnorms  = calloc((size_t)nthreads*counts_threadshift, sizeof(double));

    #pragma omp parallel num_threads(nthreads)
    {
        int elthread = omp_get_thread_num();
        double complex *thDSS = tmpComp + (size_t)elthread*ups_threadshift;   // 2 natural comps
        double complex *thRSS = thDSS + 2*ups_compshift;
        double complex *thNorm = tmpNorm + (size_t)elthread*norm_threadshift;
        double *thwc = tmpwcounts + (size_t)elthread*counts_threadshift;
        double *thwn = tmpwnorms  + (size_t)elthread*counts_threadshift;

        // (A) Data-lens centrals -> DSS numerator (+w) via shape-data legs; bin-centers.
        #pragma omp for schedule(dynamic, 256)
        for (int ig=0; ig<ngal_D; ig++){
            double c1 = pos1_D[ig], c2 = pos2_D[ig], c3 = pos3_D[ig], wc = w_D[ig];
            int zc = zbin_D[ig];
            double complex *Gn = calloc(nnvals_Gn*nbinszr_leg, sizeof(double complex));
            double complex *sumG4  = calloc(nbinszr_leg, sizeof(double complex));
            double complex *sumGabs = calloc(nbinszr_leg, sizeof(double complex));
            int *ncounts = calloc(nbinszr_leg, sizeof(int));
            int *allowedr = calloc(nbinszr_leg, sizeof(int));
            int *allowedz = calloc(nbinszr_leg, sizeof(int));
            double *wc_base = thwc + zc*nbinsz_polar*nbinsr;
            double *wn_base = thwn + zc*nbinsz_polar*nbinsr;

            ia_slab_legmultipoles(c1, c2, c3, wc,
                pos1_sD, pos2_sD, pos3_sD, w_sD, zbin_sD, e1_sD, e2_sD,
                nbinsz_polar, nslabs_sD, z0_sD, dpixz_sD, p1s_sD, p1d_sD, p1n_sD,
                p2s_sD, p2d_sD, p2n_sD, so_sD, im_sD, pgb_sD, rsb_sD, pg_sD,
                rmin, rmax, nbinsr, Pi, 0, 0, -nmax-2, nnvals_Gn,
                NULL, Gn, NULL, NULL, sumG4, sumGabs, NULL, NULL, NULL, ncounts, wc_base, wn_base);

            int nallowed = 0;
            for (int z=0; z<nbinsz_polar; z++){
                for (int r=0; r<nbinsr; r++){
                    if (ncounts[z*nbinsr+r] != 0){ allowedr[nallowed]=r; allowedz[nallowed]=z; nallowed++; }
                }
            }
            ngg_accum_upsilon(thDSS, wc, zc, Gn, sumG4, sumGabs, allowedr, allowedz,
                              nallowed, nmax, nbinsr, nbinsz_lens, nbinsz_polar, dccorr);
            free(Gn); free(sumG4); free(sumGabs); free(ncounts); free(allowedr); free(allowedz);
        }

        // (B) Random-lens centrals -> RSS numerator (+w) via shape-data legs, and
        //     the shared RRR count via the random (lens-random) legs.
        #pragma omp for schedule(dynamic, 256)
        for (int ig=0; ig<ngal_Rl; ig++){
            double c1 = pos1_Rl[ig], c2 = pos2_Rl[ig], c3 = pos3_Rl[ig], wc = w_Rl[ig];
            int zc = zbin_Rl[ig];
            // RSS numerator: shape-data G legs, weight +w (raw)
            double complex *Gn = calloc(nnvals_Gn*nbinszr_leg, sizeof(double complex));
            double complex *sumG4  = calloc(nbinszr_leg, sizeof(double complex));
            double complex *sumGabs = calloc(nbinszr_leg, sizeof(double complex));
            int *ncG = calloc(nbinszr_leg, sizeof(int));
            int *arG = calloc(nbinszr_leg, sizeof(int));
            int *azG = calloc(nbinszr_leg, sizeof(int));
            ia_slab_legmultipoles(c1, c2, c3, wc,
                pos1_sD, pos2_sD, pos3_sD, w_sD, zbin_sD, e1_sD, e2_sD,
                nbinsz_polar, nslabs_sD, z0_sD, dpixz_sD, p1s_sD, p1d_sD, p1n_sD,
                p2s_sD, p2d_sD, p2n_sD, so_sD, im_sD, pgb_sD, rsb_sD, pg_sD,
                rmin, rmax, nbinsr, Pi, 0, 0, -nmax-2, nnvals_Gn,
                NULL, Gn, NULL, NULL, sumG4, sumGabs, NULL, NULL, NULL, ncG, NULL, NULL);
            int naG = 0;
            for (int z=0; z<nbinsz_polar; z++){ for (int r=0; r<nbinsr; r++){
                if (ncG[z*nbinsr+r] != 0){ arG[naG]=r; azG[naG]=z; naG++; } } }
            ngg_accum_upsilon(thRSS, wc, zc, Gn, sumG4, sumGabs, arG, azG,
                              naG, nmax, nbinsr, nbinsz_lens, nbinsz_polar, dccorr);
            free(Gn); free(sumG4); free(sumGabs); free(ncG); free(arG); free(azG);

            // RRR normalization: random (lens-random) count legs W^R
            double complex *Wn = calloc(nnvals_Wn*nbinszr_leg, sizeof(double complex));
            double *sumW2 = calloc(nbinszr_leg, sizeof(double));
            int *ncW = calloc(nbinszr_leg, sizeof(int));
            int *arW = calloc(nbinszr_leg, sizeof(int));
            int *azW = calloc(nbinszr_leg, sizeof(int));
            ia_slab_legmultipoles(c1, c2, c3, wc,
                pos1_Rl, pos2_Rl, pos3_Rl, w_Rl, zbin_Rl, NULL, NULL,
                nbinsz_polar, nslabs_Rl, z0_Rl, dpixz_Rl, p1s_Rl, p1d_Rl, p1n_Rl,
                p2s_Rl, p2d_Rl, p2n_Rl, so_Rl, im_Rl, pgb_Rl, rsb_Rl, pg_Rl,
                rmin, rmax, nbinsr, Pi, -nmax, nnvals_Wn, 0, 0,
                Wn, NULL, NULL, sumW2, NULL, NULL, NULL, NULL, NULL, ncW, NULL, NULL);
            int naW = 0;
            for (int z=0; z<nbinsz_polar; z++){ for (int r=0; r<nbinsr; r++){
                if (ncW[z*nbinsr+r] != 0){ arW[naW]=r; azW[naW]=z; naW++; } } }
            ngg_accum_norm(thNorm, wc, zc, Wn, sumW2, arW, azW, naW,
                           nmax, nbinsr, nbinsz_lens, nbinsz_polar, dccorr);
            free(Wn); free(sumW2); free(ncW); free(arW); free(azW);
        }
    }

    // Reduce the numerator components (2 est x 2 natural) and the RRR across threads.
    #pragma omp parallel for num_threads(nthreads)
    for (int i=0; i<ups_compshift; i++){
        for (int t=0; t<nthreads; t++){
            size_t ts = (size_t)t*ups_threadshift;
            for (int c=0; c<ncomp_est*2; c++){
                Comp_n[c*ups_compshift + i] += tmpComp[ts + c*ups_compshift + i];
            }
        }
    }
    #pragma omp parallel for num_threads(nthreads)
    for (int i=0; i<norm_threadshift; i++){
        for (int t=0; t<nthreads; t++){
            RRR_n[i] += tmpNorm[(size_t)t*norm_threadshift + i];
        }
    }

    // Reduce the bin-center weighted sums (data lens x shape-data legs) and finalize.
    double *totcounts = calloc(counts_threadshift, sizeof(double));
    double *totnorms  = calloc(counts_threadshift, sizeof(double));
    for (int t=0; t<nthreads; t++){
        size_t ts = (size_t)t*counts_threadshift;
        for (int i=0; i<counts_threadshift; i++){ totcounts[i]+=tmpwcounts[ts+i]; totnorms[i]+=tmpwnorms[ts+i]; }
    }
    for (int i=0; i<counts_threadshift; i++){
        if (totnorms[i] != 0){ bin_centers[i] = totcounts[i]/totnorms[i]; }
    }

    free(tmpComp); free(tmpNorm); free(tmpwcounts); free(tmpwnorms);
    free(totcounts); free(totnorms);
}

///////////////////////////////////////////
// Slab-hashed polar-polar-polar (GGG)    //
///////////////////////////////////////////
// Slab-hashed polar-polar-polar (GGG) cross-correlator in the projected '3dbox'
// geometry (line-of-sight window |dz| < Pi). Three polar (shape) vertices,
// normalized by RRR (Vedder et al. 2026 Eq.17, S.S.S / RRR). The shape catalog is
// looped as the numerator central and hashed (nav_polar) for the two G-legs; the
// shape random is looped as the RRR central and hashed (nav_R) for the count legs.
// The 4-component numerator algebra (natural components Gamma_0..3) and its
// diagonal self-terms are ported from alloc_Gammans_discrete_ggg; the Python layer
// applies f = W_S/W_R and forms S.S.S / (f^3 RRR). n_cfs = 4, multipoles n in
// [0, nmax]. Component/norm r-index layout matches multipoles2npcf_ggg (Gamma_3 and
// the RRR count are stored transposed in the two r bins).
void alloc_Gammans_slab_GGG(const MultiresoCatalog *cat_polar, const NavHash *nav_polar,
                            const MultiresoCatalog *cat_R, const NavHash *nav_R,
                            const BinningParams *bin, int nthreads, int verbose,
                            NPCFOutput *out){
    // --- polar (spin-2) catalog: looped central + hashed G-legs (nav_polar) ---
    double *pos1_S = cat_polar->pos1_resos, *pos2_S = cat_polar->pos2_resos, *pos3_S = cat_polar->pos3_resos;
    double *w_S = cat_polar->weight_resos, *e1_S = cat_polar->e1_resos, *e2_S = cat_polar->e2_resos;
    int *zbin_S = cat_polar->zbin_resos, nbinsz_polar = cat_polar->nbinsz, ngal_S = cat_polar->ngal_resos[0];
    int *im_S = nav_polar->index_matcher, *pgb_S = nav_polar->pixs_galind_bounds, *pg_S = nav_polar->pix_gals;
    int *so_S = nav_polar->slab_offsets, *rsb_S = nav_polar->rshift_bounds;
    int nslabs_S = nav_polar->nslabs; double z0_S = nav_polar->z0, dpixz_S = nav_polar->dpix_z;
    double p1s_S = nav_polar->pix1_start, p1d_S = nav_polar->pix1_d; int p1n_S = nav_polar->pix1_n;
    double p2s_S = nav_polar->pix2_start, p2d_S = nav_polar->pix2_d; int p2n_S = nav_polar->pix2_n;
    // --- shape random: looped RRR central + hashed count legs (nav_R) ---
    double *pos1_R = cat_R->pos1_resos, *pos2_R = cat_R->pos2_resos, *pos3_R = cat_R->pos3_resos;
    double *w_R = cat_R->weight_resos; int *zbin_R = cat_R->zbin_resos, ngal_R = cat_R->ngal_resos[0];
    int *im_R = nav_R->index_matcher, *pgb_R = nav_R->pixs_galind_bounds, *pg_R = nav_R->pix_gals;
    int *so_R = nav_R->slab_offsets, *rsb_R = nav_R->rshift_bounds;
    int nslabs_R = nav_R->nslabs; double z0_R = nav_R->z0, dpixz_R = nav_R->dpix_z;
    double p1s_R = nav_R->pix1_start, p1d_R = nav_R->pix1_d; int p1n_R = nav_R->pix1_n;
    double p2s_R = nav_R->pix2_start, p2d_R = nav_R->pix2_d; int p2n_R = nav_R->pix2_n;
    // --- binning + output ---
    int nmax = bin->nmax, nbinsr = bin->nbinsr, dccorr = bin->dccorr;
    double rmin = bin->rmin, rmax = bin->rmax, Pi = bin->Pi;
    double *bin_centers = out->bin_centers;
    double complex *Comp_n = out->npcf, *RRR_n = out->norm_mp;

    int nnvals_Gn = 2*nmax+3;                  // G blocks, m in [-nmax-3, nmax-1]
    int nnvals_Wn = nmax+1;                    // W blocks, m in [0, nmax]
    int ncomp = 4;                             // Gamma_0..3
    int nbinszr_leg = nbinsz_polar*nbinsr;
    int nzcombis = nbinsz_polar*nbinsz_polar*nbinsz_polar;
    int comp_zshift = nbinsr*nbinsr;
    int comp_nshift = comp_zshift*nzcombis;
    int comp_size = (nmax+1)*comp_nshift;      // one component (all n)
    int ups_threadshift = ncomp*comp_size;
    int counts_threadshift = nbinsz_polar*nbinsz_polar*nbinsr;

    double complex *tmpComp = calloc((size_t)nthreads*ups_threadshift, sizeof(double complex));
    double complex *tmpRRR  = calloc((size_t)nthreads*comp_size, sizeof(double complex));
    double *tmpwcounts = calloc((size_t)nthreads*counts_threadshift, sizeof(double));
    double *tmpwnorms  = calloc((size_t)nthreads*counts_threadshift, sizeof(double));

    // (A) polar central -> four raw SSS natural components + bin centers.
    #pragma omp parallel num_threads(nthreads)
    {
        int elthread = omp_get_thread_num();
        double complex *thComp = tmpComp + (size_t)elthread*ups_threadshift;
        double complex *G0 = thComp, *G1 = thComp+comp_size, *G2 = thComp+2*comp_size, *G3 = thComp+3*comp_size;
        double *thwc = tmpwcounts + (size_t)elthread*counts_threadshift;
        double *thwn = tmpwnorms  + (size_t)elthread*counts_threadshift;

        #pragma omp for schedule(dynamic, 256)
        for (int ig=0; ig<ngal_S; ig++){
            double c1 = pos1_S[ig], c2 = pos2_S[ig], c3 = pos3_S[ig], cw = w_S[ig];
            int zbin_c = zbin_S[ig];
            double complex wshape = cw*(e1_S[ig] + I*e2_S[ig]);
            double complex *Gn = calloc(nnvals_Gn*nbinszr_leg, sizeof(double complex));
            double complex *sumG6 = calloc(nbinszr_leg, sizeof(double complex));
            double complex *sumG2p = calloc(nbinszr_leg, sizeof(double complex));
            double complex *sumGabsp = calloc(nbinszr_leg, sizeof(double complex));
            int *ncounts = calloc(nbinszr_leg, sizeof(int));
            int *allowedr = calloc(nbinszr_leg, sizeof(int));
            int *allowedz = calloc(nbinszr_leg, sizeof(int));
            double *wc_base = thwc + zbin_c*nbinsz_polar*nbinsr;
            double *wn_base = thwn + zbin_c*nbinsz_polar*nbinsr;

            ia_slab_legmultipoles(c1, c2, c3, cw,
                pos1_S, pos2_S, pos3_S, w_S, zbin_S, e1_S, e2_S,
                nbinsz_polar, nslabs_S, z0_S, dpixz_S, p1s_S, p1d_S, p1n_S,
                p2s_S, p2d_S, p2n_S, so_S, im_S, pgb_S, rsb_S, pg_S,
                rmin, rmax, nbinsr, Pi, 0, 0, -nmax-3, nnvals_Gn,
                NULL, Gn, NULL, NULL, NULL, NULL, sumG6, sumG2p, sumGabsp, ncounts, wc_base, wn_base);

            int nallowed = 0;
            for (int z=0; z<nbinsz_polar; z++){ for (int r=0; r<nbinsr; r++){
                if (ncounts[z*nbinsr+r] != 0){ allowedr[nallowed]=r; allowedz[nallowed]=z; nallowed++; } } }

            for (int thisn=0; thisn<nmax+1; thisn++){
                int nshift = thisn*comp_nshift;
                int blk_nm3 = (nmax+thisn)*nbinszr_leg;    // G_{n-3}
                int blk_nm1 = (nmax+thisn+2)*nbinszr_leg;  // G_{n-1}
                int blk_mnm1 = (nmax-thisn+2)*nbinszr_leg; // G_{-n-1}
                int blk_mnm3 = (nmax-thisn)*nbinszr_leg;   // G_{-n-3}
                for (int a1=0; a1<nallowed; a1++){
                    int elb1 = allowedr[a1], zbin2 = allowedz[a1];
                    int zr1 = zbin2*nbinsr + elb1;
                    double complex h0 = -wshape       * Gn[blk_nm3 + zr1];
                    double complex h1 = -conj(wshape) * Gn[blk_nm1 + zr1];
                    double complex h2 = -wshape       * conj(Gn[blk_mnm1 + zr1]);
                    double complex h3 = -wshape       * conj(Gn[blk_nm1 + zr1]);
                    if (dccorr==1){
                        int zcd = zbin_c*nbinsz_polar*nbinsz_polar + zbin2*nbinsz_polar + zbin2;
                        int gd = nshift + zcd*comp_zshift + elb1*nbinsr + elb1;
                        G0[gd] += wshape       * sumG6[zr1];
                        G1[gd] += conj(wshape) * sumG2p[zr1];
                        G2[gd] += wshape       * sumGabsp[zr1];
                        G3[gd] += wshape       * sumGabsp[zr1];
                    }
                    for (int a2=0; a2<nallowed; a2++){
                        int elb2 = allowedr[a2], zbin3 = allowedz[a2];
                        int zr2 = zbin3*nbinsr + elb2;
                        int zc = zbin_c*nbinsz_polar*nbinsz_polar + zbin2*nbinsz_polar + zbin3;
                        int gs  = nshift + zc*comp_zshift + elb1*nbinsr + elb2;
                        int gst = nshift + zc*comp_zshift + elb2*nbinsr + elb1;
                        G0[gs]  += h0 * Gn[blk_mnm3 + zr2];
                        G1[gs]  += h1 * Gn[blk_mnm1 + zr2];
                        G2[gs]  += h2 * Gn[blk_mnm3 + zr2];
                        G3[gst] += h3 * Gn[blk_nm3 + zr2];
                    }
                }
            }
            free(Gn); free(sumG6); free(sumG2p); free(sumGabsp);
            free(ncounts); free(allowedr); free(allowedz);
        }
    }

    // (B) random central -> shared RRR count (f-free; conj on t1, r-transposed).
    #pragma omp parallel num_threads(nthreads)
    {
        int elthread = omp_get_thread_num();
        double complex *thRRR = tmpRRR + (size_t)elthread*comp_size;

        #pragma omp for schedule(dynamic, 256)
        for (int ig=0; ig<ngal_R; ig++){
            double c1 = pos1_R[ig], c2 = pos2_R[ig], c3 = pos3_R[ig], cw = w_R[ig];
            int zbin_c = zbin_R[ig];
            double complex *Wn = calloc(nnvals_Wn*nbinszr_leg, sizeof(double complex));
            double *sumW2 = calloc(nbinszr_leg, sizeof(double));
            int *ncounts = calloc(nbinszr_leg, sizeof(int));
            int *allowedr = calloc(nbinszr_leg, sizeof(int));
            int *allowedz = calloc(nbinszr_leg, sizeof(int));

            ia_slab_legmultipoles(c1, c2, c3, cw,
                pos1_R, pos2_R, pos3_R, w_R, zbin_R, NULL, NULL,
                nbinsz_polar, nslabs_R, z0_R, dpixz_R, p1s_R, p1d_R, p1n_R,
                p2s_R, p2d_R, p2n_R, so_R, im_R, pgb_R, rsb_R, pg_R,
                rmin, rmax, nbinsr, Pi, 0, nnvals_Wn, 0, 0,
                Wn, NULL, NULL, sumW2, NULL, NULL, NULL, NULL, NULL, ncounts, NULL, NULL);

            int nallowed = 0;
            for (int z=0; z<nbinsz_polar; z++){ for (int r=0; r<nbinsr; r++){
                if (ncounts[z*nbinsr+r] != 0){ allowedr[nallowed]=r; allowedz[nallowed]=z; nallowed++; } } }

            for (int thisn=0; thisn<nmax+1; thisn++){
                int nshift = thisn*comp_nshift;
                int blk = thisn*nbinszr_leg;               // W_n
                for (int a1=0; a1<nallowed; a1++){
                    int elb1 = allowedr[a1], zbin2 = allowedz[a1];
                    int zr1 = zbin2*nbinsr + elb1;
                    double complex w0 = cw * conj(Wn[blk + zr1]);
                    if (dccorr==1){
                        int zcd = zbin_c*nbinsz_polar*nbinsz_polar + zbin2*nbinsz_polar + zbin2;
                        int gd = nshift + zcd*comp_zshift + elb1*nbinsr + elb1;
                        thRRR[gd] -= cw*sumW2[zr1];
                    }
                    for (int a2=0; a2<nallowed; a2++){
                        int elb2 = allowedr[a2], zbin3 = allowedz[a2];
                        int zr2 = zbin3*nbinsr + elb2;
                        int zc = zbin_c*nbinsz_polar*nbinsz_polar + zbin2*nbinsz_polar + zbin3;
                        int gst = nshift + zc*comp_zshift + elb2*nbinsr + elb1;
                        thRRR[gst] += w0 * Wn[blk + zr2];
                    }
                }
            }
            free(Wn); free(sumW2); free(ncounts); free(allowedr); free(allowedz);
        }
    }

    // Reduce the components and the RRR count across threads.
    #pragma omp parallel for num_threads(nthreads)
    for (int i=0; i<comp_size; i++){
        for (int t=0; t<nthreads; t++){
            for (int c=0; c<ncomp; c++){
                Comp_n[c*comp_size + i] += tmpComp[(size_t)t*ups_threadshift + c*comp_size + i];
            }
            RRR_n[i] += tmpRRR[(size_t)t*comp_size + i];
        }
    }

    // Reduce the bin-center weighted sums (polar central x polar-data legs) and finalize.
    double *totcounts = calloc(counts_threadshift, sizeof(double));
    double *totnorms  = calloc(counts_threadshift, sizeof(double));
    for (int t=0; t<nthreads; t++){
        size_t ts = (size_t)t*counts_threadshift;
        for (int i=0; i<counts_threadshift; i++){ totcounts[i]+=tmpwcounts[ts+i]; totnorms[i]+=tmpwnorms[ts+i]; }
    }
    for (int i=0; i<counts_threadshift; i++){
        if (totnorms[i] != 0){ bin_centers[i] = totcounts[i]/totnorms[i]; }
    }

    free(tmpComp); free(tmpRRR); free(tmpwcounts); free(tmpwnorms);
    free(totcounts); free(totnorms);
}

// DoubleTree based estimtor of Source-Lens-Lens (G3L) Correlator
void alloc_Gammans_doubletree_GNN(const MultiresoCatalog *cat_source, const NavHash *nav_source,
                                  const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                                  const TreeResoParams *tree, const BinningParams *bin,
                                  int nthreads, int verbose, NPCFOutput *out){
    // --- tree parameters (full leaf params) ---
    int nresos = tree->nresos, nresos_grid = tree->nresos_grid;
    double *dpix1_resos = tree->dpix1_resos, *dpix2_resos = tree->dpix2_resos, *reso_redges = tree->reso_redges;
    int resoshift_leafs = tree->resoshift_leafs, minresoind_leaf = tree->minresoind_leaf, maxresoind_leaf = tree->maxresoind_leaf;
    // --- multi-resolution shape (source) central + scalar lens catalogs ---
    double *isinner_source_resos = cat_source->isinner_resos, *w_source_resos = cat_source->weight_resos;
    double *pos1_source_resos = cat_source->pos1_resos, *pos2_source_resos = cat_source->pos2_resos;
    double *e1_source_resos = cat_source->e1_resos, *e2_source_resos = cat_source->e2_resos;
    int *zbin_source_resos = cat_source->zbin_resos, *ngal_source_resos = cat_source->ngal_resos, nbinsz_source = cat_source->nbinsz;
    double *isinner_lens_resos = cat_lens->isinner_resos, *w_lens_resos = cat_lens->weight_resos;
    double *pos1_lens_resos = cat_lens->pos1_resos, *pos2_lens_resos = cat_lens->pos2_resos;
    int *zbin_lens_resos = cat_lens->zbin_resos, *ngal_lens_resos = cat_lens->ngal_resos, nbinsz_lens = cat_lens->nbinsz;
    // --- binning ---
    int nmax = bin->nmax, nbinsr = bin->nbinsr, dccorr = bin->dccorr;
    double rmin = bin->rmin, rmax = bin->rmax;
    // --- navigation (source carries the occupied-region list; shared pix grid) ---
    int *index_matcher_source = nav_source->index_matcher, *pixs_galind_bounds_source = nav_source->pixs_galind_bounds, *pix_gals_source = nav_source->pix_gals;
    int *index_matcher_lens = nav_lens->index_matcher, *pixs_galind_bounds_lens = nav_lens->pixs_galind_bounds, *pix_gals_lens = nav_lens->pix_gals;
    int *index_matcher_hash = nav_source->index_matcher_hash, nregions = nav_source->nregions;
    double pix1_start = nav_lens->pix1_start, pix1_d = nav_lens->pix1_d; int pix1_n = nav_lens->pix1_n;
    double pix2_start = nav_lens->pix2_start, pix2_d = nav_lens->pix2_d; int pix2_n = nav_lens->pix2_n;
    double *bin_centers = out->bin_centers;
    double complex *Upsilon_n = out->npcf, *Norm_n = out->norm_mp;

    int _upsilonzshift = nbinsr*nbinsr;
    int _nzcombis = nbinsz_source*nbinsz_lens*nbinsz_lens;
    int _upsilonnshift = _upsilonzshift*_nzcombis;
    int _upsilonthreadshift = (nmax+1)*_upsilonnshift;
    
    double *tmpwcounts = calloc(nthreads*nbinsz_source*nbinsz_lens*nbinsr, sizeof(double));
    double *tmpwnorms  = calloc(nthreads*nbinsz_source*nbinsz_lens*nbinsr, sizeof(double));
    double *totcounts = calloc(nbinsz_source*nbinsz_lens*nbinsr, sizeof(double));
    double *totnorms  = calloc(nbinsz_source*nbinsz_lens*nbinsr, sizeof(double));
    // Temporary arrays that are allocated in parallel and later reduced
    // Shape of tmpUpsilon ~ (nthreads, nnvals, nz_source, nz_lens, nz_lens, nbinsr, nbinsr)
    double complex *tmpUpsilon = calloc(nthreads*_upsilonthreadshift, sizeof(double complex));
    double complex *tmpNorm = calloc(nthreads*_upsilonthreadshift, sizeof(double complex));
    #pragma omp parallel for num_threads(nthreads)
    for(int elthread=0;elthread<nthreads;elthread++){
        int nregions_per_thread = nregions/nthreads;
        int hasdiscrete = nresos-nresos_grid;
        int nnvals_Gn = nmax+3; // Need [-1, ..., nmax+1]
        int nnvals_Wn = nmax+1; // Need [0, ..., nmax]
        int nnvals_Ups = nmax+1;
        int nzcombis = nbinsz_source*nbinsz_lens*nbinsz_lens;
        int upsilon_zshift = nbinsr*nbinsr;
        int upsilon_nshift = upsilon_zshift*nzcombis;
        int upsilon_threadshift = nnvals_Ups*upsilon_nshift;
        int threadshift_counts = elthread*nbinsz_source*nbinsz_lens*nbinsr;
        double drbin = log(rmax/rmin)/nbinsr;
        
        // Find largest possible nshift
        int size_max_nshift = 0;
        int size_max_nshift_theo = (int) ((1+hasdiscrete+0.34)*nbinsz_lens*nbinsz_source*nbinsr*pow(4,nresos_grid-1));
        for (int elregion=0; elregion<nregions; elregion++){
            // Check if this thread is responsible for the region
            int nthread_target = mymin(elregion/nregions_per_thread, nthreads-1);
            if (nthread_target!=elthread){continue;}

            // Probe the region's cache-slot count from the source (central)
            // catalog; only the source hash + shift layout is needed here.
            int npix_hash = pix1_n*pix2_n;
            int *rshift_index_matcher_source = calloc(nresos, sizeof(int));
            int *rshift_pixs_galind_bounds_source = calloc(nresos, sizeof(int));
            int *rshift_pix_gals_source = calloc(nresos, sizeof(int));
            build_rshift_offsets(nresos, npix_hash, ngal_source_resos,
                rshift_index_matcher_source, rshift_pixs_galind_bounds_source, rshift_pix_gals_source);

            int *matchers_resoshift = calloc(nresos_grid+1, sizeof(int));
            int *ngal_in_pix = calloc(nresos*nbinsz_source, sizeof(int));
            build_region_galinpix(nresos, nresos_grid, hasdiscrete, elregion,
                pixs_galind_bounds_source, rshift_pixs_galind_bounds_source,
                pix_gals_source, rshift_pix_gals_source, zbin_source_resos,
                matchers_resoshift, ngal_in_pix);

            int *cumresoshift_z = calloc(nbinsz_source*(nresos+1), sizeof(int));
            int *thetashifts_z = calloc(nbinsz_source, sizeof(int));
            int *zbinshifts = calloc(nbinsz_source+1, sizeof(int));
            int zbin2shift, nshift;
            setup_region_shifts(nbinsz_source, nbinsz_lens, nresos, hasdiscrete, nbinsr,
                ngal_in_pix, cumresoshift_z, thetashifts_z, zbinshifts, &zbin2shift, &nshift);
            size_max_nshift = mymax(nshift, size_max_nshift);
            free(rshift_index_matcher_source);
            free(rshift_pixs_galind_bounds_source);
            free(rshift_pix_gals_source);
            free(matchers_resoshift);
            free(ngal_in_pix);
            free(cumresoshift_z);
            free(thetashifts_z);
            free(zbinshifts);
        }
        if (verbose>1){printf("Thread %i: nshift=%i, nshift_theo=%i",elthread,size_max_nshift,size_max_nshift_theo);}
            
        // Largest possible nshift: each zbin does completely fill out the lowest reso grid.
        // The remaining grids then have 1/4 + 1/16 + ... --> 0.33.... times the data of the largest grid. 
        // Now allocate the caches
        //int size_max_nshift = (int) ((1+hasdiscrete+0.34)*((float)mymax(nbinsz_lens,nbinsz_source))*nbinsr*pow(4,nresos_grid));
        double complex *Gncache = calloc(nnvals_Gn*size_max_nshift, sizeof(double complex));
        double complex *wGncache = calloc(nnvals_Gn*size_max_nshift, sizeof(double complex));
        double complex *cwGncache = calloc(nnvals_Gn*size_max_nshift, sizeof(double complex));
        double complex *Wncache = calloc(nnvals_Wn*size_max_nshift, sizeof(double complex));
        double complex *wWncache = calloc(nnvals_Wn*size_max_nshift, sizeof(double complex));
        int *Wncache_updates = calloc(size_max_nshift, sizeof(int));
        for (int elregion=0; elregion<nregions; elregion++){
            int region_debug=-1;
            // Check if this thread is responsible for the region
            int nthread_target = mymin(elregion/nregions_per_thread, nthreads-1);
            if (nthread_target!=elthread){continue;}
            bool printregdbg = (verbose>0) && (elregion==region_debug);
            bool printregdbg2 = (verbose>1) && (elregion==region_debug);
            if ((verbose>0) && (elthread==nthreads/2)){
                printf("\rDone %.2f per cent",100*((double) elregion-nregions_per_thread*(int)(nthreads/2))/nregions_per_thread);
            }
            
            // Check which sets of radii are evaluated for each resolution
            int *reso_rindedges = calloc(nresos+1, sizeof(int));
            double logrmin = log(rmin);
            build_reso_rindedges(nresos, reso_redges, rmin, rmax, nbinsr, reso_rindedges);

            // Shift variables for spatial hash of sources and lenses
            int npix_hash = pix1_n*pix2_n;
            int *rshift_index_matcher_source = calloc(nresos, sizeof(int));
            int *rshift_pixs_galind_bounds_source = calloc(nresos, sizeof(int));
            int *rshift_pix_gals_source = calloc(nresos, sizeof(int));
            build_rshift_offsets(nresos, npix_hash, ngal_source_resos,
                rshift_index_matcher_source, rshift_pixs_galind_bounds_source, rshift_pix_gals_source);
            int *rshift_index_matcher_lens = calloc(nresos, sizeof(int));
            int *rshift_pixs_galind_bounds_lens = calloc(nresos, sizeof(int));
            int *rshift_pix_gals_lens = calloc(nresos, sizeof(int));
            build_rshift_offsets(nresos, npix_hash, ngal_lens_resos,
                rshift_index_matcher_lens, rshift_pixs_galind_bounds_lens, rshift_pix_gals_lens);

            // Region layout of the source (central) catalog: per-(zbin, reso)
            // counts, reduced-grid offsets, pixel -> reduced-pixel matcher.
            int lower1, upper1, lower2, upper2;
            int *matchers_resoshift = calloc(nresos_grid+1, sizeof(int));
            int *ngal_in_pix = calloc(nresos*nbinsz_source, sizeof(int));
            int len_matcher = build_region_galinpix(nresos, nresos_grid, hasdiscrete,
                elregion, pixs_galind_bounds_source, rshift_pixs_galind_bounds_source,
                pix_gals_source, rshift_pix_gals_source, zbin_source_resos,
                matchers_resoshift, ngal_in_pix);
            double hashpix_start1, hashpix_start2;
            int *pix2redpix = calloc(nbinsz_source*len_matcher, sizeof(int));
            build_region_pix2redpix(nresos_grid, hasdiscrete, elregion, nbinsz_source,
                index_matcher_hash, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d,
                pixs_galind_bounds_source, rshift_pixs_galind_bounds_source,
                pix_gals_source, rshift_pix_gals_source, zbin_source_resos,
                pos1_source_resos, pos2_source_resos, dpix1_resos, dpix2_resos,
                matchers_resoshift, len_matcher, &hashpix_start1, &hashpix_start2, pix2redpix);

            // Setup all shift variables for the Gncache in the region
            // Gncache has structure
            // n --> zbin_lens --> zbin_source --> radius
            //   --> [ [0]*ngal_zbin1_reso1 | [0]*ngal_zbin1_reso1/2 | ... | [0]*ngal_zbin1_reson ]
            int *cumresoshift_z = calloc(nbinsz_source*(nresos+1), sizeof(int));
            int *thetashifts_z = calloc(nbinsz_source, sizeof(int));
            int *zbinshifts = calloc(nbinsz_source+1, sizeof(int));
            int zbin2shift, nshift;
            setup_region_shifts(nbinsz_source, nbinsz_lens, nresos, hasdiscrete, nbinsr,
                ngal_in_pix, cumresoshift_z, thetashifts_z, zbinshifts, &zbin2shift, &nshift);
            // Set all the cache indices that are updated in this region to zero
            //if ((elregion==region_debug)){printf("zbin2shift=%d: nshift=%d: size_max_nshift=%d \n", zbin2shift, nshift, size_max_nshift);}
            for (int _i=0; _i<nnvals_Gn*nshift; _i++){Gncache[_i] = 0; wGncache[_i] = 0; cwGncache[_i] = 0;}
            for (int _i=0; _i<nnvals_Wn*nshift; _i++){ Wncache[_i] = 0; wWncache[_i] = 0;}
            for (int _i=0; _i<nshift; _i++){ Wncache_updates[_i] = 0;}
            int Wncache_totupdates=0;
            
            
            // Now, for each resolution, loop over all the galaxies in the region and
            // allocate the Gn & Nn, as well as their caches for the corresponding 
            // set of radii
            // For elreso in resos
            //.  for gal in reso 
            //.    allocate Gn for allowed radii
            //.    allocate the Gncaches
            //.    compute the Upsilon for all combinations of the same resolution
            int ind_pix1, ind_pix2, ind_inpix1, ind_inpix2, ind_red, ind_gal1, ind_gal2, z_gal1, z_gal2;
            int ind_Gncacheshift, ind_Wncacheshift;
            int nbinszr_reso;
            double innergal, pos1_gal1, pos2_gal1, pos1_gal2, pos2_gal2, w_gal1, w_gal2, e1_gal1, e2_gal1;
            double rel1, rel2, dist;
            double complex wshape_gal1;
            double complex nphirot, phirot, phirotc;
            double rmin_reso, rmax_reso, rmin_reso_sq, rmax_reso_sq;
            int elreso_leaf, rbinmin, rbinmax, rbinmin1, rbinmax1, rbinmin2, rbinmax2;
            
            for (int elreso=0;elreso<nresos;elreso++){
                
                elreso_leaf = mymin(mymax(minresoind_leaf,elreso+resoshift_leafs),maxresoind_leaf);
                //elreso_leaf = elreso;
                rbinmin = reso_rindedges[elreso];
                rbinmax = reso_rindedges[elreso+1];
                rmin_reso = rmin*exp(rbinmin*drbin);
                rmax_reso = rmin*exp(rbinmax*drbin);
                rmin_reso_sq = rmin_reso*rmin_reso;
                rmax_reso_sq = rmax_reso*rmax_reso;
                int nbinsr_reso = rbinmax-rbinmin;
                nbinszr_reso = nbinsz_lens*nbinsr_reso;
                lower1 = pixs_galind_bounds_source[rshift_pixs_galind_bounds_source[elreso]+elregion];
                upper1 = pixs_galind_bounds_source[rshift_pixs_galind_bounds_source[elreso]+elregion+1];
                double complex *thisWns =  calloc(nnvals_Gn*nbinszr_reso, sizeof(double complex));
                double complex *thisG2ns =  calloc(nbinszr_reso, sizeof(double complex));
                double complex *thisW2ns =  calloc(nbinszr_reso, sizeof(double complex));
                int *nextncounts = calloc(nbinszr_reso, sizeof(int));
                int *allowedrinds = calloc(nbinszr_reso, sizeof(int));
                int *allowedzinds = calloc(nbinszr_reso, sizeof(int));
                //if (elregion==region_debug){printf("rbinmin=%d, rbinmax%d\n",rbinmin,rbinmax);}
                int ind_Wn, ind_counts, z1shift, z2rshift, rbin;
                for (ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                    ind_gal1 = rshift_pix_gals_source[elreso] + pix_gals_source[rshift_pix_gals_source[elreso]+ind_inpix1];
                    innergal = isinner_source_resos[ind_gal1];
                    if (innergal<1e-50){continue;}
                    z_gal1 = zbin_source_resos[ind_gal1];
                    pos1_gal1 = pos1_source_resos[ind_gal1];
                    pos2_gal1 = pos2_source_resos[ind_gal1];
                    w_gal1 = innergal*w_source_resos[ind_gal1];
                    e1_gal1 = e1_source_resos[ind_gal1];
                    e2_gal1 = e2_source_resos[ind_gal1];
                    z1shift = z_gal1*nbinsz_lens*nbinsr;
                    wshape_gal1 = (double complex) w_gal1 * (e1_gal1+I*e2_gal1);
                    
                    int pix1_lower = mymax(0, (int) floor((pos1_gal1 - (rmax_reso+pix1_d) - pix1_start)/pix1_d));
                    int pix2_lower = mymax(0, (int) floor((pos2_gal1 - (rmax_reso+pix2_d) - pix2_start)/pix2_d));
                    int pix1_upper = mymin(pix1_n-1, (int) floor((pos1_gal1 + (rmax_reso+pix1_d) - pix1_start)/pix1_d));
                    int pix2_upper = mymin(pix2_n-1, (int) floor((pos2_gal1 + (rmax_reso+pix2_d) - pix2_start)/pix2_d));
                    for (ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                        for (ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                            ind_red = index_matcher_lens[rshift_index_matcher_lens[elreso_leaf] + ind_pix2*pix1_n + ind_pix1];
                            if (ind_red==-1){continue;}
                            lower2 = pixs_galind_bounds_lens[rshift_pixs_galind_bounds_lens[elreso_leaf]+ind_red];
                            upper2 = pixs_galind_bounds_lens[rshift_pixs_galind_bounds_lens[elreso_leaf]+ind_red+1];
                            for (ind_inpix2=lower2; ind_inpix2<upper2; ind_inpix2++){
                                ind_gal2 = rshift_pix_gals_lens[elreso_leaf] + pix_gals_lens[rshift_pix_gals_lens[elreso_leaf]+ind_inpix2];
                                
                                pos1_gal2 = pos1_lens_resos[ind_gal2];
                                pos2_gal2 = pos2_lens_resos[ind_gal2];
                                w_gal2 = w_lens_resos[ind_gal2];
                                z_gal2 = zbin_lens_resos[ind_gal2];
                                
                                rel1 = pos1_gal2 - pos1_gal1;
                                rel2 = pos2_gal2 - pos2_gal1;
                                dist = rel1*rel1 + rel2*rel2;
                                if(dist < rmin_reso_sq || dist >= rmax_reso_sq) continue;
                                dist = sqrt(dist);
                                rbin = (int) floor((log(dist)-logrmin)/drbin);

                                z2rshift = z_gal2*nbinsr_reso + rbin - rbinmin;
                                ind_counts = threadshift_counts + z1shift + z_gal2*nbinsr + rbin;
                                
                                phirot = (rel1+I*rel2)/dist;
                                phirotc = conj(phirot);
                                nextncounts[z2rshift] += 1;
                                tmpwcounts[ind_counts] += w_gal1*w_gal2*dist; 
                                tmpwnorms[ind_counts] += w_gal1*w_gal2; 
                                thisG2ns[z2rshift] += wshape_gal1*w_gal2*w_gal2*phirotc*phirotc;
                                thisW2ns[z2rshift] += w_gal1*w_gal2*w_gal2;
                                
                                ind_Wn = z2rshift;
                                nphirot = phirotc;
                                for (int nextn=-1;nextn<=nmax+1;nextn++){
                                    thisWns[ind_Wn] += w_gal2*nphirot;
                                    nphirot *= phirot; 
                                    ind_Wn += nbinszr_reso;
                                }
                            }
                        }
                    }
                    // Update the Gncache and Gnnormcache
                    // Gncache in range [-1, .., nmax+1]
                    // Nncache in range [0, ..., nmax]
                    int red_reso2, npix_side_reso2, elhashpix_1_reso2, elhashpix_2_reso2, elhashpix_reso2, redpix_reso2;
                    double complex thisGn, thisNn;
                    int _tmpindcache, _tmpindGn, zrshift;
                    for (int elreso2=elreso; elreso2<nresos; elreso2++){
                        red_reso2 = elreso2 - hasdiscrete;
                        if (hasdiscrete==1 && elreso==0 && elreso2==0){red_reso2 += hasdiscrete;}
                        npix_side_reso2 = 1 << (nresos_grid-red_reso2-1);
                        elhashpix_1_reso2 = (int) floor((pos1_gal1 - hashpix_start1)/dpix1_resos[red_reso2]);
                        elhashpix_2_reso2 = (int) floor((pos2_gal1 - hashpix_start2)/dpix2_resos[red_reso2]);
                        elhashpix_reso2 = elhashpix_2_reso2*npix_side_reso2 + elhashpix_1_reso2;
                        redpix_reso2 = pix2redpix[z_gal1*len_matcher+matchers_resoshift[red_reso2]+elhashpix_reso2];
                        for (int zbin2=0; zbin2<nbinsz_lens; zbin2++){
                            if (printregdbg2){
                                printf("Gnupdates for reso1=%d reso2=%d red_reso2=%d, galindex=%d, z1=%d, z2=%d:%d radial updates; shiftstart %d = %d+%d+%d+%d+%d \n"
                                       ,elreso,elreso2,red_reso2,ind_gal1,z_gal1,zbin2,rbinmax-rbinmin,
                                       zbin2*zbin2shift + zbinshifts[z_gal1] + rbinmin*thetashifts_z[z_gal1] + 
                                       cumresoshift_z[z_gal1*(nresos+1) + elreso2] + redpix_reso2,
                                       zbin2*zbin2shift, zbinshifts[z_gal1], rbinmin*thetashifts_z[z_gal1],
                                       cumresoshift_z[z_gal1*(nresos+1) + elreso2], redpix_reso2);
                            }
                            for (int thisrbin=rbinmin; thisrbin<rbinmax; thisrbin++){
                                zrshift = zbin2*nbinsr_reso + thisrbin-rbinmin;
                                if (cabs(thisWns[nbinszr_reso+zrshift])<1e-10){continue;}
                                ind_Gncacheshift = zbin2*zbin2shift + zbinshifts[z_gal1] + thisrbin*thetashifts_z[z_gal1] + 
                                    cumresoshift_z[z_gal1*(nresos+1) + elreso2] + redpix_reso2;
                                _tmpindGn = zrshift;
                                _tmpindcache = ind_Gncacheshift;
                                for(int thisn=0; thisn<nnvals_Gn; thisn++){
                                    thisGn = thisWns[_tmpindGn];
                                    Gncache[_tmpindcache] += thisGn;
                                    wGncache[_tmpindcache] += wshape_gal1*thisGn;
                                    cwGncache[_tmpindcache] += conj(wshape_gal1)*thisGn;
                                    _tmpindGn += nbinszr_reso;
                                    _tmpindcache += nshift;
                                }
                                _tmpindGn = zrshift+nbinszr_reso;
                                _tmpindcache = ind_Gncacheshift;
                                for(int thisn=0; thisn<nnvals_Wn; thisn++){
                                    thisNn = thisWns[_tmpindGn];
                                    Wncache[_tmpindcache] += thisNn;
                                    wWncache[_tmpindcache] += w_gal1*thisNn;
                                    _tmpindGn += nbinszr_reso;
                                    _tmpindcache += nshift;
                                }
                                Wncache_updates[ind_Gncacheshift] += 1;
                                Wncache_totupdates += 1;
                            }
                            
                        } 
                    }
                    
                    // Allocate same reso Upsilon
                    // First check for zero count bins (most likely only in discrete-discrete bit)
                    int nallowedcounts = 0;
                    for (int zbin1=0; zbin1<nbinsz_lens; zbin1++){
                        for (int elb1=0; elb1<nbinsr_reso; elb1++){
                            zrshift = zbin1*nbinsr_reso + elb1;
                            if (nextncounts[zbin1*nbinsr_reso + elb1] != 0){
                                allowedrinds[nallowedcounts] = elb1;
                                allowedzinds[nallowedcounts] = zbin1;
                                nallowedcounts += 1;
                            }
                        }
                    }
                    // Now update the Upsilon_n
                    // tmpUpsilon have shape (nthreads, nmax+1, nz_source, nz_lens, nz_lens, nbinsr, nbinsr)
                    // Gns have shape (nmax+3, nbinsz_lens, nbinsr)
                    // Upsilon(thet1, thet2) ~ - we * W_{n-1}(thet1) * conj(W_{n+1})(thet2) + delta^K_{thet1,thet2} * (we * w*w*exp(-2phi))
                    for (int thisn=0; thisn<nmax+1; thisn++){
                        int elb1_full, elb2_full, _gammashift, gammashift;
                        int _wind, zrshift, _zcombi, zcombi, elb1, zbin2, elb2, zbin3;
                        double complex nextUps, nextN;
                        int thisnshift = elthread*upsilon_threadshift + thisn*upsilon_nshift;
                        for (int zrcombis1=0; zrcombis1<nallowedcounts; zrcombis1++){
                            elb1 = allowedrinds[zrcombis1];
                            zbin2 = allowedzinds[zrcombis1];
                            elb1_full = elb1 + rbinmin;
                            zrshift = zbin2*nbinsr_reso + elb1;
                            // Double counting correction
                            if (dccorr==1){
                                zcombi = z_gal1*nbinsz_lens*nbinsz_lens + zbin2*nbinsz_lens + zbin2;
                                gammashift = thisnshift + zcombi*upsilon_zshift + elb1_full*nbinsr+elb1_full;
                                tmpUpsilon[gammashift] += thisG2ns[zrshift];
                                tmpNorm[gammashift] -= thisW2ns[zrshift];  
                            }
                            _zcombi = z_gal1*nbinsz_lens*nbinsz_lens + zbin2*nbinsz_lens;
                            _wind = (thisn+1)*nbinszr_reso+zrshift;
                            _gammashift = thisnshift + elb1_full*nbinsr;
                            nextUps = -wshape_gal1*thisWns[_wind-nbinszr_reso];
                            nextN = w_gal1*thisWns[_wind];
                            for (int zrcombis2=0; zrcombis2<nallowedcounts; zrcombis2++){
                                elb2 = allowedrinds[zrcombis2];
                                zbin3 = allowedzinds[zrcombis2];
                                elb2_full = elb2 + rbinmin;
                                zcombi = _zcombi + zbin3;
                                gammashift = _gammashift + zcombi*upsilon_zshift + elb2_full;
                                _wind = (thisn+1)*nbinszr_reso + zbin3*nbinsr_reso + elb2;
                                tmpUpsilon[gammashift] += nextUps*conj(thisWns[_wind+nbinszr_reso]);
                                tmpNorm[gammashift] += nextN*conj(thisWns[_wind]);
                            }
                        }
                    }
                    
                    for (int _i=0;_i<nnvals_Gn*nbinszr_reso;_i++){thisWns[_i]=0;}
                    for (int _i=0;_i<nbinszr_reso;_i++){thisG2ns[_i]=0;}
                    for (int _i=0;_i<nbinszr_reso;_i++){
                        thisW2ns[_i]=0; nextncounts[_i]=0; allowedrinds[_i]=0; allowedzinds[_i]=0;}
                }
                free(thisWns);
                free(thisG2ns);
                free(thisW2ns);
                free(nextncounts);
                free(allowedrinds);
                free(allowedzinds);
            }
            
            
            // Allocate the Upsilon/Norms for different grid resolutions from all the cached arrays 
            //
            // Note that for different configurations of the resolutions we do the Gamman
            // allocation as follows - see eq. (xx) in yyy.zzz for the reasoning:
            // * Upsilon = -wshape * W_nm1 * conj(W_np1)
            //          --> -(wW_nm1) * conj(W_np1)    if reso1 < reso2
            //          --> - W_nm1   * conj(cwW_np1)  if reso1 > reso2
            // * Norm   =  w * W_n * conj(W_n)
            //          --> wW_n * conj(W_n)  if reso1 < reso2
            //          --> W_n  * conj(wW_n) if reso1 > reso2
            // where wW_xxx := w(shape)*W_xxx and cwG_xxx := conj(w(shape))*G_xxx
            double complex nextUps, nextN;
            int zcombi;
            for (int thisn=0; thisn<nmax+1; thisn++){
                int _upsshift;
                int thisnshift = elthread*upsilon_threadshift + thisn*upsilon_nshift;
                for (int zbin1=0; zbin1<nbinsz_source; zbin1++){
                    for (int zbin2=0; zbin2<nbinsz_lens; zbin2++){
                        for (int zbin3=0; zbin3<nbinsz_lens; zbin3++){
                            zcombi = zbin1*nbinsz_lens*nbinsz_lens + zbin2*nbinsz_lens + zbin3;
                            int _thetashift_z = thetashifts_z[zbin1]; // This is basically shift for theta_i --> theta_{i+1}
                            //if (zcombis_allowed[zcombi]==0){continue;}
                            
                            // Case max(reso1, reso2) = reso2
                            for (int thisreso1=0; thisreso1<nresos; thisreso1++){
                                rbinmin1 = reso_rindedges[thisreso1];
                                rbinmax1 = reso_rindedges[thisreso1+1];
                                for (int thisreso2=thisreso1+1; thisreso2<nresos; thisreso2++){
                                    rbinmin2 = reso_rindedges[thisreso2];
                                    rbinmax2 = reso_rindedges[thisreso2+1];
                                    for (int elgal=0; elgal<ngal_in_pix[zbin1*nresos+thisreso2]; elgal++){
                                        for (int elb1=rbinmin1; elb1<rbinmax1; elb1++){
                                            // n --> zbin2 --> zbin1 --> radius --> [ [0]*ngal_zbin1_reso1 | ... |
                                            //                                        [0]*ngal_zbin1_reson ]
                                            ind_Wncacheshift = zbin2*zbin2shift + zbinshifts[zbin1] + elb1*thetashifts_z[zbin1] 
                                                + cumresoshift_z[zbin1*(nresos+1) + thisreso2] + elgal;
                                            nextUps = -wGncache[(thisn+0)*nshift+ind_Wncacheshift];
                                            nextN = wWncache[thisn*nshift+ind_Wncacheshift];
                                            _upsshift = thisnshift + zcombi*upsilon_zshift + elb1*nbinsr;
                                            ind_Wncacheshift = zbin3*zbin2shift + zbinshifts[zbin1] + 
                                                rbinmin2*thetashifts_z[zbin1] + cumresoshift_z[zbin1*(nresos+1) + thisreso2] + 
                                                elgal;
                                            for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                                tmpUpsilon[_upsshift+elb2] += nextUps*conj(Gncache[(thisn+2)*nshift+ind_Wncacheshift]);
                                                tmpNorm[_upsshift+elb2] += nextN*conj(Wncache[thisn*nshift+ind_Wncacheshift]);
                                                ind_Wncacheshift += _thetashift_z;
                                                ind_Gncacheshift += _thetashift_z;
                                            }
                                        }
                                    }
                                }
                            }
                            
                            // Case max(reso1, reso2) = reso1
                            for (int thisreso2=0; thisreso2<nresos; thisreso2++){
                                rbinmin2 = reso_rindedges[thisreso2];
                                rbinmax2 = reso_rindedges[thisreso2+1];
                                for (int thisreso1=thisreso2+1; thisreso1<nresos; thisreso1++){
                                    rbinmin1 = reso_rindedges[thisreso1];
                                    rbinmax1 = reso_rindedges[thisreso1+1];
                                    for (int elgal=0; elgal<ngal_in_pix[zbin1*nresos+thisreso1]; elgal++){
                                        for (int elb1=rbinmin1; elb1<rbinmax1; elb1++){
                                            ind_Wncacheshift = zbin2*zbin2shift + zbinshifts[zbin1] + elb1*thetashifts_z[zbin1]
                                                + cumresoshift_z[zbin1*(nresos+1) + thisreso1] + elgal;
                                            nextUps = -Gncache[(thisn+0)*nshift+ind_Wncacheshift];
                                            nextN = Wncache[thisn*nshift+ind_Wncacheshift];
                                            _upsshift = thisnshift + zcombi*upsilon_zshift + elb1*nbinsr;
                                            ind_Wncacheshift = zbin3*zbin2shift + zbinshifts[zbin1] +
                                                rbinmin2*thetashifts_z[zbin1] +
                                                cumresoshift_z[zbin1*(nresos+1) + thisreso1] + elgal;
                                            for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                                tmpUpsilon[_upsshift+elb2] += nextUps*conj(cwGncache[(thisn+2)*nshift+ind_Wncacheshift]);
                                                tmpNorm[_upsshift+elb2] += nextN*conj(wWncache[thisn*nshift+ind_Wncacheshift]);
                                                ind_Wncacheshift += _thetashift_z;
                                                ind_Gncacheshift += _thetashift_z;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            free(reso_rindedges);
            free(rshift_index_matcher_source);
            free(rshift_pixs_galind_bounds_source);
            free(rshift_pix_gals_source);
            free(rshift_index_matcher_lens);
            free(rshift_pixs_galind_bounds_lens);
            free(rshift_pix_gals_lens);
            free(matchers_resoshift);
            free(ngal_in_pix);
            free(pix2redpix);  
            free(cumresoshift_z);
            free(thetashifts_z);
            free(zbinshifts);
        }
        free(Gncache);
        free(wGncache);
        free(cwGncache);
        free(Wncache);
        free(wWncache);
    }
    
    // Accumulate the Upsilon_n / N_n
    #pragma omp parallel for num_threads(nthreads)
    for (int thisn=0; thisn<nmax+1; thisn++){
        int iUps;
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            int thisthreadshift = thisthread*_upsilonthreadshift;
            for (int zcombi=0; zcombi<_nzcombis; zcombi++){
                for (int elb1=0; elb1<nbinsr; elb1++){
                    for (int elb2=0; elb2<nbinsr; elb2++){
                        iUps = thisn*_upsilonnshift + zcombi*_upsilonzshift + elb1*nbinsr + elb2;
                        Upsilon_n[iUps] += tmpUpsilon[thisthreadshift+iUps];
                        Norm_n[iUps] += tmpNorm[thisthreadshift+iUps];
                    }
                }
            }
        }
    }
    
    // Accumulate the bin distances and weights
    for (int thisthread=0; thisthread<nthreads; thisthread++){
        int tmpind;
        int thisthreadshift = thisthread*nbinsz_source*nbinsz_lens*nbinsr; 
        for (int elbinz=0; elbinz<nbinsz_source*nbinsz_lens; elbinz++){
            for (int elbinr=0; elbinr<nbinsr; elbinr++){
                tmpind = elbinz*nbinsr + elbinr;
                totcounts[tmpind] += tmpwcounts[thisthreadshift+tmpind];
                totnorms[tmpind] += tmpwnorms[thisthreadshift+tmpind];
            }
        }
    }
    
    // Get bin centers
    for (int elbinz=0; elbinz<nbinsz_source*nbinsz_lens; elbinz++){
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            int tmpind = elbinz*nbinsr + elbinr;
            if (totnorms[tmpind] != 0){
                bin_centers[tmpind] = totcounts[tmpind]/totnorms[tmpind];
            }
        }
    }
    free(tmpUpsilon);
    free(tmpNorm);
    free(tmpwcounts);
    free(tmpwnorms);
    free(totcounts);
    free(totnorms);
}

// Discrete estimator of Lens-Source-Source Correlator
void alloc_Gammans_discrete_NGG(const MultiresoCatalog *cat_source, const NavHash *nav_source,
                                const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                                const BinningParams *bin, int nthreads, int verbose,
                                NPCFOutput *out){
    // --- shear (source) field + scalar-position central lens, both nresos=1 ---
    double *w_source = cat_source->weight_resos, *pos1_source = cat_source->pos1_resos, *pos2_source = cat_source->pos2_resos;
    double *e1_source = cat_source->e1_resos, *e2_source = cat_source->e2_resos;
    int *zbin_source = cat_source->zbin_resos, nbinsz_source = cat_source->nbinsz, ngal_source = cat_source->ngal_resos[0];
    double *isinner_lens = cat_lens->isinner_resos, *w_lens = cat_lens->weight_resos;
    double *pos1_lens = cat_lens->pos1_resos, *pos2_lens = cat_lens->pos2_resos;
    int *zbin_lens = cat_lens->zbin_resos, nbinsz_lens = cat_lens->nbinsz, ngal_lens = cat_lens->ngal_resos[0];
    int nmax = bin->nmax, nbinsr = bin->nbinsr, dccorr = bin->dccorr;
    double rmin = bin->rmin, rmax = bin->rmax;
    int *index_matcher_source = nav_source->index_matcher, *pixs_galind_bounds_source = nav_source->pixs_galind_bounds, *pix_gals_source = nav_source->pix_gals;
    int *index_matcher_lens = nav_lens->index_matcher, *pixs_galind_bounds_lens = nav_lens->pixs_galind_bounds, *pix_gals_lens = nav_lens->pix_gals;
    int nregions = nav_lens->nregions;
    double pix1_start = nav_lens->pix1_start, pix1_d = nav_lens->pix1_d; int pix1_n = nav_lens->pix1_n;
    double pix2_start = nav_lens->pix2_start, pix2_d = nav_lens->pix2_d; int pix2_n = nav_lens->pix2_n;
    double *bin_centers = out->bin_centers;
    double complex *Upsilon_n = out->npcf, *Norm_n = out->norm_mp;

    int _nzcombis = nbinsz_lens*nbinsz_source*nbinsz_source;
    int _upsilonzshift = nbinsr*nbinsr;
    int _upsilonnshift = _upsilonzshift*_nzcombis;
    int _upsiloncompshift = (2*nmax+1)*_upsilonnshift;
    int _upsilonthreadshift = 2*_upsiloncompshift;
    int _normzshift = nbinsr*nbinsr;
    int _normnshift = _normzshift*_nzcombis;
    int _normthreadshift = (2*nmax+1)*_normnshift;    
    
    double *tmpwcounts = calloc(nthreads*nbinsz_lens*nbinsz_source*nbinsr, sizeof(double));
    double *tmpwnorms  = calloc(nthreads*nbinsz_lens*nbinsz_source*nbinsr, sizeof(double));
    double *totcounts = calloc(nbinsz_lens*nbinsz_source*nbinsr, sizeof(double));
    double *totnorms  = calloc(nbinsz_lens*nbinsz_source*nbinsr, sizeof(double));
    // Temporary arrays that are allocated in parallel and later reduced
    // Shape of tmpUpsilon ~ (nthreads, nnvals, nz_source, nz_lens, nz_lens, nbinsr, nbinsr)
    double complex *tmpUpsilon = calloc(nthreads*_upsilonthreadshift, sizeof(double complex));
    double complex *tmpNorm = calloc(nthreads*_normthreadshift, sizeof(double complex));
    #pragma omp parallel for num_threads(nthreads)
    for(int elthread=0;elthread<nthreads;elthread++){
        int nregions_per_thread = nregions/nthreads;
        int nnvals_Gn = 2*nmax+5; // Need [-nmax-2, ..., nmax+2]
        int nnvals_Wn = 2*nmax+1; // Need [-nmax, ..., nmax]
        int nnvals_Ups = 2*nmax+1;
        int nnvals_Norm = 2*nmax+1;
        int nzcombis = nbinsz_lens*nbinsz_source*nbinsz_source;
        int upsilon_zshift = nbinsr*nbinsr;
        int upsilon_nshift = upsilon_zshift*nzcombis;
        int upsilon_compshift = nnvals_Ups*upsilon_nshift;
        int threadshift_upsilon = 2*elthread*nnvals_Ups*upsilon_nshift;
        int norm_zshift = nbinsr*nbinsr;
        int norm_nshift = norm_zshift*nzcombis;
        int threadshift_norm = elthread*nnvals_Norm*norm_nshift;
        int threadshift_counts = elthread*nbinsz_lens*nbinsz_source*nbinsr;
        int nbinszr_Gn = nbinsz_source*nbinsr;
        int nbinszr_Wn = nbinsz_source*nbinsr;
        double rmin_sq = rmin*rmin;
        double rmax_sq = rmax*rmax;
        double drbin = log(rmax/rmin)/nbinsr;
        
        for (int elregion=0; elregion<nregions; elregion++){
            int region_debug=(int) (nthreads/2) * nregions_per_thread;
            //int region_debug=99999;
            // Check if this thread is responsible for the region
            int nthread_target = mymin(elregion/nregions_per_thread, nthreads-1);
            if (nthread_target!=elthread){continue;}
            bool printregdbg = (verbose>0) && (elregion==region_debug);
            bool printregdbg2 = (verbose>1) && (elregion==region_debug); 
            //if (elregion==region_debug){printf("Region %d is in thread %d\n",elregion,elthread);}
            if ((verbose>0) && (elthread==nthreads/2)){
                int elreg_inthread = elregion-nregions_per_thread*(nthreads/2);
                printf("\rDone %.2f per cent",100*((double) elreg_inthread)/nregions_per_thread);
            }
            
            int zbin_gal1, zbin_gal2;
            double isinner_gal1, pos1_gal1, pos2_gal1, w_gal1;
            double pos1_gal2, pos2_gal2, w_gal2, e1_gal2, e2_gal2;
            double complex wshape_gal2;
            int ind_red, ind_gal1, ind_gal2, lower1, upper1, lower2, upper2;
            int pix1_lower, pix2_lower, pix1_upper, pix2_upper;
            lower1 = pixs_galind_bounds_lens[elregion];
            upper1 = pixs_galind_bounds_lens[elregion+1];
            for (int ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                
                // Load lens galaxy info
                ind_gal1 = pix_gals_lens[ind_inpix1];
                isinner_gal1 = isinner_lens[ind_gal1];
                if(isinner_gal1<1e-5){continue;}
                pos1_gal1 = pos1_lens[ind_gal1];
                pos2_gal1 = pos2_lens[ind_gal1];
                w_gal1 = isinner_gal1*w_lens[ind_gal1];
                zbin_gal1 = zbin_lens[ind_gal1];
                zbin_gal1 = zbin_lens[ind_gal1];
                
                // Allocate the G_n and W_n coefficients + Double-counting correction factors
                double complex phirot, nphirot;
                double rel1, rel2, dist;
                int ind_Wnp, ind_Wnm, ind_Gnp, ind_Gnm, ind_counts, z1shift, z2rshift, rbin;
                double complex *thisGns = calloc(nnvals_Gn*nbinszr_Gn, sizeof(double complex)); 
                double complex *thisWns = calloc(nnvals_Wn*nbinszr_Wn, sizeof(double complex)); 
                double complex *thisG2ns = calloc(2*nbinszr_Gn, sizeof(double complex));
                double complex *thisW2ns = calloc(nbinszr_Wn, sizeof(double complex));
                int *thisncounts = calloc(nbinszr_Wn, sizeof(int));
                int *allowedrinds = calloc(nbinszr_Wn, sizeof(int));
                int *allowedzinds = calloc(nbinszr_Wn, sizeof(int));
                z1shift = zbin_gal1*nbinsz_source*nbinsr;
                pix1_lower = mymax(0, (int) floor((pos1_gal1 - (rmax+pix1_d) - pix1_start)/pix1_d));
                pix2_lower = mymax(0, (int) floor((pos2_gal1 - (rmax+pix2_d) - pix2_start)/pix2_d));
                pix1_upper = mymin(pix1_n-1, (int) floor((pos1_gal1 + (rmax+pix1_d) - pix1_start)/pix1_d));
                pix2_upper = mymin(pix2_n-1, (int) floor((pos2_gal1 + (rmax+pix2_d) - pix2_start)/pix2_d));
                for (int ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                    for (int ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                        ind_red = index_matcher_source[ind_pix2*pix1_n + ind_pix1];
                        if (ind_red==-1){continue;}
                        lower2 = pixs_galind_bounds_source[ind_red];
                        upper2 = pixs_galind_bounds_source[ind_red+1];
                        for (int ind_inpix=lower2; ind_inpix<upper2; ind_inpix++){
                            ind_gal2 = pix_gals_source[ind_inpix];
                            pos1_gal2 = pos1_source[ind_gal2];
                            pos2_gal2 = pos2_source[ind_gal2];
                            rel1 = pos1_gal2 - pos1_gal1;
                            rel2 = pos2_gal2 - pos2_gal1;
                            dist = rel1*rel1 + rel2*rel2;
                            if(dist < rmin_sq || dist >= rmax_sq) continue;
                            dist = sqrt(dist);
                            rbin = (int) floor(log(dist/rmin)/drbin);

                            w_gal2 = w_source[ind_gal2];
                            zbin_gal2 = zbin_source[ind_gal2];
                            e1_gal2 = e1_source[ind_gal2];
                            e2_gal2 = e2_source[ind_gal2];
                            wshape_gal2 = w_gal2*(e1_gal2+I*e2_gal2);
                            
                            z2rshift = zbin_gal2*nbinsr + rbin;
                            ind_counts = threadshift_counts + z1shift + z2rshift;
                            
                            
                            phirot = (rel1+I*rel2)/dist;
                            thisncounts[z2rshift] += 1;
                            tmpwcounts[ind_counts] += w_gal1*w_gal2*dist; 
                            tmpwnorms[ind_counts] += w_gal1*w_gal2; 
                            thisG2ns[z2rshift] += w_gal1*wshape_gal2*wshape_gal2*conj(phirot*phirot*phirot*phirot);
                            thisG2ns[nbinszr_Gn+z2rshift] += w_gal1*wshape_gal2*conj(wshape_gal2);
                            thisW2ns[z2rshift] += w_gal1*w_gal2*w_gal2;
                            
                            // n=0
                            ind_Wnp = nmax*nbinszr_Wn + z2rshift;
                            ind_Wnm = ind_Wnp;
                            ind_Gnp = (nmax+2)*nbinszr_Gn+z2rshift;
                            ind_Gnm = ind_Gnp;
                            nphirot = 1;
                            thisGns[ind_Gnp] += wshape_gal2;
                            thisWns[ind_Wnp] += w_gal2;
                            // n \in {-nmax, ..., -1, 1, ...,  nmax}
                            for (int nextn=1;nextn<=nmax;nextn++){
                                nphirot *= phirot; 
                                ind_Wnp += nbinszr_Wn;
                                ind_Wnm -= nbinszr_Wn;
                                ind_Gnp += nbinszr_Gn;
                                ind_Gnm -= nbinszr_Gn;
                                thisGns[ind_Gnp] += wshape_gal2*nphirot;
                                thisGns[ind_Gnm] += wshape_gal2*conj(nphirot);
                                thisWns[ind_Wnp] += w_gal2*nphirot;
                                thisWns[ind_Wnm] += w_gal2*conj(nphirot);
                            }
                            
                            // n \in {-nmax-2, -nmax-1, nmax+1, nmax+2}
                            nphirot *= phirot; 
                            ind_Gnp += nbinszr_Gn;
                            ind_Gnm -= nbinszr_Gn;
                            thisGns[ind_Gnp] += wshape_gal2*nphirot;
                            thisGns[ind_Gnm] += wshape_gal2*conj(nphirot);
                            nphirot *= phirot; 
                            ind_Gnp += nbinszr_Gn;
                            ind_Gnm -= nbinszr_Gn;
                            thisGns[ind_Gnp] += wshape_gal2*nphirot;
                            thisGns[ind_Gnm] += wshape_gal2*conj(nphirot);
                        }
                    }
                }
                
                // Update the Upsilon_n & N_n for this galaxy
                // shape (nthreads, nmax+1, nbinsz_lens, nbinsz_source, nbinsz_source, nbinsr, nbinsr)
                // First check for zero count bins
                // Note: Expected number of tracers in tomobin: <N> ~ 2*pi*nbar*drbin*<rbin>
                //   --> If we put lenses (with nbar<~1/arcmin^2) in tomo bins, most 3pcf bins will be empty...
                int nallowedcounts = 0;
                for (int zbin1=0; zbin1<nbinsz_source; zbin1++){
                    for (int elb1=0; elb1<nbinsr; elb1++){
                        z2rshift = zbin1*nbinsr + elb1;
                        if (thisncounts[z2rshift] != 0){
                            allowedrinds[nallowedcounts] = elb1;
                            allowedzinds[nallowedcounts] = zbin1;
                            nallowedcounts += 1;
                        }
                    }
                }
                // Now allocate only nonzero bins
                // Upsilon_-(thet1, thet2) ~ w * G_{+n-2}(thet1) * G_{-n-2}(thet2) - delta^K_{thet1,thet2} * (w * (we)^2*exp(-4*phi))
                // Upsilon_+(thet1, thet2) ~ w * G_{+n-2}(thet1) * conj(G_{+n-2})(thet2) - delta^K_{thet1,thet2} * (w * |we|^2)
                // Norm(thet1, thet2)    ~   w  * W_{n}(thet1)   * W_{-n}(thet2)   - delta^K_{thet1,thet2} * (w  * w*w)
                // Note that here we allocate also the negative multipoles as Upsilon_- does not have a symmetry connecting the 
                // negative multipoles to the positive one (for this we would need also a <n gamma^* gamma> correlator, but this
                // does not carry any additional information as compared to <n gamma gamma^*>...). 
                
                for (int thisn=-nmax; thisn<=nmax; thisn++){
                    int thisnshift_ups = threadshift_upsilon + (thisn+nmax)*upsilon_nshift;
                    int thisnshift_norm = threadshift_norm + (thisn+nmax)*norm_nshift;
                    int _wind, _upsind1, _upsind2, zrshift, zcombi, upsilon_indshift, norm_indshift, elb1, zbin2, elb2, zbin3;
                    double complex nextUps1, nextUps2, nextN;
                    for (int zrcombis1=0; zrcombis1<nallowedcounts; zrcombis1++){
                        elb1 = allowedrinds[zrcombis1];
                        zbin2 = allowedzinds[zrcombis1];
                        zrshift = zbin2*nbinsr + elb1;
                        // Double counting correction
                        if (dccorr==1){
                            zcombi = zbin_gal1*nbinsz_source*nbinsz_source + zbin2*nbinsz_source + zbin2;
                            upsilon_indshift = thisnshift_ups + zcombi*upsilon_zshift + elb1*nbinsr+elb1;
                            norm_indshift = thisnshift_norm + zcombi*norm_zshift + elb1*nbinsr+elb1;
                            tmpUpsilon[upsilon_indshift] -= thisG2ns[zrshift];
                            tmpUpsilon[upsilon_indshift+upsilon_compshift] -= thisG2ns[nbinszr_Gn+zrshift];
                            tmpNorm[norm_indshift] -= thisW2ns[zrshift];
                        }
                        _wind = (nmax+thisn)*nbinszr_Wn+zrshift;
                        _upsind1 = (nmax+0+thisn)*nbinszr_Gn+zrshift;
                        _upsind2 = (nmax+0+thisn)*nbinszr_Gn+zrshift;
                        nextUps1 = w_gal1*thisGns[_upsind1];
                        nextUps2 = w_gal1*thisGns[_upsind2];
                        nextN = w_gal1*thisWns[_wind];
                        for (int zrcombis2=0; zrcombis2<nallowedcounts; zrcombis2++){
                            elb2 = allowedrinds[zrcombis2];
                            zbin3 = allowedzinds[zrcombis2];
                            zrshift = zbin3*nbinsr + elb2;
                            _wind = (nmax-thisn)*nbinszr_Wn+zrshift;
                            _upsind1 = (nmax-thisn+0)*nbinszr_Gn+zrshift;
                            _upsind2 = (nmax+thisn+0)*nbinszr_Gn+zrshift;
                            zcombi = zbin_gal1*nbinsz_source*nbinsz_source + zbin2*nbinsz_source + zbin3;
                            upsilon_indshift = thisnshift_ups + elb1*nbinsr + zcombi*upsilon_zshift + elb2;
                            norm_indshift = thisnshift_norm + elb1*nbinsr + zcombi*upsilon_zshift + elb2;
                            tmpUpsilon[upsilon_indshift] += nextUps1*thisGns[_upsind1];
                            tmpUpsilon[upsilon_indshift+upsilon_compshift] += nextUps2*conj(thisGns[_upsind2]);
                            tmpNorm[norm_indshift] += nextN*thisWns[_wind];
                        }
                    }
                }
                free(thisWns);
                free(thisGns);
                free(thisG2ns);
                free(thisW2ns);
                free(thisncounts);
                free(allowedrinds);
                free(allowedzinds);
            }
        }
    }
    
    // Accumulate the Upsilon_n / N_n
    #pragma omp parallel for num_threads(nthreads)
    for (int thisn=0; thisn<2*nmax+1; thisn++){
        int iUps;
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            int thisthreadshift_ups = thisthread*_upsilonthreadshift;
            int thisthreadshift_norm = thisthread*_normthreadshift;
            for (int zcombi=0; zcombi<_nzcombis; zcombi++){
                for (int elb1=0; elb1<nbinsr; elb1++){
                    for (int elb2=0; elb2<nbinsr; elb2++){
                        iUps = thisn*_upsilonnshift + zcombi*_upsilonzshift + elb1*nbinsr + elb2;
                        Upsilon_n[iUps] += tmpUpsilon[thisthreadshift_ups+iUps];
                        Upsilon_n[iUps+_upsiloncompshift] += tmpUpsilon[thisthreadshift_ups+_upsiloncompshift+iUps];
                        Norm_n[iUps] += tmpNorm[thisthreadshift_norm+iUps];
                    }
                }
            }
        }
    }
    
    // Accumulate the bin distances and weights
    for (int thisthread=0; thisthread<nthreads; thisthread++){
        int tmpind;
        int thisthreadshift = thisthread*nbinsz_source*nbinsz_lens*nbinsr; 
        for (int elbinz=0; elbinz<nbinsz_source*nbinsz_lens; elbinz++){
            for (int elbinr=0; elbinr<nbinsr; elbinr++){
                tmpind = elbinz*nbinsr + elbinr;
                totcounts[tmpind] += tmpwcounts[thisthreadshift+tmpind];
                totnorms[tmpind] += tmpwnorms[thisthreadshift+tmpind];
            }
        }
    }
    
    // Get bin centers
    for (int elbinz=0; elbinz<nbinsz_source*nbinsz_lens; elbinz++){
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            int tmpind = elbinz*nbinsr + elbinr;
            if (totnorms[tmpind] != 0){
                bin_centers[tmpind] = totcounts[tmpind]/totnorms[tmpind];
            }
        }
    } 
    free(tmpUpsilon);
    free(tmpNorm);
    free(tmpwcounts);
    free(tmpwnorms);
    free(totcounts);
    free(totnorms);
}      

// Discrete estimator of Lens-Source-Source Correlator
void alloc_Gammans_tree_NGG(const MultiresoCatalog *cat_source, const NavHash *nav_source,
                            const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                            const TreeResoParams *tree, const BinningParams *bin,
                            int nthreads, int verbose, NPCFOutput *out){
    // --- reduced per-reso shear (source) field + base scalar-position central lens ---
    int nresos = tree->nresos; double *reso_redges = tree->reso_redges;
    double *w_source_resos = cat_source->weight_resos, *pos1_source_resos = cat_source->pos1_resos, *pos2_source_resos = cat_source->pos2_resos;
    double *e1_source_resos = cat_source->e1_resos, *e2_source_resos = cat_source->e2_resos;
    int *zbin_source_resos = cat_source->zbin_resos, nbinsz_source = cat_source->nbinsz, *ngal_source_resos = cat_source->ngal_resos;
    double *isinner_lens = cat_lens->isinner_resos, *w_lens = cat_lens->weight_resos;
    double *pos1_lens = cat_lens->pos1_resos, *pos2_lens = cat_lens->pos2_resos;
    int *zbin_lens = cat_lens->zbin_resos, nbinsz_lens = cat_lens->nbinsz, ngal_lens = cat_lens->ngal_resos[0];
    int *index_matcher_source = nav_source->index_matcher, *pixs_galind_bounds_source = nav_source->pixs_galind_bounds, *pix_gals_source = nav_source->pix_gals;
    int *index_matcher_lens = nav_lens->index_matcher, *pixs_galind_bounds_lens = nav_lens->pixs_galind_bounds, *pix_gals_lens = nav_lens->pix_gals;
    int nregions = nav_lens->nregions;
    double pix1_start = nav_lens->pix1_start, pix1_d = nav_lens->pix1_d; int pix1_n = nav_lens->pix1_n;
    double pix2_start = nav_lens->pix2_start, pix2_d = nav_lens->pix2_d; int pix2_n = nav_lens->pix2_n;
    int nmax = bin->nmax, nbinsr = bin->nbinsr, dccorr = bin->dccorr;
    double rmin = bin->rmin, rmax = bin->rmax;
    double *bin_centers = out->bin_centers;
    double complex *Upsilon_n = out->npcf, *Norm_n = out->norm_mp;

    int _nzcombis = nbinsz_lens*nbinsz_source*nbinsz_source;
    int _upsilonzshift = nbinsr*nbinsr;
    int _upsilonnshift = _upsilonzshift*_nzcombis;
    int _upsiloncompshift = (2*nmax+1)*_upsilonnshift;
    int _upsilonthreadshift = 2*_upsiloncompshift;
    int _normzshift = nbinsr*nbinsr;
    int _normnshift = _normzshift*_nzcombis;
    int _normthreadshift = (2*nmax+1)*_normnshift;    
    
    double *tmpwcounts = calloc(nthreads*nbinsz_lens*nbinsz_source*nbinsr, sizeof(double));
    double *tmpwnorms  = calloc(nthreads*nbinsz_lens*nbinsz_source*nbinsr, sizeof(double));
    double *totcounts = calloc(nbinsz_lens*nbinsz_source*nbinsr, sizeof(double));
    double *totnorms  = calloc(nbinsz_lens*nbinsz_source*nbinsr, sizeof(double));
    // Temporary arrays that are allocated in parallel and later reduced
    // Shape of tmpUpsilon ~ (nthreads, nnvals, nz_source, nz_lens, nz_lens, nbinsr, nbinsr)
    double complex *tmpUpsilon = calloc(nthreads*_upsilonthreadshift, sizeof(double complex));
    double complex *tmpNorm = calloc(nthreads*_normthreadshift, sizeof(double complex));
    #pragma omp parallel for num_threads(nthreads)
    for(int elthread=0;elthread<nthreads;elthread++){
        int nregions_per_thread = nregions/nthreads;
        int nnvals_Gn = 2*nmax+5; // Need [-nmax-2, ..., nmax+2]
        int nnvals_Wn = 2*nmax+1; // Need [-nmax, ..., nmax]
        int nnvals_Ups = 2*nmax+1;
        int nnvals_Norm = 2*nmax+1;
        int nzcombis = nbinsz_lens*nbinsz_source*nbinsz_source;
        int upsilon_zshift = nbinsr*nbinsr;
        int upsilon_nshift = upsilon_zshift*nzcombis;
        int upsilon_compshift = nnvals_Ups*upsilon_nshift;
        int threadshift_upsilon = 2*elthread*nnvals_Ups*upsilon_nshift;
        int norm_zshift = nbinsr*nbinsr;
        int norm_nshift = norm_zshift*nzcombis;
        int threadshift_norm = elthread*nnvals_Norm*norm_nshift;
        int threadshift_counts = elthread*nbinsz_lens*nbinsz_source*nbinsr;
        int nbinszr_Gn = nbinsz_source*nbinsr;
        int nbinszr_Wn = nbinsz_source*nbinsr;
        double drbin = log(rmax/rmin)/nbinsr;
        int npix_hash = pix1_n*pix2_n;
        
        int *rshift_index_matcher = calloc(nresos, sizeof(int));
        int *rshift_pixs_galind_bounds = calloc(nresos, sizeof(int));
        int *rshift_pix_gals = calloc(nresos, sizeof(int));
        build_rshift_offsets(nresos, npix_hash, ngal_source_resos,
            rshift_index_matcher, rshift_pixs_galind_bounds, rshift_pix_gals);
        
        
        for (int elregion=0; elregion<nregions; elregion++){
            int region_debug=(int) (nthreads/2) * nregions_per_thread;
            //int region_debug=99999;
            // Check if this thread is responsible for the region
            int nthread_target = mymin(elregion/nregions_per_thread, nthreads-1);
            if (nthread_target!=elthread){continue;}
            bool printregdbg = (verbose>0) && (elregion==region_debug);
            bool printregdbg2 = (verbose>1) && (elregion==region_debug); 
            //if (elregion==region_debug){printf("Region %d is in thread %d\n",elregion,elthread);}
            if ((verbose>0) && (elthread==nthreads/2)){
                int elreg_inthread = elregion-nregions_per_thread*(nthreads/2);
                printf("\rDone %.2f per cent",100*((double) elreg_inthread)/nregions_per_thread);
            }
            
            int zbin_gal1, zbin_gal2;
            double isinner_gal1, pos1_gal1, pos2_gal1, w_gal1;
            double pos1_gal2, pos2_gal2, w_gal2, e1_gal2, e2_gal2;
            double complex wshape_gal2;
            int ind_red, ind_gal1, ind_gal2, lower1, upper1, lower2, upper2;
            int ind_inpix1, ind_inpix2;
            lower1 = pixs_galind_bounds_lens[elregion];
            upper1 = pixs_galind_bounds_lens[elregion+1];
            for (ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                
                // Load lens galaxy info
                ind_gal1 = pix_gals_lens[ind_inpix1];
                isinner_gal1 = isinner_lens[ind_gal1];
                if(isinner_gal1==0){continue;}
                pos1_gal1 = pos1_lens[ind_gal1];
                pos2_gal1 = pos2_lens[ind_gal1];
                w_gal1 = isinner_gal1*w_lens[ind_gal1];
                zbin_gal1 = zbin_lens[ind_gal1];
                zbin_gal1 = zbin_lens[ind_gal1];
                
                // Allocate the G_n and W_n coefficients + Double-counting correction factors
                double complex phirot, nphirot;
                double rel1, rel2, dist;
                int ind_Wnp, ind_Wnm, ind_Gnp, ind_Gnm, ind_counts, z1shift, z2rshift, rbin;
                double complex *thisGns = calloc(nnvals_Gn*nbinszr_Gn, sizeof(double complex)); 
                double complex *thisWns = calloc(nnvals_Wn*nbinszr_Wn, sizeof(double complex)); 
                double complex *thisG2ns = calloc(2*nbinszr_Gn, sizeof(double complex));
                double complex *thisW2ns = calloc(nbinszr_Wn, sizeof(double complex));
                int *thisncounts = calloc(nbinszr_Wn, sizeof(int));
                int *allowedrinds = calloc(nbinszr_Wn, sizeof(int));
                int *allowedzinds = calloc(nbinszr_Wn, sizeof(int));
                z1shift = zbin_gal1*nbinsz_source*nbinsr;
                /////
                
                
                for (int elreso=0;elreso<nresos;elreso++){
                    int pix1_lower = mymax(0, (int) floor((pos1_gal1 - 
                                                           (reso_redges[elreso+1]+pix1_d) - pix1_start)/pix1_d));
                    int pix2_lower = mymax(0, (int) floor((pos2_gal1 - 
                                                           (reso_redges[elreso+1]+pix2_d) - pix2_start)/pix2_d));
                    int pix1_upper = mymin(pix1_n-1, (int) floor((pos1_gal1 + 
                                                                  (reso_redges[elreso+1]+pix1_d) - pix1_start)/pix1_d));
                    int pix2_upper = mymin(pix2_n-1, (int) floor((pos2_gal1 + 
                                                                  (reso_redges[elreso+1]+pix2_d) - pix2_start)/pix2_d));
                    int ind_pix1, ind_pix2;
                    double rmin_sq = reso_redges[elreso]*reso_redges[elreso];
                    double rmax_sq = reso_redges[elreso+1]*reso_redges[elreso+1];
                    for (ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                        for (ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                            ind_red = index_matcher_source[rshift_index_matcher[elreso] + ind_pix2*pix1_n + ind_pix1];
                            if (ind_red==-1){continue;}
                            lower2 = pixs_galind_bounds_source[rshift_pixs_galind_bounds[elreso]+ind_red];
                            upper2 = pixs_galind_bounds_source[rshift_pixs_galind_bounds[elreso]+ind_red+1];
                            for (ind_inpix2=lower2; ind_inpix2<upper2; ind_inpix2++){
                                ind_gal2 = rshift_pix_gals[elreso] + pix_gals_source[rshift_pix_gals[elreso]+ind_inpix2];
                                pos1_gal2 = pos1_source_resos[ind_gal2];
                                pos2_gal2 = pos2_source_resos[ind_gal2];
                                rel1 = pos1_gal2 - pos1_gal1;
                                rel2 = pos2_gal2 - pos2_gal1;
                                dist = rel1*rel1 + rel2*rel2;
                                if(dist < rmin_sq || dist >= rmax_sq) continue;
                                dist = sqrt(dist);
                                rbin = (int) floor((log(dist)-log(rmin))/drbin);
                                w_gal2 = w_source_resos[ind_gal2];
                                zbin_gal2 = zbin_source_resos[ind_gal2];
                                e1_gal2 = e1_source_resos[ind_gal2];
                                e2_gal2 = e2_source_resos[ind_gal2];
                                wshape_gal2 = w_gal2*(e1_gal2+I*e2_gal2);
                                z2rshift = zbin_gal2*nbinsr + rbin;
                                ind_counts = threadshift_counts + z1shift + z2rshift;

                                phirot = (rel1+I*rel2)/dist;
                                thisncounts[z2rshift] += 1;
                                tmpwcounts[ind_counts] += w_gal1*w_gal2*dist; 
                                tmpwnorms[ind_counts] += w_gal1*w_gal2; 
                                thisG2ns[z2rshift] += w_gal1*wshape_gal2*wshape_gal2*conj(phirot*phirot*phirot*phirot);
                                thisG2ns[nbinszr_Gn+z2rshift] += w_gal1*wshape_gal2*conj(wshape_gal2);
                                thisW2ns[z2rshift] += w_gal1*w_gal2*w_gal2;

                                // n=0
                                ind_Wnp = nmax*nbinszr_Wn + z2rshift;
                                ind_Wnm = ind_Wnp;
                                ind_Gnp = (nmax+2)*nbinszr_Gn+z2rshift;
                                ind_Gnm = ind_Gnp;
                                nphirot = 1;
                                thisGns[ind_Gnp] += wshape_gal2;
                                thisWns[ind_Wnp] += w_gal2;
                                // n \in {-nmax, ..., -1, 1, ...,  nmax}
                                for (int nextn=1;nextn<=nmax;nextn++){
                                    nphirot *= phirot; 
                                    ind_Wnp += nbinszr_Wn;
                                    ind_Wnm -= nbinszr_Wn;
                                    ind_Gnp += nbinszr_Gn;
                                    ind_Gnm -= nbinszr_Gn;
                                    thisGns[ind_Gnp] += wshape_gal2*nphirot;
                                    thisGns[ind_Gnm] += wshape_gal2*conj(nphirot);
                                    thisWns[ind_Wnp] += w_gal2*nphirot;
                                    thisWns[ind_Wnm] += w_gal2*conj(nphirot);
                                }

                                // n \in {-nmax-2, -nmax-1, nmax+1, nmax+2}
                                nphirot *= phirot; 
                                ind_Gnp += nbinszr_Gn;
                                ind_Gnm -= nbinszr_Gn;
                                thisGns[ind_Gnp] += wshape_gal2*nphirot;
                                thisGns[ind_Gnm] += wshape_gal2*conj(nphirot);
                                nphirot *= phirot; 
                                ind_Gnp += nbinszr_Gn;
                                ind_Gnm -= nbinszr_Gn;
                                thisGns[ind_Gnp] += wshape_gal2*nphirot;
                                thisGns[ind_Gnm] += wshape_gal2*conj(nphirot);
                            }
                        }
                    }
                }
                
                // Update the Upsilon_n & N_n for this galaxy
                // shape (nthreads, nmax+1, nbinsz_lens, nbinsz_source, nbinsz_source, nbinsr, nbinsr)
                // First check for zero count bins
                // Note: Expected number of tracers in tomobin: <N> ~ 2*pi*nbar*drbin*<rbin>
                //   --> If we put lenses (with nbar<~1/arcmin^2) in tomo bins, most 3pcf bins will be empty...
                int nallowedcounts = 0;
                for (int zbin1=0; zbin1<nbinsz_source; zbin1++){
                    for (int elb1=0; elb1<nbinsr; elb1++){
                        z2rshift = zbin1*nbinsr + elb1;
                        if (thisncounts[z2rshift] != 0){
                            allowedrinds[nallowedcounts] = elb1;
                            allowedzinds[nallowedcounts] = zbin1;
                            nallowedcounts += 1;
                        }
                    }
                }
                // Now allocate only nonzero bins
                // Upsilon_-(thet1, thet2) ~ w * G_{+n-2}(thet1) * G_{-n-2}(thet2) - delta^K_{thet1,thet2} * (w * (we)^2*exp(4*phi))
                // Upsilon_+(thet1, thet2) ~ w * G_{+n-2}(thet1) * conj(G_{+n-2})(thet2) - delta^K_{thet1,thet2} * (w * |we|^2)
                // Norm(thet1, thet2)    ~   w  * W_{n}(thet1)   * W_{-n}(thet2)   - delta^K_{thet1,thet2} * (w  * w*w)
                // Note that here we allocate also the negative multipoles as Upsilon_- does not have a symmetry connecting the 
                // negative multipoles to the positive one (for this we would need also a <n gamma^* gamma> correlator, but this
                // does not carry any additional information as compared to <n gamma gamma^*>...). 
                
                for (int thisn=-nmax; thisn<=nmax; thisn++){
                    int thisnshift_ups = threadshift_upsilon + (thisn+nmax)*upsilon_nshift;
                    int thisnshift_norm = threadshift_norm + (thisn+nmax)*norm_nshift;
                    int _wind, _upsind1, _upsind2, zrshift, zcombi, upsilon_indshift, norm_indshift, elb1, zbin2, elb2, zbin3;
                    double complex nextUps1, nextUps2, nextN;
                    for (int zrcombis1=0; zrcombis1<nallowedcounts; zrcombis1++){
                        elb1 = allowedrinds[zrcombis1];
                        zbin2 = allowedzinds[zrcombis1];
                        zrshift = zbin2*nbinsr + elb1;
                        // Double counting correction
                        if (dccorr==1){
                            zcombi = zbin_gal1*nbinsz_source*nbinsz_source + zbin2*nbinsz_source + zbin2;
                            upsilon_indshift = thisnshift_ups + zcombi*upsilon_zshift + elb1*nbinsr+elb1;
                            norm_indshift = thisnshift_norm + zcombi*norm_zshift + elb1*nbinsr+elb1;
                            tmpUpsilon[upsilon_indshift] -= thisG2ns[zrshift];
                            tmpUpsilon[upsilon_indshift+upsilon_compshift] -= thisG2ns[nbinszr_Gn+zrshift];
                            tmpNorm[norm_indshift] -= thisW2ns[zrshift];
                        }
                        _wind = (nmax+thisn)*nbinszr_Wn+zrshift;
                        _upsind1 = (nmax+0+thisn)*nbinszr_Gn+zrshift;
                        _upsind2 = (nmax+0+thisn)*nbinszr_Gn+zrshift;
                        nextUps1 = w_gal1*thisGns[_upsind1];
                        nextUps2 = w_gal1*thisGns[_upsind2];
                        nextN = w_gal1*thisWns[_wind];
                        for (int zrcombis2=0; zrcombis2<nallowedcounts; zrcombis2++){
                            elb2 = allowedrinds[zrcombis2];
                            zbin3 = allowedzinds[zrcombis2];
                            zrshift = zbin3*nbinsr + elb2;
                            _wind = (nmax-thisn)*nbinszr_Wn+zrshift;
                            _upsind1 = (nmax-thisn+0)*nbinszr_Gn+zrshift;
                            _upsind2 = (nmax+thisn+0)*nbinszr_Gn+zrshift;
                            zcombi = zbin_gal1*nbinsz_source*nbinsz_source + zbin2*nbinsz_source + zbin3;
                            upsilon_indshift = thisnshift_ups + elb1*nbinsr + zcombi*upsilon_zshift + elb2;
                            norm_indshift = thisnshift_norm + elb1*nbinsr + zcombi*upsilon_zshift + elb2;
                            tmpUpsilon[upsilon_indshift] += nextUps1*thisGns[_upsind1];
                            tmpUpsilon[upsilon_indshift+upsilon_compshift] += nextUps2*conj(thisGns[_upsind2]);
                            tmpNorm[norm_indshift] += nextN*thisWns[_wind];
                        }
                    }
                }
                free(thisWns);
                free(thisGns);
                free(thisG2ns);
                free(thisW2ns);
                free(thisncounts);
                free(allowedrinds);
                free(allowedzinds);
            }
        }
        free(rshift_index_matcher);
        free(rshift_pixs_galind_bounds);
        free(rshift_pix_gals);
    }
    
    // Accumulate the Upsilon_n / N_n
    #pragma omp parallel for num_threads(nthreads)
    for (int thisn=0; thisn<2*nmax+1; thisn++){
        int iUps;
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            int thisthreadshift_ups = thisthread*_upsilonthreadshift;
            int thisthreadshift_norm = thisthread*_normthreadshift;
            for (int zcombi=0; zcombi<_nzcombis; zcombi++){
                for (int elb1=0; elb1<nbinsr; elb1++){
                    for (int elb2=0; elb2<nbinsr; elb2++){
                        iUps = thisn*_upsilonnshift + zcombi*_upsilonzshift + elb1*nbinsr + elb2;
                        Upsilon_n[iUps] += tmpUpsilon[thisthreadshift_ups+iUps];
                        Upsilon_n[iUps+_upsiloncompshift] += tmpUpsilon[thisthreadshift_ups+_upsiloncompshift+iUps];
                        Norm_n[iUps] += tmpNorm[thisthreadshift_norm+iUps];
                    }
                }
            }
        }
    }
    
    // Accumulate the bin distances and weights
    for (int thisthread=0; thisthread<nthreads; thisthread++){
        int tmpind;
        int thisthreadshift = thisthread*nbinsz_source*nbinsz_lens*nbinsr; 
        for (int elbinz=0; elbinz<nbinsz_source*nbinsz_lens; elbinz++){
            for (int elbinr=0; elbinr<nbinsr; elbinr++){
                tmpind = elbinz*nbinsr + elbinr;
                totcounts[tmpind] += tmpwcounts[thisthreadshift+tmpind];
                totnorms[tmpind] += tmpwnorms[thisthreadshift+tmpind];
            }
        }
    }
    
    // Get bin centers
    for (int elbinz=0; elbinz<nbinsz_source*nbinsz_lens; elbinz++){
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            int tmpind = elbinz*nbinsr + elbinr;
            if (totnorms[tmpind] != 0){
                bin_centers[tmpind] = totcounts[tmpind]/totnorms[tmpind];
            }
        }
    } 
    free(tmpUpsilon);
    free(tmpNorm);
    free(tmpwcounts);
    free(tmpwnorms);
    free(totcounts);
    free(totnorms);
}

// DoubleTree based estimtor of Lens-Source-Source Correlator
void alloc_Gammans_doubletree_NGG(const MultiresoCatalog *cat_source, const NavHash *nav_source,
                                  const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                                  const TreeResoParams *tree, const BinningParams *bin,
                                  int nthreads, int verbose, NPCFOutput *out){
    // --- tree parameters (full leaf params) ---
    int nresos = tree->nresos, nresos_grid = tree->nresos_grid;
    double *dpix1_resos = tree->dpix1_resos, *dpix2_resos = tree->dpix2_resos, *reso_redges = tree->reso_redges;
    int resoshift_leafs = tree->resoshift_leafs, minresoind_leaf = tree->minresoind_leaf, maxresoind_leaf = tree->maxresoind_leaf;
    // --- multi-resolution shear (source) field + scalar-position central lens ---
    double *isinner_source_resos = cat_source->isinner_resos, *w_source_resos = cat_source->weight_resos;
    double *pos1_source_resos = cat_source->pos1_resos, *pos2_source_resos = cat_source->pos2_resos;
    double *e1_source_resos = cat_source->e1_resos, *e2_source_resos = cat_source->e2_resos;
    int *zbin_source_resos = cat_source->zbin_resos, *ngal_source_resos = cat_source->ngal_resos, nbinsz_source = cat_source->nbinsz;
    double *isinner_lens_resos = cat_lens->isinner_resos, *w_lens_resos = cat_lens->weight_resos;
    double *pos1_lens_resos = cat_lens->pos1_resos, *pos2_lens_resos = cat_lens->pos2_resos;
    int *zbin_lens_resos = cat_lens->zbin_resos, *ngal_lens_resos = cat_lens->ngal_resos, nbinsz_lens = cat_lens->nbinsz;
    // --- binning ---
    int nmax = bin->nmax, nbinsr = bin->nbinsr, dccorr = bin->dccorr;
    double rmin = bin->rmin, rmax = bin->rmax;
    // --- navigation (lens central carries the occupied-region list; shared pix grid) ---
    int *index_matcher_source = nav_source->index_matcher, *pixs_galind_bounds_source = nav_source->pixs_galind_bounds, *pix_gals_source = nav_source->pix_gals;
    int *index_matcher_lens = nav_lens->index_matcher, *pixs_galind_bounds_lens = nav_lens->pixs_galind_bounds, *pix_gals_lens = nav_lens->pix_gals;
    int *index_matcher_hash = nav_lens->index_matcher_hash, nregions = nav_lens->nregions;
    double pix1_start = nav_lens->pix1_start, pix1_d = nav_lens->pix1_d; int pix1_n = nav_lens->pix1_n;
    double pix2_start = nav_lens->pix2_start, pix2_d = nav_lens->pix2_d; int pix2_n = nav_lens->pix2_n;
    double *bin_centers = out->bin_centers;
    double complex *Upsilon_n = out->npcf, *Norm_n = out->norm_mp;

    int _ncomp_Upsilon = 2; // [Upsilon-,Upsilon+]
    int _nzcombis = nbinsz_lens*nbinsz_source*nbinsz_source;
    int _upsilonzshift = nbinsr*nbinsr;
    int _upsilonnshift = _upsilonzshift*_nzcombis;
    int _upsiloncompshift = (2*nmax+1)*_upsilonnshift;
    int _upsilonthreadshift = _ncomp_Upsilon*_upsiloncompshift;
    int _normzshift = nbinsr*nbinsr;
    int _normnshift = _normzshift*_nzcombis;
    int _normthreadshift = (2*nmax+1)*_normnshift;  
    
    int *regionsdone = calloc(nregions, sizeof(int));
    int nregionsdone = 0;
    
    double *tmpwcounts = calloc(nthreads*nbinsz_lens*nbinsz_source*nbinsr, sizeof(double));
    double *tmpwnorms  = calloc(nthreads*nbinsz_lens*nbinsz_source*nbinsr, sizeof(double));
    double *totcounts = calloc(nbinsz_lens*nbinsz_source*nbinsr, sizeof(double));
    double *totnorms  = calloc(nbinsz_lens*nbinsz_source*nbinsr, sizeof(double));
    // Temporary arrays that are allocated in parallel and later reduced
    // Shape of tmpUpsilon ~ (nthreads, 2, nnvals, nz_source, nz_lens, nz_lens, nbinsr, nbinsr)
    double complex *tmpUpsilon = calloc(nthreads*_upsilonthreadshift, sizeof(double complex));
    double complex *tmpNorm = calloc(nthreads*_normthreadshift, sizeof(double complex));
    #pragma omp parallel for num_threads(nthreads)
    for(int elthread=0;elthread<nthreads;elthread++){
        int nregions_per_thread = nregions/nthreads;
        int hasdiscrete = nresos-nresos_grid;
        int nnvals_Gn = 2*nmax+5; // Need [-nmax-2, ..., nmax+2]
        int nnvals_Wn = 2*nmax+1; // Need [-nmax, ..., nmax]
        int nnvals_Ups = 2*nmax+1;
        int nnvals_Norm = 2*nmax+1;
        int ncomp_Upsilon = 2;
        int nzcombis = nbinsz_lens*nbinsz_source*nbinsz_source;
        int upsilon_zshift = nbinsr*nbinsr;
        int upsilon_nshift = upsilon_zshift*nzcombis;
        int upsilon_compshift = nnvals_Ups*upsilon_nshift;
        int upsilon_threadshift = elthread*ncomp_Upsilon*upsilon_compshift;
        int norm_zshift = nbinsr*nbinsr;
        int norm_nshift = norm_zshift*nzcombis;
        int norm_threadshift = elthread*nnvals_Norm*norm_nshift;
        int counts_threadshift = elthread*nbinsz_lens*nbinsz_source*nbinsr;
        double drbin = log(rmax/rmin)/nbinsr;
        
        // Largest possible nshift: each zbin does completely fill out the lowest reso grid.
        // The remaining grids then have 1/4 + 1/16 + ... --> 0.33.... times the data of the largest grid. 
        // Now allocate the caches
        int size_max_nshift = ((1+hasdiscrete+0.34)*nbinsz_lens*nbinsz_source*nbinsr*pow(4,nresos_grid-1));
        double complex *Gncache = calloc(nnvals_Gn*size_max_nshift, sizeof(double complex));
        double complex *wGncache = calloc(nnvals_Gn*size_max_nshift, sizeof(double complex));
        double complex *cwGncache = calloc(nnvals_Gn*size_max_nshift, sizeof(double complex));
        double complex *Wncache = calloc(nnvals_Wn*size_max_nshift, sizeof(double complex));
        double complex *wWncache = calloc(nnvals_Wn*size_max_nshift, sizeof(double complex));
        int *Wncache_updates = calloc(size_max_nshift, sizeof(int));
        for (int _elregion=0; _elregion<2*nregions; _elregion++){
            int region_debug=-1;

            // Check if this thread is responsible for the region
            int elregion = _elregion%nregions;
            int wasdone = 0;
            if (_elregion<nregions){
                int nthread_target = mymin(elregion/nregions_per_thread, nthreads-1);
                if (nthread_target!=elthread){continue;}
            }
            #pragma omp critical
            {   
                if (regionsdone[elregion]==1){wasdone = 1;}
                else{
                    regionsdone[elregion]=1;
                    nregionsdone+=1; 
                }
            }
            if (wasdone==1){continue;}

            bool printregdbg = (verbose>0) && (elregion==region_debug);
            bool printregdbg2 = (verbose>1) && (elregion==region_debug); 
            if (printregdbg2){printf("Region %d is in thread %d\n",elregion,elthread);}
            
            // Check which sets of radii are evaluated for each resolution
            int *reso_rindedges = calloc(nresos+1, sizeof(int));
            double logrmin = log(rmin);
            build_reso_rindedges(nresos, reso_redges, rmin, rmax, nbinsr, reso_rindedges);

            // Shift variables for spatial hash of sources and lenses
            int npix_hash = pix1_n*pix2_n;
            int *rshift_index_matcher_source = calloc(nresos, sizeof(int));
            int *rshift_pixs_galind_bounds_source = calloc(nresos, sizeof(int));
            int *rshift_pix_gals_source = calloc(nresos, sizeof(int));
            build_rshift_offsets(nresos, npix_hash, ngal_source_resos,
                rshift_index_matcher_source, rshift_pixs_galind_bounds_source, rshift_pix_gals_source);
            int *rshift_index_matcher_lens = calloc(nresos, sizeof(int));
            int *rshift_pixs_galind_bounds_lens = calloc(nresos, sizeof(int));
            int *rshift_pix_gals_lens = calloc(nresos, sizeof(int));
            build_rshift_offsets(nresos, npix_hash, ngal_lens_resos,
                rshift_index_matcher_lens, rshift_pixs_galind_bounds_lens, rshift_pix_gals_lens);

            // Region layout of the lens (central) catalog: per-(zbin, reso)
            // counts, reduced-grid offsets, pixel -> reduced-pixel matcher.
            int lower1, upper1, lower2, upper2;
            int *matchers_resoshift = calloc(nresos_grid+1, sizeof(int));
            int *ngal_in_pix = calloc(nresos*nbinsz_lens, sizeof(int));
            int len_matcher = build_region_galinpix(nresos, nresos_grid, hasdiscrete,
                elregion, pixs_galind_bounds_lens, rshift_pixs_galind_bounds_lens,
                pix_gals_lens, rshift_pix_gals_lens, zbin_lens_resos,
                matchers_resoshift, ngal_in_pix);
            double hashpix_start1, hashpix_start2;
            int *pix2redpix = calloc(nbinsz_lens*len_matcher, sizeof(int));
            build_region_pix2redpix(nresos_grid, hasdiscrete, elregion, nbinsz_lens,
                index_matcher_hash, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d,
                pixs_galind_bounds_lens, rshift_pixs_galind_bounds_lens,
                pix_gals_lens, rshift_pix_gals_lens, zbin_lens_resos,
                pos1_lens_resos, pos2_lens_resos, dpix1_resos, dpix2_resos,
                matchers_resoshift, len_matcher, &hashpix_start1, &hashpix_start2, pix2redpix);

            // Setup all shift variables for the Gncache in the region
            // Gncache has structure
            // n --> zbin_source --> zbin_lens --> radius
            //   --> [ [0]*ngal_zbin1_reso1 | [0]*ngal_zbin1_reso1/2 | ... | [0]*ngal_zbin1_reson ]
            int *cumresoshift_z = calloc(nbinsz_lens*(nresos+1), sizeof(int));
            int *thetashifts_z = calloc(nbinsz_lens, sizeof(int));
            int *zbinshifts = calloc(nbinsz_lens+1, sizeof(int));
            int zbin2shift, nshift_cache;
            setup_region_shifts(nbinsz_lens, nbinsz_source, nresos, hasdiscrete, nbinsr,
                ngal_in_pix, cumresoshift_z, thetashifts_z, zbinshifts, &zbin2shift, &nshift_cache);
            // Set all the cache indices that are updated in this region to zero
            if (printregdbg2){printf("zbin2shift=%d: nshift_cache=%d: size_max_nshift=%d \n", zbin2shift, nshift_cache, size_max_nshift);}
            for (int _i=0; _i<nnvals_Gn*nshift_cache; _i++){Gncache[_i] = 0; wGncache[_i] = 0; cwGncache[_i] = 0;}
            for (int _i=0; _i<nnvals_Wn*nshift_cache; _i++){ Wncache[_i] = 0; wWncache[_i] = 0;}
            for (int _i=0; _i<nshift_cache; _i++){ Wncache_updates[_i] = 0;}
            int Wncache_totupdates=0;
            
            
            // Now, for each resolution, loop over all the galaxies in the region and
            // allocate the Gn & Nn, as well as their caches for the corresponding 
            // set of radii
            // For elreso in resos
            //.  for gal in reso 
            //.    allocate Gn for allowed radii
            //.    allocate the Gncaches
            //.    compute the Upsilon for all combinations of the same resolution
            int ind_pix1, ind_pix2, ind_inpix1, ind_inpix2, ind_red, ind_gal1, ind_gal2, z_gal1, z_gal2;
            int ind_Gncacheshift, ind_Wncacheshift;
            int nbinszr_reso;
            double innergal, pos1_gal1, pos2_gal1, pos1_gal2, pos2_gal2, w_gal1, w_gal2, e1_gal2, e2_gal2;
            double rel1, rel2, dist;
            double complex wshape_gal2;
            double complex nphirot, phirot;
            double rmin_reso, rmax_reso, rmin_reso_sq, rmax_reso_sq;
            int elreso_leaf, rbinmin, rbinmax, rbinmin1, rbinmax1, rbinmin2, rbinmax2;
            
            for (int elreso=0;elreso<nresos;elreso++){
                
                elreso_leaf = mymin(mymax(minresoind_leaf,elreso+resoshift_leafs),maxresoind_leaf);
                //elreso_leaf = elreso;
                rbinmin = reso_rindedges[elreso];
                rbinmax = reso_rindedges[elreso+1];
                rmin_reso = rmin*exp(rbinmin*drbin);
                rmax_reso = rmin*exp(rbinmax*drbin);
                rmin_reso_sq = rmin_reso*rmin_reso;
                rmax_reso_sq = rmax_reso*rmax_reso;
                int nbinsr_reso = rbinmax-rbinmin;
                nbinszr_reso = nbinsz_source*nbinsr_reso;
                lower1 = pixs_galind_bounds_lens[rshift_pixs_galind_bounds_lens[elreso]+elregion];
                upper1 = pixs_galind_bounds_lens[rshift_pixs_galind_bounds_lens[elreso]+elregion+1];
                double complex *thisWns =  calloc(nnvals_Wn*nbinszr_reso, sizeof(double complex));
                double complex *thisGns =  calloc(nnvals_Gn*nbinszr_reso, sizeof(double complex));
                double complex *thisG2ns =  calloc(2*nbinszr_reso, sizeof(double complex));
                double complex *thisW2ns =  calloc(nbinszr_reso, sizeof(double complex));
                int *thisncounts = calloc(nbinszr_reso, sizeof(int));
                int *allowedrinds = calloc(nbinszr_reso, sizeof(int));
                int *allowedzinds = calloc(nbinszr_reso, sizeof(int));
                int ind_Wnp, ind_Wnm, ind_Gnp, ind_Gnm, ind_counts, z1shift, z2rshift, rbin;
                for (ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                    ind_gal1 = rshift_pix_gals_lens[elreso] + pix_gals_lens[rshift_pix_gals_lens[elreso]+ind_inpix1];
                    innergal = isinner_lens_resos[ind_gal1];
                    if (innergal<1e-5){continue;}
                    z_gal1 = zbin_lens_resos[ind_gal1];
                    pos1_gal1 = pos1_lens_resos[ind_gal1];
                    pos2_gal1 = pos2_lens_resos[ind_gal1];
                    w_gal1 = innergal*w_lens_resos[ind_gal1];
                    z1shift = z_gal1*nbinsz_source*nbinsr;
                    
                    int pix1_lower = mymax(0, (int) floor((pos1_gal1 - (rmax_reso+pix1_d) - pix1_start)/pix1_d));
                    int pix2_lower = mymax(0, (int) floor((pos2_gal1 - (rmax_reso+pix2_d) - pix2_start)/pix2_d));
                    int pix1_upper = mymin(pix1_n-1, (int) floor((pos1_gal1 + (rmax_reso+pix1_d) - pix1_start)/pix1_d));
                    int pix2_upper = mymin(pix2_n-1, (int) floor((pos2_gal1 + (rmax_reso+pix2_d) - pix2_start)/pix2_d));
                    for (ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                        for (ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                            ind_red = index_matcher_source[rshift_index_matcher_source[elreso_leaf] + ind_pix2*pix1_n + ind_pix1];
                            if (ind_red==-1){continue;}
                            lower2 = pixs_galind_bounds_source[rshift_pixs_galind_bounds_source[elreso_leaf]+ind_red];
                            upper2 = pixs_galind_bounds_source[rshift_pixs_galind_bounds_source[elreso_leaf]+ind_red+1];
                            for (ind_inpix2=lower2; ind_inpix2<upper2; ind_inpix2++){
                                ind_gal2 = rshift_pix_gals_source[elreso_leaf] + pix_gals_source[rshift_pix_gals_source[elreso_leaf]+ind_inpix2];
                                
                                pos1_gal2 = pos1_source_resos[ind_gal2];
                                pos2_gal2 = pos2_source_resos[ind_gal2];
                                rel1 = pos1_gal2 - pos1_gal1;
                                rel2 = pos2_gal2 - pos2_gal1;
                                dist = rel1*rel1 + rel2*rel2;
                                if(dist < rmin_reso_sq || dist >= rmax_reso_sq) continue;
                                w_gal2 = w_source_resos[ind_gal2];
                                z_gal2 = zbin_source_resos[ind_gal2];
                                e1_gal2 = e1_source_resos[ind_gal2];
                                e2_gal2 = e2_source_resos[ind_gal2];
                                wshape_gal2 = w_gal2*(e1_gal2+I*e2_gal2);
                                
                                dist = sqrt(dist);
                                rbin = (int) floor((log(dist)-logrmin)/drbin);
                                z2rshift = z_gal2*nbinsr_reso + rbin - rbinmin;
                                ind_counts = counts_threadshift + z1shift + z_gal2*nbinsr + rbin;
                                
                                // New
                                phirot = (rel1+I*rel2)/dist;
                                thisncounts[z2rshift] += 1;
                                tmpwcounts[ind_counts] += w_gal1*w_gal2*dist; 
                                tmpwnorms[ind_counts] += w_gal1*w_gal2; 
                                thisG2ns[z2rshift] += w_gal1*wshape_gal2*wshape_gal2*conj(phirot*phirot*phirot*phirot);
                                thisG2ns[nbinszr_reso+z2rshift] += w_gal1*wshape_gal2*conj(wshape_gal2);
                                thisW2ns[z2rshift] += w_gal1*w_gal2*w_gal2;
                                
                                // n=0
                                ind_Wnp = nmax*nbinszr_reso + z2rshift;
                                ind_Wnm = ind_Wnp;
                                ind_Gnp = (nmax+2)*nbinszr_reso+z2rshift;
                                ind_Gnm = ind_Gnp;
                                nphirot = 1;
                                thisGns[ind_Gnp] += wshape_gal2;
                                thisWns[ind_Wnp] += w_gal2;
                                // n \in {-nmax, ..., -1, 1, ...,  nmax}
                                for (int nextn=1;nextn<=nmax;nextn++){
                                    nphirot *= phirot; 
                                    ind_Wnp += nbinszr_reso;
                                    ind_Wnm -= nbinszr_reso;
                                    ind_Gnp += nbinszr_reso;
                                    ind_Gnm -= nbinszr_reso;
                                    thisGns[ind_Gnp] += wshape_gal2*nphirot;
                                    thisGns[ind_Gnm] += wshape_gal2*conj(nphirot);
                                    thisWns[ind_Wnp] += w_gal2*nphirot;
                                    thisWns[ind_Wnm] += w_gal2*conj(nphirot);
                                }
                                // n \in {-nmax-2, -nmax-1, nmax+1, nmax+2}
                                nphirot *= phirot; 
                                ind_Gnp += nbinszr_reso;
                                ind_Gnm -= nbinszr_reso;
                                thisGns[ind_Gnp] += wshape_gal2*nphirot;
                                thisGns[ind_Gnm] += wshape_gal2*conj(nphirot);
                                nphirot *= phirot; 
                                ind_Gnp += nbinszr_reso;
                                ind_Gnm -= nbinszr_reso;
                                thisGns[ind_Gnp] += wshape_gal2*nphirot;
                                thisGns[ind_Gnm] += wshape_gal2*conj(nphirot);
                            }
                        }
                    }
                    
                    // Update the Gncache and Gnnormcache
                    // Gncache in range [-1, .., nmax+1]
                    // Nncache in range [0, ..., nmax]
                    int red_reso2, npix_side_reso2, elhashpix_1_reso2, elhashpix_2_reso2, elhashpix_reso2, redpix_reso2;
                    double complex thisGn, thisNn;
                    int _tmpindcache, _tmpindGn, _tmpindWn, zrshift;
                    for (int elreso2=elreso; elreso2<nresos; elreso2++){
                        red_reso2 = elreso2 - hasdiscrete;
                        if (hasdiscrete==1 && elreso==0 && elreso2==0){red_reso2 += hasdiscrete;}
                        npix_side_reso2 = 1 << (nresos_grid-red_reso2-1);
                        elhashpix_1_reso2 = (int) floor((pos1_gal1 - hashpix_start1)/dpix1_resos[red_reso2]);
                        elhashpix_2_reso2 = (int) floor((pos2_gal1 - hashpix_start2)/dpix2_resos[red_reso2]);
                        elhashpix_reso2 = elhashpix_2_reso2*npix_side_reso2 + elhashpix_1_reso2;
                        redpix_reso2 = pix2redpix[z_gal1*len_matcher+matchers_resoshift[red_reso2]+elhashpix_reso2];
                        for (int zbin2=0; zbin2<nbinsz_source; zbin2++){
                            //if (elregion==-1){
                            //    printf("Gnupdates for elregion=%d reso1=%d reso2=%d red_reso2=%d, galindex=%d, z1=%d, z2=%d:%d radial updates; shiftstart %d = %d+%d+%d+%d+%d, size_max_nshift=%d\n"
                            //           ,elregion,elreso,elreso2,red_reso2,ind_gal1,z_gal1,zbin2,rbinmax-rbinmin,
                            //           zbin2*zbin2shift + zbinshifts[z_gal1] + rbinmin*thetashifts_z[z_gal1] + 
                            //           cumresoshift_z[z_gal1*(nresos+1) + elreso2] + redpix_reso2,
                            //           zbin2*zbin2shift, zbinshifts[z_gal1], rbinmin*thetashifts_z[z_gal1],
                            //           cumresoshift_z[z_gal1*(nresos+1) + elreso2], redpix_reso2, size_max_nshift);
                            //}
                            for (int thisrbin=rbinmin; thisrbin<rbinmax; thisrbin++){
                                zrshift = zbin2*nbinsr_reso + thisrbin-rbinmin;
                                if (cabs(thisWns[nbinszr_reso+zrshift])<1e-10){continue;}
                                ind_Gncacheshift = zbin2*zbin2shift + zbinshifts[z_gal1] + thisrbin*thetashifts_z[z_gal1] + 
                                    cumresoshift_z[z_gal1*(nresos+1) + elreso2] + redpix_reso2;
                                _tmpindGn = zrshift;
                                _tmpindcache = ind_Gncacheshift;
                                for(int thisn=0; thisn<nnvals_Gn; thisn++){
                                    thisGn = thisGns[_tmpindGn];
                                    Gncache[_tmpindcache] += thisGn;
                                    wGncache[_tmpindcache] += w_gal1*thisGn;
                                    //cwGncache[_tmpindcache] += conj(w_gal1)*thisGn;
                                    _tmpindGn += nbinszr_reso;
                                    _tmpindcache += nshift_cache;
                                }
                                _tmpindWn = zrshift;
                                _tmpindcache = ind_Gncacheshift;
                                for(int thisn=0; thisn<nnvals_Wn; thisn++){
                                    thisNn = thisWns[_tmpindWn];
                                    Wncache[_tmpindcache] += thisNn;
                                    wWncache[_tmpindcache] += w_gal1*thisNn;
                                    _tmpindWn += nbinszr_reso;
                                    _tmpindcache += nshift_cache;
                                }
                                Wncache_updates[ind_Gncacheshift] += 1;
                                Wncache_totupdates += 1;
                                //printf("Done Gns/Nns for rbin %d/%d",thisrbin,rbinmax-rbinmin);
                            }
                        } 
                    }
                    
                    //if (elregion==-1){printf("Doing rbin-thinning (same reso)");}
                    // Allocate same reso Upsilon
                    // First check for zero count bins (most likely only in discrete-discrete bit)
                    int nallowedcounts = 0;
                    for (int zbin1=0; zbin1<nbinsz_source; zbin1++){
                        for (int elb1=0; elb1<nbinsr_reso; elb1++){
                            zrshift = zbin1*nbinsr_reso + elb1;
                            if (thisncounts[zbin1*nbinsr_reso + elb1] != 0){
                                allowedrinds[nallowedcounts] = elb1;
                                allowedzinds[nallowedcounts] = zbin1;
                                nallowedcounts += 1;
                            }
                        }
                    }
                    
                    //if (elregion==region_debug){printf("Doing Upsilon update (same reso, ncounts=%d)",nallowedcounts);}
                    // Now update the Upsilon_n
                    // tmpUpsilon have shape (nthreads, 2, 2*nmax+1, nz_lens, nz_source, nz_source, nbinsr, nbinsr)
                    // Gns have shape (nmax+5, nbinsz_source, nbinsr)
                    // Upsilon_-(thet1, thet2) ~ w * G_{+n-2}(thet1) * G_{-n-2}(thet2) - delta^K_{thet1,thet2} * (w * (we)^2*exp(-4*phi))
                    // Upsilon_+(thet1, thet2) ~ w * G_{+n-2}(thet1) * conj(G_{+n-2})(thet2) - delta^K_{thet1,thet2} * (w * |we|^2)
                    // Norm(thet1, thet2)    ~   w  * W_{n}(thet1)   * W_{-n}(thet2)   - delta^K_{thet1,thet2} * (w  * w*w)
                    for (int thisn=-nmax; thisn<=nmax; thisn++){
                        int elb1_full, elb2_full, z3r2shift, gammashift_ups, gammashift_norm;
                        int _wind, _upsind1p, _upsind1m, _upsind2p, _upsind2m, zrshift, _zcombi, zcombi, elb1, zbin2, elb2, zbin3;
                        double complex nextUpsp, nextUpsm, nextN;
                        int thisnshift_ups = upsilon_threadshift + (nmax+thisn)*upsilon_nshift;
                        int thisnshift_norm = norm_threadshift + (nmax+thisn)*upsilon_nshift;
                        for (int zrcombis1=0; zrcombis1<nallowedcounts; zrcombis1++){
                            elb1 = allowedrinds[zrcombis1];
                            zbin2 = allowedzinds[zrcombis1];
                            elb1_full = elb1 + rbinmin;
                            zrshift = zbin2*nbinsr_reso + elb1;
                            // Double counting correction
                            if (dccorr==1){
                                zcombi = z_gal1*nbinsz_source*nbinsz_source + zbin2*nbinsz_source + zbin2;
                                gammashift_ups = thisnshift_ups + zcombi*upsilon_zshift + elb1_full*nbinsr+elb1_full;
                                gammashift_norm = thisnshift_norm + zcombi*upsilon_zshift + elb1_full*nbinsr+elb1_full;
                                tmpUpsilon[gammashift_ups] -= thisG2ns[zrshift];
                                tmpUpsilon[upsilon_compshift+gammashift_ups] -= thisG2ns[nbinszr_reso+zrshift];
                                tmpNorm[gammashift_norm] -= thisW2ns[zrshift];  
                            }
                            _zcombi = z_gal1*nbinsz_source*nbinsz_source + zbin2*nbinsz_source;
                            _wind = (nmax+thisn)*nbinszr_reso+zrshift;
                            _upsind1m = (nmax+0+thisn)*nbinszr_reso+zrshift; // For Upsilon-: n-2; Gn from [nmax-2,...,nmax+2]
                            _upsind1p = (nmax+0+thisn)*nbinszr_reso+zrshift; // For Upsilon+: n-2; Gn from [nmax-2,...,nmax+2]
                            nextUpsp = w_gal1*thisGns[_upsind1p];
                            nextUpsm = w_gal1*thisGns[_upsind1m];
                            nextN = w_gal1*thisWns[_wind];
                            for (int zrcombis2=0; zrcombis2<nallowedcounts; zrcombis2++){
                                elb2 = allowedrinds[zrcombis2];
                                zbin3 = allowedzinds[zrcombis2];
                                elb2_full = elb2 + rbinmin;
                                zrshift = zbin3*nbinsr_reso + elb2;
                                zcombi = _zcombi + zbin3;
                                z3r2shift = zcombi*upsilon_zshift + elb1_full*nbinsr + elb2_full;
                                gammashift_ups = thisnshift_ups + z3r2shift;
                                gammashift_norm = thisnshift_norm + z3r2shift;
                                _wind =     (nmax-thisn)*nbinszr_reso + zrshift;
                                _upsind2p = (nmax+thisn+0)*nbinszr_reso + zrshift; // For Upsilon+: +n-2; Gn from [nmax-2,...,nmax+2]
                                _upsind2m = (nmax-thisn+0)*nbinszr_reso + zrshift; // For Upsilon-: -n-2; Gn from [nmax-2,...,nmax+2]
                                tmpUpsilon[gammashift_ups] += nextUpsm*thisGns[_upsind2m]; // Upsilon-
                                tmpUpsilon[upsilon_compshift+gammashift_ups] += nextUpsp*conj(thisGns[_upsind2p]);// Upsilon+
                                tmpNorm[gammashift_norm] += nextN*thisWns[_wind];
                            }
                        }
                    }
                    //if(elregion%100==0){printf("Setting stuff to 0 for region %d",elregion);}
                    for (int _i=0;_i<nnvals_Wn*nbinszr_reso;_i++){thisWns[_i]=0;}
                    for (int _i=0;_i<nnvals_Gn*nbinszr_reso;_i++){thisGns[_i]=0;}
                    for (int _i=0;_i<2*nbinszr_reso;_i++){thisG2ns[_i]=0;}
                    for (int _i=0;_i<nbinszr_reso;_i++){
                        thisW2ns[_i]=0; thisncounts[_i]=0; allowedrinds[_i]=0; allowedzinds[_i]=0;}
                    //if(elregion%100==0){printf("Finished stuff to 0 for region %d",elregion);}
                }
                //if(elregion%100==0)printf("Freeing stuff for region %d",elregion);
                free(thisGns);
                free(thisWns);
                free(thisG2ns);
                free(thisW2ns);
                free(thisncounts);
                free(allowedrinds);
                free(allowedzinds);
            }
            // Allocate the Upsilon/Norms for different grid resolutions from all the cached arrays 
            //
            // Note that for different configurations of the resolutions we do the Gamman
            // allocation as follows - see eq. (xx) in yyy.zzz for the reasoning:
            // * Upsilon_- = wshape  * G_nm2 * G_mnm2
            //          --> (wG_nm2) * G_mnm2    if reso1 < reso2
            //          -->  G_nm2   * wG_mnm2   if reso1 > reso2
            // * Upsilon_+ = wshape  * G_nm2 * conj(G_nm2)
            //          --> (wG_nm2) * conj(G_nm2)    if reso1 < reso2
            //          -->  G_nm2   * conj(wG_nm2)   if reso1 > reso2
            // * Norm   =  w * W_n * conj(W_n)
            //          --> wW_n * conj(W_n)  if reso1 < reso2
            //          --> W_n  * conj(wW_n) if reso1 > reso2
            // where wW_xxx := w(shape)*W_xxx and cwG_xxx := conj(w(shape))*G_xxx
            double complex nextUpsp, nextUpsm, nextN;
            int zcombi;
            //if(elregion==region_debug)printf("Allocating different reso stuff for region %d",elregion);
            for (int thisn=-nmax; thisn<=nmax; thisn++){
                int _upsshift, _normshift;
                //int thisnshift = upsilon_threadshift + (nmax+thisn)*upsilon_nshift;
                int thisnshift_ups = upsilon_threadshift + (nmax+thisn)*upsilon_nshift;
                int thisnshift_norm = norm_threadshift + (nmax+thisn)*upsilon_nshift;
                for (int zbin1=0; zbin1<nbinsz_lens; zbin1++){
                    for (int zbin2=0; zbin2<nbinsz_source; zbin2++){
                        for (int zbin3=0; zbin3<nbinsz_source; zbin3++){
                            zcombi = zbin1*nbinsz_source*nbinsz_source + zbin2*nbinsz_source + zbin3;
                            int _thetashift_z = thetashifts_z[zbin1]; // This is basically shift for theta_i --> theta_{i+1}
                            //if (zcombis_allowed[zcombi]==0){continue;}
                            // Case max(reso1, reso2) = reso2
                            for (int thisreso1=0; thisreso1<nresos; thisreso1++){
                                rbinmin1 = reso_rindedges[thisreso1];
                                rbinmax1 = reso_rindedges[thisreso1+1];
                                for (int thisreso2=thisreso1+1; thisreso2<nresos; thisreso2++){
                                    rbinmin2 = reso_rindedges[thisreso2];
                                    rbinmax2 = reso_rindedges[thisreso2+1];
                                    for (int elgal=0; elgal<ngal_in_pix[zbin1*nresos+thisreso2]; elgal++){
                                        for (int elb1=rbinmin1; elb1<rbinmax1; elb1++){
                                            // n --> zbin2 --> zbin1 --> radius --> [ [0]*ngal_zbin1_reso1 | ... |
                                            //                                        | ...  | [0]*ngal_zbin1_reson ]
                                            ind_Wncacheshift = zbin2*zbin2shift + zbinshifts[zbin1] + elb1*thetashifts_z[zbin1]+
                                                cumresoshift_z[zbin1*(nresos+1) + thisreso2] + elgal;
                                            nextUpsp = wGncache[(nmax+thisn+0)*nshift_cache+ind_Wncacheshift];
                                            nextUpsm = wGncache[(nmax+thisn+0)*nshift_cache+ind_Wncacheshift];
                                            nextN = wWncache[(nmax+thisn)*nshift_cache+ind_Wncacheshift];
                                            _upsshift = thisnshift_ups + zcombi*upsilon_zshift + elb1*nbinsr;
                                            _normshift = thisnshift_norm+ zcombi*upsilon_zshift + elb1*nbinsr;
                                            ind_Wncacheshift = zbin3*zbin2shift+zbinshifts[zbin1]+rbinmin2*thetashifts_z[zbin1]+
                                                cumresoshift_z[zbin1*(nresos+1) + thisreso2] + elgal;
                                            for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                                tmpUpsilon[_upsshift+elb2] += nextUpsm *  
                                                    Gncache[(nmax-thisn+0)*nshift_cache+ind_Wncacheshift];
                                                tmpUpsilon[_upsshift+upsilon_compshift+elb2] += nextUpsp *  
                                                    conj(Gncache[(nmax+thisn+0)*nshift_cache+ind_Wncacheshift]);
                                                tmpNorm[_normshift+elb2] += nextN * 
                                                    conj(Wncache[(nmax+thisn)*nshift_cache+ind_Wncacheshift]);
                                                ind_Wncacheshift += _thetashift_z;
                                                ind_Gncacheshift += _thetashift_z;
                                            }
                                        }
                                    }
                                }
                            }
                            
                            
                            // Case max(reso1, reso2) = reso1
                            for (int thisreso2=0; thisreso2<nresos; thisreso2++){
                                rbinmin2 = reso_rindedges[thisreso2];
                                rbinmax2 = reso_rindedges[thisreso2+1];
                                for (int thisreso1=thisreso2+1; thisreso1<nresos; thisreso1++){
                                    rbinmin1 = reso_rindedges[thisreso1];
                                    rbinmax1 = reso_rindedges[thisreso1+1];
                                    for (int elgal=0; elgal<ngal_in_pix[zbin1*nresos+thisreso1]; elgal++){
                                        for (int elb1=rbinmin1; elb1<rbinmax1; elb1++){
                                            ind_Wncacheshift = zbin2*zbin2shift + zbinshifts[zbin1] + elb1*thetashifts_z[zbin1]+
                                                cumresoshift_z[zbin1*(nresos+1) + thisreso1] + elgal;
                                            nextUpsp = Gncache[(nmax+thisn+0)*nshift_cache+ind_Wncacheshift];
                                            nextUpsm = Gncache[(nmax+thisn+0)*nshift_cache+ind_Wncacheshift];
                                            nextN = Wncache[(nmax+thisn)*nshift_cache+ind_Wncacheshift];
                                            _upsshift = thisnshift_ups + zcombi*upsilon_zshift + elb1*nbinsr;
                                            _normshift = thisnshift_norm+ zcombi*upsilon_zshift + elb1*nbinsr;
                                            ind_Wncacheshift = zbin3*zbin2shift+zbinshifts[zbin1]+rbinmin2*thetashifts_z[zbin1]+
                                                cumresoshift_z[zbin1*(nresos+1) + thisreso1] + elgal;
                                            for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                                tmpUpsilon[_upsshift+elb2] += nextUpsm *
                                                    wGncache[(nmax-thisn+0)*nshift_cache+ind_Wncacheshift];
                                                tmpUpsilon[_upsshift+upsilon_compshift+elb2] += nextUpsp *
                                                    conj(wGncache[(nmax+thisn+0)*nshift_cache+ind_Wncacheshift]);
                                                tmpNorm[_normshift+elb2] += nextN * 
                                                    conj(wWncache[(nmax+thisn)*nshift_cache+ind_Wncacheshift]);
                                                ind_Wncacheshift += _thetashift_z;
                                                ind_Gncacheshift += _thetashift_z;
                                            }
                                        }
                                    }
                                }
                            }
                            
                        }
                    }
                }
            } 
            free(reso_rindedges);
            free(rshift_index_matcher_source);
            free(rshift_pixs_galind_bounds_source);
            free(rshift_pix_gals_source);
            free(rshift_index_matcher_lens);
            free(rshift_pixs_galind_bounds_lens);
            free(rshift_pix_gals_lens);
            free(matchers_resoshift);
            free(ngal_in_pix);
            free(pix2redpix);  
            free(cumresoshift_z);
            free(thetashifts_z);
            free(zbinshifts);

            print_progress(nregionsdone, nregions, verbose);
        }
        free(Gncache);
        free(wGncache);
        free(cwGncache);
        free(Wncache);
        free(wWncache);
        free(Wncache_updates);
    }
    
    // Accumulate the Upsilon_n / N_n
    #pragma omp parallel for num_threads(nthreads)
    for (int thisn=0; thisn<2*nmax+1; thisn++){
        int iUps;
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            int thisthreadshift = thisthread*_upsilonthreadshift;
            for (int zcombi=0; zcombi<_nzcombis; zcombi++){
                for (int elb1=0; elb1<nbinsr; elb1++){
                    for (int elb2=0; elb2<nbinsr; elb2++){
                        iUps = thisn*_upsilonnshift + zcombi*_upsilonzshift + elb1*nbinsr + elb2;
                        Upsilon_n[iUps] += tmpUpsilon[thisthreadshift+iUps];
                        Upsilon_n[_upsiloncompshift+iUps] += tmpUpsilon[thisthreadshift+_upsiloncompshift+iUps];
                        Norm_n[iUps] += tmpNorm[thisthread*_normthreadshift+iUps];
                    }
                }
            }
        }
    }
    
    // Accumulate the bin distances and weights
    for (int thisthread=0; thisthread<nthreads; thisthread++){
        int tmpind;
        int thisthreadshift = thisthread*nbinsz_lens*nbinsz_source*nbinsr; 
        for (int elbinz=0; elbinz<nbinsz_lens*nbinsz_source; elbinz++){
            for (int elbinr=0; elbinr<nbinsr; elbinr++){
                tmpind = elbinz*nbinsr + elbinr;
                totcounts[tmpind] += tmpwcounts[thisthreadshift+tmpind];
                totnorms[tmpind] += tmpwnorms[thisthreadshift+tmpind];
            }
        }
    }
    
    // Get bin centers
    for (int elbinz=0; elbinz<nbinsz_lens*nbinsz_source; elbinz++){
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            int tmpind = elbinz*nbinsr + elbinr;
            if (totnorms[tmpind] != 0){
                bin_centers[tmpind] = totcounts[tmpind]/totnorms[tmpind];
            }
        }
    }
    free(tmpUpsilon);
    free(tmpNorm);
    free(tmpwcounts);
    free(tmpwnorms);
    free(totcounts);
    free(totnorms);
    free(regionsdone);
}