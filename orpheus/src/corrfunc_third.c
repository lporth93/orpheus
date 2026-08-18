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


///////////////////////
/// General helpers ///
///////////////////////

// (A) Shared multi-resolution region setup (used for flat basetree/doubletree kernels). //

// Find radial indices for which resolutions are swapped
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

// Cumulative per-reso offsets into the stacked multihash arrays.
// Only useable for flat geometries using tree/basetree/doubletree approximations
static void build_rshift_offsets(int nresos, int npix_hash, const int *ngal_resos,
    int *rshift_index_matcher, int *rshift_pixs_galind_bounds, int *rshift_pix_gals){
    for (int elreso=1;elreso<nresos;elreso++){
        rshift_index_matcher[elreso] = rshift_index_matcher[elreso-1] + npix_hash;
        rshift_pixs_galind_bounds[elreso] = rshift_pixs_galind_bounds[elreso-1] + ngal_resos[elreso-1]+1;
        rshift_pix_gals[elreso] = rshift_pix_gals[elreso-1] + ngal_resos[elreso-1];
    }
}

// Get per-region galaxy counts per (zbin, reso) of the hashed base catalog and
// cumulative pixel offsets of the reduced grids.
// Only usable for flat geometries using basetree/doubletree approximations
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
// resolution, plus the regions hash-pixel origin.
// Only usable for flat geometries using basetree/doubletree approximations
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

// Region cache-slot layout from the base catalog's per-(zbin, reso) counts.
// The discrete band shares the reso-1 grid slots, so its own count is skipped. That
// sharing needs a reso 1 to exist: with a single resolution the discrete band counts its
// own galaxies, as ngal_in_pix only holds nresos entries per zbin.
// Only used for flat geometries and doubletree approximations
static void setup_region_shifts(int nbinsz_base, int nbinsz_leaf, int nresos,
    int hasdiscrete, int nbinsr, const int *ngal_in_pix,
    int *cumresoshift_z, int *thetashifts_z, int *zbinshifts,
    int *zbin2shift, int *nshift){
    for (int elz=0; elz<nbinsz_base; elz++){
        for (int elreso=0; elreso<nresos; elreso++){
            if (hasdiscrete==1 && elreso==0 && nresos>1){
                cumresoshift_z[elz*(nresos+1) + elreso+1] = ngal_in_pix[elz*nresos + elreso+1];
            } else {
                cumresoshift_z[elz*(nresos+1) + elreso+1] =
                    cumresoshift_z[elz*(nresos+1) + elreso] + ngal_in_pix[elz*nresos + elreso];
            }
        }
        thetashifts_z[elz] = cumresoshift_z[elz*(nresos+1) + nresos];
        zbinshifts[elz+1] = zbinshifts[elz] + nbinsr*thetashifts_z[elz];
    }
    *zbin2shift = zbinshifts[nbinsz_base];
    *nshift = nbinsz_leaf*(*zbin2shift);
}

// Dense per-zbin cache index of a base's coarse reduced galaxy at every
// resolution elreso2 >= elreso. Discrete band shares the reso-1 grid.
// Only used for flat geometries and bastree/doubletree approximations
static inline void build_redpix_by_reso2(int elreso, int nresos, int nresos_grid,
    int hasdiscrete, int z_gal1, double pos1_gal1, double pos2_gal1,
    double hashpix_start1, double hashpix_start2,
    const double *dpix1_resos, const double *dpix2_resos,
    const int *matchers_resoshift, int len_matcher, const int *pix2redpix,
    int *redpix_by_reso2){
    // The reduced pixels and this cache exist only for *_accum_crossreso, which has no
    // resolution pair to visit for a single-resolution tree. Returning here also keeps the
    // grid arrays, which are empty in that case, from being indexed at all.
    if (nresos<=1){return;}
    for (int elreso2=elreso; elreso2<nresos; elreso2++){
        int red_reso2 = elreso2 - hasdiscrete;
        if (hasdiscrete==1 && elreso==0 && elreso2==0){red_reso2 += hasdiscrete;}
        int npix_side_reso2 = 1 << (nresos_grid-red_reso2-1);
        int elhashpix_1_reso2 = (int) floor((pos1_gal1 - hashpix_start1)/dpix1_resos[red_reso2]);
        int elhashpix_2_reso2 = (int) floor((pos2_gal1 - hashpix_start2)/dpix2_resos[red_reso2]);
        int elhashpix_reso2 = elhashpix_2_reso2*npix_side_reso2 + elhashpix_1_reso2;
        redpix_by_reso2[elreso2] = pix2redpix[z_gal1*len_matcher+matchers_resoshift[red_reso2]+elhashpix_reso2];
    }
}

//////////////////////////////////
/// CORRELATOR SPECIFIC HELPERS //
//////////////////////////////////

// (A) NNN HELPERS //

// Context for the scalar NNN doubletree kernel.
typedef struct {
    int nbinsz, nbinsr, nmax, nresos;
    int nnvals_Nn;
    int gamma_zshift, gamma_nshift, gamma_compshift;
    int dccorr, elthread;
    int *reso_rindedges, *ngal_in_pix, *cumresoshift_z, *thetashifts_z, *zbinshifts;
    int zbin2shift, nshift;
    double complex *Nncache, *wNncache;
    double complex *tmpTriplets_n;
} NnnContext;

static void nnn_zero_caches(NnnContext *c){
    for (int _i=0; _i<c->nnvals_Nn*c->nshift; _i++){c->Nncache[_i]=0; c->wNncache[_i]=0;}
}

// Scatter the multipoles nextWns into the region caches at higher resos
static void nnn_update_nncache(NnnContext *c, int elreso, int rbinmin, int rbinmax,
    int nbinsr_reso, int z_gal1, double w_gal1,
    const int *redpix_by_reso2, const double complex *nextWns){
    // The reduced pixels and this cache exist only for *_accum_crossreso, which has no
    // resolution pair to visit for a single-resolution tree. Returning here also keeps the
    // grid arrays, which are empty in that case, from being indexed at all.
    if (c->nresos<=1){return;}
    int nbinszr_reso = c->nbinsz*nbinsr_reso;
    for (int elreso2=elreso; elreso2<c->nresos; elreso2++){
        int redpix_reso2 = redpix_by_reso2[elreso2];
        for (int zbin2=0; zbin2<c->nbinsz; zbin2++){
            for (int thisrbin=rbinmin; thisrbin<rbinmax; thisrbin++){
                int zrshift = zbin2*nbinsr_reso + thisrbin-rbinmin;
                if (cabs(nextWns[zrshift])<1e-10){continue;}
                int ind_Nncacheshift = zbin2*c->zbin2shift + c->zbinshifts[z_gal1] +
                    thisrbin*c->thetashifts_z[z_gal1] +
                    c->cumresoshift_z[z_gal1*(c->nresos+1) + elreso2] + redpix_reso2;
                int _tmpindWn = zrshift;
                int _tmpindcache = ind_Nncacheshift;
                for(int thisn=0; thisn<c->nnvals_Nn; thisn++){
                    double complex thisWn = nextWns[_tmpindWn];
                    c->Nncache[_tmpindcache] += thisWn;
                    c->wNncache[_tmpindcache] += w_gal1*thisWn;
                    _tmpindWn += nbinszr_reso;
                    _tmpindcache += c->nshift;
                }
            }
        }
    }
}

// Same-resolution multipole allocation for a region. 
// Equation: Norm_n ~ w*N_n(t1)*conj(N_n)(t2) - w*W2n delta_{t1,t2},
static void nnn_accum_samereso(NnnContext *c, int rbinmin, int nbinsr_reso,
    int z_gal1, double w_gal1, const double complex *nextWns,
    const double complex *nextW2ns, const int *nextncounts,
    int *allowedrinds, int *allowedzinds){
    int nbinsz=c->nbinsz, nbinsr=c->nbinsr, nmax=c->nmax;
    int nbinszr_reso = nbinsz*nbinsr_reso;
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
                c->tmpTriplets_n[gammashift1 + elb1_full] -= w_gal1*nextW2ns[zrshift];
            }
            double complex w0 = w_gal1 * nextWns[ind_norm + zrshift];
            int _zcombi = z_gal1*nbinsz*nbinsz+zbin2*nbinsz;
            int _gammashift1 = thisnshift + elb1_full*nbinsr;
            for (int zrcombis2=0; zrcombis2<nallowedcounts; zrcombis2++){
                int zcombi = _zcombi+allowedzinds[zrcombis2];
                int gammashift1 = _gammashift1 + zcombi*c->gamma_zshift;
                int elb2_full = allowedrinds[zrcombis2] + rbinmin;
                int zrshift2 = allowedzinds[zrcombis2]*nbinsr_reso + allowedrinds[zrcombis2];
                c->tmpTriplets_n[gammashift1 + elb2_full] += w0*conj(nextWns[ind_norm + zrshift2]);
            }
        }
    }
}

// Cross-resolution multipole allocation ffor a region
// Equation same as for equal-reso, but the used caches depend on how the resos are ordered:
// reso1<reso2 --> wN_n*conj(N_n) vs.  reso1>reso2 --> N_n*conj(wN_n)
static void nnn_accum_crossreso(NnnContext *c){
    int nbinsz=c->nbinsz, nbinsr=c->nbinsr, nmax=c->nmax, nresos=c->nresos, nshift=c->nshift;
    int elthread=c->elthread, gamma_compshift=c->gamma_compshift;
    int gamma_nshift=c->gamma_nshift, gamma_zshift=c->gamma_zshift, zbin2shift=c->zbin2shift;
    const int *zbinshifts=c->zbinshifts, *thetashifts_z=c->thetashifts_z;
    const int *cumresoshift_z=c->cumresoshift_z, *reso_rindedges=c->reso_rindedges;
    const int *ngal_in_pix=c->ngal_in_pix;
    double complex *Nncache=c->Nncache, *wNncache=c->wNncache;
    double complex *tmpTriplets_n=c->tmpTriplets_n;
    for (int thisn=0; thisn<nmax+1; thisn++){
        int thisnshift = elthread*gamma_compshift + thisn*gamma_nshift;
        for (int zbin1=0; zbin1<nbinsz; zbin1++){
            int zbinshift_z1 = zbinshifts[zbin1], thetashift_z1 = thetashifts_z[zbin1];
            const int *cumresoshift_z1 = cumresoshift_z + zbin1*(nresos+1);
            for (int zbin2=0; zbin2<nbinsz; zbin2++){
                for (int zbin3=0; zbin3<nbinsz; zbin3++){
                    int zcombi = zbin1*nbinsz*nbinsz + zbin2*nbinsz + zbin3;
                    int _thetashift_z = thetashift_z1;
                    // Case max(reso1, reso2) = reso2
                    for (int thisreso1=0; thisreso1<nresos; thisreso1++){
                        int rbinmin1 = reso_rindedges[thisreso1];
                        int rbinmax1 = reso_rindedges[thisreso1+1];
                        for (int thisreso2=thisreso1+1; thisreso2<nresos; thisreso2++){
                            int rbinmin2 = reso_rindedges[thisreso2];
                            int rbinmax2 = reso_rindedges[thisreso2+1];
                            int cumshift2 = cumresoshift_z1[thisreso2];
                            for (int elgal=0; elgal<ngal_in_pix[zbin1*nresos+thisreso2]; elgal++){
                                for (int elb1=rbinmin1; elb1<rbinmax1; elb1++){
                                    int gammashift1 = thisnshift + zcombi*gamma_zshift + elb1*nbinsr;
                                    int ind_Nncacheshift = zbin2*zbin2shift + zbinshift_z1 + elb1*thetashift_z1 +
                                        cumshift2 + elgal;
                                    double complex w0 = wNncache[thisn*nshift + ind_Nncacheshift];
                                    ind_Nncacheshift = zbin3*zbin2shift + zbinshift_z1 + rbinmin2*thetashift_z1 +
                                            cumshift2 + elgal;
                                    int _in = thisn*nshift + ind_Nncacheshift;
                                    for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                        tmpTriplets_n[gammashift1 + elb2] += w0*conj(Nncache[_in]);
                                        _in += _thetashift_z;
                                    }
                                }
                            }
                        }
                    }
                    // Case max(reso1, reso2) = reso1
                    for (int thisreso2=0; thisreso2<nresos; thisreso2++){
                        int rbinmin2 = reso_rindedges[thisreso2];
                        int rbinmax2 = reso_rindedges[thisreso2+1];
                        for (int thisreso1=thisreso2+1; thisreso1<nresos; thisreso1++){
                            int rbinmin1 = reso_rindedges[thisreso1];
                            int rbinmax1 = reso_rindedges[thisreso1+1];
                            int cumshift1 = cumresoshift_z1[thisreso1];
                            for (int elgal=0; elgal<ngal_in_pix[zbin1*nresos+thisreso1]; elgal++){
                                for (int elb1=rbinmin1; elb1<rbinmax1; elb1++){
                                    int gammashift1 = thisnshift + zcombi*gamma_zshift + elb1*nbinsr;
                                    int ind_Nncacheshift = zbin2*zbin2shift + zbinshift_z1 + elb1*thetashift_z1 +
                                        cumshift1 + elgal;
                                    double complex w0 = Nncache[thisn*nshift + ind_Nncacheshift];
                                    ind_Nncacheshift = zbin3*zbin2shift + zbinshift_z1 + rbinmin2*thetashift_z1 +
                                            cumshift1 + elgal;
                                    int _in = thisn*nshift + ind_Nncacheshift;
                                    for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                        tmpTriplets_n[gammashift1 + elb2] += w0*conj(wNncache[_in]);
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
}

// Accumulates the per-thread multipoles and from there computes bin centers and norms
// Note that we pass the counts in both, the out->npcf and the out->norm_mp arrays.
static void nnn_reduce(int nbinsz, int nbinsr, int nmax, int nthreads,
    const double complex *tmpTriplets_n, const double *tmpwcounts,
    const double *tmpwnorms, NPCFOutput *out){
    int nzcombis = nbinsz*nbinsz*nbinsz;
    int gamma_zshift = nbinsr*nbinsr;
    int gamma_nshift = gamma_zshift*nzcombis;
    int gamma_compshift = (nmax+1)*gamma_nshift;
    double complex *Triplets_n = out->npcf;
    double complex *Triplets_norm = out->norm_mp;
    #pragma omp parallel for num_threads(nthreads)
    for (int thisn=0; thisn<nmax+1; thisn++){
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            for (int zcombi=0; zcombi<nzcombis; zcombi++){
                for (int elb1=0; elb1<nbinsr; elb1++){
                    for (int elb2=0; elb2<nbinsr; elb2++){
                        int iGamma = thisn*gamma_nshift + zcombi*gamma_zshift + elb1*nbinsr + elb2;
                        int itmpGamma = iGamma + thisthread*gamma_compshift;
                        double complex v = tmpTriplets_n[itmpGamma];
                        if (Triplets_n) Triplets_n[iGamma] += v;
                        if (Triplets_norm) Triplets_norm[iGamma] += v;
                    }
                }
            }
        }
    }
    double *totcounts = calloc(nbinsz*nbinsr, sizeof(double));
    double *totnorms = calloc(nbinsz*nbinsr, sizeof(double));
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

// Number of trailing zero bits of a power-of-two nside == log2(nside).
static inline int ggg_nside_level(long nside){ return (nside>0) ? __builtin_ctzl((unsigned long)nside) : 0; }

// Lower bound: first index i in cp[0..n) with cp[i] >= key.
static inline int ggg_lower_bound_long(const long *cp, int n, long key){
    int lo=0, hi=n;
    while (lo<hi){ int m=(lo+hi)>>1; if (cp[m]<key){ lo=m+1; } else { hi=m; } }
    return lo;
}

// (B) GGG HELPERS //

// Per-pair fill of the spin-2 shape multipoles G_n and count multipoles W_n
// and their double counting corrections, assuming nmin=0
//  * Used for all approximations in a flat geometry
//  * Shear has already been projected, wshape is rotated by e^{-2i*beta}
static inline void ggg_fill_GnWn_projected(
    double complex *nextGns, double complex *nextWns,
    double complex *nextG2ns, double complex *nextW2ns,
    int zrshift, int nbinszr, int nmax, double w2, double complex wshape,
    double complex phirot, double complex twophirotc){
    int nzero = nmax+1;
    int ind_Gn = nzero*nbinszr + zrshift;
    nextGns[ind_Gn] += wshape;
    nextWns[zrshift] += w2;
    nextG2ns[0*nbinszr+zrshift] += wshape*wshape*twophirotc;
    nextG2ns[1*nbinszr+zrshift] += wshape*wshape*conj(twophirotc);
    nextG2ns[2*nbinszr+zrshift] += wshape*conj(wshape)*twophirotc;
    nextG2ns[3*nbinszr+zrshift] += wshape*conj(wshape)*twophirotc;
    nextW2ns[zrshift] += w2*w2;
    double complex nphirot = phirot;
    double wr = creal(wshape), wi = cimag(wshape);
    for (int m=1; m<=nmax+1; m++){
        double nr = creal(nphirot), ni = cimag(nphirot);
        double a = wr*nr, b = wi*ni, c = wr*ni, d = wi*nr;
        nextGns[ind_Gn + m*nbinszr] += (a-b) + I*(c+d);
        nextGns[ind_Gn - m*nbinszr] += (a+b) + I*(d-c);
        if (m <= nmax){ nextWns[zrshift + m*nbinszr] += w2*nphirot; }
        nphirot *= phirot;
    }
}

// Same as above for nmin>3
// Right now only used in discrete/tree GGG without double conting corrs
static inline void ggg_fill_GnWn_nminband(
    double complex *nextGns, double complex *nextWns,
    int zrshift, int nbinszr, int nmin, int nmax, double w2, double complex wshape,
    double complex phirot, double complex phirotc){
    double complex phirotm = cpow(phirotc,nmax+3);
    double complex phirotp = cpow(phirot,nmin-3);
    double complex phirotn = phirotp*phirot*phirot*phirot;
    int pshift = (nmax-nmin+3)*nbinszr;
    int nextnshift = zrshift;
    for (int nextn=0;nextn<nmax-nmin+1;nextn++){
        nextGns[nextnshift] += wshape*phirotm;
        nextGns[pshift+nextnshift] += wshape*phirotp;
        nextWns[nextnshift] += w2*phirotn;
        phirotm *= phirot;
        phirotp *= phirot;
        phirotn *= phirot;
        nextnshift += nbinszr;
    }
    nextGns[nextnshift] += wshape*phirotm;
    nextGns[pshift+nextnshift] += wshape*phirotp;
    phirotm *= phirot;
    phirotp *= phirot;
    nextnshift += nbinszr;
    nextGns[nextnshift] += wshape*phirotm;
    nextGns[pshift+nextnshift] += wshape*phirotp;
}

// Context shared by the GGG DoubleTree helpers. 
typedef struct {
    int nbinsz, nbinsr, nmax, nresos;
    int nnvals_Gn, nnvals_Nn; // 2*nmax+3, nmax+1
    int gamma_zshift, gamma_nshift, gamma_compshift;
    int dccorr;
    int elthread;
    int *reso_rindedges; 
    int *ngal_in_pix;
    int *cumresoshift_z;
    int *thetashifts_z;
    int *zbinshifts; // [nbinsz+1]
    int zbin2shift, nshift;
    double complex *Gncache, *wGncache, *cwGncache, *Nncache, *wNncache;
    double complex *tmpGamma0s, *tmpGamma1s, *tmpGamma2s, *tmpGamma3s, *tmpGammans_norm;
} GggContext;

// Cache-slot layout for the GGG caches.
static void ggg_setup_shifts(GggContext *c, int hasdiscrete){
    setup_region_shifts(c->nbinsz, c->nbinsz, c->nresos, hasdiscrete, c->nbinsr,
        c->ngal_in_pix, c->cumresoshift_z, c->thetashifts_z, c->zbinshifts,
        &c->zbin2shift, &c->nshift);
}

static void ggg_zero_caches(GggContext *c){
    for (int _i=0; _i<c->nnvals_Gn*c->nshift; _i++){c->Gncache[_i]=0; c->wGncache[_i]=0; c->cwGncache[_i]=0;}
    for (int _i=0; _i<c->nnvals_Nn*c->nshift; _i++){c->Nncache[_i]=0; c->wNncache[_i]=0;}
}

// Scatter the multipoles nextGns/nextWns into the region caches at higher resos
static void ggg_update_gnwncache(GggContext *c, int elreso, int rbinmin, int rbinmax,
    int nbinsr_reso, int z_gal1, double w_gal1, double complex wshape_gal1,
    const int *redpix_by_reso2,
    const double complex *nextGns, const double complex *nextWns){
    // The reduced pixels and this cache exist only for *_accum_crossreso, which has no
    // resolution pair to visit for a single-resolution tree. Returning here also keeps the
    // grid arrays, which are empty in that case, from being indexed at all.
    if (c->nresos<=1){return;}
    int nbinsz = c->nbinsz, nresos = c->nresos, nshift = c->nshift;
    int nnvals_Gn = c->nnvals_Gn, nnvals_Nn = c->nnvals_Nn, zbin2shift = c->zbin2shift;
    int zbinshift_z1 = c->zbinshifts[z_gal1], thetashift_z1 = c->thetashifts_z[z_gal1];
    const int *cumresoshift_z1 = c->cumresoshift_z + z_gal1*(nresos+1);
    double complex *Gncache = c->Gncache, *wGncache = c->wGncache, *cwGncache = c->cwGncache;
    double complex *Nncache = c->Nncache, *wNncache = c->wNncache;
    double complex conj_wshape_gal1 = conj(wshape_gal1);
    int nbinszr_reso = nbinsz*nbinsr_reso;
    for (int elreso2=elreso; elreso2<nresos; elreso2++){
        int cumshift = cumresoshift_z1[elreso2] + redpix_by_reso2[elreso2];
        for (int zbin2=0; zbin2<nbinsz; zbin2++){
            int zbase = zbin2*zbin2shift + zbinshift_z1 + cumshift;
            for (int thisrbin=rbinmin; thisrbin<rbinmax; thisrbin++){
                int zrshift = zbin2*nbinsr_reso + thisrbin-rbinmin;
                if (cabs(nextWns[zrshift])<1e-10){continue;}
                int ind_Gncacheshift = zbase + thisrbin*thetashift_z1;
                int _tmpindGn = zrshift;
                int _tmpindcache = ind_Gncacheshift;
                for(int thisn=0; thisn<nnvals_Gn; thisn++){
                    double complex thisGn = nextGns[_tmpindGn];
                    Gncache[_tmpindcache] += thisGn;
                    wGncache[_tmpindcache] += wshape_gal1*thisGn;
                    cwGncache[_tmpindcache] += conj_wshape_gal1*thisGn;
                    _tmpindGn += nbinszr_reso;
                    _tmpindcache += nshift;
                }
                _tmpindGn = zrshift;
                _tmpindcache = ind_Gncacheshift;
                for(int thisn=0; thisn<nnvals_Nn; thisn++){
                    double complex thisGnnorm = nextWns[_tmpindGn];
                    Nncache[_tmpindcache] += thisGnnorm;
                    wNncache[_tmpindcache] += w_gal1*thisGnnorm;
                    _tmpindGn += nbinszr_reso;
                    _tmpindcache += nshift;
                }
            }
        }
    }
}

// Same-resolution multipole allocation for a region. 
// Equations from eq. (32) in 2309.08601, including double counting corrections: 
//  * Gamma0_n(t1,t2) ~ wshape * G_{n-3}(t1) * G_{-n-3}(t2) - dccorr delta_{t1,t2} 
//  * Gamma1_n(t1,t2) ~ conj(wshape) * G_{n-1}(t1) * G_{-n-1}(t2) - dccorr delta_{t1,t2} 
//  * Gamma2_n(t1,t2) ~ wshape * conj(G_{-n-1}(t1)) * G_{-n-3}(t2) - dccorr delta_{t1,t2} 
//  * Gamma3_n(t1,t2) ~ wshape * G_{n-3}(t1) * conj(G_{n-1}(t2)) - dccorr delta_{t1,t2} 
//  * Norm_n(t1,t2)   ~ w * N_{n}(t1) * conj(N_{n})(t2) - w*W2n delta_{t1,t2},
static void ggg_accum_samereso(GggContext *c, int rbinmin, int nbinsr_reso,
    int z_gal1, double w_gal1, double complex wshape_gal1,
    const double complex *nextGns, const double complex *nextWns,
    const double complex *nextG2ns, const double complex *nextW2ns,
    const int *nextncounts, int *allowedrinds, int *allowedzinds){
    int nbinsz=c->nbinsz, nbinsr=c->nbinsr, nmax=c->nmax, dccorr=c->dccorr;
    int elthread=c->elthread, gamma_compshift=c->gamma_compshift;
    int gamma_nshift=c->gamma_nshift, gamma_zshift=c->gamma_zshift;
    double complex *tmpGamma0s=c->tmpGamma0s, *tmpGamma1s=c->tmpGamma1s;
    double complex *tmpGamma2s=c->tmpGamma2s, *tmpGamma3s=c->tmpGamma3s;
    double complex *tmpGammans_norm=c->tmpGammans_norm;
    double complex conj_wshape_gal1 = conj(wshape_gal1);
    int nbinszr_reso = nbinsz*nbinsr_reso;
    int nzero = nmax+3;

    // First check for zero count bins (most likely only in discrete-discrete bit)
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
    // Now update the Gammans
    for (int thisn=0; thisn<nmax+1; thisn++){
        int ind_mnm3 = (nzero-thisn-3)*nbinszr_reso;
        int ind_mnm1 = (nzero-thisn-1)*nbinszr_reso;
        int ind_nm3 = (nzero+thisn-3)*nbinszr_reso;
        int ind_nm1 = (nzero+thisn-1)*nbinszr_reso;
        int ind_norm = thisn*nbinszr_reso;
        int thisnshift = elthread*gamma_compshift + thisn*gamma_nshift;
        for (int zrcombis1=0; zrcombis1<nallowedcounts; zrcombis1++){
            int elb1 = allowedrinds[zrcombis1];
            int zbin2 = allowedzinds[zrcombis1];
            int elb1_full = elb1 + rbinmin;
            int zrshift = zbin2*nbinsr_reso + elb1;
            if (dccorr==1){
                int zcombi = z_gal1*nbinsz*nbinsz + zbin2*nbinsz + zbin2;
                int gammashift1 = thisnshift + zcombi*gamma_zshift + elb1_full*nbinsr;
                int gammashift = gammashift1 + elb1_full;
                tmpGamma0s[gammashift] += wshape_gal1*nextG2ns[0*nbinszr_reso + zrshift];
                tmpGamma1s[gammashift] += conj_wshape_gal1*nextG2ns[1*nbinszr_reso + zrshift];
                tmpGamma2s[gammashift] += wshape_gal1*nextG2ns[2*nbinszr_reso + zrshift];
                tmpGamma3s[gammashift] += wshape_gal1*nextG2ns[3*nbinszr_reso + zrshift];
                tmpGammans_norm[gammashift1 + elb1_full] -= w_gal1*nextW2ns[zrshift];
            }
            double complex h0 = -wshape_gal1 * nextGns[ind_nm3 + zrshift];
            double complex h1 = -conj_wshape_gal1 * nextGns[ind_nm1 + zrshift];
            double complex h2 = -wshape_gal1 * conj(nextGns[ind_mnm1 + zrshift]);
            double complex h3 = -wshape_gal1 * nextGns[ind_nm3 + zrshift];
            double complex w0 = w_gal1 * nextWns[ind_norm + zrshift];
            int _zcombi = z_gal1*nbinsz*nbinsz+zbin2*nbinsz;
            int _gammashift1 = thisnshift + elb1_full*nbinsr;
            for (int zrcombis2=0; zrcombis2<nallowedcounts; zrcombis2++){
                int zcombi = _zcombi+allowedzinds[zrcombis2];
                int gammashift1 = _gammashift1 + zcombi*gamma_zshift;
                int elb2_full = allowedrinds[zrcombis2] + rbinmin;
                int zrshift2 = allowedzinds[zrcombis2]*nbinsr_reso + allowedrinds[zrcombis2];
                int gammashift = gammashift1 + elb2_full;
                double complex Gmnm3 = nextGns[ind_mnm3 + zrshift2];
                tmpGamma0s[gammashift] += h0*Gmnm3;
                tmpGamma1s[gammashift] += h1*nextGns[ind_mnm1 + zrshift2];
                tmpGamma2s[gammashift] += h2*Gmnm3;
                tmpGamma3s[gammashift] += h3*conj(nextGns[ind_nm1 + zrshift2]);
                tmpGammans_norm[gammashift1 + elb2_full] += w0*conj(nextWns[ind_norm + zrshift2]);
            }
        }
    }
}

// Cross-resolution multipole allocation for a region (sect 3.3 in https://arxiv.org/pdf/2309.08601)
// Equation same as for equal-reso, but the used caches depend on how the resos are ordered.
// Example of Gamma0: reso1<reso2 --> (wG_nm3)*G_mnm3  vs.  reso1>reso2 -->  G_nm3*(wG_mnm3)
// where wG_xxx := wshape*G_xxx and cwG_xxx := conj(wshape)*G_xxx.,
static void ggg_accum_crossreso(GggContext *c){
    int nbinsz=c->nbinsz, nbinsr=c->nbinsr, nmax=c->nmax, nresos=c->nresos, nshift=c->nshift;
    int elthread=c->elthread, gamma_compshift=c->gamma_compshift;
    int gamma_nshift=c->gamma_nshift, gamma_zshift=c->gamma_zshift, zbin2shift=c->zbin2shift;
    const int *zbinshifts=c->zbinshifts, *thetashifts_z=c->thetashifts_z;
    const int *cumresoshift_z=c->cumresoshift_z, *reso_rindedges=c->reso_rindedges;
    const int *ngal_in_pix=c->ngal_in_pix;
    double complex *Gncache=c->Gncache, *wGncache=c->wGncache, *cwGncache=c->cwGncache;
    double complex *Nncache=c->Nncache, *wNncache=c->wNncache;
    double complex *tmpGamma0s=c->tmpGamma0s, *tmpGamma1s=c->tmpGamma1s;
    double complex *tmpGamma2s=c->tmpGamma2s, *tmpGamma3s=c->tmpGamma3s;
    double complex *tmpGammans_norm=c->tmpGammans_norm;
    for (int thisn=0; thisn<nmax+1; thisn++){
        int thisnshift = elthread*gamma_compshift + thisn*gamma_nshift;
        for (int zbin1=0; zbin1<nbinsz; zbin1++){
            int zbinshift_z1 = zbinshifts[zbin1], thetashift_z1 = thetashifts_z[zbin1];
            const int *cumresoshift_z1 = cumresoshift_z + zbin1*(nresos+1);
            for (int zbin2=0; zbin2<nbinsz; zbin2++){
                for (int zbin3=0; zbin3<nbinsz; zbin3++){
                    int zcombi = zbin1*nbinsz*nbinsz + zbin2*nbinsz + zbin3;
                    int _thetashift_z = thetashift_z1;
                    // Case max(reso1, reso2) = reso2
                    for (int thisreso1=0; thisreso1<nresos; thisreso1++){
                        int rbinmin1 = reso_rindedges[thisreso1];
                        int rbinmax1 = reso_rindedges[thisreso1+1];
                        for (int thisreso2=thisreso1+1; thisreso2<nresos; thisreso2++){
                            int rbinmin2 = reso_rindedges[thisreso2];
                            int rbinmax2 = reso_rindedges[thisreso2+1];
                            int cumshift2 = cumresoshift_z1[thisreso2];
                            for (int elgal=0; elgal<ngal_in_pix[zbin1*nresos+thisreso2]; elgal++){
                                for (int elb1=rbinmin1; elb1<rbinmax1; elb1++){
                                    int gammashift1 = thisnshift + zcombi*gamma_zshift + elb1*nbinsr;
                                    int ind_Nncacheshift = zbin2*zbin2shift + zbinshift_z1 + elb1*thetashift_z1 +
                                        cumshift2 + elgal;
                                    int ind_Gncacheshift = (nmax+3)*nshift + ind_Nncacheshift;
                                    double complex h0 = -wGncache[(thisn-3)*nshift + ind_Gncacheshift];
                                    double complex h1 = -cwGncache[(thisn-1)*nshift + ind_Gncacheshift];
                                    double complex h2 = -conj(cwGncache[(-thisn-1)*nshift + ind_Gncacheshift]);
                                    double complex h3 = -wGncache[(thisn-3)*nshift + ind_Gncacheshift];
                                    double complex w0 = wNncache[thisn*nshift + ind_Nncacheshift];
                                    ind_Nncacheshift = zbin3*zbin2shift + zbinshift_z1 + rbinmin2*thetashift_z1 +
                                            cumshift2 + elgal;
                                    ind_Gncacheshift = (nmax+3)*nshift + ind_Nncacheshift;
                                    int _imnm3 = (-thisn-3)*nshift + ind_Gncacheshift;
                                    int _imnm1 = (-thisn-1)*nshift + ind_Gncacheshift;
                                    int _inm1 = (thisn-1)*nshift + ind_Gncacheshift;
                                    int _in = thisn*nshift + ind_Nncacheshift;
                                    for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                        int gammashift = gammashift1 + elb2;
                                        double complex Gmnm3 = Gncache[_imnm3];
                                        tmpGamma0s[gammashift] += h0*Gmnm3;
                                        tmpGamma1s[gammashift] += h1*Gncache[_imnm1];
                                        tmpGamma2s[gammashift] += h2*Gmnm3;
                                        tmpGamma3s[gammashift] += h3*conj(Gncache[_inm1]);
                                        tmpGammans_norm[gammashift1 + elb2] += w0*conj(Nncache[_in]);
                                        ind_Nncacheshift += _thetashift_z; ind_Gncacheshift += _thetashift_z;
                                        _imnm3 += _thetashift_z; _imnm1 += _thetashift_z; _inm1 += _thetashift_z; _in += _thetashift_z;
                                    }
                                }
                            }
                        }
                    }
                    // Case max(reso1, reso2) = reso1
                    for (int thisreso2=0; thisreso2<nresos; thisreso2++){
                        int rbinmin2 = reso_rindedges[thisreso2];
                        int rbinmax2 = reso_rindedges[thisreso2+1];
                        for (int thisreso1=thisreso2+1; thisreso1<nresos; thisreso1++){
                            int rbinmin1 = reso_rindedges[thisreso1];
                            int rbinmax1 = reso_rindedges[thisreso1+1];
                            int cumshift1 = cumresoshift_z1[thisreso1];
                            for (int elgal=0; elgal<ngal_in_pix[zbin1*nresos+thisreso1]; elgal++){
                                for (int elb1=rbinmin1; elb1<rbinmax1; elb1++){
                                    int gammashift1 = thisnshift + zcombi*gamma_zshift + elb1*nbinsr;
                                    int ind_Nncacheshift = zbin2*zbin2shift + zbinshift_z1 + elb1*thetashift_z1 +
                                        cumshift1 + elgal;
                                    int ind_Gncacheshift = (nmax+3)*nshift + ind_Nncacheshift;
                                    double complex h0 = -Gncache[(thisn-3)*nshift + ind_Gncacheshift];
                                    double complex h1 = -Gncache[(thisn-1)*nshift + ind_Gncacheshift];
                                    double complex h2 = -conj(Gncache[(-thisn-1)*nshift + ind_Gncacheshift]);
                                    double complex h3 = -Gncache[(thisn-3)*nshift + ind_Gncacheshift];
                                    double complex w0 = Nncache[thisn*nshift + ind_Nncacheshift];
                                    ind_Nncacheshift = zbin3*zbin2shift + zbinshift_z1 + rbinmin2*thetashift_z1 +
                                            cumshift1 + elgal;
                                    ind_Gncacheshift = (nmax+3)*nshift + ind_Nncacheshift;
                                    int _imnm3 = (-thisn-3)*nshift + ind_Gncacheshift;
                                    int _imnm1 = (-thisn-1)*nshift + ind_Gncacheshift;
                                    int _inm1 = (thisn-1)*nshift + ind_Gncacheshift;
                                    int _in = thisn*nshift + ind_Nncacheshift;
                                    for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                        int gammashift = gammashift1 + elb2;
                                        double complex wGmnm3 = wGncache[_imnm3];
                                        tmpGamma0s[gammashift] += h0*wGmnm3;
                                        tmpGamma1s[gammashift] += h1*cwGncache[_imnm1];
                                        tmpGamma2s[gammashift] += h2*wGmnm3;
                                        tmpGamma3s[gammashift] += h3*conj(cwGncache[_inm1]);
                                        tmpGammans_norm[gammashift1 + elb2] += w0*conj(wNncache[_in]);
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

// Accumulates the per-thread multipoles and from there computes bin centers and norms
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

// (C) GNN HELPERS //

// Context shared by the GNN (G3L) helpers.
typedef struct {
    int nbinsz_source, nbinsz_lens, nbinsr, nmax, nresos;
    int nnvals_Gn, nnvals_Wn;
    int upsilon_zshift, upsilon_nshift, upsilon_threadshift;
    int dccorr, elthread;
    int *reso_rindedges, *ngal_in_pix, *cumresoshift_z, *thetashifts_z, *zbinshifts;
    int zbin2shift, nshift;
    double complex *Gncache, *wGncache, *cwGncache;// all nmax+3 W-slots
    double complex *Wncache, *wWncache;  // W_0..W_nmax
    double complex *tmpUpsilon, *tmpNorm;
} GnnContext;

static void gnn_zero_caches(GnnContext *c){
    for (int _i=0; _i<c->nnvals_Gn*c->nshift; _i++){c->Gncache[_i]=0; c->wGncache[_i]=0; c->cwGncache[_i]=0;}
    for (int _i=0; _i<c->nnvals_Wn*c->nshift; _i++){c->Wncache[_i]=0; c->wWncache[_i]=0;}
}

// Scatter a base's leaf multipoles thisWns into the region caches
static void gnn_update_wncache(GnnContext *c, int elreso, int rbinmin, int rbinmax,
    int nbinsr_reso, int z_gal1, double w_gal1, double complex wshape_gal1,
    const int *redpix_by_reso2, const double complex *thisWns){
    // The reduced pixels and this cache exist only for *_accum_crossreso, which has no
    // resolution pair to visit for a single-resolution tree. Returning here also keeps the
    // grid arrays, which are empty in that case, from being indexed at all.
    if (c->nresos<=1){return;}
    int nbinszr_reso = c->nbinsz_lens*nbinsr_reso;
    for (int elreso2=elreso; elreso2<c->nresos; elreso2++){
        int redpix_reso2 = redpix_by_reso2[elreso2];
        for (int zbin2=0; zbin2<c->nbinsz_lens; zbin2++){
            for (int thisrbin=rbinmin; thisrbin<rbinmax; thisrbin++){
                int zrshift = zbin2*nbinsr_reso + thisrbin-rbinmin;
                if (cabs(thisWns[nbinszr_reso+zrshift])<1e-10){continue;}
                int ind_Gncacheshift = zbin2*c->zbin2shift + c->zbinshifts[z_gal1] +
                    thisrbin*c->thetashifts_z[z_gal1] +
                    c->cumresoshift_z[z_gal1*(c->nresos+1) + elreso2] + redpix_reso2;
                int _tmpindGn = zrshift;
                int _tmpindcache = ind_Gncacheshift;
                for(int thisn=0; thisn<c->nnvals_Gn; thisn++){
                    double complex thisGn = thisWns[_tmpindGn];
                    c->Gncache[_tmpindcache] += thisGn;
                    c->wGncache[_tmpindcache] += wshape_gal1*thisGn;
                    c->cwGncache[_tmpindcache] += conj(wshape_gal1)*thisGn;
                    _tmpindGn += nbinszr_reso;
                    _tmpindcache += c->nshift;
                }
                _tmpindGn = zrshift+nbinszr_reso;
                _tmpindcache = ind_Gncacheshift;
                for(int thisn=0; thisn<c->nnvals_Wn; thisn++){
                    double complex thisNn = thisWns[_tmpindGn];
                    c->Wncache[_tmpindcache] += thisNn;
                    c->wWncache[_tmpindcache] += w_gal1*thisNn;
                    _tmpindGn += nbinszr_reso;
                    _tmpindcache += c->nshift;
                }
            }
        }
    }
}

// Same-resolution Upsilon_n / N_n allocation (the discrete kernel reuses it
// with rbinmin=0, nbinsr_reso=nbinsr):
// Upsilon(t1,t2) ~ -we * W_{n-1}(t1) * conj(W_{n+1})(t2) + delta^K_{t1,t2} self
// Norm(t1,t2)    ~   w * W_n(t1)     * conj(W_n)(t2)     - delta^K_{t1,t2} self
static void gnn_accum_samereso(GnnContext *c, int rbinmin, int nbinsr_reso,
    int z_gal1, double w_gal1, double complex wshape_gal1,
    const double complex *thisWns, const double complex *thisG2ns,
    const double complex *thisW2ns, const int *nextncounts,
    int *allowedrinds, int *allowedzinds){
    int nbinsz_lens=c->nbinsz_lens, nbinsr=c->nbinsr, nmax=c->nmax;
    int nbinszr_reso = nbinsz_lens*nbinsr_reso;
    int nallowedcounts = 0;
    for (int zbin1=0; zbin1<nbinsz_lens; zbin1++){
        for (int elb1=0; elb1<nbinsr_reso; elb1++){
            if (nextncounts[zbin1*nbinsr_reso + elb1] != 0){
                allowedrinds[nallowedcounts] = elb1;
                allowedzinds[nallowedcounts] = zbin1;
                nallowedcounts += 1;
            }
        }
    }
    for (int thisn=0; thisn<nmax+1; thisn++){
        int thisnshift = c->elthread*c->upsilon_threadshift + thisn*c->upsilon_nshift;
        for (int zrcombis1=0; zrcombis1<nallowedcounts; zrcombis1++){
            int elb1 = allowedrinds[zrcombis1];
            int zbin2 = allowedzinds[zrcombis1];
            int elb1_full = elb1 + rbinmin;
            int zrshift = zbin2*nbinsr_reso + elb1;
            if (c->dccorr==1){
                int zcombi = z_gal1*nbinsz_lens*nbinsz_lens + zbin2*nbinsz_lens + zbin2;
                int gammashift = thisnshift + zcombi*c->upsilon_zshift + elb1_full*nbinsr+elb1_full;
                c->tmpUpsilon[gammashift] += thisG2ns[zrshift];
                c->tmpNorm[gammashift] -= thisW2ns[zrshift];
            }
            int _zcombi = z_gal1*nbinsz_lens*nbinsz_lens + zbin2*nbinsz_lens;
            int _wind = (thisn+1)*nbinszr_reso+zrshift;
            int _gammashift = thisnshift + elb1_full*nbinsr;
            double complex nextUps = -wshape_gal1*thisWns[_wind-nbinszr_reso];
            double complex nextN = w_gal1*thisWns[_wind];
            for (int zrcombis2=0; zrcombis2<nallowedcounts; zrcombis2++){
                int elb2 = allowedrinds[zrcombis2];
                int zbin3 = allowedzinds[zrcombis2];
                int elb2_full = elb2 + rbinmin;
                int zcombi = _zcombi + zbin3;
                int gammashift = _gammashift + zcombi*c->upsilon_zshift + elb2_full;
                _wind = (thisn+1)*nbinszr_reso + zbin3*nbinsr_reso + elb2;
                c->tmpUpsilon[gammashift] += nextUps*conj(thisWns[_wind+nbinszr_reso]);
                c->tmpNorm[gammashift] += nextN*conj(thisWns[_wind]);
            }
        }
    }
}

// Cross-resolution Upsilon_n / N_n allocation from the region caches. 
// * Upsilon = -wshape * W_nm1 * conj(W_np1) --> -(wW_nm1)*conj(W_np1) if reso1 < reso2
//                                           --> -W_nm1*conj(cwW_np1)  if reso1 > reso2
// * Norm    =  w * W_n * conj(W_n)          --> wW_n*conj(W_n)        if reso1 < reso2
//                                           --> W_n*conj(wW_n)        if reso1 > reso2
// where wW := w(shape)*W and cwW := conj(wshape)*W. In the polar caches slot k
// holds W_{k-1}, so W_{n-1} sits at slot n and W_{n+1} at slot n+2.
static void gnn_accum_crossreso(GnnContext *c){
    int nbinsz_source=c->nbinsz_source, nbinsz_lens=c->nbinsz_lens, nbinsr=c->nbinsr;
    int nmax=c->nmax, nresos=c->nresos, nshift=c->nshift;
    for (int thisn=0; thisn<nmax+1; thisn++){
        int thisnshift = c->elthread*c->upsilon_threadshift + thisn*c->upsilon_nshift;
        for (int zbin1=0; zbin1<nbinsz_source; zbin1++){
            for (int zbin2=0; zbin2<nbinsz_lens; zbin2++){
                for (int zbin3=0; zbin3<nbinsz_lens; zbin3++){
                    int zcombi = zbin1*nbinsz_lens*nbinsz_lens + zbin2*nbinsz_lens + zbin3;
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
                                    int ind_Wncacheshift = zbin2*c->zbin2shift + c->zbinshifts[zbin1]
                                        + elb1*c->thetashifts_z[zbin1]
                                        + c->cumresoshift_z[zbin1*(nresos+1) + thisreso2] + elgal;
                                    double complex nextUps = -c->wGncache[(thisn+0)*nshift+ind_Wncacheshift];
                                    double complex nextN = c->wWncache[thisn*nshift+ind_Wncacheshift];
                                    int _upsshift = thisnshift + zcombi*c->upsilon_zshift + elb1*nbinsr;
                                    ind_Wncacheshift = zbin3*c->zbin2shift + c->zbinshifts[zbin1]
                                        + rbinmin2*c->thetashifts_z[zbin1]
                                        + c->cumresoshift_z[zbin1*(nresos+1) + thisreso2] + elgal;
                                    for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                        c->tmpUpsilon[_upsshift+elb2] += nextUps*conj(c->Gncache[(thisn+2)*nshift+ind_Wncacheshift]);
                                        c->tmpNorm[_upsshift+elb2] += nextN*conj(c->Wncache[thisn*nshift+ind_Wncacheshift]);
                                        ind_Wncacheshift += _thetashift_z;
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
                                    int ind_Wncacheshift = zbin2*c->zbin2shift + c->zbinshifts[zbin1]
                                        + elb1*c->thetashifts_z[zbin1]
                                        + c->cumresoshift_z[zbin1*(nresos+1) + thisreso1] + elgal;
                                    double complex nextUps = -c->Gncache[(thisn+0)*nshift+ind_Wncacheshift];
                                    double complex nextN = c->Wncache[thisn*nshift+ind_Wncacheshift];
                                    int _upsshift = thisnshift + zcombi*c->upsilon_zshift + elb1*nbinsr;
                                    ind_Wncacheshift = zbin3*c->zbin2shift + c->zbinshifts[zbin1]
                                        + rbinmin2*c->thetashifts_z[zbin1]
                                        + c->cumresoshift_z[zbin1*(nresos+1) + thisreso1] + elgal;
                                    for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                        c->tmpUpsilon[_upsshift+elb2] += nextUps*conj(c->cwGncache[(thisn+2)*nshift+ind_Wncacheshift]);
                                        c->tmpNorm[_upsshift+elb2] += nextN*conj(c->wWncache[thisn*nshift+ind_Wncacheshift]);
                                        ind_Wncacheshift += _thetashift_z;
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

// Reduce the per-thread Upsilon_n / N_n accumulators into the NPCFOutput and
// fill the (zbin_source, zbin_lens) bin_centers. Shared by discrete/doubletree.
static void gnn_reduce(int nbinsz_source, int nbinsz_lens, int nbinsr, int nmax,
    int nthreads, const double complex *tmpUpsilon, const double complex *tmpNorm,
    const double *tmpwcounts, const double *tmpwnorms, NPCFOutput *out){
    int nzcombis = nbinsz_source*nbinsz_lens*nbinsz_lens;
    int upsilon_zshift = nbinsr*nbinsr;
    int upsilon_nshift = upsilon_zshift*nzcombis;
    int upsilon_threadshift = (nmax+1)*upsilon_nshift;
    #pragma omp parallel for num_threads(nthreads)
    for (int thisn=0; thisn<nmax+1; thisn++){
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            int thisthreadshift = thisthread*upsilon_threadshift;
            for (int zcombi=0; zcombi<nzcombis; zcombi++){
                for (int elb1=0; elb1<nbinsr; elb1++){
                    for (int elb2=0; elb2<nbinsr; elb2++){
                        int iUps = thisn*upsilon_nshift + zcombi*upsilon_zshift + elb1*nbinsr + elb2;
                        out->npcf[iUps] += tmpUpsilon[thisthreadshift+iUps];
                        out->norm_mp[iUps] += tmpNorm[thisthreadshift+iUps];
                    }
                }
            }
        }
    }
    double *totcounts = calloc(nbinsz_source*nbinsz_lens*nbinsr, sizeof(double));
    double *totnorms = calloc(nbinsz_source*nbinsz_lens*nbinsr, sizeof(double));
    for (int thisthread=0; thisthread<nthreads; thisthread++){
        int thisthreadshift = thisthread*nbinsz_source*nbinsz_lens*nbinsr;
        for (int elbinz=0; elbinz<nbinsz_source*nbinsz_lens; elbinz++){
            for (int elbinr=0; elbinr<nbinsr; elbinr++){
                int tmpind = elbinz*nbinsr + elbinr;
                totcounts[tmpind] += tmpwcounts[thisthreadshift+tmpind];
                totnorms[tmpind] += tmpwnorms[thisthreadshift+tmpind];
            }
        }
    }
    for (int elbinz=0; elbinz<nbinsz_source*nbinsz_lens; elbinz++){
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            int tmpind = elbinz*nbinsr + elbinr;
            if (totnorms[tmpind] != 0){ out->bin_centers[tmpind] = totcounts[tmpind]/totnorms[tmpind]; }
        }
    }
    free(totcounts); free(totnorms);
}

// Per-pair fill of the GNN leaf count multipoles W_m.
// Shared by the discrete/doubletree GNN kernels
static inline void gnn_fill_wn(
    double complex *thisWns, double complex *thisG2ns, double complex *thisW2ns,
    int z2rshift, int nbinszr, int nmax, double w_gal1, double w_gal2,
    double complex wshape_gal1, double complex phirot, double complex phirotc){
    thisG2ns[z2rshift] += wshape_gal1*w_gal2*w_gal2*phirotc*phirotc;
    thisW2ns[z2rshift] += w_gal1*w_gal2*w_gal2;
    int ind_Wn = z2rshift;
    double complex nphirot = phirotc;
    for (int nextn=-1;nextn<=nmax+1;nextn++){
        thisWns[ind_Wn] += w_gal2*nphirot;
        nphirot *= phirot;
        ind_Wn += nbinszr;
    }
}

// Accumulate one neighbour's contribution into a block of angular multipoles:
//   Xn[n] += leaf_weight * phirot^n where leaf_weight is w or wshape
static inline void slab_fill_Xn(double complex leaf_weight, double complex phirot,
        double complex *Xn, int zr_offset, int nbinszr, int n_min, int nnvals){
    int n_zero_block = -n_min;
    if (n_zero_block >= 0 && n_zero_block < nnvals){
        int pos_idx = zr_offset + n_zero_block*nbinszr, neg_idx = pos_idx;
        Xn[pos_idx] += leaf_weight;
        int nblocks_above = nnvals-1-n_zero_block, nblocks_below = n_zero_block;
        int nshared = (nblocks_above < nblocks_below) ? nblocks_above : nblocks_below;
        double complex phase_pow = 1.;
        int n = 1;
        for (; n<=nshared; n++){
            phase_pow *= phirot;
            pos_idx += nbinszr; neg_idx -= nbinszr;
            Xn[pos_idx] += leaf_weight*phase_pow;
            Xn[neg_idx] += leaf_weight*conj(phase_pow);
        }
        for (; n<=nblocks_above; n++){ phase_pow *= phirot; pos_idx += nbinszr; Xn[pos_idx] += leaf_weight*phase_pow; }
        for (; n<=nblocks_below; n++){ phase_pow *= phirot; neg_idx -= nbinszr; Xn[neg_idx] += leaf_weight*conj(phase_pow); }
    } else {
        double complex phase_pow = 1.;
        if (n_min >= 0){ for (int q=0; q<n_min; q++){ phase_pow *= phirot; } }
        else           { double complex phirotc = conj(phirot); for (int q=0; q<-n_min; q++){ phase_pow *= phirotc; } }
        int idx = zr_offset;
        for (int b=0; b<nnvals; b++){ Xn[idx] += leaf_weight*phase_pow; phase_pow *= phirot; idx += nbinszr; }
    }
}

// One double-counting self-term for a leaf pass
// Accumulate into accum[zrshift] as
//   sum (use_modulus_sq ? |value|^2 : value^2) * phirotc^phirotc_pow
typedef struct { int phirotc_pow; bool use_modulus_sq; double complex *accum; } LeafSelfTerm;

// Iterate the slab-hashed leaf neighbours of a base (C1,C2,C3): catalogue
// members with r_perp in [RMIN,RMAX) and |dz| < PI. 
// Recall to not nest two invocations in one block (internal _sn_* names collide).
#define SLAB_NEIGHBORS_FOREACH(C1, C2, C3, LPOS1, LPOS2, LPOS3, LZBIN, \
        NSLABS, Z0, DPIX_Z, PIX1_START, PIX1_D, PIX1_N, PIX2_START, PIX2_D, PIX2_N, \
        SLAB_OFFSETS, INDEX_MATCHER, BOUNDS, RSHIFT_BOUNDS, PIX_GALS, \
        RMIN, RMAX, NBINSR, PI, J, REL1, REL2, D2, DIST, RBIN, ZRSHIFT, PHIROT, PHIROTC) \
    for (int _sn_s = mymax(0, (int) floor(((C3)-(PI)-(Z0))/(DPIX_Z))), \
             _sn_shi = mymin((NSLABS)-1, (int) floor(((C3)+(PI)-(Z0))/(DPIX_Z))), \
             _sn_npix = (PIX1_N)*(PIX2_N), \
             _sn_p1lo = mymax(0, (int) floor(((C1)-((RMAX)+(PIX1_D))-(PIX1_START))/(PIX1_D))), \
             _sn_p1hi = mymin((PIX1_N)-1, (int) floor(((C1)+((RMAX)+(PIX1_D))-(PIX1_START))/(PIX1_D))), \
             _sn_p2lo = mymax(0, (int) floor(((C2)-((RMAX)+(PIX2_D))-(PIX2_START))/(PIX2_D))), \
             _sn_p2hi = mymin((PIX2_N)-1, (int) floor(((C2)+((RMAX)+(PIX2_D))-(PIX2_START))/(PIX2_D))); \
         _sn_s <= _sn_shi; _sn_s++) \
      for (int _sn_ip1 = _sn_p1lo, _sn_mshift = _sn_s*_sn_npix, \
               _sn_bshift = (RSHIFT_BOUNDS)[_sn_s], _sn_gshift = (SLAB_OFFSETS)[_sn_s]; \
           _sn_ip1 <= _sn_p1hi; _sn_ip1++) \
        for (int _sn_ip2 = _sn_p2lo, _sn_ir = -2; \
             _sn_ip2 <= _sn_p2hi && (_sn_ir = (INDEX_MATCHER)[_sn_mshift + _sn_ip2*(PIX1_N) + _sn_ip1], 1); \
             _sn_ip2++) \
          if (_sn_ir != -1) \
            for (int _sn_k = (BOUNDS)[_sn_bshift + _sn_ir], \
                     _sn_kup = (BOUNDS)[_sn_bshift + _sn_ir + 1]; \
                 _sn_k < _sn_kup; _sn_k++) \
              if ( ((J) = (PIX_GALS)[_sn_gshift + _sn_k], \
                    (REL1) = (LPOS1)[J] - (C1), (REL2) = (LPOS2)[J] - (C2), \
                    (D2) = (REL1)*(REL1) + (REL2)*(REL2), \
                    (D2) >= (RMIN)*(RMIN)) \
                && ((D2) < (RMAX)*(RMAX)) \
                && (fabs((LPOS3)[J] - (C3)) < (PI)) \
                && ((DIST) = sqrt(D2), \
                    (RBIN) = (int) floor(log((DIST)/(RMIN)) * ((NBINSR)/log((RMAX)/(RMIN)))), \
                    (RBIN) >= 0 && (RBIN) < (NBINSR)) \
                && ((ZRSHIFT) = (LZBIN)[J]*(NBINSR) + (RBIN), \
                    (PHIROT) = ((REL1) + I*(REL2))/(DIST), (PHIROTC) = conj(PHIROT), 1) )

// Build the phirotc power table shared by both leaf passes: pc[n] = phirotc^n for
// even n up to nmax (n in {0,2,4,6}), grouped so the values match the old hand-
// unrolled self-terms to floating-point rounding.
static inline void slab_fill_pcpow(double complex phirotc, int nmax, double complex *pc){
    pc[0] = 1.;
    if (nmax >= 2){ pc[2] = phirotc*phirotc;
        if (nmax >= 4){ pc[4] = pc[2]*pc[2];
            if (nmax >= 6){ pc[6] = pc[4]*pc[2]; } 
        } 
    }
}

// Largest phirotc exponent requested across a term list.
static inline int slab_terms_nmax(const LeafSelfTerm *terms, int nterms){
    int nmax = 0;
    for (int t=0; t<nterms; t++){
        if (terms[t].phirotc_pow > nmax){
             nmax = terms[t].phirotc_pow; 
        } 
    }
    return nmax;
}

// Accumulate the Wn for scalar leafs in the 3dbox slab geometry.
static inline __attribute__((always_inline)) void slab_count_leafmultipoles(
    double c_pos1, double c_pos2, double c_pos3, double c_w,
    const double *pos1_leaf, const double *pos2_leaf, const double *pos3_leaf,
    const double *w_leaf, const int *zbin_leaf,
    int nbinsz_leaf, int nslabs, double z0, double dpix_z,
    double pix1_start, double pix1_d, int pix1_n,
    double pix2_start, double pix2_d, int pix2_n,
    const int *slab_offsets, const int *index_matcher, const int *pixs_galind_bounds,
    const int *rshift_bounds, const int *pix_gals,
    double rmin, double rmax, int nbinsr, double Pi,
    int n_min, int n, double complex *Wn,
    const LeafSelfTerm *terms, int nterms,
    int *ncounts, double *wcounts, double *wnorms){

    int nbinszr_leaf = nbinsz_leaf*nbinsr;
    int nmax = slab_terms_nmax(terms, nterms);

    int j, rbin, zrshift;
    double rel1, rel2, d2, dist;
    double complex phirot, phirotc;
    SLAB_NEIGHBORS_FOREACH(c_pos1, c_pos2, c_pos3, pos1_leaf, pos2_leaf, pos3_leaf, zbin_leaf,
            nslabs, z0, dpix_z, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n,
            slab_offsets, index_matcher, pixs_galind_bounds, rshift_bounds, pix_gals,
            rmin, rmax, nbinsr, Pi, j, rel1, rel2, d2, dist, rbin, zrshift, phirot, phirotc){
        double w = w_leaf[j];
        ncounts[zrshift] += 1;
        if (nterms){
            double complex pc[7];
            slab_fill_pcpow(phirotc, nmax, pc);
            double w2 = w*w;
            for (int t=0; t<nterms; t++){ terms[t].accum[zrshift] += w2 * pc[terms[t].phirotc_pow]; }
        }
        slab_fill_Xn(w, phirot, Wn, zrshift, nbinszr_leaf, n_min, n);
        if (wcounts){
            wcounts[zrshift] += c_w*w*dist;
            wnorms[zrshift]  += c_w*w;
        }
    }
}

// Accumulate the Gn for polar leafs in the 3dbox slab geometry.
static inline __attribute__((always_inline)) void slab_polar_leafmultipoles(
    double c_pos1, double c_pos2, double c_pos3, double c_w,
    const double *pos1_leaf, const double *pos2_leaf, const double *pos3_leaf,
    const double *w_leaf, const int *zbin_leaf, const double *e1_leaf, const double *e2_leaf,
    int nbinsz_leaf, int nslabs, double z0, double dpix_z,
    double pix1_start, double pix1_d, int pix1_n,
    double pix2_start, double pix2_d, int pix2_n,
    const int *slab_offsets, const int *index_matcher, const int *pixs_galind_bounds,
    const int *rshift_bounds, const int *pix_gals,
    double rmin, double rmax, int nbinsr, double Pi,
    int n_min, int n, double complex *Gn,
    const LeafSelfTerm *terms, int nterms,
    int *ncounts, double *wcounts, double *wnorms){

    int nbinszr_leaf = nbinsz_leaf*nbinsr;
    int nmax = slab_terms_nmax(terms, nterms);

    int j, rbin, zrshift;
    double rel1, rel2, d2, dist;
    double complex phirot, phirotc;
    SLAB_NEIGHBORS_FOREACH(c_pos1, c_pos2, c_pos3, pos1_leaf, pos2_leaf, pos3_leaf, zbin_leaf,
            nslabs, z0, dpix_z, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n,
            slab_offsets, index_matcher, pixs_galind_bounds, rshift_bounds, pix_gals,
            rmin, rmax, nbinsr, Pi, j, rel1, rel2, d2, dist, rbin, zrshift, phirot, phirotc){
        double w = w_leaf[j];
        double complex wshape = w*(e1_leaf[j] + I*e2_leaf[j]);
        ncounts[zrshift] += 1;
        if (nterms){
            double complex pc[7];
            slab_fill_pcpow(phirotc, nmax, pc);
            double complex wsq = wshape*wshape;
            double complex wabs = wshape*conj(wshape);
            for (int t=0; t<nterms; t++){
                terms[t].accum[zrshift] += (terms[t].use_modulus_sq ? wabs : wsq) * pc[terms[t].phirotc_pow];
            }
        }
        slab_fill_Xn(wshape, phirot, Gn, zrshift, nbinszr_leaf, n_min, n);
        if (wcounts){
            wcounts[zrshift] += c_w*w*dist;
            wnorms[zrshift]  += c_w*w;
        }
    }
}

// (C) NGG HELPERS //

// Context shared by the NGG helpers: 
typedef struct {
    int nbinsz_lens, nbinsz_source, nbinsr, nmax, nresos;
    int nnvals_Gn, nnvals_Wn; // 2*nmax+5, 2*nmax+1
    int upsilon_zshift, upsilon_nshift, upsilon_compshift;
    int upsilon_threadshift, norm_threadshift;
    int dccorr, elthread;
    int *reso_rindedges, *ngal_in_pix, *cumresoshift_z, *thetashifts_z, *zbinshifts;
    int zbin2shift, nshift;
    double complex *Gncache, *wGncache;
    double complex *Wncache, *wWncache;
    double complex *tmpUpsilon, *tmpNorm;
} NggContext;

static void ngg_zero_caches(NggContext *c){
    for (int _i=0; _i<c->nnvals_Gn*c->nshift; _i++){c->Gncache[_i]=0; c->wGncache[_i]=0;}
    for (int _i=0; _i<c->nnvals_Wn*c->nshift; _i++){c->Wncache[_i]=0; c->wWncache[_i]=0;}
}

// Scatter a base's leaf multipoles thisGns into the region caches
static void ngg_update_gnwncache(NggContext *c, int elreso, int rbinmin, int rbinmax,
    int nbinsr_reso, int z_gal1, double w_gal1, const int *redpix_by_reso2,
    const double complex *thisGns, const double complex *thisWns){
    // The reduced pixels and this cache exist only for *_accum_crossreso, which has no
    // resolution pair to visit for a single-resolution tree. Returning here also keeps the
    // grid arrays, which are empty in that case, from being indexed at all.
    if (c->nresos<=1){return;}
    int nbinszr_reso = c->nbinsz_source*nbinsr_reso;
    for (int elreso2=elreso; elreso2<c->nresos; elreso2++){
        int redpix_reso2 = redpix_by_reso2[elreso2];
        for (int zbin2=0; zbin2<c->nbinsz_source; zbin2++){
            for (int thisrbin=rbinmin; thisrbin<rbinmax; thisrbin++){
                int zrshift = zbin2*nbinsr_reso + thisrbin-rbinmin;
                if (cabs(thisWns[c->nmax*nbinszr_reso+zrshift])<1e-10){continue;}
                int ind_Gncacheshift = zbin2*c->zbin2shift + c->zbinshifts[z_gal1] +
                    thisrbin*c->thetashifts_z[z_gal1] +
                    c->cumresoshift_z[z_gal1*(c->nresos+1) + elreso2] + redpix_reso2;
                int _tmpindGn = zrshift;
                int _tmpindcache = ind_Gncacheshift;
                for(int thisn=0; thisn<c->nnvals_Gn; thisn++){
                    double complex thisGn = thisGns[_tmpindGn];
                    c->Gncache[_tmpindcache] += thisGn;
                    c->wGncache[_tmpindcache] += w_gal1*thisGn;
                    _tmpindGn += nbinszr_reso;
                    _tmpindcache += c->nshift;
                }
                int _tmpindWn = zrshift;
                _tmpindcache = ind_Gncacheshift;
                for(int thisn=0; thisn<c->nnvals_Wn; thisn++){
                    double complex thisNn = thisWns[_tmpindWn];
                    c->Wncache[_tmpindcache] += thisNn;
                    c->wWncache[_tmpindcache] += w_gal1*thisNn;
                    _tmpindWn += nbinszr_reso;
                    _tmpindcache += c->nshift;
                }
            }
        }
    }
}

// Same-resolution Upsilon-/Upsilon+/N allocation (the discrete/tree kernels
// reuse it with rbinmin=0, nbinsr_reso=nbinsr):
// Upsilon-(t1,t2) ~ w * G_{+n-2}(t1) * G_{-n-2}(t2) - delta^K self
// Upsilon+(t1,t2) ~ w * G_{+n-2}(t1) * conj(G_{+n-2})(t2) - delta^K self
// Norm(t1,t2)     ~ w * W_n(t1)      * W_{-n}(t2)         - delta^K self
static void ngg_accum_samereso(NggContext *c, int rbinmin, int nbinsr_reso,
    int z_gal1, double w_gal1,
    const double complex *thisGns, const double complex *thisWns,
    const double complex *thisG2ns, const double complex *thisW2ns,
    const int *thisncounts, int *allowedrinds, int *allowedzinds){
    int nbinsz_source=c->nbinsz_source, nbinsr=c->nbinsr, nmax=c->nmax;
    int nbinszr_reso = nbinsz_source*nbinsr_reso;
    int nallowedcounts = 0;
    for (int zbin1=0; zbin1<nbinsz_source; zbin1++){
        for (int elb1=0; elb1<nbinsr_reso; elb1++){
            if (thisncounts[zbin1*nbinsr_reso + elb1] != 0){
                allowedrinds[nallowedcounts] = elb1;
                allowedzinds[nallowedcounts] = zbin1;
                nallowedcounts += 1;
            }
        }
    }
    for (int thisn=-nmax; thisn<=nmax; thisn++){
        int thisnshift_ups = c->elthread*c->upsilon_threadshift + (nmax+thisn)*c->upsilon_nshift;
        int thisnshift_norm = c->elthread*c->norm_threadshift + (nmax+thisn)*c->upsilon_nshift;
        for (int zrcombis1=0; zrcombis1<nallowedcounts; zrcombis1++){
            int elb1 = allowedrinds[zrcombis1];
            int zbin2 = allowedzinds[zrcombis1];
            int elb1_full = elb1 + rbinmin;
            int zrshift = zbin2*nbinsr_reso + elb1;
            if (c->dccorr==1){
                int zcombi = z_gal1*nbinsz_source*nbinsz_source + zbin2*nbinsz_source + zbin2;
                int gammashift_ups = thisnshift_ups + zcombi*c->upsilon_zshift + elb1_full*nbinsr+elb1_full;
                int gammashift_norm = thisnshift_norm + zcombi*c->upsilon_zshift + elb1_full*nbinsr+elb1_full;
                c->tmpUpsilon[gammashift_ups] -= thisG2ns[zrshift];
                c->tmpUpsilon[c->upsilon_compshift+gammashift_ups] -= thisG2ns[nbinszr_reso+zrshift];
                c->tmpNorm[gammashift_norm] -= thisW2ns[zrshift];
            }
            int _zcombi = z_gal1*nbinsz_source*nbinsz_source + zbin2*nbinsz_source;
            int _wind = (nmax+thisn)*nbinszr_reso+zrshift;
            int _upsind1m = (nmax+thisn)*nbinszr_reso+zrshift;
            int _upsind1p = (nmax+thisn)*nbinszr_reso+zrshift;
            double complex nextUpsp = w_gal1*thisGns[_upsind1p];
            double complex nextUpsm = w_gal1*thisGns[_upsind1m];
            double complex nextN = w_gal1*thisWns[_wind];
            for (int zrcombis2=0; zrcombis2<nallowedcounts; zrcombis2++){
                int elb2 = allowedrinds[zrcombis2];
                int zbin3 = allowedzinds[zrcombis2];
                int elb2_full = elb2 + rbinmin;
                int zrshift2 = zbin3*nbinsr_reso + elb2;
                int zcombi = _zcombi + zbin3;
                int z3r2shift = zcombi*c->upsilon_zshift + elb1_full*nbinsr + elb2_full;
                int gammashift_ups = thisnshift_ups + z3r2shift;
                int gammashift_norm = thisnshift_norm + z3r2shift;
                int _wind2 = (nmax-thisn)*nbinszr_reso + zrshift2;
                int _upsind2p = (nmax+thisn)*nbinszr_reso + zrshift2;
                int _upsind2m = (nmax-thisn)*nbinszr_reso + zrshift2;
                c->tmpUpsilon[gammashift_ups] += nextUpsm*thisGns[_upsind2m];
                c->tmpUpsilon[c->upsilon_compshift+gammashift_ups] += nextUpsp*conj(thisGns[_upsind2p]);
                c->tmpNorm[gammashift_norm] += nextN*thisWns[_wind2];
            }
        }
    }
}

// Cross-resolution Upsilon-/Upsilon+/N allocation from the region caches. The
// base-weighted cache is picked depending on which band holds the base:
// * Upsilon_- = w * G_nm2 * G_mnm2 --> (wG_nm2)*G_mnm2  if reso1 < reso2
//                                  -->  G_nm2*(wG_mnm2) if reso1 > reso2
// * Upsilon_+ = w * G_nm2 * conj(G_nm2) --> (wG_nm2)*conj(G_nm2) if reso1 < reso2
//                                       -->  G_nm2*conj(wG_nm2)  if reso1 > reso2
// * Norm      = w * W_n * conj(W_n) --> wW_n*conj(W_n)  if reso1 < reso2
//                                   --> W_n*conj(wW_n)  if reso1 > reso2
// where wG := w(lens)*G and wW := w(lens)*W.
static void ngg_accum_crossreso(NggContext *c){
    int nbinsz_lens=c->nbinsz_lens, nbinsz_source=c->nbinsz_source, nbinsr=c->nbinsr;
    int nmax=c->nmax, nresos=c->nresos, nshift_cache=c->nshift;
    for (int thisn=-nmax; thisn<=nmax; thisn++){
        int thisnshift_ups = c->elthread*c->upsilon_threadshift + (nmax+thisn)*c->upsilon_nshift;
        int thisnshift_norm = c->elthread*c->norm_threadshift + (nmax+thisn)*c->upsilon_nshift;
        for (int zbin1=0; zbin1<nbinsz_lens; zbin1++){
            for (int zbin2=0; zbin2<nbinsz_source; zbin2++){
                for (int zbin3=0; zbin3<nbinsz_source; zbin3++){
                    int zcombi = zbin1*nbinsz_source*nbinsz_source + zbin2*nbinsz_source + zbin3;
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
                                    int ind_Wncacheshift = zbin2*c->zbin2shift + c->zbinshifts[zbin1] + elb1*c->thetashifts_z[zbin1]+
                                        c->cumresoshift_z[zbin1*(nresos+1) + thisreso2] + elgal;
                                    double complex nextUpsp = c->wGncache[(nmax+thisn)*nshift_cache+ind_Wncacheshift];
                                    double complex nextUpsm = c->wGncache[(nmax+thisn)*nshift_cache+ind_Wncacheshift];
                                    double complex nextN = c->wWncache[(nmax+thisn)*nshift_cache+ind_Wncacheshift];
                                    int _upsshift = thisnshift_ups + zcombi*c->upsilon_zshift + elb1*nbinsr;
                                    int _normshift = thisnshift_norm+ zcombi*c->upsilon_zshift + elb1*nbinsr;
                                    ind_Wncacheshift = zbin3*c->zbin2shift+c->zbinshifts[zbin1]+rbinmin2*c->thetashifts_z[zbin1]+
                                        c->cumresoshift_z[zbin1*(nresos+1) + thisreso2] + elgal;
                                    for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                        c->tmpUpsilon[_upsshift+elb2] += nextUpsm *
                                            c->Gncache[(nmax-thisn)*nshift_cache+ind_Wncacheshift];
                                        c->tmpUpsilon[_upsshift+c->upsilon_compshift+elb2] += nextUpsp *
                                            conj(c->Gncache[(nmax+thisn)*nshift_cache+ind_Wncacheshift]);
                                        c->tmpNorm[_normshift+elb2] += nextN *
                                            conj(c->Wncache[(nmax+thisn)*nshift_cache+ind_Wncacheshift]);
                                        ind_Wncacheshift += _thetashift_z;
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
                                    int ind_Wncacheshift = zbin2*c->zbin2shift + c->zbinshifts[zbin1] + elb1*c->thetashifts_z[zbin1]+
                                        c->cumresoshift_z[zbin1*(nresos+1) + thisreso1] + elgal;
                                    double complex nextUpsp = c->Gncache[(nmax+thisn)*nshift_cache+ind_Wncacheshift];
                                    double complex nextUpsm = c->Gncache[(nmax+thisn)*nshift_cache+ind_Wncacheshift];
                                    double complex nextN = c->Wncache[(nmax+thisn)*nshift_cache+ind_Wncacheshift];
                                    int _upsshift = thisnshift_ups + zcombi*c->upsilon_zshift + elb1*nbinsr;
                                    int _normshift = thisnshift_norm+ zcombi*c->upsilon_zshift + elb1*nbinsr;
                                    ind_Wncacheshift = zbin3*c->zbin2shift+c->zbinshifts[zbin1]+rbinmin2*c->thetashifts_z[zbin1]+
                                        c->cumresoshift_z[zbin1*(nresos+1) + thisreso1] + elgal;
                                    for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                        c->tmpUpsilon[_upsshift+elb2] += nextUpsm *
                                            c->wGncache[(nmax-thisn)*nshift_cache+ind_Wncacheshift];
                                        c->tmpUpsilon[_upsshift+c->upsilon_compshift+elb2] += nextUpsp *
                                            conj(c->wGncache[(nmax+thisn)*nshift_cache+ind_Wncacheshift]);
                                        c->tmpNorm[_normshift+elb2] += nextN *
                                            conj(c->wWncache[(nmax+thisn)*nshift_cache+ind_Wncacheshift]);
                                        ind_Wncacheshift += _thetashift_z;
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

// Reduce the per-thread Upsilon_n / N_n accumulators into the NPCFOutput and
// fill the (zbin_source, zbin_lens) bin_centers. Shared by discrete/tree/doubletree.
static void ngg_reduce(int nbinsz_lens, int nbinsz_source, int nbinsr, int nmax,
    int nthreads, int upsiloncompshift, int upsilonthreadshift, int normthreadshift,
    const double complex *tmpUpsilon, const double complex *tmpNorm,
    const double *tmpwcounts, const double *tmpwnorms, NPCFOutput *out){
    int nzcombis = nbinsz_lens*nbinsz_source*nbinsz_source;
    int upsilon_zshift = nbinsr*nbinsr;
    int upsilon_nshift = upsilon_zshift*nzcombis;
    #pragma omp parallel for num_threads(nthreads)
    for (int thisn=0; thisn<2*nmax+1; thisn++){
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            int thisthreadshift = thisthread*upsilonthreadshift;
            for (int zcombi=0; zcombi<nzcombis; zcombi++){
                for (int elb1=0; elb1<nbinsr; elb1++){
                    for (int elb2=0; elb2<nbinsr; elb2++){
                        int iUps = thisn*upsilon_nshift + zcombi*upsilon_zshift + elb1*nbinsr + elb2;
                        out->npcf[iUps] += tmpUpsilon[thisthreadshift+iUps];
                        out->npcf[upsiloncompshift+iUps] += tmpUpsilon[thisthreadshift+upsiloncompshift+iUps];
                        out->norm_mp[iUps] += tmpNorm[thisthread*normthreadshift+iUps];
                    }
                }
            }
        }
    }
    double *totcounts = calloc(nbinsz_lens*nbinsz_source*nbinsr, sizeof(double));
    double *totnorms = calloc(nbinsz_lens*nbinsz_source*nbinsr, sizeof(double));
    for (int thisthread=0; thisthread<nthreads; thisthread++){
        int thisthreadshift = thisthread*nbinsz_source*nbinsz_lens*nbinsr;
        for (int elbinz=0; elbinz<nbinsz_source*nbinsz_lens; elbinz++){
            for (int elbinr=0; elbinr<nbinsr; elbinr++){
                int tmpind = elbinz*nbinsr + elbinr;
                totcounts[tmpind] += tmpwcounts[thisthreadshift+tmpind];
                totnorms[tmpind] += tmpwnorms[thisthreadshift+tmpind];
            }
        }
    }
    for (int elbinz=0; elbinz<nbinsz_source*nbinsz_lens; elbinz++){
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            int tmpind = elbinz*nbinsr + elbinr;
            if (totnorms[tmpind] != 0){ out->bin_centers[tmpind] = totcounts[tmpind]/totnorms[tmpind]; }
        }
    }
    free(totcounts); free(totnorms);
}

// Per-pair fill of the spin-2 shape multipoles G_n and count multipoles W_n
// and their double counting corrections, assuming nmin=0. Used for all 
// approximations in a flat geometry
static inline void ngg_fill_gnwn(
    double complex *thisGns, double complex *thisWns,
    double complex *thisG2ns, double complex *thisW2ns,
    int z2rshift, int nbinszr_Gn, int nbinszr_Wn, int nmax,
    double w_gal1, double w_gal2, double complex wshape_gal2, double complex phirot){
    thisG2ns[z2rshift] += w_gal1*wshape_gal2*wshape_gal2*conj(phirot*phirot*phirot*phirot);
    thisG2ns[nbinszr_Gn+z2rshift] += w_gal1*wshape_gal2*conj(wshape_gal2);
    thisW2ns[z2rshift] += w_gal1*w_gal2*w_gal2;
    int ind_Wnp = nmax*nbinszr_Wn + z2rshift;
    int ind_Wnm = ind_Wnp;
    int ind_Gnp = (nmax+2)*nbinszr_Gn+z2rshift;
    int ind_Gnm = ind_Gnp;
    double complex nphirot = 1;
    thisGns[ind_Gnp] += wshape_gal2;
    thisWns[ind_Wnp] += w_gal2;
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

// Accumulation of NGG numerator in multiple space, used for slab geom accumulation.
//   Ups_-(t1,t2) += fac_c [G_{n-2}(t1) G_{-n-2}(t2) - dc]
//   Ups_+(t1,t2) += fac_c [G_{n-2}(t1) conj(G_{n-2}(t2)) - dc]
// TODO: This is super clunky at the moment...only used once...
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
        int blk1 = (nmax+thisn)*nbinszr;
        int blk2 = (nmax-thisn)*nbinszr;
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

// Accumulate normalisation function for NGG, used for slab geom accumulation.
// TODO: This is super clunky at the moment...only used once...
static void ngg_accum_norm(
    double complex *thNorm, double w_c, int zc,
    const double complex *Wn, const double complex *sumW2,
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
        int blk1 = (nmax+thisn)*nbinszr;
        int blk2 = (nmax-thisn)*nbinszr;
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


////////////////////////////
// NNN CORRELATOR CLASSES //
////////////////////////////

// Scalar NNN (triplet-count) DoubleTree, flat geometry. Spin-0 reduction of
// alloc_ggg_doubletree_flat: only the count multipoles N_n = sum_g2 w_g2 e^{i n phi}
// are built (no shear G_n leaf), and the per-thread accumulator is the single complex
// Triplets_n cache. Reuses the generic region-setup helpers and the nnn_* helpers.
static void alloc_nnn_doubletree_flat(const MultiresoCatalog *cat, const NavHash *nav,
                      const TreeResoParams *tree, const BinningParams *bin,
                      int nthreads, int verbose, NPCFOutput *out){
    // Dereference passed structures
    int nresos = tree->nresos, nresos_grid = tree->nresos_grid;
    double *dpix1_resos = tree->dpix1_resos, *dpix2_resos = tree->dpix2_resos;
    double *reso_redges = tree->reso_redges;
    int resoshift_leafs = tree->resoshift_leafs;
    int minresoind_leaf = tree->minresoind_leaf, maxresoind_leaf = tree->maxresoind_leaf;
    int *ngal_resos = cat->ngal_resos, nbinsz = cat->nbinsz;
    double *isinner_resos = cat->isinner_resos, *weight_resos = cat->weight_resos;
    double *pos1_resos = cat->pos1_resos, *pos2_resos = cat->pos2_resos;
    int *zbin_resos = cat->zbin_resos;
    int *index_matcher = nav->index_matcher;
    int *pixs_galind_bounds = nav->pixs_galind_bounds, *pix_gals = nav->pix_gals;
    double pix1_start = nav->pix1_start, pix1_d = nav->pix1_d; int pix1_n = nav->pix1_n;
    double pix2_start = nav->pix2_start, pix2_d = nav->pix2_d; int pix2_n = nav->pix2_n;
    int *index_matcher_hash = nav->index_matcher_hash;
    int *filledregions = nav->filledregions, nfilledregions = nav->nfilledregions;
    int nmax = bin->nmax, nbinsr = bin->nbinsr, dccorr = bin->dccorr;
    double rmin = bin->rmin, rmax = bin->rmax;

    // Index shift for the triplet counts
    int _gamma_zshift = nbinsr*nbinsr;
    int _gamma_nshift = _gamma_zshift*nbinsz*nbinsz*nbinsz;
    int _gamma_compshift = (nmax+1)*_gamma_nshift;

    // Temporary arrays that are allocated in parallel region and later reduced
    int nregionsdone = 0;
    reset_progress();
    double *tmpwcounts = calloc(nthreads*nbinsz*nbinsr, sizeof(double));
    double *tmpwnorms = calloc(nthreads*nbinsz*nbinsr, sizeof(double));
    double complex *tmpTriplets_n = calloc(nthreads*_gamma_compshift, sizeof(double complex));

    #pragma omp parallel num_threads(nthreads)
    {
        int elthread = omp_get_thread_num();
        int hasdiscrete = nresos-nresos_grid;
        int nnvals_Nn = nmax+1;

        // Nn caches grown on demand to the region's nshift.
        long cache_cap = 0;
        double complex *Nncache=NULL, *wNncache=NULL;

        NnnContext ctx;
        ctx.nbinsz=nbinsz; ctx.nbinsr=nbinsr; ctx.nmax=nmax; ctx.nresos=nresos;
        ctx.nnvals_Nn=nnvals_Nn;
        ctx.gamma_zshift=_gamma_zshift; ctx.gamma_nshift=_gamma_nshift; ctx.gamma_compshift=_gamma_compshift;
        ctx.dccorr=dccorr; ctx.elthread=elthread;
        ctx.tmpTriplets_n=tmpTriplets_n;

        #pragma omp for schedule(dynamic, 8)
        for (int _elregion=0; _elregion<nfilledregions; _elregion++){
            int elregion = filledregions[_elregion];

            // Check which sets of radii are evaluated for each resolution
            double logrmin = log(rmin);
            double drbin = (log(rmax)-logrmin)/(nbinsr);
            int *reso_rindedges = calloc(nresos+1, sizeof(int));
            build_reso_rindedges(nresos, reso_redges, rmin, rmax, nbinsr, reso_rindedges);
            ctx.reso_rindedges = reso_rindedges;

            // Shift variables for spatial hash
            int npix_hash = pix1_n*pix2_n;
            int *rshift_index_matcher = calloc(nresos, sizeof(int));
            int *rshift_pixs_galind_bounds = calloc(nresos, sizeof(int));
            int *rshift_pix_gals = calloc(nresos, sizeof(int));
            build_rshift_offsets(nresos, npix_hash, ngal_resos,
                rshift_index_matcher, rshift_pixs_galind_bounds, rshift_pix_gals);

            // Shift variables for the matching between the pixel grids
            int *matchers_resoshift = calloc(nresos_grid+1, sizeof(int));
            int *ngal_in_pix = calloc(nresos*nbinsz, sizeof(int));
            int len_matcher = build_region_galinpix(nresos, nresos_grid, hasdiscrete,
                elregion, pixs_galind_bounds, rshift_pixs_galind_bounds,
                pix_gals, rshift_pix_gals, zbin_resos, matchers_resoshift, ngal_in_pix);
            ctx.ngal_in_pix = ngal_in_pix;

            // Build the matcher from pixels to reduced pixels in the region
            double hashpix_start1, hashpix_start2;
            int *pix2redpix = calloc(nbinsz*len_matcher, sizeof(int));
            build_region_pix2redpix(nresos_grid, hasdiscrete, elregion, nbinsz,
                index_matcher_hash, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d,
                pixs_galind_bounds, rshift_pixs_galind_bounds, pix_gals, rshift_pix_gals,
                zbin_resos, pos1_resos, pos2_resos, dpix1_resos, dpix2_resos,
                matchers_resoshift, len_matcher, &hashpix_start1, &hashpix_start2, pix2redpix);

            // Setup all shift variables for the Nncache in the region
            int *cumresoshift_z = calloc(nbinsz*(nresos+1), sizeof(int));
            int *thetashifts_z = calloc(nbinsz, sizeof(int));
            int *zbinshifts = calloc(nbinsz+1, sizeof(int));
            ctx.cumresoshift_z = cumresoshift_z; ctx.thetashifts_z = thetashifts_z; ctx.zbinshifts = zbinshifts;
            setup_region_shifts(nbinsz, nbinsz, nresos, hasdiscrete, nbinsr, ngal_in_pix,
                cumresoshift_z, thetashifts_z, zbinshifts, &ctx.zbin2shift, &ctx.nshift);
            long need = (long)nnvals_Nn * ctx.nshift;
            if (need > cache_cap){
                cache_cap = need;
                Nncache = realloc(Nncache, cache_cap*sizeof(double complex));
                wNncache = realloc(wNncache, cache_cap*sizeof(double complex));
            }
            ctx.Nncache=Nncache; ctx.wNncache=wNncache;
            nnn_zero_caches(&ctx);

            // For each resolution, loop over all galaxies in the region and allocate
            // the Nn + their caches for the corresponding set of radii.
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
                double complex *nextWns = calloc(nnvals_Nn*nbinszr_reso, sizeof(double complex));
                double complex *nextW2ns = calloc(nbinszr_reso, sizeof(double complex));
                int *nextncounts = calloc(nbinszr_reso, sizeof(int));
                int *allowedrinds = calloc(nbinszr_reso, sizeof(int));
                int *allowedzinds = calloc(nbinszr_reso, sizeof(int));

                // Find leaf resolutions for the current base resolution
                int _leaf_lo, _leaf_hi;
                if (resoshift_leafs < 0) {
                    _leaf_lo = mymax(minresoind_leaf, elreso + resoshift_leafs);
                    _leaf_hi = mymin(elreso, maxresoind_leaf);
                    _leaf_lo = mymin(_leaf_lo, _leaf_hi);
                } else {
                    _leaf_lo = _leaf_hi = mymin(mymax(minresoind_leaf, elreso + resoshift_leafs), maxresoind_leaf);
                }
                // Needs two grid resolutions to form a ratio; with fewer, the leaf band
                // is a single resolution and the value is never used.
                double _dpix_ratio = (nresos_grid>1) ?
                    dpix1_resos[nresos_grid-1] / dpix1_resos[nresos_grid-2] : 1.;

                for (int ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                    int ind_gal1 = rshift_pix_gals[elreso] + pix_gals[rshift_pix_gals[elreso]+ind_inpix1];
                    double innergal = isinner_resos[ind_gal1];
                    if (innergal<1e-5){continue;}
                    int z_gal1 = zbin_resos[ind_gal1];
                    double pos1_gal1 = pos1_resos[ind_gal1];
                    double pos2_gal1 = pos2_resos[ind_gal1];
                    double w_gal1 = innergal*weight_resos[ind_gal1];

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
                        int lower2, upper2;
                        FLATCELL_FOREACH(
                            index_matcher, rshift_index_matcher[elreso_leaf], pixs_galind_bounds, rshift_pixs_galind_bounds[elreso_leaf],
                            pos1_gal1, pos2_gal1, _rmax_sub, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, lower2, upper2){
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
                                double complex phirot = (rel1+I*rel2)/dist;
                                int zrshift = z_gal2*nbinsr_reso + rbin;
                                int ind_rbin = elthread*nbinsz*nbinsr + z_gal2*nbinsr + rbin+rbinmin;
                                nextncounts[zrshift] += 1;
                                tmpwcounts[ind_rbin] += w_gal1*w_gal2*dist;
                                tmpwnorms[ind_rbin] += w_gal1*w_gal2;
                                // Scalar count multipoles N_n = sum w_gal2 e^{i n phi}, n=0..nmax
                                double complex nphirot = 1.+0.*I;
                                int ind_Wn = zrshift;
                                for (int nextn=0; nextn<=nmax; nextn++){
                                    nextWns[ind_Wn + nextn*nbinszr_reso] += w_gal2*nphirot;
                                    nphirot *= phirot;
                                }
                                nextW2ns[zrshift] += w_gal2*w_gal2;
                            }
                        }
                    }

                    build_redpix_by_reso2(elreso, nresos, nresos_grid, hasdiscrete,
                        z_gal1, pos1_gal1, pos2_gal1, hashpix_start1, hashpix_start2,
                        dpix1_resos, dpix2_resos, matchers_resoshift, len_matcher,
                        pix2redpix, redpix_by_reso2);
                    nnn_update_nncache(&ctx, elreso, rbinmin, rbinmax, nbinsr_reso, z_gal1, w_gal1,
                                       redpix_by_reso2, nextWns);
                    nnn_accum_samereso(&ctx, rbinmin, nbinsr_reso, z_gal1, w_gal1,
                                       nextWns, nextW2ns, nextncounts, allowedrinds, allowedzinds);

                    for (int _i=0;_i<nnvals_Nn*nbinszr_reso;_i++){nextWns[_i]=0;}
                    for (int _i=0;_i<nbinszr_reso;_i++){nextW2ns[_i]=0; nextncounts[_i]=0; allowedrinds[_i]=0; allowedzinds[_i]=0;}
                }
                free(nextWns); free(nextW2ns);
                free(nextncounts); free(allowedrinds); free(allowedzinds);
            }

            nnn_accum_crossreso(&ctx);

            free(reso_rindedges);
            free(rshift_index_matcher); free(rshift_pixs_galind_bounds); free(rshift_pix_gals);
            free(matchers_resoshift); free(ngal_in_pix); free(pix2redpix);
            free(cumresoshift_z); free(thetashifts_z); free(zbinshifts);
            free(redpix_by_reso2);
            #pragma omp atomic
            nregionsdone += 1;
            print_progress(nregionsdone, nfilledregions, verbose);
        }
        free(Nncache); free(wNncache);
    }

    nnn_reduce(nbinsz, nbinsr, nmax, nthreads, tmpTriplets_n, tmpwcounts, tmpwnorms, out);
    if (verbose>0){printf("\n");}
    free(tmpwcounts); free(tmpwnorms); free(tmpTriplets_n);
}

// Scalar NNN (triplet-count) DoubleTree, curved-sky geometry.
static void alloc_nnn_doubletree_spherical(const MultiresoCatalog *cat, const NavHash *nav,
                           const TreeResoParams *tree, const BinningParams *bin,
                           int nthreads, int verbose, NPCFOutput *out){

    // Dereference the passed structs
    int nresos = tree->nresos, nresos_grid = tree->nresos_grid;
    double *reso_redges = tree->reso_redges;
    int resoshift_leafs = tree->resoshift_leafs;
    int minresoind_leaf = tree->minresoind_leaf, maxresoind_leaf = tree->maxresoind_leaf;
    int nbinsz = cat->nbinsz;
    double *isinner = cat->isinner_resos, *weight = cat->weight_resos;
    double *vx = cat->vx_resos, *vy = cat->vy_resos, *vz = cat->vz_resos;
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
    double complex *tmpTriplets_n = calloc(nthreads*_gamma_compshift, sizeof(double complex));

    // Per-reso nested levels; regions = cells of the coarsest band (smallest nside).
    int *level = calloc(nresos, sizeof(int));
    int r_region = 0;
    for (int r=0;r<nresos;r++){ level[r] = ggg_nside_level(nside_nav[r]); if (level[r] < level[r_region]) r_region = r; }
    int l_region = level[r_region];
    int nregions = ncells_resos[r_region];
    const long *region_cellpix = cell_pix + rshift_cellpix[r_region];

    int nregionsdone = 0;
    reset_progress();
    double logrmin = log(rmin);
    double drbin = (log(rmax)-logrmin)/(nbinsr);
    #pragma omp parallel num_threads(nthreads)
    {
        int elthread = omp_get_thread_num();
        int nnvals_Nn = nmax+1;

        long cache_cap = 0;
        double complex *Nncache=NULL, *wNncache=NULL;

        int *reso_rindedges = calloc(nresos+1, sizeof(int));
        build_reso_rindedges(nresos, reso_redges, rmin, rmax, nbinsr, reso_rindedges);

        int *ngal_in_pix = calloc(nresos*nbinsz, sizeof(int));
        int *cumresoshift_z = calloc(nbinsz*(nresos+1), sizeof(int));
        int *thetashifts_z = calloc(nbinsz, sizeof(int));
        int *zbinshifts = calloc(nbinsz+1, sizeof(int));
        int *slice_clo = calloc(nresos, sizeof(int));
        int *slice_chi = calloc(nresos, sizeof(int));
        int **cellzidx = calloc(nresos, sizeof(int*));

        NnnContext ctx;
        ctx.nbinsz=nbinsz; ctx.nbinsr=nbinsr; ctx.nmax=nmax; ctx.nresos=nresos;
        ctx.nnvals_Nn=nnvals_Nn;
        ctx.gamma_zshift=_gamma_zshift; ctx.gamma_nshift=_gamma_nshift; ctx.gamma_compshift=_gamma_compshift;
        ctx.dccorr=dccorr; ctx.elthread=elthread;
        ctx.tmpTriplets_n=tmpTriplets_n;
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

            setup_region_shifts(nbinsz, nbinsz, nresos, hasdiscrete, nbinsr, ngal_in_pix,
                cumresoshift_z, thetashifts_z, zbinshifts, &ctx.zbin2shift, &ctx.nshift);
            long need = (long)nnvals_Nn * ctx.nshift;
            if (need > cache_cap){
                cache_cap = need;
                Nncache = realloc(Nncache, cache_cap*sizeof(double complex));
                wNncache = realloc(wNncache, cache_cap*sizeof(double complex));
            }
            ctx.Nncache=Nncache; ctx.wNncache=wNncache;
            nnn_zero_caches(&ctx);

            for (int elreso=0; elreso<nresos; elreso++){
                int rbinmin = reso_rindedges[elreso];
                int rbinmax = reso_rindedges[elreso+1];
                if (rbinmax <= rbinmin){ continue; }
                double rmin_reso = rmin*exp(rbinmin*drbin);
                double rmax_reso = rmin*exp(rbinmax*drbin);
                int nbinsr_reso = rbinmax-rbinmin;
                int nbinszr_reso = nbinsz*nbinsr_reso;
                int elreso_leaf = mymin(mymax(minresoind_leaf, elreso+resoshift_leafs), maxresoind_leaf);
                long ns_leaf = nside_nav[elreso_leaf];
                long redleaf_off = rshift_red[elreso_leaf];
                const long *cellpix_leaf = cell_pix + rshift_cellpix[elreso_leaf];
                const int *bounds_leaf = cell_redbounds + rshift_cellbounds[elreso_leaf];
                int ncells_leaf = ncells_resos[elreso_leaf];

                double complex *nextWns = calloc(nnvals_Nn*nbinszr_reso, sizeof(double complex));
                double complex *nextW2ns = calloc(nbinszr_reso, sizeof(double complex));
                int *nextncounts = calloc(nbinszr_reso, sizeof(int));
                int *allowedrinds = calloc(nbinszr_reso, sizeof(int));
                int *allowedzinds = calloc(nbinszr_reso, sizeof(int));

                // Base galaxies of this band in the region = its cell slice.
                const int *cb1 = cell_redbounds + rshift_cellbounds[elreso];
                for (int cc=slice_clo[elreso]; cc<slice_chi[elreso]; cc++){
                    for (int j1=cb1[cc]; j1<cb1[cc+1]; j1++){
                        long g1 = rshift_red[elreso] + j1;
                        double innergal = isinner[g1];
                        if (innergal<1e-5){continue;}
                        int z_gal1 = zbin[g1];
                        double cx = vx[g1], cy = vy[g1], cz = vz[g1];
                        double w_gal1 = innergal*weight[g1];

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
                                    double w_gal2 = weight[g2];
                                    int z_gal2 = zbin[g2];
                                    // Scalar polar angle of g2 about the base (spin-0):
                                    // phi = bearing(center->g2), matching the flat kernel's angle of (g2-center).
                                    BearingAB g12 = bearing_AB_cart(cx,cy,cz, vx[g2],vy[g2],vz[g2]);
                                    double complex phirot = bearing_phirot(g12);
                                    int zrshift = z_gal2*nbinsr_reso + rbin;
                                    int ind_rbin = elthread*nbinsz*nbinsr + z_gal2*nbinsr + rbin+rbinmin;
                                    nextncounts[zrshift] += 1;
                                    tmpwcounts[ind_rbin] += w_gal1*w_gal2*dist;
                                    tmpwnorms[ind_rbin] += w_gal1*w_gal2;
                                    double complex nphirot = 1.+0.*I;
                                    int ind_Wn = zrshift;
                                    for (int nextn=0; nextn<=nmax; nextn++){
                                        nextWns[ind_Wn + nextn*nbinszr_reso] += w_gal2*nphirot;
                                        nphirot *= phirot;
                                    }
                                    nextW2ns[zrshift] += w_gal2*w_gal2;
                                }
                                ci++;
                            }
                        }

                        // Cross-reso cache index: map the base to its coarse reduced galaxy
                        // cell at each reso2 (ang2pix), then its dense per-zbin slot.
                        for (int elreso2=elreso; elreso2<nresos; elreso2++){
                            int grid_reso = elreso2 - hasdiscrete;
                            if (hasdiscrete==1 && elreso==0 && elreso2==0){ grid_reso += hasdiscrete; }
                            int map_reso = grid_reso + hasdiscrete;
                            int redpix = 0;
                            if (map_reso >= nresos){ redpix_by_reso2[elreso2] = 0; continue; }
                            if (map_reso == elreso){
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
                        nnn_update_nncache(&ctx, elreso, rbinmin, rbinmax, nbinsr_reso, z_gal1, w_gal1,
                                           redpix_by_reso2, nextWns);
                        nnn_accum_samereso(&ctx, rbinmin, nbinsr_reso, z_gal1, w_gal1,
                                           nextWns, nextW2ns, nextncounts, allowedrinds, allowedzinds);

                        for (int _i=0;_i<nnvals_Nn*nbinszr_reso;_i++){nextWns[_i]=0;}
                        for (int _i=0;_i<nbinszr_reso;_i++){nextW2ns[_i]=0; nextncounts[_i]=0; allowedrinds[_i]=0; allowedzinds[_i]=0;}
                    }
                }
                free(nextWns); free(nextW2ns);
                free(nextncounts); free(allowedrinds); free(allowedzinds);
            }

            nnn_accum_crossreso(&ctx);

            for (int r=0;r<nresos;r++){ free(cellzidx[r]); }
            #pragma omp atomic
            nregionsdone += 1;
            print_progress(nregionsdone, nregions, verbose);
        }
        free(ranges); free(redpix_by_reso2);
        free(Nncache); free(wNncache);
        free(reso_rindedges); free(ngal_in_pix);
        free(cumresoshift_z); free(thetashifts_z); free(zbinshifts);
        free(slice_clo); free(slice_chi); free(cellzidx);
    }
    free(level);

    nnn_reduce(nbinsz, nbinsr, nmax, nthreads, tmpTriplets_n, tmpwcounts, tmpwnorms, out);
    if (verbose>0){printf("\n");}
    free(tmpwcounts); free(tmpwnorms); free(tmpTriplets_n);
}

// Public entry point: choose the worker based on the catalog metric.
void alloc_nnn_doubletree(const MultiresoCatalog *cat, const NavHash *nav,
                          const TreeResoParams *tree, const BinningParams *bin,
                          int nthreads, int verbose, NPCFOutput *out){
    switch (cat->metric) {
        case METRIC_SPHERICAL:
            alloc_nnn_doubletree_spherical(cat, nav, tree, bin, nthreads, verbose, out);
            break;
        case METRIC_FLAT:
        default:
            alloc_nnn_doubletree_flat(cat, nav, tree, bin, nthreads, verbose, out);
            break;
    }
}


////////////////////////////
// GGG CORRELATOR CLASSES //
////////////////////////////

// GGG USING THE DISCRETE ESTIMATOR //
void alloc_Gammans_discrete_ggg(const MultiresoCatalog *cat, const NavHash *nav,
                                const BinningParams *bin, int nthreads, int verbose,
                                NPCFOutput *out){
    // Unpack passed structs
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
    int nregionsdone = 0;
    reset_progress();
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
                #pragma omp atomic
                nregionsdone += 1;
                print_progress(nregionsdone, ngal, verbose);

                int ind_inpix, ind_gal2;
                int lower, upper;
                double  p21, p22, w2, z2, e21, e22;
                double rel1, rel2, dist;
                double complex wshape;
                int nnvals, nnvals_norm, nzero;
                double complex twophirotc, phirot, phirotc;
                
                // Check how many ns we need for Gn
                // Gns have shape (nnvals, nbinsz, nbinsr)
                // where the ns are ordered as 
                //  * [-nmax-3, ..., nmax-1] / [0, ..., nmax]
                if (nmin<4){nmin=0;}
                if (nmin==0){nnvals=2*nmax+3;nnvals_norm=nmax+1;}
                else{nnvals=2*(nmax-nmin+3);nnvals_norm=nmax-nmin+1;}
                double complex *nextGns =  calloc(nnvals*nbinsr*nbinsz, sizeof(double complex));
                double complex *nextWns =  calloc(nnvals_norm*nbinsr*nbinsz, sizeof(double complex));
                double complex *nextG2ns =  calloc(4*nbinsz*nbinsr, sizeof(double complex));
                double complex *nextW2ns =  calloc(nbinsz*nbinsr, sizeof(double complex));

                int ind_rbin, rbin;
                int zrshift;
                int nbinszr = nbinsz*nbinsr;
                double drbin = (log(rmax)-log(rmin))/(nbinsr);
                /*if (ind_gal%10000==0){
                    printf("%d %d %d %d %d \n",nmin,nmax,nnvals,nbinsr,nbinsz);
                }*/
                FLATCELL_FOREACH(
                    index_matcher, 0, pixs_galind_bounds, 0, p11, p12, rmax,
                    pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, lower, upper){
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
                        //   -> Gns axis: [-(nmax+1), ..., nmax+1] (projected)
                        //   -> Wns axis: [0,...,nmax]
                        if (nmin==0){
                            tmpwcounts[ind_rbin] += w1*w2*dist;
                            tmpwnorms[ind_rbin] += w1*w2;
                            ggg_fill_GnWn_projected(nextGns, nextWns, nextG2ns, nextW2ns,
                                zrshift, nbinszr, nmax, w2, wshape*twophirotc, phirot, twophirotc);
                        }
                        else{
                            ggg_fill_GnWn_nminband(nextGns, nextWns, zrshift, nbinszr,
                                nmin, nmax, w2, wshape, phirot, phirotc);
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
                            w0 = w1 * conj(nextWns[ind_norm + zrshift]);
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
                                    tmpGammans_norm[gammashiftt] -= w1*nextW2ns[zrshift];
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
                                    tmpGammans_norm[gammashiftt] += w0*nextWns[ind_norm + zrshift];
                                    //if(thisthread==0 && ind_gal%1000==0){
                                    //    if (cabs(tmpGammans[gammashift] )>1e-5){nonzero_tmpGammas += 1;}
                                    //}
                                }
                            }
                        }
                    }
                }
                
                free(nextGns);
                free(nextWns);
                free(nextG2ns);
                free(nextW2ns);
                nextGns = NULL;
                nextWns = NULL;
                nextG2ns = NULL;
                nextW2ns = NULL;
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
    if (verbose>0){ printf("\n"); }
}

// GGG using the tree-based approximation //
void alloc_Gammans_tree_ggg(const MultiresoCatalog *cat, const MultiresoCatalog *cat_field,
                            const NavHash *nav, const TreeResoParams *tree,
                            const BinningParams *bin, int nthreads, int verbose,
                            NPCFOutput *out){
    // Unpack the passed structs
    double *isinner = cat->isinner_resos, *weight = cat->weight_resos;
    double *pos1 = cat->pos1_resos, *pos2 = cat->pos2_resos;
    double *e1 = cat->e1_resos, *e2 = cat->e2_resos;
    int *zbins = cat->zbin_resos, nbinsz = cat->nbinsz, ngal = cat->ngal_resos[0];
    int nresos = tree->nresos; double *reso_redges = tree->reso_redges;
    int *ngal_resos = cat_field->ngal_resos, *zbin_resos = cat_field->zbin_resos;
    double *weight_resos = cat_field->weight_resos, *pos1_resos = cat_field->pos1_resos, *pos2_resos = cat_field->pos2_resos;
    double *e1_resos = cat_field->e1_resos, *e2_resos = cat_field->e2_resos, *weightsq_resos = cat_field->weightsq_resos;
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
    int nregionsdone = 0;
    reset_progress();
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
                
                #pragma omp atomic
                nregionsdone += 1;
                print_progress(nregionsdone, ngal, verbose);
                
                
                int ind_inpix, ind_gal2;
                int lower, upper;
                double  p21, p22, w2, z2, e21, e22;
                double rel1, rel2, dist;
                double complex wshape;
                int nnvals, nnvals_norm, nzero;
                double complex twophirotc, phirot, phirotc;

                // Check how many ns we need for Gn
                // Gns have shape (nnvals, nbinsz, nbinsr)
                // where the ns are ordered as
                //  * [-nmax-3, ..., nmax-1] / [0, ..., nmax]
                if (nmin<4){nmin=0;}
                if (nmin==0){nnvals=2*nmax+3;nnvals_norm=nmax+1;}
                else{nnvals=2*(nmax-nmin+3);nnvals_norm=nmax-nmin+1;}
                double complex *nextGns =  calloc(nnvals*nbinsr*nbinsz, sizeof(double complex));
                double complex *nextWns =  calloc(nnvals_norm*nbinsr*nbinsz, sizeof(double complex));
                double complex *nextG2ns =  calloc(4*nbinsz*nbinsr, sizeof(double complex));
                double complex *nextW2ns =  calloc(nbinsz*nbinsr, sizeof(double complex));

                int ind_rbin, rbin;
                int zrshift;
                int nbinszr = nbinsz*nbinsr;
                double drbin = (log(rmax)-log(rmin))/(nbinsr);
                /*if (ind_gal%10000==0){
                    printf("%d %d %d %d %d \n",nmin,nmax,nnvals,nbinsr,nbinsz);
                }*/

                for (int elreso=0;elreso<nresos;elreso++){
                    FLATCELL_FOREACH(
                        index_matcher, rshift_index_matcher[elreso], pixs_galind_bounds, rshift_pixs_galind_bounds[elreso],
                        p11, p12, reso_redges[elreso+1], pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, lower, upper){
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
                            //   -> Gns axis: [-(nmax+1), ..., nmax+1] (projected)
                            //   -> Wns axis: [0,...,nmax]
                            if (nmin==0){
                                tmpwcounts[ind_rbin] += w1*w2*dist;
                                tmpwnorms[ind_rbin] += w1*w2;
                                ggg_fill_GnWn_projected(nextGns, nextWns, nextG2ns, nextW2ns,
                                    zrshift, nbinszr, nmax, w2, wshape*twophirotc, phirot, twophirotc);
                            }
                            else{
                                ggg_fill_GnWn_nminband(nextGns, nextWns, zrshift, nbinszr,
                                    nmin, nmax, w2, wshape, phirot, phirotc);
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
                            w0 = w1 * conj(nextWns[ind_norm + zrshift]);
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
                                    tmpGammans_norm[gammashiftt] -= w1*nextW2ns[zrshift];
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
                                    tmpGammans_norm[gammashiftt] += w0*nextWns[ind_norm + zrshift];
                                    //if(thisthread==0 && ind_gal%1000==0){
                                    //    if (cabs(tmpGammans[gammashift] )>1e-5){nonzero_tmpGammas += 1;}
                                    //}
                                }
                            }
                        }
                    }
                }
                
                free(nextGns);
                free(nextWns);
                free(nextG2ns);
                free(nextW2ns);
                nextGns = NULL;
                nextWns = NULL;
                nextG2ns = NULL;
                nextW2ns = NULL;
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
    if (verbose>0){ printf("\n"); }
}

// GGG using the BaseTree approximation //
// Exactly the same as doubletree, but here we bruteforce the calculation of the Gn
// --> ~Same speed as tree and accurate on the diagonals!
void alloc_Gammans_basetree_ggg(const MultiresoCatalog *cat, const NavHash *nav,
                                const TreeResoParams *tree, const BinningParams *bin,
                                int nthreads, int verbose, NPCFOutput *out){
    // Unpack passe structs
    int nresos = tree->nresos, nresos_grid = tree->nresos_grid;
    double *dpix1_resos = tree->dpix1_resos, *dpix2_resos = tree->dpix2_resos, *reso_redges = tree->reso_redges;
    int *ngal_resos = cat->ngal_resos, nbinsz = cat->nbinsz, *zbin_resos = cat->zbin_resos;
    double *isinner_resos = cat->isinner_resos, *weight_resos = cat->weight_resos;
    double *pos1_resos = cat->pos1_resos, *pos2_resos = cat->pos2_resos;
    double *e1_resos = cat->e1_resos, *e2_resos = cat->e2_resos, *weightsq_resos = cat->weightsq_resos;
    int *index_matcher = nav->index_matcher, *pixs_galind_bounds = nav->pixs_galind_bounds, *pix_gals = nav->pix_gals;
    double pix1_start = nav->pix1_start, pix1_d = nav->pix1_d; int pix1_n = nav->pix1_n;
    double pix2_start = nav->pix2_start, pix2_d = nav->pix2_d; int pix2_n = nav->pix2_n;
    int *index_matcher_hash = nav->index_matcher_hash, nregions = nav->nregions;
    int *filledregions = nav->filledregions, nfilledregions = nav->nfilledregions;
    int nmax = bin->nmax, nbinsr = bin->nbinsr, dccorr = bin->dccorr;
    double rmin = bin->rmin, rmax = bin->rmax;

    // Index shift for the Gamman
    int _gamma_zshift = nbinsr*nbinsr;
    int _gamma_nshift = _gamma_zshift*nbinsz*nbinsz*nbinsz;
    int _gamma_compshift = (nmax+1)*_gamma_nshift;
    
    // Temporary arrays that are allocated in parallel and later reduced
    double *tmpwcounts = calloc(nthreads*nbinsz*nbinsr, sizeof(double));
    double *tmpwnorms = calloc(nthreads*nbinsz*nbinsr, sizeof(double));
    double complex *tmpGamma0s = calloc(nthreads*_gamma_compshift, sizeof(double complex));
    double complex *tmpGamma1s = calloc(nthreads*_gamma_compshift, sizeof(double complex));
    double complex *tmpGamma2s = calloc(nthreads*_gamma_compshift, sizeof(double complex));
    double complex *tmpGamma3s = calloc(nthreads*_gamma_compshift, sizeof(double complex));
    double complex *tmpGammans_norm = calloc(nthreads*_gamma_compshift, sizeof(double complex));
    
    int nregionsdone = 0;
    reset_progress();
    #pragma omp parallel for num_threads(nthreads)
    for(int elthread=0;elthread<nthreads;elthread++){
        int nregions_per_thread = nregions/nthreads;
        int hasdiscrete = nresos-nresos_grid;
        int nnvals_Gn = 2*nmax+3;
        int nnvals_Nn = nmax+1;
        
        // Gn caches grown on demand to the region's nshift.
        long cache_cap = 0;
        double complex *Gncache=NULL, *wGncache=NULL, *cwGncache=NULL, *Nncache=NULL, *wNncache=NULL;

        GggContext ctx;
        ctx.nbinsz=nbinsz; ctx.nbinsr=nbinsr; ctx.nmax=nmax; ctx.nresos=nresos;
        ctx.nnvals_Gn=nnvals_Gn; ctx.nnvals_Nn=nnvals_Nn;
        ctx.gamma_zshift=_gamma_zshift; ctx.gamma_nshift=_gamma_nshift; ctx.gamma_compshift=_gamma_compshift;
        ctx.dccorr=dccorr; ctx.elthread=elthread;
        ctx.tmpGamma0s=tmpGamma0s; ctx.tmpGamma1s=tmpGamma1s; ctx.tmpGamma2s=tmpGamma2s;
        ctx.tmpGamma3s=tmpGamma3s; ctx.tmpGammans_norm=tmpGammans_norm;

        for (int elregion=0; elregion<nregions; elregion++){
            int region_debug=99999;
            bool printregdbg = (verbose>0) && (elregion==region_debug);
            // Check if this thread is responsible for the region
            int nthread_target = mymin(elregion/nregions_per_thread, nthreads-1);
            if (nthread_target!=elthread){continue;}
            // printf("Region %d is in thread %d\n",elregion,elthread);
            if (printregdbg){printf("Region %d is in thread %d\n",elregion,elthread);}
            #pragma omp atomic
            nregionsdone += 1;
            print_progress(nregionsdone, nregions, verbose);
            
            // Check which sets of radii are evaluated for each resolution
            int *reso_rindedges = calloc(nresos+1, sizeof(int));
            double logrmin = log(rmin);
            double drbin = (log(rmax)-logrmin)/(nbinsr);
            build_reso_rindedges(nresos, reso_redges, rmin, rmax, nbinsr, reso_rindedges);
                        
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
            ctx.reso_rindedges = reso_rindedges; ctx.ngal_in_pix = ngal_in_pix;
            ctx.cumresoshift_z = cumresoshift_z; ctx.thetashifts_z = thetashifts_z; ctx.zbinshifts = zbinshifts;
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
            int *redpix_by_reso2 = calloc(nresos, sizeof(int));
            
            // Now, for each resolution, loop over all the galaxies in the region and
            // allocate the Gn & Nn, as well as their caches  for the corresponding 
            // set of radii
            // For elreso in resos
            //.  for gal in reso 
            //.    allocate Gn for allowed radii
            //.    allocate the Gncaches
            //.    compute the Gamman for all combinations of the same resolution
            int ind_inpix1, ind_inpix2, ind_gal1, ind_gal2, z_gal1, z_gal2;
            int rbin, nbinszr, nbinszr_reso, zrshift, ind_rbin;
            double innergal, pos1_gal1, pos2_gal1, pos1_gal2, pos2_gal2, w_gal1, w_gal2, e1_gal1, e2_gal1, e1_gal2, e2_gal2;
            double rel1, rel2, dist;
            double complex wshape_gal1, wshape_gal2;
            double complex twophirotc, phirot, phirotc;
            double rmin_reso, rmax_reso, rmin_reso2, rmax_reso2;
            int rbinmin, rbinmax;
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
                double complex *nextWns =  calloc(nnvals_Nn*nbinszr_reso, sizeof(double complex));
                double complex *nextG2ns =  calloc(4*nbinszr_reso, sizeof(double complex));
                double complex *nextW2ns =  calloc(nbinszr_reso, sizeof(double complex));
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

                    FLATCELL_FOREACH(
                        index_matcher, rshift_index_matcher[elreso_leaf], pixs_galind_bounds, rshift_pixs_galind_bounds[elreso_leaf],
                        pos1_gal1, pos2_gal1, rmax_reso, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, lower2, upper2){
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
                            //   -> Gns axis: [-(nmax+1), ..., nmax+1] (projected)
                            //   -> Wns axis: [0,...,nmax]
                            nextncounts[zrshift] += 1;
                            tmpwcounts[ind_rbin] += w_gal1*w_gal2*dist;
                            tmpwnorms[ind_rbin] += w_gal1*w_gal2;
                            ggg_fill_GnWn_projected(nextGns, nextWns, nextG2ns, nextW2ns,
                                zrshift, nbinszr_reso, nmax, w_gal2, wshape_gal2*twophirotc, phirot, twophirotc);
                        }
                    }
                    // Update the region caches and the same-reso Upsilon_n
                    build_redpix_by_reso2(elreso, nresos, nresos_grid, hasdiscrete,
                        z_gal1, pos1_gal1, pos2_gal1, hashpix_start1, hashpix_start2,
                        dpix1_resos, dpix2_resos, matchers_resoshift, len_matcher,
                        pix2redpix, redpix_by_reso2);
                    ggg_update_gnwncache(&ctx, elreso, rbinmin, rbinmax, nbinsr_reso, z_gal1, w_gal1, wshape_gal1,
                                         redpix_by_reso2, nextGns, nextWns);
                    ggg_accum_samereso(&ctx, rbinmin, nbinsr_reso, z_gal1, w_gal1, wshape_gal1,
                                       nextGns, nextWns, nextG2ns, nextW2ns,
                                       nextncounts, allowedrinds, allowedzinds);
                    for (int _i=0;_i<nnvals_Gn*nbinszr_reso;_i++){nextGns[_i]=0;}
                    for (int _i=0;_i<nnvals_Nn*nbinszr_reso;_i++){nextWns[_i]=0;}
                    for (int _i=0;_i<4*nbinszr_reso;_i++){nextG2ns[_i]=0;}
                    for (int _i=0;_i<nbinszr_reso;_i++){nextW2ns[_i]=0; 
                                                        nextncounts[_i]=0; allowedrinds[_i]=0; allowedzinds[_i]=0;}
                }
                free(nextGns);
                free(nextWns);
                free(nextG2ns);
                free(nextW2ns);
                free(nextncounts);
                free(allowedrinds);
                free(allowedzinds);
            }            
            
            ggg_accum_crossreso(&ctx);

            free(redpix_by_reso2);
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
    }

    ggg_reduce(nbinsz, nbinsr, nmax, nthreads, tmpGamma0s, tmpGamma1s, tmpGamma2s, tmpGamma3s,
               tmpGammans_norm, tmpwcounts, tmpwnorms, out);
    if (verbose>0){ printf("\n"); }

    free(tmpwcounts);
    free(tmpwnorms);
    free(tmpGamma0s);
    free(tmpGamma1s);
    free(tmpGamma2s);
    free(tmpGamma3s);
    free(tmpGammans_norm);
}

// GGG using the DoubleTree approximation //

// Using the flat-sky geometry
static void alloc_ggg_doubletree_flat(const MultiresoCatalog *cat, const NavHash *nav,
                      const TreeResoParams *tree, const BinningParams *bin,
                      int nthreads, int verbose, NPCFOutput *out){
    // Dereference passed structures
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
    
    // Index shift for the Gamman
    int _gamma_zshift = nbinsr*nbinsr;
    int _gamma_nshift = _gamma_zshift*nbinsz*nbinsz*nbinsz;
    int _gamma_compshift = (nmax+1)*_gamma_nshift;
    
    // Temporary arrays that are allocated in parallel region and later reduced
    int nregionsdone = 0;
    reset_progress();
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

        // Gn caches grown on demand to the region's nshift.
        long cache_cap = 0;
        double complex *Gncache=NULL, *wGncache=NULL, *cwGncache=NULL, *Nncache=NULL, *wNncache=NULL;

        GggContext ctx;
        ctx.nbinsz=nbinsz; ctx.nbinsr=nbinsr; ctx.nmax=nmax; ctx.nresos=nresos;
        ctx.nnvals_Gn=nnvals_Gn; ctx.nnvals_Nn=nnvals_Nn;
        ctx.gamma_zshift=_gamma_zshift; ctx.gamma_nshift=_gamma_nshift; ctx.gamma_compshift=_gamma_compshift;
        ctx.dccorr=dccorr; ctx.elthread=elthread;
        ctx.tmpGamma0s=tmpGamma0s; ctx.tmpGamma1s=tmpGamma1s; ctx.tmpGamma2s=tmpGamma2s;
        ctx.tmpGamma3s=tmpGamma3s; ctx.tmpGammans_norm=tmpGammans_norm;

        #pragma omp for schedule(dynamic, 8)
        for (int _elregion=0; _elregion<nfilledregions; _elregion++){
            int elregion = filledregions[_elregion];

            // Check which sets of radii are evaluated for each resolution
            double logrmin = log(rmin);
            double drbin = (log(rmax)-logrmin)/(nbinsr);
            int *reso_rindedges = calloc(nresos+1, sizeof(int));
            build_reso_rindedges(nresos, reso_redges, rmin, rmax, nbinsr, reso_rindedges);
            ctx.reso_rindedges = reso_rindedges;
            
            // Shift variables for spatial hash
            int npix_hash = pix1_n*pix2_n;
            int *rshift_index_matcher = calloc(nresos, sizeof(int));
            int *rshift_pixs_galind_bounds = calloc(nresos, sizeof(int));
            int *rshift_pix_gals = calloc(nresos, sizeof(int));
            build_rshift_offsets(nresos, npix_hash, ngal_resos,
                rshift_index_matcher, rshift_pixs_galind_bounds, rshift_pix_gals);
            
            // Shift variables for the matching between the pixel grids
            int *matchers_resoshift = calloc(nresos_grid+1, sizeof(int));
            int *ngal_in_pix = calloc(nresos*nbinsz, sizeof(int));
            int len_matcher = build_region_galinpix(nresos, nresos_grid, hasdiscrete,
                elregion, pixs_galind_bounds, rshift_pixs_galind_bounds,
                pix_gals, rshift_pix_gals, zbin_resos, matchers_resoshift, ngal_in_pix);
            ctx.ngal_in_pix = ngal_in_pix;
            
            // Build the matcher from pixels to reduced pixels in the region
            double hashpix_start1, hashpix_start2;
            int *pix2redpix = calloc(nbinsz*len_matcher, sizeof(int));
            build_region_pix2redpix(nresos_grid, hasdiscrete, elregion, nbinsz,
                index_matcher_hash, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d,
                pixs_galind_bounds, rshift_pixs_galind_bounds, pix_gals, rshift_pix_gals,
                zbin_resos, pos1_resos, pos2_resos, dpix1_resos, dpix2_resos,
                matchers_resoshift, len_matcher, &hashpix_start1, &hashpix_start2, pix2redpix);
            
            // Setup all shift variables for the Gncache in the region
            // Gncache has structure
            // n --> zbin2 --> zbin1 --> radius 
            //   --> [ [0]*ngal_zbin1_reso1 | [0]*ngal_zbin1_reso1/2 | ... | [0]*ngal_zbin1_reson ]
            int *cumresoshift_z = calloc(nbinsz*(nresos+1), sizeof(int)); // Cumulative shift index for resolution at z1
            int *thetashifts_z = calloc(nbinsz, sizeof(int)); // Shift index for theta given z1
            int *zbinshifts = calloc(nbinsz+1, sizeof(int)); // Shifts for z2 index and n index
            ctx.cumresoshift_z = cumresoshift_z; ctx.thetashifts_z = thetashifts_z; ctx.zbinshifts = zbinshifts;
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

            // Now, for each resolution, loop over all the galaxies in the region and
            // allocate the Gn & Nn, as well as their caches  for the corresponding 
            // set of radii
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
                double complex *nextWns = calloc(nnvals_Nn*nbinszr_reso, sizeof(double complex));
                double complex *nextG2ns = calloc(4*nbinszr_reso, sizeof(double complex));
                double complex *nextW2ns = calloc(nbinszr_reso, sizeof(double complex));
                int *nextncounts = calloc(nbinszr_reso, sizeof(int));
                int *allowedrinds = calloc(nbinszr_reso, sizeof(int));
                int *allowedzinds = calloc(nbinszr_reso, sizeof(int));

                // Find leaf resolutions for current base resolution
                // This munches togehter thre resoshift_leafs parameter as well as the specified parameters for min/maxresoind_leaf
                // In particular, if the leafs are evaluated at a higher precision than the base this applies once the coarsest
                // base resolution is reached.
                int _leaf_lo, _leaf_hi;
                if (resoshift_leafs < 0) {
                    _leaf_lo = mymax(minresoind_leaf, elreso + resoshift_leafs);
                    _leaf_hi = mymin(elreso, maxresoind_leaf);
                    _leaf_lo = mymin(_leaf_lo, _leaf_hi);
                } else {
                    _leaf_lo = _leaf_hi = mymin(mymax(minresoind_leaf, elreso + resoshift_leafs), maxresoind_leaf);
                }
                // Needs two grid resolutions to form a ratio; with fewer, the leaf band
                // is a single resolution and the value is never used.
                double _dpix_ratio = (nresos_grid>1) ?
                    dpix1_resos[nresos_grid-1] / dpix1_resos[nresos_grid-2] : 1.;

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

                    for (int elreso_leaf = _leaf_lo; elreso_leaf <= _leaf_hi; elreso_leaf++) {
                        double _rmin_sub2, _rmax_sub2;
                        double _rmin_sub = rmin_reso, _rmax_sub = rmax_reso;
                        // In case we need to travese multiple leaf resolution we compute their bounds here
                        // We assume a logarithmic radial binning
                        if (resoshift_leafs < 0) {
                            // k=0: finest leaf covers [rmin_reso, rmin_reso*ratio]
                            // k=|shift|: coarsest leaf covers [rmin_reso*ratio^|shift|, rmax_reso]
                            int k = elreso_leaf - _leaf_lo;
                            _rmin_sub = rmin_reso * pow(_dpix_ratio, (double)k);
                            _rmax_sub = (elreso_leaf < _leaf_hi) ? rmin_reso * pow(_dpix_ratio, (double)(k+1)) : rmax_reso;
                            _rmin_sub = fmax(_rmin_sub, rmin_reso);
                            _rmax_sub = fmin(_rmax_sub, rmax_reso);
                            if (_rmin_sub >= _rmax_sub) continue;
                        }
                        _rmin_sub2=_rmin_sub*_rmin_sub;
                        _rmax_sub2=_rmax_sub*_rmax_sub;
                        int lower2, upper2;
                        FLATCELL_FOREACH(
                            index_matcher, rshift_index_matcher[elreso_leaf], pixs_galind_bounds, rshift_pixs_galind_bounds[elreso_leaf],
                            pos1_gal1, pos2_gal1, _rmax_sub, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, lower2, upper2){
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
                                nextncounts[zrshift] += 1;
                                tmpwcounts[ind_rbin] += w_gal1*w_gal2*dist;
                                tmpwnorms[ind_rbin] += w_gal1*w_gal2;
                                ggg_fill_GnWn_projected(nextGns, nextWns, nextG2ns, nextW2ns,
                                    zrshift, nbinszr_reso, nmax, w_gal2, wshape_gal2*twophirotc, phirot, twophirotc);
                            }
                        }
                    }

                    build_redpix_by_reso2(elreso, nresos, nresos_grid, hasdiscrete,
                        z_gal1, pos1_gal1, pos2_gal1, hashpix_start1, hashpix_start2,
                        dpix1_resos, dpix2_resos, matchers_resoshift, len_matcher,
                        pix2redpix, redpix_by_reso2);
                    ggg_update_gnwncache(&ctx, elreso, rbinmin, rbinmax, nbinsr_reso, z_gal1, w_gal1, wshape_gal1,
                                         redpix_by_reso2, nextGns, nextWns);
                    ggg_accum_samereso(&ctx, rbinmin, nbinsr_reso, z_gal1, w_gal1, wshape_gal1,
                                       nextGns, nextWns, nextG2ns, nextW2ns,
                                       nextncounts, allowedrinds, allowedzinds);

                    for (int _i=0;_i<nnvals_Gn*nbinszr_reso;_i++){nextGns[_i]=0;}
                    for (int _i=0;_i<nnvals_Nn*nbinszr_reso;_i++){nextWns[_i]=0;}
                    for (int _i=0;_i<4*nbinszr_reso;_i++){nextG2ns[_i]=0;}
                    for (int _i=0;_i<nbinszr_reso;_i++){nextW2ns[_i]=0; nextncounts[_i]=0; allowedrinds[_i]=0; allowedzinds[_i]=0;}
                }
                free(nextGns); free(nextWns); free(nextG2ns); free(nextW2ns);
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

// Using the full-sky geometry
static void alloc_ggg_doubletree_spherical(const MultiresoCatalog *cat, const NavHash *nav,
                           const TreeResoParams *tree, const BinningParams *bin,
                           int nthreads, int verbose, NPCFOutput *out){
    // Dereference the passed structs
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

    // Per-reso nested levels
    int *level = calloc(nresos, sizeof(int));
    int r_region = 0;
    for (int r=0;r<nresos;r++){ level[r] = ggg_nside_level(nside_nav[r]); if (level[r] < level[r_region]) r_region = r; }
    int l_region = level[r_region];
    int nregions = ncells_resos[r_region];
    const long *region_cellpix = cell_pix + rshift_cellpix[r_region];

    int nregionsdone = 0;
    reset_progress();
    double logrmin = log(rmin);
    double drbin = (log(rmax)-logrmin)/(nbinsr);
    #pragma omp parallel num_threads(nthreads)
    {
        int elthread = omp_get_thread_num();
        int nnvals_Gn = 2*nmax+3;
        int nnvals_Nn = nmax+1;

        // Gn caches grown on demand to the regions nshift.
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
        // Per-reso, per-cell-in-slice dense reduced galaxy index per zbin (cellzidx[r]).
        int **cellzidx = calloc(nresos, sizeof(int*));

        GggContext ctx;
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

            // Per-reso region slice + reduced-galaxy enumeration --> ngal_in_pix + cellzidx.
            for (int _i=0;_i<nresos*nbinsz;_i++){ ngal_in_pix[_i]=0; }
            for (int _i=0;_i<nbinsz*(nresos+1);_i++){ cumresoshift_z[_i]=0; }
            for (int _i=0;_i<nbinsz;_i++){ thetashifts_z[_i]=0; }
            for (int _i=0;_i<=nbinsz;_i++){ zbinshifts[_i]=0; }
            int has_inner = 0;
            for (int r=0;r<nresos;r++){
                const long *cp = cell_pix + rshift_cellpix[r];
                const int *cb = cell_redbounds + rshift_cellbounds[r];
                int nc = ncells_resos[r];

                // In nested ordering, all the descendant pixels of a coarse pixel at some nside 
                // form a contiguous range of indices at any finer nside (fourfold split --> appends 2 bits to the index).
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
                int elreso_leaf = mymin(mymax(minresoind_leaf, elreso+resoshift_leafs), maxresoind_leaf);
                long ns_leaf = nside_nav[elreso_leaf];
                long redleaf_off = rshift_red[elreso_leaf];
                const long *cellpix_leaf = cell_pix + rshift_cellpix[elreso_leaf];
                const int *bounds_leaf = cell_redbounds + rshift_cellbounds[elreso_leaf];
                int ncells_leaf = ncells_resos[elreso_leaf];

                double complex *nextGns = calloc(nnvals_Gn*nbinszr_reso, sizeof(double complex));
                double complex *nextWns = calloc(nnvals_Nn*nbinszr_reso, sizeof(double complex));
                double complex *nextG2ns = calloc(4*nbinszr_reso, sizeof(double complex));
                double complex *nextW2ns = calloc(nbinszr_reso, sizeof(double complex));
                int *nextncounts = calloc(nbinszr_reso, sizeof(int));
                int *allowedrinds = calloc(nbinszr_reso, sizeof(int));
                int *allowedzinds = calloc(nbinszr_reso, sizeof(int));

                // Base galaxies of this band in the region = its cell slice.
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

                        // Get neighbours at the leaf reso.
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
                                    double vx2 = vx[g2], vy2 = vy[g2], vz2 = vz[g2];
                                    double sd2 = sindec[g2], cd2 = cosdec[g2];
                                    double w_gal2 = weight[g2];
                                    int z_gal2 = zbin[g2];
                                    double complex wshape_gal2 = ((double complex) w_gal2 * (e1[g2]+I*e2[g2]));
                                    BearingAB g12 = bearing_AB_cart(cx,cy,cz, vx2,vy2,vz2);
                                    BearingAB g21 = bearing_AB_cart(vx2,vy2,vz2, cx,cy,cz);
                                    double complex phirot = bearing_phirot(g12);
                                    double complex phirotc = conj(phirot);
                                    double complex twophirotc = phirotc*phirotc;
                                    wshape_gal2 *= bearing_rc(g21);

                                    int zrshift = z_gal2*nbinsr_reso + rbin;
                                    int ind_rbin = elthread*nbinsz*nbinsr + z_gal2*nbinsr + rbin+rbinmin;

                                    nextncounts[zrshift] += 1;
                                    tmpwcounts[ind_rbin] += w_gal1*w_gal2*dist;
                                    tmpwnorms[ind_rbin] += w_gal1*w_gal2;
                                    ggg_fill_GnWn_projected(nextGns, nextWns, nextG2ns, nextW2ns,
                                        zrshift, nbinszr_reso, nmax, w_gal2, wshape_gal2, phirot, twophirotc);
                                }
                                ci++;
                            }
                        }

                        // Cross-reso cache index: map the base to its coarse reduced galaxy
                        // cell at each reso2 (ang2pix), then its dense per-zbin slot.
                        for (int elreso2=elreso; elreso2<nresos; elreso2++){
                            int grid_reso = elreso2 - hasdiscrete;
                            if (hasdiscrete==1 && elreso==0 && elreso2==0){ grid_reso += hasdiscrete; }
                            int map_reso = grid_reso + hasdiscrete;
                            int redpix = 0;
                            if (map_reso >= nresos){ redpix_by_reso2[elreso2] = 0; continue; }  // degenerate single-band guard
                            if (map_reso == elreso){
                                // Same band: the base is its own reduced galaxy.
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
                        ggg_update_gnwncache(&ctx, elreso, rbinmin, rbinmax, nbinsr_reso, z_gal1, w_gal1, wshape_gal1,
                                             redpix_by_reso2, nextGns, nextWns);
                        ggg_accum_samereso(&ctx, rbinmin, nbinsr_reso, z_gal1, w_gal1, wshape_gal1,
                                           nextGns, nextWns, nextG2ns, nextW2ns,
                                           nextncounts, allowedrinds, allowedzinds);

                        for (int _i=0;_i<nnvals_Gn*nbinszr_reso;_i++){nextGns[_i]=0;}
                        for (int _i=0;_i<nnvals_Nn*nbinszr_reso;_i++){nextWns[_i]=0;}
                        for (int _i=0;_i<4*nbinszr_reso;_i++){nextG2ns[_i]=0;}
                        for (int _i=0;_i<nbinszr_reso;_i++){nextW2ns[_i]=0; nextncounts[_i]=0; allowedrinds[_i]=0; allowedzinds[_i]=0;}
                    }
                }
                free(nextGns); free(nextWns); free(nextG2ns); free(nextW2ns);
                free(nextncounts); free(allowedrinds); free(allowedzinds);
            }

            ggg_accum_crossreso(&ctx);

            for (int r=0;r<nresos;r++){ free(cellzidx[r]); }
            #pragma omp atomic
            nregionsdone += 1;
            print_progress(nregionsdone, nregions, verbose);
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

// Public entry point: Choose function based on passed metric.
void alloc_ggg_doubletree(const MultiresoCatalog *cat, const NavHash *nav,
                          const TreeResoParams *tree, const BinningParams *bin,
                          int nthreads, int verbose, NPCFOutput *out){
    switch (cat->metric) {
        case METRIC_SPHERICAL:
            alloc_ggg_doubletree_spherical(cat, nav, tree, bin, nthreads, verbose, out);
            break;
        case METRIC_FLAT:
        default:
            alloc_ggg_doubletree_flat(cat, nav, tree, bin, nthreads, verbose, out);
            break;
    }
}

// Discrete GGG using the 3dbox geometry and restricting measurement to slabs
// Note that in this setup we have a polar, a (clustered) counts, and a random catalog;
// here we just allocate all the correlators; for the detailed accumulation consult the python layer
void alloc_Gammans_slab_GGG(const MultiresoCatalog *cat_polar, const NavHash *nav_polar,
                            const MultiresoCatalog *cat_R, const NavHash *nav_R,
                            const BinningParams *bin, int nthreads, int verbose,
                            NPCFOutput *out){
    // Dereference input args
    double *pos1_S = cat_polar->pos1_resos, *pos2_S = cat_polar->pos2_resos, *pos3_S = cat_polar->pos3_resos;
    double *w_S = cat_polar->weight_resos, *e1_S = cat_polar->e1_resos, *e2_S = cat_polar->e2_resos;
    int *zbin_S = cat_polar->zbin_resos, nbinsz_polar = cat_polar->nbinsz, ngal_S = cat_polar->ngal_resos[0];
    int *im_S = nav_polar->index_matcher, *pgb_S = nav_polar->pixs_galind_bounds, *pg_S = nav_polar->pix_gals;
    int *so_S = nav_polar->slab_offsets, *rsb_S = nav_polar->rshift_bounds;
    int nslabs_S = nav_polar->nslabs; double z0_S = nav_polar->z0, dpixz_S = nav_polar->dpix_z;
    double p1s_S = nav_polar->pix1_start, p1d_S = nav_polar->pix1_d; int p1n_S = nav_polar->pix1_n;
    double p2s_S = nav_polar->pix2_start, p2d_S = nav_polar->pix2_d; int p2n_S = nav_polar->pix2_n;
    double *pos1_R = cat_R->pos1_resos, *pos2_R = cat_R->pos2_resos, *pos3_R = cat_R->pos3_resos;
    double *w_R = cat_R->weight_resos; int *zbin_R = cat_R->zbin_resos, ngal_R = cat_R->ngal_resos[0];
    int *im_R = nav_R->index_matcher, *pgb_R = nav_R->pixs_galind_bounds, *pg_R = nav_R->pix_gals;
    int *so_R = nav_R->slab_offsets, *rsb_R = nav_R->rshift_bounds;
    int nslabs_R = nav_R->nslabs; double z0_R = nav_R->z0, dpixz_R = nav_R->dpix_z;
    double p1s_R = nav_R->pix1_start, p1d_R = nav_R->pix1_d; int p1n_R = nav_R->pix1_n;
    double p2s_R = nav_R->pix2_start, p2d_R = nav_R->pix2_d; int p2n_R = nav_R->pix2_n;
    int nmax = bin->nmax, nbinsr = bin->nbinsr, dccorr = bin->dccorr;
    double rmin = bin->rmin, rmax = bin->rmax, Pi = bin->Pi;
    double *bin_centers = out->bin_centers;
    double complex *Comp_n = out->npcf, *RRR_n = out->norm_mp;

    int nnvals_Gn = 2*nmax+3;
    int nnvals_Wn = nmax+1;
    int ncomp = 4; 
    int nbinszr_leaf = nbinsz_polar*nbinsr;
    int nzcombis = nbinsz_polar*nbinsz_polar*nbinsz_polar;
    int comp_zshift = nbinsr*nbinsr;
    int comp_nshift = comp_zshift*nzcombis;
    int comp_size = (nmax+1)*comp_nshift;
    int ups_threadshift = ncomp*comp_size;
    int counts_threadshift = nbinsz_polar*nbinsz_polar*nbinsr;

    double complex *tmpComp = calloc((size_t)nthreads*ups_threadshift, sizeof(double complex));
    double complex *tmpRRR  = calloc((size_t)nthreads*comp_size, sizeof(double complex));
    double *tmpwcounts = calloc((size_t)nthreads*counts_threadshift, sizeof(double));
    double *tmpwnorms  = calloc((size_t)nthreads*counts_threadshift, sizeof(double));

    // (A) polar base -> four raw SSS natural components + bin centers.
    // Progress is tracked per base galaxy across both the polar and the random loop
    int nregionsdone = 0, progtot = ngal_S + ngal_R;
    reset_progress();
    #pragma omp parallel num_threads(nthreads)
    {
        int elthread = omp_get_thread_num();
        double complex *thComp = tmpComp + (size_t)elthread*ups_threadshift;
        double complex *G0 = thComp, *G1 = thComp+comp_size, *G2 = thComp+2*comp_size, *G3 = thComp+3*comp_size;
        double *thwc = tmpwcounts + (size_t)elthread*counts_threadshift;
        double *thwn = tmpwnorms  + (size_t)elthread*counts_threadshift;

        #pragma omp for schedule(dynamic, 256)
        for (int ig=0; ig<ngal_S; ig++){
            #pragma omp atomic
            nregionsdone += 1;
            print_progress(nregionsdone, progtot, verbose);
            double c1 = pos1_S[ig], c2 = pos2_S[ig], c3 = pos3_S[ig], cw = w_S[ig];
            int zbin_c = zbin_S[ig];
            double complex wshape = cw*(e1_S[ig] + I*e2_S[ig]);
            double complex *Gn = calloc(nnvals_Gn*nbinszr_leaf, sizeof(double complex));
            double complex *sumG6 = calloc(nbinszr_leaf, sizeof(double complex));
            double complex *sumG2p = calloc(nbinszr_leaf, sizeof(double complex));
            double complex *sumGabsp = calloc(nbinszr_leaf, sizeof(double complex));
            int *ncounts = calloc(nbinszr_leaf, sizeof(int));
            int *allowedr = calloc(nbinszr_leaf, sizeof(int));
            int *allowedz = calloc(nbinszr_leaf, sizeof(int));
            double *wc_base = thwc + zbin_c*nbinsz_polar*nbinsr;
            double *wn_base = thwn + zbin_c*nbinsz_polar*nbinsr;

            slab_polar_leafmultipoles(c1, c2, c3, cw,
                pos1_S, pos2_S, pos3_S, w_S, zbin_S, e1_S, e2_S,
                nbinsz_polar, nslabs_S, z0_S, dpixz_S, p1s_S, p1d_S, p1n_S,
                p2s_S, p2d_S, p2n_S, so_S, im_S, pgb_S, rsb_S, pg_S,
                rmin, rmax, nbinsr, Pi, -nmax-3, nnvals_Gn, Gn,
                (LeafSelfTerm[]){ {6, false, sumG6}, {2, false, sumG2p}, {2, true, sumGabsp} }, 3, ncounts, wc_base, wn_base);

            int nallowed = 0;
            for (int z=0; z<nbinsz_polar; z++){ for (int r=0; r<nbinsr; r++){
                if (ncounts[z*nbinsr+r] != 0){ allowedr[nallowed]=r; allowedz[nallowed]=z; nallowed++; } } }

            for (int thisn=0; thisn<nmax+1; thisn++){
                int nshift = thisn*comp_nshift;
                int blk_nm3 = (nmax+thisn)*nbinszr_leaf; 
                int blk_nm1 = (nmax+thisn+2)*nbinszr_leaf; 
                int blk_mnm1 = (nmax-thisn+2)*nbinszr_leaf; 
                int blk_mnm3 = (nmax-thisn)*nbinszr_leaf; 
                for (int a1=0; a1<nallowed; a1++){
                    int elb1 = allowedr[a1], zbin2 = allowedz[a1];
                    int zr1 = zbin2*nbinsr + elb1;
                    double complex h0 = -wshape * Gn[blk_nm3 + zr1];
                    double complex h1 = -conj(wshape) * Gn[blk_nm1 + zr1];
                    double complex h2 = -wshape * conj(Gn[blk_mnm1 + zr1]);
                    double complex h3 = -wshape * conj(Gn[blk_nm1 + zr1]);
                    if (dccorr==1){
                        int zcd = zbin_c*nbinsz_polar*nbinsz_polar + zbin2*nbinsz_polar + zbin2;
                        int gd = nshift + zcd*comp_zshift + elb1*nbinsr + elb1;
                        G0[gd] += wshape * sumG6[zr1];
                        G1[gd] += conj(wshape) * sumG2p[zr1];
                        G2[gd] += wshape * sumGabsp[zr1];
                        G3[gd] += wshape * sumGabsp[zr1];
                    }
                    for (int a2=0; a2<nallowed; a2++){
                        int elb2 = allowedr[a2], zbin3 = allowedz[a2];
                        int zr2 = zbin3*nbinsr + elb2;
                        int zc = zbin_c*nbinsz_polar*nbinsz_polar + zbin2*nbinsz_polar + zbin3;
                        int gs  = nshift + zc*comp_zshift + elb1*nbinsr + elb2;
                        int gst = nshift + zc*comp_zshift + elb2*nbinsr + elb1;
                        G0[gs] += h0 * Gn[blk_mnm3 + zr2];
                        G1[gs] += h1 * Gn[blk_mnm1 + zr2];
                        G2[gs] += h2 * Gn[blk_mnm3 + zr2];
                        G3[gst] += h3 * Gn[blk_nm3 + zr2];
                    }
                }
            }
            free(Gn); free(sumG6); free(sumG2p); free(sumGabsp);
            free(ncounts); free(allowedr); free(allowedz);
        }
    }

    // (B) Random base --> RRR counts
    #pragma omp parallel num_threads(nthreads)
    {
        int elthread = omp_get_thread_num();
        double complex *thRRR = tmpRRR + (size_t)elthread*comp_size;

        #pragma omp for schedule(dynamic, 256)
        for (int ig=0; ig<ngal_R; ig++){
            #pragma omp atomic
            nregionsdone += 1;
            print_progress(nregionsdone, progtot, verbose);
            double c1 = pos1_R[ig], c2 = pos2_R[ig], c3 = pos3_R[ig], cw = w_R[ig];
            int zbin_c = zbin_R[ig];
            double complex *Wn = calloc(nnvals_Wn*nbinszr_leaf, sizeof(double complex));
            double complex *sumW2 = calloc(nbinszr_leaf, sizeof(double complex));
            int *ncounts = calloc(nbinszr_leaf, sizeof(int));
            int *allowedr = calloc(nbinszr_leaf, sizeof(int));
            int *allowedz = calloc(nbinszr_leaf, sizeof(int));

            slab_count_leafmultipoles(c1, c2, c3, cw,
                pos1_R, pos2_R, pos3_R, w_R, zbin_R,
                nbinsz_polar, nslabs_R, z0_R, dpixz_R, p1s_R, p1d_R, p1n_R,
                p2s_R, p2d_R, p2n_R, so_R, im_R, pgb_R, rsb_R, pg_R,
                rmin, rmax, nbinsr, Pi, 0, nnvals_Wn, Wn,
                (LeafSelfTerm[]){ {0, false, sumW2} }, 1, ncounts, NULL, NULL);

            int nallowed = 0;
            for (int z=0; z<nbinsz_polar; z++){ for (int r=0; r<nbinsr; r++){
                if (ncounts[z*nbinsr+r] != 0){ allowedr[nallowed]=r; allowedz[nallowed]=z; nallowed++; } } }

            for (int thisn=0; thisn<nmax+1; thisn++){
                int nshift = thisn*comp_nshift;
                int blk = thisn*nbinszr_leaf;
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

    // Reduce the bin-center weighted sums (polar base x polar-data leafs) and finalize.
    double *totcounts = calloc(counts_threadshift, sizeof(double));
    double *totnorms  = calloc(counts_threadshift, sizeof(double));
    for (int t=0; t<nthreads; t++){
        size_t ts = (size_t)t*counts_threadshift;
        for (int i=0; i<counts_threadshift; i++){ totcounts[i]+=tmpwcounts[ts+i]; totnorms[i]+=tmpwnorms[ts+i]; }
    }
    for (int i=0; i<counts_threadshift; i++){
        if (totnorms[i] != 0){ bin_centers[i] = totcounts[i]/totnorms[i]; }
    }

    if (verbose>0){ printf("\n"); }
    free(tmpComp); free(tmpRRR); free(tmpwcounts); free(tmpwnorms);
    free(totcounts); free(totnorms);
}


///////////////////////////////
/// GNN CORRELATOR CLASSES ///
//////////////////////////////

// Discrete estimtor of Source-Lens-Lens (G3L) Correlator
void alloc_Gammans_discrete_GNN(const MultiresoCatalog *cat_source, const NavHash *nav_source,
                                const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                                const BinningParams *bin, int nthreads, int verbose,
                                NPCFOutput *out){
    // Dereference input
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

    int _upsilonzshift = nbinsr*nbinsr;
    int _nzcombis = nbinsz_source*nbinsz_lens*nbinsz_lens;
    int _upsilonnshift = _upsilonzshift*_nzcombis;
    int _upsilonthreadshift = (nmax+1)*_upsilonnshift;

    double *tmpwcounts = calloc(nthreads*nbinsz_source*nbinsz_lens*nbinsr, sizeof(double));
    double *tmpwnorms  = calloc(nthreads*nbinsz_source*nbinsz_lens*nbinsr, sizeof(double));
    // Temporary arrays that are allocated in parallel and later reduced
    // Shape of tmpUpsilon ~ (nthreads, nnvals, nz_source, nz_lens, nz_lens, nbinsr, nbinsr)
    double complex *tmpUpsilon = calloc(nthreads*_upsilonthreadshift, sizeof(double complex));
    double complex *tmpNorm = calloc(nthreads*_upsilonthreadshift, sizeof(double complex));
    int nregionsdone = 0;
    reset_progress();
    #pragma omp parallel for num_threads(nthreads)
    for(int elthread=0;elthread<nthreads;elthread++){
        int nregions_per_thread = nregions/nthreads;
        int nnvals_Gn = nmax+3;
        //int nnvals_Wn = nmax+1;
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

        GnnContext ctx;
        ctx.nbinsz_source=nbinsz_source; ctx.nbinsz_lens=nbinsz_lens; ctx.nbinsr=nbinsr;
        ctx.nmax=nmax; ctx.upsilon_zshift=upsilon_zshift; ctx.upsilon_nshift=upsilon_nshift;
        ctx.upsilon_threadshift=upsilon_threadshift; ctx.dccorr=dccorr; ctx.elthread=elthread;
        ctx.tmpUpsilon=tmpUpsilon; ctx.tmpNorm=tmpNorm;

        for (int elregion=0; elregion<nregions; elregion++){
            int region_debug=99999;
            // Check if this thread is responsible for the region
            int nthread_target = mymin(elregion/nregions_per_thread, nthreads-1);
            if (nthread_target!=elthread){continue;}
            bool printregdbg = (verbose>0) && (elregion==region_debug);
            // printf("Region %d is in thread %d\n",elregion,elthread);
            if (printregdbg){printf("Region %d is in thread %d\n",elregion,elthread);}
            #pragma omp atomic
            nregionsdone += 1;
            print_progress(nregionsdone, nregions, verbose);
            
            int zbin_gal1, zbin_gal2;
            double isinner_gal1, pos1_gal1, pos2_gal1, w_gal1, e1_gal1, e2_gal1;
            double pos1_gal2, pos2_gal2, w_gal2;
            double complex wshape_gal1;
            int ind_gal1, ind_gal2, lower1, upper1, lower2, upper2;
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
                double complex phirot, phirotc;
                double rel1, rel2, dist;
                int ind_counts, z1shift, z2rshift, rbin;
                double complex *thisWns = calloc(nnvals_Gn*nbinszr_Gn, sizeof(double complex)); // Here we do not need Gns!
                double complex *thisG2ns = calloc(nbinszr_Gn, sizeof(double complex));
                double complex *thisW2ns = calloc(nbinszr_Gn, sizeof(double complex));
                int *thisncounts = calloc(nbinszr_Gn, sizeof(int));
                int *allowedrinds = calloc(nbinszr_Gn, sizeof(int));
                int *allowedzinds = calloc(nbinszr_Gn, sizeof(int));
                z1shift = zbin_gal1*nbinsz_lens*nbinsr;
                FLATCELL_FOREACH(
                    index_matcher_lens, 0, pixs_galind_bounds_lens, 0, pos1_gal1, pos2_gal1, rmax,
                    pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, lower2, upper2){
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
                        gnn_fill_wn(thisWns, thisG2ns, thisW2ns, z2rshift, nbinszr_Gn, nmax,
                            w_gal1, w_gal2, wshape_gal1, phirot, phirotc);
                    }
                }

                // Update the Upsilon_n & N_n for this galaxy (nbinsr_reso=nbinsr,
                // rbinmin=0 since the discrete kernel has a single resolution).
                gnn_accum_samereso(&ctx, 0, nbinsr, zbin_gal1, w_gal1, wshape_gal1,
                                   thisWns, thisG2ns, thisW2ns, thisncounts,
                                   allowedrinds, allowedzinds);
                free(thisWns);
                free(thisG2ns);
                free(thisW2ns);
                free(thisncounts);
                free(allowedrinds);
                free(allowedzinds);
            }
        }
    }

    gnn_reduce(nbinsz_source, nbinsz_lens, nbinsr, nmax, nthreads,
               tmpUpsilon, tmpNorm, tmpwcounts, tmpwnorms, out);
    if (verbose>0){ printf("\n"); }
    free(tmpUpsilon);
    free(tmpNorm);
    free(tmpwcounts);
    free(tmpwnorms);
}


// DoubleTree based estimtor of Source-Lens-Lens (G3L) Correlator
void alloc_Gammans_doubletree_GNN(const MultiresoCatalog *cat_source, const NavHash *nav_source,
                                  const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                                  const TreeResoParams *tree, const BinningParams *bin,
                                  int nthreads, int verbose, NPCFOutput *out){
    // Dereference input args
    int nresos = tree->nresos, nresos_grid = tree->nresos_grid;
    double *dpix1_resos = tree->dpix1_resos, *dpix2_resos = tree->dpix2_resos, *reso_redges = tree->reso_redges;
    int resoshift_leafs = tree->resoshift_leafs, minresoind_leaf = tree->minresoind_leaf, maxresoind_leaf = tree->maxresoind_leaf;
    double *isinner_source_resos = cat_source->isinner_resos, *w_source_resos = cat_source->weight_resos;
    double *pos1_source_resos = cat_source->pos1_resos, *pos2_source_resos = cat_source->pos2_resos;
    double *e1_source_resos = cat_source->e1_resos, *e2_source_resos = cat_source->e2_resos;
    int *zbin_source_resos = cat_source->zbin_resos, *ngal_source_resos = cat_source->ngal_resos, nbinsz_source = cat_source->nbinsz;
    double *isinner_lens_resos = cat_lens->isinner_resos, *w_lens_resos = cat_lens->weight_resos;
    double *pos1_lens_resos = cat_lens->pos1_resos, *pos2_lens_resos = cat_lens->pos2_resos;
    int *zbin_lens_resos = cat_lens->zbin_resos, *ngal_lens_resos = cat_lens->ngal_resos, nbinsz_lens = cat_lens->nbinsz;
    int nmax = bin->nmax, nbinsr = bin->nbinsr, dccorr = bin->dccorr;
    double rmin = bin->rmin, rmax = bin->rmax;
    int *index_matcher_source = nav_source->index_matcher, *pixs_galind_bounds_source = nav_source->pixs_galind_bounds, *pix_gals_source = nav_source->pix_gals;
    int *index_matcher_lens = nav_lens->index_matcher, *pixs_galind_bounds_lens = nav_lens->pixs_galind_bounds, *pix_gals_lens = nav_lens->pix_gals;
    int *index_matcher_hash = nav_source->index_matcher_hash, nregions = nav_source->nregions;
    double pix1_start = nav_lens->pix1_start, pix1_d = nav_lens->pix1_d; int pix1_n = nav_lens->pix1_n;
    double pix2_start = nav_lens->pix2_start, pix2_d = nav_lens->pix2_d; int pix2_n = nav_lens->pix2_n;

    int _upsilonzshift = nbinsr*nbinsr;
    int _nzcombis = nbinsz_source*nbinsz_lens*nbinsz_lens;
    int _upsilonnshift = _upsilonzshift*_nzcombis;
    int _upsilonthreadshift = (nmax+1)*_upsilonnshift;
    
    double *tmpwcounts = calloc(nthreads*nbinsz_source*nbinsz_lens*nbinsr, sizeof(double));
    double *tmpwnorms  = calloc(nthreads*nbinsz_source*nbinsz_lens*nbinsr, sizeof(double));
    // Temporary arrays that are allocated in parallel and later reduced
    // Shape of tmpUpsilon ~ (nthreads, nnvals, nz_source, nz_lens, nz_lens, nbinsr, nbinsr)
    double complex *tmpUpsilon = calloc(nthreads*_upsilonthreadshift, sizeof(double complex));
    double complex *tmpNorm = calloc(nthreads*_upsilonthreadshift, sizeof(double complex));
    int nregionsdone = 0;
    reset_progress();
    #pragma omp parallel for num_threads(nthreads)
    for(int elthread=0;elthread<nthreads;elthread++){
        int nregions_per_thread = nregions/nthreads;
        int hasdiscrete = nresos-nresos_grid;
        int nnvals_Gn = nmax+3; 
        int nnvals_Wn = nmax+1; 
        int nnvals_Ups = nmax+1;
        int nzcombis = nbinsz_source*nbinsz_lens*nbinsz_lens;
        int upsilon_zshift = nbinsr*nbinsr;
        int upsilon_nshift = upsilon_zshift*nzcombis;
        int upsilon_threadshift = nnvals_Ups*upsilon_nshift;
        int threadshift_counts = elthread*nbinsz_source*nbinsz_lens*nbinsr;
        double drbin = log(rmax/rmin)/nbinsr;
        
        // Gn/Wn caches grown on demand to the region's nshift.
        long cache_cap = 0;
        double complex *Gncache=NULL, *wGncache=NULL, *cwGncache=NULL, *Wncache=NULL, *wWncache=NULL;

        GnnContext ctx;
        ctx.nbinsz_source=nbinsz_source; ctx.nbinsz_lens=nbinsz_lens; ctx.nbinsr=nbinsr;
        ctx.nmax=nmax; ctx.nresos=nresos;
        ctx.nnvals_Gn=nnvals_Gn; ctx.nnvals_Wn=nnvals_Wn;
        ctx.upsilon_zshift=upsilon_zshift; ctx.upsilon_nshift=upsilon_nshift;
        ctx.upsilon_threadshift=upsilon_threadshift;
        ctx.dccorr=dccorr; ctx.elthread=elthread;
        ctx.tmpUpsilon=tmpUpsilon; ctx.tmpNorm=tmpNorm;

        for (int elregion=0; elregion<nregions; elregion++){
            // Check if this thread is responsible for the region
            int nthread_target = mymin(elregion/nregions_per_thread, nthreads-1);
            if (nthread_target!=elthread){continue;}
            #pragma omp atomic
            nregionsdone += 1;
            print_progress(nregionsdone, nregions, verbose);
            
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

            // Region layout of the source (base) catalog: per-(zbin, reso)
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
            ctx.reso_rindedges = reso_rindedges; ctx.ngal_in_pix = ngal_in_pix;
            ctx.cumresoshift_z = cumresoshift_z; ctx.thetashifts_z = thetashifts_z; ctx.zbinshifts = zbinshifts;
            setup_region_shifts(nbinsz_source, nbinsz_lens, nresos, hasdiscrete, nbinsr,
                ngal_in_pix, cumresoshift_z, thetashifts_z, zbinshifts, &ctx.zbin2shift, &ctx.nshift);
            long need = (long)nnvals_Gn * ctx.nshift;
            if (need > cache_cap){
                cache_cap = need;
                Gncache = realloc(Gncache, cache_cap*sizeof(double complex));
                wGncache = realloc(wGncache, cache_cap*sizeof(double complex));
                cwGncache = realloc(cwGncache, cache_cap*sizeof(double complex));
                Wncache = realloc(Wncache, cache_cap*sizeof(double complex));
                wWncache = realloc(wWncache, cache_cap*sizeof(double complex));
            }
            ctx.Gncache=Gncache; ctx.wGncache=wGncache; ctx.cwGncache=cwGncache;
            ctx.Wncache=Wncache; ctx.wWncache=wWncache;
            gnn_zero_caches(&ctx);
            int *redpix_by_reso2 = calloc(nresos, sizeof(int));


            // Now, for each resolution, loop over all the galaxies in the region and
            // allocate the Gn & Nn, as well as their caches for the corresponding 
            // set of radii
            // For elreso in resos
            //.  for gal in reso 
            //.    allocate Gn for allowed radii
            //.    allocate the Gncaches
            //.    compute the Upsilon for all combinations of the same resolution
            int ind_inpix1, ind_inpix2, ind_gal1, ind_gal2, z_gal1, z_gal2;
            int nbinszr_reso;
            double innergal, pos1_gal1, pos2_gal1, pos1_gal2, pos2_gal2, w_gal1, w_gal2, e1_gal1, e2_gal1;
            double rel1, rel2, dist;
            double complex wshape_gal1;
            double complex phirot, phirotc;
            double rmin_reso, rmax_reso, rmin_reso_sq, rmax_reso_sq;
            int elreso_leaf, rbinmin, rbinmax;
            
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
                int ind_counts, z1shift, z2rshift, rbin;
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
                    
                    FLATCELL_FOREACH(
                        index_matcher_lens, rshift_index_matcher_lens[elreso_leaf], pixs_galind_bounds_lens, rshift_pixs_galind_bounds_lens[elreso_leaf],
                        pos1_gal1, pos2_gal1, rmax_reso, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, lower2, upper2){
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
                            gnn_fill_wn(thisWns, thisG2ns, thisW2ns, z2rshift, nbinszr_reso, nmax,
                                w_gal1, w_gal2, wshape_gal1, phirot, phirotc);
                        }
                    }
                    // Update the region caches and the same-reso Upsilon_n / N_n
                    build_redpix_by_reso2(elreso, nresos, nresos_grid, hasdiscrete,
                        z_gal1, pos1_gal1, pos2_gal1, hashpix_start1, hashpix_start2,
                        dpix1_resos, dpix2_resos, matchers_resoshift, len_matcher,
                        pix2redpix, redpix_by_reso2);
                    gnn_update_wncache(&ctx, elreso, rbinmin, rbinmax, nbinsr_reso, z_gal1, w_gal1, wshape_gal1,
                                       redpix_by_reso2, thisWns);
                    gnn_accum_samereso(&ctx, rbinmin, nbinsr_reso, z_gal1, w_gal1, wshape_gal1,
                                       thisWns, thisG2ns, thisW2ns, nextncounts,
                                       allowedrinds, allowedzinds);

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
             
            
            gnn_accum_crossreso(&ctx);

            free(redpix_by_reso2);
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

    gnn_reduce(nbinsz_source, nbinsz_lens, nbinsr, nmax, nthreads,
               tmpUpsilon, tmpNorm, tmpwcounts, tmpwnorms, out);
    if (verbose>0){ printf("\n"); }
    free(tmpUpsilon);
    free(tmpNorm);
    free(tmpwcounts);
    free(tmpwnorms);
}


// Discrete GNN using the 3dbox geometry and restricting measurement to slabs
// Note that in this setup we have a polar, a (clustered) counts, and a random catalog;
// here we just allocate all the correlators; for the detailed accumulation consult the python layer
void alloc_Gammans_slab_GNN(const MultiresoCatalog *cat_polar, const MultiresoCatalog *cat_D,
                            const NavHash *nav_D, const MultiresoCatalog *cat_R,
                            const NavHash *nav_R, const BinningParams *bin,
                            int nthreads, int verbose, NPCFOutput *out){
    // Dereference input
    double *pos1_shape = cat_polar->pos1_resos, *pos2_shape = cat_polar->pos2_resos, *pos3_shape = cat_polar->pos3_resos;
    double *w_shape = cat_polar->weight_resos, *e1_shape = cat_polar->e1_resos, *e2_shape = cat_polar->e2_resos;
    int *zbin_shape = cat_polar->zbin_resos, nbinsz_shape = cat_polar->nbinsz, ngal_shape = cat_polar->ngal_resos[0];
    double *pos1_D = cat_D->pos1_resos, *pos2_D = cat_D->pos2_resos, *pos3_D = cat_D->pos3_resos, *w_D = cat_D->weight_resos;
    int *zbin_D = cat_D->zbin_resos, nbinsz_pos = cat_D->nbinsz;
    int *slab_offsets_D = nav_D->slab_offsets, *index_matcher_D = nav_D->index_matcher, *pixs_galind_bounds_D = nav_D->pixs_galind_bounds;
    int *rshift_bounds_D = nav_D->rshift_bounds, *pix_gals_D = nav_D->pix_gals;
    double *pos1_R = cat_R->pos1_resos, *pos2_R = cat_R->pos2_resos, *pos3_R = cat_R->pos3_resos, *w_R = cat_R->weight_resos;
    int *zbin_R = cat_R->zbin_resos, ngal_R = cat_R->ngal_resos[0];
    int *slab_offsets_R = nav_R->slab_offsets, *index_matcher_R = nav_R->index_matcher, *pixs_galind_bounds_R = nav_R->pixs_galind_bounds;
    int *rshift_bounds_R = nav_R->rshift_bounds, *pix_gals_R = nav_R->pix_gals;
    int nslabs = nav_D->nslabs; double z0 = nav_D->z0, dpix_z = nav_D->dpix_z;
    double pix1_start = nav_D->pix1_start, pix1_d = nav_D->pix1_d; int pix1_n = nav_D->pix1_n;
    double pix2_start = nav_D->pix2_start, pix2_d = nav_D->pix2_d; int pix2_n = nav_D->pix2_n;
    int nmax = bin->nmax, nbinsr = bin->nbinsr, dccorr = bin->dccorr;
    double rmin = bin->rmin, rmax = bin->rmax, Pi = bin->Pi;
    double *bin_centers = out->bin_centers;
    double complex *Comp_n = out->npcf, *RRR_n = out->norm_mp;

    int nnvals = nmax+3;  
    int ncomp = 4; // SDD, SDR, SRD, SRR
    int nbinszr_leaf = nbinsz_pos*nbinsr;
    int nzcombis = nbinsz_shape*nbinsz_pos*nbinsz_pos;
    int comp_zshift = nbinsr*nbinsr;
    int comp_nshift = comp_zshift*nzcombis;
    int comp_size = (nmax+1)*comp_nshift; 
    int ups_threadshift = ncomp*comp_size;
    int counts_threadshift = nbinsz_shape*nbinsz_pos*nbinsr;

    double complex *tmpComp = calloc((size_t)nthreads*ups_threadshift, sizeof(double complex));
    double complex *tmpRRR  = calloc((size_t)nthreads*comp_size, sizeof(double complex));
    double *tmpwcounts = calloc((size_t)nthreads*counts_threadshift, sizeof(double));
    double *tmpwnorms  = calloc((size_t)nthreads*counts_threadshift, sizeof(double));

    // (A) polar base -> four raw numerator components S.(D/R).(D/R) + bin centers.
    // Progress is tracked per base galaxy across both the polar and the random loop
    int nregionsdone = 0, progtot = ngal_shape + ngal_R;
    reset_progress();
    #pragma omp parallel num_threads(nthreads)
    {
        int elthread = omp_get_thread_num();
        double complex *thComp = tmpComp + (size_t)elthread*ups_threadshift;
        double complex *SDD = thComp, *SDR = thComp+comp_size, *SRD = thComp+2*comp_size, *SRR = thComp+3*comp_size;
        double *thwcounts = tmpwcounts + (size_t)elthread*counts_threadshift;
        double *thwnorms  = tmpwnorms  + (size_t)elthread*counts_threadshift;

        #pragma omp for schedule(dynamic, 256)
        for (int ig=0; ig<ngal_shape; ig++){
            #pragma omp atomic
            nregionsdone += 1;
            print_progress(nregionsdone, progtot, verbose);
            double c_pos1 = pos1_shape[ig];
            double c_pos2 = pos2_shape[ig];
            double c_pos3 = pos3_shape[ig];
            double c_w = w_shape[ig];
            int zbin_c = zbin_shape[ig];
            double complex wshape = c_w*(e1_shape[ig] + I*e2_shape[ig]);

            double complex *Wn_D = calloc(nnvals*nbinszr_leaf, sizeof(double complex));
            double complex *Wn_R = calloc(nnvals*nbinszr_leaf, sizeof(double complex));
            double complex *sumG2_D = calloc(nbinszr_leaf, sizeof(double complex));
            double complex *sumG2_R = calloc(nbinszr_leaf, sizeof(double complex));
            int *ncounts = calloc(nbinszr_leaf, sizeof(int));
            int *allowedr = calloc(nbinszr_leaf, sizeof(int));
            int *allowedz = calloc(nbinszr_leaf, sizeof(int));

            double *wc_base = thwcounts + zbin_c*nbinsz_pos*nbinsr;
            double *wn_base = thwnorms  + zbin_c*nbinsz_pos*nbinsr;

            // D leafs + bin_centers allocation
            slab_count_leafmultipoles(c_pos1, c_pos2, c_pos3, c_w,
                pos1_D, pos2_D, pos3_D, w_D, zbin_D,
                nbinsz_pos, nslabs, z0, dpix_z, pix1_start, pix1_d, pix1_n,
                pix2_start, pix2_d, pix2_n, slab_offsets_D, index_matcher_D,
                pixs_galind_bounds_D, rshift_bounds_D, pix_gals_D,
                rmin, rmax, nbinsr, Pi, -1, nnvals, Wn_D,
                (LeafSelfTerm[]){ {2, false, sumG2_D} }, 1, ncounts, wc_base, wn_base);
            // R leafs
            slab_count_leafmultipoles(c_pos1, c_pos2, c_pos3, c_w,
                pos1_R, pos2_R, pos3_R, w_R, zbin_R,
                nbinsz_pos, nslabs, z0, dpix_z, pix1_start, pix1_d, pix1_n,
                pix2_start, pix2_d, pix2_n, slab_offsets_R, index_matcher_R,
                pixs_galind_bounds_R, rshift_bounds_R, pix_gals_R,
                rmin, rmax, nbinsr, Pi, -1, nnvals, Wn_R,
                (LeafSelfTerm[]){ {2, false, sumG2_R} }, 1, ncounts, NULL, NULL);

            // Nonzero (zbin_leaf, r) bins.
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
                    double complex D1 = Wn_D[thisn*nbinszr_leaf + zr1];
                    double complex R1 = Wn_R[thisn*nbinszr_leaf + zr1];
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
                        double complex cD2 = conj(Wn_D[(thisn+2)*nbinszr_leaf + zr2]);
                        double complex cR2 = conj(Wn_R[(thisn+2)*nbinszr_leaf + zr2]);
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

    // (B) RRR counts
    #pragma omp parallel num_threads(nthreads)
    {
        int elthread = omp_get_thread_num();
        double complex *thRRR = tmpRRR + (size_t)elthread*comp_size;

        #pragma omp for schedule(dynamic, 256)
        for (int ig=0; ig<ngal_R; ig++){
            #pragma omp atomic
            nregionsdone += 1;
            print_progress(nregionsdone, progtot, verbose);
            double c_pos1 = pos1_R[ig], c_pos2 = pos2_R[ig], c_pos3 = pos3_R[ig], c_w = w_R[ig];
            int zbin_c = zbin_R[ig];
            double complex *Wn = calloc(nnvals*nbinszr_leaf, sizeof(double complex));
            double complex *sumW2 = calloc(nbinszr_leaf, sizeof(double complex));
            int *ncounts = calloc(nbinszr_leaf, sizeof(int));
            int *allowedr = calloc(nbinszr_leaf, sizeof(int));
            int *allowedz = calloc(nbinszr_leaf, sizeof(int));

            slab_count_leafmultipoles(c_pos1, c_pos2, c_pos3, c_w,
                pos1_R, pos2_R, pos3_R, w_R, zbin_R,
                nbinsz_pos, nslabs, z0, dpix_z, pix1_start, pix1_d, pix1_n,
                pix2_start, pix2_d, pix2_n, slab_offsets_R, index_matcher_R,
                pixs_galind_bounds_R, rshift_bounds_R, pix_gals_R,
                rmin, rmax, nbinsr, Pi, -1, nnvals, Wn,
                (LeafSelfTerm[]){ {0, false, sumW2} }, 1, ncounts, NULL, NULL);

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
                    double complex Rn1 = Wn[(thisn+1)*nbinszr_leaf + zr1];
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
                        thRRR[gs] += c_w * Rn1 * conj(Wn[(thisn+1)*nbinszr_leaf + zr2]);
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

    if (verbose>0){ printf("\n"); }
    free(tmpComp); free(tmpRRR); free(tmpwcounts); free(tmpwnorms);
    free(totcounts); free(totnorms);
}


//////////////////////////////
/// NGG CORRELATOR CLASSES ///
//////////////////////////////

// Discrete estimator of Lens-Source-Source Correlator
void alloc_Gammans_discrete_NGG(const MultiresoCatalog *cat_source, const NavHash *nav_source,
                                const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                                const BinningParams *bin, int nthreads, int verbose,
                                NPCFOutput *out){
    // Dereference input args
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
    // Temporary arrays that are allocated in parallel and later reduced
    // Shape of tmpUpsilon ~ (nthreads, nnvals, nz_source, nz_lens, nz_lens, nbinsr, nbinsr)
    double complex *tmpUpsilon = calloc(nthreads*_upsilonthreadshift, sizeof(double complex));
    double complex *tmpNorm = calloc(nthreads*_normthreadshift, sizeof(double complex));
    int nregionsdone = 0;
    reset_progress();
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
        int norm_zshift = nbinsr*nbinsr;
        int norm_nshift = norm_zshift*nzcombis;
        int threadshift_counts = elthread*nbinsz_lens*nbinsz_source*nbinsr;
        int nbinszr_Gn = nbinsz_source*nbinsr;
        int nbinszr_Wn = nbinsz_source*nbinsr;
        double rmin_sq = rmin*rmin;
        double rmax_sq = rmax*rmax;
        double drbin = log(rmax/rmin)/nbinsr;

        NggContext ctx;
        ctx.nbinsz_lens=nbinsz_lens; ctx.nbinsz_source=nbinsz_source; ctx.nbinsr=nbinsr;
        ctx.nmax=nmax; ctx.upsilon_zshift=upsilon_zshift; ctx.upsilon_nshift=upsilon_nshift;
        ctx.upsilon_compshift=upsilon_compshift;
        ctx.upsilon_threadshift=2*upsilon_compshift; ctx.norm_threadshift=nnvals_Norm*norm_nshift;
        ctx.dccorr=dccorr; ctx.elthread=elthread;
        ctx.tmpUpsilon=tmpUpsilon; ctx.tmpNorm=tmpNorm;

        for (int elregion=0; elregion<nregions; elregion++){
            int region_debug=(int) (nthreads/2) * nregions_per_thread;
            //int region_debug=99999;
            // Check if this thread is responsible for the region
            int nthread_target = mymin(elregion/nregions_per_thread, nthreads-1);
            if (nthread_target!=elthread){continue;}
            bool printregdbg = (verbose>0) && (elregion==region_debug);
            bool printregdbg2 = (verbose>1) && (elregion==region_debug); 
            //if (elregion==region_debug){printf("Region %d is in thread %d\n",elregion,elthread);}
            #pragma omp atomic
            nregionsdone += 1;
            print_progress(nregionsdone, nregions, verbose);
            
            int zbin_gal1, zbin_gal2;
            double isinner_gal1, pos1_gal1, pos2_gal1, w_gal1;
            double pos1_gal2, pos2_gal2, w_gal2, e1_gal2, e2_gal2;
            double complex wshape_gal2;
            int ind_gal1, ind_gal2, lower1, upper1, lower2, upper2;
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
                double complex phirot;
                double rel1, rel2, dist;
                int ind_counts, z1shift, z2rshift, rbin;
                double complex *thisGns = calloc(nnvals_Gn*nbinszr_Gn, sizeof(double complex)); 
                double complex *thisWns = calloc(nnvals_Wn*nbinszr_Wn, sizeof(double complex)); 
                double complex *thisG2ns = calloc(2*nbinszr_Gn, sizeof(double complex));
                double complex *thisW2ns = calloc(nbinszr_Wn, sizeof(double complex));
                int *thisncounts = calloc(nbinszr_Wn, sizeof(int));
                int *allowedrinds = calloc(nbinszr_Wn, sizeof(int));
                int *allowedzinds = calloc(nbinszr_Wn, sizeof(int));
                z1shift = zbin_gal1*nbinsz_source*nbinsr;
                FLATCELL_FOREACH(
                    index_matcher_source, 0, pixs_galind_bounds_source, 0, pos1_gal1, pos2_gal1, rmax,
                    pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, lower2, upper2){
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
                        ngg_fill_gnwn(thisGns, thisWns, thisG2ns, thisW2ns, z2rshift,
                            nbinszr_Gn, nbinszr_Wn, nmax, w_gal1, w_gal2, wshape_gal2, phirot);
                    }
                }

                // Update the Upsilon-/Upsilon+/N_n for this galaxy 
                ngg_accum_samereso(&ctx, 0, nbinsr, zbin_gal1, w_gal1,
                                   thisGns, thisWns, thisG2ns, thisW2ns, thisncounts,
                                   allowedrinds, allowedzinds);
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

    ngg_reduce(nbinsz_lens, nbinsz_source, nbinsr, nmax, nthreads,
               _upsiloncompshift, _upsilonthreadshift, _normthreadshift,
               tmpUpsilon, tmpNorm, tmpwcounts, tmpwnorms, out);
    if (verbose>0){ printf("\n"); }
    free(tmpUpsilon);
    free(tmpNorm);
    free(tmpwcounts);
    free(tmpwnorms);
}

// Discrete estimator of Lens-Source-Source Correlator
void alloc_Gammans_tree_NGG(const MultiresoCatalog *cat_source, const NavHash *nav_source,
                            const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                            const TreeResoParams *tree, const BinningParams *bin,
                            int nthreads, int verbose, NPCFOutput *out){
    // Dereference input args
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
    // Temporary arrays that are allocated in parallel and later reduced
    // Shape of tmpUpsilon ~ (nthreads, nnvals, nz_source, nz_lens, nz_lens, nbinsr, nbinsr)
    double complex *tmpUpsilon = calloc(nthreads*_upsilonthreadshift, sizeof(double complex));
    double complex *tmpNorm = calloc(nthreads*_normthreadshift, sizeof(double complex));
    int nregionsdone = 0;
    reset_progress();
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
        int norm_zshift = nbinsr*nbinsr;
        int norm_nshift = norm_zshift*nzcombis;
        int threadshift_counts = elthread*nbinsz_lens*nbinsz_source*nbinsr;
        int nbinszr_Gn = nbinsz_source*nbinsr;
        int nbinszr_Wn = nbinsz_source*nbinsr;
        double drbin = log(rmax/rmin)/nbinsr;
        int npix_hash = pix1_n*pix2_n;

        NggContext ctx;
        ctx.nbinsz_lens=nbinsz_lens; ctx.nbinsz_source=nbinsz_source; ctx.nbinsr=nbinsr;
        ctx.nmax=nmax; ctx.upsilon_zshift=upsilon_zshift; ctx.upsilon_nshift=upsilon_nshift;
        ctx.upsilon_compshift=upsilon_compshift;
        ctx.upsilon_threadshift=2*upsilon_compshift; ctx.norm_threadshift=nnvals_Norm*norm_nshift;
        ctx.dccorr=dccorr; ctx.elthread=elthread;
        ctx.tmpUpsilon=tmpUpsilon; ctx.tmpNorm=tmpNorm;

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
            #pragma omp atomic
            nregionsdone += 1;
            print_progress(nregionsdone, nregions, verbose);
            
            int zbin_gal1, zbin_gal2;
            double isinner_gal1, pos1_gal1, pos2_gal1, w_gal1;
            double pos1_gal2, pos2_gal2, w_gal2, e1_gal2, e2_gal2;
            double complex wshape_gal2;
            int ind_gal1, ind_gal2, lower1, upper1, lower2, upper2;
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
                double complex phirot;
                double rel1, rel2, dist;
                int ind_counts, z1shift, z2rshift, rbin;
                double complex *thisGns = calloc(nnvals_Gn*nbinszr_Gn, sizeof(double complex)); 
                double complex *thisWns = calloc(nnvals_Wn*nbinszr_Wn, sizeof(double complex)); 
                double complex *thisG2ns = calloc(2*nbinszr_Gn, sizeof(double complex));
                double complex *thisW2ns = calloc(nbinszr_Wn, sizeof(double complex));
                int *thisncounts = calloc(nbinszr_Wn, sizeof(int));
                int *allowedrinds = calloc(nbinszr_Wn, sizeof(int));
                int *allowedzinds = calloc(nbinszr_Wn, sizeof(int));
                z1shift = zbin_gal1*nbinsz_source*nbinsr;                
                
                for (int elreso=0;elreso<nresos;elreso++){
                    double rmin_sq = reso_redges[elreso]*reso_redges[elreso];
                    double rmax_sq = reso_redges[elreso+1]*reso_redges[elreso+1];
                    FLATCELL_FOREACH(
                        index_matcher_source, rshift_index_matcher[elreso], pixs_galind_bounds_source, rshift_pixs_galind_bounds[elreso],
                        pos1_gal1, pos2_gal1, reso_redges[elreso+1], pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, lower2, upper2){
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
                            ngg_fill_gnwn(thisGns, thisWns, thisG2ns, thisW2ns, z2rshift,
                                nbinszr_Gn, nbinszr_Wn, nmax, w_gal1, w_gal2, wshape_gal2, phirot);
                        }
                    }
                }

                // Update the Upsilon-/Upsilon+/N_n for this galaxy
                ngg_accum_samereso(&ctx, 0, nbinsr, zbin_gal1, w_gal1,
                                   thisGns, thisWns, thisG2ns, thisW2ns, thisncounts,
                                   allowedrinds, allowedzinds);
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

    ngg_reduce(nbinsz_lens, nbinsz_source, nbinsr, nmax, nthreads,
               _upsiloncompshift, _upsilonthreadshift, _normthreadshift,
               tmpUpsilon, tmpNorm, tmpwcounts, tmpwnorms, out);
    if (verbose>0){ printf("\n"); }
    free(tmpUpsilon);
    free(tmpNorm);
    free(tmpwcounts);
    free(tmpwnorms);
}

// DoubleTree based estimtor of Lens-Source-Source Correlator
void alloc_Gammans_doubletree_NGG(const MultiresoCatalog *cat_source, const NavHash *nav_source,
                                  const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                                  const TreeResoParams *tree, const BinningParams *bin,
                                  int nthreads, int verbose, NPCFOutput *out){
    // Dereference input args
    int nresos = tree->nresos, nresos_grid = tree->nresos_grid;
    double *dpix1_resos = tree->dpix1_resos, *dpix2_resos = tree->dpix2_resos, *reso_redges = tree->reso_redges;
    int resoshift_leafs = tree->resoshift_leafs, minresoind_leaf = tree->minresoind_leaf, maxresoind_leaf = tree->maxresoind_leaf;
    double *isinner_source_resos = cat_source->isinner_resos, *w_source_resos = cat_source->weight_resos;
    double *pos1_source_resos = cat_source->pos1_resos, *pos2_source_resos = cat_source->pos2_resos;
    double *e1_source_resos = cat_source->e1_resos, *e2_source_resos = cat_source->e2_resos;
    int *zbin_source_resos = cat_source->zbin_resos, *ngal_source_resos = cat_source->ngal_resos, nbinsz_source = cat_source->nbinsz;
    double *isinner_lens_resos = cat_lens->isinner_resos, *w_lens_resos = cat_lens->weight_resos;
    double *pos1_lens_resos = cat_lens->pos1_resos, *pos2_lens_resos = cat_lens->pos2_resos;
    int *zbin_lens_resos = cat_lens->zbin_resos, *ngal_lens_resos = cat_lens->ngal_resos, nbinsz_lens = cat_lens->nbinsz;
    int nmax = bin->nmax, nbinsr = bin->nbinsr, dccorr = bin->dccorr;
    double rmin = bin->rmin, rmax = bin->rmax;
    int *index_matcher_source = nav_source->index_matcher, *pixs_galind_bounds_source = nav_source->pixs_galind_bounds, *pix_gals_source = nav_source->pix_gals;
    int *index_matcher_lens = nav_lens->index_matcher, *pixs_galind_bounds_lens = nav_lens->pixs_galind_bounds, *pix_gals_lens = nav_lens->pix_gals;
    int *index_matcher_hash = nav_lens->index_matcher_hash, nregions = nav_lens->nregions;
    double pix1_start = nav_lens->pix1_start, pix1_d = nav_lens->pix1_d; int pix1_n = nav_lens->pix1_n;
    double pix2_start = nav_lens->pix2_start, pix2_d = nav_lens->pix2_d; int pix2_n = nav_lens->pix2_n;

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
    reset_progress();
    
    double *tmpwcounts = calloc(nthreads*nbinsz_lens*nbinsz_source*nbinsr, sizeof(double));
    double *tmpwnorms  = calloc(nthreads*nbinsz_lens*nbinsz_source*nbinsr, sizeof(double));
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
        int norm_zshift = nbinsr*nbinsr;
        int norm_nshift = norm_zshift*nzcombis;
        int counts_threadshift = elthread*nbinsz_lens*nbinsz_source*nbinsr;
        double drbin = log(rmax/rmin)/nbinsr;
        
        // Gn/Wn caches grown on demand to the region's nshift.
        long cache_cap = 0;
        double complex *Gncache=NULL, *wGncache=NULL, *Wncache=NULL, *wWncache=NULL;

        NggContext ctx;
        ctx.nbinsz_lens=nbinsz_lens; ctx.nbinsz_source=nbinsz_source; ctx.nbinsr=nbinsr;
        ctx.nmax=nmax; ctx.nresos=nresos;
        ctx.nnvals_Gn=nnvals_Gn; ctx.nnvals_Wn=nnvals_Wn;
        ctx.upsilon_zshift=upsilon_zshift; ctx.upsilon_nshift=upsilon_nshift;
        ctx.upsilon_compshift=upsilon_compshift;
        ctx.upsilon_threadshift=ncomp_Upsilon*upsilon_compshift; ctx.norm_threadshift=nnvals_Norm*norm_nshift;
        ctx.dccorr=dccorr; ctx.elthread=elthread;
        ctx.tmpUpsilon=tmpUpsilon; ctx.tmpNorm=tmpNorm;

        for (int _elregion=0; _elregion<2*nregions; _elregion++){
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

            // Region layout of the lens (base) catalog: per-(zbin, reso)
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
            ctx.reso_rindedges = reso_rindedges; ctx.ngal_in_pix = ngal_in_pix;
            ctx.cumresoshift_z = cumresoshift_z; ctx.thetashifts_z = thetashifts_z; ctx.zbinshifts = zbinshifts;
            setup_region_shifts(nbinsz_lens, nbinsz_source, nresos, hasdiscrete, nbinsr,
                ngal_in_pix, cumresoshift_z, thetashifts_z, zbinshifts, &ctx.zbin2shift, &ctx.nshift);
            long need = (long)nnvals_Gn * ctx.nshift;
            if (need > cache_cap){
                cache_cap = need;
                Gncache = realloc(Gncache, cache_cap*sizeof(double complex));
                wGncache = realloc(wGncache, cache_cap*sizeof(double complex));
                Wncache = realloc(Wncache, cache_cap*sizeof(double complex));
                wWncache = realloc(wWncache, cache_cap*sizeof(double complex));
            }
            ctx.Gncache=Gncache; ctx.wGncache=wGncache;
            ctx.Wncache=Wncache; ctx.wWncache=wWncache;
            ngg_zero_caches(&ctx);
            int *redpix_by_reso2 = calloc(nresos, sizeof(int));


            // Now, for each resolution, loop over all the galaxies in the region and
            // allocate the Gn & Nn, as well as their caches for the corresponding 
            // set of radii
            // For elreso in resos
            //.  for gal in reso 
            //.    allocate Gn for allowed radii
            //.    allocate the Gncaches
            //.    compute the Upsilon for all combinations of the same resolution
            int ind_inpix1, ind_inpix2, ind_gal1, ind_gal2, z_gal1, z_gal2;
            int nbinszr_reso;
            double innergal, pos1_gal1, pos2_gal1, pos1_gal2, pos2_gal2, w_gal1, w_gal2, e1_gal2, e2_gal2;
            double rel1, rel2, dist;
            double complex wshape_gal2;
            double complex phirot;
            double rmin_reso, rmax_reso, rmin_reso_sq, rmax_reso_sq;
            int elreso_leaf, rbinmin, rbinmax;

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
                int ind_counts, z1shift, z2rshift, rbin;
                for (ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                    ind_gal1 = rshift_pix_gals_lens[elreso] + pix_gals_lens[rshift_pix_gals_lens[elreso]+ind_inpix1];
                    innergal = isinner_lens_resos[ind_gal1];
                    if (innergal<1e-5){continue;}
                    z_gal1 = zbin_lens_resos[ind_gal1];
                    pos1_gal1 = pos1_lens_resos[ind_gal1];
                    pos2_gal1 = pos2_lens_resos[ind_gal1];
                    w_gal1 = innergal*w_lens_resos[ind_gal1];
                    z1shift = z_gal1*nbinsz_source*nbinsr;
                    
                    FLATCELL_FOREACH(
                        index_matcher_source, rshift_index_matcher_source[elreso_leaf], pixs_galind_bounds_source, rshift_pixs_galind_bounds_source[elreso_leaf],
                        pos1_gal1, pos2_gal1, rmax_reso, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, lower2, upper2){
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
                            ngg_fill_gnwn(thisGns, thisWns, thisG2ns, thisW2ns, z2rshift,
                                nbinszr_reso, nbinszr_reso, nmax, w_gal1, w_gal2, wshape_gal2, phirot);
                        }
                    }

                    // Update the region caches and the same-reso Upsilon-/Upsilon+/N_n
                    build_redpix_by_reso2(elreso, nresos, nresos_grid, hasdiscrete,
                        z_gal1, pos1_gal1, pos2_gal1, hashpix_start1, hashpix_start2,
                        dpix1_resos, dpix2_resos, matchers_resoshift, len_matcher,
                        pix2redpix, redpix_by_reso2);
                    ngg_update_gnwncache(&ctx, elreso, rbinmin, rbinmax, nbinsr_reso, z_gal1, w_gal1,
                                         redpix_by_reso2, thisGns, thisWns);
                    ngg_accum_samereso(&ctx, rbinmin, nbinsr_reso, z_gal1, w_gal1,
                                       thisGns, thisWns, thisG2ns, thisW2ns, thisncounts,
                                       allowedrinds, allowedzinds);
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
            ngg_accum_crossreso(&ctx);

            free(redpix_by_reso2);
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
        free(Wncache);
        free(wWncache);
    }

    ngg_reduce(nbinsz_lens, nbinsz_source, nbinsr, nmax, nthreads,
               _upsiloncompshift, _upsilonthreadshift, _normthreadshift,
               tmpUpsilon, tmpNorm, tmpwcounts, tmpwnorms, out);
    if (verbose>0){ printf("\n"); }
    free(tmpUpsilon);
    free(tmpNorm);
    free(tmpwcounts);
    free(tmpwnorms);
    free(regionsdone);
}


// Discrete NGG using the 3dbox geometry and restricting measurement to slabs
// Note that in this setup we have a polar, a (clustered) counts, and a random catalog;
// here we just allocate all the correlators; for the detailed accumulation consult the python layer
void alloc_Gammans_slab_NGG(const MultiresoCatalog *cat_lensD, const MultiresoCatalog *cat_lensR,
                            const MultiresoCatalog *cat_shapeD, const NavHash *nav_shapeD,
                            const NavHash *nav_lensR, const BinningParams *bin,
                            int nthreads, int verbose, NPCFOutput *out){
                                
    // Dereference input args
    double *pos1_D = cat_lensD->pos1_resos, *pos2_D = cat_lensD->pos2_resos, *pos3_D = cat_lensD->pos3_resos;
    double *w_D = cat_lensD->weight_resos; int *zbin_D = cat_lensD->zbin_resos;
    int ngal_D = cat_lensD->ngal_resos[0], nbinsz_lens = cat_lensD->nbinsz;
    double *pos1_Rl = cat_lensR->pos1_resos, *pos2_Rl = cat_lensR->pos2_resos, *pos3_Rl = cat_lensR->pos3_resos;
    double *w_Rl = cat_lensR->weight_resos; int *zbin_Rl = cat_lensR->zbin_resos;
    int ngal_Rl = cat_lensR->ngal_resos[0];
    double *pos1_sD = cat_shapeD->pos1_resos, *pos2_sD = cat_shapeD->pos2_resos, *pos3_sD = cat_shapeD->pos3_resos;
    double *w_sD = cat_shapeD->weight_resos, *e1_sD = cat_shapeD->e1_resos, *e2_sD = cat_shapeD->e2_resos;
    int *zbin_sD = cat_shapeD->zbin_resos, nbinsz_polar = cat_shapeD->nbinsz;
    int *im_sD = nav_shapeD->index_matcher, *pgb_sD = nav_shapeD->pixs_galind_bounds, *pg_sD = nav_shapeD->pix_gals;
    int *so_sD = nav_shapeD->slab_offsets, *rsb_sD = nav_shapeD->rshift_bounds;
    int nslabs_sD = nav_shapeD->nslabs; double z0_sD = nav_shapeD->z0, dpixz_sD = nav_shapeD->dpix_z;
    double p1s_sD = nav_shapeD->pix1_start, p1d_sD = nav_shapeD->pix1_d; int p1n_sD = nav_shapeD->pix1_n;
    double p2s_sD = nav_shapeD->pix2_start, p2d_sD = nav_shapeD->pix2_d; int p2n_sD = nav_shapeD->pix2_n;
    int *im_Rl = nav_lensR->index_matcher, *pgb_Rl = nav_lensR->pixs_galind_bounds, *pg_Rl = nav_lensR->pix_gals;
    int *so_Rl = nav_lensR->slab_offsets, *rsb_Rl = nav_lensR->rshift_bounds;
    int nslabs_Rl = nav_lensR->nslabs; double z0_Rl = nav_lensR->z0, dpixz_Rl = nav_lensR->dpix_z;
    double p1s_Rl = nav_lensR->pix1_start, p1d_Rl = nav_lensR->pix1_d; int p1n_Rl = nav_lensR->pix1_n;
    double p2s_Rl = nav_lensR->pix2_start, p2d_Rl = nav_lensR->pix2_d; int p2n_Rl = nav_lensR->pix2_n;
    int nmax = bin->nmax, nbinsr = bin->nbinsr, dccorr = bin->dccorr;
    double rmin = bin->rmin, rmax = bin->rmax, Pi = bin->Pi;
    double *bin_centers = out->bin_centers;
    double complex *Comp_n = out->npcf, *RRR_n = out->norm_mp;

    int nmp = 2*nmax+1;
    int nnvals_Gn = 2*nmax+5; // m in [-nmax-2, nmax+2]
    int nnvals_Wn = 2*nmax+1;
    int nbinszr_leaf = nbinsz_polar*nbinsr;
    int nzcombis = nbinsz_lens*nbinsz_polar*nbinsz_polar;
    int ups_zshift = nbinsr*nbinsr;
    int ups_nshift = ups_zshift*nzcombis;
    int ups_compshift = nmp*ups_nshift;
    int ncomp_est = 2; // DSS, RSS
    int ups_threadshift = ncomp_est*2*ups_compshift; // 2 est + 2 natural
    int norm_threadshift = nmp*ups_nshift;
    int counts_threadshift = nbinsz_lens*nbinsz_polar*nbinsr;

    double complex *tmpComp = calloc((size_t)nthreads*ups_threadshift, sizeof(double complex));
    double complex *tmpNorm = calloc((size_t)nthreads*norm_threadshift, sizeof(double complex));
    double *tmpwcounts = calloc((size_t)nthreads*counts_threadshift, sizeof(double));
    double *tmpwnorms  = calloc((size_t)nthreads*counts_threadshift, sizeof(double));

    // Progress is tracked per base galaxy across both the lens-data and the random loop
    int nregionsdone = 0, progtot = ngal_D + ngal_Rl;
    reset_progress();
    #pragma omp parallel num_threads(nthreads)
    {
        int elthread = omp_get_thread_num();
        double complex *thDSS = tmpComp + (size_t)elthread*ups_threadshift;
        double complex *thRSS = thDSS + 2*ups_compshift;
        double complex *thNorm = tmpNorm + (size_t)elthread*norm_threadshift;
        double *thwc = tmpwcounts + (size_t)elthread*counts_threadshift;
        double *thwn = tmpwnorms  + (size_t)elthread*counts_threadshift;

        // Get DSS numerator
        #pragma omp for schedule(dynamic, 256)
        for (int ig=0; ig<ngal_D; ig++){
            #pragma omp atomic
            nregionsdone += 1;
            print_progress(nregionsdone, progtot, verbose);
            double c1 = pos1_D[ig], c2 = pos2_D[ig], c3 = pos3_D[ig], wc = w_D[ig];
            int zc = zbin_D[ig];
            double complex *Gn = calloc(nnvals_Gn*nbinszr_leaf, sizeof(double complex));
            double complex *sumG4  = calloc(nbinszr_leaf, sizeof(double complex));
            double complex *sumGabs = calloc(nbinszr_leaf, sizeof(double complex));
            int *ncounts = calloc(nbinszr_leaf, sizeof(int));
            int *allowedr = calloc(nbinszr_leaf, sizeof(int));
            int *allowedz = calloc(nbinszr_leaf, sizeof(int));
            double *wc_base = thwc + zc*nbinsz_polar*nbinsr;
            double *wn_base = thwn + zc*nbinsz_polar*nbinsr;

            slab_polar_leafmultipoles(c1, c2, c3, wc,
                pos1_sD, pos2_sD, pos3_sD, w_sD, zbin_sD, e1_sD, e2_sD,
                nbinsz_polar, nslabs_sD, z0_sD, dpixz_sD, p1s_sD, p1d_sD, p1n_sD,
                p2s_sD, p2d_sD, p2n_sD, so_sD, im_sD, pgb_sD, rsb_sD, pg_sD,
                rmin, rmax, nbinsr, Pi, -nmax-2, nnvals_Gn, Gn,
                (LeafSelfTerm[]){ {4, false, sumG4}, {0, true, sumGabs} }, 2, ncounts, wc_base, wn_base);

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

        // Get RSS numerator and RRR normalisation
        #pragma omp for schedule(dynamic, 256)
        for (int ig=0; ig<ngal_Rl; ig++){
            #pragma omp atomic
            nregionsdone += 1;
            print_progress(nregionsdone, progtot, verbose);
            double c1 = pos1_Rl[ig], c2 = pos2_Rl[ig], c3 = pos3_Rl[ig], wc = w_Rl[ig];
            int zc = zbin_Rl[ig];
            // RSS numerator: shape-data G leafs, weight +w (raw)
            double complex *Gn = calloc(nnvals_Gn*nbinszr_leaf, sizeof(double complex));
            double complex *sumG4  = calloc(nbinszr_leaf, sizeof(double complex));
            double complex *sumGabs = calloc(nbinszr_leaf, sizeof(double complex));
            int *ncG = calloc(nbinszr_leaf, sizeof(int));
            int *arG = calloc(nbinszr_leaf, sizeof(int));
            int *azG = calloc(nbinszr_leaf, sizeof(int));
            slab_polar_leafmultipoles(c1, c2, c3, wc,
                pos1_sD, pos2_sD, pos3_sD, w_sD, zbin_sD, e1_sD, e2_sD,
                nbinsz_polar, nslabs_sD, z0_sD, dpixz_sD, p1s_sD, p1d_sD, p1n_sD,
                p2s_sD, p2d_sD, p2n_sD, so_sD, im_sD, pgb_sD, rsb_sD, pg_sD,
                rmin, rmax, nbinsr, Pi, -nmax-2, nnvals_Gn, Gn,
                (LeafSelfTerm[]){ {4, false, sumG4}, {0, true, sumGabs} }, 2, ncG, NULL, NULL);
            int naG = 0;
            for (int z=0; z<nbinsz_polar; z++){ for (int r=0; r<nbinsr; r++){
                if (ncG[z*nbinsr+r] != 0){ arG[naG]=r; azG[naG]=z; naG++; } } }
            ngg_accum_upsilon(thRSS, wc, zc, Gn, sumG4, sumGabs, arG, azG,
                              naG, nmax, nbinsr, nbinsz_lens, nbinsz_polar, dccorr);
            free(Gn); free(sumG4); free(sumGabs); free(ncG); free(arG); free(azG);

            // RRR normalization
            double complex *Wn = calloc(nnvals_Wn*nbinszr_leaf, sizeof(double complex));
            double complex *sumW2 = calloc(nbinszr_leaf, sizeof(double complex));
            int *ncW = calloc(nbinszr_leaf, sizeof(int));
            int *arW = calloc(nbinszr_leaf, sizeof(int));
            int *azW = calloc(nbinszr_leaf, sizeof(int));
            slab_count_leafmultipoles(c1, c2, c3, wc,
                pos1_Rl, pos2_Rl, pos3_Rl, w_Rl, zbin_Rl,
                nbinsz_polar, nslabs_Rl, z0_Rl, dpixz_Rl, p1s_Rl, p1d_Rl, p1n_Rl,
                p2s_Rl, p2d_Rl, p2n_Rl, so_Rl, im_Rl, pgb_Rl, rsb_Rl, pg_Rl,
                rmin, rmax, nbinsr, Pi, -nmax, nnvals_Wn, Wn,
                (LeafSelfTerm[]){ {0, false, sumW2} }, 1, ncW, NULL, NULL);
            int naW = 0;
            for (int z=0; z<nbinsz_polar; z++){ for (int r=0; r<nbinsr; r++){
                if (ncW[z*nbinsr+r] != 0){ arW[naW]=r; azW[naW]=z; naW++; } } }
            ngg_accum_norm(thNorm, wc, zc, Wn, sumW2, arW, azW, naW,
                           nmax, nbinsr, nbinsz_lens, nbinsz_polar, dccorr);
            free(Wn); free(sumW2); free(ncW); free(arW); free(azW);
        }
    }

    // Reduce all components
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

    // Get bin centers and finalize.
    double *totcounts = calloc(counts_threadshift, sizeof(double));
    double *totnorms  = calloc(counts_threadshift, sizeof(double));
    for (int t=0; t<nthreads; t++){
        size_t ts = (size_t)t*counts_threadshift;
        for (int i=0; i<counts_threadshift; i++){ totcounts[i]+=tmpwcounts[ts+i]; totnorms[i]+=tmpwnorms[ts+i]; }
    }
    for (int i=0; i<counts_threadshift; i++){
        if (totnorms[i] != 0){ bin_centers[i] = totcounts[i]/totnorms[i]; }
    }

    if (verbose>0){ printf("\n"); }
    free(tmpComp); free(tmpNorm); free(tmpwcounts); free(tmpwnorms);
    free(totcounts); free(totnorms);
}