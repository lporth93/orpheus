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
#include "corrfunc_fourth.h"
#include "corrfunc_fourth_derived.h"
#include "utils.h"
#include "healpix_utils.h"

#define M_PI      3.14159265358979323846
#define INV_2PI   0.15915494309189534561

///////////////////////
/// General helpers ///
///////////////////////

// Cumulative per-reso offsets into the stacked multihash arrays.
// Only useable for flat geometries using tree/basetree/doubletree approximations
static void build_rshift_offsets(int nresos, int npix_hash, const int *ngal_resos,
    int *rshift_index_matcher, int *rshift_pixs_galind_bounds, int *rshift_pix_gals){
    for (int elreso=1; elreso<nresos; elreso++){
        rshift_index_matcher[elreso] = rshift_index_matcher[elreso-1] + npix_hash;
        rshift_pixs_galind_bounds[elreso] = rshift_pixs_galind_bounds[elreso-1] + ngal_resos[elreso-1]+1;
        rshift_pix_gals[elreso] = rshift_pix_gals[elreso-1] + ngal_resos[elreso-1];
    }
}

// The six orderings of the radial bin triple (r1,r2,r3) and the number of them that are
// distinct. Returns that number and fills the permutation table.
static inline int build_bincombi_trafos(int elb1, int elb2, int elb3, int trafos[6][3]){
    trafos[0][0]=elb1; trafos[0][1]=elb2; trafos[0][2]=elb3;
    trafos[1][0]=elb2; trafos[1][1]=elb3; trafos[1][2]=elb1;
    trafos[2][0]=elb3; trafos[2][1]=elb1; trafos[2][2]=elb2;
    trafos[3][0]=elb1; trafos[3][1]=elb3; trafos[3][2]=elb2;
    trafos[4][0]=elb2; trafos[4][1]=elb1; trafos[4][2]=elb3;
    trafos[5][0]=elb3; trafos[5][1]=elb2; trafos[5][2]=elb1;
    if ((elb1==elb2)&&(elb1==elb3)){return 1;}
    else if ((elb1==elb2)&&(elb1!=elb3)){return 3;}
    else if ((elb1==elb3)&&(elb1!=elb2)){return 3;}
    else if ((elb2==elb3)&&(elb2!=elb1)){return 3;}
    else{return 6;}
}

// Unpack one batch of radial-bin triples: split each packed combination into its three bin
// indices, build the radial bin edges, and work out which resolution bands the batch spans.
static void build_thetabatch(int elthetbatch, int batch_nthetas, int nbinsr, int nresos,
    double rmin, double rmax, const int *thetacombis_batches, const int *cumthetacombis_batches,
    const double *reso_redges,
    int *elb1s_batch, int *elb2s_batch, int *elb3s_batch, double *bin_edges,
    int *rbin_min_batch, int *rbin_max_batch, int *reso_min_batch, int *reso_max_batch){
    double drbin = (log(rmax)-log(rmin))/(nbinsr);
    for (int elb=0; elb<batch_nthetas; elb++){
        int thisrcombi = thetacombis_batches[cumthetacombis_batches[elthetbatch]+elb];
        elb1s_batch[elb] = thisrcombi/(nbinsr*nbinsr);
        elb2s_batch[elb] = (thisrcombi-elb1s_batch[elb]*nbinsr*nbinsr)/nbinsr;
        elb3s_batch[elb] = thisrcombi-elb1s_batch[elb]*nbinsr*nbinsr-elb2s_batch[elb]*nbinsr;
        *rbin_min_batch = mymin(*rbin_min_batch, elb1s_batch[elb]);
        *rbin_max_batch = mymax(*rbin_max_batch, elb3s_batch[elb]);
    }
    bin_edges[0] = rmin;
    for (int elb=0; elb<nbinsr; elb++){
        bin_edges[elb+1] = bin_edges[elb]*exp(drbin);
    }
    for (int elreso=1; elreso<nresos; elreso++){
        if (reso_redges[elreso] <= bin_edges[*rbin_min_batch  ]){*reso_min_batch += 1;}
        if (reso_redges[elreso] <  bin_edges[*rbin_max_batch+1]){*reso_max_batch += 1;}
    }
}


///////////////////////////////////
/// CORRELATOR SPECIFIC HELPERS ///
///////////////////////////////////

// (A) GGGG HELPERS //

// Per-pair update of the Gn/Wn caches and their multiple-counting corrections
// * Assumes the multipole structure for GGGG correlators
// * nband is the upper edge of the n-loop, basically 2*nmax_alloc
static inline __attribute__((always_inline)) void gggg_fill_gnwn(
    double complex *nextGns, double complex *nextG2ns_gg, double complex *nextG2ns_ggc,
    double complex *nextWns, double complex *nextW2ns, double complex *nextW3ns,
    double complex *nextG3ns_ggg, double complex *nextG3ns_gggc,
    int nband, int nbinszr, int zrshift, int ind_Gn, int ind_G2n, int ind_Wn,
    double w2, double w2_sq, double complex wshape2, double complex wshape_sq,
    double complex wshape_cube, double complex wshapewshapec, double complex wshapesqwshapec,
    double complex phirot, double complex phirotc, double complex fourphirotc){
    double complex nphirot = 1+I*0;
    double complex nphirotc = 1+I*0;
    int nextnshift = 0;

    // Triple-counting corr
    nextW3ns[zrshift] += w2_sq*w2;
    nextG3ns_ggg[zrshift] += wshape_cube*fourphirotc;
    nextG3ns_ggg[nbinszr + zrshift] += wshape_cube*fourphirotc*fourphirotc;
    nextG3ns_gggc[zrshift] += wshapesqwshapec;
    nextG3ns_gggc[nbinszr + zrshift] += wshapesqwshapec*fourphirotc;

    // Nominal G and double-counting corr
    // n = 0
    nextGns[ind_Gn] += wshape2*nphirot;
    nextG2ns_gg[ind_G2n] += wshape_sq*nphirot;
    nextG2ns_ggc[ind_G2n] += wshapewshapec*nphirot;
    nextWns[ind_Wn] += w2*nphirot;
    nextW2ns[ind_Wn] += w2_sq*nphirot;
    // n \in [-2*nmax+1,2*nmax-1]
    nphirot *= phirot;
    nphirotc *= phirotc;
    // n in [1, ..., nband-1] x {+1,-1}
    for (int nextn=1; nextn<nband; nextn++){
        nextnshift = nextn*nbinszr;
        nextGns[ind_Gn+nextnshift] += wshape2*nphirot;
        nextGns[ind_Gn-nextnshift] += wshape2*nphirotc;
        nextG2ns_gg[ind_G2n+nextnshift] += wshape_sq*nphirot;
        nextG2ns_gg[ind_G2n-nextnshift] += wshape_sq*nphirotc;
        nextG2ns_ggc[ind_G2n+nextnshift] += wshapewshapec*nphirot;
        nextG2ns_ggc[ind_G2n-nextnshift] += wshapewshapec*nphirotc;
        nextWns[ind_Wn+nextnshift] += w2*nphirot;
        nextWns[ind_Wn-nextnshift] += w2*nphirotc;
        nextW2ns[ind_Wn+nextnshift] += w2_sq*nphirot;
        nextW2ns[ind_Wn-nextnshift] += w2_sq*nphirotc;
        nphirot *= phirot;
        nphirotc *= phirotc;
    }

    // n = \pm 2*nmax
    nextnshift += nbinszr;
    nextGns[ind_Gn-nextnshift] += wshape2*nphirotc;
    nextG2ns_gg[ind_G2n+nextnshift] += wshape_sq*nphirot;
    nextG2ns_gg[ind_G2n-nextnshift] += wshape_sq*nphirotc;
    nextG2ns_ggc[ind_G2n+nextnshift] += wshapewshapec*nphirot;
    nextG2ns_ggc[ind_G2n-nextnshift] += wshapewshapec*nphirotc;
    nextWns[ind_Wn+nextnshift] += w2*nphirot;
    nextWns[ind_Wn-nextnshift] += w2*nphirotc;
    nextW2ns[ind_Wn+nextnshift] += w2_sq*nphirot;
    nextW2ns[ind_Wn-nextnshift] += w2_sq*nphirotc;
    nphirot *= phirot;
    nphirotc *= phirotc;

    // n = \pm 2*nmax+1
    nextnshift += nbinszr;
    nextGns[ind_Gn-nextnshift] += wshape2*nphirotc;
    nextG2ns_gg[ind_G2n+nextnshift] += wshape_sq*nphirot;
    nextG2ns_gg[ind_G2n-nextnshift] += wshape_sq*nphirotc;
    nextG2ns_ggc[ind_G2n+nextnshift] += wshapewshapec*nphirot;
    nextG2ns_ggc[ind_G2n-nextnshift] += wshapewshapec*nphirotc;
    nphirotc *= phirotc;
    // n =  -2*nmax-2
    nextnshift += nbinszr;
    nextGns[ind_Gn-nextnshift] += wshape2*nphirotc;
    nextG2ns_gg[ind_G2n-nextnshift] += wshape_sq*nphirotc;
    nextG2ns_ggc[ind_G2n-nextnshift] += wshapewshapec*nphirotc;
    nphirotc *= phirotc;
    // n =  -2*nmax-3
    nextnshift += nbinszr;
    nextGns[ind_Gn-nextnshift] += wshape2*nphirotc;
    nextG2ns_gg[ind_G2n-nextnshift] += wshape_sq*nphirotc;
    nextG2ns_ggc[ind_G2n-nextnshift] += wshapewshapec*nphirotc;
    nphirotc *= phirotc;
    // n =  -2*nmax-4
    nextnshift += nbinszr;
    nextG2ns_gg[ind_G2n-nextnshift] += wshape_sq*nphirotc;
    nextG2ns_ggc[ind_G2n-nextnshift] += wshapewshapec*nphirotc;
    nphirotc *= phirotc;
    // n =  -2*nmax-5
    nextnshift += nbinszr;
    nextG2ns_gg[ind_G2n-nextnshift] += wshape_sq*nphirotc;
    nextG2ns_ggc[ind_G2n-nextnshift] += wshapewshapec*nphirotc;
}

// Accumulate batched Upsilon components + their scalar norm for one base galaxy
// See Porth+25 for the multipole structure of the nominal allocation and for the multiple-counting corrections.
static inline __attribute__((always_inline)) void gggg_accum_batchUpsilon(
    double complex *batchUpsilon_n, double complex *batchN_n,
    int batch_nthetas, int batchups_compshift, int thisnshift,
    const int *elb1s_batch, const int *elb2s_batch, const int *elb3s_batch,
    const double complex *nextGns, const double complex *nextG2ns_gg,
    const double complex *nextG2ns_ggc, const double complex *nextG3ns_ggg,
    const double complex *nextG3ns_gggc, const double complex *nextWns,
    const double complex *nextW2ns, const double complex *nextW3ns,
    double complex wshape1, double complex wshape1c, double w1,
    int nbinsr, int nzero_G2n, int nzero_Wn, int thisn2, int thisn3,
    int thisGshift_mn2m2, int thisGshift_n2m2, int thisWshift_n2,
    int thisGshift_mn3m3, int thisGshift_mn3m1, int thisGshift_n3m3, int thisGshift_n3m1,
    int thisWshift_n3, int thisGshift_mn2mn3m3, int thisGshift_mn2mn3m1,
    int thisGshift_n2n3m3, int thisGshift_n2n3m1, int thisWshift_n2n3){
    int elb1, elb2, elb3, thisnrshift;
    double complex gGG0, gGG1, gGG2, gGG3, gGG4, gGG5, gGG6, gGG7, wNN;
    for (int elb=0; elb<batch_nthetas; elb++){
        elb1 = elb1s_batch[elb];
        elb2 = elb2s_batch[elb];
        elb3 = elb3s_batch[elb];
        thisnrshift = thisnshift + elb;
        // Multiple counting corrections:
        // sum_(i neq j neq k) = sum_(i,j,k) - ( sum_(i, j, i=k) + 2perm ) + 2 * sum_(i, i=j, i=k)
        // Triple-counting corr
        if ((elb1==elb2) && (elb1==elb3) && (elb2==elb3)){
            batchUpsilon_n[0*batchups_compshift+thisnrshift] += 2*wshape1  * nextG3ns_ggg[1*nbinsr+elb1];
            batchUpsilon_n[1*batchups_compshift+thisnrshift] += 2*wshape1c * nextG3ns_ggg[0*nbinsr+elb1];
            batchUpsilon_n[2*batchups_compshift+thisnrshift] += 2*wshape1  * nextG3ns_gggc[1*nbinsr+elb1];
            batchUpsilon_n[3*batchups_compshift+thisnrshift] += 2*wshape1  * nextG3ns_gggc[1*nbinsr+elb1];
            batchUpsilon_n[4*batchups_compshift+thisnrshift] += 2*wshape1  * nextG3ns_gggc[1*nbinsr+elb1];
            batchUpsilon_n[5*batchups_compshift+thisnrshift] += 2*wshape1c * nextG3ns_gggc[0*nbinsr+elb1];
            batchUpsilon_n[6*batchups_compshift+thisnrshift] += 2*wshape1c * nextG3ns_gggc[0*nbinsr+elb1];
            batchUpsilon_n[7*batchups_compshift+thisnrshift] += 2*wshape1c * nextG3ns_gggc[0*nbinsr+elb1];
            batchN_n[thisnrshift] += 2 * w1*nextW3ns[elb1];
        }
        // Double-counting corr for theta1==theta2
        if (elb1==elb2){
            batchUpsilon_n[0*batchups_compshift+thisnrshift] -= wshape1  *
                nextG2ns_gg[(nzero_G2n+thisn3-5)*nbinsr+elb1]  * nextGns[thisGshift_mn3m3+elb3];
            batchUpsilon_n[1*batchups_compshift+thisnrshift] -= wshape1c *
                nextG2ns_gg[(nzero_G2n+thisn3-3)*nbinsr+elb1]  * nextGns[thisGshift_mn3m1+elb3];
            batchUpsilon_n[2*batchups_compshift+thisnrshift] -= wshape1  *
                nextG2ns_ggc[(nzero_G2n+thisn3-1)*nbinsr+elb1] * nextGns[thisGshift_mn3m3+elb3];
            batchUpsilon_n[3*batchups_compshift+thisnrshift] -= wshape1  *
                nextG2ns_ggc[(nzero_G2n+thisn3-1)*nbinsr+elb1] * nextGns[thisGshift_mn3m3+elb3];
            batchUpsilon_n[4*batchups_compshift+thisnrshift] -= wshape1  *
                nextG2ns_gg[(nzero_G2n+thisn3-5)*nbinsr+elb1]  * conj(nextGns[thisGshift_n3m1+elb3]);
            batchUpsilon_n[5*batchups_compshift+thisnrshift] -= wshape1c *
                nextG2ns_ggc[(nzero_G2n+thisn3+1)*nbinsr+elb1] * nextGns[thisGshift_mn3m1+elb3];
            batchUpsilon_n[6*batchups_compshift+thisnrshift] -= wshape1c *
                nextG2ns_ggc[(nzero_G2n+thisn3+1)*nbinsr+elb1] * nextGns[thisGshift_mn3m1+elb3];
            batchUpsilon_n[7*batchups_compshift+thisnrshift] -= wshape1c *
                nextG2ns_gg[(nzero_G2n+thisn3-3)*nbinsr+elb1]  * conj(nextGns[thisGshift_n3m3+elb3]);
            batchN_n[thisnrshift] -= w1 * 
                nextW2ns[(nzero_Wn+thisn3)*nbinsr+elb1]*conj(nextWns[thisWshift_n3+elb3]);
        }
        // Double-counting corr for theta1==theta3  
        if (elb1==elb3){
            batchUpsilon_n[0*batchups_compshift+thisnrshift] -= wshape1  *
                nextG2ns_gg[(nzero_G2n+thisn2-6)*nbinsr+elb1]  * nextGns[thisGshift_mn2m2+elb2];
            batchUpsilon_n[1*batchups_compshift+thisnrshift] -= wshape1c *
                nextG2ns_gg[(nzero_G2n+thisn2-2)*nbinsr+elb1]  * nextGns[thisGshift_mn2m2+elb2];
            batchUpsilon_n[2*batchups_compshift+thisnrshift] -= wshape1  *
                nextG2ns_ggc[(nzero_G2n+thisn2-2)*nbinsr+elb1] * nextGns[thisGshift_mn2m2+elb2];
            batchUpsilon_n[3*batchups_compshift+thisnrshift] -= wshape1  *
                nextG2ns_gg[(nzero_G2n+thisn2-6)*nbinsr+elb1]  * conj(nextGns[thisGshift_n2m2+elb2]);
            batchUpsilon_n[4*batchups_compshift+thisnrshift] -= wshape1  *
                nextG2ns_ggc[(nzero_G2n+thisn2-2)*nbinsr+elb1] * nextGns[thisGshift_mn2m2+elb2];
            batchUpsilon_n[5*batchups_compshift+thisnrshift] -= wshape1c *
                nextG2ns_ggc[(nzero_G2n+thisn2+2)*nbinsr+elb1] * nextGns[thisGshift_mn2m2+elb2];
            batchUpsilon_n[6*batchups_compshift+thisnrshift] -= wshape1c *
                nextG2ns_gg[(nzero_G2n+thisn2-2)*nbinsr+elb1]  * conj(nextGns[thisGshift_n2m2+elb2]);
            batchUpsilon_n[7*batchups_compshift+thisnrshift] -= wshape1c *
                nextG2ns_ggc[(nzero_G2n+thisn2+2)*nbinsr+elb1] * nextGns[thisGshift_mn2m2+elb2];
            batchN_n[thisnrshift] -= w1 * 
                nextW2ns[(nzero_Wn+thisn2)*nbinsr+elb1] * conj(nextWns[thisWshift_n2+elb2]);
        }
        // Double-counting corr for theta2==theta3
        if (elb2==elb3){
            batchUpsilon_n[0*batchups_compshift+thisnrshift] -= wshape1  *
                nextG2ns_gg[(nzero_G2n-thisn2-thisn3-5)*nbinsr+elb2]  * 
                nextGns[thisGshift_n2n3m3+elb1];
            batchUpsilon_n[1*batchups_compshift+thisnrshift] -= wshape1c *
                nextG2ns_gg[(nzero_G2n-thisn2-thisn3-3)*nbinsr+elb2]  * 
                nextGns[thisGshift_n2n3m1+elb1];
            batchUpsilon_n[2*batchups_compshift+thisnrshift] -= wshape1  *
                nextG2ns_gg[(nzero_G2n-thisn2-thisn3-5)*nbinsr+elb2]  * 
                conj(nextGns[thisGshift_mn2mn3m1+elb1]);
            batchUpsilon_n[3*batchups_compshift+thisnrshift] -= wshape1  *
                nextG2ns_ggc[(nzero_G2n-thisn2-thisn3-1)*nbinsr+elb2] * 
                nextGns[thisGshift_n2n3m3+elb1];
            batchUpsilon_n[4*batchups_compshift+thisnrshift] -= wshape1  *
                nextG2ns_ggc[(nzero_G2n-thisn2-thisn3-1)*nbinsr+elb2] * 
                nextGns[thisGshift_n2n3m3+elb1];
            batchUpsilon_n[5*batchups_compshift+thisnrshift] -= wshape1c *
                nextG2ns_gg[(nzero_G2n-thisn2-thisn3-3)*nbinsr+elb2]  * 
                conj(nextGns[thisGshift_mn2mn3m3+elb1]);
            batchUpsilon_n[6*batchups_compshift+thisnrshift] -= wshape1c *
                nextG2ns_ggc[(nzero_G2n-thisn2-thisn3+1)*nbinsr+elb2] * 
                nextGns[thisGshift_n2n3m1+elb1];
            batchUpsilon_n[7*batchups_compshift+thisnrshift] -= wshape1c *
                nextG2ns_ggc[(nzero_G2n-thisn2-thisn3+1)*nbinsr+elb2] *
                nextGns[thisGshift_n2n3m1+elb1];
            batchN_n[thisnrshift] -= w1 * 
                nextW2ns[(nzero_Wn-thisn2-thisn3)*nbinsr+elb2] * nextWns[thisWshift_n2n3+elb1];
        }
        // Nominal allocation
        gGG0 = wshape1*nextGns[thisGshift_n2n3m3+elb1]*nextGns[thisGshift_mn2m2+elb2];
        gGG1 = wshape1c*nextGns[thisGshift_n2n3m1+elb1]*nextGns[thisGshift_mn2m2+elb2];
        gGG2 = wshape1*conj(nextGns[thisGshift_mn2mn3m1+elb1])*nextGns[thisGshift_mn2m2+elb2];
        gGG3 = wshape1*nextGns[thisGshift_n2n3m3+elb1]*conj(nextGns[thisGshift_n2m2+elb2]);
        gGG4 = wshape1*nextGns[thisGshift_n2n3m3+elb1]*nextGns[thisGshift_mn2m2+elb2];
        gGG5 = wshape1c*conj(nextGns[thisGshift_mn2mn3m3+elb1])*nextGns[thisGshift_mn2m2+elb2];
        gGG6 = wshape1c*nextGns[thisGshift_n2n3m1+elb1]*conj(nextGns[thisGshift_n2m2+elb2]);
        gGG7 = wshape1c*nextGns[thisGshift_n2n3m1+elb1]*nextGns[thisGshift_mn2m2+elb2];
        wNN = w1*nextWns[thisWshift_n2n3+elb1]*conj(nextWns[thisWshift_n2+elb2]);
        batchUpsilon_n[0*batchups_compshift+thisnrshift] += gGG0*nextGns[thisGshift_mn3m3+elb3];
        batchUpsilon_n[1*batchups_compshift+thisnrshift] += gGG1*nextGns[thisGshift_mn3m1+elb3];
        batchUpsilon_n[2*batchups_compshift+thisnrshift] += gGG2*nextGns[thisGshift_mn3m3+elb3];
        batchUpsilon_n[3*batchups_compshift+thisnrshift] += gGG3*nextGns[thisGshift_mn3m3+elb3];
        batchUpsilon_n[4*batchups_compshift+thisnrshift] += gGG4*conj(nextGns[thisGshift_n3m1+elb3]);
        batchUpsilon_n[5*batchups_compshift+thisnrshift] += gGG5*nextGns[thisGshift_mn3m1+elb3];
        batchUpsilon_n[6*batchups_compshift+thisnrshift] += gGG6*nextGns[thisGshift_mn3m1+elb3];
        batchUpsilon_n[7*batchups_compshift+thisnrshift] += gGG7*conj(nextGns[thisGshift_n3m3+elb3]);
        batchN_n[thisnrshift] += wNN*conj(nextWns[thisWshift_n3+elb3]);
    }
}

// (B) NNNN HELPERS //

// Leaves per chunk in the moment fill and galaxies per chunk in the triple accumulation,
// chosen such that the operands of the inner loops stay in L1/L2.
#define NNNN_LEAFCHUNK 512
#define NNNN_GALCHUNK 256

// Moments of a batch of base galaxies
// * We use the W_{-n} = conj(W_n) symmetry so only use n>=0 for W_n and W2_n
// * Layout is [rbin][n][galaxy] such that the triple accumulation runs over contiguous galaxies.
typedef struct {
    int cap, nbinsr, nmax_Wn;
    double *Wns_re, *Wns_im, *W2ns_re, *W2ns_im; // [nbinsr][nmax_Wn+1][cap]
    double *W3ns; // [nbinsr][cap]
    double *w1; // [cap], weight of the base galaxy
} NnnnBatch;

static void nnnn_batch_alloc(NnnnBatch *batch, int cap, int nbinsr, int nmax_Wn){
    long len = mymax((long)nbinsr*(nmax_Wn+1)*cap, 1L);
    batch->cap = cap; batch->nbinsr = nbinsr; batch->nmax_Wn = nmax_Wn;
    batch->Wns_re = orpheus_malloc(len*sizeof(double));
    batch->Wns_im = orpheus_malloc(len*sizeof(double));
    batch->W2ns_re = orpheus_malloc(len*sizeof(double));
    batch->W2ns_im = orpheus_malloc(len*sizeof(double));
    batch->W3ns = orpheus_malloc(mymax((long)nbinsr*cap, 1L)*sizeof(double));
    batch->w1 = orpheus_malloc((long)cap*sizeof(double));
}

static void nnnn_batch_free(NnnnBatch *batch){
    free(batch->Wns_re); free(batch->Wns_im); free(batch->W2ns_re); free(batch->W2ns_im);
    free(batch->W3ns); free(batch->w1);
}

// Base galaxies per batch: 64 per thread for a balanced moment fill, capped by memory_bound (GiB)
static int nnnn_batch_cap(long ncache, int nbinsr, int nmax_Wn, double memory_bound, int nthreads){
    long per_gal = ((long)4*nbinsr*(nmax_Wn+1) + nbinsr + 1)*(long)sizeof(double);
    long cap = 64L*nthreads;
    if (memory_bound > 0){ cap = mymin(cap, (long)(memory_bound*1073741824.)/per_gal); }
    cap = mymin(cap, ncache);
    return (int)mymax(cap, 1L);
}

// Leaves of a single base galaxy (radial bin, weight, phase e^{i phi}), their copy sorted
// by radial bin and the running powers of the phases. One per thread, grown on demand.
typedef struct {
    int nleaves, cap, nbinsr;
    int *rbin, *binstart, *cursor;
    double *w, *phirot_re, *phirot_im, *w_sorted, *phirot_re_sorted, *phirot_im_sorted;
    double *nphirot_re, *nphirot_im;
} NnnnLeaves;

static void nnnn_leaves_alloc(NnnnLeaves *lv, int nbinsr){
    lv->nleaves = 0; lv->cap = 4096; lv->nbinsr = nbinsr;
    lv->rbin = orpheus_malloc(lv->cap*sizeof(int));
    lv->binstart = orpheus_malloc((nbinsr+1)*sizeof(int));
    lv->cursor = orpheus_malloc(nbinsr*sizeof(int));
    lv->w = orpheus_malloc(lv->cap*sizeof(double));
    lv->phirot_re = orpheus_malloc(lv->cap*sizeof(double));
    lv->phirot_im = orpheus_malloc(lv->cap*sizeof(double));
    lv->w_sorted = orpheus_malloc(lv->cap*sizeof(double));
    lv->phirot_re_sorted = orpheus_malloc(lv->cap*sizeof(double));
    lv->phirot_im_sorted = orpheus_malloc(lv->cap*sizeof(double));
    lv->nphirot_re = orpheus_malloc(NNNN_LEAFCHUNK*sizeof(double));
    lv->nphirot_im = orpheus_malloc(NNNN_LEAFCHUNK*sizeof(double));
}

static void nnnn_leaves_free(NnnnLeaves *lv){
    free(lv->rbin); free(lv->binstart); free(lv->cursor); free(lv->w);
    free(lv->phirot_re); free(lv->phirot_im); free(lv->w_sorted);
    free(lv->phirot_re_sorted); free(lv->phirot_im_sorted); free(lv->nphirot_re); free(lv->nphirot_im);
}

static inline void nnnn_leaves_push(NnnnLeaves *lv, int rbin, double w, double phirot_re, double phirot_im){
    if (lv->nleaves == lv->cap){
        lv->cap *= 2;
        lv->rbin = realloc(lv->rbin, lv->cap*sizeof(int));
        lv->w = realloc(lv->w, lv->cap*sizeof(double));
        lv->phirot_re = realloc(lv->phirot_re, lv->cap*sizeof(double));
        lv->phirot_im = realloc(lv->phirot_im, lv->cap*sizeof(double));
        lv->w_sorted = realloc(lv->w_sorted, lv->cap*sizeof(double));
        lv->phirot_re_sorted = realloc(lv->phirot_re_sorted, lv->cap*sizeof(double));
        lv->phirot_im_sorted = realloc(lv->phirot_im_sorted, lv->cap*sizeof(double));
    }
    lv->rbin[lv->nleaves] = rbin; lv->w[lv->nleaves] = w;
    lv->phirot_re[lv->nleaves] = phirot_re; lv->phirot_im[lv->nleaves] = phirot_im;
    lv->nleaves += 1;
}

// Work list of the batched NNNN accumulation
// * The wanted ordered radial triples, grouped by their two smaller bins (t1,t2). Pair p owns
//   the triples pair_start[p]..pair_start[p+1]-1, sorted by t3, and pair_order lists the
//   pairs by decreasing work.
// * The half of the needed (n2,n3) with n3>0 or n3=0<=n2, grouped by n3. The other half
//   follows from conjugation for real weights, nindices_half/nindices_conj map every nindex onto it.
typedef struct {
    int npairs, *pair_elb1, *pair_elb2, *pair_start, *pair_order;
    int *triple_elb3; long *triple_combi;
    int nnvals_n3, n2n3combis_half, nmax_Wn, nrowmax, ntriplemax;
    int *n3_start, *n2s, *nindices_half, *nindices_conj;
} NnnnPlan;

static void nnnn_plan_build(NnnnPlan *p, const int *combis, long ncombis, int nbinsr,
    const int *nindices, int len_nindices, int nmax){
    int nbinsr2 = nbinsr*nbinsr;

    // Group the triples by (t1,t2) and sort them by t3 within a pair
    int *paircount = orpheus_calloc(nbinsr2+1, sizeof(int));
    for (long elcombi=0; elcombi<ncombis; elcombi++){ paircount[combis[elcombi]/nbinsr+1] += 1; }
    for (int k=0; k<nbinsr2; k++){ paircount[k+1] += paircount[k]; }
    p->triple_elb3 = orpheus_malloc((ncombis>0?ncombis:1)*sizeof(int));
    p->triple_combi = orpheus_malloc((ncombis>0?ncombis:1)*sizeof(long));
    int *fill = orpheus_calloc(nbinsr2, sizeof(int));
    for (int elb3=0; elb3<nbinsr; elb3++){
        for (long elcombi=0; elcombi<ncombis; elcombi++){
            if (combis[elcombi]%nbinsr != elb3){ continue; }
            int elb12 = combis[elcombi]/nbinsr;
            long k = paircount[elb12] + fill[elb12]++;
            p->triple_elb3[k] = elb3; p->triple_combi[k] = elcombi;
        }
    }
    p->npairs = 0;
    for (int elb12=0; elb12<nbinsr2; elb12++){ if (paircount[elb12+1]>paircount[elb12]){ p->npairs += 1; } }
    p->pair_elb1 = orpheus_malloc((p->npairs>0?p->npairs:1)*sizeof(int));
    p->pair_elb2 = orpheus_malloc((p->npairs>0?p->npairs:1)*sizeof(int));
    p->pair_start = orpheus_malloc((p->npairs+1)*sizeof(int));
    p->pair_order = orpheus_malloc((p->npairs>0?p->npairs:1)*sizeof(int));
    int ip = 0;
    p->ntriplemax = 1;
    for (int elb12=0; elb12<nbinsr2; elb12++){
        if (paircount[elb12+1] == paircount[elb12]){ continue; }
        p->pair_elb1[ip] = elb12/nbinsr; p->pair_elb2[ip] = elb12%nbinsr;
        p->pair_start[ip] = paircount[elb12];
        p->ntriplemax = mymax(p->ntriplemax, paircount[elb12+1]-paircount[elb12]);
        ip += 1;
    }
    p->pair_start[p->npairs] = (int)ncombis;
    // Largest pairs first for the dynamic schedule
    for (int k=0; k<p->npairs; k++){ p->pair_order[k] = k; }
    for (int k=1; k<p->npairs; k++){
        int cur = p->pair_order[k], ncur = p->pair_start[cur+1]-p->pair_start[cur], j = k-1;
        while (j>=0 && p->pair_start[p->pair_order[j]+1]-p->pair_start[p->pair_order[j]] < ncur){
            p->pair_order[j+1] = p->pair_order[j]; j--;
        }
        p->pair_order[j+1] = cur;
    }
    free(paircount); free(fill);

    // Half of the multipole pairs, grouped by n3, and the map of every nindex onto it. Each
    // (n2,n3) is represented by itself if n3>0 or n3=0<=n2, else by (-n2,-n3).
    int nmax_alloc = 2*nmax+1, nnvals_Nn = 2*nmax_alloc+1, nzero_Nn = nmax_alloc;
    int *hmap = orpheus_malloc(nnvals_Nn*nnvals_Nn*sizeof(int));
    for (int i=0; i<nnvals_Nn*nnvals_Nn; i++){ hmap[i] = -1; }
    p->nnvals_n3 = 0;
    for (int i=0; i<len_nindices; i++){
        int n2 = nindices[i]/nnvals_Nn - nzero_Nn, n3 = nindices[i]%nnvals_Nn - nzero_Nn;
        int direct = (n3>0 || (n3==0 && n2>=0));
        if (!direct){ n2 = -n2; n3 = -n3; }
        p->nnvals_n3 = mymax(p->nnvals_n3, n3+1);
        hmap[(n2+nzero_Nn)*nnvals_Nn + n3+nzero_Nn] = 0;
    }
    p->n3_start = orpheus_calloc(p->nnvals_n3+1, sizeof(int));
    p->n2n3combis_half = 0; p->nmax_Wn = 0; p->nrowmax = 1;
    for (int n3=0; n3<p->nnvals_n3; n3++){
        p->n3_start[n3] = p->n2n3combis_half;
        for (int n2=-nzero_Nn; n2<=nzero_Nn; n2++){
            int idx = (n2+nzero_Nn)*nnvals_Nn + n3+nzero_Nn;
            if (hmap[idx] != 0){ continue; }
            hmap[idx] = 1 + p->n2n3combis_half++;
            p->nmax_Wn = mymax(p->nmax_Wn, mymax(abs(n2), mymax(abs(n3), abs(n2+n3))));
        }
        p->nrowmax = mymax(p->nrowmax, p->n2n3combis_half-p->n3_start[n3]);
    }
    p->n3_start[p->nnvals_n3] = p->n2n3combis_half;
    p->n2s = orpheus_malloc((p->n2n3combis_half>0?p->n2n3combis_half:1)*sizeof(int));
    for (int idx=0; idx<nnvals_Nn*nnvals_Nn; idx++){
        if (hmap[idx] > 0){ p->n2s[hmap[idx]-1] = idx/nnvals_Nn - nzero_Nn; }
    }
    p->nindices_half = orpheus_malloc((len_nindices>0?len_nindices:1)*sizeof(int));
    p->nindices_conj = orpheus_malloc((len_nindices>0?len_nindices:1)*sizeof(int));
    for (int i=0; i<len_nindices; i++){
        int n2 = nindices[i]/nnvals_Nn - nzero_Nn, n3 = nindices[i]%nnvals_Nn - nzero_Nn;
        int direct = (n3>0 || (n3==0 && n2>=0));
        int idx = direct ? nindices[i] : (-n2+nzero_Nn)*nnvals_Nn + (-n3+nzero_Nn);
        p->nindices_half[i] = hmap[idx]-1;
        p->nindices_conj[i] = !direct;
    }
    free(hmap);
}

static void nnnn_plan_free(NnnnPlan *p){
    free(p->pair_elb1); free(p->pair_elb2); free(p->pair_start); free(p->pair_order);
    free(p->triple_elb3); free(p->triple_combi); free(p->n3_start); free(p->n2s);
    free(p->nindices_half); free(p->nindices_conj);
}

// Apex objects of a batch per resolution band, with their moments. 
// * Tree uses a single band holding all radial bins. 
// * DoubleTree has one band per tree resolution: 
//   * The moments of its apex objects (galaxies, or cells of that resolution) for the bins of the band
//   * The w-weighted moments of all finer bands summed per cell (wWns rows [bin][n], bin<rindedges[band])
//   * The parent of every object in the next coarser band.
typedef struct {
    int nbands;
    const int *rindedges, *bin2band; // [nbands+1] first radial bin per band, [nbinsr] band of a bin
    NnnnBatch *batch; // [nbands]
    int *nobj; // [nbands]
    int **parent; // [nbands]
    double **wWns_re, **wWns_im; // [nbands]
    long *wWns_cap; // [nbands] row length of wWns
} NnnnBands;

// Moments and summed moments of the DoubleTree bands, grown to the objects of the current batch
static void nnnn_bands_reserve(NnnnBands *bd, int nmax_Wn){
    for (int elband=0; elband<bd->nbands; elband++){
        int nbins = bd->rindedges[elband+1]-bd->rindedges[elband];
        if (bd->batch[elband].cap < bd->nobj[elband]){
            if (bd->batch[elband].cap > 0){ nnnn_batch_free(bd->batch+elband); }
            nnnn_batch_alloc(bd->batch+elband, (int)(1.5*bd->nobj[elband])+1, nbins, nmax_Wn);
        }
        if (elband>0 && bd->wWns_cap[elband] < bd->nobj[elband]){
            bd->wWns_cap[elband] = (long)(1.5*bd->nobj[elband]) + 1;
            free(bd->wWns_re[elband]); free(bd->wWns_im[elband]);
            bd->wWns_re[elband] = orpheus_malloc((long)bd->rindedges[elband]*(nmax_Wn+1)*bd->wWns_cap[elband]*sizeof(double));
            bd->wWns_im[elband] = orpheus_malloc((long)bd->rindedges[elband]*(nmax_Wn+1)*bd->wWns_cap[elband]*sizeof(double));
        }
    }
}

// Per-thread scratch of nnnn_accum_pair. The rows of A summed per cell of band k are kept in
// Asum_re[k]/Asum_im[k], nrowmax rows of length Asum_cap[k].
typedef struct {
    double *A_re, *A_im; // nrowmax x NNNN_GALCHUNK
    double *C_re, *C_im; // nrowmax x ntriplemax
    const double **B_re, **B_im; // ntriplemax
    double **Asum_re, **Asum_im;
    long *Asum_cap;
} NnnnScratch;

// Per-thread scratch for nbands bands, the summed rows of A initially empty
static NnnnScratch *nnnn_scratch_alloc(int nthreads, int nbands, const NnnnPlan *p){
    NnnnScratch *sc = orpheus_calloc(nthreads, sizeof(NnnnScratch));
    for (int elthread=0; elthread<nthreads; elthread++){
        sc[elthread].A_re = orpheus_malloc((long)p->nrowmax*NNNN_GALCHUNK*sizeof(double));
        sc[elthread].A_im = orpheus_malloc((long)p->nrowmax*NNNN_GALCHUNK*sizeof(double));
        sc[elthread].C_re = orpheus_malloc((long)p->nrowmax*p->ntriplemax*sizeof(double));
        sc[elthread].C_im = orpheus_malloc((long)p->nrowmax*p->ntriplemax*sizeof(double));
        sc[elthread].B_re = orpheus_malloc(p->ntriplemax*sizeof(double*));
        sc[elthread].B_im = orpheus_malloc(p->ntriplemax*sizeof(double*));
        sc[elthread].Asum_re = orpheus_calloc(nbands, sizeof(double*));
        sc[elthread].Asum_im = orpheus_calloc(nbands, sizeof(double*));
        sc[elthread].Asum_cap = orpheus_calloc(nbands, sizeof(long));
    }
    return sc;
}

// Grow the summed rows of A of every band >= 1 to the objects of the current batch
static void nnnn_scratch_reserve(NnnnScratch *sc, int nthreads, int nbands, const int *nobj, int nrowmax){
    for (int elthread=0; elthread<nthreads; elthread++){
        for (int elband=1; elband<nbands; elband++){
            if (sc[elthread].Asum_cap[elband] >= nobj[elband]){ continue; }
            sc[elthread].Asum_cap[elband] = (long)(1.5*nobj[elband]) + 1;
            free(sc[elthread].Asum_re[elband]); free(sc[elthread].Asum_im[elband]);
            sc[elthread].Asum_re[elband] = orpheus_malloc(nrowmax*sc[elthread].Asum_cap[elband]*sizeof(double));
            sc[elthread].Asum_im[elband] = orpheus_malloc(nrowmax*sc[elthread].Asum_cap[elband]*sizeof(double));
        }
    }
}

static void nnnn_scratch_free(NnnnScratch *sc, int nthreads, int nbands){
    for (int elthread=0; elthread<nthreads; elthread++){
        for (int elband=0; elband<nbands; elband++){ free(sc[elthread].Asum_re[elband]); free(sc[elthread].Asum_im[elband]); }
        free(sc[elthread].A_re); free(sc[elthread].A_im); free(sc[elthread].C_re); free(sc[elthread].C_im);
        free(sc[elthread].B_re); free(sc[elthread].B_im); free(sc[elthread].Asum_re); free(sc[elthread].Asum_im); free(sc[elthread].Asum_cap);
    }
    free(sc);
}

// Apex objects of a batch of DoubleTree regions per band: their index into the multireso
// catalogue and the index of their parent within the next coarser band of the batch
typedef struct {
    int nbands, *nobj, *cap;
    long **ind_obj;
    int **parent;
} NnnnRegions;

static void nnnn_regions_alloc(NnnnRegions *rg, int nbands){
    rg->nbands = nbands;
    rg->nobj = orpheus_calloc(nbands, sizeof(int));
    rg->cap = orpheus_malloc(nbands*sizeof(int));
    rg->ind_obj = orpheus_malloc(nbands*sizeof(long*));
    rg->parent = orpheus_malloc(nbands*sizeof(int*));
    for (int elband=0; elband<nbands; elband++){
        rg->cap[elband] = 1024;
        rg->ind_obj[elband] = orpheus_malloc(rg->cap[elband]*sizeof(long));
        rg->parent[elband] = orpheus_malloc(rg->cap[elband]*sizeof(int));
    }
}

static void nnnn_regions_free(NnnnRegions *rg){
    for (int elband=0; elband<rg->nbands; elband++){ free(rg->ind_obj[elband]); free(rg->parent[elband]); }
    free(rg->nobj); free(rg->cap); free(rg->ind_obj); free(rg->parent);
}

static inline void nnnn_regions_push(NnnnRegions *rg, int band, long ind_obj, int parent){
    if (rg->nobj[band] == rg->cap[band]){
        rg->cap[band] *= 2;
        rg->ind_obj[band] = realloc(rg->ind_obj[band], rg->cap[band]*sizeof(long));
        rg->parent[band] = realloc(rg->parent[band], rg->cap[band]*sizeof(int));
    }
    rg->ind_obj[band][rg->nobj[band]] = ind_obj;
    rg->parent[band][rg->nobj[band]] = parent;
    rg->nobj[band] += 1;
}

// Per-pair update of the Wn caches and their multiple-counting corrections
// * Assumes the multipole structure for NNNN correlators
// * nband is the upper edge of the n-loop, basically 2*nmax_alloc
static inline void nnnn_fill_wn(
    double complex *nextWns, double complex *nextW2ns,
    int nband, int nbinszr, int ind_Wn, double w2, double w2_sq,
    double complex phirot, double complex phirotc){
    double complex nphirot = 1+I*0;
    double complex nphirotc = 1+I*0;
    nextWns[ind_Wn] += w2*nphirot;
    nextW2ns[ind_Wn] += w2_sq*nphirot;
    nphirot *= phirot;
    nphirotc *= phirotc;
    for (int nextn=1; nextn<=nband; nextn++){
        int nextnshift = nextn*nbinszr;
        nextWns[ind_Wn+nextnshift] += w2*nphirot;
        nextWns[ind_Wn-nextnshift] += w2*nphirotc;
        nextW2ns[ind_Wn+nextnshift] += w2_sq*nphirot;
        nextW2ns[ind_Wn-nextnshift] += w2_sq*nphirotc;
        nphirot *= phirot;
        nphirotc *= phirotc;
    }
}

// Accumulate batched count components for one base galaxy
// See Porth+25 for the multipole structure of the nominal allocation and for the multiple-counting corrections.
static inline __attribute__((always_inline)) void nnnn_accum_batchNn(
    double complex *batchN_n, int batch_nthetas, int batchN_nshift,
    const int *elb1s_batch, const int *elb2s_batch, const int *elb3s_batch,
    const double complex *nextWns, const double complex *nextW2ns, const double complex *nextW3ns,
    int *nindices, int len_nindices, int nnvals_Nn, int nzero_Nn, int nzero_Wn, int nbinsr,
    double w1, int diagnose, int elregion, int elthetbatch){
    for (int nindex=0; nindex<len_nindices; nindex++){
        int thisn2 = nindices[nindex]/nnvals_Nn - nzero_Nn;
        int thisn3 = nindices[nindex]%nnvals_Nn - nzero_Nn;
        if (thisn2>nzero_Nn || -thisn2>nzero_Nn || thisn3>nzero_Nn || -thisn3>nzero_Nn){
            if (diagnose && elregion==0 && elthetbatch==0){
                printf("Error at elregion=%d batch=%d nindex=%d: nindices[nindex]=%d n2=%d n3=%d",
                       elregion, elthetbatch, nindex, nindices[nindex], thisn2, thisn3);}
            continue;
        }
        int thisn = thisn2+thisn3;
        int thisWshift_n2 = (nzero_Wn+thisn2)*nbinsr;
        int thisWshift_n3 = (nzero_Wn+thisn3)*nbinsr;
        int thisWshift_n2n3 = (nzero_Wn+thisn)*nbinsr;
        int thisnshift = ((thisn2+nzero_Nn)*nnvals_Nn + (thisn3+nzero_Nn)) * batchN_nshift;
        for (int elb=0; elb<batch_nthetas; elb++){
            int elb1 = elb1s_batch[elb];
            int elb2 = elb2s_batch[elb];
            int elb3 = elb3s_batch[elb];
            int thisnrshift = thisnshift + elb;
            // Triple-counting corr
            if ((elb1==elb2) && (elb1==elb3) && (elb2==elb3)){
                batchN_n[thisnrshift] += 2 * w1*nextW3ns[elb1];
            }
            // Double-counting corr for theta1==theta2
            if (elb1==elb2){
                batchN_n[thisnrshift] -= w1*nextW2ns[(nzero_Wn+thisn3)*nbinsr+elb1]*conj(nextWns[thisWshift_n3+elb3]);
            }
            // Double-counting corr for theta1==theta3
            if (elb1==elb3){
                batchN_n[thisnrshift] -= w1*nextW2ns[(nzero_Wn+thisn2)*nbinsr+elb1]*conj(nextWns[thisWshift_n2+elb2]);
            }
            // Double-counting corr for theta2==theta3
            if (elb2==elb3){
                batchN_n[thisnrshift] -= w1*nextW2ns[(nzero_Wn-thisn2-thisn3)*nbinsr+elb2]*nextWns[thisWshift_n2n3+elb1];
            }
            // Nominal allocation
            double complex wNN = w1*nextWns[thisWshift_n2n3+elb1]*conj(nextWns[thisWshift_n2+elb2]);
            batchN_n[thisnrshift] += wNN*conj(nextWns[thisWshift_n3+elb3]);
        }
    }
}

// Post-process the NNNN multipoles of a theta batch:
// First construct the whole multipoles set from symmetry d´conditions and then optionally
// convert to the real-space 4pcf and to the aperture statistics
static void nnnn_reconstruct_batch(
    const double complex *batchN_n, int batch_nthetas, int batchN_nshift,
    const int *elb1s_batch, const int *elb2s_batch, const int *elb3s_batch,
    int nmax, int *nindices, int len_nindices, int n2n3combis, int n2n3combis_rec,
    double complex *thisN_n, double complex *thisN_n_rec,
    double complex *N_n, int N_nshift, int nbinsr, int alloc_4pcfmultipoles, int accumulate,
    int alloc_4pcfreal, double complex *Counts, double complex *thisnpcf,
    double *phibins, double *dbinsphi, int nbinsphi, int batchgamma_thetshift,
    int nnapradii, double *napradii, double *bin_centers_batch, double *bin_edges,
    double complex *allN4correlators, double complex *nextN4correlators,
    int thisthread, int verbose){
    for (int elb=0; elb<batch_nthetas; elb++){
        if ((verbose>1) && (thisthread==0)){
            printf("Done %.4f per cent of multipole-to-Nap4 conversion\r",100.* (float) elb/batch_nthetas);}
        // 1)
        int bincombi_trafos[6][3];
        // 2)
        int ntrafos = build_bincombi_trafos(elb1s_batch[elb], elb2s_batch[elb], elb3s_batch[elb],
                                            bincombi_trafos);
        for (int eltrafo=0; eltrafo<ntrafos; eltrafo++){
            int elb1t = bincombi_trafos[eltrafo][0];
            int elb2t = bincombi_trafos[eltrafo][1];
            int elb3t = bincombi_trafos[eltrafo][2];

            // 2a)
            for (int eln12=0; eln12<n2n3combis; eln12++){
                thisN_n[eln12] = batchN_n[eln12*batchN_nshift+elb];
            }

            getMultipolesFromSymm_NNNN(thisN_n, nmax, eltrafo, nindices, len_nindices, thisN_n_rec);
            // OPTIONAL: Allocate 4PCF in multipole basis
            if (alloc_4pcfmultipoles==1){
                for (int eln12=0; eln12<n2n3combis_rec; eln12++){
                    long thisnrshift = (long)eln12*N_nshift + elb1t*nbinsr*nbinsr + elb2t*nbinsr + elb3t;
                    if (accumulate==1){ N_n[thisnrshift] += thisN_n_rec[eln12]; }
                    else{ N_n[thisnrshift] = thisN_n_rec[eln12]; }
                }
            }

            // 2b)
            // Only required for the real-space 4pcf and for the aperture integration
            if ((alloc_4pcfreal==1) || (nnapradii>0)){
                multipoles2npcf_nnnn_singletheta(thisN_n_rec, nmax, nmax,
                                                 elb1t, elb2t, elb3t,
                                                 phibins, phibins, nbinsphi, nbinsphi,
                                                 thisnpcf);
            }

            // OPTIONAL: Allocate 4pcf in real basis (Shape: (8,ntheta,ntheta,ntheta,nphi,nphi)
            if (alloc_4pcfreal==1){
                for (int elphi12=0; elphi12<batchgamma_thetshift; elphi12++){
                    long gamma_rshift = nbinsphi*nbinsphi;
                    long gamma_phircombi = gamma_rshift*(elb1t*nbinsr*nbinsr+elb2t*nbinsr+elb3t)+elphi12;
                    Counts[gamma_phircombi] = thisnpcf[elphi12];
                }
            }

            // 2c)
            int nap4threadshift = thisthread*nnapradii;
            for (int elnapr=0; elnapr<nnapradii; elnapr++){
                double y1 = bin_centers_batch[elb1t]/napradii[elnapr];
                double y2 = bin_centers_batch[elb2t]/napradii[elnapr];
                double y3 = bin_centers_batch[elb3t]/napradii[elnapr];
                double dy1 = (bin_edges[elb1t+1]-bin_edges[elb1t])/napradii[elnapr];
                double dy2 = (bin_edges[elb2t+1]-bin_edges[elb2t])/napradii[elnapr];
                double dy3 = (bin_edges[elb3t+1]-bin_edges[elb3t])/napradii[elnapr];

                fourpcf2N4correlators(1,
                                      y1, y2, y3, dy1, dy2, dy3,
                                      phibins, phibins, dbinsphi, dbinsphi, nbinsphi, nbinsphi,
                                      thisnpcf, nextN4correlators);

                if (isfinite(cabs(nextN4correlators[0]))){
                    allN4correlators[nap4threadshift+elnapr] += nextN4correlators[0];
                }
                nextN4correlators[0] = 0;
            }

            // Reset 4pcf placeholders to zero
            for (int i=0; i<batchgamma_thetshift; i++){ thisnpcf[i] = 0; }
            for (int i=0; i<n2n3combis; i++){ thisN_n[i] = 0; }
            for (int i=0; i<n2n3combis_rec; i++){ thisN_n_rec[i] = 0; }
        }
    }
}

// Moments of one base galaxy from its leaves into slot ig of the batch. All bins of the slot
// are written, so it needs no reset. The powers z^n are vectorised over the leaves of a bin.
static void nnnn_fill_slot(NnnnLeaves *lv, NnnnBatch *batch, int ig, double w1){
    int nbinsr = batch->nbinsr, nmax_Wn = batch->nmax_Wn;
    long cap = batch->cap;
    int *binstart = lv->binstart;

    // Counting sort of the leaves by radial bin
    for (int elbinr=0; elbinr<=nbinsr; elbinr++){ binstart[elbinr] = 0; }
    for (int j=0; j<lv->nleaves; j++){ binstart[lv->rbin[j]+1] += 1; }
    for (int elbinr=0; elbinr<nbinsr; elbinr++){
        binstart[elbinr+1] += binstart[elbinr];
        lv->cursor[elbinr] = binstart[elbinr];
    }
    for (int j=0; j<lv->nleaves; j++){
        int k = lv->cursor[lv->rbin[j]]++;
        lv->w_sorted[k] = lv->w[j];
        lv->phirot_re_sorted[k] = lv->phirot_re[j];
        lv->phirot_im_sorted[k] = lv->phirot_im[j];
    }

    batch->w1[ig] = w1;
    for (int elbinr=0; elbinr<nbinsr; elbinr++){
        long rshift = (long)elbinr*(nmax_Wn+1)*cap + ig;
        double W3n = 0.;
        for (int k=binstart[elbinr]; k<binstart[elbinr+1]; k++){
            W3n += lv->w_sorted[k]*lv->w_sorted[k]*lv->w_sorted[k];
        }
        batch->W3ns[(long)elbinr*cap+ig] = W3n;
        for (int thisn=0; thisn<=nmax_Wn; thisn++){
            batch->Wns_re[rshift+thisn*cap] = 0.; batch->Wns_im[rshift+thisn*cap] = 0.;
            batch->W2ns_re[rshift+thisn*cap] = 0.; batch->W2ns_im[rshift+thisn*cap] = 0.;
        }
        for (int k0=binstart[elbinr]; k0<binstart[elbinr+1]; k0+=NNNN_LEAFCHUNK){
            int nk = mymin(NNNN_LEAFCHUNK, binstart[elbinr+1]-k0);
            const double *w2 = lv->w_sorted+k0;
            const double *phirot_re = lv->phirot_re_sorted+k0, *phirot_im = lv->phirot_im_sorted+k0;
            double *nphirot_re = lv->nphirot_re, *nphirot_im = lv->nphirot_im;
            for (int k=0; k<nk; k++){ nphirot_re[k] = 1.; nphirot_im[k] = 0.; }
            for (int thisn=0; thisn<=nmax_Wn; thisn++){
                double Wn_re = 0., Wn_im = 0., W2n_re = 0., W2n_im = 0.;
                #pragma omp simd reduction(+:Wn_re,Wn_im,W2n_re,W2n_im)
                for (int k=0; k<nk; k++){
                    double wnphirot_re = w2[k]*nphirot_re[k];
                    double wnphirot_im = w2[k]*nphirot_im[k];
                    Wn_re += wnphirot_re;
                    Wn_im += wnphirot_im;
                    W2n_re += w2[k]*wnphirot_re;
                    W2n_im += w2[k]*wnphirot_im;
                    // Next power of the phase
                    double nphirot_re_next = nphirot_re[k]*phirot_re[k] - nphirot_im[k]*phirot_im[k];
                    nphirot_im[k] = nphirot_re[k]*phirot_im[k] + nphirot_im[k]*phirot_re[k];
                    nphirot_re[k] = nphirot_re_next;
                }
                batch->Wns_re[rshift+thisn*cap] += Wn_re;
                batch->Wns_im[rshift+thisn*cap] += Wn_im;
                batch->W2ns_re[rshift+thisn*cap] += W2n_re;
                batch->W2ns_im[rshift+thisn*cap] += W2n_im;
            }
        }
    }
    lv->nleaves = 0;
}

// C[r][cshift+c] += sum_g A[r][g] conj(B_c[g]) for the nrow rows of A and the ncol rows B_c over
// ng objects. The product runs over blocks of two rows and two columns, so that every element of
// A and B loaded is used twice. For an odd nrow or ncol the last block repeats its row or column,
// whose result is not stored.
static inline void nnnn_gemm_chunk(const double *A_re, const double *A_im, long stride_A, int nrow,
    const double **B_re, const double **B_im, int ncol, int ng,
    double *C_re, double *C_im, int stride_C, int cshift){
    for (int r0=0; r0<nrow; r0+=2){
        int r1 = mymin(r0+1, nrow-1);
        const double *A0_re = A_re + r0*stride_A, *A0_im = A_im + r0*stride_A;
        const double *A1_re = A_re + r1*stride_A, *A1_im = A_im + r1*stride_A;
        for (int c0=0; c0<ncol; c0+=2){
            int c1 = mymin(c0+1, ncol-1);
            const double *B0_re = B_re[c0], *B0_im = B_im[c0];
            const double *B1_re = B_re[c1], *B1_im = B_im[c1];

            // C_ij = sum_g A_i[g] conj(B_j[g]) for the 2x2 block
            double C00_re = 0., C00_im = 0., C01_re = 0., C01_im = 0.;
            double C10_re = 0., C10_im = 0., C11_re = 0., C11_im = 0.;
            #pragma omp simd reduction(+:C00_re,C00_im,C01_re,C01_im,C10_re,C10_im,C11_re,C11_im)
            for (int g=0; g<ng; g++){
                C00_re += A0_re[g]*B0_re[g] + A0_im[g]*B0_im[g];
                C00_im += A0_im[g]*B0_re[g] - A0_re[g]*B0_im[g];
                C01_re += A0_re[g]*B1_re[g] + A0_im[g]*B1_im[g];
                C01_im += A0_im[g]*B1_re[g] - A0_re[g]*B1_im[g];
                C10_re += A1_re[g]*B0_re[g] + A1_im[g]*B0_im[g];
                C10_im += A1_im[g]*B0_re[g] - A1_re[g]*B0_im[g];
                C11_re += A1_re[g]*B1_re[g] + A1_im[g]*B1_im[g];
                C11_im += A1_im[g]*B1_re[g] - A1_re[g]*B1_im[g];
            }

            // Add the block to C, without the repeated row or column of an odd edge
            C_re[r0*stride_C + cshift + c0] += C00_re;
            C_im[r0*stride_C + cshift + c0] += C00_im;
            if (c1 != c0){
                C_re[r0*stride_C + cshift + c1] += C01_re;
                C_im[r0*stride_C + cshift + c1] += C01_im;
            }
            if (r1 != r0){
                C_re[r1*stride_C + cshift + c0] += C10_re;
                C_im[r1*stride_C + cshift + c0] += C10_im;
                if (c1 != c0){
                    C_re[r1*stride_C + cshift + c1] += C11_re;
                    C_im[r1*stride_C + cshift + c1] += C11_im;
                }
            }
        }
    }
}

// Pointers to x_m(t1) for objects g0.. of band2: w W_m(t1) for t1 in band2 (weighted with
// w in the caller), else the summed weighted moments of the finer band. sgn conjugates m<0.
static inline void nnnn_xrow(const NnnnBands *bd, int elb1, int band2, int m, int g0,
    const double **x_re, const double **x_im, double *sgn){
    const NnnnBatch *batch = bd->batch + band2;
    int nnvals_Wn = batch->nmax_Wn+1;
    *sgn = (m<0) ? -1. : 1.;
    if (bd->bin2band[elb1] == band2){
        long shift = ((long)(elb1-bd->rindedges[band2])*nnvals_Wn + abs(m))*batch->cap + g0;
        *x_re = batch->Wns_re + shift; *x_im = batch->Wns_im + shift;
    } else {
        long shift = ((long)elb1*nnvals_Wn + abs(m))*bd->wWns_cap[band2] + g0;
        *x_re = bd->wWns_re[band2] + shift; *x_im = bd->wWns_im[band2] + shift;
    }
}

// Accumulate the batch into the triples of radial pair ip. Per n3 this is a complex GEMM over
// the apex objects of band2, the band of t2,
//   A[n2][o] = x_{n2+n3}(t1) conj(W_{n2}(t2)) - delta_{t1t2} w W2_{n3}(t1),
// with x = w W(t1), or the summed weighted moments of the finer band of t1 (DoubleTree), times
// B[o][t3] = conj(W_{n3}(t3)) for t3 in band2. For a coarser t3 the rows of A are first summed
// into the cells of its band (Porth+25, App. F). The remaining multiple-counting corrections
// (Porth+25, Tab. A.1) only arise for t3=t2.
static void nnnn_accum_pair(const NnnnBands *bd, const NnnnPlan *p, int ip, NnnnScratch *sc,
    double complex *acc){
    int elb1 = p->pair_elb1[ip], elb2 = p->pair_elb2[ip];
    int band2 = bd->bin2band[elb2];
    int triple0 = p->pair_start[ip], ntriples = p->pair_start[ip+1]-triple0;
    const NnnnBatch *batch = bd->batch + band2;
    long cap = batch->cap, nshift = (long)(batch->nmax_Wn+1)*cap;
    int nobj = bd->nobj[band2];
    int elb2_inband = elb2-bd->rindedges[band2];
    int same12 = (elb1 == elb2), has_same23 = (p->triple_elb3[triple0] == elb2);
    if (nobj == 0){ return; }
    // The triples are sorted by t3, so those within band2 come first
    int ntriples_inband = 0;
    while (ntriples_inband<ntriples && bd->bin2band[p->triple_elb3[triple0+ntriples_inband]] == band2){ ntriples_inband++; }
    double *A_re = sc->A_re, *A_im = sc->A_im, *C_re = sc->C_re, *C_im = sc->C_im;

    for (int n3=0; n3<p->nnvals_n3; n3++){
        int halfshift = p->n3_start[n3], nrow = p->n3_start[n3+1]-halfshift;
        if (nrow == 0){ continue; }
        for (int i=0; i<nrow*ntriples; i++){ C_re[i] = 0.; C_im[i] = 0.; }
        if (ntriples_inband < ntriples){
            long len = (long)nrow*sc->Asum_cap[band2+1];
            for (long i=0; i<len; i++){ sc->Asum_re[band2+1][i] = 0.; sc->Asum_im[band2+1][i] = 0.; }
        }
        for (int g0=0; g0<nobj; g0+=NNNN_GALCHUNK){
            int ng = mymin(NNNN_GALCHUNK, nobj-g0);
            const double *w1 = batch->w1+g0;

            // Rows of A for this chunk of objects
            for (int r=0; r<nrow; r++){
                int n2 = p->n2s[halfshift+r];
                double sgn_x;
                double sgn_W = (n2<0) ? 1. : -1.;
                const double *x_re, *x_im;
                nnnn_xrow(bd, elb1, band2, n2+n3, g0, &x_re, &x_im, &sgn_x);
                const double *Wn2_re = batch->Wns_re + elb2_inband*nshift + abs(n2)*cap + g0;
                const double *Wn2_im = batch->Wns_im + elb2_inband*nshift + abs(n2)*cap + g0;
                double *Arow_re = A_re + (long)r*NNNN_GALCHUNK, *Arow_im = A_im + (long)r*NNNN_GALCHUNK;
                #pragma omp simd
                for (int g=0; g<ng; g++){
                    double x_im_g = sgn_x*x_im[g];
                    double Wn2_im_g = sgn_W*Wn2_im[g];
                    Arow_re[g] = x_re[g]*Wn2_re[g] - x_im_g*Wn2_im_g;
                    Arow_im[g] = x_re[g]*Wn2_im_g + x_im_g*Wn2_re[g];
                }
                if (bd->bin2band[elb1] == band2){
                    #pragma omp simd
                    for (int g=0; g<ng; g++){
                        Arow_re[g] *= w1[g];
                        Arow_im[g] *= w1[g];
                    }
                }
                if (same12){
                    const double *W2_re = batch->W2ns_re + elb2_inband*nshift + n3*cap + g0;
                    const double *W2_im = batch->W2ns_im + elb2_inband*nshift + n3*cap + g0;
                    #pragma omp simd
                    for (int g=0; g<ng; g++){
                        Arow_re[g] -= w1[g]*W2_re[g];
                        Arow_im[g] -= w1[g]*W2_im[g];
                    }
                }
            }

            // Triples with t3 in band2
            for (int c=0; c<ntriples_inband; c++){
                long Wshift = (long)(p->triple_elb3[triple0+c]-bd->rindedges[band2])*nshift + n3*cap + g0;
                sc->B_re[c] = batch->Wns_re + Wshift;
                sc->B_im[c] = batch->Wns_im + Wshift;
            }
            nnnn_gemm_chunk(A_re, A_im, NNNN_GALCHUNK, nrow, sc->B_re, sc->B_im, ntriples_inband, ng, C_re, C_im, ntriples, 0);

            // Corrections for t3=t2: - x_{n2+n3}(t1) W2_{-n2-n3}(t2), and for t1=t2=t3 also
            // - w W2_{n2}(t1) conj(W_{n2}(t2)) + 2 w W3(t1)
            if (has_same23){
                for (int r=0; r<nrow; r++){
                    int n2 = p->n2s[halfshift+r], m = n2+n3;
                    double sgn_x;
                    const double *x_re, *x_im;
                    nnnn_xrow(bd, elb1, band2, m, g0, &x_re, &x_im, &sgn_x);
                    const double *W2_re = batch->W2ns_re + elb2_inband*nshift + abs(m)*cap + g0;
                    const double *W2_im = batch->W2ns_im + elb2_inband*nshift + abs(m)*cap + g0;
                    double weight_x = (bd->bin2band[elb1] == band2);
                    double corr_re = 0., corr_im = 0.;
                    #pragma omp simd reduction(+:corr_re,corr_im)
                    for (int g=0; g<ng; g++){
                        double x_im_g = sgn_x*x_im[g];
                        double W2_im_g = -sgn_x*W2_im[g];
                        double fac = weight_x ? w1[g] : 1.;
                        corr_re -= fac*(x_re[g]*W2_re[g] - x_im_g*W2_im_g);
                        corr_im -= fac*(x_re[g]*W2_im_g + x_im_g*W2_re[g]);
                    }
                    if (same12){
                        double sgn_W = (n2<0) ? -1. : 1.;
                        const double *W2n2_re = batch->W2ns_re + elb2_inband*nshift + abs(n2)*cap + g0;
                        const double *W2n2_im = batch->W2ns_im + elb2_inband*nshift + abs(n2)*cap + g0;
                        const double *Wn2_re = batch->Wns_re + elb2_inband*nshift + abs(n2)*cap + g0;
                        const double *Wn2_im = batch->Wns_im + elb2_inband*nshift + abs(n2)*cap + g0;
                        const double *W3n = batch->W3ns + elb2_inband*cap + g0;
                        #pragma omp simd reduction(+:corr_re,corr_im)
                        for (int g=0; g<ng; g++){
                            double W2n2_im_g = sgn_W*W2n2_im[g];
                            double Wn2_im_g = -sgn_W*Wn2_im[g];
                            corr_re -= w1[g]*(W2n2_re[g]*Wn2_re[g] - W2n2_im_g*Wn2_im_g) - 2.*w1[g]*W3n[g];
                            corr_im -= w1[g]*(W2n2_re[g]*Wn2_im_g + W2n2_im_g*Wn2_re[g]);
                        }
                    }
                    C_re[r*ntriples] += corr_re;
                    C_im[r*ntriples] += corr_im;
                }
            }

            // Sum the rows of A into the cells of the next band
            if (ntriples_inband < ntriples){
                const int *parent = bd->parent[band2] + g0;
                long stride_coarse = sc->Asum_cap[band2+1];
                for (int r=0; r<nrow; r++){
                    const double *Arow_re = A_re + (long)r*NNNN_GALCHUNK, *Arow_im = A_im + (long)r*NNNN_GALCHUNK;
                    double *coarse_re = sc->Asum_re[band2+1] + r*stride_coarse;
                    double *coarse_im = sc->Asum_im[band2+1] + r*stride_coarse;
                    for (int g=0; g<ng; g++){
                        coarse_re[parent[g]] += Arow_re[g];
                        coarse_im[parent[g]] += Arow_im[g];
                    }
                }
            }
        }

        // Triples with t3 in a coarser band, from the rows of A summed into its cells
        int elband = band2+1;
        for (int eltriple=ntriples_inband; eltriple<ntriples;){
            int band3 = bd->bin2band[p->triple_elb3[triple0+eltriple]], eltriple_end = eltriple;
            while (eltriple_end<ntriples && bd->bin2band[p->triple_elb3[triple0+eltriple_end]] == band3){ eltriple_end++; }
            for (; elband<band3; elband++){
                long stride_fine = sc->Asum_cap[elband], stride_coarse = sc->Asum_cap[elband+1];
                const int *parent = bd->parent[elband];
                for (long i=0; i<(long)nrow*stride_coarse; i++){ sc->Asum_re[elband+1][i] = 0.; sc->Asum_im[elband+1][i] = 0.; }
                for (int r=0; r<nrow; r++){
                    const double *fine_re = sc->Asum_re[elband] + r*stride_fine;
                    const double *fine_im = sc->Asum_im[elband] + r*stride_fine;
                    double *coarse_re = sc->Asum_re[elband+1] + r*stride_coarse;
                    double *coarse_im = sc->Asum_im[elband+1] + r*stride_coarse;
                    for (int g=0; g<bd->nobj[elband]; g++){
                        coarse_re[parent[g]] += fine_re[g];
                        coarse_im[parent[g]] += fine_im[g];
                    }
                }
            }
            const NnnnBatch *batch3 = bd->batch + band3;
            long cap3 = batch3->cap, nshift3 = (long)(batch3->nmax_Wn+1)*cap3;
            for (int g0=0; g0<bd->nobj[band3]; g0+=NNNN_GALCHUNK){
                int ng = mymin(NNNN_GALCHUNK, bd->nobj[band3]-g0);
                for (int c=eltriple; c<eltriple_end; c++){
                    long Wshift = (long)(p->triple_elb3[triple0+c]-bd->rindedges[band3])*nshift3 + n3*cap3 + g0;
                    sc->B_re[c-eltriple] = batch3->Wns_re + Wshift;
                    sc->B_im[c-eltriple] = batch3->Wns_im + Wshift;
                }
                nnnn_gemm_chunk(sc->Asum_re[band3]+g0, sc->Asum_im[band3]+g0, sc->Asum_cap[band3], nrow, sc->B_re, sc->B_im,
                                eltriple_end-eltriple, ng, C_re, C_im, ntriples, eltriple);
            }
            eltriple = eltriple_end;
        }

        for (int r=0; r<nrow; r++){
            for (int c=0; c<ntriples; c++){
                acc[p->triple_combi[triple0+c]*p->n2n3combis_half + halfshift + r] += C_re[r*ntriples+c] + I*C_im[r*ntriples+c];
            }
        }
    }
}

// Expand the half multipole set by conjugation and add all orderings of every triple to N_n
static void nnnn_reconstruct_all(const double complex *acc, const NnnnPlan *p, const int *combis,
    long ncombis, int nbinsr, int nmax, int *nindices, int len_nindices,
    double complex *N_n, int nthreads){
    int nmax_alloc = 2*nmax+1;
    int nnvals_Nn = 2*nmax_alloc+1, nnvals_Nn_rec = 2*nmax+1;
    int n2n3combis = nnvals_Nn*nnvals_Nn, n2n3combis_rec = nnvals_Nn_rec*nnvals_Nn_rec;
    int N_nshift = nbinsr*nbinsr*nbinsr;
    int chunk = 64;
    #pragma omp parallel num_threads(nthreads)
    {
        double complex *batchN_n = orpheus_calloc((long)n2n3combis*chunk, sizeof(double complex));
        double complex *thisN_n = orpheus_calloc(n2n3combis, sizeof(double complex));
        double complex *thisN_n_rec = orpheus_calloc(n2n3combis_rec, sizeof(double complex));
        int *elb1s = orpheus_malloc(chunk*sizeof(int));
        int *elb2s = orpheus_malloc(chunk*sizeof(int));
        int *elb3s = orpheus_malloc(chunk*sizeof(int));
        #pragma omp for schedule(dynamic, 1)
        for (long combi0=0; combi0<ncombis; combi0+=chunk){
            int ncombis_chunk = (int)mymin((long)chunk, ncombis-combi0);
            for (int ic=0; ic<ncombis_chunk; ic++){
                int combi = combis[combi0+ic];
                elb1s[ic] = combi/(nbinsr*nbinsr);
                elb2s[ic] = (combi-elb1s[ic]*nbinsr*nbinsr)/nbinsr;
                elb3s[ic] = combi%nbinsr;
                const double complex *acc_combi = acc + (combi0+ic)*p->n2n3combis_half;
                for (int nindex=0; nindex<len_nindices; nindex++){
                    double complex Nn = acc_combi[p->nindices_half[nindex]];
                    batchN_n[(long)nindices[nindex]*ncombis_chunk+ic] = p->nindices_conj[nindex] ? conj(Nn) : Nn;
                }
            }
            nnnn_reconstruct_batch(batchN_n, ncombis_chunk, ncombis_chunk, elb1s, elb2s, elb3s,
                                   nmax, nindices, len_nindices, n2n3combis, n2n3combis_rec,
                                   thisN_n, thisN_n_rec, N_n, N_nshift, nbinsr, 1, 1,
                                   0, NULL, NULL, NULL, NULL, 0, 0, 0, NULL, NULL, NULL,
                                   NULL, NULL, 0, 0);
        }
        free(batchN_n); free(thisN_n); free(thisN_n_rec); free(elb1s); free(elb2s); free(elb3s);
    }
}

// First radial bin of every resolution band: a band starts at the bin edge closest to its
// switch radius, and the band changes at most once per bin.
static void nnnn_reso_rindedges(int nresos, const double *reso_redges, double rmin, double drbin,
    int nbinsr, int *reso_rindedges){
    for (int r=0; r<=nresos; r++){ reso_rindedges[r] = 0; }
    int tmpreso = 0; double tmpr = rmin;
    for (int elr=0; elr<nbinsr; elr++){
        tmpr *= exp(drbin);
        double thisredge = reso_redges[mymin(nresos, tmpreso+1)];
        if (thisredge < tmpr){
            reso_rindedges[mymin(nresos, tmpreso+1)] = elr;
            if ((tmpr-thisredge) < (thisredge-(tmpr/exp(drbin)))){ reso_rindedges[mymin(nresos, tmpreso+1)] += 1; }
            tmpreso += 1;
        }
    }
    reso_rindedges[nresos] = nbinsr;
}

// Sum the w-weighted moments of the finer bands into the cells of every band >= 1 (the
// scatter of D.2.3 in the flat DoubleTree), band by band through the parent links
static void nnnn_bands_aggregate(NnnnBands *bd, int nthreads){
    for (int elband=1; elband<bd->nbands; elband++){
        const NnnnBatch *batch_fine = bd->batch + (elband-1);
        int nnvals_Wn = batch_fine->nmax_Wn+1;
        int nobj_fine = bd->nobj[elband-1], nobj_coarse = bd->nobj[elband];
        long stride_fine = bd->wWns_cap[elband-1], stride_coarse = bd->wWns_cap[elband];
        const int *parent = bd->parent[elband-1];
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (int row=0; row<bd->rindedges[elband]*nnvals_Wn; row++){
            int elbinr = row/nnvals_Wn, thisn = row%nnvals_Wn;
            double *coarse_re = bd->wWns_re[elband] + row*stride_coarse;
            double *coarse_im = bd->wWns_im[elband] + row*stride_coarse;
            for (int g=0; g<nobj_coarse; g++){ coarse_re[g] = 0.; coarse_im[g] = 0.; }
            if (elbinr >= bd->rindedges[elband-1]){
                const double *fine_re = batch_fine->Wns_re + ((long)(elbinr-bd->rindedges[elband-1])*nnvals_Wn+thisn)*batch_fine->cap;
                const double *fine_im = batch_fine->Wns_im + ((long)(elbinr-bd->rindedges[elband-1])*nnvals_Wn+thisn)*batch_fine->cap;
                for (int g=0; g<nobj_fine; g++){
                    coarse_re[parent[g]] += batch_fine->w1[g]*fine_re[g];
                    coarse_im[parent[g]] += batch_fine->w1[g]*fine_im[g];
                }
            } else {
                const double *fine_re = bd->wWns_re[elband-1] + row*stride_fine;
                const double *fine_im = bd->wWns_im[elband-1] + row*stride_fine;
                for (int g=0; g<nobj_fine; g++){
                    coarse_re[parent[g]] += fine_re[g];
                    coarse_im[parent[g]] += fine_im[g];
                }
            }
        }
    }
}

// Triples of a batch of DoubleTree regions: summed moments, then every radial pair in
// parallel. Pairs whose t2 lies in a finer band run first, as they carry most of the work.
static void nnnn_bands_triples(NnnnBands *bd, const NnnnPlan *p, const int *pair_order_dt,
    NnnnScratch *sc, double complex *acc, int nthreads){
    nnnn_bands_aggregate(bd, nthreads);
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1)
    for (int elpair=0; elpair<p->npairs; elpair++){
        nnnn_accum_pair(bd, p, pair_order_dt[elpair], sc+omp_get_thread_num(), acc);
    }
}

// Order of the radial pairs for the DoubleTree: by the band of t2, then by decreasing triples
static int *nnnn_dt_pair_order(const NnnnPlan *p, const int *bin2band){
    int *order = orpheus_malloc((p->npairs>0?p->npairs:1)*sizeof(int));
    int nordered = 0, bandmax = 0;
    for (int elpair=0; elpair<p->npairs; elpair++){ bandmax = mymax(bandmax, bin2band[p->pair_elb2[elpair]]); }
    for (int band=0; band<=bandmax; band++){
        for (int elpair=0; elpair<p->npairs; elpair++){
            int ip = p->pair_order[elpair];
            if (bin2band[p->pair_elb2[ip]] == band){ order[nordered++] = ip; }
        }
    }
    return order;
}

// Number of finest-band apex objects per batch of regions: enough work for the threads in
// every radial pair, capped by memory_bound (GiB) for the moments of the finest band
static int nnnn_dt_batch_target(int nbins0, int nmax_Wn, double memory_bound, int nthreads){
    long per_obj = ((long)4*nbins0*(nmax_Wn+1) + nbins0 + 1)*(long)sizeof(double);
    long target = mymax(16384L, 128L*nthreads);
    if (memory_bound > 0){ target = mymin(target, (long)(memory_bound*1073741824.)/(2*per_obj)); }
    return (int)mymax(target, 1L);
}

static inline int nnnn_lower_bound_long(const long *cp, int n, long key){
    int lo = 0, hi = n;
    while (lo < hi){ int m = (lo+hi)>>1; if (cp[m] < key){ lo = m+1; } else { hi = m; } }
    return lo;
}


/////////////////////////////
// GGGG CORRELATOR CLASSES //
/////////////////////////////

// Non-tomo 4pcf using discrete estimator
// Very basic, no use of symmetry properties
void alloc_notomoGammans_discrete_gggg(const MultiresoCatalog *cat, const NavHash *nav,
                                       const BinningParams *bin, const FourthParams *fourth,
                                       int nthreads, int verbose, NPCFOutput *out){
    // Dereference input structs
    double *isinner = cat->isinner_resos, *weight = cat->weight_resos;
    double *pos1 = cat->pos1_resos, *pos2 = cat->pos2_resos;
    double *e1 = cat->e1_resos, *e2 = cat->e2_resos;
    double *rbins = bin->rbins;
    int nmax = bin->nmax, nbinsr = bin->nbinsr;
    double rmin = bin->rmin, rmax = bin->rmax;
    int *index_matcher_hash = nav->index_matcher, *pixs_galind_bounds = nav->pixs_galind_bounds, *pix_gals = nav->pix_gals;
    int nregions = nav->nregions;
    double pix1_start = nav->pix1_start, pix1_d = nav->pix1_d; int pix1_n = nav->pix1_n;
    double pix2_start = nav->pix2_start, pix2_d = nav->pix2_d; int pix2_n = nav->pix2_n;
    double *bin_centers = out->bin_centers;
    double complex *Upsilon_n = out->npcf, *N_n = out->norm_mp;
    (void)fourth;
    
    // Temporary arrays that are allocated in parallel and later reduced
    int _nnvals_Upsn = 2*nmax+1;
    int _ups_nshift = nbinsr*nbinsr*nbinsr;
    int _n2n3combis = _nnvals_Upsn*_nnvals_Upsn;
    int _ups_compshift = _n2n3combis*_ups_nshift;
    double *tmpwcounts = orpheus_calloc(nthreads*nbinsr, sizeof(double));
    double *tmpwnorms = orpheus_calloc(nthreads*nbinsr, sizeof(double));
    double complex *tmpUpsilon0_n = orpheus_calloc(nthreads*_ups_compshift, sizeof(double complex));
    double complex *tmpUpsilon1_n = orpheus_calloc(nthreads*_ups_compshift, sizeof(double complex));
    double complex *tmpUpsilon2_n = orpheus_calloc(nthreads*_ups_compshift, sizeof(double complex));
    double complex *tmpUpsilon3_n = orpheus_calloc(nthreads*_ups_compshift, sizeof(double complex));
    double complex *tmpUpsilon4_n = orpheus_calloc(nthreads*_ups_compshift, sizeof(double complex));
    double complex *tmpUpsilon5_n = orpheus_calloc(nthreads*_ups_compshift, sizeof(double complex));
    double complex *tmpUpsilon6_n = orpheus_calloc(nthreads*_ups_compshift, sizeof(double complex));
    double complex *tmpUpsilon7_n = orpheus_calloc(nthreads*_ups_compshift, sizeof(double complex));
    double complex *tmpN_n = orpheus_calloc(nthreads*_ups_compshift, sizeof(double complex));
    
    double *totcounts = orpheus_calloc(nbinsr, sizeof(double));
    double *totnorms = orpheus_calloc(nbinsr, sizeof(double));
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){
        free(tmpwcounts); free(tmpwnorms); free(tmpUpsilon0_n); free(tmpUpsilon1_n);
        free(tmpUpsilon2_n); free(tmpUpsilon3_n); free(tmpUpsilon4_n); free(tmpUpsilon5_n);
        free(tmpUpsilon6_n); free(tmpUpsilon7_n); free(tmpN_n); free(totcounts); free(totnorms);
        return;
    }
    
    int nregionsdone = 0;
    reset_progress();
    #pragma omp parallel for num_threads(nthreads)
    for (int elthread=0; elthread<nthreads; elthread++){
        int nregions_per_thread = nregions/nthreads;
        int nbinsz = 1;
        int nnvals_Gn = 4*nmax+3; // Need to cover [-n1-n2-3, n1+n2-1]
        int nnvals_G2n = 4*nmax+7; // Need to cover [-n1-n2-5, n1+n2+1]
        int nnvals_Wn = 4*nmax+1; // Need to cover [-n1-n2, n1+n2]
        int nnvals_Upsn = 2*nmax+1; // Need tocover [-nmax,+nmax]
        int nzero_Gn = 2*nmax+3;
        int nzero_G2n = 2*nmax+5;
        int nzero_Wn = 2*nmax;
        int nzero_Ups = nmax;
        
        int ups_nshift = nbinsr*nbinsr*nbinsr;
        int n2n3combis = nnvals_Upsn*nnvals_Upsn;
        int ups_compshift = n2n3combis*ups_nshift;

        int nbinszr = nbinsz*nbinsr;
        double complex *nextGns =  orpheus_calloc(nnvals_Gn*nbinszr, sizeof(double complex));
        double complex *nextG2ns_gg =  orpheus_calloc(nnvals_G2n*nbinszr, sizeof(double complex));
        double complex *nextG2ns_ggc =  orpheus_calloc(nnvals_G2n*nbinszr, sizeof(double complex));
        double complex *nextG3ns_ggg = orpheus_calloc(2*nbinszr, sizeof(double complex));
        double complex *nextG3ns_gggc = orpheus_calloc(2*nbinszr, sizeof(double complex));
        double complex *nextWns = orpheus_calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW2ns = orpheus_calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW3ns = orpheus_calloc(nbinszr, sizeof(double complex));
        
        for (int elregion=0; elregion<nregions; elregion++){
            int region_debug = mymin(500,nregions-1);
            // Check if this thread is responsible for the region
            int nthread_target = mymin(elregion/nregions_per_thread, nthreads-1);
            if (nthread_target!=elthread){continue;}
            // printf("Region %d is in thread %d\n",elregion,elthread);
            
            if ((verbose>1) && (elregion==region_debug)){printf("Region %d is in thread %d (%i regions in total)\n",
                                               elregion,elthread,nregions);}
            #pragma omp atomic
            nregionsdone += 1;
            print_progress(nregionsdone, nregions, verbose);
            int lower1 = pixs_galind_bounds[elregion];
            int upper1 = pixs_galind_bounds[elregion+1];
            for (int ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                int ind_gal = pix_gals[ind_inpix1];
                double p11, p12, w1, e11, e12;
                double innergal = isinner[ind_gal];
                if (innergal<1e-5){continue;}
                p11 = pos1[ind_gal];
                p12 = pos2[ind_gal];
                w1 = innergal*weight[ind_gal];
                e11 = e1[ind_gal];
                e12 = e2[ind_gal];     
                
                // Gns have shape (nnvals, nbinsz, nbinsr)
                // where the ns are ordered as 
                // [-nmax_1-nmax_2-3, ..., nmax_1+nmax_2+3]
                for (int i=0; i<nnvals_Gn*nbinszr; i++){nextGns[i]=0;}
                for (int i=0; i<nnvals_G2n*nbinszr; i++){nextG2ns_gg[i]=0;nextG2ns_ggc[i]=0;}
                for (int i=0; i<2*nbinszr; i++){nextG3ns_ggg[i]=0;nextG3ns_gggc[i]=0;}
                for (int i=0; i<nnvals_Wn*nbinszr; i++){nextWns[i]=0;nextW2ns[i]=0;}
                for (int i=0; i<nbinszr; i++){nextW3ns[i]=0;}
                
                int ind_gal2;
                int lower, upper;
                double  p21, p22, w2, w2_sq, e21, e22, rel1, rel2, dist2, dist, dphi;
                double complex wshape1, wshape1c, wshape2, wshape_sq, wshape_cube, wshapewshapec, wshapesqwshapec;
                double complex phirot, phirotc, twophirotc, fourphirotc;

                int ind_rbin, rbin, zrshift, ind_Gn, ind_G2n, ind_Wn;
                double drbin = (log(rmax)-log(rmin))/(nbinsr);
                double rmin2 = rmin*rmin;
                double rmax2 = rmax*rmax;
                FLATCELL_FOREACH(
                    index_matcher_hash, 0, pixs_galind_bounds, 0, p11, p12, rmax,
                    pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, lower, upper){
                    for (int ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                        ind_gal2 = pix_gals[ind_inpix];
                        p21 = pos1[ind_gal2];
                        p22 = pos2[ind_gal2];
                        w2 = weight[ind_gal2];
                        e21 = e1[ind_gal2];
                        e22 = e2[ind_gal2];

                        rel1 = p21 - p11;
                        rel2 = p22 - p12;
                        dist2 = rel1*rel1 + rel2*rel2;
                        if (dist2 < rmin2 || dist2 >= rmax2){continue;}
                        dist = sqrt(dist2);
                        if (rbins[0] < 0){
                            rbin = (int) floor((log(dist)-log(rmin))/drbin);
                        }
                        else{
                            rbin=0;
                            while (rbins[rbin+1] <= dist){rbin+=1;}
                        }
                        w2_sq = w2*w2;
                        wshape2 = (double complex) w2 * (e21+I*e22);
                        wshape_sq = wshape2*wshape2;
                        wshape_cube = wshape_sq*wshape2;
                        wshapewshapec = wshape2*conj(wshape2);
                        wshapesqwshapec = wshape_sq*conj(wshape2);
                        dphi = atan2(rel2,rel1);
                        phirot = cexp(I*dphi);
                        phirotc = conj(phirot);
                        twophirotc = phirotc*phirotc;
                        fourphirotc = twophirotc*twophirotc;
                        zrshift = 0*nbinsr + rbin;
                        ind_rbin = elthread*nbinszr + zrshift;
                        ind_Gn = nzero_Gn*nbinszr + zrshift;
                        ind_G2n = nzero_G2n*nbinszr + zrshift;
                        ind_Wn = nzero_Wn*nbinszr + zrshift;
                        tmpwcounts[ind_rbin] += w1*w2*dist; 
                        tmpwnorms[ind_rbin] += w1*w2; 
                        gggg_fill_gnwn(nextGns, nextG2ns_gg, nextG2ns_ggc,
                                        nextWns, nextW2ns, nextW3ns, nextG3ns_ggg, nextG3ns_gggc,
                                        2*nmax, nbinszr, zrshift, ind_Gn, ind_G2n, ind_Wn,
                                        w2, w2_sq, wshape2, wshape_sq, wshape_cube,
                                        wshapewshapec, wshapesqwshapec, phirot, phirotc, fourphirotc);
                    }
                }

                // Allocate Upsilon
                // Upsilon have shape
                // (8,(2*nmax+1),(2*nmax+1),nbinsr,nbinsr,nbinsr)
                double complex gGG0, gGG1, gGG2, gGG3, gGG4, gGG5, gGG6, gGG7, wNN;
                int thisn, thisnshift, thisnrshift;
                int thisGshift_mn2m2, thisGshift_n2m2, thisWshift_n2;
                int thisGshift_mn3m3, thisGshift_mn3m1, thisGshift_n3m3, thisGshift_n3m1, thisWshift_n3;
                int thisGshift_mn2mn3m3, thisGshift_mn2mn3m1, thisGshift_n2n3m3, thisGshift_n2n3m1, thisWshift_n2n3;
                double complex triplecorrA, triplecorrB;
                wshape1 = w1 * (e11+I*e12);  
                wshape1c = conj(wshape1);
                for (int thisn2=-nmax; thisn2<=nmax; thisn2++){
                    thisGshift_mn2m2 = (nzero_Gn-thisn2-2)*nbinsr;
                    thisGshift_n2m2 = (nzero_Gn+thisn2-2)*nbinsr;
                    thisWshift_n2 = (nzero_Wn+thisn2)*nbinsr;
                    for (int thisn3=-nmax; thisn3<=nmax; thisn3++){
                        thisn = thisn2+thisn3;
                        thisGshift_mn3m3 = (nzero_Gn-thisn3-3)*nbinsr;
                        thisGshift_mn3m1 = (nzero_Gn-thisn3-1)*nbinsr;
                        thisGshift_n3m3 = (nzero_Gn+thisn3-3)*nbinsr;
                        thisGshift_n3m1 = (nzero_Gn+thisn3-1)*nbinsr;
                        thisWshift_n3 = (nzero_Wn+thisn3)*nbinsr;
                        thisGshift_mn2mn3m3 = (nzero_Gn-thisn-3)*nbinsr;
                        thisGshift_mn2mn3m1 = (nzero_Gn-thisn-1)*nbinsr;
                        thisGshift_n2n3m3 = (nzero_Gn+thisn-3)*nbinsr;
                        thisGshift_n2n3m1 = (nzero_Gn+thisn-1)*nbinsr;
                        thisWshift_n2n3 = (nzero_Wn+thisn)*nbinsr;
                        thisnshift = ((thisn2+nzero_Ups)*nnvals_Upsn + (thisn3+nzero_Ups)) * ups_nshift;
                        for (int elb1=0; elb1<nbinsr; elb1++){
                            // Triple-counting corr
                            thisnrshift = elthread*ups_compshift + thisnshift + elb1*nbinsr*nbinsr + elb1*nbinsr + elb1;
                            triplecorrA = 2 * wshape1  * nextG3ns_gggc[1*nbinsr+elb1];
                            triplecorrB = 2 * wshape1c * nextG3ns_gggc[0*nbinsr+elb1];
                            tmpUpsilon0_n[thisnrshift] +=  2 * wshape1  * nextG3ns_ggg[1*nbinsr+elb1];
                            tmpUpsilon1_n[thisnrshift] +=  2 * wshape1c * nextG3ns_ggg[0*nbinsr+elb1];
                            tmpUpsilon2_n[thisnrshift] +=  triplecorrA;
                            tmpUpsilon3_n[thisnrshift] +=  triplecorrA;
                            tmpUpsilon4_n[thisnrshift] +=  triplecorrA;
                            tmpUpsilon5_n[thisnrshift] +=  triplecorrB;
                            tmpUpsilon6_n[thisnrshift] +=  triplecorrB;
                            tmpUpsilon7_n[thisnrshift] +=  triplecorrB;
                            tmpN_n[thisnrshift] += 2 * w1*nextW3ns[elb1];

                            for (int elb2=0; elb2<nbinsr; elb2++){
                                // Double-counting corr for theta1==theta2
                                thisnrshift = elthread*ups_compshift + thisnshift + elb1*nbinsr*nbinsr + elb1*nbinsr + elb2;
                                tmpUpsilon0_n[thisnrshift] -= wshape1  *
                                    nextG2ns_gg[(nzero_G2n+thisn3-5)*nbinsr+elb1]  * nextGns[thisGshift_mn3m3+elb2];
                                tmpUpsilon1_n[thisnrshift] -= wshape1c  *
                                    nextG2ns_gg[(nzero_G2n+thisn3-3)*nbinsr+elb1]  * nextGns[thisGshift_mn3m1+elb2];
                                tmpUpsilon2_n[thisnrshift] -= wshape1  *
                                    nextG2ns_ggc[(nzero_G2n+thisn3-1)*nbinsr+elb1] * nextGns[thisGshift_mn3m3+elb2];
                                tmpUpsilon3_n[thisnrshift] -= wshape1  *
                                    nextG2ns_ggc[(nzero_G2n+thisn3-1)*nbinsr+elb1] * nextGns[thisGshift_mn3m3+elb2];
                                tmpUpsilon4_n[thisnrshift] -= wshape1  *
                                    nextG2ns_gg[(nzero_G2n+thisn3-5)*nbinsr+elb1]  * conj(nextGns[thisGshift_n3m1+elb2]);
                                tmpUpsilon5_n[thisnrshift] -= wshape1c  *
                                    nextG2ns_ggc[(nzero_G2n+thisn3+1)*nbinsr+elb1] * nextGns[thisGshift_mn3m1+elb2];
                                tmpUpsilon6_n[thisnrshift] -= wshape1c  *
                                    nextG2ns_ggc[(nzero_G2n+thisn3+1)*nbinsr+elb1] * nextGns[thisGshift_mn3m1+elb2];
                                tmpUpsilon7_n[thisnrshift] -= wshape1c  *
                                    nextG2ns_gg[(nzero_G2n+thisn3-3)*nbinsr+elb1]  * conj(nextGns[thisGshift_n3m3+elb2]);
                                tmpN_n[thisnrshift] -= w1 *
                                    nextW2ns[(nzero_Wn+thisn3)*nbinsr+elb1]*conj(nextWns[thisWshift_n3+elb2]);
                                // Double-counting corr for theta1==theta3
                                thisnrshift = elthread*ups_compshift + thisnshift + elb1*nbinsr*nbinsr + elb2*nbinsr + elb1;
                                tmpUpsilon0_n[thisnrshift] -= wshape1  *
                                    nextG2ns_gg[(nzero_G2n+thisn2-6)*nbinsr+elb1]  * nextGns[thisGshift_mn2m2+elb2];
                                tmpUpsilon1_n[thisnrshift] -= wshape1c  *
                                    nextG2ns_gg[(nzero_G2n+thisn2-2)*nbinsr+elb1]  * nextGns[thisGshift_mn2m2+elb2];
                                tmpUpsilon2_n[thisnrshift] -= wshape1  *
                                   nextG2ns_ggc[(nzero_G2n+thisn2-2)*nbinsr+elb1] * nextGns[thisGshift_mn2m2+elb2];
                                tmpUpsilon3_n[thisnrshift] -= wshape1  *
                                    nextG2ns_gg[(nzero_G2n+thisn2-6)*nbinsr+elb1]  * conj(nextGns[thisGshift_n2m2+elb2]);
                                tmpUpsilon4_n[thisnrshift] -= wshape1  *
                                    nextG2ns_ggc[(nzero_G2n+thisn2-2)*nbinsr+elb1] * nextGns[thisGshift_mn2m2+elb2];
                                tmpUpsilon5_n[thisnrshift] -= wshape1c  *
                                    nextG2ns_ggc[(nzero_G2n+thisn2+2)*nbinsr+elb1] * nextGns[thisGshift_mn2m2+elb2];
                                tmpUpsilon6_n[thisnrshift] -= wshape1c  *
                                    nextG2ns_gg[(nzero_G2n+thisn2-2)*nbinsr+elb1]  * conj(nextGns[thisGshift_n2m2+elb2]);
                                tmpUpsilon7_n[thisnrshift] -= wshape1c  *
                                    nextG2ns_ggc[(nzero_G2n+thisn2+2)*nbinsr+elb1] * nextGns[thisGshift_mn2m2+elb2];
                                tmpN_n[thisnrshift] -= w1 * 
                                    nextW2ns[(nzero_Wn+thisn2)*nbinsr+elb1] * conj(nextWns[thisWshift_n2+elb2]);
                                // Double-counting corr for theta2==theta3
                                thisnrshift = elthread*ups_compshift + thisnshift + elb1*nbinsr*nbinsr + elb2*nbinsr + elb2;
                                tmpUpsilon0_n[thisnrshift] -= wshape1  *
                                    nextG2ns_gg[(nzero_G2n-thisn2-thisn3-5)*nbinsr+elb2]  * nextGns[thisGshift_n2n3m3+elb1];
                                tmpUpsilon1_n[thisnrshift] -= wshape1c  *
                                    nextG2ns_gg[(nzero_G2n-thisn2-thisn3-3)*nbinsr+elb2]  * nextGns[thisGshift_n2n3m1+elb1];
                                tmpUpsilon2_n[thisnrshift] -= wshape1  *
                                    nextG2ns_gg[(nzero_G2n-thisn2-thisn3-5)*nbinsr+elb2]  * conj(nextGns[thisGshift_mn2mn3m1+elb1]);
                                tmpUpsilon3_n[thisnrshift] -= wshape1  *
                                    nextG2ns_ggc[(nzero_G2n-thisn2-thisn3-1)*nbinsr+elb2] * nextGns[thisGshift_n2n3m3+elb1];
                                tmpUpsilon4_n[thisnrshift] -= wshape1  *
                                    nextG2ns_ggc[(nzero_G2n-thisn2-thisn3-1)*nbinsr+elb2] * nextGns[thisGshift_n2n3m3+elb1];
                                tmpUpsilon5_n[thisnrshift] -= wshape1c  *
                                    nextG2ns_gg[(nzero_G2n-thisn2-thisn3-3)*nbinsr+elb2]  * conj(nextGns[thisGshift_mn2mn3m3+elb1]);
                                tmpUpsilon6_n[thisnrshift] -= wshape1c  *
                                    nextG2ns_ggc[(nzero_G2n-thisn2-thisn3+1)*nbinsr+elb2] * nextGns[thisGshift_n2n3m1+elb1];
                                tmpUpsilon7_n[thisnrshift] -= wshape1c  *
                                    nextG2ns_ggc[(nzero_G2n-thisn2-thisn3+1)*nbinsr+elb2] * nextGns[thisGshift_n2n3m1+elb1];
                                tmpN_n[thisnrshift] -= w1 * 
                                    nextW2ns[(nzero_Wn-thisn2-thisn3)*nbinsr+elb2] * nextWns[thisWshift_n2n3+elb1];

                                // Allocation of first three complex products for Upsilon/Norm updates
                                gGG0 = wshape1*nextGns[thisGshift_n2n3m3+elb1]*nextGns[thisGshift_mn2m2+elb2];
                                gGG1 = wshape1c*nextGns[thisGshift_n2n3m1+elb1]*nextGns[thisGshift_mn2m2+elb2];
                                gGG2 = wshape1*conj(nextGns[thisGshift_mn2mn3m1+elb1])*nextGns[thisGshift_mn2m2+elb2];
                                gGG3 = wshape1*nextGns[thisGshift_n2n3m3+elb1]*conj(nextGns[thisGshift_n2m2+elb2]);
                                gGG4 = wshape1*nextGns[thisGshift_n2n3m3+elb1]*nextGns[thisGshift_mn2m2+elb2];
                                gGG5 = wshape1c*conj(nextGns[thisGshift_mn2mn3m3+elb1])*nextGns[thisGshift_mn2m2+elb2];
                                gGG6 = wshape1c*nextGns[thisGshift_n2n3m1+elb1]*conj(nextGns[thisGshift_n2m2+elb2]);
                                gGG7 = wshape1c*nextGns[thisGshift_n2n3m1+elb1]*nextGns[thisGshift_mn2m2+elb2];
                                wNN = w1*nextWns[thisWshift_n2n3+elb1]*conj(nextWns[thisWshift_n2+elb2]);
                                for (int elb3=0; elb3<nbinsr; elb3++){
                                    thisnrshift = elthread*ups_compshift + thisnshift + elb1*nbinsr*nbinsr + elb2*nbinsr + elb3;
                                    // Allocation of Upsilon and Norm
                                    tmpUpsilon0_n[thisnrshift] += gGG0*nextGns[thisGshift_mn3m3+elb3];
                                    tmpUpsilon1_n[thisnrshift] += gGG1*nextGns[thisGshift_mn3m1+elb3];
                                    tmpUpsilon2_n[thisnrshift] += gGG2*nextGns[thisGshift_mn3m3+elb3];
                                    tmpUpsilon3_n[thisnrshift] += gGG3*nextGns[thisGshift_mn3m3+elb3];
                                    tmpUpsilon4_n[thisnrshift] += gGG4*conj(nextGns[thisGshift_n3m1+elb3]);
                                    tmpUpsilon5_n[thisnrshift] += gGG5*nextGns[thisGshift_mn3m1+elb3];
                                    tmpUpsilon6_n[thisnrshift] += gGG6*nextGns[thisGshift_mn3m1+elb3];
                                    tmpUpsilon7_n[thisnrshift] += gGG7*conj(nextGns[thisGshift_n3m3+elb3]);
                                    tmpN_n[thisnrshift] += wNN*conj(nextWns[thisWshift_n3+elb3]);
                                }
                            }
                        }
                    }
                }
            }
        }

        free(nextGns);
        free(nextG2ns_gg);
        free(nextG2ns_ggc);
        free(nextG3ns_ggg);
        free(nextG3ns_gggc);
        free(nextWns);
        free(nextW2ns);
        free(nextW3ns);
    }

    // Accumulate Upsilon_n and N_n
    #pragma omp parallel for num_threads(nthreads)
    for (int thisn=0; thisn<_n2n3combis; thisn++){
        int thisnshift = thisn*_ups_nshift;
        for (int elb1=0; elb1<nbinsr; elb1++){
            for (int elb2=0; elb2<nbinsr; elb2++){
                for (int elb3=0; elb3<nbinsr; elb3++){
                    for (int elthread=0; elthread<nthreads; elthread++){
                        int thisnrshift = thisnshift + elb1*nbinsr*nbinsr + elb2*nbinsr + elb3;
                        int thistmpnshift = elthread*_ups_compshift+thisnrshift;
                        Upsilon_n[0*_ups_compshift+thisnrshift] += tmpUpsilon0_n[thistmpnshift];
                        Upsilon_n[1*_ups_compshift+thisnrshift] += tmpUpsilon1_n[thistmpnshift];
                        Upsilon_n[2*_ups_compshift+thisnrshift] += tmpUpsilon2_n[thistmpnshift];
                        Upsilon_n[3*_ups_compshift+thisnrshift] += tmpUpsilon3_n[thistmpnshift];
                        Upsilon_n[4*_ups_compshift+thisnrshift] += tmpUpsilon4_n[thistmpnshift];
                        Upsilon_n[5*_ups_compshift+thisnrshift] += tmpUpsilon5_n[thistmpnshift];
                        Upsilon_n[6*_ups_compshift+thisnrshift] += tmpUpsilon6_n[thistmpnshift];
                        Upsilon_n[7*_ups_compshift+thisnrshift] += tmpUpsilon7_n[thistmpnshift];
                        N_n[thisnrshift] += tmpN_n[thistmpnshift];
                    }
                }
            }
        }
    }

    // Accumulate the bin distances and weights
    for (int thisthread=0; thisthread<nthreads; thisthread++){
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            totcounts[elbinr] += tmpwcounts[thisthread*nbinsr+elbinr];
            totnorms[elbinr] += tmpwnorms[thisthread*nbinsr+elbinr];
        }
    }
    // Get bin centers
    for (int elbinr=0; elbinr<nbinsr; elbinr++){
        if (totnorms[elbinr] != 0){
            bin_centers[elbinr] = totcounts[elbinr]/totnorms[elbinr];
        }
    }
    
    free(tmpUpsilon0_n);
    free(tmpUpsilon1_n);
    free(tmpUpsilon2_n);
    free(tmpUpsilon3_n);
    free(tmpUpsilon4_n);
    free(tmpUpsilon5_n);
    free(tmpUpsilon6_n);
    free(tmpUpsilon7_n);
    free(tmpN_n);
    free(tmpwcounts);
    free(tmpwnorms);
    free(totcounts);
    free(totnorms);
    if (verbose>0){ printf("\n"); }
}

// Non-tomo 4pcf using tree-based estimator
void alloc_notomoGammans_tree_gggg(const MultiresoCatalog *cat_base, const MultiresoCatalog *cat_leaf,
                                   const NavHash *nav, const TreeResoParams *tree,
                                   const BinningParams *bin, const FourthParams *fourth,
                                   int nthreads, int verbose, NPCFOutput *out){
    // Dereference passed structures
    double *isinner = cat_base->isinner_resos, *weight = cat_base->weight_resos;
    double *pos1 = cat_base->pos1_resos, *pos2 = cat_base->pos2_resos;
    double *e1 = cat_base->e1_resos, *e2 = cat_base->e2_resos;
    int nresos = tree->nresos; double *reso_redges = tree->reso_redges;
    int *ngal_resos = cat_leaf->ngal_resos;
    double *weight_resos = cat_leaf->weight_resos;
    double *pos1_resos = cat_leaf->pos1_resos, *pos2_resos = cat_leaf->pos2_resos;
    double *e1_resos = cat_leaf->e1_resos, *e2_resos = cat_leaf->e2_resos;
    int *index_matcher_hash = nav->index_matcher, *pixs_galind_bounds = nav->pixs_galind_bounds, *pix_gals = nav->pix_gals;
    int nregions = nav->nregions;
    double pix1_start = nav->pix1_start, pix1_d = nav->pix1_d; int pix1_n = nav->pix1_n;
    double pix2_start = nav->pix2_start, pix2_d = nav->pix2_d; int pix2_n = nav->pix2_n;
    int nmax = bin->nmax, nbinsr = bin->nbinsr;
    double rmin = bin->rmin, rmax = bin->rmax;
    int nthetacombis = fourth->nthetacombis;
    int *nindices = fourth->nindices, len_nindices = fourth->len_nindices;
    double *bin_centers = out->bin_centers;
    double complex *Upsilon_n = out->npcf, *N_n = out->norm_mp;
    
    // Temporary arrays that are allocated in parallel and later reduced
    double *tmpwcounts = orpheus_calloc(nthreads*nbinsr, sizeof(double));
    double *tmpwnorms = orpheus_calloc(nthreads*nbinsr, sizeof(double));
    double complex *tmpUpsilon0_n = orpheus_calloc(nthreads*len_nindices*nthetacombis, sizeof(double complex));
    double complex *tmpUpsilon1_n = orpheus_calloc(nthreads*len_nindices*nthetacombis, sizeof(double complex));
    double complex *tmpUpsilon2_n = orpheus_calloc(nthreads*len_nindices*nthetacombis, sizeof(double complex));
    double complex *tmpUpsilon3_n = orpheus_calloc(nthreads*len_nindices*nthetacombis, sizeof(double complex));
    double complex *tmpUpsilon4_n = orpheus_calloc(nthreads*len_nindices*nthetacombis, sizeof(double complex));
    double complex *tmpUpsilon5_n = orpheus_calloc(nthreads*len_nindices*nthetacombis, sizeof(double complex));
    double complex *tmpUpsilon6_n = orpheus_calloc(nthreads*len_nindices*nthetacombis, sizeof(double complex));
    double complex *tmpUpsilon7_n = orpheus_calloc(nthreads*len_nindices*nthetacombis, sizeof(double complex));
    double complex *tmpN_n = orpheus_calloc(nthreads*len_nindices*nthetacombis, sizeof(double complex));
    
    double *totcounts = orpheus_calloc(nbinsr, sizeof(double));
    double *totnorms = orpheus_calloc(nbinsr, sizeof(double));

    // Helper array that checks how many regions have been already computed
    int *regionsdone = orpheus_calloc(nregions, sizeof(int));
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){
        free(tmpwcounts); free(tmpwnorms); free(tmpUpsilon0_n); free(tmpUpsilon1_n);
        free(tmpUpsilon2_n); free(tmpUpsilon3_n); free(tmpUpsilon4_n); free(tmpUpsilon5_n);
        free(tmpUpsilon6_n); free(tmpUpsilon7_n); free(tmpN_n); free(totcounts); free(totnorms);
        free(regionsdone);
        return;
    }
    int nregionsdone = 0;
    reset_progress();
    
    #pragma omp parallel for num_threads(nthreads)
    for (int elthread=0; elthread<nthreads; elthread++){
        int nregions_per_thread = nregions/nthreads;
        int nmax_alloc = 2*nmax+1;
        int nbinsz = 1;
        int nnvals_Gn = 4*nmax_alloc+3; // Need to cover [-n1-n2-3, n1+n2-1]
        int nnvals_G2n = 4*nmax_alloc+7; // Need to cover [-n1-n2-5, n1+n2+1]
        int nnvals_Wn = 4*nmax_alloc+1; // Need to cover [-n1-n2, n1+n2]
        int nnvals_Upsn = 2*nmax_alloc+1; // Need tocover [-nmax,+nmax]
        int nzero_Gn = 2*nmax_alloc+3;
        int nzero_G2n = 2*nmax_alloc+5;
        int nzero_Wn = 2*nmax_alloc;
        int nzero_Ups = nmax_alloc;
        int ups_compshift = len_nindices*nthetacombis;

        int nbinszr = nbinsz*nbinsr;
        double complex *nextGns =  orpheus_calloc(nnvals_Gn*nbinszr, sizeof(double complex));
        double complex *nextG2ns_gg =  orpheus_calloc(nnvals_G2n*nbinszr, sizeof(double complex));
        double complex *nextG2ns_ggc =  orpheus_calloc(nnvals_G2n*nbinszr, sizeof(double complex));
        double complex *nextG3ns_ggg = orpheus_calloc(2*nbinszr, sizeof(double complex));
        double complex *nextG3ns_gggc = orpheus_calloc(2*nbinszr, sizeof(double complex));
        double complex *nextWns = orpheus_calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW2ns = orpheus_calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW3ns = orpheus_calloc(nbinszr, sizeof(double complex));

        int npix_hash = pix1_n*pix2_n;
        int *rshift_index_matcher_hash = orpheus_calloc(nresos, sizeof(int));
        int *rshift_pixs_galind_bounds = orpheus_calloc(nresos, sizeof(int));
        int *rshift_pix_gals = orpheus_calloc(nresos, sizeof(int));
        build_rshift_offsets(nresos, npix_hash, ngal_resos,
                             rshift_index_matcher_hash, rshift_pixs_galind_bounds, rshift_pix_gals);

        double drbin = (log(rmax)-log(rmin))/(nbinsr);
        
        for (int _elregion=0; _elregion<2*nregions; _elregion++){

            // Check if this thread needs to allocate the region. In the first pass we split the work evenly 
            // while in the second pass we just work on the next best region, s.t. the 'fast' threads will
            // steal work from the 'slow' threads.
            int wasdone = 0;
            if (_elregion<nregions){
                int nthread_target = mymin(_elregion/nregions_per_thread, nthreads-1);
                if (nthread_target!=elthread){continue;}
            }
            int elregion = _elregion%nregions;
            #pragma omp critical
            {   
                if (regionsdone[_elregion%nregions]==1){wasdone = 1;}
                else{
                    regionsdone[_elregion%nregions]=1;
                    nregionsdone+=1; 
                }
            }
            if (wasdone==1){continue;}
            int region_debug = mymin(500,nregions-1);
            bool printregdbg = (verbose>1) && (elregion==region_debug);
            if (printregdbg){printf("Region %d is in thread %d (%i regions in total)\n",
                elregion,elthread,nregions);}
            
            int lower1 = pixs_galind_bounds[elregion];
            int upper1 = pixs_galind_bounds[elregion+1];
            for (int ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                int ind_gal = pix_gals[ind_inpix1];
                double p11, p12, w1, e11, e12;
                double innergal = isinner[ind_gal];
                if (innergal<1e-5){continue;}
                p11 = pos1[ind_gal];
                p12 = pos2[ind_gal];
                w1 = innergal*weight[ind_gal];
                e11 = e1[ind_gal];
                e12 = e2[ind_gal];     
                
                // Gns have shape (nnvals, nbinsz, nbinsr)
                // where the ns are ordered as 
                // [-nmax_1-nmax_2-3, ..., nmax_1+nmax_2+3]
                for (int i=0; i<nnvals_Gn*nbinszr; i++){nextGns[i]=0;}
                for (int i=0; i<nnvals_G2n*nbinszr; i++){nextG2ns_gg[i]=0;nextG2ns_ggc[i]=0;}
                for (int i=0; i<2*nbinszr; i++){nextG3ns_ggg[i]=0;nextG3ns_gggc[i]=0;}
                for (int i=0; i<nnvals_Wn*nbinszr; i++){nextWns[i]=0;nextW2ns[i]=0;}
                for (int i=0; i<nbinszr; i++){nextW3ns[i]=0;}
                
                int ind_gal2;
                int lower, upper;
                double  p21, p22, w2, w2_sq, e21, e22, rel1, rel2, dist2, dist, dphi;
                double complex wshape1, wshape1c, wshape2, wshape_sq, wshape_cube, wshapewshapec, wshapesqwshapec;
                double complex phirot, phirotc, twophirotc, fourphirotc;
                // Allocate Gn, Wn and their multiple-couting corrections
                for (int elreso=0; elreso<nresos; elreso++){
                    int rbin, zrshift, ind_Gn, ind_G2n, ind_Wn;
                    double rmin_reso = reso_redges[elreso];
                    double rmin_reso2 = rmin_reso*rmin_reso;
                    double rmax_reso = reso_redges[elreso+1];
                    double rmax_reso2 = rmax_reso*rmax_reso;
                    FLATCELL_FOREACH(
                        index_matcher_hash, rshift_index_matcher_hash[elreso], pixs_galind_bounds, rshift_pixs_galind_bounds[elreso],
                        p11, p12, rmax_reso, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, lower, upper){
                        for (int ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                            ind_gal2 = rshift_pix_gals[elreso] + pix_gals[rshift_pix_gals[elreso]+ind_inpix];
                            //#pragma omp critical
                            p21 = pos1_resos[ind_gal2];
                            p22 = pos2_resos[ind_gal2];
                            w2 = weight_resos[ind_gal2];
                            e21 = e1_resos[ind_gal2];
                            e22 = e2_resos[ind_gal2];
                            
                            rel1 = p21 - p11;
                            rel2 = p22 - p12;
                            dist2 = rel1*rel1 + rel2*rel2;
                            if (dist2 < rmin_reso2 || dist2 >= rmax_reso2){continue;}
                            dist = sqrt(dist2);
                            rbin = (int) floor((log(dist)-log(rmin))/drbin);
                            w2_sq = w2*w2;
                            wshape2 = (double complex) w2 * (e21+I*e22);
                            wshape_sq = wshape2*wshape2;
                            wshape_cube = wshape_sq*wshape2;
                            wshapewshapec = wshape2*conj(wshape2);
                            wshapesqwshapec = wshape_sq*conj(wshape2);
                            dphi = atan2(rel2,rel1);
                            phirot = cexp(I*dphi);
                            phirotc = conj(phirot);
                            twophirotc = phirotc*phirotc;
                            fourphirotc = twophirotc*twophirotc;
                            zrshift = 0*nbinsr + rbin;
                            ind_Gn = nzero_Gn*nbinszr + zrshift;
                            ind_G2n = nzero_G2n*nbinszr + zrshift;
                            ind_Wn = nzero_Wn*nbinszr + zrshift;
                            totcounts[zrshift] += w1*w2*dist; 
                            totnorms[zrshift] += w1*w2; 
                            gggg_fill_gnwn(nextGns, nextG2ns_gg, nextG2ns_ggc,
                                            nextWns, nextW2ns, nextW3ns, nextG3ns_ggg, nextG3ns_gggc,
                                            2*nmax_alloc, nbinszr, zrshift, ind_Gn, ind_G2n, ind_Wn,
                                            w2, w2_sq, wshape2, wshape_sq, wshape_cube,
                                            wshapewshapec, wshapesqwshapec, phirot, phirotc, fourphirotc);
                        }
                    }
                }
                
                // Allocate Upsilon; have shape (nindices, rcombis)
                double complex gGG0, gGG1, gGG2, gGG3, gGG4, gGG5, gGG6, gGG7, wNN;
                int thisn2, thisn3, thisn, thisnshift, thisnrshift, elbcombi;
                int thisGshift_mn2m2, thisGshift_n2m2, thisWshift_n2;
                int thisGshift_mn3m3, thisGshift_mn3m1, thisGshift_n3m3, thisGshift_n3m1, thisWshift_n3;
                int thisGshift_mn2mn3m3, thisGshift_mn2mn3m1, thisGshift_n2n3m3, thisGshift_n2n3m1, thisWshift_n2n3;
                double complex triplecorrA, triplecorrB;
                wshape1 = w1 * (e11+I*e12);  
                wshape1c = conj(wshape1);
                for (int nindex=0; nindex<len_nindices; nindex++){
                    thisn2 = nindices[nindex]/nnvals_Upsn - nzero_Ups;
                    thisn3 = nindices[nindex]%nnvals_Upsn - nzero_Ups;
                    if (thisn2>nzero_Ups || -thisn2>nzero_Ups || thisn3>nzero_Ups || -thisn3>nzero_Ups){
                        if (elregion==0){
                            printf("Error at elregion=%d nindex=%d: nindices[nindex]=%d n2=%d n3=%d",
                                   elregion, nindex, nindices[nindex], thisn2, thisn3);}
                        continue;
                    }
                    thisn = thisn2+thisn3;
                    thisGshift_mn2m2 = (nzero_Gn-thisn2-2)*nbinsr;
                    thisGshift_n2m2 = (nzero_Gn+thisn2-2)*nbinsr;
                    thisWshift_n2 = (nzero_Wn+thisn2)*nbinsr;
                    thisGshift_mn3m3 = (nzero_Gn-thisn3-3)*nbinsr;
                    thisGshift_mn3m1 = (nzero_Gn-thisn3-1)*nbinsr;
                    thisGshift_n3m3 = (nzero_Gn+thisn3-3)*nbinsr;
                    thisGshift_n3m1 = (nzero_Gn+thisn3-1)*nbinsr;
                    thisWshift_n3 = (nzero_Wn+thisn3)*nbinsr;
                    thisGshift_mn2mn3m3 = (nzero_Gn-thisn-3)*nbinsr;
                    thisGshift_mn2mn3m1 = (nzero_Gn-thisn-1)*nbinsr;
                    thisGshift_n2n3m3 = (nzero_Gn+thisn-3)*nbinsr;
                    thisGshift_n2n3m1 = (nzero_Gn+thisn-1)*nbinsr;
                    thisWshift_n2n3 = (nzero_Wn+thisn)*nbinsr;
                    thisnshift = nindex * nthetacombis;
                    elbcombi = 0;
                    for (int elb1=0; elb1<nbinsr; elb1++){
                        thisnrshift = elthread*ups_compshift + thisnshift + elbcombi;
                        // Triple-counting corr
                        triplecorrA = 2 * wshape1  * nextG3ns_gggc[1*nbinsr+elb1];
                        triplecorrB = 2 * wshape1c * nextG3ns_gggc[0*nbinsr+elb1];
                        tmpUpsilon0_n[thisnrshift] +=  2 * wshape1  * nextG3ns_ggg[1*nbinsr+elb1];
                        tmpUpsilon1_n[thisnrshift] +=  2 * wshape1c * nextG3ns_ggg[0*nbinsr+elb1];
                        tmpUpsilon2_n[thisnrshift] +=  triplecorrA;
                        tmpUpsilon3_n[thisnrshift] +=  triplecorrA;
                        tmpUpsilon4_n[thisnrshift] +=  triplecorrA;
                        tmpUpsilon5_n[thisnrshift] +=  triplecorrB;
                        tmpUpsilon6_n[thisnrshift] +=  triplecorrB;
                        tmpUpsilon7_n[thisnrshift] +=  triplecorrB;
                        tmpN_n[thisnrshift] += 2 * w1*nextW3ns[elb1];

                        for (int elb2=elb1; elb2<nbinsr; elb2++){
                            // Allocation of first three complex products for Upsilon/Norm updates
                            gGG0 = wshape1*nextGns[thisGshift_n2n3m3+elb1]*nextGns[thisGshift_mn2m2+elb2];
                            gGG1 = wshape1c*nextGns[thisGshift_n2n3m1+elb1]*nextGns[thisGshift_mn2m2+elb2];
                            gGG2 = wshape1*conj(nextGns[thisGshift_mn2mn3m1+elb1])*nextGns[thisGshift_mn2m2+elb2];
                            gGG3 = wshape1*nextGns[thisGshift_n2n3m3+elb1]*conj(nextGns[thisGshift_n2m2+elb2]);
                            gGG4 = wshape1*nextGns[thisGshift_n2n3m3+elb1]*nextGns[thisGshift_mn2m2+elb2];
                            gGG5 = wshape1c*conj(nextGns[thisGshift_mn2mn3m3+elb1])*nextGns[thisGshift_mn2m2+elb2];
                            gGG6 = wshape1c*nextGns[thisGshift_n2n3m1+elb1]*conj(nextGns[thisGshift_n2m2+elb2]);
                            gGG7 = wshape1c*nextGns[thisGshift_n2n3m1+elb1]*nextGns[thisGshift_mn2m2+elb2];
                            wNN = w1*nextWns[thisWshift_n2n3+elb1]*conj(nextWns[thisWshift_n2+elb2]);
                            
                            for (int elb3=elb2; elb3<nbinsr; elb3++){
                                thisnrshift = elthread*ups_compshift + thisnshift + elbcombi;
                                // Double-counting corr for theta1==theta2
                                if (elb1==elb2){
                                    tmpUpsilon0_n[thisnrshift] -= wshape1  *
                                        nextG2ns_gg[(nzero_G2n+thisn3-5)*nbinsr+elb1]  * nextGns[thisGshift_mn3m3+elb3];
                                    tmpUpsilon1_n[thisnrshift] -= wshape1c  *
                                        nextG2ns_gg[(nzero_G2n+thisn3-3)*nbinsr+elb1]  * nextGns[thisGshift_mn3m1+elb3];
                                    tmpUpsilon2_n[thisnrshift] -= wshape1  *
                                        nextG2ns_ggc[(nzero_G2n+thisn3-1)*nbinsr+elb1] * nextGns[thisGshift_mn3m3+elb3];
                                    tmpUpsilon3_n[thisnrshift] -= wshape1  *
                                        nextG2ns_ggc[(nzero_G2n+thisn3-1)*nbinsr+elb1] * nextGns[thisGshift_mn3m3+elb3];
                                    tmpUpsilon4_n[thisnrshift] -= wshape1  *
                                        nextG2ns_gg[(nzero_G2n+thisn3-5)*nbinsr+elb1]  * conj(nextGns[thisGshift_n3m1+elb3]);
                                    tmpUpsilon5_n[thisnrshift] -= wshape1c  *
                                        nextG2ns_ggc[(nzero_G2n+thisn3+1)*nbinsr+elb1] * nextGns[thisGshift_mn3m1+elb3];
                                    tmpUpsilon6_n[thisnrshift] -= wshape1c  *
                                        nextG2ns_ggc[(nzero_G2n+thisn3+1)*nbinsr+elb1] * nextGns[thisGshift_mn3m1+elb3];
                                    tmpUpsilon7_n[thisnrshift] -= wshape1c  *
                                        nextG2ns_gg[(nzero_G2n+thisn3-3)*nbinsr+elb1]  * conj(nextGns[thisGshift_n3m3+elb3]);
                                    tmpN_n[thisnrshift] -= w1 *
                                        nextW2ns[(nzero_Wn+thisn3)*nbinsr+elb1]*conj(nextWns[thisWshift_n3+elb3]);
                                }
                                // Double-counting corr for theta1==theta3
                                if (elb1==elb3){
                                    tmpUpsilon0_n[thisnrshift] -= wshape1  *
                                        nextG2ns_gg[(nzero_G2n+thisn2-6)*nbinsr+elb1]  * nextGns[thisGshift_mn2m2+elb2];
                                    tmpUpsilon1_n[thisnrshift] -= wshape1c  *
                                        nextG2ns_gg[(nzero_G2n+thisn2-2)*nbinsr+elb1]  * nextGns[thisGshift_mn2m2+elb2];
                                    tmpUpsilon2_n[thisnrshift] -= wshape1  *
                                        nextG2ns_ggc[(nzero_G2n+thisn2-2)*nbinsr+elb1] * nextGns[thisGshift_mn2m2+elb2];
                                    tmpUpsilon3_n[thisnrshift] -= wshape1  *
                                        nextG2ns_gg[(nzero_G2n+thisn2-6)*nbinsr+elb1]  * conj(nextGns[thisGshift_n2m2+elb2]);
                                    tmpUpsilon4_n[thisnrshift] -= wshape1  *
                                        nextG2ns_ggc[(nzero_G2n+thisn2-2)*nbinsr+elb1] * nextGns[thisGshift_mn2m2+elb2];
                                    tmpUpsilon5_n[thisnrshift] -= wshape1c  *
                                        nextG2ns_ggc[(nzero_G2n+thisn2+2)*nbinsr+elb1] * nextGns[thisGshift_mn2m2+elb2];
                                    tmpUpsilon6_n[thisnrshift] -= wshape1c  *
                                        nextG2ns_gg[(nzero_G2n+thisn2-2)*nbinsr+elb1]  * conj(nextGns[thisGshift_n2m2+elb2]);
                                    tmpUpsilon7_n[thisnrshift] -= wshape1c  *
                                        nextG2ns_ggc[(nzero_G2n+thisn2+2)*nbinsr+elb1] * nextGns[thisGshift_mn2m2+elb2];
                                    tmpN_n[thisnrshift] -= w1 * 
                                        nextW2ns[(nzero_Wn+thisn2)*nbinsr+elb1] * conj(nextWns[thisWshift_n2+elb2]);
                                }
                                // Double-counting corr for theta2==theta3
                                if (elb2==elb3){
                                    tmpUpsilon0_n[thisnrshift] -= wshape1  *
                                        nextG2ns_gg[(nzero_G2n-thisn2-thisn3-5)*nbinsr+elb2]  * nextGns[thisGshift_n2n3m3+elb1];
                                    tmpUpsilon1_n[thisnrshift] -= wshape1c  *
                                        nextG2ns_gg[(nzero_G2n-thisn2-thisn3-3)*nbinsr+elb2]  * nextGns[thisGshift_n2n3m1+elb1];
                                    tmpUpsilon2_n[thisnrshift] -= wshape1  *
                                        nextG2ns_gg[(nzero_G2n-thisn2-thisn3-5)*nbinsr+elb2]  * conj(nextGns[thisGshift_mn2mn3m1+elb1]);
                                    tmpUpsilon3_n[thisnrshift] -= wshape1  *
                                        nextG2ns_ggc[(nzero_G2n-thisn2-thisn3-1)*nbinsr+elb2] * nextGns[thisGshift_n2n3m3+elb1];
                                    tmpUpsilon4_n[thisnrshift] -= wshape1  *
                                        nextG2ns_ggc[(nzero_G2n-thisn2-thisn3-1)*nbinsr+elb2] * nextGns[thisGshift_n2n3m3+elb1];
                                    tmpUpsilon5_n[thisnrshift] -= wshape1c  *
                                        nextG2ns_gg[(nzero_G2n-thisn2-thisn3-3)*nbinsr+elb2]  * conj(nextGns[thisGshift_mn2mn3m3+elb1]);
                                    tmpUpsilon6_n[thisnrshift] -= wshape1c  *
                                        nextG2ns_ggc[(nzero_G2n-thisn2-thisn3+1)*nbinsr+elb2] * nextGns[thisGshift_n2n3m1+elb1];
                                    tmpUpsilon7_n[thisnrshift] -= wshape1c  *
                                        nextG2ns_ggc[(nzero_G2n-thisn2-thisn3+1)*nbinsr+elb2] * nextGns[thisGshift_n2n3m1+elb1];
                                    tmpN_n[thisnrshift] -= w1 * 
                                        nextW2ns[(nzero_Wn-thisn2-thisn3)*nbinsr+elb2] * nextWns[thisWshift_n2n3+elb1];
                                }

                                // Nominal allocation of Upsilon and Norm
                                tmpUpsilon0_n[thisnrshift] += gGG0*nextGns[thisGshift_mn3m3+elb3];
                                tmpUpsilon1_n[thisnrshift] += gGG1*nextGns[thisGshift_mn3m1+elb3];
                                tmpUpsilon2_n[thisnrshift] += gGG2*nextGns[thisGshift_mn3m3+elb3];
                                tmpUpsilon3_n[thisnrshift] += gGG3*nextGns[thisGshift_mn3m3+elb3];
                                tmpUpsilon4_n[thisnrshift] += gGG4*conj(nextGns[thisGshift_n3m1+elb3]);
                                tmpUpsilon5_n[thisnrshift] += gGG5*nextGns[thisGshift_mn3m1+elb3];
                                tmpUpsilon6_n[thisnrshift] += gGG6*nextGns[thisGshift_mn3m1+elb3];
                                tmpUpsilon7_n[thisnrshift] += gGG7*conj(nextGns[thisGshift_n3m3+elb3]);
                                tmpN_n[thisnrshift] += wNN*conj(nextWns[thisWshift_n3+elb3]);

                                elbcombi += 1;
                            }
                        }
                    }
                }
            }
            print_progress(nregionsdone, nregions, verbose);
        }

        free(nextGns);
        free(nextG2ns_gg);
        free(nextG2ns_ggc);
        free(nextG3ns_ggg);
        free(nextG3ns_gggc);
        free(nextWns);
        free(nextW2ns);
        free(nextW3ns);

        free(rshift_index_matcher_hash);
        free(rshift_pixs_galind_bounds);
        free(rshift_pix_gals);
    }

    /*
    // DBG make sure that we allocated all bin elements
    int totlen=0;
    int nonempty=0;
    for (int elthread=0;elthread<nthreads;elthread++){
        for (int nindex=0;nindex<len_nindices;nindex++){
            for (int elb=0;elb<nthetacombis;elb++){
                int thisnrshift = elthread*len_nindices*nthetacombis + nindex*nthetacombis + elb;
                //if (tmpN_n[thisnrshift] != 0){nonempty += 1;}
                totlen += 1;
            }
        }
    }
    printf('\n We allocated %d / %d entries of Nn in first round\n',nonempty,totlen);
    */

    // Accumulate Upsilon_n and N_n
    // 1) Build arrays that hold bin combis for b1<=b2<=b3
    // 2) Get bin edges and bin centers of the combinations
    // 3) Find all (theta1,theta2,theta3) combis that can be reconstructed via the symmetries
    // 4) Get the Gamma_mu(theta1,theta2,theta3,phi12,phi13)

    // 1)
    int elbcombi = 0;
    int *elb1_inds = orpheus_calloc(nthetacombis, sizeof(int));
    int *elb2_inds = orpheus_calloc(nthetacombis, sizeof(int));
    int *elb3_inds = orpheus_calloc(nthetacombis, sizeof(int));
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){
        free(tmpwcounts); free(tmpwnorms); free(tmpUpsilon0_n); free(tmpUpsilon1_n);
        free(tmpUpsilon2_n); free(tmpUpsilon3_n); free(tmpUpsilon4_n); free(tmpUpsilon5_n);
        free(tmpUpsilon6_n); free(tmpUpsilon7_n); free(tmpN_n); free(totcounts); free(totnorms);
        free(regionsdone); free(elb1_inds); free(elb2_inds); free(elb3_inds);
        return;
    }
    for (int elb1=0; elb1<nbinsr; elb1++){
        for (int elb2=elb1; elb2<nbinsr; elb2++){
            for (int elb3=elb2; elb3<nbinsr; elb3++){
                elb1_inds[elbcombi] = elb1;
                elb2_inds[elbcombi] = elb2;
                elb3_inds[elbcombi] = elb3;
                elbcombi += 1;
            }
        }
    }

    #pragma omp parallel for num_threads(nthreads)
    for (int elb=0; elb<nthetacombis; elb++){

        int ntrafos, tnrshift, nbshift, elb1, elb2, elb3, elb1t, elb2t, elb3t;
        int thisn2, thisn3;
        int nmax_alloc = 2*nmax+1;
        int nnvals_Upsn_rec = 2*nmax+1; 
        int nnvals_Upsn = 2*nmax_alloc+1; 
        int nzero_Ups = nmax_alloc;
        int ups_nshift = nbinsr*nbinsr*nbinsr;
        int n2n3combis = nnvals_Upsn*nnvals_Upsn;
        int n2n3combis_rec = nnvals_Upsn_rec*nnvals_Upsn_rec;
        int ups_rec_compshift = n2n3combis_rec*ups_nshift;

        double complex *thisUpsilon_n = orpheus_calloc(8*n2n3combis, sizeof(double complex));
        double complex *thisN_n = orpheus_calloc(n2n3combis, sizeof(double complex));
        double complex *thisUpsilon_n_rec = orpheus_calloc(8*n2n3combis_rec, sizeof(double complex));
        double complex *thisN_n_rec = orpheus_calloc(n2n3combis_rec, sizeof(double complex));

        // 2)
        elb1 = elb1_inds[elb];
        elb2 = elb2_inds[elb];
        elb3 = elb3_inds[elb];
        int bincombi_trafos[6][3];
        
        // 3)
        ntrafos = build_bincombi_trafos(elb1, elb2, elb3, bincombi_trafos);
        for (int eltrafo=0; eltrafo<ntrafos; eltrafo++){
            elb1t = bincombi_trafos[eltrafo][0];
            elb2t = bincombi_trafos[eltrafo][1];
            elb3t = bincombi_trafos[eltrafo][2];
            for (int nindex=0; nindex<len_nindices; nindex++){
                thisn2 = nindices[nindex]/nnvals_Upsn - nzero_Ups;
                thisn3 = nindices[nindex]%nnvals_Upsn - nzero_Ups;
                nbshift = ((thisn2+nzero_Ups)*nnvals_Upsn + (thisn3+nzero_Ups));
                for (int elthread=0; elthread<nthreads; elthread++){
                    tnrshift = elthread*len_nindices*nthetacombis + nindex*nthetacombis + elb;
                    thisUpsilon_n[0*n2n3combis+nbshift] += tmpUpsilon0_n[tnrshift];
                    thisUpsilon_n[1*n2n3combis+nbshift] += tmpUpsilon1_n[tnrshift];
                    thisUpsilon_n[2*n2n3combis+nbshift] += tmpUpsilon2_n[tnrshift];
                    thisUpsilon_n[3*n2n3combis+nbshift] += tmpUpsilon3_n[tnrshift];
                    thisUpsilon_n[4*n2n3combis+nbshift] += tmpUpsilon4_n[tnrshift];
                    thisUpsilon_n[5*n2n3combis+nbshift] += tmpUpsilon5_n[tnrshift];
                    thisUpsilon_n[6*n2n3combis+nbshift] += tmpUpsilon6_n[tnrshift];
                    thisUpsilon_n[7*n2n3combis+nbshift] += tmpUpsilon7_n[tnrshift];
                    thisN_n[nbshift] += tmpN_n[tnrshift];
                }
            }

            getMultipolesFromSymm(
                thisUpsilon_n, thisN_n, nmax, eltrafo, nindices, len_nindices,
                thisUpsilon_n_rec, thisN_n_rec);

            // 4)
            for (int eln12=0; eln12<n2n3combis_rec; eln12++){
                int thisnrshift = eln12*ups_nshift + elb1t*nbinsr*nbinsr + elb2t*nbinsr + elb3t;
                for (int elcomp=0; elcomp<8; elcomp++){
                    Upsilon_n[elcomp*ups_rec_compshift+thisnrshift] =  thisUpsilon_n_rec[elcomp*n2n3combis_rec+eln12];
                }
                N_n[thisnrshift] = thisN_n_rec[eln12];
            }  

            // Reset 4pcf placeholders to zero
            for (int i=0; i<n2n3combis; i++){
                thisN_n[i] = 0;
                for (int elcomp=0; elcomp<8; elcomp++){
                    thisUpsilon_n[elcomp*n2n3combis+i] = 0;
                }
            }
            for (int i=0; i<n2n3combis_rec; i++){
                thisN_n_rec[i] = 0;
                for (int elcomp=0; elcomp<8; elcomp++){
                    thisUpsilon_n_rec[elcomp*n2n3combis_rec+i] = 0;
                }
            }
        }
        free(thisUpsilon_n);
        free(thisUpsilon_n_rec);
        free(thisN_n);
        free(thisN_n_rec);
    }

    // Accumulate the bin distances and weights
    for (int thisthread=0; thisthread<nthreads; thisthread++){
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            totcounts[elbinr] += tmpwcounts[thisthread*nbinsr+elbinr];
            totnorms[elbinr] += tmpwnorms[thisthread*nbinsr+elbinr];
        }
    }
    // Get bin centers
    for (int elbinr=0; elbinr<nbinsr; elbinr++){
        if (totnorms[elbinr] != 0){
            bin_centers[elbinr] = totcounts[elbinr]/totnorms[elbinr];
        }
    }
    
    free(tmpUpsilon0_n);
    free(tmpUpsilon1_n);
    free(tmpUpsilon2_n);
    free(tmpUpsilon3_n);
    free(tmpUpsilon4_n);
    free(tmpUpsilon5_n);
    free(tmpUpsilon6_n);
    free(tmpUpsilon7_n);
    free(tmpN_n);
    free(tmpwcounts);
    free(tmpwnorms);
    free(totcounts);
    free(totnorms);

    free(elb1_inds);
    free(elb2_inds);
    free(elb3_inds);

    free(regionsdone);
}

// If thread==0 --> For final two threads allocate double/triple counting corrs
// thetacombis_batches: array of length nbinsr^3 with the indices of all possible (r1,r2,r3) combinations
//                      most likely it is simply range(nbinsr^3), but we leave some freedom here for 
//                      potential cost-based implementations
// nthetacombis_batches: array of length nthetbatches with the number of theta-combis in each batch
// cumthetacombis_batches : array of length (nthetbatches+1) with is cumsum of nthetacombis_batches
// nthetbatches: the number of theta batches
void alloc_notomoMap4_disc_gggg(const MultiresoCatalog *cat, const NavHash *nav,
    const BinningParams *bin, const FourthParams *fourth,
    int projection, double *mapradii, int nmapradii, double complex *M4correlators, int alloc_4pcfmultipoles, int alloc_4pcfreal,
    int nthreads, int verbose,
    double *bin_centers, double complex *Upsilon_n, double complex *N_n, double complex *Gammas, double complex *Norms){

    // Dereference passed input structs
    double *isinner = cat->isinner_resos, *weight = cat->weight_resos;
    double *pos1 = cat->pos1_resos, *pos2 = cat->pos2_resos;
    double *e1 = cat->e1_resos, *e2 = cat->e2_resos;
    int nmax = bin->nmax, nbinsr = bin->nbinsr;
    double rmin = bin->rmin, rmax = bin->rmax;
    double *phibins = fourth->phibins1, *dbinsphi = fourth->dbinsphi1; int nbinsphi = fourth->nbinsphi1;
    int *index_matcher_hash = nav->index_matcher, *pixs_galind_bounds = nav->pixs_galind_bounds, *pix_gals = nav->pix_gals;
    int nregions = nav->nregions;
    double pix1_start = nav->pix1_start, pix1_d = nav->pix1_d; int pix1_n = nav->pix1_n;
    double pix2_start = nav->pix2_start, pix2_d = nav->pix2_d; int pix2_n = nav->pix2_n;
    int *thetacombis_batches = fourth->thetacombis_batches, *nthetacombis_batches = fourth->nthetacombis_batches;
    int *cumthetacombis_batches = fourth->cumthetacombis_batches; int nthetbatches = fourth->nthetbatches;
               
    double complex *allM4correlators = orpheus_calloc(nthreads*8*1*nmapradii, sizeof(double complex));
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){free(allM4correlators); return;}
    int nregionsdone = 0;
    reset_progress();
    #pragma omp parallel for num_threads(nthreads)
    for (int elthetbatch=0; elthetbatch<nthetbatches; elthetbatch++){
        int nregions_skip_print = mymax(1, nregions / 100);
        
        int thisthread = omp_get_thread_num();
        //printf("Doing thetabatch %d/%d on thread %d\n",elthetbatch,nthetbatches,thisthread);
        int nbinsz = 1;
        int ncomp = 8;
        int nnvals_Gn = 4*nmax+3; // Need to cover [-n1-n2-3, n1+n2-1]
        int nnvals_G2n = 4*nmax+7; // Need to cover [-n1-n2-5, n1+n2+1]
        int nnvals_Wn = 4*nmax+1; // Need to cover [-n1-n2, n1+n2]
        int nnvals_Upsn = 2*nmax+1; // Need tocover [-nmax,+nmax]
        int nzero_Gn = 2*nmax+3;
        int nzero_G2n = 2*nmax+5;
        int nzero_Wn = 2*nmax;
        int nzero_Ups = nmax;
        
        int ups_nshift = nbinsr*nbinsr*nbinsr;
        int n2n3combis = nnvals_Upsn*nnvals_Upsn;
        int ups_compshift = n2n3combis*ups_nshift;
        
        int batch_nthetas = nthetacombis_batches[elthetbatch];
        int batchups_nshift = batch_nthetas;
        int batchups_compshift = n2n3combis*batchups_nshift;
        int batchgamma_thetshift = nbinsphi*nbinsphi;
        
        double *totcounts = orpheus_calloc(nbinsr, sizeof(double));
        double *totnorms = orpheus_calloc(nbinsr, sizeof(double));
        double *bin_centers_batch = orpheus_calloc(nbinsr, sizeof(double));
        double complex *batchUpsilon_n = orpheus_calloc(ncomp*batchups_compshift, sizeof(double complex));
        double complex *batchN_n = orpheus_calloc(batchups_compshift, sizeof(double complex));
        double complex *batchfourpcf = orpheus_calloc(ncomp*batchups_compshift, sizeof(double complex));
        double complex *batchfourpcf_norm = orpheus_calloc(batchups_compshift, sizeof(double complex));
        double *batch_thetas1 = orpheus_calloc(batch_nthetas, sizeof(double));
        double *batch_thetas2 = orpheus_calloc(batch_nthetas, sizeof(double));
        double *batch_thetas3 = orpheus_calloc(batch_nthetas, sizeof(double));
        
        int nbinszr = nbinsz*nbinsr;
        double complex *nextGns =  orpheus_calloc(nnvals_Gn*nbinszr, sizeof(double complex));
        double complex *nextG2ns_gg =  orpheus_calloc(nnvals_G2n*nbinszr, sizeof(double complex));
        double complex *nextG2ns_ggc =  orpheus_calloc(nnvals_G2n*nbinszr, sizeof(double complex));
        double complex *nextG3ns_ggg = orpheus_calloc(2*nbinszr, sizeof(double complex));
        double complex *nextG3ns_gggc = orpheus_calloc(2*nbinszr, sizeof(double complex));
        double complex *nextWns = orpheus_calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW2ns = orpheus_calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW3ns = orpheus_calloc(nbinszr, sizeof(double complex));
        
        double drbin = (log(rmax)-log(rmin))/(nbinsr);
        int *elb1s_batch = orpheus_calloc(batch_nthetas, sizeof(int));
        int *elb2s_batch = orpheus_calloc(batch_nthetas, sizeof(int));
        int *elb3s_batch = orpheus_calloc(batch_nthetas, sizeof(int));
        double *bin_edges = orpheus_calloc(nbinsr+1, sizeof(double));
        #pragma omp critical
        {
        for (int elb=0; elb<batch_nthetas; elb++){
            int thisrcombi = thetacombis_batches[cumthetacombis_batches[elthetbatch]+elb];
            elb1s_batch[elb] = thisrcombi/(nbinsr*nbinsr);
            elb2s_batch[elb] = (thisrcombi-elb1s_batch[elb]*nbinsr*nbinsr)/nbinsr;
            elb3s_batch[elb] = thisrcombi-elb1s_batch[elb]*nbinsr*nbinsr-elb2s_batch[elb]*nbinsr;
        }
        bin_edges[0] = rmin;
        for (int elb=0; elb<nbinsr; elb++){
            bin_edges[elb+1] = bin_edges[elb]*exp(drbin);
        }
        }
           
        // Allocate the 4pcf multipoles for this batch of radii 
        int offset_per_thread = nregions/nthreads;
        int offset = offset_per_thread*thisthread;
        for (int _elregion=0; _elregion<nregions; _elregion++){
            int elregion = (_elregion+offset)%nregions; // Try to evade collisions
            #pragma omp atomic
            nregionsdone += 1;
            print_progress(nregionsdone, nthetbatches*nregions, verbose);
            //int region_debug = mymin(500,nregions-1);
            int lower1, upper1;
            lower1 = pixs_galind_bounds[elregion];
            upper1 = pixs_galind_bounds[elregion+1];
            for (int ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                double time1, time2;
                time1 = omp_get_wtime();
                int ind_gal = pix_gals[ind_inpix1];
                double p11, p12, w1, e11, e12;
                double innergal;
                p11 = pos1[ind_gal];
                p12 = pos2[ind_gal];
                w1 = weight[ind_gal];
                e11 = e1[ind_gal];
                e12 = e2[ind_gal];
                innergal = isinner[ind_gal];
                if (innergal<1e-5){continue;}
                
                int ind_gal2;
                int lower, upper;
                double  p21, p22, w2, w2_sq, e21, e22, rel1, rel2, dist, dphi;
                double complex wshape1, wshape1c, wshape2, wshape_sq, wshape_cube, wshapewshapec, wshapesqwshapec;
                double complex phirot, phirotc, twophirotc, fourphirotc;
                
                // Check how many ns we need for Gn
                // Gns have shape (nnvals, nbinsz, nbinsr)
                // where the ns are ordered as 
                // [-nmax_1-nmax_2-3, ..., nmax_1+nmax_2+3]
                for (int i=0; i<nnvals_Gn*nbinszr; i++){nextGns[i]=0;}
                for (int i=0; i<nnvals_G2n*nbinszr; i++){nextG2ns_gg[i]=0;nextG2ns_ggc[i]=0;}
                for (int i=0; i<2*nbinszr; i++){nextG3ns_ggg[i]=0;nextG3ns_gggc[i]=0;}
                for (int i=0; i<nnvals_Wn*nbinszr; i++){nextWns[i]=0;nextW2ns[i]=0;}
                for (int i=0; i<nbinszr; i++){nextW3ns[i]=0;}

                int rbin, zrshift, ind_Gn, ind_G2n, ind_Wn;
                FLATCELL_FOREACH(
                    index_matcher_hash, 0, pixs_galind_bounds, 0, p11, p12, rmax,
                    pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, lower, upper){
                    for (int ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                        ind_gal2 = pix_gals[ind_inpix];
                        p21 = pos1[ind_gal2];
                        p22 = pos2[ind_gal2];
                        w2 = weight[ind_gal2];
                        e21 = e1[ind_gal2];
                        e22 = e2[ind_gal2];

                        rel1 = p21 - p11;
                        rel2 = p22 - p12;
                        dist = sqrt(rel1*rel1 + rel2*rel2);
                        if (dist < rmin || dist >= rmax) continue;
                        rbin = (int) floor((log(dist)-log(rmin))/drbin);
                        w2_sq = w2*w2;
                        wshape2 = (double complex) w2 * (e21+I*e22);
                        wshape_sq = wshape2*wshape2;
                        wshape_cube = wshape_sq*wshape2;
                        wshapewshapec = wshape2*conj(wshape2);
                        wshapesqwshapec = wshape_sq*conj(wshape2);
                        dphi = atan2(rel2,rel1);
                        phirot = cexp(I*dphi);
                        phirotc = conj(phirot);
                        twophirotc = phirotc*phirotc;
                        fourphirotc = twophirotc*twophirotc;
                        zrshift = 0*nbinsr + rbin;
                        ind_Gn = nzero_Gn*nbinszr + zrshift;
                        ind_G2n = nzero_G2n*nbinszr + zrshift;
                        ind_Wn = nzero_Wn*nbinszr + zrshift;
                        totcounts[zrshift] += w1*w2*dist; 
                        totnorms[zrshift] += w1*w2; 
                        gggg_fill_gnwn(nextGns, nextG2ns_gg, nextG2ns_ggc,
                                        nextWns, nextW2ns, nextW3ns, nextG3ns_ggg, nextG3ns_gggc,
                                        2*nmax, nbinszr, zrshift, ind_Gn, ind_G2n, ind_Wn,
                                        w2, w2_sq, wshape2, wshape_sq, wshape_cube,
                                        wshapewshapec, wshapesqwshapec, phirot, phirotc, fourphirotc);
                    }
                }
                time2 = omp_get_wtime();
                if ((verbose>1) && (elregion%nregions_skip_print == 0)&&(thisthread==0)&&(ind_inpix1==lower1)){
                    printf("Computed Gn for first gal in region %d/%d for thetabatch %d/%d in %.4f seconds\n",
                           elregion,nregions,elthetbatch,nthetbatches,(time2-time1));}
                
                time1 = omp_get_wtime();
                // Allocate Upsilon
                // Upsilon have shape 
                // (ncomp,(2*nmax+1),(2*nmax+1),nthetas)

                int thisn, thisnshift;
                int thisGshift_mn2m2, thisGshift_n2m2, thisWshift_n2;
                int thisGshift_mn3m3, thisGshift_mn3m1, thisGshift_n3m3, thisGshift_n3m1, thisWshift_n3;
                int thisGshift_mn2mn3m3, thisGshift_mn2mn3m1, thisGshift_n2n3m3, thisGshift_n2n3m1, thisWshift_n2n3;
                wshape1 = w1 * (e11+I*e12);  
                wshape1c = conj(wshape1);
                for (int thisn2=-nmax; thisn2<=nmax; thisn2++){
                    thisGshift_mn2m2 = (nzero_Gn-thisn2-2)*nbinsr;
                    thisGshift_n2m2 = (nzero_Gn+thisn2-2)*nbinsr;
                    thisWshift_n2 = (nzero_Wn+thisn2)*nbinsr;
                    for (int thisn3=-nmax; thisn3<=nmax; thisn3++){
                        thisn = thisn2+thisn3;
                        thisGshift_mn3m3 = (nzero_Gn-thisn3-3)*nbinsr;
                        thisGshift_mn3m1 = (nzero_Gn-thisn3-1)*nbinsr;
                        thisGshift_n3m3 = (nzero_Gn+thisn3-3)*nbinsr;
                        thisGshift_n3m1 = (nzero_Gn+thisn3-1)*nbinsr;
                        thisWshift_n3 = (nzero_Wn+thisn3)*nbinsr;
                        thisGshift_mn2mn3m3 = (nzero_Gn-thisn-3)*nbinsr;
                        thisGshift_mn2mn3m1 = (nzero_Gn-thisn-1)*nbinsr;
                        thisGshift_n2n3m3 = (nzero_Gn+thisn-3)*nbinsr;
                        thisGshift_n2n3m1 = (nzero_Gn+thisn-1)*nbinsr;
                        thisWshift_n2n3 = (nzero_Wn+thisn)*nbinsr;
                        thisnshift = ((thisn2+nzero_Ups)*nnvals_Upsn + (thisn3+nzero_Ups)) * batchups_nshift;
                        gggg_accum_batchUpsilon(batchUpsilon_n, batchN_n,
                                                batch_nthetas, batchups_compshift, thisnshift,
                                                elb1s_batch, elb2s_batch, elb3s_batch,
                                                nextGns, nextG2ns_gg, nextG2ns_ggc, nextG3ns_ggg, nextG3ns_gggc,
                                                nextWns, nextW2ns, nextW3ns,
                                                wshape1, wshape1c, w1,
                                                nbinsr, nzero_G2n, nzero_Wn, thisn2, thisn3,
                                                thisGshift_mn2m2, thisGshift_n2m2, thisWshift_n2,
                                                thisGshift_mn3m3, thisGshift_mn3m1, thisGshift_n3m3, thisGshift_n3m1,
                                                thisWshift_n3, thisGshift_mn2mn3m3, thisGshift_mn2mn3m1,
                                                thisGshift_n2n3m3, thisGshift_n2n3m1, thisWshift_n2n3);
                    }
                }
                time2 = omp_get_wtime();
                if ((verbose>1) && (elregion%nregions_skip_print == 0)&&(thisthread==0)&&(ind_inpix1==lower1)){
                    printf("Allocated Ups for first gal in region %d/%d for thetabatch %d/%d in %.4f seconds for %d theta-combis\n",
                           elregion,nregions,elthetbatch,nthetbatches,(time2-time1),batch_nthetas);}
            }
            if ((verbose>1) && (elregion%nregions_skip_print == 0)&&(thisthread==0)){
                printf("Done region %d/%d for thetabatch %d/%d\n",elregion,nregions,elthetbatch,nthetbatches);}
        }
        
        // Get bin centers
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            if (totnorms[elbinr] != 0){
                // Note that the bin centers are the same for every batch!
                bin_centers_batch[elbinr] = totcounts[elbinr]/totnorms[elbinr]; 
                if (elthetbatch==0){bin_centers[elbinr] = bin_centers_batch[elbinr];} // Debug
            }
        }
        
        // For each theta combination (theta1,theta2,theta3) in this batch 
        // 1) Get bin edges and bin centers of the combinations
        // 2) Get the Gamma_mu(theta1,theta2,theta3,phi12,phi13)
        // 3) Transform the Gamma_mu to the target basis
        // 4) Update the aperture Map^4 integral
        double complex *nextM4correlators = orpheus_calloc(8, sizeof(double complex));
        double complex *thisUpsilon_n = orpheus_calloc(8*n2n3combis, sizeof(double complex));
        double complex *thisN_n = orpheus_calloc(n2n3combis, sizeof(double complex));
        double complex *thisnpcf = orpheus_calloc(8*batchgamma_thetshift, sizeof(double complex));
        double complex *thisnpcf_norm = orpheus_calloc(batchgamma_thetshift, sizeof(double complex));
        for (int elb=0; elb<batch_nthetas; elb++){
            // 1)
            int nbshift, elb1, elb2, elb3;
            elb1 = elb1s_batch[elb];
            elb2 = elb2s_batch[elb];
            elb3 = elb3s_batch[elb];
            // 2)
            for (int eln12=0; eln12<n2n3combis; eln12++){
                nbshift = eln12*batchups_nshift+elb;
                for (int elcomp=0; elcomp<8; elcomp++){
                    thisUpsilon_n[elcomp*n2n3combis+eln12] = batchUpsilon_n[elcomp*batchups_compshift+nbshift];
                }
                thisN_n[eln12] = batchN_n[nbshift];
                // OPTIONAL: Allocate 4PCF in multipole basis
                if (alloc_4pcfmultipoles==1){
                    int thisnrshift = eln12*ups_nshift + elb1*nbinsr*nbinsr + elb2*nbinsr + elb3;
                    for (int elcomp=0; elcomp<8; elcomp++){
                        Upsilon_n[elcomp*ups_compshift+thisnrshift] = 
                                thisUpsilon_n[elcomp*n2n3combis+eln12];
                    }
                    N_n[thisnrshift] = thisN_n[eln12];
                }
            }
            // 3)
            // Only required for the real-space 4pcf and for the aperture integration
            if ((alloc_4pcfreal==1) || (nmapradii>0)){
                multipoles2npcf_gggg_singletheta(thisUpsilon_n, thisN_n, nmax, nmax,
                                                 bin_centers_batch[elb1], bin_centers_batch[elb2], bin_centers_batch[elb3],
                                                 phibins, phibins, nbinsphi, nbinsphi,
                                                 projection, fourth->count_floor,
                                                 thisnpcf, thisnpcf_norm);
            }
            
            // OPTIONAL: Allocate 4pcf in real basis (Shape: (8,ntheta,ntheta,ntheta,nphi,nphi)
            if (alloc_4pcfreal==1){
                for (int elphi12=0; elphi12<batchgamma_thetshift; elphi12++){
                    int gamma_rshift = nbinsphi*nbinsphi;
                    int gamma_phircombi = gamma_rshift*(elb1*nbinsr*nbinsr+elb2*nbinsr+elb3)+elphi12;
                    int gamma_compshift = nbinsr*nbinsr*nbinsr*gamma_rshift;
                    for (int elcomp=0; elcomp<8; elcomp++){
                        Gammas[elcomp*gamma_compshift+gamma_phircombi] = thisnpcf[elcomp*batchgamma_thetshift+elphi12];
                    }
                    Norms[gamma_phircombi] = thisnpcf_norm[elphi12];
                }
            }
            
            // 4)
            double y1, y2, y3, dy1, dy2, dy3;
            int map4ind;
            int map4threadshift = thisthread*8*nmapradii;
            for (int elmapr=0; elmapr<nmapradii; elmapr++){
                y1=bin_centers_batch[elb1]/mapradii[elmapr];
                y2=bin_centers_batch[elb2]/mapradii[elmapr];
                y3=bin_centers_batch[elb3]/mapradii[elmapr];
                dy1 = (bin_edges[elb1+1]-bin_edges[elb1])/mapradii[elmapr];
                dy2 = (bin_edges[elb2+1]-bin_edges[elb2])/mapradii[elmapr];
                dy3 = (bin_edges[elb3+1]-bin_edges[elb3])/mapradii[elmapr];
                fourpcf2M4correlators(1,
                                      y1, y2, y3, dy1, dy2, dy3,
                                      phibins, phibins, dbinsphi, dbinsphi, nbinsphi, nbinsphi,
                                      thisnpcf, nextM4correlators);
                for (int elcomp=0; elcomp<8; elcomp++){
                    map4ind = elcomp*nmapradii+elmapr;
                    if (isfinite(cabs(nextM4correlators[elcomp]))){
                        allM4correlators[map4threadshift+map4ind] += nextM4correlators[elcomp];
                    }
                    nextM4correlators[elcomp] = 0;
                }
            }
        
            // Reset 4pcf placeholders to zero
            for (int i=0; i<batchgamma_thetshift; i++){
                thisnpcf_norm[i] = 0;
                for (int elcomp=0; elcomp<8; elcomp++){
                    thisnpcf[elcomp*batchgamma_thetshift+i] = 0;
                }
            }
            for (int i=0; i<n2n3combis; i++){
                thisN_n[i] = 0;
                for (int elcomp=0; elcomp<8; elcomp++){
                    thisUpsilon_n[elcomp*n2n3combis+i] = 0;
                }
            }
        }
        
        for (int elmapr=0; elmapr<nmapradii; elmapr++){
            for (int elcomp=0; elcomp<8; elcomp++){
                int map4ind = elcomp*nmapradii+elmapr;
                int map4threadshift = thisthread*8*nmapradii;
                if (verbose>1){ printf("\nthread %d, elr %d, elcomp %d, allM4cont=%.20f ",
                               thisthread, elmapr, elcomp, creal(allM4correlators[map4threadshift+map4ind])); }
            }
        }
        if (verbose>1){printf("Done allocating 4pcfs for thetabatch %d/%d\n",elthetbatch,nthetbatches);}
            
        free(totcounts);
        free(totnorms);
        free(bin_centers_batch);
        free(batch_thetas1);
        free(batch_thetas2);
        free(batch_thetas3);
        free(batchUpsilon_n);
        free(batchN_n);
        free(batchfourpcf);
        free(batchfourpcf_norm);
        
        free(nextGns);
        free(nextG2ns_gg);
        free(nextG2ns_ggc);
        free(nextG3ns_ggg);
        free(nextG3ns_gggc);
        free(nextWns);
        free(nextW2ns);
        free(nextW3ns);
        
        free(elb1s_batch);
        free(elb2s_batch);
        free(elb3s_batch);
        free(bin_edges);
        
        free(nextM4correlators);
        free(thisUpsilon_n);
        free(thisN_n);
        free(thisnpcf);
        free(thisnpcf_norm);                
    }
    
    // Accummulate the Map^4 integral
    for (int elthread=0; elthread<nthreads; elthread++){
        int map4ind;
        int map4threadshift = elthread*8*nmapradii;
        for (int elcomp=0; elcomp<8; elcomp++){
            for (int elmapr=0; elmapr<nmapradii; elmapr++){
                map4ind = elcomp*nmapradii+elmapr;
                M4correlators[map4ind] += allM4correlators[map4threadshift+map4ind];
            }
        }
    }    
    if (verbose>0){ printf("\n"); }
    free(allM4correlators);
}

// If thread==0 --> For final two threads allocate double/triple counting corrs
// thetacombis_batches: array of length nbinsr^3 with the indices of all possible (r1,r2,r3) combinations
//                      most likely it is simply range(nbinsr^3), but we leave some freedom here for 
//                      potential cost-based implementations
// nthetacombis_batches: array of length nthetbatches with the number of theta-combis in each batch
// cumthetacombis_batches : array of length (nthetbatches+1) with is cumsum of nthetacombis_batches
// nthetbatches: the number of theta batches
void alloc_notomoMap4_tree_gggg(const MultiresoCatalog *cat_base, const MultiresoCatalog *cat_leaf,
    const NavHash *nav, const TreeResoParams *tree,
    const BinningParams *bin, const FourthParams *fourth,
    int projection, double *mapradii, int nmapradii, double complex *M4correlators, int alloc_4pcfmultipoles, int alloc_4pcfreal,
    int nthreads, int verbose,
    double *bin_centers, double complex *Upsilon_n, double complex *N_n, double complex *Gammas, double complex *Norms){
    
    // Dereference input structures
    double *isinner = cat_base->isinner_resos, *weight = cat_base->weight_resos;
    double *pos1 = cat_base->pos1_resos, *pos2 = cat_base->pos2_resos;
    double *e1 = cat_base->e1_resos, *e2 = cat_base->e2_resos;
    int nresos = tree->nresos; double *reso_redges = tree->reso_redges;
    int *ngal_resos = cat_leaf->ngal_resos;
    double *weight_resos = cat_leaf->weight_resos;
    double *pos1_resos = cat_leaf->pos1_resos, *pos2_resos = cat_leaf->pos2_resos;
    double *e1_resos = cat_leaf->e1_resos, *e2_resos = cat_leaf->e2_resos;
    int nmax = bin->nmax, nbinsr = bin->nbinsr;
    double rmin = bin->rmin, rmax = bin->rmax;
    int *nindices = fourth->nindices, len_nindices = fourth->len_nindices;
    double *phibins = fourth->phibins1, *dbinsphi = fourth->dbinsphi1; int nbinsphi = fourth->nbinsphi1;
    int *index_matcher_hash = nav->index_matcher, *pixs_galind_bounds = nav->pixs_galind_bounds, *pix_gals = nav->pix_gals;
    int nregions = nav->nregions;
    double pix1_start = nav->pix1_start, pix1_d = nav->pix1_d; int pix1_n = nav->pix1_n;
    double pix2_start = nav->pix2_start, pix2_d = nav->pix2_d; int pix2_n = nav->pix2_n;
    int *thetacombis_batches = fourth->thetacombis_batches, *nthetacombis_batches = fourth->nthetacombis_batches;
    int *cumthetacombis_batches = fourth->cumthetacombis_batches; int nthetbatches = fourth->nthetbatches;
               
    double complex *allM4correlators = orpheus_calloc(nthreads*8*1*nmapradii, sizeof(double complex));
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){free(allM4correlators); return;}
    int nregionsdone = 0;
    reset_progress();
    #pragma omp parallel for num_threads(nthreads)
    for (int elthetbatch=0; elthetbatch<nthetbatches; elthetbatch++){
        int nregions_skip_print = mymax(1, nregions / 100);
        
        // * nmax_alloc specifies the largest multipole that needs to be allocated when wanting 
        //   to allocate the Upsn/Nn while making use of the symmetry properties
        // * All quantities that are updated at the galaxy level are computed until nmax_alloc
        // * Once we are done iterating over the cat we apply the symmetries and allocate the
        //   reconstructed quantities having a suffix _rec
        int thisthread = omp_get_thread_num();
        //printf("Doing thetabatch %d/%d on thread %d\n",elthetbatch,nthetbatches,thisthread);
        int nmax_alloc = 2*nmax+1;
        int nbinsz = 1;
        int ncomp = 8;
        int nnvals_Gn = 4*nmax_alloc+3; // Need to cover [-n1-n2-3, n1+n2-1]
        int nnvals_G2n = 4*nmax_alloc+7; // Need to cover [-n1-n2-5, n1+n2+1]
        int nnvals_Wn = 4*nmax_alloc+1; // Need to cover [-n1-n2, n1+n2]
        int nnvals_Upsn = 2*nmax_alloc+1;  // Need tocover [-2*nmax_alloc,+2*nmax_alloc]
        int nnvals_Upsn_rec = 2*nmax+1; // Need tocover [-nmax,+nmax]
        int nzero_Gn = 2*nmax_alloc+3;
        int nzero_G2n = 2*nmax_alloc+5;
        int nzero_Wn = 2*nmax_alloc;
        int nzero_Ups = nmax_alloc;
        
        int ups_nshift = nbinsr*nbinsr*nbinsr;
        int n2n3combis = nnvals_Upsn*nnvals_Upsn;
        int n2n3combis_rec = nnvals_Upsn_rec*nnvals_Upsn_rec;
        int ups_rec_compshift = n2n3combis_rec*ups_nshift;
        
        int batch_nthetas = nthetacombis_batches[elthetbatch];
        int batchups_nshift = batch_nthetas;
        int batchups_compshift = n2n3combis*batchups_nshift;
        int batchgamma_thetshift = nbinsphi*nbinsphi;
        
        int npix_hash = pix1_n*pix2_n;
        int *rshift_index_matcher_hash = orpheus_calloc(nresos, sizeof(int));
        int *rshift_pixs_galind_bounds = orpheus_calloc(nresos, sizeof(int));
        int *rshift_pix_gals = orpheus_calloc(nresos, sizeof(int));
        build_rshift_offsets(nresos, npix_hash, ngal_resos,
                             rshift_index_matcher_hash, rshift_pixs_galind_bounds, rshift_pix_gals);

        double *totcounts = orpheus_calloc(nbinsr, sizeof(double));
        double *totnorms = orpheus_calloc(nbinsr, sizeof(double));
        double *bin_centers_batch = orpheus_calloc(nbinsr, sizeof(double));
        double complex *batchUpsilon_n = orpheus_calloc(ncomp*batchups_compshift, sizeof(double complex));
        double complex *batchN_n = orpheus_calloc(batchups_compshift, sizeof(double complex));
        double complex *batchfourpcf = orpheus_calloc(ncomp*batchups_compshift, sizeof(double complex));
        double complex *batchfourpcf_norm = orpheus_calloc(batchups_compshift, sizeof(double complex));
        double *batch_thetas1 = orpheus_calloc(batch_nthetas, sizeof(double));
        double *batch_thetas2 = orpheus_calloc(batch_nthetas, sizeof(double));
        double *batch_thetas3 = orpheus_calloc(batch_nthetas, sizeof(double));
        
        int nbinszr = nbinsz*nbinsr;
        double complex *nextGns =  orpheus_calloc(nnvals_Gn*nbinszr, sizeof(double complex));
        double complex *nextG2ns_gg =  orpheus_calloc(nnvals_G2n*nbinszr, sizeof(double complex));
        double complex *nextG2ns_ggc =  orpheus_calloc(nnvals_G2n*nbinszr, sizeof(double complex));
        double complex *nextG3ns_ggg = orpheus_calloc(2*nbinszr, sizeof(double complex));
        double complex *nextG3ns_gggc = orpheus_calloc(2*nbinszr, sizeof(double complex));
        double complex *nextWns = orpheus_calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW2ns = orpheus_calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW3ns = orpheus_calloc(nbinszr, sizeof(double complex));
        
        double drbin = (log(rmax)-log(rmin))/(nbinsr);
        int rbin_min_batch=nbinsr;int rbin_max_batch=0;
        int reso_min_batch=0; int reso_max_batch=0;
        int *elb1s_batch = orpheus_calloc(batch_nthetas, sizeof(int));
        int *elb2s_batch = orpheus_calloc(batch_nthetas, sizeof(int));
        int *elb3s_batch = orpheus_calloc(batch_nthetas, sizeof(int));
        double *bin_edges = orpheus_calloc(nbinsr+1, sizeof(double));
        #pragma omp critical
        {
            build_thetabatch(elthetbatch, batch_nthetas, nbinsr, nresos, rmin, rmax,
                             thetacombis_batches, cumthetacombis_batches, reso_redges,
                             elb1s_batch, elb2s_batch, elb3s_batch, bin_edges,
                             &rbin_min_batch, &rbin_max_batch, &reso_min_batch, &reso_max_batch);
        }
        
        // Allocate the 4pcf multipoles for this batch of radii 
        int offset_per_thread = nregions/nthreads;
        int offset = offset_per_thread*thisthread;
        for (int _elregion=0; _elregion<nregions; _elregion++){
            int elregion = (_elregion+offset)%nregions; // Try to evade collisions
            #pragma omp atomic
            nregionsdone += 1;
            print_progress(nregionsdone, nthetbatches*nregions, verbose);
            //int region_debug = mymin(500,nregions-1);
            int lower1, upper1;
            lower1 = pixs_galind_bounds[elregion];
            upper1 = pixs_galind_bounds[elregion+1];
            for (int ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                double time1, time2;
                time1 = omp_get_wtime();
                int ind_gal = pix_gals[ind_inpix1];
                double p11, p12, w1, e11, e12;
                double innergal = isinner[ind_gal];
                if (innergal<1e-5){continue;}
                p11 = pos1[ind_gal];
                p12 = pos2[ind_gal];
                w1 = innergal*weight[ind_gal];
                e11 = e1[ind_gal];
                e12 = e2[ind_gal];                
                
                int ind_gal2;
                int lower, upper;
                double  p21, p22, w2, w2_sq, e21, e22, rel1, rel2, dist2, dist, dphi;
                double complex wshape1, wshape1c, wshape2, wshape_sq, wshape_cube, wshapewshapec, wshapesqwshapec;
                double complex phirot, phirotc, twophirotc, fourphirotc;
                
                // Check how many ns we need for Gn
                // Gns have shape (nnvals, nbinsz, nbinsr)
                // where the ns are ordered as 
                // [-nmax_1-nmax_2-3, ..., nmax_1+nmax_2+3]
                for (int i=0; i<nnvals_Gn*nbinszr; i++){nextGns[i]=0;}
                for (int i=0; i<nnvals_G2n*nbinszr; i++){nextG2ns_gg[i]=0;nextG2ns_ggc[i]=0;}
                for (int i=0; i<2*nbinszr; i++){nextG3ns_ggg[i]=0;nextG3ns_gggc[i]=0;}
                for (int i=0; i<nnvals_Wn*nbinszr; i++){nextWns[i]=0;nextW2ns[i]=0;}
                for (int i=0; i<nbinszr; i++){nextW3ns[i]=0;}
                for (int elreso=reso_min_batch; elreso<=reso_max_batch; elreso++){
                    int rbin, zrshift, ind_Gn, ind_G2n, ind_Wn;
                    double rmin_reso = reso_redges[elreso];
                    double rmin_reso2 = rmin_reso*rmin_reso;
                    double rmax_reso = reso_redges[elreso+1];
                    double rmax_reso2 = rmax_reso*rmax_reso;
                    FLATCELL_FOREACH(
                        index_matcher_hash, rshift_index_matcher_hash[elreso], pixs_galind_bounds, rshift_pixs_galind_bounds[elreso],
                        p11, p12, rmax_reso, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, lower, upper){
                        for (int ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                            ind_gal2 = rshift_pix_gals[elreso] + pix_gals[rshift_pix_gals[elreso]+ind_inpix];
                            //#pragma omp critical
                            p21 = pos1_resos[ind_gal2];
                            p22 = pos2_resos[ind_gal2];
                            w2 = weight_resos[ind_gal2];
                            e21 = e1_resos[ind_gal2];
                            e22 = e2_resos[ind_gal2];
                            
                            rel1 = p21 - p11;
                            rel2 = p22 - p12;
                            dist2 = rel1*rel1 + rel2*rel2;
                            if (dist2 < rmin_reso2 || dist2 >= rmax_reso2){continue;}
                            dist = sqrt(dist2);
                            rbin = (int) floor((log(dist)-log(rmin))/drbin);
                            w2_sq = w2*w2;
                            wshape2 = (double complex) w2 * (e21+I*e22);
                            wshape_sq = wshape2*wshape2;
                            wshape_cube = wshape_sq*wshape2;
                            wshapewshapec = wshape2*conj(wshape2);
                            wshapesqwshapec = wshape_sq*conj(wshape2);
                            dphi = atan2(rel2,rel1);
                            phirot = cexp(I*dphi);
                            phirotc = conj(phirot);
                            twophirotc = phirotc*phirotc;
                            fourphirotc = twophirotc*twophirotc;
                            zrshift = 0*nbinsr + rbin;
                            ind_Gn = nzero_Gn*nbinszr + zrshift;
                            ind_G2n = nzero_G2n*nbinszr + zrshift;
                            ind_Wn = nzero_Wn*nbinszr + zrshift;
                            totcounts[zrshift] += w1*w2*dist; 
                            totnorms[zrshift] += w1*w2; 
                            gggg_fill_gnwn(nextGns, nextG2ns_gg, nextG2ns_ggc,
                                            nextWns, nextW2ns, nextW3ns, nextG3ns_ggg, nextG3ns_gggc,
                                            2*nmax_alloc, nbinszr, zrshift, ind_Gn, ind_G2n, ind_Wn,
                                            w2, w2_sq, wshape2, wshape_sq, wshape_cube,
                                            wshapewshapec, wshapesqwshapec, phirot, phirotc, fourphirotc);
                        }
                    }
                }
                time2 = omp_get_wtime();
                if ((verbose>1) && (elregion%nregions_skip_print == 0)&&(thisthread==0)&&(ind_inpix1==lower1)){
                    printf("Computed Gn for first gal in region %d/%d for thetabatch %d/%d in %.4f seconds\n",
                           elregion,nregions,elthetbatch,nthetbatches,(time2-time1));}                
                
                // Allocate Upsilon; have shape (ncomp,(2*nmax_alloc+1),(2*nmax_alloc+1),nthetas)
                time1 = omp_get_wtime();

                int thisn2, thisn3, thisn, thisnshift;
                int thisGshift_mn2m2, thisGshift_n2m2, thisWshift_n2;
                int thisGshift_mn3m3, thisGshift_mn3m1, thisGshift_n3m3, thisGshift_n3m1, thisWshift_n3;
                int thisGshift_mn2mn3m3, thisGshift_mn2mn3m1, thisGshift_n2n3m3, thisGshift_n2n3m1, thisWshift_n2n3;
                wshape1 = w1 * (e11+I*e12);  
                wshape1c = conj(wshape1);
                for (int nindex=0; nindex<len_nindices; nindex++){
                    thisn2 = nindices[nindex]/nnvals_Upsn - nzero_Ups;
                    thisn3 = nindices[nindex]%nnvals_Upsn - nzero_Ups;
                    if (thisn2>nzero_Ups || -thisn2>nzero_Ups || thisn3>nzero_Ups || -thisn3>nzero_Ups){
                        if (elregion==0 && elthetbatch==0){
                            printf("Error at elregion=%d batch=%d nindex=%d: nindices[nindex]=%d n2=%d n3=%d",
                                   elregion, elthetbatch, nindex, nindices[nindex], thisn2, thisn3);}
                        continue;
                    }
                        
                    thisn = thisn2+thisn3;
                    if ((verbose>1) && (elregion==0 && elthetbatch==0)){printf("nindex %d: n2=%d n3=%d\n",nindex,thisn2,thisn3);}
                    thisGshift_mn2m2 = (nzero_Gn-thisn2-2)*nbinsr;
                    thisGshift_n2m2 = (nzero_Gn+thisn2-2)*nbinsr;
                    thisWshift_n2 = (nzero_Wn+thisn2)*nbinsr;
                    thisGshift_mn3m3 = (nzero_Gn-thisn3-3)*nbinsr;
                    thisGshift_mn3m1 = (nzero_Gn-thisn3-1)*nbinsr;
                    thisGshift_n3m3 = (nzero_Gn+thisn3-3)*nbinsr;
                    thisGshift_n3m1 = (nzero_Gn+thisn3-1)*nbinsr;
                    thisWshift_n3 = (nzero_Wn+thisn3)*nbinsr;
                    thisGshift_mn2mn3m3 = (nzero_Gn-thisn-3)*nbinsr;
                    thisGshift_mn2mn3m1 = (nzero_Gn-thisn-1)*nbinsr;
                    thisGshift_n2n3m3 = (nzero_Gn+thisn-3)*nbinsr;
                    thisGshift_n2n3m1 = (nzero_Gn+thisn-1)*nbinsr;
                    thisWshift_n2n3 = (nzero_Wn+thisn)*nbinsr;
                    thisnshift = ((thisn2+nzero_Ups)*nnvals_Upsn + (thisn3+nzero_Ups)) * batchups_nshift;
                    gggg_accum_batchUpsilon(batchUpsilon_n, batchN_n,
                                            batch_nthetas, batchups_compshift, thisnshift,
                                            elb1s_batch, elb2s_batch, elb3s_batch,
                                            nextGns, nextG2ns_gg, nextG2ns_ggc, nextG3ns_ggg, nextG3ns_gggc,
                                            nextWns, nextW2ns, nextW3ns,
                                            wshape1, wshape1c, w1,
                                            nbinsr, nzero_G2n, nzero_Wn, thisn2, thisn3,
                                            thisGshift_mn2m2, thisGshift_n2m2, thisWshift_n2,
                                            thisGshift_mn3m3, thisGshift_mn3m1, thisGshift_n3m3, thisGshift_n3m1,
                                            thisWshift_n3, thisGshift_mn2mn3m3, thisGshift_mn2mn3m1,
                                            thisGshift_n2n3m3, thisGshift_n2n3m1, thisWshift_n2n3);
                }
                time2 = omp_get_wtime();
                if ((verbose>1) && (elregion%nregions_skip_print == 0)&&(thisthread==0)&&(ind_inpix1==lower1)){
                    printf("Allocated Ups for first gal in region %d/%d for thetabatch %d/%d in %.4f seconds for %d theta-combis\n",
                           elregion,nregions,elthetbatch,nthetbatches,(time2-time1),batch_nthetas);}
            }
            if ((verbose>1) && (elregion%nregions_skip_print == 0)&&(thisthread==0)){
                printf("Done region %d/%d for thetabatch %d/%d\n",elregion,nregions,elthetbatch,nthetbatches);}
        }
        
        // Get bin centers
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            if (totnorms[elbinr] != 0){
                // Note that the bin centers are the same for every batch!
                bin_centers_batch[elbinr] = totcounts[elbinr]/totnorms[elbinr]; 
                if (elthetbatch==0){bin_centers[elbinr] = bin_centers_batch[elbinr];} // Debug
            }
        }
        
        // For each theta combination (theta1,theta2,theta3) in this batch 
        // 1) Get bin edges and bin centers of the combinations
        // 2) Find all (theta1,theta2,theta3) combis that can be reconstructed via the symmetries
        //   2a) Get the Gamma_mu(theta1,theta2,theta3,phi12,phi13)
        //   2b) Transform the Gamma_mu to the target basis
        //   2c) Update the aperture Map^4 integral
        int ntrafos;
        double complex *nextM4correlators = orpheus_calloc(8, sizeof(double complex));
        double complex *thisUpsilon_n = orpheus_calloc(8*n2n3combis, sizeof(double complex));
        double complex *thisN_n = orpheus_calloc(n2n3combis, sizeof(double complex));
        double complex *thisUpsilon_n_rec = orpheus_calloc(8*n2n3combis_rec, sizeof(double complex));
        double complex *thisN_n_rec = orpheus_calloc(n2n3combis_rec, sizeof(double complex));
        double complex *thisnpcf = orpheus_calloc(8*batchgamma_thetshift, sizeof(double complex));
        double complex *thisnpcf_norm = orpheus_calloc(batchgamma_thetshift, sizeof(double complex));
        for (int elb=0; elb<batch_nthetas; elb++){
            if ((verbose>1) && (thisthread==0)){
                printf("Done %.4f per cent of multipole-to-Map4 conversion\r",100.* (float) elb/batch_nthetas);}
            // 1)
            int nbshift, elb1, elb2, elb3, elb1t, elb2t, elb3t;
            elb1 = elb1s_batch[elb];
            elb2 = elb2s_batch[elb];
            elb3 = elb3s_batch[elb];
            int bincombi_trafos[6][3];
            // 2)
            ntrafos = build_bincombi_trafos(elb1, elb2, elb3, bincombi_trafos);
            for (int eltrafo=0; eltrafo<ntrafos; eltrafo++){
                elb1t = bincombi_trafos[eltrafo][0];
                elb2t = bincombi_trafos[eltrafo][1];
                elb3t = bincombi_trafos[eltrafo][2];
                //printf("elb1=%d eln2=%d elb3=%d: eltrafo=%d/%d\n",elb1,elb2,elb3,eltrafo,ntrafos+1);
                // 2a)
                for (int eln12=0; eln12<n2n3combis; eln12++){
                    nbshift = eln12*batchups_nshift+elb;
                    for (int elcomp=0; elcomp<8; elcomp++){
                        thisUpsilon_n[elcomp*n2n3combis+eln12] = batchUpsilon_n[elcomp*batchups_compshift+nbshift];
                    }
                    thisN_n[eln12] = batchN_n[nbshift];
                }
                getMultipolesFromSymm(thisUpsilon_n, thisN_n, nmax, eltrafo, nindices, len_nindices,
                                       thisUpsilon_n_rec, thisN_n_rec);
                // OPTIONAL: Allocate 4PCF in multipole basis
                for (int eln12=0; eln12<n2n3combis_rec; eln12++){
                    if (alloc_4pcfmultipoles==1){
                        int thisnrshift = eln12*ups_nshift + elb1t*nbinsr*nbinsr + elb2t*nbinsr + elb3t;
                        for (int elcomp=0; elcomp<8; elcomp++){
                            Upsilon_n[elcomp*ups_rec_compshift+thisnrshift] = 
                                    thisUpsilon_n_rec[elcomp*n2n3combis_rec+eln12];
                        }
                        N_n[thisnrshift] = thisN_n_rec[eln12];
                    }
                }
                // 2b)
                // Only required for the real-space 4pcf and for the aperture integration
                if ((alloc_4pcfreal==1) || (nmapradii>0)){
                    multipoles2npcf_gggg_singletheta(thisUpsilon_n_rec, thisN_n_rec, nmax, nmax,
                                                     elb1t, elb2t, elb3t,
                                                     phibins, phibins, nbinsphi, nbinsphi,
                                                     projection, fourth->count_floor,
                                                     thisnpcf, thisnpcf_norm);
                }

                // OPTIONAL: Allocate 4pcf in real basis (Shape: (8,ntheta,ntheta,ntheta,nphi,nphi)
                if (alloc_4pcfreal==1){
                    for (int elphi12=0; elphi12<batchgamma_thetshift; elphi12++){
                        int gamma_rshift = nbinsphi*nbinsphi;
                        int gamma_phircombi = gamma_rshift*(elb1t*nbinsr*nbinsr+elb2t*nbinsr+elb3t)+elphi12;
                        int gamma_compshift = nbinsr*nbinsr*nbinsr*gamma_rshift;
                        for (int elcomp=0; elcomp<8; elcomp++){
                            Gammas[elcomp*gamma_compshift+gamma_phircombi] = thisnpcf[elcomp*batchgamma_thetshift+elphi12];
                        }
                        Norms[gamma_phircombi] = thisnpcf_norm[elphi12];
                    }
                }

                // 2c)
                double y1, y2, y3, dy1, dy2, dy3;
                int map4ind;
                int map4threadshift = thisthread*8*nmapradii;
                for (int elmapr=0; elmapr<nmapradii; elmapr++){
                    y1=bin_centers_batch[elb1t]/mapradii[elmapr];
                    y2=bin_centers_batch[elb2t]/mapradii[elmapr];
                    y3=bin_centers_batch[elb3t]/mapradii[elmapr];
                    dy1 = (bin_edges[elb1t+1]-bin_edges[elb1t])/mapradii[elmapr];
                    dy2 = (bin_edges[elb2t+1]-bin_edges[elb2t])/mapradii[elmapr];
                    dy3 = (bin_edges[elb3t+1]-bin_edges[elb3t])/mapradii[elmapr];
                    fourpcf2M4correlators(1,
                                          y1, y2, y3, dy1, dy2, dy3,
                                          phibins, phibins, dbinsphi, dbinsphi, nbinsphi, nbinsphi,
                                          thisnpcf, nextM4correlators);
                    for (int elcomp=0; elcomp<8; elcomp++){
                        map4ind = elcomp*nmapradii+elmapr;
                        if (isfinite(cabs(nextM4correlators[elcomp]))){
                            allM4correlators[map4threadshift+map4ind] += nextM4correlators[elcomp];
                        }
                        nextM4correlators[elcomp] = 0;
                    }
                }

                // Reset 4pcf placeholders to zero
                for (int i=0; i<batchgamma_thetshift; i++){
                    thisnpcf_norm[i] = 0;
                    for (int elcomp=0; elcomp<8; elcomp++){
                        thisnpcf[elcomp*batchgamma_thetshift+i] = 0;
                    }
                }
                for (int i=0; i<n2n3combis; i++){
                    thisN_n[i] = 0;
                    for (int elcomp=0; elcomp<8; elcomp++){
                        thisUpsilon_n[elcomp*n2n3combis+i] = 0;
                    }
                }
                for (int i=0; i<n2n3combis_rec; i++){
                    thisN_n_rec[i] = 0;
                    for (int elcomp=0; elcomp<8; elcomp++){
                        thisUpsilon_n_rec[elcomp*n2n3combis_rec+i] = 0;
                    }
                }
            }
        }
        
        if (verbose>1){
            for (int elmapr=0; elmapr<nmapradii; elmapr++){
                for (int elcomp=0; elcomp<8; elcomp++){
                    int map4ind = elcomp*nmapradii+elmapr;
                    int map4threadshift = thisthread*8*nmapradii;
                    if (verbose>1){ printf("\nthread %d, elr %d, elcomp %d, allM4cont=%.20f ",
                                thisthread, elmapr, elcomp, creal(allM4correlators[map4threadshift+map4ind])); }
                }
            }
        }
        if (verbose>1){printf("Done allocating 4pcfs for thetabatch %d/%d\n",elthetbatch,nthetbatches);}
            
        free(rshift_index_matcher_hash);
        free(rshift_pixs_galind_bounds);
        free(rshift_pix_gals);
            
        free(totcounts);
        free(totnorms);
        free(bin_centers_batch);
        free(batch_thetas1);
        free(batch_thetas2);
        free(batch_thetas3);
        free(batchUpsilon_n);
        free(batchN_n);
        free(batchfourpcf);
        free(batchfourpcf_norm);
        
        free(nextGns);
        free(nextG2ns_gg);
        free(nextG2ns_ggc);
        free(nextG3ns_ggg);
        free(nextG3ns_gggc);
        free(nextWns);
        free(nextW2ns);
        free(nextW3ns);
        
        free(elb1s_batch);
        free(elb2s_batch);
        free(elb3s_batch);
        free(bin_edges);
        
        free(nextM4correlators);
        free(thisUpsilon_n);
        free(thisN_n);
        free(thisUpsilon_n_rec);
        free(thisN_n_rec);
        free(thisnpcf);
        free(thisnpcf_norm);                
    }
    
    // Accummulate the Map^4 integral
    for (int elthread=0; elthread<nthreads; elthread++){
        int map4ind;
        int map4threadshift = elthread*8*nmapradii;
        for (int elcomp=0; elcomp<8; elcomp++){
            for (int elmapr=0; elmapr<nmapradii; elmapr++){
                map4ind = elcomp*nmapradii+elmapr;
                M4correlators[map4ind] += allM4correlators[map4threadshift+map4ind];
            }
        }
    }    
    if (verbose>0){ printf("\n"); }
    free(allM4correlators);
}


/////////////////////////////
// GNNN CORRELATOR CLASSES //
/////////////////////////////

void alloc_notomoGammans_discrete_gnnn(const MultiresoCatalog *cat_source, const NavHash *nav_source,
                                       const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                                       const BinningParams *bin, const FourthParams *fourth,
                                       int nthreads, int verbose, NPCFOutput *out){
    // Dereference input structs
    double *isinner_source = cat_source->isinner_resos, *weight_source = cat_source->weight_resos;
    double *pos1_source = cat_source->pos1_resos, *pos2_source = cat_source->pos2_resos;
    double *e1_source = cat_source->e1_resos, *e2_source = cat_source->e2_resos;
    int ngal_source = cat_source->ngal_resos[0];
    double *weight_lens = cat_lens->weight_resos, *pos1_lens = cat_lens->pos1_resos, *pos2_lens = cat_lens->pos2_resos;
    int *index_matcher_lens = nav_lens->index_matcher, *pixs_galind_bounds_lens = nav_lens->pixs_galind_bounds, *pix_gals_lens = nav_lens->pix_gals;
    double pix1_start = nav_lens->pix1_start, pix1_d = nav_lens->pix1_d; int pix1_n = nav_lens->pix1_n;
    double pix2_start = nav_lens->pix2_start, pix2_d = nav_lens->pix2_d; int pix2_n = nav_lens->pix2_n;
    int nmax = bin->nmax, nbinsr = bin->nbinsr, dccorr = bin->dccorr;
    double rmin = bin->rmin, rmax = bin->rmax;
    double *bin_centers = out->bin_centers;
    double complex *Gtilde_n = out->npcf, *N_n = out->norm_mp;

    int thisnshift, thisnrshift, thisthreadnrshift;
    int _nnvals_Upsn = 2*nmax+1;
    int _threadshift = _nnvals_Upsn*_nnvals_Upsn*nbinsr*nbinsr*nbinsr;

    double complex *allGtilden = orpheus_calloc(nthreads*_threadshift, sizeof(double complex));
    double complex *allNormn = orpheus_calloc(nthreads*_threadshift, sizeof(double complex));
    double *allcounts = orpheus_calloc(nthreads*nbinsr, sizeof(double));
    double *allnorms = orpheus_calloc(nthreads*nbinsr, sizeof(double));
    double *totcounts = orpheus_calloc(nbinsr, sizeof(double));
    double *totnorms = orpheus_calloc(nbinsr, sizeof(double));
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){
        free(allGtilden); free(allNormn); free(allcounts); free(allnorms); free(totcounts);
        free(totnorms);
        return;
    }

    int nregionsdone = 0;
    reset_progress();
    #pragma omp parallel for num_threads(nthreads)
    for (int ind_gal=0; ind_gal<ngal_source; ind_gal++){
        int thisthread = omp_get_thread_num();
        double drbin = (log(rmax)-log(rmin))/(nbinsr);
        //if (isource%nthreads!=thisthread){continue;}
        //printf("Doing thetabatch %d/%d on thread %d\n",elthetbatch,nthetbatches,thisthread);
        int nbinsz = 1;
        int nbinszr = nbinsz*nbinsr;
        int nnvals_Wn = 4*nmax+3; // Need to cover [-n1-n2-1, n1+n2+1] (We use the Wn for both, the nominator and the denominator)
        int nnvals_W2n = 4*nmax+3;
        int nnvals_Gtilden = 2*nmax+1;  // Need tocover [-2*nmax,+2*nmax]
        int nzero_Gtilden = nmax;
        int nzero_Wn = 2*nmax+1;
        
        int Gtilde_nshift = nbinsr*nbinsr*nbinsr;
        int n2n3combis = nnvals_Gtilden*nnvals_Gtilden;
        int Gtilde_threadshift = thisthread*Gtilde_nshift*n2n3combis;

        double p11, p12, w1, e11, e12;
        double innergal;
        p11 = pos1_source[ind_gal];
        p12 = pos2_source[ind_gal];
        w1 = weight_source[ind_gal];
        e11 = e1_source[ind_gal];
        e12 = e2_source[ind_gal];
        innergal = isinner_source[ind_gal];
        if (innergal<1e-5){continue;}
        
        int ind_gal2;
        int lower, upper;
        double  p21, p22, w2, w2_sq, rel1, rel2, dist, dphi;
        double complex phirot, phirotc, twophirot;
    
        // Check how many ns we need for Gn
        // Gns have shape (nnvals, nbinsz, nbinsr)
        // where the ns are ordered as 
        // [-nmax_1-nmax_2-3, ..., nmax_1+nmax_2+3]
        double complex *nextWns =  orpheus_calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW2ns = orpheus_calloc(nnvals_W2n*nbinszr, sizeof(double complex));
        double complex *nextW3ns = orpheus_calloc(2*nbinszr, sizeof(double complex)); // [W3n_gtilde, W3n_norm]
        //for (int i=0;i<nnvals_Wn*nbinszr;i++){nextWns[i]=0;}
        //for (int i=0;i<nnvals_W2n*nbinszr;i++){nextW2ns[i]=0;}
        //for (int i=0;i<nnvals_W3n*nbinszr;i++){nextW3ns[i]=0;}

        int rbin, zrshift, ind_Wn;
        FLATCELL_FOREACH(
            index_matcher_lens, 0, pixs_galind_bounds_lens, 0,
            p11, p12, rmax, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, lower, upper){
            for (int ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                ind_gal2 = pix_gals_lens[ind_inpix];
                //#pragma omp critical
                {p21 = pos1_lens[ind_gal2];
                p22 = pos2_lens[ind_gal2];
                w2 = weight_lens[ind_gal2];}
                
                rel1 = p21 - p11;
                rel2 = p22 - p12;
                dist = sqrt(rel1*rel1 + rel2*rel2);
                if (dist < rmin || dist >= rmax) continue;
                rbin = (int) floor((log(dist)-log(rmin))/drbin);
                w2_sq = w2*w2;
                dphi = atan2(rel2,rel1);
                phirot = cexp(I*dphi);
                phirotc = conj(phirot);
                twophirot = phirot*phirot;
                zrshift = 0*nbinsr + rbin;
                ind_Wn = nzero_Wn*nbinszr + zrshift;
                allcounts[thisthread*nbinszr + zrshift] += w1*w2*dist; 
                allnorms[thisthread*nbinszr + zrshift]  += w1*w2; 
                nextW3ns[zrshift] += w2_sq*w2;
                nextW3ns[nbinszr+zrshift] += w2_sq*w2*conj(twophirot);                          
                nnnn_fill_wn(nextWns, nextW2ns, 2*nmax+1, nbinszr, ind_Wn, w2, w2_sq, phirot, phirotc);
            }
        }

        // Allocate Upsilon
        // Upsilon have shape 
        // (ncomp,(2*nmax+1),(2*nmax+1),nthetas)
        int thisnshift, thisnrshift;
        int thisWshift_n2, thisWshift_n3, thisWshift_n2pn3;
        double complex wshape1 = - w1 * (e11+I*e12);  
        for (int n2=-nmax; n2<=nmax; n2++){
            for (int n3=-nmax; n3<=nmax; n3++){
                double complex wWW, wNN;
                //if (elregion==0 && elthetbatch==0){printf("nindex %d: n2=%d n3=%d\n",nindex,thisn2,thisn3);}
                int thisWshift_mn2mn3m1 = (nzero_Wn-n2-n3-1)*nbinsr;
                int thisWshift_n2n3m1 = (nzero_Wn+n2+n3-1)*nbinsr;
                int thisWshift_mn2 = (nzero_Wn-n2)*nbinsr;
                int thisWshift_mn3m1 = (nzero_Wn-n3-1)*nbinsr;
                thisWshift_n2 = (nzero_Wn+n2)*nbinsr;
                thisWshift_n3 = (nzero_Wn+n3)*nbinsr;
                int thisWshift_mn3 = (nzero_Wn-n3)*nbinsr;
                thisWshift_n2pn3 = (nzero_Wn+n2+n3)*nbinsr;
                thisnshift = Gtilde_threadshift + ((n2+nzero_Gtilden)*nnvals_Gtilden + (n3+nzero_Gtilden)) * Gtilde_nshift;
                for (int elb1=0; elb1<nbinsr; elb1++){
                    // elb1 = elb2 = elb3
                    if (dccorr==1){
                        thisnrshift = thisnshift + elb1*nbinsr*nbinsr + elb1*nbinsr + elb1;
                        allGtilden[thisnrshift] += 2 * wshape1*nextW3ns[1*nbinsr+elb1];
                        allNormn[thisnrshift] += 2 * w1*nextW3ns[0*nbinsr+elb1];
                    }
                    for (int elb2=0; elb2<nbinsr; elb2++){
                        if (dccorr==1){
                            // elb1 = elb2
                            thisnrshift = thisnshift + elb1*nbinsr*nbinsr + elb1*nbinsr + elb2;
                            allGtilden[thisnrshift] -= wshape1*nextW2ns[(nzero_Wn+n3-1)*nbinsr+elb1]*nextWns[thisWshift_mn3m1+elb2];
                            allNormn[thisnrshift] -= w1*nextW2ns[(nzero_Wn+n3)*nbinsr+elb1]*conj(nextWns[thisWshift_n3+elb2]);
                            // elb1 = elb3
                            thisnrshift = thisnshift + elb1*nbinsr*nbinsr + elb2*nbinsr + elb1;
                            allGtilden[thisnrshift] -= wshape1*nextW2ns[(nzero_Wn+n2-2)*nbinsr+elb1]*nextWns[thisWshift_mn2+elb2];
                            allNormn[thisnrshift] -= w1*nextW2ns[(nzero_Wn+n2)*nbinsr+elb1]*nextWns[thisWshift_mn2+elb2];
                            //elb2 = elb3
                            thisnrshift = thisnshift + elb2*nbinsr*nbinsr + elb1*nbinsr + elb1;
                            allGtilden[thisnrshift] -= wshape1*nextW2ns[thisWshift_mn2mn3m1+elb1]*nextWns[thisWshift_n2n3m1+elb2];
                            allNormn[thisnrshift] -= w1*nextW2ns[(nzero_Wn-n2-n3)*nbinsr+elb1] * nextWns[thisWshift_n2pn3+elb2];
                        }
                        wWW = wshape1 * nextWns[thisWshift_n2n3m1+elb1] *  nextWns[thisWshift_mn2+elb2];
                        wNN =  w1 * nextWns[thisWshift_n2pn3+elb1] * conj(nextWns[thisWshift_n2+elb2]);
                        for (int elb3=0; elb3<nbinsr; elb3++){
                            thisnrshift = thisnshift + elb1*nbinsr*nbinsr + elb2*nbinsr + elb3;
                            allGtilden[thisnrshift] += wWW * nextWns[thisWshift_mn3m1+elb3];
                            allNormn[thisnrshift] += wNN * nextWns[thisWshift_mn3+elb3];
                        }
                    }
                }
            }
        }
        free(nextWns);
        free(nextW2ns);
        free(nextW3ns);

        #pragma omp atomic
        nregionsdone += 1;
        print_progress(nregionsdone, ngal_source, verbose);
    }
    if (verbose>0){ printf("\n"); }
    
    // Accumulate Upsilon_n and N_n
    // Given the openmp implementation this needs to be done sequentially...however,
    // as the threads will reach this step at different points in time, it will
    // most likely not be a severe bottleneck.        
    #pragma omp parallel for num_threads(nthreads)
    for (int thisn=0; thisn<_nnvals_Upsn*_nnvals_Upsn; thisn++){
        thisnshift = thisn*nbinsr*nbinsr*nbinsr;
        for (int elb1=0; elb1<nbinsr; elb1++){
            for (int elb2=0; elb2<nbinsr; elb2++){
                for (int elb3=0; elb3<nbinsr; elb3++){
                    for (int elthread=0; elthread<nthreads; elthread++){
                        thisnrshift = thisnshift + elb1*nbinsr*nbinsr + elb2*nbinsr + elb3;
                        thisthreadnrshift = elthread*_threadshift + thisnrshift;
                        Gtilde_n[thisnrshift] += allGtilden[thisthreadnrshift];
                        N_n[thisnrshift] += allNormn[thisthreadnrshift];
                    }
                }
            }
        }
    }
    if (verbose>1){ printf("\nDone parallel accumulation of Gtilden \n"); }

    // Accumulate the bin distances and weights
    for (int thisthread=0; thisthread<nthreads; thisthread++){
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            totcounts[elbinr] += allcounts[thisthread*nbinsr+elbinr];
            totnorms[elbinr] += allnorms[thisthread*nbinsr+elbinr];
        }
    }
    // Get bin centers
    for (int elbinr=0; elbinr<nbinsr; elbinr++){
        if (totnorms[elbinr] != 0){
            bin_centers[elbinr] = totcounts[elbinr]/totnorms[elbinr];
            if (verbose>1){ printf("%.3f ",bin_centers[elbinr]); }
        }
        if (verbose>1){ printf("-1 "); }
    }
    if (verbose>1){ printf("\nDone accumulation of bin centers \n"); }
    free(allcounts);
    free(allnorms);
    free(totcounts);
    free(totnorms);
    free(allGtilden);
    free(allNormn);
    if (verbose>1){ printf("\nDone freeing stuff. \n"); }
}

// Non-tomo 4pcf using tree-based estimator
void alloc_notomoGammans_tree_gnnn(const MultiresoCatalog *cat_source, const NavHash *nav_source,
                                   const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
                                   const TreeResoParams *tree, const BinningParams *bin,
                                   const FourthParams *fourth, int nthreads, int verbose, NPCFOutput *out){

    // Dereference input args
    int nresos = tree->nresos; double *reso_redges = tree->reso_redges;
    double *isinner_source = cat_source->isinner_resos, *weight_source = cat_source->weight_resos;
    double *pos1_source = cat_source->pos1_resos, *pos2_source = cat_source->pos2_resos;
    double *e1_source = cat_source->e1_resos, *e2_source = cat_source->e2_resos;
    double *weight_lens_resos = cat_lens->weight_resos, *pos1_lens_resos = cat_lens->pos1_resos, *pos2_lens_resos = cat_lens->pos2_resos;
    int *ngal_lens_resos = cat_lens->ngal_resos;
    int *pixs_galind_bounds_source = nav_source->pixs_galind_bounds, *pix_gals_source = nav_source->pix_gals;
    int *index_matcher_lens = nav_lens->index_matcher, *pixs_galind_bounds_lens = nav_lens->pixs_galind_bounds, *pix_gals_lens = nav_lens->pix_gals;
    int nregions = nav_source->nregions;
    double pix1_start = nav_lens->pix1_start, pix1_d = nav_lens->pix1_d; int pix1_n = nav_lens->pix1_n;
    double pix2_start = nav_lens->pix2_start, pix2_d = nav_lens->pix2_d; int pix2_n = nav_lens->pix2_n;
    int nmax = bin->nmax, nbinsr = bin->nbinsr, dccorr = bin->dccorr;
    double rmin = bin->rmin, rmax = bin->rmax;
    int nthetacombis = fourth->nthetacombis;
    int *nindices = fourth->nindices, len_nindices = fourth->len_nindices;
    double *bin_centers = out->bin_centers;
    double complex *Gtilde_n = out->npcf, *N_n = out->norm_mp;

    int n_cfs = 1;
    // Temporary arrays that are allocated in parallel and later reduced
    double *tmpwcounts = orpheus_calloc(nthreads*nbinsr, sizeof(double));
    double *tmpwnorms = orpheus_calloc(nthreads*nbinsr, sizeof(double));
    double complex *tmpUpsilon_n = orpheus_calloc(nthreads*len_nindices*nthetacombis, sizeof(double complex));
    double complex *tmpN_n = orpheus_calloc(nthreads*len_nindices*nthetacombis, sizeof(double complex));
    double *totcounts = orpheus_calloc(nbinsr, sizeof(double));
    double *totnorms = orpheus_calloc(nbinsr, sizeof(double));

    // Helper array that checks how many regions have been already computed
    int *regionsdone = orpheus_calloc(nregions, sizeof(int));
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){
        free(tmpwcounts); free(tmpwnorms); free(tmpUpsilon_n); free(tmpN_n); free(totcounts);
        free(totnorms); free(regionsdone);
        return;
    }
    int nregionsdone = 0;
    reset_progress();
    
    #pragma omp parallel for num_threads(nthreads)
    for (int elthread=0; elthread<nthreads; elthread++){

        int nregions_per_thread = nregions/nthreads;
        int nbinsz = 1;
        int nmax_alloc = 2*nmax+1;
        int nnvals_Wn = 4*nmax_alloc+3; // Need to cover [-n1-n2-1, n1+n2+1] (We use the Wn for both, the nominator and the denominator)
        int nnvals_W2n = 4*nmax_alloc+3;
        int nnvals_W3n = 2;
        int nnvals_Gtilden = 2*nmax_alloc+1;  // Need tocover [-2*nmax_alloc,+2*nmax_alloc]
        int nzero_Gtilden = nmax_alloc;
        int nzero_Wn = 2*nmax_alloc+1;

        int ups_compshift = len_nindices*nthetacombis;

        int nbinszr = nbinsz*nbinsr;
        double complex *nextWns =  orpheus_calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW2ns = orpheus_calloc(nnvals_W2n*nbinszr, sizeof(double complex));
        double complex *nextW3ns = orpheus_calloc(2*nbinszr, sizeof(double complex)); // [W3n_gtilde, W3n_norm]

        int npix_hash = pix1_n*pix2_n;
        int *rshift_index_matcher = orpheus_calloc(nresos, sizeof(int));
        int *rshift_pixs_galind_bounds = orpheus_calloc(nresos, sizeof(int));
        int *rshift_pix_gals = orpheus_calloc(nresos, sizeof(int));
        for (int elreso=1; elreso<nresos; elreso++){
            rshift_index_matcher[elreso] = rshift_index_matcher[elreso-1] + npix_hash;
            rshift_pixs_galind_bounds[elreso] = rshift_pixs_galind_bounds[elreso-1] + ngal_lens_resos[elreso-1]+1;
            rshift_pix_gals[elreso] = rshift_pix_gals[elreso-1] + ngal_lens_resos[elreso-1];
        }
        
        double drbin = (log(rmax)-log(rmin))/(nbinsr);

        for (int _elregion=0; _elregion<2*nregions; _elregion++){

            // Check if this thread needs to allocate the region. In the first pass we split the work evenly 
            // while in the second pass we just work on the next best region, s.t. the 'fast' threads will
            // steal work from the 'slow' threads.
            int wasdone = 0;
            if (_elregion<nregions){
                int nthread_target = mymin(_elregion/nregions_per_thread, nthreads-1);
                if (nthread_target!=elthread){continue;}
            }
            int elregion = _elregion%nregions;
            #pragma omp critical
            {   
                if (regionsdone[_elregion%nregions]==1){wasdone = 1;}
                else{
                    regionsdone[_elregion%nregions]=1;
                    nregionsdone+=1; 
                }
            }
            if (wasdone==1){continue;}
            int region_debug = mymin(1000,nregions-1);
            bool printregdbg = (verbose>1) && (elregion==region_debug);
            if (printregdbg){printf("Region %d is in thread %d (%i regions in total)\n",
                elregion,elthread,nregions);}

            int lower1, upper1;
            lower1 = pixs_galind_bounds_source[elregion];
            upper1 = pixs_galind_bounds_source[elregion+1];
            for (int ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                int ind_gal = pix_gals_source[ind_inpix1];
                double p11, p12, w1, e11, e12;
                double innergal;
                p11 = pos1_source[ind_gal];
                p12 = pos2_source[ind_gal];
                w1 = weight_source[ind_gal];
                e11 = e1_source[ind_gal];
                e12 = e2_source[ind_gal];
                innergal = isinner_source[ind_gal];
                if (innergal<1e-5){continue;}
                
                int ind_gal2;
                int lower, upper;
                double  p21, p22, w2, w2_sq, rel1, rel2, dist, dphi;
                double complex phirot, phirotc, twophirot;
            
                // Check how many ns we need for Gn
                // Gns have shape (nnvals, nbinsz, nbinsr)
                // where the ns are ordered as 
                // [-nmax_1-nmax_2-3, ..., nmax_1+nmax_2+3]
                for (int i=0; i<nnvals_Wn*nbinszr; i++){nextWns[i]=0;}
                for (int i=0; i<nnvals_W2n*nbinszr; i++){nextW2ns[i]=0;}
                for (int i=0; i<nnvals_W3n*nbinszr; i++){nextW3ns[i]=0;}
                for (int elreso=0; elreso<nresos; elreso++){
                    int rbin, zrshift, ind_Wn;
                    double rmin_reso = reso_redges[elreso];
                    double rmax_reso = reso_redges[elreso+1];
                    FLATCELL_FOREACH(
                        index_matcher_lens, rshift_index_matcher[elreso], pixs_galind_bounds_lens, rshift_pixs_galind_bounds[elreso],
                        p11, p12, rmax_reso, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, lower, upper){
                        for (int ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                            ind_gal2 = rshift_pix_gals[elreso] + pix_gals_lens[rshift_pix_gals[elreso]+ind_inpix];
                            //#pragma omp critical
                            {p21 = pos1_lens_resos[ind_gal2];
                            p22 = pos2_lens_resos[ind_gal2];
                            w2 = weight_lens_resos[ind_gal2];}
                            
                            rel1 = p21 - p11;
                            rel2 = p22 - p12;
                            dist = sqrt(rel1*rel1 + rel2*rel2);
                            if (dist < rmin_reso || dist >= rmax_reso) continue;
                            rbin = (int) floor((log(dist)-log(rmin))/drbin);
                            w2_sq = w2*w2;
                            dphi = atan2(rel2,rel1);
                            phirot = cexp(I*dphi);
                            phirotc = conj(phirot);
                            twophirot = phirot*phirot;
                            zrshift = 0*nbinsr + rbin;
                            ind_Wn = nzero_Wn*nbinszr + zrshift;
                            nextW3ns[zrshift] += w2_sq*w2;
                            nextW3ns[nbinszr+zrshift] += w2_sq*w2*conj(twophirot);                          
                            tmpwcounts[elthread*nbinszr+zrshift] += w1*w2*dist; 
                            tmpwnorms[elthread*nbinszr+zrshift] += w1*w2; 
                            nnnn_fill_wn(nextWns, nextW2ns, 2*nmax_alloc+1, nbinszr, ind_Wn, w2, w2_sq, phirot, phirotc);
                        }
                    }
                }
                
                // Allocate Upsilon
                // Upsilon have shape 
                // (ncomp,(2*nmax+1),(2*nmax+1),nthetas)
                int n2, n3, thisnshift, thisnrshift, elbcombi;
                double complex wshape1 = - w1 * (e11+I*e12);  
                for (int nindex=0; nindex<len_nindices; nindex++){
                    n2 = nindices[nindex]/nnvals_Gtilden - nzero_Gtilden;
                    n3 = nindices[nindex]%nnvals_Gtilden - nzero_Gtilden;
                    if (n2>nnvals_Gtilden || -n2>nnvals_Gtilden || n3>nnvals_Gtilden || -n3>nnvals_Gtilden){
                        if (elregion==0){
                            printf("Error at elregion=%d nindex=%d: nindices[nindex]=%d n2=%d n3=%d",
                                   elregion, nindex, nindices[nindex], n2, n3);}
                        continue;
                    }
                    //if (elregion==0 && elthetbatch==0){printf("nindex %d: n2=%d n3=%d\n",nindex,n2,n3);}
                    double complex wWW, wNN; 
                    int thisWshift_n2 = (nzero_Wn+n2)*nbinsr;
                    int thisWshift_mn2 = (nzero_Wn-n2)*nbinsr;
                    int thisWshift_n3 = (nzero_Wn+n3)*nbinsr;
                    int thisWshift_mn3 = (nzero_Wn-n3)*nbinsr;
                    int thisWshift_n2pn3 = (nzero_Wn+n2+n3)*nbinsr;
                    int thisWshift_mn2mn3 = (nzero_Wn-n2-n3)*nbinsr;
                    int thisWshift_mn2mn3m1 = (nzero_Wn-n2-n3-1)*nbinsr;
                    int thisWshift_n2pn3m1 = (nzero_Wn+n2+n3-1)*nbinsr;
                    int thisWshift_n2m2 = (nzero_Wn+n2-2)*nbinsr;
                    int thisWshift_n3m1 = (nzero_Wn+n3-1)*nbinsr;
                    int thisWshift_mn3m1 = (nzero_Wn-n3-1)*nbinsr;
                      
                    thisnshift = elthread*ups_compshift + nindex*nthetacombis;
                    double complex Gcorr123, Gcorr12, Gcorr13, Gcorr23, Ncorr123, Ncorr12, Ncorr13, Ncorr23;
                    elbcombi = 0;
                    for (int elb1=0; elb1<nbinsr; elb1++){
                        // Precomputations for inner loop
                        Gcorr123 = + 2 * wshape1*nextW3ns[1*nbinsr+elb1];
                        Ncorr123 = + 2 * w1*nextW3ns[0*nbinsr+elb1];
                        for (int elb2=elb1; elb2<nbinsr; elb2++){
                            // Precomputations for inner loop
                            Gcorr12 = - wshape1 * nextW2ns[thisWshift_n3m1+elb1];
                            Ncorr12 = - w1 * nextW2ns[thisWshift_n3+elb1];
                            Gcorr13 = - wshape1 * nextW2ns[thisWshift_n2m2+elb1];
                            Ncorr13 = - w1 * nextW2ns[thisWshift_n2+elb1];
                            Gcorr23 = - wshape1 *  nextW2ns[thisWshift_mn2mn3m1+elb2];
                            Ncorr23 = - w1 * nextW2ns[thisWshift_mn2mn3+elb2];
                            wWW = wshape1 * nextWns[thisWshift_n2pn3m1+elb1] * nextWns[thisWshift_mn2+elb2];
                            wNN = w1 * nextWns[thisWshift_n2pn3+elb1] * nextWns[thisWshift_mn2+elb2];
                            for (int elb3=elb2; elb3<nbinsr; elb3++){
                                thisnrshift = thisnshift + elbcombi;
                                // Multiple countig corrections
                                if ((elb1==elb2) && (elb1==elb3) && (dccorr==1)){
                                    tmpUpsilon_n[thisnrshift] += Gcorr123;
                                    tmpN_n[thisnrshift] += Ncorr123;
                                }
                                if ((elb1==elb2) && (dccorr==1)){
                                    tmpUpsilon_n[thisnrshift] += Gcorr12*nextWns[thisWshift_mn3m1+elb3];
                                    tmpN_n[thisnrshift] += Ncorr12*nextWns[thisWshift_mn3+elb3];
                                }
                                if ((elb1==elb3) && (dccorr==1)){
                                    tmpUpsilon_n[thisnrshift] += Gcorr13*nextWns[thisWshift_mn2+elb2];
                                    tmpN_n[thisnrshift] += Ncorr13*nextWns[thisWshift_mn2+elb2];
                                }
                                if ((elb2==elb3) && (dccorr==1)){
                                    tmpUpsilon_n[thisnrshift] += Gcorr23*nextWns[thisWshift_n2pn3m1+elb1];
                                    tmpN_n[thisnrshift] += Ncorr23*nextWns[thisWshift_n2pn3+elb1];
                                }
                                // Nominal allocation
                                tmpUpsilon_n[thisnrshift] += wWW * nextWns[thisWshift_mn3m1+elb3];
                                tmpN_n[thisnrshift] += wNN * nextWns[thisWshift_mn3+elb3];
                                elbcombi += 1;
                            }
                        }
                    }
                }
            }
            print_progress(nregionsdone, nregions, verbose);
        }
        free(nextWns);
        free(nextW2ns);
        free(nextW3ns);
        free(rshift_index_matcher);
        free(rshift_pixs_galind_bounds);
        free(rshift_pix_gals);
    }
    if (verbose>0){ printf("\n"); }

    // Accumulate Upsilon_n and N_n
    // 1) Build arrays that hold bin combis for b1<=b2<=b3
    // 2) Get bin edges and bin centers of the combinations
    // 3) Find all (theta1,theta2,theta3) combis that can be reconstructed via the symmetries
    // 4) Get the Gamma_mu(theta1,theta2,theta3,phi12,phi13)
    // 1)
    int elbcombi = 0;
    int *elb1_inds = orpheus_calloc(nthetacombis, sizeof(int));
    int *elb2_inds = orpheus_calloc(nthetacombis, sizeof(int));
    int *elb3_inds = orpheus_calloc(nthetacombis, sizeof(int));
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){
        free(tmpwcounts); free(tmpwnorms); free(tmpUpsilon_n); free(tmpN_n); free(totcounts);
        free(totnorms); free(regionsdone); free(elb1_inds); free(elb2_inds); free(elb3_inds);
        return;
    }
    for (int elb1=0; elb1<nbinsr; elb1++){
        for (int elb2=elb1; elb2<nbinsr; elb2++){
            for (int elb3=elb2; elb3<nbinsr; elb3++){
                elb1_inds[elbcombi] = elb1;
                elb2_inds[elbcombi] = elb2;
                elb3_inds[elbcombi] = elb3;
                elbcombi += 1;
            }
        }
    }

    #pragma omp parallel for num_threads(nthreads)
    for (int elb=0; elb<nthetacombis; elb++){

        int ntrafos, tnrshift, nbshift, elb1, elb2, elb3, elb1t, elb2t, elb3t;
        int thisn2, thisn3;
        int nmax_alloc = 2*nmax+1;
        int nnvals_Upsn_rec = 2*nmax+1; 
        int nnvals_Upsn = 2*nmax_alloc+1; 
        int nzero_Ups = nmax_alloc;
        int ups_nshift = nbinsr*nbinsr*nbinsr;
        int n2n3combis = nnvals_Upsn*nnvals_Upsn;
        int n2n3combis_rec = nnvals_Upsn_rec*nnvals_Upsn_rec;
        int ups_rec_compshift = n2n3combis_rec*ups_nshift;

        double complex *thisUpsilon_n = orpheus_calloc(n_cfs*n2n3combis, sizeof(double complex));
        double complex *thisN_n = orpheus_calloc(n2n3combis, sizeof(double complex));
        double complex *thisUpsilon_n_rec = orpheus_calloc(n_cfs*n2n3combis_rec, sizeof(double complex));
        double complex *thisN_n_rec = orpheus_calloc(n2n3combis_rec, sizeof(double complex));

        // 2)
        elb1 = elb1_inds[elb];
        elb2 = elb2_inds[elb];
        elb3 = elb3_inds[elb];
        int bincombi_trafos[6][3];
        
        // 3)
        ntrafos = build_bincombi_trafos(elb1, elb2, elb3, bincombi_trafos);
        for (int eltrafo=0; eltrafo<ntrafos; eltrafo++){
            elb1t = bincombi_trafos[eltrafo][0];
            elb2t = bincombi_trafos[eltrafo][1];
            elb3t = bincombi_trafos[eltrafo][2];
            for (int nindex=0; nindex<len_nindices; nindex++){
                thisn2 = nindices[nindex]/nnvals_Upsn - nzero_Ups;
                thisn3 = nindices[nindex]%nnvals_Upsn - nzero_Ups;
                nbshift = ((thisn2+nzero_Ups)*nnvals_Upsn + (thisn3+nzero_Ups));
                for (int elthread=0; elthread<nthreads; elthread++){
                    tnrshift = elthread*len_nindices*nthetacombis + nindex*nthetacombis + elb;
                    thisUpsilon_n[0*n2n3combis+nbshift] += tmpUpsilon_n[tnrshift];
                    thisN_n[nbshift] += tmpN_n[tnrshift];
                }
            }

            getMultipolesFromSymm_GNNN(
                thisUpsilon_n, thisN_n, nmax, eltrafo, nindices, len_nindices,
                thisUpsilon_n_rec, thisN_n_rec);

            // 4)
            for (int eln12=0; eln12<n2n3combis_rec; eln12++){
                int thisnrshift = eln12*ups_nshift + elb1t*nbinsr*nbinsr + elb2t*nbinsr + elb3t;
                for (int elcomp=0; elcomp<n_cfs; elcomp++){
                    Gtilde_n[elcomp*ups_rec_compshift+thisnrshift] =  thisUpsilon_n_rec[elcomp*n2n3combis_rec+eln12];
                }
                N_n[thisnrshift] = thisN_n_rec[eln12];
            }  

            // Reset 4pcf placeholders to zero
            for (int i=0; i<n2n3combis; i++){
                thisN_n[i] = 0;
                for (int elcomp=0; elcomp<n_cfs; elcomp++){
                    thisUpsilon_n[elcomp*n2n3combis+i] = 0;
                }
            }
            for (int i=0; i<n2n3combis_rec; i++){
                thisN_n_rec[i] = 0;
                for (int elcomp=0; elcomp<n_cfs; elcomp++){
                    thisUpsilon_n_rec[elcomp*n2n3combis_rec+i] = 0;
                }
            }
        }
        free(thisUpsilon_n);
        free(thisUpsilon_n_rec);
        free(thisN_n);
        free(thisN_n_rec);
    }

    // Accumulate the bin distances and weights
    for (int thisthread=0; thisthread<nthreads; thisthread++){
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            totcounts[elbinr] += tmpwcounts[thisthread*nbinsr+elbinr];
            totnorms[elbinr] += tmpwnorms[thisthread*nbinsr+elbinr];
        }
    }
    // Get bin centers
    for (int elbinr=0; elbinr<nbinsr; elbinr++){
        if (totnorms[elbinr] != 0){
            bin_centers[elbinr] = totcounts[elbinr]/totnorms[elbinr];
        }
    }
    
    free(tmpUpsilon_n);
    free(tmpN_n);
    free(tmpwcounts);
    free(tmpwnorms);
    free(totcounts);
    free(totnorms);

    free(elb1_inds);
    free(elb2_inds);
    free(elb3_inds);

    free(regionsdone);
}
    
void alloc_notomoMapNap3_tree_gnnn(const MultiresoCatalog *cat_source, const NavHash *nav_source,
    const MultiresoCatalog *cat_lens, const NavHash *nav_lens,
    const TreeResoParams *tree, const BinningParams *bin, const FourthParams *fourth,
    const ClustCorr *clustcorr,
    double *apradii, int napradii, int alloc_4pcfmultipoles, int alloc_4pcfreal,
    int nthreads, int verbose,
    double *bin_centers, double complex *Gtilde_n, double complex *N_n,
    double complex *Gtilde, double complex *Norms, double complex *NM3correlator){

    // Dereference input args
    int nresos = tree->nresos; double *reso_redges = tree->reso_redges;
    double *isinner_source = cat_source->isinner_resos, *weight_source = cat_source->weight_resos;
    double *pos1_source = cat_source->pos1_resos, *pos2_source = cat_source->pos2_resos;
    double *e1_source = cat_source->e1_resos, *e2_source = cat_source->e2_resos;
    double *weight_lens_resos = cat_lens->weight_resos, *pos1_lens_resos = cat_lens->pos1_resos, *pos2_lens_resos = cat_lens->pos2_resos;
    int *ngal_lens_resos = cat_lens->ngal_resos;
    int *pixs_galind_bounds_source = nav_source->pixs_galind_bounds, *pix_gals_source = nav_source->pix_gals;
    int *index_matcher_lens = nav_lens->index_matcher, *pixs_galind_bounds_lens = nav_lens->pixs_galind_bounds, *pix_gals_lens = nav_lens->pix_gals;
    int nregions = nav_source->nregions;
    double pix1_start = nav_lens->pix1_start, pix1_d = nav_lens->pix1_d; int pix1_n = nav_lens->pix1_n;
    double pix2_start = nav_lens->pix2_start, pix2_d = nav_lens->pix2_d; int pix2_n = nav_lens->pix2_n;
    int nmax = bin->nmax, nbinsr = bin->nbinsr, dccorr = bin->dccorr;
    double rmin = bin->rmin, rmax = bin->rmax;
    int *nindices = fourth->nindices, len_nindices = fourth->len_nindices;
    double *phibins = fourth->phibins1, *dbinsphi = fourth->dbinsphi1; int nbinsphi = fourth->nbinsphi1;
    int *thetacombis_batches = fourth->thetacombis_batches, *nthetacombis_batches = fourth->nthetacombis_batches;
    int *cumthetacombis_batches = fourth->cumthetacombis_batches; int nthetbatches = fourth->nthetbatches;

    double complex *allNM3correlator = orpheus_calloc(nthreads*1*1*napradii, sizeof(double complex));
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){free(allNM3correlator); return;}
    int nregionsdone = 0;
    reset_progress();
    #pragma omp parallel for num_threads(nthreads)
    for (int elthetbatch=0; elthetbatch<nthetbatches; elthetbatch++){
        int nregions_skip_print = mymax(1, nregions / 100);

        // * nmax_alloc specifies the largest multipole that needs to be allocated when wanting 
        //   to allocate the Upsn/Nn while making use of the symmetry properties
        // * All quantities that are updated at the galaxy level are computed until nmax_alloc
        // * Once we are done iterating over the cat we apply the symmetries and allocate the
        //   reconstructed quantities having a suffix _rec
        int thisthread = omp_get_thread_num();
        //printf("Doing thetabatch %d/%d on thread %d\n",elthetbatch,nthetbatches,thisthread);
        int nmax_alloc = 2*nmax+1;
        int nbinsz = 1;
        int nnvals_Wn = 4*nmax_alloc+3; // Need to cover [-n1-n2-1, n1+n2+1] (We use the Wn for both, the nominator and the denominator)
        int nnvals_W2n = 4*nmax_alloc+3;
        int nnvals_W3n = 2;
        int nnvals_Gtilden = 2*nmax_alloc+1;  // Need tocover [-2*nmax_alloc,+2*nmax_alloc]
        int nnvals_Gtilden_rec = 2*nmax+1; // Need tocover [-nmax,+nmax]
        int nzero_Gtilden = nmax_alloc;
        int nzero_Wn = 2*nmax_alloc+1;
        
        int Gtilde_nshift = nbinsr*nbinsr*nbinsr;
        int n2n3combis = nnvals_Gtilden*nnvals_Gtilden;
        int n2n3combis_rec = nnvals_Gtilden_rec*nnvals_Gtilden_rec;
        
        int batch_nthetas = nthetacombis_batches[elthetbatch];
        int batchGtilde_nshift = batch_nthetas;
        int batchGtilde_compshift = n2n3combis*batchGtilde_nshift;
        int batchGtilde_thetshift = nbinsphi*nbinsphi;
        
        int npix_hash = pix1_n*pix2_n;
        int *rshift_index_matcher = orpheus_calloc(nresos, sizeof(int));
        int *rshift_pixs_galind_bounds = orpheus_calloc(nresos, sizeof(int));
        int *rshift_pix_gals = orpheus_calloc(nresos, sizeof(int));
        for (int elreso=1; elreso<nresos; elreso++){
            rshift_index_matcher[elreso] = rshift_index_matcher[elreso-1] + npix_hash;
            rshift_pixs_galind_bounds[elreso] = rshift_pixs_galind_bounds[elreso-1] + ngal_lens_resos[elreso-1]+1;
            rshift_pix_gals[elreso] = rshift_pix_gals[elreso-1] + ngal_lens_resos[elreso-1];
        }

        double *totcounts = orpheus_calloc(nbinsr, sizeof(double));
        double *totnorms = orpheus_calloc(nbinsr, sizeof(double));
        double *bin_centers_batch = orpheus_calloc(nbinsr, sizeof(double));
        double complex *batchGtilde_n = orpheus_calloc(batchGtilde_compshift, sizeof(double complex));
        double complex *batchN_n = orpheus_calloc(batchGtilde_compshift, sizeof(double complex));
        double *batch_thetas1 = orpheus_calloc(batch_nthetas, sizeof(double));
        double *batch_thetas2 = orpheus_calloc(batch_nthetas, sizeof(double));
        double *batch_thetas3 = orpheus_calloc(batch_nthetas, sizeof(double));
        
        int nbinszr = nbinsz*nbinsr;
        double complex *nextWns =  orpheus_calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW2ns = orpheus_calloc(nnvals_W2n*nbinszr, sizeof(double complex));
        double complex *nextW3ns = orpheus_calloc(2*nbinszr, sizeof(double complex)); // [W3n_gtilde, W3n_norm]
        
        double drbin = (log(rmax)-log(rmin))/(nbinsr);
        int rbin_min_batch=nbinsr;int rbin_max_batch=0;
        int reso_min_batch=0; int reso_max_batch=0;
        int *elb1s_batch = orpheus_calloc(batch_nthetas, sizeof(int));
        int *elb2s_batch = orpheus_calloc(batch_nthetas, sizeof(int));
        int *elb3s_batch = orpheus_calloc(batch_nthetas, sizeof(int));
        double *bin_edges = orpheus_calloc(nbinsr+1, sizeof(double));
        #pragma omp critical
        {
            build_thetabatch(elthetbatch, batch_nthetas, nbinsr, nresos, rmin, rmax,
                             thetacombis_batches, cumthetacombis_batches, reso_redges,
                             elb1s_batch, elb2s_batch, elb3s_batch, bin_edges,
                             &rbin_min_batch, &rbin_max_batch, &reso_min_batch, &reso_max_batch);
        }
        
        // Allocate the 4pcf multipoles for this batch of radii 
        int offset_per_thread = nregions/nthreads;
        int offset = offset_per_thread*thisthread;
        for (int _elregion=0; _elregion<nregions; _elregion++){
            int elregion = (_elregion+offset)%nregions; // Try to evade collisions
            #pragma omp atomic
            nregionsdone += 1;
            print_progress(nregionsdone, nthetbatches*nregions, verbose);
            //int region_debug = mymin(500,nregions-1);
            int lower1, upper1;
            lower1 = pixs_galind_bounds_source[elregion];
            upper1 = pixs_galind_bounds_source[elregion+1];
            for (int ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                double time1, time2;
                time1 = omp_get_wtime();
                int ind_gal = pix_gals_source[ind_inpix1];
                double p11, p12, w1, e11, e12;
                double innergal;
                p11 = pos1_source[ind_gal];
                p12 = pos2_source[ind_gal];
                w1 = weight_source[ind_gal];
                e11 = e1_source[ind_gal];
                e12 = e2_source[ind_gal];
                innergal = isinner_source[ind_gal];
                if (innergal<1e-5){continue;}
                
                int ind_gal2;
                int lower, upper;
                double  p21, p22, w2, w2_sq, rel1, rel2, dist, dphi;
                double complex phirot, phirotc, twophirot;
            
                // Check how many ns we need for Gn
                // Gns have shape (nnvals, nbinsz, nbinsr)
                // where the ns are ordered as 
                // [-nmax_1-nmax_2-3, ..., nmax_1+nmax_2+3]
                for (int i=0; i<nnvals_Wn*nbinszr; i++){nextWns[i]=0;}
                for (int i=0; i<nnvals_W2n*nbinszr; i++){nextW2ns[i]=0;}
                for (int i=0; i<nnvals_W3n*nbinszr; i++){nextW3ns[i]=0;}
                for (int elreso=reso_min_batch; elreso<=reso_max_batch; elreso++){
                    int rbin, zrshift, ind_Wn;
                    double rmin_reso = reso_redges[elreso];
                    double rmax_reso = reso_redges[elreso+1];
                    FLATCELL_FOREACH(
                        index_matcher_lens, rshift_index_matcher[elreso], pixs_galind_bounds_lens, rshift_pixs_galind_bounds[elreso],
                        p11, p12, rmax_reso, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, lower, upper){
                        for (int ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                            ind_gal2 = rshift_pix_gals[elreso] + pix_gals_lens[rshift_pix_gals[elreso]+ind_inpix];
                            //#pragma omp critical
                            {p21 = pos1_lens_resos[ind_gal2];
                            p22 = pos2_lens_resos[ind_gal2];
                            w2 = weight_lens_resos[ind_gal2];}
                            
                            rel1 = p21 - p11;
                            rel2 = p22 - p12;
                            dist = sqrt(rel1*rel1 + rel2*rel2);
                            if (dist < rmin_reso || dist >= rmax_reso) continue;
                            rbin = (int) floor((log(dist)-log(rmin))/drbin);
                            w2_sq = w2*w2;
                            dphi = atan2(rel2,rel1);
                            phirot = cexp(I*dphi);
                            phirotc = conj(phirot);
                            twophirot = phirot*phirot;
                            zrshift = 0*nbinsr + rbin;
                            ind_Wn = nzero_Wn*nbinszr + zrshift;
                            nextW3ns[zrshift] += w2_sq*w2;
                            nextW3ns[nbinszr+zrshift] += w2_sq*w2*twophirot;                          
                            totcounts[zrshift] += w1*w2*dist; 
                            totnorms[zrshift] += w1*w2; 
                            nnnn_fill_wn(nextWns, nextW2ns, 2*nmax_alloc+1, nbinszr, ind_Wn, w2, w2_sq, phirot, phirotc);
                        }
                    }
                }
                time2 = omp_get_wtime();
                if ((verbose>1) && (elregion%nregions_skip_print == 0)&&(thisthread==0)&&(ind_inpix1==lower1)){
                    printf("Computed Wn for first gal in region %d/%d for thetabatch %d/%d in %.4f seconds\n",
                           elregion,nregions,elthetbatch,nthetbatches,(time2-time1));}                
                
                // Allocate Upsilon; have shape (ncomp,(2*nmax_alloc+1),(2*nmax_alloc+1),nthetas)
                time1 = omp_get_wtime();
                int thisn2, thisn3, thisnshift, thisnrshift, elb1, elb2, elb3;
                int thisWshift_n2, thisWshift_n3, thisWshift_mn2, thisWshift_mn3m1;
                int thisWshift_n2pn3, thisWshift_n2n3m1, thisWshift_mn2mn3m1;
                double complex wshape1 = - w1 * (e11+I*e12);
                for (int nindex=0; nindex<len_nindices; nindex++){
                    thisn2 = nindices[nindex]/nnvals_Gtilden - nzero_Gtilden;
                    thisn3 = nindices[nindex]%nnvals_Gtilden - nzero_Gtilden;
                    if (thisn2>nzero_Gtilden || -thisn2>nzero_Gtilden || thisn3>nzero_Gtilden || -thisn3>nzero_Gtilden){
                        if (elregion==0 && elthetbatch==0){
                            printf("Error at elregion=%d batch=%d nindex=%d: nindices[nindex]=%d n2=%d n3=%d",
                                   elregion, elthetbatch, nindex, nindices[nindex], thisn2, thisn3);}
                        continue;
                    }
                    //if (elregion==0 && elthetbatch==0){printf("nindex %d: n2=%d n3=%d\n",nindex,thisn2,thisn3);}
                    // Same convention as alloc_notomoGammans_discrete_gnnn:
                    // Upsilon_n = wshape1 * W_{n2+n3-1}(b1) W_{-n2}(b2) W_{-n3-1}(b3)
                    thisWshift_n2n3m1 = (nzero_Wn+thisn2+thisn3-1)*nbinsr;
                    thisWshift_mn2 = (nzero_Wn-thisn2)*nbinsr;
                    thisWshift_mn3m1 = (nzero_Wn-thisn3-1)*nbinsr;
                    thisWshift_mn2mn3m1 = (nzero_Wn-thisn2-thisn3-1)*nbinsr;
                    thisWshift_n2 = (nzero_Wn+thisn2)*nbinsr;
                    thisWshift_n3 = (nzero_Wn+thisn3)*nbinsr;
                    thisWshift_n2pn3 = (nzero_Wn+thisn2+thisn3)*nbinsr;
                    thisnshift = ((thisn2+nzero_Gtilden)*nnvals_Gtilden + (thisn3+nzero_Gtilden)) * batchGtilde_nshift;
                    for (int elb=0; elb<batch_nthetas; elb++){
                        elb1 = elb1s_batch[elb];
                        elb2 = elb2s_batch[elb];
                        elb3 = elb3s_batch[elb];
                        thisnrshift = thisnshift + elb;
                        // Multiple counting corrections:
                        // sum_(i neq j neq k) = sum_(i,j,k) - ( sum_(i, j, i=k) + 2perm ) + 2 * sum_(i, i=j, i=k)
                        // Triple-counting corr
                        if ((elb1==elb2) && (elb1==elb3) && (elb2==elb3) && (dccorr==1)){
                            batchGtilde_n[thisnrshift] += 2 * wshape1*nextW3ns[1*nbinsr+elb1];
                            batchN_n[thisnrshift] += 2 * w1*nextW3ns[0*nbinsr+elb1];
                        }
                        // Double-counting corr for theta1==theta2
                        if ((elb1==elb2) && (dccorr==1)){
                            batchGtilde_n[thisnrshift] -= wshape1*nextW2ns[(nzero_Wn+thisn3-1)*nbinsr+elb1]*nextWns[thisWshift_mn3m1+elb3];
                            batchN_n[thisnrshift] -= w1*nextW2ns[(nzero_Wn+thisn3)*nbinsr+elb1]*conj(nextWns[thisWshift_n3+elb3]);
                        }
                        // Double-counting corr for theta1==theta3
                        if ((elb1==elb3) && (dccorr==1)){
                            batchGtilde_n[thisnrshift] -= wshape1*nextW2ns[(nzero_Wn+thisn2-2)*nbinsr+elb1]*nextWns[thisWshift_mn2+elb2];
                            batchN_n[thisnrshift] -= w1*nextW2ns[(nzero_Wn+thisn2)*nbinsr+elb1]*conj(nextWns[thisWshift_n2+elb2]);
                        }
                        // Double-counting corr for theta2==theta3
                        if ((elb2==elb3) && (dccorr==1)){
                            batchGtilde_n[thisnrshift] -= wshape1*nextW2ns[thisWshift_mn2mn3m1+elb2]*nextWns[thisWshift_n2n3m1+elb1];
                            batchN_n[thisnrshift] -= w1*nextW2ns[(nzero_Wn-thisn2-thisn3)*nbinsr+elb2] * nextWns[thisWshift_n2pn3+elb1];
                        }
                        // Nominal allocation
                        batchGtilde_n[thisnrshift] += wshape1 * nextWns[thisWshift_n2n3m1+elb1] *
                                                      nextWns[thisWshift_mn2+elb2] * nextWns[thisWshift_mn3m1+elb3];
                        batchN_n[thisnrshift] += w1 * nextWns[thisWshift_n2pn3+elb1] *
                                                 conj(nextWns[thisWshift_n2+elb2]) * conj(nextWns[thisWshift_n3+elb3]);
                    }
                }
                time2 = omp_get_wtime();
                if ((verbose>1) && (elregion%nregions_skip_print == 0)&&(thisthread==0)&&(ind_inpix1==lower1)){
                    printf("Allocated Gtilden for first gal in region %d/%d for thetabatch %d/%d in %.4f seconds for %d theta-combis\n",
                           elregion,nregions,elthetbatch,nthetbatches,(time2-time1),batch_nthetas);}
            }
            if ((verbose>1) && (elregion%nregions_skip_print == 0)&&(thisthread==0)){
                printf("Done region %d/%d for thetabatch %d/%d\n",elregion,nregions,elthetbatch,nthetbatches);}
        }
        
        // Get bin centers
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            if (totnorms[elbinr] != 0){
                // Note that the bin centers are the same for every batch!
                bin_centers_batch[elbinr] = totcounts[elbinr]/totnorms[elbinr]; 
                bin_centers[elbinr] = totcounts[elbinr]/totnorms[elbinr]; 
            }
        }
        
        // For each theta combination (theta1,theta2,theta3) in this batch 
        // 1) Get bin edges and bin centers of the combinations
        // 2) Find all (theta1,theta2,theta3) combis that can be reconstructed via the symmetries
        //   2a) Get the Gammatilde(theta1,theta2,theta3,phi12,phi13)
        //   2b) Transform the Gammatilde to the target basis & apply clustering correction
        //   2c) Update the aperture MapNap3 integral
        int ntrafos;
        double complex *nextNM3correlator = orpheus_calloc(1, sizeof(double complex));
        double complex *thisGtilde_n = orpheus_calloc(1*n2n3combis, sizeof(double complex));
        double complex *thisN_n = orpheus_calloc(n2n3combis, sizeof(double complex));
        double complex *thisGtilde_n_rec = orpheus_calloc(1*n2n3combis_rec, sizeof(double complex));
        double complex *thisN_n_rec = orpheus_calloc(n2n3combis_rec, sizeof(double complex));
        double complex *thisnpcf = orpheus_calloc(1*batchGtilde_thetshift, sizeof(double complex));
        double complex *thisnpcf_norm = orpheus_calloc(batchGtilde_thetshift, sizeof(double complex));
        for (int elb=0; elb<batch_nthetas; elb++){
            // 1)
            int nbshift, elb1, elb2, elb3, elb1t, elb2t, elb3t;
            elb1 = elb1s_batch[elb];
            elb2 = elb2s_batch[elb];
            elb3 = elb3s_batch[elb];
            int bincombi_trafos[6][3];
            // 2)
            ntrafos = build_bincombi_trafos(elb1, elb2, elb3, bincombi_trafos);
            for (int eltrafo=0; eltrafo<ntrafos; eltrafo++){
                elb1t = bincombi_trafos[eltrafo][0];
                elb2t = bincombi_trafos[eltrafo][1];
                elb3t = bincombi_trafos[eltrafo][2];
                //printf("elb1=%d eln2=%d elb3=%d: eltrafo=%d/%d\n",elb1,elb2,elb3,eltrafo,ntrafos+1);
                // 2a)
                for (int eln12=0; eln12<n2n3combis; eln12++){
                    nbshift = eln12*batchGtilde_nshift+elb;
                    thisGtilde_n[eln12] = batchGtilde_n[nbshift];
                    thisN_n[eln12] = batchN_n[nbshift];
                }
                getMultipolesFromSymm_GNNN(thisGtilde_n, thisN_n, nmax, eltrafo, nindices, len_nindices,
                                            thisGtilde_n_rec, thisN_n_rec);
                // OPTIONAL: Allocate 4PCF in multipole basis
                for (int eln12=0; eln12<n2n3combis_rec; eln12++){
                    if (alloc_4pcfmultipoles==1){
                        int thisnrshift = eln12*Gtilde_nshift + elb1t*nbinsr*nbinsr + elb2t*nbinsr + elb3t;
                        Gtilde_n[thisnrshift] = thisGtilde_n_rec[eln12];
                        N_n[thisnrshift] = thisN_n_rec[eln12];
                    }
                }
                // 2b) OPTIONAL: Allocate 4pcf in real basis (Shape: (1,ntheta,ntheta,ntheta,nphi,nphi)
                // The clustering correction is evaluated at the geometric bin centers
                // Only required for the real-space 4pcf and for the aperture integration
                if ((alloc_4pcfreal==1) || (napradii>0)){
                    multipoles2npcf_gnnn_singletheta(thisGtilde_n_rec, thisN_n_rec, nmax, nmax,
                                                     sqrt(bin_edges[elb1t]*bin_edges[elb1t+1]),
                                                     sqrt(bin_edges[elb2t]*bin_edges[elb2t+1]),
                                                     sqrt(bin_edges[elb3t]*bin_edges[elb3t+1]),
                                                     phibins, phibins, nbinsphi, nbinsphi,
                                                     clustcorr,
                                                     thisnpcf, thisnpcf_norm);
                }

                if (alloc_4pcfreal==1){
                    for (int elphi12=0; elphi12<batchGtilde_thetshift; elphi12++){
                        int Gtilde_rshift = nbinsphi*nbinsphi;
                        int Gtilde_phircombi = Gtilde_rshift*(elb1t*nbinsr*nbinsr+elb2t*nbinsr+elb3t)+elphi12;
                        Gtilde[Gtilde_phircombi] = thisnpcf[elphi12];
                        Norms[Gtilde_phircombi] = thisnpcf_norm[elphi12];
                    }
                }

                // 2c) Update MapNap3 integral
                // Filter evaluated at the geometric bin centers, consistent with
                double y1, y2, y3, dy1, dy2, dy3;
                int mapnap3threadshift = thisthread*napradii;
                for (int elapr=0; elapr<napradii; elapr++){
                    y1=sqrt(bin_edges[elb1t]*bin_edges[elb1t+1])/apradii[elapr];
                    y2=sqrt(bin_edges[elb2t]*bin_edges[elb2t+1])/apradii[elapr];
                    y3=sqrt(bin_edges[elb3t]*bin_edges[elb3t+1])/apradii[elapr];
                    dy1 = (bin_edges[elb1t+1]-bin_edges[elb1t])/apradii[elapr];
                    dy2 = (bin_edges[elb2t+1]-bin_edges[elb2t])/apradii[elapr];
                    dy3 = (bin_edges[elb3t+1]-bin_edges[elb3t])/apradii[elapr];
                    fourpcf2MN3correlator(
                         1, y1, y2, y3, dy1, dy2, dy3,
                         phibins, phibins, dbinsphi, dbinsphi, nbinsphi, nbinsphi, thisnpcf, nextNM3correlator);
                    if (isfinite(cabs(nextNM3correlator[0]))){
                        allNM3correlator[mapnap3threadshift+elapr] += nextNM3correlator[0];
                    }
                    //if (true){printf("\nthread %d, elr %d, allMN3cont=%.20f ", thisthread, elapr, creal(nextNM3correlator[0]));}
                    nextNM3correlator[0] = 0;
                }

                // Reset 4pcf placeholders to zero
                for (int i=0; i<batchGtilde_thetshift; i++){
                    thisnpcf_norm[i] = 0;
                    thisnpcf[i] = 0;
                }
                for (int i=0; i<n2n3combis; i++){
                    thisN_n[i] = 0;
                    thisGtilde_n[i] = 0;
                }
                for (int i=0; i<n2n3combis_rec; i++){
                    thisN_n_rec[i] = 0;
                    thisGtilde_n_rec[i] = 0;
                }
            }
        }
        
        if (verbose>1){printf("Done allocating 4pcfs for thetabatch %d/%d\n",elthetbatch,nthetbatches);}
            
        free(rshift_index_matcher);
        free(rshift_pixs_galind_bounds);
        free(rshift_pix_gals);
        free(totcounts);
        free(totnorms);
        free(bin_centers_batch);
        free(batchGtilde_n);
        free(batchN_n);
        free(batch_thetas1);
        free(batch_thetas2);
        free(batch_thetas3);
        free(nextWns);
        free(nextW2ns);
        free(nextW3ns);
        free(elb1s_batch);
        free(elb2s_batch);
        free(elb3s_batch);
        free(bin_edges);
        free(nextNM3correlator);
        free(thisGtilde_n);
        free(thisN_n);
        free(thisGtilde_n_rec);
        free(thisN_n_rec);
        free(thisnpcf);
        free(thisnpcf_norm);                
    }
    
    // Accummulate the Map^4 integral
    for (int elthread=0; elthread<nthreads; elthread++){
        for (int elapr=0; elapr<napradii; elapr++){
            NM3correlator[elapr] += allNM3correlator[elthread*napradii+elapr];
        }
    }    
    if (verbose>0){ printf("\n"); }
    free(allNM3correlator);
}

/////////////////////////////
// NNNN CORRELATOR CLASSES //
/////////////////////////////

// NNNN correlator using the tree approximation
// This is the "low-memory-implenentaiont": It minimizes memory usage by distributing the individual combinations of 
// radial bin combis to different threads. This might hurt runtime as each thread needs to do the full Wn computation.
void alloc_notomoNap4_tree_nnnn(const MultiresoCatalog *cat_base, const MultiresoCatalog *cat_leaf,
    const NavHash *nav, const TreeResoParams *tree, const BinningParams *bin, const FourthParams *fourth,
    double *napradii, int nnapradii, double complex *N4correlators, int alloc_4pcfmultipoles, int alloc_4pcfreal,
    int nthreads, int verbose,
    double *bin_centers, double complex *N_n, double complex *Counts){

    // Dereference input args
    double *isinner = cat_base->isinner_resos, *weight = cat_base->weight_resos;
    double *pos1 = cat_base->pos1_resos, *pos2 = cat_base->pos2_resos;
    int nmax = bin->nmax, nbinsr = bin->nbinsr;
    double rmin = bin->rmin, rmax = bin->rmax;
    int *nindices = fourth->nindices, len_nindices = fourth->len_nindices;
    double *phibins = fourth->phibins1, *dbinsphi = fourth->dbinsphi1; int nbinsphi = fourth->nbinsphi1;
    int nresos = tree->nresos; double *reso_redges = tree->reso_redges;
    int *ngal_resos = cat_leaf->ngal_resos;
    double *weight_resos = cat_leaf->weight_resos;
    double *pos1_resos = cat_leaf->pos1_resos, *pos2_resos = cat_leaf->pos2_resos;
    int *index_matcher_hash = nav->index_matcher, *pixs_galind_bounds = nav->pixs_galind_bounds, *pix_gals = nav->pix_gals;
    int nregions = nav->nregions;
    double pix1_start = nav->pix1_start, pix1_d = nav->pix1_d; int pix1_n = nav->pix1_n;
    double pix2_start = nav->pix2_start, pix2_d = nav->pix2_d; int pix2_n = nav->pix2_n;
    int *thetacombis_batches = fourth->thetacombis_batches, *nthetacombis_batches = fourth->nthetacombis_batches;
    int *cumthetacombis_batches = fourth->cumthetacombis_batches; int nthetbatches = fourth->nthetbatches;
               
    double complex *allN4correlators = orpheus_calloc(nthreads*1*nnapradii, sizeof(double complex));
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){free(allN4correlators); return;}
    int nregionsdone = 0;
    reset_progress();
    #pragma omp parallel for num_threads(nthreads)
    for (int elthetbatch=0; elthetbatch<nthetbatches; elthetbatch++){
        int nregions_skip_print = mymax(1, nregions / 100);
        
        // * nmax_alloc specifies the largest multipole that needs to be allocated when wanting 
        //   to allocate the Upsn/Nn while making use of the symmetry properties
        // * All quantities that are updated at the galaxy level are computed until nmax_alloc
        // * Once we are done iterating over the cat we apply the symmetries and allocate the
        //   reconstructed quantities having a suffix _rec
        int thisthread = omp_get_thread_num();
        //printf("Doing thetabatch %d/%d on thread %d\n",elthetbatch,nthetbatches,thisthread);
        int nmax_alloc = 2*nmax+1;
        int nbinsz = 1;
        int nnvals_Wn = 4*nmax_alloc+1; // Need to cover [-n1-n2, n1+n2]
        int nnvals_Nn = 2*nmax_alloc+1;  // Need tocover [-2*nmax_alloc,+2*nmax_alloc]
        int nnvals_Nn_rec = 2*nmax+1; // Need tocover [-nmax,+nmax]
        int nzero_Wn = 2*nmax_alloc;
        int nzero_Nn = nmax_alloc;
        
        int N_nshift = nbinsr*nbinsr*nbinsr;
        int n2n3combis = nnvals_Nn*nnvals_Nn;
        int n2n3combis_rec = nnvals_Nn_rec*nnvals_Nn_rec;
        
        int batch_nthetas = nthetacombis_batches[elthetbatch];
        int batchN_nshift = batch_nthetas;
        int batchN_compshift = n2n3combis*batchN_nshift;
        int batchgamma_thetshift = nbinsphi*nbinsphi;
        
        int npix_hash = pix1_n*pix2_n;
        int *rshift_index_matcher_hash = orpheus_calloc(nresos, sizeof(int));
        int *rshift_pixs_galind_bounds = orpheus_calloc(nresos, sizeof(int));
        int *rshift_pix_gals = orpheus_calloc(nresos, sizeof(int));
        build_rshift_offsets(nresos, npix_hash, ngal_resos,
                             rshift_index_matcher_hash, rshift_pixs_galind_bounds, rshift_pix_gals);

        double *totcounts = orpheus_calloc(nbinsr, sizeof(double));
        double *totnorms = orpheus_calloc(nbinsr, sizeof(double));
        double *bin_centers_batch = orpheus_calloc(nbinsr, sizeof(double));
        double complex *batchN_n = orpheus_calloc(batchN_compshift, sizeof(double complex));
        double *batch_thetas1 = orpheus_calloc(batch_nthetas, sizeof(double));
        double *batch_thetas2 = orpheus_calloc(batch_nthetas, sizeof(double));
        double *batch_thetas3 = orpheus_calloc(batch_nthetas, sizeof(double));
        
        int nbinszr = nbinsz*nbinsr;
        double complex *nextWns = orpheus_calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW2ns = orpheus_calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW3ns = orpheus_calloc(nbinszr, sizeof(double complex));
        
        double drbin = (log(rmax)-log(rmin))/(nbinsr);
        int rbin_min_batch=nbinsr;int rbin_max_batch=0;
        int reso_min_batch=0; int reso_max_batch=0;
        int *elb1s_batch = orpheus_calloc(batch_nthetas, sizeof(int));
        int *elb2s_batch = orpheus_calloc(batch_nthetas, sizeof(int));
        int *elb3s_batch = orpheus_calloc(batch_nthetas, sizeof(int));
        double *bin_edges = orpheus_calloc(nbinsr+1, sizeof(double));
        
        #pragma omp critical
        {
            build_thetabatch(elthetbatch, batch_nthetas, nbinsr, nresos, rmin, rmax,
                             thetacombis_batches, cumthetacombis_batches, reso_redges,
                             elb1s_batch, elb2s_batch, elb3s_batch, bin_edges,
                             &rbin_min_batch, &rbin_max_batch, &reso_min_batch, &reso_max_batch);
        }
        
        // Allocate the 4pcf multipoles for this batch of radii 
        int offset_per_thread = nregions/nthreads;
        int offset = offset_per_thread*thisthread;
        for (int _elregion=0; _elregion<nregions; _elregion++){
            int elregion = (_elregion+offset)%nregions; // Try to evade collisions
            #pragma omp atomic
            nregionsdone += 1;
            print_progress(nregionsdone, nthetbatches*nregions, verbose);
            //int region_debug = mymin(500,nregions-1);
            int lower1, upper1;
            lower1 = pixs_galind_bounds[elregion];
            upper1 = pixs_galind_bounds[elregion+1];
            for (int ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                double time1, time2;
                time1 = omp_get_wtime();
                int ind_gal = pix_gals[ind_inpix1];
                double p11, p12, w1;
                double innergal;
                p11 = pos1[ind_gal];
                p12 = pos2[ind_gal];
                w1 = weight[ind_gal];
                innergal = isinner[ind_gal];
                if (innergal<1e-5){continue;}
                
                int ind_gal2;
                int lower, upper;
                double  p21, p22, w2, w2_sq, rel1, rel2, dist, dphi;
                double complex phirot, phirotc;
                
                // Check how many ns we need for Gn
                // Gns have shape (nnvals, nbinsz, nbinsr)
                // where the ns are ordered as 
                // [-nmax_1-nmax_2-3, ..., nmax_1+nmax_2+3]
                for (int i=0; i<nnvals_Wn*nbinszr; i++){nextWns[i]=0;nextW2ns[i]=0;}
                for (int i=0; i<nbinszr; i++){nextW3ns[i]=0;}
                for (int elreso=reso_min_batch; elreso<=reso_max_batch; elreso++){
                    int rbin, zrshift, ind_Wn;
                    double rmin_reso = reso_redges[elreso];
                    double rmax_reso = reso_redges[elreso+1];
                    FLATCELL_FOREACH(
                        index_matcher_hash, rshift_index_matcher_hash[elreso], pixs_galind_bounds, rshift_pixs_galind_bounds[elreso],
                        p11, p12, rmax_reso, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, lower, upper){
                        for (int ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                            ind_gal2 = rshift_pix_gals[elreso] + pix_gals[rshift_pix_gals[elreso]+ind_inpix];
                            //#pragma omp critical
                            {
                            p21 = pos1_resos[ind_gal2];
                            p22 = pos2_resos[ind_gal2];
                            w2 = weight_resos[ind_gal2];
                            }
                            
                            rel1 = p21 - p11;
                            rel2 = p22 - p12;
                            dist = sqrt(rel1*rel1 + rel2*rel2);
                            if (dist < rmin_reso || dist >= rmax_reso) continue;
                            rbin = (int) floor((log(dist)-log(rmin))/drbin);
                            w2_sq = w2*w2;
                            dphi = atan2(rel2,rel1);
                            phirot = cexp(I*dphi);
                            phirotc = conj(phirot);
                            zrshift = 0*nbinsr + rbin;
                            ind_Wn = nzero_Wn*nbinszr + zrshift;
                            nextW3ns[zrshift] += w2_sq*w2;
                            totcounts[zrshift] += w1*w2*dist; 
                            totnorms[zrshift] += w1*w2; 
                            nnnn_fill_wn(nextWns, nextW2ns, 2*nmax_alloc, nbinszr, ind_Wn, w2, w2_sq, phirot, phirotc);
                        }
                    }
                }
                time2 = omp_get_wtime();
                if ((verbose>1) && (elregion%nregions_skip_print == 0)&&(thisthread==0)&&(ind_inpix1==lower1)){
                    printf("Computed Wn for first gal in region %d/%d for thetabatch %d/%d in %.4f seconds\n",
                           elregion,nregions,elthetbatch,nthetbatches,(time2-time1));
                } 

                // Allocate Upsilon
                // Upsilon have shape 
                // (ncomp,(2*nmax_alloc+1),(2*nmax_alloc+1),nthetas)
                time1 = omp_get_wtime();

                nnnn_accum_batchNn(batchN_n, batch_nthetas, batchN_nshift,
                                   elb1s_batch, elb2s_batch, elb3s_batch,
                                   nextWns, nextW2ns, nextW3ns,
                                   nindices, len_nindices, nnvals_Nn, nzero_Nn, nzero_Wn, nbinsr,
                                   w1, 1, elregion, elthetbatch);
                
                time2 = omp_get_wtime();
                if ((verbose>1) && (elregion%nregions_skip_print == 0)&&(thisthread==0)&&(ind_inpix1==lower1)){
                    printf("Allocated Nns for first gal in region %d/%d for thetabatch %d/%d in %.4f seconds for %d theta-combis\n",
                           elregion,nregions,elthetbatch,nthetbatches,(time2-time1),batch_nthetas);
                }
                if ((verbose>1) && (elregion%nregions_skip_print == 0)&&(thisthread==0)&&(ind_inpix1==lower1)){
                    printf("Done region %d/%d for thetabatch %d/%d\n",elregion,nregions,elthetbatch,nthetbatches);
                }
                
            }
        }
        
        // Get bin centers
        // Inside the parallel loop
        if (elthetbatch==0 && omp_get_thread_num() == 0){
            for (int elbinr=0; elbinr<nbinsr; elbinr++){
                if (totnorms[elbinr] != 0){
                    bin_centers[elbinr] = totcounts[elbinr]/totnorms[elbinr]; 
                }
            }
        }
        
        // For each theta combination (theta1,theta2,theta3) in this batch 
        // 1) Get bin edges and bin centers of the combinations
        // 2) Find all (theta1,theta2,theta3) combis that can be reconstructed via the symmetries
        //   2a) Get the Gamma_mu(theta1,theta2,theta3,phi12,phi13)
        //   2b) Transform the Gamma_mu to the target basis
        //   2c) Update the aperture Map^4 integral
        double complex *nextN4correlators = orpheus_calloc(1, sizeof(double complex));
        double complex *thisN_n = orpheus_calloc(n2n3combis, sizeof(double complex));
        double complex *thisN_n_rec = orpheus_calloc(n2n3combis_rec, sizeof(double complex));
        double complex *thisnpcf = orpheus_calloc(batchgamma_thetshift, sizeof(double complex));
        nnnn_reconstruct_batch(batchN_n, batch_nthetas, batchN_nshift,
                              elb1s_batch, elb2s_batch, elb3s_batch,
                              nmax, nindices, len_nindices, n2n3combis, n2n3combis_rec,
                              thisN_n, thisN_n_rec,
                              N_n, N_nshift, nbinsr, alloc_4pcfmultipoles, 0,
                              alloc_4pcfreal, Counts, thisnpcf,
                              phibins, dbinsphi, nbinsphi, batchgamma_thetshift,
                              nnapradii, napradii, bin_centers_batch, bin_edges,
                              allN4correlators, nextN4correlators,
                              thisthread, verbose);
        
        for (int elnapr=0; elnapr<nnapradii; elnapr++){
            int nap4ind = elnapr;
            int nap4threadshift = thisthread*nnapradii;
            if (verbose>1){ printf("\nthread %d, elr %d, elcomp %d, allN4cont=%.20f ",
                           thisthread, elnapr, 0, creal(allN4correlators[nap4threadshift+nap4ind])); }
        }
        
        if (verbose>1){printf("Done allocating 4pcfs for thetabatch %d/%d\n",elthetbatch,nthetbatches);}

        free(rshift_index_matcher_hash);
        free(rshift_pixs_galind_bounds);
        free(rshift_pix_gals);
            
        free(totcounts);
        free(totnorms);
        free(bin_centers_batch);
        free(batch_thetas1);
        free(batch_thetas2);
        free(batch_thetas3);
        free(batchN_n);
        
        free(nextWns);
        free(nextW2ns);
        free(nextW3ns);
        
        free(elb1s_batch);
        free(elb2s_batch);
        free(elb3s_batch);
        free(bin_edges);

        free(nextN4correlators);
        free(thisN_n);
        free(thisN_n_rec);
        free(thisnpcf);
    }
   
    // Accummulate the Nap^4 integral
    for (int elthread=0; elthread<nthreads; elthread++){
        int nap4ind;
        int nap4threadshift = elthread*nnapradii;
        for (int elnapr=0; elnapr<nnapradii; elnapr++){
            nap4ind = elnapr;
            N4correlators[nap4ind] += allN4correlators[nap4threadshift+nap4ind];
        }
    }
    if (verbose>0){ printf("\n"); }
    free(allN4correlators);
}

// NNNN correlator using the tree approximation
// This is the "runtime-optimised-implenentaiont": It first builds and stores all Wn moments
// for each base galaxy. In a second pass it accumulates the Nn fully in parallel.
//
// Note that the moment cache consists of 2*ncache*nnvals_Wn*nbinsr complex doubles
// where ncache = pixs_galind_bounds[nregions]. As this can exceed 2e9 elements we split it 
// into chunks to avoid segfaults.
void alloc_notomoNap4_tree_nnnn_highmem(const MultiresoCatalog *cat_base, const MultiresoCatalog *cat_leaf,
    const NavHash *nav, const TreeResoParams *tree, const BinningParams *bin, const FourthParams *fourth,
    double *napradii, int nnapradii, double complex *N4correlators, int alloc_4pcfmultipoles, int alloc_4pcfreal,
    int nthreads, int verbose,
    double *bin_centers, double complex *N_n, double complex *Counts){

    // Dereference input args
    double *isinner = cat_base->isinner_resos, *weight = cat_base->weight_resos;
    double *pos1 = cat_base->pos1_resos, *pos2 = cat_base->pos2_resos;
    int nmax = bin->nmax, nbinsr = bin->nbinsr;
    double rmin = bin->rmin, rmax = bin->rmax;
    int *nindices = fourth->nindices, len_nindices = fourth->len_nindices;
    double *phibins = fourth->phibins1, *dbinsphi = fourth->dbinsphi1; int nbinsphi = fourth->nbinsphi1;
    int nresos = tree->nresos; double *reso_redges = tree->reso_redges;
    int *ngal_resos = cat_leaf->ngal_resos;
    double *weight_resos = cat_leaf->weight_resos;
    double *pos1_resos = cat_leaf->pos1_resos, *pos2_resos = cat_leaf->pos2_resos;
    int *index_matcher_hash = nav->index_matcher, *pixs_galind_bounds = nav->pixs_galind_bounds, *pix_gals = nav->pix_gals;
    int nregions = nav->nregions;
    double pix1_start = nav->pix1_start, pix1_d = nav->pix1_d; int pix1_n = nav->pix1_n;
    double pix2_start = nav->pix2_start, pix2_d = nav->pix2_d; int pix2_n = nav->pix2_n;
    int *thetacombis_batches = fourth->thetacombis_batches, *nthetacombis_batches = fourth->nthetacombis_batches;
    int *cumthetacombis_batches = fourth->cumthetacombis_batches; int nthetbatches = fourth->nthetbatches;

    int nmax_alloc = 2*nmax+1;
    int nbinsz = 1;
    int nnvals_Wn = 4*nmax_alloc+1; // Need to cover [-n1-n2, n1+n2]
    int nnvals_Nn = 2*nmax_alloc+1; // Need to cover [-2*nmax_alloc,+2*nmax_alloc]
    int nnvals_Nn_rec = 2*nmax+1;   // Need to cover [-nmax,+nmax]
    int nzero_Wn = 2*nmax_alloc;
    int nzero_Nn = nmax_alloc;
    int N_nshift = nbinsr*nbinsr*nbinsr;
    int n2n3combis = nnvals_Nn*nnvals_Nn;
    int n2n3combis_rec = nnvals_Nn_rec*nnvals_Nn_rec;
    int nbinszr = nbinsz*nbinsr;
    double drbin = (log(rmax)-log(rmin))/(nbinsr);
    int npix_hash = pix1_n*pix2_n;

    int *rshift_index_matcher_hash = orpheus_calloc(nresos, sizeof(int));
    int *rshift_pixs_galind_bounds = orpheus_calloc(nresos, sizeof(int));
    int *rshift_pix_gals = orpheus_calloc(nresos, sizeof(int));
    build_rshift_offsets(nresos, npix_hash, ngal_resos,
                         rshift_index_matcher_hash, rshift_pixs_galind_bounds, rshift_pix_gals);

    double *bin_edges = orpheus_calloc(nbinsr+1, sizeof(double));
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){
        free(rshift_index_matcher_hash); free(rshift_pixs_galind_bounds); free(rshift_pix_gals);
        free(bin_edges);
        return;
    }
    bin_edges[0] = rmin;
    for (int elb=0; elb<nbinsr; elb++){ bin_edges[elb+1] = bin_edges[elb]*exp(drbin); }

    // Build moment cache //
    // Determine how many chunks we need an then allocate the cache 
    long ncache = (long) pixs_galind_bounds[nregions];
    long wn_per_gal = (long) nnvals_Wn*nbinsr;
    int gal_per_chunk = (int)(1000000000L / wn_per_gal); if (gal_per_chunk<1){gal_per_chunk=1;}
    int nchunks = (int)((ncache + (long)gal_per_chunk - 1)/(long)gal_per_chunk);
    double complex **Wncache  = orpheus_malloc(nchunks*sizeof(double complex*));
    double complex **W2ncache = orpheus_malloc(nchunks*sizeof(double complex*));
    double complex **W3ncache = orpheus_malloc(nchunks*sizeof(double complex*));
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){
        free(rshift_index_matcher_hash); free(rshift_pixs_galind_bounds); free(rshift_pix_gals);
        free(bin_edges); free(Wncache); free(W2ncache); free(W3ncache);
        return;
    }
    for (int c=0; c<nchunks; c++){
        long chunkgals = gal_per_chunk;
        if ((long)(c+1)*gal_per_chunk > ncache){ chunkgals = ncache - (long)c*gal_per_chunk; }
        Wncache[c]  = orpheus_calloc(chunkgals*wn_per_gal, sizeof(double complex));
        W2ncache[c] = orpheus_calloc(chunkgals*wn_per_gal, sizeof(double complex));
        W3ncache[c] = orpheus_calloc(chunkgals*nbinsr, sizeof(double complex));
    }

    double *tmp_totcounts = orpheus_calloc(nthreads*nbinsr, sizeof(double));
    double *tmp_totnorms  = orpheus_calloc(nthreads*nbinsr, sizeof(double));
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){
        free(rshift_index_matcher_hash); free(rshift_pixs_galind_bounds); free(rshift_pix_gals);
        free(bin_edges); free(Wncache); free(W2ncache); free(W3ncache); free(tmp_totcounts);
        free(tmp_totnorms);
        return;
    }

    // Fill the moment cache in parallel
    // Progress spans both the cache fill and the batch accumulation below, which reads
    // the whole cache once per theta batch; both are counted in cache-touch units.
    // progscale keeps the int arguments of print_progress in range for large caches.
    long nregionsdone = 0, progtot = ncache*(1L+nthetbatches);
    long progscale = 1L + progtot/1000000000L;
    reset_progress();
    #pragma omp parallel for num_threads(nthreads)
    for (long ic=0; ic<ncache; ic++){
        #pragma omp atomic
        nregionsdone += 1;
        print_progress((int)(nregionsdone/progscale), (int)(progtot/progscale), verbose);
        int thisthread = omp_get_thread_num();
        int ind_inpix1 = (int) ic;
        int ind_gal = pix_gals[ind_inpix1];
        double p11 = pos1[ind_gal];
        double p12 = pos2[ind_gal];
        double w1 = weight[ind_gal];
        double innergal = isinner[ind_gal];
        if (innergal<1e-5){continue;}

        int chunk = (int)(ic/gal_per_chunk);
        int loc = (int)(ic - (long)chunk*gal_per_chunk);
        double complex *nextWns  = Wncache[chunk]  + (long)loc*wn_per_gal;
        double complex *nextW2ns = W2ncache[chunk] + (long)loc*wn_per_gal;
        double complex *nextW3ns = W3ncache[chunk] + (long)loc*nbinsr;

        int ind_gal2, lower, upper;
        double p21, p22, w2, w2_sq, rel1, rel2, dist, dphi;
        double complex phirot, phirotc;
        for (int elreso=0; elreso<nresos; elreso++){
            int rbin, zrshift, ind_Wn;
            double rmin_reso = reso_redges[elreso];
            double rmax_reso = reso_redges[elreso+1];
            FLATCELL_FOREACH(
                index_matcher_hash, rshift_index_matcher_hash[elreso], pixs_galind_bounds, rshift_pixs_galind_bounds[elreso],
                p11, p12, rmax_reso, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, lower, upper){
                for (int ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                    ind_gal2 = rshift_pix_gals[elreso] + pix_gals[rshift_pix_gals[elreso]+ind_inpix];
                    p21 = pos1_resos[ind_gal2];
                    p22 = pos2_resos[ind_gal2];
                    w2 = weight_resos[ind_gal2];
                    rel1 = p21 - p11;
                    rel2 = p22 - p12;
                    dist = sqrt(rel1*rel1 + rel2*rel2);
                    if (dist < rmin_reso || dist >= rmax_reso) continue;
                    rbin = (int) floor((log(dist)-log(rmin))/drbin);
                    w2_sq = w2*w2;
                    dphi = atan2(rel2,rel1);
                    phirot = cexp(I*dphi);
                    phirotc = conj(phirot);
                    zrshift = 0*nbinsr + rbin;
                    ind_Wn = nzero_Wn*nbinszr + zrshift;
                    nextW3ns[zrshift] += w2_sq*w2;
                    tmp_totcounts[thisthread*nbinsr+zrshift] += w1*w2*dist;
                    tmp_totnorms[thisthread*nbinsr+zrshift]  += w1*w2;
                    nnnn_fill_wn(nextWns, nextW2ns, 2*nmax_alloc, nbinszr, ind_Wn, w2, w2_sq, phirot, phirotc);
                }
            }
        }
    }

    // Reduce bin centers
    double *totcounts = orpheus_calloc(nbinsr, sizeof(double));
    double *totnorms  = orpheus_calloc(nbinsr, sizeof(double));
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){
        free(rshift_index_matcher_hash); free(rshift_pixs_galind_bounds); free(rshift_pix_gals);
        free(bin_edges); free(Wncache); free(W2ncache); free(W3ncache); free(tmp_totcounts);
        free(tmp_totnorms); free(totcounts); free(totnorms);
        return;
    }
    for (int t=0; t<nthreads; t++){
        for (int b=0; b<nbinsr; b++){
            totcounts[b] += tmp_totcounts[t*nbinsr+b];
            totnorms[b]  += tmp_totnorms[t*nbinsr+b];
        }
    }
    for (int b=0; b<nbinsr; b++){ if (totnorms[b]!=0){ bin_centers[b] = totcounts[b]/totnorms[b]; } }

    // Allocation of NNNN & optionally further transforms //
    double complex *allN4correlators = orpheus_calloc(nthreads*nnapradii, sizeof(double complex));
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){
        free(rshift_index_matcher_hash); free(rshift_pixs_galind_bounds); free(rshift_pix_gals);
        free(bin_edges); free(Wncache); free(W2ncache); free(W3ncache); free(tmp_totcounts);
        free(tmp_totnorms); free(totcounts); free(totnorms); free(allN4correlators);
        return;
    }
    #pragma omp parallel for num_threads(nthreads)
    for (int elthetbatch=0; elthetbatch<nthetbatches; elthetbatch++){
        int thisthread = omp_get_thread_num();
        int batch_nthetas = nthetacombis_batches[elthetbatch];
        int batchN_nshift = batch_nthetas;
        long batchN_compshift = (long)n2n3combis*batchN_nshift;
        int batchgamma_thetshift = nbinsphi*nbinsphi;

        double complex *batchN_n = orpheus_calloc(batchN_compshift, sizeof(double complex));
        int *elb1s_batch = orpheus_calloc(batch_nthetas, sizeof(int));
        int *elb2s_batch = orpheus_calloc(batch_nthetas, sizeof(int));
        int *elb3s_batch = orpheus_calloc(batch_nthetas, sizeof(int));
        double *bin_centers_batch = orpheus_calloc(nbinsr, sizeof(double));
        for (int b=0; b<nbinsr; b++){ bin_centers_batch[b] = bin_centers[b]; }
        for (int elb=0; elb<batch_nthetas; elb++){
            int thisrcombi = thetacombis_batches[cumthetacombis_batches[elthetbatch]+elb];
            elb1s_batch[elb] = thisrcombi/(nbinsr*nbinsr);
            elb2s_batch[elb] = (thisrcombi-elb1s_batch[elb]*nbinsr*nbinsr)/nbinsr;
            elb3s_batch[elb] = thisrcombi-elb1s_batch[elb]*nbinsr*nbinsr-elb2s_batch[elb]*nbinsr;
        }

        // Accumulate the multipole 4PCF for this batch by reading cached moments
        for (long ic=0; ic<ncache; ic++){
            int ind_inpix1 = (int) ic;
            int ind_gal = pix_gals[ind_inpix1];
            double w1 = weight[ind_gal];
            double innergal = isinner[ind_gal];
            if (innergal<1e-5){continue;}
            int chunk = (int)(ic/gal_per_chunk);
            int loc = (int)(ic - (long)chunk*gal_per_chunk);
            double complex *nextWns  = Wncache[chunk]  + (long)loc*wn_per_gal;
            double complex *nextW2ns = W2ncache[chunk] + (long)loc*wn_per_gal;
            double complex *nextW3ns = W3ncache[chunk] + (long)loc*nbinsr;

            nnnn_accum_batchNn(batchN_n, batch_nthetas, batchN_nshift,
                               elb1s_batch, elb2s_batch, elb3s_batch,
                               nextWns, nextW2ns, nextW3ns,
                               nindices, len_nindices, nnvals_Nn, nzero_Nn, nzero_Wn, nbinsr,
                               w1, 0, 0, 0);
        }
        #pragma omp atomic
        nregionsdone += ncache;
        print_progress((int)(nregionsdone/progscale), (int)(progtot/progscale), verbose);

        // Reconstruct each theta combi via symmetries, optionally transform to Nap^4
        double complex *nextN4correlators = orpheus_calloc(1, sizeof(double complex));
        double complex *thisN_n = orpheus_calloc(n2n3combis, sizeof(double complex));
        double complex *thisN_n_rec = orpheus_calloc(n2n3combis_rec, sizeof(double complex));
        double complex *thisnpcf = orpheus_calloc(batchgamma_thetshift, sizeof(double complex));
        nnnn_reconstruct_batch(batchN_n, batch_nthetas, batchN_nshift,
                              elb1s_batch, elb2s_batch, elb3s_batch,
                              nmax, nindices, len_nindices, n2n3combis, n2n3combis_rec,
                              thisN_n, thisN_n_rec,
                              N_n, N_nshift, nbinsr, alloc_4pcfmultipoles, 0,
                              alloc_4pcfreal, Counts, thisnpcf,
                              phibins, dbinsphi, nbinsphi, batchgamma_thetshift,
                              nnapradii, napradii, bin_centers_batch, bin_edges,
                              allN4correlators, nextN4correlators,
                              thisthread, verbose);

        free(batchN_n);
        free(elb1s_batch);
        free(elb2s_batch);
        free(elb3s_batch);
        free(bin_centers_batch);
        free(nextN4correlators);
        free(thisN_n);
        free(thisN_n_rec);
        free(thisnpcf);
    }

    // Accummulate the Nap^4 integral
    for (int elthread=0; elthread<nthreads; elthread++){
        int nap4threadshift = elthread*nnapradii;
        for (int elnapr=0; elnapr<nnapradii; elnapr++){
            N4correlators[elnapr] += allN4correlators[nap4threadshift+elnapr];
        }
    }
    free(allN4correlators);
    if (verbose>0){ printf("\n"); }

    for (int c=0; c<nchunks; c++){ free(Wncache[c]); free(W2ncache[c]); free(W3ncache[c]); }
    free(Wncache); free(W2ncache); free(W3ncache);
    free(tmp_totcounts); free(tmp_totnorms); free(totcounts); free(totnorms);
    free(rshift_index_matcher_hash); free(rshift_pixs_galind_bounds); free(rshift_pix_gals);
    free(bin_edges);
}


// NNNN correlator using the tree approximation
// * A:  Work list of the radial triples and multipoles, and batches of base galaxies whose
//       moments fit into memory_bound
// * B1: For each batch, the moments of its base galaxies in parallel over the galaxies
// * B2: From these, update the multipoles of all triples in parallel over the radial pairs,
//       each summing over all galaxies of the batch (nnnn_accum_pair)
// * C:  Reconstruct all orderings of the triples once at the end
void alloc_nnnn_tree(const MultiresoCatalog *cat_base, const MultiresoCatalog *cat_leaf,
    const NavHash *nav, const TreeResoParams *tree, const BinningParams *bin, const FourthParams *fourth,
    double memory_bound, int nthreads, int verbose, NPCFOutput *out){

    // Dereference input args
    double *isinner = cat_base->isinner_resos, *weight = cat_base->weight_resos;
    double *pos1 = cat_base->pos1_resos, *pos2 = cat_base->pos2_resos;
    int nmax = bin->nmax, nbinsr = bin->nbinsr;
    double rmin = bin->rmin, rmax = bin->rmax;
    int *nindices = fourth->nindices, len_nindices = fourth->len_nindices;
    int nresos = tree->nresos; double *reso_redges = tree->reso_redges;
    int *ngal_resos = cat_leaf->ngal_resos;
    double *weight_resos = cat_leaf->weight_resos;
    double *pos1_resos = cat_leaf->pos1_resos, *pos2_resos = cat_leaf->pos2_resos;
    int *index_matcher_hash = nav->index_matcher, *pixs_galind_bounds = nav->pixs_galind_bounds, *pix_gals = nav->pix_gals;
    int nregions = nav->nregions;
    double pix1_start = nav->pix1_start, pix1_d = nav->pix1_d; int pix1_n = nav->pix1_n;
    double pix2_start = nav->pix2_start, pix2_d = nav->pix2_d; int pix2_n = nav->pix2_n;
    int *thetacombis_batches = fourth->thetacombis_batches;
    int *cumthetacombis_batches = fourth->cumthetacombis_batches; int nthetbatches = fourth->nthetbatches;
    double *bin_centers = out->bin_centers;
    double complex *N_n = out->npcf;

    long ncombis = cumthetacombis_batches[nthetbatches];
    double drbin = (log(rmax)-log(rmin))/(nbinsr);
    int npix_hash = pix1_n*pix2_n;

    int *rshift_index_matcher_hash = orpheus_calloc(nresos, sizeof(int));
    int *rshift_pixs_galind_bounds = orpheus_calloc(nresos, sizeof(int));
    int *rshift_pix_gals = orpheus_calloc(nresos, sizeof(int));
    double *tmp_totcounts = orpheus_calloc(nthreads*nbinsr, sizeof(double));
    double *tmp_totnorms  = orpheus_calloc(nthreads*nbinsr, sizeof(double));
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){
        free(rshift_index_matcher_hash); free(rshift_pixs_galind_bounds); free(rshift_pix_gals);
        free(tmp_totcounts); free(tmp_totnorms);
        return;
    }
    build_rshift_offsets(nresos, npix_hash, ngal_resos,
                         rshift_index_matcher_hash, rshift_pixs_galind_bounds, rshift_pix_gals);

    // A: Work list, base galaxies and batches //

    // Only inner galaxies are base galaxies
    long ngal_all = (long) pixs_galind_bounds[nregions];
    int *baseinds = orpheus_malloc((ngal_all>0?ngal_all:1)*sizeof(int));
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){
        free(rshift_index_matcher_hash); free(rshift_pixs_galind_bounds); free(rshift_pix_gals);
        free(tmp_totcounts); free(tmp_totnorms); free(baseinds);
        return;
    }
    long ncache = 0;
    for (long ig=0; ig<ngal_all; ig++){
        if (isinner[pix_gals[(int)ig]] >= 1e-5){ baseinds[ncache++] = (int)ig; }
    }

    NnnnPlan plan;
    nnnn_plan_build(&plan, thetacombis_batches, ncombis, nbinsr, nindices, len_nindices, nmax);
    int cap = nnnn_batch_cap(ncache, nbinsr, plan.nmax_Wn, memory_bound, nthreads);
    NnnnBatch batch;
    nnnn_batch_alloc(&batch, cap, nbinsr, plan.nmax_Wn);
    double complex *acc = orpheus_calloc(ncombis*plan.n2n3combis_half, sizeof(double complex));
    NnnnLeaves *leaves = orpheus_malloc(nthreads*sizeof(NnnnLeaves));
    NnnnScratch *sc = nnnn_scratch_alloc(nthreads, 1, &plan);
    // A single band holding all radial bins
    int band_rindedges[2] = {0, nbinsr}, band_nobj[1] = {0};
    int *bin2band = orpheus_calloc(nbinsr, sizeof(int));
    NnnnBands bd = {1, band_rindedges, bin2band, &batch, band_nobj, NULL, NULL, NULL, NULL};
    for (int elthread=0; elthread<nthreads; elthread++){ nnnn_leaves_alloc(leaves+elthread, nbinsr); }
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){
        for (int elthread=0; elthread<nthreads; elthread++){ nnnn_leaves_free(leaves+elthread); }
        free(rshift_index_matcher_hash); free(rshift_pixs_galind_bounds); free(rshift_pix_gals);
        free(tmp_totcounts); free(tmp_totnorms); free(baseinds); nnnn_plan_free(&plan);
        nnnn_batch_free(&batch); free(acc); free(leaves); nnnn_scratch_free(sc, nthreads, 1);
        free(bin2band);
        return;
    }
    if (verbose>1){ printf("alloc_nnnn_tree: %ld inner / %ld total galaxies in batches of %d, %d radial pairs, "
           "%d multipoles (%.2f GiB accumulator)\n", ncache, ngal_all, cap, plan.npairs, plan.n2n3combis_half,
           ncombis*plan.n2n3combis_half*16./1073741824.); }

    // B: Iterate over the batches of base galaxies //
    // Progress counts every galaxy once in B1 and once in B2
    long ndone = 0, progtot = 2*ncache;
    long progscale = 1L + progtot/1000000000L;
    reset_progress();
    for (long g0=0; g0<ncache; g0+=cap){
        int nbatch = (int)mymin((long)cap, ncache-g0);

        // B1: Moments of the batch //
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 4)
        for (int ig=0; ig<nbatch; ig++){
            #pragma omp atomic
            ndone += 1;
            print_progress((int)(ndone/progscale), (int)(progtot/progscale), verbose);
            int thisthread = omp_get_thread_num();
            NnnnLeaves *lv = leaves + thisthread;
            int ind_gal = pix_gals[baseinds[g0+ig]];
            double p11 = pos1[ind_gal];
            double p12 = pos2[ind_gal];
            double w1 = weight[ind_gal];
            int lower, upper;
            for (int elreso=0; elreso<nresos; elreso++){
                double rmin_reso = reso_redges[elreso];
                double rmax_reso = reso_redges[elreso+1];
                FLATCELL_FOREACH(
                    index_matcher_hash, rshift_index_matcher_hash[elreso], pixs_galind_bounds, rshift_pixs_galind_bounds[elreso],
                    p11, p12, rmax_reso, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, lower, upper){
                    for (int ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                        int ind_gal2 = rshift_pix_gals[elreso] + pix_gals[rshift_pix_gals[elreso]+ind_inpix];
                        double rel1 = pos1_resos[ind_gal2] - p11;
                        double rel2 = pos2_resos[ind_gal2] - p12;
                        double dist = sqrt(rel1*rel1 + rel2*rel2);
                        if (dist < rmin_reso || dist >= rmax_reso){ continue; }
                        int rbin = mymin((int) floor((log(dist)-log(rmin))/drbin), nbinsr-1);
                        double w2 = weight_resos[ind_gal2];
                        tmp_totcounts[thisthread*nbinsr+rbin] += w1*w2*dist;
                        tmp_totnorms[thisthread*nbinsr+rbin]  += w1*w2;
                        nnnn_leaves_push(lv, rbin, w2, rel1/dist, rel2/dist);
                    }
                }
            }
            nnnn_fill_slot(lv, &batch, ig, w1);
        }

        // B2: Update the multipoles of all triples from the batch //
        band_nobj[0] = nbatch;
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1)
        for (int elpair=0; elpair<plan.npairs; elpair++){
            nnnn_accum_pair(&bd, &plan, plan.pair_order[elpair], sc+omp_get_thread_num(), acc);
        }
        ndone += nbatch;
        print_progress((int)(ndone/progscale), (int)(progtot/progscale), verbose);
    }

    // C: All orderings of every triple //
    nnnn_reconstruct_all(acc, &plan, thetacombis_batches, ncombis, nbinsr, nmax,
                         nindices, len_nindices, N_n, nthreads);

    // Get bin centers
    for (int elthread=1; elthread<nthreads; elthread++){
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            tmp_totcounts[elbinr] += tmp_totcounts[elthread*nbinsr+elbinr];
            tmp_totnorms[elbinr]  += tmp_totnorms[elthread*nbinsr+elbinr];
        }
    }
    for (int elbinr=0; elbinr<nbinsr; elbinr++){
        if (tmp_totnorms[elbinr] != 0){
            bin_centers[elbinr] = tmp_totcounts[elbinr]/tmp_totnorms[elbinr];
        }
    }

    // Free all allocated quantities
    for (int elthread=0; elthread<nthreads; elthread++){ nnnn_leaves_free(leaves+elthread); }
    free(rshift_index_matcher_hash); free(rshift_pixs_galind_bounds); free(rshift_pix_gals);
    free(tmp_totcounts); free(tmp_totnorms); free(baseinds); nnnn_plan_free(&plan);
    nnnn_batch_free(&batch); free(acc); free(leaves); nnnn_scratch_free(sc, nthreads, 1);
    free(bin2band);
    if (verbose>0){ printf("\n"); }
}


// NNNN correlator using the tree approximation in spherical geometry
// Besides gathering the leaves, which uses healpix navigation and spherical geometry
// for the distances and bearing angles, it is the same as alloc_nnnn_tree.
// Recall that all separations are in radians.
void alloc_nnnn_tree_spherical(const MultiresoCatalog *cat_base, const MultiresoCatalog *cat_leaf,
    const NavHash *nav, const TreeResoParams *tree, const BinningParams *bin, const FourthParams *fourth,
    double memory_bound, int nthreads, int verbose, NPCFOutput *out){

    // Dereference input args
    double *isinner = cat_base->isinner_resos, *weight = cat_base->weight_resos;
    double *pos_vx = cat_base->vx_resos, *pos_vy = cat_base->vy_resos, *pos_vz = cat_base->vz_resos;
    int ngal = cat_base->ngal_resos[0];
    int nmax = bin->nmax, nbinsr = bin->nbinsr;
    double rmin = bin->rmin, rmax = bin->rmax;
    int *nindices = fourth->nindices, len_nindices = fourth->len_nindices;
    int nresos = tree->nresos; double *reso_redges = tree->reso_redges;
    int *ncells_resos = nav->ncells_resos;
    long *nside_nav = nav->nside_nav;
    double *weight_resos = cat_leaf->weight_resos;
    double *vx_resos = cat_leaf->vx_resos, *vy_resos = cat_leaf->vy_resos, *vz_resos = cat_leaf->vz_resos;
    int *band_offset_leafgals = nav->rshift_red;
    long *cell_pixids = nav->cell_pix;
    int *cell_leafgal_bounds = nav->cell_redbounds, *band_offset_cellpixids = nav->rshift_cellpix, *band_offset_leafgal_bounds = nav->rshift_cellbounds;
    int *thetacombis_batches = fourth->thetacombis_batches;
    int *cumthetacombis_batches = fourth->cumthetacombis_batches; int nthetbatches = fourth->nthetbatches;
    double *bin_centers = out->bin_centers;
    double complex *N_n = out->npcf;

    long ncombis = cumthetacombis_batches[nthetbatches];
    double drbin = (log(rmax)-log(rmin))/(nbinsr);

    // A: Work list, base galaxies and batches //

    // Only inner galaxies are base galaxies
    int *baseinds = orpheus_malloc((ngal>0?ngal:1)*sizeof(int));
    double *tmp_totcounts = orpheus_calloc(nthreads*nbinsr, sizeof(double));
    double *tmp_totnorms  = orpheus_calloc(nthreads*nbinsr, sizeof(double));
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){free(baseinds); free(tmp_totcounts); free(tmp_totnorms); return;}
    long ncache = 0;
    for (int ig=0; ig<ngal; ig++){
        if (isinner[ig] >= 1e-5){ baseinds[ncache++] = ig; }
    }

    NnnnPlan plan;
    nnnn_plan_build(&plan, thetacombis_batches, ncombis, nbinsr, nindices, len_nindices, nmax);
    int cap = nnnn_batch_cap(ncache, nbinsr, plan.nmax_Wn, memory_bound, nthreads);
    NnnnBatch batch;
    nnnn_batch_alloc(&batch, cap, nbinsr, plan.nmax_Wn);
    double complex *acc = orpheus_calloc(ncombis*plan.n2n3combis_half, sizeof(double complex));
    NnnnLeaves *leaves = orpheus_malloc(nthreads*sizeof(NnnnLeaves));
    NnnnScratch *sc = nnnn_scratch_alloc(nthreads, 1, &plan);
    // A single band holding all radial bins
    int band_rindedges[2] = {0, nbinsr}, band_nobj[1] = {0};
    int *bin2band = orpheus_calloc(nbinsr, sizeof(int));
    NnnnBands bd = {1, band_rindedges, bin2band, &batch, band_nobj, NULL, NULL, NULL, NULL};
    // Ranges returned by query_disc, one buffer per thread grown on demand
    long *query_cap = orpheus_malloc(nthreads*sizeof(long));
    long **ranges = orpheus_malloc(nthreads*sizeof(long*));
    for (int elthread=0; elthread<nthreads; elthread++){
        nnnn_leaves_alloc(leaves+elthread, nbinsr);
        query_cap[elthread] = 2048;
        ranges[elthread] = orpheus_malloc(2*query_cap[elthread]*sizeof(long));
    }
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){
        for (int elthread=0; elthread<nthreads; elthread++){ nnnn_leaves_free(leaves+elthread); free(ranges[elthread]); }
        free(baseinds); free(tmp_totcounts); free(tmp_totnorms); nnnn_plan_free(&plan);
        nnnn_batch_free(&batch); free(acc); free(leaves); nnnn_scratch_free(sc, nthreads, 1);
        free(bin2band); free(query_cap); free(ranges);
        return;
    }
    if (verbose>1){ printf("alloc_nnnn_tree_spherical: %ld inner / %d total galaxies in batches of %d, "
           "%d radial pairs, %d multipoles (%.2f GiB accumulator)\n", ncache, ngal, cap, plan.npairs,
           plan.n2n3combis_half, ncombis*plan.n2n3combis_half*16./1073741824.); }

    // B: Iterate over the batches of base galaxies //
    // Progress counts every galaxy once in B1 and once in B2
    long ndone = 0, progtot = 2*ncache;
    long progscale = 1L + progtot/1000000000L;
    reset_progress();
    for (long g0=0; g0<ncache; g0+=cap){
        int nbatch = (int)mymin((long)cap, ncache-g0);

        // B1: Moments of the batch //
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 4)
        for (int ig=0; ig<nbatch; ig++){
            #pragma omp atomic
            ndone += 1;
            print_progress((int)(ndone/progscale), (int)(progtot/progscale), verbose);
            int thisthread = omp_get_thread_num();
            NnnnLeaves *lv = leaves + thisthread;
            int ind_gal = baseinds[g0+ig];
            double bx = pos_vx[ind_gal], by = pos_vy[ind_gal], bz = pos_vz[ind_gal];
            double w1 = weight[ind_gal];
            double bvec[3] = {bx, by, bz};
            for (int elreso=0; elreso<nresos; elreso++){
                double rmin_reso = reso_redges[elreso];
                double rmax_reso = reso_redges[elreso+1];
                long ns_nav_r = nside_nav[elreso];
                // Run query_disc to get the ranges and optionally resize
                long nranges = hpx_query_disc_nest_ranges(ns_nav_r, bvec, rmax_reso, ranges[thisthread], query_cap[thisthread]);
                if (nranges > query_cap[thisthread]){
                    query_cap[thisthread] = nranges;
                    ranges[thisthread] = realloc(ranges[thisthread], 2*nranges*sizeof(long));
                    nranges = hpx_query_disc_nest_ranges(ns_nav_r, bvec, rmax_reso, ranges[thisthread], query_cap[thisthread]);
                }
                const long *thisranges = ranges[thisthread];

                // For each range returned by query_disc, go through its stored cells and
                // collect the leaves within this band.
                const long *band_cellpixids = cell_pixids + band_offset_cellpixids[elreso];
                const int  *band_leafgal_bounds = cell_leafgal_bounds + band_offset_leafgal_bounds[elreso];
                int ncells_reso = ncells_resos[elreso];
                long leafgal_offset = band_offset_leafgals[elreso];
                int cell_idx = 0;
                for (long range_idx=0; range_idx<nranges; range_idx++){
                    long range_lo = thisranges[2*range_idx], range_hi = thisranges[2*range_idx+1];
                    int loi = cell_idx, hii = ncells_reso;
                    // Only the cells with galaxies in them are stored, so binary search for the
                    // first stored cell within this range
                    while (loi < hii){
                         int m=(loi+hii)>>1;
                         if (band_cellpixids[m] < range_lo){ loi=m+1; }
                         else { hii=m; }
                    }
                    cell_idx = loi;
                    while (cell_idx < ncells_reso && band_cellpixids[cell_idx] < range_hi){
                        int clo = band_leafgal_bounds[cell_idx], chi = band_leafgal_bounds[cell_idx+1];
                        for (int j=clo; j<chi; j++){
                            long ind_leafgal = leafgal_offset + j;
                            double vx2 = vx_resos[ind_leafgal], vy2 = vy_resos[ind_leafgal], vz2 = vz_resos[ind_leafgal];
                            double dist = sphere_dist(bx, by, bz, vx2, vy2, vz2);
                            if (dist < rmin_reso || dist >= rmax_reso) continue;
                            int rbin = mymin((int) floor((log(dist)-log(rmin))/drbin), nbinsr-1);
                            double w2 = weight_resos[ind_leafgal];
                            double complex phirot = bearing_phirot(bearing_AB_cart(bx, by, bz, vx2, vy2, vz2));
                            tmp_totcounts[thisthread*nbinsr+rbin] += w1*w2*dist;
                            tmp_totnorms[thisthread*nbinsr+rbin]  += w1*w2;
                            nnnn_leaves_push(lv, rbin, w2, creal(phirot), cimag(phirot));
                        }
                        cell_idx++;
                    }
                }
            }
            nnnn_fill_slot(lv, &batch, ig, w1);
        }

        // B2: Update the multipoles of all triples from the batch //
        band_nobj[0] = nbatch;
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1)
        for (int elpair=0; elpair<plan.npairs; elpair++){
            nnnn_accum_pair(&bd, &plan, plan.pair_order[elpair], sc+omp_get_thread_num(), acc);
        }
        ndone += nbatch;
        print_progress((int)(ndone/progscale), (int)(progtot/progscale), verbose);
    }

    // C: All orderings of every triple //
    nnnn_reconstruct_all(acc, &plan, thetacombis_batches, ncombis, nbinsr, nmax,
                         nindices, len_nindices, N_n, nthreads);

    // Get bin centers
    for (int elthread=1; elthread<nthreads; elthread++){
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            tmp_totcounts[elbinr] += tmp_totcounts[elthread*nbinsr+elbinr];
            tmp_totnorms[elbinr]  += tmp_totnorms[elthread*nbinsr+elbinr];
        }
    }
    for (int elbinr=0; elbinr<nbinsr; elbinr++){
        if (tmp_totnorms[elbinr] != 0){
            bin_centers[elbinr] = tmp_totcounts[elbinr]/tmp_totnorms[elbinr];
        }
    }

    // Free all allocated quantities
    for (int elthread=0; elthread<nthreads; elthread++){ nnnn_leaves_free(leaves+elthread); free(ranges[elthread]); }
    free(baseinds); free(tmp_totcounts); free(tmp_totnorms); nnnn_plan_free(&plan);
    nnnn_batch_free(&batch); free(acc); free(leaves); nnnn_scratch_free(sc, nthreads, 1);
    free(bin2band); free(query_cap); free(ranges);
    if (verbose>0){ printf("\n"); }
}


// Flat-sky NNNN DoubleTree (Porth+25, App. F). Regions are the cells of the spatial hash. A batch
// of regions is processed by all threads at once:
// * B1: Moments of every apex object (galaxies of the discrete band, cells of the grid bands)
//       for the bins of its band, from the leaves at the leaf resolution of that band
// * B2: Weighted moments of the finer bands summed per cell, then the multipoles of all
//       triples per radial pair (nnnn_accum_pair)
// * C:  Reconstruct all orderings of the triples once at the end
static void alloc_nnnn_doubletree_flat(const MultiresoCatalog *cat, const NavHash *nav,
    const TreeResoParams *tree, const BinningParams *bin, const FourthParams *fourth,
    double memory_bound, int nthreads, int verbose, NPCFOutput *out){

    // Dereference input args
    int nresos = tree->nresos, nresos_grid = tree->nresos_grid;
    double *dpix1_resos = tree->dpix1_resos, *dpix2_resos = tree->dpix2_resos, *reso_redges = tree->reso_redges;
    int resoshift_leafs = tree->resoshift_leafs, minresoind_leaf = tree->minresoind_leaf, maxresoind_leaf = tree->maxresoind_leaf;
    double *isinner_resos = cat->isinner_resos, *weight_resos = cat->weight_resos;
    double *pos1_resos = cat->pos1_resos, *pos2_resos = cat->pos2_resos;
    int *ngal_resos = cat->ngal_resos;
    int nmax = bin->nmax, nbinsr = bin->nbinsr;
    double rmin = bin->rmin, rmax = bin->rmax;
    int *nindices = fourth->nindices, len_nindices = fourth->len_nindices;
    int *index_matcher_hash = nav->index_matcher, *region_to_fullhash = nav->index_matcher_hash;
    int *pixs_galind_bounds = nav->pixs_galind_bounds, *pix_gals = nav->pix_gals;
    int *filledregions = nav->filledregions, nfilledregions = nav->nfilledregions;
    double pix1_start = nav->pix1_start, pix1_d = nav->pix1_d; int pix1_n = nav->pix1_n;
    double pix2_start = nav->pix2_start, pix2_d = nav->pix2_d; int pix2_n = nav->pix2_n;
    int *thetacombis_batches = fourth->thetacombis_batches;
    int *cumthetacombis_batches = fourth->cumthetacombis_batches; int nthetbatches = fourth->nthetbatches;
    double *bin_centers = out->bin_centers;
    double complex *N_n = out->npcf;

    int hasdiscrete = nresos - nresos_grid;
    long ncombis = cumthetacombis_batches[nthetbatches];
    double drbin = (log(rmax)-log(rmin))/(nbinsr);
    double logrmin = log(rmin);
    int npix_hash = pix1_n*pix2_n;

    // A: Bands, work list and scratch //
    int *rshift_index_matcher_hash = orpheus_calloc(nresos, sizeof(int));
    int *rshift_pixs_galind_bounds = orpheus_calloc(nresos, sizeof(int));
    int *rshift_pix_gals = orpheus_calloc(nresos, sizeof(int));
    int *reso_rindedges = orpheus_calloc(nresos+1, sizeof(int));
    int *bin2band = orpheus_calloc(nbinsr, sizeof(int));
    double *tmp_totcounts = orpheus_calloc(nthreads*nbinsr, sizeof(double));
    double *tmp_totnorms = orpheus_calloc(nthreads*nbinsr, sizeof(double));
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){
        free(rshift_index_matcher_hash); free(rshift_pixs_galind_bounds); free(rshift_pix_gals);
        free(reso_rindedges); free(bin2band); free(tmp_totcounts); free(tmp_totnorms);
        return;
    }
    build_rshift_offsets(nresos, npix_hash, ngal_resos,
                         rshift_index_matcher_hash, rshift_pixs_galind_bounds, rshift_pix_gals);
    nnnn_reso_rindedges(nresos, reso_redges, rmin, drbin, nbinsr, reso_rindedges);
    for (int elreso=0; elreso<nresos; elreso++){
        for (int elbinr=reso_rindedges[elreso]; elbinr<reso_rindedges[elreso+1]; elbinr++){ bin2band[elbinr] = elreso; }
    }

    NnnnPlan plan;
    nnnn_plan_build(&plan, thetacombis_batches, ncombis, nbinsr, nindices, len_nindices, nmax);
    int *pair_order_dt = nnnn_dt_pair_order(&plan, bin2band);
    double complex *acc = orpheus_calloc(ncombis*plan.n2n3combis_half, sizeof(double complex));
    NnnnScratch *sc = nnnn_scratch_alloc(nthreads, nresos, &plan);
    NnnnLeaves *leaves = orpheus_malloc(nthreads*sizeof(NnnnLeaves));
    for (int elthread=0; elthread<nthreads; elthread++){ nnnn_leaves_alloc(leaves+elthread, nbinsr); }
    NnnnRegions rg;
    nnnn_regions_alloc(&rg, nresos);
    NnnnBands bd = {nresos, reso_rindedges, bin2band, orpheus_calloc(nresos, sizeof(NnnnBatch)), rg.nobj,
                    rg.parent, orpheus_calloc(nresos, sizeof(double*)),
                    orpheus_calloc(nresos, sizeof(double*)), orpheus_calloc(nresos, sizeof(long))};
    // Region-local index of the object in every grid pixel and the batch slot of every object
    int **pix2redpix = orpheus_calloc(nresos_grid, sizeof(int*));
    for (int elreso_grid=0; elreso_grid<nresos_grid; elreso_grid++){
        int npix_side = 1 << (nresos_grid-elreso_grid-1);
        pix2redpix[elreso_grid] = orpheus_malloc((long)npix_side*npix_side*sizeof(int));
    }
    int **slot_of_local = orpheus_calloc(nresos, sizeof(int*));
    int *slot_cap = orpheus_calloc(nresos, sizeof(int));
    int target = nnnn_dt_batch_target(reso_rindedges[1]-reso_rindedges[0], plan.nmax_Wn, memory_bound, nthreads);
    long nmissing = 0;
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){
        fprintf(stderr, "alloc_nnnn_doubletree: allocation of the work list or scratch failed\n");
        exit(1);
    }
    if (verbose>1){ printf("alloc_nnnn_doubletree: %d filled regions in batches of >=%d apices, %d radial pairs, "
           "%d multipoles (%.2f GiB accumulator)\n", nfilledregions, target, plan.npairs, plan.n2n3combis_half,
           ncombis*plan.n2n3combis_half*16./1073741824.); }

    // B: Iterate over the batches of regions //
    double time_moments = 0., time_triples = 0.;
    int _elregion = 0;
    reset_progress();
    while (_elregion < nfilledregions){

        // Collect regions until the batch holds enough apex objects of the finest band
        for (int elreso=0; elreso<nresos; elreso++){ rg.nobj[elreso] = 0; }
        while (_elregion < nfilledregions && rg.nobj[0] < target){
            int elregion = filledregions[_elregion];
            int elregion_fullhash = region_to_fullhash[elregion];
            double hashpix_start1 = pix1_start + (elregion_fullhash%pix1_n)*pix1_d;
            double hashpix_start2 = pix2_start + (elregion_fullhash/pix1_n)*pix2_d;
            // Coarsest band first, such that every apex finds the slot of its parent
            for (int elreso=nresos-1; elreso>=0; elreso--){
                int lower = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion];
                int upper = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion+1];
                if (slot_cap[elreso] < upper-lower){
                    slot_cap[elreso] = upper-lower;
                    free(slot_of_local[elreso]); slot_of_local[elreso] = orpheus_malloc(slot_cap[elreso]*sizeof(int));
                }
                // Map the grid pixels of this region to the region-local index of their object
                int elreso_grid = elreso - hasdiscrete;
                if (elreso_grid >= 0){
                    int npix_side = 1 << (nresos_grid-elreso_grid-1);
                    for (int ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                        int ind_gal = rshift_pix_gals[elreso] + pix_gals[rshift_pix_gals[elreso]+ind_inpix];
                        int elhashpix_1 = (int) floor((pos1_resos[ind_gal]-hashpix_start1)/dpix1_resos[elreso_grid]);
                        int elhashpix_2 = (int) floor((pos2_resos[ind_gal]-hashpix_start2)/dpix2_resos[elreso_grid]);
                        if (elhashpix_1<0){elhashpix_1=0;} if (elhashpix_1>=npix_side){elhashpix_1=npix_side-1;}
                        if (elhashpix_2<0){elhashpix_2=0;} if (elhashpix_2>=npix_side){elhashpix_2=npix_side-1;}
                        pix2redpix[elreso_grid][elhashpix_2*npix_side + elhashpix_1] = ind_inpix-lower;
                    }
                }
                for (int ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                    int ind_gal = rshift_pix_gals[elreso] + pix_gals[rshift_pix_gals[elreso]+ind_inpix];
                    slot_of_local[elreso][ind_inpix-lower] = -1;
                    if (isinner_resos[ind_gal] < 1e-5){ continue; }
                    int parent = -1;
                    if (elreso < nresos-1){
                        int elreso_grid2 = elreso+1 - hasdiscrete;
                        int npix_side = 1 << (nresos_grid-elreso_grid2-1);
                        int elhashpix_1 = (int) floor((pos1_resos[ind_gal]-hashpix_start1)/dpix1_resos[elreso_grid2]);
                        int elhashpix_2 = (int) floor((pos2_resos[ind_gal]-hashpix_start2)/dpix2_resos[elreso_grid2]);
                        if (elhashpix_1<0){elhashpix_1=0;} if (elhashpix_1>=npix_side){elhashpix_1=npix_side-1;}
                        if (elhashpix_2<0){elhashpix_2=0;} if (elhashpix_2>=npix_side){elhashpix_2=npix_side-1;}
                        parent = slot_of_local[elreso+1][pix2redpix[elreso_grid2][elhashpix_2*npix_side + elhashpix_1]];
                        if (parent < 0){ nmissing += 1; continue; }
                    }
                    slot_of_local[elreso][ind_inpix-lower] = rg.nobj[elreso];
                    nnnn_regions_push(&rg, elreso, ind_gal, parent);
                }
            }
            _elregion += 1;
        }
        nnnn_bands_reserve(&bd, plan.nmax_Wn);
        nnnn_scratch_reserve(sc, nthreads, nresos, rg.nobj, plan.nrowmax);
        // Bail out rather than dereference a failed allocation
        if (orpheus_get_error()){
            fprintf(stderr, "alloc_nnnn_doubletree: allocation of the batch moments failed\n");
            exit(1);
        }

        // B1: Moments of every apex object //
        double time1 = omp_get_wtime();
        long nobj_all = 0;
        for (int elreso=0; elreso<nresos; elreso++){ nobj_all += rg.nobj[elreso]; }
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 4)
        for (long elobj=0; elobj<nobj_all; elobj++){
            int thisthread = omp_get_thread_num();
            NnnnLeaves *lv = leaves + thisthread;
            int elreso = 0;
            long ig = elobj;
            while (ig >= rg.nobj[elreso]){ ig -= rg.nobj[elreso]; elreso++; }
            int ind_gal = (int)rg.ind_obj[elreso][ig];
            int rbinmin = reso_rindedges[elreso], rbinmax = reso_rindedges[elreso+1];
            if (rbinmax <= rbinmin){ bd.batch[elreso].w1[ig] = weight_resos[ind_gal]; continue; }
            int elreso_leaf = mymin(mymax(minresoind_leaf, elreso+resoshift_leafs), maxresoind_leaf);
            double rmin_reso = rmin*exp(rbinmin*drbin), rmax_reso = rmin*exp(rbinmax*drbin);
            double p11 = pos1_resos[ind_gal], p12 = pos2_resos[ind_gal];
            int lower, upper;
            FLATCELL_FOREACH(
                index_matcher_hash, rshift_index_matcher_hash[elreso_leaf], pixs_galind_bounds, rshift_pixs_galind_bounds[elreso_leaf],
                p11, p12, rmax_reso, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, lower, upper){
                for (int ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                    int ind_gal2 = rshift_pix_gals[elreso_leaf] + pix_gals[rshift_pix_gals[elreso_leaf]+ind_inpix];
                    double rel1 = pos1_resos[ind_gal2]-p11, rel2 = pos2_resos[ind_gal2]-p12;
                    double dist = sqrt(rel1*rel1 + rel2*rel2);
                    if (dist<rmin_reso || dist>=rmax_reso){ continue; }
                    int rbin = (int) floor((log(dist)-logrmin)/drbin);
                    if (rbin<rbinmin || rbin>=rbinmax){ continue; }
                    double w2 = weight_resos[ind_gal2];
                    tmp_totcounts[thisthread*nbinsr+rbin] += w2*dist;
                    tmp_totnorms[thisthread*nbinsr+rbin]  += w2;
                    nnnn_leaves_push(lv, rbin-rbinmin, w2, rel1/dist, rel2/dist);
                }
            }
            nnnn_fill_slot(lv, bd.batch+elreso, (int)ig, weight_resos[ind_gal]);
        }
        double time2 = omp_get_wtime();

        // B2: Multipoles of all triples from the batch //
        nnnn_bands_triples(&bd, &plan, pair_order_dt, sc, acc, nthreads);
        time_moments += time2-time1;
        time_triples += omp_get_wtime()-time2;
        print_progress(_elregion, nfilledregions, verbose);
    }
    if (verbose>1){ printf("\nalloc_nnnn_doubletree: moments %.1fs, triples %.1fs (wall)", time_moments, time_triples); }
    if (nmissing > 0){ fprintf(stderr, "alloc_nnnn_doubletree: %ld apices without a parent cell were skipped\n", nmissing); }

    // C: All orderings of every triple //
    nnnn_reconstruct_all(acc, &plan, thetacombis_batches, ncombis, nbinsr, nmax,
                         nindices, len_nindices, N_n, nthreads);

    // Get bin centers
    for (int elthread=1; elthread<nthreads; elthread++){
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            tmp_totcounts[elbinr] += tmp_totcounts[elthread*nbinsr+elbinr];
            tmp_totnorms[elbinr]  += tmp_totnorms[elthread*nbinsr+elbinr];
        }
    }
    for (int elbinr=0; elbinr<nbinsr; elbinr++){
        if (tmp_totnorms[elbinr] != 0){
            bin_centers[elbinr] = tmp_totcounts[elbinr]/tmp_totnorms[elbinr];
        }
    }

    // Free all allocated quantities
    for (int elthread=0; elthread<nthreads; elthread++){ nnnn_leaves_free(leaves+elthread); }
    for (int elreso=0; elreso<nresos; elreso++){
        if (bd.batch[elreso].cap > 0){ nnnn_batch_free(bd.batch+elreso); }
        free(bd.wWns_re[elreso]); free(bd.wWns_im[elreso]); free(slot_of_local[elreso]);
    }
    for (int elreso_grid=0; elreso_grid<nresos_grid; elreso_grid++){ free(pix2redpix[elreso_grid]); }
    free(bd.batch); free(bd.wWns_re); free(bd.wWns_im); free(bd.wWns_cap);
    nnnn_regions_free(&rg); nnnn_scratch_free(sc, nthreads, nresos); nnnn_plan_free(&plan);
    free(pair_order_dt); free(acc); free(leaves); free(pix2redpix);
    free(slot_of_local); free(slot_cap);
    free(rshift_index_matcher_hash); free(rshift_pixs_galind_bounds); free(rshift_pix_gals);
    free(reso_rindedges); free(bin2band); free(tmp_totcounts); free(tmp_totnorms);
    if (verbose>0){ printf("\n"); }
}

// Curved-sky NNNN DoubleTree. As alloc_nnnn_doubletree_flat, but the regions are healpix cells
// at the coarsest navigation level, the parents follow from the nested pixel indices and the
// leaves from query_disc. Recall that all separations are in radians.
static void alloc_nnnn_doubletree_spherical(const MultiresoCatalog *cat, const NavHash *nav,
    const TreeResoParams *tree, const BinningParams *bin, const FourthParams *fourth,
    double memory_bound, int nthreads, int verbose, NPCFOutput *out){

    // Dereference input args
    int nresos = tree->nresos, nresos_grid = tree->nresos_grid;
    double *reso_redges = tree->reso_redges;
    int resoshift_leafs = tree->resoshift_leafs, minresoind_leaf = tree->minresoind_leaf, maxresoind_leaf = tree->maxresoind_leaf;
    double *isinner = cat->isinner_resos, *weight = cat->weight_resos;
    double *vx = cat->vx_resos, *vy = cat->vy_resos, *vz = cat->vz_resos;
    int *ncells_resos = nav->ncells_resos;
    long *nside_nav = nav->nside_nav, *cell_pix = nav->cell_pix;
    int *cell_redbounds = nav->cell_redbounds;
    int *rshift_red = nav->rshift_red, *rshift_cellpix = nav->rshift_cellpix, *rshift_cellbounds = nav->rshift_cellbounds;
    int nmax = bin->nmax, nbinsr = bin->nbinsr;
    double rmin = bin->rmin, rmax = bin->rmax;
    int *nindices = fourth->nindices, len_nindices = fourth->len_nindices;
    int *thetacombis_batches = fourth->thetacombis_batches;
    int *cumthetacombis_batches = fourth->cumthetacombis_batches; int nthetbatches = fourth->nthetbatches;
    double *bin_centers = out->bin_centers;
    double complex *N_n = out->npcf;

    int hasdiscrete = nresos - nresos_grid;
    long ncombis = cumthetacombis_batches[nthetbatches];
    double drbin = (log(rmax)-log(rmin))/(nbinsr);
    double logrmin = log(rmin);

    // A: Bands, regions, work list and scratch //
    int *reso_rindedges = orpheus_calloc(nresos+1, sizeof(int));
    int *bin2band = orpheus_calloc(nbinsr, sizeof(int));
    int *level = orpheus_calloc(nresos, sizeof(int));
    int *slice_clo = orpheus_calloc(nresos, sizeof(int));
    int *slice_chi = orpheus_calloc(nresos, sizeof(int));
    double *tmp_totcounts = orpheus_calloc(nthreads*nbinsr, sizeof(double));
    double *tmp_totnorms = orpheus_calloc(nthreads*nbinsr, sizeof(double));
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){
        free(reso_rindedges); free(bin2band); free(level); free(slice_clo); free(slice_chi);
        free(tmp_totcounts); free(tmp_totnorms);
        return;
    }
    nnnn_reso_rindedges(nresos, reso_redges, rmin, drbin, nbinsr, reso_rindedges);
    for (int elreso=0; elreso<nresos; elreso++){
        for (int elbinr=reso_rindedges[elreso]; elbinr<reso_rindedges[elreso+1]; elbinr++){ bin2band[elbinr] = elreso; }
    }
    // Regions are the occupied cells of the coarsest navigation level. In nested ordering the
    // descendants of a cell are a contiguous range of indices at any finer level.
    int r_region = 0;
    for (int elreso=0; elreso<nresos; elreso++){
        level[elreso] = (nside_nav[elreso]>0) ? __builtin_ctzl((unsigned long)nside_nav[elreso]) : 0;
        if (level[elreso] < level[r_region]){ r_region = elreso; }
    }
    int nregions = ncells_resos[r_region];
    const long *region_cellpix = cell_pix + rshift_cellpix[r_region];

    NnnnPlan plan;
    nnnn_plan_build(&plan, thetacombis_batches, ncombis, nbinsr, nindices, len_nindices, nmax);
    int *pair_order_dt = nnnn_dt_pair_order(&plan, bin2band);
    double complex *acc = orpheus_calloc(ncombis*plan.n2n3combis_half, sizeof(double complex));
    NnnnScratch *sc = nnnn_scratch_alloc(nthreads, nresos, &plan);
    NnnnLeaves *leaves = orpheus_malloc(nthreads*sizeof(NnnnLeaves));
    long *query_cap = orpheus_malloc(nthreads*sizeof(long));
    long **ranges = orpheus_malloc(nthreads*sizeof(long*));
    for (int elthread=0; elthread<nthreads; elthread++){
        nnnn_leaves_alloc(leaves+elthread, nbinsr);
        query_cap[elthread] = 2048;
        ranges[elthread] = orpheus_malloc(2*query_cap[elthread]*sizeof(long));
    }
    NnnnRegions rg;
    nnnn_regions_alloc(&rg, nresos);
    NnnnBands bd = {nresos, reso_rindedges, bin2band, orpheus_calloc(nresos, sizeof(NnnnBatch)), rg.nobj,
                    rg.parent, orpheus_calloc(nresos, sizeof(double*)),
                    orpheus_calloc(nresos, sizeof(double*)), orpheus_calloc(nresos, sizeof(long))};
    // Slot in the batch of the apex in every cell of the region's slice (grid bands)
    int **slot_of_cell = orpheus_calloc(nresos, sizeof(int*));
    int *slot_cap = orpheus_calloc(nresos, sizeof(int));
    int target = nnnn_dt_batch_target(reso_rindedges[1]-reso_rindedges[0], plan.nmax_Wn, memory_bound, nthreads);
    long nmissing = 0;
    // Bail out rather than dereference a failed allocation
    if (orpheus_get_error()){
        fprintf(stderr, "alloc_nnnn_doubletree: allocation of the work list or scratch failed\n");
        exit(1);
    }
    if (verbose>1){ printf("alloc_nnnn_doubletree: %d regions at nside %ld in batches of >=%d apices, %d radial pairs, "
           "%d multipoles (%.2f GiB accumulator)\n", nregions, nside_nav[r_region], target, plan.npairs,
           plan.n2n3combis_half, ncombis*plan.n2n3combis_half*16./1073741824.); }

    // B: Iterate over the batches of regions //
    double time_moments = 0., time_triples = 0.;
    int elregion = 0;
    reset_progress();
    while (elregion < nregions){

        // Collect regions until the batch holds enough apex objects of the finest band
        for (int elreso=0; elreso<nresos; elreso++){ rg.nobj[elreso] = 0; }
        while (elregion < nregions && rg.nobj[0] < target){
            long region_id = region_cellpix[elregion];
            // Coarsest band first, such that every apex finds the slot of its parent
            for (int elreso=nresos-1; elreso>=0; elreso--){
                const long *cp = cell_pix + rshift_cellpix[elreso];
                const int *cb = cell_redbounds + rshift_cellbounds[elreso];
                int shift = 2*(level[elreso]-level[r_region]);
                slice_clo[elreso] = nnnn_lower_bound_long(cp, ncells_resos[elreso], region_id << shift);
                slice_chi[elreso] = nnnn_lower_bound_long(cp, ncells_resos[elreso], (region_id+1) << shift);
                int ncells_slice = slice_chi[elreso]-slice_clo[elreso];
                if (slot_cap[elreso] < ncells_slice){
                    slot_cap[elreso] = ncells_slice;
                    free(slot_of_cell[elreso]); slot_of_cell[elreso] = orpheus_malloc(slot_cap[elreso]*sizeof(int));
                }
                for (int c=slice_clo[elreso]; c<slice_chi[elreso]; c++){
                    slot_of_cell[elreso][c-slice_clo[elreso]] = -1;
                    for (int j=cb[c]; j<cb[c+1]; j++){
                        long g = rshift_red[elreso] + j;
                        if (isinner[g] < 1e-5){ continue; }
                        int parent = -1;
                        if (elreso < nresos-1){
                            long parent_pix;
                            if (elreso < hasdiscrete){
                                double v[3] = {vx[g], vy[g], vz[g]};
                                parent_pix = hpx_ang2pix_nest(nside_nav[elreso+1], v);
                            } else {
                                parent_pix = cp[c] >> 2*(level[elreso]-level[elreso+1]);
                            }
                            const long *cp2 = cell_pix + rshift_cellpix[elreso+1];
                            int c2 = nnnn_lower_bound_long(cp2+slice_clo[elreso+1], slice_chi[elreso+1]-slice_clo[elreso+1], parent_pix) + slice_clo[elreso+1];
                            if (c2 < slice_chi[elreso+1] && cp2[c2] == parent_pix){ parent = slot_of_cell[elreso+1][c2-slice_clo[elreso+1]]; }
                            if (parent < 0){ nmissing += 1; continue; }
                        }
                        slot_of_cell[elreso][c-slice_clo[elreso]] = rg.nobj[elreso];
                        nnnn_regions_push(&rg, elreso, g, parent);
                    }
                }
            }
            elregion += 1;
        }
        nnnn_bands_reserve(&bd, plan.nmax_Wn);
        nnnn_scratch_reserve(sc, nthreads, nresos, rg.nobj, plan.nrowmax);
        // Bail out rather than dereference a failed allocation
        if (orpheus_get_error()){
            fprintf(stderr, "alloc_nnnn_doubletree: allocation of the batch moments failed\n");
            exit(1);
        }

        // B1: Moments of every apex object //
        double time1 = omp_get_wtime();
        long nobj_all = 0;
        for (int elreso=0; elreso<nresos; elreso++){ nobj_all += rg.nobj[elreso]; }
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 4)
        for (long elobj=0; elobj<nobj_all; elobj++){
            int thisthread = omp_get_thread_num();
            NnnnLeaves *lv = leaves + thisthread;
            int elreso = 0;
            long ig = elobj;
            while (ig >= rg.nobj[elreso]){ ig -= rg.nobj[elreso]; elreso++; }
            long g1 = rg.ind_obj[elreso][ig];
            int rbinmin = reso_rindedges[elreso], rbinmax = reso_rindedges[elreso+1];
            if (rbinmax <= rbinmin){ bd.batch[elreso].w1[ig] = weight[g1]; continue; }
            int elreso_leaf = mymin(mymax(minresoind_leaf, elreso+resoshift_leafs), maxresoind_leaf);
            double rmin_reso = rmin*exp(rbinmin*drbin), rmax_reso = rmin*exp(rbinmax*drbin);
            double bx = vx[g1], by = vy[g1], bz = vz[g1];
            double bvec[3] = {bx, by, bz};
            long ns_leaf = nside_nav[elreso_leaf];
            long nranges = hpx_query_disc_nest_ranges(ns_leaf, bvec, rmax_reso, ranges[thisthread], query_cap[thisthread]);
            if (nranges > query_cap[thisthread]){
                query_cap[thisthread] = nranges;
                ranges[thisthread] = realloc(ranges[thisthread], 2*nranges*sizeof(long));
                nranges = hpx_query_disc_nest_ranges(ns_leaf, bvec, rmax_reso, ranges[thisthread], query_cap[thisthread]);
            }
            const long *thisranges = ranges[thisthread];
            const long *cp = cell_pix + rshift_cellpix[elreso_leaf];
            const int *cb = cell_redbounds + rshift_cellbounds[elreso_leaf];
            int ncells_leaf = ncells_resos[elreso_leaf];
            long leafgal_offset = rshift_red[elreso_leaf];
            int ci = 0;
            for (long rr=0; rr<nranges; rr++){
                long plo = thisranges[2*rr], phi = thisranges[2*rr+1];
                ci = nnnn_lower_bound_long(cp+ci, ncells_leaf-ci, plo) + ci;
                while (ci < ncells_leaf && cp[ci] < phi){
                    for (int j=cb[ci]; j<cb[ci+1]; j++){
                        long g2 = leafgal_offset + j;
                        double vx2 = vx[g2], vy2 = vy[g2], vz2 = vz[g2];
                        double dist = sphere_dist(bx, by, bz, vx2, vy2, vz2);
                        if (dist<rmin_reso || dist>=rmax_reso){ continue; }
                        int rbin = (int) floor((log(dist)-logrmin)/drbin);
                        if (rbin<rbinmin || rbin>=rbinmax){ continue; }
                        double w2 = weight[g2];
                        double complex phirot = bearing_phirot(bearing_AB_cart(bx, by, bz, vx2, vy2, vz2));
                        tmp_totcounts[thisthread*nbinsr+rbin] += w2*dist;
                        tmp_totnorms[thisthread*nbinsr+rbin]  += w2;
                        nnnn_leaves_push(lv, rbin-rbinmin, w2, creal(phirot), cimag(phirot));
                    }
                    ci++;
                }
            }
            nnnn_fill_slot(lv, bd.batch+elreso, (int)ig, weight[g1]);
        }
        double time2 = omp_get_wtime();

        // B2: Multipoles of all triples from the batch //
        nnnn_bands_triples(&bd, &plan, pair_order_dt, sc, acc, nthreads);
        time_moments += time2-time1;
        time_triples += omp_get_wtime()-time2;
        print_progress(elregion, nregions, verbose);
    }
    if (verbose>1){ printf("\nalloc_nnnn_doubletree: moments %.1fs, triples %.1fs (wall)", time_moments, time_triples); }
    if (nmissing > 0){ fprintf(stderr, "alloc_nnnn_doubletree: %ld apices without a parent cell were skipped\n", nmissing); }

    // C: All orderings of every triple //
    nnnn_reconstruct_all(acc, &plan, thetacombis_batches, ncombis, nbinsr, nmax,
                         nindices, len_nindices, N_n, nthreads);

    // Get bin centers
    for (int elthread=1; elthread<nthreads; elthread++){
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            tmp_totcounts[elbinr] += tmp_totcounts[elthread*nbinsr+elbinr];
            tmp_totnorms[elbinr]  += tmp_totnorms[elthread*nbinsr+elbinr];
        }
    }
    for (int elbinr=0; elbinr<nbinsr; elbinr++){
        if (tmp_totnorms[elbinr] != 0){
            bin_centers[elbinr] = tmp_totcounts[elbinr]/tmp_totnorms[elbinr];
        }
    }

    // Free all allocated quantities
    for (int elthread=0; elthread<nthreads; elthread++){ nnnn_leaves_free(leaves+elthread); free(ranges[elthread]); }
    for (int elreso=0; elreso<nresos; elreso++){
        if (bd.batch[elreso].cap > 0){ nnnn_batch_free(bd.batch+elreso); }
        free(bd.wWns_re[elreso]); free(bd.wWns_im[elreso]); free(slot_of_cell[elreso]);
    }
    free(bd.batch); free(bd.wWns_re); free(bd.wWns_im); free(bd.wWns_cap);
    nnnn_regions_free(&rg); nnnn_scratch_free(sc, nthreads, nresos); nnnn_plan_free(&plan);
    free(pair_order_dt); free(acc); free(leaves); free(query_cap); free(ranges);
    free(slot_of_cell); free(slot_cap);
    free(reso_rindedges); free(bin2band); free(level); free(slice_clo); free(slice_chi);
    free(tmp_totcounts); free(tmp_totnorms);
    if (verbose>0){ printf("\n"); }
}

// Public entry point: Choose function based on passed metric.
void alloc_nnnn_doubletree(const MultiresoCatalog *cat, const NavHash *nav,
    const TreeResoParams *tree, const BinningParams *bin, const FourthParams *fourth,
    double memory_bound, int nthreads, int verbose, NPCFOutput *out){
    switch (cat->metric) {
        case METRIC_SPHERICAL:
            alloc_nnnn_doubletree_spherical(cat, nav, tree, bin, fourth, memory_bound, nthreads, verbose, out);
            break;
        case METRIC_FLAT:
        default:
            alloc_nnnn_doubletree_flat(cat, nav, tree, bin, fourth, memory_bound, nthreads, verbose, out);
            break;
    }
}