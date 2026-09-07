#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <omp.h>
#include "spatialhash.h"
#include "utils.h"
#include "healpix_utils.h"

#define _PI_ 3.14159265358979323846
#define FLAG_NOGAL -1  
#define FLAG_OUTSIDE -1  
#define SQUARE(x) ((x)*(x))

void build_spatialhash(double *pos_1, double *pos_2, int ngal,
    double mask_d1, double mask_d2, double mask_min1, double mask_min2, int mask_n1, int mask_n2,
    int *result){

    int npix, npixs_with_gals, pix_1, pix_2, index, noutside;
    int noutsiders, index_raw, index_red;
    int ind_gal, ind_pix;

    int start_isoutside, start_matcher, start_bounds, start_pixgals, start_ngalinpix;

    // First step: Allocate number of galaxies per pixel
    // ngals_in_pix = [ngals_in_pix1, ngals_in_pix2, ..., ngals_in_pix-1]
    // s.t. sum(ngals_in_poix) == ngal_tot --> at most ngal_tot non-zero elements
    npix = mask_n1*mask_n2;
    start_isoutside = 0;
    start_matcher = ngal;
    start_bounds = ngal+npix;
    start_pixgals = ngal+npix+ngal+1;
    start_ngalinpix=ngal+npix+ngal+1+ngal;

    npixs_with_gals = 0;
    noutside = 0;
    for (ind_gal=0; ind_gal<ngal; ind_gal++){
        pix_1 = (int) floor((pos_1[ind_gal]-mask_min1)/mask_d1);
        pix_2 = (int) floor((pos_2[ind_gal]-mask_min2)/mask_d2);
        index = pix_2*mask_n1+pix_1;
        if (pix_1 > mask_n1 || pix_2 > mask_n2 || pix_1<0 || pix_2<0){
            result[start_isoutside+ind_gal] = 0;//true
            noutside += 1;}
        else{
            if (result[start_ngalinpix+index] == 0){npixs_with_gals+=1;}
            result[start_isoutside+ind_gal] = 1;//false
            result[start_ngalinpix+index] += 1;
        }
    } 

    // Second step: Allocate pixels with galaxies in them and their bounds
    // index_matcher = [flag_nogal, ..., 0, ..., 1, 2, ..., nrelpixs, flag_nogal, ...]
    //     --> length npix
    // pixs_galind_bounds = [0, ngals_in_pix_a, ngals_in_pix_a + ngals_in_pix_b, ..., ngal_tot, g.a.r.b.a.g.e]
    //     --> length ngal+1
    int nrelpix = 0;
    result[start_bounds+0] = 0;
    for (ind_pix=0; ind_pix<npix; ind_pix++){
        if (result[start_ngalinpix+ind_pix] == 0){result[start_matcher+ind_pix] = FLAG_NOGAL;}
        else{
            result[start_matcher+ind_pix] = nrelpix;
            result[start_bounds+nrelpix+1] = result[start_bounds+nrelpix] + result[start_ngalinpix+ind_pix];
            nrelpix += 1;
        }
    }

    // Third step: Put galaxy indices into pixels
    // pix_gals = [gal1_in_pix_a, ..., gal-1_in_pix_a, ..., gal1_in_pix_n, ..., gal-1_in_pix_n, e.m.p.t.y.g.a.l.s]
    //     --> length ngal
    noutsiders = 0;
    for (ind_gal=0; ind_gal<ngal; ind_gal++){
        if (result[start_isoutside+ind_gal] == 0){
            result[start_pixgals+ngal-noutsiders-1] = FLAG_OUTSIDE;
            noutsiders += 1;
        }
        else{
            pix_1 = (int) floor((pos_1[ind_gal]-mask_min1)/mask_d1);
            pix_2 = (int) floor((pos_2[ind_gal]-mask_min2)/mask_d2);
            index_raw = pix_2*mask_n1+pix_1;
            index_red = result[start_matcher+index_raw];
            index = result[start_bounds+index_red] + result[start_ngalinpix+index_raw]-1;
            result[start_pixgals+index] =  ind_gal;
            result[start_ngalinpix+index_raw] -= 1;
        }
    }
}

void _gen_pixmeans(double *pos_1, double *pos_2, double *e1, double *e2, double *w, double *wc, int ngal,
    double mask_d1, double mask_d2, double mask_min1, double mask_min2, int mask_n1, int mask_n2,
    double *result){

    int npix, pix_1, pix_2, index;
    int ind_gal, ind_1, ind_2;
    double tmp_tot, tmp_rot;
    
    npix = mask_n1*mask_n2;

    for (ind_gal=0; ind_gal<ngal; ind_gal++){
        pix_1 = (int) floor((pos_1[ind_gal]-(mask_min1-.5*mask_d1))/mask_d1);
        pix_2 = (int) floor((pos_2[ind_gal]-(mask_min2-.5*mask_d2))/mask_d2);
        index = pix_2*mask_n1+pix_1;
        if (pix_1 > mask_n1 || pix_2 > mask_n2 || pix_1<0 || pix_2<0){}
        else{
            result[0*npix+index] += wc[ind_gal]*pos_1[ind_gal];
            result[1*npix+index] += wc[ind_gal]*pos_2[ind_gal];
            result[2*npix+index] += w[ind_gal];
            result[3*npix+index] += wc[ind_gal];
            result[4*npix+index] += w[ind_gal]*e1[ind_gal];
            result[5*npix+index] += w[ind_gal]*e2[ind_gal];
        }
    } 
    
    for (ind_1=0; ind_1<mask_n1; ind_1++){
        for (ind_2=0; ind_2<mask_n2; ind_2++){
            index = ind_2*mask_n1+ind_1;
            result[0*npix+index] /= result[3*npix+index];
            result[1*npix+index] /= result[3*npix+index];
            result[4*npix+index] /= result[2*npix+index];
            result[5*npix+index] /= result[2*npix+index];
            
            tmp_tot = sqrt(SQUARE(result[4*npix+index])+SQUARE(result[5*npix+index]));
            if (result[4*npix+index] == 0 && result[5*npix+index] >= 0){tmp_rot = _PI_/4;}
            else if (result[4*npix+index] == 0 && result[5*npix+index] <  0){tmp_rot = -_PI_/4;}
            else if (result[4*npix+index] >  0 && result[5*npix+index] == 0){tmp_rot = 0;}
            else if (result[4*npix+index] <  0 && result[5*npix+index] == 0){tmp_rot = _PI_/2;}
            else {tmp_rot = -atan(-(result[4*npix+index]-tmp_tot)/result[5*npix+index]);}
            result[6*npix+index] = tmp_tot;
            result[7*npix+index] = tmp_rot;
        }
    }  
}

// Parameter shuffle specifies how the pixel center is chosen. Options are:
// 0: Use center of mass
// 1: Do a random shift
// 2: Use pixel center
// 3: Use galaxy with largest weight 
// 4: Use random galaxy (unweighted) 
// 5: Use random galaxy (weighted) (TODO)
void reducecat(double *isinner, double *w, double *pos_1, double *pos_2, double *scalarquants, int ngal, int nscalarquants, 
               int normed,
               double mask_d1, double mask_d2, double mask_min1, double mask_min2, int mask_n1, int mask_n2, int shuffle,
               double *isinner_red, double *w_red, double *pos1_red, double *pos2_red, double *scalarquants_red, int ngal_red){
    
    // Build spatial hash
    int npix = mask_n1*mask_n2;
    //int start_isoutside = 0;
    int start_matcher = ngal;
    int start_bounds = ngal+npix;
    int start_pixgals = ngal+npix+ngal+1;
    //int start_ngalinpix=ngal+npix+ngal+1+ngal;
    int *spatialhash = orpheus_calloc(2*npix+3*ngal+1, sizeof(int));
    build_spatialhash(pos_1, pos_2, ngal,
                      mask_d1, mask_d2, mask_min1, mask_min2, mask_n1, mask_n2,
                      spatialhash);
    
    // Allocate pixelized catalog from spatial hash
    int ind_pix1, ind_pix2, ind_red, lower, upper, ind_inpix, ind_gal, elscalarquant;
    int ind_maxw;
    double tmpisinner, tmppos_1, tmppos_2, tmpw, maxw, tmpdenom, shift_1, shift_2;
    double *tmpscalarquants;
    int rseed=42;
    srand(rseed);  
    for (ind_pix2=0; ind_pix2<mask_n2; ind_pix2++){
        for (ind_pix1=0; ind_pix1<mask_n1; ind_pix1++){
            ind_red = spatialhash[start_matcher + ind_pix2*mask_n1 + ind_pix1];
            if (ind_red==FLAG_NOGAL){continue;}
            lower = spatialhash[start_bounds+ind_red];
            upper = spatialhash[start_bounds+ind_red+1];
            tmpisinner = 0;
            tmpw = 0;
            maxw = 0;
            tmppos_1 = 0;
            tmppos_2 = 0;
            ind_maxw = 0;
            tmpscalarquants = orpheus_calloc(nscalarquants, sizeof(double));
            for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                ind_gal = spatialhash[start_pixgals+ind_inpix];
                tmpisinner += w[ind_gal]*isinner[ind_gal];
                tmpw += w[ind_gal];
                tmppos_1 += w[ind_gal]*pos_1[ind_gal];
                tmppos_2 += w[ind_gal]*pos_2[ind_gal];
                for (elscalarquant=0; elscalarquant<nscalarquants; elscalarquant++){
                    tmpscalarquants[elscalarquant] +=  w[ind_gal]*scalarquants[elscalarquant*ngal+ind_gal];
                }
                if(w[ind_gal]>maxw){ind_maxw=ind_gal;}
            }
            if (tmpw==0){continue;}
            w_red[ngal_red] = tmpw;
            isinner_red[ngal_red] = tmpisinner/tmpw;
            if (shuffle==0){
                pos1_red[ngal_red] = tmppos_1/tmpw;
                pos2_red[ngal_red] = tmppos_2/tmpw;}
            else if (shuffle==1){
                shift_1 = ((double)rand()/(double)(RAND_MAX)) * mask_d1;
                shift_2 = ((double)rand()/(double)(RAND_MAX)) * mask_d2;
                pos1_red[ngal_red] = mask_min1+ind_pix1*mask_d1 + shift_1;
                pos2_red[ngal_red] = mask_min2+ind_pix2*mask_d2 + shift_2;}
            else if (shuffle==2){
                pos1_red[ngal_red] = mask_min1+ind_pix1*mask_d1 + mask_d1/2;
                pos2_red[ngal_red] = mask_min2+ind_pix2*mask_d2 + mask_d2/2;}
            else if (shuffle==3){
                pos1_red[ngal_red] = pos_1[ind_maxw];
                pos2_red[ngal_red] = pos_2[ind_maxw];}
            else if (shuffle==4){
                ind_inpix = (int) (((double) rand()/(double)(RAND_MAX)) * (upper-lower+1));
                ind_gal = spatialhash[start_pixgals+ind_inpix];
                pos1_red[ngal_red] = pos_1[ind_gal];
                pos2_red[ngal_red] = pos_2[ind_gal];}
            for (elscalarquant=0; elscalarquant<nscalarquants; elscalarquant++){
                // It depends on the statistics whether we want to normalize:
                // i) Shear 3pcf
                // Here we need to average each of the quantities in order to retain the correct
                // normalization of the NPCF - i.e. for a polar field we would have
                // Upsilon_n,pix ~ w_pix * G1_pix * G2_pix
                //               ~ w_pix * (w_pix * shape_pix * g1) * (w_pix * shape_pix * g2)
                //               ~ w_pix^3 * shape_pix^2
                // This means that shape_pix should be independent of the size of the pixel, i.e. that
                // we should normalize shape_pix ~ (sum_i w_i*gamma_i) / (sum_i w_i)
                // ii) w^2ww correlators
                // Here we simply compute w_pix^2*w_pix*w_pix, meaning that we should not normalie
                tmpdenom = 1;
                if (normed==1){tmpdenom=tmpw;}
                scalarquants_red[elscalarquant*ngal+ngal_red] =  tmpscalarquants[elscalarquant]/tmpdenom;
            }
            ngal_red += 1;
            free(tmpscalarquants);
        }
    }
    free(spatialhash);
}

// Counter-based PRNG: 
// This is a super basic PRNG that randomizes bits of an input 32 bit integer.
// So here this is a basic thead safe and reproducible option when using openmp.
static inline unsigned int _hash_u32(unsigned int x){
    x += 0x9e3779b9u; // 32 bit golden ratio constant
    x = (x ^ (x >> 16)) * 0x21f0aaadu;
    x = (x ^ (x >> 15)) * 0x735a2d97u;
    return x ^ (x >> 15);
}
// Maps a 32 bit integer to the interval [0,1)
static inline double _u01(unsigned int h){ return (double)h * (1./4294967296.); }

// Reduce a tomographic catalog by aggregating galaxies within each spatial pixel.
// Each occupied (pixel, zbin)-pair produces one reduced galaxy.
void reducecat_tomo(double *isinner, double *w, double *pos_1, double *pos_2, double *scalarquants,
               int *zbins, int ngal, int nscalarquants, int nbinsz, int normed,
               double mask_d1, double mask_d2, double mask_min1, double mask_min2, int mask_n1, int mask_n2, int shuffle,
               int nthreads,
               double *isinner_red, double *w_red, double *pos1_red, double *pos2_red, int *zbins_red, double *scalarquants_red){

    // Single spatial hash over the whole catalog
    int npix = mask_n1*mask_n2;
    int start_matcher = ngal;
    int start_bounds = ngal+npix;
    int start_pixgals = ngal+npix+ngal+1;
    int *spatialhash = orpheus_calloc(2*npix+3*ngal+1, sizeof(int));
    build_spatialhash(pos_1, pos_2, ngal,
                      mask_d1, mask_d2, mask_min1, mask_min2, mask_n1, mask_n2,
                      spatialhash);

    // Build mapping from compact occupied-pixel index back to original pixel index. 
    // We use this for shuffle modes 1 and 2 where the reduced galaxy position is derived from
    // the pixel location.
    int noccupied = 0;
    for (int p=0; p<npix; p++){ if (spatialhash[start_matcher+p]!=FLAG_NOGAL){ noccupied += 1; } }
    if (noccupied==0){ free(spatialhash); return; }
    int *pix_of_red = orpheus_malloc(noccupied*sizeof(int));
    for (int p=0; p<npix; p++){
        int ir = spatialhash[start_matcher+p];
        if (ir!=FLAG_NOGAL){ pix_of_red[ir] = p; }
    }

    // First pass: Count number of occupied zbins in each occupied pixel. A prefix sum over
    // these counts assigns each pixel a disjoint output range [outoffset[ir], outoffset[ir+1]).
    int *pix_nzocc = orpheus_malloc(noccupied*sizeof(int));
    #pragma omp parallel num_threads(nthreads)
    {
        // This array checks whether a zbin was already allocated for the pixel. Does only need
        // to be intialised once as we define allocated by the pixel index which gets refreshed
        // for each new pixel   
        int *counted = orpheus_calloc(nbinsz, sizeof(int)); // Check
        #pragma omp for schedule(dynamic, 64)
        for (int ir=0; ir<noccupied; ir++){
            int lower = spatialhash[start_bounds+ir];
            int upper = spatialhash[start_bounds+ir+1];
            int marker = ir+1;
            int tmpcount = 0;
            for (int j=lower; j<upper; j++){
                int z = zbins[spatialhash[start_pixgals+j]];
                if (counted[z]!=marker){ counted[z]=marker; tmpcount+=1; }
            }
            pix_nzocc[ir] = tmpcount;
        }
        free(counted);
    }
    int *outoffset = orpheus_malloc((noccupied+1)*sizeof(int));
    outoffset[0] = 0;
    for (int ir=0; ir<noccupied; ir++){ outoffset[ir+1] = outoffset[ir] + pix_nzocc[ir]; }

    // Second pass: accumulate weighted quantities for each zbin within a pixel and write
    // one reduced galaxy per occupied zbin.
    #pragma omp parallel num_threads(nthreads)
    {
        double *tmpw  = orpheus_malloc(nbinsz*sizeof(double));
        double *tmpis = orpheus_malloc(nbinsz*sizeof(double));
        double *tmpp1 = orpheus_malloc(nbinsz*sizeof(double));
        double *tmpp2 = orpheus_malloc(nbinsz*sizeof(double));
        double *tmpmw = orpheus_malloc(nbinsz*sizeof(double));
        int *largestwind = orpheus_malloc(nbinsz*sizeof(int));
        int *randind = orpheus_malloc(nbinsz*sizeof(int));
        int *tmpcounts = orpheus_malloc(nbinsz*sizeof(int));
        double *asq = (nscalarquants>0) ? orpheus_malloc((size_t)nbinsz*nscalarquants*sizeof(double)) : NULL;

        #pragma omp for schedule(dynamic, 64)
        for (int ir=0; ir<noccupied; ir++){
            int lower = spatialhash[start_bounds+ir];
            int upper = spatialhash[start_bounds+ir+1];
            int pix = pix_of_red[ir];
            int ind_pix1 = pix % mask_n1;
            int ind_pix2 = pix / mask_n1;

            memset(tmpw,  0, nbinsz*sizeof(double));
            memset(tmpis, 0, nbinsz*sizeof(double));
            memset(tmpp1, 0, nbinsz*sizeof(double));
            memset(tmpp2, 0, nbinsz*sizeof(double));
            if (asq){ memset(asq, 0, (size_t)nbinsz*nscalarquants*sizeof(double)); }
            if (shuffle==3){ memset(tmpmw, 0, nbinsz*sizeof(double)); for (int z=0; z<nbinsz; z++){ largestwind[z]=-1; } }
            if (shuffle==4){ memset(tmpcounts, 0, nbinsz*sizeof(int)); for (int z=0; z<nbinsz; z++){ randind[z]=-1; } }

            for (int j=lower; j<upper; j++){
                int ind_gal = spatialhash[start_pixgals+j];
                int z = zbins[ind_gal];
                double wg = w[ind_gal];
                tmpis[z] += wg*isinner[ind_gal];
                tmpw[z]  += wg;
                tmpp1[z] += wg*pos_1[ind_gal];
                tmpp2[z] += wg*pos_2[ind_gal];
                for (int e=0; e<nscalarquants; e++){
                    asq[(size_t)z*nscalarquants+e] += wg*scalarquants[(size_t)e*ngal+ind_gal];
                }
                if (shuffle==3 && wg>tmpmw[z]){ tmpmw[z]=wg; largestwind[z]=ind_gal; }
                if (shuffle==4){ tmpcounts[z]+=1;
                    unsigned int hr = _hash_u32(_hash_u32((unsigned int)(ir*nbinsz+z)) + (unsigned int)tmpcounts[z]);
                    if (hr % (unsigned int)tmpcounts[z] == 0){ randind[z]=ind_gal; } }
            }

            int slot = outoffset[ir];
            for (int z=0; z<nbinsz; z++){
                if (tmpw[z]==0.){ continue; }
                w_red[slot] = tmpw[z];
                isinner_red[slot] = tmpis[z]/tmpw[z];
                zbins_red[slot] = z;
                if (shuffle==0){
                    pos1_red[slot] = tmpp1[z]/tmpw[z];
                    pos2_red[slot] = tmpp2[z]/tmpw[z]; }
                else if (shuffle==1){
                    // Get random shift by first, scrambling the bits of the output slot to
                    // random 32 bit integer, then map this to [0,1) and normalise by mask_d.
                    // So this is pretty random but still deterministic independent of threads etc.
                    double s1 = _u01(_hash_u32((unsigned int)slot*2u))      * mask_d1;
                    double s2 = _u01(_hash_u32((unsigned int)slot*2u + 1u)) * mask_d2;
                    pos1_red[slot] = mask_min1+ind_pix1*mask_d1 + s1;
                    pos2_red[slot] = mask_min2+ind_pix2*mask_d2 + s2; }
                else if (shuffle==2){
                    pos1_red[slot] = mask_min1+ind_pix1*mask_d1 + mask_d1/2;
                    pos2_red[slot] = mask_min2+ind_pix2*mask_d2 + mask_d2/2; }
                else if (shuffle==3){
                    pos1_red[slot] = pos_1[largestwind[z]];
                    pos2_red[slot] = pos_2[largestwind[z]]; }
                else if (shuffle==4){
                    pos1_red[slot] = pos_1[randind[z]];
                    pos2_red[slot] = pos_2[randind[z]]; }
                for (int e=0; e<nscalarquants; e++){
                    double denom = (normed==1) ? tmpw[z] : 1.;
                    scalarquants_red[(size_t)e*ngal+slot] = asq[(size_t)z*nscalarquants+e]/denom;
                }
                slot += 1;
            }
        }
        free(tmpw); free(tmpis); free(tmpp1); free(tmpp2); free(tmpmw);
        free(largestwind); free(randind); free(tmpcounts);
        if (asq){ free(asq); }
    }

    free(pix_of_red);
    free(pix_nzocc);
    free(outoffset);
    free(spatialhash);
}

//////////////////////////////
/// Spherical spatial hash ///
//////////////////////////////

// Map sky positions in degrees to unit vectors and allocate some trig helpers needed lateron
void sphericalhash_positions(
    const double *ra_deg, const double *dec_deg, long ngal,
    double *vx, double *vy, double *vz, double *ra, double *sindec, double *cosdec,
    int nthreads){

    const double deg2rad = _PI_/180.;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (long i=0; i<ngal; i++){
        double thisra = ra_deg[i]*deg2rad, thisdec = dec_deg[i]*deg2rad;
        double cd = cos(thisdec), sd = sin(thisdec);
        ra[i] = thisra; sindec[i] = sd; cosdec[i] = cd;
        vx[i] = cd*cos(thisra); vy[i] = cd*sin(thisra); vz[i] = sd;
    }
}

// Nested healpix cells at nside, with the tomographic bin folded in for nz>1
void sphericalhash_keys(
    const double *vx, const double *vy, const double *vz, long ngal,
    long nside, const int *zbins, int nz, long *key, int nthreads){

    // The discrete band groups by cell alone and so passes nz=1 even for a tomographic catalog.
    // Folding the bin in regardless would read as pix+zbins there and merge neighbouring cells.
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (long i=0; i<ngal; i++){
        double v1[3] = {vx[i], vy[i], vz[i]};
        long pix = hpx_ang2pix_nest(nside, v1);
        key[i] = (nz>1) ? pix*nz + zbins[i] : pix;
    }
}

// One stable LSD radix pass on the byte at `shift`.
// The basic workflow histogram -> prefix sum -> stable distribution structure is
// classical LSD radix sorting, see i.e. Knuth, TAOCP Vol. 3, §5.2.5). 
// Here we parallelize using the per-chunk histograms and their (bucket, chunk) 
// prefix ordering.
static void radix_pass(const long *kin, const long *pin, long *kout, long *pout,
                       long n, int shift, int nthreads){

    const int nbuckets = 256; // radix = one byte

    // Each input chunk gets its own histogram, avoiding
    // synchronization while counting.
    long *hist = orpheus_calloc((size_t)nthreads*nbuckets, sizeof(long));
    long *offsets = orpheus_malloc((size_t)nthreads*nbuckets*sizeof(long));
    if (orpheus_get_error()){ free(hist); free(offsets); return; }

    // Get counting-sort histogram, performed independently per chunk.
    #pragma omp parallel for num_threads(nthreads) schedule(static, 1)
    for (int c=0; c<nthreads; c++){
        long *tmphist = hist + (long)c*nbuckets;
        for (long i=(n*c)/nthreads; i<(n*(c+1))/nthreads; i++){
            tmphist[(kin[i]>>shift)&255] += 1;
        }
    }
    // Get counting-sort prefix sum. As we process in (bucket, chunk)
    // order assigns each chunk a contiguous part of each bucket; since
    // the chunks are in input order this preserves stability across chunks.
    long cumul = 0;
    for (int b=0; b<nbuckets; b++){
        for (int c=0; c<nthreads; c++){
            offsets[(long)c*nbuckets+b] = cumul;
            cumul += hist[(long)c*nbuckets+b];
        }
    }
    // Stable distribution: records are scanned in input order within each
    // chunk and written at successive positions in that chunk's bucket range.
    #pragma omp parallel for num_threads(nthreads) schedule(static, 1)
    for (int c=0; c<nthreads; c++){
        long *tmpoffsets = offsets + (long)c*nbuckets;
        for (long i=(n*c)/nthreads; i<(n*(c+1))/nthreads; i++){
            long ind = tmpoffsets[(kin[i]>>shift)&255]++;
            kout[ind] = kin[i];
            // First pass starts from the identity permutation; later passes carry it along.
            pout[ind] = (pin==NULL) ? i : pin[i];
        }
    }
    free(hist); free(offsets);
}

// To sort over the whole healpix key range withoug too much memory pressure 
// we sort the keys using LSD radix sort
long sphericalhash_sort(const long *key, long ngal, int nbits,
                        long *order, long *key_sorted, int nthreads){

    if (ngal<=0){ return 0; }

    // Only process bytes containing meaningful key bits.
    int npass = mymax(1, (nbits+7)/8);

    long *tmpkey = orpheus_malloc((size_t)ngal*sizeof(long));
    long *tmporder = orpheus_malloc((size_t)ngal*sizeof(long));
    if (orpheus_get_error()){ free(tmpkey); free(tmporder); return 0; }

    // Standard radix-sort ping-pong buffering.  Choose the destination on each
    // pass so that, after the final pass, the result resides in the caller's
    // key_sorted and order arrays.
    long *keybuf[2] = {tmpkey, key_sorted};
    long *orderbuf[2] = {tmporder, order};
    for (int p=0; p<npass; p++){
        int dest = 1 - ((npass-1-p)&1);
        radix_pass(p==0 ? key : keybuf[1-dest],
                   p==0 ? NULL : orderbuf[1-dest],
                   keybuf[dest], orderbuf[dest],
                   ngal, 8*p, nthreads);
    }
    free(tmpkey); free(tmporder);

    // In sorted order, each distinct spherical hash value begins a new run, 
    // so the number of runs is the number of occupied cells.
    long nocc = 0;
    #pragma omp parallel for num_threads(nthreads) schedule(static) reduction(+:nocc)
    for (long i=0; i<ngal; i++){
        if (i==0 || key_sorted[i]!=key_sorted[i-1]){ nocc += 1; }
    }
    return nocc;
}

// Allocate Discrete band of spherical hash 
// --> The reduced tracers are the tracers themselves, so the band is only permuted
//     into hash order and cut into cells
void sphericalhash_gather(
    const long *order, const long *key_sorted, long ngal,
    const double *vx, const double *vy, const double *vz,
    const double *ra, const double *sindec, const double *cosdec,
    const double *w, const double *isinner, const int *zbins,
    const double *e1, const double *e2, int do_shear, int do_wsq,
    double *red_vx, double *red_vy, double *red_vz,
    double *red_ra, double *red_sindec, double *red_cosdec,
    double *red_w, double *red_isinner, int *red_zbin,
    double *red_e1, double *red_e2, double *red_wsq,
    long *cell_pix, long *cell_redbounds, int nthreads){

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (long i=0; i<ngal; i++){
        long g = order[i];
        red_vx[i] = vx[g]; red_vy[i] = vy[g]; red_vz[i] = vz[g];
        red_ra[i] = ra[g]; red_sindec[i] = sindec[g]; red_cosdec[i] = cosdec[g];
        red_w[i] = w[g]; red_isinner[i] = isinner[g]; red_zbin[i] = zbins[g];
        if (do_shear){ red_e1[i] = e1[g]; red_e2[i] = e2[g]; }
        if (do_wsq){ red_wsq[i] = w[g]*w[g]; }
    }

    long ncells = 0;
    for (long i=0; i<ngal; i++){
        if (i==0 || key_sorted[i]!=key_sorted[i-1]){
            cell_pix[ncells] = key_sorted[i];
            cell_redbounds[ncells] = i;
            ncells += 1;
        }
    }
    cell_redbounds[ncells] = ngal;
}

// Helpers for shuffling
// Get seed based on pixel index
static inline unsigned int cell_seed(long pix){
    return _hash_u32((unsigned int)(pix & 0xffffffffu) ^ _hash_u32((unsigned int)(pix>>32)));
}
// Uniformy draw a random subpixel. As those are equal-area by definition this is exact
static inline long cell_subpix(long pix, int nsplit){
    return (pix << (2*nsplit)) + (long)(cell_seed(pix) & (unsigned int)((1u<<(2*nsplit)) - 1u));
}

// Allocate a reduced band of spherical hash 
// --> Each sorted key is one tracer, placed at the chosen center based on shuffle
//     and transport the shear of the discrete galaxies to it
long sphericalhash_reduce(
    const long *order, const long *key_sorted, long ngal, long nocc,
    long nside, int nz, int shuffle, int navshift,
    const double *vx, const double *vy, const double *vz,
    const double *ra, const double *sindec, const double *cosdec,
    const double *w, const double *isinner,
    const double *e1, const double *e2, int do_shear, int do_wsq,
    double *red_vx, double *red_vy, double *red_vz,
    double *red_ra, double *red_sindec, double *red_cosdec,
    double *red_w, double *red_isinner, int *red_zbin,
    double *red_e1, double *red_e2, double *red_wsq,
    long *cell_pix, long *cell_redbounds, int nthreads){

    long *occbounds = orpheus_malloc((size_t)(nocc+1)*sizeof(long));
    if (orpheus_get_error()){ free(occbounds); return 0; }
    long iocc = 0;
    for (long i=0; i<ngal; i++){
        if (i==0 || key_sorted[i]!=key_sorted[i-1]){ occbounds[iocc] = i; iocc += 1; }
    }
    occbounds[nocc] = ngal;
    // Subcell depth for shuffle 1, capped so that nside<<nsplit stays a legal healpix resolution
    int nsplit = 10;
    while (nsplit>0 && (nside<<nsplit) > (1L<<29)){ nsplit -= 1; }

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (long ic=0; ic<nocc; ic++){

        // Compute helpers for centroid computation within the pixel
        double sumw = 0., sumis = 0., sumwsq = 0., sumx = 0., sumy = 0., sumz = 0.;
        for (long i=occbounds[ic]; i<occbounds[ic+1]; i++){
            long g = order[i];
            double wg = w[g];
            sumw += wg; sumis += isinner[g];
            sumx += wg*vx[g]; sumy += wg*vy[g]; sumz += wg*vz[g];
            if (do_wsq){ sumwsq += wg*wg; }
        }

        // Get pixel index 
        long pix = key_sorted[occbounds[ic]]/nz;

        // Get pixel center based on shuffling convention
        double cx = 0., cy = 0., cz = 0., v1[3];
        long pick = -1;
        switch (shuffle){
            case 0: // Weighted centroid of the members
                {   double norm = sqrt(sumx*sumx + sumy*sumy + sumz*sumz);
                    if (norm==0.){ norm = 1.; }
                    cx = sumx/norm; cy = sumy/norm; cz = sumz/norm; }
                break;
            case 1: // Uniformly random point of the cell
                hpx_pix2vec_nest(nside<<nsplit, cell_subpix(pix, nsplit), v1);
                cx = v1[0]; cy = v1[1]; cz = v1[2];
                break;
            case 2: // Cell center
                hpx_pix2vec_nest(nside, pix, v1);
                cx = v1[0]; cy = v1[1]; cz = v1[2];
                break;
            case 3: // One member drawn at random
                {   long seen = 0;
                    pick = order[occbounds[ic]];
                    for (long i=occbounds[ic]; i<occbounds[ic+1]; i++){
                        seen += 1;
                        if (_hash_u32(cell_seed(pix) + (unsigned int)seen)
                            % (unsigned int)seen == 0u){ pick = order[i]; }
                    }
                    cx = vx[pick]; cy = vy[pick]; cz = vz[pick]; }
                break;
        }
        // Get trig helpers for chosen center
        double cra, csindec, ccosdec;
        if (pick >= 0){
             cra = ra[pick]; csindec = sindec[pick]; ccosdec = cosdec[pick]; 
        }
        else{
            cra = atan2(cy, cx); if (cra<0.){ cra += 2.*_PI_; } csindec = cz; ccosdec = sqrt(mymax(0., 1.-cz*cz));
        }

        // Update the relevant arrays with the chosen center 
        red_vx[ic] = cx; red_vy[ic] = cy; red_vz[ic] = cz;
        red_ra[ic] = cra; red_sindec[ic] = csindec; red_cosdec[ic] = ccosdec;
        red_w[ic] = sumw;
        red_isinner[ic] = (sumis>0.) ? 1. : 0.;
        red_zbin[ic] = (int) (key_sorted[occbounds[ic]]%nz);
        if (do_wsq){ red_wsq[ic] = sumwsq; }

        // Parallel transport the shear of each discrete galaxy to the chosen pixel center
        if (do_shear){
            double sume1 = 0., sume2 = 0.;
            for (long i=occbounds[ic]; i<occbounds[ic+1]; i++){
                long g = order[i];
                double dphi = 2.*(sphere_bearing(cra, csindec, ccosdec,
                                                 ra[g], sindec[g], cosdec[g])
                                  + _PI_
                                  - sphere_bearing(ra[g], sindec[g], cosdec[g],
                                                   cra, csindec, ccosdec));
                double complex wshape = w[g]*(e1[g] + I*e2[g])*cexp(I*dphi);
                sume1 += creal(wshape); sume2 += cimag(wshape);
            }
            double norm = (sumw==0.) ? 1. : sumw;
            red_e1[ic] = sume1/norm; red_e2[ic] = sume2/norm;
        }
    }

    // Update the bookkeeping of occupied cells and ranges 
    long ncells = 0;
    for (long ic=0; ic<nocc; ic++){
        long navpix = (key_sorted[occbounds[ic]]/nz) >> navshift;
        if (ic==0 || navpix!=cell_pix[ncells-1]){
            cell_pix[ncells] = navpix;
            cell_redbounds[ncells] = ic;
            ncells += 1;
        }
    }
    cell_redbounds[ncells] = nocc;

    free(occbounds);

    return ncells;
}
