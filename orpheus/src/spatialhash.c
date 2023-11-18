/* External C librabry for the Measurements_dev.py script
   Here, some WL estimators are computed. No firlefanz!
 */

/* Compile with gcc-7 -fopenmp $(gsl-config --cflags) -fPIC -shared -o spatialhash.so spatialhash.c $(gsl-config --libs) -std=c99 (This uses the newer gcc-7 having openmp support maually installed in /usr/local/bin) */
/* On Apollo: gcc -shared -o spatialhash.so -fPIC -fopenmp spatialhash.c  -std=c99 */

/* --- includes --- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "spatialhash.h"

#define _PI_ 3.14159265358979323846
#define FLAG_NOGAL -1  
#define FLAG_OUTSIDE -1  
#define SQUARE(x) ((x)*(x))
#define mymax(x,y) ((x) >= (y)) ? (x) : (y)


void build_spatialhash(double *pos_1, double *pos_2, int ngal,
    double mask_d1, double mask_d2, double mask_min1, double mask_min2, int mask_n1, int mask_n2,
    int *result){

    int npix, npixs_with_gals, pix_1, pix_2, index, noutside;
    int nrelpix, cumsum;
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
    nrelpix = 0;
    cumsum = 0;
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

void reducecat(double *w, double *pos_1, double *pos_2, double *scalarquants, int ngal, int nscalarquants,
               double mask_d1, double mask_d2, double mask_min1, double mask_min2, int mask_n1, int mask_n2,
               double *w_red, double *pos1_red, double *pos2_red, double *scalarquants_red, int ngal_red){
    
    // Build spatial hash
    int npix = mask_n1*mask_n2;
    int start_isoutside = 0;
    int start_matcher = ngal;
    int start_bounds = ngal+npix;
    int start_pixgals = ngal+npix+ngal+1;
    int start_ngalinpix=ngal+npix+ngal+1+ngal;
    int *spatialhash = calloc(2*npix+3*ngal+1, sizeof(int));
    build_spatialhash(pos_1, pos_2, ngal,
                      mask_d1, mask_d2, mask_min1, mask_min2, mask_n1, mask_n2,
                      spatialhash);
    
    // Allocate pixelized catalog from spatial hash
    int ind_pix1, ind_pix2, ind_red, lower, upper, ind_inpix, ind_gal, elscalarquant;
    double tmppos_1, tmppos_2, tmpw;
    double *tmpscalarquants;
    for (ind_pix1=0; ind_pix1<mask_n1; ind_pix1++){
        for (ind_pix2=0; ind_pix2<mask_n2; ind_pix2++){
            ind_red = spatialhash[start_matcher + ind_pix2*mask_n1 + ind_pix1];
            if (ind_red==FLAG_NOGAL){continue;}
            lower = spatialhash[start_bounds+ind_red];
            upper = spatialhash[start_bounds+ind_red+1];
            tmpw = 0;
            tmppos_1 = 0;
            tmppos_2 = 0;
            tmpscalarquants = calloc(nscalarquants, sizeof(double));            
            for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                ind_gal = spatialhash[start_pixgals+ind_inpix];
                tmpw += w[ind_gal];
                tmppos_1 += w[ind_gal]*pos_1[ind_gal];
                tmppos_2 += w[ind_gal]*pos_2[ind_gal];
                for (elscalarquant=0; elscalarquant<nscalarquants; elscalarquant++){
                    tmpscalarquants[elscalarquant] +=  w[ind_gal]*scalarquants[elscalarquant*ngal+ind_gal];
                }
            }
            if (tmpw==0){continue;}
            w_red[ngal_red] = tmpw;
            pos1_red[ngal_red] = tmppos_1/tmpw;
            pos2_red[ngal_red] = tmppos_2/tmpw;
            for (elscalarquant=0; elscalarquant<nscalarquants; elscalarquant++){
                // Here we need to average each of the quantities in order to retain the correct
                // normalization of the NPCF - i.e. for a polar field we would have
                // Upsilon_n,pix ~ w_pix * G1_pix * G2_pix
                //               ~ w_pix * (w_pix * shape_pix * g1) * (w_pix * shape_pix * g2)
                //               ~ w_pix^3 * shape_pix^2
                // This means that shape_pix should be independent of the size of the pixel, i.e. that
                // we should normalize shape_pix ~ (sum_i w_i*gamma_i) / (sum_i w_i)
                scalarquants_red[elscalarquant*ngal+ngal_red] =  tmpscalarquants[elscalarquant]/tmpw;
            }
            ngal_red += 1;
        }
    }
}

void reducecat2(double *pos_1, double *pos_2, double *e1, double *e2, double *w, int ngal,
               double mask_d1, double mask_d2, double mask_min1, double mask_min2, int mask_n1, int mask_n2,
               double *pos1_red, double *pos2_red, double *e1_red, double *e2_red, double *w_red){
    
    int pix_1, pix_2, index;
    for (int ind_gal=0; ind_gal<ngal; ind_gal++){
        pix_1 = (int) floor((pos_1[ind_gal]-mask_min1)/mask_d1);
        pix_2 = (int) floor((pos_2[ind_gal]-mask_min2)/mask_d2);
        index = pix_2*mask_n1+pix_1;
        if (pix_1 > mask_n1 || pix_2 > mask_n2 || pix_1<0 || pix_2<0){continue;}
        else{
            pos1_red[index] += w[ind_gal]*pos_1[ind_gal];
            pos2_red[index] += w[ind_gal]*pos_2[ind_gal];
            e1_red[index] += w[ind_gal]*e1[ind_gal];
            e2_red[index] += w[ind_gal]*e2[ind_gal];
            w_red[index] += w[ind_gal];
        }
    } 
    
    for (index=0; index<mask_n1*mask_n2; index++){
        if (w_red[index]>0){
            pos1_red[index] /= w_red[index];
            pos2_red[index] /= w_red[index];
        }
    }
}