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

#define mymin(x,y) ((x) <= (y)) ? (x) : (y)
#define mymax(x,y) ((x) >= (y)) ? (x) : (y)
#define M_PI      3.14159265358979323846
#define INV_2PI   0.15915494309189534561

// Non-tomo 4pcf using discrete estimator
// Very basic, no use of symmetry properties
void alloc_notomoGammans_discrete_gggg(
    double *isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int ngal, 
    int nmax, double rmin, double rmax, double *rbins, int nbinsr, int dccorr,
    int *index_matcher_hash, int *pixs_galind_bounds, int *pix_gals, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, int verbose, double *bin_centers, double complex *Upsilon_n, double complex *N_n){
    
    // Temporary arrays that are allocated in parallel and later reduced
    int _nnvals_Upsn = 2*nmax+1;
    int _ups_nshift = nbinsr*nbinsr*nbinsr;
    int _n2n3combis = _nnvals_Upsn*_nnvals_Upsn;
    int _ups_compshift = _n2n3combis*_ups_nshift;
    double *tmpwcounts = calloc(nthreads*nbinsr, sizeof(double));
    double *tmpwnorms = calloc(nthreads*nbinsr, sizeof(double));
    double complex *tmpUpsilon0_n = calloc(nthreads*_ups_compshift, sizeof(double complex));
    double complex *tmpUpsilon1_n = calloc(nthreads*_ups_compshift, sizeof(double complex));
    double complex *tmpUpsilon2_n = calloc(nthreads*_ups_compshift, sizeof(double complex));
    double complex *tmpUpsilon3_n = calloc(nthreads*_ups_compshift, sizeof(double complex));
    double complex *tmpUpsilon4_n = calloc(nthreads*_ups_compshift, sizeof(double complex));
    double complex *tmpUpsilon5_n = calloc(nthreads*_ups_compshift, sizeof(double complex));
    double complex *tmpUpsilon6_n = calloc(nthreads*_ups_compshift, sizeof(double complex));
    double complex *tmpUpsilon7_n = calloc(nthreads*_ups_compshift, sizeof(double complex));
    double complex *tmpN_n = calloc(nthreads*_ups_compshift, sizeof(double complex));
    
    double *totcounts = calloc(nbinsr, sizeof(double));
    double *totnorms = calloc(nbinsr, sizeof(double));
    
    #pragma omp parallel for num_threads(nthreads)
    for(int elthread=0;elthread<nthreads;elthread++){
        int nregions_per_thread = nregions/nthreads;
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

        int nbinszr = nbinsz*nbinsr;
        double complex *nextGns =  calloc(nnvals_Gn*nbinszr, sizeof(double complex));
        double complex *nextG2ns_gg =  calloc(nnvals_G2n*nbinszr, sizeof(double complex));
        double complex *nextG2ns_ggc =  calloc(nnvals_G2n*nbinszr, sizeof(double complex));
        double complex *nextG3ns_ggg = calloc(2*nbinszr, sizeof(double complex));
        double complex *nextG3ns_gggc = calloc(2*nbinszr, sizeof(double complex));
        double complex *nextWns = calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW2ns = calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW3ns = calloc(nbinszr, sizeof(double complex));
        
        for (int elregion=0; elregion<nregions; elregion++){
            int region_debug = mymin(500,nregions-1);
            // Check if this thread is responsible for the region
            int nthread_target = mymin(elregion/nregions_per_thread, nthreads-1);
            if (nthread_target!=elthread){continue;}
            // printf("Region %d is in thread %d\n",elregion,elthread);
            
            if (elregion==region_debug){printf("Region %d is in thread %d (%i regions in total)\n",
                                               elregion,elthread,nregions);}
            if (elthread==nthreads/2){
                printf("\rDone %.2f per cent",
                       100*((double) elregion-nregions_per_thread*(int)(nthreads/2))/nregions_per_thread);
            }
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
                for (int i=0;i<nnvals_Gn*nbinszr;i++){nextGns[i]=0;}
                for (int i=0;i<nnvals_G2n*nbinszr;i++){nextG2ns_gg[i]=0;nextG2ns_ggc[i]=0;}
                for (int i=0;i<2*nbinszr;i++){nextG3ns_ggg[i]=0;nextG3ns_gggc[i]=0;}
                for (int i=0;i<nnvals_Wn*nbinszr;i++){nextWns[i]=0;nextW2ns[i]=0;}
                for (int i=0;i<nbinszr;i++){nextW3ns[i]=0;}
                
                int ind_gal2;
                int ind_red, lower, upper; 
                double  p21, p22, w2, w2_sq, e21, e22, rel1, rel2, dist2, dist, dphi;
                double complex wshape1, wshape1c, wshape2, wshape_sq, wshape_cube, wshapewshapec, wshapesqwshapec;
                double complex phirot, phirotc, twophirotc, fourphirotc, nphirot, nphirotc;

                int ind_rbin, rbin, zrshift, nextnshift, ind_Gn, ind_G2n, ind_Wn;
                double drbin = (log(rmax)-log(rmin))/(nbinsr);
                double rmin2 = rmin*rmin; 
                double rmax2 = rmax*rmax;
                int pix1_lower = mymax(0, (int) floor((p11 - (rmax+pix1_d) - pix1_start)/pix1_d));
                int pix2_lower = mymax(0, (int) floor((p12 - (rmax+pix2_d) - pix2_start)/pix2_d));
                int pix1_upper = mymin(pix1_n-1, (int) floor((p11 + (rmax+pix1_d) - pix1_start)/pix1_d));
                int pix2_upper = mymin(pix2_n-1, (int) floor((p12 + (rmax+pix2_d) - pix2_start)/pix2_d));
                for (int ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                    for (int ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                        ind_red = index_matcher_hash[ind_pix2*pix1_n + ind_pix1];
                        if (ind_red==-1){continue;}
                        lower = pixs_galind_bounds[ind_red];
                        upper = pixs_galind_bounds[ind_red+1];
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
                            if(dist2 < rmin2 || dist2 >= rmax2){continue;}
                            dist = sqrt(dist2);
                            if (rbins[0] < 0){
                                rbin = (int) floor((log(dist)-log(rmin))/drbin);
                            }
                            else{
                                rbin=0;
                                while(rbins[rbin+1] <= dist){rbin+=1;}
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
                            nphirot = 1+I*0;
                            nphirotc = 1+I*0;
                            
                            // Triple-counting corr
                            nextW3ns[zrshift] += w2_sq*w2;
                            nextG3ns_ggg[zrshift] += wshape_cube*fourphirotc;
                            nextG3ns_ggg[nbinszr + zrshift] += wshape_cube*fourphirotc*fourphirotc;
                            nextG3ns_gggc[zrshift] += wshapesqwshapec;
                            nextG3ns_gggc[nbinszr + zrshift] += wshapesqwshapec*fourphirotc;                            
                            
                            // Nominal G and double-counting corr
                            // n = 0
                            tmpwcounts[ind_rbin] += w1*w2*dist; 
                            tmpwnorms[ind_rbin] += w1*w2; 
                            nextGns[ind_Gn] += wshape2*nphirot;
                            nextG2ns_gg[ind_G2n] += wshape_sq*nphirot;
                            nextG2ns_ggc[ind_G2n] += wshapewshapec*nphirot;
                            nextWns[ind_Wn] += w2*nphirot;  
                            nextW2ns[ind_Wn] += w2_sq*nphirot;
                            // /*
                            // n \in [-2*nmax+1,2*nmax-1]                          
                            nphirot *= phirot;
                            nphirotc *= phirotc; 
                            // n in [1, ..., nmax-1] x {+1,-1}
                            nextnshift = 0;
                            for (int nextn=1;nextn<2*nmax;nextn++){
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
                    }
                }
                
                // Allocate Upsilon
                // Upsilon have shape 
                // (8,(2*nmax+1),(2*nmax+1),nbinsr,nbinsr,nbinsr)
                // Ups_0 ~ wgamma  *  G_{n2+n3-3}  *  G_{-n2-2}  *  G_{-n3-3}
                // Ups_1 ~ wgammac *  G_{n2+n3-1}  *  G_{-n2-2}  *  G_{-n3-1}
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
                                    nextG2ns_gg[(nzero_G2n+thisn3-5)*nbinsr+elb1]  * nextGns[thisGshift_mn3m3+elb2];
                                tmpUpsilon7_n[thisnrshift] -= wshape1c  *
                                    nextG2ns_gg[(nzero_G2n+thisn3-5)*nbinsr+elb1]  * nextGns[thisGshift_mn3m3+elb2];
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
    // Given the openmp implementation this needs to be done sequentially...however,
    // as the threads will reach this step at different points in time, it will
    // most likely not be a severe bottleneck.        
    int thisn_thread, thisn2, thisn3, thistmpnshift, thisnshift, thisnrshift, ind_Upsn;
    double complex toadd;
    #pragma omp parallel for num_threads(nthreads)
    for (int thisn=0; thisn<_n2n3combis; thisn++){
        thisn2 = thisn_thread/_nnvals_Upsn;
        thisn3 = thisn_thread%_nnvals_Upsn;
        thisnshift = thisn*_ups_nshift;
        for (int elb1=0; elb1<nbinsr; elb1++){
            for (int elb2=0; elb2<nbinsr; elb2++){
                for (int elb3=0; elb3<nbinsr; elb3++){
                    for (int elthread=0; elthread<nthreads; elthread++){
                        thisnrshift = thisnshift + elb1*nbinsr*nbinsr + elb2*nbinsr + elb3;
                        thistmpnshift = elthread*_ups_compshift+thisnrshift;
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
}

// Non-tomo 4pcf using tree-based estimator
void alloc_notomoGammans_tree_gggg(
    double *isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int ngal, 
    int nmax, double rmin, double rmax, int nbinsr, int nthetacombis, int dccorr,
    int *nindices, int len_nindices, 
    int nresos, double *reso_redges, int *ngal_resos, 
    double *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos, 
    double *e1_resos, double *e2_resos,
    int *index_matcher_hash, int *pixs_galind_bounds, int *pix_gals, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, int verbose, double *bin_centers, double complex *Upsilon_n, double complex *N_n){
    
    // Temporary arrays that are allocated in parallel and later reduced
    double *tmpwcounts = calloc(nthreads*nbinsr, sizeof(double));
    double *tmpwnorms = calloc(nthreads*nbinsr, sizeof(double));
    double complex *tmpUpsilon0_n = calloc(nthreads*len_nindices*nthetacombis, sizeof(double complex));
    double complex *tmpUpsilon1_n = calloc(nthreads*len_nindices*nthetacombis, sizeof(double complex));
    double complex *tmpUpsilon2_n = calloc(nthreads*len_nindices*nthetacombis, sizeof(double complex));
    double complex *tmpUpsilon3_n = calloc(nthreads*len_nindices*nthetacombis, sizeof(double complex));
    double complex *tmpUpsilon4_n = calloc(nthreads*len_nindices*nthetacombis, sizeof(double complex));
    double complex *tmpUpsilon5_n = calloc(nthreads*len_nindices*nthetacombis, sizeof(double complex));
    double complex *tmpUpsilon6_n = calloc(nthreads*len_nindices*nthetacombis, sizeof(double complex));
    double complex *tmpUpsilon7_n = calloc(nthreads*len_nindices*nthetacombis, sizeof(double complex));
    double complex *tmpN_n = calloc(nthreads*len_nindices*nthetacombis, sizeof(double complex));
    
    double *totcounts = calloc(nbinsr, sizeof(double));
    double *totnorms = calloc(nbinsr, sizeof(double));

    // Helper array that checks how many regions have been already computed
    int *regionsdone = calloc(nregions, sizeof(int));
    int nregionsdone = 0;
    
    #pragma omp parallel for num_threads(nthreads)
    for(int elthread=0;elthread<nthreads;elthread++){
        int nregions_per_thread = nregions/nthreads;
        int nmax_alloc = 2*nmax+1;
        int nbinsz = 1;
        int ncomp = 8;
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
        double complex *nextGns =  calloc(nnvals_Gn*nbinszr, sizeof(double complex));
        double complex *nextG2ns_gg =  calloc(nnvals_G2n*nbinszr, sizeof(double complex));
        double complex *nextG2ns_ggc =  calloc(nnvals_G2n*nbinszr, sizeof(double complex));
        double complex *nextG3ns_ggg = calloc(2*nbinszr, sizeof(double complex));
        double complex *nextG3ns_gggc = calloc(2*nbinszr, sizeof(double complex));
        double complex *nextWns = calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW2ns = calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW3ns = calloc(nbinszr, sizeof(double complex));

        int npix_hash = pix1_n*pix2_n;
        int *rshift_index_matcher_hash = calloc(nresos, sizeof(int));
        int *rshift_pixs_galind_bounds = calloc(nresos, sizeof(int));
        int *rshift_pix_gals = calloc(nresos, sizeof(int));
        for (int elreso=1;elreso<nresos;elreso++){
            rshift_index_matcher_hash[elreso] = rshift_index_matcher_hash[elreso-1] + npix_hash;
            rshift_pixs_galind_bounds[elreso] = rshift_pixs_galind_bounds[elreso-1] + ngal_resos[elreso-1]+1;
            rshift_pix_gals[elreso] = rshift_pix_gals[elreso-1] + ngal_resos[elreso-1];
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
                for (int i=0;i<nnvals_Gn*nbinszr;i++){nextGns[i]=0;}
                for (int i=0;i<nnvals_G2n*nbinszr;i++){nextG2ns_gg[i]=0;nextG2ns_ggc[i]=0;}
                for (int i=0;i<2*nbinszr;i++){nextG3ns_ggg[i]=0;nextG3ns_gggc[i]=0;}
                for (int i=0;i<nnvals_Wn*nbinszr;i++){nextWns[i]=0;nextW2ns[i]=0;}
                for (int i=0;i<nbinszr;i++){nextW3ns[i]=0;}
                
                int ind_gal2;
                int ind_red, lower, upper; 
                double  p21, p22, w2, w2_sq, e21, e22, rel1, rel2, dist2, dist, dphi;
                double complex wshape1, wshape1c, wshape2, wshape_sq, wshape_cube, wshapewshapec, wshapesqwshapec;
                double complex phirot, phirotc, twophirotc, fourphirotc, nphirot, nphirotc;
                // Allocate Gn, Wn and their multiple-couting corrections
                for (int elreso=0;elreso<=nresos;elreso++){
                    int ind_rbin, rbin, zrshift, nextnshift, ind_Gn, ind_G2n, ind_Wn;
                    double rmin_reso = reso_redges[elreso];
                    double rmin_reso2 = rmin_reso*rmin_reso;
                    double rmax_reso = reso_redges[elreso+1];
                    double rmax_reso2 = rmax_reso*rmax_reso;
                    int pix1_lower = mymax(0, (int) floor((p11 - (rmax_reso+pix1_d) - pix1_start)/pix1_d));
                    int pix2_lower = mymax(0, (int) floor((p12 - (rmax_reso+pix2_d) - pix2_start)/pix2_d));
                    int pix1_upper = mymin(pix1_n-1, (int) floor((p11 + (rmax_reso+pix1_d) - pix1_start)/pix1_d));
                    int pix2_upper = mymin(pix2_n-1, (int) floor((p12 + (rmax_reso+pix2_d) - pix2_start)/pix2_d));
                    for (int ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                        for (int ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                            ind_red = index_matcher_hash[rshift_index_matcher_hash[elreso] + ind_pix2*pix1_n + ind_pix1];
                            if (ind_red==-1){continue;}
                            lower = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+ind_red];
                            upper = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+ind_red+1];
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
                                if(dist2 < rmin_reso2 || dist2 >= rmax_reso2){continue;}
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
                                nphirot = 1+I*0;
                                nphirotc = 1+I*0;

                                // Triple-counting corr
                                nextW3ns[zrshift] += w2_sq*w2;
                                nextG3ns_ggg[zrshift] += wshape_cube*fourphirotc;
                                nextG3ns_ggg[nbinszr + zrshift] += wshape_cube*fourphirotc*fourphirotc;
                                nextG3ns_gggc[zrshift] += wshapesqwshapec;
                                nextG3ns_gggc[nbinszr + zrshift] += wshapesqwshapec*fourphirotc;                            

                                // Nominal G and double-counting corr
                                // n = 0
                                totcounts[zrshift] += w1*w2*dist; 
                                totnorms[zrshift] += w1*w2; 
                                nextGns[ind_Gn] += wshape2*nphirot;
                                nextG2ns_gg[ind_G2n] += wshape_sq*nphirot;
                                nextG2ns_ggc[ind_G2n] += wshapewshapec*nphirot;
                                nextWns[ind_Wn] += w2*nphirot;  
                                nextW2ns[ind_Wn] += w2_sq*nphirot;
                                // /*
                                // n \in [-2*nmax+1,2*nmax-1]                          
                                nphirot *= phirot;
                                nphirotc *= phirotc; 
                                // n in [1, ..., 2*nmax_alloc-1] x {+1,-1}
                                nextnshift = 0;
                                for (int nextn=1;nextn<2*nmax_alloc;nextn++){
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

                                // n = \pm 2*nmax_alloc
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

                                // n = \pm 2*nmax_alloc+1 
                                nextnshift += nbinszr;
                                nextGns[ind_Gn-nextnshift] += wshape2*nphirotc;
                                nextG2ns_gg[ind_G2n+nextnshift] += wshape_sq*nphirot;
                                nextG2ns_gg[ind_G2n-nextnshift] += wshape_sq*nphirotc;
                                nextG2ns_ggc[ind_G2n+nextnshift] += wshapewshapec*nphirot;
                                nextG2ns_ggc[ind_G2n-nextnshift] += wshapewshapec*nphirotc;
                                nphirotc *= phirotc;
                                // n =  -2*nmax_alloc-2
                                nextnshift += nbinszr;
                                nextGns[ind_Gn-nextnshift] += wshape2*nphirotc;
                                nextG2ns_gg[ind_G2n-nextnshift] += wshape_sq*nphirotc;
                                nextG2ns_ggc[ind_G2n-nextnshift] += wshapewshapec*nphirotc;
                                nphirotc *= phirotc;
                                // n =  -2*nmax_alloc-3
                                nextnshift += nbinszr;
                                nextGns[ind_Gn-nextnshift] += wshape2*nphirotc;
                                nextG2ns_gg[ind_G2n-nextnshift] += wshape_sq*nphirotc;
                                nextG2ns_ggc[ind_G2n-nextnshift] += wshapewshapec*nphirotc;
                                nphirotc *= phirotc;
                                // n =  -2*nmax_alloc-4
                                nextnshift += nbinszr;
                                nextG2ns_gg[ind_G2n-nextnshift] += wshape_sq*nphirotc;
                                nextG2ns_ggc[ind_G2n-nextnshift] += wshapewshapec*nphirotc;
                                nphirotc *= phirotc;
                                // n =  -2*nmax_alloc-5
                                nextnshift += nbinszr;
                                nextG2ns_gg[ind_G2n-nextnshift] += wshape_sq*nphirotc;
                                nextG2ns_ggc[ind_G2n-nextnshift] += wshapewshapec*nphirotc;
                            }
                        }
                    }
                }
                
                // Allocate Upsilon
                // Upsilon_mu have shape 
                // (nindices, rcombis)
                // Ups_0 ~ wgamma  *  G_{n2+n3-3}  *  G_{-n2-2}  *  G_{-n3-3}
                // Ups_1 ~ wgammac *  G_{n2+n3-1}  *  G_{-n2-2}  *  G_{-n3-1}
                double complex gGG0, gGG1, gGG2, gGG3, gGG4, gGG5, gGG6, gGG7, wNN;
                int thisn2, thisn3, thisn, thisnshift, thisnrshift, elbcombi, elb1, elb2, elb3;
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
                            thisnrshift = elthread*ups_compshift + thisnshift + elbcombi;
                            // Double-counting corr for theta1==theta2
                            if (elb1==elb2){
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
                                    nextG2ns_gg[(nzero_G2n+thisn3-5)*nbinsr+elb1]  * nextGns[thisGshift_mn3m3+elb2];
                                tmpUpsilon7_n[thisnrshift] -= wshape1c  *
                                    nextG2ns_gg[(nzero_G2n+thisn3-5)*nbinsr+elb1]  * nextGns[thisGshift_mn3m3+elb2];
                                tmpN_n[thisnrshift] -= w1 * 
                                    nextW2ns[(nzero_Wn+thisn3)*nbinsr+elb1]*conj(nextWns[thisWshift_n3+elb2]);
                            }

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
                                // Double-counting corr for theta1==theta3 
                                if ((elb1==elb3) && (elb1!=elb2)){ 
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
                                if ((elb2==elb3) && (elb1!=elb2)){ 
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
    int *elb1_inds = calloc(nthetacombis, sizeof(int));
    int *elb2_inds = calloc(nthetacombis, sizeof(int));
    int *elb3_inds = calloc(nthetacombis, sizeof(int));
    for (int elb1=0;elb1<nbinsr;elb1++){
        for (int elb2=elb1;elb2<nbinsr;elb2++){
            for (int elb3=elb2;elb3<nbinsr;elb3++){
                elb1_inds[elbcombi] = elb1;
                elb2_inds[elbcombi] = elb2;
                elb3_inds[elbcombi] = elb3;
                elbcombi += 1;
            }
        }
    }

    
    #pragma omp parallel for num_threads(nthreads)
    for (int elb=0;elb<nthetacombis;elb++){

        int ntrafos, tnrshift, nbshift, nbshift_tmp, elb1, elb2, elb3, elb1t, elb2t, elb3t;
        int thisn2, thisn3, thisn;
        int nmax_alloc = 2*nmax+1;
        int nnvals_Upsn_rec = 2*nmax+1; 
        int nnvals_Upsn = 2*nmax_alloc+1; 
        int nzero_Ups = nmax_alloc;
        int ups_nshift = nbinsr*nbinsr*nbinsr;
        int n2n3combis = nnvals_Upsn*nnvals_Upsn;
        int n2n3combis_rec = nnvals_Upsn_rec*nnvals_Upsn_rec;
        int ups_rec_compshift = n2n3combis_rec*ups_nshift;

        double complex *thisUpsilon_n = calloc(8*n2n3combis, sizeof(double complex));
        double complex *thisN_n = calloc(n2n3combis, sizeof(double complex));
        double complex *thisUpsilon_n_rec = calloc(8*n2n3combis_rec, sizeof(double complex));
        double complex *thisN_n_rec = calloc(n2n3combis_rec, sizeof(double complex));

        // 2)
        elb1 = elb1_inds[elb];
        elb2 = elb2_inds[elb];
        elb3 = elb3_inds[elb];
        int bincombi_trafos[6][3] = {{elb1,elb2,elb3}, {elb2,elb3,elb1}, {elb3,elb1,elb2},
                                     {elb1,elb3,elb2}, {elb2,elb1,elb3}, {elb3,elb2,elb1}}; 
        
        // 3)
        if ((elb1==elb2)&&(elb1==elb3)){ntrafos=1;}
        else if ((elb1==elb2)&&(elb1!=elb3)){ntrafos=3;}
        else if ((elb1==elb3)&&(elb1!=elb2)){ntrafos=3;}
        else if ((elb2==elb3)&&(elb2!=elb1)){ntrafos=3;}
        else{ntrafos=6;}
        for (int eltrafo=0;eltrafo<ntrafos;eltrafo++){
            elb1t = bincombi_trafos[eltrafo][0];
            elb2t = bincombi_trafos[eltrafo][1];
            elb3t = bincombi_trafos[eltrafo][2];
            for (int nindex=0;nindex<len_nindices;nindex++){
                thisn2 = nindices[nindex]/nnvals_Upsn - nzero_Ups;
                thisn3 = nindices[nindex]%nnvals_Upsn - nzero_Ups;
                nbshift_tmp = nindex*nthetacombis+elb;
                nbshift = ((thisn2+nzero_Ups)*nnvals_Upsn + (thisn3+nzero_Ups));
                for (int elthread=0;elthread<nthreads;elthread++){
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
            for(int eln12=0;eln12<n2n3combis_rec;eln12++){
                int thisnrshift = eln12*ups_nshift + elb1t*nbinsr*nbinsr + elb2t*nbinsr + elb3t;
                for (int elcomp=0;elcomp<8;elcomp++){
                    Upsilon_n[elcomp*ups_rec_compshift+thisnrshift] =  thisUpsilon_n_rec[elcomp*n2n3combis_rec+eln12];
                }
                N_n[thisnrshift] = thisN_n_rec[eln12];
            }  

            // Reset 4pcf placeholders to zero
            for(int i=0;i<n2n3combis;i++){
                thisN_n[i] = 0;
                for (int elcomp=0;elcomp<8;elcomp++){
                    thisUpsilon_n[elcomp*n2n3combis+i] = 0;
                }
            }
            for(int i=0;i<n2n3combis_rec;i++){
                thisN_n_rec[i] = 0;
                for (int elcomp=0;elcomp<8;elcomp++){
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
void alloc_notomoMap4_disc_gggg(
    double *isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int ngal, 
    int nmax, double rmin, double rmax, int nbinsr, int dccorr, double *phibins, double *dbinsphi, int nbinsphi,
    int *index_matcher_hash, int *pixs_galind_bounds, int *pix_gals, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, int verbose, int projection, double *mapradii, int nmapradii, double complex *M4correlators, 
    int alloc_4pcfmultipoles, int alloc_4pcfreal,
    double *bin_centers, double complex *Upsilon_n, double complex *N_n, double complex *Gammas, double complex *Norms){
               
    double complex *allM4correlators = calloc(nthreads*8*1*nmapradii, sizeof(double complex));
    #pragma omp parallel for num_threads(nthreads)
    for(int elthetbatch=0;elthetbatch<nthetbatches;elthetbatch++){
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
        
        double *totcounts = calloc(nbinsr, sizeof(double));
        double *totnorms = calloc(nbinsr, sizeof(double));
        double *bin_centers_batch = calloc(nbinsr, sizeof(double));
        double complex *batchUpsilon_n = calloc(ncomp*batchups_compshift, sizeof(double complex));
        double complex *batchN_n = calloc(batchups_compshift, sizeof(double complex));
        double complex *batchfourpcf = calloc(ncomp*batchups_compshift, sizeof(double complex));
        double complex *batchfourpcf_norm = calloc(batchups_compshift, sizeof(double complex));
        double *batch_thetas1 = calloc(batch_nthetas, sizeof(double));
        double *batch_thetas2 = calloc(batch_nthetas, sizeof(double));
        double *batch_thetas3 = calloc(batch_nthetas, sizeof(double));
        
        int nbinszr = nbinsz*nbinsr;
        double complex *nextGns =  calloc(nnvals_Gn*nbinszr, sizeof(double complex));
        double complex *nextG2ns_gg =  calloc(nnvals_G2n*nbinszr, sizeof(double complex));
        double complex *nextG2ns_ggc =  calloc(nnvals_G2n*nbinszr, sizeof(double complex));
        double complex *nextG3ns_ggg = calloc(2*nbinszr, sizeof(double complex));
        double complex *nextG3ns_gggc = calloc(2*nbinszr, sizeof(double complex));
        double complex *nextWns = calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW2ns = calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW3ns = calloc(nbinszr, sizeof(double complex));
        
        double drbin = (log(rmax)-log(rmin))/(nbinsr);
        int *elb1s_batch = calloc(batch_nthetas, sizeof(int));
        int *elb2s_batch = calloc(batch_nthetas, sizeof(int));
        int *elb3s_batch = calloc(batch_nthetas, sizeof(int));
        double *bin_edges = calloc(nbinsr+1, sizeof(double));
        #pragma omp critical
        {
        for (int elb=0;elb<batch_nthetas;elb++){
            int thisrcombi = thetacombis_batches[cumthetacombis_batches[elthetbatch]+elb];
            elb1s_batch[elb] = thisrcombi/(nbinsr*nbinsr);
            elb2s_batch[elb] = (thisrcombi-elb1s_batch[elb]*nbinsr*nbinsr)/nbinsr;
            elb3s_batch[elb] = thisrcombi-elb1s_batch[elb]*nbinsr*nbinsr-elb2s_batch[elb]*nbinsr;
        }
        bin_edges[0] = rmin;
        for (int elb=0;elb<nbinsr;elb++){
            bin_edges[elb+1] = bin_edges[elb]*exp(drbin);
        }
        }
           
        // Allocate the 4pcf multipoles for this batch of radii 
        int offset_per_thread = nregions/nthreads;
        int offset = offset_per_thread*thisthread;
        for (int _elregion=0; _elregion<nregions; _elregion++){
            int elregion = (_elregion+offset)%nregions; // Try to evade collisions
            if ((elregion%nregions_skip_print == 0)&&(thisthread==0)){
                printf("Doing region %d/%d for thetabatch %d/%d\n",elregion,nregions,elthetbatch,nthetbatches);
            }
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
                int ind_red, lower, upper; 
                double  p21, p22, w2, w2_sq, e21, e22, rel1, rel2, dist, dphi;
                double complex wshape1, wshape1c, wshape2, wshape_sq, wshape_cube, wshapewshapec, wshapesqwshapec;
                double complex phirot, phirotc, twophirotc, fourphirotc, nphirot, nphirotc;
                
                // Check how many ns we need for Gn
                // Gns have shape (nnvals, nbinsz, nbinsr)
                // where the ns are ordered as 
                // [-nmax_1-nmax_2-3, ..., nmax_1+nmax_2+3]
                
                for (int i=0;i<nnvals_Gn*nbinszr;i++){nextGns[i]=0;}
                for (int i=0;i<nnvals_G2n*nbinszr;i++){nextG2ns_gg[i]=0;nextG2ns_ggc[i]=0;}
                for (int i=0;i<2*nbinszr;i++){nextG3ns_ggg[i]=0;nextG3ns_gggc[i]=0;}
                for (int i=0;i<nnvals_Wn*nbinszr;i++){nextWns[i]=0;nextW2ns[i]=0;}
                for (int i=0;i<nbinszr;i++){nextW3ns[i]=0;}

                int rbin, zrshift, nextnshift, ind_Gn, ind_G2n, ind_Wn;
                int pix1_lower = mymax(0, (int) floor((p11 - (rmax+pix1_d) - pix1_start)/pix1_d));
                int pix2_lower = mymax(0, (int) floor((p12 - (rmax+pix2_d) - pix2_start)/pix2_d));
                int pix1_upper = mymin(pix1_n-1, (int) floor((p11 + (rmax+pix1_d) - pix1_start)/pix1_d));
                int pix2_upper = mymin(pix2_n-1, (int) floor((p12 + (rmax+pix2_d) - pix2_start)/pix2_d));
                
                for (int ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                    for (int ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                        ind_red = index_matcher_hash[ind_pix2*pix1_n + ind_pix1];
                        if (ind_red==-1){continue;}
                        lower = pixs_galind_bounds[ind_red];
                        upper = pixs_galind_bounds[ind_red+1];
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
                            if(dist < rmin || dist >= rmax) continue;
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
                            nphirot = 1+I*0;
                            nphirotc = 1+I*0;
                            
                            // Triple-counting corr
                            nextW3ns[zrshift] += w2_sq*w2;
                            nextG3ns_ggg[zrshift] += wshape_cube*fourphirotc;
                            nextG3ns_ggg[nbinszr + zrshift] += wshape_cube*fourphirotc*fourphirotc;
                            nextG3ns_gggc[zrshift] += wshapesqwshapec;
                            nextG3ns_gggc[nbinszr + zrshift] += wshapesqwshapec*fourphirotc;                            
                            
                            // Nominal G and double-counting corr
                            // n = 0
                            totcounts[zrshift] += w1*w2*dist; 
                            totnorms[zrshift] += w1*w2; 
                            nextGns[ind_Gn] += wshape2*nphirot;
                            nextG2ns_gg[ind_G2n] += wshape_sq*nphirot;
                            nextG2ns_ggc[ind_G2n] += wshapewshapec*nphirot;
                            nextWns[ind_Wn] += w2*nphirot;  
                            nextW2ns[ind_Wn] += w2_sq*nphirot;
                            // /*
                            // n \in [-2*nmax+1,2*nmax-1]                          
                            nphirot *= phirot;
                            nphirotc *= phirotc; 
                            // n in [1, ..., nmax-1] x {+1,-1}
                            nextnshift = 0;
                            for (int nextn=1;nextn<2*nmax;nextn++){
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
                    }
                }
                time2 = omp_get_wtime();
                if ((elregion%nregions_skip_print == 0)&&(thisthread==0)&&(ind_inpix1==lower1)){
                    printf("Computed Gn for first gal in region %d/%d for thetabatch %d/%d in %.4f seconds\n",
                           elregion,nregions,elthetbatch,nthetbatches,(time2-time1));}
                
                time1 = omp_get_wtime();
                // Allocate Upsilon
                // Upsilon have shape 
                // (ncomp,(2*nmax+1),(2*nmax+1),nthetas)
                double complex gGG0, gGG1, gGG2, gGG3, gGG4, gGG5, gGG6, gGG7, wNN;
                int thisn, thisnshift, thisnrshift, elb1, elb2, elb3;
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
                        for (int elb=0;elb<batch_nthetas;elb++){
                            //int thisrcombi = thetacombis_batches[cumthetacombis_batches[elthetbatch]+elb];
                            //elb1 = thisrcombi/(nbinsr*nbinsr);
                            //elb2 = (thisrcombi-elb1*nbinsr*nbinsr)/nbinsr;
                            //elb3 = thisrcombi-elb1*nbinsr*nbinsr-elb2*nbinsr;
                            elb1 = elb1s_batch[elb];
                            elb2 = elb2s_batch[elb];
                            elb3 = elb3s_batch[elb];
                            thisnrshift = thisnshift + elb;
                            // Multiple counting corrections:
                            // sum_(i neq j neq k) = sum_(i,j,k) - ( sum_(i, j, i=k) + 2perm ) + 2 * sum_(i, i=j, i=k)
                            // Triple-counting corr
                            if ((elb1==elb2) && (elb1==elb3) && (elb2==elb3)){
                                batchUpsilon_n[0*batchups_compshift+thisnrshift] += 
                                    2 * wshape1  * nextG3ns_ggg[1*nbinsr+elb1];
                                batchUpsilon_n[1*batchups_compshift+thisnrshift] += 
                                    2 * wshape1c * nextG3ns_ggg[0*nbinsr+elb1];
                                batchUpsilon_n[2*batchups_compshift+thisnrshift] += 
                                    2 * wshape1  * nextG3ns_gggc[1*nbinsr+elb1];
                                batchUpsilon_n[3*batchups_compshift+thisnrshift] +=
                                    2 * wshape1  * nextG3ns_gggc[1*nbinsr+elb1];
                                batchUpsilon_n[4*batchups_compshift+thisnrshift] += 
                                    2 * wshape1  * nextG3ns_gggc[1*nbinsr+elb1];
                                batchUpsilon_n[5*batchups_compshift+thisnrshift] += 
                                    2 * wshape1c * nextG3ns_gggc[0*nbinsr+elb1];
                                batchUpsilon_n[6*batchups_compshift+thisnrshift] += 
                                    2 * wshape1c * nextG3ns_gggc[0*nbinsr+elb1];
                                batchUpsilon_n[7*batchups_compshift+thisnrshift] += 
                                    2 * wshape1c * nextG3ns_gggc[0*nbinsr+elb1];
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
                }
                time2 = omp_get_wtime();
                if ((elregion%nregions_skip_print == 0)&&(thisthread==0)&&(ind_inpix1==lower1)){
                    printf("Allocated Ups for first gal in region %d/%d for thetabatch %d/%d in %.4f seconds for %d theta-combis\n",
                           elregion,nregions,elthetbatch,nthetbatches,(time2-time1),batch_nthetas);}
            }
            if ((elregion%nregions_skip_print == 0)&&(thisthread==0)){
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
        double complex *nextM4correlators = calloc(8, sizeof(double complex));
        double complex *thisUpsilon_n = calloc(8*n2n3combis, sizeof(double complex));
        double complex *thisN_n = calloc(n2n3combis, sizeof(double complex));
        double complex *thisnpcf = calloc(8*batchgamma_thetshift, sizeof(double complex));
        double complex *thisnpcf_norm = calloc(batchgamma_thetshift, sizeof(double complex));
        for (int elb=0;elb<batch_nthetas;elb++){
            // 1)
            int nbshift, elb1, elb2, elb3;
            elb1 = elb1s_batch[elb];
            elb2 = elb2s_batch[elb];
            elb3 = elb3s_batch[elb];
            // 2)
            for(int eln12=0;eln12<n2n3combis;eln12++){
                nbshift = eln12*batchups_nshift+elb;
                for (int elcomp=0;elcomp<8;elcomp++){
                    thisUpsilon_n[elcomp*n2n3combis+eln12] = batchUpsilon_n[elcomp*batchups_compshift+nbshift];
                }
                thisN_n[eln12] = batchN_n[nbshift];
                // OPTIONAL: Allocate 4PCF in multipole basis
                if (alloc_4pcfmultipoles==1){
                    int thisnrshift = eln12*ups_nshift + elb1*nbinsr*nbinsr + elb2*nbinsr + elb3;
                    for (int elcomp=0;elcomp<8;elcomp++){
                        Upsilon_n[elcomp*ups_compshift+thisnrshift] = 
                                thisUpsilon_n[elcomp*n2n3combis+eln12];
                    }
                    N_n[thisnrshift] = thisN_n[eln12];
                }
            }
            // 3)
            multipoles2npcf_gggg_singletheta(thisUpsilon_n, thisN_n, nmax, nmax,
                                             bin_centers_batch[elb1], bin_centers_batch[elb2], bin_centers_batch[elb3],
                                             phibins, phibins, nbinsphi, nbinsphi,
                                             projection, thisnpcf, thisnpcf_norm);
            
            // OPTIONAL: Allocate 4pcf in real basis (Shape: (8,ntheta,ntheta,ntheta,nphi,nphi)
            if (alloc_4pcfreal==1){
                for (int elphi12=0;elphi12<batchgamma_thetshift;elphi12++){
                    int gamma_rshift = nbinsphi*nbinsphi;
                    int gamma_phircombi = gamma_rshift*(elb1*nbinsr*nbinsr+elb2*nbinsr+elb3)+elphi12;
                    int gamma_compshift = nbinsr*nbinsr*nbinsr*gamma_rshift;
                    for (int elcomp=0;elcomp<8;elcomp++){
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
                for (int elcomp=0;elcomp<8;elcomp++){
                    map4ind = elcomp*nmapradii+elmapr;
                    if (isnan(cabs(nextM4correlators[elcomp]))==false){
                        allM4correlators[map4threadshift+map4ind] += nextM4correlators[elcomp];
                    }
                    nextM4correlators[elcomp] = 0;
                }
            }
        
            // Reset 4pcf placeholders to zero
            for(int i=0;i<batchgamma_thetshift;i++){
                thisnpcf_norm[i] = 0;
                for (int elcomp=0;elcomp<8;elcomp++){
                    thisnpcf[elcomp*batchgamma_thetshift+i] = 0;
                }
            }
            for(int i=0;i<n2n3combis;i++){
                thisN_n[i] = 0;
                for (int elcomp=0;elcomp<8;elcomp++){
                    thisUpsilon_n[elcomp*n2n3combis+i] = 0;
                }
            }
        }
        
        for (int elmapr=0; elmapr<nmapradii; elmapr++){
            for (int elcomp=0;elcomp<8;elcomp++){
                int map4ind = elcomp*nmapradii+elmapr;
                int map4threadshift = thisthread*8*nmapradii;
                printf("\nthread %d, elr %d, elcomp %d, allM4cont=%.20f ",
                               thisthread, elmapr, elcomp, creal(allM4correlators[map4threadshift+map4ind]));
            }
        }
        if (thisthread>-1){printf("Done allocating 4pcfs for thetabatch %d/%d\n",elthetbatch,nthetbatches);}
            
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
    for (int elthread=0;elthread<nthreads;elthread++){
        int map4ind;
        int map4threadshift = elthread*8*nmapradii;
        for (int elcomp=0;elcomp<8;elcomp++){
            for (int elmapr=0; elmapr<nmapradii; elmapr++){
                map4ind = elcomp*nmapradii+elmapr;
                M4correlators[map4ind] += allM4correlators[map4threadshift+map4ind];
            }
        }
    }    
    free(allM4correlators);
}

// If thread==0 --> For final two threads allocate double/triple counting corrs
// thetacombis_batches: array of length nbinsr^3 with the indices of all possible (r1,r2,r3) combinations
//                      most likely it is simply range(nbinsr^3), but we leave some freedom here for 
//                      potential cost-based implementations
// nthetacombis_batches: array of length nthetbatches with the number of theta-combis in each batch
// cumthetacombis_batches : array of length (nthetbatches+1) with is cumsum of nthetacombis_batches
// nthetbatches: the number of theta batches
void alloc_notomoMap4_tree_gggg(
    double *isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int ngal, 
    int nmax, double rmin, double rmax, int nbinsr, int dccorr, 
    int *nindices, int len_nindices, double *phibins, double *dbinsphi, int nbinsphi,
    int nresos, double *reso_redges, int *ngal_resos, 
    double *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos, 
    double *e1_resos, double *e2_resos,
    int *index_matcher_hash, int *pixs_galind_bounds, int *pix_gals, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, int verbose, int projection, double *mapradii, int nmapradii, double complex *M4correlators, 
    int alloc_4pcfmultipoles, int alloc_4pcfreal,
    double *bin_centers, double complex *Upsilon_n, double complex *N_n, double complex *Gammas, double complex *Norms){
               
    double complex *allM4correlators = calloc(nthreads*8*1*nmapradii, sizeof(double complex));
    #pragma omp parallel for num_threads(nthreads)
    for (int elthetbatch=0;elthetbatch<nthetbatches;elthetbatch++){
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
        int *rshift_index_matcher_hash = calloc(nresos, sizeof(int));
        int *rshift_pixs_galind_bounds = calloc(nresos, sizeof(int));
        int *rshift_pix_gals = calloc(nresos, sizeof(int));
        for (int elreso=1;elreso<nresos;elreso++){
            rshift_index_matcher_hash[elreso] = rshift_index_matcher_hash[elreso-1] + npix_hash;
            rshift_pixs_galind_bounds[elreso] = rshift_pixs_galind_bounds[elreso-1] + ngal_resos[elreso-1]+1;
            rshift_pix_gals[elreso] = rshift_pix_gals[elreso-1] + ngal_resos[elreso-1];
        }

        double *totcounts = calloc(nbinsr, sizeof(double));
        double *totnorms = calloc(nbinsr, sizeof(double));
        double *bin_centers_batch = calloc(nbinsr, sizeof(double));
        double complex *batchUpsilon_n = calloc(ncomp*batchups_compshift, sizeof(double complex));
        double complex *batchN_n = calloc(batchups_compshift, sizeof(double complex));
        double complex *batchfourpcf = calloc(ncomp*batchups_compshift, sizeof(double complex));
        double complex *batchfourpcf_norm = calloc(batchups_compshift, sizeof(double complex));
        double *batch_thetas1 = calloc(batch_nthetas, sizeof(double));
        double *batch_thetas2 = calloc(batch_nthetas, sizeof(double));
        double *batch_thetas3 = calloc(batch_nthetas, sizeof(double));
        
        int nbinszr = nbinsz*nbinsr;
        double complex *nextGns =  calloc(nnvals_Gn*nbinszr, sizeof(double complex));
        double complex *nextG2ns_gg =  calloc(nnvals_G2n*nbinszr, sizeof(double complex));
        double complex *nextG2ns_ggc =  calloc(nnvals_G2n*nbinszr, sizeof(double complex));
        double complex *nextG3ns_ggg = calloc(2*nbinszr, sizeof(double complex));
        double complex *nextG3ns_gggc = calloc(2*nbinszr, sizeof(double complex));
        double complex *nextWns = calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW2ns = calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW3ns = calloc(nbinszr, sizeof(double complex));
        
        double drbin = (log(rmax)-log(rmin))/(nbinsr);
        int rbin_min_batch=nbinsr;int rbin_max_batch=0;
        int reso_min_batch=0; int reso_max_batch=0;
        int *elb1s_batch = calloc(batch_nthetas, sizeof(int));
        int *elb2s_batch = calloc(batch_nthetas, sizeof(int));
        int *elb3s_batch = calloc(batch_nthetas, sizeof(int));
        double *bin_edges = calloc(nbinsr+1, sizeof(double));
        #pragma omp critical
        {
            for (int elb=0;elb<batch_nthetas;elb++){
                int thisrcombi = thetacombis_batches[cumthetacombis_batches[elthetbatch]+elb];
                elb1s_batch[elb] = thisrcombi/(nbinsr*nbinsr);
                elb2s_batch[elb] = (thisrcombi-elb1s_batch[elb]*nbinsr*nbinsr)/nbinsr;
                elb3s_batch[elb] = thisrcombi-elb1s_batch[elb]*nbinsr*nbinsr-elb2s_batch[elb]*nbinsr;
                rbin_min_batch = mymin(rbin_min_batch, elb1s_batch[elb]); 
                rbin_max_batch = mymax(rbin_max_batch, elb3s_batch[elb]); 
            }
            bin_edges[0] = rmin;
            for (int elb=0;elb<nbinsr;elb++){
                bin_edges[elb+1] = bin_edges[elb]*exp(drbin);
            }
            for (int elreso=1;elreso<nresos;elreso++){
                if (reso_redges[elreso] <= bin_edges[rbin_min_batch  ]){reso_min_batch += 1;}
                if (reso_redges[elreso] <  bin_edges[rbin_max_batch+1]){reso_max_batch += 1;}
            }
            //printf("For batch %d with imin=%d imax=%d we have resomin=%d resomax=%d",
            //       elthetbatch, rbin_min_batch, rbin_max_batch, reso_min_batch, reso_max_batch);
        }
        
        // Allocate the 4pcf multipoles for this batch of radii 
        int offset_per_thread = nregions/nthreads;
        int offset = offset_per_thread*thisthread;
        for (int _elregion=0; _elregion<nregions; _elregion++){
            int elregion = (_elregion+offset)%nregions; // Try to evade collisions
            if ((verbose>0) && (elregion%nregions_skip_print == 0)&&(thisthread==0)){
                printf("Doing region %d/%d for thetabatch %d/%d\n",elregion,nregions,elthetbatch,nthetbatches);
            }
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
                int ind_red, lower, upper; 
                double  p21, p22, w2, w2_sq, e21, e22, rel1, rel2, dist2, dist, dphi;
                double complex wshape1, wshape1c, wshape2, wshape_sq, wshape_cube, wshapewshapec, wshapesqwshapec;
                double complex phirot, phirotc, twophirotc, fourphirotc, nphirot, nphirotc;
                
                // Check how many ns we need for Gn
                // Gns have shape (nnvals, nbinsz, nbinsr)
                // where the ns are ordered as 
                // [-nmax_1-nmax_2-3, ..., nmax_1+nmax_2+3]
                for (int i=0;i<nnvals_Gn*nbinszr;i++){nextGns[i]=0;}
                for (int i=0;i<nnvals_G2n*nbinszr;i++){nextG2ns_gg[i]=0;nextG2ns_ggc[i]=0;}
                for (int i=0;i<2*nbinszr;i++){nextG3ns_ggg[i]=0;nextG3ns_gggc[i]=0;}
                for (int i=0;i<nnvals_Wn*nbinszr;i++){nextWns[i]=0;nextW2ns[i]=0;}
                for (int i=0;i<nbinszr;i++){nextW3ns[i]=0;}
                for (int elreso=reso_min_batch;elreso<=reso_max_batch;elreso++){
                    int rbin, zrshift, nextnshift, ind_Gn, ind_G2n, ind_Wn;
                    double rmin_reso = reso_redges[elreso];
                    double rmin_reso2 = rmin_reso*rmin_reso;
                    double rmax_reso = reso_redges[elreso+1];
                    double rmax_reso2 = rmax_reso*rmax_reso;
                    int pix1_lower = mymax(0, (int) floor((p11 - (rmax_reso+pix1_d) - pix1_start)/pix1_d));
                    int pix2_lower = mymax(0, (int) floor((p12 - (rmax_reso+pix2_d) - pix2_start)/pix2_d));
                    int pix1_upper = mymin(pix1_n-1, (int) floor((p11 + (rmax_reso+pix1_d) - pix1_start)/pix1_d));
                    int pix2_upper = mymin(pix2_n-1, (int) floor((p12 + (rmax_reso+pix2_d) - pix2_start)/pix2_d));
                    for (int ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                        for (int ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                            ind_red = index_matcher_hash[rshift_index_matcher_hash[elreso] + ind_pix2*pix1_n + ind_pix1];
                            if (ind_red==-1){continue;}
                            lower = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+ind_red];
                            upper = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+ind_red+1];
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
                                if(dist2 < rmin_reso2 || dist2 >= rmax_reso2){continue;}
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
                                nphirot = 1+I*0;
                                nphirotc = 1+I*0;

                                // Triple-counting corr
                                nextW3ns[zrshift] += w2_sq*w2;
                                nextG3ns_ggg[zrshift] += wshape_cube*fourphirotc;
                                nextG3ns_ggg[nbinszr + zrshift] += wshape_cube*fourphirotc*fourphirotc;
                                nextG3ns_gggc[zrshift] += wshapesqwshapec;
                                nextG3ns_gggc[nbinszr + zrshift] += wshapesqwshapec*fourphirotc;                            

                                // Nominal G and double-counting corr
                                // n = 0
                                totcounts[zrshift] += w1*w2*dist; 
                                totnorms[zrshift] += w1*w2; 
                                nextGns[ind_Gn] += wshape2*nphirot;
                                nextG2ns_gg[ind_G2n] += wshape_sq*nphirot;
                                nextG2ns_ggc[ind_G2n] += wshapewshapec*nphirot;
                                nextWns[ind_Wn] += w2*nphirot;  
                                nextW2ns[ind_Wn] += w2_sq*nphirot;
                                // /*
                                // n \in [-2*nmax+1,2*nmax-1]                          
                                nphirot *= phirot;
                                nphirotc *= phirotc; 
                                // n in [1, ..., 2*nmax_alloc-1] x {+1,-1}
                                nextnshift = 0;
                                for (int nextn=1;nextn<2*nmax_alloc;nextn++){
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

                                // n = \pm 2*nmax_alloc
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

                                // n = \pm 2*nmax_alloc+1 
                                nextnshift += nbinszr;
                                nextGns[ind_Gn-nextnshift] += wshape2*nphirotc;
                                nextG2ns_gg[ind_G2n+nextnshift] += wshape_sq*nphirot;
                                nextG2ns_gg[ind_G2n-nextnshift] += wshape_sq*nphirotc;
                                nextG2ns_ggc[ind_G2n+nextnshift] += wshapewshapec*nphirot;
                                nextG2ns_ggc[ind_G2n-nextnshift] += wshapewshapec*nphirotc;
                                nphirotc *= phirotc;
                                // n =  -2*nmax_alloc-2
                                nextnshift += nbinszr;
                                nextGns[ind_Gn-nextnshift] += wshape2*nphirotc;
                                nextG2ns_gg[ind_G2n-nextnshift] += wshape_sq*nphirotc;
                                nextG2ns_ggc[ind_G2n-nextnshift] += wshapewshapec*nphirotc;
                                nphirotc *= phirotc;
                                // n =  -2*nmax_alloc-3
                                nextnshift += nbinszr;
                                nextGns[ind_Gn-nextnshift] += wshape2*nphirotc;
                                nextG2ns_gg[ind_G2n-nextnshift] += wshape_sq*nphirotc;
                                nextG2ns_ggc[ind_G2n-nextnshift] += wshapewshapec*nphirotc;
                                nphirotc *= phirotc;
                                // n =  -2*nmax_alloc-4
                                nextnshift += nbinszr;
                                nextG2ns_gg[ind_G2n-nextnshift] += wshape_sq*nphirotc;
                                nextG2ns_ggc[ind_G2n-nextnshift] += wshapewshapec*nphirotc;
                                nphirotc *= phirotc;
                                // n =  -2*nmax_alloc-5
                                nextnshift += nbinszr;
                                nextG2ns_gg[ind_G2n-nextnshift] += wshape_sq*nphirotc;
                                nextG2ns_ggc[ind_G2n-nextnshift] += wshapewshapec*nphirotc;
                            }
                        }
                    }
                }
                time2 = omp_get_wtime();
                if ((verbose>0) && (elregion%nregions_skip_print == 0)&&(thisthread==0)&&(ind_inpix1==lower1)){
                    printf("Computed Gn for first gal in region %d/%d for thetabatch %d/%d in %.4f seconds\n",
                           elregion,nregions,elthetbatch,nthetbatches,(time2-time1));}                
                
                // Allocate Upsilon
                // Upsilon have shape 
                // (ncomp,(2*nmax_alloc+1),(2*nmax_alloc+1),nthetas)
                time1 = omp_get_wtime();
                double complex gGG0, gGG1, gGG2, gGG3, gGG4, gGG5, gGG6, gGG7, wNN;
                int thisn2, thisn3, thisn, thisnshift, thisnrshift, elb1, elb2, elb3;
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
                    for (int elb=0;elb<batch_nthetas;elb++){
                        elb1 = elb1s_batch[elb];
                        elb2 = elb2s_batch[elb];
                        elb3 = elb3s_batch[elb];
                        thisnrshift = thisnshift + elb;
                        // Multiple counting corrections:
                        // sum_(i neq j neq k) = sum_(i,j,k) - ( sum_(i, j, i=k) + 2perm ) + 2 * sum_(i, i=j, i=k)
                        // Triple-counting corr
                        if ((elb1==elb2) && (elb1==elb3) && (elb2==elb3)){
                            batchUpsilon_n[0*batchups_compshift+thisnrshift] += 
                                2 * wshape1  * nextG3ns_ggg[1*nbinsr+elb1];
                            batchUpsilon_n[1*batchups_compshift+thisnrshift] += 
                                2 * wshape1c * nextG3ns_ggg[0*nbinsr+elb1];
                            batchUpsilon_n[2*batchups_compshift+thisnrshift] += 
                                2 * wshape1  * nextG3ns_gggc[1*nbinsr+elb1];
                            batchUpsilon_n[3*batchups_compshift+thisnrshift] +=
                                2 * wshape1  * nextG3ns_gggc[1*nbinsr+elb1];
                            batchUpsilon_n[4*batchups_compshift+thisnrshift] += 
                                2 * wshape1  * nextG3ns_gggc[1*nbinsr+elb1];
                            batchUpsilon_n[5*batchups_compshift+thisnrshift] += 
                                2 * wshape1c * nextG3ns_gggc[0*nbinsr+elb1];
                            batchUpsilon_n[6*batchups_compshift+thisnrshift] += 
                                2 * wshape1c * nextG3ns_gggc[0*nbinsr+elb1];
                            batchUpsilon_n[7*batchups_compshift+thisnrshift] += 
                                2 * wshape1c * nextG3ns_gggc[0*nbinsr+elb1];
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
                time2 = omp_get_wtime();
                if ((verbose>0) && (elregion%nregions_skip_print == 0)&&(thisthread==0)&&(ind_inpix1==lower1)){
                    printf("Allocated Ups for first gal in region %d/%d for thetabatch %d/%d in %.4f seconds for %d theta-combis\n",
                           elregion,nregions,elthetbatch,nthetbatches,(time2-time1),batch_nthetas);}
            }
            if ((verbose>0) && (elregion%nregions_skip_print == 0)&&(thisthread==0)){
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
        double complex *nextM4correlators = calloc(8, sizeof(double complex));
        double complex *thisUpsilon_n = calloc(8*n2n3combis, sizeof(double complex));
        double complex *thisN_n = calloc(n2n3combis, sizeof(double complex));
        double complex *thisUpsilon_n_rec = calloc(8*n2n3combis_rec, sizeof(double complex));
        double complex *thisN_n_rec = calloc(n2n3combis_rec, sizeof(double complex));
        double complex *thisnpcf = calloc(8*batchgamma_thetshift, sizeof(double complex));
        double complex *thisnpcf_norm = calloc(batchgamma_thetshift, sizeof(double complex));
        for (int elb=0;elb<batch_nthetas;elb++){
            if ((verbose>1) && (thisthread==0)){
                printf("Done %.4f per cent of multipole-to-Map4 conversion\r",100.* (float) elb/batch_nthetas);}
            // 1)
            int nbshift, elb1, elb2, elb3, elb1t, elb2t, elb3t;
            elb1 = elb1s_batch[elb];
            elb2 = elb2s_batch[elb];
            elb3 = elb3s_batch[elb];
            int bincombi_trafos[6][3] = {{elb1,elb2,elb3}, {elb2,elb3,elb1}, {elb3,elb1,elb2},
                                         {elb1,elb3,elb2}, {elb2,elb1,elb3}, {elb3,elb2,elb1}}; 
            // 2)
            if ((elb1==elb2)&&(elb1==elb3)){ntrafos=1;}
            else if ((elb1==elb2)&&(elb1!=elb3)){ntrafos=3;}
            else if ((elb1==elb3)&&(elb1!=elb2)){ntrafos=3;}
            else if ((elb2==elb3)&&(elb2!=elb1)){ntrafos=3;}
            else{ntrafos=6;}
            for (int eltrafo=0;eltrafo<ntrafos;eltrafo++){
                elb1t = bincombi_trafos[eltrafo][0];
                elb2t = bincombi_trafos[eltrafo][1];
                elb3t = bincombi_trafos[eltrafo][2];
                //printf("elb1=%d eln2=%d elb3=%d: eltrafo=%d/%d\n",elb1,elb2,elb3,eltrafo,ntrafos+1);
                // 2a)
                for(int eln12=0;eln12<n2n3combis;eln12++){
                    nbshift = eln12*batchups_nshift+elb;
                    for (int elcomp=0;elcomp<8;elcomp++){
                        thisUpsilon_n[elcomp*n2n3combis+eln12] = batchUpsilon_n[elcomp*batchups_compshift+nbshift];
                    }
                    thisN_n[eln12] = batchN_n[nbshift];
                }
                getMultipolesFromSymm(thisUpsilon_n, thisN_n, nmax, eltrafo, nindices, len_nindices,
                                       thisUpsilon_n_rec, thisN_n_rec);
                // OPTIONAL: Allocate 4PCF in multipole basis
                for(int eln12=0;eln12<n2n3combis_rec;eln12++){
                    if (alloc_4pcfmultipoles==1){
                        int thisnrshift = eln12*ups_nshift + elb1t*nbinsr*nbinsr + elb2t*nbinsr + elb3t;
                        for (int elcomp=0;elcomp<8;elcomp++){
                            Upsilon_n[elcomp*ups_rec_compshift+thisnrshift] = 
                                    thisUpsilon_n_rec[elcomp*n2n3combis_rec+eln12];
                        }
                        N_n[thisnrshift] = thisN_n_rec[eln12];
                    }
                }
                // 2b)
                multipoles2npcf_gggg_singletheta(thisUpsilon_n_rec, thisN_n_rec, nmax, nmax,
                                                 elb1t, elb2t, elb3t,
                                                 phibins, phibins, nbinsphi, nbinsphi,
                                                 projection, thisnpcf, thisnpcf_norm);

                // OPTIONAL: Allocate 4pcf in real basis (Shape: (8,ntheta,ntheta,ntheta,nphi,nphi)
                if (alloc_4pcfreal==1){
                    for (int elphi12=0;elphi12<batchgamma_thetshift;elphi12++){
                        int gamma_rshift = nbinsphi*nbinsphi;
                        int gamma_phircombi = gamma_rshift*(elb1t*nbinsr*nbinsr+elb2t*nbinsr+elb3t)+elphi12;
                        int gamma_compshift = nbinsr*nbinsr*nbinsr*gamma_rshift;
                        for (int elcomp=0;elcomp<8;elcomp++){
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
                    for (int elcomp=0;elcomp<8;elcomp++){
                        map4ind = elcomp*nmapradii+elmapr;
                        if (isnan(cabs(nextM4correlators[elcomp]))==false){
                            allM4correlators[map4threadshift+map4ind] += nextM4correlators[elcomp];
                        }
                        nextM4correlators[elcomp] = 0;
                    }
                }

                // Reset 4pcf placeholders to zero
                for(int i=0;i<batchgamma_thetshift;i++){
                    thisnpcf_norm[i] = 0;
                    for (int elcomp=0;elcomp<8;elcomp++){
                        thisnpcf[elcomp*batchgamma_thetshift+i] = 0;
                    }
                }
                for(int i=0;i<n2n3combis;i++){
                    thisN_n[i] = 0;
                    for (int elcomp=0;elcomp<8;elcomp++){
                        thisUpsilon_n[elcomp*n2n3combis+i] = 0;
                    }
                }
                for(int i=0;i<n2n3combis_rec;i++){
                    thisN_n_rec[i] = 0;
                    for (int elcomp=0;elcomp<8;elcomp++){
                        thisUpsilon_n_rec[elcomp*n2n3combis_rec+i] = 0;
                    }
                }
            }
        }
        
        if (verbose>1){
            for (int elmapr=0; elmapr<nmapradii; elmapr++){
                for (int elcomp=0;elcomp<8;elcomp++){
                    int map4ind = elcomp*nmapradii+elmapr;
                    int map4threadshift = thisthread*8*nmapradii;
                    printf("\nthread %d, elr %d, elcomp %d, allM4cont=%.20f ",
                                thisthread, elmapr, elcomp, creal(allM4correlators[map4threadshift+map4ind]));
                }
            }
        }
        if ((verbose>0) && (thisthread>-1)){printf("Done allocating 4pcfs for thetabatch %d/%d\n",elthetbatch,nthetbatches);}
            
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
    for (int elthread=0;elthread<nthreads;elthread++){
        int map4ind;
        int map4threadshift = elthread*8*nmapradii;
        for (int elcomp=0;elcomp<8;elcomp++){
            for (int elmapr=0; elmapr<nmapradii; elmapr++){
                map4ind = elcomp*nmapradii+elmapr;
                M4correlators[map4ind] += allM4correlators[map4threadshift+map4ind];
            }
        }
    }    
    free(allM4correlators);
}

void alloc_notomoGammans_discrete_gnnn(
    double *isinner_source, double *weight_source, double *pos1_source, double *pos2_source, double *e1_source, double *e2_source, int ngal_source, 
    double *weight_lens, double *pos1_lens, double *pos2_lens, int ngal_lens, 
    int *index_matcher_lens, int *pixs_galind_bounds_lens, int *pix_gals_lens, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int nmax, double rmin, double rmax, int nbinsr, int dccorr,
    int nthreads, double *bin_centers, double complex *Gtilde_n, double complex *N_n){

    int thistmpnshift, thisnshift, thisnrshift, thisthreadnrshift;
    int _nnvals_Upsn = 2*nmax+1;
    int _threadshift = _nnvals_Upsn*_nnvals_Upsn*nbinsr*nbinsr*nbinsr;

    double complex *allGtilden = calloc(nthreads*_threadshift, sizeof(double complex));
    double complex *allNormn = calloc(nthreads*_threadshift, sizeof(double complex));
    double *allcounts = calloc(nthreads*nbinsr, sizeof(double));
    double *allnorms = calloc(nthreads*nbinsr, sizeof(double));
    double *totcounts = calloc(nbinsr, sizeof(double));
    double *totnorms = calloc(nbinsr, sizeof(double));

    int ndone = 0;
    int ngal_per_update = mymax(1,ngal_source/1000);
    #pragma omp parallel for num_threads(nthreads)
    for (int ind_gal=0;ind_gal<ngal_source;ind_gal++){
        int thisthread = omp_get_thread_num();
        double drbin = (log(rmax)-log(rmin))/(nbinsr);
        //if (isource%nthreads!=thisthread){continue;}
        //printf("Doing thetabatch %d/%d on thread %d\n",elthetbatch,nthetbatches,thisthread);
        int nbinsz = 1;
        int ncomp = 1;
        int nbinszr = nbinsz*nbinsr;
        int nnvals_Wn = 4*nmax+3; // Need to cover [-n1-n2-1, n1+n2+1] (We use the Wn for both, the nominator and the denominator)
        int nnvals_W2n = 4*nmax+3;
        int nnvals_W3n = 2;
        int nnvals_Gtilden = 2*nmax+1;  // Need tocover [-2*nmax,+2*nmax]
        int nzero_Gtilden = nmax;
        int nzero_Wn = 2*nmax+1;
        int nzero_W2n = 2*nmax+1;
        
        int Gtilde_nshift = nbinsr*nbinsr*nbinsr;
        int n2n3combis = nnvals_Gtilden*nnvals_Gtilden;
        int Gtilde_threadshift = thisthread*Gtilde_nshift*n2n3combis;
        int npix_hash = pix1_n*pix2_n;


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
        int ind_red, lower, upper; 
        double  p21, p22, w2, w2_sq, rel1, rel2, dist, dphi;
        double complex phirot, phirotc, twophirot, nphirot, nphirotc;
    
        // Check how many ns we need for Gn
        // Gns have shape (nnvals, nbinsz, nbinsr)
        // where the ns are ordered as 
        // [-nmax_1-nmax_2-3, ..., nmax_1+nmax_2+3]
        double complex *nextWns =  calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW2ns = calloc(nnvals_W2n*nbinszr, sizeof(double complex));
        double complex *nextW3ns = calloc(2*nbinszr, sizeof(double complex)); // [W3n_gtilde, W3n_norm]
        //for (int i=0;i<nnvals_Wn*nbinszr;i++){nextWns[i]=0;}
        //for (int i=0;i<nnvals_W2n*nbinszr;i++){nextW2ns[i]=0;}
        //for (int i=0;i<nnvals_W3n*nbinszr;i++){nextW3ns[i]=0;}

        int rbin, zrshift, nextnshift, ind_Wn;
        int pix1_lower = mymax(0, (int) floor((p11 - (rmax+pix1_d) - pix1_start)/pix1_d));
        int pix2_lower = mymax(0, (int) floor((p12 - (rmax+pix2_d) - pix2_start)/pix2_d));
        int pix1_upper = mymin(pix1_n-1, (int) floor((p11 + (rmax+pix1_d) - pix1_start)/pix1_d));
        int pix2_upper = mymin(pix2_n-1, (int) floor((p12 + (rmax+pix2_d) - pix2_start)/pix2_d));
        for (int ind_pix1=pix1_lower; ind_pix1<=pix1_upper; ind_pix1++){
            for (int ind_pix2=pix2_lower; ind_pix2<=pix2_upper; ind_pix2++){
                ind_red = index_matcher_lens[ind_pix2*pix1_n + ind_pix1];
                if (ind_red==-1){continue;}
                lower = pixs_galind_bounds_lens[ind_red];
                upper = pixs_galind_bounds_lens[ind_red+1];
                for (int ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                    ind_gal2 = pix_gals_lens[ind_inpix];
                    //#pragma omp critical
                    {p21 = pos1_lens[ind_gal2];
                    p22 = pos2_lens[ind_gal2];
                    w2 = weight_lens[ind_gal2];}
                    
                    rel1 = p21 - p11;
                    rel2 = p22 - p12;
                    dist = sqrt(rel1*rel1 + rel2*rel2);
                    if(dist < rmin || dist >= rmax) continue;
                    rbin = (int) floor((log(dist)-log(rmin))/drbin);
                    w2_sq = w2*w2;
                    dphi = atan2(rel2,rel1);
                    phirot = cexp(I*dphi);
                    phirotc = conj(phirot);
                    twophirot = phirot*phirot;
                    zrshift = 0*nbinsr + rbin;
                    ind_Wn = nzero_Wn*nbinszr + zrshift;
                    nphirot = 1+I*0;
                    nphirotc = 1+I*0;

                    allcounts[thisthread*nbinszr + zrshift] += w1*w2*dist; 
                    allnorms[thisthread*nbinszr + zrshift]  += w1*w2; 

                    // Triple-counting corr
                    nextW3ns[zrshift] += w2_sq*w2;
                    nextW3ns[nbinszr+zrshift] += w2_sq*w2*conj(twophirot);                          

                    // Nominal G and double-counting corr
                    // n = 0
                    nextWns[ind_Wn] += w2*nphirot;  
                    nextW2ns[ind_Wn] += w2_sq*nphirot;
                    // /*
                    // n \in [-2*nmax+1,2*nmax-1]                          
                    nphirot *= phirot;
                    nphirotc *= phirotc; 
                    // n in [1, ..., 2*nmax-1] x {+1,-1}
                    for (int nextn=1;nextn<=2*nmax+1;nextn++){
                        nextnshift = nextn*nbinszr;
                        nextWns[ind_Wn+nextnshift] += w2*nphirot;
                        nextWns[ind_Wn-nextnshift] += w2*nphirotc;
                        nextW2ns[ind_Wn+nextnshift] += w2_sq*nphirot;
                        nextW2ns[ind_Wn-nextnshift] += w2_sq*nphirotc;
                        nphirot *= phirot;
                        nphirotc *= phirotc; 
                    }
                }
            }
        }

        // Allocate Upsilon
        // Upsilon have shape 
        // (ncomp,(2*nmax+1),(2*nmax+1),nthetas)
        int thisn2, thisn3, thisnshift, thisnrshift, elb1, elb2, elb3;
        int thisWshift_n2, thisWshift_n3, thisWshift_n3p1;
        int thisWshift_n2pn3, thisWshift_mn2mn3p1;
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
                thisWshift_n3p1 = (nzero_Wn+n3+1)*nbinsr;
                thisWshift_n2pn3 = (nzero_Wn+n2+n3)*nbinsr;
                thisnshift = Gtilde_threadshift + ((n2+nzero_Gtilden)*nnvals_Gtilden + (n3+nzero_Gtilden)) * Gtilde_nshift;
                for (int elb1=0;elb1<nbinsr;elb1++){
                    // elb1 = elb2 = elb3
                    thisnrshift = thisnshift + elb1*nbinsr*nbinsr + elb1*nbinsr + elb1;
                    allGtilden[thisnrshift] += 2 * wshape1*nextW3ns[1*nbinsr+elb1];
                    allNormn[thisnrshift] += 2 * w1*nextW3ns[0*nbinsr+elb1];
                    for (int elb2=0;elb2<nbinsr;elb2++){
                        // elb1 = elb2
                        thisnrshift = thisnshift + elb1*nbinsr*nbinsr + elb1*nbinsr + elb2;
                        allGtilden[thisnrshift] -= wshape1 * 
                            nextW2ns[(nzero_Wn+n3-1)*nbinsr+elb1]*nextWns[thisWshift_mn3m1+elb2];
                        allNormn[thisnrshift] -= w1 * 
                            nextW2ns[(nzero_Wn+n3)*nbinsr+elb1]*conj(nextWns[thisWshift_n3+elb2]);
                        // elb1 = elb3
                        thisnrshift = thisnshift + elb1*nbinsr*nbinsr + elb2*nbinsr + elb1;
                        allGtilden[thisnrshift] -= wshape1 * 
                            nextW2ns[(nzero_Wn+n2-2)*nbinsr+elb1]*nextWns[thisWshift_mn2+elb2];
                        allNormn[thisnrshift] -= w1 * 
                            nextW2ns[(nzero_Wn+n2)*nbinsr+elb1]*nextWns[thisWshift_mn2+elb2];
                        //elb2 = elb3
                        thisnrshift = thisnshift + elb2*nbinsr*nbinsr + elb1*nbinsr + elb1;
                        allGtilden[thisnrshift] -= wshape1 * 
                            nextW2ns[thisWshift_mn2mn3m1+elb1]*nextWns[thisWshift_n2n3m1+elb2];
                        allNormn[thisnrshift] -= w1 * 
                            nextW2ns[(nzero_Wn-n2-n3)*nbinsr+elb1] * nextWns[thisWshift_n2pn3+elb2];
                        wWW = wshape1 * nextWns[thisWshift_n2n3m1+elb1] *  nextWns[thisWshift_mn2+elb2];
                        wNN =  w1 * nextWns[thisWshift_n2pn3+elb1] * conj(nextWns[thisWshift_n2+elb2]);
                        for (int elb3=0;elb3<nbinsr;elb3++){
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


        #pragma omp critical
        {
            ndone += 1;
            if (ndone%ngal_per_update == 0){printf("\nDone %.2f percent",100 * (double) (ndone)/(double)(ngal_source));}
        }
    }
    printf("\nDone parallel allocation of Gtilden \n");
    
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
    printf("\nDone parallel accumulation of Gtilden \n");
    
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
            printf("%.3f ",bin_centers[elbinr]);
        }
        printf("-1 ");
    }
    printf("\nDone accumulation of bin centers \n");
    free(allcounts);
    free(allnorms);
    free(totcounts);
    free(totnorms);
    free(allGtilden);
    free(allNormn);
    printf("\nDone freeing stuff. \n");
}


// Non-tomo 4pcf using tree-based estimator
void alloc_notomoGammans_tree_gnnn(
    int nresos, double *reso_redges,
    double *isinner_source, double *weight_source, double *pos1_source, double *pos2_source, double *e1_source, double *e2_source, int ngal_source, 
    double *weight_lens_resos, double *pos1_lens_resos, double *pos2_lens_resos, int *ngal_lens_resos, 
    int *index_matcher_source, int *pixs_galind_bounds_source, int *pix_gals_source, 
    int *index_matcher_lens, int *pixs_galind_bounds_lens, int *pix_gals_lens, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int nmax, double rmin, double rmax, int nbinsr, int dccorr, int nthetacombis, 
    int *nindices, int len_nindices, 
    int nthreads, int verbose, double *bin_centers, double complex *Gtilde_n, double complex *N_n){
    
    int n_cfs = 1;
    // Temporary arrays that are allocated in parallel and later reduced
    double *tmpwcounts = calloc(nthreads*nbinsr, sizeof(double));
    double *tmpwnorms = calloc(nthreads*nbinsr, sizeof(double));
    double complex *tmpUpsilon_n = calloc(nthreads*len_nindices*nthetacombis, sizeof(double complex));
    double complex *tmpN_n = calloc(nthreads*len_nindices*nthetacombis, sizeof(double complex));
    double *totcounts = calloc(nbinsr, sizeof(double));
    double *totnorms = calloc(nbinsr, sizeof(double));

    // Helper array that checks how many regions have been already computed
    int *regionsdone = calloc(nregions, sizeof(int));
    int nregionsdone = 0;
    
    #pragma omp parallel for num_threads(nthreads)
    for(int elthread=0;elthread<nthreads;elthread++){

        int nregions_per_thread = nregions/nthreads;
        int nbinsz = 1;
        int ncomp = 1;
        int nmax_alloc = 2*nmax+1;
        int nnvals_Wn = 4*nmax_alloc+3; // Need to cover [-n1-n2-1, n1+n2+1] (We use the Wn for both, the nominator and the denominator)
        int nnvals_W2n = 4*nmax_alloc+3;
        int nnvals_W3n = 2;
        int nnvals_Gtilden = 2*nmax_alloc+1;  // Need tocover [-2*nmax_alloc,+2*nmax_alloc]
        int nnvals_Gtilden_rec = 2*nmax+1; // Need tocover [-nmax,+nmax]
        int nzero_Gtilden = nmax_alloc;
        int nzero_Wn = 2*nmax_alloc+1;
        int nzero_W2n = 2*nmax_alloc+1;

        int ups_compshift = len_nindices*nthetacombis;

        int nbinszr = nbinsz*nbinsr;
        double complex *nextWns =  calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW2ns = calloc(nnvals_W2n*nbinszr, sizeof(double complex));
        double complex *nextW3ns = calloc(2*nbinszr, sizeof(double complex)); // [W3n_gtilde, W3n_norm]

        int npix_hash = pix1_n*pix2_n;
        int *rshift_index_matcher = calloc(nresos, sizeof(int));
        int *rshift_pixs_galind_bounds = calloc(nresos, sizeof(int));
        int *rshift_pix_gals = calloc(nresos, sizeof(int));
        for (int elreso=1;elreso<nresos;elreso++){
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
                int ind_red, lower, upper; 
                double  p21, p22, w2, w2_sq, rel1, rel2, dist, dphi;
                double complex phirot, phirotc, twophirot, nphirot, nphirotc;
            
                // Check how many ns we need for Gn
                // Gns have shape (nnvals, nbinsz, nbinsr)
                // where the ns are ordered as 
                // [-nmax_1-nmax_2-3, ..., nmax_1+nmax_2+3]
                for (int i=0;i<nnvals_Wn*nbinszr;i++){nextWns[i]=0;}
                for (int i=0;i<nnvals_W2n*nbinszr;i++){nextW2ns[i]=0;}
                for (int i=0;i<nnvals_W3n*nbinszr;i++){nextW3ns[i]=0;}
                for (int elreso=0;elreso<=nresos;elreso++){
                    int rbin, zrshift, nextnshift, ind_Wn;
                    double rmin_reso = reso_redges[elreso];
                    double rmax_reso = reso_redges[elreso+1];
                    int pix1_lower = mymax(0, (int) floor((p11 - (rmax_reso+pix1_d) - pix1_start)/pix1_d));
                    int pix2_lower = mymax(0, (int) floor((p12 - (rmax_reso+pix2_d) - pix2_start)/pix2_d));
                    int pix1_upper = mymin(pix1_n-1, (int) floor((p11 + (rmax_reso+pix1_d) - pix1_start)/pix1_d));
                    int pix2_upper = mymin(pix2_n-1, (int) floor((p12 + (rmax_reso+pix2_d) - pix2_start)/pix2_d));
                    for (int ind_pix1=pix1_lower; ind_pix1<=pix1_upper; ind_pix1++){
                        for (int ind_pix2=pix2_lower; ind_pix2<=pix2_upper; ind_pix2++){
                            ind_red = index_matcher_lens[rshift_index_matcher[elreso] + ind_pix2*pix1_n + ind_pix1];
                            if (ind_red==-1){continue;}
                            lower = pixs_galind_bounds_lens[rshift_pixs_galind_bounds[elreso]+ind_red];
                            upper = pixs_galind_bounds_lens[rshift_pixs_galind_bounds[elreso]+ind_red+1];
                            for (int ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                                ind_gal2 = rshift_pix_gals[elreso] + pix_gals_lens[rshift_pix_gals[elreso]+ind_inpix];
                                //#pragma omp critical
                                {p21 = pos1_lens_resos[ind_gal2];
                                p22 = pos2_lens_resos[ind_gal2];
                                w2 = weight_lens_resos[ind_gal2];}
                                
                                rel1 = p21 - p11;
                                rel2 = p22 - p12;
                                dist = sqrt(rel1*rel1 + rel2*rel2);
                                if(dist < rmin_reso || dist >= rmax_reso) continue;
                                rbin = (int) floor((log(dist)-log(rmin))/drbin);
                                w2_sq = w2*w2;
                                dphi = atan2(rel2,rel1);
                                phirot = cexp(I*dphi);
                                phirotc = conj(phirot);
                                twophirot = phirot*phirot;
                                zrshift = 0*nbinsr + rbin;
                                ind_Wn = nzero_Wn*nbinszr + zrshift;
                                nphirot = 1+I*0;
                                nphirotc = 1+I*0;

                                // Triple-counting corr
                                nextW3ns[zrshift] += w2_sq*w2;
                                nextW3ns[nbinszr+zrshift] += w2_sq*w2*conj(twophirot);                          

                                // Nominal G and double-counting corr
                                // n = 0
                                tmpwcounts[elthread*nbinszr+zrshift] += w1*w2*dist; 
                                tmpwnorms[elthread*nbinszr+zrshift] += w1*w2; 
                                nextWns[ind_Wn] += w2*nphirot;  
                                nextW2ns[ind_Wn] += w2_sq*nphirot;
                                // /*
                                // n \in [-2*nmax+1,2*nmax-1]                          
                                nphirot *= phirot;
                                nphirotc *= phirotc; 
                                // n in [1, ..., 2*nmax_alloc-1] x {+1,-1}
                                nextnshift = 0;
                                for (int nextn=1;nextn<=2*nmax_alloc+1;nextn++){
                                    nextnshift = nextn*nbinszr;
                                    nextWns[ind_Wn+nextnshift] += w2*nphirot;
                                    nextWns[ind_Wn-nextnshift] += w2*nphirotc;
                                    nextW2ns[ind_Wn+nextnshift] += w2_sq*nphirot;
                                    nextW2ns[ind_Wn-nextnshift] += w2_sq*nphirotc;
                                    nphirot *= phirot;
                                    nphirotc *= phirotc; 
                                }
                            }
                        }
                    }
                }
                
                // Allocate Upsilon
                // Upsilon have shape 
                // (ncomp,(2*nmax+1),(2*nmax+1),nthetas)
                int n2, n3, thisnshift, thisnrshift, elbcombi, elb1, elb2, elb3;
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
                    for (int elb1=0;elb1<nbinsr;elb1++){
                        // Precomputations for inner loop
                        Gcorr123 = + 2 * wshape1*nextW3ns[1*nbinsr+elb1];
                        Ncorr123 = + 2 * w1*nextW3ns[0*nbinsr+elb1];
                        for (int elb2=elb1;elb2<nbinsr;elb2++){
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

    // Accumulate Upsilon_n and N_n
    // 1) Build arrays that hold bin combis for b1<=b2<=b3
    // 2) Get bin edges and bin centers of the combinations
    // 3) Find all (theta1,theta2,theta3) combis that can be reconstructed via the symmetries
    // 4) Get the Gamma_mu(theta1,theta2,theta3,phi12,phi13)
    // 1)
    int elbcombi = 0;
    int *elb1_inds = calloc(nthetacombis, sizeof(int));
    int *elb2_inds = calloc(nthetacombis, sizeof(int));
    int *elb3_inds = calloc(nthetacombis, sizeof(int));
    for (int elb1=0;elb1<nbinsr;elb1++){
        for (int elb2=elb1;elb2<nbinsr;elb2++){
            for (int elb3=elb2;elb3<nbinsr;elb3++){
                elb1_inds[elbcombi] = elb1;
                elb2_inds[elbcombi] = elb2;
                elb3_inds[elbcombi] = elb3;
                elbcombi += 1;
            }
        }
    }

    #pragma omp parallel for num_threads(nthreads)
    for (int elb=0;elb<nthetacombis;elb++){

        int ntrafos, tnrshift, nbshift, nbshift_tmp, elb1, elb2, elb3, elb1t, elb2t, elb3t;
        int thisn2, thisn3, thisn;
        int nmax_alloc = 2*nmax+1;
        int nnvals_Upsn_rec = 2*nmax+1; 
        int nnvals_Upsn = 2*nmax_alloc+1; 
        int nzero_Ups = nmax_alloc;
        int ups_nshift = nbinsr*nbinsr*nbinsr;
        int n2n3combis = nnvals_Upsn*nnvals_Upsn;
        int n2n3combis_rec = nnvals_Upsn_rec*nnvals_Upsn_rec;
        int ups_rec_compshift = n2n3combis_rec*ups_nshift;

        double complex *thisUpsilon_n = calloc(n_cfs*n2n3combis, sizeof(double complex));
        double complex *thisN_n = calloc(n2n3combis, sizeof(double complex));
        double complex *thisUpsilon_n_rec = calloc(n_cfs*n2n3combis_rec, sizeof(double complex));
        double complex *thisN_n_rec = calloc(n2n3combis_rec, sizeof(double complex));

        // 2)
        elb1 = elb1_inds[elb];
        elb2 = elb2_inds[elb];
        elb3 = elb3_inds[elb];
        int bincombi_trafos[6][3] = {{elb1,elb2,elb3}, {elb2,elb3,elb1}, {elb3,elb1,elb2},
                                     {elb1,elb3,elb2}, {elb2,elb1,elb3}, {elb3,elb2,elb1}}; 
        
        // 3)
        if ((elb1==elb2)&&(elb1==elb3)){ntrafos=1;}
        else if ((elb1==elb2)&&(elb1!=elb3)){ntrafos=3;}
        else if ((elb1==elb3)&&(elb1!=elb2)){ntrafos=3;}
        else if ((elb2==elb3)&&(elb2!=elb1)){ntrafos=3;}
        else{ntrafos=6;}
        for (int eltrafo=0;eltrafo<ntrafos;eltrafo++){
            elb1t = bincombi_trafos[eltrafo][0];
            elb2t = bincombi_trafos[eltrafo][1];
            elb3t = bincombi_trafos[eltrafo][2];
            for (int nindex=0;nindex<len_nindices;nindex++){
                thisn2 = nindices[nindex]/nnvals_Upsn - nzero_Ups;
                thisn3 = nindices[nindex]%nnvals_Upsn - nzero_Ups;
                nbshift_tmp = nindex*nthetacombis+elb;
                nbshift = ((thisn2+nzero_Ups)*nnvals_Upsn + (thisn3+nzero_Ups));
                for (int elthread=0;elthread<nthreads;elthread++){
                    tnrshift = elthread*len_nindices*nthetacombis + nindex*nthetacombis + elb;
                    thisUpsilon_n[0*n2n3combis+nbshift] += tmpUpsilon_n[tnrshift];
                    thisN_n[nbshift] += tmpN_n[tnrshift];
                }
            }

            getMultipolesFromSymm_GNNN(
                thisUpsilon_n, thisN_n, nmax, eltrafo, nindices, len_nindices,
                thisUpsilon_n_rec, thisN_n_rec);

            // 4)
            for(int eln12=0;eln12<n2n3combis_rec;eln12++){
                int thisnrshift = eln12*ups_nshift + elb1t*nbinsr*nbinsr + elb2t*nbinsr + elb3t;
                for (int elcomp=0;elcomp<n_cfs;elcomp++){
                    Gtilde_n[elcomp*ups_rec_compshift+thisnrshift] =  thisUpsilon_n_rec[elcomp*n2n3combis_rec+eln12];
                }
                N_n[thisnrshift] = thisN_n_rec[eln12];
            }  

            // Reset 4pcf placeholders to zero
            for(int i=0;i<n2n3combis;i++){
                thisN_n[i] = 0;
                for (int elcomp=0;elcomp<n_cfs;elcomp++){
                    thisUpsilon_n[elcomp*n2n3combis+i] = 0;
                }
            }
            for(int i=0;i<n2n3combis_rec;i++){
                thisN_n_rec[i] = 0;
                for (int elcomp=0;elcomp<n_cfs;elcomp++){
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
    

void alloc_notomoMapNap3_tree_gnnn(
    int nresos, double *reso_redges,
    double *isinner_source, double *weight_source, double *pos1_source, double *pos2_source, double *e1_source, double *e2_source, int ngal_source, 
    double *weight_lens_resos, double *pos1_lens_resos, double *pos2_lens_resos, int *ngal_lens_resos, 
    int *index_matcher_source, int *pixs_galind_bounds_source, int *pix_gals_source, 
    int *index_matcher_lens, int *pixs_galind_bounds_lens, int *pix_gals_lens, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int nmax, double rmin, double rmax, int nbinsr, int dccorr,
    int *nindices, int len_nindices, double *phibins, double *dbinsphi, int nbinsphi,
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, double *apradii, int napradii, double complex *NM3correlator, 
    int alloc_4pcfmultipoles, int alloc_4pcfreal,
    double *bin_centers, double complex *Gtilde_n, double complex *N_n, double complex *Gtilde, double complex *Norms){
               
    double complex *allNM3correlator = calloc(nthreads*1*1*napradii, sizeof(double complex));
    #pragma omp parallel for num_threads(nthreads)
    for (int elthetbatch=0;elthetbatch<nthetbatches;elthetbatch++){
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
        int ncomp = 1;
        int nnvals_Wn = 4*nmax_alloc+3; // Need to cover [-n1-n2-1, n1+n2+1] (We use the Wn for both, the nominator and the denominator)
        int nnvals_W2n = 4*nmax_alloc+3;
        int nnvals_W3n = 2;
        int nnvals_Gtilden = 2*nmax_alloc+1;  // Need tocover [-2*nmax_alloc,+2*nmax_alloc]
        int nnvals_Gtilden_rec = 2*nmax+1; // Need tocover [-nmax,+nmax]
        int nzero_Gtilden = nmax_alloc;
        int nzero_Wn = 2*nmax_alloc+1;
        int nzero_W2n = 2*nmax_alloc+1;
        
        int Gtilde_nshift = nbinsr*nbinsr*nbinsr;
        int n2n3combis = nnvals_Gtilden*nnvals_Gtilden;
        int n2n3combis_rec = nnvals_Gtilden_rec*nnvals_Gtilden_rec;
        
        int batch_nthetas = nthetacombis_batches[elthetbatch];
        int batchGtilde_nshift = batch_nthetas;
        int batchGtilde_compshift = n2n3combis*batchGtilde_nshift;
        int batchGtilde_thetshift = nbinsphi*nbinsphi;
        
        int npix_hash = pix1_n*pix2_n;
        int *rshift_index_matcher = calloc(nresos, sizeof(int));
        int *rshift_pixs_galind_bounds = calloc(nresos, sizeof(int));
        int *rshift_pix_gals = calloc(nresos, sizeof(int));
        for (int elreso=1;elreso<nresos;elreso++){
            rshift_index_matcher[elreso] = rshift_index_matcher[elreso-1] + npix_hash;
            rshift_pixs_galind_bounds[elreso] = rshift_pixs_galind_bounds[elreso-1] + ngal_lens_resos[elreso-1]+1;
            rshift_pix_gals[elreso] = rshift_pix_gals[elreso-1] + ngal_lens_resos[elreso-1];
        }

        double *totcounts = calloc(nbinsr, sizeof(double));
        double *totnorms = calloc(nbinsr, sizeof(double));
        double *bin_centers_batch = calloc(nbinsr, sizeof(double));
        double complex *batchGtilde_n = calloc(batchGtilde_compshift, sizeof(double complex));
        double complex *batchN_n = calloc(batchGtilde_compshift, sizeof(double complex));
        double *batch_thetas1 = calloc(batch_nthetas, sizeof(double));
        double *batch_thetas2 = calloc(batch_nthetas, sizeof(double));
        double *batch_thetas3 = calloc(batch_nthetas, sizeof(double));
        
        int nbinszr = nbinsz*nbinsr;
        double complex *nextWns =  calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW2ns = calloc(nnvals_W2n*nbinszr, sizeof(double complex));
        double complex *nextW3ns = calloc(2*nbinszr, sizeof(double complex)); // [W3n_gtilde, W3n_norm]
        
        double drbin = (log(rmax)-log(rmin))/(nbinsr);
        int rbin_min_batch=nbinsr;int rbin_max_batch=0;
        int reso_min_batch=0; int reso_max_batch=0;
        int *elb1s_batch = calloc(batch_nthetas, sizeof(int));
        int *elb2s_batch = calloc(batch_nthetas, sizeof(int));
        int *elb3s_batch = calloc(batch_nthetas, sizeof(int));
        double *bin_edges = calloc(nbinsr+1, sizeof(double));
        #pragma omp critical
        {
            for (int elb=0;elb<batch_nthetas;elb++){
                int thisrcombi = thetacombis_batches[cumthetacombis_batches[elthetbatch]+elb];
                elb1s_batch[elb] = thisrcombi/(nbinsr*nbinsr);
                elb2s_batch[elb] = (thisrcombi-elb1s_batch[elb]*nbinsr*nbinsr)/nbinsr;
                elb3s_batch[elb] = thisrcombi-elb1s_batch[elb]*nbinsr*nbinsr-elb2s_batch[elb]*nbinsr;
                rbin_min_batch = mymin(rbin_min_batch, elb1s_batch[elb]); 
                rbin_max_batch = mymax(rbin_max_batch, elb3s_batch[elb]); 
            }
            bin_edges[0] = rmin;
            for (int elb=0;elb<nbinsr;elb++){
                bin_edges[elb+1] = bin_edges[elb]*exp(drbin);
            }
            for (int elreso=1;elreso<nresos;elreso++){
                if (reso_redges[elreso] <= bin_edges[rbin_min_batch  ]){reso_min_batch += 1;}
                if (reso_redges[elreso] <  bin_edges[rbin_max_batch+1]){reso_max_batch += 1;}
            }
            //printf("For batch %d with imin=%d imax=%d we have resomin=%d resomax=%d",
            //       elthetbatch, rbin_min_batch, rbin_max_batch, reso_min_batch, reso_max_batch);
        }
        
        // Allocate the 4pcf multipoles for this batch of radii 
        int offset_per_thread = nregions/nthreads;
        int offset = offset_per_thread*thisthread;
        for (int _elregion=0; _elregion<nregions; _elregion++){
            int elregion = (_elregion+offset)%nregions; // Try to evade collisions
            if ((elregion%nregions_skip_print == 0)&&(thisthread==0)){
                printf("Doing region %d/%d for thetabatch %d/%d\n",elregion,nregions,elthetbatch,nthetbatches);
            }
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
                int ind_red, lower, upper; 
                double  p21, p22, w2, w2_sq, rel1, rel2, dist, dphi;
                double complex phirot, phirotc, twophirot, nphirot, nphirotc;
            
                // Check how many ns we need for Gn
                // Gns have shape (nnvals, nbinsz, nbinsr)
                // where the ns are ordered as 
                // [-nmax_1-nmax_2-3, ..., nmax_1+nmax_2+3]
                for (int i=0;i<nnvals_Wn*nbinszr;i++){nextWns[i]=0;}
                for (int i=0;i<nnvals_W2n*nbinszr;i++){nextW2ns[i]=0;}
                for (int i=0;i<nnvals_W3n*nbinszr;i++){nextW3ns[i]=0;}
                for (int elreso=reso_min_batch;elreso<=reso_max_batch;elreso++){
                    int rbin, zrshift, nextnshift, ind_Wn;
                    double rmin_reso = reso_redges[elreso];
                    double rmax_reso = reso_redges[elreso+1];
                    int pix1_lower = mymax(0, (int) floor((p11 - (rmax_reso+pix1_d) - pix1_start)/pix1_d));
                    int pix2_lower = mymax(0, (int) floor((p12 - (rmax_reso+pix2_d) - pix2_start)/pix2_d));
                    int pix1_upper = mymin(pix1_n-1, (int) floor((p11 + (rmax_reso+pix1_d) - pix1_start)/pix1_d));
                    int pix2_upper = mymin(pix2_n-1, (int) floor((p12 + (rmax_reso+pix2_d) - pix2_start)/pix2_d));
                    for (int ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                        for (int ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                            ind_red = index_matcher_lens[rshift_index_matcher[elreso] + ind_pix2*pix1_n + ind_pix1];
                            if (ind_red==-1){continue;}
                            lower = pixs_galind_bounds_lens[rshift_pixs_galind_bounds[elreso]+ind_red];
                            upper = pixs_galind_bounds_lens[rshift_pixs_galind_bounds[elreso]+ind_red+1];
                            for (int ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                                ind_gal2 = rshift_pix_gals[elreso] + pix_gals_lens[rshift_pix_gals[elreso]+ind_inpix];
                                //#pragma omp critical
                                {p21 = pos1_lens_resos[ind_gal2];
                                p22 = pos2_lens_resos[ind_gal2];
                                w2 = weight_lens_resos[ind_gal2];}
                                
                                rel1 = p21 - p11;
                                rel2 = p22 - p12;
                                dist = sqrt(rel1*rel1 + rel2*rel2);
                                if(dist < rmin_reso || dist >= rmax_reso) continue;
                                rbin = (int) floor((log(dist)-log(rmin))/drbin);
                                w2_sq = w2*w2;
                                dphi = atan2(rel2,rel1);
                                phirot = cexp(I*dphi);
                                phirotc = conj(phirot);
                                twophirot = phirot*phirot;
                                zrshift = 0*nbinsr + rbin;
                                ind_Wn = nzero_Wn*nbinszr + zrshift;
                                nphirot = 1+I*0;
                                nphirotc = 1+I*0;

                                // Triple-counting corr
                                nextW3ns[zrshift] += w2_sq*w2;
                                nextW3ns[nbinszr+zrshift] += w2_sq*w2*twophirot;                          

                                // Nominal G and double-counting corr
                                // n = 0
                                totcounts[zrshift] += w1*w2*dist; 
                                totnorms[zrshift] += w1*w2; 
                                nextWns[ind_Wn] += w2*nphirot;  
                                nextW2ns[ind_Wn] += w2_sq*nphirot;
                                // /*
                                // n \in [-2*nmax+1,2*nmax-1]                          
                                nphirot *= phirot;
                                nphirotc *= phirotc; 
                                // n in [1, ..., 2*nmax_alloc-1] x {+1,-1}
                                nextnshift = 0;
                                for (int nextn=1;nextn<=2*nmax_alloc+1;nextn++){
                                    nextnshift = nextn*nbinszr;
                                    nextWns[ind_Wn+nextnshift] += w2*nphirot;
                                    nextWns[ind_Wn-nextnshift] += w2*nphirotc;
                                    nextW2ns[ind_Wn+nextnshift] += w2_sq*nphirot;
                                    nextW2ns[ind_Wn-nextnshift] += w2_sq*nphirotc;
                                    nphirot *= phirot;
                                    nphirotc *= phirotc; 
                                }
                            }
                        }
                    }
                }
                time2 = omp_get_wtime();
                if ((elregion%nregions_skip_print == 0)&&(thisthread==0)&&(ind_inpix1==lower1)){
                    printf("Computed Wn for first gal in region %d/%d for thetabatch %d/%d in %.4f seconds\n",
                           elregion,nregions,elthetbatch,nthetbatches,(time2-time1));}                
                
                // Allocate Upsilon
                // Upsilon have shape 
                // (ncomp,(2*nmax_alloc+1),(2*nmax_alloc+1),nthetas)
                time1 = omp_get_wtime();
                int thisn2, thisn3, thisnshift, thisnrshift, elb1, elb2, elb3;
                int thisWshift_n2, thisWshift_n3, thisWshift_n3p1;
                int thisWshift_n2pn3, thisWshift_mn2mn3p1;
                double complex wshape1 = w1 * (e11+I*e12);  
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
                    thisWshift_mn2mn3p1 = (nzero_Wn-thisn2-thisn3+1)*nbinsr;
                    thisWshift_n2 = (nzero_Wn+thisn2)*nbinsr;
                    thisWshift_n3 = (nzero_Wn+thisn3)*nbinsr;
                    thisWshift_n3p1 = (nzero_Wn+thisn3+1)*nbinsr;
                    thisWshift_n2pn3 = (nzero_Wn+thisn2+thisn3)*nbinsr;
                    thisnshift = ((thisn2+nzero_Gtilden)*nnvals_Gtilden + (thisn3+nzero_Gtilden)) * batchGtilde_nshift;
                    for (int elb=0;elb<batch_nthetas;elb++){
                        elb1 = elb1s_batch[elb];
                        elb2 = elb2s_batch[elb];
                        elb3 = elb3s_batch[elb];
                        thisnrshift = thisnshift + elb;
                        // Multiple counting corrections:
                        // sum_(i neq j neq k) = sum_(i,j,k) - ( sum_(i, j, i=k) + 2perm ) + 2 * sum_(i, i=j, i=k)
                        // Triple-counting corr
                        if ((elb1==elb2) && (elb1==elb3) && (elb2==elb3)){
                            batchGtilde_n[thisnrshift] += wshape1*nextW3ns[1*nbinsr+elb1];
                            batchN_n[thisnrshift] += 2 * w1*nextW3ns[0*nbinsr+elb1];
                        }
                        // Double-counting corr for theta1==theta2
                        if ((elb1==elb2)){
                            batchGtilde_n[thisnrshift] -= wshape1 * 
                                nextW2ns[(nzero_Wn+thisn3-1)*nbinsr+elb1]*nextWns[thisWshift_n3p1+elb3];
                            batchN_n[thisnrshift] -= w1 * 
                                nextW2ns[(nzero_Wn+thisn3)*nbinsr+elb1]*conj(nextWns[thisWshift_n3+elb3]);
                        }
                        // Double-counting corr for theta1==theta3  
                        if ((elb1==elb3)){
                            batchGtilde_n[thisnrshift] -= wshape1 * 
                                nextW2ns[(nzero_Wn+thisn2-2)*nbinsr+elb1]*nextWns[thisWshift_n2+elb2];
                            batchN_n[thisnrshift] -= w1 * 
                                nextW2ns[(nzero_Wn+thisn2)*nbinsr+elb1]*conj(nextWns[thisWshift_n2+elb2]);
                        }
                        // Double-counting corr for theta2==theta3
                        if ((elb2==elb3)){
                            batchGtilde_n[thisnrshift] -= wshape1 * 
                                nextW2ns[(nzero_Wn+thisn2+thisn3+1)*nbinsr+elb2]*nextWns[thisWshift_mn2mn3p1+elb1];
                            batchN_n[thisnrshift] -= w1 * 
                                nextW2ns[(nzero_Wn-thisn2-thisn3)*nbinsr+elb2] * nextWns[thisWshift_n2pn3+elb1];
                        }
                        // Nominal allocation
                        batchGtilde_n[thisnrshift] += wshape1 * nextWns[thisWshift_mn2mn3p1+elb1] * 
                                                      nextWns[thisWshift_n2+elb2] * nextWns[thisWshift_n3p1+elb3];
                        batchN_n[thisnrshift] += w1 * nextWns[thisWshift_n2pn3+elb1] * 
                                                 conj(nextWns[thisWshift_n2+elb2]) * conj(nextWns[thisWshift_n3+elb3]);
                    }                             
                }
                time2 = omp_get_wtime();
                if ((elregion%nregions_skip_print == 0)&&(thisthread==0)&&(ind_inpix1==lower1)){
                    printf("Allocated Gtilden for first gal in region %d/%d for thetabatch %d/%d in %.4f seconds for %d theta-combis\n",
                           elregion,nregions,elthetbatch,nthetbatches,(time2-time1),batch_nthetas);}
            }
            if ((elregion%nregions_skip_print == 0)&&(thisthread==0)){
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
        //   2b) Transform the Gammatilde to the target basis
        //   2c) Update the aperture MapNap3 integral
        int ntrafos;
        double complex *nextNM3correlator = calloc(1, sizeof(double complex));
        double complex *thisGtilde_n = calloc(1*n2n3combis, sizeof(double complex));
        double complex *thisN_n = calloc(n2n3combis, sizeof(double complex));
        double complex *thisGtilde_n_rec = calloc(1*n2n3combis_rec, sizeof(double complex));
        double complex *thisN_n_rec = calloc(n2n3combis_rec, sizeof(double complex));
        double complex *thisnpcf = calloc(1*batchGtilde_thetshift, sizeof(double complex));
        double complex *thisnpcf_norm = calloc(batchGtilde_thetshift, sizeof(double complex));
        for (int elb=0;elb<batch_nthetas;elb++){
            if (thisthread==0){
                printf("Done %.4f per cent of multipole-to-MapNap3 conversion\r",100.* (float) elb/batch_nthetas);}
            // 1)
            int nbshift, elb1, elb2, elb3, elb1t, elb2t, elb3t;
            elb1 = elb1s_batch[elb];
            elb2 = elb2s_batch[elb];
            elb3 = elb3s_batch[elb];
            int bincombi_trafos[6][3] = {{elb1,elb2,elb3}, {elb2,elb3,elb1}, {elb3,elb1,elb2},
                                         {elb1,elb3,elb2}, {elb2,elb1,elb3}, {elb3,elb2,elb1}}; 
            // 2)
            if ((elb1==elb2)&&(elb1==elb3)){ntrafos=1;}
            else if ((elb1==elb2)&&(elb1!=elb3)){ntrafos=3;}
            else if ((elb1==elb3)&&(elb1!=elb2)){ntrafos=3;}
            else if ((elb2==elb3)&&(elb2!=elb1)){ntrafos=3;}
            else{ntrafos=6;}
            for (int eltrafo=0;eltrafo<ntrafos;eltrafo++){
                elb1t = bincombi_trafos[eltrafo][0];
                elb2t = bincombi_trafos[eltrafo][1];
                elb3t = bincombi_trafos[eltrafo][2];
                //printf("elb1=%d eln2=%d elb3=%d: eltrafo=%d/%d\n",elb1,elb2,elb3,eltrafo,ntrafos+1);
                // 2a)
                for(int eln12=0;eln12<n2n3combis;eln12++){
                    nbshift = eln12*batchGtilde_nshift+elb;
                    thisGtilde_n[eln12] = batchGtilde_n[nbshift];
                    thisN_n[eln12] = batchN_n[nbshift];
                }
                getMultipolesFromSymm_GNNN(thisGtilde_n, thisN_n, nmax, eltrafo, nindices, len_nindices,
                                            thisGtilde_n_rec, thisN_n_rec);
                // OPTIONAL: Allocate 4PCF in multipole basis
                for(int eln12=0;eln12<n2n3combis_rec;eln12++){
                    if (alloc_4pcfmultipoles==1){
                        int thisnrshift = eln12*Gtilde_nshift + elb1t*nbinsr*nbinsr + elb2t*nbinsr + elb3t;
                        Gtilde_n[thisnrshift] = thisGtilde_n_rec[eln12];
                        N_n[thisnrshift] = thisN_n_rec[eln12];
                    }
                }
                // 2b)
                multipoles2npcf_gnnn_singletheta(thisGtilde_n_rec, thisN_n_rec, nmax, nmax,
                                                 elb1t, elb2t, elb3t,
                                                 phibins, phibins, nbinsphi, nbinsphi,
                                                 thisnpcf, thisnpcf_norm);

                // OPTIONAL: Allocate 4pcf in real basis (Shape: (1,ntheta,ntheta,ntheta,nphi,nphi)
                if (alloc_4pcfreal==1){
                    for (int elphi12=0;elphi12<batchGtilde_thetshift;elphi12++){
                        int Gtilde_rshift = nbinsphi*nbinsphi;
                        int Gtilde_phircombi = Gtilde_rshift*(elb1t*nbinsr*nbinsr+elb2t*nbinsr+elb3t)+elphi12;
                        Gtilde[Gtilde_phircombi] = thisnpcf[elphi12];
                        Norms[Gtilde_phircombi] = thisnpcf_norm[elphi12];
                    }
                }

                // 2c)
                double y1, y2, y3, dy1, dy2, dy3;
                int mapnap3threadshift = thisthread*napradii;
                for (int elapr=0; elapr<napradii; elapr++){
                    y1=bin_centers_batch[elb1t]/apradii[elapr];
                    y2=bin_centers_batch[elb2t]/apradii[elapr];
                    y3=bin_centers_batch[elb3t]/apradii[elapr];
                    dy1 = (bin_edges[elb1t+1]-bin_edges[elb1t])/apradii[elapr];
                    dy2 = (bin_edges[elb2t+1]-bin_edges[elb2t])/apradii[elapr];
                    dy3 = (bin_edges[elb3t+1]-bin_edges[elb3t])/apradii[elapr];
                    fourpcf2MN3correlator(
                         1, y1, y2, y3, dy1, dy2, dy3,
                         phibins, phibins, dbinsphi, dbinsphi, nbinsphi, nbinsphi, thisnpcf, nextNM3correlator);
                    if (isnan(cabs(nextNM3correlator[0]))==false){
                        allNM3correlator[mapnap3threadshift+elapr] += nextNM3correlator[0];
                    }
                    //if (true){printf("\nthread %d, elr %d, allMN3cont=%.20f ", thisthread, elapr, creal(nextNM3correlator[0]));}
                    nextNM3correlator[0] = 0;
                }

                // Reset 4pcf placeholders to zero
                for(int i=0;i<batchGtilde_thetshift;i++){
                    thisnpcf_norm[i] = 0;
                    thisnpcf[i] = 0;
                }
                for(int i=0;i<n2n3combis;i++){
                    thisN_n[i] = 0;
                    thisGtilde_n[i] = 0;
                }
                for(int i=0;i<n2n3combis_rec;i++){
                    thisN_n_rec[i] = 0;
                    thisGtilde_n_rec[i] = 0;
                }
            }
        }
        
        if (thisthread>-1){printf("Done allocating 4pcfs for thetabatch %d/%d\n",elthetbatch,nthetbatches);}
            
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
    for (int elthread=0;elthread<nthreads;elthread++){
        int mapnap3ind;
        int mapnap3threadshift = elthread*napradii;
        for (int elapr=0; elapr<napradii; elapr++){
            NM3correlator[elapr] += allNM3correlator[elthread*napradii+elapr];
        }
    }    
    free(allNM3correlator);
}

// Here we implement a runtime-optimised implementation when only subselecting
// a limited range of radial bin configurations
// To keep the memory as low as possible we further restrict the parallel allocation 
// to the (thet1 <= thet2 <= thet3) combis configurations and lateron allocate the
// other permutations based on the symmetry properties.
void alloc_notomoGammans_tree_nnnn(
    double *isinner, double *weight, double *pos1, double *pos2, int ngal, 
    int nmax, double rmin, double rmax, int nbinsr, int nthetacombis, int dccorr, 
    int *nindices, int len_nindices, 
    int nresos, double *reso_redges, int *ngal_resos, 
    double *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos, 
    int *index_matcher_hash, int *pixs_galind_bounds, int *pix_gals, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, int verbose,
    double *bin_centers, double complex *N_n){

    // Temporary arrays that are allocated in parallel and later reduced
    double *tmpwcounts = calloc(nthreads*nbinsr, sizeof(double));
    double *tmpwnorms = calloc(nthreads*nbinsr, sizeof(double));
    double complex *tmpN_n = calloc(nthreads*len_nindices*nthetacombis, sizeof(double complex));
    
    double *totcounts = calloc(nbinsr, sizeof(double));
    double *totnorms = calloc(nbinsr, sizeof(double));

    // Helper array that checks how many regions have been already computed
    int *regionsdone = calloc(nregions, sizeof(int));
    int nregionsdone = 0;
    
    #pragma omp parallel for num_threads(nthreads)
    for(int elthread=0;elthread<nthreads;elthread++){
        int nregions_per_thread = nregions/nthreads;
        int nmax_alloc = 2*nmax+1;
        int nbinsz = 1;
        int nnvals_Wn = 4*nmax_alloc+1; // Need to cover [-n1-n2, n1+n2]
        int nnvals_Upsn = 2*nmax_alloc+1; // Need tocover [-nmax,+nmax]
        int nzero_Wn = 2*nmax_alloc;
        int nzero_Ups = nmax_alloc;
        int ups_compshift = len_nindices*nthetacombis;

        int nbinszr = nbinsz*nbinsr;
        double complex *nextWns = calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW2ns = calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW3ns = calloc(nbinszr, sizeof(double complex));

        int npix_hash = pix1_n*pix2_n;
        int *rshift_index_matcher_hash = calloc(nresos, sizeof(int));
        int *rshift_pixs_galind_bounds = calloc(nresos, sizeof(int));
        int *rshift_pix_gals = calloc(nresos, sizeof(int));
        for (int elreso=1;elreso<nresos;elreso++){
            rshift_index_matcher_hash[elreso] = rshift_index_matcher_hash[elreso-1] + npix_hash;
            rshift_pixs_galind_bounds[elreso] = rshift_pixs_galind_bounds[elreso-1] + ngal_resos[elreso-1]+1;
            rshift_pix_gals[elreso] = rshift_pix_gals[elreso-1] + ngal_resos[elreso-1];
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
            int region_debug = mymin(500,nregions-1);
            bool printregdbg = (verbose>1) && (elregion==region_debug);
            if (printregdbg){printf("Region %d is in thread %d (%i regions in total)\n",
                elregion,elthread,nregions);}
            
            int lower1 = pixs_galind_bounds[elregion];
            int upper1 = pixs_galind_bounds[elregion+1];
            for (int ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                int ind_gal = pix_gals[ind_inpix1];
                double p11, p12, w1;
                double innergal = isinner[ind_gal];
                if (innergal<1e-5){continue;}
                p11 = pos1[ind_gal];
                p12 = pos2[ind_gal];
                w1 = innergal*weight[ind_gal];    
                
                // Gns have shape (nnvals, nbinsz, nbinsr)
                // where the ns are ordered as 
                // [-nmax_1-nmax_2-3, ..., nmax_1+nmax_2+3]
                for (int i=0;i<nnvals_Wn*nbinszr;i++){nextWns[i]=0;nextW2ns[i]=0;}
                for (int i=0;i<nbinszr;i++){nextW3ns[i]=0;}
                
                int ind_gal2;
                int ind_red, lower, upper; 
                double  p21, p22, w2, w2_sq,rel1, rel2, dist2, dist, dphi;
                double complex phirot, phirotc, nphirot, nphirotc, wadd, w2add;
                // Allocate Gn, Wn and their multiple-couting corrections
                for (int elreso=0;elreso<=nresos;elreso++){
                    int ind_rbin, rbin, zrshift, nextnshift, ind_Gn, ind_G2n, ind_Wn;
                    double rmin_reso = reso_redges[elreso];
                    double rmin_reso2 = rmin_reso*rmin_reso;
                    double rmax_reso = reso_redges[elreso+1];
                    double rmax_reso2 = rmax_reso*rmax_reso;
                    int pix1_lower = mymax(0, (int) floor((p11 - (rmax_reso+pix1_d) - pix1_start)/pix1_d));
                    int pix2_lower = mymax(0, (int) floor((p12 - (rmax_reso+pix2_d) - pix2_start)/pix2_d));
                    int pix1_upper = mymin(pix1_n-1, (int) floor((p11 + (rmax_reso+pix1_d) - pix1_start)/pix1_d));
                    int pix2_upper = mymin(pix2_n-1, (int) floor((p12 + (rmax_reso+pix2_d) - pix2_start)/pix2_d));
                    for (int ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                        for (int ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                            ind_red = index_matcher_hash[rshift_index_matcher_hash[elreso] + ind_pix2*pix1_n + ind_pix1];
                            if (ind_red==-1){continue;}
                            lower = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+ind_red];
                            upper = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+ind_red+1];
                            for (int ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                                ind_gal2 = rshift_pix_gals[elreso] + pix_gals[rshift_pix_gals[elreso]+ind_inpix];
                                //#pragma omp critical
                                p21 = pos1_resos[ind_gal2];
                                p22 = pos2_resos[ind_gal2];
                                w2 = weight_resos[ind_gal2];
                                
                                rel1 = p21 - p11;
                                rel2 = p22 - p12;
                                dist2 = rel1*rel1 + rel2*rel2;
                                if(dist2 < rmin_reso2 || dist2 >= rmax_reso2){continue;}
                                dist = sqrt(dist2);
                                rbin = (int) floor((log(dist)-log(rmin))/drbin);
                                w2_sq = w2*w2;
                                dphi = atan2(rel2,rel1);
                                phirot = cexp(I*dphi);
                                phirotc = conj(phirot);
                                zrshift = 0*nbinsr + rbin;
                                ind_Wn = nzero_Wn*nbinszr + zrshift;
                                nphirot = 1+I*0;
                                nphirotc = 1+I*0;

                                // Triple-counting corr
                                nextW3ns[zrshift] += w2_sq*w2;                        

                                // Nominal G and double-counting corr
                                // n = 0
                                totcounts[zrshift] += w1*w2*dist; 
                                totnorms[zrshift] += w1*w2; 
                                nextWns[ind_Wn] += w2*nphirot;  
                                nextW2ns[ind_Wn] += w2_sq*nphirot;
                                // /*
                                // n \in [-2*nmax+1,2*nmax-1]                          
                                nphirot *= phirot;
                                nphirotc *= phirotc; 
                                // n in [1, ..., 2*nmax_alloc-1] x {+1,-1}
                                nextnshift = 0;
                                for (int nextn=1;nextn<=2*nmax_alloc;nextn++){
                                    nextnshift = nextn*nbinszr;
                                    wadd=w2*nphirot; w2add=w2_sq*nphirot;
                                    nextWns[ind_Wn+nextnshift] += wadd;
                                    nextWns[ind_Wn-nextnshift] += conj(wadd);
                                    nextW2ns[ind_Wn+nextnshift] += w2add;
                                    nextW2ns[ind_Wn-nextnshift] += conj(w2add);
                                    nphirot *= phirot;
                                    nphirotc *= phirotc; 
                                }  
                            }
                        }
                    }
                }
                
                // Allocate Upsilon
                // Upsilon_mu have shape 
                // (nindices, rcombis)
                // Ups_0 ~ wgamma  *  G_{n2+n3-3}  *  G_{-n2-2}  *  G_{-n3-3}
                // Ups_1 ~ wgammac *  G_{n2+n3-1}  *  G_{-n2-2}  *  G_{-n3-1}
                double complex gGG0, gGG1, gGG2, gGG3, gGG4, gGG5, gGG6, gGG7, wNN;
                int thisn2, thisn3, thisn, thisnshift, thisnrshift, elbcombi, elb1, elb2, elb3;
                int thisWshift_n2, thisWshift_n3, thisWshift_n2n3;
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
                    thisWshift_n2n3 = (nzero_Wn+thisn)*nbinsr;
                    thisnshift = nindex * nthetacombis;
                    elbcombi = 0;
                    for (int elb1=0; elb1<nbinsr; elb1++){
                        thisnrshift = elthread*ups_compshift + thisnshift + elbcombi;
                        // Triple-counting corr
                        tmpN_n[thisnrshift] += 2 * w1*nextW3ns[elb1];

                        for (int elb2=elb1; elb2<nbinsr; elb2++){
                            thisnrshift = elthread*ups_compshift + thisnshift + elbcombi;
                            // Double-counting corr for theta1==theta2
                            if (elb1==elb2){
                                tmpN_n[thisnrshift] -= w1 * 
                                    nextW2ns[(nzero_Wn+thisn3)*nbinsr+elb1]*conj(nextWns[thisWshift_n3+elb2]);
                            }

                            // Allocation of first three complex products for Norm updates
                            wNN = w1*nextWns[thisWshift_n2n3+elb1]*conj(nextWns[thisWshift_n2+elb2]);
                            
                            for (int elb3=elb2; elb3<nbinsr; elb3++){
                                thisnrshift = elthread*ups_compshift + thisnshift + elbcombi;
                                // Double-counting corr for theta1==theta3 
                                if ((elb1==elb3) && (elb1!=elb2)){ 
                                    tmpN_n[thisnrshift] -= w1 * 
                                        nextW2ns[(nzero_Wn+thisn2)*nbinsr+elb1] * conj(nextWns[thisWshift_n2+elb2]);
                                }
                                // Double-counting corr for theta2==theta3
                                if ((elb2==elb3) && (elb1!=elb2)){ 
                                    tmpN_n[thisnrshift] -= w1 * 
                                        nextW2ns[(nzero_Wn-thisn2-thisn3)*nbinsr+elb2] * nextWns[thisWshift_n2n3+elb1];
                                }

                                // Nominal allocation of Upsilon and Norm
                                tmpN_n[thisnrshift] += wNN*conj(nextWns[thisWshift_n3+elb3]);

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

        free(rshift_index_matcher_hash);
        free(rshift_pixs_galind_bounds);
        free(rshift_pix_gals);
    }

    // Accumulate Upsilon_n and N_n
    // 1) Build arrays that hold bin combis for b1<=b2<=b3
    // 2) Get bin edges and bin centers of the combinations
    // 3) Find all (theta1,theta2,theta3) combis that can be reconstructed via the symmetries
    // 4) Get the Gamma_mu(theta1,theta2,theta3,phi12,phi13)

    // 1)
    int elbcombi = 0;
    int *elb1_inds = calloc(nthetacombis, sizeof(int));
    int *elb2_inds = calloc(nthetacombis, sizeof(int));
    int *elb3_inds = calloc(nthetacombis, sizeof(int));
    for (int elb1=0;elb1<nbinsr;elb1++){
        for (int elb2=elb1;elb2<nbinsr;elb2++){
            for (int elb3=elb2;elb3<nbinsr;elb3++){
                elb1_inds[elbcombi] = elb1;
                elb2_inds[elbcombi] = elb2;
                elb3_inds[elbcombi] = elb3;
                elbcombi += 1;
            }
        }
    }

    
    #pragma omp parallel for num_threads(nthreads)
    for (int elb=0;elb<nthetacombis;elb++){

        int ntrafos, tnrshift, nbshift, nbshift_tmp, elb1, elb2, elb3, elb1t, elb2t, elb3t;
        int thisn2, thisn3, thisn;
        int nmax_alloc = 2*nmax+1;
        int nnvals_Upsn_rec = 2*nmax+1; 
        int nnvals_Upsn = 2*nmax_alloc+1; 
        int nzero_Ups = nmax_alloc;
        int ups_nshift = nbinsr*nbinsr*nbinsr;
        int n2n3combis = nnvals_Upsn*nnvals_Upsn;
        int n2n3combis_rec = nnvals_Upsn_rec*nnvals_Upsn_rec;
        int ups_rec_compshift = n2n3combis_rec*ups_nshift;

        double complex *thisN_n = calloc(n2n3combis, sizeof(double complex));
        double complex *thisN_n_rec = calloc(n2n3combis_rec, sizeof(double complex));

        // 2)
        elb1 = elb1_inds[elb];
        elb2 = elb2_inds[elb];
        elb3 = elb3_inds[elb];
        int bincombi_trafos[6][3] = {{elb1,elb2,elb3}, {elb2,elb3,elb1}, {elb3,elb1,elb2},
                                     {elb1,elb3,elb2}, {elb2,elb1,elb3}, {elb3,elb2,elb1}}; 
        
        // 3)
        if ((elb1==elb2)&&(elb1==elb3)){ntrafos=1;}
        else if ((elb1==elb2)&&(elb1!=elb3)){ntrafos=3;}
        else if ((elb1==elb3)&&(elb1!=elb2)){ntrafos=3;}
        else if ((elb2==elb3)&&(elb2!=elb1)){ntrafos=3;}
        else{ntrafos=6;}
        for (int eltrafo=0;eltrafo<ntrafos;eltrafo++){
            elb1t = bincombi_trafos[eltrafo][0];
            elb2t = bincombi_trafos[eltrafo][1];
            elb3t = bincombi_trafos[eltrafo][2];
            for (int nindex=0;nindex<len_nindices;nindex++){
                thisn2 = nindices[nindex]/nnvals_Upsn - nzero_Ups;
                thisn3 = nindices[nindex]%nnvals_Upsn - nzero_Ups;
                nbshift_tmp = nindex*nthetacombis+elb;
                nbshift = ((thisn2+nzero_Ups)*nnvals_Upsn + (thisn3+nzero_Ups));
                for (int elthread=0;elthread<nthreads;elthread++){
                    tnrshift = elthread*len_nindices*nthetacombis + nindex*nthetacombis + elb;
                    thisN_n[nbshift] += tmpN_n[tnrshift];
                }
            }
            getMultipolesFromSymm_NNNN(thisN_n, nmax, eltrafo, nindices, len_nindices, thisN_n_rec);

            // 4)
            for(int eln12=0;eln12<n2n3combis_rec;eln12++){
                int thisnrshift = eln12*ups_nshift + elb1t*nbinsr*nbinsr + elb2t*nbinsr + elb3t;
                N_n[thisnrshift] = thisN_n_rec[eln12];
            }  

            // Reset 4pcf placeholders to zero
            for(int i=0;i<n2n3combis;i++){
                thisN_n[i] = 0;
            }
            for(int i=0;i<n2n3combis_rec;i++){
                thisN_n_rec[i] = 0;
            }
        }
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
void alloc_notomoNap4_tree_nnnn(
    double *isinner, double *weight, double *pos1, double *pos2, int ngal, 
    int nmax, double rmin, double rmax, int nbinsr, int dccorr, 
    int *nindices, int len_nindices, double *phibins, double *dbinsphi, int nbinsphi,
    int nresos, double *reso_redges, int *ngal_resos, 
    double *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos, 
    int *index_matcher_hash, int *pixs_galind_bounds, int *pix_gals, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, double *napradii, int nnapradii, double complex *N4correlators, 
    int alloc_4pcfmultipoles, int alloc_4pcfreal,
    double *bin_centers, double complex *N_n, double complex *Counts){
               
    double complex *allN4correlators = calloc(nthreads*1*nnapradii, sizeof(double complex));
    #pragma omp parallel for num_threads(nthreads)
    for (int elthetbatch=0;elthetbatch<nthetbatches;elthetbatch++){
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
        int N_rec_compshift = n2n3combis_rec*N_nshift;
        
        int batch_nthetas = nthetacombis_batches[elthetbatch];
        int batchN_nshift = batch_nthetas;
        int batchN_compshift = n2n3combis*batchN_nshift;
        int batchgamma_thetshift = nbinsphi*nbinsphi;
        
        int npix_hash = pix1_n*pix2_n;
        int *rshift_index_matcher_hash = calloc(nresos, sizeof(int));
        int *rshift_pixs_galind_bounds = calloc(nresos, sizeof(int));
        int *rshift_pix_gals = calloc(nresos, sizeof(int));
        for (int elreso=1;elreso<nresos;elreso++){
            rshift_index_matcher_hash[elreso] = rshift_index_matcher_hash[elreso-1] + npix_hash;
            rshift_pixs_galind_bounds[elreso] = rshift_pixs_galind_bounds[elreso-1] + ngal_resos[elreso-1]+1;
            rshift_pix_gals[elreso] = rshift_pix_gals[elreso-1] + ngal_resos[elreso-1];
        }

        double *totcounts = calloc(nbinsr, sizeof(double));
        double *totnorms = calloc(nbinsr, sizeof(double));
        double *bin_centers_batch = calloc(nbinsr, sizeof(double));
        double complex *batchN_n = calloc(batchN_compshift, sizeof(double complex));
        double *batch_thetas1 = calloc(batch_nthetas, sizeof(double));
        double *batch_thetas2 = calloc(batch_nthetas, sizeof(double));
        double *batch_thetas3 = calloc(batch_nthetas, sizeof(double));
        
        int nbinszr = nbinsz*nbinsr;
        double complex *nextWns = calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW2ns = calloc(nnvals_Wn*nbinszr, sizeof(double complex));
        double complex *nextW3ns = calloc(nbinszr, sizeof(double complex));
        
        double drbin = (log(rmax)-log(rmin))/(nbinsr);
        int rbin_min_batch=nbinsr;int rbin_max_batch=0;
        int reso_min_batch=0; int reso_max_batch=0;
        int *elb1s_batch = calloc(batch_nthetas, sizeof(int));
        int *elb2s_batch = calloc(batch_nthetas, sizeof(int));
        int *elb3s_batch = calloc(batch_nthetas, sizeof(int));
        double *bin_edges = calloc(nbinsr+1, sizeof(double));
        
        
        #pragma omp critical
        {
            for (int elb=0;elb<batch_nthetas;elb++){
                int thisrcombi = thetacombis_batches[cumthetacombis_batches[elthetbatch]+elb];
                elb1s_batch[elb] = thisrcombi/(nbinsr*nbinsr);
                elb2s_batch[elb] = (thisrcombi-elb1s_batch[elb]*nbinsr*nbinsr)/nbinsr;
                elb3s_batch[elb] = thisrcombi-elb1s_batch[elb]*nbinsr*nbinsr-elb2s_batch[elb]*nbinsr;
                rbin_min_batch = mymin(rbin_min_batch, elb1s_batch[elb]); 
                rbin_max_batch = mymax(rbin_max_batch, elb3s_batch[elb]); 
            }
            bin_edges[0] = rmin;
            for (int elb=0;elb<nbinsr;elb++){
                bin_edges[elb+1] = bin_edges[elb]*exp(drbin);
            }
            for (int elreso=1;elreso<nresos;elreso++){
                if (reso_redges[elreso] <= bin_edges[rbin_min_batch  ]){reso_min_batch += 1;}
                if (reso_redges[elreso] <  bin_edges[rbin_max_batch+1]){reso_max_batch += 1;}
            }
            //printf("For batch %d with imin=%d imax=%d we have resomin=%d resomax=%d",
            //       elthetbatch, rbin_min_batch, rbin_max_batch, reso_min_batch, reso_max_batch);
        }
        
        
        
        // Allocate the 4pcf multipoles for this batch of radii 
        int offset_per_thread = nregions/nthreads;
        int offset = offset_per_thread*thisthread;
        for (int _elregion=0; _elregion<nregions; _elregion++){
            int elregion = (_elregion+offset)%nregions; // Try to evade collisions
            if ((elregion%nregions_skip_print == 0)&&(thisthread==0)){
                printf("Doing region %d/%d for thetabatch %d/%d\n",elregion,nregions,elthetbatch,nthetbatches);
            }
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
                int ind_red, lower, upper; 
                double  p21, p22, w2, w2_sq, rel1, rel2, dist, dphi;
                double complex phirot, phirotc, twophirotc, nphirot, nphirotc;
                
                // Check how many ns we need for Gn
                // Gns have shape (nnvals, nbinsz, nbinsr)
                // where the ns are ordered as 
                // [-nmax_1-nmax_2-3, ..., nmax_1+nmax_2+3]
                for (int i=0;i<nnvals_Wn*nbinszr;i++){nextWns[i]=0;nextW2ns[i]=0;}
                for (int i=0;i<nbinszr;i++){nextW3ns[i]=0;}
                for (int elreso=reso_min_batch;elreso<=reso_max_batch;elreso++){
                    int rbin, zrshift, nextnshift, ind_Wn;
                    double rmin_reso = reso_redges[elreso];
                    double rmax_reso = reso_redges[elreso+1];
                    int pix1_lower = mymax(0, (int) floor((p11 - (rmax_reso+pix1_d) - pix1_start)/pix1_d));
                    int pix2_lower = mymax(0, (int) floor((p12 - (rmax_reso+pix2_d) - pix2_start)/pix2_d));
                    int pix1_upper = mymin(pix1_n-1, (int) floor((p11 + (rmax_reso+pix1_d) - pix1_start)/pix1_d));
                    int pix2_upper = mymin(pix2_n-1, (int) floor((p12 + (rmax_reso+pix2_d) - pix2_start)/pix2_d));
                    for (int ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                        for (int ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                            ind_red = index_matcher_hash[rshift_index_matcher_hash[elreso] + ind_pix2*pix1_n + ind_pix1];
                            if (ind_red==-1){continue;}
                            lower = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+ind_red];
                            upper = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+ind_red+1];
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
                                if(dist < rmin_reso || dist >= rmax_reso) continue;
                                rbin = (int) floor((log(dist)-log(rmin))/drbin);
                                w2_sq = w2*w2;
                                dphi = atan2(rel2,rel1);
                                phirot = cexp(I*dphi);
                                phirotc = conj(phirot);
                                twophirotc = phirotc*phirotc;
                                zrshift = 0*nbinsr + rbin;
                                ind_Wn = nzero_Wn*nbinszr + zrshift;
                                nphirot = 1+I*0;
                                nphirotc = 1+I*0;

                                // Triple-counting corr
                                nextW3ns[zrshift] += w2_sq*w2;

                                // Nominal G and double-counting corr
                                // n = 0
                                totcounts[zrshift] += w1*w2*dist; 
                                totnorms[zrshift] += w1*w2; 
                                nextWns[ind_Wn] += w2*nphirot;  
                                nextW2ns[ind_Wn] += w2_sq*nphirot;
                                // /*
                                // n \in [-2*nmax+1,2*nmax-1]                          
                                nphirot *= phirot;
                                nphirotc *= phirotc; 
                                // n in [1, ..., 2*nmax_alloc-1] x {+1,-1}
                                nextnshift = 0;
                                for (int nextn=1;nextn<=2*nmax_alloc;nextn++){
                                    nextnshift = nextn*nbinszr;
                                    nextWns[ind_Wn+nextnshift] += w2*nphirot;
                                    nextWns[ind_Wn-nextnshift] += w2*nphirotc;
                                    nextW2ns[ind_Wn+nextnshift] += w2_sq*nphirot;
                                    nextW2ns[ind_Wn-nextnshift] += w2_sq*nphirotc;
                                    nphirot *= phirot;
                                    nphirotc *= phirotc; 
                                }  
                            }
                        }
                    }
                }
                time2 = omp_get_wtime();
                if ((elregion%nregions_skip_print == 0)&&(thisthread==0)&&(ind_inpix1==lower1)){
                    printf("Computed Wn for first gal in region %d/%d for thetabatch %d/%d in %.4f seconds\n",
                           elregion,nregions,elthetbatch,nthetbatches,(time2-time1));
                } 

                // Allocate Upsilon
                // Upsilon have shape 
                // (ncomp,(2*nmax_alloc+1),(2*nmax_alloc+1),nthetas)
                time1 = omp_get_wtime();
                double complex wNN;
                int thisn2, thisn3, thisn, thisnshift, thisnrshift, elb1, elb2, elb3;
                int thisWshift_n2, thisWshift_n3, thisWshift_n2n3;
                for (int nindex=0; nindex<len_nindices; nindex++){
                    thisn2 = nindices[nindex]/nnvals_Nn - nzero_Nn;
                    thisn3 = nindices[nindex]%nnvals_Nn - nzero_Nn;
                    if (thisn2>nzero_Nn || -thisn2>nzero_Nn || thisn3>nzero_Nn || -thisn3>nzero_Nn){
                        if (elregion==0 && elthetbatch==0){
                            printf("Error at elregion=%d batch=%d nindex=%d: nindices[nindex]=%d n2=%d n3=%d",
                                   elregion, elthetbatch, nindex, nindices[nindex], thisn2, thisn3);}
                        continue;
                    }
                        
                    thisn = thisn2+thisn3;
                    //if (elregion==0 && elthetbatch==0){printf("nindex %d: n2=%d n3=%d\n",nindex,thisn2,thisn3);}
                    thisWshift_n2 = (nzero_Wn+thisn2)*nbinsr;
                    thisWshift_n3 = (nzero_Wn+thisn3)*nbinsr;
                    thisWshift_n2n3 = (nzero_Wn+thisn)*nbinsr;
                    thisnshift = ((thisn2+nzero_Nn)*nnvals_Nn + (thisn3+nzero_Nn)) * batchN_nshift;
                    for (int elb=0;elb<batch_nthetas;elb++){
                        elb1 = elb1s_batch[elb];
                        elb2 = elb2s_batch[elb];
                        elb3 = elb3s_batch[elb];
                        thisnrshift = thisnshift + elb;
                        // Multiple counting corrections:
                        // sum_(i neq j neq k) = sum_(i,j,k) - ( sum_(i, j, i=k) + 2perm ) + 2 * sum_(i, i=j, i=k)
                        // Triple-counting corr
                        if ((elb1==elb2) && (elb1==elb3) && (elb2==elb3)){
                            batchN_n[thisnrshift] += 2 * w1*nextW3ns[elb1];
                        }
                        // Double-counting corr for theta1==theta2
                        if (elb1==elb2){
                            batchN_n[thisnrshift] -= w1 * 
                                nextW2ns[(nzero_Wn+thisn3)*nbinsr+elb1]*conj(nextWns[thisWshift_n3+elb3]);
                        }
                        // Double-counting corr for theta1==theta3  
                        if (elb1==elb3){
                            batchN_n[thisnrshift] -= w1 * 
                                nextW2ns[(nzero_Wn+thisn2)*nbinsr+elb1] * conj(nextWns[thisWshift_n2+elb2]);
                        }
                        // Double-counting corr for theta2==theta3
                        if (elb2==elb3){
                            batchN_n[thisnrshift] -= w1 * 
                                nextW2ns[(nzero_Wn-thisn2-thisn3)*nbinsr+elb2] * nextWns[thisWshift_n2n3+elb1];
                        }
                        // Nominal allocation
                        wNN = w1*nextWns[thisWshift_n2n3+elb1]*conj(nextWns[thisWshift_n2+elb2]);
                        batchN_n[thisnrshift] += wNN*conj(nextWns[thisWshift_n3+elb3]);
                    }
                }
                
                time2 = omp_get_wtime();
                if ((elregion%nregions_skip_print == 0)&&(thisthread==0)&&(ind_inpix1==lower1)){
                    printf("Allocated Nns for first gal in region %d/%d for thetabatch %d/%d in %.4f seconds for %d theta-combis\n",
                           elregion,nregions,elthetbatch,nthetbatches,(time2-time1),batch_nthetas);
                }
                if ((elregion%nregions_skip_print == 0)&&(thisthread==0)&&(ind_inpix1==lower1)){
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
        int ntrafos;
        double complex *nextN4correlators = calloc(1, sizeof(double complex));
        double complex *thisN_n = calloc(n2n3combis, sizeof(double complex));
        double complex *thisN_n_rec = calloc(n2n3combis_rec, sizeof(double complex));
        double complex *thisnpcf = calloc(batchgamma_thetshift, sizeof(double complex));
        for (int elb=0;elb<batch_nthetas;elb++){
            if (thisthread==0){
                printf("Done %.4f per cent of multipole-to-Nap4 conversion\r",100.* (float) elb/batch_nthetas);}
            // 1)
            
            int nbshift, elb1, elb2, elb3, elb1t, elb2t, elb3t;
            elb1 = elb1s_batch[elb];
            elb2 = elb2s_batch[elb];
            elb3 = elb3s_batch[elb];
            int bincombi_trafos[6][3] = {{elb1,elb2,elb3}, {elb2,elb3,elb1}, {elb3,elb1,elb2},
                                         {elb1,elb3,elb2}, {elb2,elb1,elb3}, {elb3,elb2,elb1}}; 
            // 2)
            if ((elb1==elb2)&&(elb1==elb3)){ntrafos=1;}
            else if ((elb1==elb2)&&(elb1!=elb3)){ntrafos=3;}
            else if ((elb1==elb3)&&(elb1!=elb2)){ntrafos=3;}
            else if ((elb2==elb3)&&(elb2!=elb1)){ntrafos=3;}
            else{ntrafos=6;}
            for (int eltrafo=0;eltrafo<ntrafos;eltrafo++){
                elb1t = bincombi_trafos[eltrafo][0];
                elb2t = bincombi_trafos[eltrafo][1];
                elb3t = bincombi_trafos[eltrafo][2];
                //printf("elb1=%d eln2=%d elb3=%d: eltrafo=%d/%d\n",elb1,elb2,elb3,eltrafo,ntrafos+1);

                // 2a)
                for(int eln12=0;eln12<n2n3combis;eln12++){
                    nbshift = eln12*batchN_nshift+elb;
                    thisN_n[eln12] = batchN_n[nbshift];
                }

                getMultipolesFromSymm_NNNN(thisN_n, nmax, eltrafo, nindices, len_nindices, thisN_n_rec);
                // OPTIONAL: Allocate 4PCF in multipole basis
                for(int eln12=0;eln12<n2n3combis_rec;eln12++){
                    if (alloc_4pcfmultipoles==1){
                        int thisnrshift = eln12*N_nshift + elb1t*nbinsr*nbinsr + elb2t*nbinsr + elb3t;
                        N_n[thisnrshift] = thisN_n_rec[eln12];
                    }
                }

                // 2b)
                multipoles2npcf_nnnn_singletheta(thisN_n_rec, nmax, nmax,
                                                 elb1t, elb2t, elb3t,
                                                 phibins, phibins, nbinsphi, nbinsphi,
                                                 thisnpcf);

                // OPTIONAL: Allocate 4pcf in real basis (Shape: (8,ntheta,ntheta,ntheta,nphi,nphi)
                if (alloc_4pcfreal==1){
                    for (int elphi12=0;elphi12<batchgamma_thetshift;elphi12++){
                        int gamma_rshift = nbinsphi*nbinsphi;
                        int gamma_phircombi = gamma_rshift*(elb1t*nbinsr*nbinsr+elb2t*nbinsr+elb3t)+elphi12;
                        Counts[gamma_phircombi] = thisnpcf[elphi12];
                    }
                }

                // 2c)
                double y1, y2, y3, dy1, dy2, dy3;
                int nap4ind;
                int nap4threadshift = thisthread*nnapradii;
                for (int elnapr=0; elnapr<nnapradii; elnapr++){
                    y1=bin_centers_batch[elb1t]/napradii[elnapr];
                    y2=bin_centers_batch[elb2t]/napradii[elnapr];
                    y3=bin_centers_batch[elb3t]/napradii[elnapr];
                    dy1 = (bin_edges[elb1t+1]-bin_edges[elb1t])/napradii[elnapr];
                    dy2 = (bin_edges[elb2t+1]-bin_edges[elb2t])/napradii[elnapr];
                    dy3 = (bin_edges[elb3t+1]-bin_edges[elb3t])/napradii[elnapr];
                    
                    fourpcf2N4correlators(1,
                                          y1, y2, y3, dy1, dy2, dy3,
                                          phibins, phibins, dbinsphi, dbinsphi, nbinsphi, nbinsphi,
                                          thisnpcf, nextN4correlators);
                    
                    nap4ind = elnapr;
                    if (isnan(cabs(nextN4correlators[0]))==false){
                        allN4correlators[nap4threadshift+nap4ind] += nextN4correlators[0];
                    }
                    nextN4correlators[0] = 0;
                }

                // Reset 4pcf placeholders to zero
                for(int i=0;i<batchgamma_thetshift;i++){
                    thisnpcf[i] = 0;
                }
                for(int i=0;i<n2n3combis;i++){
                    thisN_n[i] = 0;
                }
                for(int i=0;i<n2n3combis_rec;i++){
                    thisN_n_rec[i] = 0;
                }
            }
        }
        
        
        for (int elnapr=0; elnapr<nnapradii; elnapr++){
            int nap4ind = elnapr;
            int nap4threadshift = thisthread*nnapradii;
            printf("\nthread %d, elr %d, elcomp %d, allN4cont=%.20f ",
                           thisthread, elnapr, 0, creal(allN4correlators[nap4threadshift+nap4ind]));
        }
        
            
        if (thisthread>-1){printf("Done allocating 4pcfs for thetabatch %d/%d\n",elthetbatch,nthetbatches);}

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
    for (int elthread=0;elthread<nthreads;elthread++){
        int nap4ind;
        int nap4threadshift = elthread*nnapradii;
        for (int elnapr=0; elnapr<nnapradii; elnapr++){
            nap4ind = elnapr;
            N4correlators[nap4ind] += allN4correlators[nap4threadshift+nap4ind];
        }
    }
    free(allN4correlators);
}


//////////////////////////////
// DOUBLETREE 4PCF (notomo)  //
//////////////////////////////
// Memory-efficient acceleration of alloc_notomoNap4_tree_nnnn.
// The single-leg multipole moments (Wn, W2n, W3n) that build the multipole 4PCF
// depend only on the central galaxy, not on the (theta1,theta2,theta3) batch. The
// "Tree" version recomputes them once per theta-batch (the batch loop is outermost),
// which is the dominant cost. Here we instead:
//   Phase 1: compute each central's moments exactly once and store them in a cache.
//   Phase 2: the (unchanged) per-batch combination + multipole->Map^4 transform reads
//            the cached moments instead of re-scanning the catalogue.
// This is exact (no additional approximation w.r.t. the Tree). The output multipoles
// N_n are identical to the Tree; the Nap^4 integral additionally fixes a latent bug in
// the Tree where bin_centers_batch was never populated (left at 0).
//
// Memory contract: the moment cache is the dominant buffer with
//   2 * ncache * nnvals_Wn * nbinsr  complex doubles   (Wn and W2n caches)
// where ncache = pixs_galind_bounds[nregions]. It is split into chunks of < 1e9
// elements each so that no single allocation / int32 index reaches the 2e9 wall.
void alloc_notomoNap4_doubletree_nnnn(
    double *isinner, double *weight, double *pos1, double *pos2, int ngal,
    int nmax, double rmin, double rmax, int nbinsr, int dccorr,
    int *nindices, int len_nindices, double *phibins, double *dbinsphi, int nbinsphi,
    int nresos, double *reso_redges, int *ngal_resos,
    double *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos,
    int *index_matcher_hash, int *pixs_galind_bounds, int *pix_gals, int nregions,
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, double *napradii, int nnapradii, double complex *N4correlators,
    int alloc_4pcfmultipoles, int alloc_4pcfreal,
    double *bin_centers, double complex *N_n, double complex *Counts){

    // * nmax_alloc specifies the largest multipole that needs to be allocated when wanting
    //   to allocate the Upsn/Nn while making use of the symmetry properties
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

    // Per-resolution offsets into the flattened hierarchical-grid arrays (shared, read-only)
    int *rshift_index_matcher_hash = calloc(nresos, sizeof(int));
    int *rshift_pixs_galind_bounds = calloc(nresos, sizeof(int));
    int *rshift_pix_gals = calloc(nresos, sizeof(int));
    for (int elreso=1;elreso<nresos;elreso++){
        rshift_index_matcher_hash[elreso] = rshift_index_matcher_hash[elreso-1] + npix_hash;
        rshift_pixs_galind_bounds[elreso] = rshift_pixs_galind_bounds[elreso-1] + ngal_resos[elreso-1]+1;
        rshift_pix_gals[elreso] = rshift_pix_gals[elreso-1] + ngal_resos[elreso-1];
    }

    // Radial bin edges (shared, read-only)
    double *bin_edges = calloc(nbinsr+1, sizeof(double));
    bin_edges[0] = rmin;
    for (int elb=0;elb<nbinsr;elb++){ bin_edges[elb+1] = bin_edges[elb]*exp(drbin); }

    //////////////////////////////
    // Phase 1: build moment cache
    //////////////////////////////
    // One slot per central-galaxy index ind_inpix1 in [0, pixs_galind_bounds[nregions]).
    // The cache is chunked so each allocation stays well below the 2e9-element int32 wall.
    long ncache = (long) pixs_galind_bounds[nregions];
    long wn_per_gal = (long) nnvals_Wn*nbinsr;
    int gpc = (int)(1000000000L / wn_per_gal); if (gpc<1){gpc=1;} // galaxies per chunk
    int nchunks = (int)((ncache + (long)gpc - 1)/(long)gpc);
    double complex **Wncache  = malloc(nchunks*sizeof(double complex*));
    double complex **W2ncache = malloc(nchunks*sizeof(double complex*));
    double complex **W3ncache = malloc(nchunks*sizeof(double complex*));
    for (int c=0;c<nchunks;c++){
        long chunkgals = gpc;
        if ((long)(c+1)*gpc > ncache){ chunkgals = ncache - (long)c*gpc; }
        Wncache[c]  = calloc(chunkgals*wn_per_gal, sizeof(double complex));
        W2ncache[c] = calloc(chunkgals*wn_per_gal, sizeof(double complex));
        W3ncache[c] = calloc(chunkgals*nbinsr, sizeof(double complex));
    }

    // Bin-center accumulators (reduced after Phase 1). The ratio counts/norms is the bin
    // center; accumulating over all resolutions counts each pair exactly once.
    double *tmp_totcounts = calloc(nthreads*nbinsr, sizeof(double));
    double *tmp_totnorms  = calloc(nthreads*nbinsr, sizeof(double));

    #pragma omp parallel for num_threads(nthreads)
    for (long ic=0; ic<ncache; ic++){
        int thisthread = omp_get_thread_num();
        int ind_inpix1 = (int) ic;
        int ind_gal = pix_gals[ind_inpix1];
        double p11 = pos1[ind_gal];
        double p12 = pos2[ind_gal];
        double w1 = weight[ind_gal];
        double innergal = isinner[ind_gal];
        if (innergal<1e-5){continue;}

        int chunk = (int)(ic/gpc);
        int loc = (int)(ic - (long)chunk*gpc);
        double complex *nextWns  = Wncache[chunk]  + (long)loc*wn_per_gal;
        double complex *nextW2ns = W2ncache[chunk] + (long)loc*wn_per_gal;
        double complex *nextW3ns = W3ncache[chunk] + (long)loc*nbinsr;

        int ind_gal2, ind_red, lower, upper;
        double p21, p22, w2, w2_sq, rel1, rel2, dist, dphi;
        double complex phirot, phirotc, nphirot, nphirotc;
        // Loop over ALL resolutions / radial bins (cache is a superset of any batch's needs)
        for (int elreso=0;elreso<nresos;elreso++){
            int rbin, zrshift, nextnshift, ind_Wn;
            double rmin_reso = reso_redges[elreso];
            double rmax_reso = reso_redges[elreso+1];
            int pix1_lower = mymax(0, (int) floor((p11 - (rmax_reso+pix1_d) - pix1_start)/pix1_d));
            int pix2_lower = mymax(0, (int) floor((p12 - (rmax_reso+pix2_d) - pix2_start)/pix2_d));
            int pix1_upper = mymin(pix1_n-1, (int) floor((p11 + (rmax_reso+pix1_d) - pix1_start)/pix1_d));
            int pix2_upper = mymin(pix2_n-1, (int) floor((p12 + (rmax_reso+pix2_d) - pix2_start)/pix2_d));
            for (int ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                for (int ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                    ind_red = index_matcher_hash[rshift_index_matcher_hash[elreso] + ind_pix2*pix1_n + ind_pix1];
                    if (ind_red==-1){continue;}
                    lower = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+ind_red];
                    upper = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+ind_red+1];
                    for (int ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                        ind_gal2 = rshift_pix_gals[elreso] + pix_gals[rshift_pix_gals[elreso]+ind_inpix];
                        p21 = pos1_resos[ind_gal2];
                        p22 = pos2_resos[ind_gal2];
                        w2 = weight_resos[ind_gal2];
                        rel1 = p21 - p11;
                        rel2 = p22 - p12;
                        dist = sqrt(rel1*rel1 + rel2*rel2);
                        if(dist < rmin_reso || dist >= rmax_reso) continue;
                        rbin = (int) floor((log(dist)-log(rmin))/drbin);
                        w2_sq = w2*w2;
                        dphi = atan2(rel2,rel1);
                        phirot = cexp(I*dphi);
                        phirotc = conj(phirot);
                        zrshift = 0*nbinsr + rbin;
                        ind_Wn = nzero_Wn*nbinszr + zrshift;
                        nphirot = 1+I*0;
                        nphirotc = 1+I*0;
                        // Triple-counting corr
                        nextW3ns[zrshift] += w2_sq*w2;
                        // Nominal G and double-counting corr, n = 0
                        tmp_totcounts[thisthread*nbinsr+zrshift] += w1*w2*dist;
                        tmp_totnorms[thisthread*nbinsr+zrshift]  += w1*w2;
                        nextWns[ind_Wn]  += w2*nphirot;
                        nextW2ns[ind_Wn] += w2_sq*nphirot;
                        nphirot *= phirot;
                        nphirotc *= phirotc;
                        // n in [1, ..., 2*nmax_alloc] x {+1,-1}
                        nextnshift = 0;
                        for (int nextn=1;nextn<=2*nmax_alloc;nextn++){
                            nextnshift = nextn*nbinszr;
                            nextWns[ind_Wn+nextnshift]  += w2*nphirot;
                            nextWns[ind_Wn-nextnshift]  += w2*nphirotc;
                            nextW2ns[ind_Wn+nextnshift] += w2_sq*nphirot;
                            nextW2ns[ind_Wn-nextnshift] += w2_sq*nphirotc;
                            nphirot *= phirot;
                            nphirotc *= phirotc;
                        }
                    }
                }
            }
        }
    }

    // Reduce bin centers
    double *totcounts = calloc(nbinsr, sizeof(double));
    double *totnorms  = calloc(nbinsr, sizeof(double));
    for (int t=0;t<nthreads;t++){
        for (int b=0;b<nbinsr;b++){
            totcounts[b] += tmp_totcounts[t*nbinsr+b];
            totnorms[b]  += tmp_totnorms[t*nbinsr+b];
        }
    }
    for (int b=0;b<nbinsr;b++){ if (totnorms[b]!=0){ bin_centers[b] = totcounts[b]/totnorms[b]; } }

    ////////////////////////////////////////////////////
    // Phase 2: combination + multipole->Map^4 transform
    ////////////////////////////////////////////////////
    double complex *allN4correlators = calloc(nthreads*nnapradii, sizeof(double complex));
    #pragma omp parallel for num_threads(nthreads)
    for (int elthetbatch=0; elthetbatch<nthetbatches; elthetbatch++){
        int thisthread = omp_get_thread_num();
        int batch_nthetas = nthetacombis_batches[elthetbatch];
        int batchN_nshift = batch_nthetas;
        long batchN_compshift = (long)n2n3combis*batchN_nshift;
        int batchgamma_thetshift = nbinsphi*nbinsphi;

        double complex *batchN_n = calloc(batchN_compshift, sizeof(double complex));
        int *elb1s_batch = calloc(batch_nthetas, sizeof(int));
        int *elb2s_batch = calloc(batch_nthetas, sizeof(int));
        int *elb3s_batch = calloc(batch_nthetas, sizeof(int));
        double *bin_centers_batch = calloc(nbinsr, sizeof(double));
        for (int b=0;b<nbinsr;b++){ bin_centers_batch[b] = bin_centers[b]; }
        for (int elb=0;elb<batch_nthetas;elb++){
            int thisrcombi = thetacombis_batches[cumthetacombis_batches[elthetbatch]+elb];
            elb1s_batch[elb] = thisrcombi/(nbinsr*nbinsr);
            elb2s_batch[elb] = (thisrcombi-elb1s_batch[elb]*nbinsr*nbinsr)/nbinsr;
            elb3s_batch[elb] = thisrcombi-elb1s_batch[elb]*nbinsr*nbinsr-elb2s_batch[elb]*nbinsr;
        }

        // Combination: accumulate the multipole 4PCF for this batch by reading cached moments
        for (long ic=0; ic<ncache; ic++){
            int ind_inpix1 = (int) ic;
            int ind_gal = pix_gals[ind_inpix1];
            double w1 = weight[ind_gal];
            double innergal = isinner[ind_gal];
            if (innergal<1e-5){continue;}
            int chunk = (int)(ic/gpc);
            int loc = (int)(ic - (long)chunk*gpc);
            double complex *nextWns  = Wncache[chunk]  + (long)loc*wn_per_gal;
            double complex *nextW2ns = W2ncache[chunk] + (long)loc*wn_per_gal;
            double complex *nextW3ns = W3ncache[chunk] + (long)loc*nbinsr;

            double complex wNN;
            int thisn2, thisn3, thisn, thisnshift, thisnrshift, elb1, elb2, elb3;
            int thisWshift_n2, thisWshift_n3, thisWshift_n2n3;
            for (int nindex=0; nindex<len_nindices; nindex++){
                thisn2 = nindices[nindex]/nnvals_Nn - nzero_Nn;
                thisn3 = nindices[nindex]%nnvals_Nn - nzero_Nn;
                if (thisn2>nzero_Nn || -thisn2>nzero_Nn || thisn3>nzero_Nn || -thisn3>nzero_Nn){ continue; }
                thisn = thisn2+thisn3;
                thisWshift_n2 = (nzero_Wn+thisn2)*nbinsr;
                thisWshift_n3 = (nzero_Wn+thisn3)*nbinsr;
                thisWshift_n2n3 = (nzero_Wn+thisn)*nbinsr;
                thisnshift = ((thisn2+nzero_Nn)*nnvals_Nn + (thisn3+nzero_Nn)) * batchN_nshift;
                for (int elb=0;elb<batch_nthetas;elb++){
                    elb1 = elb1s_batch[elb];
                    elb2 = elb2s_batch[elb];
                    elb3 = elb3s_batch[elb];
                    thisnrshift = thisnshift + elb;
                    // Multiple counting corrections:
                    // sum_(i neq j neq k) = sum_(i,j,k) - ( sum_(i, j, i=k) + 2perm ) + 2 * sum_(i, i=j, i=k)
                    if ((elb1==elb2) && (elb1==elb3) && (elb2==elb3)){
                        batchN_n[thisnrshift] += 2 * w1*nextW3ns[elb1];
                    }
                    if (elb1==elb2){
                        batchN_n[thisnrshift] -= w1 *
                            nextW2ns[(nzero_Wn+thisn3)*nbinsr+elb1]*conj(nextWns[thisWshift_n3+elb3]);
                    }
                    if (elb1==elb3){
                        batchN_n[thisnrshift] -= w1 *
                            nextW2ns[(nzero_Wn+thisn2)*nbinsr+elb1] * conj(nextWns[thisWshift_n2+elb2]);
                    }
                    if (elb2==elb3){
                        batchN_n[thisnrshift] -= w1 *
                            nextW2ns[(nzero_Wn-thisn2-thisn3)*nbinsr+elb2] * nextWns[thisWshift_n2n3+elb1];
                    }
                    // Nominal allocation
                    wNN = w1*nextWns[thisWshift_n2n3+elb1]*conj(nextWns[thisWshift_n2+elb2]);
                    batchN_n[thisnrshift] += wNN*conj(nextWns[thisWshift_n3+elb3]);
                }
            }
        }

        // For each theta combination: reconstruct via symmetries, transform to Map^4
        int ntrafos;
        double complex *nextN4correlators = calloc(1, sizeof(double complex));
        double complex *thisN_n = calloc(n2n3combis, sizeof(double complex));
        double complex *thisN_n_rec = calloc(n2n3combis_rec, sizeof(double complex));
        double complex *thisnpcf = calloc(batchgamma_thetshift, sizeof(double complex));
        for (int elb=0;elb<batch_nthetas;elb++){
            int nbshift, elb1, elb2, elb3, elb1t, elb2t, elb3t;
            elb1 = elb1s_batch[elb];
            elb2 = elb2s_batch[elb];
            elb3 = elb3s_batch[elb];
            int bincombi_trafos[6][3] = {{elb1,elb2,elb3}, {elb2,elb3,elb1}, {elb3,elb1,elb2},
                                         {elb1,elb3,elb2}, {elb2,elb1,elb3}, {elb3,elb2,elb1}};
            if ((elb1==elb2)&&(elb1==elb3)){ntrafos=1;}
            else if ((elb1==elb2)&&(elb1!=elb3)){ntrafos=3;}
            else if ((elb1==elb3)&&(elb1!=elb2)){ntrafos=3;}
            else if ((elb2==elb3)&&(elb2!=elb1)){ntrafos=3;}
            else{ntrafos=6;}
            for (int eltrafo=0;eltrafo<ntrafos;eltrafo++){
                elb1t = bincombi_trafos[eltrafo][0];
                elb2t = bincombi_trafos[eltrafo][1];
                elb3t = bincombi_trafos[eltrafo][2];
                for(int eln12=0;eln12<n2n3combis;eln12++){
                    nbshift = eln12*batchN_nshift+elb;
                    thisN_n[eln12] = batchN_n[nbshift];
                }
                getMultipolesFromSymm_NNNN(thisN_n, nmax, eltrafo, nindices, len_nindices, thisN_n_rec);
                // OPTIONAL: Allocate 4PCF in multipole basis
                for(int eln12=0;eln12<n2n3combis_rec;eln12++){
                    if (alloc_4pcfmultipoles==1){
                        int thisnrshift = eln12*N_nshift + elb1t*nbinsr*nbinsr + elb2t*nbinsr + elb3t;
                        N_n[thisnrshift] = thisN_n_rec[eln12];
                    }
                }
                multipoles2npcf_nnnn_singletheta(thisN_n_rec, nmax, nmax,
                                                 elb1t, elb2t, elb3t,
                                                 phibins, phibins, nbinsphi, nbinsphi,
                                                 thisnpcf);
                // OPTIONAL: Allocate 4pcf in real basis
                if (alloc_4pcfreal==1){
                    for (int elphi12=0;elphi12<batchgamma_thetshift;elphi12++){
                        int gamma_rshift = nbinsphi*nbinsphi;
                        int gamma_phircombi = gamma_rshift*(elb1t*nbinsr*nbinsr+elb2t*nbinsr+elb3t)+elphi12;
                        Counts[gamma_phircombi] = thisnpcf[elphi12];
                    }
                }
                // Update the aperture Map^4 integral
                double y1, y2, y3, dy1, dy2, dy3;
                int nap4ind;
                int nap4threadshift = thisthread*nnapradii;
                for (int elnapr=0; elnapr<nnapradii; elnapr++){
                    y1=bin_centers_batch[elb1t]/napradii[elnapr];
                    y2=bin_centers_batch[elb2t]/napradii[elnapr];
                    y3=bin_centers_batch[elb3t]/napradii[elnapr];
                    dy1 = (bin_edges[elb1t+1]-bin_edges[elb1t])/napradii[elnapr];
                    dy2 = (bin_edges[elb2t+1]-bin_edges[elb2t])/napradii[elnapr];
                    dy3 = (bin_edges[elb3t+1]-bin_edges[elb3t])/napradii[elnapr];
                    fourpcf2N4correlators(1,
                                          y1, y2, y3, dy1, dy2, dy3,
                                          phibins, phibins, dbinsphi, dbinsphi, nbinsphi, nbinsphi,
                                          thisnpcf, nextN4correlators);
                    nap4ind = elnapr;
                    if (isnan(cabs(nextN4correlators[0]))==false){
                        allN4correlators[nap4threadshift+nap4ind] += nextN4correlators[0];
                    }
                    nextN4correlators[0] = 0;
                }
                for(int i=0;i<batchgamma_thetshift;i++){ thisnpcf[i] = 0; }
                for(int i=0;i<n2n3combis;i++){ thisN_n[i] = 0; }
                for(int i=0;i<n2n3combis_rec;i++){ thisN_n_rec[i] = 0; }
            }
        }

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
    for (int elthread=0;elthread<nthreads;elthread++){
        int nap4threadshift = elthread*nnapradii;
        for (int elnapr=0; elnapr<nnapradii; elnapr++){
            N4correlators[elnapr] += allN4correlators[nap4threadshift+elnapr];
        }
    }
    free(allN4correlators);

    for (int c=0;c<nchunks;c++){ free(Wncache[c]); free(W2ncache[c]); free(W3ncache[c]); }
    free(Wncache); free(W2ncache); free(W3ncache);
    free(tmp_totcounts); free(tmp_totnorms); free(totcounts); free(totnorms);
    free(rshift_index_matcher_hash); free(rshift_pixs_galind_bounds); free(rshift_pix_gals);
    free(bin_edges);
}


//////////////////////////////////
// TREE 4PCF multipoles only     //
//////////////////////////////////
// Multipoles-only estimator with the same acceleration as
// alloc_notomoNap4_doubletree_nnnn: each discrete galaxy's single-leg moments
// (Wn/W2n/W3n) are computed exactly once into a cache (Phase 1); the multipole 4PCF
// is then formed per theta-batch from the cache and the routine STOPS right after the
// multipole reconstruction (getMultipolesFromSymm_NNNN) -- no real-space transform,
// no Map^4 integral. Exact w.r.t. the Tree (this caching is just a faster tree, not a
// true double tree: the central vertex is not gridded).
//
// Memory: the cache dominates at 2*ncache*nnvals_Wn*nbinsr complex doubles
// (ncache = pixs_galind_bounds[nregions]), split into <1e9-element chunks so that no
// allocation / int32 index reaches the 2e9 wall.
void alloc_nnnn_tree(
    double *isinner, double *weight, double *pos1, double *pos2, int ngal,
    int nmax, double rmin, double rmax, int nbinsr, int dccorr,
    int *nindices, int len_nindices,
    int nresos, double *reso_redges, int *ngal_resos,
    double *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos,
    int *index_matcher_hash, int *pixs_galind_bounds, int *pix_gals, int nregions,
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, double memory_bound, double *bin_centers, double complex *N_n){

    int nmax_alloc = 2*nmax+1;
    int nbinsz = 1;
    int nnvals_Wn = 4*nmax_alloc+1;
    int nnvals_Nn = 2*nmax_alloc+1;
    int nnvals_Nn_rec = 2*nmax+1;
    int nzero_Wn = 2*nmax_alloc;
    int nzero_Nn = nmax_alloc;
    int N_nshift = nbinsr*nbinsr*nbinsr;
    int n2n3combis = nnvals_Nn*nnvals_Nn;
    int n2n3combis_rec = nnvals_Nn_rec*nnvals_Nn_rec;
    int nbinszr = nbinsz*nbinsr;
    double drbin = (log(rmax)-log(rmin))/(nbinsr);
    int npix_hash = pix1_n*pix2_n;

    int *rshift_index_matcher_hash = calloc(nresos, sizeof(int));
    int *rshift_pixs_galind_bounds = calloc(nresos, sizeof(int));
    int *rshift_pix_gals = calloc(nresos, sizeof(int));
    for (int elreso=1;elreso<nresos;elreso++){
        rshift_index_matcher_hash[elreso] = rshift_index_matcher_hash[elreso-1] + npix_hash;
        rshift_pixs_galind_bounds[elreso] = rshift_pixs_galind_bounds[elreso-1] + ngal_resos[elreso-1]+1;
        rshift_pix_gals[elreso] = rshift_pix_gals[elreso-1] + ngal_resos[elreso-1];
    }

    ///////////////////////////////////////////////////////////////////
    // Plan the galaxy-blocking so the moment cache stays <= memory_bound
    ///////////////////////////////////////////////////////////////////
    // The cache holds the per-galaxy moments (Wn, W2n, W3n) for one block of galaxies.
    // We process the catalogue in n_iter blocks; each galaxy lands in exactly one block
    // so its moments are still computed once. memory_bound (GiB) caps the cache; we
    // reserve room for the per-thread Phase-2 buffers (which scale with nthreads).
    //
    // Only INNER galaxies act as central vertices (border galaxies are used solely as
    // neighbours). We build a compact list of their catalogue indices so the cache and
    // the blocking are sized by the inner count, not by the (border-inflated) total --
    // critical when patchextend is large, where most galaxies are border.
    long ngal_all = (long) pixs_galind_bounds[nregions];
    int *centralinds = malloc((ngal_all>0?ngal_all:1)*sizeof(int));
    long ncache = 0;
    for (long ig=0; ig<ngal_all; ig++){
        if (isinner[pix_gals[(int)ig]] >= 1e-5){ centralinds[ncache++] = (int)ig; }
    }
    centralinds = realloc(centralinds, (ncache>0?ncache:1)*sizeof(int));
    long wn_per_gal = (long) nnvals_Wn*nbinsr;
    long bytes_per_gal = (2*wn_per_gal + nbinsr) * (long)sizeof(double complex);
    int max_batch = 0;
    for (int b=0;b<nthetbatches;b++){ if (nthetacombis_batches[b]>max_batch){ max_batch = nthetacombis_batches[b]; } }
    long phase2_per_thread = ((long)max_batch*n2n3combis + n2n3combis + n2n3combis_rec) * (long)sizeof(double complex);
    long reserve = (long)nthreads * phase2_per_thread;
    long gals_per_iter;
    if (memory_bound <= 0){
        gals_per_iter = ncache; // unbounded
    } else {
        long budget = (long)(memory_bound * 1073741824.0); // GiB -> bytes
        long avail = budget - reserve;
        if (avail < bytes_per_gal){ avail = bytes_per_gal; } // at least one galaxy
        gals_per_iter = avail / bytes_per_gal;
    }
    if (gals_per_iter > ncache){ gals_per_iter = ncache; }
    if (gals_per_iter < 1){ gals_per_iter = 1; }
    int n_iter = (int)((ncache + gals_per_iter - 1)/gals_per_iter);

    // Within a block, split into int32-safe chunks (< 1e9 elements each)
    int gpc = (int)(1000000000L / wn_per_gal); if (gpc<1){gpc=1;}
    if ((long)gpc > gals_per_iter){ gpc = (int)gals_per_iter; }
    int nchunks = (int)((gals_per_iter + (long)gpc - 1)/(long)gpc);

    // Allocate the cache once (sized for one block); reused across iterations.
    double complex **Wncache  = malloc(nchunks*sizeof(double complex*));
    double complex **W2ncache = malloc(nchunks*sizeof(double complex*));
    double complex **W3ncache = malloc(nchunks*sizeof(double complex*));
    for (int c=0;c<nchunks;c++){
        long chunkgals = gpc;
        if ((long)(c+1)*gpc > gals_per_iter){ chunkgals = gals_per_iter - (long)c*gpc; }
        Wncache[c]  = calloc(chunkgals*wn_per_gal, sizeof(double complex));
        W2ncache[c] = calloc(chunkgals*wn_per_gal, sizeof(double complex));
        W3ncache[c] = calloc(chunkgals*nbinsr, sizeof(double complex));
        if (Wncache[c]==NULL || W2ncache[c]==NULL || W3ncache[c]==NULL){
            fprintf(stderr, "alloc_nnnn_tree: FAILED to allocate moment cache chunk %d/%d "
                    "(~%.1f GiB/chunk, memory_bound=%.1f GiB). Lower memory_bound or the number "
                    "of parallel workers.\n", c, nchunks,
                    chunkgals*bytes_per_gal/1073741824.0, memory_bound);
            exit(1);
        }
    }
    printf("alloc_nnnn_tree: %ld inner / %ld total galaxies, %d block(s) of <=%ld gal (cache ~%.1f GiB), %d chunk(s)/block\n",
           ncache, ngal_all, n_iter, gals_per_iter, gals_per_iter*bytes_per_gal/1073741824.0, nchunks);

    double *tmp_totcounts = calloc(nthreads*nbinsr, sizeof(double));
    double *tmp_totnorms  = calloc(nthreads*nbinsr, sizeof(double));

    // Outer loop over galaxy blocks. N_n is accumulated across blocks (the multipole
    // reconstruction getMultipolesFromSymm_NNNN is linear, so summing per-block
    // reconstructions equals reconstructing the full sum). N_n starts zeroed.
    for (int it=0; it<n_iter; it++){
        long g0 = (long)it*gals_per_iter;
        long g1 = g0 + gals_per_iter; if (g1>ncache){ g1 = ncache; }
        long nblock = g1 - g0;

        //////////////////////////////
        // Phase 1: build moment cache
        //////////////////////////////
        #pragma omp parallel for num_threads(nthreads)
        for (long il=0; il<nblock; il++){
            int thisthread = omp_get_thread_num();
            int ind_inpix1 = centralinds[g0 + il];
            int ind_gal = pix_gals[ind_inpix1];
            double p11 = pos1[ind_gal];
            double p12 = pos2[ind_gal];
            double w1 = weight[ind_gal];

            int chunk = (int)(il/gpc);
            int loc = (int)(il - (long)chunk*gpc);
            double complex *nextWns  = Wncache[chunk]  + (long)loc*wn_per_gal;
            double complex *nextW2ns = W2ncache[chunk] + (long)loc*wn_per_gal;
            double complex *nextW3ns = W3ncache[chunk] + (long)loc*nbinsr;
            // Reset this galaxy's cache slot (reused across blocks; kernel accumulates)
            for (long k=0;k<wn_per_gal;k++){ nextWns[k]=0; nextW2ns[k]=0; }
            for (int k=0;k<nbinsr;k++){ nextW3ns[k]=0; }

            int ind_gal2, ind_red, lower, upper;
            double p21, p22, w2, w2_sq, rel1, rel2, dist, dphi;
            double complex phirot, phirotc, nphirot, nphirotc;
            for (int elreso=0;elreso<nresos;elreso++){
                int rbin, zrshift, nextnshift, ind_Wn;
                double rmin_reso = reso_redges[elreso];
                double rmax_reso = reso_redges[elreso+1];
                int pix1_lower = mymax(0, (int) floor((p11 - (rmax_reso+pix1_d) - pix1_start)/pix1_d));
                int pix2_lower = mymax(0, (int) floor((p12 - (rmax_reso+pix2_d) - pix2_start)/pix2_d));
                int pix1_upper = mymin(pix1_n-1, (int) floor((p11 + (rmax_reso+pix1_d) - pix1_start)/pix1_d));
                int pix2_upper = mymin(pix2_n-1, (int) floor((p12 + (rmax_reso+pix2_d) - pix2_start)/pix2_d));
                for (int ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                    for (int ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                        ind_red = index_matcher_hash[rshift_index_matcher_hash[elreso] + ind_pix2*pix1_n + ind_pix1];
                        if (ind_red==-1){continue;}
                        lower = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+ind_red];
                        upper = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+ind_red+1];
                        for (int ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                            ind_gal2 = rshift_pix_gals[elreso] + pix_gals[rshift_pix_gals[elreso]+ind_inpix];
                            p21 = pos1_resos[ind_gal2];
                            p22 = pos2_resos[ind_gal2];
                            w2 = weight_resos[ind_gal2];
                            rel1 = p21 - p11;
                            rel2 = p22 - p12;
                            dist = sqrt(rel1*rel1 + rel2*rel2);
                            if(dist < rmin_reso || dist >= rmax_reso) continue;
                            rbin = (int) floor((log(dist)-log(rmin))/drbin);
                            w2_sq = w2*w2;
                            dphi = atan2(rel2,rel1);
                            phirot = cexp(I*dphi);
                            phirotc = conj(phirot);
                            zrshift = 0*nbinsr + rbin;
                            ind_Wn = nzero_Wn*nbinszr + zrshift;
                            nphirot = 1+I*0;
                            nphirotc = 1+I*0;
                            nextW3ns[zrshift] += w2_sq*w2;
                            tmp_totcounts[thisthread*nbinsr+zrshift] += w1*w2*dist;
                            tmp_totnorms[thisthread*nbinsr+zrshift]  += w1*w2;
                            nextWns[ind_Wn] += w2*nphirot;
                            nextW2ns[ind_Wn] += w2_sq*nphirot;
                            nphirot *= phirot;
                            nphirotc *= phirotc;
                            nextnshift = 0;
                            for (int nextn=1;nextn<=2*nmax_alloc;nextn++){
                                nextnshift = nextn*nbinszr;
                                nextWns[ind_Wn+nextnshift] += w2*nphirot;
                                nextWns[ind_Wn-nextnshift] += w2*nphirotc;
                                nextW2ns[ind_Wn+nextnshift] += w2_sq*nphirot;
                                nextW2ns[ind_Wn-nextnshift] += w2_sq*nphirotc;
                                nphirot *= phirot;
                                nphirotc *= phirotc;
                            }
                        }
                    }
                }
            }
        }

        /////////////////////////////////////////////////////////
        // Phase 2: combination + multipole reconstruction (stop)
        /////////////////////////////////////////////////////////
        #pragma omp parallel for num_threads(nthreads)
        for (int elthetbatch=0; elthetbatch<nthetbatches; elthetbatch++){
            int batch_nthetas = nthetacombis_batches[elthetbatch];
            int batchN_nshift = batch_nthetas;
            long batchN_compshift = (long)n2n3combis*batchN_nshift;
            double complex *batchN_n = calloc(batchN_compshift, sizeof(double complex));
            int *elb1s_batch = calloc(batch_nthetas, sizeof(int));
            int *elb2s_batch = calloc(batch_nthetas, sizeof(int));
            int *elb3s_batch = calloc(batch_nthetas, sizeof(int));
            for (int elb=0;elb<batch_nthetas;elb++){
                int thisrcombi = thetacombis_batches[cumthetacombis_batches[elthetbatch]+elb];
                elb1s_batch[elb] = thisrcombi/(nbinsr*nbinsr);
                elb2s_batch[elb] = (thisrcombi-elb1s_batch[elb]*nbinsr*nbinsr)/nbinsr;
                elb3s_batch[elb] = thisrcombi-elb1s_batch[elb]*nbinsr*nbinsr-elb2s_batch[elb]*nbinsr;
            }

            // Combination: read this block's cached moments
            for (long il=0; il<nblock; il++){
                int ind_inpix1 = centralinds[g0 + il];
                int ind_gal = pix_gals[ind_inpix1];
                double w1 = weight[ind_gal];
                int chunk = (int)(il/gpc);
                int loc = (int)(il - (long)chunk*gpc);
                double complex *nextWns  = Wncache[chunk]  + (long)loc*wn_per_gal;
                double complex *nextW2ns = W2ncache[chunk] + (long)loc*wn_per_gal;
                double complex *nextW3ns = W3ncache[chunk] + (long)loc*nbinsr;

                double complex wNN;
                int thisn2, thisn3, thisn, thisnshift, thisnrshift, elb1, elb2, elb3;
                int thisWshift_n2, thisWshift_n3, thisWshift_n2n3;
                for (int nindex=0; nindex<len_nindices; nindex++){
                    thisn2 = nindices[nindex]/nnvals_Nn - nzero_Nn;
                    thisn3 = nindices[nindex]%nnvals_Nn - nzero_Nn;
                    if (thisn2>nzero_Nn || -thisn2>nzero_Nn || thisn3>nzero_Nn || -thisn3>nzero_Nn){ continue; }
                    thisn = thisn2+thisn3;
                    thisWshift_n2 = (nzero_Wn+thisn2)*nbinsr;
                    thisWshift_n3 = (nzero_Wn+thisn3)*nbinsr;
                    thisWshift_n2n3 = (nzero_Wn+thisn)*nbinsr;
                    thisnshift = ((thisn2+nzero_Nn)*nnvals_Nn + (thisn3+nzero_Nn)) * batchN_nshift;
                    for (int elb=0;elb<batch_nthetas;elb++){
                        elb1 = elb1s_batch[elb];
                        elb2 = elb2s_batch[elb];
                        elb3 = elb3s_batch[elb];
                        thisnrshift = thisnshift + elb;
                        if ((elb1==elb2) && (elb1==elb3) && (elb2==elb3)){
                            batchN_n[thisnrshift] += 2 * w1*nextW3ns[elb1];
                        }
                        if (elb1==elb2){
                            batchN_n[thisnrshift] -= w1 *
                                nextW2ns[(nzero_Wn+thisn3)*nbinsr+elb1]*conj(nextWns[thisWshift_n3+elb3]);
                        }
                        if (elb1==elb3){
                            batchN_n[thisnrshift] -= w1 *
                                nextW2ns[(nzero_Wn+thisn2)*nbinsr+elb1] * conj(nextWns[thisWshift_n2+elb2]);
                        }
                        if (elb2==elb3){
                            batchN_n[thisnrshift] -= w1 *
                                nextW2ns[(nzero_Wn-thisn2-thisn3)*nbinsr+elb2] * nextWns[thisWshift_n2n3+elb1];
                        }
                        wNN = w1*nextWns[thisWshift_n2n3+elb1]*conj(nextWns[thisWshift_n2+elb2]);
                        batchN_n[thisnrshift] += wNN*conj(nextWns[thisWshift_n3+elb3]);
                    }
                }
            }

            // Reconstruct via symmetries and accumulate into N_n (STOP -- multipoles only)
            double complex *thisN_n = calloc(n2n3combis, sizeof(double complex));
            double complex *thisN_n_rec = calloc(n2n3combis_rec, sizeof(double complex));
            for (int elb=0;elb<batch_nthetas;elb++){
                int nbshift, elb1, elb2, elb3, elb1t, elb2t, elb3t, ntrafos;
                elb1 = elb1s_batch[elb];
                elb2 = elb2s_batch[elb];
                elb3 = elb3s_batch[elb];
                int bincombi_trafos[6][3] = {{elb1,elb2,elb3}, {elb2,elb3,elb1}, {elb3,elb1,elb2},
                                             {elb1,elb3,elb2}, {elb2,elb1,elb3}, {elb3,elb2,elb1}};
                if ((elb1==elb2)&&(elb1==elb3)){ntrafos=1;}
                else if ((elb1==elb2)&&(elb1!=elb3)){ntrafos=3;}
                else if ((elb1==elb3)&&(elb1!=elb2)){ntrafos=3;}
                else if ((elb2==elb3)&&(elb2!=elb1)){ntrafos=3;}
                else{ntrafos=6;}
                for (int eltrafo=0;eltrafo<ntrafos;eltrafo++){
                    elb1t = bincombi_trafos[eltrafo][0];
                    elb2t = bincombi_trafos[eltrafo][1];
                    elb3t = bincombi_trafos[eltrafo][2];
                    for(int eln12=0;eln12<n2n3combis;eln12++){
                        nbshift = eln12*batchN_nshift+elb;
                        thisN_n[eln12] = batchN_n[nbshift];
                    }
                    getMultipolesFromSymm_NNNN(thisN_n, nmax, eltrafo, nindices, len_nindices, thisN_n_rec);
                    for(int eln12=0;eln12<n2n3combis_rec;eln12++){
                        int thisnrshift = eln12*N_nshift + elb1t*nbinsr*nbinsr + elb2t*nbinsr + elb3t;
                        N_n[thisnrshift] += thisN_n_rec[eln12];
                    }
                    for(int i=0;i<n2n3combis;i++){ thisN_n[i] = 0; }
                    for(int i=0;i<n2n3combis_rec;i++){ thisN_n_rec[i] = 0; }
                }
            }

            free(batchN_n);
            free(elb1s_batch);
            free(elb2s_batch);
            free(elb3s_batch);
            free(thisN_n);
            free(thisN_n_rec);
        }
    }

    double *totcounts = calloc(nbinsr, sizeof(double));
    double *totnorms  = calloc(nbinsr, sizeof(double));
    for (int t=0;t<nthreads;t++){
        for (int b=0;b<nbinsr;b++){
            totcounts[b] += tmp_totcounts[t*nbinsr+b];
            totnorms[b]  += tmp_totnorms[t*nbinsr+b];
        }
    }
    for (int b=0;b<nbinsr;b++){ if (totnorms[b]!=0){ bin_centers[b] = totcounts[b]/totnorms[b]; } }

    for (int c=0;c<nchunks;c++){ free(Wncache[c]); free(W2ncache[c]); free(W3ncache[c]); }
    free(Wncache); free(W2ncache); free(W3ncache);
    free(tmp_totcounts); free(tmp_totnorms); free(totcounts); free(totnorms);
    free(rshift_index_matcher_hash); free(rshift_pixs_galind_bounds); free(rshift_pix_gals);
    free(centralinds);
}


////////////////////////////////////////
// CURVED-SKY (full-sky) TREE: NNNN     //
////////////////////////////////////////
// Full-sky scalar 4PCF multipoles. Identical in *structure* to alloc_nnnn_tree
// (the flat oracle) -- same moment cache, same Phase-2 combination and symmetry
// reconstruction -- and differs only in the three local geometry kernels:
//   (1) leg navigation: a precomputed nested-HEALPix neighbour CSR (built by
//       Catalog.multihash_spherical) replaces the flat pixel-box double loop;
//   (2) separation:  geodesic sphere_dist() replaces sqrt(rel1^2+rel2^2);
//   (3) apex phase:  sphere_bearing() replaces atan2(rel2,rel1).
// A scalar field has no spin-2 parallel transport, so no holonomy phase enters.
// All separations are in RADIANS here (rmin/rmax/reso_redges and bin_centers);
// the caller converts to/from the catalogue's angular unit. See
// Tutorials_private/fullsky_covariance_notes.md (sections 1-2).
//
// Leg catalogues (per band, concatenated, offset by rshift_leg) carry both the
// unit vector (vx,vy,vz; for the distance dot/cross) and (ra,sindec,cosdec; for
// the bearing). Centrals (apexes) are the full-resolution inner galaxies.
void alloc_nnnn_tree_spherical(
    double *cen_isinner, double *cen_w,
    double *cen_vx, double *cen_vy, double *cen_vz,
    double *cen_ra, double *cen_sindec, double *cen_cosdec, int ngal,
    int nmax, double rmin, double rmax, int nbinsr, int dccorr,
    int *nindices, int len_nindices,
    int nresos, double *reso_redges, int *ngal_resos, int *ncells_resos,
    long *nside_nav,
    double *red_w, double *red_vx, double *red_vy, double *red_vz,
    double *red_ra, double *red_sindec, double *red_cosdec, int *rshift_red,
    long *cell_pix, int *cell_redbounds, int *rshift_cellpix, int *rshift_cellbounds,
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, double memory_bound, double *bin_centers, double complex *N_n){

    int nmax_alloc = 2*nmax+1;
    int nbinsz = 1;
    int nnvals_Wn = 4*nmax_alloc+1;
    int nnvals_Nn = 2*nmax_alloc+1;
    int nnvals_Nn_rec = 2*nmax+1;
    int nzero_Wn = 2*nmax_alloc;
    int nzero_Nn = nmax_alloc;
    int N_nshift = nbinsr*nbinsr*nbinsr;
    int n2n3combis = nnvals_Nn*nnvals_Nn;
    int n2n3combis_rec = nnvals_Nn_rec*nnvals_Nn_rec;
    int nbinszr = nbinsz*nbinsr;
    double drbin = (log(rmax)-log(rmin))/(nbinsr);

    ///////////////////////////////////////////////////////////////////
    // Plan the galaxy-blocking so the moment cache stays <= memory_bound
    ///////////////////////////////////////////////////////////////////
    // Centrals are the inner galaxies directly; neighbours per band are found on the
    // fly with query_disc over the reduced-catalogue nested-HEALPix hash (see Phase 1).
    int *centralinds = malloc((ngal>0?ngal:1)*sizeof(int));
    long ncache = 0;
    for (int ig=0; ig<ngal; ig++){
        if (cen_isinner[ig] >= 1e-5){ centralinds[ncache++] = ig; }
    }
    centralinds = realloc(centralinds, (ncache>0?ncache:1)*sizeof(int));
    long wn_per_gal = (long) nnvals_Wn*nbinsr;
    long bytes_per_gal = (2*wn_per_gal + nbinsr) * (long)sizeof(double complex);
    int max_batch = 0;
    for (int b=0;b<nthetbatches;b++){ if (nthetacombis_batches[b]>max_batch){ max_batch = nthetacombis_batches[b]; } }
    long phase2_per_thread = ((long)max_batch*n2n3combis + n2n3combis + n2n3combis_rec) * (long)sizeof(double complex);
    long reserve = (long)nthreads * phase2_per_thread;
    long gals_per_iter;
    if (memory_bound <= 0){
        gals_per_iter = ncache;
    } else {
        long budget = (long)(memory_bound * 1073741824.0);
        long avail = budget - reserve;
        if (avail < bytes_per_gal){ avail = bytes_per_gal; }
        gals_per_iter = avail / bytes_per_gal;
    }
    if (gals_per_iter > ncache){ gals_per_iter = ncache; }
    if (gals_per_iter < 1){ gals_per_iter = 1; }
    int n_iter = (int)((ncache + gals_per_iter - 1)/gals_per_iter);

    int gpc = (int)(1000000000L / wn_per_gal); if (gpc<1){gpc=1;}
    if ((long)gpc > gals_per_iter){ gpc = (int)gals_per_iter; }
    int nchunks = (int)((gals_per_iter + (long)gpc - 1)/(long)gpc);

    double complex **Wncache  = malloc(nchunks*sizeof(double complex*));
    double complex **W2ncache = malloc(nchunks*sizeof(double complex*));
    double complex **W3ncache = malloc(nchunks*sizeof(double complex*));
    for (int c=0;c<nchunks;c++){
        long chunkgals = gpc;
        if ((long)(c+1)*gpc > gals_per_iter){ chunkgals = gals_per_iter - (long)c*gpc; }
        Wncache[c]  = calloc(chunkgals*wn_per_gal, sizeof(double complex));
        W2ncache[c] = calloc(chunkgals*wn_per_gal, sizeof(double complex));
        W3ncache[c] = calloc(chunkgals*nbinsr, sizeof(double complex));
        if (Wncache[c]==NULL || W2ncache[c]==NULL || W3ncache[c]==NULL){
            fprintf(stderr, "alloc_nnnn_tree_spherical: FAILED to allocate moment cache chunk %d/%d "
                    "(~%.1f GiB/chunk, memory_bound=%.1f GiB).\n", c, nchunks,
                    chunkgals*bytes_per_gal/1073741824.0, memory_bound);
            exit(1);
        }
    }
    printf("alloc_nnnn_tree_spherical: %ld inner / %d total galaxies, %d block(s) of <=%ld gal (cache ~%.1f GiB), %d chunk(s)/block\n",
           ncache, ngal, n_iter, gals_per_iter, gals_per_iter*bytes_per_gal/1073741824.0, nchunks);

    double *tmp_totcounts = calloc(nthreads*nbinsr, sizeof(double));
    double *tmp_totnorms  = calloc(nthreads*nbinsr, sizeof(double));

    for (int it=0; it<n_iter; it++){
        long g0 = (long)it*gals_per_iter;
        long g1 = g0 + gals_per_iter; if (g1>ncache){ g1 = ncache; }
        long nblock = g1 - g0;

        //////////////////////////////
        // Phase 1: build moment cache
        //////////////////////////////
        #pragma omp parallel for num_threads(nthreads)
        for (long il=0; il<nblock; il++){
            int thisthread = omp_get_thread_num();
            int ic = centralinds[g0 + il];        // central (apex) catalogue index
            double cx = cen_vx[ic], cy = cen_vy[ic], cz = cen_vz[ic];
            double cra = cen_ra[ic], csdec = cen_sindec[ic], ccdec = cen_cosdec[ic];

            int chunk = (int)(il/gpc);
            int loc = (int)(il - (long)chunk*gpc);
            double complex *nextWns  = Wncache[chunk]  + (long)loc*wn_per_gal;
            double complex *nextW2ns = W2ncache[chunk] + (long)loc*wn_per_gal;
            double complex *nextW3ns = W3ncache[chunk] + (long)loc*nbinsr;
            for (long k=0;k<wn_per_gal;k++){ nextWns[k]=0; nextW2ns[k]=0; }
            for (int k=0;k<nbinsr;k++){ nextW3ns[k]=0; }

            double w2, w2_sq, dist, dphi;
            double complex phirot, phirotc, nphirot, nphirotc;
            long qcap = 2048;
            long *ranges = malloc(2*qcap*sizeof(long));
            double v1[3] = {cx, cy, cz};
            for (int elreso=0;elreso<nresos;elreso++){
                int rbin, zrshift, nextnshift, ind_Wn;
                double rmin_reso = reso_redges[elreso];
                double rmax_reso = reso_redges[elreso+1];
                // Curved-sky single-tree navigation: query_disc at this band's nav nside,
                // merge-join the occupied cells against cell_pix, iterate their reduced
                // galaxies. Inclusive query + the distance cut below make this exact and
                // robust to nav coarsening (nside_nav may be coarser than the reduction).
                long ns_nav_r = nside_nav[elreso];
                long nr = hpx_query_disc_nest_ranges(ns_nav_r, v1, rmax_reso, ranges, qcap);
                if (nr > qcap){ qcap = nr; ranges = realloc(ranges, 2*qcap*sizeof(long));
                                nr = hpx_query_disc_nest_ranges(ns_nav_r, v1, rmax_reso, ranges, qcap); }
                const long *cp = cell_pix + rshift_cellpix[elreso];
                const int  *cb = cell_redbounds + rshift_cellbounds[elreso];
                int ncells_r = ncells_resos[elreso];
                long red_off = rshift_red[elreso];
                int ci = 0;
                for (long qr=0; qr<nr; qr++){
                    long plo = ranges[2*qr], phr = ranges[2*qr+1];
                    int loi = ci, hii = ncells_r;
                    while (loi < hii){ int m=(loi+hii)>>1; if (cp[m] < plo){ loi=m+1; } else { hii=m; } }
                    ci = loi;
                    while (ci < ncells_r && cp[ci] < phr){
                        int clo = cb[ci], chi = cb[ci+1];
                        for (int j=clo; j<chi; j++){
                            long lg = red_off + j;
                            dist = sphere_dist(cx, cy, cz, red_vx[lg], red_vy[lg], red_vz[lg]);
                            if (dist < rmin_reso || dist >= rmax_reso) continue;
                            w2 = red_w[lg];
                            rbin = (int) floor((log(dist)-log(rmin))/drbin);
                            w2_sq = w2*w2;
                            dphi = sphere_bearing(cra, csdec, ccdec,
                                                  red_ra[lg], red_sindec[lg], red_cosdec[lg]);
                            phirot = cexp(I*dphi);
                            phirotc = conj(phirot);
                            zrshift = 0*nbinsr + rbin;
                            ind_Wn = nzero_Wn*nbinszr + zrshift;
                            nphirot = 1+I*0;
                            nphirotc = 1+I*0;
                            nextW3ns[zrshift] += w2_sq*w2;
                            tmp_totcounts[thisthread*nbinsr+zrshift] += cen_w[ic]*w2*dist;
                            tmp_totnorms[thisthread*nbinsr+zrshift]  += cen_w[ic]*w2;
                            nextWns[ind_Wn] += w2*nphirot;
                            nextW2ns[ind_Wn] += w2_sq*nphirot;
                            nphirot *= phirot;
                            nphirotc *= phirotc;
                            nextnshift = 0;
                            for (int nextn=1;nextn<=2*nmax_alloc;nextn++){
                                nextnshift = nextn*nbinszr;
                                nextWns[ind_Wn+nextnshift] += w2*nphirot;
                                nextWns[ind_Wn-nextnshift] += w2*nphirotc;
                                nextW2ns[ind_Wn+nextnshift] += w2_sq*nphirot;
                                nextW2ns[ind_Wn-nextnshift] += w2_sq*nphirotc;
                                nphirot *= phirot;
                                nphirotc *= phirotc;
                            }
                        }
                        ci++;
                    }
                }
            }
            free(ranges);
        }

        /////////////////////////////////////////////////////////
        // Phase 2: combination + multipole reconstruction (stop)
        // (identical to alloc_nnnn_tree -- geometry-independent)
        /////////////////////////////////////////////////////////
        #pragma omp parallel for num_threads(nthreads)
        for (int elthetbatch=0; elthetbatch<nthetbatches; elthetbatch++){
            int batch_nthetas = nthetacombis_batches[elthetbatch];
            int batchN_nshift = batch_nthetas;
            long batchN_compshift = (long)n2n3combis*batchN_nshift;
            double complex *batchN_n = calloc(batchN_compshift, sizeof(double complex));
            int *elb1s_batch = calloc(batch_nthetas, sizeof(int));
            int *elb2s_batch = calloc(batch_nthetas, sizeof(int));
            int *elb3s_batch = calloc(batch_nthetas, sizeof(int));
            for (int elb=0;elb<batch_nthetas;elb++){
                int thisrcombi = thetacombis_batches[cumthetacombis_batches[elthetbatch]+elb];
                elb1s_batch[elb] = thisrcombi/(nbinsr*nbinsr);
                elb2s_batch[elb] = (thisrcombi-elb1s_batch[elb]*nbinsr*nbinsr)/nbinsr;
                elb3s_batch[elb] = thisrcombi-elb1s_batch[elb]*nbinsr*nbinsr-elb2s_batch[elb]*nbinsr;
            }

            for (long il=0; il<nblock; il++){
                int ic = centralinds[g0 + il];
                double w1 = cen_w[ic];
                int chunk = (int)(il/gpc);
                int loc = (int)(il - (long)chunk*gpc);
                double complex *nextWns  = Wncache[chunk]  + (long)loc*wn_per_gal;
                double complex *nextW2ns = W2ncache[chunk] + (long)loc*wn_per_gal;
                double complex *nextW3ns = W3ncache[chunk] + (long)loc*nbinsr;

                double complex wNN;
                int thisn2, thisn3, thisn, thisnshift, thisnrshift, elb1, elb2, elb3;
                int thisWshift_n2, thisWshift_n3, thisWshift_n2n3;
                for (int nindex=0; nindex<len_nindices; nindex++){
                    thisn2 = nindices[nindex]/nnvals_Nn - nzero_Nn;
                    thisn3 = nindices[nindex]%nnvals_Nn - nzero_Nn;
                    if (thisn2>nzero_Nn || -thisn2>nzero_Nn || thisn3>nzero_Nn || -thisn3>nzero_Nn){ continue; }
                    thisn = thisn2+thisn3;
                    thisWshift_n2 = (nzero_Wn+thisn2)*nbinsr;
                    thisWshift_n3 = (nzero_Wn+thisn3)*nbinsr;
                    thisWshift_n2n3 = (nzero_Wn+thisn)*nbinsr;
                    thisnshift = ((thisn2+nzero_Nn)*nnvals_Nn + (thisn3+nzero_Nn)) * batchN_nshift;
                    for (int elb=0;elb<batch_nthetas;elb++){
                        elb1 = elb1s_batch[elb];
                        elb2 = elb2s_batch[elb];
                        elb3 = elb3s_batch[elb];
                        thisnrshift = thisnshift + elb;
                        if ((elb1==elb2) && (elb1==elb3) && (elb2==elb3)){
                            batchN_n[thisnrshift] += 2 * w1*nextW3ns[elb1];
                        }
                        if (elb1==elb2){
                            batchN_n[thisnrshift] -= w1 *
                                nextW2ns[(nzero_Wn+thisn3)*nbinsr+elb1]*conj(nextWns[thisWshift_n3+elb3]);
                        }
                        if (elb1==elb3){
                            batchN_n[thisnrshift] -= w1 *
                                nextW2ns[(nzero_Wn+thisn2)*nbinsr+elb1] * conj(nextWns[thisWshift_n2+elb2]);
                        }
                        if (elb2==elb3){
                            batchN_n[thisnrshift] -= w1 *
                                nextW2ns[(nzero_Wn-thisn2-thisn3)*nbinsr+elb2] * nextWns[thisWshift_n2n3+elb1];
                        }
                        wNN = w1*nextWns[thisWshift_n2n3+elb1]*conj(nextWns[thisWshift_n2+elb2]);
                        batchN_n[thisnrshift] += wNN*conj(nextWns[thisWshift_n3+elb3]);
                    }
                }
            }

            double complex *thisN_n = calloc(n2n3combis, sizeof(double complex));
            double complex *thisN_n_rec = calloc(n2n3combis_rec, sizeof(double complex));
            for (int elb=0;elb<batch_nthetas;elb++){
                int nbshift, elb1, elb2, elb3, elb1t, elb2t, elb3t, ntrafos;
                elb1 = elb1s_batch[elb];
                elb2 = elb2s_batch[elb];
                elb3 = elb3s_batch[elb];
                int bincombi_trafos[6][3] = {{elb1,elb2,elb3}, {elb2,elb3,elb1}, {elb3,elb1,elb2},
                                             {elb1,elb3,elb2}, {elb2,elb1,elb3}, {elb3,elb2,elb1}};
                if ((elb1==elb2)&&(elb1==elb3)){ntrafos=1;}
                else if ((elb1==elb2)&&(elb1!=elb3)){ntrafos=3;}
                else if ((elb1==elb3)&&(elb1!=elb2)){ntrafos=3;}
                else if ((elb2==elb3)&&(elb2!=elb1)){ntrafos=3;}
                else{ntrafos=6;}
                for (int eltrafo=0;eltrafo<ntrafos;eltrafo++){
                    elb1t = bincombi_trafos[eltrafo][0];
                    elb2t = bincombi_trafos[eltrafo][1];
                    elb3t = bincombi_trafos[eltrafo][2];
                    for(int eln12=0;eln12<n2n3combis;eln12++){
                        nbshift = eln12*batchN_nshift+elb;
                        thisN_n[eln12] = batchN_n[nbshift];
                    }
                    getMultipolesFromSymm_NNNN(thisN_n, nmax, eltrafo, nindices, len_nindices, thisN_n_rec);
                    for(int eln12=0;eln12<n2n3combis_rec;eln12++){
                        int thisnrshift = eln12*N_nshift + elb1t*nbinsr*nbinsr + elb2t*nbinsr + elb3t;
                        N_n[thisnrshift] += thisN_n_rec[eln12];
                    }
                    for(int i=0;i<n2n3combis;i++){ thisN_n[i] = 0; }
                    for(int i=0;i<n2n3combis_rec;i++){ thisN_n_rec[i] = 0; }
                }
            }

            free(batchN_n);
            free(elb1s_batch);
            free(elb2s_batch);
            free(elb3s_batch);
            free(thisN_n);
            free(thisN_n_rec);
        }
    }

    double *totcounts = calloc(nbinsr, sizeof(double));
    double *totnorms  = calloc(nbinsr, sizeof(double));
    for (int t=0;t<nthreads;t++){
        for (int b=0;b<nbinsr;b++){
            totcounts[b] += tmp_totcounts[t*nbinsr+b];
            totnorms[b]  += tmp_totnorms[t*nbinsr+b];
        }
    }
    for (int b=0;b<nbinsr;b++){ if (totnorms[b]!=0){ bin_centers[b] = totcounts[b]/totnorms[b]; } }

    for (int c=0;c<nchunks;c++){ free(Wncache[c]); free(W2ncache[c]); free(W3ncache[c]); }
    free(Wncache); free(W2ncache); free(W3ncache);
    free(tmp_totcounts); free(tmp_totnorms); free(totcounts); free(totnorms);
    free(centralinds);
}


//////////////////////////////////
// DOUBLE TREE 4PCF multipoles    //
//////////////////////////////////
// Genuine double tree (App. F): the central vertex is gridded too, via the reduced
// central catalogues at each resolution band. For NNNN the central and legs are the same
// (number-count) field, so the reso catalogues (`*_resos`) serve as both. Validated against
// alloc_nnnn_tree (the exact oracle); see claude_optimisations.txt for the full design log.
//
// One region-parallel loop over filled regions (schedule(dynamic)); each thread owns a
// private N_n copy (reduced at the end; nthreads_cross capped by memory_bound). Per region:
//   A. per band: scan that band's reduced centrals at the leaf resolution -> per-central
//      Wn/W2n/W3n; aggregate Wn,W2n into the region cell caches; (discrete band) also store
//      the per-galaxy moments Wdisc/W2disc/W3disc.
//   B. SAME-BAND combination (all three legs one band): grid bands inline during the scan;
//      the discrete band via a nested (t1,t2,t3) batched combine that caches the two-leg
//      product P12 = w*W_{n2+n3}(t1)*conj(W_n2)(t2) and reuses it across t3 (branch-free
//      inner loop). Full W2n/W3n multiple-counting corrections.
//   C. CROSS-BAND combination (Eq. F.6 nesting): finest leg aggregated into the middle
//      band's cells, coarser legs as single cell reps; STREAMED over (n2,n3) to keep the
//      partial product at O(cells). e1==e2 / e2==e3 multiple-counting corrections included.
// Speedup over alloc_nnnn_tree grows with nbar (~sqrt(N) vs ~N); best for large-scale /
// few-discrete-bin configs. See the per-phase notes inline below.
void alloc_nnnn_doubletree(
    int nresos, int nresos_grid, double *dpix1_resos, double *dpix2_resos, double *reso_redges,
    int resoshift_leafs, int minresoind_leaf, int maxresoind_leaf,
    double *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos, int *ngal_resos,
    int nmax, double rmin, double rmax, int nbinsr, int dccorr,
    int *nindices, int len_nindices,
    int *index_matcher_hash, int *index_matcher_full, int *pixs_galind_bounds, int *pix_gals,
    int *filledregions, int nfilledregions, int nregions,
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, double memory_bound, int verbose, double *bin_centers, double complex *N_n){

    int nmax_alloc = 2*nmax+1;
    int nbinsz = 1;
    int nnvals_Wn = 4*nmax_alloc+1;
    int nnvals_Nn = 2*nmax_alloc+1;
    int nnvals_Nn_rec = 2*nmax+1;
    int nzero_Wn = 2*nmax_alloc;
    int nzero_Nn = nmax_alloc;
    int N_nshift = nbinsr*nbinsr*nbinsr;
    int n2n3combis = nnvals_Nn*nnvals_Nn;
    int n2n3combis_rec = nnvals_Nn_rec*nnvals_Nn_rec;
    int nbinszr = nbinsz*nbinsr;
    double drbin = (log(rmax)-log(rmin))/(nbinsr);
    double logrmin = log(rmin);
    int npix_hash = pix1_n*pix2_n;

    // Per-band offsets into the concatenated reso catalogues
    int *rshift_index_matcher_hash = calloc(nresos, sizeof(int));
    int *rshift_pixs_galind_bounds = calloc(nresos, sizeof(int));
    int *rshift_pix_gals = calloc(nresos, sizeof(int));
    for (int elreso=1;elreso<nresos;elreso++){
        rshift_index_matcher_hash[elreso] = rshift_index_matcher_hash[elreso-1] + npix_hash;
        rshift_pixs_galind_bounds[elreso] = rshift_pixs_galind_bounds[elreso-1] + ngal_resos[elreso-1]+1;
        rshift_pix_gals[elreso] = rshift_pix_gals[elreso-1] + ngal_resos[elreso-1];
    }

    // Map each resolution band to its range of radial bins (port of the 3pt reso_rindedges)
    int *reso_rindedges = calloc(nresos+1, sizeof(int));
    {
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
    if (verbose){
        printf("alloc_nnnn_doubletree: nresos=%d nbinsr=%d -- band radial-bin ranges:\n", nresos, nbinsr);
        for (int r=0;r<nresos;r++){
            int leaf = mymin(mymax(minresoind_leaf, r+resoshift_leafs), maxresoind_leaf);
            printf("  band %d: bins [%d,%d) leaf=%d\n", r, reso_rindedges[r], reso_rindedges[r+1], leaf);
        }
    }

    double *tmp_totcounts = calloc(nthreads*nbinsr, sizeof(double));
    double *tmp_totnorms  = calloc(nthreads*nbinsr, sizeof(double));

    // (same-band combination is now folded into the unified region-parallel loop below)

    ///////////////////////////////////////////////////////////////
    // CROSS-BAND triplets (two OR three distinct bands), Eq. F.6.
    // Region-parallel; region-local cell-aggregated cache; thread-
    // private N_n (reduced at the end). The combination NESTS the
    // legs (finest aggregated into the middle band's cells, coarser
    // legs as single cell reps) so no spurious inter-central cross-
    // terms appear; see the combination block below. Multiple-counting
    // corrections ARE included: e1==e2 (b1==b2<b3) is folded into the finer
    // partial product (-w*W2_n3(theta1)); e2==e3 (b1<b2==b3) is a boundary
    // term (-w*W_{n2+n3}(theta1)*W2_{-n2-n3}(theta2)); e1==e3/all-equal cannot
    // occur cross-band (would force all three legs into one band).
    ///////////////////////////////////////////////////////////////
    int hasdiscrete = nresos - nresos_grid;
    int *bin2band = calloc(nbinsr, sizeof(int));
    for (int r=0;r<nresos;r++){ for (int b=reso_rindedges[r]; b<reso_rindedges[r+1]; b++){ bin2band[b]=r; } }
    long ntheta_tot = (long) cumthetacombis_batches[nthetbatches];
    long Nn_size = (long)n2n3combis_rec * N_nshift;

    // Membership mask of the wanted ordered triplets (respects custom_thetacombis). The streamed
    // cross-band combination iterates (e1<=e2<=e3) directly and consults this mask.
    char *wanted = calloc((long)N_nshift, 1);
    for (long elt=0; elt<ntheta_tot; elt++){ wanted[thetacombis_batches[elt]] = 1; }

    // Within-band (same-band) wanted triplets, grouped by band -- for the inline same-band
    // combination in the unified region loop. sb_band_lo/hi[b] index into sb_e1/e2/e3.
    long n_sameband = 0;
    for (long elt=0; elt<ntheta_tot; elt++){
        int c = thetacombis_batches[elt];
        int e1=c/(nbinsr*nbinsr), e2=(c-e1*nbinsr*nbinsr)/nbinsr, e3=c-e1*nbinsr*nbinsr-e2*nbinsr;
        if (bin2band[e1]==bin2band[e2] && bin2band[e1]==bin2band[e3]){ n_sameband++; }
    }
    int *sb_e1 = malloc((n_sameband>0?n_sameband:1)*sizeof(int));
    int *sb_e2 = malloc((n_sameband>0?n_sameband:1)*sizeof(int));
    int *sb_e3 = malloc((n_sameband>0?n_sameband:1)*sizeof(int));
    int *sb_band_lo = calloc(nresos, sizeof(int));
    int *sb_band_hi = calloc(nresos, sizeof(int));
    {
        long k = 0;
        for (int b=0;b<nresos;b++){
            sb_band_lo[b] = (int)k;
            for (long elt=0; elt<ntheta_tot; elt++){
                int c = thetacombis_batches[elt];
                int e1=c/(nbinsr*nbinsr), e2=(c-e1*nbinsr*nbinsr)/nbinsr, e3=c-e1*nbinsr*nbinsr-e2*nbinsr;
                if (bin2band[e1]==b && bin2band[e2]==b && bin2band[e3]==b){ sb_e1[k]=e1; sb_e2[k]=e2; sb_e3[k]=e3; k++; }
            }
            sb_band_hi[b] = (int)k;
        }
    }

    // Worst-case per-region sizes, to bound the cross-band thread count by memory_bound. Each
    // cross-band thread holds: its N_n slice + the region cell cache (Wn/wWn) + the per-galaxy
    // discrete moment cache Wdisc + the streamed n-accumulator + small scratch. Streaming over
    // (n2,n3) is what keeps the partial product at O(cells) instead of O(n2n3combis*cells).
    long max_thetashift = 1; // max over regions of sum_{grid resos} ngal_in_pix
    long max_ndisc = 1;      // max over regions of discrete-band galaxies (ngal_in_pix[0])
    for (int fr=0; fr<nfilledregions; fr++){
        int elregion = filledregions[fr];
        long ts = 0;
        for (int r=hasdiscrete; r<nresos; r++){
            ts += pixs_galind_bounds[rshift_pixs_galind_bounds[r]+elregion+1]
                - pixs_galind_bounds[rshift_pixs_galind_bounds[r]+elregion];
        }
        if (ts > max_thetashift){ max_thetashift = ts; }
        if (hasdiscrete>=1){
            long nd = pixs_galind_bounds[rshift_pixs_galind_bounds[0]+elregion+1]
                    - pixs_galind_bounds[rshift_pixs_galind_bounds[0]+elregion];
            if (nd > max_ndisc){ max_ndisc = nd; }
        }
    }
    int ndb_disc = (hasdiscrete==1 && reso_rindedges[1] > reso_rindedges[0])
                 ? (reso_rindedges[1]-reso_rindedges[0]) : 0;
    long per_thread = Nn_size                                  // tmpN_n slice
        + 4L*nnvals_Wn*nbinsr*max_thetashift                  // Wncache + wWncache + W2ncache + wW2ncache
        + 2L*nnvals_Wn*ndb_disc*max_ndisc                     // Wdisc + W2disc (per-galaxy discrete)
        + (long)nbinsr*n2n3combis                             // streamed n-accumulator
        + (long)(n_sameband - sb_band_lo[hasdiscrete<nresos?hasdiscrete:nresos-1])*n2n3combis // grid same-band acc
        + (long)max_ndisc*n2n3combis                          // P12disc (discrete-discrete W12 cache)
        + 3L*nnvals_Wn*nbinsr;                                // scan scratch (Wn/W2n) + small
    per_thread *= 16;
    int nthreads_cross = nthreads;
    if (memory_bound > 0){
        long cap = (long)(memory_bound*1073741824.0) / per_thread;
        if (cap < 1){ cap = 1; }
        if (cap < nthreads_cross){ nthreads_cross = (int)cap; }
    }
    double complex *tmpN_n = calloc((long)nthreads_cross*Nn_size, sizeof(double complex));
    if (tmpN_n==NULL){
        fprintf(stderr,"alloc_nnnn_doubletree: cross-band N_n accumulator alloc failed "
                "(%d threads x %.2f GiB). Lower memory_bound/threads.\n",
                nthreads_cross, Nn_size*16/1073741824.0); exit(1);
    }
    if (verbose){
        printf("alloc_nnnn_doubletree: cross-band using %d/%d threads (%d filled regions) "
               "(per-thread est. %.2f GiB: N_n %.2f + cache %.2f + Wdisc %.2f)\n",
               nthreads_cross, nthreads, nfilledregions, per_thread/1073741824.0,
               Nn_size*16/1073741824.0, 2.0*nnvals_Wn*nbinsr*max_thetashift*16/1073741824.0,
               (double)nnvals_Wn*ndb_disc*max_ndisc*16/1073741824.0);
    }

    // per-thread wall accumulators to split cross-band into aggregation vs combination (CPU = sum)
    // per-(thread,band) timers: aperture scan, cell aggregation, same-band combine; plus
    // per-thread same-band reconstruct and cross-band combine. Summed over threads -> CPU.
    double *t_scan = calloc((long)nthreads_cross*nresos, sizeof(double));
    double *t_aggc = calloc((long)nthreads_cross*nresos, sizeof(double));
    double *t_sbc  = calloc((long)nthreads_cross*nresos, sizeof(double));
    double *t_sbrec = calloc(nthreads_cross, sizeof(double));
    double *t_xband = calloc(nthreads_cross, sizeof(double));
    double t_cb_wall0 = omp_get_wtime(); clock_t t_cb_cpu0 = clock();

    #pragma omp parallel for num_threads(nthreads_cross) schedule(dynamic)
    for (int fr=0; fr<nfilledregions; fr++){
        int thisthread = omp_get_thread_num();
        int elregion = filledregions[fr];
        double complex *myN_n = tmpN_n + (long)thisthread*Nn_size;

        // ---- region-local cell-cache setup ----
        int *ngal_in_pix = calloc(nresos, sizeof(int));
        for (int r=0;r<nresos;r++){
            ngal_in_pix[r] = pixs_galind_bounds[rshift_pixs_galind_bounds[r]+elregion+1]
                           - pixs_galind_bounds[rshift_pixs_galind_bounds[r]+elregion];
        }
        // grid-pixel matcher offsets (grid resolutions only)
        int *matchers_resoshift = calloc(nresos_grid+1, sizeof(int));
        for (int eg=0; eg<nresos_grid; eg++){
            int npix_side = 1 << (nresos_grid-eg-1);
            matchers_resoshift[eg+1] = matchers_resoshift[eg] + npix_side*npix_side;
        }
        int len_matcher = matchers_resoshift[nresos_grid];
        // cell offsets per gridding resolution (grid resolutions carry cells, discrete carries none)
        int *cumresoshift = calloc(nresos+1, sizeof(int));
        for (int r=0;r<nresos;r++){
            cumresoshift[r+1] = cumresoshift[r] + ((r>=hasdiscrete) ? ngal_in_pix[r] : 0);
        }
        long thetashift = cumresoshift[nresos];
        long nshift_cache = (long)nbinsr*thetashift;

        // pixel -> reduced-cell map for each grid resolution
        int elregion_fullhash = index_matcher_full[elregion];
        double hashpix_start1 = pix1_start + (elregion_fullhash%pix1_n)*pix1_d;
        double hashpix_start2 = pix2_start + (elregion_fullhash/pix1_n)*pix2_d;
        int *pix2redpix = calloc(len_matcher>0?len_matcher:1, sizeof(int));
        for (int eg=0; eg<nresos_grid; eg++){
            int r = eg + hasdiscrete;
            int npix_side = 1 << (nresos_grid-eg-1);
            int lo = pixs_galind_bounds[rshift_pixs_galind_bounds[r]+elregion];
            int hi = pixs_galind_bounds[rshift_pixs_galind_bounds[r]+elregion+1];
            int cnt = 0;
            for (int ip=lo; ip<hi; ip++){
                int ind_gal = rshift_pix_gals[r] + pix_gals[rshift_pix_gals[r]+ip];
                int eh1 = (int) floor((pos1_resos[ind_gal]-hashpix_start1)/dpix1_resos[eg]);
                int eh2 = (int) floor((pos2_resos[ind_gal]-hashpix_start2)/dpix2_resos[eg]);
                // Guard against boundary / non-power-of-2 grids (assumes 2x-spaced resos)
                if (eh1<0){eh1=0;} if (eh1>=npix_side){eh1=npix_side-1;}
                if (eh2<0){eh2=0;} if (eh2>=npix_side){eh2=npix_side-1;}
                pix2redpix[matchers_resoshift[eg] + eh2*npix_side + eh1] = cnt;
                cnt += 1;
            }
        }

        double complex *Wncache  = calloc(nnvals_Wn*nshift_cache, sizeof(double complex));
        double complex *wWncache = calloc(nnvals_Wn*nshift_cache, sizeof(double complex));
        // W2 cell caches for the cross-band multiple-counting corrections (small vs N_n):
        // W2ncache (unweighted, for e2==e3), wW2ncache (weighted, for e1==e2 grid).
        double complex *W2ncache  = calloc(nnvals_Wn*nshift_cache, sizeof(double complex));
        double complex *wW2ncache = calloc(nnvals_Wn*nshift_cache, sizeof(double complex));
        if (Wncache==NULL || wWncache==NULL || W2ncache==NULL || wW2ncache==NULL){
            fprintf(stderr,"alloc_nnnn_doubletree: cross-band cell cache alloc failed (region %d)\n", elregion); exit(1);
        }

        // Per-galaxy discrete-band moments (filled during the band-0 scan below): needed for the
        // b1==b2==discrete cross-band combination, where the two finer legs are un-griddable.
        int db_lo = (hasdiscrete>=1) ? reso_rindedges[0] : 0;
        int db_hi = (hasdiscrete>=1) ? reso_rindedges[1] : 0;
        int ndb = db_hi - db_lo;
        long ndisc = (hasdiscrete>=1) ? ngal_in_pix[0] : 0;
        double *disc_p1=NULL, *disc_p2=NULL, *disc_w=NULL;
        double complex *Wdisc=NULL, *W2disc=NULL, *W3disc=NULL;
        if (ndb>0 && ndisc>0){
            disc_p1 = calloc(ndisc, sizeof(double));
            disc_p2 = calloc(ndisc, sizeof(double));
            disc_w  = calloc(ndisc, sizeof(double));
            Wdisc   = calloc((long)ndisc*nnvals_Wn*ndb, sizeof(double complex));
            W2disc  = calloc((long)ndisc*nnvals_Wn*ndb, sizeof(double complex)); // for same-band corrections
            W3disc  = calloc((long)ndisc*ndb, sizeof(double complex));
            if (Wdisc==NULL || W2disc==NULL || W3disc==NULL){
                fprintf(stderr,"alloc_nnnn_doubletree: Wdisc alloc failed (region %d, %.2f GiB). "
                        "Lower memory_bound/nmax or widen tree_resos.\n",
                        elregion, (double)ndisc*nnvals_Wn*ndb*16/1073741824.0); exit(1);
            }
        }

        // ---- scan each central; same-band combine inline + aggregate into coarse cells ----
        // sameband_N holds only the GRID-band same-band triplets (per-central scatter, small);
        // discrete-band same-band is done by the cache-friendly batched pass after the scan.
        int gtrip0 = sb_band_lo[hasdiscrete<nresos?hasdiscrete:nresos-1]; // first grid-band triplet
        long n_gridtrip = n_sameband - gtrip0;
        double complex *nextWns  = calloc(nnvals_Wn*nbinsr, sizeof(double complex));
        double complex *nextW2ns = calloc(nnvals_Wn*nbinsr, sizeof(double complex));
        double complex *nextW3ns = calloc(nbinsr, sizeof(double complex));
        double complex *sameband_N = calloc(n_gridtrip>0?(long)n_gridtrip*n2n3combis:1, sizeof(double complex));
        for (int aband=0; aband<nresos; aband++){
            int rbinmin = reso_rindedges[aband], rbinmax = reso_rindedges[aband+1];
            if (rbinmax<=rbinmin){ continue; }
            int aleaf = mymin(mymax(minresoind_leaf, aband+resoshift_leafs), maxresoind_leaf);
            double rmin_a = rmin*exp(rbinmin*drbin), rmax_a = rmin*exp(rbinmax*drbin);
            int lo1 = pixs_galind_bounds[rshift_pixs_galind_bounds[aband]+elregion];
            int hi1 = pixs_galind_bounds[rshift_pixs_galind_bounds[aband]+elregion+1];
            for (int ip=lo1; ip<hi1; ip++){
                int ind_gal1 = rshift_pix_gals[aband] + pix_gals[rshift_pix_gals[aband]+ip];
                if (isinner_resos[ind_gal1] < 1e-5){ continue; }
                double p11 = pos1_resos[ind_gal1], p12 = pos2_resos[ind_gal1], w1 = weight_resos[ind_gal1];
                double tt0 = omp_get_wtime();
                for (int i=0;i<nnvals_Wn*nbinsr;i++){ nextWns[i]=0; nextW2ns[i]=0; }
                for (int b=rbinmin;b<rbinmax;b++){ nextW3ns[b]=0; }
                // band-a neighbour scan at the leaf resolution
                int pix1_lower = mymax(0, (int) floor((p11 - (rmax_a+pix1_d) - pix1_start)/pix1_d));
                int pix2_lower = mymax(0, (int) floor((p12 - (rmax_a+pix2_d) - pix2_start)/pix2_d));
                int pix1_upper = mymin(pix1_n-1, (int) floor((p11 + (rmax_a+pix1_d) - pix1_start)/pix1_d));
                int pix2_upper = mymin(pix2_n-1, (int) floor((p12 + (rmax_a+pix2_d) - pix2_start)/pix2_d));
                for (int ipx1=pix1_lower; ipx1<pix1_upper; ipx1++){
                    for (int ipx2=pix2_lower; ipx2<pix2_upper; ipx2++){
                        int ind_red = index_matcher_hash[rshift_index_matcher_hash[aleaf] + ipx2*pix1_n + ipx1];
                        if (ind_red==-1){continue;}
                        int lo2 = pixs_galind_bounds[rshift_pixs_galind_bounds[aleaf]+ind_red];
                        int hi2 = pixs_galind_bounds[rshift_pixs_galind_bounds[aleaf]+ind_red+1];
                        for (int ip2=lo2; ip2<hi2; ip2++){
                            int ind_gal2 = rshift_pix_gals[aleaf] + pix_gals[rshift_pix_gals[aleaf]+ip2];
                            double rel1 = pos1_resos[ind_gal2]-p11, rel2 = pos2_resos[ind_gal2]-p12;
                            double dist = sqrt(rel1*rel1+rel2*rel2);
                            if (dist<rmin_a || dist>=rmax_a){ continue; }
                            int rbin = (int) floor((log(dist)-logrmin)/drbin);
                            if (rbin<rbinmin || rbin>=rbinmax){ continue; }
                            double w2 = weight_resos[ind_gal2], w2_sq = w2*w2;
                            double dphi = atan2(rel2,rel1);
                            double complex phirot = cexp(I*dphi), phirotc = conj(phirot);
                            double complex nphirot = 1+I*0, nphirotc = 1+I*0;
                            int ind_Wn = nzero_Wn*nbinsr + rbin;
                            nextW3ns[rbin] += w2_sq*w2;
                            tmp_totcounts[thisthread*nbinsr+rbin] += w2*dist;
                            tmp_totnorms[thisthread*nbinsr+rbin]  += w2;
                            nextWns[ind_Wn]  += w2*nphirot;
                            nextW2ns[ind_Wn] += w2_sq*nphirot;
                            nphirot *= phirot; nphirotc *= phirotc;
                            for (int nextn=1;nextn<=2*nmax_alloc;nextn++){
                                nextWns[ind_Wn+nextn*nbinsr]  += w2*nphirot;
                                nextWns[ind_Wn-nextn*nbinsr]  += w2*nphirotc;
                                nextW2ns[ind_Wn+nextn*nbinsr] += w2_sq*nphirot;
                                nextW2ns[ind_Wn-nextn*nbinsr] += w2_sq*nphirotc;
                                nphirot *= phirot; nphirotc *= phirotc;
                            }
                        }
                    }
                }
                double tt1 = omp_get_wtime(); t_scan[(long)thisthread*nresos+aband] += tt1-tt0;
                if (aband < hasdiscrete && Wdisc!=NULL){
                    long il = ip - lo1; // local discrete-galaxy index
                    disc_p1[il]=p11; disc_p2[il]=p12; disc_w[il]=w1;
                    for (int n=0;n<nnvals_Wn;n++){
                        for (int b=db_lo;b<db_hi;b++){
                            Wdisc[((long)il*nnvals_Wn + n)*ndb + (b-db_lo)]  = nextWns[n*nbinsr+b];
                            W2disc[((long)il*nnvals_Wn + n)*ndb + (b-db_lo)] = nextW2ns[n*nbinsr+b];
                        }
                    }
                    for (int b=db_lo;b<db_hi;b++){ W3disc[(long)il*ndb + (b-db_lo)] = nextW3ns[b]; }
                }
                // aggregate into the cell cache at every coarser gridding resolution
                for (int r=mymax(aband,hasdiscrete); r<nresos; r++){
                    int eg = r - hasdiscrete;
                    int npix_side = 1 << (nresos_grid-eg-1);
                    int eh1 = (int) floor((p11-hashpix_start1)/dpix1_resos[eg]);
                    int eh2 = (int) floor((p12-hashpix_start2)/dpix2_resos[eg]);
                    if (eh1<0){eh1=0;} if (eh1>=npix_side){eh1=npix_side-1;}
                    if (eh2<0){eh2=0;} if (eh2>=npix_side){eh2=npix_side-1;}
                    int cell = pix2redpix[matchers_resoshift[eg] + eh2*npix_side + eh1];
                    for (int n=0;n<nnvals_Wn;n++){
                        long base = (long)n*nshift_cache + cumresoshift[r] + cell;
                        for (int b=rbinmin;b<rbinmax;b++){
                            double complex v = nextWns[n*nbinsr+b], v2 = nextW2ns[n*nbinsr+b];
                            Wncache[base + (long)b*thetashift]   += v;
                            wWncache[base + (long)b*thetashift]  += w1*v;
                            W2ncache[base + (long)b*thetashift]  += v2;
                            wW2ncache[base + (long)b*thetashift] += w1*v2;
                        }
                    }
                } 
                double tt2 = omp_get_wtime(); t_aggc[(long)thisthread*nresos+aband] += tt2-tt1;
                // ---- same-band combination for GRID bands (per-central scatter; small) ----
                // The discrete band is handled by the cache-friendly batched pass after the scan
                // (its huge per-central scatter into sameband_N is the bottleneck we are avoiding).
                if (aband >= hasdiscrete){
                  for (int k=sb_band_lo[aband]; k<sb_band_hi[aband]; k++){
                    int e1=sb_e1[k], e2=sb_e2[k], e3=sb_e3[k];
                    long sbase = (long)(k-gtrip0)*n2n3combis;
                    int c12=(e1==e2), c13=(e1==e3), c23=(e2==e3); // coincidence pattern (hoisted)
                    if (!c12 && !c13 && !c23){
                        for (int nindex=0; nindex<len_nindices; nindex++){
                            int thisn2 = nindices[nindex]/nnvals_Nn - nzero_Nn;
                            int thisn3 = nindices[nindex]%nnvals_Nn - nzero_Nn;
                            if (thisn2>nzero_Nn || -thisn2>nzero_Nn || thisn3>nzero_Nn || -thisn3>nzero_Nn){ continue; }
                            int thisn = thisn2+thisn3;
                            sameband_N[sbase + (thisn2+nzero_Nn)*nnvals_Nn + (thisn3+nzero_Nn)] +=
                                w1*nextWns[(nzero_Wn+thisn)*nbinsr+e1]*conj(nextWns[(nzero_Wn+thisn2)*nbinsr+e2])*conj(nextWns[(nzero_Wn+thisn3)*nbinsr+e3]);
                        }
                    } else {
                        for (int nindex=0; nindex<len_nindices; nindex++){
                            int thisn2 = nindices[nindex]/nnvals_Nn - nzero_Nn;
                            int thisn3 = nindices[nindex]%nnvals_Nn - nzero_Nn;
                            if (thisn2>nzero_Nn || -thisn2>nzero_Nn || thisn3>nzero_Nn || -thisn3>nzero_Nn){ continue; }
                            int thisn = thisn2+thisn3;
                            int wsh_n2=(nzero_Wn+thisn2)*nbinsr, wsh_n3=(nzero_Wn+thisn3)*nbinsr, wsh_n=(nzero_Wn+thisn)*nbinsr;
                            double complex acc = w1*nextWns[wsh_n+e1]*conj(nextWns[wsh_n2+e2])*conj(nextWns[wsh_n3+e3]);
                            if (c12&&c13){ acc += 2*w1*nextW3ns[e1]; }
                            if (c12){ acc -= w1*nextW2ns[(nzero_Wn+thisn3)*nbinsr+e1]*conj(nextWns[wsh_n3+e3]); }
                            if (c13){ acc -= w1*nextW2ns[(nzero_Wn+thisn2)*nbinsr+e1]*conj(nextWns[wsh_n2+e2]); }
                            if (c23){ acc -= w1*nextW2ns[(nzero_Wn-thisn2-thisn3)*nbinsr+e2]*nextWns[wsh_n+e1]; }
                            sameband_N[sbase + (thisn2+nzero_Nn)*nnvals_Nn + (thisn3+nzero_Nn)] += acc;
                        }
                    }
                  }
                }
                t_sbc[(long)thisthread*nresos+aband] += omp_get_wtime()-tt2;
            }
        }
        free(nextWns); free(nextW2ns); free(nextW3ns);

        // ---- parent maps: grid cell at finer reso eg2 -> grid cell at coarser reso eg3 ----
        // Built from the eg2 reduced catalogue (one rep per eg2 cell); used to nest the
        // F.6 combination (sum finer-band cells into their coarser-band parent cell).
        int **parent = calloc((long)nresos_grid*nresos_grid, sizeof(int*));
        for (int eg2=0; eg2<nresos_grid; eg2++){
            int r2 = eg2 + hasdiscrete;
            int nps2 = 1 << (nresos_grid-eg2-1);
            int plo = pixs_galind_bounds[rshift_pixs_galind_bounds[r2]+elregion];
            int phi = pixs_galind_bounds[rshift_pixs_galind_bounds[r2]+elregion+1];
            for (int eg3=eg2+1; eg3<nresos_grid; eg3++){
                int nps3 = 1 << (nresos_grid-eg3-1);
                int *map = calloc(ngal_in_pix[r2]>0?ngal_in_pix[r2]:1, sizeof(int));
                for (int ip=plo; ip<phi; ip++){
                    int ind_gal = rshift_pix_gals[r2] + pix_gals[rshift_pix_gals[r2]+ip];
                    double pp1 = pos1_resos[ind_gal], pp2 = pos2_resos[ind_gal];
                    int a1 = (int) floor((pp1-hashpix_start1)/dpix1_resos[eg2]);
                    int a2 = (int) floor((pp2-hashpix_start2)/dpix2_resos[eg2]);
                    if (a1<0){a1=0;} if (a1>=nps2){a1=nps2-1;}
                    if (a2<0){a2=0;} if (a2>=nps2){a2=nps2-1;}
                    int C2 = pix2redpix[matchers_resoshift[eg2] + a2*nps2 + a1];
                    int d1 = (int) floor((pp1-hashpix_start1)/dpix1_resos[eg3]);
                    int d2 = (int) floor((pp2-hashpix_start2)/dpix2_resos[eg3]);
                    if (d1<0){d1=0;} if (d1>=nps3){d1=nps3-1;}
                    if (d2<0){d2=0;} if (d2>=nps3){d2=nps3-1;}
                    map[C2] = pix2redpix[matchers_resoshift[eg3] + d2*nps3 + d1];
                }
                parent[eg2*nresos_grid+eg3] = map;
            }
        }
        int maxcells = 1;
        for (int r=0;r<nresos;r++){ if (ngal_in_pix[r]>maxcells){ maxcells = ngal_in_pix[r]; } }

        // Precompute each discrete galaxy's cell at every grid resolution (used by the streamed
        // discrete combination; avoids recomputing the floor/clamp per (n2,n3)).
        int *disc_c3_all = NULL;
        if (Wdisc!=NULL){
            disc_c3_all = malloc((long)nresos_grid*ndisc*sizeof(int));
            for (int eg=0; eg<nresos_grid; eg++){
                int nps = 1 << (nresos_grid-eg-1);
                for (long il=0; il<ndisc; il++){
                    int a1 = (int) floor((disc_p1[il]-hashpix_start1)/dpix1_resos[eg]);
                    int a2 = (int) floor((disc_p2[il]-hashpix_start2)/dpix2_resos[eg]);
                    if (a1<0){a1=0;} if (a1>=nps){a1=nps-1;}
                    if (a2<0){a2=0;} if (a2>=nps){a2=nps-1;}
                    disc_c3_all[(long)eg*ndisc+il] = pix2redpix[matchers_resoshift[eg] + a2*nps + a1];
                }
            }
        }

        // ---- combine cross-band ordered triplets via the Eq. F.6 nesting (streamed over n) ----
        // Outer loop over the two finer legs (theta1<=theta2). For each (theta1,theta2,b3) the
        // partial product is built ONE (n2,n3) at a time -- streaming over n keeps it at
        // O(cells) instead of O(n2n3combis*cells) -- and accumulated into accum[e3][n2,n3];
        // the n-stream finishes, then each triplet is reconstructed. The partial product is
        // reused across all theta3 in band b3 (the F.6 reuse):
        //   Dsmall[C3] = sum_{Cf->C3} conj(W_n2(theta2,Cf)) * [sum_{x in Cf} w_x W_{n2+n3}(theta1,x)]
        //   N[n2,n3]  += sum_{C3} conj(W_n3(theta3,C3)) * Dsmall[C3]
        // Grid finer legs read the cell cache at b2-cells; two discrete finer legs
        // (b1==b2==discrete) use the per-galaxy Wdisc cache built during the band-0 scan.
        double complex *thisN_n = calloc(n2n3combis, sizeof(double complex));
        double complex *thisN_n_rec = calloc(n2n3combis_rec, sizeof(double complex));
        double complex *Dsmall = calloc(maxcells, sizeof(double complex));               // per-(n2,n3) cell partial
        double complex *accum = calloc((long)nbinsr*n2n3combis, sizeof(double complex));  // [e3][n2,n3] over the n-stream
        // discrete-discrete cross-band: cache the per-galaxy product P12disc[ncombi][il] once per
        // (theta1,theta2) and reuse across all coarse bands b3 (W12 reuse), layout ncombi-major.
        double complex *P12disc = (Wdisc!=NULL) ? malloc((long)ndisc*n2n3combis*sizeof(double complex)) : NULL;

        // ---- discrete-band same-band: nested (theta1,theta2,theta3) combine with W12 reuse ----
        // For each (theta1<=theta2) cache P12[n2,n3] = w*W_{n2+n3}(theta1)*conj(W_n2)(theta2)
        // (folding the e1==e2 correction into it) and reuse it across all theta3 -- the nominal
        // theta3 loop is then branch-free. The e2==e3 / e1==e3 / triple corrections only touch
        // the boundary theta3==theta2, so no multiple-counting checks live in the inner loop.
        if (hasdiscrete==1 && Wdisc!=NULL && db_hi>db_lo){
            double t_db0 = omp_get_wtime();
            int *t3list = malloc(ndb*sizeof(int));
            double complex *batchN = calloc((long)ndb*n2n3combis, sizeof(double complex));
            double complex *P12 = malloc((long)n2n3combis*sizeof(double complex));
            for (int t1=db_lo; t1<db_hi; t1++){
                for (int t2=t1; t2<db_hi; t2++){
                    int nt3=0;
                    for (int t3=t2; t3<db_hi; t3++){
                        if (wanted[t1*nbinsr*nbinsr + t2*nbinsr + t3]){ t3list[nt3++]=t3; }
                    }
                    if (nt3==0){ continue; }
                    for (long i=0;i<(long)nt3*n2n3combis;i++){ batchN[i]=0; }
                    int e1d=t1-db_lo, e2d=t2-db_lo, c12=(t1==t2);
                    int do_corr = (t3list[0]==t2); // the boundary triplet theta3==theta2 exists
                    for (long il=0; il<ndisc; il++){
                        double w1 = disc_w[il];
                        if (w1==0){ continue; }
                        const double complex *Wx  = Wdisc  + (long)il*nnvals_Wn*ndb;
                        const double complex *W2x = W2disc + (long)il*nnvals_Wn*ndb;
                        const double complex *W3x = W3disc + (long)il*ndb;
                        // P12 (cached, reused across theta3); folds the e1==e2 correction when t1==t2
                        for (int nindex=0; nindex<len_nindices; nindex++){
                            int thisn2 = nindices[nindex]/nnvals_Nn - nzero_Nn;
                            int thisn3 = nindices[nindex]%nnvals_Nn - nzero_Nn;
                            if (thisn2>nzero_Nn || -thisn2>nzero_Nn || thisn3>nzero_Nn || -thisn3>nzero_Nn){ continue; }
                            int thisn = thisn2+thisn3;
                            double complex v = w1*Wx[(nzero_Wn+thisn)*ndb+e1d]*conj(Wx[(nzero_Wn+thisn2)*ndb+e2d]);
                            if (c12){ v -= w1*W2x[(nzero_Wn+thisn3)*ndb+e1d]; }
                            P12[(thisn2+nzero_Nn)*nnvals_Nn+(thisn3+nzero_Nn)] = v;
                        }
                        // nominal over theta3 -- branch-free inner loop
                        for (int jj=0; jj<nt3; jj++){
                            int e3d = t3list[jj]-db_lo;
                            long bbase = (long)jj*n2n3combis;
                            for (int nindex=0; nindex<len_nindices; nindex++){
                                int thisn2 = nindices[nindex]/nnvals_Nn - nzero_Nn;
                                int thisn3 = nindices[nindex]%nnvals_Nn - nzero_Nn;
                                if (thisn2>nzero_Nn || -thisn2>nzero_Nn || thisn3>nzero_Nn || -thisn3>nzero_Nn){ continue; }
                                int ncombi = (thisn2+nzero_Nn)*nnvals_Nn+(thisn3+nzero_Nn);
                                batchN[bbase+ncombi] += P12[ncombi]*conj(Wx[(nzero_Wn+thisn3)*ndb+e3d]);
                            }
                        }
                        // corrections only at theta3==theta2 (jj==0): e2==e3, plus all-equal if t1==t2
                        if (do_corr){
                            for (int nindex=0; nindex<len_nindices; nindex++){
                                int thisn2 = nindices[nindex]/nnvals_Nn - nzero_Nn;
                                int thisn3 = nindices[nindex]%nnvals_Nn - nzero_Nn;
                                if (thisn2>nzero_Nn || -thisn2>nzero_Nn || thisn3>nzero_Nn || -thisn3>nzero_Nn){ continue; }
                                int thisn = thisn2+thisn3;
                                int ncombi = (thisn2+nzero_Nn)*nnvals_Nn+(thisn3+nzero_Nn);
                                double complex corr = - w1*W2x[(nzero_Wn-thisn)*ndb+e2d]*Wx[(nzero_Wn+thisn)*ndb+e1d]; // e2==e3
                                if (c12){
                                    corr -= w1*W2x[(nzero_Wn+thisn2)*ndb+e1d]*conj(Wx[(nzero_Wn+thisn2)*ndb+e2d]);    // e1==e3 (all equal)
                                    corr += 2*w1*W3x[e1d];                                                              // triple
                                }
                                batchN[ncombi] += corr;
                            }
                        }
                    }
                    for (int jj=0; jj<nt3; jj++){
                        int e1=t1, e2=t2, e3=t3list[jj];
                        for (int i=0;i<n2n3combis;i++){ thisN_n[i] = batchN[(long)jj*n2n3combis+i]; }
                        int tr[6][3] = {{e1,e2,e3},{e2,e3,e1},{e3,e1,e2},{e1,e3,e2},{e2,e1,e3},{e3,e2,e1}};
                        int ntrafos;
                        if ((e1==e2)&&(e1==e3)){ntrafos=1;}
                        else if ((e1==e2)&&(e1!=e3)){ntrafos=3;}
                        else if ((e1==e3)&&(e1!=e2)){ntrafos=3;}
                        else if ((e2==e3)&&(e2!=e1)){ntrafos=3;}
                        else{ntrafos=6;}
                        for (int t=0;t<ntrafos;t++){
                            int e1t=tr[t][0], e2t=tr[t][1], e3t=tr[t][2];
                            getMultipolesFromSymm_NNNN(thisN_n, nmax, t, nindices, len_nindices, thisN_n_rec);
                            for(int kk=0;kk<n2n3combis_rec;kk++){
                                myN_n[(long)kk*N_nshift + e1t*nbinsr*nbinsr + e2t*nbinsr + e3t] += thisN_n_rec[kk];
                            }
                            for(int kk=0;kk<n2n3combis_rec;kk++){ thisN_n_rec[kk]=0; }
                        }
                    }
                }
            }
            free(t3list); free(batchN); free(P12);
            t_sbc[(long)thisthread*nresos+0] += omp_get_wtime()-t_db0;
        }

        // ---- reconstruct the GRID-band same-band triplets (per-central scatter) into N_n ----
        double t_sbrec0 = omp_get_wtime();
        for (long k=gtrip0;k<n_sameband;k++){
            int e1=sb_e1[k], e2=sb_e2[k], e3=sb_e3[k];
            for (int i=0;i<n2n3combis;i++){ thisN_n[i] = sameband_N[(long)(k-gtrip0)*n2n3combis+i]; }
            int tr[6][3] = {{e1,e2,e3},{e2,e3,e1},{e3,e1,e2},{e1,e3,e2},{e2,e1,e3},{e3,e2,e1}};
            int ntrafos;
            if ((e1==e2)&&(e1==e3)){ntrafos=1;}
            else if ((e1==e2)&&(e1!=e3)){ntrafos=3;}
            else if ((e1==e3)&&(e1!=e2)){ntrafos=3;}
            else if ((e2==e3)&&(e2!=e1)){ntrafos=3;}
            else{ntrafos=6;}
            for (int t=0;t<ntrafos;t++){
                int e1t=tr[t][0], e2t=tr[t][1], e3t=tr[t][2];
                getMultipolesFromSymm_NNNN(thisN_n, nmax, t, nindices, len_nindices, thisN_n_rec);
                for(int kk=0;kk<n2n3combis_rec;kk++){
                    myN_n[(long)kk*N_nshift + e1t*nbinsr*nbinsr + e2t*nbinsr + e3t] += thisN_n_rec[kk];
                }
                for(int kk=0;kk<n2n3combis_rec;kk++){ thisN_n_rec[kk]=0; }
            }
        }
        free(sameband_N);
        t_sbrec[thisthread] += omp_get_wtime() - t_sbrec0;

        double t_xb0 = omp_get_wtime();
        for (int e1=0; e1<nbinsr; e1++){
            int b1 = bin2band[e1];
            for (int e2=e1; e2<nbinsr; e2++){
                int b2 = bin2band[e2];
                int discdisc = (b1==b2 && b1<hasdiscrete);  // two finer legs in the un-gridded band
                int egf = discdisc ? 0 : (b2 - hasdiscrete); // finer-cell grid resolution (b2 grid otherwise)
                if (discdisc && P12disc){
                    // cache w_x * W_{n2+n3}^x(e1) * conj(W_n2^x(e2)) per galaxy (reused across all b3)
                    for (int nindex=0; nindex<len_nindices; nindex++){
                        int thisn2 = nindices[nindex]/nnvals_Nn - nzero_Nn;
                        int thisn3 = nindices[nindex]%nnvals_Nn - nzero_Nn;
                        if (thisn2>nzero_Nn || -thisn2>nzero_Nn || thisn3>nzero_Nn || -thisn3>nzero_Nn){ continue; }
                        int thisn=thisn2+thisn3, mi=nzero_Wn+thisn, n2i=nzero_Wn+thisn2, n3i=nzero_Wn+thisn3;
                        double complex *p = P12disc + (long)((thisn2+nzero_Nn)*nnvals_Nn+(thisn3+nzero_Nn))*ndisc;
                        if (e1==e2){
                            // fold the e1==e2 multiple-counting correction: -w*W2_{n3}(theta1)
                            for (long il=0; il<ndisc; il++){
                                p[il] = disc_w[il] * (Wdisc[((long)il*nnvals_Wn + mi)*ndb + (e1-db_lo)]
                                                       * conj(Wdisc[((long)il*nnvals_Wn + n2i)*ndb + (e2-db_lo)])
                                                     - W2disc[((long)il*nnvals_Wn + n3i)*ndb + (e1-db_lo)]);
                            }
                        } else {
                            for (long il=0; il<ndisc; il++){
                                p[il] = disc_w[il] * Wdisc[((long)il*nnvals_Wn + mi)*ndb + (e1-db_lo)]
                                                  * conj(Wdisc[((long)il*nnvals_Wn + n2i)*ndb + (e2-db_lo)]);
                            }
                        }
                    }
                }
                for (int b3=mymax(b2,hasdiscrete); b3<nresos; b3++){
                    int rb3lo = reso_rindedges[b3], rb3hi = reso_rindedges[b3+1];
                    if (rb3hi<=rb3lo){ continue; }
                    if (1 + (b2!=b1) + ((b3!=b2)&&(b3!=b1)) < 2){ continue; } // all same band -> same-band phase
                    int eg3 = b3 - hasdiscrete;
                    int nC3 = ngal_in_pix[b3];
                    long coff3 = cumresoshift[b3];
                    int *mapf = (egf==eg3) ? NULL : parent[egf*nresos_grid + eg3]; // finer cell -> b3 cell (grid path)
                    for (int e3=rb3lo; e3<rb3hi; e3++){
                        for (int k=0;k<n2n3combis;k++){ accum[(long)e3*n2n3combis+k]=0; }
                    }
                    // stream over (n2,n3): build the small per-cell partial product, then fold leg3
                    for (int nindex=0; nindex<len_nindices; nindex++){
                        int thisn2 = nindices[nindex]/nnvals_Nn - nzero_Nn;
                        int thisn3 = nindices[nindex]%nnvals_Nn - nzero_Nn;
                        if (thisn2>nzero_Nn || -thisn2>nzero_Nn || thisn3>nzero_Nn || -thisn3>nzero_Nn){ continue; }
                        int thisn = thisn2+thisn3;
                        int ncombi = (thisn2+nzero_Nn)*nnvals_Nn + (thisn3+nzero_Nn);
                        for (int c=0;c<nC3;c++){ Dsmall[c]=0; }
                        if (!discdisc){
                            // grid finer legs: P2 from the cell cache at b2-cells, aggregated to b3
                            int nC2 = ngal_in_pix[b2];
                            long off1 = (long)(nzero_Wn+thisn)*nshift_cache  + (long)e1*thetashift + cumresoshift[b2];
                            long off2 = (long)(nzero_Wn+thisn2)*nshift_cache + (long)e2*thetashift + cumresoshift[b2];
                            for (int c2=0;c2<nC2;c2++){
                                int c3 = mapf ? mapf[c2] : c2;
                                Dsmall[c3] += conj(Wncache[off2+c2]) * wWncache[off1+c2];
                            }
                            if (e1==e2){
                                // fold the e1==e2 grid multiple-counting correction: -w*W2_{n3}(theta1)
                                long offw2 = (long)(nzero_Wn+thisn3)*nshift_cache + (long)e1*thetashift + cumresoshift[b2];
                                for (int c2=0;c2<nC2;c2++){
                                    int c3 = mapf ? mapf[c2] : c2;
                                    Dsmall[c3] -= wW2ncache[offw2+c2];
                                }
                            }
                        } else {
                            // two discrete finer legs: read the cached W12 product, aggregate into b3-cells
                            const int *c3_eg3 = disc_c3_all + (long)eg3*ndisc;
                            const double complex *p = P12disc + (long)ncombi*ndisc;
                            for (long il=0; il<ndisc; il++){ Dsmall[c3_eg3[il]] += p[il]; }
                        }
                        for (int e3=rb3lo; e3<rb3hi; e3++){
                            if (e3<e2){ continue; }
                            if (!wanted[e1*nbinsr*nbinsr + e2*nbinsr + e3]){ continue; }
                            int b3b = bin2band[e3];
                            if (1 + (b2!=b1) + ((b3b!=b2)&&(b3b!=b1)) < 2){ continue; }
                            long off3 = (long)(nzero_Wn+thisn3)*nshift_cache + (long)e3*thetashift + coff3;
                            double complex acc = 0;
                            for (int c3=0;c3<nC3;c3++){ acc += conj(Wncache[off3+c3]) * Dsmall[c3]; }
                            if (e3==e2){
                                // e2==e3 multiple-counting correction (b1<b2==b3): -w*W_{n2+n3}(theta1)*W2_{-n2-n3}(theta2)
                                int nC2 = ngal_in_pix[b2];
                                long o1 = (long)(nzero_Wn+thisn)*nshift_cache + (long)e1*thetashift + cumresoshift[b2];
                                long o2 = (long)(nzero_Wn-thisn)*nshift_cache + (long)e2*thetashift + cumresoshift[b2];
                                double complex cc = 0;
                                for (int c=0;c<nC2;c++){ cc += wWncache[o1+c] * W2ncache[o2+c]; }
                                acc -= cc;
                            }
                            accum[(long)e3*n2n3combis + ncombi] += acc;
                        }
                    }
                    // reconstruct each wanted triplet (e1,e2,e3) in this band into thread-private N_n
                    for (int e3=rb3lo; e3<rb3hi; e3++){
                        if (e3<e2){ continue; }
                        if (!wanted[e1*nbinsr*nbinsr + e2*nbinsr + e3]){ continue; }
                        int b3b = bin2band[e3];
                        if (1 + (b2!=b1) + ((b3b!=b2)&&(b3b!=b1)) < 2){ continue; }
                        for (int k=0;k<n2n3combis;k++){ thisN_n[k] = accum[(long)e3*n2n3combis+k]; }
                        int tr[6][3] = {{e1,e2,e3},{e2,e3,e1},{e3,e1,e2},{e1,e3,e2},{e2,e1,e3},{e3,e2,e1}};
                        int ntrafos;
                        if ((e1==e2)&&(e1==e3)){ntrafos=1;}
                        else if ((e1==e2)&&(e1!=e3)){ntrafos=3;}
                        else if ((e1==e3)&&(e1!=e2)){ntrafos=3;}
                        else if ((e2==e3)&&(e2!=e1)){ntrafos=3;}
                        else{ntrafos=6;}
                        for (int t=0;t<ntrafos;t++){
                            int e1t=tr[t][0], e2t=tr[t][1], e3t=tr[t][2];
                            getMultipolesFromSymm_NNNN(thisN_n, nmax, t, nindices, len_nindices, thisN_n_rec);
                            for(int k=0;k<n2n3combis_rec;k++){
                                myN_n[(long)k*N_nshift + e1t*nbinsr*nbinsr + e2t*nbinsr + e3t] += thisN_n_rec[k];
                            }
                            for(int k=0;k<n2n3combis_rec;k++){ thisN_n_rec[k]=0; }
                        }
                    }
                }
            }
        }
        t_xband[thisthread] += omp_get_wtime() - t_xb0;

        // (b1==b2==discrete triplets are handled inline in the streamed combination above,
        // via the per-galaxy Wdisc cache -- no separate pass / re-scan needed.)

        for (int eg2=0; eg2<nresos_grid; eg2++){
            for (int eg3=eg2+1; eg3<nresos_grid; eg3++){ free(parent[eg2*nresos_grid+eg3]); }
        }
        free(parent); free(Dsmall); free(accum); if (P12disc){ free(P12disc); }
        if (Wdisc){ free(Wdisc); free(W2disc); free(W3disc); free(disc_p1); free(disc_p2); free(disc_w); free(disc_c3_all); }
        free(thisN_n); free(thisN_n_rec);
        free(Wncache); free(wWncache); free(W2ncache); free(wW2ncache); free(pix2redpix);
        free(ngal_in_pix); free(matchers_resoshift); free(cumresoshift);
    }
    // Per-phase timers (CPU-s = summed over threads). Only printed at verbosity>=2;
    // the per-(thread,band) accumulators are always collected (negligible overhead).
    if (verbose){
        double t_cb_wall = omp_get_wtime() - t_cb_wall0;
        double t_cb_cpu  = (double)(clock() - t_cb_cpu0)/CLOCKS_PER_SEC;
        double cpu_sbrec=0, cpu_xband=0;
        for (int t=0;t<nthreads_cross;t++){ cpu_sbrec += t_sbrec[t]; cpu_xband += t_xband[t]; }
        printf("alloc_nnnn_doubletree TIMERS [region-loop wall %.1fs | cpu %.1fs | cores %.1f] CPU-s breakdown:\n",
               t_cb_wall, t_cb_cpu, t_cb_cpu/(t_cb_wall>0?t_cb_wall:1));
        double tot_scan=0, tot_aggc=0, tot_sbc=0;
        for (int b=0;b<nresos;b++){
            double s=0, a=0, c=0;
            for (int t=0;t<nthreads_cross;t++){ s+=t_scan[(long)t*nresos+b]; a+=t_aggc[(long)t*nresos+b]; c+=t_sbc[(long)t*nresos+b]; }
            tot_scan+=s; tot_aggc+=a; tot_sbc+=c;
            printf("  band %d [bins %d-%d]: aperture-scan %.1f | cell-agg %.1f | sameband-combine %.1f\n",
                   b, reso_rindedges[b], reso_rindedges[b+1], s, a, c);
        }
        printf("  TOTALS: scan %.1f | cell-agg %.1f | sameband-combine %.1f | sameband-recon %.1f | cross-band %.1f\n",
               tot_scan, tot_aggc, tot_sbc, cpu_sbrec, cpu_xband);
    }
    free(t_scan); free(t_aggc); free(t_sbc); free(t_sbrec); free(t_xband);

    // Reduce the cross-band thread-private contributions into N_n
    for (int t=0;t<nthreads_cross;t++){
        double complex *src = tmpN_n + (long)t*Nn_size;
        for (long i=0;i<Nn_size;i++){ N_n[i] += src[i]; }
    }
    free(tmpN_n); free(bin2band); free(wanted);

    // Bin centers from the accumulated counts
    double *totcounts = calloc(nbinsr, sizeof(double));
    double *totnorms  = calloc(nbinsr, sizeof(double));
    for (int t=0;t<nthreads;t++){
        for (int b=0;b<nbinsr;b++){ totcounts[b]+=tmp_totcounts[t*nbinsr+b]; totnorms[b]+=tmp_totnorms[t*nbinsr+b]; }
    }
    for (int b=0;b<nbinsr;b++){ if (totnorms[b]!=0){ bin_centers[b] = totcounts[b]/totnorms[b]; } }

    free(rshift_index_matcher_hash); free(rshift_pixs_galind_bounds); free(rshift_pix_gals);
    free(reso_rindedges); free(tmp_totcounts); free(tmp_totnorms); free(totcounts); free(totnorms);
    free(sb_e1); free(sb_e2); free(sb_e3); free(sb_band_lo); free(sb_band_hi);
}