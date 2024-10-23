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

#define mymin(x,y) ((x) <= (y)) ? (x) : (y)
#define mymax(x,y) ((x) >= (y)) ? (x) : (y)
#define M_PI      3.14159265358979323846
#define INV_2PI   0.15915494309189534561

// Non-tomo 4pcf using discrete estimator
// Very basic, no use of symmetry properties
void alloc_notomoGammans_discrete_gggg(
    int *isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int ngal, 
    int nmax, double rmin, double rmax, double *rbins, int nbinsr, int dccorr,
    int *index_matcher_hash, int *pixs_galind_bounds, int *pix_gals, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, double *bin_centers, double complex *Upsilon_n, double complex *N_n){
    
    // Temporary arrays that are allocated in parallel and later reduced
    double *tmpwcounts = calloc(nthreads*nbinsr, sizeof(double));
    double *tmpwnorms = calloc(nthreads*nbinsr, sizeof(double));
    
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
        double complex *tmpUpsilon_n = calloc(8*ups_compshift, sizeof(double complex));
        double complex *tmpN_n = calloc(ups_compshift, sizeof(double complex));
        
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
                int innergal;
                #pragma omp critical
                {p11 = pos1[ind_gal];
                p12 = pos2[ind_gal];
                w1 = weight[ind_gal];
                e11 = e1[ind_gal];
                e12 = e2[ind_gal];
                innergal = isinner[ind_gal];}
                if (innergal==0){continue;}
                
                int ind_gal2;
                int ind_red, lower, upper; 
                double  p21, p22, w2, w2_sq, e21, e22, rel1, rel2, dist, dphi;
                double complex wshape1, wshape1c, wshape2, wshape_sq, wshape_cube, wshapewshapec, wshapesqwshapec;
                double complex phirot, phirotc, twophirotc, fourphirotc, nphirot, nphirotc;
                
                // Check how many ns we need for Gn
                // Gns have shape (nnvals, nbinsz, nbinsr)
                // where the ns are ordered as 
                // [-nmax_1-nmax_2-3, ..., nmax_1+nmax_2+3]
                int nbinszr = nbinsz*nbinsr;
                double complex *nextGns =  calloc(ncomp*nnvals_Gn*nbinszr, sizeof(double complex));
                double complex *nextG2ns_gg =  calloc(nnvals_G2n*nbinszr, sizeof(double complex));
                double complex *nextG2ns_ggc =  calloc(nnvals_G2n*nbinszr, sizeof(double complex));
                double complex *nextG3ns_ggg = calloc(2*nbinszr, sizeof(double complex));
                double complex *nextG3ns_gggc = calloc(2*nbinszr, sizeof(double complex));
                double complex *nextWns = calloc(nnvals_Wn*nbinszr, sizeof(double complex));
                double complex *nextW2ns = calloc(nnvals_Wn*nbinszr, sizeof(double complex));
                double complex *nextW3ns = calloc(nbinszr, sizeof(double complex));

                int ind_rbin, rbin, zrshift, nextnshift, ind_Gn, ind_G2n, ind_Wn;
                double drbin = (log(rmax)-log(rmin))/(nbinsr);
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
                            // Double-counting corr for theta1==theta2
                            // Double-counting corr for theta1==theta3 
                            for (int elb2=0; elb2<nbinsr; elb2++){
                                // Double-counting corr for theta2==theta3
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
                                    thisnrshift = thisnshift + elb1*nbinsr*nbinsr + elb2*nbinsr + elb3;
                                    // Allocation of Upsilon and Norm
                                    tmpUpsilon_n[0*ups_compshift+thisnrshift] += gGG0*nextGns[thisGshift_mn3m3+elb3];
                                    tmpUpsilon_n[1*ups_compshift+thisnrshift] += gGG1*nextGns[thisGshift_mn3m1+elb3];
                                    tmpUpsilon_n[2*ups_compshift+thisnrshift] += gGG2*nextGns[thisGshift_mn3m3+elb3];
                                    tmpUpsilon_n[3*ups_compshift+thisnrshift] += gGG3*nextGns[thisGshift_mn3m3+elb3];
                                    tmpUpsilon_n[4*ups_compshift+thisnrshift] += gGG4*conj(nextGns[thisGshift_n3m1+elb3]);
                                    tmpUpsilon_n[5*ups_compshift+thisnrshift] += gGG5*nextGns[thisGshift_mn3m1+elb3];
                                    tmpUpsilon_n[6*ups_compshift+thisnrshift] += gGG6*nextGns[thisGshift_mn3m1+elb3];
                                    tmpUpsilon_n[7*ups_compshift+thisnrshift] += gGG7*conj(nextGns[thisGshift_n3m3+elb3]);
                                    tmpN_n[thisnrshift] += wNN*conj(nextWns[thisWshift_n3+elb3]);
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
                nextGns = NULL;
                nextG2ns_gg = NULL;
                nextG2ns_ggc = NULL;
                nextG3ns_ggg = NULL;
                nextG3ns_gggc = NULL;
                nextWns = NULL;
                nextW2ns = NULL;
                nextW3ns = NULL;
            }
        }
        
        // Accumulate Upsilon_n and N_n
        // Given the openmp implementation this needs to be done sequentially...however,
        // as the threads will reach this step at different points in time, it will
        // most likely not be a severe bottleneck.        
        int thisn_thread, thisn2, thisn3, thisnshift, thisnrshift, ind_Upsn;
        double complex toadd;
        for (int thisn=0; thisn<n2n3combis; thisn++){
            thisn_thread = (thisn+elthread)%n2n3combis;
            thisn2 = thisn_thread/nnvals_Upsn;
            thisn3 = thisn_thread%nnvals_Upsn;
            thisnshift = (thisn2*nnvals_Upsn + thisn3) * ups_nshift;
            for (int elb1=0; elb1<nbinsr; elb1++){
                for (int elb2=0; elb2<nbinsr; elb2++){
                    for (int elb3=0; elb3<nbinsr; elb3++){
                        thisnrshift = thisnshift + elb1*nbinsr*nbinsr + elb2*nbinsr + elb3;
                        N_n[thisnrshift] += tmpN_n[thisnrshift];
                        for (int elcomp=0; elcomp<8; elcomp++){
                            ind_Upsn = elcomp*ups_compshift+thisnrshift;
                            toadd = tmpUpsilon_n[elcomp*ups_compshift+thisnrshift];
                            //#pragma omp atomic update
                            Upsilon_n[ind_Upsn] += toadd;
                        }
                    }
                }
            }
        }
        
        free(tmpUpsilon_n);
        free(tmpN_n);
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
    
    free(tmpwcounts);
    free(tmpwnorms);
    free(totcounts);
    free(totnorms);
}




// If thread==0 --> For final two threads allocate double/triple counting corrs
// thetacombis_batches: array of length nbinsr^3 with the indices of all possible (r1,r2,r3) combinations
//                      most likely it is simply range(nbinsr^3), but we leave some freedom here for 
//                      potential cost-based implementations
// nthetacombis_batches: array of length nthetbatches with the number of theta-combis in each batch
// cumthetacombis_batches : array of length (nthetbatches+1) with is cumsum of nthetacombis_batches
// nthetbatches: the number of theta batches
void alloc_notomoMap4_disc_gggg(
    int *isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int ngal, 
    int nmax, double rmin, double rmax, int nbinsr, int dccorr, double *phibins, double *dbinsphi, int nbinsphi,
    int *index_matcher_hash, int *pixs_galind_bounds, int *pix_gals, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, int projection, double *mapradii, int nmapradii, double complex *M4correlators, 
    int alloc_4pcfmultipoles, int alloc_4pcfreal,
    double *bin_centers, double complex *Upsilon_n, double complex *N_n, double complex *Gammas, double complex *Norms){
               
    double complex *allM4correlators = calloc(nthreads*8*1*nmapradii, sizeof(double complex));
    #pragma omp parallel for
    for(int elthetbatch=0;elthetbatch<nthetbatches;elthetbatch++){
        int nregions_skip_print = nregions/1000;
        
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
                int innergal;
                p11 = pos1[ind_gal];
                p12 = pos2[ind_gal];
                w1 = weight[ind_gal];
                e11 = e1[ind_gal];
                e12 = e2[ind_gal];
                innergal = isinner[ind_gal];
                if (innergal==0){continue;}
                
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
    int *isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int ngal, 
    int nmax, double rmin, double rmax, int nbinsr, int dccorr, 
    int *nindices, int len_nindices, double *phibins, double *dbinsphi, int nbinsphi,
    int nresos, double *reso_redges, int *ngal_resos, 
    int *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos, 
    double *e1_resos, double *e2_resos,
    int *index_matcher_hash, int *pixs_galind_bounds, int *pix_gals, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, int projection, double *mapradii, int nmapradii, double complex *M4correlators, 
    int alloc_4pcfmultipoles, int alloc_4pcfreal,
    double *bin_centers, double complex *Upsilon_n, double complex *N_n, double complex *Gammas, double complex *Norms){
               
    double complex *allM4correlators = calloc(nthreads*8*1*nmapradii, sizeof(double complex));
    #pragma omp parallel for
    for (int elthetbatch=0;elthetbatch<nthetbatches;elthetbatch++){
        int nregions_skip_print = nregions/1000;
        
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
                int innergal;
                p11 = pos1[ind_gal];
                p12 = pos2[ind_gal];
                w1 = weight[ind_gal];
                e11 = e1[ind_gal];
                e12 = e2[ind_gal];
                innergal = isinner[ind_gal];
                if (innergal==0){continue;}
                
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
                for (int elreso=reso_min_batch;elreso<=reso_max_batch;elreso++){
                    int rbin, zrshift, nextnshift, ind_Gn, ind_G2n, ind_Wn;
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
                                {p21 = pos1_resos[ind_gal2];
                                p22 = pos2_resos[ind_gal2];
                                w2 = weight_resos[ind_gal2];
                                e21 = e1_resos[ind_gal2];
                                e22 = e2_resos[ind_gal2];}
                                
                                rel1 = p21 - p11;
                                rel2 = p22 - p12;
                                dist = sqrt(rel1*rel1 + rel2*rel2);
                                if(dist < rmin_reso || dist >= rmax_reso) continue;
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
                if ((elregion%nregions_skip_print == 0)&&(thisthread==0)&&(ind_inpix1==lower1)){
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
                    if (elregion==0 && elthetbatch==0){printf("nindex %d: n2=%d n3=%d\n",nindex,thisn2,thisn3);}
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
            if (thisthread==0){
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
                getMultiplolesFromSymm(thisUpsilon_n, thisN_n, nmax, eltrafo, nindices, len_nindices,
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
        
        for (int elmapr=0; elmapr<nmapradii; elmapr++){
            for (int elcomp=0;elcomp<8;elcomp++){
                int map4ind = elcomp*nmapradii+elmapr;
                int map4threadshift = thisthread*8*nmapradii;
                printf("\nthread %d, elr %d, elcomp %d, allM4cont=%.20f ",
                               thisthread, elmapr, elcomp, creal(allM4correlators[map4threadshift+map4ind]));
            }
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

// Computes Ups(theta1,theta2,theta1|z1,z2,z2,z2)
// --> For this all the symmetries & multiple-counting corrs are the same as for the nontomo case
// If thread==0 --> For final two threads allocate double/triple counting corrs
// thetacombis_batches: array of length nbinsr^3 with the indices of all possible (r1,r2,r3) combinations
//                      most likely it is simply range(nbinsr^3), but we leave some freedom here for 
//                      potential cost-based implementations
// nthetacombis_batches: array of length nthetbatches with the number of theta-combis in each batch
// cumthetacombis_batches : array of length (nthetbatches+1) with is cumsum of nthetacombis_batches
// nthetbatches: the number of theta batches
void alloc_twotomoMap4_tree_gggg(
    int *isinner_z1, double *weight_z1, double *pos1_z1, double *pos2_z1, double *e1_z1, double *e2_z1, int ngal_z1, 
    int *index_matcher_hash_z1, int *pixs_galind_bounds_z1, int *pix_gals_z1, int nregions_z1, 
    int nmax, double rmin, double rmax, int nbinsr, int dccorr, 
    int *nindices, int len_nindices, double *phibins, double *dbinsphi, int nbinsphi,
    int nresos, double *reso_redges, int *ngal_resos_z2, 
    double *weight_resos_z2, double *pos1_resos_z2, double *pos2_resos_z2, 
    double *e1_resos_z2, double *e2_resos_z2,
    int *index_matcher_hash_z2, int *pixs_galind_bounds_z2, int *pix_gals_z2, int nregions_z2, 
    double pix1_start_z2, double pix1_d_z2, int pix1_n_z2, double pix2_start_z2, double pix2_d_z2, int pix2_n_z2, 
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, int projection, double *mapradii, int nmapradii, double complex *M4correlators, 
    int alloc_4pcfmultipoles, int alloc_4pcfreal,
    double *bin_centers, double complex *Upsilon_n, double complex *N_n, double complex *Gammas, double complex *Norms){
               
    double complex *allM4correlators = calloc(nthreads*8*1*nmapradii, sizeof(double complex));
    #pragma omp parallel for
    for (int elthetbatch=0;elthetbatch<nthetbatches;elthetbatch++){
        int nregions_skip_print = nregions_z1/1000;
        
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
        
        int npix_hash_z2 = pix1_n_z2*pix2_n_z2;
        int *rshift_index_matcher_hash_z2 = calloc(nresos, sizeof(int));
        int *rshift_pixs_galind_bounds_z2 = calloc(nresos, sizeof(int));
        int *rshift_pix_gals_z2 = calloc(nresos, sizeof(int));
        for (int elreso=1;elreso<nresos;elreso++){
            rshift_index_matcher_hash_z2[elreso] = rshift_index_matcher_hash_z2[elreso-1] + npix_hash_z2;
            rshift_pixs_galind_bounds_z2[elreso] = rshift_pixs_galind_bounds_z2[elreso-1] + ngal_resos_z2[elreso-1]+1;
            rshift_pix_gals_z2[elreso] = rshift_pix_gals_z2[elreso-1] + ngal_resos_z2[elreso-1];
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
        int offset_per_thread = nregions_z1/nthreads;
        int offset = offset_per_thread*thisthread;
        for (int _elregion=0; _elregion<nregions_z1; _elregion++){
            int elregion = (_elregion+offset)%nregions_z1; // Try to evade collisions
            if ((elregion%nregions_skip_print == 0)&&(thisthread==0)){
                printf("Doing region %d/%d for thetabatch %d/%d\n",elregion,nregions_z1,elthetbatch,nthetbatches);
            }
            //int region_debug = mymin(500,nregions-1);
            int lower1, upper1;
            lower1 = pixs_galind_bounds_z1[elregion];
            upper1 = pixs_galind_bounds_z1[elregion+1];
            for (int ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                double time1, time2;
                time1 = omp_get_wtime();
                int ind_gal = pix_gals_z1[ind_inpix1];
                double p11, p12, w1, e11, e12;
                int innergal;
                p11 = pos1_z1[ind_gal];
                p12 = pos2_z1[ind_gal];
                w1 = weight_z1[ind_gal];
                e11 = e1_z1[ind_gal];
                e12 = e2_z1[ind_gal];
                innergal = isinner_z1[ind_gal];
                if (innergal==0){continue;}
                
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
                for (int elreso=reso_min_batch;elreso<=reso_max_batch;elreso++){
                    int rbin, zrshift, nextnshift, ind_Gn, ind_G2n, ind_Wn;
                    double rmin_reso = reso_redges[elreso];
                    double rmax_reso = reso_redges[elreso+1];
                    int pix1_lower = mymax(0, (int) floor((p11 - (rmax_reso+pix1_d_z2) - pix1_start_z2)/pix1_d_z2));
                    int pix2_lower = mymax(0, (int) floor((p12 - (rmax_reso+pix2_d_z2) - pix2_start_z2)/pix2_d_z2));
                    int pix1_upper = mymin(pix1_n_z2-1, (int) floor((p11 + (rmax_reso+pix1_d_z2) - pix1_start_z2)/pix1_d_z2));
                    int pix2_upper = mymin(pix2_n_z2-1, (int) floor((p12 + (rmax_reso+pix2_d_z2) - pix2_start_z2)/pix2_d_z2));
                    for (int ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                        for (int ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                            ind_red = index_matcher_hash_z2[rshift_index_matcher_hash_z2[elreso] + ind_pix2*pix1_n_z2 + ind_pix1];
                            if (ind_red==-1){continue;}
                            lower = pixs_galind_bounds_z2[rshift_pixs_galind_bounds_z2[elreso]+ind_red];
                            upper = pixs_galind_bounds_z2[rshift_pixs_galind_bounds_z2[elreso]+ind_red+1];
                            for (int ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                                ind_gal2 = rshift_pix_gals_z2[elreso] + pix_gals_z2[rshift_pix_gals_z2[elreso]+ind_inpix];
                                //#pragma omp critical
                                {p21 = pos1_resos_z2[ind_gal2];
                                p22 = pos2_resos_z2[ind_gal2];
                                w2 = weight_resos_z2[ind_gal2];
                                e21 = e1_resos_z2[ind_gal2];
                                e22 = e2_resos_z2[ind_gal2];}
                                
                                rel1 = p21 - p11;
                                rel2 = p22 - p12;
                                dist = sqrt(rel1*rel1 + rel2*rel2);
                                if(dist < rmin_reso || dist >= rmax_reso) continue;
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
                if ((elregion%nregions_skip_print == 0)&&(thisthread==0)&&(ind_inpix1==lower1)){
                    printf("Computed Gn for first gal in region %d/%d for thetabatch %d/%d in %.4f seconds\n",
                           elregion,nregions_z1,elthetbatch,nthetbatches,(time2-time1));}                
                
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
                    if (elregion==0 && elthetbatch==0){printf("nindex %d: n2=%d n3=%d\n",nindex,thisn2,thisn3);}
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
                if ((elregion%nregions_skip_print == 0)&&(thisthread==0)&&(ind_inpix1==lower1)){
                    printf("Allocated Ups for first gal in region %d/%d for thetabatch %d/%d in %.4f seconds for %d theta-combis\n",
                           elregion,nregions_z1,elthetbatch,nthetbatches,(time2-time1),batch_nthetas);}
            }
            if ((elregion%nregions_skip_print == 0)&&(thisthread==0)){
                printf("Done region %d/%d for thetabatch %d/%d\n",elregion,nregions_z1,elthetbatch,nthetbatches);}
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
            if (thisthread==0){
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
                getMultiplolesFromSymm(thisUpsilon_n, thisN_n, nmax, eltrafo, nindices, len_nindices,
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
        
        for (int elmapr=0; elmapr<nmapradii; elmapr++){
            for (int elcomp=0;elcomp<8;elcomp++){
                int map4ind = elcomp*nmapradii+elmapr;
                int map4threadshift = thisthread*8*nmapradii;
                printf("\nthread %d, elr %d, elcomp %d, allM4cont=%.20f ",
                               thisthread, elmapr, elcomp, creal(allM4correlators[map4threadshift+map4ind]));
            }
        }
        if (thisthread>-1){printf("Done allocating 4pcfs for thetabatch %d/%d\n",elthetbatch,nthetbatches);}
            
        free(rshift_index_matcher_hash_z2);
        free(rshift_pixs_galind_bounds_z2);
        free(rshift_pix_gals_z2);
            
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

