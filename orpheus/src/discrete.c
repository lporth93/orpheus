#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <complex.h>

#include "spatialhash.h"
#include "assign.h"
#include "discrete.h"

#define mymin(x,y) ((x) <= (y)) ? (x) : (y)
#define mymax(x,y) ((x) >= (y)) ? (x) : (y)
#define M_PI           3.14159265358979323846  


// Allocates multipoles of shape catalog via discrete estimator
void alloc_Gammans_discrete_ggg(
    int *isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int *zbins, int nbinsz, int ngal, 
    int nmin, int nmax, double rmin, double rmax, double *rbins, int nbinsr, int dccorr,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, double *bin_centers, double complex *Gammans, double complex *Gammans_norm){
    
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
                int zbin1, innergal;
                #pragma omp critical
                {p11 = pos1[ind_gal];
                p12 = pos2[ind_gal];
                w1 = weight[ind_gal];
                zbin1 = zbins[ind_gal];
                e11 = e1[ind_gal];
                e12 = e2[ind_gal];
                innergal = isinner[ind_gal];}
                if (innergal==0){continue;}
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
                                // Nomminal allocation
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
        
        /*
        int nonzero = 0;
        for (int thisn=0; thisn<nmax-nmin+1; thisn++){
            for (int zcombi=0; zcombi<nbinsz*nbinsz*nbinsz; zcombi++){
                for (int elb1=0; elb1<nbinsr; elb1++){
                    for (int elb2=0; elb2<nbinsr; elb2++){
                        int iGamma = thisn*_gamma_nshift + zcombi*_gamma_zshift + elb1*nbinsr + elb2;
                        if (cabs(Gammans_norm[iGamma])>1e-5){
                            nonzero += 1;
                        }
                    }
                }
            }
        }
        */
        
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

void alloc_Gammans_tree_ggg(
    int *isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int *zbins, int nbinsz, int ngal, 
    int nresos, double *reso_redges, int *ngal_resos, 
    double *weight_resos, double *pos1_resos, double *pos2_resos, double *e1_resos, double *e2_resos, int *zbin_resos,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int nmin, int nmax, double rmin, double rmax, double *rbins, int nbinsr, int dccorr, 
    int nthreads, double *bin_centers, double complex *Gammans, double complex *Gammans_norm){
    
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
            
            int npix_hash = pix1_n*pix2_n;
            int *rshift_index_matcher = calloc(nresos, sizeof(int));
            int *rshift_pixs_galind_bounds = calloc(nresos, sizeof(int));
            int *rshift_pix_gals = calloc(nresos, sizeof(int));
            for (int elreso=1;elreso<nresos;elreso++){
                rshift_index_matcher[elreso] = rshift_index_matcher[elreso-1] + npix_hash;
                rshift_pixs_galind_bounds[elreso] = rshift_pixs_galind_bounds[elreso-1] + ngal_resos[elreso-1]+1;
                rshift_pix_gals[elreso] = rshift_pix_gals[elreso-1] + ngal_resos[elreso-1];
            }
                
            for (int ind_gal=0; ind_gal<ngal; ind_gal++){
                // Check if galaxy falls in stripe used in this process
                double p11, p12, w1, e11, e12;
                int zbin1, innergal;
                #pragma omp critical
                {p11 = pos1[ind_gal];
                p12 = pos2[ind_gal];
                w1 = weight[ind_gal];
                zbin1 = zbins[ind_gal];
                e11 = e1[ind_gal];
                e12 = e2[ind_gal];
                innergal = isinner[ind_gal];}
                if (innergal==0){continue;}
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
                                // Nomminal allocation
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


// Allocates multipoles of shape catalog via discrete estimator
void alloc_Gammans_discrete_gggg(
    int *isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int *zbins, int nbinsz, int ngal, 
    int nmax, double rmin, double rmax, double *rbins, int nbinsr, int dccorr,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, double *bin_centers, double complex *Gammans, double complex *Gammans_norm){
    
    int ncomp = 8;
    // Index shift for the Gamman
    int _gamma_zshift = nbinsr*nbinsr*nbinsr;
    int _gamma_nshift = _gamma_zshift*nbinsz*nbinsz*nbinsz*nbinsz;
    int _gamma_compshift = (nmax+1)*(nmax+1)*_gamma_nshift;
    
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
        double complex *tmpGammans = calloc(nthreads*ncomp*_gamma_compshift, sizeof(double complex));
        double complex *tmpGammans_norm = calloc(nthreads*_gamma_compshift, sizeof(double complex));
        #pragma omp parallel for num_threads(nthreads)
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            int gamma_zshift = nbinsr*nbinsr*nbinsr;
            int gamma_nshift = _gamma_zshift*nbinsz*nbinsz*nbinsz*nbinsz;
            int gamma_compshift = (nmax+1)*_gamma_nshift;
            for (int ind_gal=0; ind_gal<ngal; ind_gal++){
                // Check if galaxy falls in stripe used in this process
                double p11, p12, w1, e11, e12;
                int zbin1, innergal;
                #pragma omp critical
                {p11 = pos1[ind_gal];
                p12 = pos2[ind_gal];
                w1 = weight[ind_gal];
                zbin1 = zbins[ind_gal];
                e11 = e1[ind_gal];
                e12 = e2[ind_gal];
                innergal = isinner[ind_gal];}
                if (innergal==0){continue;}
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
                // [-nmax_1-nmax_2-3, ..., nmax_1+nmax_2+3]
                nnvals=4*nmax+7;
                nnvals_norm=4*nmax+1;
                double complex *nextGns =  calloc(nnvals*nbinsr*nbinsz, sizeof(double complex));
                double complex *nextGns_norm =  calloc(nnvals_norm*nbinsr*nbinsz, sizeof(double complex));
                double complex *nextG2ns =  calloc(ncomp*nbinsz*nbinsr, sizeof(double complex));
                double complex *nextG2ns_norm =  calloc(nbinsz*nbinsr, sizeof(double complex));
                double complex *nextG3ns =  calloc(ncomp*nbinsz*nbinsr, sizeof(double complex));
                double complex *nextG3ns_norm =  calloc(nbinsz*nbinsr, sizeof(double complex));

                int ind_rbin, rbin;
                int ind_Gn, ind_Gnnorm, zrshift, nextnshift;
                int nbinszr = nbinsz*nbinsr;
                double drbin = (log(rmax)-log(rmin))/(nbinsr);
                
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
                            
                            nzero = 2*nmax+3;
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
                            nextG3ns_norm[zrshift] += w2*w2*w2;
                            nphirot *= phirot;
                            nphirotc *= phirotc; 
                            // n in [1, ..., nmax-1] x {+1,-1}
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
                    }
                }
                
                // Now update the Gammans
                // tmpGammas have shape (nthreads, nmax+1, nzcombis3, r*r, ncomp)
                // Gns have shape (nnvals, nbinsz, nbinsr)
                //int nonzero_tmpGammas = 0;
                double complex h0, h1, h2, h3, w0;
                int thisnshift, r12shift;
                int gammashift1, gammashiftt1, gammashift, gammashiftt;
                int ind_mnm3, ind_mnm1, ind_nm3, ind_nm1, ind_norm;
                int elb2, zbin3, zcombi;
                wshape = w1 * (e11+I*e12);
                for (int thisn=0; thisn<nmax+1; thisn++){
                    nzero = nmax+3;
                    ind_mnm3 = (nzero-thisn-3)*nbinszr;
                    ind_mnm1 = (nzero-thisn-1)*nbinszr;
                    ind_nm3 = (nzero+thisn-3)*nbinszr;
                    ind_nm1 = (nzero+thisn-1)*nbinszr;
                    ind_norm = thisn*nbinszr;
                    
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
                                    gammashift = ncomp*(gammashift1 + elb1);
                                    gammashiftt = gammashiftt1 + r12shift;
                                    //phirotm = wshape*nextG2ns[zrshift];
                                    tmpGammans[gammashift] += wshape*nextG2ns[zrshift];
                                    tmpGammans[gammashift+1] += conj(wshape)*nextG2ns[nbinszr+zrshift];
                                    tmpGammans[gammashift+2] += wshape*nextG2ns[2*nbinszr+zrshift];
                                    tmpGammans[ncomp*gammashiftt+3] += wshape*nextG2ns[3*nbinszr+zrshift];
                                    tmpGammans_norm[gammashiftt] -= w1*nextG2ns_norm[zrshift];
                                }
                                // Nomminal allocation
                                for (elb2=0; elb2<nbinsr; elb2++){
                                    zrshift = zbin3*nbinsr + elb2;
                                    r12shift = elb2*nbinsr+elb1;
                                    gammashift = ncomp*(gammashift1 + elb2);
                                    gammashiftt = gammashiftt1 + r12shift;
                                    //phirotm = h0*nextGns[ind_mnm3 + zrshift];
                                    tmpGammans[gammashift] += h0*nextGns[ind_mnm3 + zrshift];
                                    tmpGammans[gammashift+1] += h1*nextGns[ind_mnm1 + zrshift];
                                    tmpGammans[gammashift+2] += h2*nextGns[ind_mnm3 + zrshift];
                                    tmpGammans[ncomp*gammashiftt+3] += h3*nextGns[ind_nm3 + zrshift];
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
        for (int thisn=0; thisn<nmax+1; thisn++){
            int itmpGamma, iGamma;
            for (int thisthread=0; thisthread<nthreads; thisthread++){
                for (int zcombi=0; zcombi<nbinsz*nbinsz*nbinsz; zcombi++){
                    for (int elb1=0; elb1<nbinsr; elb1++){
                        for (int elb2=0; elb2<nbinsr; elb2++){
                            iGamma = thisn*_gamma_nshift + zcombi*_gamma_zshift + elb1*nbinsr + elb2;
                            itmpGamma = iGamma + thisthread*_gamma_compshift;
                            for (int elcomp=0; elcomp<ncomp; elcomp++){
                                Gammans[elcomp*_gamma_compshift+iGamma] += tmpGammans[ncomp*itmpGamma+elcomp];
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
