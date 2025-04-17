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

#define mymin(x,y) ((x) <= (y)) ? (x) : (y)
#define mymax(x,y) ((x) >= (y)) ? (x) : (y)
#define M_PI      3.14159265358979323846
#define INV_2PI   0.15915494309189534561


///////////////////////////////////////////////
/// THIRD-ORDER SHEAR CORRELATION FUNCTIONS ///
///////////////////////////////////////////////
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
    double *weight_resos, double *pos1_resos, double *pos2_resos, 
    double *e1_resos, double *e2_resos, int *zbin_resos, double *weightsq_resos,
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
            int ngalproc = 0;
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
                
                if (thisthread==nthreads/2){
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

void alloc_Gammans_doubletree_ggg(
    int nresos, int nresos_grid, double *dpix1_resos,  double *dpix2_resos, double *reso_redges, 
    int resoshift_leafs, int minresoind_leaf, int maxresoind_leaf,
    int *ngal_resos, int nbinsz, int *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos, 
    double *e1_resos, double *e2_resos, int *zbin_resos, double *weightsq_resos,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int *index_matcher_hash, int nregions, int *filledregions, int nfilledregions, 
    int nmax, double rmin, double rmax, int nbinsr, int dccorr, 
    int nthreads, double *bin_centers, double complex *Gammans, double complex *Gammans_norm){
        
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
        int nregions_per_thread = nfilledregions/nthreads;
        int hasdiscrete = nresos-nresos_grid;
        int nnvals_Gn = 2*nmax+3;
        int nnvals_Nn = nmax+1;
        
        // Compute how large the caches have to be at most for this thread
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
        for (int _elregion=0; _elregion<nfilledregions; _elregion++){
            int region_debug=-99999;
            // Check if this thread is responsible for the region
            int nthread_target = mymin(_elregion/nregions_per_thread, nthreads-1);
            if (nthread_target!=elthread){continue;}
            // printf("Region %d is in thread %d\n",elregion,elthread);
            if (_elregion==region_debug){printf("Region %d is in thread %d\n",_elregion,elthread);}
            if (elthread==nthreads/2){
                printf("\rDone %.2f per cent",100*((double) _elregion-nregions_per_thread*(int)(nthreads/2))/nregions_per_thread);
            }
            int elregion = filledregions[_elregion];
            
            // Check which sets of radii are evaluated for each resolution
            int *reso_rindedges = calloc(nresos+1, sizeof(int));
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
            if (elregion==region_debug){
                printf("Bin edges:\n");
                for (int elreso=0;elreso<nresos;elreso++){
                    printf("  reso=%d: index_start=%d, rtarget_start=%.2f, rtrue_start=%.2f\n",
                           elreso, reso_rindedges[elreso], reso_redges[elreso], rmin*exp(reso_rindedges[elreso]*drbin));
                    printf("           index_end=%d, rtarget_end=%.2f, rtrue_end=%.2f\n",
                           reso_rindedges[elreso+1], reso_redges[elreso+1], rmin*exp(reso_rindedges[elreso+1]*drbin));
                }
            }
                        
            // Shift variables for 3pcf quantities
            int gamma_zshift = nbinsr*nbinsr;
            int gamma_nshift = gamma_zshift*nbinsz*nbinsz*nbinsz;
            int gamma_compshift = (nmax+1)*gamma_nshift;
            
            // Shift variables for spatial hash
            int npix_hash = pix1_n*pix2_n;
            int *rshift_index_matcher = calloc(nresos, sizeof(int));
            int *rshift_pixs_galind_bounds = calloc(nresos, sizeof(int));
            int *rshift_pix_gals = calloc(nresos, sizeof(int));
            for (int elreso=1;elreso<nresos;elreso++){
                rshift_index_matcher[elreso] = rshift_index_matcher[elreso-1] + npix_hash;
                rshift_pixs_galind_bounds[elreso] = rshift_pixs_galind_bounds[elreso-1] + ngal_resos[elreso-1]+1;
                rshift_pix_gals[elreso] = rshift_pix_gals[elreso-1] + ngal_resos[elreso-1];
            }
            
            // Shift variables for the matching between the pixel grids
            int lower, upper, lower1, upper1, lower2, upper2, ind_inpix, ind_gal, zbin_gal;
            int npix_side, thisreso, elreso_grid, len_matcher;
            int *matchers_resoshift = calloc(nresos_grid+1, sizeof(int));
            int *ngal_in_pix = calloc(nresos*nbinsz, sizeof(int));
            for (int elreso=0;elreso<nresos;elreso++){
                elreso_grid = elreso - hasdiscrete;
                lower = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion];
                upper = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion+1];
                for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                    ind_gal = rshift_pix_gals[elreso] + pix_gals[rshift_pix_gals[elreso]+ind_inpix];
                    ngal_in_pix[zbin_resos[ind_gal]*nresos+elreso] += 1;
                }
                if (elregion==region_debug){
                    for (int elbinz=0; elbinz<nbinsz; elbinz++){
                        printf("ngal_in_pix[elreso=%d][elz=%d] = %d \n",
                               elreso,elbinz,ngal_in_pix[elbinz*nresos+elreso]);
                    }
                }
                if (elreso_grid>=0){
                    npix_side = 1 << (nresos_grid-elreso_grid-1);
                    matchers_resoshift[elreso_grid+1] = matchers_resoshift[elreso_grid] + npix_side*npix_side; 
                }
                if (elregion==region_debug){printf("matchers_resoshift[elreso=%d] = %d \n", elreso,matchers_resoshift[elreso_grid+1]);}
            }
            len_matcher = matchers_resoshift[nresos_grid];
            
            
            // Build the matcher from pixels to reduced pixels in the region
            int elregion_fullhash, elhashpix_1, elhashpix_2, elhashpix;
            double hashpix_start1, hashpix_start2;
            double pos1_gal, pos2_gal;
            elregion_fullhash = index_matcher_hash[elregion];
            hashpix_start1 = pix1_start + (elregion_fullhash%pix1_n)*pix1_d;
            hashpix_start2 = pix2_start + (elregion_fullhash/pix1_n)*pix2_d;
            if (elregion==region_debug){
                printf("pix1_start=%.2f pix2_start=%.2f \n", pix1_start,pix2_start);
                printf("hashpix_start1=%.2f hashpix_start2=%.2f \n", hashpix_start1,hashpix_start2);}
            int *pix2redpix = calloc(nbinsz*len_matcher, sizeof(int)); // For each z matches pixel in unreduced grid to index in reduced grid
            for (int elreso=0;elreso<nresos_grid;elreso++){
                thisreso = elreso + hasdiscrete;
                lower = pixs_galind_bounds[rshift_pixs_galind_bounds[thisreso]+elregion];
                upper = pixs_galind_bounds[rshift_pixs_galind_bounds[thisreso]+elregion+1];
                npix_side = 1 << (nresos_grid-elreso-1);
                int *tmpcounts = calloc(nbinsz, sizeof(int));
                for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                    ind_gal = rshift_pix_gals[thisreso] + pix_gals[rshift_pix_gals[thisreso]+ind_inpix];
                    zbin_gal = zbin_resos[ind_gal];
                    pos1_gal = pos1_resos[ind_gal];
                    pos2_gal = pos2_resos[ind_gal];
                    elhashpix_1 = (int) floor((pos1_gal - hashpix_start1)/dpix1_resos[elreso]);
                    elhashpix_2 = (int) floor((pos2_gal - hashpix_start2)/dpix2_resos[elreso]);
                    elhashpix = elhashpix_2*npix_side + elhashpix_1;
                    //pix2redpix[zbin_gal*len_matcher+matchers_resoshift[elreso]+elhashpix] = ind_inpix-lower;
                    pix2redpix[zbin_gal*len_matcher+matchers_resoshift[elreso]+elhashpix] = tmpcounts[zbin_gal];
                    tmpcounts[zbin_gal] += 1;
                    if (elregion==region_debug){
                        printf("elreso=%d, lower=%d, thispix=%d, zgal=%d: pix2redpix[%d]=%d  \n",
                               elreso,lower,ind_inpix,zbin_gal,zbin_gal*len_matcher+matchers_resoshift[elreso]+elhashpix,ind_inpix-lower);
                    }
                }
                free(tmpcounts);
            }
            
            // Resopix2resopix
            // [resopix_reso0 --> [...id........., resopix_reso1, resopix_reso2, ..., resopix_reson],
            //  resopix_reso1 --> [....0........., ...id........, resopix_reso2, ..., resopix_reson],
            //. ...
            //  resopix_reson --> [....0........., ....0........, ....0........, ..., resopix_reson]
            // ] --> nreso*
                        
            // Setup all shift variables for the Gncache in the region
            // Gncache has structure
            // n --> zbin2 --> zbin1 --> radius 
            //   --> [ [0]*ngal_zbin1_reso1 | [0]*ngal_zbin1_reso1/2 | ... | [0]*ngal_zbin1_reson ]
            int *cumresoshift_z = calloc(nbinsz*(nresos+1), sizeof(int)); // Cumulative shift index for resolution at z1
            int *thetashifts_z = calloc(nbinsz, sizeof(int)); // Shift index for theta given z1
            int *zbinshifts = calloc(nbinsz+1, sizeof(int)); // Cumulative shift index for z1
            int zbin2shift, nshift; // Shifts for z2 index and n index
            for (int elz=0; elz<nbinsz; elz++){
                if (elregion==region_debug){printf("z=%d/%d: \n", elz,nbinsz);}
                for (int elreso=0; elreso<nresos; elreso++){
                    if (elregion==region_debug){printf("  reso=%d/%d: \n", elreso,nresos);}
                    if (hasdiscrete==1 && elreso==0){
                        cumresoshift_z[elz*(nresos+1) + elreso+1] = ngal_in_pix[elz*nresos + elreso+1];
                    }
                    else{
                        cumresoshift_z[elz*(nresos+1) + elreso+1] = cumresoshift_z[elz*(nresos+1) + elreso] + ngal_in_pix[elz*nresos + elreso];
                    }
                    if (elregion==region_debug){printf("  cumresoshift_z[z][reso+1]=%d: \n", cumresoshift_z[elz*(nresos+1) + elreso+1]);}
                }
                thetashifts_z[elz] = cumresoshift_z[elz*(nresos+1) + nresos];
                zbinshifts[elz+1] = zbinshifts[elz] + nbinsr*thetashifts_z[elz];
                if ((elregion==region_debug)){printf("thetashifts_z[z]=%d: \nzbinshifts[z+1]=%d: \n", thetashifts_z[elz],  zbinshifts[elz+1]);}
            }
            zbin2shift = zbinshifts[nbinsz];
            nshift = nbinsz*zbin2shift;
            // Set all the cache indices that are updated in this region to zero
            if ((elregion==region_debug)){printf("zbin2shift=%d: nshift=%d: \n", zbin2shift,  nshift);}
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
            int innergal, rbin, nextn, nextnshift, nbinszr, nbinszr_reso, zrshift, ind_rbin;
            double pos1_gal1, pos2_gal1, pos1_gal2, pos2_gal2, w_gal1, w_gal2, wsq_gal2, e1_gal1, e2_gal1, e1_gal2, e2_gal2;
            double rel1, rel2, dist;
            double complex wshape_gal1, wshape_gal2;
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
                double complex *nextGns =  calloc(nnvals_Gn*nbinszr_reso, sizeof(double complex));
                double complex *nextGns_norm =  calloc(nnvals_Nn*nbinszr_reso, sizeof(double complex));
                double complex *nextG2ns =  calloc(4*nbinszr_reso, sizeof(double complex));
                double complex *nextG2ns_norm =  calloc(nbinszr_reso, sizeof(double complex));
                double complex *nextG2ndiscs_norm =  calloc(nbinszr_reso, sizeof(double complex));
                int *nextncounts = calloc(nbinszr_reso, sizeof(int));
                int *allowedrinds = calloc(nbinszr_reso, sizeof(int));
                int *allowedzinds = calloc(nbinszr_reso, sizeof(int));
                if (elregion==region_debug){printf("rbinmin=%d, rbinmax%d\n",rbinmin,rbinmax);}
                for (ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                    ind_gal1 = rshift_pix_gals[elreso] + pix_gals[rshift_pix_gals[elreso]+ind_inpix1];
                    innergal = isinner_resos[ind_gal1];
                    if (innergal==0){continue;}
                    z_gal1 = zbin_resos[ind_gal1];
                    pos1_gal1 = pos1_resos[ind_gal1];
                    pos2_gal1 = pos2_resos[ind_gal1];
                    w_gal1 = weight_resos[ind_gal1];
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
                                w_gal2 = weight_resos[ind_gal2];
                                wsq_gal2 = weightsq_resos[ind_gal2];
                                z_gal2 = zbin_resos[ind_gal2];
                                e1_gal2 = e1_resos[ind_gal2];
                                e2_gal2 = e2_resos[ind_gal2];
                                
                                rel1 = pos1_gal2 - pos1_gal1;
                                rel2 = pos2_gal2 - pos2_gal1;
                                dist = sqrt(rel1*rel1 + rel2*rel2);
                                if(dist < rmin_reso || dist >= rmax_reso) continue;
                                rbin = (int) floor((log(dist)-logrmin)/drbin) - rbinmin;
                                
                                wshape_gal2 = (double complex) w_gal2 * (e1_gal2+I*e2_gal2);
                                //double dphi = atan2(rel2,rel1);
                                //phirot = cexp(I*dphi);
                                phirot = (rel1+I*rel2)/dist;// * fabs(rel1)/rel1;
                                //if (rel1<0){phirot*=-1;}
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
                                nextG2ndiscs_norm[zrshift] += wsq_gal2;
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
                            if (elregion==region_debug){
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
                    for (int _i=0;_i<nbinszr_reso;_i++){nextG2ns_norm[_i]=0; nextG2ndiscs_norm[_i]=0; 
                                                        nextncounts[_i]=0; allowedrinds[_i]=0; allowedzinds[_i]=0;}
                }
                free(nextGns);
                free(nextGns_norm);
                free(nextG2ns);
                free(nextG2ns_norm);
                free(nextG2ndiscs_norm);
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
            int zcombi;
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
                                                
                                                //thisthetshift = elb2*thetashifts_z[zbin1];
                                                //gammashift = 4*(gammashift1 + elb2);
                                                //tmpGammans[gammashift] += h0*Gncache[thisthetshift + ind_Gncacheshift_mnm3];
                                                //tmpGammans[gammashift+1] += h1*Gncache[thisthetshift + ind_Gncacheshift_mnm1];
                                                //tmpGammans[gammashift+2] += h2*Gncache[thisthetshift + ind_Gncacheshift_mnm3];
                                                //tmpGammans[gammashift+3] += h3*conj(Gncache[thisthetshift + ind_Gncacheshift_nm1]);
                                                //tmpGammans_norm[gammashift1 + elb2] += w0*conj(Nncache[thisthetshift + ind_Nncacheshift]);
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
                                            //ind_Nncacheshift = thisn*nshift + zbin3*zbin2shift + zbinshifts[zbin1] +
                                            //        cumresoshift_z[zbin1*(nresos+1) + thisreso1] + elgal;
                                            //ind_Gncacheshift_mnm3 = ind_Nncacheshift + (nmax-thisn)*nshift;
                                            //ind_Gncacheshift_mnm1 = ind_Nncacheshift + (nmax-thisn+2)*nshift;
                                            //ind_Gncacheshift_nm1 = ind_Nncacheshift + (nmax+thisn+2)*nshift;
                                            //ind_Nncacheshift += thisn*nshift;
                                            ind_Nncacheshift = zbin3*zbin2shift + zbinshifts[zbin1] + rbinmin2*thetashifts_z[zbin1] +
                                                    cumresoshift_z[zbin1*(nresos+1) + thisreso1] + elgal;
                                            ind_Gncacheshift = (nmax+3)*nshift + ind_Nncacheshift;
                                            _imnm3 = (-thisn-3)*nshift + ind_Gncacheshift;
                                            _imnm1 = (-thisn-1)*nshift + ind_Gncacheshift;
                                            _inm1 = (thisn-1)*nshift + ind_Gncacheshift;
                                            _in = thisn*nshift + ind_Nncacheshift;
                                            for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                                gammashift = 4*(gammashift1 + elb2);
                                                //ind_Nncacheshift = zbin3*zbin2shift + zbinshifts[zbin1] + elb2*thetashifts_z[zbin1] +
                                                //    cumresoshift_z[zbin1*(nresos+1) + thisreso1] + elgal;
                                                //ind_Gncacheshift = (nmax+3)*nshift + ind_Nncacheshift;
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
                                                //thisthetshift = elb2*thetashifts_z[zbin1];
                                                //tmpGammans[gammashift] += mycmul(h0, wGncache[thisthetshift + ind_Gncacheshift_mnm3]);
                                                //tmpGammans[gammashift+1] += mycmul(h1,cwGncache[thisthetshift + ind_Gncacheshift_mnm1]);
                                                //tmpGammans[gammashift+2] += mycmul(h2,wGncache[thisthetshift + ind_Gncacheshift_mnm3]);
                                                //tmpGammans[gammashift+3] += mycmul_zzc(h3,cwGncache[thisthetshift + ind_Gncacheshift_nm1]);
                                                //tmpGammans_norm[gammashift1 + elb2] += mycmul_zzc(w0,wNncache[thisthetshift + ind_Nncacheshift]);
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

// Exactly the same as doubletree, but here we bruteforce the calculation of the Gn
// --> Same speed as tree and accurate on the diagonals!
void alloc_Gammans_basetree_ggg(
    int nresos, int nresos_grid, double *dpix1_resos,  double *dpix2_resos, double *reso_redges, int *ngal_resos, int nbinsz,
    int *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos, 
    double *e1_resos, double *e2_resos, int *zbin_resos, double *weightsq_resos,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int *index_matcher_hash, int nregions, int *filledregions, int nfilledregions, 
    int nmax, double rmin, double rmax, int nbinsr, int dccorr, 
    int nthreads, double *bin_centers, double complex *Gammans, double complex *Gammans_norm){
    
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
            // Check if this thread is responsible for the region
            int nthread_target = mymin(elregion/nregions_per_thread, nthreads-1);
            if (nthread_target!=elthread){continue;}
            // printf("Region %d is in thread %d\n",elregion,elthread);
            if (elregion==region_debug){printf("Region %d is in thread %d\n",elregion,elthread);}
            if (elthread==nthreads/2){
                printf("\rDone %.2f per cent",100*((double) elregion-nregions_per_thread*(int)(nthreads/2))/nregions_per_thread);
            }
            
            
            // Check which sets of radii are evaluated for each resolution
            int *reso_rindedges = calloc(nresos+1, sizeof(int));
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
            if (elregion==region_debug){
                printf("Bin edges:\n");
                for (int elreso=0;elreso<nresos;elreso++){
                    printf("  reso=%d: index_start=%d, rtarget_start=%.2f, rtrue_start=%.2f\n",
                           elreso, reso_rindedges[elreso], reso_redges[elreso], rmin*exp(reso_rindedges[elreso]*drbin));
                    printf("           index_end=%d, rtarget_end=%.2f, rtrue_end=%.2f\n",
                           reso_rindedges[elreso+1], reso_redges[elreso+1], rmin*exp(reso_rindedges[elreso+1]*drbin));
                }
            }
                        
            // Shift variables for 3pcf quantities
            int gamma_zshift = nbinsr*nbinsr;
            int gamma_nshift = gamma_zshift*nbinsz*nbinsz*nbinsz;
            int gamma_compshift = (nmax+1)*gamma_nshift;
            
            // Shift variables for spatial hash
            int npix_hash = pix1_n*pix2_n;
            int *rshift_index_matcher = calloc(nresos, sizeof(int));
            int *rshift_pixs_galind_bounds = calloc(nresos, sizeof(int));
            int *rshift_pix_gals = calloc(nresos, sizeof(int));
            for (int elreso=1;elreso<nresos;elreso++){
                rshift_index_matcher[elreso] = rshift_index_matcher[elreso-1] + npix_hash;
                rshift_pixs_galind_bounds[elreso] = rshift_pixs_galind_bounds[elreso-1] + ngal_resos[elreso-1]+1;
                rshift_pix_gals[elreso] = rshift_pix_gals[elreso-1] + ngal_resos[elreso-1];
            }
            
            // Shift variables for the matching between the pixel grids
            int lower, upper, lower1, upper1, lower2, upper2, ind_inpix, ind_gal, zbin_gal;
            int npix_side, thisreso, elreso_grid, len_matcher;
            int *matchers_resoshift = calloc(nresos_grid+1, sizeof(int));
            int *ngal_in_pix = calloc(nresos*nbinsz, sizeof(int));
            for (int elreso=0;elreso<nresos;elreso++){
                elreso_grid = elreso - hasdiscrete;
                lower = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion];
                upper = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion+1];
                for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                    ind_gal = rshift_pix_gals[elreso] + pix_gals[rshift_pix_gals[elreso]+ind_inpix];
                    ngal_in_pix[zbin_resos[ind_gal]*nresos+elreso] += 1;
                }
                if (elregion==region_debug){
                    for (int elbinz=0; elbinz<nbinsz; elbinz++){
                        printf("ngal_in_pix[elreso=%d][elz=%d] = %d \n",
                               elreso,elbinz,ngal_in_pix[elbinz*nresos+elreso]);
                    }
                }
                if (elreso_grid>=0){
                    npix_side = 1 << (nresos_grid-elreso_grid-1);
                    matchers_resoshift[elreso_grid+1] = matchers_resoshift[elreso_grid] + npix_side*npix_side; 
                }
                if (elregion==region_debug){printf("matchers_resoshift[elreso=%d] = %d \n", elreso,matchers_resoshift[elreso_grid+1]);}
            }
            len_matcher = matchers_resoshift[nresos_grid];
            
            
            // Build the matcher from pixels to reduced pixels in the region
            int elregion_fullhash, elhashpix_1, elhashpix_2, elhashpix;
            double hashpix_start1, hashpix_start2;
            double pos1_gal, pos2_gal;
            elregion_fullhash = index_matcher_hash[elregion];
            hashpix_start1 = pix1_start + (elregion_fullhash%pix1_n)*pix1_d;
            hashpix_start2 = pix2_start + (elregion_fullhash/pix1_n)*pix2_d;
            if (elregion==region_debug){
                printf("pix1_start=%.2f pix2_start=%.2f \n", pix1_start,pix2_start);
                printf("hashpix_start1=%.2f hashpix_start2=%.2f \n", hashpix_start1,hashpix_start2);}
            int *pix2redpix = calloc(nbinsz*len_matcher, sizeof(int)); // For each z matches pixel in unreduced grid to index in reduced grid
            for (int elreso=0;elreso<nresos_grid;elreso++){
                thisreso = elreso + hasdiscrete;
                lower = pixs_galind_bounds[rshift_pixs_galind_bounds[thisreso]+elregion];
                upper = pixs_galind_bounds[rshift_pixs_galind_bounds[thisreso]+elregion+1];
                npix_side = 1 << (nresos_grid-elreso-1);
                int *tmpcounts = calloc(nbinsz, sizeof(int));
                for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                    ind_gal = rshift_pix_gals[thisreso] + pix_gals[rshift_pix_gals[thisreso]+ind_inpix];
                    zbin_gal = zbin_resos[ind_gal];
                    pos1_gal = pos1_resos[ind_gal];
                    pos2_gal = pos2_resos[ind_gal];
                    elhashpix_1 = (int) floor((pos1_gal - hashpix_start1)/dpix1_resos[elreso]);
                    elhashpix_2 = (int) floor((pos2_gal - hashpix_start2)/dpix2_resos[elreso]);
                    elhashpix = elhashpix_2*npix_side + elhashpix_1;
                    //pix2redpix[zbin_gal*len_matcher+matchers_resoshift[elreso]+elhashpix] = ind_inpix-lower;
                    pix2redpix[zbin_gal*len_matcher+matchers_resoshift[elreso]+elhashpix] = tmpcounts[zbin_gal];
                    tmpcounts[zbin_gal] += 1;
                    if (elregion==region_debug){
                        printf("elreso=%d, lower=%d, thispix=%d, zgal=%d: pix2redpix[%d]=%d  \n",
                               elreso,lower,ind_inpix,zbin_gal,zbin_gal*len_matcher+matchers_resoshift[elreso]+elhashpix,ind_inpix-lower);
                    }
                }
                free(tmpcounts);
            }
            
            // Resopix2resopix
            // [resopix_reso0 --> [...id........., resopix_reso1, resopix_reso2, ..., resopix_reson],
            //  resopix_reso1 --> [....0........., ...id........, resopix_reso2, ..., resopix_reson],
            //. ...
            //  resopix_reson --> [....0........., ....0........, ....0........, ..., resopix_reson]
            // ] --> nreso*
            
            
            
            // Setup all shift variables for the Gncache in the region
            // Gncache has structure
            // n --> zbin2 --> zbin1 --> radius 
            //   --> [ [0]*ngal_zbin1_reso1 | [0]*ngal_zbin1_reso1/2 | ... | [0]*ngal_zbin1_reson ]
            int *cumresoshift_z = calloc(nbinsz*(nresos+1), sizeof(int)); // Cumulative shift index for resolution at z1
            int *thetashifts_z = calloc(nbinsz, sizeof(int)); // Shift index for theta given z1
            int *zbinshifts = calloc(nbinsz+1, sizeof(int)); // Cumulative shift index for z1
            int zbin2shift, nshift; // Shifts for z2 index and n index
            for (int elz=0; elz<nbinsz; elz++){
                if (elregion==region_debug){printf("z=%d/%d: \n", elz,nbinsz);}
                for (int elreso=0; elreso<nresos; elreso++){
                    if (elregion==region_debug){printf("  reso=%d/%d: \n", elreso,nresos);}
                    if (hasdiscrete==1 && elreso==0){
                        cumresoshift_z[elz*(nresos+1) + elreso+1] = ngal_in_pix[elz*nresos + elreso+1];
                    }
                    else{
                        cumresoshift_z[elz*(nresos+1) + elreso+1] = cumresoshift_z[elz*(nresos+1) + elreso] + ngal_in_pix[elz*nresos + elreso];
                    }
                    if (elregion==region_debug){printf("  cumresoshift_z[z][reso+1]=%d: \n", cumresoshift_z[elz*(nresos+1) + elreso+1]);}
                }
                thetashifts_z[elz] = cumresoshift_z[elz*(nresos+1) + nresos];
                zbinshifts[elz+1] = zbinshifts[elz] + nbinsr*thetashifts_z[elz];
                if ((elregion==region_debug)){printf("thetashifts_z[z]=%d: \nzbinshifts[z+1]=%d: \n", thetashifts_z[elz],  zbinshifts[elz+1]);}
            }
            zbin2shift = zbinshifts[nbinsz];
            nshift = nbinsz*zbin2shift;
            // Set all the cache indeces that are updated in this region to zero
            if ((elregion==region_debug)){printf("zbin2shift=%d: nshift=%d: \n", zbin2shift,  nshift);}
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
            int innergal, rbin, nextn, nextnshift, nbinszr, nbinszr_reso, zrshift, ind_rbin;
            double pos1_gal1, pos2_gal1, pos1_gal2, pos2_gal2, w_gal1, w_gal2, e1_gal1, e2_gal1, e1_gal2, e2_gal2;
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
                if (elregion==region_debug){printf("rbinmin=%d, rbinmax%d\n",rbinmin,rbinmax);}
                for (ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                    ind_gal1 = rshift_pix_gals[elreso] + pix_gals[rshift_pix_gals[elreso]+ind_inpix1];
                    innergal = isinner_resos[ind_gal1];
                    if (innergal==0){continue;}
                    z_gal1 = zbin_resos[ind_gal1];
                    pos1_gal1 = pos1_resos[ind_gal1];
                    pos2_gal1 = pos2_resos[ind_gal1];
                    w_gal1 = weight_resos[ind_gal1];
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
                            if (elregion==region_debug){
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
void alloc_Gammans_discrete_GNN(
    int *isinner_source, double *w_source, double *pos1_source, double *pos2_source, double *e1_source, double *e2_source, int *zbin_source, int nbinsz_source, int ngal_source,
    double *w_lens, double *pos1_lens, double *pos2_lens, int *zbin_lens, int nbinsz_lens, int ngal_lens, 
    int nmax, double rmin, double rmax, int nbinsr, int dccorr,
    int *index_matcher_source, int *pixs_galind_bounds_source, int *pix_gals_source, 
    int *index_matcher_lens, int *pixs_galind_bounds_lens, int *pix_gals_lens, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int nthreads, double *bin_centers, double complex *Upsilon_n, double complex *Norm_n){
    
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
            // printf("Region %d is in thread %d\n",elregion,elthread);
            if (elregion==region_debug){printf("Region %d is in thread %d\n",elregion,elthread);}
            if (elthread==nthreads/2){
                int elreg_inthread = elregion-nregions_per_thread*(nthreads/2);
                printf("\rDone %.2f per cent",100*((double) elreg_inthread/nregions_per_thread));
            }
            
            int zbin_gal1, zbin_gal2;
            double pos1_gal1, pos2_gal1, w_gal1, e1_gal1, e2_gal1;
            double pos1_gal2, pos2_gal2, w_gal2;
            double complex wshape_gal1;
            int ind_red, ind_gal1, ind_gal2, lower1, upper1, isinner_gal1, lower2, upper2;
            int pix1_lower, pix2_lower, pix1_upper, pix2_upper;
            lower1 = pixs_galind_bounds_source[elregion];
            upper1 = pixs_galind_bounds_source[elregion+1];
            for (int ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                
                // Load source galaxy info
                ind_gal1 = pix_gals_source[ind_inpix1];
                #pragma omp critical
                {pos1_gal1 = pos1_source[ind_gal1];
                pos2_gal1 = pos2_source[ind_gal1];
                w_gal1 = w_source[ind_gal1];
                zbin_gal1 = zbin_source[ind_gal1];
                e1_gal1 = e1_source[ind_gal1];
                e2_gal1 = e2_source[ind_gal1];
                zbin_gal1 = zbin_source[ind_gal1];
                isinner_gal1 = isinner_source[ind_gal1];}
                if(isinner_gal1==0){continue;}
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

// DoubleTree based estimtor of Source-Lens-Lens (G3L) Correlator
void alloc_Gammans_doubletree_GNN(
    int nresos, int nresos_grid, double *dpix1_resos,  double *dpix2_resos, double *reso_redges, 
    int resoshift_leafs, int minresoind_leaf, int maxresoind_leaf,
    int *isinner_source_resos, double *w_source_resos, double *pos1_source_resos, double *pos2_source_resos, 
    double *e1_source_resos, double *e2_source_resos, int *zbin_source_resos, int *ngal_source_resos, int nbinsz_source, 
    int *isinner_lens_resos, double *w_lens_resos, double *pos1_lens_resos, double *pos2_lens_resos, 
    int *zbin_lens_resos, int *ngal_lens_resos, int nbinsz_lens, 
    int nmax, double rmin, double rmax, int nbinsr, int dccorr,
    int *index_matcher_source, int *pixs_galind_bounds_source, int *pix_gals_source, 
    int *index_matcher_lens, int *pixs_galind_bounds_lens, int *pix_gals_lens, int *index_matcher_hash, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int nthreads, double *bin_centers, double complex *Upsilon_n, double complex *Norm_n){
    
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
            int region_debug=nregions/2;
            // Check if this thread is responsible for the region
            int nthread_target = mymin(elregion/nregions_per_thread, nthreads-1);
            if (nthread_target!=elthread){continue;}
            // printf("Region %d is in thread %d\n",elregion,elthread);
            if (elregion==region_debug){printf("Region %d is in thread %d\n",elregion,elthread);}
            if (elthread==nthreads/2){
                //printf("\rDone %.2f per cent",100*((double) elregion-nregions_per_thread*(int)(nthreads/2))/nregions_per_thread);
            }
            
            // Check which sets of radii are evaluated for each resolution
            int *reso_rindedges = calloc(nresos+1, sizeof(int));
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
            if (elregion==region_debug){
                printf("Bin edges:\n");
                for (int elreso=0;elreso<nresos;elreso++){
                    printf("  reso=%d: index_start=%d, rtarget_start=%.2f, rtrue_start=%.2f\n",
                           elreso, reso_rindedges[elreso], reso_redges[elreso], rmin*exp(reso_rindedges[elreso]*drbin));
                    printf("           index_end=%d, rtarget_end=%.2f, rtrue_end=%.2f\n",
                           reso_rindedges[elreso+1], reso_redges[elreso+1], rmin*exp(reso_rindedges[elreso+1]*drbin));
                }
            }
            
            // Shift variables for spatial hash of sources and lenses
            int npix_hash = pix1_n*pix2_n;
            int *rshift_index_matcher_source = calloc(nresos, sizeof(int));
            int *rshift_pixs_galind_bounds_source = calloc(nresos, sizeof(int));
            int *rshift_pix_gals_source = calloc(nresos, sizeof(int));
            for (int elreso=1;elreso<nresos;elreso++){
                rshift_index_matcher_source[elreso] = rshift_index_matcher_source[elreso-1] + npix_hash;
                rshift_pixs_galind_bounds_source[elreso] = rshift_pixs_galind_bounds_source[elreso-1] + ngal_source_resos[elreso-1]+1;
                rshift_pix_gals_source[elreso] = rshift_pix_gals_source[elreso-1] + ngal_source_resos[elreso-1];
            }
            int *rshift_index_matcher_lens = calloc(nresos, sizeof(int));
            int *rshift_pixs_galind_bounds_lens = calloc(nresos, sizeof(int));
            int *rshift_pix_gals_lens = calloc(nresos, sizeof(int));
            for (int elreso=1;elreso<nresos;elreso++){
                rshift_index_matcher_lens[elreso] = rshift_index_matcher_lens[elreso-1] + npix_hash;
                rshift_pixs_galind_bounds_lens[elreso] = rshift_pixs_galind_bounds_lens[elreso-1] + ngal_lens_resos[elreso-1]+1;
                rshift_pix_gals_lens[elreso] = rshift_pix_gals_lens[elreso-1] + ngal_lens_resos[elreso-1];
            }
            
            // Shift variables for the matching between the pixel grids (only needed for sources!)
            int lower, upper, ind_inpix, ind_gal, zbin_gal;
            int npix_side, thisreso, elreso_grid, len_matcher;
            int *matchers_resoshift = calloc(nresos_grid+1, sizeof(int));
            int *ngal_in_pix = calloc(nresos*nbinsz_source, sizeof(int));
            for (int elreso=0;elreso<nresos;elreso++){
                elreso_grid = elreso - hasdiscrete;
                lower = pixs_galind_bounds_source[rshift_pixs_galind_bounds_source[elreso]+elregion];
                upper = pixs_galind_bounds_source[rshift_pixs_galind_bounds_source[elreso]+elregion+1];
                for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                    ind_gal = rshift_pix_gals_source[elreso] + pix_gals_source[rshift_pix_gals_source[elreso]+ind_inpix];
                    ngal_in_pix[zbin_source_resos[ind_gal]*nresos+elreso] += 1;
                }
                if (elregion==region_debug){
                    for (int elbinz=0; elbinz<nbinsz_source; elbinz++){
                        printf("ngal_in_pix[elreso=%d][elz=%d] = %d \n",
                               elreso,elbinz,ngal_in_pix[elbinz*nresos+elreso]);
                    }
                }
                if (elreso_grid>=0){
                    npix_side = 1 << (nresos_grid-elreso_grid-1);
                    matchers_resoshift[elreso_grid+1] = matchers_resoshift[elreso_grid] + npix_side*npix_side; 
                }
                if (elregion==region_debug){printf("matchers_resoshift[elreso=%d] = %d \n", elreso,matchers_resoshift[elreso_grid+1]);}
            }
            len_matcher = matchers_resoshift[nresos_grid];
            
            // Build the matcher from pixels to reduced pixels in the region (only needed for sources!)
            int elregion_fullhash, elhashpix_1, elhashpix_2, elhashpix;
            double hashpix_start1, hashpix_start2;
            double pos1_gal, pos2_gal;
            elregion_fullhash = index_matcher_hash[elregion];
            hashpix_start1 = pix1_start + (elregion_fullhash%pix1_n)*pix1_d;
            hashpix_start2 = pix2_start + (elregion_fullhash/pix1_n)*pix2_d;
            if (elregion==region_debug){
                printf("elregion=%d, elregion_fullhash=%d, pix1_start=%.2f pix2_start=%.2f \n", elregion,elregion_fullhash,pix1_start,pix2_start);
                printf("hashpix_start1=%.2f hashpix_start2=%.2f \n", hashpix_start1,hashpix_start2);}
            int *pix2redpix = calloc(nbinsz_source*len_matcher, sizeof(int)); // For each z matches pixel in unreduced grid to index in reduced grid
            if (elregion==region_debug){printf("pix2redpix has length = %d \n",nbinsz_source*len_matcher);}
            for (int elreso=0;elreso<nresos_grid;elreso++){
                thisreso = elreso + hasdiscrete;
                lower = pixs_galind_bounds_source[rshift_pixs_galind_bounds_source[thisreso]+elregion];
                upper = pixs_galind_bounds_source[rshift_pixs_galind_bounds_source[thisreso]+elregion+1];
                npix_side = 1 << (nresos_grid-elreso-1);
                int *tmpcounts = calloc(nbinsz_source, sizeof(int));
                for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                    ind_gal = rshift_pix_gals_source[thisreso] + pix_gals_source[rshift_pix_gals_source[thisreso]+ind_inpix];
                    zbin_gal = zbin_source_resos[ind_gal];
                    pos1_gal = pos1_source_resos[ind_gal];
                    pos2_gal = pos2_source_resos[ind_gal];
                    elhashpix_1 = (int) floor((pos1_gal - hashpix_start1)/dpix1_resos[elreso]);
                    elhashpix_2 = (int) floor((pos2_gal - hashpix_start2)/dpix2_resos[elreso]);
                    elhashpix = elhashpix_2*npix_side + elhashpix_1;
                    pix2redpix[zbin_gal*len_matcher+matchers_resoshift[elreso]+elhashpix] = tmpcounts[zbin_gal];
                    tmpcounts[zbin_gal] += 1;
                    if (elregion==region_debug){
                        printf("elreso=%d, lower=%d, thispix=%d, elhashpix=%d %d %d, zgal=%d: pix2redpix[%d=%d+%d+%d*%d+%d]=%d  \n", 
                               elreso, lower,ind_inpix,elhashpix_1,elhashpix_2,elhashpix, zbin_gal,
                               zbin_gal*len_matcher+matchers_resoshift[elreso]+elhashpix,
                               zbin_gal*len_matcher,matchers_resoshift[elreso],elhashpix_2,npix_side,elhashpix_1,
                               ind_inpix-lower);
                    }
                }
                free(tmpcounts);
            }
            
            // Setup all shift variables for the Gncache in the region
            // Gncache has structure
            // n --> zbin_lens --> zbin_source --> radius 
            //   --> [ [0]*ngal_zbin1_reso1 | [0]*ngal_zbin1_reso1/2 | ... | [0]*ngal_zbin1_reson ]
            int *cumresoshift_z = calloc(nbinsz_source*(nresos+1), sizeof(int)); // Cumulative shift index for resolution at z1
            int *thetashifts_z = calloc(nbinsz_source, sizeof(int)); // Shift index for theta given z1
            int *zbinshifts = calloc(nbinsz_source+1, sizeof(int)); // Cumulative shift index for z1
            int zbin2shift, nshift; // Shifts for z2 index and n index
            for (int elz=0; elz<nbinsz_source; elz++){
                if (elregion==region_debug){printf("z=%d/%d: \n", elz,nbinsz_source);}
                for (int elreso=0; elreso<nresos; elreso++){
                    if (elregion==region_debug){printf("  reso=%d/%d: \n", elreso,nresos);}
                    if (hasdiscrete==1 && elreso==0){
                        cumresoshift_z[elz*(nresos+1) + elreso+1] = ngal_in_pix[elz*nresos + elreso+1];
                    }
                    else{
                        cumresoshift_z[elz*(nresos+1) + elreso+1] = cumresoshift_z[elz*(nresos+1) + elreso] + ngal_in_pix[elz*nresos + elreso];
                    }
                    if (elregion==region_debug){printf("  cumresoshift_z[z][reso+1]=%d: \n", cumresoshift_z[elz*(nresos+1) + elreso+1]);}
                }
                thetashifts_z[elz] = cumresoshift_z[elz*(nresos+1) + nresos];
                zbinshifts[elz+1] = zbinshifts[elz] + nbinsr*thetashifts_z[elz];
                if ((elregion==region_debug)){printf("thetashifts_z[z]=%d: \nzbinshifts[z+1]=%d: \n", thetashifts_z[elz],  zbinshifts[elz+1]);}
            }
            zbin2shift = zbinshifts[nbinsz_source];
            nshift = nbinsz_lens*zbin2shift;
            size_max_nshift = mymax(nshift, size_max_nshift);
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
        printf("Thread %i: nshift=%i, nshift_theo=%i",elthread,size_max_nshift,size_max_nshift_theo);
            
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
            // printf("Region %d is in thread %d\n",elregion,elthread);
            if (elregion==region_debug){printf("Region %d is in thread %d\n",elregion,elthread);}
            if (elthread==nthreads/2){
                //printf("\rDone %.2f per cent",100*((double) elregion-nregions_per_thread*(int)(nthreads/2))/nregions_per_thread);
            }
            
            // Check which sets of radii are evaluated for each resolution
            int *reso_rindedges = calloc(nresos+1, sizeof(int));
            double logrmin = log(rmin);
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
            if (elregion==region_debug){
                printf("Bin edges:\n");
                for (int elreso=0;elreso<nresos;elreso++){
                    printf("  reso=%d: index_start=%d, rtarget_start=%.2f, rtrue_start=%.2f\n",
                           elreso, reso_rindedges[elreso], reso_redges[elreso], rmin*exp(reso_rindedges[elreso]*drbin));
                    printf("           index_end=%d, rtarget_end=%.2f, rtrue_end=%.2f\n",
                           reso_rindedges[elreso+1], reso_redges[elreso+1], rmin*exp(reso_rindedges[elreso+1]*drbin));
                }
            }
            
            // Shift variables for spatial hash of sources and lenses
            int npix_hash = pix1_n*pix2_n;
            int *rshift_index_matcher_source = calloc(nresos, sizeof(int));
            int *rshift_pixs_galind_bounds_source = calloc(nresos, sizeof(int));
            int *rshift_pix_gals_source = calloc(nresos, sizeof(int));
            for (int elreso=1;elreso<nresos;elreso++){
                rshift_index_matcher_source[elreso] = rshift_index_matcher_source[elreso-1] + npix_hash;
                rshift_pixs_galind_bounds_source[elreso] = rshift_pixs_galind_bounds_source[elreso-1] + ngal_source_resos[elreso-1]+1;
                rshift_pix_gals_source[elreso] = rshift_pix_gals_source[elreso-1] + ngal_source_resos[elreso-1];
            }
            int *rshift_index_matcher_lens = calloc(nresos, sizeof(int));
            int *rshift_pixs_galind_bounds_lens = calloc(nresos, sizeof(int));
            int *rshift_pix_gals_lens = calloc(nresos, sizeof(int));
            for (int elreso=1;elreso<nresos;elreso++){
                rshift_index_matcher_lens[elreso] = rshift_index_matcher_lens[elreso-1] + npix_hash;
                rshift_pixs_galind_bounds_lens[elreso] = rshift_pixs_galind_bounds_lens[elreso-1] + ngal_lens_resos[elreso-1]+1;
                rshift_pix_gals_lens[elreso] = rshift_pix_gals_lens[elreso-1] + ngal_lens_resos[elreso-1];
            }
            
            // Shift variables for the matching between the pixel grids (only needed for sources!)
            int lower, upper, lower1, upper1, lower2, upper2, ind_inpix, ind_gal, zbin_gal;
            int npix_side, thisreso, elreso_grid, len_matcher;
            int *matchers_resoshift = calloc(nresos_grid+1, sizeof(int));
            int *ngal_in_pix = calloc(nresos*nbinsz_source, sizeof(int));
            for (int elreso=0;elreso<nresos;elreso++){
                elreso_grid = elreso - hasdiscrete;
                lower = pixs_galind_bounds_source[rshift_pixs_galind_bounds_source[elreso]+elregion];
                upper = pixs_galind_bounds_source[rshift_pixs_galind_bounds_source[elreso]+elregion+1];
                for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                    ind_gal = rshift_pix_gals_source[elreso] + pix_gals_source[rshift_pix_gals_source[elreso]+ind_inpix];
                    ngal_in_pix[zbin_source_resos[ind_gal]*nresos+elreso] += 1;
                }
                if (elregion==region_debug){
                    for (int elbinz=0; elbinz<nbinsz_source; elbinz++){
                        printf("ngal_in_pix[elreso=%d][elz=%d] = %d \n",
                               elreso,elbinz,ngal_in_pix[elbinz*nresos+elreso]);
                    }
                }
                if (elreso_grid>=0){
                    npix_side = 1 << (nresos_grid-elreso_grid-1);
                    matchers_resoshift[elreso_grid+1] = matchers_resoshift[elreso_grid] + npix_side*npix_side; 
                }
                if (elregion==region_debug){printf("matchers_resoshift[elreso=%d] = %d \n", elreso,matchers_resoshift[elreso_grid+1]);}
            }
            len_matcher = matchers_resoshift[nresos_grid];
            
            // Build the matcher from pixels to reduced pixels in the region (only needed for sources!)
            int elregion_fullhash, elhashpix_1, elhashpix_2, elhashpix;
            double hashpix_start1, hashpix_start2;
            double pos1_gal, pos2_gal;
            elregion_fullhash = index_matcher_hash[elregion];
            hashpix_start1 = pix1_start + (elregion_fullhash%pix1_n)*pix1_d;
            hashpix_start2 = pix2_start + (elregion_fullhash/pix1_n)*pix2_d;
            if (elregion==region_debug){
                printf("elregion=%d, elregion_fullhash=%d, pix1_start=%.2f pix2_start=%.2f \n", elregion,elregion_fullhash,pix1_start,pix2_start);
                printf("hashpix_start1=%.2f hashpix_start2=%.2f \n", hashpix_start1,hashpix_start2);}
            int *pix2redpix = calloc(nbinsz_source*len_matcher, sizeof(int)); // For each z matches pixel in unreduced grid to index in reduced grid
            
            for (int elreso=0;elreso<nresos_grid;elreso++){
                thisreso = elreso + hasdiscrete;
                lower = pixs_galind_bounds_source[rshift_pixs_galind_bounds_source[thisreso]+elregion];
                upper = pixs_galind_bounds_source[rshift_pixs_galind_bounds_source[thisreso]+elregion+1];
                npix_side = 1 << (nresos_grid-elreso-1);
                int *tmpcounts = calloc(nbinsz_source, sizeof(int));
                for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                    ind_gal = rshift_pix_gals_source[thisreso] + pix_gals_source[rshift_pix_gals_source[thisreso]+ind_inpix];
                    zbin_gal = zbin_source_resos[ind_gal];
                    pos1_gal = pos1_source_resos[ind_gal];
                    pos2_gal = pos2_source_resos[ind_gal];
                    elhashpix_1 = (int) floor((pos1_gal - hashpix_start1)/dpix1_resos[elreso]);
                    elhashpix_2 = (int) floor((pos2_gal - hashpix_start2)/dpix2_resos[elreso]);
                    elhashpix = elhashpix_2*npix_side + elhashpix_1;
                    pix2redpix[zbin_gal*len_matcher+matchers_resoshift[elreso]+elhashpix] = tmpcounts[zbin_gal];
                    tmpcounts[zbin_gal] += 1;
                    if (elregion==region_debug){
                        printf("elreso=%d, lower=%d, thispix=%d, elhashpix=%d %d %d, zgal=%d: pix2redpix[%d]=%d  \n",
                               elreso,lower,ind_inpix,elhashpix_1,elhashpix_2,elhashpix,zbin_gal,zbin_gal*len_matcher+matchers_resoshift[elreso]+elhashpix,ind_inpix-lower);
                    }
                }
                free(tmpcounts);
            }
            
            // Setup all shift variables for the Gncache in the region
            // Gncache has structure
            // n --> zbin_lens --> zbin_source --> radius 
            //   --> [ [0]*ngal_zbin1_reso1 | [0]*ngal_zbin1_reso1/2 | ... | [0]*ngal_zbin1_reson ]
            int *cumresoshift_z = calloc(nbinsz_source*(nresos+1), sizeof(int)); // Cumulative shift index for resolution at z1
            int *thetashifts_z = calloc(nbinsz_source, sizeof(int)); // Shift index for theta given z1
            int *zbinshifts = calloc(nbinsz_source+1, sizeof(int)); // Cumulative shift index for z1
            int zbin2shift, nshift; // Shifts for z2 index and n index
            for (int elz=0; elz<nbinsz_source; elz++){
                if (elregion==region_debug){printf("z=%d/%d: \n", elz,nbinsz_source);}
                for (int elreso=0; elreso<nresos; elreso++){
                    if (elregion==region_debug){printf("  reso=%d/%d: \n", elreso,nresos);}
                    if (hasdiscrete==1 && elreso==0){
                        cumresoshift_z[elz*(nresos+1) + elreso+1] = ngal_in_pix[elz*nresos + elreso+1];
                    }
                    else{
                        cumresoshift_z[elz*(nresos+1) + elreso+1] = cumresoshift_z[elz*(nresos+1) + elreso] + ngal_in_pix[elz*nresos + elreso];
                    }
                    if (elregion==region_debug){printf("  cumresoshift_z[z][reso+1]=%d: \n", cumresoshift_z[elz*(nresos+1) + elreso+1]);}
                }
                thetashifts_z[elz] = cumresoshift_z[elz*(nresos+1) + nresos];
                zbinshifts[elz+1] = zbinshifts[elz] + nbinsr*thetashifts_z[elz];
                if ((elregion==region_debug)){printf("thetashifts_z[z]=%d: \nzbinshifts[z+1]=%d: \n", thetashifts_z[elz],  zbinshifts[elz+1]);}
            }
            zbin2shift = zbinshifts[nbinsz_source];
            nshift = nbinsz_lens*zbin2shift;
            // Set all the cache indices that are updated in this region to zero
            if ((elregion==region_debug)){printf("zbin2shift=%d: nshift=%d: size_max_nshift=%d \n", zbin2shift, nshift, size_max_nshift);}
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
            int innergal, nbinszr_reso;
            double pos1_gal1, pos2_gal1, pos1_gal2, pos2_gal2, w_gal1, w_gal2, e1_gal1, e2_gal1;
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
                if (elregion==region_debug){printf("rbinmin=%d, rbinmax%d\n",rbinmin,rbinmax);}
                int ind_Wn, ind_counts, z1shift, z2rshift, rbin;
                for (ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                    ind_gal1 = rshift_pix_gals_source[elreso] + pix_gals_source[rshift_pix_gals_source[elreso]+ind_inpix1];
                    innergal = isinner_source_resos[ind_gal1];
                    if (innergal==0){continue;}
                    z_gal1 = zbin_source_resos[ind_gal1];
                    pos1_gal1 = pos1_source_resos[ind_gal1];
                    pos2_gal1 = pos2_source_resos[ind_gal1];
                    w_gal1 = w_source_resos[ind_gal1];
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
                                
                                //if (rbin<0 || rbin>=nbinsr){
                                //    printf("%.2f %d",dist,rbin);
                                //    continue;}

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
                            if (elregion==-1){
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
                                                //ind_Wncacheshift = zbin3*zbin2shift + zbinshifts[zbin1] + elb2*thetashifts_z[zbin1] +
                                                //    cumresoshift_z[zbin1*(nresos+1) + thisreso2] + elgal;
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
void alloc_Gammans_discrete_NGG(
    double *w_source, double *pos1_source, double *pos2_source, double *e1_source, double *e2_source, int *zbin_source, int nbinsz_source, int ngal_source,
    int *isinner_lens, double *w_lens, double *pos1_lens, double *pos2_lens, int *zbin_lens, int nbinsz_lens, int ngal_lens, 
    int nmax, double rmin, double rmax, int nbinsr, int dccorr,
    int *index_matcher_source, int *pixs_galind_bounds_source, int *pix_gals_source, 
    int *index_matcher_lens, int *pixs_galind_bounds_lens, int *pix_gals_lens, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int nthreads, double *bin_centers, double complex *Upsilon_n, double complex *Norm_n){
    
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
            //int region_debug=(int) (nthreads/2) * nregions_per_thread;
            //int region_debug=99999;
            // Check if this thread is responsible for the region
            int nthread_target = mymin(elregion/nregions_per_thread, nthreads-1);
            if (nthread_target!=elthread){continue;}
            //if (elregion==region_debug){printf("Region %d is in thread %d\n",elregion,elthread);}
            if (elthread==nthreads/2){
                int elreg_inthread = elregion-nregions_per_thread*(nthreads/2);
                printf("\rDone %.2f per cent",100*((double) elreg_inthread)/nregions_per_thread);
            }
            
            int zbin_gal1, zbin_gal2;
            double pos1_gal1, pos2_gal1, w_gal1;
            double pos1_gal2, pos2_gal2, w_gal2, e1_gal2, e2_gal2;
            double complex wshape_gal2;
            int ind_red, ind_gal1, ind_gal2, lower1, upper1, isinner_gal1, lower2, upper2;
            int pix1_lower, pix2_lower, pix1_upper, pix2_upper;
            lower1 = pixs_galind_bounds_lens[elregion];
            upper1 = pixs_galind_bounds_lens[elregion+1];
            for (int ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                
                // Load lens galaxy info
                ind_gal1 = pix_gals_lens[ind_inpix1];
                #pragma omp critical
                {pos1_gal1 = pos1_lens[ind_gal1];
                pos2_gal1 = pos2_lens[ind_gal1];
                w_gal1 = w_lens[ind_gal1];
                zbin_gal1 = zbin_lens[ind_gal1];
                zbin_gal1 = zbin_lens[ind_gal1];
                isinner_gal1 = isinner_lens[ind_gal1];}
                if(isinner_gal1==0){continue;}
                
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
                            if (rbin<0 || rbin>=nbinsr){
                                printf("%.2f %d",dist,rbin);
                                continue;
                            }
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
                // Upsilon_-(thet1, thet2) ~ w * G_{+n-2}(thet1) * G_{-n-2}(thet2) + delta^K_{thet1,thet2} * (w * (we)^2*exp(4*phi))
                // Upsilon_+(thet1, thet2) ~ w * G_{+n-2}(thet1) * conj(G_{+n-2})(thet2) + delta^K_{thet1,thet2} * (w * |we|^2)
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
    printf("Accumulate Upsilon\n");
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
    printf("Accumulate bin centers \n");
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
    printf("Finalize bin centers \n");
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

// Discrete estimator of Lens-Source-Source Correlator using a fixed grid of angles within the annuli
// This helps keeping a good angular resolution while not having to care much about the radial resolution
// The problem is that we still need to assign the full catlalog to the phigrid at runtime. This means 
// that while the Gn allocation only happens on the `reduced' catalog, the preparation of this `reduced'
// catalog still scales as O(N_gal). Even worse, the size of the reduced catalog is nbinsr*nphi*nbinsz which
// can very well approach the size of the full catalog....thus, this is not intended to be used.
/*void alloc_Gammans_phitree_NGG(
    double *w_source, double *pos1_source, double *pos2_source, double *e1_source, double *e2_source, int *zbin_source, int nbinsz_source, int ngal_source,
    int *isinner_lens, double *w_lens, double *pos1_lens, double *pos2_lens, int *zbin_lens, int nbinsz_lens, int ngal_lens, 
    int nmax, double rmin, double rmax, int nbinsr, int dccorr,
    int *index_matcher_source, int *pixs_galind_bounds_source, int *pix_gals_source, 
    int *index_matcher_lens, int *pixs_galind_bounds_lens, int *pix_gals_lens, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int *phitree_bins, int phitree_nphi, int phitree_nx, int phitree_ny,
    int nthreads, double *bin_centers, double complex *Upsilon_n, double complex *Norm_n){
    
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
        int ngalmax_ap = nbinsz_source*nbinsr*phitree_nphi;
        
        int *inds_galinap = calloc(ngalmax_ap, sizeof(int));
        double *apgals_w = calloc(ngalmax_ap, sizeof(double));
        double *apgals_rel1 = calloc(ngalmax_ap, sizeof(double));
        double *apgals_rel2 = calloc(ngalmax_ap, sizeof(double));
        double complex *apgals_wshape = calloc(ngalmax_ap, sizeof(double complex));                                
        
        for (int elregion=0; elregion<nregions; elregion++){
            int region_debug=(int) (nthreads/2) * nregions_per_thread;
            //int region_debug=99999;
            // Check if this thread is responsible for the region
            int nthread_target = mymin(elregion/nregions_per_thread, nthreads-1);
            if (nthread_target!=elthread){continue;}
            //if (elregion==region_debug){printf("Region %d is in thread %d\n",elregion,elthread);}
            if (elthread==nthreads/2){
                int elreg_inthread = elregion-nregions_per_thread*(nthreads/2);
                printf("\rDone %.2f per cent",100*((double) elreg_inthread)/nregions_per_thread);
            }
            
            int zbin_gal1, zbin_gal2;
            double pos1_gal1, pos2_gal1, w_gal1;
            double pos1_gal2, pos2_gal2, w_gal2, e1_gal2, e2_gal2;
            double complex wshape_gal2;
            int ind_red, ind_gal1, ind_gal2, lower1, upper1, isinner_gal1, lower2, upper2;
            int pix1_lower, pix2_lower, pix1_upper, pix2_upper;
            lower1 = pixs_galind_bounds_lens[elregion];
            upper1 = pixs_galind_bounds_lens[elregion+1];
            for (int ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                
                // Load lens galaxy info
                ind_gal1 = pix_gals_lens[ind_inpix1];
                #pragma omp critical
                {pos1_gal1 = pos1_lens[ind_gal1];
                pos2_gal1 = pos2_lens[ind_gal1];
                w_gal1 = w_lens[ind_gal1];
                zbin_gal1 = zbin_lens[ind_gal1];
                zbin_gal1 = zbin_lens[ind_gal1];
                isinner_gal1 = isinner_lens[ind_gal1];}
                if(isinner_gal1==0){continue;}
                
                // Allocate the G_n and W_n coefficients + Double-counting correction factors
                double complex phirot, nphirot;
                double rel1, rel2, dist;
                int ind_Wnp, ind_Wnm, ind_Gnp, ind_Gnm, ind_counts, z1shift, z2rshift, rbin, rbin_gal2;
                int phibin1, phibin2, phibin, ind_galinap;
                double complex *thisGns = calloc(nnvals_Gn*nbinszr_Gn, sizeof(double complex)); 
                double complex *thisWns = calloc(nnvals_Wn*nbinszr_Wn, sizeof(double complex)); 
                double complex *thisG2ns = calloc(2*nbinszr_Gn, sizeof(double complex));
                double complex *thisW2ns = calloc(nbinszr_Wn, sizeof(double complex));
                int *thisncounts = calloc(nbinszr_Wn, sizeof(int));
                int *allowedrinds = calloc(nbinszr_Wn, sizeof(int));
                int *allowedzinds = calloc(nbinszr_Wn, sizeof(int));
                z1shift = zbin_gal1*nbinsz_source*nbinsr;
                
                // First step: Put individual galaxies in phi bins
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
                            phibin1 = (int) ((rel1/dist)*phitree_nx);
                            phibin2 = (int) ((rel2/dist)*phitree_ny);
                            phibin = phitree_bins[phibin2*phitree_nx+phibin1];
                            // Add to galaxy in phibin
                            ind_galinap = zbin_gal2*nbinsr*phitree_nphi + rbin*phitree_nphi + phibin;
                            apgals_w[ind_galinap] += w_gal2;
                            apgals_rel1[ind_galinap] += w_gal2*rel1;
                            apgals_rel2[ind_galinap] += w_gal2*rel2;
                            apgals_wshape[ind_galinap] += w_gal2*(e1_gal2+I*e2_gal2);
                        }
                    }
                }
                // Second step: Filter for nonempty bins
                int nnonempty = 0;
                for (int elbinz=0; elbinz<nbinsz_source; elbinz++){ 
                    for (int rbin=0; rbin<nbinsr; rbin++){ 
                        for (int phibin=0; phibin<phitree_nphi; phibin++){ 
                            ind_galinap = elbinz*nbinsr*phitree_nphi + rbin*phitree_nphi + phibin;
                            if (apgals_w>0){
                                inds_galinap[nnonempty] = ind_galinap;
                                apgals_rel1[ind_galinap] /= apgals_w[ind_galinap];
                                apgals_rel2[ind_galinap] /= apgals_w[ind_galinap];
                                nnonempty += 1;
                            }
                        }
                    }
                }
                // Third step: Allocate the Gn
                for (int elgal=0; elgal<nnonempty; elgal++){
                    ind_galinap = inds_galinap[elgal];
                    zbin_gal2 = (int) (ind_galinap/(nbinsr*phitree_nphi));
                    rbin_gal2 = (int) ((ind_galinap-zbin_gal2*nbinsr*phitree_nphi)/phitree_nphi);
                    
                    rel1 = apgals_rel1[ind_galinap];
                    rel2 = apgals_rel2[ind_galinap];
                    dist = sqrt(rel1*rel1 + rel2*rel2);
                    w_gal2 = apgals_w[ind_galinap];
                    wshape_gal2 = apgals_wshape[ind_galinap];
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
                // Upsilon_-(thet1, thet2) ~ w * G_{+n-2}(thet1) * G_{-n-2}(thet2) + delta^K_{thet1,thet2} * (w * (we)^2*exp(4*phi))
                // Upsilon_+(thet1, thet2) ~ w * G_{+n-2}(thet1) * conj(G_{+n-2})(thet2) + delta^K_{thet1,thet2} * (w * |we|^2)
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
                // Set phigrids to zero
                for (int tmpind=0; tmpind<nnonempty; tmpind++){
                    inds_galinap[tmpind]=0;
                    apgals_w[tmpind]=0;
                    apgals_rel1[tmpind]=0;
                    apgals_rel2[tmpind]=0;
                    apgals_wshape[tmpind]=0;
                }
                
                
                //free stuff
                free(thisWns);
                free(thisGns);
                free(thisG2ns);
                free(thisW2ns);
                free(thisncounts);
                free(allowedrinds);
                free(allowedzinds);
                
                
                
            }
        }
        free(inds_galinap);
        free(apgals_w);
        free(apgals_rel1);
        free(apgals_rel2);
        free(apgals_wshape);
    }
    
    // Accumulate the Upsilon_n / N_n
    printf("Accumulate Upsilon\n");
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
    printf("Accumulate bin centers \n");
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
    printf("Finalize bin centers \n");
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
*/        

// Discrete estimator of Lens-Source-Source Correlator
void alloc_Gammans_tree_NGG(
        int nresos, double *reso_redges, 
        double *w_source_resos, double *pos1_source_resos, double *pos2_source_resos,
        double *e1_source_resos, double *e2_source_resos, int *zbin_source_resos, int nbinsz_source, int *ngal_source_resos,
        int *isinner_lens, double *w_lens, double *pos1_lens, double *pos2_lens, int *zbin_lens, int nbinsz_lens, int ngal_lens, 
        int *index_matcher_source, int *pixs_galind_bounds_source, int *pix_gals_source, 
        int *index_matcher_lens, int *pixs_galind_bounds_lens, int *pix_gals_lens, int nregions, 
        double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
        int nmax, double rmin, double rmax, int nbinsr, int dccorr,
        int nthreads, double *bin_centers, double complex *Upsilon_n, double complex *Norm_n){
    
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
        for (int elreso=1;elreso<nresos;elreso++){
            rshift_index_matcher[elreso] = rshift_index_matcher[elreso-1] + npix_hash;
            rshift_pixs_galind_bounds[elreso] = rshift_pixs_galind_bounds[elreso-1] + ngal_source_resos[elreso-1]+1;
            rshift_pix_gals[elreso] = rshift_pix_gals[elreso-1] + ngal_source_resos[elreso-1];
        }
        
        
        for (int elregion=0; elregion<nregions; elregion++){
            //int region_debug=(int) (nthreads/2) * nregions_per_thread;
            //int region_debug=99999;
            // Check if this thread is responsible for the region
            int nthread_target = mymin(elregion/nregions_per_thread, nthreads-1);
            if (nthread_target!=elthread){continue;}
            //if (elregion==region_debug){printf("Region %d is in thread %d\n",elregion,elthread);}
            if (elthread==nthreads/2){
                int elreg_inthread = elregion-nregions_per_thread*(nthreads/2);
                printf("\rDone %.2f per cent",100*((double) elreg_inthread)/nregions_per_thread);
            }
            
            int zbin_gal1, zbin_gal2;
            double pos1_gal1, pos2_gal1, w_gal1;
            double pos1_gal2, pos2_gal2, w_gal2, e1_gal2, e2_gal2;
            double complex wshape_gal2;
            int ind_red, ind_gal1, ind_gal2, lower1, upper1, isinner_gal1, lower2, upper2;
            int ind_inpix1, ind_inpix2;
            lower1 = pixs_galind_bounds_lens[elregion];
            upper1 = pixs_galind_bounds_lens[elregion+1];
            for (ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                
                // Load lens galaxy info
                ind_gal1 = pix_gals_lens[ind_inpix1];
                #pragma omp critical
                {pos1_gal1 = pos1_lens[ind_gal1];
                pos2_gal1 = pos2_lens[ind_gal1];
                w_gal1 = w_lens[ind_gal1];
                zbin_gal1 = zbin_lens[ind_gal1];
                zbin_gal1 = zbin_lens[ind_gal1];
                isinner_gal1 = isinner_lens[ind_gal1];}
                if(isinner_gal1==0){continue;}
                
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
                // Upsilon_-(thet1, thet2) ~ w * G_{+n-2}(thet1) * G_{-n-2}(thet2) + delta^K_{thet1,thet2} * (w * (we)^2*exp(4*phi))
                // Upsilon_+(thet1, thet2) ~ w * G_{+n-2}(thet1) * conj(G_{+n-2})(thet2) + delta^K_{thet1,thet2} * (w * |we|^2)
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
    printf("Accumulate Upsilon\n");
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
    printf("Accumulate bin centers \n");
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
    printf("Finalize bin centers \n");
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
void alloc_Gammans_doubletree_NGG(
    int nresos, int nresos_grid, double *dpix1_resos,  double *dpix2_resos, double *reso_redges, 
    int resoshift_leafs, int minresoind_leaf, int maxresoind_leaf,
    int *isinner_source_resos, double *w_source_resos, double *pos1_source_resos, double *pos2_source_resos, 
    double *e1_source_resos, double *e2_source_resos, int *zbin_source_resos, int *ngal_source_resos, int nbinsz_source, 
    int *isinner_lens_resos, double *w_lens_resos, double *pos1_lens_resos, double *pos2_lens_resos, 
    int *zbin_lens_resos, int *ngal_lens_resos, int nbinsz_lens, 
    int nmax, double rmin, double rmax, int nbinsr, int dccorr,
    int *index_matcher_source, int *pixs_galind_bounds_source, int *pix_gals_source, 
    int *index_matcher_lens, int *pixs_galind_bounds_lens, int *pix_gals_lens, int *index_matcher_hash, int nregions, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int nthreads, double *bin_centers, double complex *Upsilon_n, double complex *Norm_n){
    
    int _ncomp_Upsilon = 2;
    int _nzcombis = nbinsz_lens*nbinsz_source*nbinsz_source;
    int _upsilonzshift = nbinsr*nbinsr;
    int _upsilonnshift = _upsilonzshift*_nzcombis;
    int _upsiloncompshift = (2*nmax+1)*_upsilonnshift;
    int _upsilonthreadshift = _ncomp_Upsilon*_upsiloncompshift;
    int _normzshift = nbinsr*nbinsr;
    int _normnshift = _normzshift*_nzcombis;
    int _normthreadshift = (2*nmax+1)*_normnshift;   
    
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
        for (int elregion=0; elregion<nregions; elregion++){
            int region_debug=-1;
            // Check if this thread is responsible for the region
            int nthread_target = mymin(elregion/nregions_per_thread, nthreads-1);
            if (nthread_target!=elthread){continue;}
            // printf("Region %d is in thread %d\n",elregion,elthread);
            //if (elregion==region_debug){printf("Region %d is in thread %d\n",elregion,elthread);}
            if (elthread==nthreads/2){
                //printf("\rDone %.2f per cent",100*((double) elregion-nregions_per_thread*(int)(nthreads/2))/nregions_per_thread);
            }
            
            // Check which sets of radii are evaluated for each resolution
            int *reso_rindedges = calloc(nresos+1, sizeof(int));
            double logrmin = log(rmin);
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
            if (elregion==region_debug){
                printf("Bin edges:\n");
                for (int elreso=0;elreso<nresos;elreso++){
                    printf("  reso=%d: index_start=%d, rtarget_start=%.2f, rtrue_start=%.2f\n",
                           elreso, reso_rindedges[elreso], reso_redges[elreso], rmin*exp(reso_rindedges[elreso]*drbin));
                    printf("           index_end=%d, rtarget_end=%.2f, rtrue_end=%.2f\n",
                           reso_rindedges[elreso+1], reso_redges[elreso+1], rmin*exp(reso_rindedges[elreso+1]*drbin));
                }
            }
            
            // Shift variables for spatial hash of sources and lenses
            int npix_hash = pix1_n*pix2_n;
            int *rshift_index_matcher_source = calloc(nresos, sizeof(int));
            int *rshift_pixs_galind_bounds_source = calloc(nresos, sizeof(int));
            int *rshift_pix_gals_source = calloc(nresos, sizeof(int));
            for (int elreso=1;elreso<nresos;elreso++){
                rshift_index_matcher_source[elreso] = rshift_index_matcher_source[elreso-1] + npix_hash;
                rshift_pixs_galind_bounds_source[elreso] = rshift_pixs_galind_bounds_source[elreso-1] + ngal_source_resos[elreso-1]+1;
                rshift_pix_gals_source[elreso] = rshift_pix_gals_source[elreso-1] + ngal_source_resos[elreso-1];
            }
            int *rshift_index_matcher_lens = calloc(nresos, sizeof(int));
            int *rshift_pixs_galind_bounds_lens = calloc(nresos, sizeof(int));
            int *rshift_pix_gals_lens = calloc(nresos, sizeof(int));
            for (int elreso=1;elreso<nresos;elreso++){
                rshift_index_matcher_lens[elreso] = rshift_index_matcher_lens[elreso-1] + npix_hash;
                rshift_pixs_galind_bounds_lens[elreso] = rshift_pixs_galind_bounds_lens[elreso-1] + ngal_lens_resos[elreso-1]+1;
                rshift_pix_gals_lens[elreso] = rshift_pix_gals_lens[elreso-1] + ngal_lens_resos[elreso-1];
            }
            
            // Shift variables for the matching between the pixel grids (only needed for lenses!)
            int lower, upper, lower1, upper1, lower2, upper2, ind_inpix, ind_gal, zbin_gal;
            int npix_side, thisreso, elreso_grid, len_matcher;
            int *matchers_resoshift = calloc(nresos_grid+1, sizeof(int));
            int *ngal_in_pix = calloc(nresos*nbinsz_lens, sizeof(int));
            for (int elreso=0;elreso<nresos;elreso++){
                elreso_grid = elreso - hasdiscrete;
                lower = pixs_galind_bounds_lens[rshift_pixs_galind_bounds_lens[elreso]+elregion];
                upper = pixs_galind_bounds_lens[rshift_pixs_galind_bounds_lens[elreso]+elregion+1];
                for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                    ind_gal = rshift_pix_gals_lens[elreso] + pix_gals_lens[rshift_pix_gals_lens[elreso]+ind_inpix];
                    ngal_in_pix[zbin_lens_resos[ind_gal]*nresos+elreso] += 1;
                }
                if (elregion==region_debug){
                    for (int elbinz=0; elbinz<nbinsz_lens; elbinz++){
                        printf("ngal_in_pix[elreso=%d][elz=%d] = %d \n",
                               elreso,elbinz,ngal_in_pix[elbinz*nresos+elreso]);
                    }
                }
                if (elreso_grid>=0){
                    npix_side = 1 << (nresos_grid-elreso_grid-1);
                    matchers_resoshift[elreso_grid+1] = matchers_resoshift[elreso_grid] + npix_side*npix_side; 
                }
                if (elregion==region_debug){printf("matchers_resoshift[elreso=%d] = %d \n", elreso,matchers_resoshift[elreso_grid+1]);}
            }
            len_matcher = matchers_resoshift[nresos_grid];
            
            // Build the matcher from pixels to reduced pixels in the region (only needed for lenses!)
            int elregion_fullhash, elhashpix_1, elhashpix_2, elhashpix;
            double hashpix_start1, hashpix_start2;
            double pos1_gal, pos2_gal;
            elregion_fullhash = index_matcher_hash[elregion];
            hashpix_start1 = pix1_start + (elregion_fullhash%pix1_n)*pix1_d;
            hashpix_start2 = pix2_start + (elregion_fullhash/pix1_n)*pix2_d;
            if (elregion==region_debug){
                printf("elregion=%d, elregion_fullhash=%d, pix1_start=%.2f pix2_start=%.2f \n", elregion,elregion_fullhash,pix1_start,pix2_start);
                printf("hashpix_start1=%.2f hashpix_start2=%.2f \n", hashpix_start1,hashpix_start2);}
            int *pix2redpix = calloc(nbinsz_lens*len_matcher, sizeof(int)); // For each z matches pixel in unreduced grid to index in reduced grid
            
            for (int elreso=0;elreso<nresos_grid;elreso++){
                thisreso = elreso + hasdiscrete;
                lower = pixs_galind_bounds_lens[rshift_pixs_galind_bounds_lens[thisreso]+elregion];
                upper = pixs_galind_bounds_lens[rshift_pixs_galind_bounds_lens[thisreso]+elregion+1];
                npix_side = 1 << (nresos_grid-elreso-1);
                int *tmpcounts = calloc(nbinsz_lens, sizeof(int));
                for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                    ind_gal = rshift_pix_gals_lens[thisreso] + pix_gals_lens[rshift_pix_gals_lens[thisreso]+ind_inpix];
                    zbin_gal = zbin_lens_resos[ind_gal];
                    pos1_gal = pos1_lens_resos[ind_gal];
                    pos2_gal = pos2_lens_resos[ind_gal];
                    elhashpix_1 = (int) floor((pos1_gal - hashpix_start1)/dpix1_resos[elreso]);
                    elhashpix_2 = (int) floor((pos2_gal - hashpix_start2)/dpix2_resos[elreso]);
                    elhashpix = elhashpix_2*npix_side + elhashpix_1;
                    pix2redpix[zbin_gal*len_matcher+matchers_resoshift[elreso]+elhashpix] = tmpcounts[zbin_gal];
                    tmpcounts[zbin_gal] += 1;
                    if (elregion==region_debug){
                        printf("elreso=%d, lower=%d, thispix=%d, elhashpix=%d %d %d, zgal=%d: pix2redpix[%d]=%d  \n",
                               elreso,lower,ind_inpix,elhashpix_1,elhashpix_2,elhashpix,zbin_gal,zbin_gal*len_matcher+matchers_resoshift[elreso]+elhashpix,ind_inpix-lower);
                    }
                }
                free(tmpcounts);
            }
            
            // Setup all shift variables for the Gncache in the region
            // Gncache has structure
            // n --> zbin_source --> zbin_lens --> radius 
            //   --> [ [0]*ngal_zbin1_reso1 | [0]*ngal_zbin1_reso1/2 | ... | [0]*ngal_zbin1_reson ]
            int *cumresoshift_z = calloc(nbinsz_lens*(nresos+1), sizeof(int)); // Cumulative shift index for resolution at z1
            int *thetashifts_z = calloc(nbinsz_lens, sizeof(int)); // Shift index for theta given z1
            int *zbinshifts = calloc(nbinsz_lens+1, sizeof(int)); // Cumulative shift index for z1
            int zbin2shift, nshift_cache; // Shifts for z2 index and n index
            for (int elz=0; elz<nbinsz_lens; elz++){
                if (elregion==region_debug){printf("z=%d/%d: \n", elz,nbinsz_lens);}
                for (int elreso=0; elreso<nresos; elreso++){
                    //if (elregion==region_debug){printf("  reso=%d/%d: \n", elreso,nresos);}
                    if (hasdiscrete==1 && elreso==0){
                        cumresoshift_z[elz*(nresos+1) + elreso+1] = ngal_in_pix[elz*nresos + elreso+1];
                    }
                    else{
                        cumresoshift_z[elz*(nresos+1) + elreso+1] = cumresoshift_z[elz*(nresos+1) + elreso] + ngal_in_pix[elz*nresos + elreso];
                    }
                    if (elregion==region_debug){printf("  cumresoshift_z[z][reso+1]=%d: \n", cumresoshift_z[elz*(nresos+1) + elreso+1]);}
                }
                thetashifts_z[elz] = cumresoshift_z[elz*(nresos+1) + nresos];
                zbinshifts[elz+1] = zbinshifts[elz] + nbinsr*thetashifts_z[elz];
                if ((elregion==region_debug)){printf("thetashifts_z[z]=%d: \nzbinshifts[z+1]=%d: \n", 
                                                     thetashifts_z[elz],  zbinshifts[elz+1]);}
            }            
            zbin2shift = zbinshifts[nbinsz_lens];
            nshift_cache = nbinsz_source*zbin2shift;
            // Set all the cache indices that are updated in this region to zero
            if ((elregion==region_debug)){printf("zbin2shift=%d: nshift_cache=%d: size_max_nshift=%d \n", 
                                                 zbin2shift, nshift_cache, size_max_nshift);}
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
            int innergal, nbinszr_reso;
            double pos1_gal1, pos2_gal1, pos1_gal2, pos2_gal2, w_gal1, w_gal2, e1_gal2, e2_gal2;
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
                //if (elregion==region_debug){printf("rbinmin=%d, rbinmax%d\n",rbinmin,rbinmax);}
                int ind_Wnp, ind_Wnm, ind_Gnp, ind_Gnm, ind_counts, z1shift, z2rshift, rbin;
                for (ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                    ind_gal1 = rshift_pix_gals_lens[elreso] + pix_gals_lens[rshift_pix_gals_lens[elreso]+ind_inpix1];
                    innergal = isinner_lens_resos[ind_gal1];
                    if (innergal==0){continue;}
                    z_gal1 = zbin_lens_resos[ind_gal1];
                    pos1_gal1 = pos1_lens_resos[ind_gal1];
                    pos2_gal1 = pos2_lens_resos[ind_gal1];
                    w_gal1 = w_lens_resos[ind_gal1];
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
                            if (elregion==-1){
                                printf("Gnupdates for elregion=%d reso1=%d reso2=%d red_reso2=%d, galindex=%d, z1=%d, z2=%d:%d radial updates; shiftstart %d = %d+%d+%d+%d+%d, size_max_nshift=%d\n"
                                       ,elregion,elreso,elreso2,red_reso2,ind_gal1,z_gal1,zbin2,rbinmax-rbinmin,
                                       zbin2*zbin2shift + zbinshifts[z_gal1] + rbinmin*thetashifts_z[z_gal1] + 
                                       cumresoshift_z[z_gal1*(nresos+1) + elreso2] + redpix_reso2,
                                       zbin2*zbin2shift, zbinshifts[z_gal1], rbinmin*thetashifts_z[z_gal1],
                                       cumresoshift_z[z_gal1*(nresos+1) + elreso2], redpix_reso2, size_max_nshift);
                            }
                            for (int thisrbin=rbinmin; thisrbin<rbinmax; thisrbin++){
                                zrshift = zbin2*nbinsr_reso + thisrbin-rbinmin;
                                if (cabs(thisWns[nbinszr_reso+zrshift])<1e-10){continue;}
                                //printf("Doing Gns/Nns for rbin %i/%i",thisrbin,rbinmax-rbinmin);
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
                    // Upsilon_-(thet1, thet2) ~ w * G_{+n-2}(thet1) * G_{-n-2}(thet2) + delta^K_{thet1,thet2} * (w * (we)^2*exp(4*phi))
                    // Upsilon_+(thet1, thet2) ~ w * G_{+n-2}(thet1) * conj(G_{+n-2})(thet2) + delta^K_{thet1,thet2} * (w * |we|^2)
                    // Norm(thet1, thet2)    ~   w  * W_{n}(thet1)   * W_{-n}(thet2)   - delta^K_{thet1,thet2} * (w  * w*w)
                    for (int thisn=-nmax; thisn<=nmax; thisn++){
                        int elb1_full, elb2_full, z3r2shift, gammashift_ups, gammashift_norm;
                        int _wind, _upsind1, _upsind2p, _upsind2m, zrshift, _zcombi, zcombi, elb1, zbin2, elb2, zbin3;
                        double complex nextUps, nextN;
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
                            _upsind1 = (nmax+0+thisn)*nbinszr_reso+zrshift;
                            nextUps = w_gal1*thisGns[_upsind1];
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
                                _upsind2p = (nmax-thisn)*nbinszr_reso + zrshift;
                                _upsind2m = (nmax+thisn)*nbinszr_reso + zrshift;
                                tmpUpsilon[gammashift_ups] += nextUps*thisGns[_upsind2p];
                                tmpUpsilon[upsilon_compshift+gammashift_ups] += nextUps*conj(thisGns[_upsind2m]);
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
            double complex nextUps, nextN;
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
                                            nextUps = wGncache[(nmax+thisn)*nshift_cache+ind_Wncacheshift];
                                            nextN = wWncache[(nmax+thisn)*nshift_cache+ind_Wncacheshift];
                                            _upsshift = thisnshift_ups + zcombi*upsilon_zshift + elb1*nbinsr;
                                            _normshift = thisnshift_norm+ zcombi*upsilon_zshift + elb1*nbinsr;
                                            ind_Wncacheshift = zbin3*zbin2shift+zbinshifts[zbin1]+rbinmin2*thetashifts_z[zbin1]+
                                                cumresoshift_z[zbin1*(nresos+1) + thisreso2] + elgal;
                                            for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                                tmpUpsilon[_upsshift+elb2] += nextUps *  
                                                    Gncache[(nmax-thisn)*nshift_cache+ind_Wncacheshift];
                                                tmpUpsilon[_upsshift+upsilon_compshift+elb2] += nextUps *  
                                                    conj(Gncache[(nmax+thisn)*nshift_cache+ind_Wncacheshift]);
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
                                            nextUps = Gncache[(nmax+thisn)*nshift_cache+ind_Wncacheshift];
                                            nextN = Wncache[(nmax+thisn)*nshift_cache+ind_Wncacheshift];
                                            _upsshift = thisnshift_ups + zcombi*upsilon_zshift + elb1*nbinsr;
                                            _normshift = thisnshift_norm+ zcombi*upsilon_zshift + elb1*nbinsr;
                                            ind_Wncacheshift = zbin3*zbin2shift+zbinshifts[zbin1]+rbinmin2*thetashifts_z[zbin1]+
                                                cumresoshift_z[zbin1*(nresos+1) + thisreso1] + elgal;
                                            for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                                tmpUpsilon[_upsshift+elb2] += nextUps *
                                                    wGncache[(nmax-thisn)*nshift_cache+ind_Wncacheshift];
                                                tmpUpsilon[_upsshift+upsilon_compshift+elb2] += nextUps *
                                                    conj(wGncache[(nmax+thisn)*nshift_cache+ind_Wncacheshift]);
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
}
  