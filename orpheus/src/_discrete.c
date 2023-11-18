// gcc -fopenmp $(gsl-config --cflags) -fPIC -ffast-math -O3 -shared -o discrete.so discrete.c $(gsl-config --libs) -std=c99
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
                /*
                if (ind_gal%10000==0){
                    int nonzero_Gn = 0;
                    int nonzero_Gnnorm = 0;
                    for (int thisn=0; thisn<2*nmax+3; thisn++){
                         for (int thisz=0; thisz<nbinsz; thisz++){
                             for (int thisr=0; thisr<nbinsr; thisr++){
                                if (cabs(nextGns[thisn*nbinsr*nbinsz+thisz*nbinsr+thisr]) >1e-5){
                                    nonzero_Gn += 1;
                                }
                                if (thisn<nmax+1 && cabs(nextGns_norm[thisn*nbinsr*nbinsz+thisz*nbinsr+thisr]) >1e-5){
                                    nonzero_Gnnorm += 1;
                                }
                            }
                         }
                    }
                    printf("%d %d %d %d %d %d %d %d \n",
                           ind_gal,odd,thisthread,thisstripe,galstripe,nonzero_Gn,nonzero_Gnnorm,nonzero_tmpGammas);
                }
                */
                
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



////////////////////////
/// UNUSED FUNCTIONS ///
////////////////////////

typedef struct {
    double real;
    double imag;
} CT_Complex;

// Example: Discrete cat of N=1e6 gals & two pixelized ones with N_1 = 1e5 and N_2 = 4e4 gals
//          --> * Each array has 1.14e6 entries
//              * ngals_start = [0,1e6,1e6+1e5,]
//              * ncats = 3
//              * Similar things hold for the spatialhash-based arrays
void alloc_Gammans_NNN_(int *isinner, double *weight, double *pos1, double *pos2, int *zbins, 
                int nbinsz, int ncats, int *ngals_catshift, 
                int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
                double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
                int *index_matchers_catshift, int* pixs_galind_bounds_catshift, int *pix_gals_shift,
                double *rbin_edges, double *rbinind_cuts, int nbinsr, int nmax
                int nthreads
               )

void alloc_Gammans_discrete_gd_dd(
    int *isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int *zbins, int nbinsz, int ngal, 
    int nmin, int nmax, double rmin, double rmax, int irmin_gd, double *rbins, int nbinsr, int do_dd,
    int nreso, int *nbinsr_reso, int *Gnsgrid_nredpixs, int *Gnsgrid_resoshift, int Gnsgrid_zshift, int Gnsgrid_compshift,
    int *galpixinds, double *wgrids, double *e1grids, double *e2grids, 
    double *Gnsgrids_re, double *Gnsgrids_im, double *Gnnormsgrids_re, double *Gnnormsgrids_im, 
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, 
    double *bin_centers, 
    double complex *Gammans_dg, double complex *Gammans_norm_dg,
    double complex *Gammans_gd, double complex *Gammans_norm_gd,
    double complex *Gammans_dd, double complex *Gammans_norm_dd);

void alloc_Gamman_grid_multireso(double *Gncache_re, double *Gncache_im, int* inds_cache,
                                 double *Gnnorm_re, double *Gnnorm_im, 
                                 double *weights_pix, double *e1_pix, double *e2_pix, 
                                 int nreso, int nbinsr, int nbinsz, int *nbinsr_reso, int *nGnpix_reso,
                                 int *Gnindices_z_reso, int *nfieldpix_z_reso, int allzperms,
                                 int nthreads, double complex *Gamman,  double complex *Gamman_norm);
    
void alloc_Gamman_grid_singlereso(double *Gns_re, double *Gns_im, double *Gnnorm_re, double *Gnnorm_im, 
                                  double *weights_pix, double *e1_pix, double *e2_pix,
                                  int nbinsr, int nbinsz, int npix_red_grid, 
                                  int *npixs_red_zs, int *cumnpixs_red_zs, int *redgrid2redzs,
                                  int nthreads, double complex *Gamman,  double complex *Gamman_norm);

void add_list_of_arrs(double **arr1, double **arr2, int narrs, int lenarr, double *res);
void add_arrays(int** arr1_ptr, int** arr2_ptr, int size, int num_arrays, int** result_ptr);


void add_list_of_arrs(double **arr1, double **arr2, int narrs, int lenarr, double *res){
    for (int eln=0; eln<narrs; eln++){
        for (int elr=0; elr<lenarr; elr++){
            res[eln*lenarr+elr] = arr1[eln][elr] + arr2[eln][elr];
        }
    }
}

void add_arrays(int** arr1_ptr, int** arr2_ptr, int size, int num_arrays, int** result_ptr) {
    int i, j;
    for (i = 0; i < num_arrays; i++) {
        int* arr1 = arr1_ptr[i];
        int* arr2 = arr2_ptr[i];
        int* result = result_ptr[i];
        for (j = 0; j < size; j++) {
            result[j] = arr1[j] + arr2[j];
        }
    }
}

// * weights_pix, e1_pix, e2_pix are the entries of the reduced grid for each redshift slice, 
//   i.e weights_pix ~ [weights_pix_redgridz0, ..., weights_pix_redgridzn], where
//   |weights_pix_redgridzi| = npixs_red_zs[i]
// * Gns have shape (4, nbinsz, nbinsr, npix_red_grid) where the indices are [-n-3,-n-1,n-3,n-1]
//   Gnnorm has shape (nbinsz, nbinsr, npix_red_grid) where the index is n
// * redgrid2redzs is of shape [(redgrid2redzs[0]),..., (redgrid2redzs[n])]
//   where each block stores the indices of the reduced grid for which the weight of the corresponding
//   zbin is not zero --> len(redgrid2redzs) = len(weights_pix) etc.
// * Gamman is of shape (4, nzcombis, nbinsr, nbinsr)
void alloc_Gamman_grid_singlereso(double *Gns_re, double *Gns_im, double *Gnnorm_re, double *Gnnorm_im, 
                                  double *weights_pix, double *e1_pix, double *e2_pix,
                                  int nbinsr, int nbinsz, int npix_red_grid, 
                                  int *npixs_red_zs, int *cumnpixs_red_zs, int *redgrid2redzs,
                                  int nthreads, double complex *Gamman,  double complex *Gamman_norm){
    
    int gammaax0 = nbinsz*nbinsz*nbinsz*nbinsr*nbinsr;
    int gammaax1 = nbinsr*nbinsr;
    int gammancomp = 4*gammaax0;
    
    int Gnax0 = nbinsz*nbinsr*npix_red_grid;
    int Gnax1 = nbinsr*npix_red_grid;
    double complex *tmpGamman = calloc(nthreads*4*nbinsz*nbinsz*nbinsz*nbinsr*nbinsr, sizeof(double complex));
    double complex *tmpGamman_norm = calloc(nthreads*nbinsz*nbinsz*nbinsz*nbinsr*nbinsr, sizeof(double complex));
    for (int zbin1=0; zbin1<nbinsz; zbin1++){
        int shiftz1_cat = cumnpixs_red_zs[zbin1];
        #pragma omp parallel for num_threads(nthreads)
        for (int elpix=0; elpix<npixs_red_zs[zbin1]; elpix++){
            int tid = omp_get_thread_num();
            int thiszcombi = zbin1*nbinsz*nbinsz;
            for (int zbin2=0; zbin2<nbinsz; zbin2++){
                for (int elb1=0; elb1<nbinsr; elb1++){
                    double complex wshape, h0, h1, h2, h3, w0;
                    int pixind_cat, gind0_z2, gind1_z2, gind2_z2, gind3_z2;
                    int gind0_z3, gind1_z3, gind2_z3, gind3_z3;
                    pixind_cat = shiftz1_cat + elpix;
                    gind0_z2 = zbin2*Gnax1 + elb1*npix_red_grid + redgrid2redzs[pixind_cat];
                    gind1_z2 = gind0_z2 + Gnax0;
                    gind2_z2 = gind1_z2 + Gnax0;
                    gind3_z2 = gind2_z2 + Gnax0;
                    wshape = weights_pix[pixind_cat] * (e1_pix[pixind_cat]+I*e2_pix[pixind_cat]);
                    h0 = -wshape * (Gns_re[gind0_z2]+I*Gns_re[gind0_z2]);
                    h1 = -conj(wshape) * (Gns_re[gind2_z2]+I*Gns_im[gind2_z2]);
                    h2 = -wshape * (Gns_re[gind3_z2]-I*Gns_im[gind3_z2]);
                    h3 = -wshape * (Gns_re[gind2_z2]-I*Gns_im[gind2_z2]);
                    w0 = weights_pix[pixind_cat] * (Gnnorm_re[gind0_z2]-I*Gnnorm_im[gind0_z2]);
                    for (int zbin3=0; zbin3<nbinsz; zbin3++){
                        for (int elb2=0; elb2<nbinsr; elb2++){ 
                            gind0_z3 = zbin3*Gnax1 + elb1*npix_red_grid + redgrid2redzs[pixind_cat];
                            gind1_z3 = gind0_z3 + Gnax0;
                            gind2_z3 = gind1_z3 + Gnax0;
                            gind3_z3 = gind2_z3 + Gnax0;
                            int gammashift = tid*gammancomp + thiszcombi*gammaax1+elb1*nbinsr+elb2;
                            int gammashiftt = tid*gammancomp + thiszcombi*gammaax1+elb2*nbinsr+elb1;
                            tmpGamman[gammashift] += h0*(Gns_re[gind1_z3]+I*Gns_im[gind1_z3]);
                            tmpGamman[gammashift+gammaax0] += h1*(Gns_re[gind3_z3]+I*Gns_im[gind3_z3]);
                            tmpGamman[gammashift+2*gammaax0] += h2*(Gns_re[gind1_z3]+I*Gns_im[gind1_z3]);
                            tmpGamman[gammashiftt+3*gammaax0] += h3*(Gns_re[gind0_z3]+I*Gns_im[gind0_z3]);
                            tmpGamman_norm[gammashiftt] += w0*(Gnnorm_re[gind0_z3]+I*Gnnorm_im[gind0_z3]);
                        }
                    }
                }
                if (elpix==npixs_red_zs[zbin1]-1){thiszcombi += 1;}  
            } 
        }
    } 
    
    // Accummulate Gamman
    #pragma omp parallel for num_threads(mymin(nbinsr*nbinsr,nthreads))
    for (int elb=0; elb<nbinsr*nbinsr; elb++){ 
        int elb1 = (int) floor(elb/nbinsr);
        int elb2 = elb%nbinsr;
        for (int thiszcombi=0; thiszcombi<nbinsz*nbinsz*nbinsz; thiszcombi++){
            for (int ncomp=0; ncomp<4; ncomp++){
                for (int nthread=0; nthread<nthreads; nthread++){
                    int gammashift = thiszcombi*gammaax1+elb1*nbinsr+elb2;
                    int tmpgammashift = gammashift+nthread*gammancomp;
                    Gamman[gammashift] += tmpGamman[tmpgammashift];
                    Gamman_norm[gammashift] += tmpGamman_norm[tmpgammashift];
                    Gamman[gammashift+gammaax0] += tmpGamman[tmpgammashift+gammaax0];
                    Gamman[gammashift+2*gammaax0] += tmpGamman[tmpgammashift+2*gammaax0];
                    Gamman[gammashift+3*gammaax0] += tmpGamman[tmpgammashift+3*gammaax0];
                }
            }
        }
    }
    free(tmpGamman);
    free(tmpGamman_norm);       
}

// Gamma_n(R1, R2, z1, z2, z3) ~ \sum_pix gamma_pix(z1) * Gn'_pix(R1,z2) * Gn''_pix(R2,z3)
// * weights_pix, e1_pix, e2_pix are the entries of the reduced grid for each redshift slice, 
//   i.e weights_pix ~ [[weights_pix_redgridz0_res0, ..., weights_pix_redgridzn_res0], 
//                      ..., 
//                      [weights_pix_redgridz0_resi, ..., weights_pix_redgridzn_resi]]
//   where |weights_pix_redgridzi_resk| = nfieldpix_z_reso[i][k]
// * Gn is passed as a cache of size (6, nbinsz, cumpixs_r)
//   inds_cache has the components of the 0th axis belonging to n = [-n-3,-n-1,n-3,n-1]
// * Gns have structure G[n][z][reso][r][pixs] where the indices are n \in [-n-3,-n-1,n-3,n-1]
//   and 
//   Gnnorm has shape (nbinsz, nbinsr, npix_red_grid) where the index is n
// * nbinsr_reso has shape (nreso) and stores how many radial bins we have for each resolution
// * nfieldpix_z_reso has shape (nreso, nz) and stores how many nonempty pixels we have for the data grid
//   at redshift z for each resolution reso
// * nGnpix_reso has shape (nreso) and stores how many nonzero pixels we have for the Gn at each resolution
//   for each resolution scale.
// * Gnindices_z_reso has shape (nreso, nzs, sum_reso nfieldpix_z_reso[reso,z]) and stores which pixels of the Gn data
//   correspond to nonzero field pixels. Length < sum_reso nGnpix_reso*nzs
// * Gamman is of shape (4, nzcombis, nbinsr, nbinsr)
// TODO Better way to handle nzcombis2/nzcombis3
void alloc_Gamman_grid_multireso(double *Gncache_re, double *Gncache_im, int* inds_cache,
                                 double *Gnnorm_re, double *Gnnorm_im, 
                                 double *weights_pix, double *e1_pix, double *e2_pix, 
                                 int nreso, int nbinsr, int nbinsz, int *nbinsr_reso, int *nGnpix_reso,
                                 int *Gnindices_z_reso, int *nfieldpix_z_reso, int allzperms,
                                 int nthreads, double complex *Gamman,  double complex *Gamman_norm){
    
    int nzcombis2, nzcombis3;
    if (allzperms==0){
        nzcombis3 = (int) ((nbinsz*(nbinsz-1)*(nbinsz-2))/6 + nbinsz*(nbinsz-1) + nbinsz);
        nzcombis2 = (int) ((nbinsz*(nbinsz+1))/2);}
    else{
        nzcombis3=nbinsz*nbinsz*nbinsz;
        nzcombis2=nbinsz*nbinsz;}
    int Gamma_compshift = nzcombis3*nbinsr*nbinsr;
    int Gamma_rshift = nbinsr*nbinsr;
    
    double complex *tmpGamman = calloc(nthreads*4*Gamma_compshift, sizeof(double complex));
    double complex *tmpGamman_norm = calloc(nthreads*Gamma_compshift, sizeof(double complex));
    //printf("Allocated output\n");
    //printf("nthreads=%d nloop=%d\n",nthreads, nzcombis2*nbinsr);

    #pragma omp parallel for num_threads(nthreads) default(shared)
    //(tmpGamman, tmpGamman_norm) private(nzcombis2, nzcombis3, Gamma_compshift, Gamma_rshift)
    for (int elthread=0; elthread<nzcombis2*nbinsr; elthread++){
        int tid, rbin1, zbin1, zbin2, zbin3, thiszcombi, thiszcombi2;
        int tmpzbinshift, thiszbin1shift, zbin2shift, minz2, minz3;
        int reso1, reso2, thisreso;
        int fieldpix0, iGnshift, ihn0, ihn;
        int iGn_base, iGn_0, iGn_1, iGn_2, iGn_3;
        int iGamma0, iGamma1, iGamma2, iGamma3, iGammanorm, gammashift;    
        
        int tnbinsz = nbinsz;
        int tnzcombis2 = nzcombis2;
        int tnzcombis3 = nzcombis3;
        
        tid = omp_get_thread_num();
        //printf("Starting loop index %d on process %d\n",elthread, tid);
        // Allocate zbin1shift ~ [0,\tau_{nbinsz,3}, \tau_{nbinsz,3}+\tau_{nbinsz-1,3}, ..., ]
        int *zbin1shift = calloc(nbinsz+1, sizeof(int)); // = [0,15,25,31,34,35] for nbinsz=5
        int toadd1;
        for (int tmpz=0; tmpz<nbinsz; tmpz++){
            if (allzperms==0){toadd1 = ((nbinsz-tmpz)*(nbinsz-tmpz+1))/2;}
            else{toadd1 = nbinsz*nbinsz;}
            zbin1shift[1+tmpz] = zbin1shift[tmpz] + toadd1;
        }
        //printf("Thread %d: Allocated zbin1shift\n",elthread);
        // Get indices (zbin1, zbin2, rbin1) index of this process
        thiszcombi2 = (int) floor(elthread/nbinsr);
        rbin1 = elthread%nbinsr;
        //printf("Thread %d: Got rbin1=%d zcombi2=%d\n",elthread,rbin1,thiszcombi2);
        tmpzbinshift = 0;
        thiszbin1shift = 0;
        for (zbin1=0; zbin1<nbinsz; zbin1++){
            tmpzbinshift += nbinsz - (1-allzperms)*zbin1;
            if (thiszcombi2<tmpzbinshift){break;}
            else{thiszbin1shift=tmpzbinshift;}
        }
        //printf("Thread %d: Got zbin1=%d\n",elthread,zbin1);
        thiszcombi2 -= thiszbin1shift;
        minz2 = (1-allzperms)*zbin1;
        tmpzbinshift = 0;
        zbin2shift = 0;
        zbin2 = minz2 + thiszcombi2;
        for (int tmpz=0; tmpz<thiszcombi2; tmpz++){
            zbin2shift += nbinsz - (1-allzperms)*(minz2+tmpz);}
        //printf("Thread %d: Got zbin2=%d\n",elthread,zbin2);
        minz3 = (1-allzperms)*zbin2;

        if (tid==0){
            //printf("Indices for thread %d/%d: tid:%d zbin1:%d zbin2:%d rbin1:%d\n",
            //   elthread,nbinsz*nbinsz*nbinsr,tid,zbin1,zbin2,rbin1);
            //printf("\rDone %.2f per cent of Gamman computation",
            //    (float) nthreads*elthread/ ((float) nbinsz*nbinsz*nbinsr) * 100);
        }
        // Allocate some helpers for index counting
        int *cumnbinsr = calloc(nreso+1, sizeof(int));
        int *resoofrbin = calloc(nbinsr, sizeof(int));
        int *Gn_resoshift = calloc(nreso+1, sizeof(int)); 
        int *field_cumpixshifts = calloc(nreso*nbinsz+1, sizeof(int)); // shape (nreso, nz)
        int *field_totpixperz = calloc(nbinsz, sizeof(int)); 
        int *field_pixperreso = calloc(nreso, sizeof(int));
        int *field_resoshift = calloc(nreso+1, sizeof(int));
        int *field_cumzresoshifts = calloc(nreso*(nbinsz+1), sizeof(int)); // shape (nz+1, nreso)
        for (int i=0; i<nreso; i++){
            cumnbinsr[i+1] = cumnbinsr[i] + nbinsr_reso[i];
            Gn_resoshift[i+1] = Gn_resoshift[i] + cumnbinsr[i+1]*nGnpix_reso[i];
            for (int j=0; j<nbinsz; j++){
                field_cumpixshifts[i*nbinsz+j+1] = field_cumpixshifts[i*nbinsz+j] + nfieldpix_z_reso[i*nbinsz+j];
                field_totpixperz[j] += nfieldpix_z_reso[i*nbinsz+j];
                field_pixperreso[i] += nfieldpix_z_reso[i*nbinsz+j];
                field_cumzresoshifts[(j+1)*nreso+i] = field_cumzresoshifts[j*nreso+i] + nfieldpix_z_reso[i*nbinsz+j];
                //if (elthread==0){
                //    printf("Reso=%d, zbin=%d field_cumpixshifts = %d\n",i,j+1,field_cumpixshifts[i*nbinsz+j+1]);
                //    printf("Reso=%d, zbin=%d field_cumzresoshifts = %d\n",i,j+1,field_cumzresoshifts[(j+1)*nreso+i]);
                //    if (i==nreso-1){
                //        printf("Reso=%d, zbin=%d field_totpixperz = %d\n",i,j,field_totpixperz[j]);
                //    }
                //}
            }
            field_resoshift[i+1] = field_resoshift[i] + field_pixperreso[i];
            //if (elthread==0){
            //    printf("Reso=%d, cumnbinsr = %d\n",i+1,cumnbinsr[i+1]);
            //    printf("Reso=%d, Gn_resoshift = %d\n",i+1,Gn_resoshift[i+1]);
            //    printf("Reso=%d, field_resoshift = %d\n",i+1,field_resoshift[i+1]);
            //    printf("Reso=%d, field_pixperreso = %d\n",i,field_pixperreso[i]);
            //}
        }
        thisreso=0;
        for (int i=0; i<nbinsr; i++){
            if (i>=cumnbinsr[thisreso+1]){thisreso += 1;}
            resoofrbin[i] =  thisreso;
            //if (elthread==0){
            //    printf("resoofrbin[%d] = %d",i,resoofrbin[i]);
            //}
        }
        //if (elthread==0){
        //    printf("Allocated helpers\n");
        //}
        int Gn_zshift = Gn_resoshift[nreso];
        int Gn_nshift = nbinsz * Gn_zshift;
        int Gamma_threadshift = 4*tid*Gamma_compshift;
        int Gammanorm_threadshift = tid*Gamma_compshift;

        // Precompute h_i & w0 for all available resolutions
        // h_i have shape [field_z1_reso0, ..., field_z1_reson] with length cumnpix_z
        // For this only fill the resolutions reso >= reso(rbin1)
        //if (elthread==_thisthread){
        //    printf("Allocating hi of length %d\n",field_totpixperz[zbin1]);
        //}
        double complex *h0 = calloc(field_totpixperz[zbin1], sizeof(double complex));
        double complex *h1 = calloc(field_totpixperz[zbin1], sizeof(double complex));
        double complex *h2 = calloc(field_totpixperz[zbin1], sizeof(double complex));
        double complex *h3 = calloc(field_totpixperz[zbin1], sizeof(double complex));
        double complex *w0 = calloc(field_totpixperz[zbin1], sizeof(double complex));
        double complex wshape;
        reso1 = resoofrbin[rbin1];
        //if (elthread==_thisthread){
        //    printf("this reso=%d\n",reso1);
        //}
        for (int ireso=reso1; ireso<nreso; ireso++){
            //if (elthread==_thisthread){
            //    printf("Doing reso %d/%d\n",ireso+1,nreso);
            //}
            fieldpix0 = field_cumpixshifts[ireso*nbinsz+zbin1];
            ihn0=0;
            for (int _ireso=0; _ireso<ireso; _ireso++){
                ihn0 += nfieldpix_z_reso[_ireso*nbinsz+zbin1];
            }
            //ihn0 = field_cumzresoshifts[zbin1*nreso+ireso];//524249
            iGn_base = zbin2*Gn_zshift + Gn_resoshift[ireso] + rbin1*nGnpix_reso[ireso];
            iGn_0 = iGn_base + inds_cache[0]*Gn_nshift;
            iGn_1 = iGn_base + inds_cache[1]*Gn_nshift;
            iGn_2 = iGn_base + inds_cache[2]*Gn_nshift;
            iGn_3 = iGn_base + inds_cache[3]*Gn_nshift;
            //if (elthread==_thisthread){
            //    printf("fieldpix0=%d ihn0=%d iGn_base=%d Gn_nshift=%d\n",
            //           fieldpix0,ihn0,iGn_base,Gn_nshift);
            //    printf("Loop over %d pixels\n",nfieldpix_z_reso[ireso*nbinsz+zbin1]);
            //}
            for (int elpix=0; elpix<nfieldpix_z_reso[ireso*nbinsz+zbin1]; elpix++){
                iGnshift = Gnindices_z_reso[fieldpix0+elpix];
                ihn = ihn0 + elpix;
                //wshape = weights_pix[fieldpix0+elpix] * (e1_pix[fieldpix0+elpix]+I*e2_pix[fieldpix0+elpix]);
                wshape = e1_pix[fieldpix0+elpix]+I*e2_pix[fieldpix0+elpix];
                h0[ihn] = -wshape * (Gncache_re[iGn_2+iGnshift]+I*Gncache_im[iGn_2+iGnshift]);
                h1[ihn] = -conj(wshape) * (Gncache_re[iGn_3+iGnshift]+I*Gncache_im[iGn_3+iGnshift]);
                h2[ihn] = -wshape * (Gncache_re[iGn_1+iGnshift]-I*Gncache_im[iGn_1+iGnshift]);
                h3[ihn] = -wshape * (Gncache_re[iGn_2+iGnshift]+I*Gncache_im[iGn_2+iGnshift]);
                w0[ihn] = weights_pix[fieldpix0+elpix] * (Gnnorm_re[iGn_base+iGnshift]+I*Gnnorm_im[iGn_base+iGnshift]);
            }
        }
        //printf("thread=%d zcombi2=%d r1=%d zbin1=%d zbin2=%d minz2=%d minz3=%d\n",elthread,thiszcombi2,rbin1,zbin1,zbin2,minz2,minz3);
        for (int rbin2=0; rbin2<nbinsr; rbin2++){ 
            // Find which resolution we need to take - i.e. the larger of the two
            reso2 = resoofrbin[rbin2];
            thisreso = mymax(reso1, reso2); 
            ihn0 = 0;
            for (int _ireso=0; _ireso<thisreso; _ireso++){
                ihn0 += nfieldpix_z_reso[_ireso*nbinsz+zbin1];
            }
            for (zbin3=minz3; zbin3<nbinsz; zbin3++){
                fieldpix0 = field_cumpixshifts[thisreso*nbinsz+zbin1];
                thiszcombi = zbin1shift[zbin1] + zbin2shift + zbin3-minz3;
                gammashift = thiszcombi*Gamma_rshift+rbin1*nbinsr+rbin2;
                iGamma0 = Gamma_threadshift + gammashift;
                iGamma1 = Gamma_threadshift + gammashift+Gamma_compshift;
                iGamma2 = Gamma_threadshift + gammashift+2*Gamma_compshift;
                iGamma3 = Gamma_threadshift + gammashift+3*Gamma_compshift;
                iGammanorm = Gammanorm_threadshift + gammashift;
                iGn_base = zbin3*Gn_zshift + Gn_resoshift[thisreso] + rbin2*nGnpix_reso[thisreso];
                iGn_0 = iGn_base + inds_cache[0]*Gn_nshift;
                iGn_1 = iGn_base + inds_cache[1]*Gn_nshift;
                iGn_2 = iGn_base + inds_cache[2]*Gn_nshift;
                iGn_3 = iGn_base + inds_cache[3]*Gn_nshift;
                //if (elthread==_thisthread){
                    //printf("rbin1=%d rbin2=%d zbin1=%d zbin2=%d zbin3=%d thisreso=%d fieldpix0=%d ihn0=%d iGn_base=%d looprange=%d\n",
                   //        rbin1,rbin2,zbin1,zbin2,zbin3,thisreso,fieldpix0,ihn0,iGn_base,nfieldpix_z_reso[thisreso*nbinsz+zbin1]);
                    //printf("thiszcombi=%d gammashift=%d iGamma3=%d\n",
                    //      thiszcombi,gammashift,iGamma3);
                //}
                for (int elpix=0; elpix<nfieldpix_z_reso[thisreso*nbinsz+zbin1]; elpix++){
                    ihn = ihn0 + elpix;
                    iGnshift = Gnindices_z_reso[fieldpix0+elpix];
                    //if (rbin2==10 && zbin3==4 && elpix>=43950 && elthread==_thisthread){
                    //    printf("elpix=%d ihn=%d iGnshift=%d len(ihn)=%d\n",
                    //           elpix,ihn,iGnshift,field_totpixperz[zbin1]);
                    //}
                    tmpGamman[iGamma0] += h0[ihn]*(Gncache_re[iGn_0+iGnshift]+I*Gncache_im[iGn_0+iGnshift]);
                    tmpGamman[iGamma1] += h1[ihn]*(Gncache_re[iGn_1+iGnshift]+I*Gncache_im[iGn_1+iGnshift]);
                    tmpGamman[iGamma2] += h2[ihn]*(Gncache_re[iGn_0+iGnshift]+I*Gncache_im[iGn_0+iGnshift]);
                    tmpGamman[iGamma3] += h3[ihn]*(Gncache_re[iGn_3+iGnshift]-I*Gncache_im[iGn_3+iGnshift]);
                    tmpGamman_norm[iGammanorm] += w0[ihn]*(Gnnorm_re[iGn_base+iGnshift]-I*Gnnorm_im[iGn_base+iGnshift]);
                    //if (rbin2==10 && zbin3==4 && elpix>=43950 && elthread==_thisthread){
                    //    printf("Finished for elpix=%d \n",elpix);
                    //}
                }
            }
        }
        //if (elthread==_thisthread){
        //    printf("Allocated the Gammas for thread %d\n",elthread);
        //}

        free(zbin1shift);
        free(cumnbinsr);
        free(Gn_resoshift);
        free(field_cumpixshifts);
        free(field_totpixperz);
        free(field_pixperreso);
        free(field_resoshift);
        free(field_cumzresoshifts);
        free(h0);
        free(h1);
        free(h2);
        free(h3);
        free(w0);

        //if (elthread==_thisthread){
        //    printf("Finished thread %d\n",elthread);
        //}
    }
        
    // Accummulate the Gamma_n
    #pragma omp parallel for num_threads(nthreads) default(shared)
    for (int elb=0; elb<nbinsr*nbinsr; elb++){ 
        int elb1 = (int) floor(elb/nbinsr);
        int elb2 = elb%nbinsr;
        int Gamma_threadshift = 4*Gamma_compshift;
        int Gammanorm_threadshift = Gamma_compshift;
        for (int thiszcombi=0; thiszcombi<nzcombis3; thiszcombi++){
            for (int ncomp=0; ncomp<4; ncomp++){
                for (int nthread=0; nthread<nthreads; nthread++){
                    int gammashift = thiszcombi*Gamma_rshift+elb1*nbinsr+elb2;
                    int iGamma0 = nthread*Gamma_threadshift + gammashift;
                    Gamman[gammashift] += tmpGamman[iGamma0];
                    Gamman[gammashift+Gamma_compshift] += tmpGamman[iGamma0+Gamma_compshift];
                    Gamman[gammashift+2*Gamma_compshift] += tmpGamman[iGamma0+2*Gamma_compshift];
                    Gamman[gammashift+3*Gamma_compshift] += tmpGamman[iGamma0+3*Gamma_compshift];
                    Gamman_norm[gammashift] += tmpGamman_norm[nthread*Gammanorm_threadshift+gammashift];
                }
            }
        }
    }
    free(tmpGamman);
    free(tmpGamman_norm);  
    printf("\n");
}

// * Gns have shape (nzbins, cumnbinsr[0]*npixreso[0]+...+cumnbinsr[nreso-1]*npixreso[nreso-1]) for each multipole n
//   This means that G[n][z] ~ [r1_reso0, r2_reso0, ..., ri0_reso0,
//                              r1_reso1, r2_reso1, ..., ri0_reso1, ..., ri1_reso1,
//                              ...
//                              r1_resok, r2_resok, ..., ri0_resok, ..., ri1_reso1, ..., rik_resok, ..., rmax]
// * Gamma_n has shape (4,n,nzbins,nzbins,nzbins,nrbins,nrbins)
//         i.e. for tomo KiDS we have ~ 4*41*5^3*50^2 ~ 5e7 components
/*
void alloc_Gammans_grid(CT_Complex** Gns, int nmin, int nmax, int nbinsz, int nreso, int *cumnbinsr, int *npixreso,
                        int nthreads, double complex *Gamma_n,  double complex *Gamma_n_norm){
    
    int nrs_tot;
    int thisreso1, thisreso2, thisreso;
    int Gnshift;
    int nbinsn = 2*nmax+5;
    int nbinsr = cumnbinsr[-1];
    
    int nrs_tot = 0;
    int *cumnnrs = calloc(nbinsr, sizeof(int)) 
    for (int elbinr=0; elreso<nreso, elreso++){
    }
    nrs_tot = cumnnrs[nreso-1]

    for (int eln=0; eln<=nmax-nmin; eln++){
        for (int elz1=0; elz1<nbinsz; elz1++){
            for (int elz2=0; elz2<nbinsz; elz2++){
                for (int elz3=0; elz3<nbinsz; elz3++){
                    thisreso1 = 0;
                    thisreso2 = 0;
                    for (int elr1=0; elr1<nbinsr; elr1++){
                        if (elr1>=cumnbinsr[thisreso1]){thisreso1+=1;}
                        for (int elr2=0; elr2<nbinsr; elr2++){
                            if (elr2>=cumnbinsr[thisreso2]){thisreso2+=1;}
                            thisreso = mymax(thisreso1, thisreso2);
                            Gnshift = 
                            double complex *tmpGammans = calloc(nthreads*4, sizeof(double complex));
                            double complex *tmpGammansnorm = calloc(nthreads, sizeof(double complex));
                            #pragma omp parallel for num_threads(nthreads)
                            for (int elpix=0; elpix<npixreso[thisreso]; elpix++){
                                int Gnshift = 
                                int tid = omp_get_thread_num();
                                tmpGammans
                                
                            }
                        }   
                    }
                }
            }
        }
    }
    for (i = 0; i < num_arrays; i++) {
        double complex* arr1 = arr1_ptr[i];
        double complex* arr2 = arr2_ptr[i];
        double complex* result = result_ptr[i];
        for (j = 0; j < size; j++) {
            result[j] = arr1[j] + arr2[j];
        }
    }
} 
*/
/*void alloc_Gn_discrete_polar(
    double *weight, double *pos1, double *pos2, double *scalar, double *e1, double *e2, int *zbins, int nzbins, int ngal, 
    int *nvals, int nnvals, int nnorm, double rmin, double rmax, int nbins,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, double pix1_n, double pix1_start, double pix1_d, double pix1_n, 
    int nthreads, double *gns, double *norms);   */


// Time complexity ~ Ngals * nbar*4*rmax
// Result is double of shape (nzbins, ns, nradii, ngal)
// I.e. nzbins=5, ns=8, nradii=20, ngal=1e7 --> 4e8 entries (as ngal gets distributed across zbins)
// - weight, pos1, pos2, e1, e2, zbins are arrays from shape catalog of ngal galaxies in nzbins photometric bins
// - index_matcher, pixs_galind_bounds, pix_gals, pix1/2_start, pix1/2_d, pix1/2_n are properties of the spatial hash
// - nvals are the nnvals n values for which the Gn are allocated in this round; 
//   nnorm is the n value for which the norm gets allocated
// - rmin, rmax, nbins are the log-spaced range of radii for which the Discrete estimator is exectuted.
// - Gncache, normcache are the cached arrays that are gettting updated. Their memory is already fully allocated in python
//   inds_start_Gncache and ind_start_normcache give the starting value 
/*void alloc_Gn_discrete_polar(
    double *weight, double *pos1, double *pos2, double *e1, double *e2, int *zbins, int nzbins, int ngal, 
    int *nvals, int nnvals, int nnorm, double rmin, double rmax, int nbins,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, double pix1_n, double pix2_start, double pix2_d, double pix2_n, 
    int nthreads, double *Gncache, double *normcache, int *inds_start_Gncache, int ind_start_normcache){

    #pragma omp parallel for num_threads(nthreads)
    for (int ind_gal=0; ind_gal<ngal; ind_gal++){
        int ind_pix1, ind_pix2;
        int ind_red, ind_gal, ind_gn, lower, upper; 
        double rel1, rel2, dist2, dphi;
        complex wshape, exp_of_rot;
        int rbin;
        
        double drbin = (log(rmax)-log(rmin))/nbins;
        int mul3 = nnvals*nbins*ngal;
        int mul2 = nbins*ngal;
        
        int pix1_lower = (int) floor((pos1[ind_gal] - rmax - (pix1_start-.5*pix1_d) - pix1_d)/pix1_d);
        int pix1_upper = (int) floor((pos1[ind_gal] + rmax - (pix1_start-.5*pix1_d) + pix1_d)/pix1_d);
        int pix2_lower = (int) floor((pos2[ind_gal] - rmax - (pix2_start-.5*pix2_d) - pix2_d)/pix2_d);
        int pix2_upper = (int) floor((pos2[ind_gal] + rmax - (pix2_start-.5*pix2_d) + pix1_d)/pix2_d);
        
        for (ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
            for (ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){

                ind_red = index_matcher[ind_pix2*pix1_n + ind_pix1]
                if (ind_red==-1){continue;}
                lower = pixs_galind_bounds[ind_red];
                upper = pixs_galind_bounds[ind_red+1];
                for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                    ind_gal2 = pix_gals[ind_inpix];
                    rel1 = pos1[ind_gal2] - pos1[ind_gal];
                    rel2 = pos2[ind_gal2] - pos2[ind_gal];
                    dist2 = rel1*rel1 + rel2*rel2;
                    if(dist2 < rmin || dist >= rmax) continue;
                    rbin = (int) floor((.5*log(dist2)-log(rmin))/drbin);
                    wshape = weight[ind_gal2]*(e1[ind_gal2]+I*e2[ind_gal2])
                    dphi = atan2(rel2,rel1);
                    for (ind_n=0; ind_n<nnvals; ind_n++){
                        // Shape of gn (nzbins, ns, nradii, ngal)
                        exp_of_rot = cexp(I*dphi*nvals[ind_n]);
                        ind_gn = zbins[ind_gal2]*mul3 + ind_n*mul2 + rbin*ngal + ngal;
                        ind_norm = zbins[ind_gal2]*mul2 + rbin*ngal + ngal;
                        Gncache[inds_start_Gncache[ind_n]+ind_gn] += wshape*exp_of_rot;
                    norms[ind_start_normcache+ind_norm] += weight[ind_gal2]*cexp(I*dphi*nvals[nnorm]);
                    }
                }
            }
        }
    } 
}*/

// Calculates Gn s.t. one can allocate Gamman for [nmin, ..., nmax]
// Ranges of n are * n \in [-nmax-3, ..., -nmin+1] u [nmin-3, ..., nmax+3] (i.e. 2*(nmax-nmin+4) values) for nmin>3
//                 * n \in [-nmax-3, ..., nmax+1] (i.e. 2*nmax+5 values) otherwise
// Gns have shape (2*(nmax-nmin+4), ngal, nz, nr) / (2*nmax+4, ngal, nz, nr)
// bin_centers and counts have shape (nz, nz, nr); but are allocated first as (ngal, nz, nr) and afterwards reduced
// Memory: * ngal=1e7, nmax=40, nz=5, nr=50 --> 1e7*(3*40+5)*50*5 ~ 3.1e11 ~ 2.5TB
//         * ngal=1e6, nmax=40, nz=5, nr=10 -->                   ~ 6e9    ~ 50GB
//         * ngal=1e6, nmin=10, nmax=20, nz=5, nr=20              ~ 2.8e9  ~ 24GB
// Note that in both case we only need one cexp evaluation while most operations are cmult ones.
void alloc_Gns_discrete_basic(
    double *weight, double *pos1, double *pos2, double *e1, double *e2, int *zbins, int nbinsz, int ngal, 
    int nmin, int nmax, double rmin, double rmax, int nbinsr,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, double *bin_centers, int *counts, double complex *Gns, double complex *Gns_norm){
    
    // Need to allocate the bin centers/counts in this way to ensure parallelizability
    // At a later stage are reduced to the shape of the output
    double *bin_centers_gcount = calloc(ngal*nbinsz*nbinsr, sizeof(double));
    double *bin_centers_gnorm = calloc(ngal*nbinsz*nbinsr, sizeof(double));
    int *gcounts = calloc(ngal*nbinsz*nbinsr, sizeof(int));
    
    // Allocate Gns
    // We do this in parallel as follows:
    // * Split survey in 2*nthreads equal area stripes along x-axis
    // * Do two parallelized iterations over galaxies
    //   - In first one only consider galaxies within stripes of even number
    //   - In second one only consider galaxies within stripes of odd number
    // --> We avoid race conditions in calling the spatial hash arrays. - This
    //    is explicitly made sure by (re)setting nthreads in the python layer.
    for (int odd=0; odd<2; odd++){
        //#pragma omp parallel for private(nbinsz, ngal, nmin, nmax, rmin, rmax, nbinsr, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, nthreads)
        #pragma omp parallel for num_threads(nthreads)
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            // Index shifts for Gn
            int indshift3 = ngal*nbinsz*nbinsr;
            int indshift2 = nbinsz*nbinsr;
            for (int ind_gal=0; ind_gal<ngal; ind_gal++){

                // Check if galaxy falls in stripe used in this process
                double p11, p12;
                #pragma omp critical
                {p11 = pos1[ind_gal];
                p12 = pos2[ind_gal];}
                int thisstripe = 2*thisthread + odd;
                int galstripe = (int) floor((p11-pix1_start)/pix1_d * (2*nthreads)/pix1_n);
                if (thisstripe != galstripe){continue;}
                
                int ind_pix1, ind_pix2, ind_inpix, ind_gal2;
                int ind_red, lower, upper; 
                double  p21, p22, w2, z2, e21, e22;
                double rel1, rel2, dist, dphi;
                double complex wshape;
                int nextn, nzero;
                double complex nphirot, nphirotc, phirot, phirotc, phirotm, phirotp, phirotn;

                int rbin;
                int thisindshift;

                if (nmin<4){nmin=0;}
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
                            rbin = (int) floor((log(dist)-log(rmin))/drbin);
                            wshape = (double complex) w2 * (e21+I*e22);
                            dphi = atan2(rel2,rel1);
                            thisindshift = ind_gal*indshift2+z2*nbinsr + rbin;

                            // nmin=0 -
                            //   -> Gns axis: [-nmax-3, ..., nmax+1]
                            //   -> Gn_norm axis: [0,...,nmax]
                            if (nmin==0){
                                nzero = nmax+3;
                                nphirot = 1+I*0;
                                nphirotc = 1+I*0;
                                phirot = cexp(I*dphi);
                                phirotc = conj(phirot);
                                int tmpGindp = nzero*indshift3 + thisindshift;
                                int tmpGindm = nzero*indshift3 + thisindshift;
                                int tmpGindn = thisindshift;
                                // n = 0
                                //bin_centers_gcount[thisindshift] += w1*w2*dist; 
                                //bin_centers_gnorm[thisindshift] += w1*w2; 
                                //gcounts[thisindshift] += 1; 
                                Gns[tmpGindp] += wshape*nphirot;
                                Gns_norm[tmpGindn] += w2*nphirot;  
                                nphirot *= phirot;
                                nphirotc *= phirotc; 
                                tmpGindp += indshift3;
                                tmpGindm -= indshift3;
                                tmpGindn += indshift3;
                                // n in [1, ..., nmax] x {+1,-1}
                                for (nextn=1;nextn<nmax+1;nextn++){
                                    Gns[tmpGindp] += wshape*nphirot;
                                    Gns[tmpGindm] += wshape*nphirotc;
                                    Gns_norm[tmpGindn] += w2*nphirot;  
                                    nphirot *= phirot;
                                    nphirotc *= phirotc; 
                                    tmpGindp += indshift3;
                                    tmpGindm -= indshift3;
                                    tmpGindn += indshift3;
                                }
                                // n in [-nmax-1, nmax+1]
                                Gns[tmpGindp] += wshape*nphirot;
                                Gns[tmpGindm] += wshape*nphirotc;
                                nphirotc *= phirotc; 
                                // n in [-nmax-2, -nmax=3]
                                Gns[indshift3+thisindshift] += wshape*nphirotc;
                                nphirotc *= phirotc; 
                                Gns[thisindshift] += wshape*nphirotc;
                            }
                            // nmin>3 
                            //   --> Gns axis: [-nmax-3, ..., -nmin+1, nmin-3, ..., nmax+1]
                            //   --> Gn_norm axis: [nmin, ..., nmax]
                            else{
                                phirot = cexp(dphi);
                                phirotc = conj(phirot);
                                phirotp = cpow(phirotc,nmax+3);
                                phirotm = cpow(phirot,nmin-3);
                                phirotn = phirotm*phirot*phirot*phirot;
                                int tmpGindm = thisindshift;
                                int tmpGindp = (nmin+nmax)*indshift3 + thisindshift;
                                // n in [0, ..., nmax-nmin] + {-nmax-3, nmin-3, nmin}
                                for (nextn=0;nextn<nmax-nmin+1;nextn++){
                                    Gns[tmpGindm] += wshape*phirotm;
                                    Gns[tmpGindp] += wshape*phirotp;
                                    Gns_norm[tmpGindm] += w2*phirotn;
                                    phirotm *= phirot;
                                    phirotp *= phirot;
                                    phirotn *= phirot;
                                    tmpGindm += indshift3;
                                    tmpGindp += indshift3;
                                }
                                // n in [nmax-nmin+1, nmax-nmin+2, nmax-nmin+3] + {-nmax-3, nmin-3}
                                Gns[tmpGindm] += wshape*phirotm;
                                Gns[tmpGindp] += wshape*phirotp;
                                phirotm *= phirot;
                                phirotp *= phirot;
                                tmpGindm += indshift3;
                                tmpGindp += indshift3;
                                Gns[tmpGindm] += wshape*phirotm;
                                Gns[tmpGindp] += wshape*phirotp;
                                phirotm *= phirot;
                                phirotp *= phirot;
                                tmpGindm += indshift3;
                                tmpGindp += indshift3;
                                Gns[tmpGindm] += wshape*phirotm;
                                Gns[tmpGindp] += wshape*phirotp;
                            }               
                        }
                    }
                }
            }
        }
    }
    
    /*
    // Finish the calculation of bin centers (cannot be parallelized)
    double *bin_centers_norm = calloc(nbinsz*nbinsz*nbinsr, sizeof(double));
    int countbin, gbinind;
    for (int ind_gal=0; ind_gal<ngal; ind_gal++){
        int zbin1 = zbins[ind_gal];
        for (int elb1=0; elb1<nbinsr; elb1++){
            for (int zbin2=0; zbin2<nbinsz; zbin2++){
                countbin = zbin1*nbinsz*nbinsr + zbin2*nbinsr + elb1;
                gbinind = ind_gal*indshift2+zbin2*nbinsr + elb1;
                bin_centers[countbin] += bin_centers_gcount[gbinind];
                bin_centers_norm[countbin] += bin_centers_gnorm[gbinind];
                counts[countbin] += gcounts[gbinind];
            }
        }
    }
    for (int elb1=0; elb1<nbinsr; elb1++){
        for (int zbin1=0; zbin1<nbinsz; zbin1++){
            for (int zbin2=0; zbin2<nbinsz; zbin2++){
                countbin = zbin1*nbinsz*nbinsr + zbin2*nbinsr + elb1;
                bin_centers[countbin] /=  bin_centers_norm[countbin];
            }
        }
    }
    free(bin_centers_norm);
    */
    
    free(gcounts);
    free(bin_centers_gcount);
    free(bin_centers_gnorm);
}



// Maps element on high resolution grid to (reduced) indices of all lower resolution grids
// resomapper has form  [ 0 | 0...N_gal_DeltaN | 0...N_gal_DeltaN-1 | ... | 0...N_gal_Delta1 ]
//                         (N_gal_DeltaN-1 els) (N_gal_DeltaN-2 els)        (N_gal_disc els)
void tocourseinds(int ind_hr, int elreso_hr, int *shiftNgal_resos, int *resomapper, *inds_lr){
    ind_resomapper = shiftNgal_resos[elreso_hr] + ind_hr;
    val_resomapper = resomapper[ind_resomapper]
    for (el_red=0; elr_red<ind_hr; el_red++){
        inds_lr[el_red] = val_resomapper;
        elreso_lr = elreso_hr - (el_red+1);
        ind_resomapper = shiftNgal_resos[elreso_lr] + val_resomapper;
        val_resomapper = resomapper[ind_resomapper];
    }
}

// Allocates the discrete values for Gn for some galaxy indices, as well
void alloc_Gn_disc_gals(
    int elreso, int *inds_gal, int ngals, 
    int *nrbins_reso, int *Ngal_resos,
    int *isinner, double *weight, double *pos1, double *pos2, int *zbins, 
    int nbinsz, int ncats, int *ngals_catshift, int nrelpixs_base,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int *index_matchers_catshift, int* pixs_galind_bounds_catshift, int *pix_gals_shift,
    double *rbin_edges, double *rbinind_cuts, int nbinsr, int nmin_G, int nmax_G, int nmax_W,
    double complex *Gns, double complex *Wns
    ){
    int indshiftG = 0;    
    
    // TBD
}

void alloc_Gammans_NNN(int *isinner, double *weight, double *pos1, double *pos2, int *zbins, 
    int nbinsz, int ncats, int *ngals_catshift, int nrelpixs_base,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int *index_matchers_catshift, int* pixs_galind_bounds_catshift, int *pix_gals_shift,
    double *rbin_edges, double *rbinind_cuts, int nbinsr, int nmax,
    int nthreads,
    double complex *fullGammans){
    
    
    // Prepare Gammans that are partially allocated in each thread
    int _nncombis = 2*nmax+3;
    int _nzcombis = nbinsz*nbinsz*nbinsz;
    int _nrcombis = (nbinsr*nbinsr+nbinsr)/2;
    int _nbinsGamma = _nncombis*_nzcombis*_nrcombis;
    double complex *Gammans = calloc(nthreads*_nbinsGamma, sizeof(double complex));
    #pragma omp parallel for num_threads(nthreads) default(shared)
    for (int thisthread=0; thisthread<nthreads; thisthread++){
        elpix_base_start = ceil(((float) (thisthread*nrelpixs_base))/((float) nthreads));
        elpix_base_end = floor(((float) ((thisthread+1)*nrelpixs_base))/((float) nthreads));                         
        for (int elpix_base=elpix_base_start; elpix_base<elpix_base_end; elpix_base++){
                        
            // Allocate (cumulative) number of galaxies within the pixel for each resolution
            // The resolutions are ordered s.t. they start from the coursest one!
            int *Ngal_pix_resos = calloc(ncats);
            int *shiftNgal_pix_resos = calloc(ncats);
            for (int elreso=0; elreso<ncats; elreso++){
                lower = pixs_galind_bounds[pixs_galind_bounds_catshift[elreso] + elpix_base];
                upper = pixs_galind_bounds[pixs_galind_bounds_catshift[elreso] + elpix_base+1];
                Ngal_pix_resos[elreso] = upper-lower;
                if (elreso>0){
                    shiftNgal_pix_resos[elreso] = Ngal_pix_resos[elreso-1]+Ngal_pix_resos[elreso];
                }
            }
            totshiftNgal_pix_resos = Ngal_pix_resos[ncats-2]+Ngal_pix_resos[ncats-1];
            int *resmapper = calloc(totshiftNgal_pix_resos);
            
            // Allocate all Gn(Delta_i | Delta_j)
            for (int elreso=0; elreso<ncats; elreso++){
                lower = pixs_galind_bounds[pixs_galind_bounds_catshift[elreso] + elpix_base];
                upper = pixs_galind_bounds[pixs_galind_bounds_catshift[elreso] + elpix_base+1];
            }
            
            
            free(Ngal_pix_resos);
            free(shiftNgal_pix_resos);
            free(resmapper);
        }
    }
    free(Gammans)
}
               
// Allocates multipoles of shape catalog via discrete estimator
// Allocate discrete Gns on nbinsr bins between rmin_disc and rmax
// Total range of discrete computation is in rmin_disc,rmax_disc in nbinsr_disc bins
void alloc_Gammans_discrete_gd_dd(
    int *isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int *zbins, int nbinsz, int ngal, 
    int nmin, int nmax, double rmin, double rmax, int irmin_gd, double *rbins, int nbinsr, int do_dd,
    int nreso, int *nbinsr_reso, int *Gnsgrid_nredpixs, int *Gnsgrid_resoshift, int Gnsgrid_zshift, int Gnsgrid_compshift,
    int *galpixinds, double *wgrids, double *e1grids, double *e2grids, 
    double *Gnsgrids_re, double *Gnsgrids_im, double *Gnnormsgrids_re, double *Gnnormsgrids_im, 
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, 
    double *bin_centers, 
    double complex *Gammans_dg, double complex *Gammans_norm_dg,
    double complex *Gammans_gd, double complex *Gammans_norm_gd,
    double complex *Gammans_dd, double complex *Gammans_norm_dd){
    
    
    // Index shift for the Gamman
    int _gamma_zshift = nbinsr*nbinsr;
    int _gamma_nshift = _gamma_zshift*nbinsz*nbinsz*nbinsz;
    int _gamma_compshift = (nmax-nmin+1)*_gamma_nshift;
    int *cumGnsgridredpixs = calloc(nreso+1, sizeof(int));
    int *cumnbinsr_reso = calloc(nreso+1, sizeof(int));
    for (int _=0; _<nreso; _++){
        cumGnsgridredpixs[_+1] = cumGnsgridredpixs[_] + Gnsgrid_nredpixs[_];
        cumnbinsr_reso[_+1] = cumnbinsr_reso[_] + nbinsr_reso[_];
    } 
    int nbinsr_grid = cumnbinsr_reso[nreso];
    
    // Gnsgrid_disc have shape (nnvals, nbinsz, nbinsz, [[r0_reso0, ..., r0_resod], ..., [rn_reso0,...,rn_resod]]) 
    int Gnsgrid_disc_rshift = cumGnsgridredpixs[nreso];
    int Gnsgrid_disc_zshift = (nbinsr-irmin_gd)*Gnsgrid_disc_rshift;
    int Gnsgrid_disc_nshift = nbinsz*nbinsz*Gnsgrid_disc_zshift;
    double *totcounts = calloc(nbinsz*nbinsr, sizeof(double));
    double *totnorms = calloc(nbinsz*nbinsr, sizeof(double));
    double complex *Gnsgrid_disc = calloc((2*nmax+3)*Gnsgrid_disc_nshift, sizeof(double complex));
    double complex *conjGnsgrid_disc = calloc((2*nmax+3)*Gnsgrid_disc_nshift, sizeof(double complex));
    double complex *Gnnormsgrid_disc = calloc((nmax+1)*Gnsgrid_disc_nshift, sizeof(double complex));
    //printf("Length of grids: %d %d\n",(2*nmax+3)*Gnsgrid_disc_nshift,(nmax+1)*Gnsgrid_disc_nshift);
           
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
        #pragma omp parallel for num_threads(nthreads) default(shared)
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
                if (innergal == 0){continue;}
                int thisstripe = 2*thisthread + odd;
                int galstripe = (int) floor((p11-pix1_start)/pix1_d * (2*nthreads)/pix1_n);
                if (thisstripe != galstripe){continue;}
                
                int ind_pix1, ind_pix2, ind_inpix, ind_gal2;
                int ind_red, lower, upper; 
                double  p21, p22, w2, z2, e21, e22;
                double rel1, rel2, dist, dphi;
                double complex wshape;
                int nnvals, nnvals_norm, nextn, nzero;
                double complex nphirot, nphirotc, phirot, phirotc, phirotm, phirotp, phirotn;
                
                // Check how many ns we need for Gn
                // Gns have shape (nnvals, nbinsz, nbinsr)
                // where the ns are ordered as 
                //  * [-nmax-3, ..., nmax-1] / [0, ..., nmax]
                if (nmin<4){nmin=0;}
                if (nmin==0){nnvals=2*nmax+3;nnvals_norm=nmax+1;}
                else{nnvals=2*(nmax-nmin+3);nnvals_norm=nmax-nmin+1;}
                double complex *nextGns =  calloc(nnvals*nbinsr*nbinsz, sizeof(double complex));
                double complex *nextGns_norm =  calloc(nnvals_norm*nbinsr*nbinsz, sizeof(double complex));

                int ind_rbin, rbin;
                int ind_Gn, ind_Gnnorm, zrshift, nextnshift;
                int nbinszr = nbinsz*nbinsr;
                double drbin = (log(rmax)-log(rmin))/(nbinsr);
                double complex wshape1 = w1 * (e11 + I*e12);
                //if (ind_gal%10000==0){
                //    printf("%d %d %d %d %d \n",nmin,nmax,nnvals,nbinsr,nbinsz);
                //}
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
                            dphi = atan2(rel2,rel1);
                            
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
                                phirot = cexp(I*dphi);
                                phirotc = conj(phirot);
                                // n = 0
                                tmpwcounts[ind_rbin] += w1*w2*dist; 
                                tmpwnorms[ind_rbin] += w1*w2; 
                                nextGns[ind_Gn] += wshape*nphirot;
                                nextGns_norm[ind_Gnnorm] += w2*nphirot;  
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
                                phirot = cexp(dphi);
                                phirotc = conj(phirot);
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
                
                // Update the Gnsgrid_discrete
                // ~ (2*nmax+3) * nbinsz * nbinsz * nbinsr * nreso * Ngal
                // ~ 23 * 5 * 5 * nbinsr * 2 * Ngal 
                // ~ 1150 * nbinsr * Ngal
                // Vs allocation:
                // ~ (2*nmax+3) * nbar * (rmax^2-rmin^2)*pi * Ngal
                // ~ 23 * 5 * (10^2-5^2) * pi * Ngal
                // ~ 27000 * Ngal >~ Gnalloc :)
                //if (ind_gal>153500){printf("Allocate Gnsgrid for gal %d  \n",ind_gal);}
                for (nextn=0; nextn<nnvals; nextn++){
                    int nextGnshift, nextGndshift, Gndresoshift;
                    for (int nextz=0; nextz<nbinsz; nextz++){ 
                        for (int nextr=irmin_gd; nextr<nbinsr; nextr++){
                            nextGnshift = nextn*nbinszr + nextz*nbinsr + nextr;
                            nextGndshift = 
                                nextn*Gnsgrid_disc_nshift + 
                                zbin1*nbinsz*Gnsgrid_disc_zshift +
                                nextz*Gnsgrid_disc_zshift + 
                                (nextr-irmin_gd)*Gnsgrid_disc_rshift;
                            for (int indreso=0; indreso<nreso; indreso++){
                                Gndresoshift = 
                                    cumGnsgridredpixs[indreso] + 
                                    galpixinds[indreso*ngal+ind_gal];
                                //Gnsgrid_disc[nextGndshift+Gndresoshift] += nextGns[nextGnshift];
                                Gnsgrid_disc[nextGndshift+Gndresoshift] += wshape1*nextGns[nextGnshift];
                                conjGnsgrid_disc[nextGndshift+Gndresoshift] += conj(wshape1)*nextGns[nextGnshift];
                            }
                            
                        }
                    }
                }
                // Update the Gnnormsgrid_discrete
                for (nextn=0; nextn<nnvals_norm; nextn++){
                    int nextGnshift, nextGndshift, Gndresoshift;
                    for (int nextz=0; nextz<nbinsz; nextz++){ 
                        for (int nextr=irmin_gd; nextr<nbinsr; nextr++){
                            nextGnshift = nextn*nbinszr + nextz*nbinsr + nextr;
                            nextGndshift = 
                                nextn*Gnsgrid_disc_nshift + 
                                zbin1*nbinsz*Gnsgrid_disc_zshift +
                                nextz*Gnsgrid_disc_zshift + 
                                (nextr-irmin_gd)*Gnsgrid_disc_rshift;
                            for (int indreso=0; indreso<nreso; indreso++){
                                Gndresoshift = 
                                    cumGnsgridredpixs[indreso] + 
                                    galpixinds[indreso*ngal+ind_gal];
                                //Gnnormsgrid_disc[nextGndshift+Gndresoshift] += nextGns_norm[nextGnshift];
                                Gnnormsgrid_disc[nextGndshift+Gndresoshift] += w1*nextGns_norm[nextGnshift];

                            }
                        }
                    }
                }
                //if (ind_gal>153500){printf("Done allocating Gnnormsgrid for gal %d  \n",ind_gal);}
                
                // Update the Gammans_dd
                // tmpGammas have shape (nthreads, nmax+1, nzcombis, r*r, 4)
                // Gns have shape (nnvals, nbinsz, nbinsr)
                //int nonzero_tmpGammas = 0;
                if (do_dd==1){
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
                                w0 = weight[ind_gal] * conj(nextGns_norm[ind_norm + zrshift]);
                                for (zbin3=0; zbin3<nbinsz; zbin3++){
                                    zcombi = zbin1*nbinsz*nbinsz+zbin2*nbinsz+zbin3;
                                    gammashift1 = thisnshift + zcombi*gamma_zshift + elb1*nbinsr;
                                    gammashiftt1 = thisnshift + zcombi*gamma_zshift;
                                    for (elb2=0; elb2<nbinsr; elb2++){
                                        zrshift = zbin3*nbinsr + elb2;
                                        r12shift = elb2*nbinsr+elb1;
                                        gammashift = 4*(gammashift1 + elb2);
                                        gammashiftt = gammashiftt1 + r12shift;
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
                }
                
                //if (ind_gal>153500){printf("Freeing pointers after finishing with last gal %d\n",ind_gal);}
                free(nextGns);
                free(nextGns_norm);
                nextGns = NULL;
                nextGns_norm = NULL;
            }
        }
        
        // Accumulate the Gamman_dd
        if (do_dd==1){
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
                                    Gammans_dd[elcomp*_gamma_compshift+iGamma] += tmpGammans[4*itmpGamma+elcomp];
                                }
                                Gammans_norm_dd[iGamma] += tmpGammans_norm[itmpGamma];
                            }
                        }
                    }
                }
            }
        }
        
        // Update the bin distances and weights
        if (do_dd==1){
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
        }
        //printf("Freeing arrays for this half\n");
        free(tmpwcounts);
        free(tmpwnorms);
        free(tmpGammans);
        free(tmpGammans_norm); 
        tmpwcounts = NULL;
        tmpwnorms = NULL;
        tmpGammans = NULL;
        tmpGammans_norm = NULL;
        //printf("Done freeing arrays for this half\n");
    }
    
    // Allocate the Gamman_dg & Gamman_gd
    // Inefficient by at worst factor nbinsz as we do not mask the zcatalogs individually;
    // realistically do not expect a large difference if Npix_red < Ngal
    // We parallelize over the pixels for each resolution, i.e. each thread
    // only allocates npix_reso/nthreads pixels.
    // Gamman_dg(r1,r2) ~ sum wG_d(r1) * G_g(r2)
    // tmpGammans_dg have shape (nthreads, nmax+1, nzcombis, r_disc, r_grid, 4)
    // tmpGammans_gd have shape (nthreads, nmax+1, nzcombis, r_grid, r_disc, 4)
    _gamma_zshift = (nbinsr-irmin_gd)*nbinsr_grid;
    _gamma_nshift = _gamma_zshift*nbinsz*nbinsz*nbinsz;
    _gamma_compshift = (nmax-nmin+1)*_gamma_nshift;
    int _gamma_threadshift = 4*_gamma_compshift;
    double complex *tmpGammans_dg = calloc(nthreads*_gamma_threadshift, sizeof(double complex));
    double complex *tmpGammans_gd = calloc(nthreads*_gamma_threadshift, sizeof(double complex));
    double complex *tmpGammans_norm_dg = calloc(nthreads*_gamma_compshift, sizeof(double complex));
    double complex *tmpGammans_norm_gd = calloc(nthreads*_gamma_compshift, sizeof(double complex));
    #pragma omp parallel for num_threads(nthreads) default(shared)
    for (int thisthread=0; thisthread<nthreads; thisthread++){
        int *start_g = calloc(5, sizeof(int)); // [-n-3, -n-1, n-3, n-1, n]
        int *start_d = calloc(5, sizeof(int)); // [-n-3, -n-1, n-3, n-1, n]
        int gdshift_1, gdshift_2, dgshift_1, dgshift_2;
        int ind_Gamman_gd, ind_Gamman_gd_norm, ind_Gamman_dg, ind_Gamman_dg_norm;
        int tmpreso, npix_reso, startpix_redfield, pixind_start, pixind_end, zcombi;
        double complex Gng_mnm3, Gng_mnm1, Gng_nm3, Gng_nm1, Gngn_n;
        double complex thiswshape;
        for (int nextn=nmin; nextn<=nmax; nextn++){
            // Allocate Gamman_gd ~ wg(z1)*G^(grid)(R1,z2)*G^(disc)(R2,z3)
            for (int zbin1=0; zbin1<nbinsz; zbin1++){
                tmpreso = 0;
                for (int elb1=0; elb1<nbinsr_grid; elb1++){
                    if (elb1 >= cumnbinsr_reso[tmpreso+1]){tmpreso += 1;}
                    npix_reso = (int) ceil(Gnsgrid_nredpixs[tmpreso]/nthreads);
                    startpix_redfield = nbinsz*cumGnsgridredpixs[tmpreso] + 
                        zbin1*Gnsgrid_nredpixs[tmpreso];
                    pixind_start = thisthread*npix_reso;
                    pixind_end = mymin(pixind_start+npix_reso, Gnsgrid_nredpixs[tmpreso]);
                    for (int zbin2=0; zbin2<nbinsz; zbin2++){
                        gdshift_1 = zbin2*Gnsgrid_zshift + Gnsgrid_resoshift[tmpreso] + 
                            (elb1-cumnbinsr_reso[tmpreso])*Gnsgrid_nredpixs[tmpreso];
                        start_g[0] = (nmax-nextn)*Gnsgrid_compshift + gdshift_1;
                        start_g[1] = start_g[0] + 2*Gnsgrid_compshift;
                        start_g[2] = start_g[0] + 2*nextn*Gnsgrid_compshift;
                        start_g[3] = start_g[2] + 2*Gnsgrid_compshift;
                        start_g[4] = nextn*Gnsgrid_compshift + gdshift_1;
                        for (int elb2=0; elb2<nbinsr-irmin_gd; elb2++){
                            for (int zbin3=0; zbin3<nbinsz; zbin3++){
                                zcombi = zbin1*nbinsz*nbinsz+zbin2*nbinsz+zbin3;
                                gdshift_2 = zbin1*nbinsz*Gnsgrid_disc_zshift + zbin3*Gnsgrid_disc_zshift + 
                                    elb2*Gnsgrid_disc_rshift + cumGnsgridredpixs[tmpreso];
                                start_d[0] = (nmax-nextn)*Gnsgrid_disc_nshift + gdshift_2;  
                                start_d[1] = start_d[0] + 2*Gnsgrid_disc_nshift;
                                start_d[2] = start_d[0] + 2*nextn*Gnsgrid_disc_nshift;
                                start_d[3] = start_d[2] + 2*Gnsgrid_disc_nshift;
                                start_d[4] = nextn*Gnsgrid_disc_nshift + gdshift_2;
                                ind_Gamman_gd_norm = thisthread*_gamma_compshift + (nextn-nmin)*_gamma_nshift + 
                                    zcombi*_gamma_zshift + elb1*(nbinsr-irmin_gd) + elb2;
                                ind_Gamman_gd = 4*ind_Gamman_gd_norm;
                                for (int tmppix=pixind_start; tmppix<pixind_end; tmppix++){
                                    int _gshift = startpix_redfield + tmppix;
                                    thiswshape = (e1grids[_gshift]+I*e2grids[_gshift]); // Grid already holds w*e
                                    Gng_mnm3 = Gnsgrids_re[start_g[0]+tmppix] + I*Gnsgrids_im[start_g[0]+tmppix];
                                    Gng_mnm1 = Gnsgrids_re[start_g[1]+tmppix] + I*Gnsgrids_im[start_g[1]+tmppix];
                                    Gng_nm3 = Gnsgrids_re[start_g[2]+tmppix] + I*Gnsgrids_im[start_g[2]+tmppix];
                                    Gng_nm1 = Gnsgrids_re[start_g[3]+tmppix] + I*Gnsgrids_im[start_g[3]+tmppix];
                                    Gngn_n = Gnnormsgrids_re[start_g[4]+tmppix] + I*Gnnormsgrids_im[start_g[4]+tmppix]; 
                                    tmpGammans_gd[ind_Gamman_gd+0] +=  
                                        //-thiswshape * Gng_nm3 * Gnsgrid_disc[start_d[0]+tmppix];
                                        -Gng_nm3 * Gnsgrid_disc[start_d[0]+tmppix];
                                    tmpGammans_gd[ind_Gamman_gd+1] += 
                                        //-conj(thiswshape) * Gng_nm1 * Gnsgrid_disc[start_d[1]+tmppix];
                                        -Gng_nm1 * conjGnsgrid_disc[start_d[1]+tmppix];
                                    tmpGammans_gd[ind_Gamman_gd+2] += 
                                        //-thiswshape * conj(Gng_mnm1) * Gnsgrid_disc[start_d[0]+tmppix];
                                        -conj(Gng_mnm1) * Gnsgrid_disc[start_d[0]+tmppix];
                                    tmpGammans_gd[ind_Gamman_gd+3] += 
                                        //-thiswshape * Gng_nm3 * conj(Gnsgrid_disc[start_d[3]+tmppix]);
                                        -Gng_nm3 * conj(conjGnsgrid_disc[start_d[3]+tmppix]);
                                    tmpGammans_norm_gd[ind_Gamman_gd_norm] += 
                                        //wgrids[_gshift] * Gngn_n * conj(Gnnormsgrid_disc[start_d[4]+tmppix]);
                                        Gngn_n * conj(Gnnormsgrid_disc[start_d[4]+tmppix]);
                                }
                            }
                        }
                    }
                }
            }
            //printf("\rAllocated gd for n=%d",nextn);
            // Allocate Gamman_dg ~ wg(z1)*G^(disc)(R1,z2)*G^(grid)(R2,z3)
            for (int zbin1=0; zbin1<nbinsz; zbin1++){
                tmpreso = 0;
                for (int elbg=0; elbg<nbinsr_grid; elbg++){
                    if (elbg >= cumnbinsr_reso[tmpreso+1]){tmpreso += 1;}
                    npix_reso = (int) ceil(Gnsgrid_nredpixs[tmpreso]/nthreads);
                    startpix_redfield = nbinsz*cumGnsgridredpixs[tmpreso] + 
                        zbin1*Gnsgrid_nredpixs[tmpreso];
                    pixind_start = thisthread*npix_reso;
                    pixind_end = mymin(pixind_start+npix_reso, Gnsgrid_nredpixs[tmpreso]);
                    for (int zbin3=0; zbin3<nbinsz; zbin3++){
                        dgshift_2 = zbin3*Gnsgrid_zshift + Gnsgrid_resoshift[tmpreso] + 
                            (elbg-cumnbinsr_reso[tmpreso])*Gnsgrid_nredpixs[tmpreso];
                        start_g[0] = (nmax-nextn)*Gnsgrid_compshift + dgshift_2;
                        start_g[1] = start_g[0] + 2*Gnsgrid_compshift;
                        start_g[2] = start_g[0] + 2*nextn*Gnsgrid_compshift;
                        start_g[3] = start_g[2] + 2*Gnsgrid_compshift;
                        start_g[4] = nextn*Gnsgrid_compshift + dgshift_2;
                        for (int elbd=0; elbd<(nbinsr-irmin_gd); elbd++){
                            for (int zbin2=0; zbin2<nbinsz; zbin2++){
                                zcombi = zbin1*nbinsz*nbinsz+zbin2*nbinsz+zbin3;
                                dgshift_1 = zbin1*nbinsz*Gnsgrid_disc_zshift + zbin2*Gnsgrid_disc_zshift + 
                                    elbd*Gnsgrid_disc_rshift + cumGnsgridredpixs[tmpreso];
                                start_d[0] = (nmax-nextn)*Gnsgrid_disc_nshift + dgshift_1;
                                start_d[1] = start_d[0] + 2*Gnsgrid_disc_nshift;
                                start_d[2] = start_d[0] + 2*nextn*Gnsgrid_disc_nshift;
                                start_d[3] = start_d[2] + 2*Gnsgrid_disc_nshift;
                                start_d[4] = nextn*Gnsgrid_disc_nshift + dgshift_1;
                                ind_Gamman_dg_norm = thisthread*_gamma_compshift + (nextn-nmin)*_gamma_nshift + 
                                    zcombi*_gamma_zshift + elbd*nbinsr_grid + elbg;
                                ind_Gamman_dg = 4*ind_Gamman_dg_norm;
                                for (int tmppix=pixind_start; tmppix<pixind_end; tmppix++){
                                    int _gshift = startpix_redfield + tmppix;
                                    thiswshape = (e1grids[_gshift]+I*e2grids[_gshift]); // Grid already holds w*e
                                    Gng_mnm3 = Gnsgrids_re[start_g[0]+tmppix] + I*Gnsgrids_im[start_g[0]+tmppix];
                                    Gng_mnm1 = Gnsgrids_re[start_g[1]+tmppix] + I*Gnsgrids_im[start_g[1]+tmppix];
                                    Gng_nm3 = Gnsgrids_re[start_g[2]+tmppix] + I*Gnsgrids_im[start_g[2]+tmppix];
                                    Gng_nm1 = Gnsgrids_re[start_g[3]+tmppix] + I*Gnsgrids_im[start_g[3]+tmppix];
                                    Gngn_n = Gnnormsgrids_re[start_g[4]+tmppix] + I*Gnnormsgrids_im[start_g[4]+tmppix]; 
                                    tmpGammans_dg[ind_Gamman_dg+0] +=  
                                        //-thiswshape * Gnsgrid_disc[start_d[2]+tmppix] * Gng_mnm3;
                                        -Gnsgrid_disc[start_d[2]+tmppix] * Gng_mnm3;
                                    tmpGammans_dg[ind_Gamman_dg+1] += 
                                        //-conj(thiswshape) * Gnsgrid_disc[start_d[3]+tmppix] * Gng_mnm1;
                                        -conjGnsgrid_disc[start_d[3]+tmppix] * Gng_mnm1;
                                    tmpGammans_dg[ind_Gamman_dg+2] += 
                                        //-thiswshape * conj(Gnsgrid_disc[start_d[1]+tmppix]) * Gng_mnm3;
                                        -conj(conjGnsgrid_disc[start_d[1]+tmppix]) * Gng_mnm3;
                                    tmpGammans_dg[ind_Gamman_dg+3] += 
                                        //-thiswshape * Gnsgrid_disc[start_d[2]+tmppix] * conj(Gng_nm1);
                                        -Gnsgrid_disc[start_d[2]+tmppix] * conj(Gng_nm1);
                                    tmpGammans_norm_dg[ind_Gamman_dg_norm] +=  
                                        //wgrids[_gshift] * Gnnormsgrid_disc[start_d[4]+tmppix] * conj(Gngn_n);
                                        Gnnormsgrid_disc[start_d[4]+tmppix] * conj(Gngn_n);
                                }
                            }
                        }
                    }
                }  
            }
            //printf("\rAllocated dg for n=%d",nextn);
            
        }
        free(start_d);
        free(start_g);
    }
    
    // Accumulate the Gamman_dg & Gamman_gd
    //#pragma omp parallel for num_threads(nthreads)
    for (int thisthread=0; thisthread<nthreads; thisthread++){
        for (int thisn=0; thisn<nmax-nmin+1; thisn++){
            int itmpGamma_dg, iGamma_dg, itmpGamma_gd, iGamma_gd;
            for (int zcombi=0; zcombi<nbinsz*nbinsz*nbinsz; zcombi++){
                for (int elb1=0; elb1<nbinsr_grid; elb1++){
                    for (int elb2=0; elb2<nbinsr-irmin_gd; elb2++){
                        iGamma_dg = thisn*_gamma_nshift + zcombi*_gamma_zshift + elb2*nbinsr_grid + elb1;
                        itmpGamma_dg = iGamma_dg + thisthread*_gamma_compshift;
                        iGamma_gd = thisn*_gamma_nshift + zcombi*_gamma_zshift + elb1*(nbinsr-irmin_gd) + elb2;
                        itmpGamma_gd = iGamma_gd + thisthread*_gamma_compshift;
                        for (int elcomp=0; elcomp<4; elcomp++){
                            Gammans_gd[elcomp*_gamma_compshift+iGamma_gd] += tmpGammans_gd[4*itmpGamma_gd+elcomp];
                            Gammans_dg[elcomp*_gamma_compshift+iGamma_dg] += tmpGammans_dg[4*itmpGamma_dg+elcomp];
                        }
                        Gammans_norm_dg[iGamma_dg] += tmpGammans_norm_dg[itmpGamma_dg];
                        Gammans_norm_gd[iGamma_gd] += tmpGammans_norm_gd[itmpGamma_gd];
                        
                    }
                }
            }
        }
    }
    
    // Finalize bin centers
    //printf("Accumulating bin centers\n");
    for (int elbinz=0; elbinz<nbinsz; elbinz++){
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            int tmpind = elbinz*nbinsr + elbinr;
            if (totnorms[tmpind] != 0){
                bin_centers[tmpind] = totcounts[tmpind]/totnorms[tmpind];
            }
            
        }
    } 
    
    //printf("Freeing last patch of arrays\n");
    free(totcounts);
    free(totnorms);
    free(cumnbinsr_reso);
    free(cumGnsgridredpixs);
    free(Gnnormsgrid_disc);
    free(conjGnsgrid_disc);
    free(Gnsgrid_disc);
    free(tmpGammans_gd);
    free(tmpGammans_dg);
    free(tmpGammans_norm_gd);
    free(tmpGammans_norm_dg);
    totcounts = NULL;
    totnorms = NULL;
    cumnbinsr_reso = NULL;
    cumGnsgridredpixs = NULL;
    Gnnormsgrid_disc = NULL;
    conjGnsgrid_disc = NULL;
    Gnsgrid_disc = NULL;
    tmpGammans_gd = NULL;
    tmpGammans_dg = NULL;
    tmpGammans_norm_gd = NULL;
    tmpGammans_norm_dg = NULL;
    //printf("Done\n");
}

/* Allocates multipoles of shape catalog via discrete estimator
*  @param gal2stripe (Flat array of shape (ngal)
*         The stripe that each galaxy falls into
*  @param gal2redpix (Flat array of shape (nreso, ngal))
*         The reduced Gn pixel the each galaxy falls into (matches Gns_grid_re[stripe][reso])
*  @param nGnpix_reso_stripe (Flat array of shape (2*nthreads, nreso))
*         The number of reduced Gn pixels for each thread and resolution
*  @param nfieldpix_z_reso_stripe (Flat array of shape (2*nthreads, nreso, nbinsz))
*         The number of reduced Gn pixels for each thread and resolution
*  @param Gnindices_z_reso_stripe (Structure of shape 2*nthreads -> nreso -> indz -> ngnpi_reso))
*         Stores the reduced pixelindex of Gn on the corresponding stripe
*  @param weights_pix_stripes (Structure of shape 2*nthreads -> nreso -> nbinsz -> (nfieldpix_z_reso_stripe,))
*         Stores the weights of grided catalogs for each patch, redshift and resolution
*  @param e1_pix_stripes (Structure of shape 2*nthreads -> nreso -> nbinsz -> (nfieldpix_z_reso_stripe,))
*         Stores the first ellipticity component of grided catalogs for each stripe, redshift and resolution
*  @param e2_pix_stripes (Structure of shape 2*nthreads -> nreso -> nbinsz -> (nfieldpix_z_reso_stripe,))
*         Stores the second ellipticity component of gridded catalogs for each stripe, redshift and resolution
*/
/*
void alloc_Gammans_combined(
    int nmin, int nmax, int nthreads, 
    double *weight, double *pos1, double *pos2, double *e1, double *e2, int *zbins, int nbinsz, int ngal,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    double rmin_disc, double rmax_disc, double *rbins_disc, int nbinsr_disc,
    int *gal2stripe, int *gal2redpix, 
    double *weights_pix_stripes, double *e1_pix_stripes, double *e2_pix_stripes,
    int nreso, int nbinsr_grid, int *nbinsr_reso, int *nfieldpix_z_reso_stripe,
    double *Gns_grid_re, double *Gns_grid_im, double *Gnnorm_grid_re, double *Gnnorm_grid_im, 
    int *nGnpix_reso_stripe, int *Gnindices_z_reso_stripe,
    double *bin_centers, double complex *Gammans, double complex *Gammans_norm){
    
    // Index shift for the Gamman
    int _gamma_zshift = nbinsr_disc*nbinsr_disc;
    int _gamma_nshift = _gamma_zshift*nbinsz*nbinsz*nbinsz;
    int _gamma_compshift = (nmax-nmin+1)*_gamma_nshift;
    int _gammadg_zshift = nbinsr_grid*nbinsr_disc;
    int _gammadg_nshift = _gammadg_zshift*nbinsz*nbinsz*nbinsz;
    int _gammadg_compshift = (nmax-nmin+1)*_gammadg_nshift;
    
    double *totcounts = calloc(nbinsz*nbinsr_disc, sizeof(double));
    double *totnorms = calloc(nbinsz*nbinsr_disc, sizeof(double));
           
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
        double *tmpwcounts = calloc(nthreads*nbinsz*nbinsr_disc, sizeof(double));
        double *tmpwnorms = calloc(nthreads*nbinsz*nbinsr_disc, sizeof(double));
        double complex *tmpGammans_dd = calloc(nthreads*4*_gamma_compshift, sizeof(double complex));
        double complex *tmpGammans_dd_norm = calloc(nthreads*_gamma_compshift, sizeof(double complex));
        double complex *tmpGammans_dg = calloc(nthreads*4*_gammadg_compshift, sizeof(double complex));
        double complex *tmpGammans_dg_norm = calloc(nthreads*_gammadg_compshift, sizeof(double complex));
        double complex *tmpGammans_gd = calloc(nthreads*4*_gammadg_compshift, sizeof(double complex));
        double complex *tmpGammans_gd_norm = calloc(nthreads*_gammadg_compshift, sizeof(double complex));
        #pragma omp parallel for num_threads(nthreads)
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            // Figure out how many n values we need
            int nnvals, nnvals_norm;
            if (nmin<4){nmin=0;}
            if (nmin==0){nnvals=2*nmax+3;nnvals_norm=nmax+1;}
            else{nnvals=2*(nmax-nmin+3);nnvals_norm=nmax-nmin+1;}
            
            // Various shift indices for the Gammas
            int gamma_zshift_dd = nbinsr_disc*nbinsr_disc;
            int gamma_nshift_dd = gamma_zshift_dd*nbinsz*nbinsz*nbinsz;
            int gamma_compshift_dd = (nmax-nmin+1)*gamma_nshift_dd;
            int gamma_zshift_dg = nbinsr_disc*nbinsr_grid;
            int gamma_nshift_dg = gamma_zshift_dg*nbinsz*nbinsz*nbinsz;
            int gamma_compshift_dg = (nmax-nmin+1)*gamma_nshift_dg;
            int gamma_zshift_gg = nbinsr_grid*nbinsr_grid;
            int gamma_nshift_gg = gamma_zshift_gg*nbinsz*nbinsz*nbinsz;
            int gamma_compshift_gg = (nmax-nmin+1)*gamma_nshift_gg;
            
            // Allocate some helpers for index counting of grid based quantities
            int *cumnbinsr_grid = calloc(nreso+1, sizeof(int));
            int *resoofrbin = calloc(nbinsr_grid, sizeof(int));
            int *Gn_resoshift = calloc(nreso+1, sizeof(int));
            int *stripeGn_resoshift = calloc(nreso+1, sizeof(int));
            int *field_cumpixshifts = calloc(nreso*nbinsz+1, sizeof(int)); // shape (nreso, nz)
            int *field_totpixperz = calloc(nbinsz, sizeof(int)); 
            int *field_pixperreso = calloc(nreso, sizeof(int));
            int *field_resoshift = calloc(nreso+1, sizeof(int));
            int *field_cumzresoshifts = calloc(nreso*(nbinsz+1), sizeof(int)); // shape (nz+1, nreso)
            for (int i=0; i<nreso; i++){
                cumnbinsr_grid[i+1] = cumnbinsr_grid[i] + nbinsr_reso[i];
                Gn_resoshift[i+1] = Gn_resoshift[i] + cumnbinsr_grid[i+1]*nGnpix_reso[i];
                stripeGn_resoshift[i+1] = stripeGn_resoshift[i] + nbinsr_disc*nGnpix_reso[i];
                for (int j=0; j<nbinsz; j++){
                    field_cumpixshifts[i*nbinsz+j+1] = field_cumpixshifts[i*nbinsz+j] + nfieldpix_z_reso[i*nbinsz+j];
                    field_totpixperz[j] += nfieldpix_z_reso[i*nbinsz+j];
                    field_pixperreso[i] += nfieldpix_z_reso[i*nbinsz+j];
                    field_cumzresoshifts[(j+1)*nreso+i] = field_cumzresoshifts[j*nreso+i] + nfieldpix_z_reso[i*nbinsz+j];
                }
                field_resoshift[i+1] = field_resoshift[i] + field_pixperreso[i];
            }
            thisreso=0;
            for (int i=0; i<nbinsr_grid; i++){
                if (i>=cumnbinsr_grid[thisreso+1]){thisreso += 1;}
                resoofrbin[i] =  thisreso;
            }
            int Gn_zshift = Gn_resoshift[nreso];
            int Gn_nshift = nbinsz * Gn_zshift;
            int stripeGn_zshift = stripeGn_resoshift[nreso];
            int stripeGn_nshift = nbinsz * stripeGn_zshift;
            // Allocate Gns that the discrete data will be put onto
            double complex *Gn_discgrid = calloc(nnvals*stripeGn_nshift, sizeof(double complex));
            double complex *Gnnorm_discgrid = calloc(nnvals_norm*stripeGn_nshift, sizeof(double complex));
            for (int ind_gal=0; ind_gal<ngal; ind_gal++){
                if (gal2stripe[ind_gal] != 2*thisthread + odd;){continue;}
                // Check if galaxy falls in stripe used in this process
                double p11, p12, w1, e11, e12;
                int zbin1;
                p11 = pos1[ind_gal];
                p12 = pos2[ind_gal];
                w1 = weight[ind_gal];
                zbin1 = zbins[ind_gal];
                e11 = e1[ind_gal];
                e12 = e2[ind_gal];                
                
                int ind_pix1, ind_pix2, ind_inpix, ind_gal2;
                int elreso, ind_red, lower, upper; 
                double  p21, p22, w2, z2, e21, e22;
                double rel1, rel2, dist, dphi;
                double complex wshape;
                int nnvals, nnvals_norm, nextn, nzero, shiftbins;
                double complex nphirot, nphirotc, phirot, phirotc, phirotm, phirotp, phirotn;
                
                // Check how many ns we need for Gn
                // Gns have shape (nnvals, nbinsz, nbinsr)
                // where the ns are ordered as 
                //  * [-nmax-3, ..., nmax-1] / [0, ..., nmax]
                double complex *nextGns_d =  calloc(nnvals*nbinsr_disc*nbinsz, sizeof(double complex));
                double complex *nextGns_d_norm =  calloc(nnvals_norm*nbinsr_disc*nbinsz, sizeof(double complex));
                int ind_rbin, rbin;
                int ind_Gn, ind_Gnnorm, zrshift, nextnshift, index_alloc;
                int nbinszr = nbinsz*nbinsr_disc;
                
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
                            if(dist < rbins_disc[0] || dist >= rbins_disc[nrbins_disc-1]) continue;
                            rbin=0;
                            while(rbins_disc[rbin+1] < dist){rbin+=1;}
                            wshape = (double complex) w2 * (e21+I*e22);
                            dphi = atan2(rel2,rel1);
                            
                            zrshift = z2*nbinsr_disc + rbin;
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
                                phirot = cexp(I*dphi);
                                phirotc = conj(phirot);
                                // n = 0
                                tmpwcounts[ind_rbin] += w1*w2*dist; 
                                tmpwnorms[ind_rbin] += w1*w2; 
                                nextGns_d[ind_Gn] += wshape*nphirot;
                                nextGns_d_norm[ind_Gnnorm] += w2*nphirot;
                                nphirot *= phirot;
                                nphirotc *= phirotc; 
                                // n in [1, ..., nmax-1] x {+1,-1}
                                for (nextn=1;nextn<nmax;nextn++){
                                    nextnshift = nextn*nbinszr;
                                    nextGns_d[ind_Gn+nextnshift] += wshape*nphirot;
                                    nextGns_d[ind_Gn-nextnshift] += wshape*nphirotc;
                                    nextGns_d_norm[ind_Gnnorm+nextnshift] += w2*nphirot;
                                    nphirot *= phirot;
                                    nphirotc *= phirotc; 
                                }
                                // n in [nmax, -nmax, -nmax-1, -nmax-2, -nmax-3]
                                nextGns_d_norm[ind_Gnnorm+nextnshift+nbinszr] += w2*nphirot;  
                                nextGns_d[zrshift+3*nbinszr] += wshape*nphirotc;
                                nphirotc *= phirotc; 
                                nextGns_d[zrshift+2*nbinszr] += wshape*nphirotc;
                                nphirotc *= phirotc; 
                                nextGns_d[zrshift+nbinszr] += wshape*nphirotc;
                                nphirotc *= phirotc; 
                                nextGns_d[zrshift] += wshape*nphirotc;
                            }
                            
                            // nmin>3 
                            //   --> Gns axis: [-nmax-3, ..., -nmin+1, nmin-3, ..., nmax+1]
                            //   --> Gn_norm axis: [nmin, ..., nmax]
                            else{
                                phirot = cexp(dphi);
                                phirotc = conj(phirot);
                                phirotm = cpow(phirotc,nmax+3);
                                phirotp = cpow(phirot,nmin-3);
                                phirotn = phirotp*phirot*phirot*phirot;
                                int pshift = (nmax-nmin+3)*nbinszr;
                                nextnshift = zrshift;
                                // n in [-nmax-3, ..., -nmin-3] + [nmin-3, ..., nmax-3]
                                for (nextn=0;nextn<nmax-nmin+1;nextn++){
                                    nextGns_d[nextnshift] += wshape*phirotm;
                                    nextGns_d[pshift+nextnshift] += wshape*phirotp;
                                    nextGns_d_norm[nextnshift] += w2*phirotn;
                                    phirotm *= phirot;
                                    phirotp *= phirot;
                                    phirotn *= phirot;
                                    nextnshift += nbinszr;
                                }
                                // n in [-nmin-2, -nmin-1] + [nmax-2, nmax-1]
                                nextGns_d[nextnshift] += wshape*phirotm;
                                nextGns_d[pshift+nextnshift] += wshape*phirotp;
                                phirotm *= phirot;
                                phirotp *= phirot;
                                nextnshift += nbinszr;
                                nextGns_d[nextnshift] += wshape*phirotm;
                                nextGns_d[pshift+nextnshift] += wshape*phirotp;
                            } 
                        }
                    }
                }
                
                // Allocate the discrete Gns on the grid
                int ind_Gndiscgrid_base, ind_Gndiscgrid;
                if (nmin==0){
                    for (int indn=0; indn<nnvals; indn++){
                        for (int indz=0; indz<nbinsz; indz++){
                            for (int indreso=0; ireso<nreso; ireso++){
                                ind_Gndiscgrid_base = indn*stripeGn_nshift + 
                                    indz*stripeGn_zshift + stripeGn_resoshift[elreso] + 
                                    gal2redpix[elreso*ngal+ind_gal];
                                for (int indr=0; indr<nbinsr_reso[indreso]; ireso++){
                                    ind_Gndiscgrid = ind_Gndiscgrid_base + indr*nGnpix_reso[i];
                                    Gn_discgrid[ind_Gndiscgrid] += nextGns_d[indn*nbinszr+indz*nbinsr+indr];
                                }
                            } 
                        }  
                    }
                }
                
                // Now update the Gammans (dd)
                // tmpGammas have shape (nthreads, nmax+1, nzcombis, r*r, 4)
                // Gns have shape (nnvals, nbinsz, nbinsr)
                //int nonzero_tmpGammas = 0;
                double complex h0, h1, h2, h3, w0;
                int thisnshift, thisnshift_n, r12shift;
                int gammashiftn, gammashift1, gammashiftt1, gammashift, gammashiftt;
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
                        for (int elb1=0; elb1<nbinsr_disc; elb1++){
                            zrshift = zbin2*nbinsr_disc + elb1;
                            h0 = -wshape * nextGns_d[ind_nm3 + zrshift];
                            h1 = -conj(wshape) * nextGns_d[ind_nm1 + zrshift];
                            h2 = -wshape * conj(nextGns_d[ind_mnm1 + zrshift]);
                            h3 = -wshape * conj(nextGns_d[ind_nm1 + zrshift]);
                            w0 = weight[ind_gal] * conj(nextGns_d_norm[ind_norm + zrshift]);
                            for (zbin3=0; zbin3<nbinsz; zbin3++){
                                zcombi = zbin1*nbinsz*nbinsz+zbin2*nbinsz+zbin3;
                                gammashift1 = thisnshift + zcombi*gamma_zshift + elb1*nbinsr_disc;
                                gammashiftt1 = thisnshift + zcombi*gamma_zshift;
                                for (elb2=0; elb2<nbinsr_disc; elb2++){
                                    zrshift = zbin3*nbinsr_disc + elb2;
                                    r12shift = elb2*nbinsr_disc+elb1;
                                    gammashift = 4*(gammashift1 + elb2);
                                    gammashiftt = gammashiftt1 + r12shift;
                                    tmpGammans_dd[gammashift] += h0*nextGns_d[ind_mnm3 + zrshift];
                                    tmpGammans_dd[gammashift+1] += h1*nextGns_d[ind_mnm1 + zrshift];
                                    tmpGammans_dd[gammashift+2] += h2*nextGns_d[ind_mnm3 + zrshift];
                                    tmpGammans_dd[4*gammashiftt+3] += h3*nextGns_d[ind_nm3 + zrshift];
                                    tmpGammans_dd_norm[gammashiftt] += w0*nextGns_d_norm[ind_norm + zrshift];
                                }
                            }
                        }
                    }
                }
                free(nextGns_d);
                free(nextGns_d_norm);
                nextGns_d = NULL;
                nextGns_d_norm = NULL;
            }
            
            // Update the Gammans (dg)
            
            // Update the Gammans (gd)
            
            
            free(cumnbinsr_grid);
            free(resoofrbin);
            free(Gn_resoshift);
            free(stripeGn_resoshift);
            free(field_cumpixshifts);
            free(field_totpixperz); 
            free(field_pixperreso);
            free(field_resoshift);
            free(field_cumzresoshifts);
            free(Gn_discgrid);
            free(Gnnorm_discgrid);
        }
        
        // Accumulate the Gamman (dd)
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
                double tmpcounts = 0;
                double tmpnorms = 0;
                for (int thisthread=0; thisthread<nthreads; thisthread++){
                    int tshift = thisthread*nbinsz*nbinsr; 
                    totcounts[tmpind] += tmpwcounts[tshift+tmpind];
                    totnorms[tmpind] += tmpwnorms[tshift+tmpind];
                }
            }
        }
        free(tmpwcounts);
        free(tmpwnorms);
        free(tmpGammans_dd);
        free(tmpGammans_dd_norm); 
        free(tmpGammans_dg);
        free(tmpGammans_dg_norm); 
        free(tmpGammans_gd);
        free(tmpGammans_gd_norm); 
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
}*/

// Calculates Gn s.t. one can allocate Gamman for nth multipole
// Gns have shape (5, ngal, nz, nr) where the 0th axis has components [-n-3,-n-1,n-3,n-1,n]
// Ranges of n are * n \in [-nmax-3, ..., -nmin+1] u [nmin-3, ..., nmax+3] (i.e. 2*(nmax-nmin+4) values) for nmin>3
//                 * n \in [-nmax-3, ..., nmax+1] (i.e. 2*nmax+5 values) otherwise
// Gns have shape (2*(nmax-nmin+4), ngal, nz, nr) / (2*nmax+4, ngal, nz, nr)
// bin_centers and counts have shape (nz, nz, nr); but are allocated first as (ngal, nz, nr) and afterwards reduced
// Memory: * ngal=1e7, nmax=40, nz=5, nr=50 --> 1e7*(3*40+5)*50*5 ~ 3.1e11 ~ 2.5TB
//         * ngal=1e6, nmax=40, nz=5, nr=10 -->                   ~ 6e9    ~ 50GB
//         * ngal=1e6, nmin=10, nmax=20, nz=5, nr=20              ~ 2.8e9  ~ 24GB
// Note that in both case we only need one cexp evaluation while most operations are cmult ones.
void alloc_Gnsingle_discrete_basic(
    double *weight, double *pos1, double *pos2, double *e1, double *e2, int *zbins, int nbinsz, int ngal, 
    int n, double rmin, double rmax, int nbinsr,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, double *bin_centers, int *counts, double complex *Gns, double complex *Gn_norm){
    
    // Need to allocate the bin centers/counts in this way to ensure parallelizability
    // At a later stage are reduced to the shape of the output
    
    // Allocate Gns
    // We do this in parallel as follows:
    // * Split survey in 2*nthreads equal area stripes along x-axis
    // * Do two parallelized iterations over galaxies
    //   - In first one only consider galaxies within stripes of even number
    //   - In second one only consider galaxies within stripes of odd number
    // --> We avoid race conditions in calling the spatial hash arrays. - This
    //    is explicitly made sure by (re)setting nthreads in the python layer.
    for (int odd=0; odd<2; odd++){
        //#pragma omp parallel for private(nbinsz, ngal, nmin, nmax, rmin, rmax, nbinsr, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, nthreads)
        #pragma omp parallel for num_threads(nthreads)
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            // Index shifts for Gn
            int indshift3 = ngal*nbinsz*nbinsr;
            int indshift2 = nbinsz*nbinsr;
            for (int ind_gal=0; ind_gal<ngal; ind_gal++){

                // Check if galaxy falls in stripe used in this process
                double p11, p12, w1;
                int z1;
                #pragma omp critical
                {p11 = pos1[ind_gal];
                p12 = pos2[ind_gal];
                w1 = weight[ind_gal];
                z1 = zbins[ind_gal];}
                int thisstripe = 2*thisthread + odd;
                int galstripe = (int) floor((p11-pix1_start)/pix1_d * (2*nthreads)/pix1_n);
                if (thisstripe != galstripe){continue;}
                
                int ind_pix1, ind_pix2, ind_inpix, ind_gal2;
                int ind_red, lower, upper; 
                double  p21, p22, w2, z2, e21, e22;
                double rel1, rel2, dist, dphi;
                double complex wshape;
                double complex nphirot, threephirot, phirot;

                int rbin;
                int thisindshift;

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
                            rbin = (int) floor((log(dist)-log(rmin))/drbin);
                            wshape = (double complex) w2 * (e21+I*e22);
                            dphi = atan2(rel2,rel1);
                            thisindshift = ind_gal*indshift2+z2*nbinsr + rbin;
                            phirot = cexp(I*dphi);
                            nphirot = cexp(I*n*dphi);
                            threephirot = phirot*phirot*phirot;
                            //nphirot = cpow(phirot,n);
                            Gns[thisindshift] += wshape*conj(nphirot*threephirot);
                            Gns[indshift3+thisindshift] += wshape*conj(nphirot*phirot);
                            Gns[2*indshift3+thisindshift] += wshape*nphirot*conj(phirot);
                            Gns[3*indshift3+thisindshift] += wshape*nphirot*conj(threephirot);
                            Gn_norm[thisindshift] += w2*nphirot;              
                        }
                    }
                }
            }
        }
    }
    
    /*
    // Finish the calculation of bin centers (cannot be parallelized)
    double *bin_centers_norm = calloc(nbinsz*nbinsz*nbinsr, sizeof(double));
    int countbin, gbinind;
    for (int ind_gal=0; ind_gal<ngal; ind_gal++){
        int zbin1 = zbins[ind_gal];
        for (int elb1=0; elb1<nbinsr; elb1++){
            for (int zbin2=0; zbin2<nbinsz; zbin2++){
                countbin = zbin1*nbinsz*nbinsr + zbin2*nbinsr + elb1;
                gbinind = ind_gal*indshift2+zbin2*nbinsr + elb1;
                bin_centers[countbin] += bin_centers_gcount[gbinind];
                bin_centers_norm[countbin] += bin_centers_gnorm[gbinind];
                counts[countbin] += gcounts[gbinind];
            }
        }
    }
    for (int elb1=0; elb1<nbinsr; elb1++){
        for (int zbin1=0; zbin1<nbinsz; zbin1++){
            for (int zbin2=0; zbin2<nbinsz; zbin2++){
                countbin = zbin1*nbinsz*nbinsr + zbin2*nbinsr + elb1;
                bin_centers[countbin] /=  bin_centers_norm[countbin];
            }
        }
    }
    free(bin_centers_norm);
    */
    
}


// Calculates Gamma_n & Gamman_bor s.t. one can allocate Gamman for nth multipole
// Gns have shape (5, ngal, nz, nr) where the 0th axis has components [-n-3,-n-1,n-3,n-1,n]
// Ranges of n are * n \in [-nmax-3, ..., -nmin+1] u [nmin-3, ..., nmax+3] (i.e. 2*(nmax-nmin+4) values) for nmin>3
//                 * n \in [-nmax-3, ..., nmax+1] (i.e. 2*nmax+5 values) otherwise
// Gns have shape (2*(nmax-nmin+4), ngal, nz, nr) / (2*nmax+4, ngal, nz, nr)
// bin_centers and counts have shape (nz, nz, nr); but are allocated first as (ngal, nz, nr) and afterwards reduced
// Memory: * ngal=1e7, nmax=40, nz=5, nr=50 --> 1e7*(3*40+5)*50*5 ~ 3.1e11 ~ 2.5TB
//         * ngal=1e6, nmax=40, nz=5, nr=10 -->                   ~ 6e9    ~ 50GB
//         * ngal=1e6, nmin=10, nmax=20, nz=5, nr=20              ~ 2.8e9  ~ 24GB
// Note that in both case we only need one cexp evaluation while most operations are cmult ones.
void alloc_Gammansingle_discretemixed_basic(
    int *isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int *zbins, int nbinsz, int ngal, 
    int n, double rmin, double rmax, int nbinsr_disc,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    double stripes_start, double stripes_d,  
    int nreso, double *pix1start_reso, double *pix2start_reso, double *pixd_reso, int *nbinsr_reso, 
    int *pix1n_reso, int *pix2n_reso, int *npixred_reso, int *npixbare_reso, int *red_indices,
    double *weight_grid, double *e1_grid, double *e2_grid, double complex *Gns_grid, int nbinsr_grid,
    double *bin_centers_disc, int *counts_disc, double complex *Gamman, double complex *Gamman_norm,
    int nthreads){
    
    // Need to allocate the bin centers/counts in this way to ensure parallelizability
    // At a later stage are reduced to the shape of the output
    
    // Various shift parameters for Gns_grid
    int *cumnpixbare_grid = calloc(nreso+1, sizeof(size_t));
    int *cumnpixred_grid = calloc(nreso+1, sizeof(size_t));
    int *cumnbinsr_grid = calloc(nreso+1, sizeof(size_t));
    int *cumresoshift_grid = calloc(nreso+1, sizeof(size_t));
    for (int elreso=0; elreso<nreso; elreso++){
        cumnpixbare_grid[elreso+1] = cumnpixbare_grid[elreso] + npixbare_reso[elreso];
        cumnpixred_grid[elreso+1] = cumnpixred_grid[elreso] + npixred_reso[elreso];
        cumnbinsr_grid[elreso+1] = cumnbinsr_grid[elreso] + nbinsr_reso[elreso];
        cumresoshift_grid[elreso+1] = cumresoshift_grid[elreso] + nbinsr_reso[elreso]*npixbare_reso[elreso];
    }
    size_t resoshift_grid = cumresoshift_grid[nreso];
    size_t nshift_grid = nbinsz*resoshift_grid;
    
    
    
    // Get the Gammans in parallel as follows:
    // * Split survey in 2*nthreads equal area stripes along x-axis
    // * Do two parallelized iterations over galaxies
    //   - In first one only consider galaxies within stripes of even number
    //   - In second one only consider galaxies within stripes of odd number
    // --> We avoid race conditions in calling the spatial hash arrays. - This
    //    is explicitly made sure by (re)setting nthreads in the python layer.
    for (int odd=0; odd<2; odd++){
        double complex *tmpGammans_dd = calloc(nthreads*4*nbinsz*nbinsz*nbinsz*nbinsr_disc*nbinsr_disc, 
                                               sizeof(double complex));
        double complex *tmpGammansnorm_dd = calloc(nthreads*nbinsz*nbinsz*nbinsz*nbinsr_disc*nbinsr_disc, 
                                                   sizeof(double complex));
        double complex *tmpGammans_dg = calloc(nthreads*4*nbinsz*nbinsz*nbinsz*nbinsr_disc*nbinsr_grid, 
                                               sizeof(double complex));
        double complex *tmpGammansnorm_dg = calloc(nthreads*nbinsz*nbinsz*nbinsz*nbinsr_disc*nbinsr_grid, 
                                                   sizeof(double complex));
        double complex *tmpGammans_gd = calloc(nthreads*4*nbinsz*nbinsz*nbinsz*nbinsr_grid*nbinsr_disc, 
                                               sizeof(double complex));
        double complex *tmpGammansnorm_gd = calloc(nthreads*nbinsz*nbinsz*nbinsz*nbinsr_grid*nbinsr_disc, 
                                                   sizeof(double complex));
        #pragma omp parallel for num_threads(nthreads)
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            // Index shifts for Gn
            int indshift2 = nbinsz*nbinsr_disc;
            for (int ind_gal=0; ind_gal<ngal; ind_gal++){
                
                // 1) Get the Gn of inner galaxies
                if (isinner[ind_gal]==0){continue;}
                
                // Load info of base galaxy - this needs to be single threaded!
                double p11, p12, e11, e12, w1; 
                int z1;
                #pragma omp critical
                {p11 = pos1[ind_gal];
                 p12 = pos2[ind_gal];
                 e11 = e1[ind_gal];
                 e12 = e2[ind_gal];
                 w1 = weight[ind_gal];
                 z1 = zbins[ind_gal];}
                // Check if galaxy falls in stripe used in this process
                int thisstripe = 2*thisthread + odd;
                int galstripe = (int) floor((p11-stripes_start)/stripes_d);
                if (thisstripe != galstripe){continue;}
                
                // Allocate helper arrays
                double complex *fourGns_disc = calloc(4*nbinsz*nbinsr_disc, sizeof(double complex));
                double complex *Gnnorm_disc = calloc(nbinsz*nbinsr_disc, sizeof(double complex));
                double *thisweights_grid = calloc(nreso, sizeof(double));
                double *thise1s_grid = calloc(nreso, sizeof(double));
                double *thise2s_grid = calloc(nreso, sizeof(double));
                
                int ind_pix1, ind_pix2, ind_inpix, ind_gal2;
                int ind_red, lower, upper; 
                double  p21, p22, w2, z2, e21, e22;
                double rel1, rel2, dist, dphi;
                double complex wshape;
                double complex nphirot, threephirot, phirot;

                int rbin;
                int thisindshift;

                double drbin = (log(rmax)-log(rmin))/(nbinsr_disc);

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
                            rbin = (int) floor((log(dist)-log(rmin))/drbin);
                            wshape = (double complex) w2 * (e21+I*e22);
                            dphi = atan2(rel2,rel1);
                            thisindshift = z2*nbinsr_disc + rbin;
                            phirot = cexp(I*dphi);
                            nphirot = cexp(I*n*dphi);
                            threephirot = phirot*phirot*phirot;
                            // Need [G_{n-3}, G_{-n-3}, G_{n-1}, G_{-n-1}] for Gamman computation
                            // Need G_n for normalization
                            fourGns_disc[thisindshift] += wshape*nphirot*conj(threephirot);
                            fourGns_disc[indshift2+thisindshift] += wshape*conj(nphirot*threephirot);
                            fourGns_disc[2*indshift2+thisindshift] += wshape*nphirot*conj(phirot);
                            fourGns_disc[3*indshift2+thisindshift] += wshape*conj(nphirot*phirot);
                            Gnnorm_disc[thisindshift] += w2*nphirot;            
                        }
                    }
                }
                
                // 2) Find relevant components of Gn_grid 
                // Note that in this function we assume an NGP scheme!
                // Recall the "shape" of Gn_grid is a flattened version of the following structure:
                // [ [(nz,nr_reso1,npixred_reso1), ..., (nz,nr_resox,npixred_resox)]_{n-3}, ..., [...]_{n} ]
                int resoshift_grid;
                int galpix_x, galpix_y, gal_redpix, ind_griddata;
                int fourGns_grid_s1;
                int fourGns_grid_s2 = nbinsz*nbinsr_grid;
                size_t thisshift_grid;
                double complex *fourGns_grid = calloc(4*nbinsr_grid*nbinsz, sizeof(double complex));
                double complex *Gnnorm_grid = calloc(nbinsr_grid*nbinsz, sizeof(double complex));
                int thisreso = 0;
                galpix_x = (int) floor((p11-pix1start_reso[0])/pixd_reso[0]);
                galpix_y = (int) floor((p12-pix2start_reso[0])/pixd_reso[0]);
                resoshift_grid = 0;
                gal_redpix = red_indices[galpix_y*pix1n_reso[thisreso]+galpix_x];
                for (int elbinr=0; elbinr<nbinsr_grid; elbinr++){
                    if (elbinr >= cumnbinsr_grid[thisreso+1]){
                        galpix_x = (int) floor((p11-pix1start_reso[thisreso])/pixd_reso[thisreso]);
                        galpix_y = (int) floor((p12-pix2start_reso[thisreso])/pixd_reso[thisreso]);
                        resoshift_grid = cumresoshift_grid[thisreso];
                        gal_redpix = red_indices[cumnpixbare_grid[thisreso]+galpix_y*pix1n_reso[thisreso]+galpix_x];
                        ind_griddata = cumnpixred_grid[thisreso] + gal_redpix;
                        thisweights_grid[thisreso] = weight_grid[ind_griddata];
                        thise1s_grid[thisreso] = e1_grid[ind_griddata];
                        thise2s_grid[thisreso] = e2_grid[ind_griddata];
                    }
                    for (int elbinz=0; elbinz<nbinsz; elbinz++){
                        fourGns_grid_s1 = elbinz*nbinsr_grid+elbinr;
                        thisshift_grid = elbinz*resoshift_grid + cumresoshift_grid[thisreso] + 
                            (elbinr-cumnbinsr_grid[thisreso])*npixred_reso[thisreso] + gal_redpix;
                        fourGns_grid[fourGns_grid_s1] = Gns_grid[thisshift_grid];
                        fourGns_grid[fourGns_grid_s2+fourGns_grid_s1] = Gns_grid[nshift_grid+thisshift_grid];
                        fourGns_grid[2*fourGns_grid_s2+fourGns_grid_s1] = Gns_grid[2*nshift_grid+thisshift_grid];
                        fourGns_grid[3*fourGns_grid_s2+fourGns_grid_s1] = Gns_grid[3*nshift_grid+thisshift_grid];
                        Gnnorm_grid[fourGns_grid_s1] = Gns_grid[4*nshift_grid+thisshift_grid];
                    }  
                    if (elbinr >= cumnbinsr_grid[thisreso+1]){thisreso+=1;}
                }
                
                // 3) Update the (dd, dg, gd) blocks of Gamman for this galaxy 
                update_Gammansingle_discmixed_worker(
                        fourGns_disc, Gnnorm_disc, fourGns_grid, Gnnorm_grid, 
                        nbinsr_disc, nbinsr_grid, nbinsz, nreso,
                        thisthread, z1, w1, e11, e12,
                        thisweights_grid, thise1s_grid, thise2s_grid, cumnbinsr_grid,
                        tmpGammans_dd, tmpGammansnorm_dd, 
                        tmpGammans_dg, tmpGammansnorm_dg, 
                        tmpGammans_gd, tmpGammansnorm_gd);
                
                // 3a) If not yet done, update the gg component of this pixel 
                                
                free(thisweights_grid);
                free(thise1s_grid);
                free(thise2s_grid);
                free(fourGns_disc);
                free(Gnnorm_disc);
                free(fourGns_grid);
                free(Gnnorm_grid);
            }
        }
        // 4*nbinsz*nbinsz*nbinsz*nbinsr_disc*nbinsr_disc
        // 4*nbinsz*nbinsz*nbinsz*(nbinsr_disc+nbinsr)*(nbinsr_disc+nbinsr)
        // 4) Accumulate the (dd, dg, gd) blocks and put them in the Gamman & Gamman_norm output
        #pragma omp parallel for num_threads(mymin(4,nthreads))
        for (int thisthread=0; thisthread<4; thisthread++){
            int gammaax0, gammaax1, gammaax2, gammaax3;
            int indshiftthread, indshift0, indshift1, indshift2, indshift3, indshift4, indshift;
            int indshift0_g, indshift1_g, indshift2_g, indshift4_g, indshift_g;
            int nbinsr_tot = nbinsr_disc+nbinsr_grid;
            int gammaax0_g = nbinsz*nbinsz*nbinsz*nbinsr_tot*nbinsr_tot;
            int gammaax1_g = nbinsz*nbinsz*nbinsr_tot*nbinsr_tot;
            int gammaax2_g = nbinsz*nbinsr_tot*nbinsr_tot;
            int gammaax3_g = nbinsr_tot*nbinsr_tot;
            int indshift3_g;
            // Accumulate Gamman_dd & Gammannorm_dd
            if (thisthread==0){
                indshiftthread = 4*nbinsz*nbinsz*nbinsz*nbinsr_disc*nbinsr_disc;
                gammaax0 = nbinsz*nbinsz*nbinsz*nbinsr_disc*nbinsr_disc;
                gammaax1 = nbinsz*nbinsz*nbinsr_disc*nbinsr_disc;
                gammaax2 = nbinsz*nbinsr_disc*nbinsr_disc;
                gammaax3 = nbinsr_disc*nbinsr_disc;
                for (int elcomp=0; elcomp<4; elcomp++){
                    indshift0 = elcomp*gammaax0;
                    indshift0_g = elcomp*gammaax0_g;
                    for (int elz1=0; elz1<nbinsz; elz1++){
                        indshift1 = indshift0+elz1*gammaax1;
                        indshift1_g = indshift0_g+elz1*gammaax1_g;
                        for (int elz2=0; elz2<nbinsz; elz2++){
                            indshift2 = indshift1+elz2*gammaax2;
                            indshift2_g = indshift1_g+elz2*gammaax2_g;
                            for (int elz3=0; elz3<nbinsz; elz3++){
                                indshift3 = indshift2+elz3*gammaax3;
                                indshift3_g = indshift2_g+elz3*gammaax3_g;
                                for (int elr1=0; elr1<nbinsr_disc; elr1++){
                                    indshift4 = indshift3+elr1*nbinsr_disc;
                                    indshift4_g = indshift3+elr1*nbinsr_tot;
                                    for (int elr2=0; elr2<nbinsr_disc; elr2++){
                                        indshift = indshift4+elr2;
                                        indshift_g = indshift4_g+elr2;
                                        for (int elthread=0; elr2<nthreads; elthread++){
                                            Gamman[indshift_g] += tmpGammans_dd[elthread*indshiftthread + indshift];
                                            if (elcomp==0){
                                                Gamman_norm[indshift_g] += tmpGammansnorm_dd[elthread*indshiftthread + indshift];
                                            } 
                                        }
                                    }
                                }  
                            }
                        }
                    } 
                }
            }
            // Accumulate Gamman_dg & Gammannorm_dg
            if (thisthread==1){
                gammaax0 = nbinsz*nbinsz*nbinsz*nbinsr_disc*nbinsr_grid;
                gammaax1 = nbinsz*nbinsz*nbinsr_disc*nbinsr_grid;
                gammaax2 = nbinsz*nbinsr_disc*nbinsr_grid;
                gammaax3 = nbinsr_disc*nbinsr_grid;
                for (int elcomp=0; elcomp<4; elcomp++){
                    indshift0 = elcomp*gammaax0;
                    indshift0_g = elcomp*gammaax0_g;
                    for (int elz1=0; elz1<nbinsz; elz1++){
                        indshift1 = indshift0+elz1*gammaax1;
                        indshift1_g = indshift0_g+elz1*gammaax1_g;
                        for (int elz2=0; elz2<nbinsz; elz2++){
                            indshift2 = indshift1+elz2*gammaax2;
                            indshift2_g = indshift1_g+elz2*gammaax2_g;
                            for (int elz3=0; elz3<nbinsz; elz3++){
                                indshift3 = indshift2+elz3*gammaax3;
                                indshift3_g = indshift2_g+elz3*gammaax3_g;
                                for (int elr1=0; elr1<nbinsr_disc; elr1++){
                                    indshift4 = indshift3+elr1*nbinsr_disc;
                                    indshift4_g = indshift3+elr1*nbinsr_tot;
                                    for (int elr2=0; elr2<nbinsr_grid; elr2++){
                                        indshift = indshift4+elr2;
                                        indshift_g = indshift4_g+nbinsr_disc+elr2;
                                        for (int elthread=0; elr2<nthreads; elthread++){
                                            Gamman[indshift_g] += tmpGammans_dg[elthread*indshiftthread + indshift];
                                            if (elcomp==0){
                                                Gamman_norm[indshift_g] += tmpGammansnorm_dg[elthread*indshiftthread + indshift];
                                            } 
                                        }
                                    }
                                }  
                            }
                        }
                    } 
                }
            }
            // Accumulate Gamman_gd & Gammannorm_gd
            if (thisthread==2){
                gammaax0 = nbinsz*nbinsz*nbinsz*nbinsr_grid*nbinsr_disc;
                gammaax1 = nbinsz*nbinsz*nbinsr_grid*nbinsr_disc;
                gammaax2 = nbinsz*nbinsr_grid*nbinsr_disc;
                gammaax3 = nbinsr_grid*nbinsr_disc;
                for (int elcomp=0; elcomp<4; elcomp++){
                    indshift0 = elcomp*gammaax0;
                    indshift0_g = elcomp*gammaax0_g;
                    for (int elz1=0; elz1<nbinsz; elz1++){
                        indshift1 = indshift0+elz1*gammaax1;
                        indshift1_g = indshift0_g+elz1*gammaax1_g;
                        for (int elz2=0; elz2<nbinsz; elz2++){
                            indshift2 = indshift1+elz2*gammaax2;
                            indshift2_g = indshift1_g+elz2*gammaax2_g;
                            for (int elz3=0; elz3<nbinsz; elz3++){
                                indshift3 = indshift2+elz3*gammaax3;
                                indshift3_g = indshift2_g+elz3*gammaax3_g;
                                for (int elr1=0; elr1<nbinsr_grid; elr1++){
                                    indshift4 = indshift3+elr1*nbinsr_disc;
                                    indshift4_g = indshift3+(nbinsr_disc+elr1)*nbinsr_tot;
                                    for (int elr2=0; elr2<nbinsr_disc; elr2++){
                                        indshift = indshift4+elr2;
                                        indshift_g = indshift4_g+elr2;
                                        for (int elthread=0; elr2<nthreads; elthread++){
                                            Gamman[indshift_g] += tmpGammans_gd[elthread*indshiftthread + indshift];
                                            if (elcomp==0){
                                                Gamman_norm[indshift_g] += tmpGammansnorm_gd[elthread*indshiftthread + indshift];
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
        free(tmpGammans_dd);
        free(tmpGammansnorm_dd);
        free(tmpGammans_dg);
        free(tmpGammansnorm_dg);
        free(tmpGammans_gd);
        free(tmpGammansnorm_gd);
    }
}

/*
// Calculates Gamman for discrete/discrete and discrete/grid pairs of radial bins
// Gammans are three things:
// * Gamma_n^(disc,disc) of shape (nthreads, nnvals, 5, nz, nz, nz, nr_disc, nr_disc)
// * Gamma_n^(disc,grid) of shape (nthreads, nnvals, 5, nz, nz, nz, nr_disc, nr_grid)
// * Gamma_n^(grid,disc) of shape (nthreads, nnvals, 5, nz, nz, nz, nr_grid, nr_disc)
// * Gns on patch of shape (nnvals, nz, nz, nr_disc, npix_patch)
// Strategy is as follows
// * For (even, odd):
//   * For stripe:
//     * For each subpatch:
//       * Compute size of pixelgrid in this subpatch
//       * Allocate memory for G_ns^patch; size = (nnvals, nzbins, nzbins, nthet_res, npixs_subpatch_res)
//       * For each galaxy in subpatch
//         * Compute G_ns^discrete 
//         * Update Gamma_n^(disc,disc) ~ gamma_igal * G_ns^discrete * G_ns^discrete
//         * Paint Gns^discrete to grids with corresponding MAS scheme and update G_ns^patch
//       * Update Gamma_n^(disc,grid) on subpatch via Gamma_ns^mixed ~ gamma_c * G_ns^patch_c * G_ns^grid_c
//         Update Gamma_n^(grid,disc) on subpatch via Gamma_ns^mixed ~ gamma_c * G_ns^grid_c * G_ns^patch_c
// * Accumulate Gamma_n^(disc,disc), Gamma_n^(disc,grid) and Gamma_n^(grid,disc)
// When using NGP strategy is as follows
// * For (even, odd):
//   * For stripe:
//     * For each subpatch:
//       * Compute size of pixelgrid in this subpatch
//       * Allocate memory for G_ns^patch; size = (nnvals, nzbins, nzbins, nthet_res, npixs_subpatch_res)
//       * For each galaxy in subpatch
//         * Compute G_ns^discrete 
//         * Update Gamma_n^(disc,disc) ~ gamma_igal * G_ns^discrete * G_ns^discrete
//         * Check which reduced pixel numbers c the galaxy belongs to
//           Update Gamma_n^(disc,grid) via Gamma_ns^mixeda ~ gamma_c * G_ns^discrete * G_ns^grid_c
//           Update Gamma_n^(grid,disc) via Gamma_ns^mixeda ~ gamma_c * G_ns^grid_c * G_ns^discrete
//           where gamma_c denotes the shear value of the grid based method in pixel number c
// * Accumulate Gamma_n^(disc,disc), Gamma_n^(disc,grid) and Gamma_n^(grid,disc)
// Ranges of n are * n \in [-nmax-3, ..., -nmin+1] u [nmin-3, ..., nmax+3] (i.e. 2*(nmax-nmin+4) values) for nmin>3
//                 * n \in [-nmax-3, ..., nmax+1] (i.e. 2*nmax+5 values) otherwise
// Memory: * Euclid/LSST (tomo): ngal=1e7, nmax=40, nz=10, nr_disc=40, nr_grid=20/20, npix=1e6/3e5, npatch=1000, nthreads=20 
//           --> Gn_patch            ~ 85*10*10*(20*1e6+20*3e5)/1e3 ~ 2e8 (4e9 across threads)
//           --> Gamma_n^(disc,disc) ~ 20*41*5*10*10*10*40*40 ~ 6e9
//           --> Gamma_n^(grid,disc) ~ 20*41*5*10*10*10*(20+20)*40 ~ 6e9
//         * KiDS (tomo): ngal=1e6, nmax=20, nz=5, nr_disc=20, nr_grid=20/20, npix=1e6/3e5, npatch=1000, nthreads=20 
//           --> Gn_patch            ~ 45*5*5*(20*1e6+20*3e5)/1e3 ~ 2.5e7 (5e8 across threads)
//           --> Gamma_n^(disc,disc) ~ 20*21*5*5*5*5*20*20 ~ 1e8
//           --> Gamma_n^(grid,disc) ~ 20*21*5*5*5*5*20*40 ~ 2e8
//         * Euclid/LSST (no tomo): ngal=1e7, nmax=40, nz=1, nr_disc=40, nr_grid=20/20, npix=1e6/3e5, npatch=1000, nthreads=20 
//           --> Gn_patch            ~ 85*1*1*(20*1e6+20*3e5)/1e3 ~ 2e6 (4e6 across threads)
//           --> Gamma_n^(disc,disc) ~ 20*41*5*1*1*1*40*40 ~ 6e6
//           --> Gamma_n^(grid,disc) ~ 20*41*5*1*1*1*(20+20)*40 ~ 6e6
// Note that in both case we only need one cexp evaluation while most operations are cmult ones.
void alloc_Gammans_discretemixed(
    int *isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int *zbins, int nbinsz, int ngal, 
    int nmin, int nmax, double rmin, double rmax, int nbinsr_disc,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    double *stripe_bounds, int nstripes, int npatches_1, int npatches_2, int *galpatchinds, 
    int nreso, int mas, int *dpix_res, int *nbinsr_res, int *npixred_res, int *red_indices, double *red_weights,
    double complex *GnsFFT,
    double *bin_centers_disc, int *counts_disc, 
    double complex *Gammans_discdisc, double complex *Gammans_discgrid, double complex *Gammans_griddisc,
    int n_threads){
    
    // Need to allocate the bin centers/counts in this way to ensure parallelizability
    // At a later stage are reduced to the shape of the output
    double *bin_centers_gcount = calloc(ngal*nbinsz*nbinsr, sizeof(double));
    double *bin_centers_gnorm = calloc(ngal*nbinsz*nbinsr, sizeof(double));
    int *gcounts = calloc(ngal*nbinsz*nbinsr, sizeof(int));
    
    // Allocate Gns
    // We do this in parallel as follows:
    // * Split survey in 2*nthreads equal area stripes along x-axis
    // * Do two parallelized iterations over galaxies
    //   - In first one only consider galaxies within stripes of even number
    //   - In second one only consider galaxies within stripes of odd number
    // --> We avoid race conditions in calling the spatial hash arrays. - This
    //    is explicitly made sure by (re)setting nthreads in the python layer.
    for (int odd=0; odd<2; odd++){
        //#pragma omp parallel for private(nbinsz, ngal, nmin, nmax, rmin, rmax, nbinsr, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, nthreads)
        #pragma omp parallel for num_threads(nthreads)
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            // Index shifts for Gn
            int indshift3 = ngal*nbinsz*nbinsr;
            int indshift2 = nbinsz*nbinsr;
            for (int ind_gal=0; ind_gal<ngal; ind_gal++){

                // Check if galaxy falls in stripe used in this process
                double p11, p12, w1;
                int z1;
                #pragma omp critical
                {p11 = pos1[ind_gal];
                p12 = pos2[ind_gal];
                w1 = weight[ind_gal];
                z1 = zbins[ind_gal];}
                int thisstripe = 2*thisthread + odd;
                int galstripe = (int) floor((p11-pix1_start)/pix1_d * (2*nthreads)/pix1_n);
                if (thisstripe != galstripe){continue;}
                
                int ind_pix1, ind_pix2, ind_inpix, ind_gal2;
                int ind_red, ind_gn, lower, upper; 
                double  p21, p22, w2, z2, e21, e22;
                double rel1, rel2, dist, dphi;
                double complex wshape;
                int nextn, nzero, shiftbins;
                double complex nphirot, nphirotc, phirot, phirotc, phirotm, phirotp, phirotn;

                int rbin;
                int thisindshift, index_alloc;

                if (nmin<4){nmin=0;}
                double drbin = (log(rmax)-log(rmin))/(nbinsr);

                int pix1_lower = mymax(0, (int) floor((p11 - (rmax+pix1_d) - pix1_start)/pix1_d));
                int pix2_lower = mymax(0, (int) floor((p12 - (rmax+pix2_d) - pix2_start)/pix2_d));
                int pix1_upper = mymin(pix1_n-1, (int) floor((p11 + (rmax+pix1_d) - pix1_start)/pix1_d));
                int pix2_upper = mymin(pix2_n-1, (int) floor((p12 + (rmax+pix2_d) - pix2_start)/pix2_d));

                for (ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                    for (ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                        ind_red = index_matcher[ind_pix2*pix1_n + ind_pix1];
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
                            rbin = (int) floor((log(dist)-log(rmin))/drbin);
                            wshape = (double complex) w2 * (e21+I*e22);
                            dphi = atan2(rel2,rel1);
                            thisindshift = ind_gal*indshift2+z2*nbinsr + rbin;

                            // nmin=0 -
                            //   -> Gns axis: [-nmax-3, ..., nmax+1]
                            //   -> Gn_norm axis: [0,...,nmax]
                            if (nmin==0){
                                nzero = nmax+3;
                                nphirot = 1+I*0;
                                nphirotc = 1+I*0;
                                phirot = cexp(I*dphi);
                                phirotc = conj(phirot);
                                int tmpGindp = nzero*indshift3 + thisindshift;
                                int tmpGindm = nzero*indshift3 + thisindshift;
                                int tmpGindn = thisindshift;
                                // n = 0
                                //bin_centers_gcount[thisindshift] += w1*w2*dist; 
                                //bin_centers_gnorm[thisindshift] += w1*w2; 
                                //gcounts[thisindshift] += 1; 
                                Gns[tmpGindp] += wshape*nphirot;
                                Gns_norm[tmpGindn] += w2*nphirot;  
                                nphirot *= phirot;
                                nphirotc *= phirotc; 
                                tmpGindp += indshift3;
                                tmpGindm -= indshift3;
                                tmpGindn += indshift3;
                                // n in [1, ..., nmax] x {+1,-1}
                                for (nextn=1;nextn<nmax+1;nextn++){
                                    Gns[tmpGindp] += wshape*nphirot;
                                    Gns[tmpGindm] += wshape*nphirotc;
                                    Gns_norm[tmpGindn] += w2*nphirot;  
                                    nphirot *= phirot;
                                    nphirotc *= phirotc; 
                                    tmpGindp += indshift3;
                                    tmpGindm -= indshift3;
                                    tmpGindn += indshift3;
                                }
                                // n in [-nmax-1, nmax+1]
                                Gns[tmpGindp] += wshape*nphirot;
                                Gns[tmpGindm] += wshape*nphirotc;
                                nphirotc *= phirotc; 
                                // n in [-nmax-2, -nmax=3]
                                Gns[indshift3+thisindshift] += wshape*nphirotc;
                                nphirotc *= phirotc; 
                                Gns[thisindshift] += wshape*nphirotc;
                            }
                            // nmin>3 
                            //   --> Gns axis: [-nmax-3, ..., -nmin+1, nmin-3, ..., nmax+1]
                            //   --> Gn_norm axis: [nmin, ..., nmax]
                            else{
                                phirot = cexp(dphi);
                                phirotc = conj(phirot);
                                phirotp = cpow(phirotc,nmax+3);
                                phirotm = cpow(phirot,nmin-3);
                                phirotn = phirotm*phirot*phirot*phirot;
                                int tmpGindm = thisindshift;
                                int tmpGindp = (nmin+nmax)*indshift3 + thisindshift;
                                // n in [0, ..., nmax-nmin] + {-nmax-3, nmin-3, nmin}
                                for (nextn=0;nextn<nmax-nmin+1;nextn++){
                                    Gns[tmpGindm] += wshape*phirotm;
                                    Gns[tmpGindp] += wshape*phirotp;
                                    Gns_norm[tmpGindm] += w2*phirotn;
                                    phirotm *= phirot;
                                    phirotp *= phirot;
                                    phirotn *= phirot;
                                    tmpGindm += indshift3;
                                    tmpGindp += indshift3;
                                }
                                // n in [nmax-nmin+1, nmax-nmin+2, nmax-nmin+3] + {-nmax-3, nmin-3}
                                Gns[tmpGindm] += wshape*phirotm;
                                Gns[tmpGindp] += wshape*phirotp;
                                phirotm *= phirot;
                                phirotp *= phirot;
                                tmpGindm += indshift3;
                                tmpGindp += indshift3;
                                Gns[tmpGindm] += wshape*phirotm;
                                Gns[tmpGindp] += wshape*phirotp;
                                phirotm *= phirot;
                                phirotp *= phirot;
                                tmpGindm += indshift3;
                                tmpGindp += indshift3;
                                Gns[tmpGindm] += wshape*phirotm;
                                Gns[tmpGindp] += wshape*phirotp;
                            }               
                        }
                    }
                }
            }
        }
    }
    
    
    // Finish the calculation of bin centers (cannot be parallelized)
    double *bin_centers_norm = calloc(nbinsz*nbinsz*nbinsr, sizeof(double));
    int countbin, gbinind;
    for (int ind_gal=0; ind_gal<ngal; ind_gal++){
        int zbin1 = zbins[ind_gal];
        for (int elb1=0; elb1<nbinsr; elb1++){
            for (int zbin2=0; zbin2<nbinsz; zbin2++){
                countbin = zbin1*nbinsz*nbinsr + zbin2*nbinsr + elb1;
                gbinind = ind_gal*indshift2+zbin2*nbinsr + elb1;
                bin_centers[countbin] += bin_centers_gcount[gbinind];
                bin_centers_norm[countbin] += bin_centers_gnorm[gbinind];
                counts[countbin] += gcounts[gbinind];
            }
        }
    }
    for (int elb1=0; elb1<nbinsr; elb1++){
        for (int zbin1=0; zbin1<nbinsz; zbin1++){
            for (int zbin2=0; zbin2<nbinsz; zbin2++){
                countbin = zbin1*nbinsz*nbinsr + zbin2*nbinsr + elb1;
                bin_centers[countbin] /=  bin_centers_norm[countbin];
            }
        }
    }
    free(bin_centers_norm);
    
    
    free(gcounts);
    free(bin_centers_gcount);
    free(bin_centers_gnorm);
}
*/
    
// Calculates Gamman for discrete/discrete and discrete/grid pairs of radial bins
// Gammans are three things:
// * Gamma_n^(disc,disc) of shape (nthreads, nnvals, 5, nz, nz, nz, nr_disc, nr_disc)
// * Gamma_n^(disc,grid) of shape (nthreads, nnvals, 5, nz, nz, nz, nr_disc, nr_grid)
// * Gamma_n^(grid,disc) of shape (nthreads, nnvals, 5, nz, nz, nz, nr_grid, nr_disc)
// * Gns on patch of shape (nnvals, nz, nz, nr_disc, npix_patch)
// When using NGP strategy is as follows
// * For (even, odd):
//   * For stripe:
//     * For each subpatch:
//       * Compute size of pixelgrid in this subpatch
//       * Allocate memory for G_ns^patch; size = (nnvals, nzbins, nzbins, nthet_res, npixs_subpatch_res)
//       * For each galaxy in subpatch
//         * Compute G_ns^discrete 
//         * Update Gamma_n^(disc,disc) ~ gamma_igal * G_ns^discrete * G_ns^discrete
//         * Check which reduced pixel numbers c the galaxy belongs to
//           Update Gamma_n^(disc,grid) via Gamma_ns^mixeda ~ gamma_c * G_ns^discrete * G_ns^grid_c
//           Update Gamma_n^(grid,disc) via Gamma_ns^mixeda ~ gamma_c * G_ns^grid_c * G_ns^discrete
//           where gamma_c denotes the shear value of the grid based method in pixel number c
// * Accumulate Gamma_n^(disc,disc), Gamma_n^(disc,grid) and Gamma_n^(grid,disc)
// Ranges of n are * n \in [-nmax-3, ..., -nmin+1] u [nmin-3, ..., nmax+3] (i.e. 2*(nmax-nmin+4) values) for nmin>3
//                 * n \in [-nmax-3, ..., nmax+1] (i.e. 2*nmax+5 values) otherwise
// Memory: * Euclid/LSST (tomo): ngal=1e7, nmax=40, nz=10, nr_disc=40, nr_grid=20/20, npix=1e6/3e5, npatch=1000, nthreads=20 
//           --> Gamma_n^(disc,disc) ~ 20*41*5*10*10*10*40*40 ~ 6e9
//           --> Gamma_n^(grid,disc) ~ 20*41*5*10*10*10*(20+20)*40 ~ 6e9
//         * KiDS (tomo): ngal=1e6, nmax=20, nz=5, nr_disc=20, nr_grid=20/20, npix=1e6/3e5, npatch=1000, nthreads=20 
//           --> Gn_patch            ~ 45*5*5*(20*1e6+20*3e5)/1e3 ~ 2.5e7 (5e8 across threads)
//           --> Gamma_n^(disc,disc) ~ 20*21*5*5*5*5*20*20 ~ 1e8
//           --> Gamma_n^(grid,disc) ~ 20*21*5*5*5*5*20*40 ~ 2e8
//         * Euclid/LSST (no tomo): ngal=1e7, nmax=40, nz=1, nr_disc=40, nr_grid=20/20, npix=1e6/3e5, npatch=1000, nthreads=20 
//           --> Gn_patch            ~ 85*1*1*(20*1e6+20*3e5)/1e3 ~ 2e6 (4e6 across threads)
//           --> Gamma_n^(disc,disc) ~ 20*41*5*1*1*1*40*40 ~ 6e6
//           --> Gamma_n^(grid,disc) ~ 20*41*5*1*1*1*(20+20)*40 ~ 6e6
// Note that in both case we only need one cexp evaluation while most operations are cmult ones.
// cumnbinsr_reso at which radii resolution changes. Length of nreso and last element nbins_grid+1.
// Redindices ~ array of length npixbare_reso1 + ... + npixbare_reson in which the nonzero elements
//              contain the index of that pixel in the reduced pixelgrid of the corresponding resolution 

/*
void alloc_Gammans_discretemixedNGP(
    int *isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int *zbins, int nbinsz, int ngal, 
    int nmin, int nmax, double rmin, double rmax, int nbinsr_disc,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    double stripes_start, double stripes_d,  
    int nreso, double *pix1start_reso, double *pix2start_reso, int *pixd_reso, int *nbinsr_reso, 
    int *pix1n_reso, int *pix2n_reso, int *npixbare_reso, int *red_indices,
    double complex *Gns_grid, int nbinsr_grid,
    double *bin_centers_disc, int *counts_disc, double complex *Gammans, double complex *Gammans_norm,
    int n_threads){
    
    // Allocate helper arrays in a way to ensure parallelizability
    // At a later stage those are reduced to the shape of the output
    int nnvals = 0;
    if (nmin>3){nnvals = 2*(nmax-nmin+5);}
    else{nnvals = 2*nmax+5;}
    
    // Various shift parameters for Gns_grid
    int *cumnpixbare_Gngrid = calloc(nreso+1, sizeof(size_t));
    int *cumnbinsr_Gngrid = calloc(nreso+1, sizeof(size_t));
    int *cumresoshift_Gngrid = calloc(nreso+1, sizeof(size_t));
    for (int elreso=0; elreso<nreso; elreso++){
        cumnpixbare_Gngrid[elreso+1] = cumnpixbare_Gngrid[elreso] + npixbare_reso[elreso];
        cumnbinsr_Gngrid[elreso+1] = cumnbinsr_Gngrid[elreso] + nbinsr_reso[elreso];
        cumresoshift_Gngrid[elreso+1] = cumresoshift_Gngrid[elreso] + nbinsr_reso[elreso]*npixbare_reso[elreso];
    }
    size_t resoshift_Gngrid = cumresoshift_Gngrid[nreso];
    size_t nshift_Gngrid = nbinsz*resoshift_Gngrid;
    
    
    // Allocate Gns
    // We do this in parallel as follows:
    // * Find extent along x-axis of inner patch
    // * Split inner patch in 2*nthreads equal area stripes along x-axis
    // * Do two parallelized iterations over galaxies
    //   - In first one only consider galaxies within stripes of even number
    //   - In second one only consider galaxies within stripes of odd number
    // --> We avoid race conditions in calling the spatial hash arrays. - This
    //    is explicitly made sure by (re)setting nthreads in the python layer.
    for (int odd=0; odd<2; odd++){
        double *bin_centers_gcount = calloc(nthreads*nbinsz*nbinsz*nbinsr_disc, sizeof(double));
        double *bin_centers_gnorm = calloc(nthreads*nbinsz*nbinsz*nbinsr_disc, sizeof(double));
        double *bin_centers_norm = calloc(nthreads*nbinsz*nbinsz*nbinsr_disc, sizeof(double));
        int *gcounts = calloc(nthreads*nbinsz*nbinsz*nbinsr_disc, sizeof(int));
        int nnvals_gamma = (nmax-nmin+1);
        double complex *tmpGammans_dd = calloc(nthreads*nnvals_gamma*4*nbinsz*nbinsz*nbinsz*nbinsr_disc*nbinsr_disc, 
                                               sizeof(double complex));
        double complex *tmpGammansnorm_dd = calloc(nthreads*nnvals_gamma*nbinsz*nbinsz*nbinsz*nbinsr_disc*nbinsr_disc, 
                                                   sizeof(double complex));
        double complex *tmpGammans_dg = calloc(nthreads*nnvals_gamma*4*nbinsz*nbinsz*nbinsz*nbinsr_disc*nbinsr_grid, 
                                               sizeof(double complex));
        double complex *tmpGammansnorm_dg = calloc(nthreads*nnvals_gamma*nbinsz*nbinsz*nbinsz*nbinsr_disc*nbinsr_grid, 
                                                   sizeof(double complex));
        double complex *tmpGammans_gd = calloc(nthreads*nnvals_gamma*4*nbinsz*nbinsz*nbinsz*nbinsr_grid*nbinsr_disc, 
                                               sizeof(double complex));
        double complex *tmpGammansnorm_gd = calloc(nthreads*nnvals_gamma*nbinsz*nbinsz*nbinsz*nbinsr_grid*nbinsr_disc, 
                                                   sizeof(double complex));
        #pragma omp parallel for num_threads(nthreads)
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            for (int ind_gal=0; ind_gal<ngal; ind_gal++){
                
                if (isinner[ind_gal]==0){continue;}
                
                // Load info of base galaxy - this needs to be single threaded!
                double p11, p12, e11, e12, w1; 
                int z1;
                #pragma omp critical
                {p11 = pos1[ind_gal];
                 p12 = pos2[ind_gal];
                 e11 = e1[ind_gal];
                 e12 = e2[ind_gal];
                 w1 = weight[ind_gal];
                 z1 = zbins[ind_gal];}
                // Check if galaxy falls in stripe used in this process
                int thisstripe = 2*thisthread + odd;
                int galstripe = (int) floor((p11-stripes_start)/stripes_d);
                if (thisstripe != galstripe){continue;}
                
                // Allocate helper arrays
                double complex *thisGns = calloc(nnvals*nbinsz*nbinsr_disc, sizeof(double complex));
                double complex *thisGnsnorm = calloc((nmax-nmin+1)*nbinsz*nbinsr_disc, sizeof(double complex));
                

                int indshift4 = thisthread*nbinsz*nbinsz*nbinsr_disc;
                int indshift3 = z1*nbinsz*nbinsr_disc;
                int indshift2 = nbinsz*nbinsr_disc;
                
                int ind_pix1, ind_pix2, ind_inpix, ind_gal2;
                int ind_red, ind_gn, lower, upper; 
                double  p21, p22, w2, z2, e21, e22;
                double rel1, rel2, dist, dphi;
                double complex wshape;
                int nextn, nzero, shiftbins;
                double complex nphirot, nphirotc, phirot, phirotc, phirotm, phirotp, phirotn;
                int rbin;
                int thisindshift, index_alloc;

                if (nmin<4){nmin=0;}
                double drbin = (log(rmax)-log(rmin))/(nbinsr_disc);

                int pix1_lower = mymax(0, (int) floor((p11 - (rmax+pix1_d) - pix1_start)/pix1_d));
                int pix2_lower = mymax(0, (int) floor((p12 - (rmax+pix2_d) - pix2_start)/pix2_d));
                int pix1_upper = mymin(pix1_n-1, (int) floor((p11 + (rmax+pix1_d) - pix1_start)/pix1_d));
                int pix2_upper = mymin(pix2_n-1, (int) floor((p12 + (rmax+pix2_d) - pix2_start)/pix2_d));

                for (ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                    for (ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                        ind_red = index_matcher[ind_pix2*pix1_n + ind_pix1];
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
                            rbin = (int) floor((log(dist)-log(rmin))/drbin);
                            wshape = (double complex) w2 * (e21+I*e22);
                            dphi = atan2(rel2,rel1);
                            thisindshift = z2*nbinsr_disc + rbin;

                            // nmin=0
                            //   -> Gns axis: [-nmax-3, ..., nmax+1]
                            //   -> Gn_norm axis: [0,...,nmax]
                            if (nmin==0){
                                nzero = nmax+3;
                                nphirot = 1+I*0;
                                nphirotc = 1+I*0;
                                phirot = cexp(I*dphi);
                                phirotc = conj(phirot);
                                int tmpGindp = nzero*indshift2 + thisindshift;
                                int tmpGindm = nzero*indshift2 + thisindshift;
                                int tmpGindn = thisindshift;
                                int countsshift = indshift4 + indshift3 + thisindshift;
                                // n = 0
                                bin_centers_gcount[countsshift] += w1*w2*dist; 
                                bin_centers_gnorm[countsshift] += w1*w2; 
                                gcounts[countsshift] += 1; 
                                thisGns[tmpGindp] += wshape*nphirot;
                                thisGnsnorm[tmpGindn] += w2*nphirot;
                                nphirot *= phirot;
                                nphirotc *= phirotc; 
                                tmpGindp += indshift2;
                                tmpGindm -= indshift2;
                                tmpGindn += indshift2;
                                // n in [1, ..., nmax] x {+1,-1}
                                for (nextn=1;nextn<nmax+1;nextn++){
                                    thisGns[tmpGindp] += wshape*nphirot;
                                    thisGns[tmpGindm] += wshape*nphirotc;
                                    thisGnsnorm[tmpGindn] += w2*nphirot;  
                                    nphirot *= phirot;
                                    nphirotc *= phirotc; 
                                    tmpGindp += indshift2;
                                    tmpGindm -= indshift2;
                                    tmpGindn += indshift2;
                                }
                                // n in [-nmax-1, nmax+1]
                                thisGns[tmpGindp] += wshape*nphirot;
                                thisGns[tmpGindm] += wshape*nphirotc;
                                nphirotc *= phirotc; 
                                // n in [-nmax-2, -nmax=3]
                                thisGns[indshift2+thisindshift] += wshape*nphirotc;
                                nphirotc *= phirotc; 
                                thisGns[thisindshift] += wshape*nphirotc;
                            }
                            // nmin>3 
                            //   --> Gns axis: [-nmax-3, ..., -nmin+1, nmin-3, ..., nmax+1]
                            //   --> Gn_norm axis: [nmin, ..., nmax]
                            else{
                                phirot = cexp(dphi);
                                phirotc = conj(phirot);
                                phirotp = cpow(phirotc,nmax+3);
                                phirotm = cpow(phirot,nmin-3);
                                phirotn = phirotm*phirot*phirot*phirot;
                                int tmpGindm = thisindshift;
                                int tmpGindp = (nmin+nmax)*indshift2 + thisindshift;
                                // n in [0, ..., nmax-nmin] + {-nmax-3, nmin-3, nmin}
                                for (nextn=0;nextn<nmax-nmin+1;nextn++){
                                    thisGns[tmpGindm] += wshape*phirotm;
                                    thisGns[tmpGindp] += wshape*phirotp;
                                    thisGnsnorm[tmpGindm] += w2*phirotn;
                                    phirotm *= phirot;
                                    phirotp *= phirot;
                                    phirotn *= phirot;
                                    tmpGindm += indshift2;
                                    tmpGindp += indshift2;
                                }
                                // n in [nmax-nmin+1, nmax-nmin+2, nmax-nmin+3] + {-nmax-3, nmin-3}
                                thisGns[tmpGindm] += wshape*phirotm;
                                thisGns[tmpGindp] += wshape*phirotp;
                                phirotm *= phirot;
                                phirotp *= phirot;
                                tmpGindm += indshift2;
                                tmpGindp += indshift2;
                                thisGns[tmpGindm] += wshape*phirotm;
                                thisGns[tmpGindp] += wshape*phirotp;
                                phirotm *= phirot;
                                phirotp *= phirot;
                                tmpGindm += indshift2;
                                tmpGindp += indshift2;
                                thisGns[tmpGindm] += wshape*phirotm;
                                thisGns[tmpGindp] += wshape*phirotp;
                            }               
                        }
                    }
                }
                
                
                // Update blocks in Gamman tensor
                double complex *fourGns_disc = calloc(4*nbinsr_disc*nbinsz, sizeof(double complex));
                double complex *oneGnsnorm_disc = calloc(nbinsr_disc*nbinsz, sizeof(double complex));
                double complex *fourGns_grid = calloc(4*nbinsr_grid*nbinsz, sizeof(double complex));
                double complex *oneGnsnorm_grid = calloc(nbinsr_grid*nbinsz, sizeof(double complex));
                for (int n=nmin; n<=nmax; ind_inpix++){
                    // Need [G_{n-3}, G_{-n-3}, G_{n-1}, G_{-n-1}] for Gamman computation
                    int thisninds[4] = {nmax+n, n-nmin, nmax+n+2, n-nmin+2};
                    if (nmin>3){int thisninds[4] = {nmax-nmin+5+n-nmin, nmax-n, nmax-nmin+5+n-nmin+2, nmax-n+2};}
                    int shiftGns3, shiftGns2, shiftGns23;
                    int shiftthisGnsnorm3, shiftthisGns3, shiftthisGns23, shiftthisGnsnorm23;
                    // Find relevant components of Gn_disc
                    for (int ncomp=0; ncomp<4; ncomp++){
                        shiftGns3 = ncomp*nbinsr_disc*nbinsz;
                        shiftthisGnsnorm3 = (n-nmin)*nbinsr_disc*nbinsz;
                        shiftthisGns3 = thisninds[ncomp]*nbinsr_disc*nbinsz;
                        for (int elbinz=0; elbinz<nbinsz; elbinz++){
                            shiftGns2 = elbinz*nbinsr_disc;
                            shiftGns23 = shiftGns3+shiftGns2;
                            shiftthisGns23 = shiftthisGns3+shiftGns2;
                            shiftthisGnsnorm23 = shiftthisGnsnorm3+shiftGns2;
                            for (int elbinr=0; elbinr<nbinsr_disc; elbinr++){
                                fourGns_disc[shiftGns23+elbinr] = thisGns[shiftthisGns23+elbinr];
                                if (ncomp==0){
                                    oneGnsnorm_disc[shiftGns2+elbinr] = thisGnsnorm[shiftthisGnsnorm23+elbinr];
                                }
                            }
                        } 
                    }
                    // Find relevant components of Gn_grid 
                    // Note that in this function we assume an NGP scheme!
                    size_t resoshift_Gngrid = cumresoshift_Gngrid[nreso];
                    size_t nshift_Gngrid = nbinsz*resoshift_Gngrid;
                    
                    size_t thisnshift = (n-nmin)*nshift_Gngrid;
                    size_t resoshift = 0
                    int galpix_x, galpix_y, gal_redpix, ;
                    for (int elbinr=0; elbinr<nbinsr_grid; elbinr++){
                        if (elbinr >= cumbinsr_Gngrid[thisreso]){
                            thisreso+=1;
                            resoshift += 
                        }
                        galpix_x = (int) floor((p11-pix1start_reso[thisreso])/dpix_reso[thisreso]);
                        galpix_y = (int) floor((p12-pix2start_reso[thisreso])/dpix_reso[thisreso]);
                        gal_redpix = red_indices[cumnpixbare_Gngrid[thisreso]+galpix_y*pix1n_reso[thisreso]+galpix_x];
                         
                        for (int ncomp=0; ncomp<4; ncomp++){
                            shiftGns3 = ncomp*nbinsr_grid*nbinsz;
                            for (int elbinz=0; elbinz<nbinsz; elbinz++){
                                fourGns_grid[] = Gns_grid[shiftGns3+elbinz*nbinsr_grid+elbinr]
                            }
                        }  
                    }
                    
                    for (int ncomp=0; ncomp<4; ncomp++){
                        int shiftGns3 = ncomp*nbinsr_disc*nbinsz;
                        int shiftthisGnsnorm3 = (n-nmin)*nbinsr_disc*nbinsz;
                        int shiftthisGns3 = thisninds[ncomp]*nbinsr_disc*nbinsz;
                        for (int elbinz=0; elbinz<nbinsz; elbinz++){
                            int shiftGns2 = elbinz*nbinsr_disc;
                            int shiftGns23 = shiftGns3+shiftGns2;
                            int shiftthisGns23 = shiftthisGns3+shiftGns2;
                            int shiftthisGnsnorm23 = shiftthisGnsnorm3+shiftGns2;
                                fourGns_disc[shiftGns23+elbinr] = thisGns[shiftthisGns23+elbinr];
                                if (ncomp==0){
                                    oneGnsnorm_disc[shiftGns2+elbinr] = thisGnsnorm[shiftthisGnsnorm23+elbinr];
                                }
                        }
                    } 
                    
                    update_Gamman_discmixed_worker(
                        fourGns_disc, oneGnsnorm_disc, fourGns_grid, oneGnsnorm_grid, 
                        nmin, nmax, n, nbinsr_disc, nbinsr_grid, nbinsz, 
                        nthread, z1,
                        w1, e11, e12, w1_grid, e11_grid, e12_grid,
                        tmpGammans_dd, tmpGammansnorm_dd, 
                        tmpGammans_dg, tmpGammansnorm_dg, 
                        tmpGammans_gd, tmpGammansnorm_gd);
                }
                
                
                free(redindices);
                free(redindweights);
                free(thisGns);
                free(thisGnsnorm);
                free(fourGns_disc);
                free(oneGnsnorm_disc);
                free(fourGns_grid);
                free(oneGnsnorm_grid);
            }
        }
        
        // Accumulate parallely allocated bin_centers & counts
        int countind, gcountind;
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            for (int elbinz1=0; elbinz1<nbinsz; elbinz1++){
                for (int elbinz2=0; elbinz2<nbinsz; elbinz2++){
                    for (int nworker=0; nworker<2*nthreads; nworker++){
                        countind = elbinz1*nbinsz*nbinsr + elbinz2*nbinsr + elbinr;
                        gcountind = nworker*nbinsz*nbinsz*nbinsr + countind;
                        bin_centers[countind] += bin_centers_gcount[gcountind];
                        bin_centers_norm[countind] += bin_centers_gnorm[gcountind];
                        counts[countind] += gcounts[gcountind];
                    }
                    bin_centers[countind] /=  bin_centers_norm[countind];
                }
            }
        }
        
        // Build the Gamman & Gamman_norm multipoles
        #pragma omp parallel for num_threads(mymin(4,nthreads))
        for (int redblock=0; redblock<4; redblock++){
            
            // Accumulate Gamman_dd & Gammannorm_dd
            if (redblock==0){
                1;
            }
            // Accumulate Gamman_dg & Gammannorm_dg
            if (redblock==1){
                1;
            }
            // Accumulate Gamman_gd & Gammannorm_gd
            if (redblock==2){
                1;
            }
            // Accumulate Gamman_gg & Gammannorm_gg
            if (redblock==3){
                1;
            }   
        }
        
                
        
        
        
        free(bin_centers_gcount);
        free(bin_centers_gnorm);
        free(bin_centers_norm);
        free(gcounts);
        free(tmpGammans_dd);
        free(tmpGammansnorm_dd);
        free(tmpGammans_dg);
        free(tmpGammansnorm_dg);
        free(tmpGammans_gd);
        free(tmpGammansnorm_gd);
    }
         
    
    
    // Accumulate parallely allocated Gamman & Gammannorm
    // tmpGammans ~ (2*nthreads, (nmax-nmin+1), 4, nbinsz, nbinsz, nbinsz, nbinsr, nbinsr);
    // tmpGammansnorm ~ (2*nthreads, (nmax-nmin+1), nbinsz, nbinsz, nbinsz, nbinsr, nbinsr);
    // Gammans ~ ((nmax-nmin)+1, 4, nbinsz, nbinsz, nbinsz, nbinsr, nbinsr);
    // Gammans_norm ~ ((nmax-nmin)+1, nbinsz, nbinsz, nbinsz, nbinsr, nbinsr);
    int nzeroshift = (nmax-nmin)*nbinsz*nbinsz*nbinsz*nbinsr*nbinsr;
    int workershift = (nmax-nmin+1)*nbinsz*nbinsz*nbinsz*nbinsr*nbinsr;
    for (int eln=0; eln<=nmax-nmin; eln++){
        int ax0shift = eln*nbinsz*nbinsz*nbinsz*nbinsr*nbinsr;
        for (int elcomp=0; elcomp<4; elcomp++){ 
            int cax0shift = elcomp*nbinsz*nbinsz*nbinsz*nbinsr*nbinsr;
            for (int elbinz1=0; elbinz1<nbinsz; elbinz1++){
                int cax1shift = cax0shift+elbinz1*nbinsz*nbinsz*nbinsr*nbinsr;
                for (int elbinz2=0; elbinz2<nbinsz; elbinz2++){
                    int cax2shift = cax1shift + elbinz2*nbinsz*nbinsr*nbinsr;
                    for (int elbinz3=0; elbinz3<nbinsz; elbinz3++){
                        int cax3shift = cax2shift + elbinz3*nbinsr*nbinsr;
                        for (int elbinr1=0; elbinr1<nbinsr; elbinr1++){
                            int cax4shift = cax2shift + elbinr1*nbinsr;
                            for (int elbinr2=0; elbinr2<nbinsr; elbinr2++){
                                int cax5shift = cax4shift + elbinr2;
                                for (int elworker=0; elworker<2*nthreads; elworker++){
                                    int gammashift = 4*ax0shift+cax5shift;
                                    int tmpgammashift = 4*(elworker*workershift+ax0shift)+cax5shift;
                                    int gammashift_n = ax0shift+cax5shift;
                                    int tmpgammashift_n = elworker*workershift+ax0shift+cax5shift;
                                    Gammans[gammashift] += tmpGammans[tmpgammashift];
                                    if (elcomp==0){
                                        Gammans_norm[gammashift_n] += tmpGammansnorm[tmpgammashift_n];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    free(gcounts);
    free(bin_centers_gcount);
    free(bin_centers_gnorm);
    free(bin_centers_norm);
    free(tmpGammans);
    free(tmpGammansnorm);
    
}
*/

// Calculates Gn s.t. one can allocate Gamman for [nmin, ..., nmax]
// Ranges of n are * n \in [-nmax-3, ..., -nmin+1] u [nmin-3, ..., nmax+3] (i.e. 2*(nmax-nmin+4) values) for nmin>3
//                 * n \in [-nmax-3, ..., nmax+1] (i.e. 2*nmax+5 values) otherwise
// Gns have shape (2*(nmax-nmin+4), ngal, nz, nr) / (2*nmax+4, ngal, nz, nr)
// bin_centers and counts have shape (nz, nz, nr); but are allocated first as (ngal, nz, nr) and afterwards reduced
// Memory: * ngal=1e7, nmax=40, nz=5, nr=50 --> 1e7*(3*40+5)*50*5 ~ 3.1e11 ~ 2.5TB
//         * ngal=1e6, nmax=40, nz=5, nr=10 -->                   ~ 6e9    ~ 50GB
//         * ngal=1e6, nmin=10, nmax=20, nz=5, nr=20              ~ 2.8e9  ~ 24GB
// Note that in both case we only need one cexp evaluation while most operations are cmult ones.
void alloc_Gns_discrete_basic2(
    double *weight, double *pos1, double *pos2, double *e1, double *e2, int *zbins, int nbinsz, int ngal, 
    int nmin, int nmax, double rmin, double rmax, int nbinsr,
    SpatialHash *hash,
    int nthreads, double *bin_centers, int *counts, double complex *Gns, double complex *Gns_norm){
    
    // Need to allocate the bin centers/counts in this way to ensure parallelizability
    // At a later stage are reduced to the shape of the output
    double *bin_centers_gcount = calloc(ngal*nbinsz*nbinsr, sizeof(double));
    double *bin_centers_gnorm = calloc(ngal*nbinsz*nbinsr, sizeof(double));
    int *gcounts = calloc(ngal*nbinsz*nbinsr, sizeof(int));
    
    // Allocate Gns
    // We do this in parallel as follows:
    // * Split survey in 2*nthreads equal area stripes along x-axis
    // * Do two parallelized iterations over galaxies
    //   - In first one only consider galaxies within stripes of even number
    //   - In second one only consider galaxies within stripes of odd number
    // --> We avoid race conditions in calling the spatial hash arrays. - This
    //    is explicitly made sure by (re)setting nthreads in the python layer.
    for (int odd=0; odd<2; odd++){
        //#pragma omp parallel for private(nbinsz, ngal, nmin, nmax, rmin, rmax, nbinsr, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, nthreads)
        #pragma omp parallel for num_threads(nthreads)
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            // Index shifts for Gn
            int indshift3 = ngal*nbinsz*nbinsr;
            int indshift2 = nbinsz*nbinsr;
            for (int ind_gal=0; ind_gal<ngal; ind_gal++){
                

                // Check if galaxy falls in stripe used in this process
                double p11, p12;
                #pragma omp critical
                {p11 = pos1[ind_gal];
                p12 = pos2[ind_gal];}
                int thisstripe = 2*thisthread + odd;
                int galstripe = (int) floor((p11-hash->pixstarts[0])/hash->pixsize[0] * (2*nthreads)/hash->gridsize[0]);
                if (thisstripe != galstripe){continue;}
                
                int ind_pix1, ind_pix2, ind_inpix, ind_gal2;
                int ind_red, lower, upper; 
                double  p21, p22, w2, z2, e21, e22;
                double rel1, rel2, dist, dphi;
                double complex wshape;
                int nextn, nzero;
                double complex nphirot, nphirotc, phirot, phirotc, phirotm, phirotp, phirotn;

                int rbin;
                int thisindshift;

                if (nmin<4){nmin=0;}
                double drbin = (log(rmax)-log(rmin))/(nbinsr);             
                            
                int pix1_lower = mymax(0, (int) floor((p11 - (rmax+hash->pixsize[0]) - hash->pixstarts[0])/hash->pixsize[0]));
                int pix2_lower = mymax(0, (int) floor((p12 - (rmax+hash->pixsize[1]) - hash->pixstarts[1])/hash->pixsize[1]));
                int pix1_upper = mymin(hash->gridsize[0]-1, 
                                       (int) floor((p11 + (rmax+hash->pixsize[0]) - 
                                                    hash->pixstarts[0])/hash->pixsize[0]));
                int pix2_upper = mymin(hash->gridsize[1]-1, 
                                       (int) floor((p12 + (rmax+hash->pixsize[1]) - 
                                                    hash->pixstarts[1])/hash->pixsize[1]));
                if (ind_gal%10000==0){printf("%d %d %d %d\n",pix1_lower,pix1_upper,pix2_lower,pix2_upper);}
                for (ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                    for (ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                        ind_red = hash->index_matcher[ind_pix2*hash->gridsize[0] + ind_pix1];
                        if (ind_red==-1){continue;}
                        lower = hash->pixs_galind_bounds[ind_red];
                        upper = hash->pixs_galind_bounds[ind_red+1];
                        for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                            ind_gal2 = hash->pix_gals[ind_inpix];
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
                            rbin = (int) floor((log(dist)-log(rmin))/drbin);
                            wshape = (double complex) w2 * (e21+I*e22);
                            dphi = atan2(rel2,rel1);
                            thisindshift = ind_gal*indshift2+z2*nbinsr + rbin;

                            // nmin=0 -
                            //   -> Gns axis: [-nmax-3, ..., nmax+1]
                            //   -> Gn_norm axis: [0,...,nmax]
                            if (nmin==0){
                                nzero = nmax+3;
                                nphirot = 1+I*0;
                                nphirotc = 1+I*0;
                                phirot = cexp(I*dphi);
                                phirotc = conj(phirot);
                                int tmpGindp = nzero*indshift3 + thisindshift;
                                int tmpGindm = nzero*indshift3 + thisindshift;
                                int tmpGindn = thisindshift;
                                // n = 0
                                //bin_centers_gcount[thisindshift] += w1*w2*dist; 
                                //bin_centers_gnorm[thisindshift] += w1*w2; 
                                //gcounts[thisindshift] += 1; 
                                Gns[tmpGindp] += wshape*nphirot;
                                Gns_norm[tmpGindn] += w2*nphirot;  
                                nphirot *= phirot;
                                nphirotc *= phirotc; 
                                tmpGindp += indshift3;
                                tmpGindm -= indshift3;
                                tmpGindn += indshift3;
                                // n in [1, ..., nmax] x {+1,-1}
                                for (nextn=1;nextn<nmax+1;nextn++){
                                    Gns[tmpGindp] += wshape*nphirot;
                                    Gns[tmpGindm] += wshape*nphirotc;
                                    Gns_norm[tmpGindn] += w2*nphirot;  
                                    nphirot *= phirot;
                                    nphirotc *= phirotc; 
                                    tmpGindp += indshift3;
                                    tmpGindm -= indshift3;
                                    tmpGindn += indshift3;
                                }
                                // n in [-nmax-1, nmax+1]
                                Gns[tmpGindp] += wshape*nphirot;
                                Gns[tmpGindm] += wshape*nphirotc;
                                nphirotc *= phirotc; 
                                // n in [-nmax-2, -nmax=3]
                                Gns[indshift3+thisindshift] += wshape*nphirotc;
                                nphirotc *= phirotc; 
                                Gns[thisindshift] += wshape*nphirotc;
                            }
                            // nmin>3 
                            //   --> Gns axis: [-nmax-3, ..., -nmin+1, nmin-3, ..., nmax+1]
                            //   --> Gn_norm axis: [nmin, ..., nmax]
                            else{
                                phirot = cexp(dphi);
                                phirotc = conj(phirot);
                                phirotp = cpow(phirotc,nmax+3);
                                phirotm = cpow(phirot,nmin-3);
                                phirotn = phirotm*phirot*phirot*phirot;
                                int tmpGindm = thisindshift;
                                int tmpGindp = (nmin+nmax)*indshift3 + thisindshift;
                                // n in [0, ..., nmax-nmin] + {-nmax-3, nmin-3, nmin}
                                for (nextn=0;nextn<nmax-nmin+1;nextn++){
                                    Gns[tmpGindm] += wshape*phirotm;
                                    Gns[tmpGindp] += wshape*phirotp;
                                    Gns_norm[tmpGindm] += w2*phirotn;
                                    phirotm *= phirot;
                                    phirotp *= phirot;
                                    phirotn *= phirot;
                                    tmpGindm += indshift3;
                                    tmpGindp += indshift3;
                                }
                                // n in [nmax-nmin+1, nmax-nmin+2, nmax-nmin+3] + {-nmax-3, nmin-3}
                                Gns[tmpGindm] += wshape*phirotm;
                                Gns[tmpGindp] += wshape*phirotp;
                                phirotm *= phirot;
                                phirotp *= phirot;
                                tmpGindm += indshift3;
                                tmpGindp += indshift3;
                                Gns[tmpGindm] += wshape*phirotm;
                                Gns[tmpGindp] += wshape*phirotp;
                                phirotm *= phirot;
                                phirotp *= phirot;
                                tmpGindm += indshift3;
                                tmpGindp += indshift3;
                                Gns[tmpGindm] += wshape*phirotm;
                                Gns[tmpGindp] += wshape*phirotp;
                            }               
                        }
                    }
                }
            }
        }
    }
    
    /*
    // Finish the calculation of bin centers (cannot be parallelized)
    double *bin_centers_norm = calloc(nbinsz*nbinsz*nbinsr, sizeof(double));
    int countbin, gbinind;
    for (int ind_gal=0; ind_gal<ngal; ind_gal++){
        int zbin1 = zbins[ind_gal];
        for (int elb1=0; elb1<nbinsr; elb1++){
            for (int zbin2=0; zbin2<nbinsz; zbin2++){
                countbin = zbin1*nbinsz*nbinsr + zbin2*nbinsr + elb1;
                gbinind = ind_gal*indshift2+zbin2*nbinsr + elb1;
                bin_centers[countbin] += bin_centers_gcount[gbinind];
                bin_centers_norm[countbin] += bin_centers_gnorm[gbinind];
                counts[countbin] += gcounts[gbinind];
            }
        }
    }
    for (int elb1=0; elb1<nbinsr; elb1++){
        for (int zbin1=0; zbin1<nbinsz; zbin1++){
            for (int zbin2=0; zbin2<nbinsz; zbin2++){
                countbin = zbin1*nbinsz*nbinsr + zbin2*nbinsr + elb1;
                bin_centers[countbin] /=  bin_centers_norm[countbin];
            }
        }
    }
    free(bin_centers_norm);
    */
    
    free(gcounts);
    free(bin_centers_gcount);
    free(bin_centers_gnorm);
}


// Calculates Gn s.t. one can allocate Gamman for [nmin, ..., nmax]
// Ranges of n are * n \in [-nmax-3, ..., -nmin+1] u [nmin-3, ..., nmax+3] (i.e. nnvals=2*(nmax-nmin+4) values) for nmin>3
//                 * n \in [-nmax-3, ..., nmax+1] (i.e. nnvals=2*nmax+5 values) otherwise
// Gns have shape (nnvals, nz, nz, nr, npix_inner)
// Gns_norm have shape ((nmax-nmin)+1, nz, nz, nr, npix_inner)
// Gammans have shape (2*(nmax-nmin)+1, 4, nz, nz, nz, nr, nr)
// bin_centers and counts have shape (nz, nz, nr); but are allocated first as (ngal, nz, nr) and afterwards reduced
// Memory: * ngal=1e7, nmax=40, nz=5, nr=20, npix_red=1e6 --> Gn/Gnnorms: (85+41)*25*20*1e6 ~ 6.3e10 ~ 1.08 TB
//           nthreads=28                                  --> tmpGamman: 56*41*4*125*400 ~ 3.3e8 ~ 5.2 GB
//         * ngal=1e6, nmax=20, nz=5, nr=20, npix_red=4e5 --> Gn/Gnnorms: (45+21)*25*20*4e5 ~ 6.3e10 ~ 211GB
//           nthreads=28                                  --> tmpGamman: 56*21*4*125*400 ~ 1.7e8 ~ 2.7 GB
// Note that in both case we only need one cexp evaluation while most operations are cmult ones.
// Outputs have the following shape
// 
void alloc_GnsGammans_discrete_basic(
    double *weight, int *inner_region, double *pos1, double *pos2, double *e1, double *e2, int *zbins, int nbinsz, int ngal,
    int nmin, int nmax, double rmin, double rmax, int nbinsr,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    double inner_xmin, double inner_xmax,
    int *gridmatcher_inds, double *gridmatcher_weights, int *reducedgridmatcher, int mas_scheme, int npix_inner,
    int nthreads, double *bin_centers, int *counts, 
    double complex *Gns, double complex *Gns_norm, double complex *Gammans, double complex *Gammans_norm){
    
    // Allocate helper arrays in a way to ensure parallelizability
    // At a later stage those are reduced to the shape of the output
    int nnvals = 0;
    if (nmin>3){nnvals = 2*(nmax-nmin+5);}
    else{nnvals = 2*nmax+5;}
    double *bin_centers_gcount = calloc(2*nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
    double *bin_centers_gnorm = calloc(2*nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
    double *bin_centers_norm = calloc(nbinsz*nbinsz*nbinsr, sizeof(double));
    int *gcounts = calloc(2*nthreads*nbinsz*nbinsz*nbinsr, sizeof(int));
    double complex *tmpGammans = calloc(2*nthreads*(nmax-nmin+1)*4*nbinsz*nbinsz*nbinsz*nbinsr*nbinsr, sizeof(double complex));
    double complex *tmpGammansnorm = calloc(2*nthreads*(nmax-nmin+1)*nbinsz*nbinsz*nbinsz*nbinsr*nbinsr, sizeof(double complex));
    
    // Allocate Gns
    // We do this in parallel as follows:
    // * Find extent along x-axis of inner patch
    // * Split inner patch in 2*nthreads equal area stripes along x-axis
    // * Do two parallelized iterations over galaxies
    //   - In first one only consider galaxies within stripes of even number
    //   - In second one only consider galaxies within stripes of odd number
    // --> We avoid race conditions in calling the spatial hash arrays. - This
    //    is explicitly made sure by (re)setting nthreads in the python layer.
    double inner_lenx = inner_xmax-inner_xmin; 
    for (int odd=0; odd<2; odd++){
        #pragma omp parallel for num_threads(nthreads)
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            int ind_inner = 0;
            for (int ind_gal=0; ind_gal<ngal; ind_gal++){
                
                if (inner_region[ind_gal]==0){continue;}
                
                // Load info of base galaxy - this needs to be single threaded!
                ind_inner += 1;
                double p11, p12, e11, e12, w1; 
                int z1;
                #pragma omp critical
                {p11 = pos1[ind_gal];
                 p12 = pos2[ind_gal];
                 e11 = e1[ind_gal];
                 e12 = e2[ind_gal];
                 w1 = weight[ind_gal];
                 z1 = zbins[ind_gal];}
                // Check if galaxy falls in stripe used in this process
                int thisstripe = 2*thisthread + odd;
                int galstripe = (int) floor((p11-inner_xmin)/inner_lenx * 2*nthreads);
                if (thisstripe != galstripe){continue;}
                
                // Allocate helper arrays
                int nredindices = (2*mas_scheme+1)*(2*mas_scheme+1);
                int *redindices = calloc(nredindices, sizeof(int));
                double *redindweights = calloc(nredindices, sizeof(double));
                double complex *thisGns = calloc(nnvals*nbinsz*nbinsr, sizeof(double complex));
                double complex *thisGnsnorm = calloc((nmax-nmin+1)*nbinsz*nbinsr, sizeof(double complex));
                
                // Indices/weights for Gnallocation
                for (int redindex=0; redindex<nredindices; redindex++){
                    redindices[redindex] = reducedgridmatcher[gridmatcher_inds[ind_inner*nredindices+redindex]];
                    redindweights[redindex] = gridmatcher_weights[ind_inner*nredindices+redindex];
                }
                
                // Index shifts for Gn
                int thisworker = odd*nthreads+thisthread;
                int indshift4 = thisworker*nbinsz*nbinsz*nbinsr;
                int indshift3 = z1*nbinsz*nbinsr;
                int indshift2 = nbinsr*nbinsz;
                
                int ind_pix1, ind_pix2, ind_inpix, ind_gal2;
                int ind_red, lower, upper; 
                double  p21, p22, w2, z2, e21, e22;
                double rel1, rel2, dist, dphi;
                double complex wshape;
                int nextn, nzero;
                double complex nphirot, nphirotc, phirot, phirotc, phirotm, phirotp, phirotn;
                int rbin;
                int thisindshift;

                if (nmin<4){nmin=0;}
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
                            rbin = (int) floor((log(dist)-log(rmin))/drbin);
                            wshape = (double complex) w2 * (e21+I*e22);
                            dphi = atan2(rel2,rel1);
                            thisindshift = z2*nbinsr + rbin;

                            // nmin=0
                            //   -> Gns axis: [-nmax-3, ..., nmax+1]
                            //   -> Gn_norm axis: [0,...,nmax]
                            if (nmin==0){
                                nzero = nmax+3;
                                nphirot = 1+I*0;
                                nphirotc = 1+I*0;
                                phirot = cexp(I*dphi);
                                phirotc = conj(phirot);
                                int tmpGindp = nzero*indshift2 + thisindshift;
                                int tmpGindm = nzero*indshift2 + thisindshift;
                                int tmpGindn = thisindshift;
                                int countsshift = indshift4 + indshift3 + thisindshift;
                                // n = 0
                                bin_centers_gcount[countsshift] += w1*w2*dist; 
                                bin_centers_gnorm[countsshift] += w1*w2; 
                                gcounts[countsshift] += 1; 
                                thisGns[tmpGindp] += wshape*nphirot;
                                thisGnsnorm[tmpGindn] += w2*nphirot;
                                nphirot *= phirot;
                                nphirotc *= phirotc; 
                                tmpGindp += indshift2;
                                tmpGindm -= indshift2;
                                tmpGindn += indshift2;
                                // n in [1, ..., nmax] x {+1,-1}
                                for (nextn=1;nextn<nmax+1;nextn++){
                                    thisGns[tmpGindp] += wshape*nphirot;
                                    thisGns[tmpGindm] += wshape*nphirotc;
                                    thisGnsnorm[tmpGindn] += w2*nphirot;  
                                    nphirot *= phirot;
                                    nphirotc *= phirotc; 
                                    tmpGindp += indshift2;
                                    tmpGindm -= indshift2;
                                    tmpGindn += indshift2;
                                }
                                // n in [-nmax-1, nmax+1]
                                thisGns[tmpGindp] += wshape*nphirot;
                                thisGns[tmpGindm] += wshape*nphirotc;
                                nphirotc *= phirotc; 
                                // n in [-nmax-2, -nmax=3]
                                thisGns[indshift2+thisindshift] += wshape*nphirotc;
                                nphirotc *= phirotc; 
                                thisGns[thisindshift] += wshape*nphirotc;
                            }
                            // nmin>3 
                            //   --> Gns axis: [-nmax-3, ..., -nmin+1, nmin-3, ..., nmax+1]
                            //   --> Gn_norm axis: [nmin, ..., nmax]
                            else{
                                phirot = cexp(dphi);
                                phirotc = conj(phirot);
                                phirotp = cpow(phirotc,nmax+3);
                                phirotm = cpow(phirot,nmin-3);
                                phirotn = phirotm*phirot*phirot*phirot;
                                int tmpGindm = thisindshift;
                                int tmpGindp = (nmin+nmax)*indshift2 + thisindshift;
                                // n in [0, ..., nmax-nmin] + {-nmax-3, nmin-3, nmin}
                                for (nextn=0;nextn<nmax-nmin+1;nextn++){
                                    thisGns[tmpGindm] += wshape*phirotm;
                                    thisGns[tmpGindp] += wshape*phirotp;
                                    thisGnsnorm[tmpGindm] += w2*phirotn;
                                    phirotm *= phirot;
                                    phirotp *= phirot;
                                    phirotn *= phirot;
                                    tmpGindm += indshift2;
                                    tmpGindp += indshift2;
                                }
                                // n in [nmax-nmin+1, nmax-nmin+2, nmax-nmin+3] + {-nmax-3, nmin-3}
                                thisGns[tmpGindm] += wshape*phirotm;
                                thisGns[tmpGindp] += wshape*phirotp;
                                phirotm *= phirot;
                                phirotp *= phirot;
                                tmpGindm += indshift2;
                                tmpGindp += indshift2;
                                thisGns[tmpGindm] += wshape*phirotm;
                                thisGns[tmpGindp] += wshape*phirotp;
                                phirotm *= phirot;
                                phirotp *= phirot;
                                tmpGindm += indshift2;
                                tmpGindp += indshift2;
                                thisGns[tmpGindm] += wshape*phirotm;
                                thisGns[tmpGindp] += wshape*phirotp;
                            }               
                        }
                    }
                }
                
                // Update Gns
                // length: nnvals*nbinsr*nbinsz^2*npix_reducedgrid
                double complex thisGnval, thisGnnormval, masweight;
                for(int nindex=0; nindex<nnvals; nindex++){
                    int shiftGnax0 = nindex*nbinsr*nbinsz*nbinsz*npix_inner;
                    int shiftthisGnax0 = nindex*nbinsz*nbinsr;
                    for (int elbinz=0; elbinz<nbinsz; elbinz++){
                        int shiftGnax1 = shiftGnax0 + z1*nbinsz*nbinsr*npix_inner;
                        int shiftGnax2 = shiftGnax1 + elbinz*nbinsr*npix_inner;
                        int shiftthisGnax1 = shiftthisGnax0 + elbinz*nbinsr;
                        for (int elbinr=0; elbinr<nbinsr; elbinr++){
                            int shiftGnax3 = shiftGnax2 + elbinr*npix_inner;
                            thisGnval = thisGns[shiftthisGnax1+elbinr];
                            if (nindex<=nmax-nmin){
                                thisGnnormval = thisGnsnorm[shiftthisGnax1+elbinr];
                            }
                            for (int redindex=0; redindex<nredindices; redindex++){
                                masweight = (double complex) redindweights[redindex];
                                Gns[shiftGnax3+redindices[redindex]] += masweight*thisGnval;
                                if (nindex<=nmax-nmin){
                                    Gns_norm[shiftGnax3+redindices[redindex]] += masweight*thisGnnormval;
                                }
                            }
                        }
                    }    
                }
                
                // Update Gammans
                double complex *fourGns = calloc(4*nbinsr*nbinsz, sizeof(double complex));
                double complex *oneGnsnorm = calloc(nbinsr*nbinsz, sizeof(double complex));
                for (int n=nmin; n<=nmax; ind_inpix++){
                    // Need [G_{n-3}, G_{-n-3}, G_{n-1}, G_{-n-1}] for Gamman computation
                    // nmin==0: n \in [-nmax-3, ..., nmax+1]
                    // nmin>3 : n \in [-nmax-3, ..., -nmin+1, nmin-3, ..., nmax+1]
                    int thisninds[4] = {nmax+n, n-nmin, nmax+n+2, n-nmin+2};
                    if (nmin>3){thisninds[0] = nmax-nmin+5+n-nmin;
                                thisninds[1] = nmax-n;
                                thisninds[2] = nmax-nmin+5+n-nmin+2;
                                thisninds[3] = nmax-n+2;}
                    for (int ncomp=0; ncomp<4; ncomp++){
                        int shiftGns3 = ncomp*nbinsr*nbinsz;
                        int shiftthisGnsnorm3 = (n-nmin)*nbinsr*nbinsz;
                        int shiftthisGns3 = thisninds[ncomp]*nbinsr*nbinsz;
                        for (int elbinz=0; elbinz<nbinsz; elbinz++){
                            int shiftGns2 = elbinz*nbinsr;
                            int shiftGns23 = shiftGns3+shiftGns2;
                            int shiftthisGns23 = shiftthisGns3+shiftGns2;
                            int shiftthisGnsnorm23 = shiftthisGnsnorm3+shiftGns2;
                            for (int elbinr=0; elbinr<nbinsr; elbinr++){
                                fourGns[shiftGns23+elbinr] = thisGns[shiftthisGns23+elbinr];
                                if (ncomp==0){
                                    oneGnsnorm[shiftGns2+elbinr] = thisGnsnorm[shiftthisGnsnorm23+elbinr];
                                }
                            }
                        } 
                    }
                    update_Gamman_discrete_worker(
                        fourGns, oneGnsnorm, nmin, nmax, n, nbinsr, nbinsz, thisworker, 
                        w1, e11, e12, z1, tmpGammans, tmpGammansnorm);
                }
                free(redindices);
                free(redindweights);
                free(thisGns);
                free(thisGnsnorm);
                free(fourGns);
                free(oneGnsnorm);
            }
        }
    }
         
    // Accumulate parallely allocated bin_centers & counts
    int countind, gcountind;
    for (int elbinr=0; elbinr<nbinsr; elbinr++){
        for (int elbinz1=0; elbinz1<nbinsz; elbinz1++){
            for (int elbinz2=0; elbinz2<nbinsz; elbinz2++){
                for (int nworker=0; nworker<2*nthreads; nworker++){
                    countind = elbinz1*nbinsz*nbinsr + elbinz2*nbinsr + elbinr;
                    gcountind = nworker*nbinsz*nbinsz*nbinsr + countind;
                    bin_centers[countind] += bin_centers_gcount[gcountind];
                    bin_centers_norm[countind] += bin_centers_gnorm[gcountind];
                    counts[countind] += gcounts[gcountind];
                }
                bin_centers[countind] /=  bin_centers_norm[countind];
            }
        }
    }
    
    // Accumulate parallely allocated Gamman & Gammannorm
    // tmpGammans ~ (2*nthreads, (nmax-nmin+1), 4, nbinsz, nbinsz, nbinsz, nbinsr, nbinsr);
    // tmpGammansnorm ~ (2*nthreads, (nmax-nmin+1), nbinsz, nbinsz, nbinsz, nbinsr, nbinsr);
    // Gammans ~ ((nmax-nmin)+1, 4, nbinsz, nbinsz, nbinsz, nbinsr, nbinsr);
    // Gammans_norm ~ ((nmax-nmin)+1, nbinsz, nbinsz, nbinsz, nbinsr, nbinsr);
    int workershift = (nmax-nmin+1)*nbinsz*nbinsz*nbinsz*nbinsr*nbinsr;
    for (int eln=0; eln<=nmax-nmin; eln++){
        int ax0shift = eln*nbinsz*nbinsz*nbinsz*nbinsr*nbinsr;
        for (int elcomp=0; elcomp<4; elcomp++){ 
            int cax0shift = elcomp*nbinsz*nbinsz*nbinsz*nbinsr*nbinsr;
            for (int elbinz1=0; elbinz1<nbinsz; elbinz1++){
                int cax1shift = cax0shift+elbinz1*nbinsz*nbinsz*nbinsr*nbinsr;
                for (int elbinz2=0; elbinz2<nbinsz; elbinz2++){
                    int cax2shift = cax1shift + elbinz2*nbinsz*nbinsr*nbinsr;
                    for (int elbinz3=0; elbinz3<nbinsz; elbinz3++){
                        for (int elbinr1=0; elbinr1<nbinsr; elbinr1++){
                            int cax4shift = cax2shift + elbinr1*nbinsr;
                            for (int elbinr2=0; elbinr2<nbinsr; elbinr2++){
                                int cax5shift = cax4shift + elbinr2;
                                for (int elworker=0; elworker<2*nthreads; elworker++){
                                    int gammashift = 4*ax0shift+cax5shift;
                                    int tmpgammashift = 4*(elworker*workershift+ax0shift)+cax5shift;
                                    int gammashift_n = ax0shift+cax5shift;
                                    int tmpgammashift_n = elworker*workershift+ax0shift+cax5shift;
                                    Gammans[gammashift] += tmpGammans[tmpgammashift];
                                    if (elcomp==0){
                                        Gammans_norm[gammashift_n] += tmpGammansnorm[tmpgammashift_n];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    free(gcounts);
    free(bin_centers_gcount);
    free(bin_centers_gnorm);
    free(bin_centers_norm);
    free(tmpGammans);
    free(tmpGammansnorm);
    
}


void test_rsearch(double *pos1, double *pos2, int ngals, double center1, double center2, double *radii, int nradii,
                  int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
                  double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
                  int *indices_radii, int *rshells, int ninradii){
    
    int ind_gal, ind_pix1, ind_pix2, ind_inpix;
    int ind_red, lower, upper;
    int rbin;
    double rel1, rel2, dist;
    
    double rmin = radii[0];
    double rmax = radii[nradii-1];
    double drbin = (log(radii[nradii-1])-log(radii[0]))/(nradii-1);
    ninradii = 0;
        
    int pix1_lower = mymax(0, (int) floor((center1 - (rmax+pix1_d) - pix1_start)/pix1_d));
    int pix2_lower = mymax(0, (int) floor((center2 - (rmax+pix2_d) - pix2_start)/pix2_d));
    int pix1_upper = mymin(pix1_n-1, (int) floor((center1 + (rmax+pix1_d) - pix1_start)/pix1_d));
    int pix2_upper = mymin(pix2_n-1, (int) floor((center2 + (rmax+pix2_d) - pix2_start)/pix2_d));

    for (ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
        for (ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
            ind_red = index_matcher[ind_pix2*pix1_n + ind_pix1];
            if (ind_red==-1){continue;}
            lower = pixs_galind_bounds[ind_red];
            upper = pixs_galind_bounds[ind_red+1];
            for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                ind_gal = pix_gals[ind_inpix];
                rel1 = pos1[ind_gal] - center1;
                rel2 = pos2[ind_gal] - center2;
                dist = sqrt(rel1*rel1 + rel2*rel2);
                if(dist < rmin || dist >= rmax){continue;}
                rbin = (int) floor((log(dist)-log(rmin))/drbin);
                indices_radii[ninradii] = ind_gal;
                rshells[ninradii] = rbin;
                ninradii += 1;
            }
        }
    }  
    
}


void test_inshell(double *pos1, double *pos2, int ngals, double rmin, double rmax,
                  int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
                  double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
                  int *countshell){
    
    int ind_gal, ind_gal2, ind_pix1, ind_pix2, ind_inpix;
    int ind_red, lower, upper;
    double rel1, rel2, dist;
        
    for (ind_gal=0; ind_gal<ngals; ind_gal++){
        
        double center1 = pos1[ind_gal];
        double center2 = pos2[ind_gal];
        
        int pix1_lower = mymax(0, (int) floor((center1 - (rmax+pix1_d) - pix1_start)/pix1_d));
        int pix2_lower = mymax(0, (int) floor((center2 - (rmax+pix2_d) - pix2_start)/pix2_d));
        int pix1_upper = mymin(pix1_n-1, (int) floor((center1 + (rmax+pix1_d) - pix1_start)/pix1_d));
        int pix2_upper = mymin(pix2_n-1, (int) floor((center2 + (rmax+pix2_d) - pix2_start)/pix2_d));

        for (ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
            for (ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                ind_red = index_matcher[ind_pix2*pix1_n + ind_pix1];
                if (ind_red==-1){continue;}
                lower = pixs_galind_bounds[ind_red];
                upper = pixs_galind_bounds[ind_red+1];
                for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                    ind_gal2 = pix_gals[ind_inpix];
                    rel1 = pos1[ind_gal2] - center1;
                    rel2 = pos2[ind_gal2] - center2;
                    dist = sqrt(rel1*rel1 + rel2*rel2);
                    if(dist > rmin && dist <= rmax){
                        countshell[ind_gal] += 1;}
                }
            }
        }
    }  
    
}


// Shape (ngal, nradii, 2)       
void test_G01_discrete(
    double *weight, double *pos1, double *pos2, double *e1, double *e2, int ngal, 
    double *bin_edges, int nbinedges,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, complex *G01s, double complex *G01s_norm){
        
    // Allocate Gns
    // We do this in parallel as follows:
    // * Split survey in 2*nthreads equal area stripes along x-axis
    // * Do two parallelized iterations over galaxies
    //   - In first one only consider galaxies within stripes of even number
    //   - In second one only consider galaxies within stripes of odd number
    // --> We avoid race conditions in calling the spatial hash arrays. - This
    //    is explicitly made sure by (re)setting nthreads in the python layer.
    //for (int odd=0; odd<2; odd++){
    //    #pragma omp parallel for num_threads(nthreads)
    //    for (int thisthread=0; thisthread<nthreads; thisthread++){
            // Index shifts for Gn
            for (int ind_gal=0; ind_gal<ngal; ind_gal++){

                // Check if galaxy falls in stripe used in this process
      //          double p11;
      //          #pragma omp critical
      //          {p11 = pos1[ind_gal];}
      //          int thisstripe = 2*thisthread + odd;
      //          int galstripe = (int) floor((p11-pix1_start)/pix1_d * (2*nthreads)/pix1_n);
      //          if (thisstripe != galstripe){continue;}
                
                int ind_pix1, ind_pix2, ind_inpix, ind_gal2;
                int ind_red, lower, upper, rbin; 
                double rel1, rel2, dist, dphi;
                double complex wshape;
                double complex phirot;
                
                int nbinsr = nbinedges-1;
                double rmax = bin_edges[nbinsr];
                double rmin = bin_edges[0];
                int pix1_lower = mymax(0, (int) floor((pos1[ind_gal] - (rmax+pix1_d) - pix1_start)/pix1_d));
                int pix2_lower = mymax(0, (int) floor((pos2[ind_gal] - (rmax+pix2_d) - pix2_start)/pix2_d));
                int pix1_upper = mymin(pix1_n-1, (int) floor((pos1[ind_gal] + (rmax+pix1_d) - pix1_start)/pix1_d));
                int pix2_upper = mymin(pix2_n-1, (int) floor((pos2[ind_gal] + (rmax+pix2_d) - pix2_start)/pix2_d));
                for (ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                    for (ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                        ind_red = index_matcher[ind_pix2*pix1_n + ind_pix1];
                        if (ind_red==-1){continue;}
                        lower = pixs_galind_bounds[ind_red];
                        upper = pixs_galind_bounds[ind_red+1];
                        for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                            ind_gal2 = pix_gals[ind_inpix];
                            rel1 = pos1[ind_gal2] - pos1[ind_gal];
                            rel2 = pos2[ind_gal2] - pos2[ind_gal];
                            dist = sqrt(rel1*rel1 + rel2*rel2);
                            if(dist < rmin || dist >= rmax){continue;}
                            for (rbin=0; rbin<nbinedges-1; rbin++){
                                if (dist<=bin_edges[rbin+1]){break;}
                            }
                            wshape = weight[ind_gal2]*(e1[ind_gal2]+I*e2[ind_gal2]);
                            dphi = atan2(rel2,rel1);
                            phirot = cexp(I*dphi);
                            G01s[ind_gal*nbinsr*2 + rbin*2] += wshape;
                            G01s[ind_gal*nbinsr*2 + rbin*2+1] += wshape*phirot;
                            G01s_norm[ind_gal*nbinsr*2 + 2*rbin] += weight[ind_gal2];
                            G01s_norm[ind_gal*nbinsr*2 + 2*rbin+1] += weight[ind_gal2]*phirot;
                        }
                    }               
                }
            }
    //    }
    //}
}


// Gns have shape (4, ngal, nz, nr) (40*1e7*5*
// Gn_norm has shape (ngal, nz, nr)
// bin_centers and counts have shape (nz, nz, nr)
// No atomics involved here --> Make sure that no race conditions occure (as done in python layer)
void alloc_Gn_discrete_basic(
    double *weight, double *pos1, double *pos2, double *e1, double *e2, int *zbins, int nbinsz, int ngal, 
    int n, double rmin, double rmax, int nbinsr,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, double *bin_centers, int *counts, double complex *Gns, double complex *Gn_norm){
    
    // Index shifts for Gn
    int indshift3 = ngal*nbinsz*nbinsr;
    int indshift2 = nbinsz*nbinsr;
    
    double *bin_centers_norm = calloc(nbinsz*nbinsz*nbinsr, sizeof(double));
    
    // Allocate Gns
    //#pragma omp parallel for num_threads(nthreads)
    //#pragma omp parallel for schedule(static,chunksize)
    for (int ind_gal=0; ind_gal<ngal; ind_gal++){
        int ind_pix1, ind_pix2, ind_inpix, ind_gal2;
        int ind_red, lower, upper; 
        double p11, p12, p21, p22, w1, w2, z1, z2, e21, e22;
        double rel1, rel2, dist, dphi;
        double complex wshape;
        double complex nphirot, phirot, threephirot;
            
        int rbin, countbin;
        int thisindshift;
        
        double drbin = (log(rmax)-log(rmin))/(nbinsr);
        p11 = pos1[ind_gal];
        p12 = pos2[ind_gal];
        w1 = weight[ind_gal];
        z1 = zbins[ind_gal];
        
        int pix1_lower = mymax(0, (int) floor((p11 - (rmax+pix1_d) - pix1_start)/pix1_d));
        int pix1_upper = mymin(pix1_n-1, (int) floor((p11 + (rmax+pix1_d) - pix1_start)/pix1_d));
        int pix2_lower = mymax(0, (int) floor((p12 - (rmax+pix2_d) - pix2_start)/pix2_d));
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
                    rbin = (int) floor((log(dist)-log(rmin))/drbin);
                    wshape = w2*(e21+I*e22);
                    dphi = atan2(rel2,rel1);
                    thisindshift = ind_gal*indshift2+z2*nbinsr + rbin;
                    phirot = cexp(I*dphi);
                    nphirot = cpow(phirot, n);
                    threephirot = phirot*phirot*phirot;
                    // Allocate in order [n-3, -n-3, n-1, -n-1]
                    Gns[0*indshift3 + thisindshift] += wshape*nphirot*conj(threephirot);
                    Gns[1*indshift3 + thisindshift] += wshape*conj(nphirot*threephirot);
                    Gns[2*indshift3 + thisindshift] += wshape*nphirot*conj(phirot);
                    Gns[3*indshift3 + thisindshift] += wshape*conj(nphirot*phirot);
                    Gn_norm[thisindshift] += w2*nphirot; 
                    if (n==0){
                        countbin = z1*nbinsz*nbinsr + z2*nbinsr + rbin;
                        bin_centers[countbin] += w1*w2*dist; 
                        bin_centers_norm[countbin] += w1*w2; 
                        counts[countbin] += 1;
                        
                    }                      
                }
            }
        }
    } 
    if (n==0){
        for (int elb1=0; elb1<nbinsr; elb1++){
            for (int zbin1=0; zbin1<nbinsz; zbin1++){
                for (int zbin2=0; zbin2<nbinsz; zbin2++){
                    int countbin = zbin1*nbinsz*nbinsr + zbin2*nbinsr + elb1;
                    bin_centers[countbin] /= bin_centers_norm[countbin];
                }
            }
        }
    }
    free(bin_centers_norm);
}


// Gns ~ (4, ngals, nbinsz, nbinsr) ~ [G_{n-3}, G_{-n-3}, G_{n-1}, G_{-n-1}]
// threepcf ~ (4, nbinsz, nbinsz, nbinsz, nbinsr, nbinsr)
// gammashift (elb1, elb2, zbin1, zbin2, zbin3)
// g1/2shift  (ind_gal, elb1/2, zbin2/3)
// Parallelize over ngal in similar fashion as done for 'alloc_Gns_discrete_basic'.
// With this method we make sure that no Gns are called t
void alloc_Gamman_discrete_basic(
    double complex *Gns, double complex *norms,
    double *weight, double *e1, double *e2, int *zbins, int nbinsz, int ngal, 
    int n, int nbinsr, double complex *threepcf, double complex *threepcf_n){
    
    // Index shifts for Gn
    int indshift3 = ngal*nbinsz*nbinsr;
    
    // Indexshifts for Gamman
    int gammaax0 = nbinsz*nbinsz*nbinsz*nbinsr*nbinsr;
    int gammaax1 = nbinsz*nbinsz*nbinsr*nbinsr;
    int gammaax2 = nbinsz*nbinsr*nbinsr;
    int gammaax3 = nbinsr*nbinsr;
    
    // Threepcf has shape (4, nbinsz, nbinsz, nbinsz, nbinsr, nbinsr)
    // Only pick galaxies of fixed zbin --> helps a bit for tomo setting.
    //#pragma omp parallel for num_threads(nthreads)
    for (int ind_gal=0; ind_gal<ngal; ind_gal++){
        int zbin1 = zbins[ind_gal];
        for (int elb1=0; elb1<nbinsr; elb1++){
            for (int zbin2=0; zbin2<nbinsz; zbin2++){
                double complex h0, h1, h2, h3, w0;
                int gammashift1, gammashift12, gammashift, gammashiftt, g1shift, g2shift;
                int elb2, zbin3;
                double complex wshape = weight[ind_gal] * (e1[ind_gal]+I*e2[ind_gal]);
                gammashift1 = zbin1*gammaax1;
                gammashift12 = gammashift1+zbin2*gammaax2;
                g1shift = ind_gal*nbinsz*nbinsr+zbin2*nbinsr+elb1;
                h0 = -wshape * Gns[0*indshift3 + g1shift];
                h1 = -conj(wshape) * Gns[2*indshift3 + g1shift];
                h2 = -wshape * conj(Gns[3*indshift3 + g1shift]);
                h3 = -wshape * conj(Gns[2*indshift3 + g1shift]);
                w0 = weight[ind_gal] * conj(norms[g1shift]);
                for (elb2=0; elb2<nbinsr; elb2++){
                    for (zbin3=0; zbin3<nbinsz; zbin3++){
                        g2shift = ind_gal*nbinsz*nbinsr+ zbin3*nbinsr + elb2;
                        gammashift = gammashift12 + zbin3*gammaax3 + elb1*nbinsr + elb2;
                        gammashiftt = gammashift12 + zbin3*gammaax3 + elb2*nbinsr + elb1;
                        threepcf[gammashift] += h0*Gns[1*indshift3 + g2shift];
                        threepcf[gammaax0 + gammashift] += h1*Gns[3*indshift3 + g2shift];
                        threepcf[2*gammaax0 + gammashift] += h2*Gns[1*indshift3 + g2shift];
                        threepcf[3*gammaax0 + gammashiftt] += h3*Gns[0*indshift3 + g2shift];
                        threepcf_n[gammashiftt] += w0*norms[g2shift];
                    }
                }
            }
        }
    }
}

    
// Gammans/Gammans_norm have shape (nmax+1, nr, nr)
// No atomics involved here --> Make sure that no race conditions occure (as done in python layer)
void alloc_Gammans_discrete_G3L(
    double *w_source, double *pos1_source, double *pos2_source, double *e1, double *e2, int ngal_source,
    double *w_lens, double *pos1_lens, double *pos2_lens, int ngal_lens, 
    int nmax, double rmin, double rmax, int nbinsr,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, double *rbin_means, double complex *Gammans, double complex *Gammans_norm){
        
    // Need to allocate the bin centers/counts in this way to ensure parallelizability
    // At a later stage are reduced to the shape of the output
    double *rbin_means_denom = calloc(nbinsr, sizeof(double));
    
    // Allocate Gns
    // We do this in parallel as follows:
    // * Split survey in 2*nthreads equal area stripes along x-axis
    // * Do two parallelized iterations over galaxies
    //   - In first one only consider galaxies within stripes of even number
    //   - In second one only consider galaxies within stripes of odd number
    // --> We avoid race conditions in calling the spatial hash arrays. - This
    //    is explicitly made sure by (re)setting nthreads in the python layer.
    for (int odd=0; odd<2; odd++){
        //#pragma omp parallel for private(nbinsz, ngal, nmin, nmax, rmin, rmax, nbinsr, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, nthreads)
        double complex *tmpGamma_n = calloc(nthreads*(2*nmax+1)*nbinsr*nbinsr, sizeof(double complex));
        double complex *tmpGamma_n_norm = calloc(nthreads*(2*nmax+1)*nbinsr*nbinsr, sizeof(double complex));
        double *tmprbin_nom = calloc(nthreads*nbinsr, sizeof(double));
        double *tmprbin_denom = calloc(nthreads*nbinsr, sizeof(double));
        #pragma omp parallel for num_threads(nthreads)
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            // Index shifts for Gn
            for (int ind_gal=0; ind_gal<ngal_source; ind_gal++){

                // Check if galaxy falls in stripe used in this process
                double p11, p12, e11, e12, w_s;
                #pragma omp critical
                {p11 = pos1_source[ind_gal];
                p12 = pos2_source[ind_gal];
                w_s = w_source[ind_gal];
                e11 = e1[ind_gal];
                e12 = e2[ind_gal];}
                int thisstripe = 2*thisthread + odd;
                int galstripe = (int) floor((p11-pix1_start)/pix1_d * (2*nthreads)/pix1_n);
                if (thisstripe != galstripe){continue;}
                
                // Allocate the Gn for this galaxy
                // ordering is {-nmax-1, -nmax, ..., nmax-1} for Gn 
                //             {-nmax, ..., nmax} for Gn_norm
                int ind_pix1, ind_pix2, ind_inpix, ind_gal2;
                int ind_red, lower, upper; 
                double p21, p22, w_l;
                double rel1, rel2, dist, dphi;
                double complex wshape;
                int tmpGind, rbin;
                double complex nphirot, phirot;

                double drbin = log(rmax/rmin)/(nbinsr);

                int pix1_lower = mymax(0, (int) floor((p11 - (rmax+pix1_d) - pix1_start)/pix1_d));
                int pix2_lower = mymax(0, (int) floor((p12 - (rmax+pix2_d) - pix2_start)/pix2_d));
                int pix1_upper = mymin(pix1_n-1, (int) floor((p11 + (rmax+pix1_d) - pix1_start)/pix1_d));
                int pix2_upper = mymin(pix2_n-1, (int) floor((p12 + (rmax+pix2_d) - pix2_start)/pix2_d));
                
                double complex *thisGns = calloc((nmax+3)*nbinsr, sizeof(double complex));
                for (ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                    for (ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                        ind_red = index_matcher[ind_pix2*pix1_n + ind_pix1];
                        if (ind_red==-1){continue;}
                        lower = pixs_galind_bounds[ind_red];
                        upper = pixs_galind_bounds[ind_red+1];
                        for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                //for (ind_gal2=0; ind_gal2<ngal_lens; ind_gal2++){
                            ind_gal2 = pix_gals[ind_inpix];
                            p21 = pos1_lens[ind_gal2];
                            p22 = pos2_lens[ind_gal2];
                            w_l = w_lens[ind_gal2];

                            rel1 = p21 - p11;
                            rel2 = p22 - p12;
                            dist = sqrt(rel1*rel1 + rel2*rel2);
                            if(dist < rmin || dist >= rmax) continue;
                            rbin = (int) floor(log(dist/rmin)/drbin);
                            tmprbin_nom[thisthread*nbinsr+rbin] += w_l*w_s*dist;
                            tmprbin_denom[thisthread*nbinsr+rbin] += w_l*w_s;
                            dphi = atan2(rel2,rel1);
                            if (dphi<0){dphi += 2*M_PI;}
                            //LL
                            phirot = cexp(-I*dphi);
                            //LP
                            //phirot = cexp(I*dphi);
                            nphirot = 1;
                            tmpGind = rbin;
                            for (int nextn=0;nextn<=nmax+2;nextn++){
                                thisGns[tmpGind] += w_l*nphirot;
                                nphirot *= phirot; 
                                tmpGind += nbinsr;
                            }
                        }
                    }
                }
                
                // Update the Gamman & Gamman_norm for this galaxy
                // shape (nthreads, nmax+1, nbinsr, nbinsr)
                double complex Gn1, Gn2, Gnnorm1, Gnnorm2;
                int threadshift = thisthread*(2*nmax+1)*nbinsr*nbinsr;
            
                int Gammaind, thisnshift;
                wshape = w_s*(e11+I*e12);
                for (int thisn=-nmax; thisn<=nmax; thisn++){
                    thisnshift = (thisn+nmax)*nbinsr*nbinsr;
                    for (int elb1=0; elb1<nbinsr; elb1++){
                        //LL
                        if (thisn+1>=0){Gn1 = thisGns[(thisn+1)*nbinsr+elb1];}
                        else{Gn1 = conj(thisGns[(-thisn-1)*nbinsr+elb1]);}
                        if(thisn>=0){Gnnorm1 = thisGns[thisn*nbinsr+elb1];}
                        else{Gnnorm1 = conj(thisGns[(-thisn)*nbinsr+elb1]);}
                        //LP
                        //if (thisn-1>=0){Gn1 = thisGns[(thisn-1)*nbinsr+elb1];}
                        //else{Gn1 = conj(thisGns[(-thisn+1)*nbinsr+elb1]);}
                        //if(thisn>=0){Gnnorm1 = thisGns[thisn*nbinsr+elb1];}
                        //else{Gnnorm1 = conj(thisGns[(-thisn)*nbinsr+elb1]);}
                        for (int elb2=0; elb2<nbinsr; elb2++){
                            //LL
                            if (-thisn+1>=0){Gn2 = (thisGns[(-thisn+1)*nbinsr+elb2]);}
                            else{Gn2 = conj(thisGns[(thisn-1)*nbinsr+elb2]);}
                            if(-thisn>=0){Gnnorm2 = thisGns[(-thisn)*nbinsr+elb2];}
                            else{Gnnorm2 = conj(thisGns[thisn*nbinsr+elb2]);}
                            //LP
                            //if (-thisn-1>=0){Gn2 = (thisGns[(-thisn-1)*nbinsr+elb2]);}
                            //else{Gn2 = conj(thisGns[(thisn+1)*nbinsr+elb2]);}
                            //if(-thisn>=0){Gnnorm2 = thisGns[(-thisn)*nbinsr+elb2];}
                            //else{Gnnorm2 = conj(thisGns[thisn*nbinsr+elb2]);}
                            Gammaind = threadshift+thisnshift+elb1*nbinsr+elb2;
                            tmpGamma_n[Gammaind] += wshape*Gn1*Gn2;
                            tmpGamma_n_norm[Gammaind] += w_s*Gnnorm1*Gnnorm2;
                        }
                    }
                }
                free(thisGns);
            }
        }
        
        // Accumulate the Gammans & Gammans_norm 
        for (int nthread=0; nthread<nthreads; nthread++){
            int threadshift = nthread*(2*nmax+1)*nbinsr*nbinsr;
            for (int thisn=0; thisn<=2*nmax; thisn++){
                int nshift = thisn*nbinsr*nbinsr;
                for (int elb1=0; elb1<nbinsr; elb1++){
                    for (int elb2=0; elb2<nbinsr; elb2++){
                        Gammans[nshift+elb1*nbinsr+elb2] += tmpGamma_n[threadshift+nshift+elb1*nbinsr+elb2];
                        Gammans_norm[nshift+elb1*nbinsr+elb2]+= tmpGamma_n_norm[threadshift+nshift+elb1*nbinsr+elb2]; 
                    }
                }
            }
        } 
        
        // Accumulate the bin centers
        for (int nthread=0; nthread<nthreads; nthread++){
            for (int elb=0; elb<nbinsr; elb++){
                rbin_means[elb] += tmprbin_nom[nthread*nbinsr+elb];
                rbin_means_denom[elb] += tmprbin_denom[nthread*nbinsr+elb];
            }
        }
        
        free(tmpGamma_n);
        free(tmpGamma_n_norm);
        free(tmprbin_nom);
        free(tmprbin_denom);
    } 
    
    // Finalize radial bin centers
    for (int elb=0; elb<nbinsr; elb++){
        rbin_means[elb] /= rbin_means_denom[elb];
    }
    free(rbin_means_denom);
}

// Gammans/Gammans_norm have shape (nmax+1, nr, nr)
// No atomics involved here --> Make sure that no race conditions occure (as done in python layer)
void alloc_Gammans_discrete_SSL(
    double *w_source, double *pos1_source, double *pos2_source, double *e1, double *e2, int ngal_source,
    double *w_lens, double *pos1_lens, double *pos2_lens, int ngal_lens, 
    int nmax, double rmin, double rmax, int nbinsr,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, double *rbin_means, double complex *Gammans, double complex *Gammans_norm){
        
    // Denominator for bin center computation
    double *rbin_means_denom = calloc(nbinsr, sizeof(double));
    
    // Allocate Gns
    // We do this in parallel as follows:
    // * Split survey in 2*nthreads equal area stripes along x-axis
    // * Do two parallelized iterations over galaxies
    //   - In first one only consider galaxies within stripes of even number
    //   - In second one only consider galaxies within stripes of odd number
    // --> We avoid race conditions in calling the spatial hash arrays. - This
    //    is explicitly made sure by (re)setting nthreads in the python layer.
    for (int odd=0; odd<2; odd++){
        //#pragma omp parallel for private(nbinsz, ngal, nmin, nmax, rmin, rmax, nbinsr, pix1_start, pix1_d, pix1_n, pix2_start, pix2_d, pix2_n, nthreads)
        double complex *tmpGamma_n = calloc(nthreads*(2*nmax+1)*nbinsr*nbinsr, sizeof(double complex));
        double complex *tmpGamma_n_norm = calloc(nthreads*(2*nmax+1)*nbinsr*nbinsr, sizeof(double complex));
        double *tmprbin_nom = calloc(nthreads*nbinsr, sizeof(double));
        double *tmprbin_denom = calloc(nthreads*nbinsr, sizeof(double));
        #pragma omp parallel for num_threads(nthreads)
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            // Index shifts for Gn
            for (int ind_gal=0; ind_gal<ngal_lens; ind_gal++){

                // Check if galaxy falls in stripe used in this process
                double p11, p12, w_l;
                #pragma omp critical
                {p11 = pos1_lens[ind_gal];
                p12 = pos2_lens[ind_gal];
                w_l = w_lens[ind_gal];}
                int thisstripe = 2*thisthread + odd;
                int galstripe = (int) floor((p11-pix1_start)/pix1_d * (2*nthreads)/pix1_n);
                if (thisstripe != galstripe){continue;}
                
                // Allocate the Gn for this galaxy
                // ordering is {-nmax-1, -nmax, ..., nmax-1} for Gn 
                //             {-nmax, ..., nmax} for Gn_norm
                int ind_pix1, ind_pix2, ind_inpix, ind_gal2;
                int ind_red, lower, upper; 
                double p21, p22, e11, e12, w_s;
                double rel1, rel2, dist, dphi;
                double complex wshape;
                int tmpGind, rbin;
                double complex nphirot, phirot;

                double drbin = log(rmax/rmin)/(nbinsr);

                int pix1_lower = mymax(0, (int) floor((p11 - (rmax+pix1_d) - pix1_start)/pix1_d));
                int pix2_lower = mymax(0, (int) floor((p12 - (rmax+pix2_d) - pix2_start)/pix2_d));
                int pix1_upper = mymin(pix1_n-1, (int) floor((p11 + (rmax+pix1_d) - pix1_start)/pix1_d));
                int pix2_upper = mymin(pix2_n-1, (int) floor((p12 + (rmax+pix2_d) - pix2_start)/pix2_d));
                
                double complex *thisGns = calloc((nmax+4)*nbinsr, sizeof(double complex));
                double complex *thisGns_norm = calloc((nmax+4)*nbinsr, sizeof(double complex));
                for (ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                    for (ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                        ind_red = index_matcher[ind_pix2*pix1_n + ind_pix1];
                        if (ind_red==-1){continue;}
                        lower = pixs_galind_bounds[ind_red];
                        upper = pixs_galind_bounds[ind_red+1];
                        for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                            ind_gal2 = pix_gals[ind_inpix];
                            p21 = pos1_source[ind_gal2];
                            p22 = pos2_source[ind_gal2];
                            e11 = e1[ind_gal2];
                            e12 = e2[ind_gal2];
                            w_s = w_source[ind_gal2];
                            wshape = w_s * (e11 + I*e12);
                            rel1 = p21 - p11;
                            rel2 = p22 - p12;
                            dist = sqrt(rel1*rel1 + rel2*rel2);
                            if(dist < rmin || dist >= rmax) continue;
                            rbin = (int) floor(log(dist/rmin)/drbin);
                            tmprbin_nom[thisthread*nbinsr+rbin] += w_l*w_s*dist;
                            tmprbin_denom[thisthread*nbinsr+rbin] += w_l*w_s;
                            dphi = atan2(rel2,rel1);
                            if (dphi<0){dphi += 2*M_PI;}
                            //LL
                            phirot = cexp(-I*dphi);
                            //LP
                            //phirot = cexp(I*dphi);
                            nphirot = 1;
                            tmpGind = rbin;
                            for (int nextn=0;nextn<=nmax+3;nextn++){
                                thisGns[tmpGind] += wshape*nphirot;
                                thisGns_norm[tmpGind] += w_s*nphirot;
                                nphirot *= phirot; 
                                tmpGind += nbinsr;
                            }
                        }
                    }
                }
                
                // Update the Gamman & Gamman_norm for this galaxy
                // shape (nthreads, nmax+1, nbinsr, nbinsr)
                double complex Gn1, Gn2, Gnnorm1, Gnnorm2;
                int threadshift = thisthread*(2*nmax+1)*nbinsr*nbinsr;
            
                int Gammaind, thisnshift;
                for (int thisn=-nmax; thisn<=nmax; thisn++){
                    thisnshift = (thisn+nmax)*nbinsr*nbinsr;
                    for (int elb1=0; elb1<nbinsr; elb1++){
                        //LL
                        if (thisn-2>=0){Gn1 = thisGns[(thisn-2)*nbinsr+elb1];}
                        else{Gn1 = conj(thisGns[(-thisn+2)*nbinsr+elb1]);}
                        if(thisn>=0){Gnnorm1 = thisGns_norm[thisn*nbinsr+elb1];}
                        else{Gnnorm1 = conj(thisGns_norm[(-thisn)*nbinsr+elb1]);}
                        //LP
                        //if (thisn-1>=0){Gn1 = thisGns[(thisn-1)*nbinsr+elb1];}
                        //else{Gn1 = conj(thisGns[(-thisn+1)*nbinsr+elb1]);}
                        //if(thisn>=0){Gnnorm1 = thisGns[thisn*nbinsr+elb1];}
                        //else{Gnnorm1 = conj(thisGns[(-thisn)*nbinsr+elb1]);}
                        for (int elb2=0; elb2<nbinsr; elb2++){
                            //LL
                            if (thisn+2>=0){Gn2 = conj(thisGns[(thisn+2)*nbinsr+elb2]);}
                            else{Gn2 = thisGns[(-thisn-2)*nbinsr+elb2];}
                            if(-thisn>=0){Gnnorm2 = thisGns_norm[(-thisn)*nbinsr+elb2];}
                            else{Gnnorm2 = conj(thisGns_norm[thisn*nbinsr+elb2]);}
                            //LP
                            //if (-thisn-1>=0){Gn2 = (thisGns[(-thisn-1)*nbinsr+elb2]);}
                            //else{Gn2 = conj(thisGns[(thisn+1)*nbinsr+elb2]);}
                            //if(-thisn>=0){Gnnorm2 = thisGns[(-thisn)*nbinsr+elb2];}
                            //else{Gnnorm2 = conj(thisGns[thisn*nbinsr+elb2]);}
                            Gammaind = threadshift+thisnshift+elb1*nbinsr+elb2;
                            tmpGamma_n[Gammaind] += w_l*Gn1*Gn2;
                            tmpGamma_n_norm[Gammaind] += w_l*Gnnorm1*Gnnorm2;
                        }
                    }
                }
                free(thisGns);
                free(thisGns_norm);
            }
        }
        
        // Accumulate the Gammans & Gammans_norm (single thread...)
        for (int nthread=0; nthread<nthreads; nthread++){
            int threadshift = nthread*(2*nmax+1)*nbinsr*nbinsr;
            for (int thisn=0; thisn<=2*nmax; thisn++){
                int nshift = thisn*nbinsr*nbinsr;
                for (int elb1=0; elb1<nbinsr; elb1++){
                    for (int elb2=0; elb2<nbinsr; elb2++){
                        Gammans[nshift+elb1*nbinsr+elb2] += tmpGamma_n[threadshift+nshift+elb1*nbinsr+elb2];
                        Gammans_norm[nshift+elb1*nbinsr+elb2]+= tmpGamma_n_norm[threadshift+nshift+elb1*nbinsr+elb2]; 
                    }
                }
            }
        } 
        
        // Accumulate the Gammans & Gammans_norm (single thread...)
        for (int nthread=0; nthread<nthreads; nthread++){
            for (int elb=0; elb<nbinsr; elb++){
                rbin_means[elb] += tmprbin_nom[nthread*nbinsr+elb];
                rbin_means_denom[elb] += tmprbin_denom[nthread*nbinsr+elb];
            }
        }
        
        free(tmpGamma_n);
        free(tmpGamma_n_norm);
        free(tmprbin_nom);
        free(tmprbin_denom);
    } 
    for (int elb=0; elb<nbinsr; elb++){
        rbin_means[elb] /= rbin_means_denom[elb];
    }
    free(rbin_means_denom);
}

         
// Gns ~ (4, nbinsz, nbinsr) ~ [G_{n-3}, G_{-n-3}, G_{n-1}, G_{-n-1}]
// threepcf ~ (nworkers * 4 * nbinsz * nbinsz * nbinsz * nbinsr * nbinsr)
// threepcf_n ~ (nworkers * nbinsz * nbinsz * nbinsz * nbinsr * nbinsr)
// gammashift (elb1, elb2, zbin1, zbin2, zbin3)
// g1/2shift  (ind_gal, elb1/2, zbin2/3)
// Parallelize over ngal in similar fashion as done for 'alloc_Gns_discrete_basic'.
// With this method we make sure that no Gns are called t
void update_Gamman_discrete_worker(
    double complex *Gns, double complex *Gnnorms,
    int nmin, int nmax, int n, int nbinsr, int nbinsz, int nworker, 
    double weight, double e1, double e2, int zbin1,
    double complex *threepcf, double complex *threepcf_n){
    
    // Index shifts for Gn
    int indshift2 = nbinsz*nbinsr;
    
    // Indexshifts for Gamman
    int gammaax0 = nbinsz*nbinsz*nbinsz*nbinsr*nbinsr;
    int gammaax1 = nbinsz*nbinsz*nbinsr*nbinsr;
    int gammaax2 = nbinsz*nbinsr*nbinsr;
    int gammaax3 = nbinsr*nbinsr;
    int nshift = 4*(nmax-nmin+1)*gammaax0;
    int baseshift = nworker*nshift+4*(n-nmin)*gammaax0;
    int baseshift_n = nworker*(nmax-nmin+1)*gammaax0 + (n-nmin)*gammaax0;
    // Threepcf has shape (4, nbinsz, nbinsz, nbinsz, nbinsr, nbinsr)
    // Only pick galaxies of fixed zbin --> helps a bit for tomo setting.
    double complex wshape = weight * (e1+I*e2);
    for (int elb1=0; elb1<nbinsr; elb1++){
        for (int zbin2=0; zbin2<nbinsz; zbin2++){
            double complex h0, h1, h2, h3, w0;
            int gammashift1, gammashift12, gammashift, gammashiftt, g1shift, g2shift;
            int elb2, zbin3;
            gammashift1 = zbin1*gammaax1;
            gammashift12 = gammashift1+zbin2*gammaax2;
            g1shift = zbin2*nbinsr+elb1;
            h0 = -wshape * Gns[0*indshift2 + g1shift];
            h1 = -conj(wshape) * Gns[2*indshift2 + g1shift];
            h2 = -wshape * conj(Gns[3*indshift2 + g1shift]);
            h3 = -wshape * conj(Gns[2*indshift2 + g1shift]);
            w0 = weight * conj(Gnnorms[g1shift]);
            for (elb2=0; elb2<nbinsr; elb2++){
                for (zbin3=0; zbin3<nbinsz; zbin3++){
                    g2shift = zbin3*nbinsr + elb2;
                    gammashift = gammashift12 + zbin3*gammaax3 + elb1*nbinsr + elb2;
                    gammashiftt = gammashift12 + zbin3*gammaax3 + elb2*nbinsr + elb1;
                    threepcf[baseshift+gammashift] += h0*Gns[1*indshift2 + g2shift];
                    threepcf[baseshift+ gammaax0 + gammashift] += h1*Gns[3*indshift2 + g2shift];
                    threepcf[baseshift + 2*gammaax0 + gammashift] += h2*Gns[1*indshift2 + g2shift];
                    threepcf[baseshift + 3*gammaax0 + gammashiftt] += h3*Gns[0*indshift2 + g2shift];
                    threepcf_n[baseshift_n + gammashiftt] += w0*Gnnorms[g2shift];
                }
            }
        }
    }
}
    
    
void update_Gammansingle_discmixed_worker( 
    double complex *Gns_disc, double complex *Gnnorms_disc, 
    double complex *Gns_grid, double complex *Gnnorms_grid,
    int nbinsr_disc, int nbinsr_grid, int nbinsz, int nreso,
    int nworker, int zbin1, double weight_disc, double e1_disc, double e2_disc,
    double *weight_grid, double *e1_grid, double *e2_grid, int *cumnbinsr_grid,
    double complex *Gamman_dd, double complex *Gammannorm_dd,
    double complex *Gamman_dg, double complex *Gammannorm_dg,
    double complex *Gamman_gd, double complex *Gammannorm_gd){
    
    size_t gammaax0, gammaax1, gammaax2, gammaax3;
    size_t gammashift1, gammashift12, gammashift, g1shift, g2shift;
    size_t baseshift, baseshift_n;
    int elb2, zbin3;
    double complex h0, h1, h2, h3, w0;
    
    int thisreso;
    int gax2_disc = nbinsr_disc*nbinsr_disc;
    int gax2_grid = nbinsr_grid*nbinsr_grid;
    double complex wshape_disc = weight_disc * (e1_disc+I*e2_disc);
    double complex *wshape_grid = calloc(nreso, sizeof(double complex));
    for (int ireso=0; ireso<nreso; ireso++){
        wshape_grid[ireso] = weight_grid[ireso] * (e1_grid[ireso]+I*e2_grid[ireso]);
    }
    
    // Update Gamman_dd
    gammaax0 = nbinsz*nbinsz*nbinsz*nbinsr_disc*nbinsr_disc;
    gammaax1 = nbinsz*nbinsz*nbinsr_disc*nbinsr_disc;
    gammaax2 = nbinsz*nbinsr_disc*nbinsr_disc;
    gammaax3 = nbinsr_disc*nbinsr_disc;
    baseshift = nworker*4*gammaax0;
    baseshift_n = nworker*gammaax0;
    for (int elb1=0; elb1<nbinsr_disc; elb1++){
        for (int zbin2=0; zbin2<nbinsz; zbin2++){
            gammashift1 = zbin1*gammaax1;
            gammashift12 = gammashift1+zbin2*gammaax2;
            g1shift = zbin2*nbinsr_disc+elb1;
            h0 = -wshape_disc * Gns_disc[0*gax2_disc + g1shift];
            h1 = -conj(wshape_disc) * Gns_disc[2*gax2_disc + g1shift];
            h2 = -wshape_disc * conj(Gns_disc[3*gax2_disc + g1shift]);
            h3 = -wshape_disc * Gns_disc[0*gax2_disc + g1shift];
            w0 = weight_disc * Gnnorms_disc[g1shift];
            for (elb2=0; elb2<nbinsr_disc; elb2++){
                for (zbin3=0; zbin3<nbinsz; zbin3++){
                    g2shift = zbin3*nbinsr_disc + elb2;
                    gammashift = gammashift12 + zbin3*gammaax3 + elb1*nbinsr_disc + elb2;
                    Gamman_dd[baseshift+gammashift] += h0*Gns_disc[1*gax2_disc + g2shift];
                    Gamman_dd[baseshift+ gammaax0 + gammashift] += h1*Gns_disc[3*gax2_disc + g2shift];
                    Gamman_dd[baseshift + 2*gammaax0 + gammashift] += h2*Gns_disc[1*gax2_disc + g2shift];
                    Gamman_dd[baseshift + 3*gammaax0 + gammashift] += h3*conj(Gns_disc[2*gax2_disc + g2shift]);
                    Gammannorm_dd[baseshift_n + gammashift] += w0*conj(Gnnorms_disc[g2shift]);
                }
            }
        }
    }
    
    // Update Gamman_dg
    gammaax0 = nbinsz*nbinsz*nbinsz*nbinsr_disc*nbinsr_grid;
    gammaax1 = nbinsz*nbinsz*nbinsr_disc*nbinsr_grid;
    gammaax2 = nbinsz*nbinsr_disc*nbinsr_grid;
    gammaax3 = nbinsr_disc*nbinsr_grid;
    baseshift = nworker*4*gammaax0;
    baseshift_n = nworker*gammaax0;
    for (int elb1=0; elb1<nbinsr_disc; elb1++){
        for (int zbin2=0; zbin2<nbinsz; zbin2++){
            gammashift1 = zbin1*gammaax1;
            gammashift12 = gammashift1+zbin2*gammaax2;
            g1shift = zbin2*nbinsr_disc+elb1;
            h0 = -wshape_disc * Gns_disc[0*gax2_disc + g1shift];
            h1 = -conj(wshape_disc) * Gns_disc[2*gax2_disc + g1shift];
            h2 = -wshape_disc * conj(Gns_disc[3*gax2_disc + g1shift]);
            h3 = -wshape_disc * Gns_disc[0*gax2_disc + g1shift];
            w0 = weight_disc * Gnnorms_disc[g1shift];
            for (elb2=0; elb2<nbinsr_grid; elb2++){
                for (zbin3=0; zbin3<nbinsz; zbin3++){
                    g2shift = zbin3*nbinsr_grid + elb2;
                    gammashift = gammashift12 + zbin3*gammaax3 + elb1*nbinsr_grid + elb2;
                    Gamman_dg[baseshift+gammashift] += h0*Gns_grid[1*gax2_grid + g2shift];
                    Gamman_dg[baseshift+ gammaax0 + gammashift] += h1*Gns_grid[3*gax2_grid + g2shift];
                    Gamman_dg[baseshift + 2*gammaax0 + gammashift] += h2*Gns_grid[1*gax2_grid + g2shift];
                    Gamman_dg[baseshift + 3*gammaax0 + gammashift] += h3*conj(Gns_grid[2*gax2_grid + g2shift]);
                    Gammannorm_dg[baseshift_n + gammashift] += w0*conj(Gnnorms_grid[g2shift]);
                }
            }
        }
    }
    
    // Update Gamman_gd
    gammaax0 = nbinsz*nbinsz*nbinsz*nbinsr_disc*nbinsr_grid;
    gammaax1 = nbinsz*nbinsz*nbinsr_disc*nbinsr_grid;
    gammaax2 = nbinsz*nbinsr_disc*nbinsr_grid;
    gammaax3 = nbinsr_disc*nbinsr_grid;
    baseshift = nworker*4*gammaax0;
    baseshift_n = nworker*gammaax0;
    thisreso = 0;
    for (int elb1=0; elb1<nbinsr_grid; elb1++){
        if(elb1 >= cumnbinsr_grid[thisreso+1]){thisreso+=1;}
        for (int zbin2=0; zbin2<nbinsz; zbin2++){
            gammashift1 = zbin1*gammaax1;
            gammashift12 = gammashift1+zbin2*gammaax2;
            g1shift = zbin2*nbinsr_grid+elb1;
            h0 = -wshape_grid[thisreso] * Gns_grid[0*gax2_grid + g1shift];
            h1 = -conj(wshape_grid[thisreso]) * Gns_grid[2*gax2_grid + g1shift];
            h2 = -wshape_grid[thisreso] * conj(Gns_grid[3*gax2_grid + g1shift]);
            h3 = -wshape_grid[thisreso] * Gns_grid[0*gax2_grid + g1shift];
            w0 = weight_grid[thisreso] * Gnnorms_grid[g1shift];
            for (elb2=0; elb2<nbinsr_disc; elb2++){
                for (zbin3=0; zbin3<nbinsz; zbin3++){
                    g2shift = zbin3*nbinsr_grid + elb2;
                    gammashift = gammashift12 + zbin3*gammaax3 + elb1*nbinsr_disc + elb2;
                    Gamman_gd[baseshift+gammashift] += h0*Gns_disc[1*gax2_disc + g2shift];
                    Gamman_gd[baseshift+ gammaax0 + gammashift] += h1*Gns_disc[3*gax2_disc + g2shift];
                    Gamman_gd[baseshift + 2*gammaax0 + gammashift] += h2*Gns_disc[1*gax2_disc + g2shift];
                    Gamman_gd[baseshift + 3*gammaax0 + gammashift] += h3*conj(Gns_disc[2*gax2_disc + g2shift]);
                    Gammannorm_gd[baseshift_n + gammashift] += w0*conj(Gnnorms_disc[g2shift]);
                }
            }
        }
    }
}
    
// Gns_a and Gns_b are either the grid or discrete components
// weight, e1, e2, zbin1 correspond to:
//  * values of the FFT grid at pixel position of galaxy if Gns_a==grid
//  * values of galaxy if Gns_a==discrete
// Gns ~ (4, nbinsz, nbinsr) ~ [G_{n-3}, G_{-n-3}, G_{n-1}, G_{-n-1}]
// threepcf_dd ~ (nthreads * 4 * nbinsz * nbinsz * nbinsz * nbinsr_disc * nbinsr_disc)
// threepcfnorm_dd ~ (nthreads * nbinsz * nbinsz * nbinsz * nbinsr * nbinsr)
// gammashift (elb1, elb2, zbin1, zbin2, zbin3)
// g1/2shift  (ind_gal, elb1/2, zbin2/3)
// Parallelize over ngal in similar fashion as done for 'alloc_Gns_discrete_basic'.
// With this method we make sure that no Gns are called t
void update_Gamman_discmixed_worker(
    double complex *Gns_disc, double complex *Gnnorms_disc, 
    double complex *Gns_grid, double complex *Gnnorms_grid,
    int nmin, int nmax, int n, int nbinsr_disc, int nbinsr_grid, int nbinsz, 
    int nworker, int zbin1,
    double weight_disc, double e1_disc, double e2_disc,
    double weight_grid, double e1_grid, double e2_grid,
    double complex *threepcf_dd, double complex *threepcfnorm_dd,
    double complex *threepcf_dg, double complex *threepcfnorm_dg,
    double complex *threepcf_gd, double complex *threepcfnorm_gd){
    
    size_t gammaax0, gammaax1, gammaax2, gammaax3;
    size_t gammashift1, gammashift12, gammashift, g1shift, g2shift;
    size_t nshift, baseshift, baseshift_n;
    int elb2, zbin3;
    double complex h0, h1, h2, h3, w0;
    
    int gax2_disc = nbinsr_disc*nbinsr_disc;
    int gax2_grid = nbinsr_grid*nbinsr_grid;
    double complex wshape_disc = weight_disc * (e1_disc+I*e2_disc);
    
    // Update Gamman_dd
    gammaax0 = nbinsz*nbinsz*nbinsz*nbinsr_disc*nbinsr_disc;
    gammaax1 = nbinsz*nbinsz*nbinsr_disc*nbinsr_disc;
    gammaax2 = nbinsz*nbinsr_disc*nbinsr_disc;
    gammaax3 = nbinsr_disc*nbinsr_disc;
    nshift = 4*(nmax-nmin+1)*gammaax0;
    baseshift = nworker*nshift+4*(n-nmin)*gammaax0;
    baseshift_n = nworker*(nmax-nmin+1)*gammaax0 + (n-nmin)*gammaax0;
    for (int elb1=0; elb1<nbinsr_disc; elb1++){
        for (int zbin2=0; zbin2<nbinsz; zbin2++){
            gammashift1 = zbin1*gammaax1;
            gammashift12 = gammashift1+zbin2*gammaax2;
            g1shift = zbin2*nbinsr_disc+elb1;
            h0 = -wshape_disc * Gns_disc[0*gax2_disc + g1shift];
            h1 = -conj(wshape_disc) * Gns_disc[2*gax2_disc + g1shift];
            h2 = -wshape_disc * conj(Gns_disc[3*gax2_disc + g1shift]);
            h3 = -wshape_disc * Gns_disc[0*gax2_disc + g1shift];
            w0 = weight_disc * Gnnorms_disc[g1shift];
            for (elb2=0; elb2<nbinsr_disc; elb2++){
                for (zbin3=0; zbin3<nbinsz; zbin3++){
                    g2shift = zbin3*nbinsr_disc + elb2;
                    gammashift = gammashift12 + zbin3*gammaax3 + elb1*nbinsr_disc + elb2;
                    threepcf_dd[baseshift+gammashift] += h0*Gns_disc[1*gax2_disc + g2shift];
                    threepcf_dd[baseshift+ gammaax0 + gammashift] += h1*Gns_disc[3*gax2_disc + g2shift];
                    threepcf_dd[baseshift + 2*gammaax0 + gammashift] += h2*Gns_disc[1*gax2_disc + g2shift];
                    threepcf_dd[baseshift + 3*gammaax0 + gammashift] += h3*conj(Gns_disc[2*gax2_disc + g2shift]);
                    threepcfnorm_dd[baseshift_n + gammashift] += w0*conj(Gnnorms_disc[g2shift]);
                }
            }
        }
    }
    
    // Update Gamman_dg
    gammaax0 = nbinsz*nbinsz*nbinsz*nbinsr_disc*nbinsr_grid;
    gammaax1 = nbinsz*nbinsz*nbinsr_disc*nbinsr_grid;
    gammaax2 = nbinsz*nbinsr_disc*nbinsr_grid;
    gammaax3 = nbinsr_disc*nbinsr_grid;
    nshift = 4*(nmax-nmin+1)*gammaax0;
    baseshift = nworker*nshift+4*(n-nmin)*gammaax0;
    baseshift_n = nworker*(nmax-nmin+1)*gammaax0 + (n-nmin)*gammaax0;
    for (int elb1=0; elb1<nbinsr_disc; elb1++){
        for (int zbin2=0; zbin2<nbinsz; zbin2++){
            gammashift1 = zbin1*gammaax1;
            gammashift12 = gammashift1+zbin2*gammaax2;
            g1shift = zbin2*nbinsr_disc+elb1;
            h0 = -wshape_disc * Gns_disc[0*gax2_disc + g1shift];
            h1 = -conj(wshape_disc) * Gns_disc[2*gax2_disc + g1shift];
            h2 = -wshape_disc * conj(Gns_disc[3*gax2_disc + g1shift]);
            h3 = -wshape_disc * Gns_disc[0*gax2_disc + g1shift];
            w0 = weight_disc * Gnnorms_disc[g1shift];
            for (elb2=0; elb2<nbinsr_grid; elb2++){
                for (zbin3=0; zbin3<nbinsz; zbin3++){
                    g2shift = zbin3*nbinsr_grid + elb2;
                    gammashift = gammashift12 + zbin3*gammaax3 + elb1*nbinsr_grid + elb2;
                    threepcf_dg[baseshift+gammashift] += h0*Gns_grid[1*gax2_grid + g2shift];
                    threepcf_dg[baseshift+ gammaax0 + gammashift] += h1*Gns_grid[3*gax2_grid + g2shift];
                    threepcf_dg[baseshift + 2*gammaax0 + gammashift] += h2*Gns_grid[1*gax2_grid + g2shift];
                    threepcf_dg[baseshift + 3*gammaax0 + gammashift] += h3*conj(Gns_grid[2*gax2_grid + g2shift]);
                    threepcfnorm_dg[baseshift_n + gammashift] += w0*conj(Gnnorms_grid[g2shift]);
                }
            }
        }
    }
    
    // Update Gamman_gd
    gammaax0 = nbinsz*nbinsz*nbinsz*nbinsr_disc*nbinsr_grid;
    gammaax1 = nbinsz*nbinsz*nbinsr_disc*nbinsr_grid;
    gammaax2 = nbinsz*nbinsr_disc*nbinsr_grid;
    gammaax3 = nbinsr_disc*nbinsr_grid;
    nshift = 4*(nmax-nmin+1)*gammaax0;
    baseshift = nworker*nshift+4*(n-nmin)*gammaax0;
    baseshift_n = nworker*(nmax-nmin+1)*gammaax0 + (n-nmin)*gammaax0;
    for (int elb1=0; elb1<nbinsr_grid; elb1++){
        for (int zbin2=0; zbin2<nbinsz; zbin2++){
            gammashift1 = zbin1*gammaax1;
            gammashift12 = gammashift1+zbin2*gammaax2;
            g1shift = zbin2*nbinsr_grid+elb1;
            h0 = -wshape_disc * Gns_grid[0*gax2_grid + g1shift];
            h1 = -conj(wshape_disc) * Gns_grid[2*gax2_grid + g1shift];
            h2 = -wshape_disc * conj(Gns_grid[3*gax2_grid + g1shift]);
            h3 = -wshape_disc * Gns_grid[0*gax2_grid + g1shift];
            w0 = weight_disc * Gnnorms_grid[g1shift];
            for (elb2=0; elb2<nbinsr_disc; elb2++){
                for (zbin3=0; zbin3<nbinsz; zbin3++){
                    g2shift = zbin3*nbinsr_grid + elb2;
                    gammashift = gammashift12 + zbin3*gammaax3 + elb1*nbinsr_disc + elb2;
                    threepcf_dd[baseshift+gammashift] += h0*Gns_disc[1*gax2_disc + g2shift];
                    threepcf_dd[baseshift+ gammaax0 + gammashift] += h1*Gns_disc[3*gax2_disc + g2shift];
                    threepcf_dd[baseshift + 2*gammaax0 + gammashift] += h2*Gns_disc[1*gax2_disc + g2shift];
                    threepcf_dd[baseshift + 3*gammaax0 + gammashift] += h3*conj(Gns_disc[2*gax2_disc + g2shift]);
                    threepcfnorm_dd[baseshift_n + gammashift] += w0*conj(Gnnorms_disc[g2shift]);
                }
            }
        }
    }
}

void _alloc_GnGamman_discrete_basic(
    double *weight, double *pos1, double *pos2, double *e1, double *e2, int *zbins, int nbinsz, int ngal, 
    int n, double rmin, double rmax, int nbinsr,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nthreads, double complex *threepcf, double complex *threepcf_n){
    
    // Index shifts for Gn
    int indshift3 = ngal*nbinsz*nbinsr;
    int indshift2 = nbinsz*nbinsr;
    
    // Indexshifts for Gamman
    int gammaax0 = nbinsz*nbinsz*nbinsz*nbinsr*nbinsr;
    int gammaax1 = nbinsz*nbinsz*nbinsr*nbinsr;
    int gammaax2 = nbinsz*nbinsr*nbinsr;
    int gammaax3 = nbinsr*nbinsr;
    
    // Allocate Gns
    int *Gns = calloc(4*indshift3, sizeof(double complex));
    int *norms = calloc(indshift3, sizeof(double complex));
    #pragma omp parallel for num_threads(nthreads)
    for (int ind_gal=0; ind_gal<ngal; ind_gal++){
        int ind_pix1, ind_pix2, ind_inpix, ind_gal2;
        int ind_red, lower, upper; 
        double rel1, rel2, dist2, dphi;
        double complex wshape;
        double complex nphirot, phirot, threephirot;
            
        int rbin;
        int thisindshift;
        
        double drbin = (log(rmax)-log(rmin))/nbinsr;
        
        int pix1_lower = (int) floor((pos1[ind_gal] - rmax - (pix1_start-.5*pix1_d) - pix1_d)/pix1_d);
        int pix1_upper = (int) floor((pos1[ind_gal] + rmax - (pix1_start-.5*pix1_d) + pix1_d)/pix1_d);
        int pix2_lower = (int) floor((pos2[ind_gal] - rmax - (pix2_start-.5*pix2_d) - pix2_d)/pix2_d);
        int pix2_upper = (int) floor((pos2[ind_gal] + rmax - (pix2_start-.5*pix2_d) + pix1_d)/pix2_d);
        
        for (ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
            for (ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){

                ind_red = index_matcher[ind_pix2*pix1_n + ind_pix1];
                if (ind_red==-1){continue;}
                lower = pixs_galind_bounds[ind_red];
                upper = pixs_galind_bounds[ind_red+1];
                for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                    ind_gal2 = pix_gals[ind_inpix];
                    rel1 = pos1[ind_gal2] - pos1[ind_gal];
                    rel2 = pos2[ind_gal2] - pos2[ind_gal];
                    dist2 = rel1*rel1 + rel2*rel2;
                    if(dist2 < rmin*rmin || dist2 >= rmax*rmax) continue;
                    rbin = (int) floor((.5*log(dist2)-log(rmin))/drbin);
                    wshape = weight[ind_gal2]*(e1[ind_gal2]+I*e2[ind_gal2]);
                    dphi = atan2(rel2,rel1);
                    thisindshift = ind_gal*indshift2+zbins[ind_gal2]*nbinsr + rbin;
                    phirot = cexp(I*dphi);
                    nphirot = cpow(phirot, n);
                    threephirot = phirot*phirot*phirot;
                    // Allocate in order [n-3, -n-3, n-1, -n-1]
                    Gns[0*indshift3 + thisindshift] += wshape*nphirot*conj(threephirot);
                    Gns[1*indshift3 + thisindshift] += wshape*conj(nphirot*threephirot);
                    Gns[2*indshift3 + thisindshift] += wshape*nphirot*conj(phirot);
                    Gns[3*indshift3 + thisindshift] += wshape*conj(nphirot*phirot);
                    norms[thisindshift] += weight[ind_gal2]*nphirot;                    
                }
            }
        }
    } 
    
    // Allocate Gamma_n
    /*
    # h_n  ~ conv(field, g_n)
    # hc_n ~ conv(field_c, g_n)
    #  --> h_n_c :~ conv(field, g_n)_c
    #             ~ conv(field_c, g_n_c)
    #             ~ conv(field_c, g_{-n})
    #             ~ hc_{-n}
    #
    # Gamma0_n    ~ field   * h_{n-3}    * h_{-n-3}
    # Gamma1_n    ~ field_c * h_{n-1}    * h_{-n-1}
    # Gamma2_n    ~ field   * hc_{n+1}   * h_{-n-3}
    #             ~ field   * h_{-n-1}_c * h_{-n-3}
    # Gamma3_n    ~ field   * h_{n-3}    * hc_{-n+1}
    #             ~ field   * h_{n-3}    * h_{n-1}_c
    #
    # Gamma0t_n   ~ field   * h_{-n-3}   * h_{n-3}
    #             ~ Gamma0_{-n}
    # Gamma1t_n   ~ field_c * h_{-n-1}   * h_{n-1}
    #             ~ Gamma1_{-n}
    # Gamma2t_n   ~ field   * h_{-n-3}   * hc_{n+1} 
    #             ~ field   * h_{-n-3}   * h_{-n-1}_c 
    #             ~ Gamma3_{-n}
    # Gamma3t_n   ~ field   * hc_{-n+1} * h_{n-3}
    #             ~ field   * h_{n-1}_c * h_{n-3}
    #             ~ Gamma2_{-n}
    #
    # totnorm_n   ~ weight  * norm_n     * norm_n_c
    # totnormt_n  ~ weight  * norm_{-n}  * norm_{-n}_c
    #             ~ weight  * norm_n_c   * norm_n
    #             ~ totnorm_{-n}
    #
    #  --> We can allocate the nth and -nth multipole from the 4*nbins precomputed h_n components!
    #
    # We reorder those expressions s.t. we need to do little work in the inner loop i.e. for the Gammai:
    # Component       0n         1n           2n           3n       
    # Outer loop  f*h_{n-3}  f_c*h_{n-1}  f*h_{-n-1}_c  f*h_{n-1}_c   // f ~ field
    # Inner loop  h_{-n-3}    h_{-n-1}      h_{-n-3}      h_{n-3}  
    # For the Gamma3_{n}, and the normalization this amounts to swapping elb1 <--> elb2 in the allocation
    

    for elb1 in range(nbins):
        hfield = np.zeros((4, self.npix, self.npix), dtype=np.complex128)
        hfield[0] = -self.shear * hns[elb1,0]
        hfield[1] = -self.shear.conj() * hns[elb1,2]
        hfield[2] = -self.shear * hns[elb1,3].conj()
        hfield[3] = -self.shear * hns[elb1,2].conj()
        wnorm = self.weight * norms[elb1].conj()
        for elb2 in range(nbins):
            threepcf_n[0,nmax+n,elb1,elb2] = np.mean(hfield[0]*hns[elb2,1])
            threepcf_n[1,nmax+n,elb1,elb2] = np.mean(hfield[1]*hns[elb2,3])
            threepcf_n[2,nmax+n,elb1,elb2] = np.mean(hfield[2]*hns[elb2,1])
            threepcf_n[3,nmax+n,elb2,elb1] = np.mean(hfield[3]*hns[elb2,0])
            threepcf_n_norm[nmax+n,elb2,elb1] = np.mean(wnorm*norms[elb2])
            if n!= 0:
                threepcf_n[0,nmax-n,elb2,elb1] = threepcf_n[0,nmax+n,elb1,elb2]
                threepcf_n[1,nmax-n,elb2,elb1] = threepcf_n[1,nmax+n,elb1,elb2]
                threepcf_n[2,nmax-n,elb1,elb2] = threepcf_n[3,nmax+n,elb2,elb1]
                threepcf_n[3,nmax-n,elb2,elb1] = threepcf_n[2,nmax+n,elb1,elb2]
                threepcf_n_norm[nmax-n,elb1,elb2] = threepcf_n_norm[nmax+n,elb2,elb1] 
    */
    
    // Threepcf has shape (2, 4, nbinsz, nbinsz, nbinsz, nbinsr, nbinsr)
    #pragma omp parallel for num_threads(nthreads)
    for (int elb1=0; elb1<nbinsr; elb1++){
        for (int zbin2=0; zbin2<nbinsz; zbin2++){
            double complex h0, h1, h2, h3, w0;
            int gammashift1, gammashift2, gammashift, g1shift, g2shift;
            int elb2, zbin1, zbin3;
            for (int ind_gal=0; ind_gal<ngal; ind_gal++){
                double complex shape = (e1[ind_gal]+I*e2[ind_gal]);
                zbin1 = zbins[ind_gal];
                gammashift1 = zbin1*gammaax1;
                gammashift2 = zbin2*gammaax2;
                g1shift = ind_gal*nbinsz*nbinsr+zbin2*nbinsr+elb1;
                h0 = -shape * Gns[0*indshift3 + g1shift];
                h1 = -conj(shape) * Gns[2*indshift3 + g1shift];
                h2 = -shape * conj(Gns[3*indshift3 + g1shift]);
                h3 = -shape * conj(Gns[2*indshift3 + g1shift]);
                w0 = weight[ind_gal] * norms[g1shift];
                for (elb2=0; elb2<nbinsr; elb2++){
                    for (zbin3=0; zbin3<nbinsz; zbin3++){
                        g2shift = ind_gal*nbinsz*nbinsr+zbin3*nbinsr+elb2;
                        gammashift = gammashift1 + gammashift2 + zbin3*gammaax3 + elb1*nbinsr + elb2;
                        threepcf[0*gammaax0 + gammashift] += h0*Gns[1*indshift3 + g2shift];
                        threepcf[1*gammaax0 + gammashift] += h1*Gns[3*indshift3 + g2shift];
                        threepcf[2*gammaax0 + gammashift] += h2*Gns[1*indshift3 + g2shift];
                        threepcf[3*gammaax0 + gammashift] += h3*Gns[0*indshift3 + g2shift];
                        threepcf_n[gammashift] += w0*norms[g2shift];
                    }
                }
            }
        }
    }
    
    free(Gns);
    free(norms);
}

