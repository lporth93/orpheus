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
#include "covariance_second.h"

#define mymin(x,y) ((x) <= (y)) ? (x) : (y)
#define mymax(x,y) ((x) >= (y)) ? (x) : (y)
#define M_PI      3.14159265358979323846
#define INV_2PI   0.15915494309189534561

void alloc_triplets_tree_xipxipcov(
    int *isinner, double *weight, double *pos1, double *pos2, int *zbins, int nbinsz, int ngal, 
    int nresos, double *reso_redges, int *ngal_resos, 
    double *weight_resos, double *pos1_resos, double *pos2_resos, int *zbin_resos,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int nmin, int nmax, double rmin, double rmax, double *rbins, int nbinsr, int dccorr, 
    int nthreads, double *bin_centers, 
    double *wwcounts, double *w2wcounts, double complex *w2wwcounts, double complex *wwwcounts){
    
    // Index shift for the Gamman
    int _triplet_zshift = nbinsr*nbinsr;
    int _triplet_nshift = _triplet_zshift*nbinsz*nbinsz*nbinsz;
    int _triplet_compshift = (nmax-nmin+1)*_triplet_nshift;
    
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
        double *tmpwwcounts = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
        double *tmpw2wcounts = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
        double complex *tmpw2wwcounts = calloc(nthreads*_triplet_compshift, sizeof(double complex));
        double complex *tmpwwwcounts = calloc(nthreads*_triplet_compshift, sizeof(double complex));
        #pragma omp parallel for num_threads(nthreads)
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            int gamma_zshift = nbinsr*nbinsr;
            int gamma_nshift = _triplet_zshift*nbinsz*nbinsz*nbinsz;
            int gamma_compshift = (nmax-nmin+1)*_triplet_nshift;
            
            int npix_hash = pix1_n*pix2_n;
            int *rshift_index_matcher = calloc(nresos, sizeof(int));
            int *rshift_pixs_galind_bounds = calloc(nresos, sizeof(int));
            int *rshift_pix_gals = calloc(nresos, sizeof(int));
            double *reso_redges2 = calloc(nresos+1, sizeof(double));
            reso_redges2[0] = reso_redges[0]*reso_redges[0];
            for (int elreso=1;elreso<nresos;elreso++){
                rshift_index_matcher[elreso] = rshift_index_matcher[elreso-1] + npix_hash;
                rshift_pixs_galind_bounds[elreso] = rshift_pixs_galind_bounds[elreso-1] + ngal_resos[elreso-1]+1;
                rshift_pix_gals[elreso] = rshift_pix_gals[elreso-1] + ngal_resos[elreso-1];
                reso_redges2[1+elreso] = reso_redges[1+elreso]*reso_redges[1+elreso];
            }
                
            for (int ind_gal=0; ind_gal<ngal; ind_gal++){
                // Check if galaxy falls in stripe used in this process
                double p11, p12, w1;
                int zbin1, innergal;
                #pragma omp critical
                {p11 = pos1[ind_gal];
                p12 = pos2[ind_gal];
                w1 = weight[ind_gal];
                zbin1 = zbins[ind_gal];
                innergal = isinner[ind_gal];}
                if (innergal==0){continue;}
                int thisstripe = 2*thisthread + odd;
                int galstripe = (int) floor((p11-pix1_start)/pix1_d * (2*nthreads)/pix1_n);
                if (thisstripe != galstripe){continue;}
                
                int ind_pix1, ind_pix2, ind_inpix, ind_gal2;
                int ind_red, lower, upper; 
                double  p21, p22, w2, z2;
                double rel1, rel2, dist, dist2;
                int nnvals, nextn;
                double complex nphirot, phirot;
                
                // Check how many ns we need for Gn
                // Gns have shape (nnvals, nbinsz, nbinsr)
                nnvals = nmax+1;
                double complex *nextGns =  calloc(nnvals*nbinsr*nbinsz, sizeof(double complex));
                double complex *nextG2ns =  calloc(nbinsz*nbinsr, sizeof(double complex));

                int ind_rbin, ind_wwbin, rbin;
                int zrshift, normzrshift, _normzrshift;
                int nbinszr = nbinsz*nbinsr;
                int nbinsz2r = nbinsz*nbinsz*nbinsr;
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
                                rel1 = p21 - p11;
                                rel2 = p22 - p12;
                                dist2 = rel1*rel1 + rel2*rel2;
                                dist = sqrt(dist2);
                                if(dist < reso_redges[elreso] || dist >= reso_redges[elreso+1]){continue;}
                                //dist = sqrt(dist2);
                                if (rbins[0] < 0){
                                    rbin = (int) floor(log(dist/rmin)/drbin);
                                }
                                else{
                                    rbin=0;
                                    while(rbins[rbin+1] < dist){rbin+=1;}
                                }
                                
                                
                                phirot = (rel1+I*rel2)/dist;// * fabs(rel1)/rel1;
                                //if (rel1<0){phirot*=-1;}
                                //phirotc = conj(phirot);
                                //twophirotc = phirotc*phirotc;
                                //double dphi = atan2(rel2,rel1);
                                //phirot = cexp(I*dphi);
                                zrshift = z2*nbinsr + rbin;
                                ind_rbin = thisthread*nbinszr + zrshift;
                                ind_wwbin = thisthread*nbinsz2r+zbin1*nbinszr+zrshift;
                                nphirot = 1+I*0;
                                tmpwcounts[ind_rbin] += w1*w2*dist; 
                                tmpwnorms[ind_rbin] += w1*w2; 
                                tmpwwcounts[ind_wwbin] += w1*w2; 
                                tmpw2wcounts[ind_wwbin] += w1*w1*w2; 
                                nextGns[zrshift] += w2*nphirot;  
                                nextG2ns[zrshift] += w2*w2;
                                nphirot *= phirot;
                                
                                for (nextn=1;nextn<nmax+1;nextn++){
                                    nextGns[zrshift+nextn*nbinszr] += w2*nphirot;  
                                    nphirot *= phirot;
                                }
                            }
                        }
                    }
                }
                
                // Now update the Gammans
                // tmpGammas have shape (nthreads, nmax+1, nzcombis3, r*r, 4)
                // Gns have shape (nnvals, nbinsz, nbinsr)
                //int nonzero_tmpGammas = 0;
                double complex wsq, w1_sq, w0;
                int thisnshift, r12shift;
                int gammashift1, gammashift;
                int ind_norm;
                int thisn, elb1, elb2, zbin2, zbin3, zcombi;
                w1_sq = w1*w1;
                for (thisn=0; thisn<nmax-nmin+1; thisn++){
                    ind_norm = thisn*nbinszr;
                    thisnshift = thisthread*gamma_compshift + thisn*gamma_nshift;
                    for (zbin2=0; zbin2<nbinsz; zbin2++){
                        for (elb1=0; elb1<nbinsr; elb1++){
                            normzrshift = ind_norm + zbin2*nbinsr + elb1;
                            wsq = w1_sq * nextGns[normzrshift];
                            w0 = w1 * nextGns[normzrshift];
                            for (zbin3=0; zbin3<nbinsz; zbin3++){
                                zcombi = zbin1*nbinsz*nbinsz+zbin2*nbinsz+zbin3;
                                gammashift1 = thisnshift + zcombi*gamma_zshift;
                                // Double counting correction
                                if (zbin1==zbin2 && zbin1==zbin3 && dccorr==1){
                                    zrshift = zbin2*nbinsr + elb1;
                                    r12shift = elb1*nbinsr+elb1;
                                    gammashift = gammashift1 + r12shift;
                                    tmpw2wwcounts[gammashift] -= w1_sq*nextG2ns[zrshift];
                                    tmpwwwcounts[gammashift] -= w1*nextG2ns[zrshift];
                                }
                                // Nominal allocation
                                _normzrshift = ind_norm+zbin3*nbinsr;
                                for (elb2=0; elb2<nbinsr; elb2++){
                                    normzrshift = _normzrshift + elb2;
                                    gammashift = gammashift1 + elb1*nbinsr+elb2;
                                    //phirotm = h0*nextGns[ind_mnm3 + zrshift];
                                    tmpw2wwcounts[gammashift] += wsq*conj(nextGns[normzrshift]);
                                    tmpwwwcounts[gammashift] += w0*conj(nextGns[normzrshift]);
                                    //if(thisthread==0 && ind_gal%1000==0){
                                    //    if (cabs(tmpGammans[gammashift] )>1e-5){nonzero_tmpGammas += 1;}
                                    //}
                                }
                            }
                        }
                    }
                }
                
                free(nextGns);
                free(nextG2ns);
                nextGns = NULL;
                nextG2ns = NULL;
            }
            
            free(rshift_index_matcher);
            free(rshift_pixs_galind_bounds);
            free(rshift_pix_gals);
            free(reso_redges2);
        }
        
        // Accumulate the Gamman
        #pragma omp parallel for num_threads(nthreads)
        for (int thisn=0; thisn<nmax-nmin+1; thisn++){
            int itmpGamma, iGamma, _iGamma;
            for (int thisthread=0; thisthread<nthreads; thisthread++){
                for (int zcombi=0; zcombi<nbinsz*nbinsz*nbinsz; zcombi++){
                    for (int elb1=0; elb1<nbinsr; elb1++){
                        _iGamma = thisn*_triplet_nshift + zcombi*_triplet_zshift + elb1*nbinsr;
                        for (int elb2=0; elb2<nbinsr; elb2++){
                            iGamma = _iGamma + elb2;
                            itmpGamma = iGamma + thisthread*_triplet_compshift;
                            w2wwcounts[iGamma] += tmpw2wwcounts[itmpGamma];
                            wwwcounts[iGamma] += tmpwwwcounts[itmpGamma];
                        }
                    }
                }
            }
        }
        
        
        // Accumulate the paircounts
        int threadshift, zzrind;
        int nbinszr = nbinsz*nbinsr;
        int nbinsz2r = nbinsz*nbinsz*nbinsr;
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            threadshift = thisthread*nbinsz2r;
            for (int elbinz1=0; elbinz1<nbinsz; elbinz1++){
                for (int elbinz2=0; elbinz2<nbinsz; elbinz2++){
                    for (int elbinr=0; elbinr<nbinsr; elbinr++){
                        zzrind = elbinz1*nbinszr+elbinz2*nbinsr+elbinr;
                        wwcounts[zzrind] += tmpwwcounts[threadshift+zzrind];
                        w2wcounts[zzrind] += tmpw2wcounts[threadshift+zzrind];
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
        free(tmpwwcounts);
        free(tmpw2wcounts);
        free(tmpw2wwcounts);
        free(tmpwwwcounts); 
        tmpwcounts = NULL;
        tmpwnorms = NULL;
        tmpwwcounts = NULL;
        tmpw2wcounts = NULL;
        tmpw2wwcounts = NULL;
        tmpwwwcounts = NULL;
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


void alloc_triplets_doubletree_xipxipcov(
    int nresos, int nresos_grid, double *dpix1_resos,  double *dpix2_resos, double *reso_redges, int *ngal_resos, int nbinsz,
    int *isinner_resos, double *weight_resos, double *weight_sq_resos, double *pos1_resos, double *pos2_resos, int *zbin_resos,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, int nregions, int *index_matcher_hash,
    int nmax, double rmin, double rmax, int nbinsr, int dccorr, 
    int nthreads, double *bin_centers, 
    double *wwcounts, double *w2wcounts, double complex *w2wwcounts, double complex *wwwcounts){
    
    // Index shift for the triplets
    int _triplets_zshift = nbinsr*nbinsr;
    int _triplets_nshift = _triplets_zshift*nbinsz*nbinsz*nbinsz;
    int _triplets_compshift = (nmax+1)*_triplets_nshift;
    
    double *totcounts = calloc(nbinsz*nbinsr, sizeof(double));
    double *totnorms = calloc(nbinsz*nbinsr, sizeof(double));
    
    // Temporary arrays that are allocated in parallel and later reduced
    double *tmpwcounts = calloc(nthreads*nbinsz*nbinsr, sizeof(double));
    double *tmpwnorms = calloc(nthreads*nbinsz*nbinsr, sizeof(double));
    double *tmpwwcounts = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
    double *tmpw2wcounts = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
    double complex *tmpw2wwcounts = calloc(nthreads*_triplets_compshift, sizeof(double complex));
    double complex *tmpwwwcounts = calloc(nthreads*_triplets_compshift, sizeof(double complex));
        
    
    #pragma omp parallel for num_threads(nthreads)
    for(int elthread=0;elthread<nthreads;elthread++){
        int nregions_per_thread = nregions/nthreads;
        int hasdiscrete = nresos-nresos_grid;
        int nnvals_Nn = nmax+1;        
        
        // Initialize the caches for a region
        // Upper bound for possible nshift: each zbin does completely fill out the lowest reso grid.
        // The remaining grids then have 1/4 + 1/16 + ... --> 0.33.... times the data of the largest grid. 
        // TODO: Make this more memory-efficient by doing a dry run in which we infer each size_max_nshift
        //       per thread?
        int size_max_nshift = (int) ((1+hasdiscrete+0.34)*nbinsz*nbinsr*pow(4,nresos_grid));
        double complex *Nncache = calloc(nnvals_Nn*size_max_nshift, sizeof(double complex));
        double complex *wNncache = calloc(nnvals_Nn*size_max_nshift, sizeof(double complex));
        double complex *w2Nncache = calloc(nnvals_Nn*size_max_nshift, sizeof(double complex));
        int *Nncache_updates = calloc(size_max_nshift, sizeof(int));
        for (int elregion=0; elregion<nregions; elregion++){
            
            int region_debug=16;
            // Check if this thread is responsible for the region
            int nthread_target = mymin(elregion/nregions_per_thread, nthreads-1);
            if (nthread_target!=elthread){continue;}
            // printf("Region %d is in thread %d\n",elregion,elthread);
            if (elregion==region_debug){printf("Region %d is in thread %d\n",elregion,elthread);}
            
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
            int triplets_zshift = nbinsr*nbinsr;
            int triplets_nshift = triplets_zshift*nbinsz*nbinsz*nbinsz;
            int triplets_compshift = (nmax+1)*triplets_nshift;
            
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
                    pix2redpix[zbin_gal*len_matcher+matchers_resoshift[elreso]+elhashpix] = tmpcounts[zbin_gal];
                    tmpcounts[zbin_gal] += 1;
                    if (elregion==region_debug){
                        printf("elreso=%d, lower=%d, thispix=%d, zgal=%d: pix2redpix[%d]=%d  \n",
                               elreso,lower,ind_inpix,zbin_gal,zbin_gal*len_matcher+matchers_resoshift[elreso]+elhashpix,ind_inpix-lower);
                    }
                }
                free(tmpcounts);
            }
            
            // Setup all shift variables for the Gncache in the region
            // Nncache has structure
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
                        cumresoshift_z[elz*(nresos+1) + elreso+1] = cumresoshift_z[elz*(nresos+1) + elreso] +
                            ngal_in_pix[elz*nresos + elreso];
                    }
                    if (elregion==region_debug){printf("  cumresoshift_z[z][reso+1]=%d: \n", 
                                                       cumresoshift_z[elz*(nresos+1) + elreso+1]);}
                }
                thetashifts_z[elz] = cumresoshift_z[elz*(nresos+1) + nresos];
                zbinshifts[elz+1] = zbinshifts[elz] + nbinsr*thetashifts_z[elz];
                if ((elregion==region_debug)){printf("thetashifts_z[z]=%d: \nzbinshifts[z+1]=%d: \n", 
                                                     thetashifts_z[elz],  zbinshifts[elz+1]);}
            }
            zbin2shift = zbinshifts[nbinsz];
            nshift = nbinsz*zbin2shift;
            // Set all the cache indeces that are updated in this region to zero
            if ((elregion==region_debug)){printf("zbin2shift=%d: nshift=%d: \n", zbin2shift,  nshift);}
            for (int _i=0; _i<nnvals_Nn*nshift; _i++){ Nncache[_i] = 0; wNncache[_i] = 0; w2Nncache[_i] = 0;}
            for (int _i=0; _i<nshift; _i++){ Nncache_updates[_i] = 0;}
            int Nncache_totupdates=0;
            
            // Now, for each resolution, loop over all the galaxies in the region and
            // allocate the Nn, as well as their caches  for the corresponding 
            // set of radii
            // For elreso in resos
            //.  for gal in reso 
            //.    allocate Nn for allowed radii
            //.    allocate the Nncaches
            //.    compute the triplets for all combinations of the same resolution
            int ind_pix1, ind_pix2, ind_inpix1, ind_inpix2, ind_red, ind_gal1, ind_gal2, z_gal1, z_gal2;
            int ind_Nn, ind_Nncacheshift;
            int innergal, rbin, nextn, nbinsr_reso, nbinszr, nbinszr_reso, nbinsz2r, zrshift, ind_rbin, ind_wwbin;
            double pos1_gal1, pos2_gal1, pos1_gal2, pos2_gal2, w_gal1, w_gal1_sq, w_gal2;
            double rel1, rel2, dist;
            double complex nphirot, phirot;
            double rmin_reso, rmax_reso;
            int rbinmin, rbinmax, rbinmin1, rbinmax1, rbinmin2, rbinmax2;
            nbinszr =  nbinsz*nbinsr;
            nbinsz2r = nbinsz*nbinsz*nbinsr;
            for (int elreso=0;elreso<nresos;elreso++){
                int elreso_leaf = elreso;
                rbinmin = reso_rindedges[elreso];
                rbinmax = reso_rindedges[elreso+1];
                rmin_reso = rmin*exp(rbinmin*drbin);
                rmax_reso = rmin*exp(rbinmax*drbin);
                nbinsr_reso = rbinmax-rbinmin;
                nbinszr_reso = nbinsz*nbinsr_reso;
                lower1 = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion];
                upper1 = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion+1];
                double complex *nextNns =  calloc(nnvals_Nn*nbinszr_reso, sizeof(double complex));
                double complex *nextN2ns =  calloc(nbinszr_reso, sizeof(double complex));
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
                    w_gal1_sq = weight_sq_resos[ind_gal1];
                    
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
                                z_gal2 = zbin_resos[ind_gal2];
                                
                                rel1 = pos1_gal2 - pos1_gal1;
                                rel2 = pos2_gal2 - pos2_gal1;
                                dist = sqrt(rel1*rel1 + rel2*rel2);
                                if(dist < rmin_reso || dist >= rmax_reso) continue;
                                rbin = (int) floor((log(dist)-logrmin)/drbin) - rbinmin;
                                
                                phirot = (rel1+I*rel2)/dist;
                                zrshift = z_gal2*nbinsr_reso + rbin;
                                ind_rbin = elthread*nbinszr + z_gal2*nbinsr + rbin+rbinmin;
                                ind_wwbin = elthread*nbinsz2r+z_gal1*nbinszr+ z_gal2*nbinsr + rbin+rbinmin;
                                                                
                                // n = 0
                                ind_Nn = zrshift;
                                nphirot = 1+I*0;
                                nextncounts[zrshift] += 1;
                                tmpwcounts[ind_rbin] += w_gal1*w_gal2*dist; 
                                tmpwnorms[ind_rbin] += w_gal1*w_gal2; 
                                tmpwwcounts[ind_wwbin] += w_gal1*w_gal2; 
                                tmpw2wcounts[ind_wwbin] += w_gal1_sq*w_gal2; 
                                nextNns[ind_Nn] += w_gal2*nphirot;  
                                nextN2ns[zrshift] += w_gal2*w_gal2;
                                nphirot *= phirot;
                                ind_Nn += nbinszr_reso;
                                // n in [1, ..., nmax-1] x {+1,-1}
                                for (nextn=1;nextn<=nmax;nextn++){
                                    nextNns[ind_Nn] += w_gal2*nphirot;  
                                    nphirot *= phirot;
                                    ind_Nn += nbinszr_reso;
                                }
                            }
                        }
                    }
                        
                    // Update the Gncache and Gnnormcache
                    int red_reso2, npix_side_reso2, elhashpix_1_reso2, elhashpix_2_reso2, elhashpix_reso2, redpix_reso2;
                    double complex thisNn;
                    int _tmpindNn;
                    for (int elreso2=elreso; elreso2<nresos; elreso2++){
                        red_reso2 = elreso2 - hasdiscrete;
                        if (hasdiscrete==1 && elreso==0 && elreso2==0){red_reso2 += hasdiscrete;}
                        npix_side_reso2 = 1 << (nresos_grid-red_reso2-1);
                        elhashpix_1_reso2 = (int) floor((pos1_gal1 - hashpix_start1)/dpix1_resos[red_reso2]);
                        elhashpix_2_reso2 = (int) floor((pos2_gal1 - hashpix_start2)/dpix2_resos[red_reso2]);
                        elhashpix_reso2 = elhashpix_2_reso2*npix_side_reso2 + elhashpix_1_reso2;
                        redpix_reso2 = pix2redpix[z_gal1*len_matcher+matchers_resoshift[red_reso2]+elhashpix_reso2];
                        for (int zbin2=0; zbin2<nbinsz; zbin2++){
                            for (int thisrbin=rbinmin; thisrbin<rbinmax; thisrbin++){
                                zrshift = zbin2*nbinsr_reso + thisrbin-rbinmin;
                                if (cabs(nextNns[zrshift])<1e-10){continue;}
                                ind_Nncacheshift = zbin2*zbin2shift + zbinshifts[z_gal1] + thisrbin*thetashifts_z[z_gal1] + 
                                    cumresoshift_z[z_gal1*(nresos+1) + elreso2] + redpix_reso2;
                                Nncache_updates[ind_Nncacheshift] += 1;
                                Nncache_totupdates += 1;
                                _tmpindNn = zrshift;
                                for(int thisn=0; thisn<nnvals_Nn; thisn++){
                                    thisNn = nextNns[_tmpindNn];
                                    Nncache[ind_Nncacheshift] += thisNn;
                                    wNncache[ind_Nncacheshift] += w_gal1*thisNn;
                                    w2Nncache[ind_Nncacheshift] += w_gal1_sq*thisNn;
                                    _tmpindNn += nbinszr_reso;
                                    ind_Nncacheshift += nshift;
                                }
                            }
                            
                        } 
                    }
                    
                    // Allocate same reso triplets
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
                    // Now update the triplets
                    // tmpwxwwcounts have shape (nthreads, nmax+1, nzcombis3, r*r)
                    // Nns have shape (nnvals, nbinsz, nbinsr)
                    double complex w0, w2, cNn;
                    int thisnshift;
                    int _tripletsshift1, tripletsshift1;
                    int ind_norm;
                    int _zcombi, zcombi, elb1_full, elb2_full;
                    for (int thisn=0; thisn<nmax+1; thisn++){
                        ind_norm = thisn*nbinszr_reso;
                        thisnshift = elthread*triplets_compshift + thisn*triplets_nshift;
                        int elb1, zbin2;
                        for (int zrcombis1=0; zrcombis1<nallowedcounts; zrcombis1++){
                            elb1 = allowedrinds[zrcombis1];
                            zbin2 = allowedzinds[zrcombis1];
                            elb1_full = elb1 + rbinmin;
                            zrshift = zbin2*nbinsr_reso + elb1;
                            // Double counting correction
                            if (dccorr==1){
                                zcombi = z_gal1*nbinsz*nbinsz + zbin2*nbinsz + zbin2;
                                tripletsshift1 = thisnshift + zcombi*triplets_zshift + elb1_full*nbinsr;
                                tmpw2wwcounts[tripletsshift1 + elb1_full] -= w_gal1_sq*nextN2ns[zrshift];
                                tmpwwwcounts[tripletsshift1 + elb1_full] -= w_gal1*nextN2ns[zrshift];
                            }
                            w0 = w_gal1 * nextNns[ind_norm + zrshift];
                            w2 = w_gal1_sq * nextNns[ind_norm + zrshift];
                            _zcombi = z_gal1*nbinsz*nbinsz+zbin2*nbinsz;
                            _tripletsshift1 = thisnshift + elb1_full*nbinsr;
                            for (int zrcombis2=0; zrcombis2<nallowedcounts; zrcombis2++){
                                zcombi = _zcombi+allowedzinds[zrcombis2];
                                tripletsshift1 = _tripletsshift1 + zcombi*triplets_zshift; 
                                elb2_full = allowedrinds[zrcombis2] + rbinmin;
                                zrshift = allowedzinds[zrcombis2]*nbinsr_reso + allowedrinds[zrcombis2];
                                cNn = conj(nextNns[ind_norm + zrshift]);
                                tmpw2wwcounts[tripletsshift1 + elb2_full] += w2*cNn;
                                tmpwwwcounts[tripletsshift1 + elb2_full] += w0*cNn;
                            }
                        }
                    }
                    for (int _i=0;_i<nnvals_Nn*nbinszr_reso;_i++){nextNns[_i]=0;}
                    for (int _i=0;_i<nbinszr_reso;_i++){nextN2ns[_i]=0; nextncounts[_i]=0; 
                                                        allowedrinds[_i]=0; allowedzinds[_i]=0;}
                }
                free(nextNns);
                free(nextN2ns);
                free(nextncounts);
                free(allowedrinds);
                free(allowedzinds);
            }
                        
            // Allocate the triplets for different grid resolutions from all the cached arrays 
            //
            // Note that for different configurations of the resolutions we do the triplets
            // allocation as follows - see eq. (32) in 2309.08601 for the reasoning:
            // * w2ww = w^2 * N_n * N_mn
            //          --> (w2N_n) * conj(N_n) if reso1 < reso2
            //          --> N_n * conj(w2N_n)   if reso1 > reso2
            // where w2N_n := w^2*N_n
            double complex w0, w2;
            int thisnshift;
            int tripletsshift1;
            int zcombi;
            for (int thisn=0; thisn<nmax+1; thisn++){
                thisnshift = elthread*triplets_compshift + thisn*triplets_nshift;
                for (int zbin1=0; zbin1<nbinsz; zbin1++){
                    for (int zbin2=0; zbin2<nbinsz; zbin2++){
                        for (int zbin3=0; zbin3<nbinsz; zbin3++){
                            zcombi = zbin1*nbinsz*nbinsz + zbin2*nbinsz + zbin3;
                            int _thetashift_z = thetashifts_z[zbin1];
                            // Case max(reso1, reso2) = reso2
                            for (int thisreso1=0; thisreso1<nresos; thisreso1++){
                                rbinmin1 = reso_rindedges[thisreso1];
                                rbinmax1 = reso_rindedges[thisreso1+1];
                                for (int thisreso2=thisreso1+1; thisreso2<nresos; thisreso2++){
                                    rbinmin2 = reso_rindedges[thisreso2];
                                    rbinmax2 = reso_rindedges[thisreso2+1];
                                    for (int elgal=0; elgal<ngal_in_pix[zbin1*nresos+thisreso2]; elgal++){
                                        for (int elb1=rbinmin1; elb1<rbinmax1; elb1++){
                                            tripletsshift1 = thisnshift + zcombi*triplets_zshift + elb1*nbinsr;
                                            // n --> zbin2 --> zbin1 --> radius --> 
                                            //   --> [ [0]*ngal_zbin1_reso1 | ... | [0]*ngal_zbin1_reson ]
                                            ind_Nncacheshift = zbin2*zbin2shift + zbinshifts[zbin1] + 
                                                elb1*thetashifts_z[zbin1] +
                                                cumresoshift_z[zbin1*(nresos+1) + thisreso2] + elgal;
                                            w0 = wNncache[thisn*nshift + ind_Nncacheshift];
                                            w2 = w2Nncache[thisn*nshift + ind_Nncacheshift];
                                            ind_Nncacheshift = zbin3*zbin2shift + zbinshifts[zbin1] +
                                                rbinmin2*thetashifts_z[zbin1] + thisn*nshift +
                                                cumresoshift_z[zbin1*(nresos+1) + thisreso2] + elgal;
                                            for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                                tmpw2wwcounts[tripletsshift1 + elb2] += w2*conj(Nncache[ind_Nncacheshift]);
                                                tmpwwwcounts[tripletsshift1 + elb2] += w0*conj(Nncache[ind_Nncacheshift]);
                                                ind_Nncacheshift += _thetashift_z;
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
                                            tripletsshift1 = thisnshift + zcombi*triplets_zshift + elb1*nbinsr;
                                            ind_Nncacheshift = zbin2*zbin2shift + zbinshifts[zbin1] + 
                                                elb1*thetashifts_z[zbin1] +
                                                cumresoshift_z[zbin1*(nresos+1) + thisreso1] + elgal;
                                            w0 = Nncache[thisn*nshift + ind_Nncacheshift];
                                            ind_Nncacheshift = zbin3*zbin2shift + zbinshifts[zbin1] +
                                                rbinmin2*thetashifts_z[zbin1] + thisn*nshift + 
                                                cumresoshift_z[zbin1*(nresos+1) + thisreso1] + elgal;
                                            for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                                tmpw2wwcounts[tripletsshift1 + elb2] += w0*conj(w2Nncache[ind_Nncacheshift]);
                                                tmpwwwcounts[tripletsshift1 + elb2] += w0*conj(wNncache[ind_Nncacheshift]);
                                                ind_Nncacheshift += _thetashift_z;
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
        free(Nncache);
        free(wNncache);
        free(w2Nncache);
        free(Nncache_updates);
    }
    
    // Accumulate the triplets
    #pragma omp parallel for num_threads(nthreads)
    for (int thisn=0; thisn<nmax+1; thisn++){
        int itmptriplet, itriplet;
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            for (int zcombi=0; zcombi<nbinsz*nbinsz*nbinsz; zcombi++){
                for (int elb1=0; elb1<nbinsr; elb1++){
                    for (int elb2=0; elb2<nbinsr; elb2++){
                        itriplet = thisn*_triplets_nshift + zcombi*_triplets_zshift + elb1*nbinsr + elb2;
                        itmptriplet = itriplet + thisthread*_triplets_compshift;
                        w2wwcounts[itriplet] += tmpw2wwcounts[itmptriplet];
                        wwwcounts[itriplet] += tmpwwwcounts[itmptriplet];
                    }
                }
            }
        }
    }
    
    // Accumulate the paircounts
    int threadshift, zzrind;
    int nbinszr = nbinsz*nbinsr;
    int nbinsz2r = nbinsz*nbinsz*nbinsr;
    for (int thisthread=0; thisthread<nthreads; thisthread++){
        threadshift = thisthread*nbinsz2r;
        for (int elbinz1=0; elbinz1<nbinsz; elbinz1++){
            for (int elbinz2=0; elbinz2<nbinsz; elbinz2++){
                for (int elbinr=0; elbinr<nbinsr; elbinr++){
                    zzrind = elbinz1*nbinszr+elbinz2*nbinsr+elbinr;
                    wwcounts[zzrind] += tmpwwcounts[threadshift+zzrind];
                    w2wcounts[zzrind] += tmpw2wcounts[threadshift+zzrind];
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
    free(tmpw2wwcounts);
    free(tmpwwwcounts);
    free(totcounts);
    free(totnorms);
}


void alloc_triplets_basetree_xipxipcov(
    int nresos, int nresos_grid, double *dpix1_resos,  double *dpix2_resos, double *reso_redges, int *ngal_resos, int nbinsz,
    int *isinner_resos, double *weight_resos, double *weight_sq_resos, double *pos1_resos, double *pos2_resos, int *zbin_resos,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, int nregions, int *index_matcher_hash,
    int nmax, double rmin, double rmax, int nbinsr, int dccorr, 
    int nthreads, double *bin_centers, 
    double *wwcounts, double *w2wcounts, double complex *w2wwcounts, double complex *wwwcounts){
    
    // Index shift for the triplets
    int _triplets_zshift = nbinsr*nbinsr;
    int _triplets_nshift = _triplets_zshift*nbinsz*nbinsz*nbinsz;
    int _triplets_compshift = (nmax+1)*_triplets_nshift;
    
    double *totcounts = calloc(nbinsz*nbinsr, sizeof(double));
    double *totnorms = calloc(nbinsz*nbinsr, sizeof(double));
    
    // Temporary arrays that are allocated in parallel and later reduced
    double *tmpwcounts = calloc(nthreads*nbinsz*nbinsr, sizeof(double));
    double *tmpwnorms = calloc(nthreads*nbinsz*nbinsr, sizeof(double));
    double *tmpwwcounts = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
    double *tmpw2wcounts = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
    double complex *tmpw2wwcounts = calloc(nthreads*_triplets_compshift, sizeof(double complex));
    double complex *tmpwwwcounts = calloc(nthreads*_triplets_compshift, sizeof(double complex));
        
    
    #pragma omp parallel for num_threads(nthreads)
    for(int elthread=0;elthread<nthreads;elthread++){
        int nregions_per_thread = nregions/nthreads;
        int hasdiscrete = nresos-nresos_grid;
        int nnvals_Nn = nmax+1;        
        
        // Initialize the caches for a region
        // Upper bound for possible nshift: each zbin does completely fill out the lowest reso grid.
        // The remaining grids then have 1/4 + 1/16 + ... --> 0.33.... times the data of the largest grid. 
        // TODO: Make this more memory-efficient by doing a dry run in which we infer each size_max_nshift
        //       per thread?
        int size_max_nshift = (int) ((1+hasdiscrete+0.34)*nbinsz*nbinsr*pow(4,nresos_grid));
        double complex *Nncache = calloc(nnvals_Nn*size_max_nshift, sizeof(double complex));
        double complex *wNncache = calloc(nnvals_Nn*size_max_nshift, sizeof(double complex));
        double complex *w2Nncache = calloc(nnvals_Nn*size_max_nshift, sizeof(double complex));
        int *Nncache_updates = calloc(size_max_nshift, sizeof(int));
        for (int elregion=0; elregion<nregions; elregion++){
            
            int region_debug=16;
            // Check if this thread is responsible for the region
            int nthread_target = mymin(elregion/nregions_per_thread, nthreads-1);
            if (nthread_target!=elthread){continue;}
            // printf("Region %d is in thread %d\n",elregion,elthread);
            if (elregion==region_debug){printf("Region %d is in thread %d\n",elregion,elthread);}
            
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
            int triplets_zshift = nbinsr*nbinsr;
            int triplets_nshift = triplets_zshift*nbinsz*nbinsz*nbinsz;
            int triplets_compshift = (nmax+1)*triplets_nshift;
            
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
                    pix2redpix[zbin_gal*len_matcher+matchers_resoshift[elreso]+elhashpix] = tmpcounts[zbin_gal];
                    tmpcounts[zbin_gal] += 1;
                    if (elregion==region_debug){
                        printf("elreso=%d, lower=%d, thispix=%d, zgal=%d: pix2redpix[%d]=%d  \n",
                               elreso,lower,ind_inpix,zbin_gal,zbin_gal*len_matcher+matchers_resoshift[elreso]+elhashpix,ind_inpix-lower);
                    }
                }
                free(tmpcounts);
            }
            
            // Setup all shift variables for the Gncache in the region
            // Nncache has structure
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
                        cumresoshift_z[elz*(nresos+1) + elreso+1] = cumresoshift_z[elz*(nresos+1) + elreso] +
                            ngal_in_pix[elz*nresos + elreso];
                    }
                    if (elregion==region_debug){printf("  cumresoshift_z[z][reso+1]=%d: \n", 
                                                       cumresoshift_z[elz*(nresos+1) + elreso+1]);}
                }
                thetashifts_z[elz] = cumresoshift_z[elz*(nresos+1) + nresos];
                zbinshifts[elz+1] = zbinshifts[elz] + nbinsr*thetashifts_z[elz];
                if ((elregion==region_debug)){printf("thetashifts_z[z]=%d: \nzbinshifts[z+1]=%d: \n", 
                                                     thetashifts_z[elz],  zbinshifts[elz+1]);}
            }
            zbin2shift = zbinshifts[nbinsz];
            nshift = nbinsz*zbin2shift;
            // Set all the cache indeces that are updated in this region to zero
            if ((elregion==region_debug)){printf("zbin2shift=%d: nshift=%d: \n", zbin2shift,  nshift);}
            for (int _i=0; _i<nnvals_Nn*nshift; _i++){ Nncache[_i] = 0; wNncache[_i] = 0; w2Nncache[_i] = 0;}
            for (int _i=0; _i<nshift; _i++){ Nncache_updates[_i] = 0;}
            int Nncache_totupdates=0;
            
            // Now, for each resolution, loop over all the galaxies in the region and
            // allocate the Nn, as well as their caches  for the corresponding 
            // set of radii
            // For elreso in resos
            //.  for gal in reso 
            //.    allocate Nn for allowed radii
            //.    allocate the Nncaches
            //.    compute the triplets for all combinations of the same resolution
            int ind_pix1, ind_pix2, ind_inpix1, ind_inpix2, ind_red, ind_gal1, ind_gal2, z_gal1, z_gal2;
            int ind_Nn, ind_Nncacheshift;
            int innergal, rbin, nextn, nbinsr_reso, nbinszr, nbinszr_reso, nbinsz2r, zrshift, ind_rbin, ind_wwbin;
            double pos1_gal1, pos2_gal1, pos1_gal2, pos2_gal2, w_gal1, w_gal1_sq, w_gal2;
            double rel1, rel2, dist;
            double complex nphirot, phirot;
            double rmin_reso, rmax_reso;
            int rbinmin, rbinmax, rbinmin1, rbinmax1, rbinmin2, rbinmax2;
            nbinszr =  nbinsz*nbinsr;
            nbinsz2r = nbinsz*nbinsz*nbinsr;
            for (int elreso=0;elreso<nresos;elreso++){
                int elreso_leaf = 0;
                rbinmin = reso_rindedges[elreso];
                rbinmax = reso_rindedges[elreso+1];
                rmin_reso = rmin*exp(rbinmin*drbin);
                rmax_reso = rmin*exp(rbinmax*drbin);
                nbinsr_reso = rbinmax-rbinmin;
                nbinszr_reso = nbinsz*nbinsr_reso;
                lower1 = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion];
                upper1 = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion+1];
                double complex *nextNns =  calloc(nnvals_Nn*nbinszr_reso, sizeof(double complex));
                double complex *nextN2ns =  calloc(nbinszr_reso, sizeof(double complex));
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
                    w_gal1_sq = weight_sq_resos[ind_gal1];
                    
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
                                z_gal2 = zbin_resos[ind_gal2];
                                
                                rel1 = pos1_gal2 - pos1_gal1;
                                rel2 = pos2_gal2 - pos2_gal1;
                                dist = sqrt(rel1*rel1 + rel2*rel2);
                                if(dist < rmin_reso || dist >= rmax_reso) continue;
                                rbin = (int) floor((log(dist)-logrmin)/drbin) - rbinmin;
                                
                                phirot = (rel1+I*rel2)/dist;
                                zrshift = z_gal2*nbinsr_reso + rbin;
                                ind_rbin = elthread*nbinszr + z_gal2*nbinsr + rbin+rbinmin;
                                ind_wwbin = elthread*nbinsz2r+z_gal1*nbinszr+ z_gal2*nbinsr + rbin+rbinmin;
                                                                
                                // n = 0
                                ind_Nn = zrshift;
                                nphirot = 1+I*0;
                                nextncounts[zrshift] += 1;
                                tmpwcounts[ind_rbin] += w_gal1*w_gal2*dist; 
                                tmpwnorms[ind_rbin] += w_gal1*w_gal2; 
                                tmpwwcounts[ind_wwbin] += w_gal1*w_gal2; 
                                tmpw2wcounts[ind_wwbin] += w_gal1_sq*w_gal2; 
                                nextNns[ind_Nn] += w_gal2*nphirot;  
                                nextN2ns[zrshift] += w_gal2*w_gal2;
                                nphirot *= phirot;
                                ind_Nn += nbinszr_reso;
                                // n in [1, ..., nmax-1] x {+1,-1}
                                for (nextn=1;nextn<=nmax;nextn++){
                                    nextNns[ind_Nn] += w_gal2*nphirot;  
                                    nphirot *= phirot;
                                    ind_Nn += nbinszr_reso;
                                }
                            }
                        }
                    }
                        
                    // Update the Gncache and Gnnormcache
                    int red_reso2, npix_side_reso2, elhashpix_1_reso2, elhashpix_2_reso2, elhashpix_reso2, redpix_reso2;
                    double complex thisNn;
                    int _tmpindNn;
                    for (int elreso2=elreso; elreso2<nresos; elreso2++){
                        red_reso2 = elreso2 - hasdiscrete;
                        if (hasdiscrete==1 && elreso==0 && elreso2==0){red_reso2 += hasdiscrete;}
                        npix_side_reso2 = 1 << (nresos_grid-red_reso2-1);
                        elhashpix_1_reso2 = (int) floor((pos1_gal1 - hashpix_start1)/dpix1_resos[red_reso2]);
                        elhashpix_2_reso2 = (int) floor((pos2_gal1 - hashpix_start2)/dpix2_resos[red_reso2]);
                        elhashpix_reso2 = elhashpix_2_reso2*npix_side_reso2 + elhashpix_1_reso2;
                        redpix_reso2 = pix2redpix[z_gal1*len_matcher+matchers_resoshift[red_reso2]+elhashpix_reso2];
                        for (int zbin2=0; zbin2<nbinsz; zbin2++){
                            for (int thisrbin=rbinmin; thisrbin<rbinmax; thisrbin++){
                                zrshift = zbin2*nbinsr_reso + thisrbin-rbinmin;
                                if (cabs(nextNns[zrshift])<1e-10){continue;}
                                ind_Nncacheshift = zbin2*zbin2shift + zbinshifts[z_gal1] + thisrbin*thetashifts_z[z_gal1] + 
                                    cumresoshift_z[z_gal1*(nresos+1) + elreso2] + redpix_reso2;
                                Nncache_updates[ind_Nncacheshift] += 1;
                                Nncache_totupdates += 1;
                                _tmpindNn = zrshift;
                                for(int thisn=0; thisn<nnvals_Nn; thisn++){
                                    thisNn = nextNns[_tmpindNn];
                                    Nncache[ind_Nncacheshift] += thisNn;
                                    wNncache[ind_Nncacheshift] += w_gal1*thisNn;
                                    w2Nncache[ind_Nncacheshift] += w_gal1_sq*thisNn;
                                    _tmpindNn += nbinszr_reso;
                                    ind_Nncacheshift += nshift;
                                }
                            }
                            
                        } 
                    }
                    
                    // Allocate same reso triplets
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
                    // Now update the triplets
                    // tmpwxwwcounts have shape (nthreads, nmax+1, nzcombis3, r*r)
                    // Nns have shape (nnvals, nbinsz, nbinsr)
                    double complex w0, w2, cNn;
                    int thisnshift;
                    int _tripletsshift1, tripletsshift1;
                    int ind_norm;
                    int _zcombi, zcombi, elb1_full, elb2_full;
                    for (int thisn=0; thisn<nmax+1; thisn++){
                        ind_norm = thisn*nbinszr_reso;
                        thisnshift = elthread*triplets_compshift + thisn*triplets_nshift;
                        int elb1, zbin2;
                        for (int zrcombis1=0; zrcombis1<nallowedcounts; zrcombis1++){
                            elb1 = allowedrinds[zrcombis1];
                            zbin2 = allowedzinds[zrcombis1];
                            elb1_full = elb1 + rbinmin;
                            zrshift = zbin2*nbinsr_reso + elb1;
                            // Double counting correction
                            if (dccorr==1){
                                zcombi = z_gal1*nbinsz*nbinsz + zbin2*nbinsz + zbin2;
                                tripletsshift1 = thisnshift + zcombi*triplets_zshift + elb1_full*nbinsr;
                                tmpw2wwcounts[tripletsshift1 + elb1_full] -= w_gal1_sq*nextN2ns[zrshift];
                                tmpwwwcounts[tripletsshift1 + elb1_full] -= w_gal1*nextN2ns[zrshift];
                            }
                            w0 = w_gal1 * nextNns[ind_norm + zrshift];
                            w2 = w_gal1_sq * nextNns[ind_norm + zrshift];
                            _zcombi = z_gal1*nbinsz*nbinsz+zbin2*nbinsz;
                            _tripletsshift1 = thisnshift + elb1_full*nbinsr;
                            for (int zrcombis2=0; zrcombis2<nallowedcounts; zrcombis2++){
                                zcombi = _zcombi+allowedzinds[zrcombis2];
                                tripletsshift1 = _tripletsshift1 + zcombi*triplets_zshift; 
                                elb2_full = allowedrinds[zrcombis2] + rbinmin;
                                zrshift = allowedzinds[zrcombis2]*nbinsr_reso + allowedrinds[zrcombis2];
                                cNn = conj(nextNns[ind_norm + zrshift]);
                                tmpw2wwcounts[tripletsshift1 + elb2_full] += w2*cNn;
                                tmpwwwcounts[tripletsshift1 + elb2_full] += w0*cNn;
                            }
                        }
                    }
                    for (int _i=0;_i<nnvals_Nn*nbinszr_reso;_i++){nextNns[_i]=0;}
                    for (int _i=0;_i<nbinszr_reso;_i++){nextN2ns[_i]=0; nextncounts[_i]=0; 
                                                        allowedrinds[_i]=0; allowedzinds[_i]=0;}
                }
                free(nextNns);
                free(nextN2ns);
                free(nextncounts);
                free(allowedrinds);
                free(allowedzinds);
            }
                        
            // Allocate the triplets for different grid resolutions from all the cached arrays 
            //
            // Note that for different configurations of the resolutions we do the triplets
            // allocation as follows - see eq. (32) in 2309.08601 for the reasoning:
            // * w2ww = w^2 * N_n * N_mn
            //          --> (w2N_n) * conj(N_n) if reso1 < reso2
            //          --> N_n * conj(w2N_n)   if reso1 > reso2
            // where w2N_n := w^2*N_n
            double complex w0, w2;
            int thisnshift;
            int tripletsshift1;
            int zcombi;
            for (int thisn=0; thisn<nmax+1; thisn++){
                thisnshift = elthread*triplets_compshift + thisn*triplets_nshift;
                for (int zbin1=0; zbin1<nbinsz; zbin1++){
                    for (int zbin2=0; zbin2<nbinsz; zbin2++){
                        for (int zbin3=0; zbin3<nbinsz; zbin3++){
                            zcombi = zbin1*nbinsz*nbinsz + zbin2*nbinsz + zbin3;
                            int _thetashift_z = thetashifts_z[zbin1];
                            // Case max(reso1, reso2) = reso2
                            for (int thisreso1=0; thisreso1<nresos; thisreso1++){
                                rbinmin1 = reso_rindedges[thisreso1];
                                rbinmax1 = reso_rindedges[thisreso1+1];
                                for (int thisreso2=thisreso1+1; thisreso2<nresos; thisreso2++){
                                    rbinmin2 = reso_rindedges[thisreso2];
                                    rbinmax2 = reso_rindedges[thisreso2+1];
                                    for (int elgal=0; elgal<ngal_in_pix[zbin1*nresos+thisreso2]; elgal++){
                                        for (int elb1=rbinmin1; elb1<rbinmax1; elb1++){
                                            tripletsshift1 = thisnshift + zcombi*triplets_zshift + elb1*nbinsr;
                                            // n --> zbin2 --> zbin1 --> radius --> 
                                            //   --> [ [0]*ngal_zbin1_reso1 | ... | [0]*ngal_zbin1_reson ]
                                            ind_Nncacheshift = zbin2*zbin2shift + zbinshifts[zbin1] + 
                                                elb1*thetashifts_z[zbin1] +
                                                cumresoshift_z[zbin1*(nresos+1) + thisreso2] + elgal;
                                            w0 = wNncache[thisn*nshift + ind_Nncacheshift];
                                            w2 = w2Nncache[thisn*nshift + ind_Nncacheshift];
                                            ind_Nncacheshift = zbin3*zbin2shift + zbinshifts[zbin1] +
                                                rbinmin2*thetashifts_z[zbin1] + thisn*nshift +
                                                cumresoshift_z[zbin1*(nresos+1) + thisreso2] + elgal;
                                            for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                                tmpw2wwcounts[tripletsshift1 + elb2] += w2*conj(Nncache[ind_Nncacheshift]);
                                                tmpwwwcounts[tripletsshift1 + elb2] += w0*conj(Nncache[ind_Nncacheshift]);
                                                ind_Nncacheshift += _thetashift_z;
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
                                            tripletsshift1 = thisnshift + zcombi*triplets_zshift + elb1*nbinsr;
                                            ind_Nncacheshift = zbin2*zbin2shift + zbinshifts[zbin1] + 
                                                elb1*thetashifts_z[zbin1] +
                                                cumresoshift_z[zbin1*(nresos+1) + thisreso1] + elgal;
                                            w0 = Nncache[thisn*nshift + ind_Nncacheshift];
                                            ind_Nncacheshift = zbin3*zbin2shift + zbinshifts[zbin1] +
                                                rbinmin2*thetashifts_z[zbin1] + thisn*nshift + 
                                                cumresoshift_z[zbin1*(nresos+1) + thisreso1] + elgal;
                                            for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                                tmpw2wwcounts[tripletsshift1 + elb2] += w0*conj(w2Nncache[ind_Nncacheshift]);
                                                tmpwwwcounts[tripletsshift1 + elb2] += w0*conj(wNncache[ind_Nncacheshift]);
                                                ind_Nncacheshift += _thetashift_z;
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
        free(Nncache);
        free(wNncache);
        free(w2Nncache);
        free(Nncache_updates);
    }
    
    // Accumulate the triplets
    #pragma omp parallel for num_threads(nthreads)
    for (int thisn=0; thisn<nmax+1; thisn++){
        int itmptriplet, itriplet;
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            for (int zcombi=0; zcombi<nbinsz*nbinsz*nbinsz; zcombi++){
                for (int elb1=0; elb1<nbinsr; elb1++){
                    for (int elb2=0; elb2<nbinsr; elb2++){
                        itriplet = thisn*_triplets_nshift + zcombi*_triplets_zshift + elb1*nbinsr + elb2;
                        itmptriplet = itriplet + thisthread*_triplets_compshift;
                        w2wwcounts[itriplet] += tmpw2wwcounts[itmptriplet];
                        wwwcounts[itriplet] += tmpwwwcounts[itmptriplet];
                    }
                }
            }
        }
    }
    
    // Accumulate the paircounts
    int threadshift, zzrind;
    int nbinszr = nbinsz*nbinsr;
    int nbinsz2r = nbinsz*nbinsz*nbinsr;
    for (int thisthread=0; thisthread<nthreads; thisthread++){
        threadshift = thisthread*nbinsz2r;
        for (int elbinz1=0; elbinz1<nbinsz; elbinz1++){
            for (int elbinz2=0; elbinz2<nbinsz; elbinz2++){
                for (int elbinr=0; elbinr<nbinsr; elbinr++){
                    zzrind = elbinz1*nbinszr+elbinz2*nbinsr+elbinr;
                    wwcounts[zzrind] += tmpwwcounts[threadshift+zzrind];
                    w2wcounts[zzrind] += tmpw2wcounts[threadshift+zzrind];
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
    free(tmpw2wwcounts);
    free(tmpwwwcounts);
    free(totcounts);
    free(totnorms);
}
