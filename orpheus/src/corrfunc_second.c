#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <complex.h>

#include "utils.h"
#include "spatialhash.h"
#include "assign.h"
#include "corrfunc_second.h"

#define mymin(x,y) ((x) <= (y)) ? (x) : (y)
#define mymax(x,y) ((x) >= (y)) ? (x) : (y)
#define M_PI      3.14159265358979323846
#define INV_2PI   0.15915494309189534561

////////////////////////////////////////////////
/// SECOND-ORDER SHEAR CORRELATION FUNCTIONS ///
////////////////////////////////////////////////
// Directly use with two cats as one would do for LS-like estimator
void alloc_NNcounts_doubletree(
    int nresos, int nresos_grid, double *dpix1_resos,  double *dpix2_resos, double *reso_redges, 
    int resoshift_leafs, int minresoind_leaf, int maxresoind_leaf,
    int *ngal_resos, int nbinsz, int *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos, 
    double *scalar_tracer, int *zbin_resos, 
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int nregions, int *index_matcher_hash,
    double rmin, double rmax, int nbinsr, int do_dc,
    int nthreads, double *bin_centers, double *counts, long long int *npair){
    
    double *totcount = calloc(nbinsz*nbinsz*nbinsr, sizeof(double));
        
    // Temporary arrays that are allocated in parallel and later reduced
    int *tmpnpair = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(int));
    double *tmpwcount = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
    double *tmpwnorm = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
    
    #pragma omp parallel for num_threads(nthreads)
    for(int elthread=0;elthread<nthreads;elthread++){
        int nregions_per_thread = nregions/nthreads;
        int hasdiscrete = nresos-nresos_grid;
        
        // Check which sets of radii are evaluated for each resolution
        int *reso_rindedges = calloc(nresos+1, sizeof(int));
        double *binedges = calloc(nbinsr+2, sizeof(double));
        double logrmin = log(rmin);
        double drbin = (log(rmax)-logrmin)/(nbinsr);
        int tmpreso = 0;
        double thisredge = 0;
        double tmpr = rmin;
        for (int elr=0;elr<nbinsr;elr++){
            binedges[elr] = tmpr;
            tmpr *= exp(drbin);
            thisredge = reso_redges[mymin(nresos,tmpreso+1)];
            
            if (thisredge<tmpr){
                reso_rindedges[mymin(nresos,tmpreso+1)] = elr;
                if ((tmpr-thisredge)<(thisredge - (tmpr/exp(drbin)))){reso_rindedges[mymin(nresos,tmpreso+1)]+=1;}
                tmpreso+=1;
            }
        }
        binedges[nbinsr] = tmpr;
        binedges[nbinsr+1] = tmpr* exp(drbin);
        
        // Very fine linear array between rmin/rmax
        // We use it to get the bin indices of the log array fast.
        double dbin_lin = 0.9*rmin*(exp(drbin)-1);
        double dbin_lin_inv = 1./dbin_lin;
        int nbinsr_lin = (int) ceil(binedges[nbinsr]/dbin_lin);
        int *linarr_bins = calloc(nbinsr_lin+1, sizeof(int));
        int tmplogbin = 0;
        tmpr = rmin;
        for (int elr=0;elr<=nbinsr_lin;elr++){
            if (tmpr>binedges[tmplogbin+1]){tmplogbin+=1;}
            linarr_bins[elr] = tmplogbin;
            tmpr += dbin_lin;
            if (tmpr>=binedges[nbinsr]){break;}
        }
            
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
        
        for (int elregion=0; elregion<nregions; elregion++){
            int region_debug=99999;
            // Check if this thread is responsible for the region
            int nthread_target = mymin(elregion/nregions_per_thread, nthreads-1);
            if (nthread_target!=elthread){continue;}
            // printf("Region %d is in thread %d\n",elregion,elthread);
            if (elregion==region_debug){printf("Region %d is in thread %d\n",elregion,elthread);}
            if (elthread==nthreads/2){
                printf("\rDone %.2f per cent",100*((double) elregion-nregions_per_thread*(int)(nthreads/2))/
                       nregions_per_thread);
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
            
            // Now, for each resolution, loop over all the galaxies in the region and
            
            int ind_pix1, ind_pix2, ind_inpix1, ind_inpix2, ind_red, ind_gal1, ind_gal2, z_gal1, z_gal2;
            int pix1_lower, pix2_lower, pix1_upper, pix2_upper;
            int lower1, upper1, lower2, upper2;
            int innergal, rbin, nbinsz2r, nbinszr, ind_rbin;
            double pos1_gal1, pos2_gal1, pos1_gal2, pos2_gal2, w_gal1, w_gal2, wtracer_gal1, wtracer_gal2;
            double rel1, rel2, dist, dist_sq;
            double complex wshape_gal1, wshape_gal2;
            double complex phirotc_sq;
            double rmin_reso, rmax_reso, rmin_reso_sq, rmax_reso_sq;

            int elreso_leaf, rbinmin, rbinmax;
            nbinsz2r =  nbinsz*nbinsz*nbinsr;
            nbinszr =  nbinsz*nbinsr;
            for (int elreso=0;elreso<nresos;elreso++){
                elreso_leaf = mymin(mymax(minresoind_leaf,elreso+resoshift_leafs),maxresoind_leaf);
                //elreso_leaf = elreso;
                rbinmin = reso_rindedges[elreso];
                rbinmax = reso_rindedges[elreso+1];
                rmin_reso = rmin*exp(rbinmin*drbin);
                rmax_reso = rmin*exp(rbinmax*drbin);
                int nbinsr_reso = rbinmax-rbinmin;
                rmin_reso_sq = rmin_reso*rmin_reso;
                rmax_reso_sq = rmax_reso*rmax_reso;
                lower1 = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion];
                upper1 = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion+1];
                if (elregion==region_debug){printf("rbinmin=%d, rbinmax%d\n",rbinmin,rbinmax);}
                for (ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                    ind_gal1 = rshift_pix_gals[elreso] + pix_gals[rshift_pix_gals[elreso]+ind_inpix1];
                    innergal = isinner_resos[ind_gal1];
                    if (innergal==0){continue;}
                    pos1_gal1 = pos1_resos[ind_gal1];
                    pos2_gal1 = pos2_resos[ind_gal1];
                    z_gal1 = zbin_resos[ind_gal1];
                    w_gal1 = weight_resos[ind_gal1];
                    wtracer_gal1 = w_gal1*scalar_tracer[ind_gal1];
                    
                    pix1_lower = mymax(0, (int) floor((pos1_gal1 - 2*pix1_d - pix1_start)/pix1_d));// No DC of pairs 
                    pix2_lower = mymax(0, (int) floor((pos2_gal1 - (rmax_reso+pix2_d) - pix2_start)/pix2_d));
                    pix1_upper = mymin(pix1_n-1, (int) floor((pos1_gal1 + (rmax_reso+pix1_d) - pix1_start)/pix1_d));
                    pix2_upper = mymin(pix2_n-1, (int) floor((pos2_gal1 + (rmax_reso+pix2_d) - pix2_start)/pix2_d));
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
                                dist_sq = rel1*rel1 + rel2*rel2;
                                if(rel1<0 && do_dc==0){continue;}
                                if(dist_sq < rmin_reso_sq || dist_sq >= rmax_reso_sq){continue;}
                                dist = sqrt(dist_sq);
                                z_gal2 = zbin_resos[ind_gal2];
                                w_gal2 = weight_resos[ind_gal2];
                                wtracer_gal2 = w_gal2*scalar_tracer[ind_gal2];
                                
                                // Now get the bin index. We have multiple options
                                // Basic way to go when havin a constant logspace. However, the call to log()
                                // is pretty expensive (~50%) of runtime
                                //rbin = (int) floor((0.5*log(dist_sq)-logrmin)/drbin); 
                                // Do a basic binary search. Becomes pretty slow due to multiple comparisons
                                //rbin = binary_search(binedges, nbinsr, dist); 
                                // Search from the largest bin of the current reso backwards. Faster than
                                // binary search for medium sized arrays, but still slow with many bins.
                                //rbin = backsearch(binedges, rbinmin, rbinmax, dist);
                                // Retrieve the bin index of the log using our linear helper array. This is
                                // about twice as fast as calling log(). Instead of time complexity this method
                                // adds space complexity for very narrow log-bins
                                tmplogbin = (int) ((dist-rmin)*dbin_lin_inv);
                                rbin = linarr_bins[tmplogbin]; 
                                rbin += (dist > binedges[rbin + 1]) ? 1 : 0;
                                ind_rbin = elthread*nbinsz2r + z_gal1*nbinszr + z_gal2*nbinsr + rbin;
                                tmpnpair[ind_rbin] += 1; 
                                tmpwcount[ind_rbin] += w_gal1*w_gal2*dist; 
                                tmpwnorm[ind_rbin] += wtracer_gal1*wtracer_gal2;
                            }
                        }
                    }
                }
            }
        }
        free(reso_rindedges);
        free(binedges);
        free(linarr_bins);
        free(rshift_index_matcher);
        free(rshift_pixs_galind_bounds);
        free(rshift_pix_gals);
    }
    // Accumulate the bin distances and weights
    //#pragma omp parallel for num_threads(nthreads)
    for (int elbinr=0; elbinr<nbinsr; elbinr++){
        for (int elbinz1=0; elbinz1<nbinsz; elbinz1++){
            for (int elbinz2=0; elbinz2<nbinsz; elbinz2++){
                int tmpind = elbinz1*nbinsz*nbinsr + elbinz2*nbinsr + elbinr;
                for (int thisthread=0; thisthread<nthreads; thisthread++){
                    int tshift = thisthread*nbinsz*nbinsz*nbinsr; 
                    totcount[tmpind] += tmpwcount[tshift+tmpind];
                    npair[tmpind] += tmpnpair[tshift+tmpind];
                    counts[tmpind] += tmpwnorm[tshift+tmpind];
                }
            }
        }
    }
    // Get bin centers
    for (int elbinz1=0; elbinz1<nbinsz; elbinz1++){
        for (int elbinz2=0; elbinz2<nbinsz; elbinz2++){
            for (int elbinr=0; elbinr<nbinsr; elbinr++){
                int tmpind = elbinz1*nbinsz*nbinsr + elbinz2*nbinsr + elbinr;
                if (counts[tmpind] != 0){
                    bin_centers[tmpind] = totcount[tmpind]/counts[tmpind];
                }
            }
        }
    } 
    free(totcount);
    free(tmpwcount);
    free(tmpnpair);
    free(tmpwnorm);
}

void alloc_xipm_doubletree(
    int nresos, int nresos_grid, double *dpix1_resos,  double *dpix2_resos, double *reso_redges, 
    int resoshift_leafs, int minresoind_leaf, int maxresoind_leaf,
    int *ngal_resos, int nbinsz, int *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos, 
    double *e1_resos, double *e2_resos, int *zbin_resos, 
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int nregions, int *index_matcher_hash,
    double rmin, double rmax, int nbinsr, int do_dc,
    int nthreads, double *bin_centers, double complex *xip, double complex *xim, double *norm, long long int *npair){
    
    double *totcount = calloc(nbinsz*nbinsz*nbinsr, sizeof(double));
        
    // Temporary arrays that are allocated in parallel and later reduced
    int *tmpnpair = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(int));
    double *tmpwcount = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
    double *tmpwnorm = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
    double complex *tmpgg = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double complex));
    double complex *tmpggstar = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double complex));
    
    #pragma omp parallel for num_threads(nthreads)
    for(int elthread=0;elthread<nthreads;elthread++){
        int nregions_per_thread = nregions/nthreads;
        int hasdiscrete = nresos-nresos_grid;
        
        // Check which sets of radii are evaluated for each resolution
        int *reso_rindedges = calloc(nresos+1, sizeof(int));
        double *binedges = calloc(nbinsr+2, sizeof(double));
        double logrmin = log(rmin);
        double drbin = (log(rmax)-logrmin)/(nbinsr);
        int tmpreso = 0;
        double thisredge = 0;
        double tmpr = rmin;
        for (int elr=0;elr<nbinsr;elr++){
            binedges[elr] = tmpr;
            tmpr *= exp(drbin);
            thisredge = reso_redges[mymin(nresos,tmpreso+1)];
            
            if (thisredge<tmpr){
                reso_rindedges[mymin(nresos,tmpreso+1)] = elr;
                if ((tmpr-thisredge)<(thisredge - (tmpr/exp(drbin)))){reso_rindedges[mymin(nresos,tmpreso+1)]+=1;}
                tmpreso+=1;
            }
        }
        binedges[nbinsr] = tmpr;
        binedges[nbinsr+1] = tmpr* exp(drbin);
        
        // Very fine linear array between rmin/rmax
        // We use it to get the bin indices of the log array fast.
        double dbin_lin = 0.9*rmin*(exp(drbin)-1);
        double dbin_lin_inv = 1./dbin_lin;
        int nbinsr_lin = (int) ceil(binedges[nbinsr]/dbin_lin);
        int *linarr_bins = calloc(nbinsr_lin+1, sizeof(int));
        int tmplogbin = 0;
        tmpr = rmin;
        for (int elr=0;elr<=nbinsr_lin;elr++){
            if (tmpr>binedges[tmplogbin+1]){tmplogbin+=1;}
            linarr_bins[elr] = tmplogbin;
            tmpr += dbin_lin;
            if (tmpr>=binedges[nbinsr]){break;}
        }
            
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
        
        for (int elregion=0; elregion<nregions; elregion++){
            int region_debug=99999;
            // Check if this thread is responsible for the region
            int nthread_target = mymin(elregion/nregions_per_thread, nthreads-1);
            if (nthread_target!=elthread){continue;}
            // printf("Region %d is in thread %d\n",elregion,elthread);
            if (elregion==region_debug){printf("Region %d is in thread %d\n",elregion,elthread);}
            if (elthread==nthreads/2){
                printf("\rDone %.2f per cent",100*((double) elregion-nregions_per_thread*(int)(nthreads/2))/
                       nregions_per_thread);
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
            
            // Now, for each resolution, loop over all the galaxies in the region and
            int ind_pix1, ind_pix2, ind_inpix1, ind_inpix2, ind_red, ind_gal1, ind_gal2, z_gal1, z_gal2;
            int pix1_lower, pix2_lower, pix1_upper, pix2_upper;
            int lower1, upper1, lower2, upper2;
            int innergal, rbin, nbinsz2r, nbinszr, ind_rbin;
            double pos1_gal1, pos2_gal1, pos1_gal2, pos2_gal2, w_gal1, w_gal2, e1_gal1, e2_gal1, e1_gal2, e2_gal2;
            double rel1, rel2, dist, dist_sq;
            double complex wshape_gal1, wshape_gal2;
            double complex phirotc_sq;
            double rmin_reso, rmax_reso, rmin_reso_sq, rmax_reso_sq;

            int elreso_leaf, rbinmin, rbinmax;
            nbinsz2r =  nbinsz*nbinsz*nbinsr;
            nbinszr =  nbinsz*nbinsr;
            for (int elreso=0;elreso<nresos;elreso++){
                elreso_leaf = mymin(mymax(minresoind_leaf,elreso+resoshift_leafs),maxresoind_leaf);
                //elreso_leaf = elreso;
                rbinmin = reso_rindedges[elreso];
                rbinmax = reso_rindedges[elreso+1];
                rmin_reso = rmin*exp(rbinmin*drbin);
                rmax_reso = rmin*exp(rbinmax*drbin);
                int nbinsr_reso = rbinmax-rbinmin;
                rmin_reso_sq = rmin_reso*rmin_reso;
                rmax_reso_sq = rmax_reso*rmax_reso;
                lower1 = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion];
                upper1 = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion+1];
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
                    
                    pix1_lower = mymax(0, (int) floor((pos1_gal1 - (rmax_reso+pix1_d) - pix1_start)/pix1_d));
                    pix2_lower = mymax(0, (int) floor((pos2_gal1 - (rmax_reso+pix2_d) - pix2_start)/pix2_d));
                    pix1_upper = mymin(pix1_n-1, (int) floor((pos1_gal1 + (rmax_reso+pix1_d) - pix1_start)/pix1_d));
                    pix2_upper = mymin(pix2_n-1, (int) floor((pos2_gal1 + (rmax_reso+pix2_d) - pix2_start)/pix2_d));
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
                                dist_sq = rel1*rel1 + rel2*rel2;
                                if(rel1<0 && do_dc==0){continue;}
                                if(dist_sq < rmin_reso_sq || dist_sq >= rmax_reso_sq){continue;} 
                                dist = sqrt(dist_sq);
                                w_gal2 = weight_resos[ind_gal2];
                                z_gal2 = zbin_resos[ind_gal2];
                                e1_gal2 = e1_resos[ind_gal2];
                                e2_gal2 = e2_resos[ind_gal2];
                                
                                // Now get the bin index. We have multiple options
                                // Basic way to go when havin a constant logspace. However, the call to log()
                                // is pretty expensive (~30%) of runtime
                                //rbin = (int) floor((0.5*log(dist_sq)-logrmin)/drbin); 
                                // Do a basic binary search. Becomes pretty slow due to multiple comparisons
                                //rbin = binary_search(binedges, nbinsr, dist); 
                                // Search from the largest bin of the current reso backwards. Faster than
                                // binary search for medium sized arrays, but still slow with many bins.
                                //rbin = backsearch(binedges, rbinmin, rbinmax, dist);
                                // Retrieve the bin index of the log using our linear helper array. This is
                                // about twice as fast as calling log(). Instead of time complexity this method
                                // adds space complexity for very narrow log-bins
                                tmplogbin = (int) ((dist-rmin)*dbin_lin_inv);
                                rbin = linarr_bins[tmplogbin]; 
                                rbin += (dist > binedges[rbin + 1]) ? 1 : 0;
                                //rbin += (dist < binedges[rbin]) ? -1 : 0;
                                ind_rbin = elthread*nbinsz2r + z_gal1*nbinszr + z_gal2*nbinsr + rbin;
                                
                                wshape_gal2 = (double complex) w_gal2 * (e1_gal2+I*e2_gal2);
                                phirotc_sq = (rel1*rel1-rel2*rel2-2*I*rel1*rel2)/dist_sq;
                                tmpnpair[ind_rbin] += 1; 
                                tmpwcount[ind_rbin] += w_gal1*w_gal2*dist; 
                                tmpwnorm[ind_rbin] += w_gal1*w_gal2;
                                tmpgg[ind_rbin] += wshape_gal1*wshape_gal2*phirotc_sq*phirotc_sq;
                                tmpggstar[ind_rbin] += wshape_gal1*conj(wshape_gal2);
                            }
                        }
                    }
                }
            }
        }
        free(reso_rindedges);
        free(binedges);
        free(linarr_bins);
        free(rshift_index_matcher);
        free(rshift_pixs_galind_bounds);
        free(rshift_pix_gals);
    }
    // Accumulate the bin distances and weights
    //#pragma omp parallel for num_threads(nthreads)
    for (int elbinr=0; elbinr<nbinsr; elbinr++){
        for (int elbinz1=0; elbinz1<nbinsz; elbinz1++){
            for (int elbinz2=0; elbinz2<nbinsz; elbinz2++){
                int tmpind = elbinz1*nbinsz*nbinsr + elbinz2*nbinsr + elbinr;
                for (int thisthread=0; thisthread<nthreads; thisthread++){
                    int tshift = thisthread*nbinsz*nbinsz*nbinsr; 
                    totcount[tmpind] += tmpwcount[tshift+tmpind];
                    npair[tmpind] += tmpnpair[tshift+tmpind];
                    norm[tmpind] += tmpwnorm[tshift+tmpind];
                    xip[tmpind] += tmpggstar[tshift+tmpind];
                    xim[tmpind] += tmpgg[tshift+tmpind];
                }
            }
        }
    }
    // Get bin centers
    for (int elbinz1=0; elbinz1<nbinsz; elbinz1++){
        for (int elbinz2=0; elbinz2<nbinsz; elbinz2++){
            for (int elbinr=0; elbinr<nbinsr; elbinr++){
                int tmpind = elbinz1*nbinsz*nbinsr + elbinz2*nbinsr + elbinr;
                if (norm[tmpind] != 0){
                    bin_centers[tmpind] = totcount[tmpind]/norm[tmpind];
                    xip[tmpind] /= norm[tmpind];
                    xim[tmpind] /= norm[tmpind];
                }
            }
        }
    } 
    free(totcount);
    free(tmpwcount);
    free(tmpnpair);
    free(tmpwnorm);
    free(tmpggstar);
    free(tmpgg);
}


void __alloc_xipm_doubletree(
    int nresos, int nresos_grid, double *dpix1_resos,  double *dpix2_resos, double *reso_redges, 
    int resoshift_leafs, int minresoind_leaf, int maxresoind_leaf,
    int *ngal_resos, int nbinsz, int *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos, 
    double *e1_resos, double *e2_resos, int *zbin_resos, 
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int nregions, int *index_matcher_hash,
    double rmin, double rmax, int nbinsr, int do_dc,
    int nthreads, double *bin_centers, double complex *xip, double complex *xim, double *norm, long long int *npair){
    
    double *totcount = calloc(nbinsz*nbinsz*nbinsr, sizeof(double));
        
    // Temporary arrays that are allocated in parallel and later reduced
    int *tmpnpair = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(int));
    double *tmpwcount = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
    double *tmpwnorm = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
    //double *tmpgg_re = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
    //double *tmpgg_im = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
    double *tmpgg = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double complex));
    double *tmpggstar = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double complex));
    
    #pragma omp parallel for num_threads(nthreads)
    for(int elthread=0;elthread<nthreads;elthread++){
        int nregions_per_thread = nregions/nthreads;
        int hasdiscrete = nresos-nresos_grid;
        
        // Check which sets of radii are evaluated for each resolution
        int *reso_rindedges = calloc(nresos+1, sizeof(int));
        double *binedges = calloc(nbinsr+2, sizeof(double));
        double logrmin = log(rmin);
        double drbin = (log(rmax)-logrmin)/(nbinsr);
        int tmpreso = 0;
        double thisredge = 0;
        double tmpr = rmin;
        for (int elr=0;elr<nbinsr;elr++){
            binedges[elr] = tmpr;
            tmpr *= exp(drbin);
            thisredge = reso_redges[mymin(nresos,tmpreso+1)];
            
            if (thisredge<tmpr){
                reso_rindedges[mymin(nresos,tmpreso+1)] = elr;
                if ((tmpr-thisredge)<(thisredge - (tmpr/exp(drbin)))){reso_rindedges[mymin(nresos,tmpreso+1)]+=1;}
                tmpreso+=1;
            }
        }
        binedges[nbinsr] = tmpr;
        binedges[nbinsr+1] = tmpr* exp(drbin);
        
        // Very fine linear array between rmin/rmax
        // We use it to get the bin indices of the log array fast.
        double dbin_lin = 0.9*rmin*(exp(drbin)-1);
        double dbin_lin_inv = 1./dbin_lin;
        int nbinsr_lin = (int) ceil(binedges[nbinsr]/dbin_lin);
        int *linarr_bins = calloc(nbinsr_lin+1, sizeof(int));
        int tmplogbin = 0;
        tmpr = rmin;
        for (int elr=0;elr<=nbinsr_lin;elr++){
            if (tmpr>binedges[tmplogbin+1]){tmplogbin+=1;}
            linarr_bins[elr] = tmplogbin;
            tmpr += dbin_lin;
            if (tmpr>=binedges[nbinsr]){break;}
        }
            
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
        
        for (int elregion=0; elregion<nregions; elregion++){
            int region_debug=99999;
            // Check if this thread is responsible for the region
            int nthread_target = mymin(elregion/nregions_per_thread, nthreads-1);
            if (nthread_target!=elthread){continue;}
            // printf("Region %d is in thread %d\n",elregion,elthread);
            if (elregion==region_debug){printf("Region %d is in thread %d\n",elregion,elthread);}
            if (elthread==nthreads/2){
                printf("\rDone %.2f per cent",100*((double) elregion-nregions_per_thread*(int)(nthreads/2))/
                       nregions_per_thread);
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
            
            // Now, for each resolution, loop over all the galaxies in the region and
            int ind_pix1, ind_pix2, ind_inpix1, ind_inpix2, ind_red, ind_gal1, ind_gal2, z_gal1, z_gal2;
            int pix1_lower, pix2_lower, pix1_upper, pix2_upper;
            int lower1, upper1, lower2, upper2;
            int innergal, rbin, nbinsz2r, nbinszr, ind_rbin;
            double pos1_gal1, pos2_gal1, pos1_gal2, pos2_gal2, w_gal1, w_gal2, e1_gal1, e2_gal1, e1_gal2, e2_gal2;
            double rel1, rel2, dist, dist_sq;
            double complex wshape_gal1, wshape_gal2;
            double complex phirotc_sq;
            double rphirotc_sq;
            double rmin_reso, rmax_reso, rmin_reso_sq, rmax_reso_sq;

            int elreso_leaf, rbinmin, rbinmax;
            nbinsz2r =  nbinsz*nbinsz*nbinsr;
            nbinszr =  nbinsz*nbinsr;
            for (int elreso=0;elreso<nresos;elreso++){
                elreso_leaf = mymin(mymax(minresoind_leaf,elreso+resoshift_leafs),maxresoind_leaf);
                //elreso_leaf = elreso;
                rbinmin = reso_rindedges[elreso];
                rbinmax = reso_rindedges[elreso+1];
                rmin_reso = rmin*exp(rbinmin*drbin);
                rmax_reso = rmin*exp(rbinmax*drbin);
                int nbinsr_reso = rbinmax-rbinmin;
                rmin_reso_sq = rmin_reso*rmin_reso;
                rmax_reso_sq = rmax_reso*rmax_reso;
                lower1 = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion];
                upper1 = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion+1];
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
                    
                    double rel1_2, rel2_2, w_12, e1_12, e2_12;
                    double complex e_gal1 = (e1_gal1+I*e2_gal1);
                    
                    pix1_lower = mymax(0, (int) floor((pos1_gal1 - 2*pix1_d - pix1_start)/pix1_d));// No DC of pairs 
                    pix2_lower = mymax(0, (int) floor((pos2_gal1 - (rmax_reso+pix2_d) - pix2_start)/pix2_d));
                    pix1_upper = mymin(pix1_n-1, (int) floor((pos1_gal1 + (rmax_reso+pix1_d) - pix1_start)/pix1_d));
                    pix2_upper = mymin(pix2_n-1, (int) floor((pos2_gal1 + (rmax_reso+pix2_d) - pix2_start)/pix2_d));
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
                                dist_sq = rel1*rel1 + rel2*rel2;
                                if(rel1<0 && do_dc==0){continue;}
                                if(dist_sq < rmin_reso_sq || dist_sq >= rmax_reso_sq) continue; // DC of pairs
                                dist = sqrt(dist_sq);
                                w_gal2 = weight_resos[ind_gal2];
                                z_gal2 = zbin_resos[ind_gal2];
                                e1_gal2 = e1_resos[ind_gal2];
                                e2_gal2 = e2_resos[ind_gal2];
                                
                                // Now get the bin index. We have multiple options
                                // Basic way to go when havin a constant logspace. However, the call to log()
                                // is pretty expensive (~30%) of runtime
                                //rbin = (int) floor((0.5*log(dist_sq)-logrmin)/drbin); 
                                // Do a basic binary search. Becomes pretty slow due to multiple comparisons
                                //rbin = binary_search(binedges, nbinsr, dist); 
                                // Search from the largest bin of the current reso backwards. Faster than
                                // binary search for medium sized arrays, but still slow with many bins.
                                //rbin = backsearch(binedges, rbinmin, rbinmax, dist);
                                // Retrieve the bin index of the log using our linear helper array. This is
                                // about twice as fast as calling log(). Instead of time complexity this method
                                // adds space complexity for very narrow log-bins
                                tmplogbin = (int) ((dist-rmin)*dbin_lin_inv);
                                rbin = linarr_bins[tmplogbin]; 
                                rbin += (dist > binedges[rbin+1]) ?  1 : 0;
                                ind_rbin = elthread*nbinsz2r + z_gal1*nbinszr + z_gal2*nbinsr + rbin;
                               
                                // Basic allocation
                                wshape_gal2 = (double complex) w_gal2 * (e1_gal2+I*e2_gal2);
                                phirotc_sq = (rel1*rel1-rel2*rel2-2*I*rel1*rel2)/dist_sq;
                                tmpnpair[ind_rbin] += 1; 
                                tmpwcount[ind_rbin] += w_gal1*w_gal2*dist; 
                                tmpwnorm[ind_rbin] += w_gal1*w_gal2;
                                tmpgg[ind_rbin] += wshape_gal1*wshape_gal2*phirotc_sq*phirotc_sq;
                                tmpggstar[ind_rbin] += wshape_gal1*conj(wshape_gal2);
                                
                                /*
                                // This should be a bit faster if we do not have patch overlap and want to dc
                                // as in this case we can directly account for the pair with switched positions
                                // Practically:
                                // g1g2 -> e1*proj*e2*proj + e1*conj(proj)*e2*conj(proj)
                                //       = 2*e1*e2*Re(proj*proj)
                                // g1g2c -> e1*proj*conj(e2*proj) + e2*proj*conj(e1*proj)
                                //       = 2*Re(e1*e2)
                                // Somehow the gg_im part is wrong...
                                rel1_2=rel1*rel1;
                                rel2_2=rel2*rel2;
                                w_12 = w_gal1*w_gal2;
                                e1_12 = e1_gal1*e1_gal2;
                                e2_12 = e2_gal1*e2_gal2;
                                rphirotc_sq = 2*(rel1_2*rel1_2-6*rel1_2*rel2_2+rel2_2*rel2_2)/(dist_sq*dist_sq); 
                                tmpnpair[ind_rbin] += 1; 
                                tmpwcount[ind_rbin] += w_12*dist; 
                                tmpwnorm[ind_rbin] += w_12;
                                tmpgg_re[ind_rbin] += w_12*(e1_12-e2_12)*rphirotc_sq;
                                tmpgg_im[ind_rbin] += w_12*(e1_gal1*e2_gal2+e2_gal1*e1_gal2)*rphirotc_sq;
                                tmpggstar[ind_rbin] += w_12*(e1_12+e2_12);
                                */
                            }
                        }
                    }
                }
            }
        }
        free(reso_rindedges);
        free(binedges);
        free(linarr_bins);
        free(rshift_index_matcher);
        free(rshift_pixs_galind_bounds);
        free(rshift_pix_gals);
    }
    // Accumulate the bin distances and weights
    //#pragma omp parallel for num_threads(nthreads)
    for (int elbinr=0; elbinr<nbinsr; elbinr++){
        for (int elbinz1=0; elbinz1<nbinsz; elbinz1++){
            for (int elbinz2=0; elbinz2<nbinsz; elbinz2++){
                int tmpind = elbinz1*nbinsz*nbinsr + elbinz2*nbinsr + elbinr;
                for (int thisthread=0; thisthread<nthreads; thisthread++){
                    int tshift = thisthread*nbinsz*nbinsz*nbinsr; 
                    totcount[tmpind] += tmpwcount[tshift+tmpind];
                    npair[tmpind] += tmpnpair[tshift+tmpind];
                    norm[tmpind] += tmpwnorm[tshift+tmpind];
                    xip[tmpind] += tmpggstar[tshift+tmpind];
                    //xim[tmpind] += tmpgg_re[tshift+tmpind] + I*tmpgg_im[tshift+tmpind];
                    xim[tmpind] += tmpgg[tshift+tmpind];
                }
            }
        }
    }
    // Get bin centers
    for (int elbinz1=0; elbinz1<nbinsz; elbinz1++){
        for (int elbinz2=0; elbinz2<nbinsz; elbinz2++){
            for (int elbinr=0; elbinr<nbinsr; elbinr++){
                int tmpind = elbinz1*nbinsz*nbinsr + elbinz2*nbinsr + elbinr;
                if (norm[tmpind] != 0){
                    bin_centers[tmpind] = totcount[tmpind]/norm[tmpind];
                    xip[tmpind] /= norm[tmpind];
                    xim[tmpind] /= norm[tmpind];
                }
            }
        }
    } 
    free(totcount);
    free(tmpwcount);
    free(tmpnpair);
    free(tmpwnorm);
    free(tmpggstar);
    //free(tmpgg_re);
    //free(tmpgg_im);
    free(tmpgg);
}
