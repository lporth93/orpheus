// In case the machines are acting up, compile with 
// gcc-12 -fopenmp $(gsl-config --cflags) -fPIC -shared -o discrete.so discrete.c $(gsl-config --libs) -std=c99

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <complex.h>

#include "utils.h"
#include "combinatorics.h"
#include "spatialhash.h"
#include "directestimator.h"

#define mymin(x,y) ((x) <= (y)) ? (x) : (y)
#define mymax(x,y) ((x) >= (y)) ? (x) : (y)
#define M_PI      3.14159265358979323846
#define INV_2PI   0.15915494309189534561


double getFilterQ(int type_filter, double reff2){
    double reff, xc;
    double res=0;
    double area = 1;
    switch (type_filter){
        // Polynominal (from Schneider 1998)
        case 0:
            //reff2 = pow(dgal/R_ap,2);
            res = 6./area * reff2 * (1.-reff2);
            break;
        // Exponential (from Crittenden 2002)
        case 1:
            //reff2 = pow(dgal/R_ap,2);
            res = reff2/(4.*area)*exp(-0.5*reff2);
            break;
        // NFW (from Schirmer 2004)
        case 2:
            xc = 0.15;
            reff = sqrt(reff2);
            res = pow(area * (1+exp(6-150*reff)+exp(-47+50*reff)),-1)*(xc/reff)*tanh(reff/xc);
            break;
        // Poly-exp (More close to NFW, but there is a simple U)
        case 3:
            //reff2 = pow(dgal/R_ap,2);
            res = -1./area * (1.*(1./(150*reff2)+1)*exp(-150.*reff2) + 0.5*(1/(30*reff2)+1)*exp(-30.*reff2) + (-.0233333)*(1./(1*reff2)+1)*exp(-1.*reff2));
            break;
        default:
            printf("Error! operator is not correct");
            res = 0;
        case 4:
            //reff2 = pow(dgal/R_ap,2);
            res = reff2/(0.01*area)*exp(-0.1*reff2);
            break;
    }
    return res;
}

double getFilterU(int type_filter, double reff2){
    double res=0;
    double area = 1;
    switch (type_filter){
        // Polynominal (from Schneider 1998)
        case 0:
            //reff2 = pow(dgal/R_ap,2);
            res = 6./area * reff2 * (1.-reff2);
            break;
        // Exponential (from Crittenden 2002)
        case 1:
            //reff2 = pow(dgal/R_ap,2);
            res = INV_2PI * (1.-0.5*reff2) * exp(-0.5*reff2);
            break;
    }
    return res;
}

double getFilterSupp(int type_filter){
    double res = 1;
    switch (type_filter){
        // Polynominal Schneider
        case 0:
            res = 1.;
            break;
        // Exponential Crittenden
        case 1:
            res = 4;
            break;
        // NFW
        case 2:
            res = 1.25;
            break;
        // Poly-exp
        case 3:
            res = 2.5;
            break;
         default:
            printf("Error! operator is not correct");
            res = 0;
    }
    return res;
}

double getFilterSuppU(int type_filter){
    double res = 1;
    switch (type_filter){
        // Polynominal Schneider
        case 0:
            res = 1.;
            break;
        // Exponential Crittenden
        case 1:
            res = 4;
            break;
        // NFW
        case 2:
            res = 1.25;
            break;
        // Poly-exp
        case 3:
            res = 2.5;
            break;
         default:
            printf("Error! operator is not correct");
            res = 0;
    }
    return res;
}


// Returns Map+iMx for each zbin, as well as an aperture weight map and a coverage fraction map.
// This forms the basis of all estimators that do not do any multiple-counting corrections.
// TODO: For multi-scale estimator might be good to use tree-methods here s.t. we can overlay centers?
void ApertureMassMap_Equal(
    double R_ap, double *centers_1, double *centers_2, int ncenters_1, int ncenters_2,
    int max_order, int ind_filter, int weight_method, double weight_outer, double weight_inpainted, 
    double *weight, double *insurvey, double *pos1, double *pos2, double complex *g, int *zbins, int nbinsz, int ngal, 
    double *mask, 
    double mask1_start, double mask2_start, double mask1_d, double mask2_d, int mask1_n, int mask2_n,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    int nthreads, 
    double *out_counts, double *out_covs, double *out_Msn, double *out_Sn, double *out_Mapn, double *out_Mapn_var){
        
    #pragma omp parallel for num_threads(nthreads)
    for (int elthread=0; elthread<nthreads; elthread++){
        //printf("Entered parallel region %d\n",elthread);
        int ind_ap, elbinz, order;
        double *nextcounts = calloc(3*nbinsz, sizeof(double));
        double *nextcovs = calloc(2, sizeof(double));
        double *nextMsn = calloc(max_order*nbinsz, sizeof(double));
        double *nextSn = calloc(max_order*nbinsz, sizeof(double));
        double *nextSn_w = calloc(max_order*nbinsz, sizeof(double));
        double *nextS2n_w = calloc(max_order*nbinsz, sizeof(double));
        double *nextMapn = calloc(max_order*nbinsz, sizeof(double));
        double *nextMapn_var = calloc(max_order*nbinsz, sizeof(double));
        
        double *factorials_zcombis = calloc(max_order+nbinsz+1, sizeof(double));
        double *factorials = calloc(max_order+1, sizeof(double));
        double *bellargs_Msn = calloc(max_order, sizeof(double));
        double *bellargs_Sn = calloc(max_order, sizeof(double));
        double *bellargs_Sn_w = calloc(max_order, sizeof(double));
        double *bellargs_S2n_w = calloc(max_order, sizeof(double));
        double *nextMapn_singlez = calloc(max_order+1, sizeof(double));
        double *nextMapn_norm_singlez = calloc(max_order+1, sizeof(double));
        double *nextMapn_var_singlez = calloc(max_order+1, sizeof(double));
        double *nextMapn_var_norm_singlez = calloc(max_order+1, sizeof(double));
        
        gen_fac_table(max_order, factorials);
        gen_fac_table(max_order+nbinsz, factorials_zcombis);
        //int thread_nzcombis = zcombis_tot(nbinsz, max_order, factorials_zcombis);
        
        //printf("%d %d \n ",ncenters, thread_nzcombis);
        //printf("Done preps for parallel region %d\n",elthread);
        for (ind_ap=0; ind_ap<ncenters_1*ncenters_2; ind_ap++){
            if ((ind_ap%nthreads)!=elthread){continue;}
            int c1 = ind_ap%ncenters_1;
            int c2 = ind_ap/ncenters_1;
            //if (elthread==0){
                //for (order=1; order<=max_order; order++){
                //    printf("%d %d %d %d %d %d \n ",
                //           ncenters, ind_ap, nbinsz, order, factorials_zcombis[nbinsz+order-1],
                //           zcombis_order(nbinsz, order, factorials_zcombis));
                //}
                //printf("Tot: %d \n ", thread_nzcombis);
            //}
            
            //if (elthread==0){printf("Start setting stuff to zero for ap %d on thread %d\n",ind_ap,elthread);}
            // Reset args to zeros
            nextcovs[0]=0;nextcovs[1]=0;
            for (int i=0; i<3*nbinsz; i++){nextcounts[i]=0;}
            for (int i=0; i<max_order*nbinsz; i++){
                nextMsn[i]=0;nextSn[i]=0;nextSn_w[i]=0;nextS2n_w[i]=0;
            }
                    
            // Get all the statistics of the aperture in power sum basis
            //if (elthread==0){printf("Get power sums from ap %d on thread %d\n",ind_ap,elthread);}
            singleAp_MapnSingleEonlyDisc( R_ap, centers_1[c1], centers_2[c2], 
                max_order, ind_filter, weight_outer, weight_inpainted, 
                weight, insurvey, pos1, pos2, g, zbins, nbinsz, ngal,
                mask, mask1_start, mask2_start, mask1_d, mask2_d, mask1_n, mask2_n,
                index_matcher, pixs_galind_bounds, pix_gals,
                nextcounts, nextcovs, nextMsn, nextSn, nextSn_w, nextS2n_w);
            
            //if (elthread==0){printf("Got stats for ap %d on thread %d\n",ind_ap,elthread);}
            //printf("For aperture # %d at zbin0: x/y=%.3f/%.3f, cov=%.3f/%.3f, counts=%.3f, Ms1=%.10f S1=%.10f S2=%.10f \n",
            //                   ind_ap, centers_1[c1], centers_2[c2], nextcovs[0], nextcovs[1], nextcounts[0],
            //                   nextMsn[0*max_order+0],nextSn[0*max_order+0],nextSn[0*max_order+1]);
            
            // Transform to Mapn(zi)
            //if (elthread==0){printf("Transforming to Mapn %d on thread %d\n",ind_ap,elthread);}
            for (elbinz=0; elbinz<nbinsz; elbinz++){
                for (int i=0; i<max_order+1; i++){
                    nextMapn_singlez[i]=0; nextMapn_norm_singlez[i]=0; 
                    nextMapn_var_singlez[i]=0; nextMapn_var_norm_singlez[i]=0;
                    nextMapn[elbinz*max_order+i]=0;nextMapn_var[elbinz*max_order+i]=0;
                }
                int tmpind = elbinz*max_order+0;
                bellargs_Msn[0] = -nextMsn[tmpind];
                bellargs_Sn[0] = -nextSn[tmpind];
                bellargs_Sn_w[0] = -nextSn_w[tmpind];
                bellargs_S2n_w[0] = -nextS2n_w[tmpind];
                
                for (order=1; order<max_order; order++){ 
                    tmpind += 1;
                    bellargs_Msn[order] = -factorials[order]*nextMsn[tmpind];
                    bellargs_Sn[order] = -factorials[order]*nextSn[tmpind];
                    bellargs_Sn_w[order] = -factorials[order]*nextSn_w[tmpind];
                    bellargs_S2n_w[order] = -factorials[order]*nextS2n_w[tmpind];
                }
                getBellRecursive(max_order, bellargs_Msn, factorials, nextMapn_singlez);
                getBellRecursive(max_order, bellargs_Sn,  factorials, nextMapn_norm_singlez);
                getBellRecursive(max_order, bellargs_S2n_w, factorials, nextMapn_var_singlez);
                getBellRecursive(max_order, bellargs_Sn_w,  factorials, nextMapn_var_norm_singlez);
                for (order=0; order<max_order; order++){
                    if ((nextMapn_norm_singlez[order+1]!=0)&&(nextMapn_var_norm_singlez[order+1]!=0)){
                        nextMapn[elbinz*max_order+order] = nextMapn_singlez[order+1]/nextMapn_norm_singlez[order+1];
                        nextMapn_var[elbinz*max_order+order] = nextMapn_var_singlez[order+1] / (
                            nextMapn_var_norm_singlez[order+1]*nextMapn_var_norm_singlez[order+1]);
                        //printf("For aperture # %d: cov=%.3f, counts=%.3f, Map^%d_nom=%.10f, Multiplets^%d=%.10f, Map^%d=%.10f w_Ap=%.10f \n",
                        //       ind_ap, nextcovs[0], nextcounts[elbinz*3], order, nextMapn_singlez[order+1], order, 
                        //       nextMapn_norm_singlez[order+1], order,
                        //       nextMapn[elbinz*max_order+order],nextMapn_var[elbinz*max_order+order]);
                    }
                }   
            }
            
            // Update output grids
            int ncenters = ncenters_1*ncenters_2;
            out_covs[0*ncenters + c2*ncenters_1+c1] = nextcovs[0];
            out_covs[1*ncenters + c2*ncenters_1+c1] = nextcovs[1];
            for (int i=0; i<3*nbinsz; i++){
                out_counts[i*ncenters + c2*ncenters_1+c1] =  nextcounts[i];
            }
            for (int i=0; i<max_order*nbinsz; i++){
                out_Sn[i*ncenters + c2*ncenters_1+c1] = nextSn[i];
                out_Msn[i*ncenters + c2*ncenters_1+c1] = nextMsn[i];
                out_Mapn[i*ncenters + c2*ncenters_1+c1] = nextMapn[i];
                out_Mapn_var[i*ncenters + c2*ncenters_1+c1] = nextMapn_var[i];
                //if (i<2){
                //printf("At allocation: For ap %d/%d on thread %d at i=%i: S=%.8f  Ms=%.8f i.e. S=%.8f  Ms=%.8f \n",
                //       c1,c2,elthread,i,nextSn[i],nextMsn[i],
                //       out_Sn[i*ncenters + c2*ncenters_1+c1],
                //       out_Msn[i*ncenters + c2*ncenters_1+c1]);}
            }
            //printf("For ap %d/%d on thread %d: S1=%.8f  S2=%.8f  Ms1=%.8f  Ms2=%.8f\n\n",
            //       c1,c2,elthread,
            //       out_Sn[0*ncenters + c2*ncenters_1+c1],
            //       out_Sn[1*ncenters + c2*ncenters_1+c1],
            //       out_Msn[0*ncenters + c2*ncenters_1+c1],
            //       out_Msn[1*ncenters + c2*ncenters_1+c1]);
        }
        
        free(nextcounts);
        free(nextcovs);
        free(nextMsn);
        free(nextSn);
        free(nextSn_w);
        free(nextS2n_w);
        
        free(factorials_zcombis);
        free(factorials);
        free(bellargs_Msn);
        free(bellargs_Sn);
        free(bellargs_Sn_w);
        free(bellargs_S2n_w);
        free(nextMapn_singlez);
        free(nextMapn_norm_singlez);
        free(nextMapn_var_singlez);
        free(nextMapn_var_norm_singlez);
        free(nextMapn);
        free(nextMapn_var);
    }
}

// Computes Napn for single aperture scale, taking into account the multiple-counting corrections.  
void ApertureCountsMap_Equal(
    double R_ap, double *centers_1, double *centers_2, int ncenters_1, int ncenters_2,
    int max_order, int ind_filter, int do_subtractions, int Nbar_policy, double weight_outer, double weight_inpainted, 
    double *weight, double *insurvey, double *pos1, double *pos2, double *tracer, int *zbins, int nbinsz, int ngal, 
    double *mask, 
    double mask1_start, double mask2_start, double mask1_d, double mask2_d, int mask1_n, int mask2_n,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    int nthreads,
    double *out_counts, double *out_covs, double *out_Msn, double *out_Sn, double *out_Napn, double *out_Napn_norm){
    
    
    #pragma omp parallel for num_threads(nthreads)
    for (int elthread=0; elthread<nthreads; elthread++){
        //printf("Entered parallel region %d\n",elthread);
        int ind_ap, elbinz, elzcombi, elcov_cut, order;
        double *nextcounts = calloc(3*nbinsz, sizeof(double));
        double *nextcovs = calloc(2, sizeof(double));
        double *nextMsn = calloc(max_order*nbinsz, sizeof(double));
        double *nextSn = calloc(max_order*nbinsz, sizeof(double));
        
        double *factorials_zcombis = calloc(max_order+nbinsz+1, sizeof(double));
        double *factorials = calloc(max_order+1, sizeof(double));
        double *bellargs_Msn = calloc(max_order, sizeof(double));
        double *bellargs_Sn = calloc(max_order, sizeof(double));
        double *nextNapn_singlez = calloc(max_order+1, sizeof(double));
        double *nextNapn = calloc(max_order*nbinsz, sizeof(double));
        double *nextNapn_norm_singlez = calloc(max_order+1, sizeof(double));
        double *nextNapn_norm = calloc(max_order*nbinsz, sizeof(double));
        
        gen_fac_table(max_order, factorials);
        gen_fac_table(max_order+nbinsz, factorials_zcombis);
        
        for (ind_ap=0; ind_ap<ncenters_1*ncenters_2; ind_ap++){
            if ((ind_ap%nthreads)!=elthread){continue;}
            int c1 = ind_ap%ncenters_1;
            int c2 = ind_ap/ncenters_1;
       
            // Reset args to zeros
            for (int i=0; i<3*nbinsz; i++){nextcounts[i]=0;}
            nextcovs[0]=0;nextcovs[1]=0;
            for (int i=0; i<max_order*nbinsz; i++){
                nextMsn[i]=0;nextSn[i]=0;
            }
            
            // Get all the statistics of the aperture in power sum basis
            singleAp_NapnSingleDisc( R_ap, centers_1[c1], centers_2[c2], 
                max_order, ind_filter, Nbar_policy, weight_outer, weight_inpainted, 
                weight, insurvey, pos1, pos2, tracer, zbins, nbinsz, ngal,
                mask, mask1_start, mask2_start, mask1_d, mask2_d, mask1_n, mask2_n,
                index_matcher, pixs_galind_bounds, pix_gals,  
                nextcounts, nextcovs, nextMsn, nextSn);
            
            
            // Transform to Napn(zi)
            for (elbinz=0; elbinz<nbinsz; elbinz++){
                for (int i=0; i<max_order+1; i++){
                    nextNapn_singlez[i]=0; nextNapn_norm_singlez[i]=0; 
                    nextNapn[elbinz*max_order+i]=0;nextNapn_norm[elbinz*max_order+i]=0;
                    }
                int tmpind = elbinz*max_order+0;
                bellargs_Msn[0] = -nextMsn[tmpind];
                bellargs_Sn[0] = -nextSn[tmpind];
                for (order=1; order<max_order; order++){ 
                    tmpind += 1;
                    bellargs_Msn[order] = -factorials[order]*nextMsn[tmpind];
                    bellargs_Sn[order] = -factorials[order]*nextSn[tmpind];
                }
                getBellRecursive(max_order, bellargs_Msn, factorials, nextNapn_singlez);
                getBellRecursive(max_order, bellargs_Sn,  factorials, nextNapn_norm_singlez);
                for (order=0; order<max_order; order++){
                    if ((nextNapn_norm_singlez[order+1]!=0)){
                        nextNapn[elbinz*max_order+order] = pow(-1,order+1)*nextNapn_singlez[order+1];
                        nextNapn_norm[elbinz*max_order+order] = pow(-1,order+1)*nextNapn_norm_singlez[order+1];
                    }
                }   
            }
            
            // Update output grids
            int ncenters = ncenters_1*ncenters_2;
            out_covs[0*ncenters + c2*ncenters_1+c1] = nextcovs[0];
            out_covs[1*ncenters + c2*ncenters_1+c1] = nextcovs[1];
            for (int i=0; i<3*nbinsz; i++){
                out_counts[i*ncenters + c2*ncenters_1+c1] =  nextcounts[i];
            }
            for (int i=0; i<max_order*nbinsz; i++){
                out_Sn[i*ncenters + c2*ncenters_1+c1] = nextSn[i];
                out_Msn[i*ncenters + c2*ncenters_1+c1] = nextMsn[i];
                out_Napn[i*ncenters + c2*ncenters_1+c1] = nextNapn[i];
                out_Napn_norm[i*ncenters + c2*ncenters_1+c1] = nextNapn_norm[i];
            }
        }
        
        free(nextcounts);
        free(nextcovs);
        free(nextMsn);
        free(nextSn);
        
        free(factorials_zcombis);
        free(factorials);
        free(bellargs_Msn);
        free(bellargs_Sn);
        free(nextNapn_singlez);
        free(nextNapn_norm_singlez);
        free(nextNapn);
        free(nextNapn_norm);
    }
}

// Computes Mapn for single aperture scale, taking into account the multiple-counting corrections.
// Weight methods:
//  * 0 --> Identity weights
//  * 1 --> Inverse shape noise weights               
void MapnSingleEonlyDisc(
    double R_ap, double *centers_1, double *centers_2, int ncenters,
    int max_order, int ind_filter, int weight_method, int do_subtractions, double weight_outer, double weight_inpainted, 
    double *weight, double *insurvey, double *pos1, double *pos2, double complex *g, int *zbins, int nbinsz, int ngal, 
    double *mask, double *fraccov_cuts, int nfrac_cuts, int fraccov_method,
    double mask1_start, double mask2_start, double mask1_d, double mask2_d, int mask1_n, int mask2_n,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    int nthreads, double *Mapn, double *wtot_Mapn){
    
    //printf("Starting\n");
    double *fac_zcombis = calloc(max_order+nbinsz+1, sizeof(double));
    gen_fac_table(max_order+nbinsz, fac_zcombis);
    int nzcombis = zcombis_tot(nbinsz, max_order, fac_zcombis);
    
    // shape (nthreads, nfrac_cuts, nzcombis)
    double *tmpMapn = calloc(nthreads*nfrac_cuts*nzcombis, sizeof(double));
    double *tmpwtot_Mapn = calloc(nthreads*nfrac_cuts*nzcombis, sizeof(double));
    //printf("Now entering parallel region\n");
    
    #pragma omp parallel for num_threads(nthreads)
    for (int elthread=0; elthread<nthreads; elthread++){
        //printf("Entered parallel region %d\n",elthread);
        int ind_ap, elbinz, elzcombi, elcov_cut, order;
        double *nextcounts = calloc(3*nbinsz, sizeof(double));
        double *nextcovs = calloc(2, sizeof(double));
        double *nextMsn = calloc(max_order*nbinsz, sizeof(double));
        double *nextSn = calloc(max_order*nbinsz, sizeof(double));
        double *nextSn_w = calloc(max_order*nbinsz, sizeof(double));
        double *nextS2n_w = calloc(max_order*nbinsz, sizeof(double));
        
        double *factorials_zcombis = calloc(max_order+nbinsz+1, sizeof(double));
        double *factorials = calloc(max_order+1, sizeof(double));
        double *bellargs_Msn = calloc(max_order, sizeof(double));
        double *bellargs_Sn = calloc(max_order, sizeof(double));
        double *bellargs_Sn_w = calloc(max_order, sizeof(double));
        double *bellargs_S2n_w = calloc(max_order, sizeof(double));
        double *nextMapn_singlez = calloc(max_order+1, sizeof(double));
        double *nextMapn_norm_singlez = calloc(max_order+1, sizeof(double));
        double *nextMapn_var_singlez = calloc(max_order+1, sizeof(double));
        double *nextMapn_var_norm_singlez = calloc(max_order+1, sizeof(double));
        double *nextMapn = calloc(max_order*nbinsz, sizeof(double));
        double *nextMapn_var = calloc(max_order*nbinsz, sizeof(double));
        
        gen_fac_table(max_order, factorials);
        gen_fac_table(max_order+nbinsz, factorials_zcombis);
        int thread_nzcombis = zcombis_tot(nbinsz, max_order, factorials_zcombis);
        int thisthreadshift = elthread*nfrac_cuts*thread_nzcombis;
        
        //printf("%d %d \n ",ncenters, thread_nzcombis);
        //printf("Done preps for parallel region %d\n",elthread);
        for (ind_ap=0; ind_ap<ncenters; ind_ap++){
            if ((ind_ap%nthreads)!=elthread){continue;}
            //if (elthread==0){printf("Starting ap %d/%d on thread %d\n",ind_ap+1,ncenters,elthread);}
            if (elthread==0){
                //for (order=1; order<=max_order; order++){
                //    printf("%d %d %d %d %d %d \n ",
                //           ncenters, ind_ap, nbinsz, order, factorials_zcombis[nbinsz+order-1],
                //           zcombis_order(nbinsz, order, factorials_zcombis));
                //}
                //printf("Tot: %d \n ", thread_nzcombis);
            }
            
            //if (elthread==0){printf("Start setting stuff to zero for ap %d on thread %d\n",ind_ap,elthread);}
            // Reset args to zeros
            nextcovs[0]=0;nextcovs[1]=0;
            for (int i=0; i<max_order*nbinsz; i++){
                nextMsn[i]=0;nextSn[i]=0;nextSn_w[i]=0;nextS2n_w[i]=0;
            }
            for (int i=0; i<3*nbinsz; i++){nextcounts[i]=0;}
            
            // Get all the statistics of the aperture in power sum basis
            //if (elthread==0){printf("Get power sums from ap %d on thread %d\n",ind_ap,elthread);}
            singleAp_MapnSingleEonlyDisc( R_ap, centers_1[ind_ap], centers_2[ind_ap], 
                max_order, ind_filter, weight_outer, weight_inpainted, 
                weight, insurvey, pos1, pos2, g, zbins, nbinsz, ngal,
                mask, mask1_start, mask2_start, mask1_d, mask2_d, mask1_n, mask2_n,
                index_matcher, pixs_galind_bounds, pix_gals, 
                nextcounts, nextcovs, nextMsn, nextSn, nextSn_w, nextS2n_w);
            //if (elthread==0){printf("Got stats for ap %d on thread %d\n",ind_ap,elthread);}
            //for (order=0; order<max_order; order++){
            //printf("For aperture # %d: x/y=%.3f/%.3f, cov=%.3f/%.3f, counts=%.3f, Ms_%d=%.10f S_%d=%.10f \n",
            //                   ind_ap, centers_1[ind_ap], centers_2[ind_ap], nextcovs[0], nextcovs[0], nextcounts[0], order,
            //                   nextMsn[0*max_order+order],order,nextSn[0*max_order+order]);}
            
            // Transform to Mapn(zi)
            //if (elthread==0){printf("Transforming to Mapn %d on thread %d\n",ind_ap,elthread);}
            for (elbinz=0; elbinz<nbinsz; elbinz++){
                for (int i=0; i<max_order+1; i++){
                    nextMapn_singlez[i]=0; nextMapn_norm_singlez[i]=0; 
                    nextMapn_var_singlez[i]=0; nextMapn_var_norm_singlez[i]=0;
                    }
                double norm_var;
                int tmpind = elbinz*max_order+0;
                bellargs_Msn[0] = -nextMsn[tmpind];
                bellargs_Sn[0] = -nextSn[tmpind];
                bellargs_Sn_w[0] = -nextSn_w[tmpind];
                bellargs_S2n_w[0] = -nextS2n_w[tmpind];
                norm_var = 1;//nextSn_w[elbinz*max_order];
                // We renorm the bellargs_Sn_w/bellargs_Sn_2w variables to keep the numbers finite
                // and such that the renorm factor cancels out in the nextMapn_var allocation.
                for (order=1; order<max_order; order++){ 
                    tmpind += 1;
                    bellargs_Msn[order] = -factorials[order]*nextMsn[tmpind];
                    bellargs_Sn[order] = -factorials[order]*nextSn[tmpind];
                    bellargs_Sn_w[order] = -factorials[order]*nextSn_w[tmpind]/norm_var; 
                    bellargs_S2n_w[order] = -factorials[order]*nextS2n_w[tmpind]/norm_var/norm_var;
                    norm_var *= norm_var;
                }
                getBellRecursive(max_order, bellargs_Msn, factorials, nextMapn_singlez);
                getBellRecursive(max_order, bellargs_Sn,  factorials, nextMapn_norm_singlez);
                getBellRecursive(max_order, bellargs_S2n_w, factorials, nextMapn_var_singlez);
                getBellRecursive(max_order, bellargs_Sn_w,  factorials, nextMapn_var_norm_singlez);
                for (order=0; order<max_order; order++){
                    if ((nextMapn_norm_singlez[order+1]!=0)&&(nextMapn_var_norm_singlez[order+1]!=0)){
                        nextMapn[elbinz*max_order+order] = nextMapn_singlez[order+1]/nextMapn_norm_singlez[order+1];
                        nextMapn_var[elbinz*max_order+order] = nextMapn_var_singlez[order+1] / (
                            nextMapn_var_norm_singlez[order+1]*nextMapn_var_norm_singlez[order+1]);
                    }
                }   
            }
            
            // Build all zcombis
            // Map^n(z_1,...,z_n) = \prod_{i=1}^nbinsz Map^{alpha_i}(z_i)
            //if (elthread==0){printf("Allocating zcombis for ap %d on thread %d\n",ind_ap,elthread);}
            int elzbin, tmpzbin, tmporder, thisind;
            int cumzcombi = 0;
            double toadd_Mapn, toadd_Mapn_var, toadd_Mapn_w;
            for (order=1; order<=max_order; order++){
                int *thiszcombi = calloc(order, sizeof(int));
                //if (ind_ap==ncenters/3){printf("\nNow doing order %d\n",order);}
                for (elzcombi=0; elzcombi<zcombis_order(nbinsz,order,factorials_zcombis); elzcombi++){
                    //printf("Building zcombis at order %d (combi %d) ",order,elzcombi);
                    // Compute Map^n and its weight for this zcombi 
                    // Do double counting corrs
                    //if (ind_ap==ncenters/3){for (int _i=0;_i<order;_i++){printf("%d ",thiszcombi[_i]);}}
                    if (do_subtractions){
                        toadd_Mapn = 1;
                        toadd_Mapn_var = 1;
                        if (order>1){
                            tmpzbin = thiszcombi[0];
                            tmporder = 0;
                            for (elzbin=1; elzbin<order; elzbin++){
                                if (thiszcombi[elzbin]==tmpzbin){tmporder+=1;}
                                else{
                                    //if (ind_ap==ncenters/3){printf(" Map^%i(z_%d)",tmporder+1,tmpzbin);}
                                    toadd_Mapn *= nextMapn[tmpzbin*max_order+tmporder];
                                    toadd_Mapn_var *= nextMapn_var[tmpzbin*max_order+tmporder];
                                    tmporder = 0;
                                    tmpzbin = thiszcombi[elzbin];
                                }
                            }
                            toadd_Mapn *= nextMapn[tmpzbin*max_order+tmporder];
                            toadd_Mapn_var *= nextMapn_var[tmpzbin*max_order+tmporder];
                            //if (ind_ap==ncenters/3){printf(" Map^%i(z_%d)\n",tmporder+1,tmpzbin);}
                        }
                        else{
                            toadd_Mapn = nextMapn[elzcombi*max_order+0];
                            toadd_Mapn_var = nextMapn_var[elzcombi*max_order+0];
                        }
                    }
                    // No subtractions
                    else{
                        toadd_Mapn = 1;
                        toadd_Mapn_var = 1;
                        for (elzbin=0; elzbin<order; elzbin++){
                            tmpzbin = thiszcombi[elzbin];
                            toadd_Mapn *= nextMapn[tmpzbin*max_order+0];
                            toadd_Mapn_var *= nextMapn_var[tmpzbin*max_order+0];
                        }
                    }
                    if (weight_method==0){toadd_Mapn_w = 1.;}
                    if ((weight_method==1) && (toadd_Mapn_var!=0)){toadd_Mapn_w = 1./toadd_Mapn_var;}
                    if ((weight_method==1) && (toadd_Mapn_var==0)){toadd_Mapn_w = 0;}
                    // Apply coverage cuts
                    for (elcov_cut=1;elcov_cut<=nfrac_cuts;elcov_cut++){  
                        if ((nextcovs[0]>fraccov_cuts[nfrac_cuts-elcov_cut])|| 
                            (nextcovs[1]>fraccov_cuts[nfrac_cuts-elcov_cut])){break;}
                        thisind = thisthreadshift + (nfrac_cuts-elcov_cut)*thread_nzcombis + cumzcombi;
                        tmpMapn[thisind] += toadd_Mapn_w*toadd_Mapn;
                        tmpwtot_Mapn[thisind] += toadd_Mapn_w;
                    }
                    nextzcombination(nbinsz, order, thiszcombi);
                    cumzcombi += 1;
                    
                    //if (ind_ap%1000==0){printf("ap:%d order:%d\n  toadd_Mapn_var:%.30f\n.  toadd_Mapn_w:%.30f \n\n",
                    //                           ind_ap,order,toadd_Mapn_var,toadd_Mapn_w);}
                }
                free(thiszcombi);
            }
            //printf("Finished ap %d on thread %d\n",ind_ap,elthread);
        }
        
        free(nextcounts);
        free(nextcovs);
        free(nextMsn);
        free(nextSn);
        free(nextSn_w);
        free(nextS2n_w);
        
        free(factorials_zcombis);
        free(factorials);
        free(bellargs_Msn);
        free(bellargs_Sn);
        free(bellargs_Sn_w);
        free(bellargs_S2n_w);
        free(nextMapn_singlez);
        free(nextMapn_norm_singlez);
        free(nextMapn_var_singlez);
        free(nextMapn_var_norm_singlez);
        free(nextMapn);
        free(nextMapn_var);
    }
    
    // Accumulate the Mapn across the threads
    #pragma omp parallel for num_threads(nthreads)
    for (int fzcombi=0; fzcombi<nfrac_cuts*nzcombis; fzcombi++){
        int thisind;
        for (int elthread=0; elthread<nthreads; elthread++){
            thisind = elthread*nfrac_cuts*nzcombis+fzcombi;
            Mapn[fzcombi] += tmpMapn[thisind];
            wtot_Mapn[fzcombi] += tmpwtot_Mapn[thisind];
        }
         Mapn[fzcombi] /= wtot_Mapn[fzcombi];
    }
    free(fac_zcombis);
    free(tmpMapn);
    free(tmpwtot_Mapn);
}

// Computes statistics on single aperture
void singleAp_MapnSingleEonlyDisc(
    double R_ap, double center_1, double center_2, 
    int max_order, int ind_filter, double weight_outer, double weight_inpainted, 
    double *weight, double *insurvey, double *pos1, double *pos2, double complex *g, int *zbins, int nbinsz, int ngal, 
    double *mask,
    double mask1_start, double mask2_start, double mask1_d, double mask2_d, int mask1_n, int mask2_n,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double *counts, double *covs, double *Msn, double *Sn, double *Sn_w, double *S2n_w){

    // Variables for counting components
    int order;
    // Helper precomputations
    double R2_ap=R_ap*R_ap;
    double max_d = mymax(mask1_d,mask2_d);
    double R2_ap_d = (R_ap + max_d) * (R_ap + max_d);
    
    // Variables used for updating the mask and selecting useful galaxies
    int lower, upper;
    double Qpix, mask_frac;
    double pix_center1, pix_center2, dpix_ap_sq;
    bool pixinap, badpixel;
    // All the indices
    int ind_pix1, ind_pix2, ind_raw, ind_red, ind_inpix, ind_gal, zbin;
    // Helper variables being used for the actual Mapn computation
    double complex phirotc_sq;
    double rel1, rel2, d2gal, w, wc, frac_insurvey, et, Qval;
    double tmp_et, tmp_w, tmp_wc, tmp_etmult;
    double tmp_norm, tmp_normw, tmp_normvol, fac_norm, fac_normw, fac_normvol;
    
    int npix = mask1_n*mask2_n; 
    double npix_m=0.; double Qpix_m=0; double npix_t=0.; double Qpix_t=0;
    double supp_filter = getFilterSupp(ind_filter);
    double supp_Q2 = supp_filter*supp_filter;
    //int pix1_lower = mymin( mask1_n-1, mymax(0, (int) floor((center_1 - (supp_filter*R_ap+mask1_d) - mask1_start)/mask1_d)));
    //int pix1_upper = mymin( mask1_n-1, mymax(0, (int) floor((center_1 + (supp_filter*R_ap+mask1_d) - mask1_start)/mask1_d)));
    //int pix2_lower = mymin( mask2_n-1, mymax(0, (int) floor((center_2 - (supp_filter*R_ap+mask2_d) - mask2_start)/mask2_d)));
    //int pix2_upper = mymin( mask2_n-1, mymax(0, (int) floor((center_2 + (supp_filter*R_ap+mask2_d) - mask2_start)/mask2_d)));
    int pix1_lower = (int) floor((center_1 - (supp_filter*R_ap+mask1_d) - mask1_start)/mask1_d);
    int pix1_upper = (int) floor((center_1 + (supp_filter*R_ap+mask1_d) - mask1_start)/mask1_d);
    int pix2_lower = (int) floor((center_2 - (supp_filter*R_ap+mask2_d) - mask2_start)/mask2_d);
    int pix2_upper = (int) floor((center_2 + (supp_filter*R_ap+mask2_d) - mask2_start)/mask2_d);
   
    // Compute Map statistics for this aperture
    //  1) Check if pixel is within suitable range of aperture
    //  2) If so, update aperture coverage and loop through galaxies in that pixel to update the sums for Mapn
    //  3) Finally, check if there were sufficiently many galaxies in the aperture; if not set result to  zero
    // As we loop through the pixels in the square encompassing the aperture
    // we needlessly compute distances for ~25% of pixels (subdominant!).
    for (ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
        for (ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
            
            pix_center1 = mask1_start + ind_pix1*mask1_d;
            pix_center2 = mask2_start + ind_pix2*mask2_d;
            dpix_ap_sq = (pix_center1 - center_1)*(pix_center1 - center_1) +
                         (pix_center2 - center_2)*(pix_center2 - center_2);

            // Only care about pixels within aperture
            if (dpix_ap_sq > supp_Q2*R2_ap_d){continue;}
            
            // These out of bounds cases will occur as we are searching within a
            // rectangle that exceeds the survey footprint
            // --> Treat those as fully masked and continue;
            ind_raw = ind_pix2*mask1_n + ind_pix1;
            badpixel = (ind_raw >= npix || ind_raw < 0 ||
                        ind_pix1<0 || ind_pix2<0 || 
                        ind_pix1>=mask1_n || ind_pix2>=mask2_n);
            // Update coverage fraction
            pixinap = dpix_ap_sq <= supp_Q2*R2_ap;
            Qpix = getFilterQ(ind_filter, dpix_ap_sq/R2_ap);
            if (badpixel && pixinap){
                npix_m+=1; Qpix_m+=Qpix;npix_t+=1; Qpix_t+=Qpix;} 
            if (!badpixel && pixinap){
                mask_frac = mask[ind_raw];
                npix_m+=mask_frac; Qpix_m+=mask_frac*Qpix; npix_t+=1; Qpix_t+=Qpix; }
            if (badpixel){continue;}

            // Go through the galaxies in the pixel
            ind_red = index_matcher[ind_raw];
            if (ind_red==-1){continue;}
            lower = pixs_galind_bounds[ind_red];
            upper = pixs_galind_bounds[ind_red+1];
            //printf("Go through galaxies in this pixel %d (%d %d)\n",ind_red,lower,upper);
            for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                ind_gal = pix_gals[ind_inpix];
                rel1 = pos1[ind_gal]-center_1;
                rel2 = pos2[ind_gal]-center_2;
                d2gal = (rel1*rel1 + rel2*rel2);
                if (d2gal > supp_Q2*R2_ap){continue;} 
                if (d2gal < 1e-5){continue;}
                //printf("Start allocating stuff for next galaxy\n");
                // Get tangential ellipticity and value of Q filter
                double frac_inpainted = 0.;
                frac_insurvey = insurvey[ind_gal];
                w = weight[ind_gal];
                wc = w * (
                    (1.-frac_inpainted)*(frac_insurvey + (1.-frac_insurvey)*weight_outer) +
                    frac_inpainted*weight_inpainted*(frac_insurvey + (1.-frac_insurvey)*weight_outer));
                phirotc_sq = (rel1*rel1-rel2*rel2-2*I*rel1*rel2)/d2gal;
                et = -creal(g[ind_gal]*phirotc_sq);
                Qval = getFilterQ(ind_filter, d2gal/R2_ap);
                //printf(" Got main quantities\n");
                // Update raw/weighted counts
                zbin = zbins[ind_gal];
                counts[zbin*3 + 0] += 1;
                counts[zbin*3 + 1] += w;
                counts[zbin*3 + 2] += wc;
                //printf(" Updated counts\n");
                // Update power sums
                tmp_et = 1;
                tmp_w = 1;
                tmp_wc = 1;
                tmp_etmult = w*Qval*et;
                for (order=0; order<max_order; order++){
                    tmp_w *= w;
                    tmp_wc *= wc;
                    tmp_et *= tmp_etmult;
                    
                    Msn[zbin*max_order+order] += tmp_et;
                    Sn[zbin*max_order+order] += tmp_w;
                    Sn_w[zbin*max_order+order] += tmp_wc;
                    S2n_w[zbin*max_order+order] += tmp_wc*tmp_wc;
                //printf(" Updated powersums. Done.\n");
                }
                //if ((ind_pix1==pix1_lower+(pix1_upper-pix1_lower)/2+2*(int)(R_ap/mask1_d))&&
                //    (ind_pix2==pix2_lower+(pix2_upper-pix2_lower)/2+2*(int)(R_ap/mask2_d))&&(ind_inpix==lower)){
                //    printf("w=%.4f  reff=%.4f  Q=%.4f  ang=%.4f+i%.4ff  et=%.4f  toadd=%.4f S1=%.4f  Ms0=%.4f\n", 
                //           w,sqrt(d2gal/R2_ap),Qval,creal(phirotc_sq),cimag(phirotc_sq),et,tmp_etmult, 
                //           Sn[zbin*max_order+1], Msn[zbin*max_order]);
               // }
            }
        }
    }
    //printf("R2=%.2f R2d=%.2f, maxd=%.4f, R=%.2f, Rext=%.4f counts=%.4f  S0=%.4f  Ms0=%.4f\n",
    //       R2_ap,R2_ap_d, max_d, R_ap, R_ap+max_d, counts[1], Sn[zbin*max_order], Msn[zbin*max_order]);
    
    // Raw and Q-weighted masked coverage fraction
    covs[0] = npix_m/npix_t;
    covs[1] = Qpix_m/Qpix_t;
    //printf("lower1=%d lower2=%d upper1=%d upper2=%d, npixm=%.2f, npixt=%.2f",
    //       pix1_lower,pix2_lower,pix1_upper,pix2_upper,npix_m,npix_t);
    //printf("covs=%.2f %.2f; counts=%.2f %.2f %.2f\n",covs[0],covs[1],counts[0],counts[1],counts[2]);
    
    // Renormalize the S_n/Ms_n components s.t. they match eqns (22),(23) in 2106.04594.
    // Note that if there are not sufficient galaxies to form closed n-side polygons,
    // the transformation equations will set those components to zero in the Mapn basis,
    // so we do not worry about this here.
    
    
    int thiscomp;
    for (zbin=0; zbin<nbinsz; zbin++){
        if (Sn[zbin*max_order]!=0){fac_norm = 1./(Sn[zbin*max_order]);}else{fac_norm = 0;}
        if (Sn_w[zbin*max_order]!=0){fac_normw = 1./(Sn_w[zbin*max_order]);}else{fac_normw = 0;}
        fac_normvol = supp_Q2;
        tmp_norm = 1;
        tmp_normw = 1.;///R_ap/R_ap;
        tmp_normvol = 1;
        for (order=0; order<max_order; order++){
            thiscomp = zbin*max_order+order;
            tmp_norm *= fac_norm;
            tmp_normw *= fac_normw;
            tmp_normvol *= fac_normvol;
            Msn[thiscomp] *= tmp_normvol*tmp_norm;
            Sn[thiscomp] *= tmp_norm;
            Sn_w[thiscomp] *= tmp_normw;
            S2n_w[thiscomp] *= tmp_normw*tmp_normw;
        }
    }
    
    //printf(" * covs=%.5f %.5f counts=%.5f\n",covs[0],counts[0]);
    //printf("After: R2=%.2f R2d=%.2f, maxd=%.4f, R=%.2f, Rext=%.4f counts=%.4f  S0=%.6f  S1=%.6f  Ms0=%.6f\n",
    //           R2_ap,R2_ap_d, max_d, R_ap, R_ap+max_d, counts[1], Sn[0*max_order], Sn[0*max_order+1], Msn[0*max_order]);
}

// Computes Napn for single aperture scale, taking into account the multiple-counting corrections.              
void NapnSingleDisc(
    double R_ap, double *centers_1, double *centers_2, int ncenters,
    int max_order, int ind_filter, int do_subtractions, int Nbar_policy, double weight_outer, double weight_inpainted, 
    double *weight, double *insurvey, double *pos1, double *pos2, double *tracer, int *zbins, int nbinsz, int ngal, 
    double *mask, double *fraccov_cuts, int nfrac_cuts, int fraccov_method,
    double mask1_start, double mask2_start, double mask1_d, double mask2_d, int mask1_n, int mask2_n,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    int nthreads, double *Napn, double *wtot_Napn){
    
    double *fac_zcombis = calloc(max_order+nbinsz+1, sizeof(double));
    gen_fac_table(max_order+nbinsz, fac_zcombis);
    int nzcombis = zcombis_tot(nbinsz, max_order, fac_zcombis);
    double *tmpNapn = calloc(nthreads*nfrac_cuts*nzcombis, sizeof(double));
    double *tmpwtot_Napn = calloc(nthreads*nfrac_cuts*nzcombis, sizeof(double));
    double *tmpcountstot_Napn = calloc(nthreads*nfrac_cuts*nbinsz, sizeof(double)); // Helper for allocating global Nbar
    double *countstot_Napn = calloc(nfrac_cuts*nbinsz, sizeof(double)); // Helper for allocating global Nbar
    
    #pragma omp parallel for num_threads(nthreads)
    for (int elthread=0; elthread<nthreads; elthread++){
        int ind_ap, elbinz, elzcombi, elcov_cut, order;
        double *nextcounts = calloc(3*nbinsz, sizeof(double));
        double *nextcovs = calloc(2, sizeof(double));
        double *nextMsn = calloc(max_order*nbinsz, sizeof(double));
        double *nextSn = calloc(max_order*nbinsz, sizeof(double));
        
        double *factorials_zcombis = calloc(max_order+nbinsz+1, sizeof(double));
        double *factorials = calloc(max_order+1, sizeof(double));
        double *bellargs_Msn = calloc(max_order, sizeof(double));
        double *bellargs_Sn = calloc(max_order, sizeof(double));
        double *nextNapn_singlez = calloc(max_order+1, sizeof(double));
        double *nextNapn_norm_singlez = calloc(max_order+1, sizeof(double));
        double *nextNapn = calloc(max_order*nbinsz, sizeof(double));
        
        gen_fac_table(max_order, factorials);
        gen_fac_table(max_order+nbinsz, factorials_zcombis);
        int thread_nzcombis = zcombis_tot(nbinsz, max_order, factorials_zcombis);
        int thisthreadshift = elthread*nfrac_cuts*thread_nzcombis;
        
        for (ind_ap=0; ind_ap<ncenters; ind_ap++){
            if ((ind_ap%nthreads)!=elthread){continue;}
            
            // Reset args to zeros
            nextcovs[0]=0;nextcovs[1]=0;
            for (int i=0; i<max_order*nbinsz; i++){
                nextMsn[i]=0;nextSn[i]=0;
            }
            for (int i=0; i<3*nbinsz; i++){nextcounts[i]=0;}
            
            // Get all the statistics of the aperture in power sum basis
            singleAp_NapnSingleDisc( R_ap, centers_1[ind_ap], centers_2[ind_ap], 
                max_order, ind_filter, Nbar_policy, weight_outer, weight_inpainted, 
                weight, insurvey, pos1, pos2, tracer, zbins, nbinsz, ngal,
                mask, mask1_start, mask2_start, mask1_d, mask2_d, mask1_n, mask2_n,
                index_matcher, pixs_galind_bounds, pix_gals,  
                nextcounts, nextcovs, nextMsn, nextSn);
            
            // Transform to Napn(zi)
            for (elbinz=0; elbinz<nbinsz; elbinz++){
                for (int i=0; i<max_order+1; i++){
                    nextNapn_singlez[i]=0; nextNapn_norm_singlez[i]=0; 
                    }
                int tmpind = elbinz*max_order+0;
                bellargs_Msn[0] = -nextMsn[tmpind];
                bellargs_Sn[0] = -nextSn[tmpind];
                for (order=1; order<max_order; order++){ 
                    tmpind += 1;
                    bellargs_Msn[order] = -factorials[order]*nextMsn[tmpind];
                    bellargs_Sn[order] = -factorials[order]*nextSn[tmpind];
                }
                getBellRecursive(max_order, bellargs_Msn, factorials, nextNapn_singlez);
                getBellRecursive(max_order, bellargs_Sn,  factorials, nextNapn_norm_singlez);
                for (order=0; order<max_order; order++){
                    if ((nextNapn_norm_singlez[order+1]!=0)){
                        nextNapn[elbinz*max_order+order] = pow(-1,order+1)*nextNapn_singlez[order+1];
                    }
                }   
            }
            
            // Build all zcombis
            // Nap^n(z_1,...,z_n) = \prod_{i=1}^nbinsz Nap^{alpha_i}(z_i)
            int elzbin, tmpzbin, tmporder, thisind;
            int cumzcombi = 0;
            double toadd_Napn, toadd_countsn, toadd_Napn_w;
            //double area_ap = getFilterSuppU(ind_filter)*getFilterSuppU(ind_filter)*R_ap*R_ap*M_PI;
            double area_ap = pow(getFilterSuppU(ind_filter),2)*R_ap*R_ap*M_PI;
            for (order=1; order<=max_order; order++){
                int *thiszcombi = calloc(order, sizeof(int));
                for (elzcombi=0; elzcombi<zcombis_order(nbinsz,order,factorials_zcombis); elzcombi++){
                    // Compute Nap^n and its weight for this zcombi 
                    // Do double counting corrs
                    if (do_subtractions){
                        toadd_Napn = 1;
                        toadd_countsn = 1;
                        if (order>1){
                            tmpzbin = thiszcombi[0];
                            tmporder = 0;
                            for (elzbin=1; elzbin<order; elzbin++){
                                if (thiszcombi[elzbin]==tmpzbin){tmporder+=1;}
                                else{
                                    toadd_Napn *= nextNapn[tmpzbin*max_order+tmporder];
                                    tmporder = 0;
                                    tmpzbin = thiszcombi[elzbin];
                                }
                            }
                            toadd_Napn *= nextNapn[tmpzbin*max_order+tmporder];
                        }
                        else{
                            toadd_Napn = nextNapn[elzcombi*max_order+0];
                        }
                    }
                    // No double counting corrs
                    else{
                        toadd_Napn = 1;
                        toadd_countsn = 1;
                        for (elzbin=0; elzbin<order; elzbin++){
                            tmpzbin = thiszcombi[elzbin];
                            toadd_Napn *= nextNapn[tmpzbin*max_order+0];
                        }
                    }
                    toadd_Napn_w = 1.; // Dummy for aperture weight that has to be unity
                    int ind_counts;
                    // Apply coverage cuts
                    for (elcov_cut=1;elcov_cut<=nfrac_cuts;elcov_cut++){  
                        if ((nextcovs[0]>fraccov_cuts[nfrac_cuts-elcov_cut])){break;}
                        thisind = thisthreadshift + (nfrac_cuts-elcov_cut)*thread_nzcombis + cumzcombi;
                        tmpNapn[thisind] += toadd_Napn_w*toadd_Napn;
                        tmpwtot_Napn[thisind] += toadd_Napn_w;
                        // Update global counts mean
                        if (order==1){
                            ind_counts = elthread*nfrac_cuts*nbinsz+(nfrac_cuts-elcov_cut)*nbinsz+elzcombi;
                            tmpcountstot_Napn[ind_counts] += nextcounts[elzcombi*3+1]/area_ap;
                        }
                    }
                    nextzcombination(nbinsz, order, thiszcombi);
                    cumzcombi += 1;
                }
                free(thiszcombi);
            }            
        }
        
        free(nextcounts);
        free(nextcovs);
        free(nextMsn);
        free(nextSn);
        
        free(factorials_zcombis);
        free(factorials);
        free(bellargs_Msn);
        free(bellargs_Sn);
        free(nextNapn_singlez);
        free(nextNapn_norm_singlez);
        free(nextNapn);   
    }
    
    // Get the global Nbar per redshift bin
    for (int elcov_cut=1;elcov_cut<=nfrac_cuts;elcov_cut++){
        for (int elbinz=0; elbinz<nbinsz; elbinz++){
            for (int elthread=0; elthread<nthreads; elthread++){
                int fzcombi = (nfrac_cuts-elcov_cut)*nbinsz+elbinz;
                countstot_Napn[fzcombi] += tmpcountstot_Napn[elthread*nfrac_cuts*nbinsz+fzcombi];
            }
        }
    }
    // Accumulate the Napn across the threads
    //#pragma omp parallel for num_threads(nthreads)
    int cumzcombi = 0;
    int thread_nzcombis = zcombis_tot(nbinsz, max_order, fac_zcombis);
    for (int order=1; order<=max_order; order++){
        int *thiszcombi = calloc(order, sizeof(int));
        for (int elzcombi=0; elzcombi<zcombis_order(nbinsz,order,fac_zcombis); elzcombi++){
            for (int elcov_cut=1;elcov_cut<=nfrac_cuts;elcov_cut++){ 
                int fzcombi = (nfrac_cuts-elcov_cut)*thread_nzcombis + cumzcombi;
                // Recover global Napn and number of apertures
                for (int elthread=0; elthread<nthreads; elthread++){
                    int thisind = elthread*nfrac_cuts*nzcombis+fzcombi;
                    Napn[fzcombi] += tmpNapn[thisind];
                    wtot_Napn[fzcombi] += tmpwtot_Napn[thisind];
                }
                // Recover global mean of Nbar(z1) \cdots Nbar(zk)
                double thiscountstot = 1;
                for (int elzbin=0; elzbin<order; elzbin++){
                    thiscountstot *= countstot_Napn[(nfrac_cuts-elcov_cut)*nbinsz+thiszcombi[elzbin]];
                }
                // * Local/no nbar policy
                //   We have N_ap^n/(Nbar_ap)^n per aperture --> Only need to divide by number of apertures
                // * Global nbar policy
                //   * We computed the nbar per aperture, summed over all apertures in 
                //     countstot_Napn = sum_ap nextcounts_ap(z_i) ~ Num_aps*<Nbar(z_i)>
                //   * We computed for each zcombination the product 
                //     thiscountstot = countstot_Napn(z_i1) \cdots countstot_Napn(z_il) 
                //                     ~ Num_aps^l* <Nbar(z_i1)> \cdots <Nbar(z_il)>
                //   * As our basis we have Napn = sum_ap Nap^n 
                //                               ~ Num_aps * <Nap'^n> 
                //   * We want to get 
                //     <Nap^n(zi1,...,z_il)> 
                //           = 1/(<Nbar(z_i1)> \cdots <Nbar(z_il)>) * <Nap'^n>
                //           = 1/(Num_aps^l*<Nbar(z_i1)> \cdots <Nbar(z_il)>) * (Num_aps*<Nap'^n>) * Num_aps^(l-1)
                //           ~ wtot_Napn^(l-1) * Napn/thiscountstot        
                if ((Nbar_policy==0) || (Nbar_policy==2)){Napn[fzcombi] /= wtot_Napn[fzcombi];} 
                else if (Nbar_policy==1){
                    Napn[fzcombi] /= thiscountstot/pow(wtot_Napn[fzcombi],order-1);
                } // Use global Nbar policy
                //printf("%.5f ", countstot_Napn[fzcombi]/wtot_Napn[fzcombi]);
            }
            nextzcombination(nbinsz, order, thiszcombi);
            cumzcombi += 1;
        }
        free(thiszcombi);
    }
    
    free(tmpNapn);
    free(tmpwtot_Napn);
    free(tmpcountstot_Napn);
    free(countstot_Napn);
    free(fac_zcombis);
}

// Computes statistics on single aperture
void singleAp_NapnSingleDisc(
    double R_ap, double center_1, double center_2, 
    int max_order, int ind_filter, int Nbar_policy, double weight_outer, double weight_inpainted, 
    double *weight, double *insurvey, double *pos1, double *pos2, double *tracer, int *zbins, int nbinsz, int ngal, 
    double *mask,
    double mask1_start, double mask2_start, double mask1_d, double mask2_d, int mask1_n, int mask2_n,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double *counts, double *covs, double *Msn, double *Sn){

    // Variables for counting components
    int order;
    // Helper precomputations
    double R2_ap=R_ap*R_ap;
    double max_d = mymax(mask1_d,mask2_d);
    double R2_ap_d = (R_ap + max_d) * (R_ap + max_d);
    
    // Variables used for updating the mask and selecting useful galaxies
    int lower, upper;
    double Upix, mask_frac;
    double pix_center1, pix_center2, dpix_ap_sq;
    bool pixinap, badpixel;
    // All the indices
    int ind_pix1, ind_pix2, ind_raw, ind_red, ind_inpix, ind_gal, zbin;
    // Helper variables being used for the actual Mapn computation
    double complex phirotc_sq;
    double rel1, rel2, d2gal, w, tr, frac_insurvey, Uval;
    double tmp_n, tmp_w, tmp_nmult, tmp_wmult;
    double tmp_norm, tmp_normw, tmp_normvol, fac_norm, fac_normvol;
    
    int npix = mask1_n*mask2_n; 
    double npix_m=0.; double Upix_m=0; double npix_t=0.; double Upix_t=0;
    double supp_filter = getFilterSuppU(ind_filter);
    double supp_U2 = supp_filter*supp_filter;
    int pix1_lower = mymax(0, (int) floor((center_1 - (supp_filter*R_ap+mask1_d) - mask1_start)/mask1_d));
    int pix1_upper = mymin(mask1_n-1, (int) floor((center_1 + (supp_filter*R_ap+mask1_d) - mask1_start)/mask1_d));
    int pix2_lower = mymax(0, (int) floor((center_2 - (supp_filter*R_ap+mask2_d) - mask2_start)/mask2_d));
    int pix2_upper = mymin(mask2_n-1, (int) floor((center_2 + (supp_filter*R_ap+mask2_d) - mask2_start)/mask2_d));
   
    // Compute Nap statistics for this aperture
    //  1) Check if pixel is within suitable range of aperture
    //  2) If so, update aperture coverage and loop through galaxies in that pixel to update the sums for Mapn
    //  3) Finally, check if there were sufficiently many galaxies in the aperture; if not set result to  zero
    // As we loop through the pixels in the square encompassing the aperture
    // we needlessly compute distances for ~25% of pixels (subdominant!).
    for (ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
        for (ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
            
            pix_center1 = mask1_start + ind_pix1*mask1_d;
            pix_center2 = mask2_start + ind_pix2*mask2_d;
            dpix_ap_sq = (pix_center1 - center_1)*(pix_center1 - center_1) +
                         (pix_center2 - center_2)*(pix_center2 - center_2);

            // Only care about pixels within aperture
            if (dpix_ap_sq > supp_U2*R2_ap_d){continue;}
            
            // These out of bounds cases will occur as we are searching within a
            // rectangle that exceeds the survey footprint
            // --> Treat those as fully masked and continue;
            ind_raw = ind_pix2*mask1_n + ind_pix1;
            badpixel = (ind_raw >= npix || ind_raw < 0 ||
                        ind_pix1<0 || ind_pix2<0 || 
                        ind_pix1>=mask1_n || ind_pix2>=mask2_n);
            // Update coverage fraction
            pixinap = dpix_ap_sq <= supp_U2*R2_ap;
            Upix = getFilterU(ind_filter, dpix_ap_sq/R2_ap);
            if (badpixel && pixinap){
                npix_m+=1; Upix_m+=Upix;npix_t+=1; Upix_t+=Upix;} 
            if (!badpixel && pixinap){
                mask_frac = mask[ind_raw];
                npix_m+=mask_frac; Upix_m+=mask_frac*Upix; npix_t+=1; Upix_t+=Upix; }
            if (badpixel){continue;}

            // Go through the galaxies in the pixel
            ind_red = index_matcher[ind_raw];
            if (ind_red==-1){continue;}
            lower = pixs_galind_bounds[ind_red];
            upper = pixs_galind_bounds[ind_red+1];
            for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                ind_gal = pix_gals[ind_inpix];
                rel1 = pos1[ind_gal]-center_1;
                rel2 = pos2[ind_gal]-center_2;
                d2gal = (rel1*rel1 + rel2*rel2);
                if (d2gal > supp_U2*R2_ap){continue;} 
                if (d2gal < 1e-15){continue;}
                // Get relevant base quantities
                w = weight[ind_gal];
                tr = tracer[ind_gal];
                frac_insurvey = insurvey[ind_gal];
                Uval = getFilterU(ind_filter, d2gal/R2_ap)/R2_ap;
                // Update raw/weighted counts
                zbin = zbins[ind_gal];
                counts[zbin*3 + 0] += 1;
                counts[zbin*3 + 1] += w;
                counts[zbin*3 + 2] += w*frac_insurvey;
                // Update power sums
                tmp_w = 1;
                tmp_n = 1;
                tmp_wmult = w;
                tmp_nmult = tr*Uval;
                for (order=0; order<max_order; order++){
                    tmp_w *= tmp_wmult;
                    tmp_n *= tmp_nmult;
                    Msn[zbin*max_order+order] += tmp_n;
                    Sn[zbin*max_order+order] += tmp_w;
                }
            }
        }
    }
    
    // Raw and Q-weighted masked coverage fraction
    covs[0] = npix_m/npix_t;
    covs[1] = Upix_m/Upix_t;
    // Renormalize the Ms_n to account for the Nbar in the estimator of Schneider 1998.
    // Note that if there are not sufficient galaxies to form closed n-side polygons,
    // the transformation equations will set those components to zero in the Mapn basis,
    // so we do not worry about this here.
    int thiscomp;
    double inv_Nbar;
    for (zbin=0; zbin<nbinsz; zbin++){
        if ((Nbar_policy==0) && (counts[zbin*3+1]>0)){inv_Nbar = M_PI*supp_U2*R2_ap/counts[zbin*3+1];}
        else if ((Nbar_policy==0) && (counts[zbin*3+1]==0)){inv_Nbar=0;}
        else if ((Nbar_policy==1) || (Nbar_policy==2)){inv_Nbar=1;}
        fac_norm = 1;
        fac_normvol = inv_Nbar;
        tmp_norm = 1;
        tmp_normvol = 1;
        for (order=0; order<max_order; order++){
            thiscomp = zbin*max_order+order;
            tmp_norm *= fac_norm;
            tmp_normvol *= fac_normvol;
            Msn[thiscomp] *= tmp_normvol*tmp_norm;
            Sn[thiscomp] *= tmp_norm;
        }
    }
}


/*
// Compute MapNap, Nap2, Map2, Nap2Map, NapMap2, Nap3Map, Nap2Map2, NapMap3 and its disconnected part for two distinct radii
// output = [Nap2, MapNap, Map2, Map3, Nap2Map, NapMap2, Map3, Nap4, Nap3Map, Nap2Map2, NapMap3, Map4]
// output_weights = [w_L2, w_LS, w_S2, w_L2S, w_LS2, w_L3S, w_L2S2, w_LS3]
void NapmMapm_fourth_TwoScaleEonlyDisc_Notomo(
    double R_ap_lens, double R_ap_source, double *centers_1, double *centers_2, int ncenters,
    int ind_filter, int weight_method, int do_subtractions,
    double *mask, double *fraccov_cuts, int nfrac_cuts, int fraccov_method,
    double mask1_start, double mask2_start, double mask1_d, double mask2_d, int mask1_n, int mask2_n,
    double *weight_l, double *insurvey_l, double *pos1_l, double *pos2_l, double *kappa_g, 
    int ngal_l, int *index_matcher_l, int *pixs_galind_bounds_l, int *pix_gals_l,
    double *weight_s, double *insurvey_s, double *pos1_s, double *pos2_s, double complex *g, 
    int ngal_s, int *index_matcher_s, int *pixs_galind_bounds_s, int *pix_gals_s,
    int nthreads,
    double *output, double *output_weights){
    
    double *tmpoutput = calloc(nthreads*12*nfrac_cuts, sizeof(double));
    double *tmpoutput_weights = calloc(nthreads*12*nfrac_cuts, sizeof(double));
    
    for (int elthread=0; elthread<nthreads; elthread++){
        
        int ind_ap, elcov_cut, order, order_s, order_l;
        double *nextcounts_s = calloc(3, sizeof(double));
        double *nextcounts_l = calloc(3, sizeof(double));
        double *nextcovs_s = calloc(2, sizeof(double));
        double *nextcovs_l = calloc(2, sizeof(double));
        double *nextMsn_s = calloc(max_order, sizeof(double));
        double *nextSn_s = calloc(max_order, sizeof(double));
        double *nextSn_s_w = calloc(max_order, sizeof(double));
        double *nextS2n_s_w = calloc(max_order, sizeof(double));
        double *nextMsn_l = calloc(max_order, sizeof(double));
        double *nextSn_l = calloc(max_order, sizeof(double));
        double *nextSn_l_w = calloc(max_order, sizeof(double));
        
        double *factorials = calloc(max_order+1, sizeof(double));
        double *bellargs_Msn_s = calloc(max_order, sizeof(double));
        double *bellargs_Sn_s = calloc(max_order, sizeof(double));
        double *bellargs_Sn_s_w = calloc(max_order, sizeof(double));
        double *bellargs_S2n_s_w = calloc(max_order, sizeof(double));
        double *bellargs_Msn_l = calloc(max_order, sizeof(double));
        double *bellargs_Sn_l = calloc(max_order, sizeof(double));
        double *nextNapm_tmp = calloc(max_order+1, sizeof(double));
        double *nextMapn_tmp = calloc(max_order+1, sizeof(double));
        double *nextMapn_norm_tmp = calloc(max_order+1, sizeof(double));
        double *nextMapn_var_tmp = calloc(max_order+1, sizeof(double));
        double *nextMapn_var_norm_tmp = calloc(max_order+1, sizeof(double));
        double *nextNapm = calloc(max_order, sizeof(double));
        double *nextMapn = calloc(max_order, sizeof(double));
        double *nextMapn_var = calloc(max_order, sizeof(double));
        
        gen_fac_table(max_order, factorials);
        int thisthreadshift = elthread*nfrac_cuts*thread_nzcombis;
        
        
        //printf("%d %d \n ",ncenters, thread_nzcombis);
        //printf("Done preps for parallel region %d\n",elthread);
        for (ind_ap=0; ind_ap<ncenters_1*ncenters_2; ind_ap++){
            if ((ind_ap%nthreads)!=elthread){continue;}
            int c1 = ind_ap%ncenters_1;
            int c2 = ind_ap/ncenters_1;
            
            
            //if (elthread==0){printf("Start setting stuff to zero for ap %d on thread %d\n",ind_ap,elthread);}
            // Reset args to zeros
            nextcovs[0]=0;nextcovs[1]=0;
            for (int i=0; i<3; i++){nextcounts[i]=0;}
            for (int i=0; i<max_order; i++){
                nextMsn_s[i]=0;nextSn_s[i]=0;nextSn_w[i]=0;nextS2n_w[i]=0;
            }
                    
            // Get all the statistics of the aperture in power sum basis
            //if (elthread==0){printf("Get power sums from ap %d on thread %d\n",ind_ap,elthread);}
            singleAp_MapnSingleEonlyDisc( R_ap, centers_1[c1], centers_2[c2], 
                max_order, ind_filter, 
                weight, insurvey, pos1, pos2, g, zbins, 1, ngal,
                mask, mask1_start, mask2_start, mask1_d, mask2_d, mask1_n, mask2_n,
                index_matcher, pixs_galind_bounds, pix_gals,  
                nextcounts, nextcovs, nextMsn, nextSn, nextSn_w, nextS2n_w);
        }
        
    }
    
    
    
}

// Computes NapmMapn for single aperture scale, taking into account the multiple-counting corrections.
// Weight methods:
//  * 0 --> Identity weights
//  * 1 --> Inverse shape noise weights (for Map-part of stats) 
// We define z_{1_l}<=z_{2_l}<=...z_{zbinsl_l}<=z_{1_s}<=z_{2_s}<=...<=z_{zbins_s}
void NapmMapnSingleEonlyDisc(
    double R_ap, double *centers_1, double *centers_2, int ncenters,
    int max_order, int ind_filter, int weight_method, int do_subtractions,
    double *mask, double *fraccov_cuts, int nfrac_cuts, int fraccov_method,
    double mask1_start, double mask2_start, double mask1_d, double mask2_d, int mask1_n, int mask2_n,
    double *weight_l, double *insurvey_l, double *pos1_l, double *pos2_l, double *kappa_g, int *zbins_l, 
    int nbinsz_l, int ngal_l, int *index_matcher_l, int *pixs_galind_bounds_l, int *pix_gals_l,
    double *weight_s, double *insurvey_s, double *pos1_s, double *pos2_s, double complex *g, int *zbins_s, 
    int nbinsz_s, int ngal_s, int *index_matcher_s, int *pixs_galind_bounds_s, int *pix_gals_s,
    int nthreads, double *NapmMapn, double *wtot_NapmMapn){
    
    //printf("Starting\n");
    int _nbinsz = nbinsz_l+nbinsz_s;
    double *fac_zcombis = calloc(max_order+_nbinsz+1, sizeof(double));
    gen_fac_table(max_order+_nbinsz, fac_zcombis);
    int nzcombis = zcombis_tot(_nbinsz, max_order, fac_zcombis);
    
    // shape (nthreads, nfrac_cuts, nzcombis)
    double *tmpNapmMapn = calloc(nthreads*nfrac_cuts*nzcombis, sizeof(double));
    double *tmpwtot_NapmMapn = calloc(nthreads*nfrac_cuts*nzcombis, sizeof(double));
    //printf("Now entering parallel region\n");
    
    #pragma omp parallel for num_threads(nthreads)
    for (int elthread=0; elthread<nthreads; elthread++){
        //printf("Entered parallel region %d\n",elthread);
        int ind_ap, elbinz, elzcombi_l, elzcombi_s, elcov_cut, order, order_s, order_l;
        int nbinsz = nbinsz_l+nbinsz_s;
        double *nextcounts_s = calloc(3*nbinsz_s, sizeof(double));
        double *nextcounts_l = calloc(3*nbinsz_l, sizeof(double));
        double *nextcovs_s = calloc(2, sizeof(double));
        double *nextcovs_l = calloc(2, sizeof(double));
        double *nextMsn_s = calloc(max_order*nbinsz_s, sizeof(double));
        double *nextSn_s = calloc(max_order*nbinsz_s, sizeof(double));
        double *nextSn_s_w = calloc(max_order*nbinsz_s, sizeof(double));
        double *nextS2n_s_w = calloc(max_order*nbinsz_s, sizeof(double));
        double *nextMsn_l = calloc(max_order*nbinsz_l, sizeof(double));
        double *nextSn_l = calloc(max_order*nbinsz_l, sizeof(double));
        double *nextSn_l_w = calloc(max_order*nbinsz_l, sizeof(double));
        
        double *factorials_zcombis = calloc(max_order+nbinsz+1, sizeof(double));
        double *factorials = calloc(max_order+1, sizeof(double));
        double *bellargs_Msn_s = calloc(max_order, sizeof(double));
        double *bellargs_Sn_s = calloc(max_order, sizeof(double));
        double *bellargs_Sn_s_w = calloc(max_order, sizeof(double));
        double *bellargs_S2n_s_w = calloc(max_order, sizeof(double));
        double *bellargs_Msn_l = calloc(max_order, sizeof(double));
        double *bellargs_Sn_l = calloc(max_order, sizeof(double));
        double *nextNapm_singlez = calloc(max_order+1, sizeof(double));
        double *nextNapm_norm_singlez = calloc(max_order+1, sizeof(double));
        double *nextMapn_singlez = calloc(max_order+1, sizeof(double));
        double *nextMapn_norm_singlez = calloc(max_order+1, sizeof(double));
        double *nextMapn_var_singlez = calloc(max_order+1, sizeof(double));
        double *nextMapn_var_norm_singlez = calloc(max_order+1, sizeof(double));
        double *nextNapm = calloc(max_order*nbinsz_l, sizeof(double));
        double *nextMapn = calloc(max_order*nbinsz_s, sizeof(double));
        double *nextMapn_var = calloc(max_order*nbinsz_s, sizeof(double));
        
        gen_fac_table(max_order, factorials);
        gen_fac_table(max_order+nbinsz, factorials_zcombis);
        int thread_nzcombis = zcombis_tot(nbinsz, max_order, factorials_zcombis);
        int thisthreadshift = elthread*nfrac_cuts*thread_nzcombis;
        
        //printf("%d %d \n ",ncenters, thread_nzcombis);
        //printf("Done preps for parallel region %d\n",elthread);
        for (ind_ap=0; ind_ap<ncenters; ind_ap++){
            if ((ind_ap%nthreads)!=elthread){continue;}
            //if (elthread==0){printf("Starting ap %d/%d on thread %d\n",ind_ap+1,ncenters,elthread);}
            if (elthread==0){
                //for (order=1; order<=max_order; order++){
                //    printf("%d %d %d %d %d %d \n ",
                //           ncenters, ind_ap, nbinsz, order, factorials_zcombis[nbinsz+order-1],
                //           zcombis_order(nbinsz, order, factorials_zcombis));
                //}
                //printf("Tot: %d \n ", thread_nzcombis);
            }
            
            //if (elthread==0){printf("Start setting stuff to zero for ap %d on thread %d\n",ind_ap,elthread);}
            // Reset args to zeros
            nextcovs_s[0]=0;nextcovs_s[1]=0;
            for (int i=0; i<max_order*nbinsz_s; i++){
                nextMsn_s[i]=0;nextSn_s[i]=0;nextSn_s_w[i]=0;nextS2n_s_w[i]=0;}
            for (int i=0; i<max_order*nbinsz_l; i++){
                nextMsn_l[i]=0;nextSn_l[i]=0;nextSn_l_w[i]=0;}
            for (int i=0; i<3*nbinsz_s; i++){nextcounts_s[i]=0;}
            for (int i=0; i<3*nbinsz_l; i++){nextcounts_l[i]=0;}
            
            // Get all the statistics of the aperture in power sum basis
            //if (elthread==0){printf("Get power sums from ap %d on thread %d\n",ind_ap,elthread);}
            singleAp_MapnSingleEonlyDisc( R_ap, centers_1[ind_ap], centers_2[ind_ap], 
                max_order, ind_filter, 
                weight_s, insurvey_s, pos1_s, pos2_s, g, zbins_s, nbinsz_s, ngal_s,
                mask, mask1_start, mask2_start, mask1_d, mask2_d, mask1_n, mask2_n,
                index_matcher_s, pixs_galind_bounds_s, pix_gals_s,  
                nextcounts_s, nextcovs_s, nextMsn_s, nextSn_s, nextSn_s_w, nextS2n_s_w);
            singleAp_NapnSingleEonlyDisc( R_ap, centers_1[ind_ap], centers_2[ind_ap], 
                max_order, ind_filter, 
                weight_l, insurvey_l, pos1_l, pos2_l, kappa_g, zbins_l, nbinsz_l, ngal_l,
                mask, mask1_start, mask2_start, mask1_d, mask2_d, mask1_n, mask2_n,
                index_matcher_l, pixs_galind_bounds_l, pix_gals_l,  
                nextcounts_l, nextcovs_l, nextMsn_l, nextSn_l);

            //if (elthread==0){printf("Got stats for ap %d on thread %d\n",ind_ap,elthread);}
            //for (order=0; order<max_order; order++){
            //printf("For aperture # %d: x/y=%.3f/%.3f, cov=%.3f/%.3f, counts=%.3f, Ms_%d=%.10f S_%d=%.10f \n",
            //                   ind_ap, centers_1[ind_ap], centers_2[ind_ap], nextcovs[0], nextcovs[0], nextcounts[0], order,
            //                   nextMsn[0*max_order+order],order,nextSn[0*max_order+order]);}
                
            // Transform to Napm(zi) & Mapn(zi)
            //if (elthread==0){printf("Transforming to Mapn %d on thread %d\n",ind_ap,elthread);}
            for (elbinz=0; elbinz<nbinsz_l; elbinz++){
                for (int i=0; i<max_order+1; i++){
                    nextNapm_singlez[i]=0; nextNapm_norm_singlez[i]=0; 
                    }
                int tmpind = elbinz*max_order+0;
                bellargs_Msn_l[0] = -nextMsn_l[tmpind];
                bellargs_Sn_l[0] = -nextSn_l[tmpind];
                for (order=1; order<max_order; order++){ 
                    tmpind += 1;
                    bellargs_Msn_l[order] = -factorials[order]*nextMsn_l[tmpind];
                    bellargs_Sn_l[order] = -factorials[order]*nextSn_l[tmpind];
                }
                getBellRecursive(max_order, bellargs_Msn_l, factorials, nextNapm_singlez);
                getBellRecursive(max_order, bellargs_Sn_l,  factorials, nextNapm_norm_singlez);
                for (order=0; order<max_order; order++){
                    if (nextNapm_norm_singlez[order+1]!=0){
                        nextNapm[elbinz*max_order+order] = nextNapm_singlez[order+1];
                    }
                }   
            }
            for (elbinz=0; elbinz<nbinsz_s; elbinz++){
                for (int i=0; i<max_order+1; i++){
                    nextMapn_singlez[i]=0; nextMapn_norm_singlez[i]=0; 
                    nextMapn_var_singlez[i]=0; nextMapn_var_norm_singlez[i]=0;
                    }
                int tmpind = elbinz*max_order+0;
                bellargs_Msn_s[0] = -nextMsn_s[tmpind];
                bellargs_Sn_s[0] = -nextSn_s[tmpind];
                bellargs_Sn_s_w[0] = -nextSn_s_w[tmpind];
                bellargs_S2n_s_w[0] = -nextS2n_s_w[tmpind];
                for (order=1; order<max_order; order++){ 
                    tmpind += 1;
                    bellargs_Msn_s[order] = -factorials[order]*nextMsn_s[tmpind];
                    bellargs_Sn_s[order] = -factorials[order]*nextSn_s[tmpind];
                    bellargs_Sn_s_w[order] = -factorials[order]*nextSn_s_w[tmpind];
                    bellargs_S2n_s_w[order] = -factorials[order]*nextS2n_s_w[tmpind];
                }
                getBellRecursive(max_order, bellargs_Msn_s, factorials, nextMapn_singlez);
                getBellRecursive(max_order, bellargs_Sn_s,  factorials, nextMapn_norm_singlez);
                getBellRecursive(max_order, bellargs_S2n_s_w, factorials, nextMapn_var_singlez);
                getBellRecursive(max_order, bellargs_Sn_s_w,  factorials, nextMapn_var_norm_singlez);
                for (order=0; order<max_order; order++){
                    if ((nextMapn_norm_singlez[order+1]!=0)&&(nextMapn_var_norm_singlez[order+1]!=0)){
                        nextMapn[elbinz*max_order+order] = nextMapn_singlez[order+1]/nextMapn_norm_singlez[order+1];
                        nextMapn_var[elbinz*max_order+order] = nextMapn_var_singlez[order+1] / (
                            nextMapn_var_norm_singlez[order+1]*abs(nextMapn_var_norm_singlez[order+1]));
                    }
                }   
            }
            
            // Build all zcombis a la
            // Nap^mMap^n(z_1,...,z_m,z'_1,...,z'_n) = 
            //       \prod_{i=1}^nbinsz_s Map^{alpha_i}(z'_i) \prod_{j=1}^nbinsz_l Map^{alpha_j}(z_j)
            //if (elthread==0){printf("Allocating zcombis for ap %d on thread %d\n",ind_ap,elthread);}
            int thisind;
            int cumzcombi = 0;
            double toadd_Napm, toadd_Napm_w;
            double toadd_Mapn, toadd_Mapn_var, toadd_Mapn_w;
            for (order=1; order<=max_order; order++){
                for (order_s=0; order_s<=order; order_s++){
                    order_l = order-order_s;
                    //if (ind_ap==ncenters/3){printf("\nNow doing order %d\n",order);}
                    int *thiszcombi_l = calloc(order_s, sizeof(int));
                    for (elzcombi_l=0; elzcombi_l<zcombis_order(nbinsz_l,order_l,factorials_zcombis); elzcombi_l++){
                        toadd_Napm = nexttomoterm(order_l, max_order, nextNapm, 
                                                  thiszcombi_l, elzcombi_l, do_subtractions);
                        int *thiszcombi_s = calloc(order_s, sizeof(int));
                        for (elzcombi_s=0; elzcombi_s<zcombis_order(nbinsz_s,order_s,factorials_zcombis); elzcombi_s++){
                            // Compute Map^n and its weight for this zcombi 
                            toadd_Mapn = nexttomoterm(order_s, max_order, nextMapn, 
                                                  thiszcombi_s, elzcombi_s, do_subtractions);
                            toadd_Mapn_var = nexttomoterm(order_s, max_order, nextMapn_var, 
                                                          thiszcombi_s, elzcombi_s, do_subtractions);
                            // Get aperture weights
                            toadd_Napm_w = 1.;
                            toadd_Mapn_w = 1.;
                            if (weight_method==0){toadd_Mapn_w = 1.;}
                            if ((weight_method==1) && (toadd_Mapn_var!=0)){toadd_Mapn_w = 1./toadd_Mapn_var;}
                            if ((weight_method==1) && (toadd_Mapn_var==0)){toadd_Mapn_w = 0;}
                            
                            // Apply coverage cuts
                            for (elcov_cut=1;elcov_cut<=nfrac_cuts;elcov_cut++){  
                                if ((nextcovs_s[0]>fraccov_cuts[nfrac_cuts-elcov_cut])|| 
                                    (nextcovs_s[1]>fraccov_cuts[nfrac_cuts-elcov_cut])){break;}
                                thisind = thisthreadshift + (nfrac_cuts-elcov_cut)*thread_nzcombis + cumzcombi;
                                tmpNapmMapn[thisind] += toadd_Napm_w*toadd_Mapn_w*toadd_Napm*toadd_Mapn;
                                tmpwtot_NapmMapn[thisind] += toadd_Napm_w*toadd_Mapn_w;
                            }
                            nextzcombination(nbinsz_s, order_s, thiszcombi_s);
                            cumzcombi += 1;
                        }
                        free(thiszcombi_s);
                        nextzcombination(nbinsz_l, order_l, thiszcombi_l);
                    }
                    free(thiszcombi_l);
                }
                //printf("Finished ap %d on thread %d\n",ind_ap,elthread);
            }
        }
        
        free(nextcounts_s);
        free(nextcovs_s);
        free(nextMsn_s);
        free(nextSn_s);
        free(nextSn_s_w);
        free(nextS2n_s_w);
        free(nextcounts_l);
        free(nextcovs_l);
        free(nextMsn_l);
        free(nextSn_l);
        
        free(factorials_zcombis);
        free(factorials);
        free(bellargs_Msn_s);
        free(bellargs_Sn_s);
        free(bellargs_Sn_s_w);
        free(bellargs_S2n_s_w);
        free(bellargs_Msn_l);
        free(bellargs_Sn_l);
        free(nextMapn_singlez);
        free(nextMapn_norm_singlez);
        free(nextMapn_var_singlez);
        free(nextMapn_var_norm_singlez);
        free(nextMapn);
        free(nextMapn_var);
        free(nextNapm_singlez);
        free(nextNapm_norm_singlez);
        free(nextNapm);
    }
    
    // Accumulate the Mapn across the threads 
    #pragma omp parallel for num_threads(nthreads)
    for (int fzcombi=0; fzcombi<nfrac_cuts*nzcombis; fzcombi++){
        int thisind;
        for (int elthread=0; elthread<nthreads; elthread++){
            thisind = elthread*nfrac_cuts*nzcombis+fzcombi;
            NapmMapn[fzcombi] += tmpNapmMapn[thisind];
            wtot_NapmMapn[fzcombi] += tmpwtot_NapmMapn[thisind];
        }
         NapmMapn[fzcombi] /= wtot_NapmMapn[fzcombi];
    }
    free(fac_zcombis);
    free(tmpNapmMapn);
    free(tmpwtot_NapmMapn);
}
*/

/*
// Computes statistics on single aperture
void singleAp_NapnSingleEonlyDisc(
    double R_ap, double center_1, double center_2, 
    int max_order, int ind_filter, 
    double *weight, double *insurvey, double *pos1, double *pos2, int *zbins, int nbinsz, int ngal, double *nbar_zs, 
    double *mask,
    double mask1_start, double mask2_start, double mask1_d, double mask2_d, int mask1_n, int mask2_n,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double *counts, double *covs, double *Msn){

    // Variables for counting components
    int order;
    // Helper precomputations
    double R2_ap=R_ap*R_ap;
    double max_d = mymax(mask1_d,mask2_d);
    double R2_ap_d = (R_ap + max_d) * (R_ap + max_d);
    
    // Variables used for updating the mask and selecting useful galaxies
    int lower, upper;
    double Upix, mask_frac;
    double pix_center1, pix_center2, dpix_ap_sq;
    bool pixinap, badpixel;
    // All the indices
    int ind_pix1, ind_pix2, ind_raw, ind_red, ind_inpix, ind_gal, zbin;
    // Helper variables being used for the actual Mapn computation
    double complex phirotc_sq;
    double rel1, rel2, d2gal, w, frac_insurvey, Uval;
    double tmp_w, tmp_wm;
    
    int npix = mask1_n*mask2_n; 
    double npix_m=0.; double Upix_m=0; double npix_t=0.; double Upix_t=0;
    double supp_filter = getFilterSupp(ind_filter);
    double supp_Q2 = supp_filter*supp_filter;
    int pix1_lower = (int) floor((center_1 - (supp_filter*R_ap+mask1_d) - mask1_start)/mask1_d);
    int pix1_upper = (int) floor((center_1 + (supp_filter*R_ap+mask1_d) - mask1_start)/mask1_d);
    int pix2_lower = (int) floor((center_2 - (supp_filter*R_ap+mask2_d) - mask2_start)/mask2_d);
    int pix2_upper = (int) floor((center_2 + (supp_filter*R_ap+mask2_d) - mask2_start)/mask2_d);
   
    // Compute Map statistics for this aperture
    //  1) Check if pixel is within suitable range of aperture
    //  2) If so, update aperture coverage and loop through galaxies in that pixel to update the sums for Mapn
    //  3) Finally, check if there were sufficiently many galaxies in the aperture; if not set result to  zero
    // As we loop through the pixels in the square encompassing the aperture
    // we needlessly compute distances for ~25% of pixels (subdominant!).
    for (ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
        for (ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
            
            pix_center1 = mask1_start + ind_pix1*mask1_d;
            pix_center2 = mask2_start + ind_pix2*mask2_d;
            dpix_ap_sq = (pix_center1 - center_1)*(pix_center1 - center_1) +
                         (pix_center2 - center_2)*(pix_center2 - center_2);

            // Only care about pixels within aperture
            if (dpix_ap_sq > supp_Q2*R2_ap_d){continue;}
            
            // These out of bounds cases will occur as we are searching within a
            // rectangle that exceeds the survey footprint
            // --> Treat those as fully masked and continue;
            ind_raw = ind_pix2*mask1_n + ind_pix1;
            badpixel = (ind_raw >= npix || ind_raw < 0 ||
                        ind_pix1<0 || ind_pix2<0 || 
                        ind_pix1>=mask1_n || ind_pix2>=mask2_n);
            // Update coverage fraction
            pixinap = dpix_ap_sq <= supp_Q2*R2_ap;
            Upix = getFilterQ(ind_filter, dpix_ap_sq/R2_ap);
            if (badpixel && pixinap){
                npix_m+=1; Upix_m+=Upix;npix_t+=1; Upix_t+=Upix;} 
            if (!badpixel && pixinap){
                mask_frac = mask[ind_raw];
                npix_m+=mask_frac; Upix_m+=mask_frac*Upix; npix_t+=1; Upix_t+=Upix; }
            if (badpixel){continue;}

            // Go through the galaxies in the pixel
            ind_red = index_matcher[ind_raw];
            if (ind_red==-1){continue;}
            lower = pixs_galind_bounds[ind_red];
            upper = pixs_galind_bounds[ind_red+1];
            //printf("Go through galaxies in this pixel %d (%d %d)\n",ind_red,lower,upper);
            for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                ind_gal = pix_gals[ind_inpix];
                rel1 = pos1[ind_gal]-center_1;
                rel2 = pos2[ind_gal]-center_2;
                d2gal = (rel1*rel1 + rel2*rel2);
                if (d2gal > supp_Q2*R2_ap){continue;} 
                if (d2gal < 1e-5){continue;}
                //printf("Start allocating stuff for next galaxy\n");
                // Get tangential ellipticity and value of Q filter
                w = weight[ind_gal];
                frac_insurvey = insurvey[ind_gal];
                Uval = getFilterU(ind_filter, d2gal/R2_ap);
                //printf(" Got main quantities\n");
                // Update raw/weighted counts
                zbin = zbins[ind_gal];
                counts[zbin*3 + 0] += 1;
                counts[zbin*3 + 1] += w;
                counts[zbin*3 + 2] += w*frac_insurvey;
                //printf(" Updated counts\n");
                // Update power sums
                tmp_wU = 1;
                tmp_wUmult = w*Uval;
                for (order=0; order<max_order; order++){
                    tmp_w *= w;
                    tmp_wU *= tmp_wUmult;
                    Msn[zbin*max_order+order] += tmp_wU;
                }
            }
        }
    }
    //printf("R2=%.2f R2d=%.2f, maxd=%.4f, R=%.2f, Rext=%.4f counts=%.4f  S0=%.4f  Ms0=%.4f\n",
    //       R2_ap,R2_ap_d, max_d, R_ap, R_ap+max_d, counts[1], Sn[zbin*max_order], Msn[zbin*max_order]);
    
    // Raw and Q-weighted masked coverage fraction
    covs[0] = npix_m/npix_t;
    covs[1] = Qpix_m/Qpix_t;
    //printf("lower1=%d lower2=%d upper1=%d upper2=%d, npixm=%.2f, npixt=%.2f",
    //       pix1_lower,pix2_lower,pix1_upper,pix2_upper,npix_m,npix_t);
    //printf("covs=%.2f %.2f; counts=%.2f %.2f %.2f\n",covs[0],covs[1],counts[0],counts[1],counts[2]);
    
    // Renormalize the Ms_n components s.t. they match eqns (22),(23) in 2106.04594.
    // Note that if there are not sufficient galaxies to form closed n-side polygons,
    // the transformation equations will set those components to zero in the Mapn basis,
    // so we do not worry about this here.
    int thiscomp, fac_norm, tmp_norm;
    for (zbin=0; zbin<nbinsz; zbin++){
        fac_norm = 1/(nbar_zs[zbin]*M_PI*R_ap*R_ap);
        tmp_norm = 1;
        for (order=0; order<max_order; order++){
            thiscomp = zbin*max_order+order;
            tmp_norm *= fac_norm;
            Msn[thiscomp] *= tmp_norm;
        }
    }
}
*/
    
    

// Computes statistics on single aperture
void _singleAp_NapnSingleEonlyDisc(
    double R_ap, double center_1, double center_2, 
    int max_order, int ind_filter, 
    double *weight, double *insurvey, double *pos1, double *pos2, double *kappa_g, int *zbins, int nbinsz, int ngal, 
    double *mask,
    double mask1_start, double mask2_start, double mask1_d, double mask2_d, int mask1_n, int mask2_n,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double *counts, double *covs, double *Msn, double *Sn){

    // Variables for counting components
    int order;
    // Helper precomputations
    double R2_ap=R_ap*R_ap;
    double max_d = mymax(mask1_d,mask2_d);
    double R2_ap_d = (R_ap + max_d) * (R_ap + max_d);
    
    // Variables used for updating the mask and selecting useful galaxies
    int lower, upper;
    double Upix, mask_frac;
    double pix_center1, pix_center2, dpix_ap_sq;
    bool pixinap, badpixel;
    // All the indices
    int ind_pix1, ind_pix2, ind_raw, ind_red, ind_inpix, ind_gal, zbin;
    // Helper variables being used for the actual Mapn computation
    double rel1, rel2, d2gal, w, kg, frac_insurvey, Uval;
    double tmp_kg, tmp_w, tmp_wm, tmp_kgmult, tmp_wmmult;
    double tmp_norm, tmp_normw, tmp_normvol, fac_norm, fac_normw, fac_normvol;
    
    int npix = mask1_n*mask2_n; 
    double npix_m=0.; double Upix_m=0; double npix_w=0.; double Upix_w=0;
    double supp_filter = getFilterSupp(ind_filter);
    double supp_U2 = supp_filter*supp_filter;
    int pix1_lower = mymax(0, (int) floor((center_1 - (supp_filter*R_ap+mask1_d) - mask1_start)/mask1_d));
    int pix1_upper = mymin(mask1_n-1, (int) floor((center_1 + (supp_filter*R_ap+mask1_d) - mask1_start)/mask1_d));
    int pix2_lower = mymax(0, (int) floor((center_2 - (supp_filter*R_ap+mask2_d) - mask2_start)/mask2_d));
    int pix2_upper = mymin(mask2_n-1, (int) floor((center_2 + (supp_filter*R_ap+mask2_d) - mask2_start)/mask2_d));
    
    // Compute Map statistics for this aperture
    //  1) Check if pixel is within suitable range of aperture
    //  2) If so, update aperture coverage and loop through galaxies in that pixel to update the sums for Mapn
    //  3) Finally, check if there were sufficiently many galaxies in the aperture; if not set result to  zero
    // As we loop through the pixels in the square encompassing the aperture
    // we needlessly compute distances for ~25% of pixels (subdominant!).
    for (ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
        for (ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
            
            pix_center1 = mask1_start + ind_pix1*mask1_d;
            pix_center2 = mask2_start + ind_pix2*mask2_d;
            dpix_ap_sq = (pix_center1 - center_1)*(pix_center1 - center_1) +
                         (pix_center2 - center_2)*(pix_center2 - center_2);

            // Only care about pixels within aperture
            if (dpix_ap_sq > supp_U2*R2_ap_d){continue;}
            
            // These out of bounds cases will occur as we are searching within a
            // rectangle that exceeds the survey footprint
            // --> Treat those as fully masked and continue;
            ind_raw = ind_pix2*mask1_n + ind_pix1;
            badpixel = (ind_raw >= npix || ind_raw < 0 ||
                        ind_pix1<0 || ind_pix2<0 || 
                        ind_pix1>=mask1_n || ind_pix2>=mask2_n);
            // Update coverage fraction
            pixinap = dpix_ap_sq <= supp_U2*R2_ap;
            Upix = getFilterU(ind_filter, dpix_ap_sq/R2_ap);
            if (badpixel && pixinap){
                npix_m+=1; Upix_m+=Upix;npix_w+=1; Upix_w+=Upix;} 
            if (!badpixel && pixinap){
                mask_frac = mask[ind_raw];
                npix_m+=mask_frac; Upix_m+=mask_frac*Upix; npix_w+=1; Upix_w+=Upix; }
            if (badpixel){continue;}

            // Go through the galaxies in the pixel
            ind_red = index_matcher[ind_raw];
            if (ind_red==-1){continue;}
            lower = pixs_galind_bounds[ind_red];
            upper = pixs_galind_bounds[ind_red+1];
            //printf("Go through galaxies in this pixel %d (%d %d)\n",ind_red,lower,upper);
            for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                ind_gal = pix_gals[ind_inpix];
                rel1 = pos1[ind_gal]-center_1;
                rel2 = pos2[ind_gal]-center_2;
                d2gal = (rel1*rel1 + rel2*rel2);
                if (d2gal > supp_U2*R2_ap){continue;} 
                //printf("Start allocating stuff for next galaxy\n");
                // Get tangential ellipticity and value of Q filter
                w = weight[ind_gal];
                kg = kappa_g[ind_gal];
                
                frac_insurvey = insurvey[ind_gal];
                Uval = getFilterU(ind_filter, d2gal/R2_ap);
                //printf(" Got main quantities\n");
                // Update raw/weighted counts
                zbin = zbins[ind_gal];
                counts[zbin*3 + 0] += 1;
                counts[zbin*3 + 1] += w;
                counts[zbin*3 + 2] += w*frac_insurvey;
                //printf(" Updated counts\n");
                // Update power sums
                tmp_kg = 1;
                tmp_w = 1;
                tmp_wm = 1;
                tmp_kgmult = w*kg*Uval;
                tmp_wmmult = w*frac_insurvey;
                for (order=0; order<max_order; order++){
                    tmp_w *= w;
                    tmp_wm *= tmp_wmmult;
                    tmp_kg *= tmp_kgmult;
                    
                    Msn[zbin*max_order+order] += tmp_kg;
                    Sn[zbin*max_order+order] += tmp_w;
                //printf(" Updated powersums. Done.\n");
                }
                //if ((ind_pix1==pix1_lower+(pix1_upper-pix1_lower)/2+2*(int)(R_ap/mask1_d))&&
                //    (ind_pix2==pix2_lower+(pix2_upper-pix2_lower)/2+2*(int)(R_ap/mask2_d))&&(ind_inpix==lower)){
                //    printf("w=%.4f  reff=%.4f  Q=%.4f  ang=%.4f+i%.4ff  et=%.4f  toadd=%.4f S1=%.4f  Ms0=%.4f\n", 
                //           w,sqrt(d2gal/R2_ap),Qval,creal(phirotc_sq),cimag(phirotc_sq),et,tmp_etmult, 
                //           Sn[zbin*max_order+1], Msn[zbin*max_order]);
               // }
            }
        }
    }
    //printf("R2=%.2f R2d=%.2f, maxd=%.4f, R=%.2f, Rext=%.4f counts=%.4f  S0=%.4f  Ms0=%.4f\n",
    //       R2_ap,R2_ap_d, max_d, R_ap, R_ap+max_d, counts[1], Sn[zbin*max_order], Msn[zbin*max_order]);
    
    // Raw and Q-weighted masked coverage fraction
    covs[0] = npix_m/npix_w;
    covs[1] = Upix_m/Upix_w; // Should never be used due to compensatedness. Maybe use |U|?
    //printf("lower1=%d lower2=%d upper1=%d upper2=%d, npixm=%.2f, npixt=%.2f",
    //       pix1_lower,pix2_lower,pix1_upper,pix2_upper,npix_m,npix_t);
    //printf("covs=%.2f %.2f; counts=%.2f %.2f %.2f\n",covs[0],covs[1],counts[0],counts[1],counts[2]);
    
    // Renormalize the S_n/Ms_n components s.t. they match eqns (22),(23) in 2106.04594.
    // Note that if there are not sufficient galaxies to form closed n-side polygons,
    // the transformation equations will set those components to zero in the Mapn basis,
    // so we do not worry about this here.
    int thiscomp;
    for (zbin=0; zbin<nbinsz; zbin++){
        if (Sn[zbin*max_order]!=0){fac_norm = 1./(Sn[zbin*max_order]);}else{fac_norm = 0;}
        //if (Sn_w[zbin*max_order]!=0){fac_normw = 1./(Sn_w[zbin*max_order]);}else{fac_normw = 0;}
        fac_normvol = supp_U2;
        tmp_norm = 1;
        tmp_normw = 1;
        tmp_normvol = 1;
        for (order=0; order<max_order; order++){
            thiscomp = zbin*max_order+order;
            tmp_norm *= fac_norm;
            tmp_normw *= fac_normw;
            tmp_normvol *= fac_normvol;
            Msn[thiscomp] *= tmp_normvol*tmp_norm;
            Sn[thiscomp] *= tmp_norm;
            //Sn_w[thiscomp] *= tmp_normw;
        }
    }
    //printf("After: R2=%.2f R2d=%.2f, maxd=%.4f, R=%.2f, Rext=%.4f counts=%.4f  S0=%.6f  S1=%.6f  Ms0=%.6f\n",
    //           R2_ap,R2_ap_d, max_d, R_ap, R_ap+max_d, counts[1], Sn[0*max_order], Sn[0*max_order+1], Msn[0*max_order]);
}