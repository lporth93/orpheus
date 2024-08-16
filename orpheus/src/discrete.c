// In case the machines are acting up, compile with 
// gcc-12 -fopenmp $(gsl-config --cflags) -fPIC -shared -o discrete.so discrete.c $(gsl-config --libs) -std=c99

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
#include "discrete.h"

#define mymin(x,y) ((x) <= (y)) ? (x) : (y)
#define mymax(x,y) ((x) >= (y)) ? (x) : (y)
#define M_PI      3.14159265358979323846
#define INV_2PI   0.15915494309189534561
//#define int int32_t

void alloc_negone_int(int *arr, int length){
    for (int i=0; i<length; i++){arr[i]=-1;}
}

void set_to_zero_dcomplex(double complex *arr, int *indices, int nindices){
    for (int elind=0;elind<nindices;elind++){arr[indices[elind]] = 0;}
}

double complex mycmul(double complex z1, double complex z2){
    double a, b, c, d;
    a=creal(z1); b = cimag(z1);
    c=creal(z2); d = cimag(z2);
    return (a*c-b*d)+I*(a*d+b*c);
}

// mycmul_zzc(z1, z2) = z1 * conj(z2)
double complex mycmul_zzc(double complex z1, double complex z2){
    double a, b, c, d;
    a=creal(z1); b = cimag(z1);
    c=creal(z2); d = cimag(z2);
    return (a*c+b*d)+I*(-a*d+b*c);
}

float atan_approx(float z)
{
    const float n1 = 0.97239411f;
    const float n2 = -0.19194795f;
    return (n1 + n2 * z * z) * z;
}

float atan2_approx(float y, float x){
    const float n1 = 0.97239411f;
    const float n2 = -0.19194795f;
    const float th_max = M_PI+M_PI-1e-7;
    float ay = fabs(y)+1e-7, ax = fabs(x)+1e-7; // Avoid zero-divisions
    int invert = ay > ax;
    float z = invert ? ax/ay : ay/ax; 
    float th = (n1 + n2 * z * z) * z; // Approx of atan in [-1,1]
    if(invert) th = M_PI/2 - th;
    if(x < 0) th = M_PI - th;
    th*=(y/ay);
    if(y < 0) th = 2*M_PI + th; 
    //if(th<0||th>(2*M_PI)){printf("y=%.2f x=%.2f, arctan2=%.4f",y,x,th);}
    return mymin(th, th_max);
}

// Adapted from https://stackoverflow.com/a/44232045
// For n>~10 ln/ln_approx=1 <~ .2e-5
float ln_approx(float x) {
  unsigned int bx = * (unsigned int *) (&x);
  unsigned int ex = bx >> 23;
  signed int t = (signed int)ex-(signed int)127;
  unsigned int s = (t < 0) ? (-t) : t;
  bx = 1065353216 | (bx & 8388607);
  x = * (float *) (&bx);
  return -1.7417939 + (2.8212026 + (-1.4699568 + (0.44717955 - 0.056570851 * x) * x) * x) * x+0.6931471806*t;
}




/*    
// Non-tomo 4pcf using discrete estimator
// Very basic, no use of symmetry properties
void alloc_notomoGammans_tree_gggg(
    int *isinner, double *weight, double *pos1, double *pos2, double *e1, double *e2, int ngal, // Discrete cat
    int *index_matcher_hash, int *pixs_galind_bounds_hash, int *pix_gals_hash, int nregions, //Stuff related to regions
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nresos, double *reso_redges, int *ngal_resos, // Stuff related to tree
    double *weight_resos, double *pos1_resos, double *pos2_resos, 
    double *e1_resos, double *e2_resos, int *zbin_resos, double *weightsq_resos,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    int nmax, double rmin, double rmax, double *rbins, int nbinsr, int dccorr, // Stuff related to binning
    int nthreads, double *bin_centers, double complex *Upsilon_n, double complex *N_n){ //Stuff related to output
    
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
                        lower = pixs_galind_bounds_hash[ind_red];
                        upper = pixs_galind_bounds_hash[ind_red+1];
                        for (int ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                            ind_gal2 = pix_gals_hash[ind_inpix];
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
        #pragma omp critical
        {
            int thisn_thread, thisn2, thisn3, thisnshift, thisnrshift;
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
                                Upsilon_n[elcomp*ups_compshift+thisnrshift] += tmpUpsilon_n[elcomp*ups_compshift+thisnrshift];
                            }
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

// Upsilon_n has shape (8,nthetas,nphi12,nphi13)
void multipoles2npcf_gggg_rtriple(double complex *Upsilon_n, double complex *N_n, int n1max, int n2max,
                                  double *theta1, double *theta2, double *theta3, int nthetas,
                                  double *phis12, double *phis13, int nbinsphi12, int nbinsphi13,
                                  int projection, double complex *npcf, double complex *npcf_norm){
    int n_cfs = 8;
    int nmax = n1max;
    double complex *expphi12s = calloc(2*nmax+1, sizeof(double complex));
    double complex *expphi13s = calloc(2*nmax+1, sizeof(double complex));
    double complex *projdir = calloc(n_cfs, sizeof(double complex));
    int npcf_thetshift = nbinsphi12*nbinsphi13;
    int npcf_compshift = nthetas*npcf_thetshift;
    int ups_thetshift = (2*nmax+1)*(2*nmax+1);
    int ups_compshift =nthetas*ups_thetshift;
    
    for (int elphi12=0; elphi12<nbinsphi12; elphi12++){
        for (int elphi13=0; elphi13<nbinsphi13; elphi13++){
            // Convert multipoles to npcf
            for (int nextn=0; nextn<2*nmax+1; nextn++){ 
                expphi12s[nextn] = cexp(I*(nextn-nmax)*phis12[elphi12]);
                expphi13s[nextn] = cexp(I*(nextn-nmax)*phis13[elphi13]);
            }
            double complex nextang;
            for (int elthet=0;elthet<nthetas;elthet++){
                int ind_npcf = elthet*npcf_thetshift + elphi12*nbinsphi13 + elphi13;
                for (int nextn1=0; nextn1<2*nmax+1; nextn1++){
                    for (int nextn2=0; nextn2<2*nmax+1; nextn2++){ 
                        int ind_ups = elthet*ups_thetshift + nextn1*(2*nmax+1) + nextn2;
                        nextang = INV_2PI * expphi12s[nextn1] * expphi13s[nextn2];
                        npcf_norm[ind_npcf] += N_n[ind_ups]*nextang;
                        for (int elcf=0; elcf<n_cfs; elcf++){ 
                            npcf[elcf*npcf_compshift + ind_npcf] += 
                                Upsilon_n[elcf*ups_compshift + ind_ups]*nextang;
                        }
                    }
                }
                // Normalize: Gamma=Upsilon/N --> Make sure that we have counts, i.e. N >~ 1.
                for (int elcf=0; elcf<n_cfs; elcf++){ 
                    if (cabs(npcf_norm[ind_npcf]) > 0.1){npcf[elcf*npcf_compshift + ind_npcf] /= cabs(npcf_norm[ind_npcf]);}
                    else{npcf[elcf*npcf_compshift + ind_npcf] = 0;}
                }
                // Now transform to some projection
                if (projection==0){//X projection
                    for (int elcf=0; elcf<n_cfs; elcf++){projdir[elcf] = 1;}
                }
                else if (projection==1){//Centroid projection
                    double rcenter1, rcenter2, rcenter3;
                    double complex q1, q2, q3, q4;
                    double complex qcbyq_1, qcbyq_2, qcbyq_3, qcbyq_4, qbyqc_1, qbyqc_2, qbyqc_3, qbyqc_4;
                    rcenter1 = theta1[elthet];
                    rcenter2 = theta2[elthet];
                    rcenter3 = theta3[elthet];                        
                    q1 = -0.25*( 1*rcenter1 + 1*rcenter2*expphi12s[nmax+1] + 1*rcenter3*expphi13s[nmax+1]);
                    q2 = +0.25*( 3*rcenter1 - 1*rcenter2*expphi12s[nmax+1] - 1*rcenter3*expphi13s[nmax+1]);
                    q3 = +0.25*(-1*rcenter1 + 3*rcenter2*expphi12s[nmax+1] - 1*rcenter3*expphi13s[nmax+1]);
                    q4 = +0.25*(-1*rcenter1 - 1*rcenter2*expphi12s[nmax+1] + 3*rcenter3*expphi13s[nmax+1]);
                    qcbyq_1=conj(q1)/q1; qcbyq_2=conj(q2)/q2; qcbyq_3=conj(q3)/q3; qcbyq_4=conj(q4)/q4;
                    qbyqc_1=q1/conj(q1); qbyqc_2=q2/conj(q2); qbyqc_3=q3/conj(q3); qbyqc_4=q4/conj(q4); 
                    projdir[0] = qcbyq_1*qcbyq_2*qcbyq_3*qcbyq_4 * expphi12s[nmax+2] * expphi13s[nmax+3];
                    projdir[1] = qbyqc_1*qcbyq_2*qcbyq_3*qcbyq_4 * expphi12s[nmax+2] * expphi13s[nmax+1];
                    projdir[2] = qcbyq_1*qbyqc_2*qcbyq_3*qcbyq_4 * expphi12s[nmax+2] * expphi13s[nmax+3];
                    projdir[3] = qcbyq_1*qcbyq_2*qbyqc_3*qcbyq_4 * expphi12s[nmax-2] * expphi13s[nmax+3];
                    projdir[4] = qcbyq_1*qcbyq_2*qcbyq_3*qbyqc_4 * expphi12s[nmax+2] * expphi13s[nmax-1];
                    projdir[5] = qbyqc_1*qbyqc_2*qcbyq_3*qcbyq_4 * expphi12s[nmax+2] * expphi13s[nmax+1];
                    projdir[6] = qbyqc_1*qcbyq_2*qbyqc_3*qcbyq_4 * expphi12s[nmax-2] * expphi13s[nmax+1];
                    projdir[7] = qbyqc_1*qcbyq_2*qcbyq_3*qbyqc_4 * expphi12s[nmax+2] * expphi13s[nmax-3];
                }
                for (int elcf=0; elcf<n_cfs; elcf++){npcf[elcf*npcf_compshift + ind_npcf] *= projdir[elcf];}
            }
        }
    } 
    free(expphi12s);
    free(expphi13s);
    free(projdir);
}
*/

/*
// Very similar to last bit in the low-mem computation 
// TODO: Implement! right now just copy-paste
void multipoles2M4correlators(int n_cfs, double complex *Upsilon_n, double complex *N_n, int nmax, 
                              double *rcenters, int nbinsr, double *phis12, int nbinsphi12, double *phis13, int nbinsphi13,
                              int projection, int nthreads, double complex *M4, double complex *npcf_norm){
    
    int nrcombis = nbinsr*nbinsr*nbinsr;
    
    // For each theta combination (theta1,theta2,theta3) in this batch 
    // 1) Get bin edges and bin centers of the combinations
    // 2) Get the Gamma_mu(theta1,theta2,theta3,phi12,phi13)
    // 3) Transform the Gamma_mu to the target basis
    // 4) Update the aperture Map^4 integral
    #pragma omp parallel for
    for (int rcombi=0; rcombi<nrcombis; rcombi++){
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
    
}
*/


/*
// 4pcf
// We only allocate a single zcombination Upsilon(z1,z2,z3,z4) 
// Notes: * zsel contains [z1, z2, z3, z4]
//        * nbinsz gives the number of unique zbins in union zsel
//        * nbinsz_leafs gives the number of unique bins in zsel[1:]
//        * By convention, we force {z1,z2,z3,z4} \in {0,1,2,3} with z2<=z3<=z4
//          This does not need to relflect the actual catalog and is taken care of in the python layer.
// TODO: I suspect that when computing Upsilon_i(theta1,theta2,theta2;z1,z2,z3,z4) for 
//       n2max/n3max --> 2*n2max/2*n3max, one can reconstruct the different Upsilon_i(theta1,theta2,theta2;z1,za,zb,zc)
//       up until nmax, saving a factor of ~2-3. For no tomography the same symmetries could be employed
//       s.t. it would be sufficient to compute Upsilon_i for theta1<=theta2<=theta3. Check this!
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
        #pragma omp parallel for num_threads(nthreads)
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            int gamma_zshift = nbinsr*nbinsr*nbinsr;
            int gamma_nshift = _gamma_zshift*nbinsz*nbinsz*nbinsz*nbinsz;
            int gamma_compshift = (nmax+1)*_gamma_nshift;
            double complex *tmpGammans = calloc(ncomp*gamma_compshift, sizeof(double complex));
            double complex *tmpGammans_norm = calloc(gamma_compshift, sizeof(double complex));
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
*/

/*
// 4pcf
// We only allocate a single zcombination Upsilon(z1,z2,z3,z4) 
// Notes: * zsel contains [z1, z2, z3, z4]
//        * nbinsz gives the number of unique zbins in union zsel
//        * nbinsz_leafs gives the number of unique bins in zsel[1:]
//        * By convention, we force {z1,z2,z3,z4} \in {0,1,2,3} with z2<=z3<=z4
//          This does not need to relflect the actual catalog and is taken care of in the python layer.
// TODO: I suspect that when computing Upsilon_i(theta1,theta2,theta2;z1,z2,z3,z4) for 
//       n2max/n3max --> 2*n2max/2*n3max, one can reconstruct the different Upsilon_i(theta1,theta2,theta2;z1,za,zb,zc)
//       up until nmax, saving a factor of ~2-3. For no tomography the same symmetries could be employed
//       s.t. it would be sufficient to compute Upsilon_i for theta1<=theta2<=theta3. Check this!
void alloc_Gammans_doubletree_gggg(
    int nresos, int nresos_grid, double *dpix1_resos,  double *dpix2_resos, double *reso_redges, 
    int *ngal_resos, int *zsel, int nbinsz, int nbinsz_leafs,
    int *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos, 
    double *e1_resos, double *e2_resos, int *zbin_resos,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, 
    int nregions, int *index_matcher_hash,
    int nmax, double rmin, double rmax, int nbinsr, int dccorr, 
    int nthreads, double *bin_centers, double complex *Gammans, double complex *Gammans_norm){
    
    // Index shift for the Gamman
    int _gamma_nshift = nbinsr*nbinsr*nbinsr;
    int _gamma_compshift = (2*nmax+3)*(2*nmax+3)*_gamma_nshift;
    
    double *totcounts = calloc(nbinsz_leafs*nbinsr, sizeof(double));
    double *totnorms = calloc(nbinsz_leafs*nbinsr, sizeof(double));
    
    // Temporary arrays that are allocated in parallel and later reduced
    double *tmpwcounts = calloc(nthreads*nbinsz_leafs*nbinsr, sizeof(double));
    double *tmpwnorms = calloc(nthreads*nbinsz_leafs*nbinsr, sizeof(double));
    double complex *tmpUpsilonsn = calloc(nthreads*8*_gamma_compshift, sizeof(double complex));
    double complex *tmpGammans_norm = calloc(nthreads*_gamma_compshift, sizeof(double complex));
    
    #pragma omp parallel for num_threads(nthreads)
    for(int elthread=0;elthread<nthreads;elthread++){
        int nregions_per_thread = nregions/nthreads;
        int hasdiscrete = nresos-nresos_grid;
        int nnvals_Gn = 4*nmax+3; // Need to cover [-n1-n2-3, n1+n2-1]
        int nnvals_G2n = 4*nmax+3; // Need to cover [-n1-n2-5, n1+n2-3]
        int nnvals_Nn = 4*nmax+1; // Need to cover [-n1-n2, n1+n2]
        int nnvals_Upsn = 2*nmax+3; // Need tocover [-nmax,+nmax]
        
        // * redzs gives reduced z index for the leafs
        //   I.e. for zsel = [4,2,3,3] --> redzs = [0,1,1]
        // * zbins_leafs gives the unique zinds in the leafs (Used for the Gncaches)
        //  I.e. for zsel = [4,2,3,3] --> zbins_leafs = [2,3]
        int *redzs = calloc(3, sizeof(int));
        int *zbins_leafs = calloc(nbinsz_leafs, sizeof(int));
        if (zsel[2]>zsel[1]){redzs[1]=1;}
        if (zsel[3]>zsel[2]){redzs[2]=redzs[1]+1;}
        else{redzs[2]=redzs[1];}
        zbins_leafs[0] = zsel[1];
        if (nbinsz_leafs==2){zbins_leafs[1]=zsel[3];}
        if (nbinsz_leafs==3){zbins_leafs[1]=zsel[2]; zbins_leafs[2]=zsel[3];}
        
        // Compute how large the caches have to be at most for this thread
        
        
        // Largest possible nshift: each zbin does completely fill out the lowest reso grid.
        // The remaining grids then have 1/4 + 1/16 + ... --> 0.33.... times the data of the largest grid. 
        // Now allocate the caches
        int size_max_nshift = (int) ((1+hasdiscrete+0.34)*nbinsz_leafs*nbinsr*pow(4,nresos_grid));
        double complex *Gncache = calloc(nnvals_Gn*size_max_nshift, sizeof(double complex));
        double complex *wGncache = calloc(nnvals_Gn*size_max_nshift, sizeof(double complex));
        double complex *cwGncache = calloc(nnvals_Gn*size_max_nshift, sizeof(double complex));
        double complex *Nncache = calloc(nnvals_Nn*size_max_nshift, sizeof(double complex));
        double complex *wNncache = calloc(nnvals_Nn*size_max_nshift, sizeof(double complex));
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
            int gamma_nshift = nbinsr*nbinsr*nbinsr;
            int gamma_compshift = (2*nmax+1)*gamma_nshift;
            
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
            int indreso_hash = nresos-1;
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
                if (elregion==region_debug){printf("matchers_resoshift[elreso=%d] = %d \n",
                                                   elreso,matchers_resoshift[elreso_grid+1]);}
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
            // n --> [zleaf[0], ..., zleaf[nbinsz_leafs]] --> radius 
            //   --> [ [0]*ngal_zbin1_reso1 | [0]*ngal_zbin1_reso1/2 | ... | [0]*ngal_zbin1_reson ]
            int *cumresoshift_z = calloc(nbinsz_leafs*(nresos+1), sizeof(int)); // Cumulative shift index for resolution at z1
            int *thetashifts_z = calloc(nbinsz_leafs, sizeof(int)); // Shift index for theta given z1
            int *zbinshifts = calloc(nbinsz_leafs+1, sizeof(int)); // Cumulative shift index for z1
            int nshift; // Shifts for n index
            for (int elz=0; elz<nbinsz_leafs; elz++){
                if (elregion==region_debug){printf("z=%d/%d: \n", elz,nbinsz_leafs);}
                for (int elreso=0; elreso<nresos; elreso++){
                    if (elregion==region_debug){printf("  reso=%d/%d: \n", elreso,nresos);}
                    if (hasdiscrete==1 && elreso==0){
                        cumresoshift_z[elz*(nresos+1) + elreso+1] = ngal_in_pix[zbins_leafs[elz]*nresos + elreso+1];
                    }
                    else{
                        cumresoshift_z[elz*(nresos+1) + elreso+1] = cumresoshift_z[elz*(nresos+1) + elreso] +
                            ngal_in_pix[zbins_leafs[elz]*nresos + elreso];
                    }
                    if (elregion==region_debug){printf("  cumresoshift_z[z][reso+1]=%d: \n", 
                                                       cumresoshift_z[elz*(nresos+1) + elreso+1]);}
                }
                thetashifts_z[elz] = cumresoshift_z[elz*(nresos+1) + nresos];
                zbinshifts[elz+1] = zbinshifts[elz] + nbinsr*thetashifts_z[elz];
                if ((elregion==region_debug)){printf("thetashifts_z[z]=%d: \nzbinshifts[z+1]=%d: \n", 
                                                     thetashifts_z[elz],  zbinshifts[elz+1]);}
            }
            nshift = zbinshifts[nbinsz_leafs-1];
            // Set all the cache indeces that are updated in this region to zero
            if ((elregion==region_debug)){printf("nshift=%d: \n",  nshift);}
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
            int ind_pix1, ind_pix2, ind_inpix1, ind_inpix2, ind_red, ind_gal1, ind_gal2, z_gal1, z_gal2, redz_gal2;
            int ind_Gn, ind_Nn, ind_Gncacheshift, ind_Gncacheshift2, ind_Nncacheshift, ind_Gncache;
            int innergal, rbin, nextn, nextnshift, nbinszr, nbinszr_reso, zrshift, ind_rbin;
            double pos1_gal1, pos2_gal1, pos1_gal2, pos2_gal2, w_gal1, w_gal2, e1_gal1, e2_gal1, e1_gal2, e2_gal2;
            double w_gal2_sq;
            double rel1, rel2, dist;
            double complex wshape_gal1, wshape_gal2;
            double complex _wwphic, _wwphi;
            double complex nphirot, twophirotc, nphirotc, phirot, phirotc, phirotn;
            double rmin_reso, rmax_reso;
            int rbinmin, rbinmax, rbinmin1, rbinmax1, rbinmin2, rbinmax2;
            int nzero_Gn = 2*nmax+3;
            int nzero_Nn = 2*nmax;
            nbinszr =  nbinsz_leafs*nbinsr;
            for (int elreso=0;elreso<nresos;elreso++){
                //rmin_reso = reso_redges[elreso];
                //rmax_reso = reso_redges[elreso+1];
                //rbinmin = (int) floor((log(rmin_reso)-logrmin)/drbin);
                //rbinmax = mymin((int) floor((log(rmax_reso)-logrmin)/drbin), nbinsr-1);
                rbinmin = reso_rindedges[elreso];
                rbinmax = reso_rindedges[elreso+1];
                rmin_reso = rmin*exp(rbinmin*drbin);
                rmax_reso = rmin*exp(rbinmax*drbin);
                int nbinsr_reso = rbinmax-rbinmin;
                nbinszr_reso = nbinsz_leafs*nbinsr_reso;
                lower1 = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion];
                upper1 = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+elregion+1];
                // We have the following double/triple counting corrections terms:
                // TODO: Add expressions for Upsilon_1-Upsilon_7
                // Upsilon_0
                // ---------
                // * j=k=l --> w * G^(3)_{-8}(Theta_1)
                // * j=k   --> w * G^(2)_{n_3}(Theta_1) * G{-n_3}(Theta_3)
                //   j=l   --> w * G^(2)_{n_2}(Theta_1) * G{-n_2}(Theta_2) 
                //   k=l   --> w * G_{n_2+n_3}(Theta_1) * G^(2)_{-(n_2_n_3)}(Theta_2) 
                // Upsilon_1?
                // ---------
                // * j=k=l --> w * G^(3)_{-4}(Theta_1)
                // * j=k   --> w * G^(2)_{n_3}(Theta_1) * G{-n_3}(Theta_3)
                //   j=l   --> w * G^(2)_{n_2}(Theta_1) * G{-n_2}(Theta_2) 
                //   k=l   --> w * G_{n_2+n_3}(Theta_1) * G^(2)_{-(n_2_n_3)}(Theta_2) 
                // Upsilon_2?
                // ---------
                // * j=k=l --> w * G^(3)_{-4}(Theta_1)
                // * j=k   --> w * G^(2)_{n_3}(Theta_1) * G{-n_3}(Theta_3)
                //   j=l   --> w * G^(2)_{n_2}(Theta_1) * G{-n_2}(Theta_2) 
                //   k=l   --> w * G_{n_2+n_3}(Theta_1) * G^(2)_{-(n_2_n_3)}(Theta_2) 
                // Upsilon_3?
                // ---------
                // * j=k=l --> w * G^(3)_{-4}(Theta_1)
                // * j=k   --> w * G^(2)_{n_3}(Theta_1) * G{-n_3}(Theta_3)
                //   j=l   --> w * G^(2)_{n_2}(Theta_1) * G{-n_2}(Theta_2) 
                //   k=l   --> w * G_{n_2+n_3}(Theta_1) * G^(2)_{-(n_2_n_3)}(Theta_2) 
                // Upsilon_4?
                // ---------
                // * j=k=l --> w * G^(3)_{-4}(Theta_1)
                // * j=k   --> w * G^(2)_{n_3}(Theta_1) * G{-n_3}(Theta_3)
                //   j=l   --> w * G^(2)_{n_2}(Theta_1) * G{-n_2}(Theta_2) 
                //   k=l   --> w * G_{n_2+n_3}(Theta_1) * G^(2)_{-(n_2_n_3)}(Theta_2) 
                // Upsilon_5?
                // ---------
                // * j=k=l --> w * G^(3)_{0}(Theta_1)
                // * j=k   --> w * G^(2)_{n_3}(Theta_1) * G{-n_3}(Theta_3)
                //   j=l   --> w * G^(2)_{n_2}(Theta_1) * G{-n_2}(Theta_2) 
                //   k=l   --> w * G_{n_2+n_3}(Theta_1) * G^(2)_{-(n_2_n_3)}(Theta_2) 
                // Upsilon_6?
                // ---------
                // * j=k=l --> w * G^(3)_{0}(Theta_1)
                // * j=k   --> w * G^(2)_{n_3}(Theta_1) * G{-n_3}(Theta_3)
                //   j=l   --> w * G^(2)_{n_2}(Theta_1) * G{-n_2}(Theta_2) 
                //   k=l   --> w * G_{n_2+n_3}(Theta_1) * G^(2)_{-(n_2_n_3)}(Theta_2) 
                // Upsilon_7?
                // ---------
                // * j=k=l --> w * G^(3)_{0}(Theta_1)
                // * j=k   --> w * G^(2)_{n_3}(Theta_1) * G{-n_3}(Theta_3)
                //   j=l   --> w * G^(2)_{n_2}(Theta_1) * G{-n_2}(Theta_2) 
                //   k=l   --> w * G_{n_2+n_3}(Theta_1) * G^(2)_{-(n_2_n_3)}(Theta_2)  
                // Normalization
                // -------------
                // * j=k=l --> w * N^(3)(Theta_1)
                // * j=k   --> w * N^(2)_{n_3}(Theta_1) * N_{-n_3}(Theta_3)
                //   j=l   --> w * N^(2)_{n_2}(Theta_1) * N_{-n_2}(Theta_2) 
                //   k=l   --> w * N_{n_2+n_3}(Theta_1) * N^(2)_{-(n_2+n_3)}(Theta_2) 
                double complex *nextGns =  calloc(nnvals_Gn*nbinszr_reso, sizeof(double complex));
                double complex *nextG2ns =  calloc(nnvals_Gn*nbinszr_reso, sizeof(double complex));
                double complex *nextG3ns =  calloc(8*nbinszr_reso, sizeof(double complex));
                double complex *nextNns = calloc(nnvals_Nn*nbinszr_reso, sizeof(double complex));
                double complex *nextN2ns = calloc(nnvals_Nn*nbinszr_reso, sizeof(double complex));
                double complex *nextN3ns = calloc(nbinszr_reso, sizeof(double complex));
                int *nextncounts = calloc(nbinszr_reso, sizeof(int));
                int *allowedrinds = calloc(nbinszr_reso, sizeof(int));
                int *allowedzinds = calloc(nbinszr_reso, sizeof(int));
                if (elregion==region_debug){printf("rbinmin=%d, rbinmax%d\n",rbinmin,rbinmax);}
                for (ind_inpix1=lower1; ind_inpix1<upper1; ind_inpix1++){
                    ind_gal1 = rshift_pix_gals[elreso] + pix_gals[rshift_pix_gals[elreso]+ind_inpix1];
                    innergal = isinner_resos[ind_gal1];
                    if (innergal==0){continue;}
                    z_gal1 = zbin_resos[ind_gal1];
                    if (z_gal1!=zsel[0]) {continue;}
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
                            ind_red = index_matcher[rshift_index_matcher[elreso] + ind_pix2*pix1_n + ind_pix1];
                            if (ind_red==-1){continue;}
                            lower2 = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+ind_red];
                            upper2 = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+ind_red+1];
                            for (ind_inpix2=lower2; ind_inpix2<upper2; ind_inpix2++){
                                ind_gal2 = rshift_pix_gals[elreso] + pix_gals[rshift_pix_gals[elreso]+ind_inpix2];
                                z_gal2 = zbin_resos[ind_gal2];
                                if (z_gal2==zsel[1]){redz_gal2=redzs[0]};
                                else if (z_gal2==zsel[2]){redz_gal2=redzs[1]};
                                else if (z_gal2==zsel[3]){redz_gal2=redzs[2]};
                                else {continue;}
                                pos1_gal2 = pos1_resos[ind_gal2];
                                pos2_gal2 = pos2_resos[ind_gal2];
                                w_gal2 = weight_resos[ind_gal2];
                                e1_gal2 = e1_resos[ind_gal2];
                                e2_gal2 = e2_resos[ind_gal2];
                                
                                rel1 = pos1_gal2 - pos1_gal1;
                                rel2 = pos2_gal2 - pos2_gal1;
                                dist = sqrt(rel1*rel1 + rel2*rel2);
                                if(dist < rmin_reso || dist >= rmax_reso){continue;}
                                rbin = (int) floor((log(dist)-logrmin)/drbin) - rbinmin;
                                
                                wshape_gal2 = (double complex) w_gal2 * (e1_gal2+I*e2_gal2);
                                w_gal2_sq = w_gal2*w_gal2;
                                phirot = (rel1+I*rel2)/dist;
                                phirotc = conj(phirot);
                                twophirotc = phirotc*phirotc;
                                zrshift = redz_gal2*nbinsr_reso + rbin;
                                ind_rbin = elthread*nbinszr + redz_gal2*nbinsr + rbin+rbinmin;
                                
                                // nmin=0 
                                //   -> Gns axis: [-nmax-3, ..., -nmin-1, nmin-3, nmax-1]
                                //   -> Gn_norm axis: [0,...,nmax]
                                ind_Gn = nzero_Gn*nbinszr_reso + zrshift;
                                ind_Nn = nzero_Nn*nbinszr_reso + zrshift;
                                nphirot = 1+I*0;
                                nphirotc = 1+I*0;
                                
                                // n = 0
                                // TODO: Add all the G3 terms
                                nextncounts[zrshift] += 1;
                                tmpwcounts[ind_rbin] += w_gal1*w_gal2*dist; 
                                tmpwnorms[ind_rbin] += w_gal1*w_gal2; 
                                nextGns[ind_Gn] += wshape_gal2*nphirot;
                                nextNns[ind_Nn] += w_gal2*nphirot;  
                                nextN2ns[ind_Nn] += w_gal2_sq*nphirot;  
                                nextN3ns[zrshift] += w_gal2_sq*w_gal2;
                                nphirot *= phirot;
                                nphirotc *= phirotc; 
                                // n in [1, ..., 2*nmax-1] x {+1,-1}
                                // TODO: Add all the G2 terms
                                for (nextn=1;nextn<2*nmax;nextn++){
                                    nextnshift = nextn*nbinszr_reso;
                                    nextGns[ind_Gn+nextnshift] += wshape_gal2*nphirot;
                                    nextGns[ind_Gn-nextnshift] += wshape_gal2*nphirotc;
                                    nextNns[ind_Nn+nextnshift] += w_gal2*nphirot; 
                                    nextN2ns[ind_Nn+nextnshift] += w_gal2_sq*nphirot; 
                                    nphirot *= phirot;
                                    nphirotc *= phirotc; 
                                }
                                // n in [2*nmax, -2*nmax, -2*nmax-1, -2*nmax-2, -2*nmax-3]
                                // TODO: Add all the G2 terms
                                nextNns[ind_Nn+nextnshift+nbinszr_reso] += w_gal2*nphirot; 
                                nextN2ns[ind_Nn+nextnshift+nbinszr_reso] += w_gal2_sq*nphirot; 
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
                    // TODO: Add Caches for G2n and N2n
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
                        for (int zbin2=0; zbin2<nbinsz_leafs; zbin2++){
                            if (elregion==region_debug){
                                printf("Gnupdates for reso1=%d reso2=%d red_reso2=%d, galindex=%d, z1=%d, z2=%d:%d radial updates; shiftstart %d = %d+%d+%d+%d \n"
                                       ,elreso,elreso2,red_reso2,ind_gal1,z_gal1,zbin2,rbinmax-rbinmin,
                                       zbinshifts[z_gal1] + rbinmin*thetashifts_z[z_gal1] + 
                                       cumresoshift_z[z_gal1*(nresos+1) + elreso2] + redpix_reso2,
                                       , zbinshifts[z_gal1], rbinmin*thetashifts_z[z_gal1],
                                       cumresoshift_z[z_gal1*(nresos+1) + elreso2], redpix_reso2);
                            }
                            for (int thisrbin=rbinmin; thisrbin<rbinmax; thisrbin++){
                                zrshift = zbin2*nbinsr_reso + thisrbin-rbinmin;
                                if (cabs(nextNns[zrshift])<1e-10){continue;}
                                ind_Gncacheshift = zbinshifts[redz_gal2] + thisrbin*thetashifts_z[redz_gal2] + 
                                    cumresoshift_z[redz_gal2*(nresos+1) + elreso2] + redpix_reso2;
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
                                    thisGnnorm = nextNns[_tmpindGn];
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

                    // Allocate same reso Upsilon
                    // First check for zero count bins (most likely only in discrete-discrete bit)
                    int nallowedcounts = 0;
                    for (int zbin1=0; zbin1<nbinsz_leafs; zbin1++){
                        for (int elb1=0; elb1<nbinsr_reso; elb1++){
                            zrshift = zbin1*nbinsr_reso + elb1;
                            if (nextncounts[zbin1*nbinsr_reso + elb1] != 0){
                                allowedrinds[nallowedcounts] = elb1;
                                allowedzinds[nallowedcounts] = zbin1;
                                nallowedcounts += 1;
                            }
                        }
                    }
                    // Now update the Upsilon
                    // tmpGammas have shape (nthreads, 2*nmax_2+1, 2*nmax_3+1, r*r*r, 8)
                    // Gns have shape (nnvals, nbinsz, nbinsr)
                    // 
                    double complex h0_1, h1_1, h2_1, h3_1, w0_1;
                    double complex h0_2, h1_2, h2_2, h3_2, w0_2;
                    double complex Gmnm3, Gmnm1;
                    int thisn2, thisn3, thisnshift;
                    int _gammashift1, gammashift1, gammashift;
                    int ind_n2pn3m3, ind_n2pn3m1, ind_mn2mn3m1, ind_mn2mn3m3;
                    int ind_n2m2, ind_mn2m2, ind_n3m1, ind_n3m3, ind_mn3m1, ind_mn3m3;
                    int _zcombi, zcombi, elb1_full, elb2_full;
                    for (int ind_n2=0; ind_n2<2*nmax+1; ind_n2++){
                        thisn2 = ind_n2-nmax;
                        for (int ind_n3=0; ind_n3<2*nmax+1; ind_n3++){
                            thisn3 = ind_n3-nmax;
                            ind_n2pn3m3 = (nzero_Gn + thisn2+thisn3-3)*nbinszr_reso;
                            ind_n2pn3m1 = (nzero_Gn + thisn2+thisn3-1)*nbinszr_reso;
                            ind_mn2mn3m1 = (nzero_Gn - thisn2-thisn3-1)*nbinszr_reso;
                            ind_mn2mn3m3 = (nzero_Gn - thisn2-thisn3-3)*nbinszr_reso;
                            ind_n2m2 = (nzero_Gn + thisn2-2)*nbinszr_reso;
                            ind_mn2m2 = (nzero_Gn - thisn2-2)*nbinszr_reso;
                            ind_n3m1 = (nzero_Gn + thisn3-1)*nbinszr_reso;
                            ind_n3m3 = (nzero_Gn + thisn3-3)*nbinszr_reso;
                            ind_mn3m1 = (nzero_Gn - thisn3-1)*nbinszr_reso;
                            ind_mn3m3 = (nzero_Gn - thisn3-3)*nbinszr_reso;
                            ind_norm1 = (nzero_Nn + thisn2+thisn3)*nbinszr_reso;
                            ind_norm2 = (nzero_Nn - thisn2)*nbinszr_reso;
                            ind_norm3 = (nzero_Nn - thisn3)*nbinszr_reso;
                            thisnshift = elthread*gamma_compshift + (thisn2*nnvals_Upsn+thisn3)*gamma_nshift;
                            int elb1, zbin2;
                            for (int zrcombis1=0; zrcombis1<nallowedcounts; zrcombis1++){
                                elb1 = allowedrinds[zrcombis1];
                                zbin2 = allowedzinds[zrcombis1];
                                elb1_full = elb1 + rbinmin;
                                zrshift = zbin2*nbinsr_reso + elb1;
                                // Triple counting correction
                                if (tccorr==1){
                                    zcombi = z_gal1*nbinsz*nbinsz + zbin2*nbinsz + zbin2;
                                    gammashift1 = thisnshift + zcombi*gamma_zshift + elb1_full*nbinsr;
                                    gammashift = 4*(gammashift1 + elb1_full);
                                    //phirotm = wshape_gal1*nextG2ns[zrshift];
                                    tmpUpsilonsn[gammashift] += wshape_gal1*nextG2ns[0*nbinszr_reso + zrshift];
                                    tmpUpsilonsn[gammashift+1] += conj(wshape_gal1)*nextG2ns[1*nbinszr_reso + zrshift];
                                    tmpUpsilonsn[gammashift+2] += wshape_gal1*nextG2ns[2*nbinszr_reso + zrshift];
                                    tmpUpsilonsn[gammashift+3] += wshape_gal1*nextG2ns[3*nbinszr_reso + zrshift];
                                    tmpGammans_norm[gammashift1 + elb1_full] -= w_gal1*nextN2ns[zrshift];
                                }
                                h0 = -wshape_gal1 * nextGns[ind_nm3 + zrshift];
                                h1 = -conj(wshape_gal1) * nextGns[ind_nm1 + zrshift];
                                h2 = -wshape_gal1 * conj(nextGns[ind_mnm1 + zrshift]);
                                h3 = -wshape_gal1 * nextGns[ind_nm3 + zrshift];
                                w0 = w_gal1 * nextNns[ind_norm + zrshift];
                                _zcombi = z_gal1*nbinsz*nbinsz+zbin2*nbinsz;
                                _gammashift1 = thisnshift + elb1_full*nbinsr;
                                for (int zrcombis2=0; zrcombis2<nallowedcounts; zrcombis2++){
                                    zcombi = _zcombi+allowedzinds[zrcombis2];
                                    gammashift1 = _gammashift1 + zcombi*gamma_zshift; 
                                    elb2_full = allowedrinds[zrcombis2] + rbinmin;
                                    zrshift = allowedzinds[zrcombis2]*nbinsr_reso + allowedrinds[zrcombis2];
                                    gammashift = 4*(gammashift1 + elb2_full);
                                    Gmnm3 = nextGns[ind_mnm3 + zrshift];
                                    tmpUpsilonsn[gammashift] += h0*Gmnm3;
                                    tmpUpsilonsn[gammashift+1] += h1*nextGns[ind_mnm1 + zrshift];
                                    tmpUpsilonsn[gammashift+2] += h2*Gmnm3;
                                    tmpUpsilonsn[gammashift+3] += h3*conj(nextGns[ind_nm1 + zrshift]);
                                    tmpGammans_norm[gammashift1 + elb2_full] += w0*conj(nextNns[ind_norm + zrshift]);
                                }
                            }
                        }
                    }
                                        
                    for (int _i=0;_i<nnvals_Gn*nbinszr_reso;_i++){nextGns[_i]=0;}
                    for (int _i=0;_i<nnvals_Nn*nbinszr_reso;_i++){nextNns[_i]=0;}
                    for (int _i=0;_i<4*nbinszr_reso;_i++){nextG2ns[_i]=0;}
                    for (int _i=0;_i<nbinszr_reso;_i++){nextN2ns[_i]=0; nextncounts[_i]=0; allowedrinds[_i]=0; allowedzinds[_i]=0;}
                }
                free(nextGns);
                free(nextNns);
                free(nextG2ns);
                free(nextN2ns);
                free(nextncounts);
                free(allowedrinds);
                free(allowedzinds);
            }
            
            // Allocate the Upsilon_n for different grid resolutions from all the cached arrays 
            //
            // Note that for different configurations of the resolutions we do the Gamman
            // allocation as follows - see eq. (32) in 2309.08601 for the reasoning:
            // * Ups0 = wshape * G_n2pn3m3 * G_mn2m2 * G_mn3m3
            //          --> [(wG_n2pn3m3)_{2|1} * wG_mn2m2_{2|2}]_{3|2} * G_mn3m3_{3|3} (reso1 <= reso2 <= reso3)
            //          --> [(wG_n2pn3m3)_{2|1} * wG_mn2m2_{2|2}]_{3|2} * G_mn3m3_{3|3} (reso1 <= reso3 <= reso2)
            //          --> (wG_n2pn3m3) * (wG_mn2m2) * G_mn3m3    if max(reso1, reso2, reso3) = reso3
            // * Ups1 = conj(wshape) * G_n2pn3m1 * G_mn2m2 * G_mn3m1
            //          --> G_n2pn3m3    * (wG_mn2m2) * (cwG_mn3m1) if max(reso1, reso2, reso3) = reso1
            //          --> (wG_n2pn3m3) * (cwG_mn2m2)    * (wG_mn3m3) if max(reso1, reso2, reso3) = reso2
            //          --> (wG_n2pn3m3) * (wG_mn2m2) * G_mn3m3    if max(reso1, reso2, reso3) = reso3
            // * Ups2 = wshape * G_n2pn3m3 * G_mn2m2 * G_mn3m3
            //          --> G_n2pn3m3    * (wG_mn2m2) * (wG_mn3m3) if max(reso1, reso2, reso3) = reso1
            //          --> (wG_n2pn3m3) * G_mn2m2    * (wG_mn3m3) if max(reso1, reso2, reso3) = reso2
            //          --> (wG_n2pn3m3) * (wG_mn2m2) * G_mn3m3    if max(reso1, reso2, reso3) = reso3
            // * Ups5 = wshape * G_n2pn3m3 * G_mn2m2 * G_mn3m3
            //          --> G_n2pn3m3    * (wG_mn2m2) * (wG_mn3m3) if max(reso1, reso2, reso3) = reso1
            //          --> (wG_n2pn3m3) * G_mn2m2    * (wG_mn3m3) if max(reso1, reso2, reso3) = reso2
            //          --> (wG_n2pn3m3) * (wG_mn2m2) * G_mn3m3    if max(reso1, reso2, reso3) = reso3
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
            int ind_mnm3, ind_mnm1, ind_nm3, ind_nm1, ind_norm;
            int zbin3, zcombi, elb1_full, elb2_full;
            int red_reso1, red_reso2;
            int thisthetshift, ind_Gncacheshift_mnm3, ind_Gncacheshift_mnm1, ind_Gncacheshift_nm1;
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
                                            // n --> zbin2 --> zbin1 --> radius 
                                            //   --> [ [0]*ngal_zbin1_reso1 | ... | [0]*ngal_zbin1_reson ]
                                            ind_Nncacheshift = zbinshifts[zbin1] + 
                                                elb1*thetashifts_z[zbin1] +
                                                cumresoshift_z[zbin1*(nresos+1) + thisreso2] + elgal;
                                            ind_Gncacheshift = (nmax+3)*nshift + ind_Nncacheshift;
                                            h0 = -wGncache[(thisn-3)*nshift + ind_Gncacheshift];
                                            h1 = -cwGncache[(thisn-1)*nshift + ind_Gncacheshift];
                                            h2 = -conj(cwGncache[(-thisn-1)*nshift + ind_Gncacheshift]);
                                            h3 = -wGncache[(thisn-3)*nshift + ind_Gncacheshift];
                                            w0 = wNncache[thisn*nshift + ind_Nncacheshift];
                                            
                                            ind_Nncacheshift = + zbinshifts[zbin1] +
                                                rbinmin2*thetashifts_z[zbin1] + 
                                                cumresoshift_z[zbin1*(nresos+1) + thisreso2] + elgal;
                                            ind_Gncacheshift = (nmax+3)*nshift + ind_Nncacheshift;
                                            _imnm3 = (-thisn-3)*nshift + ind_Gncacheshift;
                                            _imnm1 = (-thisn-1)*nshift + ind_Gncacheshift;
                                            _inm1 = (thisn-1)*nshift + ind_Gncacheshift;
                                            _in = thisn*nshift + ind_Nncacheshift;
                                            for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                                gammashift = 4*(gammashift1 + elb2);
                                                tmpUpsilonsn[gammashift] += h0*Gncache[_imnm3];
                                                tmpUpsilonsn[gammashift+1] += h1*Gncache[_imnm1];
                                                tmpUpsilonsn[gammashift+2] += h2*Gncache[_imnm3];
                                                tmpUpsilonsn[gammashift+3] += h3*conj(Gncache[_inm1]);
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
                                rbinmin2 = reso_rindedges[thisreso2];
                                rbinmax2 = reso_rindedges[thisreso2+1];
                                for (int thisreso1=thisreso2+1; thisreso1<nresos; thisreso1++){
                                    rbinmin1 = reso_rindedges[thisreso1];
                                    rbinmax1 = reso_rindedges[thisreso1+1];
                                    for (int elgal=0; elgal<ngal_in_pix[zbin1*nresos+thisreso1]; elgal++){
                                        for (int elb1=rbinmin1; elb1<rbinmax1; elb1++){
                                            gammashift1 = thisnshift + zcombi*gamma_zshift + elb1*nbinsr;
                                            ind_Nncacheshift = zbinshifts[zbin1] + elb1*thetashifts_z[zbin1] +
                                                cumresoshift_z[zbin1*(nresos+1) + thisreso1] + elgal;
                                            ind_Gncacheshift = (nmax+3)*nshift + ind_Nncacheshift;
                                            h0 = -Gncache[(thisn-3)*nshift + ind_Gncacheshift];
                                            h1 = -Gncache[(thisn-1)*nshift + ind_Gncacheshift];
                                            h2 = -conj(Gncache[(-thisn-1)*nshift + ind_Gncacheshift]);
                                            h3 = -Gncache[(thisn-3)*nshift + ind_Gncacheshift];
                                            w0 = Nncache[thisn*nshift + ind_Nncacheshift];
                                            ind_Nncacheshift = zbinshifts[zbin1] + rbinmin2*thetashifts_z[zbin1] +
                                                    cumresoshift_z[zbin1*(nresos+1) + thisreso1] + elgal;
                                            ind_Gncacheshift = (nmax+3)*nshift + ind_Nncacheshift;
                                            _imnm3 = (-thisn-3)*nshift + ind_Gncacheshift;
                                            _imnm1 = (-thisn-1)*nshift + ind_Gncacheshift;
                                            _inm1 = (thisn-1)*nshift + ind_Gncacheshift;
                                            _in = thisn*nshift + ind_Nncacheshift;
                                            for (int elb2=rbinmin2; elb2<rbinmax2; elb2++){
                                                gammashift = 4*(gammashift1 + elb2);
                                                tmpUpsilonsn[gammashift] += h0*wGncache[_imnm3];
                                                tmpUpsilonsn[gammashift+1] += h1*cwGncache[_imnm1];
                                                tmpUpsilonsn[gammashift+2] += h2*wGncache[_imnm3];
                                                tmpUpsilonsn[gammashift+3] += h3*conj(cwGncache[_inm1]);
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
        free(redzs);
        free(zbins_leafs);
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
*/



///////////////////////////////////////////
// PURGATORY -- DELETE AFTER NEXT BACKUP //
///////////////////////////////////////////
// Exactly the same as doubletree, but here we bruteforce the calculation of the Gn
// and further use a course phi-binning in the annuli --> Bit faster than basetree
// but does not solve issues on diagonals...
void _alloc_Gammans_basephitree_ggg(
    int nresos, int nresos_grid, double *dpix1_resos,  double *dpix2_resos, double *reso_redges, int *ngal_resos, int nbinsz,
    int *isinner_resos, double *weight_resos, double *pos1_resos, double *pos2_resos, 
    double *e1_resos, double *e2_resos, int *zbin_resos, double *weightsq_resos,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n, int nregions, int *index_matcher_hash,
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
        
        // Caches for atan2(y=+1, x) and atan2(y=-1, x)
        // Do this in range x in [-10,10] with binsize .01
        double _tanachelim = 10;
        double _tancachesteps = 2000;
        double _tancachedx = 2*_tanachelim/_tancachesteps;
        double *_tan2cachep = calloc(_tancachesteps, sizeof(double));
        double *_tan2cachem = calloc(_tancachesteps, sizeof(double));
        for (int _i=0; _i<_tancachesteps; _i++){
            double _x = -_tanachelim + _i*_tancachedx + _tancachedx/2;
            _tan2cachep[_i] = atan2(1,_x);
            _tan2cachem[_i] = 2*M_PI+atan2(-1,_x);
        }
            
            
        // Largest possible nshift: each zbin does completely fill out the lowest reso grid.
        // The remaining grids then have 1/4 + 1/16 + ... --> 0.33.... times the data of the largest grid. 
        // Now allocate the caches
        int size_max_nshift = (int) ((1+hasdiscrete+0.34)*nbinsz*nbinsr*pow(4,nresos_grid));
        double complex *Gncache = calloc(nnvals_Gn*size_max_nshift, sizeof(double complex));
        double complex *wGncache = calloc(nnvals_Gn*size_max_nshift, sizeof(double complex));
        double complex *cwGncache = calloc(nnvals_Gn*size_max_nshift, sizeof(double complex));
        double complex *Nncache = calloc(nnvals_Nn*size_max_nshift, sizeof(double complex));
        double complex *wNncache = calloc(nnvals_Nn*size_max_nshift, sizeof(double complex));
        int *Nncache_updates = calloc(size_max_nshift, sizeof(int));
        // xxx_phibins have shape (nr, nz, nphi)
        int nbinsphi = 6000000;
        double dphi = 2*M_PI/nbinsphi;
        double nbar = 8;
        double *pos1_phibins = calloc(nbinsr*nbinsphi*nbinsz, sizeof(double));
        double *pos2_phibins = calloc(nbinsr*nbinsphi*nbinsz, sizeof(double));
        double *weight_phibins = calloc(nbinsr*nbinsphi*nbinsz, sizeof(double));
        double complex *wshape_phibins = calloc(nbinsr*nbinsphi*nbinsz, sizeof(double complex));
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
            double pos1_gal1, pos2_gal1, pos1_gal2, pos2_gal2, w_gal1, w_gal2, wsq_gal2, e1_gal1, e2_gal1, e1_gal2, e2_gal2;
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
                //rmin_reso = reso_redges[elreso];
                //rmax_reso = reso_redges[elreso+1];
                //rbinmin = (int) floor((log(rmin_reso)-logrmin)/drbin);
                //rbinmax = mymin((int) floor((log(rmax_reso)-logrmin)/drbin), nbinsr-1);
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
                    
                    int ngal_ex = (int) (2*M_PI*1.72*rbinmin*drbin*nbar*nbinsz);
                    int phibins_minshift = rbinmin*nbinsz*nbinsphi;
                    int phibins_maxshift = rbinmax*nbinsz*nbinsphi;
                    // Bin already small --> do exact calculation
                    if (ngal_ex<nbinsphi){
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
                    }
                    // Bin large --> first bin in phi and then allocate Gn over phi bins
                    else{
                        // Do this bit in
                        int phibin;
                        int rphizind;
                        float fdist;
                        float invdrbin= 1./((float)drbin );
                        float bminshift = (float) logrmin * invdrbin;
                        for (int _i=phibins_minshift; _i<phibins_maxshift; _i++){
                            pos1_phibins[_i] = 0; pos2_phibins[_i] = 0; weight_phibins[_i] = 0; wshape_phibins[_i] = 0;}
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
                                    if(dist < rmin_reso2 || dist >= rmax_reso2){continue;}
                                    fdist = sqrtf(dist);
                                    w_gal2 = weight_resos[ind_gal2];
                                    z_gal2 = zbin_resos[ind_gal2];
                                    e1_gal2 = e1_resos[ind_gal2];
                                    e2_gal2 = e2_resos[ind_gal2];
                                    wshape_gal2 = (double complex) w_gal2 * (e1_gal2+I*e2_gal2);
                                    rbin = (int) floor(logf(fdist)*invdrbin-bminshift);
                                    rbin = mymax(mymin(rbin, rbinmax-1), rbinmin);
                                    phibin = (int) floor((double) atan2_approx((float) rel2, (float) rel1)/dphi);
                                    
                                    //rbin = rbinmin+ind_gal2%nbinsr_reso;
                                    //rbin = (int) floor((log(dist)-logrmin)/drbin);
                                    //phibin = ind_gal2%nbinsphi;
                                    
                                    //xnorm = mymax(mymin(_tanachelim,rel1/(abs(rel2)+1e-5)),-_tanachelim);
                                    //tancacheind = (int) ((xnorm+_tanachelim)/_tancachesteps);
                                    //if(rel2>0){phibin = (int) (_tan2cachep[tancacheind]/dphi);}
                                    //else {phibin = (int) (_tan2cachem[tancacheind]/dphi);}
                                    
                                    //if (phibin<0 || phibin>=nbinsphi){printf("%.2f %.2f, %i, %i\n",rel1,rel2,tancacheind,phibin);}
                                    //phibin = (int) (atan2(rel2, rel1)/dphi);
                                      
                                    rphizind = rbin*nbinsz*nbinsphi+z_gal2*nbinsphi+phibin;
                                    if (rphizind<phibins_minshift || rphizind>=phibins_maxshift){
                                        printf("%d %d %d %d %d %d %d %d \n", z_gal2, rbin, rbinmin, rbinmax, 
                                               phibin, rphizind, phibins_minshift, phibins_maxshift);
                                        continue;
                                    }
                                    pos1_phibins[rphizind] += w_gal2*rel1;
                                    pos2_phibins[rphizind] += w_gal2*rel2;
                                    weight_phibins[rphizind] += w_gal2;
                                    wshape_phibins[rphizind] += wshape_gal2;
                                }
                            }
                        }
                        
                        for (int elbinz=0; elbinz<nbinsz; elbinz++){
                            for (int elbinr=rbinmin; elbinr<rbinmax; elbinr++){
                                for (int elbinphi=0; elbinphi<nbinsphi; elbinphi++){
                                    rphizind = elbinr*nbinsz*nbinsphi+elbinz*nbinsphi+elbinphi;
                                    //double dphi = atan2(rel2,rel1);
                                    //phirot = cexp(I*dphi);
                                    //if (rphizind<phibins_minshift || rphizind>=phibins_maxshift){
                                    //    printf("%d %d %d %d %d \n", elbinz, elbinr, elbinphi, phibins_minshift, phibins_maxshift);
                                    //    continue;
                                    //}
                                    w_gal2 = weight_phibins[rphizind];
                                    if (w_gal2 <= 0){continue;}
                                    
                                    wshape_gal2 = wshape_phibins[rphizind]/w_gal2;
                                    rel1 = pos1_phibins[rphizind]/w_gal2;
                                    rel2 = pos2_phibins[rphizind]/w_gal2;  
                                    dist = sqrt(rel1*rel1 + rel2*rel2);
                                    phirot = (rel1+I*rel2)/dist;// * fabs(rel1)/rel1;
                                    phirotc = conj(phirot);
                                    twophirotc = phirotc*phirotc;
                                    zrshift = elbinz*nbinsr_reso + elbinr-rbinmin;
                                    ind_rbin = elthread*nbinszr + elbinz*nbinsr;

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
        
        free(pos1_phibins);
        free(pos2_phibins);
        free(weight_phibins);
        free(wshape_phibins);
        free(_tan2cachep);
        free(_tan2cachem);
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

