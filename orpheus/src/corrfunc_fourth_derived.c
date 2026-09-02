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
#include "corrfunc_fourth_derived.h"

#define M_PI      3.14159265358979323846
#define INV_2PI   0.15915494309189534561

static inline double cabs2(double complex y) {
    double a = creal(y);
    double b = cimag(y);
    return a*a + b*b;
}


//////////////////////////////
/// GGGG STATISTICS RELATED //
//////////////////////////////

// Reconstructs all multipole components from the ones with theta1<=theta2<=theta3
// Upsn_in ~ Upsn[elthetbatch] ~ shape (8,2*nmax_alloc+1,2*nmax_alloc+1)
// Nn_in ~ Nn[elthetbatch] ~ shape (2*nmax_alloc+1,2*nmax_alloc+1)
// Upsn_out ~  shape (8,2*nmax+1,2*nmax+1)
// Nn_out ~  shape (2*nmax+1,2*nmax+1)
// 
// Ordering for eltrafo: [123, 231, 312, 132, 213, 321]
// Different configs: 
//  * All three elbs equal --> No permutations needed, i.e. eltrafo in [0]
//  * Only two elbs equal  --> Only cyclic permutations needed, i.e. eltrafo in [0,1,2]
//  * All elbs unequal     --> All permutations needed, i.e. eltrafo in [0,1,2,3,4,5]   
void getMultipolesFromSymm(double complex *Upsn_in, double complex *Nn_in,
                            int nmax, int eltrafo, int *nindices, int len_nindices,
                            double complex *Upsn_out, double complex *Nn_out){
    
    int nmax_alloc = 2*nmax+1;
    int nzero_in = nmax_alloc;
    int nzero_out = nmax;
    int n2shift_in = 2*nmax_alloc+1;
    int n2shift_out = 2*nmax+1;
    int compshift_in = n2shift_in*n2shift_in;
    int compshift_out = n2shift_out*n2shift_out;
    int comptrafos[8][6] = {{0,0,0,0,0,0}, {1,1,1,1,1,1}, 
                            {2,4,3,2,3,4}, {3,2,4,4,2,3}, {4,3,2,3,4,2}, 
                            {5,7,6,5,6,7}, {6,5,7,7,5,6}, {7,6,5,6,7,5}};
    int comptrafo0 = comptrafos[0][eltrafo];
    int comptrafo1 = comptrafos[1][eltrafo];
    int comptrafo2 = comptrafos[2][eltrafo];
    int comptrafo3 = comptrafos[3][eltrafo];
    int comptrafo4 = comptrafos[4][eltrafo];
    int comptrafo5 = comptrafos[5][eltrafo];
    int comptrafo6 = comptrafos[6][eltrafo];
    int comptrafo7 = comptrafos[7][eltrafo];
    
    int n2, n2_0234, n2_1567, n2_N;
    int n3, n3_0234, n3_1567, n3_N;
    int n23shift_in, n23shift_out;
    for (int nindex=0; nindex<len_nindices; nindex++){
        n2 = nindices[nindex]/(2*nmax_alloc+1) - nzero_in;
        n3 = nindices[nindex]%(2*nmax_alloc+1) - nzero_in;
        switch (eltrafo){
            case 0:
                n2_0234=n2; n3_0234=n3;
                n2_1567=n2; n3_1567=n3;
                n2_N=n2; n3_N=n3;
                break;
            case 1:
                n2_0234=n3+1; n3_0234=-n2-n3;
                n2_1567=n3-1; n3_1567=-n2-n3;
                n2_N=n3; n3_N=-n2-n3;
                break;
            case 2:
                n2_0234=-n2-n3+1; n3_0234=n2-1;
                n2_1567=-n2-n3-1; n3_1567=n2+1;
                n2_N=-n2-n3; n3_N=n2;
                break;
            case 3:
                n2_0234=n3+1; n3_0234=n2-1;
                n2_1567=n3-1; n3_1567=n2+1;
                n2_N=n3; n3_N=n2;
                break;
            case 4:
                n2_0234=-n2-n3+1; n3_0234=n3;
                n2_1567=-n2-n3-1; n3_1567=n3;
                n2_N=-n2-n3; n3_N=n3;
                break;
            case 5:
                n2_0234=n2; n3_0234=-n2-n3;
                n2_1567=n2; n3_1567=-n2-n3;
                n2_N=n2; n3_N=-n2-n3;
                break;
            default:
                n2_0234=0; n3_0234=0;
                n2_1567=0; n3_1567=0;
                n2_N=0; n3_N=0;
                break;
        }
        //printf("ind=%d: n2=%d n3=%d n2_0234=%d n3_0234=%d n2_1567=%d n3_1567=%d \n",
        //       nindex,n2,n3,n2_0234,n3_0234,n2_1567,n3_1567);
        if ((abs(n2_0234)<=nmax) && (abs(n3_0234)<=nmax)){
            n23shift_in = (nzero_in+n2)*n2shift_in + (nzero_in+n3);
            n23shift_out = (nzero_out+n2_0234)*n2shift_out + (nzero_out+n3_0234);
            Upsn_out[comptrafo0*compshift_out+n23shift_out] = Upsn_in[0*compshift_in+n23shift_in];
            Upsn_out[comptrafo2*compshift_out+n23shift_out] = Upsn_in[2*compshift_in+n23shift_in];
            Upsn_out[comptrafo3*compshift_out+n23shift_out] = Upsn_in[3*compshift_in+n23shift_in];
            Upsn_out[comptrafo4*compshift_out+n23shift_out] = Upsn_in[4*compshift_in+n23shift_in];
            //printf("Accepted Ups0234 update\n");
            //printf("Ups0_out[%d] is filled by Ups0_in[%d]=%.5f\n",
            //      comptrafo0*compshift_out+n23shift_out,0*compshift_in+n23shift_in,creal(Upsn_in[0*compshift_in+n23shift_in]));
        }
        if ((abs(n2_1567)<=nmax) && (abs(n3_1567)<=nmax)){
            n23shift_in = (nzero_in+n2)*n2shift_in + (nzero_in+n3);
            n23shift_out = (nzero_out+n2_1567)*n2shift_out + (nzero_out+n3_1567);
            Upsn_out[comptrafo1*compshift_out+n23shift_out] = Upsn_in[1*compshift_in+n23shift_in];
            Upsn_out[comptrafo5*compshift_out+n23shift_out] = Upsn_in[5*compshift_in+n23shift_in];
            Upsn_out[comptrafo6*compshift_out+n23shift_out] = Upsn_in[6*compshift_in+n23shift_in];
            Upsn_out[comptrafo7*compshift_out+n23shift_out] = Upsn_in[7*compshift_in+n23shift_in];
        }
        if ((abs(n2_N)<=nmax) && (abs(n3_N)<=nmax)){
            Nn_out[(nzero_out+n2_N)*n2shift_out+(nzero_out+n3_N)] = Nn_in[(nzero_in+n2)*n2shift_in+(nzero_in+n3)];
        }
    }
}

// Upsilon_n has shape (8,nphi12,nphi13)
void multipoles2npcf_gggg_singletheta(double complex *Upsilon_n, double complex *N_n, int n1max, int n2max,
                                      double theta1, double theta2, double theta3,
                                      double *phis12, double *phis13, int nbinsphi12, int nbinsphi13,
                                      int projection, double complex *npcf, double complex *npcf_norm){
    int n_cfs = 8;
    int nmax = n1max;
    int nns = 2*nmax+1;
    double complex expphi12, expphi13;
    double complex *expphi12s = orpheus_calloc(nns, sizeof(double complex));
    double complex *expphi13s = orpheus_calloc(nns, sizeof(double complex));
    double complex *projdir = orpheus_calloc(n_cfs, sizeof(double complex));
    int npcf_compshift = nbinsphi12*nbinsphi13;
    int ups_compshift = nns*nns;
    for (int elphi12=0; elphi12<nbinsphi12; elphi12++){
        for (int elphi13=0; elphi13<nbinsphi13; elphi13++){
            // Convert multipoles to npcf
            expphi12s[nmax] = 1;
            expphi13s[nmax] = 1;
            expphi12 = cexp(I*phis12[elphi12]);
            expphi13 = cexp(I*phis13[elphi13]);
            for (int nextn=1; nextn<=nmax; nextn++){ 
                expphi12s[nmax+nextn] = expphi12s[nmax+nextn-1]*expphi12;
                expphi12s[nmax-nextn] = conj(expphi12s[nmax+nextn]);
                expphi13s[nmax+nextn] = expphi13s[nmax+nextn-1]*expphi13;
                expphi13s[nmax-nextn] = conj(expphi13s[nmax+nextn]);
            }
            double complex nextang;
            int ind_npcf = elphi12*nbinsphi13 + elphi13;
            for (int nextn1=0; nextn1<nns; nextn1++){
                for (int nextn2=0; nextn2<nns; nextn2++){ 
                    int ind_ups = nextn1*nns + nextn2;
                    nextang = INV_2PI * expphi12s[nextn1] * expphi13s[nextn2];
                    npcf_norm[ind_npcf] += N_n[ind_ups]*nextang;
                    for (int elcf=0; elcf<n_cfs; elcf++){ 
                        npcf[elcf*npcf_compshift + ind_npcf] += Upsilon_n[elcf*ups_compshift + ind_ups]*nextang;
                    }
                }
            }
            // Normalize: Gamma=Upsilon/N --> Make sure that we have counts, i.e. N >~ 1.
            for (int elcf=0; elcf<n_cfs; elcf++){ 
                if (cabs(npcf_norm[ind_npcf]) > 0.){npcf[elcf*npcf_compshift + ind_npcf] /= creal(npcf_norm[ind_npcf]);}
                else{npcf[elcf*npcf_compshift + ind_npcf] = 0;}
            }
            // Now transform to some projection
            if (projection==0){//X projection
                for (int elcf=0; elcf<n_cfs; elcf++){projdir[elcf] = 1;}
            }
            else if (projection==1){//Centroid projection
                double complex y1, y2, y3;
                double complex q1, q2, q3, q4;
                double complex qcbyq_1, qcbyq_2, qcbyq_3, qcbyq_4, qbyqc_1, qbyqc_2, qbyqc_3, qbyqc_4;
                y1 = theta1;
                y2 = theta2*expphi12s[nmax+1];
                y3 = theta3*expphi13s[nmax+1];                        
                q1 = -0.25*(  y1 + y2   + y3);
                q2 = +0.25*(3*y1 - y2   - y3);
                q3 = +0.25*(- y1 + 3*y2 - y3);
                q4 = +0.25*(- y1 - 1*y2 + 3*y3);
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
    free(expphi12s);
    free(expphi13s);
    free(projdir);
}


// Upsilon_n has shape (8,n1max+1,n2max+1,nphi12,nphi13)
void multipoles2npcf_gggg_singletheta_nconvergence(
    double complex *Upsilon_n, double complex *N_n, int n1max, int n2max,
    double theta1, double theta2, double theta3,
    double *phis12, double *phis13, int nbinsphi12, int nbinsphi13,
    int projection, int verbose, double complex *npcf, double complex *npcf_norm){
    
    int n_cfs = 8;
    int nmax = n1max;
    int nns = 2*nmax+1;
    double complex expphi12, expphi13;
    double complex *expphi12s = orpheus_calloc(nns, sizeof(double complex));
    double complex *expphi13s = orpheus_calloc(nns, sizeof(double complex));
    double complex *projdir = orpheus_calloc(n_cfs, sizeof(double complex));
    int npcf_n2cutshift = nbinsphi12*nbinsphi13;
    int npcf_n1cutshift = (n2max+1)*nbinsphi12*nbinsphi13;
    int npcf_compshift = (n1max+1)*(n2max+1)*nbinsphi12*nbinsphi13;
    int ups_compshift = nns*nns;
    reset_progress();
    for (int elphi12=0; elphi12<nbinsphi12; elphi12++){
        for (int elphi13=0; elphi13<nbinsphi13; elphi13++){
            print_progress(elphi12*nbinsphi13+elphi13+1, nbinsphi12*nbinsphi13, verbose);
            // Convert multipoles to npcf
            expphi12s[nmax] = 1;
            expphi13s[nmax] = 1;
            expphi12 = cexp(I*phis12[elphi12]);
            expphi13 = cexp(I*phis13[elphi13]);
            for (int nextn=1; nextn<=nmax; nextn++){ 
                expphi12s[nmax+nextn] = expphi12s[nmax+nextn-1]*expphi12;
                expphi12s[nmax-nextn] = conj(expphi12s[nmax+nextn]);
                expphi13s[nmax+nextn] = expphi13s[nmax+nextn-1]*expphi13;
                expphi13s[nmax-nextn] = conj(expphi13s[nmax+nextn]);
            }
            double complex nextang;
            for (int n1cut=0; n1cut<=nmax; n1cut++){
                for (int n2cut=0; n2cut<=nmax; n2cut++){ 
                    //printf("Doing n1c=%d n2c=%d",n1cut,n2cut);
                    int ind_npcf = n1cut*npcf_n1cutshift + n2cut*npcf_n2cutshift + elphi12*nbinsphi13 + elphi13;
                    for (int nextn1=-n1cut; nextn1<=n1cut; nextn1++){
                        for (int nextn2=-n2cut; nextn2<=n2cut; nextn2++){ 
                            int ind_ups = (nmax+nextn1)*nns + (nmax+nextn2);
                            nextang = INV_2PI * expphi12s[nmax+nextn1] * expphi13s[nmax+nextn2];
                            npcf_norm[ind_npcf] += N_n[ind_ups]*nextang;
                            for (int elcf=0; elcf<n_cfs; elcf++){ 
                                npcf[elcf*npcf_compshift + ind_npcf] += Upsilon_n[elcf*ups_compshift + ind_ups]*nextang;
                            }
                        }
                    }
                    // Normalize: Gamma=Upsilon/N --> Make sure that we have counts, i.e. N >~ 1.
                    for (int elcf=0; elcf<n_cfs; elcf++){ 
                        if (cabs(npcf_norm[ind_npcf]) > 0.){npcf[elcf*npcf_compshift + ind_npcf] /= creal(npcf_norm[ind_npcf]);}
                        else{npcf[elcf*npcf_compshift + ind_npcf] = 0;}
                    }
                    // Now transform to some projection
                    if (projection==0){//X projection
                        for (int elcf=0; elcf<n_cfs; elcf++){projdir[elcf] = 1;}
                    }
                    else if (projection==1){//Centroid projection
                        double complex y1, y2, y3;
                        double complex q1, q2, q3, q4;
                        double complex qcbyq_1, qcbyq_2, qcbyq_3, qcbyq_4, qbyqc_1, qbyqc_2, qbyqc_3, qbyqc_4;
                        y1 = theta1;
                        y2 = theta2*expphi12s[nmax+1];
                        y3 = theta3*expphi13s[nmax+1];                        
                        q1 = -0.25*(  y1 + y2   + y3);
                        q2 = +0.25*(3*y1 - y2   - y3);
                        q3 = +0.25*(- y1 + 3*y2 - y3);
                        q4 = +0.25*(- y1 - 1*y2 + 3*y3);
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
    } 
    free(expphi12s);
    free(expphi13s);
    free(projdir);
}

void fourpcfmultipoles2M4correlators(
    int nmax, int nmax_trafo,
    double *theta_edges, double *theta_centers, int nthetas, 
    double *mapradii, int nmapradii,
    double *phis1, double *phis2, double *dphis1, double *dphis2, int nbinsphi1, int nbinsphi2,
    int projection, int nthreads, int verbose,
    double complex *Upsilon_n, double complex *N_n, double complex *m4corr){


    double complex *allm4corr = orpheus_calloc(nthreads*8*nmapradii, sizeof(double complex));
    int trafos_finished = 0;
    reset_progress();

    #pragma omp parallel for num_threads(nthreads)
    for (int thetacombi=0; thetacombi<nthetas*nthetas*nthetas; thetacombi++){
        
        int thisthread = omp_get_thread_num();
        
        int nphicombis = nbinsphi1*nbinsphi2;
        int n2n3combis = (2*nmax+1)*(2*nmax+1);
        int n2n3combis_trafo = (2*nmax_trafo+1)*(2*nmax_trafo+1);
        int nthetas2 = nthetas*nthetas;
        int nthetas3 = nthetas*nthetas*nthetas;
        int compshift = n2n3combis*nthetas3;
        int ithet1 = thetacombi/nthetas2;
        int ithet2 = (thetacombi-nthetas2*ithet1)/nthetas;
        int ithet3 = thetacombi%nthetas;
        
        double theta1, theta2, theta3, dtheta1, dtheta2, dtheta3;
        #pragma omp critical
        {
            theta1 = theta_centers[ithet1];
            theta2 = theta_centers[ithet2];
            theta3 = theta_centers[ithet3];
            dtheta1 = theta_edges[ithet1+1]-theta_edges[ithet1];
            dtheta2 = theta_edges[ithet2+1]-theta_edges[ithet2];
            dtheta3 = theta_edges[ithet3+1]-theta_edges[ithet3];
        }
        
        // Transform multipoles to 4pcf
        int thisn1, thisn2, n2n3combi_trafo;
        double complex *thisnpcf = orpheus_calloc(8*nphicombis, sizeof(double complex));
        double complex *thisnpcf_norm = orpheus_calloc(nphicombis, sizeof(double complex));
        double complex *Upsn_single = orpheus_calloc(8*n2n3combis_trafo, sizeof(double complex));
        double complex *Nn_single = orpheus_calloc(1*n2n3combis_trafo, sizeof(double complex));
        for (int elcomp=0; elcomp<8; elcomp++){
            for (int n2n3combi=0; n2n3combi<n2n3combis; n2n3combi++){
                thisn1 = n2n3combi/(2*nmax+1) - nmax;
                thisn2 = n2n3combi%(2*nmax+1) - nmax;
                if ((abs(thisn1)<=nmax_trafo) && (abs(thisn2)<=nmax_trafo)){
                    n2n3combi_trafo = (thisn1+nmax_trafo)*(2*nmax_trafo+1) + (thisn2+nmax_trafo);
                    Upsn_single[elcomp*n2n3combis_trafo+n2n3combi_trafo] = 
                        Upsilon_n[elcomp*compshift+n2n3combi*nthetas3+thetacombi];
                    Nn_single[n2n3combi_trafo] = 
                        N_n[n2n3combi*nthetas3+thetacombi];
                }
            }
        }
        multipoles2npcf_gggg_singletheta(Upsn_single, Nn_single, nmax_trafo, nmax_trafo,
                                         theta1, theta2, theta3,
                                         phis1, phis2, nbinsphi1, nbinsphi2,
                                         projection, thisnpcf, thisnpcf_norm);

        // Transform 4pcf to M4
        double complex nextm4corr[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        double y1, y2, y3, dy1, dy2, dy3, R_ap;
        int map4threadshift = thisthread*8*nmapradii;
        for (int elmapr=0; elmapr<nmapradii; elmapr++){
            #pragma omp critical
            {R_ap=mapradii[elmapr];}
            y1=theta1/R_ap; y2=theta2/R_ap; y3=theta3/R_ap;
            dy1 = dtheta1/R_ap; dy2 = dtheta2/R_ap; dy3 = dtheta3/R_ap;
            fourpcf2M4correlators(1,
                                  y1, y2, y3, dy1, dy2, dy3,
                                  phis1, phis2, dphis1, dphis2, nbinsphi1, nbinsphi2,
                                  thisnpcf, nextm4corr);
            for (int elcomp=0;elcomp<8;elcomp++){
                if (isfinite(cabs(nextm4corr[elcomp]))){
                    allm4corr[map4threadshift+elcomp*nmapradii+elmapr] += nextm4corr[elcomp];
                }
                nextm4corr[elcomp] = 0;
            }
        }
        free(thisnpcf);
        free(thisnpcf_norm);
        free(Upsn_single);
        free(Nn_single);
        
        #pragma omp atomic
        trafos_finished+=1;
        
        print_progress(trafos_finished, nthetas3, verbose);
        //int tmpprint=(int) (100.0*trafos_finished/nthetas3);
        //if (tmpprint > lastprint){
        //    printf("\rStatus after %i per cent:",tmpprint);
        //    for (int elmapr=0; elmapr<nmapradii; elmapr++){
        //        double complex thisM4 = allm4corr[map4threadshift+0*nmapradii+elmapr];
        //        printf("  M4(%.2f) = 1e12*(%.2f + i*%.2f)\n", mapradii[elmapr], 1e12*creal(thisM4), 1e12*cimag(thisM4));
        //    } 
        //    #pragma omp critical
        //    {lastprint = tmpprint;}
        //}
    }
    if (verbose>0){ printf("\n"); }

    // Accumulate the M4correlators
    for (int elthread=0;elthread<nthreads;elthread++){
        for (int elcr=0;elcr<8*nmapradii;elcr++){
            m4corr[elcr] += allm4corr[elthread*8*nmapradii+elcr];
        }
    }  

    free(allm4corr);
}


// M4 filters for a fixed aperture radius
// Note that with y==theta/R_ap the expressions do not depend on the aperture radius
// fourpcf has shape (8,nbinsz,nbinsphi,nbinsphi)
void fourpcf2M4correlators(int nzcombis,
                           double y1, double y2, double y3, double dy1, double dy2, double dy3,
                           double *phis1, double *phis2, double *dphis1, double *dphis2, int nbinsphi1, int nbinsphi2,
                           double complex *fourpcf, double complex *m4corr){
    double complex q1, q2, q3, q4, q1c, q2c, q3c, q4c, q1_2, q2_2, q3_2, q4_2, q1c_2, q2c_2, q3c_2, q4c_2;
    double complex q1cq2c, q1cq3c, q1cq4c, q2cq3c, q2cq4c, q3cq4c;
    double q1abs, q2abs, q3abs, q4abs, q1abs_2, q2abs_2, q3abs_2, q4abs_2;
    double complex _y1, _y2, _y3, _y1c, _y3c, _y1_2, _y2_2, _y3_2, _y1_3, _y3_3 ;
    double _y1abs, _y2abs, _y3abs, _y1abs_2, _y2abs_2, _y3abs_2, _y1abs_3, _y3abs_3;
    double _measure, _exp;
    double complex F_1[8]= {0, 0, 0, 0, 0, 0, 0, 0};
    double complex F_2[8]= {0, 0, 0, 0, 0, 0, 0, 0};
    double complex F_3[8]= {0, 0, 0, 0, 0, 0, 0, 0};
    double complex xproj[8]= {0, 0, 0, 0, 0, 0, 0, 0};
    double compshuffle[8] = {0, 1, 2, 3, 4, 7, 5, 6};
    double complex nextF;
    int fourpcf_compshift, ind_4pcf;
    for (int elphi1=0;elphi1<nbinsphi1;elphi1++){
        for (int elphi2=0;elphi2<nbinsphi2;elphi2++){
            _y1 = y1; _y1_2=_y1*_y1; _y1_3=_y1_2*_y1; _y1c=conj(_y1); 
            _y1abs=y1; _y1abs_2=_y1abs*_y1abs; _y1abs_3=_y1abs_2*_y1abs;
            _y2 = y2*cexp(I*phis1[elphi1]); _y2_2=_y2*_y2; _y3c=conj(_y2); 
            _y2abs=y2; _y2abs_2=_y2abs*_y2abs; 
            _y3 = y3*cexp(I*phis2[elphi2]); _y3_2=_y3*_y3; _y3_3=_y3_2*_y3; _y3c=conj(_y3); 
            _y3abs=y3; _y3abs_2=_y3abs*_y3abs; _y3abs_3=_y3abs_2*_y3abs;
            q1 =  0.25*(3*_y1-_y2-_y3); q1c=conj(q1); q1_2=q1*q1; q1c_2=q1c*q1c; q1abs=cabs(q1); q1abs_2=q1abs*q1abs;
            q2 =  0.25*(3*_y2-_y3-_y1); q2c=conj(q2); q2_2=q2*q2; q2c_2=q2c*q2c; q2abs=cabs(q2); q2abs_2=q2abs*q2abs;
            q3 =  0.25*(3*_y3-_y1-_y2); q3c=conj(q3); q3_2=q3*q3; q3c_2=q3c*q3c; q3abs=cabs(q3); q3abs_2=q3abs*q3abs;
            q4 = -0.25*(  _y1+_y2+_y3); q4c=conj(q4); q4_2=q4*q4; q4c_2=q4c*q4c; q4abs=cabs(q4); q4abs_2=q4abs*q4abs;
            q1cq2c=q1c*q2c; q1cq3c=q1c*q3c; q1cq4c=q1c*q4c; q2cq3c=q2c*q3c; q2cq4c=q2c*q4c; q3cq4c=q3c*q4c; 
            _measure = y1*y2*y3*dy1*dy2*dy3 * dphis1[elphi1]*dphis2[elphi2]*INV_2PI*INV_2PI;
            _exp = exp(-0.5*(q1abs_2+q2abs_2+q3abs_2+q4abs_2));
            F_1[0] = q1c_2*q2c_2*q3c_2*q4c_2;
            F_1[1] = q4_2*q1c_2*q2c_2*q3c_2; 
            F_1[2] = q1_2*q4c_2*q2c_2*q3c_2; 
            F_1[3] = q2_2*q1c_2*q4c_2*q3c_2; 
            F_1[4] = q3_2*q1c_2*q2c_2*q4c_2; 
            F_1[5] = q4_2*q1c_2*q2c_2*q3_2; 
            F_1[6] = q4_2*q3c_2*q2c_2*q1_2; 
            F_1[7] = q4_2*q1c_2*q3c_2*q2_2;
            F_2[0] = 0;
            F_2[1] = 2*q4*q1cq2c*q3c*(q2cq3c+q1cq3c+q1cq2c);
            F_2[2] = 2*q1*q2cq4c*q3c*(q2cq3c+q3cq4c+q2cq4c);
            F_2[3] = 2*q2*q1cq4c*q3c*(q3cq4c+q1cq3c+q1cq4c);
            F_2[4] = 2*q3*q1cq2c*q4c*(q2cq4c+q1cq4c+q1cq2c);
            F_2[5] = 2*q1cq2c*q3*q4*(q3+q4)*(q1c+q2c);
            F_2[6] = 2*q2cq3c*q1*q4*(q1+q4)*(q3c+q2c);
            F_2[7] = 2*q1cq3c*q2*q4*(q2+q4)*(q1c+q3c);
            F_3[0] = 0;
            F_3[1] = 0.5*(q1c_2*(q2c_2+4*q2cq3c)+q2c_2*(q3c_2+4*q1cq3c)+q3c_2*(q1c_2+4*q1cq2c));
            F_3[2] = 0.5*(q4c_2*(q2c_2+4*q2cq3c)+q2c_2*(q3c_2+4*q3cq4c)+q3c_2*(q4c_2+4*q2cq4c));
            F_3[3] = 0.5*(q1c_2*(q4c_2+4*q3cq4c)+q4c_2*(q3c_2+4*q1cq3c)+q3c_2*(q1c_2+4*q1cq4c));
            F_3[4] = 0.5*(q1c_2*(q2c_2+4*q2cq4c)+q2c_2*(q4c_2+4*q1cq4c)+q4c_2*(q1c_2+4*q1cq2c));
            F_3[5] = 0.5*(q3_2+4*q3*q4+q4_2)*(q1c_2+4*q1cq2c+q2c_2) + 3*(q3+q4)*(q1c+q2c) + 1.5;
            F_3[6] = 0.5*(q1_2+4*q1*q4+q4_2)*(q3c_2+4*q2cq3c+q2c_2) + 3*(q1+q4)*(q3c+q2c) + 1.5;
            F_3[7] = 0.5*(q2_2+4*q2*q4+q4_2)*(q1c_2+4*q1cq3c+q3c_2) + 3*(q2+q4)*(q1c+q3c) + 1.5;
            xproj[0] =  _y1_3*_y2_2*_y3_3/(_y1abs_3*_y2abs_2*_y3abs_3);//comp 0
            xproj[1] =  _y1*_y2_2*_y3/(_y1abs*_y2abs_2*_y3abs);//comp 1
            xproj[2] =  _y1c*_y2_2*_y3_3/(_y1abs*_y2abs_2*_y3abs_3);//comp 2
            xproj[3] =  _y1_3*conj(_y2_2)*_y3_3/(_y1abs_3*_y2abs_2*_y3abs_3);//comp 3
            xproj[4] =  _y1_3*_y2_2*_y3c/(_y1abs_3*_y2abs_2*_y3abs);//comp 4
            xproj[5] =  _y1*_y2_2*conj(_y3_3)/(_y1abs*_y2abs_2*_y3abs_3);//comp 7
            xproj[6] =  conj(_y1_3)*_y2_2*_y3/(_y1abs_3*_y2abs_2*_y3abs);//comp 5
            xproj[7] =  _y1*conj(_y2_2)*_y3/(_y1abs*_y2abs_2*_y3abs);//comp 6
            for (int elcombi=0;elcombi<8;elcombi++){
                nextF = 1./64 * _measure * (F_1[elcombi]+F_2[elcombi]+F_3[elcombi])* _exp * xproj[elcombi];
                fourpcf_compshift = compshuffle[elcombi]*nzcombis*nbinsphi1*nbinsphi2 + elphi1*nbinsphi2 + elphi2;
                for (int zcombi=0;zcombi<nzcombis;zcombi++){
                    ind_4pcf = fourpcf_compshift + zcombi*nbinsphi1*nbinsphi2;
                    m4corr[elcombi*nzcombis+zcombi] += nextF * fourpcf[ind_4pcf];
                }
            }
        }
    }
}

// No zbins as in most circumstances this will exceed the 2^32 elements barrier in the arrays...
// Additionally, we only subselect certain (phi12,phi13) bin combinations to make sure that 
// the individual chunks will never exceed 1e9 elements
// I.e. nbinsr=nbinsphi=50 --> len_arr = 8*nbinsr^3*nbinsphi^2 ~ 2.5e9
// Projections: 0:x, 1:centroid
void multipoles2npcf_gggg(double complex *upsilon_n, double complex *N_n, double *rcenters,
                          const BinningParams *bin, const FourthParams *fourth,
                          int projection, int n_cfs, int nthreads,
                          double complex *npcf, double complex *npcf_norm){
    int nbinsr = bin->nbinsr, nmax = bin->nmax;
    double *phis12 = fourth->phibins1, *phis13 = fourth->phibins2;
    int nbinsphi12 = fourth->nbinsphi1, nbinsphi13 = fourth->nbinsphi2;
    // Shape of upsilon_n: (n_cfs,2*_nmax+1,2*_nmax+1,1,nbinsr,nbinsr,nbinsr) ~ (elcomp, n1, n2, elb1, elb2, elb3)
    //          npcf     : (n_cfs,1,nbinsr,nbinsr,nbinsr,nbinsphi,nbinsphi)   ~ (elcomp, elb1, elb2, elb3, elphi12, elphi13)
    
    // We parallelize over the different phi combis
    int nrcombis = nbinsr*nbinsr*nbinsr;
    #pragma omp parallel for num_threads(nthreads)
    for (int nrcombi=0; nrcombi<nrcombis; nrcombi++){
        
        int nbinsr2 = nbinsr*nbinsr;
        int nbinsr3 = nbinsr*nbinsr*nbinsr;
        int mult_n1shift = nbinsr3;
        int mult_n2shift = (2*nmax+1)*mult_n1shift;
        int mult_compshift = (2*nmax+1)*mult_n2shift;
        int npcf_r3shift = nbinsphi12*nbinsphi13;
        int npcf_r2shift = nbinsr*npcf_r3shift;
        int npcf_r1shift = nbinsr*npcf_r2shift;
        int npcf_compshift = nbinsr*npcf_r1shift;
        
        int elr1 = nrcombi/nbinsr2;
        int elr2 = (nrcombi-elr1*nbinsr2)/nbinsr;
        int elr3 = nrcombi%nbinsr;
        double rcenter1, rcenter2, rcenter3;
        double complex *expphi12s = orpheus_calloc(2*nmax+1, sizeof(double complex));
        double complex *expphi13s = orpheus_calloc(2*nmax+1, sizeof(double complex));
        double complex *projdir = orpheus_calloc(n_cfs, sizeof(double complex));
        double complex nextang;
        int ind_multi, ind_npcf;
        double complex q1, q2, q3, q4;
        double complex qcbyq_1, qcbyq_2, qcbyq_3, qcbyq_4;
        double complex qbyqc_1, qbyqc_2, qbyqc_3, qbyqc_4;
        
        #pragma omp critical
        {rcenter1 = rcenters[elr1]; rcenter2 = rcenters[elr2]; rcenter3 = rcenters[elr3];}
        
        for (int elphi12=0; elphi12<nbinsphi12; elphi12++){
            for (int elphi13=0; elphi13<nbinsphi13; elphi13++){
                // Convert multipoles to npcf
                for (int nextn=0; nextn<2*nmax+1; nextn++){ 
                    expphi12s[nextn] = cexp(I*(nextn-nmax)*phis12[elphi12]);
                    expphi13s[nextn] = cexp(I*(nextn-nmax)*phis13[elphi13]);
                }
                ind_npcf = elr1*npcf_r1shift + elr2*npcf_r2shift + elr3*npcf_r3shift + elphi12*nbinsphi13 + elphi13;
                for (int nextn1=0; nextn1<2*nmax+1; nextn1++){
                    for (int nextn2=0; nextn2<2*nmax+1; nextn2++){ 
                        nextang = INV_2PI * expphi12s[nextn1] * expphi13s[nextn2];
                        ind_multi = nextn1*mult_n2shift + nextn2*mult_n1shift + elr1*nbinsr2 + elr2*nbinsr + elr3;
                        npcf_norm[ind_npcf] += N_n[ind_multi]*nextang;
                        for (int elcf=0; elcf<n_cfs; elcf++){ 
                            npcf[elcf*npcf_compshift + ind_npcf] += upsilon_n[elcf*mult_compshift + ind_multi]*nextang;
                        }
                    }
                }
                // Normalize: Gamma=Upsilon/N --> Make sure that we have counts, i.e. N >~ 1.
                for (int elcf=0; elcf<n_cfs; elcf++){ 
                    if (cabs(npcf_norm[ind_npcf]) > 0.){npcf[elcf*npcf_compshift + ind_npcf] /= creal(npcf_norm[ind_npcf]);}
                    else{npcf[elcf*npcf_compshift + ind_npcf] = 0;}
                }
                // Now transform to some projection
                if (projection==0){//X projection
                    for (int elcf=0; elcf<n_cfs; elcf++){projdir[elcf] = 1;}
                }
                else if (projection==1){//Centroid projection
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
        free(expphi12s);
        free(expphi13s);
        free(projdir);
    }
}

// If thread==0 --> For final two threads allocate double/triple counting corrs
// thetacombis_batches: array of length nbinsr^3 with the indices of all possible (r1,r2,r3) combinations
//                      most likely it is simply range(nbinsr^3), but we leave some freedom here for 
//                      potential cost-based implementations
// nthetacombis_batches: array of length nthetbatches with the number of theta-combis in each batch
// cumthetacombis_batches : array of length (nthetbatches+1) with is cumsum of nthetacombis_batches
// nthetbatches: the number of theta batches
void alloc_notomoMap4_analytic(
    double rmin, double rmax, int nbinsr, double *phibins, double *dbinsphi, int nbinsphi, int nsubr,
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, double *mapradii, int nmapradii, 
    double *xip, double *xim, double thetamin_xi, double thetamax_xi, int nthetabins_xi, int nsubsample_filter,
    double complex *M4correlators){
               
    
    double complex *allM4correlators = orpheus_calloc(nthreads*8*1*nmapradii, sizeof(double complex));
    #pragma omp parallel for num_threads(nthreads)
    for(int elthetbatch=0;elthetbatch<nthetbatches;elthetbatch++){
        int thisthread = omp_get_thread_num();
        //printf("Starting thetabatch %d/%d on thread %d with %d thetacombis\n",
        //       elthetbatch,nthetbatches,thisthread,nthetacombis_batches[elthetbatch]);
        //int nbinsz = 1;
        int batch_nthetas = nthetacombis_batches[elthetbatch];   
        int batchgamma_thetshift = nbinsphi*nbinsphi;
        
        double *bin_centers = orpheus_calloc(nbinsr, sizeof(double));
        double *bin_edges = orpheus_calloc(nbinsr+1, sizeof(double));
        double drbin = exp((log(rmax)-log(rmin))/(nbinsr));
        int *elb1s_batch = orpheus_calloc(batch_nthetas, sizeof(int));
        int *elb2s_batch = orpheus_calloc(batch_nthetas, sizeof(int));
        int *elb3s_batch = orpheus_calloc(batch_nthetas, sizeof(int));
        
        double complex *nextM4correlators = orpheus_calloc(8, sizeof(double complex));
        double complex *thisnpcf = orpheus_calloc(8*batchgamma_thetshift, sizeof(double complex));
        //printf("Done allocations for thetabatch %d/%d on thread %d with %d thetacombis\n",
        //       elthetbatch,nthetbatches,thisthread,nthetacombis_batches[elthetbatch]);
        
        #pragma omp critical
        {
            for (int elb=0;elb<batch_nthetas;elb++){
                int thisrcombi = thetacombis_batches[cumthetacombis_batches[elthetbatch]+elb];
                elb1s_batch[elb] = thisrcombi/(nbinsr*nbinsr);
                elb2s_batch[elb] = (thisrcombi-elb1s_batch[elb]*nbinsr*nbinsr)/nbinsr;
                elb3s_batch[elb] = thisrcombi-elb1s_batch[elb]*nbinsr*nbinsr-elb2s_batch[elb]*nbinsr;
                if ((elb1s_batch[elb]>=nbinsr) || (elb2s_batch[elb]>=nbinsr) || (elb3s_batch[elb]>=nbinsr) 
                   || (elb1s_batch[elb]<0) || (elb2s_batch[elb]<0) || (elb3s_batch[elb]<0)){
                    printf("Error for index %d in thetabatch %d (rcombi %d) with el1=%d el2=%d el3=%d",
                         elb,elthetbatch,thisrcombi,elb1s_batch[elb],elb2s_batch[elb],elb3s_batch[elb]);}
            }
            bin_edges[0] = rmin;
            for (int elb=0;elb<nbinsr;elb++){
                bin_edges[elb+1] = bin_edges[elb]*drbin;
                bin_centers[elb] = .5*(bin_edges[elb]+bin_edges[elb+1]);
            }
        }
        double dtheta_xi = (thetamax_xi-thetamin_xi)/nthetabins_xi;
        
        // For each theta combination (theta1,theta2,theta3) in this batch 
        for (int elb=0;elb<batch_nthetas;elb++){
            
            // Get the analytic gaussian 4pcf from the 2pcf
            if (nsubr==1){
                gauss4pcf_analytic(bin_centers[elb1s_batch[elb]],
                                   bin_centers[elb2s_batch[elb]],
                                   bin_centers[elb3s_batch[elb]], phibins, nbinsphi,
                                   xip, xim, thetamin_xi, thetamax_xi, dtheta_xi, 
                                   thisnpcf);
            }
            else{
                gauss4pcf_analytic_integrated(elb1s_batch[elb], 
                                              elb2s_batch[elb], 
                                              elb3s_batch[elb], 
                                              nsubr,
                                              bin_edges, nbinsr,
                                              phibins, nbinsphi,
                                              xip, xim, thetamin_xi, thetamax_xi, dtheta_xi,
                                              thisnpcf);
            }
            
            // Update the aperture Map^4 integral
            double y1, y2, y3, dy1, dy2, dy3;
            int map4ind;
            int map4threadshift = thisthread*8*nmapradii;
            for (int elmapr=0; elmapr<nmapradii; elmapr++){
                y1=bin_centers[elb1s_batch[elb]]/mapradii[elmapr];
                y2=bin_centers[elb2s_batch[elb]]/mapradii[elmapr];
                y3=bin_centers[elb3s_batch[elb]]/mapradii[elmapr];
                dy1 = (bin_edges[elb1s_batch[elb]+1]-bin_edges[elb1s_batch[elb]])/mapradii[elmapr];
                dy2 = (bin_edges[elb2s_batch[elb]+1]-bin_edges[elb2s_batch[elb]])/mapradii[elmapr];
                dy3 = (bin_edges[elb3s_batch[elb]+1]-bin_edges[elb3s_batch[elb]])/mapradii[elmapr];
                fourpcf2M4correlators(1,
                                      y1, y2, y3, dy1, dy2, dy3,
                                      phibins, phibins, dbinsphi, dbinsphi, nbinsphi, nbinsphi,
                                      thisnpcf, nextM4correlators);
                for (int elcomp=0;elcomp<8;elcomp++){
                    map4ind = elcomp*nmapradii+elmapr;
                    if (isfinite(cabs(nextM4correlators[elcomp]))){
                        allM4correlators[map4threadshift+map4ind] += nextM4correlators[elcomp];
                    }
                    nextM4correlators[elcomp] = 0;
                }
            }
            
            // Reset 4pcf placeholders to zero
            for(int i=0;i<8*batchgamma_thetshift;i++){thisnpcf[i] = 0;}
        }        
        
        //if (thisthread>-1){printf("Done allocating 4pcfs for thetabatch %d/%d\n",elthetbatch,nthetbatches);}
                
        free(bin_centers);
        free(bin_edges);
        free(elb1s_batch);
        free(elb2s_batch);
        free(elb3s_batch);
        free(nextM4correlators);
        free(thisnpcf);
        bin_centers = NULL;
        bin_edges = NULL;
        elb1s_batch = NULL;
        elb2s_batch = NULL;
        elb3s_batch = NULL;
        nextM4correlators = NULL;
        thisnpcf = NULL;
                
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
    
void gauss4pcf_analytic_integrated(
    int indbin1, int indbin2, int indbin3, int nsubr, double *rbin_edges, int nbinsr, double *phis, int nphis, 
    double *xip, double *xim, double thetamin_xi, double thetamax_xi, double dtheta_xi,
    double complex *gaussfourpcf){
    
    double dtheta1, dtheta2, dtheta3, subshift, subsubshift, thisw, wtot;
    double *theta1_subs = orpheus_calloc(nsubr, sizeof(double));
    double *theta2_subs = orpheus_calloc(nsubr, sizeof(double));
    double *theta3_subs = orpheus_calloc(nsubr, sizeof(double));
    
    // We define the subsampling in a way s.t. the subsampled bin values are different for the different thetas.
    // In particular, the values for theta2 are the `true` subsampled ones, while we shift the values of
    // theta1 and theta3 by +-1/3 of the subsampling bin width.
    dtheta1 = rbin_edges[indbin1+1] - rbin_edges[indbin1];
    dtheta2 = rbin_edges[indbin2+1] - rbin_edges[indbin2];
    dtheta3 = rbin_edges[indbin3+1] - rbin_edges[indbin3];
    subsubshift = 1./(3*nsubr);
    for (int elsub=0; elsub<nsubr; elsub++){
        subshift = (1.+2*elsub)/(2*nsubr);
        theta1_subs[elsub] = rbin_edges[indbin1] + dtheta1*(subshift + 0*subsubshift);
        theta2_subs[elsub] = rbin_edges[indbin2] + dtheta2*(subshift + 0);
        theta3_subs[elsub] = rbin_edges[indbin3] + dtheta3*(subshift - 0*subsubshift);    
        //printf("%.2f %.2f\n ",subshift,theta2_subs[elsub]);
    }
    
    // Run through all possible combinations of subsampled bin centers, evaluate the 
    // corresponding 4pcf and add it to the bin-averaged 4pcf
    wtot = 0;
    for (int elsub1=0; elsub1<nsubr; elsub1++){
        for (int elsub2=0; elsub2<nsubr; elsub2++){
            for (int elsub3=0; elsub3<nsubr; elsub3++){
                thisw = 1;
                double complex *nextfourpcf = orpheus_calloc(8*nphis*nphis, sizeof(double complex));
                gauss4pcf_analytic(theta1_subs[elsub1],
                                   theta2_subs[elsub2],
                                   theta3_subs[elsub3], phis, nphis,
                                   xip, xim, thetamin_xi, thetamax_xi, dtheta_xi, nextfourpcf);
                for (int ind=0; ind<8*nphis*nphis; ind++){gaussfourpcf[ind] += thisw*nextfourpcf[ind];}
                free(nextfourpcf);
                wtot += thisw;
            }
        }
    }
    
    // Normalize bin-averaged 4pcf
    for (int ind=0; ind<8*nphis*nphis; ind++){gaussfourpcf[ind] /= wtot;}
    
    free(theta1_subs);
    free(theta2_subs);
    free(theta3_subs);
}


// gaussfourpcf has shape (8,nphis,nphis)
// assume xip, xim have linear bin-edges
// X4 --> X1 --> X2 --> X3 convention
void gauss4pcf_analytic(double theta1, double theta2, double theta3, double *phis, int nphis,
                        double *xip, double *xim, double thetamin_xis, double thetamax_xis, double dtheta_xis,
                        double complex *gaussfourpcf){
    double complex y1, y2, y3;
    double complex ang1, ang2, ang3, ang1_3, ang1_4, ang2_2, ang2_4, ang3_3, ang3_4;
    double complex ang12, ang12_4, ang13, ang13_4, ang23, ang23_4;
    double absy12, absy13, absy23;
    double xip_1, xip_2, xip_3, xip_12, xip_13, xip_23, xim_1, xim_2, xim_3, xim_12, xim_13, xim_23;
    int phishift;
    double complex xprojs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int nphis_2 = nphis*nphis;
    
    y1 = (double complex) theta1;
    ang1 = y1/cabs(y1);
    ang1_3 = ang1*ang1*ang1;
    ang1_4 = ang1_3*ang1;
    xip_1 = linint(xip, theta1, thetamin_xis, thetamax_xis, dtheta_xis);//xip[eltheta1];
    xim_1 = linint(xim, theta1, thetamin_xis, thetamax_xis, dtheta_xis);
    xip_2 = linint(xip, theta2, thetamin_xis, thetamax_xis, dtheta_xis);
    xim_2 = linint(xim, theta2, thetamin_xis, thetamax_xis, dtheta_xis);
    xip_3 = linint(xip, theta3, thetamin_xis, thetamax_xis, dtheta_xis);
    xim_3 = linint(xim, theta3, thetamin_xis, thetamax_xis, dtheta_xis);
    for (int elphi1=0;elphi1<nphis;elphi1++){
        y2 = theta2*cexp(I*phis[elphi1]);
        ang2 = y2/cabs(y2);
        ang2_2 = ang2*ang2;
        ang2_4 = ang2_2*ang2_2;
        absy12 = cabs(y2-y1);
        ang12 = (y2-y1)/absy12;
        ang12_4 = ang12*ang12*ang12*ang12;
        xip_12 = linint(xip, absy12, thetamin_xis, thetamax_xis, dtheta_xis);
        xim_12 = linint(xim, absy12, thetamin_xis, thetamax_xis, dtheta_xis);
        for (int elphi2=0;elphi2<nphis;elphi2++){
            phishift = elphi1*nphis+elphi2;
            y3 = theta3*cexp(I*phis[elphi2]);
            ang3 = y3/cabs(y3);
            ang3_3 =ang3*ang3*ang3;
            ang3_4=ang3_3*ang3;
            absy13 = cabs(y3-y1);
            ang13 = (y3-y1)/absy13;
            ang13_4 = ang13*ang13*ang13*ang13;
            xip_13 = linint(xip, absy13, thetamin_xis, thetamax_xis, dtheta_xis);
            xim_13 = linint(xim, absy13, thetamin_xis, thetamax_xis, dtheta_xis);
            absy23 = cabs(y3-y2);
            ang23 = (y3-y2)/absy23;
            ang23_4 = ang23*ang23*ang23*ang23;
            if ((absy12>1e-3) && (absy23>1e-3) && (absy13>1e-3)){
                xip_23 = linint(xip, absy23, thetamin_xis, thetamax_xis, dtheta_xis);
                xim_23 = linint(xim, absy23, thetamin_xis, thetamax_xis, dtheta_xis);
                xprojs[0] = ang1_3       * ang2_2       * ang3_3;
                xprojs[1] = ang1         * ang2_2       * ang3;
                xprojs[2] = conj(ang1)   * ang2_2       * ang3_3;
                xprojs[3] = ang1_3       * conj(ang2_2) * ang3_3;
                xprojs[4] = ang1_3       * ang2_2       * conj(ang3);
                xprojs[5] = conj(ang1_3) * ang2_2       * ang3;
                xprojs[6] = ang1         * conj(ang2_2) * ang3;
                xprojs[7] = ang1         * ang2_2       * conj(ang3_3);
                gaussfourpcf[0*nphis_2+phishift] = conj(xprojs[0]) * (
                    ang23_4*ang1_4*xim_23*xim_1 + ang13_4*ang2_4*xim_13*xim_2 + ang12_4*ang3_4*xim_12*xim_3);
                gaussfourpcf[1*nphis_2+phishift] = conj(xprojs[1]) * (
                    ang23_4*xim_23*xip_1 + ang13_4*xim_13*xip_2 + ang12_4*xim_12*xip_3);
                gaussfourpcf[2*nphis_2+phishift] = conj(xprojs[2]) * (
                    ang23_4*xip_1*xim_23 + ang2_4*xim_2*xip_13 + ang3_4*xim_3*xip_12);
                gaussfourpcf[3*nphis_2+phishift] = conj(xprojs[3]) * (
                    ang1_4*xim_1*xip_23 + ang13_4*xip_2*xim_13 + ang3_4*xim_3*xip_12);
                gaussfourpcf[4*nphis_2+phishift] = conj(xprojs[4]) * (
                    ang1_4*xim_1*xip_23 + ang2_4*xim_2*xip_13 + ang12_4*xip_3*xim_12);
                gaussfourpcf[5*nphis_2+phishift] = conj(xprojs[5]) * (
                    conj(ang1_4)*ang23_4*xim_1*xim_23 + xip_2*xip_13 + xip_3*xip_12);
                gaussfourpcf[6*nphis_2+phishift] = conj(xprojs[6]) * (
                    xip_1*xip_23 + conj(ang2_4)*ang13_4*xim_2*xim_13 + xip_3*xip_12);
                gaussfourpcf[7*nphis_2+phishift] = conj(xprojs[7]) * (
                    xip_1*xip_23 + xip_2*xip_13 + conj(ang3_4)*ang12_4*xim_3*xim_12);
                for (int elcomp=0;elcomp<8;elcomp++){
                    if (!isfinite(cabs(gaussfourpcf[elcomp*nphis_2+phishift]))){gaussfourpcf[elcomp*nphis_2+phishift]=0;}
                }
            }
        }
    }
}

////////////////////////////////////
/// GNNN (G4L) STATISTICS RELATED //
////////////////////////////////////
// Reconstructs all multipole components from the ones with theta1<=theta2<=theta3
// Gtilde_in ~ Nn[elthetbatch] ~ shape (2*nmax_alloc+1,2*nmax_alloc+1)
// Nn_in ~ Nn[elthetbatch] ~ shape (2*nmax_alloc+1,2*nmax_alloc+1)
// Gtilde_out ~  shape (2*nmax+1,2*nmax+1)
// Nn_out ~  shape (2*nmax+1,2*nmax+1)
// 
// Ordering for eltrafo: [123, 231, 312, 132, 213, 321]
// Different configs: 
//  * All three elbs equal --> No permutations needed, i.e. eltrafo in [0]
//  * Only two elbs equal  --> Only cyclic permutations needed, i.e. eltrafo in [0,1,2]
//  * All elbs unequal     --> All permutations needed, i.e. eltrafo in [0,1,2,3,4,5]   
void getMultipolesFromSymm_GNNN(double complex *Gtilden_in, double complex *Nn_in,
                                int nmax, int eltrafo, int *nindices, int len_nindices,
                                double complex *Gtilden_out, double complex *Nn_out){
    
    int nmax_alloc = 2*nmax+1;
    int nzero_in = nmax_alloc;
    int nzero_out = nmax;
    int n2shift_in = 2*nmax_alloc+1;
    int n2shift_out = 2*nmax+1;
    
    int n2, n2_N, n2_Gtilde;
    int n3, n3_N, n3_Gtilde;
    for (int nindex=0; nindex<len_nindices; nindex++){
        n2 = nindices[nindex]/(2*nmax_alloc+1) - nzero_in;
        n3 = nindices[nindex]%(2*nmax_alloc+1) - nzero_in;
        switch (eltrafo){
            case 0:
                n2_N=n2; n3_N=n3;
                n2_Gtilde=n2; n3_Gtilde=n3;
                break;
            case 1:
                n2_N=n3; n3_N=-n2-n3;
                n2_Gtilde=n3+1; n3_Gtilde=-n2-n3;
                break;
            case 2:
                n2_N=-n2-n3; n3_N=n2;
                n2_Gtilde=-n2-n3+1; n3_Gtilde=n2-1;
                break;
            case 3:
                n2_N=n3; n3_N=n2;
                n2_Gtilde=n3+1; n3_Gtilde=n2-1;
                break;
            case 4:
                n2_N=-n2-n3; n3_N=n3;
                n2_Gtilde=-n2-n3+1; n3_Gtilde=n3;
                break;
            case 5:
                n2_N=n2; n3_N=-n2-n3;
                n2_Gtilde=n2; n3_Gtilde=-n2-n3;
                break;
            default:
                n2_N=0; n3_N=0;
                n2_Gtilde=0; n3_Gtilde=0;
                break;
        }
        if ((abs(n2_N)<=nmax) && (abs(n3_N)<=nmax)){
            Nn_out[(nzero_out+n2_N)*n2shift_out+(nzero_out+n3_N)] = Nn_in[(nzero_in+n2)*n2shift_in+(nzero_in+n3)];
        }
         if ((abs(n2_Gtilde)<=nmax) && (abs(n3_Gtilde)<=nmax)){
            Gtilden_out[(nzero_out+n2_Gtilde)*n2shift_out+(nzero_out+n3_Gtilde)] = Gtilden_in[(nzero_in+n2)*n2shift_in+(nzero_in+n3)];
        }
    }
}

// Get index and linear interpolation weight on monotonically increasing grid.
static int locate_lin(double *arr, int n, double x, int *ind, double *w){
    if (x<arr[0] || x>arr[n-1]){return 0;}
    int lo = 0;
    int hi = n-1;
    while (hi-lo>1){
        int mid = (lo+hi)/2;
        if (x<arr[mid]){hi = mid;}
        else{lo = mid;}
    }
    *ind = lo;
    *w = (x-arr[lo])/(arr[lo+1]-arr[lo]);
    return 1;
}

// Clustering correction Ccirr = 1 + omega(d12) + omega(d13) + omega(d23) + zeta(d12,d13,d23)
// to recover the unbiased GNNN correlator Gtilde = C * Upsilon/N 
// xi is passed as a 1D table on a linear theta-grid (zero-filled outside its range), zeta is passed on flattened (r1,r2,phi) grid
double gnnn_clustering_corr(double theta1, double theta2, double theta3,
                            double phi12, double phi13,
                            double *xi_nn, double thetamin_xi, double thetamax_xi, double dtheta_xi, int has_xi,
                            double *zeta, double *zeta_rbins, int zeta_nr, double *zeta_phis, int zeta_nphi, int has_zeta){

    // Get sideslengths
    double cphi12 = cos(phi12);
    double cphi13 = cos(phi13);
    double cphi23 = cos(phi13-phi12);
    double d12 = sqrt(fmax(0., theta1*theta1 + theta2*theta2 - 2*theta1*theta2*cphi12));
    double d13 = sqrt(fmax(0., theta1*theta1 + theta3*theta3 - 2*theta1*theta3*cphi13));
    double d23 = sqrt(fmax(0., theta2*theta2 + theta3*theta3 - 2*theta2*theta3*cphi23));
    double corr = 1.;
    // Build second-order correction
    if (has_xi==1){
        corr += linint(xi_nn, d12, thetamin_xi, thetamax_xi, dtheta_xi);
        corr += linint(xi_nn, d13, thetamin_xi, thetamax_xi, dtheta_xi);
        corr += linint(xi_nn, d23, thetamin_xi, thetamax_xi, dtheta_xi);
    }
    // Build third-order correction
    if (has_zeta==1 && d12>0 && d13>0){
        int i1, i2, i3;
        double w1, w2, w3;
        if ((locate_lin(zeta_rbins, zeta_nr, d12, &i1, &w1)==1) && (locate_lin(zeta_rbins, zeta_nr, d13, &i2, &w2)==1)){
            double cpsi = (d12*d12 + d13*d13 - d23*d23)/(2*d12*d13);
            if (cpsi>1.){cpsi = 1.;}
            if (cpsi<-1.){cpsi = -1.;}
            double psi = acos(cpsi);
            if (psi<=zeta_phis[0]){i3 = 0; w3 = 0.;}
            else if (psi>=zeta_phis[zeta_nphi-1]){i3 = zeta_nphi-2; w3 = 1.;}
            // The bracketing above already implies success, but locate_lin leaves its
            // outputs untouched on failure, so an unchecked call would index zeta with
            // an uninitialised i3 should that invariant ever break
            else if (locate_lin(zeta_phis, zeta_nphi, psi, &i3, &w3)!=1){i3 = 0; w3 = 0.;}
            double z_ll = (1-w3)*zeta[(i1*zeta_nr+i2)*zeta_nphi+i3] + w3*zeta[(i1*zeta_nr+i2)*zeta_nphi+i3+1];
            double z_lu = (1-w3)*zeta[(i1*zeta_nr+i2+1)*zeta_nphi+i3] + w3*zeta[(i1*zeta_nr+i2+1)*zeta_nphi+i3+1];
            double z_ul = (1-w3)*zeta[((i1+1)*zeta_nr+i2)*zeta_nphi+i3] + w3*zeta[((i1+1)*zeta_nr+i2)*zeta_nphi+i3+1];
            double z_uu = (1-w3)*zeta[((i1+1)*zeta_nr+i2+1)*zeta_nphi+i3] + w3*zeta[((i1+1)*zeta_nr+i2+1)*zeta_nphi+i3+1];
            corr += (1-w1)*((1-w2)*z_ll + w2*z_lu) + w1*((1-w2)*z_ul + w2*z_uu);
        }
    }
    return corr;
}

// Upsilon_n has shape (8,nphi12,nphi13)
void multipoles2npcf_gnnn_singletheta(double complex *Gtilde_n, double complex *N_n, int n1max, int n2max,
                                      double theta1, double theta2, double theta3,
                                      double *phis12, double *phis13, int nbinsphi12, int nbinsphi13,
                                      const ClustCorr *cc,
                                      double complex *npcf, double complex *npcf_norm){
    double count_floor = cc->count_floor;
    double *xi_nn = cc->xi_nn; double thetamin_xi = cc->thetamin_xi, thetamax_xi = cc->thetamax_xi, dtheta_xi = cc->dtheta_xi; int has_xi = cc->has_xi;
    double *zeta = cc->zeta, *zeta_rbins = cc->zeta_rbins; int zeta_nr = cc->zeta_nr; double *zeta_phis = cc->zeta_phis; int zeta_nphi = cc->zeta_nphi; int has_zeta = cc->has_zeta;
    int nmax = n1max;
    int nns = 2*nmax+1;
    double complex expphi12, expphi13;
    double complex *expphi12s = orpheus_calloc(nns, sizeof(double complex));
    double complex *expphi13s = orpheus_calloc(nns, sizeof(double complex));
    for (int elphi12=0; elphi12<nbinsphi12; elphi12++){
        for (int elphi13=0; elphi13<nbinsphi13; elphi13++){
            // Convert multipoles to npcf
            expphi12s[nmax] = 1;
            expphi13s[nmax] = 1;
            expphi12 = cexp(I*phis12[elphi12]);
            expphi13 = cexp(I*phis13[elphi13]);
            for (int nextn=1; nextn<=nmax; nextn++){
                expphi12s[nmax+nextn] = expphi12s[nmax+nextn-1]*expphi12;
                expphi12s[nmax-nextn] = conj(expphi12s[nmax+nextn]);
                expphi13s[nmax+nextn] = expphi13s[nmax+nextn-1]*expphi13;
                expphi13s[nmax-nextn] = conj(expphi13s[nmax+nextn]);
            }
            double complex nextang;
            int ind_npcf = elphi12*nbinsphi13 + elphi13;
            for (int nextn1=0; nextn1<nns; nextn1++){
                for (int nextn2=0; nextn2<nns; nextn2++){
                    int ind_ups = nextn1*nns + nextn2;
                    nextang = INV_2PI * expphi12s[nextn1] * expphi13s[nextn2];
                    npcf[ind_npcf] += Gtilde_n[ind_ups]*nextang;
                    npcf_norm[ind_npcf] += N_n[ind_ups]*nextang;
                }
            }

            // Normalize Gtilde--> Make sure that we have counts, i.e. N >~ 1.
            // We treat set small values as zero, as those could be interpreted as ringing effects from the multipole-based reconstruction, i.e. they are oscillating around zero.
            if (cabs(npcf_norm[ind_npcf]) > count_floor){npcf[ind_npcf] /= creal(npcf_norm[ind_npcf]);}
            else{npcf[ind_npcf] = 0;}

            // Optionally debias the estimator by applying the 2pt & 3pt clustering correction
            if (has_xi==1 || has_zeta==1){
                npcf[ind_npcf] *= gnnn_clustering_corr(
                    theta1, theta2, theta3,
                    phis12[elphi12], phis13[elphi13],
                    xi_nn, thetamin_xi, thetamax_xi, dtheta_xi, has_xi,
                    zeta, zeta_rbins, zeta_nr, zeta_phis, zeta_nphi, has_zeta);
            }
        }
    }
    free(expphi12s);
    free(expphi13s);
}

// Upsilon_n has shape (8,n1max+1,n2max+1,nphi12,nphi13)
void multipoles2npcf_gnnn_singletheta_nconvergence(
    double complex *Upsilon_n, double complex *N_n, int n1max, int n2max,
    double theta1, double theta2, double theta3,
    double *phis12, double *phis13, int nbinsphi12, int nbinsphi13,
    const ClustCorr *cc,
    double complex *npcf, double complex *npcf_norm){

    double count_floor = cc->count_floor;
    double *xi_nn = cc->xi_nn; double thetamin_xi = cc->thetamin_xi, thetamax_xi = cc->thetamax_xi, dtheta_xi = cc->dtheta_xi; int has_xi = cc->has_xi;
    double *zeta = cc->zeta, *zeta_rbins = cc->zeta_rbins; int zeta_nr = cc->zeta_nr; double *zeta_phis = cc->zeta_phis; int zeta_nphi = cc->zeta_nphi; int has_zeta = cc->has_zeta;
    int n_cfs = 1;
    int nmax = n1max;
    int nns = 2*nmax+1;
    double complex expphi12, expphi13;
    double complex *expphi12s = orpheus_calloc(nns, sizeof(double complex));
    double complex *expphi13s = orpheus_calloc(nns, sizeof(double complex));
    double complex *projdir = orpheus_calloc(n_cfs, sizeof(double complex));
    int npcf_n2cutshift = nbinsphi12*nbinsphi13;
    int npcf_n1cutshift = (n2max+1)*nbinsphi12*nbinsphi13;
    int npcf_compshift = (n1max+1)*(n2max+1)*nbinsphi12*nbinsphi13;
    int ups_compshift = nns*nns;
    for (int elphi12=0; elphi12<nbinsphi12; elphi12++){
        for (int elphi13=0; elphi13<nbinsphi13; elphi13++){
            // Convert multipoles to npcf
            expphi12s[nmax] = 1;
            expphi13s[nmax] = 1;
            expphi12 = cexp(I*phis12[elphi12]);
            expphi13 = cexp(I*phis13[elphi13]);
            for (int nextn=1; nextn<=nmax; nextn++){ 
                expphi12s[nmax+nextn] = expphi12s[nmax+nextn-1]*expphi12;
                expphi12s[nmax-nextn] = conj(expphi12s[nmax+nextn]);
                expphi13s[nmax+nextn] = expphi13s[nmax+nextn-1]*expphi13;
                expphi13s[nmax-nextn] = conj(expphi13s[nmax+nextn]);
            }
            double complex nextang;
            for (int n1cut=0; n1cut<=nmax; n1cut++){
                for (int n2cut=0; n2cut<=nmax; n2cut++){ 
                    //printf("Doing n1c=%d n2c=%d",n1cut,n2cut);
                    int ind_npcf = n1cut*npcf_n1cutshift + n2cut*npcf_n2cutshift + elphi12*nbinsphi13 + elphi13;
                    for (int nextn1=-n1cut; nextn1<=n1cut; nextn1++){
                        for (int nextn2=-n2cut; nextn2<=n2cut; nextn2++){ 
                            int ind_ups = (nmax+nextn1)*nns + (nmax+nextn2);
                            nextang = INV_2PI * expphi12s[nmax+nextn1] * expphi13s[nmax+nextn2];
                            npcf_norm[ind_npcf] += N_n[ind_ups]*nextang;
                            for (int elcf=0; elcf<n_cfs; elcf++){ 
                                npcf[elcf*npcf_compshift + ind_npcf] += Upsilon_n[elcf*ups_compshift + ind_ups]*nextang;
                            }
                        }
                    }
                    // Normalize: Gamma=Upsilon/N --> Make sure that we have counts, i.e. N >~ 1.
                    for (int elcf=0; elcf<n_cfs; elcf++){
                        if (cabs(npcf_norm[ind_npcf]) > count_floor){npcf[elcf*npcf_compshift + ind_npcf] /= creal(npcf_norm[ind_npcf]);}
                        else{npcf[elcf*npcf_compshift + ind_npcf] = 0;}
                        if (has_xi==1 || has_zeta==1){
                            npcf[elcf*npcf_compshift + ind_npcf] *= gnnn_clustering_corr(
                                theta1, theta2, theta3,
                                phis12[elphi12], phis13[elphi13],
                                xi_nn, thetamin_xi, thetamax_xi, dtheta_xi, has_xi,
                                zeta, zeta_rbins, zeta_nr, zeta_phis, zeta_nphi, has_zeta);
                        }
                    }
                }
            }
        }
    }
    free(expphi12s);
    free(expphi13s);
    free(projdir);
}

// Convert full GNNN 4pcf multipoles to the real-space basis
void multipoles2npcf_gnnn(const double complex *Gtilde_n, const double complex *N_n,
                          const BinningParams *bin, const FourthParams *fourth, const ClustCorr *clust_corr,
                          int nthreads, double complex *npcf, double complex *npcf_norm){
    int nmax = bin->nmax, nbinsr = bin->nbinsr;
    double rmin = bin->rmin, rmax = bin->rmax;
    double *rbins = bin->rbins;
    int nbinsphi1 = fourth->nbinsphi1, nbinsphi2 = fourth->nbinsphi2;
    double *phibins1 = fourth->phibins1, *phibins2 = fourth->phibins2;
    int nns = 2*nmax+1;
    int nbinsr2 = nbinsr*nbinsr;
    int nbinsr3 = nbinsr*nbinsr*nbinsr;
    int nphicombis = nbinsphi1*nbinsphi2;

    // Geometric bin centers use the explicit edges when passed (bin->rbins), else log-spaced
    double *bin_edges = orpheus_calloc(nbinsr+1, sizeof(double));
    if (rbins!=NULL){
        for (int elb=0;elb<=nbinsr;elb++){bin_edges[elb] = rbins[elb];}
    }
    else{
        double drbin = log(rmax/rmin)/nbinsr;
        bin_edges[0] = rmin;
        for (int elb=0;elb<nbinsr;elb++){bin_edges[elb+1] = bin_edges[elb]*exp(drbin);}
    }

    #pragma omp parallel for num_threads(nthreads)
    for (int nrcombi=0; nrcombi<nbinsr3; nrcombi++){
        int elr1 = nrcombi/nbinsr2;
        int elr2 = (nrcombi-elr1*nbinsr2)/nbinsr;
        int elr3 = nrcombi%nbinsr;
        double complex *npcf_slice = orpheus_calloc(nns*nns, sizeof(double complex));
        double complex *norm_slice = orpheus_calloc(nns*nns, sizeof(double complex));
        for (int n1=0;n1<nns;n1++){
            for (int n2=0;n2<nns;n2++){
                int ind = (n1*nns+n2)*nbinsr3 + elr1*nbinsr2 + elr2*nbinsr + elr3;
                npcf_slice[n1*nns+n2] = Gtilde_n[ind];
                norm_slice[n1*nns+n2] = N_n[ind];
            }
        }
        int off = nrcombi*nphicombis;
        multipoles2npcf_gnnn_singletheta(npcf_slice, norm_slice, nmax, nmax,
                                         sqrt(bin_edges[elr1]*bin_edges[elr1+1]),
                                         sqrt(bin_edges[elr2]*bin_edges[elr2+1]),
                                         sqrt(bin_edges[elr3]*bin_edges[elr3+1]),
                                         phibins1, phibins2, nbinsphi1, nbinsphi2,
                                         clust_corr,
                                         npcf+off, npcf_norm+off);
        free(npcf_slice);
        free(norm_slice);
    }
    free(bin_edges);
}

void fourpcfmultipoles2MN3correlators(
    int nmax, int nmax_trafo,
    double *theta_edges, double *theta_centers, int nthetas,
    double *apradii_N, double *apradii_M, int napradii,
    double *phis1, double *phis2, double *dphis1, double *dphis2, int nbinsphi1, int nbinsphi2,
    int projection, int nthreads, int verbose,
    const ClustCorr *cc,
    double complex *Upsilon_n, double complex *N_n, double complex *mn3corr){

    int n_cfs = 1;
    double complex *allmn3corr = orpheus_calloc(nthreads*n_cfs*napradii, sizeof(double complex));

    int trafos_finished = 0;
    reset_progress();
    #pragma omp parallel for num_threads(nthreads)
    for (int thetacombi=0; thetacombi<nthetas*nthetas*nthetas; thetacombi++){
        
        int thisthread = omp_get_thread_num();
        
        int nphicombis = nbinsphi1*nbinsphi2;
        int n2n3combis = (2*nmax+1)*(2*nmax+1);
        int n2n3combis_trafo = (2*nmax_trafo+1)*(2*nmax_trafo+1);
        int nthetas2 = nthetas*nthetas;
        int nthetas3 = nthetas*nthetas*nthetas;
        int compshift = n2n3combis*nthetas3;
        int ithet1 = thetacombi/nthetas2;
        int ithet2 = (thetacombi-nthetas2*ithet1)/nthetas;
        int ithet3 = thetacombi%nthetas;

        double theta1, theta2, theta3, dtheta1, dtheta2, dtheta3;
        #pragma omp critical
        {
            theta1 = sqrt(theta_edges[ithet1]*theta_edges[ithet1+1]);
            theta2 = sqrt(theta_edges[ithet2]*theta_edges[ithet2+1]);
            theta3 = sqrt(theta_edges[ithet3]*theta_edges[ithet3+1]);
            dtheta1 = theta_edges[ithet1+1]-theta_edges[ithet1];
            dtheta2 = theta_edges[ithet2+1]-theta_edges[ithet2];
            dtheta3 = theta_edges[ithet3+1]-theta_edges[ithet3];
        }

        // Transform multipoles to 4pcf
        int thisn1, thisn2, n2n3combi_trafo;
        double complex *thisnpcf = orpheus_calloc(n_cfs*nphicombis, sizeof(double complex));
        double complex *thisnpcf_norm = orpheus_calloc(nphicombis, sizeof(double complex));
        double complex *Upsn_single = orpheus_calloc(n_cfs*n2n3combis_trafo, sizeof(double complex));
        double complex *Nn_single = orpheus_calloc(1*n2n3combis_trafo, sizeof(double complex));
        for (int elcomp=0; elcomp<n_cfs; elcomp++){
            for (int n2n3combi=0; n2n3combi<n2n3combis; n2n3combi++){
                thisn1 = n2n3combi/(2*nmax+1) - nmax;
                thisn2 = n2n3combi%(2*nmax+1) - nmax;
                if ((abs(thisn1)<=nmax_trafo) && (abs(thisn2)<=nmax_trafo)){
                    n2n3combi_trafo = (thisn1+nmax_trafo)*(2*nmax_trafo+1) + (thisn2+nmax_trafo);
                    Upsn_single[elcomp*n2n3combis_trafo+n2n3combi_trafo] = 
                        Upsilon_n[elcomp*compshift+n2n3combi*nthetas3+thetacombi];
                    Nn_single[n2n3combi_trafo] = 
                        N_n[n2n3combi*nthetas3+thetacombi];
                }
            }
        }
        multipoles2npcf_gnnn_singletheta(Upsn_single, Nn_single, nmax_trafo, nmax_trafo,
                                         theta1, theta2, theta3,
                                         phis1, phis2, nbinsphi1, nbinsphi2,
                                         cc,
                                         thisnpcf, thisnpcf_norm);


        // Update the MN3 statistics
        double complex nextmn3corr[1] = {0};
        double R_N, R_M;
        int mn3threadshift = thisthread*n_cfs*napradii;
        for (int elapr=0; elapr<napradii; elapr++){
            R_N = apradii_N[elapr];
            R_M = apradii_M[elapr];
            fourpcf2MN3correlatormulti(1, R_N, R_N, R_N, R_M,
                                    theta1, theta2, theta3, dtheta1, dtheta2, dtheta3,
                                    phis1, phis2, dphis1, dphis2, nbinsphi1, nbinsphi2,
                                    thisnpcf, nextmn3corr);
            for (int elcomp=0;elcomp<n_cfs;elcomp++){
                if (isfinite(cabs(nextmn3corr[elcomp]))){
                    allmn3corr[mn3threadshift+elcomp*napradii+elapr] += nextmn3corr[elcomp];
                }
                nextmn3corr[elcomp] = 0;
            }
        }

        /*
        // Transform 4pcf to M4
        double complex nextmn3corr[1] = {0};
        double y1, y2, y3, dy1, dy2, dy3, R_ap;
        int mn3threadshift = thisthread*n_cfs*napradii;
        for (int elmapr=0; elmapr<napradii; elmapr++){
            #pragma omp critical
            {R_ap=apradii[elmapr];}
            y1=theta1/R_ap; y2=theta2/R_ap; y3=theta3/R_ap;
            dy1 = dtheta1/R_ap; dy2 = dtheta2/R_ap; dy3 = dtheta3/R_ap;
            fourpcf2MN3correlator(1,
                                  y1, y2, y3, dy1, dy2, dy3,
                                  phis1, phis2, dphis1, dphis2, nbinsphi1, nbinsphi2,
                                  thisnpcf, nextmn3corr);
            for (int elcomp=0;elcomp<n_cfs;elcomp++){
                if (isfinite(cabs(nextmn3corr[elcomp]))){
                    allmn3corr[mn3threadshift+elcomp*napradii+elmapr] += nextmn3corr[elcomp];
                }
                nextmn3corr[elcomp] = 0;
            }
        }*/
        free(thisnpcf);
        free(thisnpcf_norm);
        free(Upsn_single);
        free(Nn_single);
        
        #pragma omp atomic
        trafos_finished+=1;
        print_progress(trafos_finished, nthetas3, verbose);
    }
    if (verbose>0){ printf("\n"); }

    // Accumulate the MN3correlators
    for (int elthread=0;elthread<nthreads;elthread++){
        for (int elcr=0;elcr<n_cfs*napradii;elcr++){
            mn3corr[elcr] += allmn3corr[elthread*n_cfs*napradii+elcr];
        }
    }

    free(allmn3corr);
}

void fourpcf2MN3correlator(int nzcombis,
                           double y1, double y2, double y3, double dy1, double dy2, double dy3,
                           double *phis1, double *phis2, double *dphis1, double *dphis2, int nbinsphi1, int nbinsphi2,
                           double complex *fourpcf, double complex *mn3corr){
    int ind_4pcf;
    double complex exp12_sqrt, exp13_sqrt, exp23_sqrt, exp2312_sqrt, exp12, exp13, exp23, exp12c, exp13c, exp23c;
    double complex g1, g2, g3, g1c, g2c, g3c, h, g1g2cexp12_re, g1g3cexp13_re, g2g3cexp23_re;
    double F1, F2, F3, csq, b, measure;
    double complex t1, t2, t3, nextF;
    for (int elphi1=0;elphi1<nbinsphi1;elphi1++){
        exp12_sqrt=cexp(I*.5*phis1[elphi1]); exp12=exp12_sqrt*exp12_sqrt; exp12c = conj(exp12);
        for (int elphi2=0;elphi2<nbinsphi2;elphi2++){
            exp13_sqrt=cexp(I*.5*phis2[elphi2]); exp13=exp13_sqrt*exp13_sqrt; exp13c = conj(exp13);
            exp23_sqrt=exp13_sqrt*conj(exp12_sqrt); exp23=exp23_sqrt*exp23_sqrt; exp23c = conj(exp23); exp2312_sqrt=exp23_sqrt*conj(exp12_sqrt);
            g1 = y1 - 0.25*(y1        + y2*exp12c + y3*exp13c); g1c = conj(g1);
            g2 = y2 - 0.25*(y1*exp12  + y2        + y3*exp23c); g2c = conj(g2);
            g3 = y3 - 0.25*(y1*exp13  + y2*exp23  + y3       ); g3c = conj(g3);
            h = 0.25*(y1*exp13_sqrt + y2*exp2312_sqrt + y3*conj(exp13_sqrt));
            F1=2-creal(g1*g1c); F2=2-creal(g2*g2c); F3=2-creal(g3*g3c);
            g1g2cexp12_re=creal(g1*g2c*exp12); g1g3cexp13_re=creal(g1*g3c*exp13); g2g3cexp23_re=creal(g2*g3c*exp23);
            t1 = 2*g1g2cexp12_re*(F3-1) + 2*g1g3cexp13_re*(F2-1) + 2*g2g3cexp23_re*(F1-1) + 2*F1*F2*F3 - (F1*F2+F1*F3+F2*F3) + (F1+F2+F3-1.5);
            t2 = g1*exp13_sqrt*(g2g3cexp23_re+F2*F3-(F2+F3-1.5)) + g2*exp2312_sqrt*(g1g3cexp13_re+F1*F3-(F1+F3-1.5)) + g3*conj(exp13_sqrt)*(g1g2cexp12_re+F1*F2-(F1+F2-1.5));
            t3 = g1*g2*exp23*(F3-1.5) + g1*g3*(F2-1.5) + g2*g3*exp12c*(F1-1.5);
            csq = 1./16 * ( y1*y1+y2*y2+y3*y3 + 2*y1*y2*creal(exp12)+2*y1*y3*creal(exp13)+2*y2*y3*creal(exp23) );
            b = 0.5*(y1*y1+y2*y2+y3*y3) - 2*csq;
            measure = y1*y2*y3*dy1*dy2*dy3 * dphis1[elphi1]*dphis2[elphi2]*INV_2PI*INV_2PI;
            nextF = exp(-b)/128 * measure * (h*h*t1 + 2*h*t2 + t3);
            for (int zcombi=0;zcombi<nzcombis;zcombi++){
                ind_4pcf = zcombi*nbinsphi1*nbinsphi2 + elphi1*nbinsphi2 + elphi2;
                mn3corr[zcombi] += nextF * fourpcf[ind_4pcf];
            } 
        }
    } 
}

void fourpcf2MN3correlatormulti(int nzcombis, double R1, double R2, double R3, double R4,
                           double theta1, double theta2, double theta3, double dtheta1, double dtheta2, double dtheta3,
                           double *phis1, double *phis2, double *dphis1, double *dphis2, int nbinsphi1, int nbinsphi2,
                           double complex *fourpcf, double complex *mn3corr){
    int ind_4pcf;
    double complex exp12_sqrt, exp13_sqrt, exp23_sqrt, exp2312_sqrt, exp12, exp13, exp23, exp12c, exp13c, exp23c;
    double cos12, cos13, cos23;
    double complex g1, g2, g3, g2c, g3c, h, g1g2cexp12_re, g1g3cexp13_re, g2g3cexp23_re;
    double F1, F2, F3, b, measure;
    double complex t1, t2, t3, nextF;
    
    // Helpers that do not depend on angles
    double R1_2 = R1*R1; double R2_2 =R2*R2; double R3_2 = R3*R3; double R4_2 = R4*R4; 
    double Theta_6 = 0.25*(R1_2*R2_2*R3_2 + R1_2*R2_2*R4_2 + R1_2*R3_2*R4_2 + R2_2*R3_2*R4_2); double Theta_12=Theta_6*Theta_6;
    double a = 2./(1./R1_2 + 1./R2_2 + 1./R3_2 + 1./R4_2);double a2=2*a; double a3=3*a;
    double b1 = 0.5*( theta1*theta1/R1_2*(1-a/(2*R1_2)) + theta2*theta2/R2_2*(1-a/(2*R2_2)) + theta3*theta3/R3_2*(1-a/(2*R3_2)) );
    double b2_12 = 2*theta1*theta2/R1_2/R2_2; double b2_13 = 2*theta1*theta3/R1_2/R3_2; double b2_23 = 2*theta2*theta3/R2_2/R3_2; 
    double h_1 = theta1/R1_2; double h_2 = theta2/R2_2; double h_3 = theta3/R3_2; 
    for (int elphi1=0;elphi1<nbinsphi1;elphi1++){
        exp12_sqrt=cexp(I*.5*phis1[elphi1]); exp12=exp12_sqrt*exp12_sqrt; exp12c = conj(exp12);
        for (int elphi2=0;elphi2<nbinsphi2;elphi2++){
            // Helpers that depend on angles
            exp13_sqrt=cexp(I*.5*phis2[elphi2]); exp13=exp13_sqrt*exp13_sqrt; exp13c = conj(exp13);
            exp23_sqrt=exp13_sqrt*conj(exp12_sqrt); exp23=exp23_sqrt*exp23_sqrt; exp23c = conj(exp23); exp2312_sqrt=exp23_sqrt*conj(exp12_sqrt);
            cos12=creal(exp12); cos13=creal(exp13); cos23=creal(exp23);
            b = b1 - 0.25*a*(b2_12*cos12 + b2_13*cos13 + b2_23*cos23);
            h = 0.5*a*( h_1*exp13_sqrt + h_2*exp2312_sqrt + h_3*conj(exp13_sqrt) );
            g1 = theta1 - 0.5*a*(h_1        + h_2*exp12c + h_3*exp13c);
            g2 = theta2 - 0.5*a*(h_1*exp12  + h_2        + h_3*exp23c); g2c = conj(g2);
            g3 = theta3 - 0.5*a*(h_1*exp13  + h_2*exp23  + h_3       ); g3c = conj(g3);
            g1g2cexp12_re=creal(g1*g2c*exp12); g1g3cexp13_re=creal(g1*g3c*exp13); g2g3cexp23_re=creal(g2*g3c*exp23);
            F1=2*R1_2-cabs2(g1); F2=2*R2_2-cabs2(g2); F3=2*R3_2-cabs2(g3);
            // Final expression
            t1 = 2*g1g2cexp12_re*(F3-a2) + 2*g1g3cexp13_re*(F2-a2) + 2*g2g3cexp23_re*(F1-a2) + F1*F2*F3/a - (F1*F2+F1*F3+F2*F3) + a2*(F1+F2+F3-a3);
            t2 = g1*exp13_sqrt*(a2*g2g3cexp23_re+F2*F3-a2*(F2+F3-a3)) + g2*exp2312_sqrt*(a2*g1g3cexp13_re+F1*F3-a2*(F1+F3-a3)) + g3*conj(exp13_sqrt)*(a2*g1g2cexp12_re+F1*F2-a2*(F1+F2-a3));
            t3 = g1*g2*exp23*(F3-a3) + g1*g3*(F2-a3) + g2*g3*exp12c*(F1-a3);
            measure = theta1*theta2*theta3*dtheta1*dtheta2*dtheta3 * dphis1[elphi1]*dphis2[elphi2]*INV_2PI*INV_2PI;
            nextF = exp(-b)/128/Theta_12 * measure * (h*h*t1 + 2*h*t2 + a2*t3);
            for (int zcombi=0;zcombi<nzcombis;zcombi++){
                ind_4pcf = zcombi*nbinsphi1*nbinsphi2 + elphi1*nbinsphi2 + elphi2;
                mn3corr[zcombi] += nextF * fourpcf[ind_4pcf];
            } 
        }
    } 
}

// If thread==0 --> For final two threads allocate double/triple counting corrs
// thetacombis_batches: array of length nbinsr^3 with the indices of all possible (r1,r2,r3) combinations
//                      most likely it is simply range(nbinsr^3), but we leave some freedom here for 
//                      potential cost-based implementations
// nthetacombis_batches: array of length nthetbatches with the number of theta-combis in each batch
// cumthetacombis_batches : array of length (nthetbatches+1) with is cumsum of nthetacombis_batches
// nthetbatches: the number of theta batches
void alloc_notomoMapNap3_corrections(
    double *theta_edges, double *theta_centers, int nbinsr, double *phibins, double *dbinsphi, int nbinsphi, int nmax,
    int nthreads, double *apradii, int napradii, 
    double *xing, double complex *Gtilde_third, 
    int include_second, int include_third, double complex *MN3correlators){
    
    double complex *allMN3correlators = orpheus_calloc(nthreads*1*napradii, sizeof(double complex));
    #pragma omp parallel for num_threads(nthreads)
    for(int thisrcombi=0;thisrcombi<nbinsr*nbinsr*nbinsr;thisrcombi++){
        int thisthread = omp_get_thread_num();
        //printf("Starting thetabatch %d/%d on thread %d with %d thetacombis\n",
        //       elthetbatch,nthetbatches,thisthread,nthetacombis_batches[elthetbatch]);
        //int nbinsz = 1;
        int elb1, elb2, elb3;
        int n_cfs = 1;
        int batchgamma_thetshift = nbinsphi*nbinsphi;
        double complex *nextMN3correlators = orpheus_calloc(n_cfs, sizeof(double complex));
        double complex *thisnpcf = orpheus_calloc(n_cfs*batchgamma_thetshift, sizeof(double complex));
        //printf("Done allocations for thetabatch %d/%d on thread %d with %d thetacombis\n",
        //       elthetbatch,nthetbatches,thisthread,nthetacombis_batches[elthetbatch]);
        
        elb1 = thisrcombi/(nbinsr*nbinsr);
        elb2 = (thisrcombi-elb1*nbinsr*nbinsr)/nbinsr;
        elb3 = thisrcombi-elb1*nbinsr*nbinsr-elb2*nbinsr;
        gtilde4pcf_corrections(
            elb1, elb2, elb3, nbinsr, phibins, nbinsphi, nmax,
            include_second, include_third, xing, Gtilde_third, 
            thisnpcf);
        
        // Update the aperture MapNap^3 integral
        double R_ap, theta1, theta2, theta3, dtheta1, dtheta2, dtheta3;
        int mapnap3ind;
        int mapnap3threadshift = thisthread*n_cfs*napradii;
        theta1=theta_centers[elb1];
        theta2=theta_centers[elb2];
        theta3=theta_centers[elb3];
        dtheta1 = theta_edges[elb1+1]-theta_edges[elb1];
        dtheta2 = theta_edges[elb2+1]-theta_edges[elb2];
        dtheta3 = theta_edges[elb3+1]-theta_edges[elb3];
        for (int elapr=0; elapr<napradii; elapr++){
            R_ap = apradii[elapr];
            fourpcf2MN3correlatormulti(1, R_ap, R_ap, R_ap, R_ap,
                                    theta1, theta2, theta3, dtheta1, dtheta2, dtheta3,
                                    phibins, phibins, dbinsphi, dbinsphi, nbinsphi, nbinsphi,
                                    thisnpcf, nextMN3correlators);
            for (int elcomp=0;elcomp<n_cfs;elcomp++){
                mapnap3ind = elcomp*napradii+elapr;
                if (isfinite(cabs(nextMN3correlators[elcomp]))){
                    allMN3correlators[mapnap3threadshift+mapnap3ind] += nextMN3correlators[elcomp];
                }
                nextMN3correlators[elcomp] = 0;
            }
        }     
        // Reset 4pcf placeholders to zero
        for(int i=0;i<n_cfs*batchgamma_thetshift;i++){thisnpcf[i] = 0;}   
        
        //if (thisthread>-1){printf("Done allocating 4pcfs for thetabatch %d/%d\n",elthetbatch,nthetbatches);}
                
        free(nextMN3correlators);
        free(thisnpcf);
        nextMN3correlators = NULL;
        thisnpcf = NULL;       
    }
    // Accummulate the MapNap^3 integral
    int n_cfs = 1;
    for (int elthread=0;elthread<nthreads;elthread++){
        int mapnap3ind;
        int mapnap3threadshift = elthread*n_cfs*napradii;
        for (int elcomp=0;elcomp<n_cfs;elcomp++){
            for (int elmapr=0; elmapr<napradii; elmapr++){
                mapnap3ind = elcomp*napradii+elmapr;
                MN3correlators[mapnap3ind] += allMN3correlators[mapnap3threadshift+mapnap3ind];
            }
        }
    }
    free(allMN3correlators);
}


// If thread==0 --> For final two threads allocate double/triple counting corrs
// thetacombis_batches: array of length nbinsr^3 with the indices of all possible (r1,r2,r3) combinations
//                      most likely it is simply range(nbinsr^3), but we leave some freedom here for 
//                      potential cost-based implementations
// nthetacombis_batches: array of length nthetbatches with the number of theta-combis in each batch
// cumthetacombis_batches : array of length (nthetbatches+1) with is cumsum of nthetacombis_batches
// nthetbatches: the number of theta batches
void alloc_notomoMapNap3_analytic(
    double rmin, double rmax, int nbinsr, double *phibins, double *dbinsphi, int nbinsphi, int nsubr,
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, double *apradii, int napradii, 
    double *xing, double *xinn, double thetamin_xi, double thetamax_xi, int nthetabins_xi, int nsubsample_filter,
    double complex *MN3correlators){
               
    
    double complex *allMN3correlators = orpheus_calloc(nthreads*1*napradii, sizeof(double complex));
    #pragma omp parallel for num_threads(nthreads)
    for(int elthetbatch=0;elthetbatch<nthetbatches;elthetbatch++){
        int thisthread = omp_get_thread_num();
        //printf("Starting thetabatch %d/%d on thread %d with %d thetacombis\n",
        //       elthetbatch,nthetbatches,thisthread,nthetacombis_batches[elthetbatch]);
        //int nbinsz = 1;
        int n_cfs = 1;
        int batch_nthetas = nthetacombis_batches[elthetbatch];   
        int batchgamma_thetshift = nbinsphi*nbinsphi;
        double drbin = exp((log(rmax)-log(rmin))/(nbinsr));

        double *bin_centers = orpheus_calloc(nbinsr, sizeof(double));
        double *bin_edges = orpheus_calloc(nbinsr+1, sizeof(double));
        int *elb1s_batch = orpheus_calloc(batch_nthetas, sizeof(int));
        int *elb2s_batch = orpheus_calloc(batch_nthetas, sizeof(int));
        int *elb3s_batch = orpheus_calloc(batch_nthetas, sizeof(int));
        
        double complex *nextMN3correlators = orpheus_calloc(n_cfs, sizeof(double complex));
        double complex *thisnpcf = orpheus_calloc(n_cfs*batchgamma_thetshift, sizeof(double complex));
        //printf("Done allocations for thetabatch %d/%d on thread %d with %d thetacombis\n",
        //       elthetbatch,nthetbatches,thisthread,nthetacombis_batches[elthetbatch]);
        
        #pragma omp critical
        {
            for (int elb=0;elb<batch_nthetas;elb++){
                int thisrcombi = thetacombis_batches[cumthetacombis_batches[elthetbatch]+elb];
                elb1s_batch[elb] = thisrcombi/(nbinsr*nbinsr);
                elb2s_batch[elb] = (thisrcombi-elb1s_batch[elb]*nbinsr*nbinsr)/nbinsr;
                elb3s_batch[elb] = thisrcombi-elb1s_batch[elb]*nbinsr*nbinsr-elb2s_batch[elb]*nbinsr;
                if ((elb1s_batch[elb]>=nbinsr) || (elb2s_batch[elb]>=nbinsr) || (elb3s_batch[elb]>=nbinsr) 
                   || (elb1s_batch[elb]<0) || (elb2s_batch[elb]<0) || (elb3s_batch[elb]<0)){
                    printf("Error for index %d in thetabatch %d (rcombi %d) with el1=%d el2=%d el3=%d",
                         elb,elthetbatch,thisrcombi,elb1s_batch[elb],elb2s_batch[elb],elb3s_batch[elb]);}
            }
            bin_edges[0] = rmin;
            for (int elb=0;elb<nbinsr;elb++){
                bin_edges[elb+1] = bin_edges[elb]*drbin;
                bin_centers[elb] = .5*(bin_edges[elb]+bin_edges[elb+1]);
            }
        }
        double dtheta_xi = (thetamax_xi-thetamin_xi)/nthetabins_xi;
        
        // For each theta combination (theta1,theta2,theta3) in this batch 
        for (int elb=0;elb<batch_nthetas;elb++){
            
            // Get the analytic gaussian 4pcf from the 2pcf
            if (nsubr<2){
                gtilde4pcf_analytic(bin_centers[elb1s_batch[elb]],
                                    bin_centers[elb2s_batch[elb]],
                                    bin_centers[elb3s_batch[elb]], phibins, nbinsphi,
                                    xing, xinn, thetamin_xi, thetamax_xi, dtheta_xi, 
                                    thisnpcf);
            }
            else{
                gtilde4pcf_analytic_integrated(elb1s_batch[elb], 
                                               elb2s_batch[elb], 
                                               elb3s_batch[elb], 
                                               nsubr,
                                               bin_edges, nbinsr,
                                               phibins, nbinsphi,
                                               xing, xinn, thetamin_xi, thetamax_xi, dtheta_xi,
                                               thisnpcf);
            }
            
            // Update the aperture MapNap^3 integral
            double R_ap, theta1, theta2, theta3, dtheta1, dtheta2, dtheta3;
            int mapnap3ind;
            int mapnap3threadshift = thisthread*n_cfs*napradii;
            for (int elapr=0; elapr<napradii; elapr++){
                R_ap = apradii[elapr];
                theta1=bin_centers[elb1s_batch[elb]];
                theta2=bin_centers[elb2s_batch[elb]];
                theta3=bin_centers[elb3s_batch[elb]];
                dtheta1 = (bin_edges[elb1s_batch[elb]+1]-bin_edges[elb1s_batch[elb]]);
                dtheta2 = (bin_edges[elb2s_batch[elb]+1]-bin_edges[elb2s_batch[elb]]);
                dtheta3 = (bin_edges[elb3s_batch[elb]+1]-bin_edges[elb3s_batch[elb]]);
                fourpcf2MN3correlatormulti(1, R_ap, R_ap, R_ap, R_ap,
                                      theta1, theta2, theta3, dtheta1, dtheta2, dtheta3,
                                      phibins, phibins, dbinsphi, dbinsphi, nbinsphi, nbinsphi,
                                      thisnpcf, nextMN3correlators);
                for (int elcomp=0;elcomp<n_cfs;elcomp++){
                    mapnap3ind = elcomp*napradii+elapr;
                    if (isfinite(cabs(nextMN3correlators[elcomp]))){
                        allMN3correlators[mapnap3threadshift+mapnap3ind] += nextMN3correlators[elcomp];
                    }
                    nextMN3correlators[elcomp] = 0;
                }
            }
            
            // Reset 4pcf placeholders to zero
            for(int i=0;i<n_cfs*batchgamma_thetshift;i++){thisnpcf[i] = 0;}
        }        
        
        //if (thisthread>-1){printf("Done allocating 4pcfs for thetabatch %d/%d\n",elthetbatch,nthetbatches);}
        free(bin_centers);
        free(bin_edges);
        free(elb1s_batch);
        free(elb2s_batch);
        free(elb3s_batch);
        free(nextMN3correlators);
        free(thisnpcf);
        bin_centers = NULL;
        bin_edges = NULL;
        elb1s_batch = NULL;
        elb2s_batch = NULL;
        elb3s_batch = NULL;
        nextMN3correlators = NULL;
        thisnpcf = NULL;       
    }
    // Accummulate the MapNap^3 integral
    int n_cfs = 1;
    for (int elthread=0;elthread<nthreads;elthread++){
        int mapnap3ind;
        int mapnap3threadshift = elthread*n_cfs*napradii;
        for (int elcomp=0;elcomp<n_cfs;elcomp++){
            for (int elmapr=0; elmapr<napradii; elmapr++){
                mapnap3ind = elcomp*napradii+elmapr;
                MN3correlators[mapnap3ind] += allMN3correlators[mapnap3threadshift+mapnap3ind];
            }
        }
    }
    free(allMN3correlators);
} 

void gtilde4pcf_analytic_integrated(
    int indbin1, int indbin2, int indbin3, int nsubr, double *rbin_edges, int nbinsr, double *phis, int nphis, 
    double *xing, double *xinn, double thetamin_xi, double thetamax_xi, double dtheta_xi,
    double complex *gaussfourpcf){
    int n_cfs = 1;
    double dtheta1, dtheta2, dtheta3, subshift, subsubshift, thisw, wtot;
    double *theta1_subs = orpheus_calloc(nsubr, sizeof(double));
    double *theta2_subs = orpheus_calloc(nsubr, sizeof(double));
    double *theta3_subs = orpheus_calloc(nsubr, sizeof(double));
    
    // We define the subsampling in a way s.t. the subsampled bin values are different for the different thetas.
    // In particular, the values for theta2 are the true subsampled ones, while we shift the values of
    // theta1 and theta3 by +-1/3 of the subsampling bin width.
    dtheta1 = rbin_edges[indbin1+1] - rbin_edges[indbin1];
    dtheta2 = rbin_edges[indbin2+1] - rbin_edges[indbin2];
    dtheta3 = rbin_edges[indbin3+1] - rbin_edges[indbin3];
    subsubshift = 1./(3*nsubr);
    for (int elsub=0; elsub<nsubr; elsub++){
        subshift = (1.+2*elsub)/(2*nsubr);
        theta1_subs[elsub] = rbin_edges[indbin1] + dtheta1*(subshift + 0*subsubshift);
        theta2_subs[elsub] = rbin_edges[indbin2] + dtheta2*(subshift + 0);
        theta3_subs[elsub] = rbin_edges[indbin3] + dtheta3*(subshift - 0*subsubshift);    
        //printf("%.2f %.2f\n ",subshift,theta2_subs[elsub]);
    }
    
    // Run through all possible combinations of subsampled bin centers, evaluate the 
    // corresponding 4pcf and add it to the bin-averaged 4pcf
    wtot = 0;
    for (int elsub1=0; elsub1<nsubr; elsub1++){
        for (int elsub2=0; elsub2<nsubr; elsub2++){
            for (int elsub3=0; elsub3<nsubr; elsub3++){
                thisw = 1;
                double complex *nextfourpcf = orpheus_calloc(n_cfs*nphis*nphis, sizeof(double complex));
                gtilde4pcf_analytic(theta1_subs[elsub1],
                                    theta2_subs[elsub2],
                                    theta3_subs[elsub3], phis, nphis,
                                    xing, xinn, thetamin_xi, thetamax_xi, dtheta_xi, nextfourpcf);
                for (int ind=0; ind<n_cfs*nphis*nphis; ind++){gaussfourpcf[ind] += thisw*nextfourpcf[ind];}
                free(nextfourpcf);
                wtot += thisw;
            }
        }
    }

    // Normalize bin-averaged 4pcf
    for (int ind=0; ind<n_cfs*nphis*nphis; ind++){gaussfourpcf[ind] /= wtot;}
    
    free(theta1_subs);
    free(theta2_subs);
    free(theta3_subs);
}

void gtilde4pcf_analytic(
    double theta1, double theta2, double theta3, double *phis, int nphis,
    double *xing, double *xinn, double thetamin_xis, double thetamax_xis, double dtheta_xis,
    double complex *gaussfourpcf){
    double complex y1, y2, y3;
    double complex ang12, ang13, ang23;
    double absy12, absy13, absy23;
    double xing_1, xing_2, xing_3, xinn_12, xinn_13, xinn_23;
    int phishift;
    
    y1 = (double complex) theta1;
    xing_1 = linint(xing, theta1, thetamin_xis, thetamax_xis, dtheta_xis);
    xing_2 = linint(xing, theta2, thetamin_xis, thetamax_xis, dtheta_xis);
    xing_3 = linint(xing, theta3, thetamin_xis, thetamax_xis, dtheta_xis);
    for (int elphi12=0;elphi12<nphis;elphi12++){
        ang12 = cexp(I*phis[elphi12]);
        y2 = theta2*ang12;
        absy12 = cabs(y2-y1);
        xinn_12 = linint(xinn, absy12, thetamin_xis, thetamax_xis, dtheta_xis);
        for (int elphi13=0;elphi13<nphis;elphi13++){
            phishift = elphi12*nphis+elphi13;
            ang13 = cexp(I*phis[elphi13]);
            y3 = theta3*ang13;
            absy13 = cabs(y3-y1);
            xinn_13 = linint(xinn, absy13, thetamin_xis, thetamax_xis, dtheta_xis);
            absy23 = cabs(y3-y2);
            ang23 = ang13*conj(ang12);
            xinn_23 = linint(xinn, absy23, thetamin_xis, thetamax_xis, dtheta_xis);
            gaussfourpcf[phishift] = xinn_12*xing_3*ang13 + xinn_13*xing_2*ang12*conj(ang23) + xinn_23*xing_1*conj(ang13);
            //gaussfourpcf[phishift] = xing_1*conj(ang13) +  xing_2*ang12*conj(ang23) +  xing_3*ang13; //2pt correction
            if (!isfinite(cabs(gaussfourpcf[phishift]))){gaussfourpcf[phishift]=0;}
        }
    }
}

// <gNNN> = <gkkk> + (<gNN> + 2 perm) + (<gN> + 2 perm)
//        = <gkk> + ((<gkk>_a + <gk>_a1 + <gk>_a2) + 2 perm) + (<gk>_a + 2 perm)
//        := <gkkk> + <gNNN>_corr_third + <gNNN>_corr_second
// where in the x-projection for 3pt & 4pt functions we have (<gk>_a1 + <gk>_a2 + 2 perm) = 2*(<gk>_a + 2 perm)
void gtilde4pcf_corrections(
    int itheta1, int itheta2, int itheta3, int nthetas, double *phis, int nphis, int nmax, 
    int include_second, int include_third,  double *xi_ng, double complex *Gtilde_third, 
    double complex *fourpcf_corr){

    double complex ang12, ang13, ang23, corr_second, corr_third, Gtilde_12, Gtilde_13, Gtilde_23;
    int elphi23;

    for (int elphi12=0;elphi12<nphis;elphi12++){
        ang12 = cexp(I*phis[elphi12]);
        for (int elphi13=0;elphi13<nphis;elphi13++){
            elphi23 = (elphi13-elphi12+nphis)%nphis;
            ang13 = cexp(I*phis[elphi13]);
            ang23 = ang13*conj(ang12);
            Gtilde_12 = Gtilde_third[itheta1*nthetas*nphis+itheta2*nphis+elphi12];
            Gtilde_13 = Gtilde_third[itheta1*nthetas*nphis+itheta3*nphis+elphi13];
            Gtilde_23 = Gtilde_third[itheta2*nthetas*nphis+itheta3*nphis+elphi23];
            corr_second = xi_ng[itheta1]*conj(ang13) + xi_ng[itheta2]*ang12*conj(ang23) +  xi_ng[itheta3]*ang13; 
            corr_third = Gtilde_12*conj(ang23) + Gtilde_13 + Gtilde_23*ang12;
            fourpcf_corr[elphi12*nphis+elphi13] = include_second*corr_second + include_third*(corr_third-2*corr_second);
        }
    }
}

////////////////////////////////////
/// NNNN STATISTICS RELATED //
////////////////////////////////////

// Reconstructs all multipole components from the ones with theta1<=theta2<=theta3
// Nn_in ~ Nn[elthetbatch] ~ shape (2*nmax_alloc+1,2*nmax_alloc+1)
// Nn_out ~  shape (2*nmax+1,2*nmax+1)
// 
// Ordering for eltrafo: [123, 231, 312, 132, 213, 321]
// Different configs: 
//  * All three elbs equal --> No permutations needed, i.e. eltrafo in [0]
//  * Only two elbs equal  --> Only cyclic permutations needed, i.e. eltrafo in [0,1,2]
//  * All elbs unequal     --> All permutations needed, i.e. eltrafo in [0,1,2,3,4,5]   
void getMultipolesFromSymm_NNNN(double complex *Nn_in,
                                 int nmax, int eltrafo, int *nindices, int len_nindices,
                                 double complex *Nn_out){
    
    int nmax_alloc = 2*nmax+1;
    int nzero_in = nmax_alloc;
    int nzero_out = nmax;
    int n2shift_in = 2*nmax_alloc+1;
    int n2shift_out = 2*nmax+1;
    
    int n2, n2_N;
    int n3, n3_N;
    for (int nindex=0; nindex<len_nindices; nindex++){
        n2 = nindices[nindex]/(2*nmax_alloc+1) - nzero_in;
        n3 = nindices[nindex]%(2*nmax_alloc+1) - nzero_in;
        switch (eltrafo){
            case 0:
                n2_N=n2; n3_N=n3;
                break;
            case 1:
                n2_N=n3; n3_N=-n2-n3;
                break;
            case 2:
                n2_N=-n2-n3; n3_N=n2;
                break;
            case 3:
                n2_N=n3; n3_N=n2;
                break;
            case 4:
                n2_N=-n2-n3; n3_N=n3;
                break;
            case 5:
                n2_N=n2; n3_N=-n2-n3;
                break;
            default:
                n2_N=0; n3_N=0;
                break;
        }
        if ((abs(n2_N)<=nmax) && (abs(n3_N)<=nmax)){
            Nn_out[(nzero_out+n2_N)*n2shift_out+(nzero_out+n3_N)] = Nn_in[(nzero_in+n2)*n2shift_in+(nzero_in+n3)];
        }
    }
}

// Upsilon_n has shape (1,nphi12,nphi13)
void multipoles2npcf_nnnn(double complex *N_n, const BinningParams *bin, const FourthParams *fourth,
                          double *theta_centers, double complex *npcf, int nthreads){
    int n1max = bin->nmax, n2max = bin->nmax;
    int nbinsstheta = bin->nbinsr;
    double *phis12 = fourth->phibins1, *phis13 = fourth->phibins2;
    int nbinsphi12 = fourth->nbinsphi1, nbinsphi13 = fourth->nbinsphi2;
    
    #pragma omp parallel for num_threads(nthreads)
    for (int nthetacombi=0; nthetacombi<nbinsstheta*nbinsstheta*nbinsstheta; nthetacombi++){
        int n12combis = (2*n1max+1)*(2*n2max+1);
        int phi23combis = nbinsphi12*nbinsphi13;
        int ntheta2 = nbinsstheta*nbinsstheta;
        int ntheta3 = ntheta2*nbinsstheta;
        int eltheta1=nthetacombi/ntheta2;
        int eltheta2=(nthetacombi-eltheta1*ntheta2)/nbinsstheta;
        int eltheta3=nthetacombi%nbinsstheta;
        double complex *tmpnpcf = orpheus_calloc(phi23combis, sizeof(double complex));
        double complex *tmpNn = orpheus_calloc(n12combis, sizeof(double complex));
        for (int n12=0;n12<n12combis;n12++){
            tmpNn[n12] = N_n[n12*ntheta3+nthetacombi];
        }
        multipoles2npcf_nnnn_singletheta(
            tmpNn, n1max,n2max, theta_centers[eltheta1], theta_centers[eltheta2], theta_centers[eltheta3],
            phis12, phis13, nbinsphi12, nbinsphi13, tmpnpcf);
        for (int elphi12=0; elphi12<nbinsphi12; elphi12++){
            for (int elphi13=0; elphi13<nbinsphi13; elphi13++){
                npcf[nthetacombi*phi23combis+elphi12*nbinsphi13+elphi13] += tmpnpcf[elphi12*nbinsphi13+elphi13];
            }
        }
        free(tmpnpcf);
        free(tmpNn);
    }
}

// Upsilon_n has shape (1,nphi12,nphi13)
void multipoles2npcf_nnnn_singletheta(double complex *N_n, int n1max, int n2max,
                                      double theta1, double theta2, double theta3,
                                      double *phis12, double *phis13, int nbinsphi12, int nbinsphi13,
                                      double complex *npcf){
    int nmax = n1max;
    int nns = 2*nmax+1;
    double complex expphi12, expphi13;
    double complex *expphi12s = orpheus_calloc(nns, sizeof(double complex));
    double complex *expphi13s = orpheus_calloc(nns, sizeof(double complex));
    for (int elphi12=0; elphi12<nbinsphi12; elphi12++){
        for (int elphi13=0; elphi13<nbinsphi13; elphi13++){
            // Convert multipoles to npcf
            expphi12s[nmax] = 1;
            expphi13s[nmax] = 1;
            expphi12 = cexp(I*phis12[elphi12]);
            expphi13 = cexp(I*phis13[elphi13]);
            for (int nextn=1; nextn<=nmax; nextn++){ 
                expphi12s[nmax+nextn] = expphi12s[nmax+nextn-1]*expphi12;
                expphi12s[nmax-nextn] = conj(expphi12s[nmax+nextn]);
                expphi13s[nmax+nextn] = expphi13s[nmax+nextn-1]*expphi13;
                expphi13s[nmax-nextn] = conj(expphi13s[nmax+nextn]);
            }
            double complex nextang;
            int ind_npcf = elphi12*nbinsphi13 + elphi13;
            for (int nextn1=0; nextn1<nns; nextn1++){
                for (int nextn2=0; nextn2<nns; nextn2++){ 
                    int ind_ups = nextn1*nns + nextn2;
                    nextang = INV_2PI * expphi12s[nextn1] * expphi13s[nextn2];
                    npcf[ind_npcf] += N_n[ind_ups]*nextang;
                }
            }
        }
    } 
    free(expphi12s);
    free(expphi13s);
}

// N4 filters for a fixed aperture radius
// Note that with y==theta/R_ap the expressions do not depend on the aperture radius
// TODO: F_1, F_2 and F_3 are still placeholders set to unity, 
// fourpcf has shape (1,nbinsz,nbinsphi,nbinsphi)
void fourpcf2N4correlators(int nzcombis,
                           double y1, double y2, double y3, double dy1, double dy2, double dy3,
                           double *phis1, double *phis2, double *dphis1, double *dphis2, int nbinsphi1, int nbinsphi2,
                           double complex *fourpcf, double complex *n4corr){
    double complex q1, q2, q3, q4, q1c, q2c, q3c, q4c, q1_2, q2_2, q3_2, q4_2, q1c_2, q2c_2, q3c_2, q4c_2;
    double complex q1cq2c, q1cq3c, q1cq4c, q2cq3c, q2cq4c, q3cq4c;
    double q1abs, q2abs, q3abs, q4abs, q1abs_2, q2abs_2, q3abs_2, q4abs_2;
    double complex _y1, _y2, _y3, _y1c, _y3c, _y1_2, _y2_2, _y3_2, _y1_3, _y3_3 ;
    double _y1abs, _y2abs, _y3abs, _y1abs_2, _y2abs_2, _y3abs_2, _y1abs_3, _y3abs_3;
    double _measure, _exp;
    double complex F_1, F_2, F_3;
    double complex nextF;
    int fourpcf_compshift, ind_4pcf;
    for (int elphi1=0;elphi1<nbinsphi1;elphi1++){
        for (int elphi2=0;elphi2<nbinsphi2;elphi2++){
            _y1 = y1; _y1_2=_y1*_y1; _y1_3=_y1_2*_y1; _y1c=conj(_y1); 
            _y1abs=y1; _y1abs_2=_y1abs*_y1abs; _y1abs_3=_y1abs_2*_y1abs;
            _y2 = y2*cexp(I*phis1[elphi1]); _y2_2=_y2*_y2; _y3c=conj(_y2); 
            _y2abs=y2; _y2abs_2=_y2abs*_y2abs; 
            _y3 = y3*cexp(I*phis2[elphi2]); _y3_2=_y3*_y3; _y3_3=_y3_2*_y3; _y3c=conj(_y3); 
            _y3abs=y3; _y3abs_2=_y3abs*_y3abs; _y3abs_3=_y3abs_2*_y3abs;
            q1 =  0.25*(3*_y1-_y2-_y3); q1c=conj(q1); q1_2=q1*q1; q1c_2=q1c*q1c; q1abs=cabs(q1); q1abs_2=q1abs*q1abs;
            q2 =  0.25*(3*_y2-_y3-_y1); q2c=conj(q2); q2_2=q2*q2; q2c_2=q2c*q2c; q2abs=cabs(q2); q2abs_2=q2abs*q2abs;
            q3 =  0.25*(3*_y3-_y1-_y2); q3c=conj(q3); q3_2=q3*q3; q3c_2=q3c*q3c; q3abs=cabs(q3); q3abs_2=q3abs*q3abs;
            q4 = -0.25*(  _y1+_y2+_y3); q4c=conj(q4); q4_2=q4*q4; q4c_2=q4c*q4c; q4abs=cabs(q4); q4abs_2=q4abs*q4abs;
            q1cq2c=q1c*q2c; q1cq3c=q1c*q3c; q1cq4c=q1c*q4c; q2cq3c=q2c*q3c; q2cq4c=q2c*q4c; q3cq4c=q3c*q4c; 
            _measure = y1*y2*y3*dy1*dy2*dy3 * dphis1[elphi1]*dphis2[elphi2]*INV_2PI*INV_2PI;
            _exp = exp(-0.5*(q1abs_2+q2abs_2+q3abs_2+q4abs_2));
            F_1 = 1;
            F_2 = 1;
            F_3 = 1;
            nextF = 1./64 * _measure * (F_1+F_2+F_3)* _exp;
            fourpcf_compshift = elphi1*nbinsphi2 + elphi2;
            for (int zcombi=0;zcombi<nzcombis;zcombi++){
                ind_4pcf = fourpcf_compshift + zcombi*nbinsphi1*nbinsphi2;
                n4corr[zcombi] += nextF * fourpcf[ind_4pcf];
            }
        }
    }
}