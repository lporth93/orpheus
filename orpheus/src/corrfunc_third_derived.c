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
#include "corrfunc_third_derived.h"

#define M_PI      3.14159265358979323846

// TODO: THIS IMPLEMENTATION IS OK BUT CONCEPTUALLY HORRIBLE AS IT BASICALLY COPIES THE 
//       SAME LOOP TWICE. ALSO THE PARALLELIZATION IS BAD. SHOULD BE FIXED...
void multipoles2npcf_ggg(double complex *Upsilon_n, double complex *N_n, int nmax, int nbinsz,
                         double *theta_centers, int nbinstheta, double *phi_centers, int nbinsphi,
                         int projection, int nthreads,
                         double complex *npcf, double complex *npcf_norm){

    // We split the calculation in two separate loops, one for theta1<=theta2
    // and one for theta2<theta1. We do this as due to the symmetry properties
    // we need to call both, (theta1,theta2) and (theta2,theta1) combinations of
    // the Upsilon_n in the innermost loop which might introduce race conditions.
    // Notes:
    // 1) In principle, running both loops may not be neccessary due to the symmetry property
    //      Gamma_mu(thet1,thet2,z1,z2,z3,phi) = Gamma_mu'(thet2,thet1,z1,z3,z2,-phi),
    //    but as we do not enforce a symmetric phi-binning this may not hold.
    // 2) In contrast to P24 we normalise the individual triplet correlators by the number of
    //    angular bins (nphi) instead of 2*pi. For the Quotient leading to the natural
    //    components this does not matter, but it gives the correctly normalised counts
    
    // Run first batch (theta1<=theta2)
    #pragma omp parallel for num_threads(nthreads)
    for (int itheta1=0; itheta1<nbinstheta; itheta1++){
        int z1, z2, z3, zcombi_t;
        int thetcombi, thetcombi_t;
        int indNn_p, indNn_m, ind_Norm;
        int indexpphi_p, indexpphi_m;
        int n_cfs=4;
        int nthetcombis=nbinstheta*nbinstheta;
        int nzcombis=nbinsz*nbinsz*nbinsz;
        int nns=nmax+1;
        int ups_zshift=nthetcombis;
        int ups_nshift=nzcombis*ups_zshift;
        int ups_compshift=nns*ups_nshift;
        int gam_thetshift=nbinsphi;
        int gam_zshift=nthetcombis*gam_thetshift;
        int gam_compshift=nzcombis*gam_zshift;
        double norm_triplets = 1./nbinsphi;
        // Allocate lookup table for all exponential factors appearing
        double complex *expphis = calloc((2*nmax+1)*nbinsphi, sizeof(double complex));
        for (int nextn=0;nextn<=nmax;nextn++){
            for (int elphi=0;elphi<nbinsphi;elphi++){
                expphis[(nmax+nextn)*nbinsphi+elphi] = cexp(I*nextn*phi_centers[elphi]);
                expphis[(nmax-nextn)*nbinsphi+elphi] = conj(expphis[(nmax+nextn)*nbinsphi+elphi]);
            }
        }
        // Continue with transformation 
        for (int itheta2=itheta1; itheta2<nbinstheta; itheta2++){
            thetcombi   = itheta1*nbinstheta+itheta2;
            thetcombi_t = itheta2*nbinstheta+itheta1;
            for (int zcombi=0;zcombi<nbinsz*nbinsz*nbinsz; zcombi++){
                z1 = zcombi/(nbinsz*nbinsz);
                z2 = (zcombi-z1*nbinsz*nbinsz)/nbinsz;
                z3 = zcombi%nbinsz;
                zcombi_t = z1*nbinsz*nbinsz+z3*nbinsz+z2;
                for (int elphi=0; elphi<nbinsphi; elphi++){
                    // Convert multipoles to npcf
                    ind_Norm = zcombi*gam_zshift+thetcombi*gam_thetshift+elphi;
                    // Base case (n=0)
                    indNn_p = zcombi*ups_zshift+thetcombi;
                    npcf_norm[ind_Norm] += N_n[indNn_p];                    
                    npcf[0*gam_compshift+ind_Norm] += Upsilon_n[0*ups_compshift+indNn_p];
                    npcf[1*gam_compshift+ind_Norm] += Upsilon_n[1*ups_compshift+indNn_p];
                    npcf[2*gam_compshift+ind_Norm] += Upsilon_n[2*ups_compshift+indNn_p];
                    npcf[3*gam_compshift+ind_Norm] += Upsilon_n[3*ups_compshift+indNn_p];
                    //if (itheta1==10 && itheta2==20 && zcombi==33 && elphi==44){
                    //    printf("Added %.4f+i*%.4f to norm\n",creal(N_n[indNn_p]),cimag(N_n[indNn_p]));
                    //    printf("Current value at index %d/%d is %.4f+i*%.4f to norm\n",ind_Norm,indNn_p,creal(npcf_norm[ind_Norm]),cimag(npcf_norm[ind_Norm]));
                    //}
                    // Remaining cases (n=\pm 1,... \pm nmax)
                    for (int nextn=1; nextn<=nmax; nextn++){ 
                        indexpphi_p = (nmax+nextn)*nbinsphi+elphi;
                        indexpphi_m = (nmax-nextn)*nbinsphi+elphi;
                        indNn_p = nextn*ups_nshift+zcombi*ups_zshift+thetcombi;
                        indNn_m = nextn*ups_nshift+zcombi_t*ups_zshift+thetcombi_t;
                        npcf_norm[ind_Norm] += N_n[indNn_p]*expphis[indexpphi_p];
                        npcf_norm[ind_Norm] += N_n[indNn_m]*expphis[indexpphi_m];
                        npcf[0*gam_compshift+ind_Norm] += Upsilon_n[0*ups_compshift+indNn_p]*expphis[indexpphi_p];
                        npcf[0*gam_compshift+ind_Norm] += Upsilon_n[0*ups_compshift+indNn_m]*expphis[indexpphi_m];
                        npcf[1*gam_compshift+ind_Norm] += Upsilon_n[1*ups_compshift+indNn_p]*expphis[indexpphi_p];
                        npcf[1*gam_compshift+ind_Norm] += Upsilon_n[1*ups_compshift+indNn_m]*expphis[indexpphi_m];
                        npcf[2*gam_compshift+ind_Norm] += Upsilon_n[2*ups_compshift+indNn_p]*expphis[indexpphi_p];
                        npcf[2*gam_compshift+ind_Norm] += Upsilon_n[3*ups_compshift+indNn_m]*expphis[indexpphi_m];
                        npcf[3*gam_compshift+ind_Norm] += Upsilon_n[3*ups_compshift+indNn_p]*expphis[indexpphi_p];
                        npcf[3*gam_compshift+ind_Norm] += Upsilon_n[2*ups_compshift+indNn_m]*expphis[indexpphi_m];
                    }
                    // Normalize: Gamma=Upsilon/N --> Make sure that we have counts, i.e. N >~ 1.
                    npcf_norm[ind_Norm] *= norm_triplets;
                    for (int elcf=0; elcf<n_cfs; elcf++){ 
                        npcf[elcf*gam_compshift+ind_Norm] *= norm_triplets;
                        if (cabs(npcf_norm[ind_Norm]) > 0.1){npcf[elcf*gam_compshift+ind_Norm] /= cabs(npcf_norm[ind_Norm]);}
                        else{npcf[elcf*gam_compshift+ind_Norm] = 0;}
                    }
                } 
            }
        }
        free(expphis);
    }
    
    // Run second batch (theta1>theta2)
    #pragma omp parallel for num_threads(nthreads)
    for (int itheta2=0; itheta2<nbinstheta; itheta2++){
        int z1, z2, z3, zcombi_t;
        int thetcombi, thetcombi_t;
        int indNn_p, indNn_m, ind_Norm;
        int indexpphi_p, indexpphi_m;
        int n_cfs=4;
        int nthetcombis=nbinstheta*nbinstheta;
        int nzcombis=nbinsz*nbinsz*nbinsz;
        int nns=nmax+1;
        int ups_zshift=nthetcombis;
        int ups_nshift=nzcombis*ups_zshift;
        int ups_compshift=nns*ups_nshift;
        int gam_thetshift=nbinsphi;
        int gam_zshift=nthetcombis*gam_thetshift;
        int gam_compshift=nzcombis*gam_zshift;        
        double norm_triplets = 1./nbinsphi;
        // Allocate lookup table for all exponential factors appearing
        double complex *expphis = calloc((2*nmax+1)*nbinsphi, sizeof(double complex));
        for (int nextn=0;nextn<=nmax;nextn++){
            for (int elphi=0;elphi<nbinsphi;elphi++){
                expphis[(nmax+nextn)*nbinsphi+elphi] = cexp(I*nextn*phi_centers[elphi]);
                expphis[(nmax-nextn)*nbinsphi+elphi] = conj(expphis[(nmax+nextn)*nbinsphi+elphi]);
            }
        }
        // Continue with transformation 
        for (int itheta1=itheta2+1; itheta1<nbinstheta; itheta1++){
            thetcombi   = itheta1*nbinstheta+itheta2;
            thetcombi_t = itheta2*nbinstheta+itheta1;
            for (int zcombi=0;zcombi<nbinsz*nbinsz*nbinsz; zcombi++){
                z1 = zcombi/(nbinsz*nbinsz);
                z2 = (zcombi-z1*nbinsz*nbinsz)/nbinsz;
                z3 = zcombi%nbinsz;
                zcombi_t = z1*nbinsz*nbinsz+z3*nbinsz+z2;
                for (int elphi=0; elphi<nbinsphi; elphi++){
                    // Convert multipoles to npcf
                    ind_Norm = zcombi*gam_zshift+thetcombi*gam_thetshift+elphi;
                    // Base case (n=0)
                    indNn_p = zcombi*ups_zshift+thetcombi;
                    npcf_norm[ind_Norm] += N_n[indNn_p];
                    npcf[0*gam_compshift+ind_Norm] += Upsilon_n[0*ups_compshift+indNn_p];
                    npcf[1*gam_compshift+ind_Norm] += Upsilon_n[1*ups_compshift+indNn_p];
                    npcf[2*gam_compshift+ind_Norm] += Upsilon_n[2*ups_compshift+indNn_p];
                    npcf[3*gam_compshift+ind_Norm] += Upsilon_n[3*ups_compshift+indNn_p];
                    // Remaining cases (n=\pm 1,... \pm nmax)
                    for (int nextn=1; nextn<=nmax; nextn++){ 
                        indexpphi_p = (nmax+nextn)*nbinsphi+elphi;
                        indexpphi_m = (nmax-nextn)*nbinsphi+elphi;
                        indNn_p = nextn*ups_nshift+zcombi*ups_zshift+thetcombi;
                        indNn_m = nextn*ups_nshift+zcombi_t*ups_zshift+thetcombi_t;
                        npcf_norm[ind_Norm] += N_n[indNn_p]*expphis[indexpphi_p];
                        npcf_norm[ind_Norm] += N_n[indNn_m]*expphis[indexpphi_m];
                        npcf[0*gam_compshift+ind_Norm] += Upsilon_n[0*ups_compshift+indNn_p]*expphis[indexpphi_p];
                        npcf[0*gam_compshift+ind_Norm] += Upsilon_n[0*ups_compshift+indNn_m]*expphis[indexpphi_m];
                        npcf[1*gam_compshift+ind_Norm] += Upsilon_n[1*ups_compshift+indNn_p]*expphis[indexpphi_p];
                        npcf[1*gam_compshift+ind_Norm] += Upsilon_n[1*ups_compshift+indNn_m]*expphis[indexpphi_m];
                        npcf[2*gam_compshift+ind_Norm] += Upsilon_n[2*ups_compshift+indNn_p]*expphis[indexpphi_p];
                        npcf[2*gam_compshift+ind_Norm] += Upsilon_n[3*ups_compshift+indNn_m]*expphis[indexpphi_m];
                        npcf[3*gam_compshift+ind_Norm] += Upsilon_n[3*ups_compshift+indNn_p]*expphis[indexpphi_p];
                        npcf[3*gam_compshift+ind_Norm] += Upsilon_n[2*ups_compshift+indNn_m]*expphis[indexpphi_m];
                    }
                    // Normalize: Gamma=Upsilon/N --> Make sure that we have counts, i.e. N >~ 1.
                    npcf_norm[ind_Norm] *= norm_triplets;
                    for (int elcf=0; elcf<n_cfs; elcf++){ 
                        npcf[elcf*gam_compshift+ind_Norm] *= norm_triplets;
                        if (cabs(npcf_norm[ind_Norm]) > 0.1){npcf[elcf*gam_compshift+ind_Norm] /= cabs(npcf_norm[ind_Norm]);}
                        else{npcf[elcf*gam_compshift+ind_Norm] = 0;}
                    }
                } 
            }
        }
        free(expphis);
    }  
    
    // Optionally transform to different basis
    if (projection==1){
        _x2centroid_ggg(npcf, nbinsz, 
                        theta_centers, nbinstheta, phi_centers, nbinsphi,
                        nthreads);
    }
}

void _x2centroid_ggg(double complex *npcf, int nbinsz, 
                     double *theta_centers, int nbinstheta, double *phi_centers, int nbinsphi,
                     int nthreads){
    
    double *thetas_buffer = calloc(nthreads*nbinstheta, sizeof(double));
    for (int elthread=0;elthread<nthreads; elthread++){
        for (int eltheta=0;eltheta<nbinstheta; eltheta++){
            thetas_buffer[elthread*nbinstheta+eltheta] = theta_centers[eltheta];
        }
    }
    #pragma omp parallel for num_threads(nthreads)
    for (int elphi=0; elphi<nbinsphi; elphi++){
        double bin1, bin2;
        double complex prod1, prod2, prod3, prod1_inv, prod2_inv, prod3_inv;
        double complex rot0, rot1, rot2, rot3;
        int ind_gam, ithet1, ithet2;
        int nthetcombis=nbinstheta*nbinstheta;
        int nzcombis=nbinsz*nbinsz*nbinsz;
        int gam_thetshift=nbinsphi;
        int gam_zshift=nthetcombis*gam_thetshift;
        int gam_compshift=nzcombis*gam_zshift;
        double complex phiexp = cexp(I*phi_centers[elphi]);
        double complex phiexp_c = conj(phiexp);
        double complex phiexp3 = phiexp*phiexp*phiexp;
        int thisthread = omp_get_thread_num();
        for (int thetcombi=0; thetcombi<nbinstheta*nbinstheta; thetcombi++){
            ithet1 = thetcombi/nbinstheta;
            ithet2 = thetcombi%nbinstheta;
            bin1 = thetas_buffer[thisthread*nbinstheta+ithet1];
            bin2 = thetas_buffer[thisthread*nbinstheta+ithet2];
            prod1 = (bin1 + bin2*phiexp_c)/(bin1 + bin2*phiexp); //q1
            prod2 = (2*bin1 - bin2*phiexp_c)/(2*bin1 - bin2*phiexp); //q2
            prod3 = (2*bin2*phiexp_c - bin1)/(2*bin2*phiexp - bin1); //q3
            prod1_inv = conj(prod1)/cabs(prod1);
            prod2_inv = conj(prod2)/cabs(prod2);
            prod3_inv = conj(prod3)/cabs(prod3);
            rot0 = prod1*prod2*prod3*phiexp3;
            rot1 = prod1_inv*prod2*prod3*phiexp;
            rot2 = prod1*prod2_inv*prod3*phiexp3;
            rot3 = prod1*prod2*prod3_inv*phiexp_c;
            for (int zcombi=0;zcombi<nzcombis; zcombi++){
                ind_gam = zcombi*gam_zshift+thetcombi*gam_thetshift+elphi;
                npcf[0*gam_compshift+ind_gam] *= rot0;
                npcf[1*gam_compshift+ind_gam] *= rot1;
                npcf[2*gam_compshift+ind_gam] *= rot2;
                npcf[3*gam_compshift+ind_gam] *= rot3;
            } 
        }
    }
    free(thetas_buffer);
}

void threepcf2M3correlators_singlescale(double *npcf, 
                                        double *theta_edges, double *theta_centers, int nbinstheta, 
                                        double *phi_centers, int nbinsphi, int nbinsz,
                                        int nthreads,
                                        double *mapradii, int nmapradii, double complex *M3correlators){
    
    double *thetas_center_buffer = calloc(nthreads*nbinstheta, sizeof(double));
    double *thetas_edges_buffer = calloc(nthreads*(nbinstheta+1), sizeof(double));
    for (int elthread=0;elthread<nthreads; elthread++){
        for (int eltheta=0;eltheta<nbinstheta; eltheta++){
            thetas_center_buffer[elthread*nbinstheta+eltheta] = theta_centers[eltheta];
            thetas_edges_buffer[elthread*(nbinstheta+1)+eltheta] = theta_edges[eltheta];
        }
        thetas_edges_buffer[elthread*(nbinstheta+1)+nbinstheta] = theta_edges[nbinstheta];
    }
    double complex *M3correlators_buffer = calloc(nthreads*4*nbinsz*nbinsz*nbinsz*nmapradii, sizeof(double complex));
    double dphi = phi_centers[1]-phi_centers[0];
    int map3_threadshift=4*nbinsz*nbinsz*nbinsz*nmapradii;
    #pragma omp parallel for num_threads(nthreads)
    for (int elphi=0; elphi<nbinsphi; elphi++){
        int thisthread = omp_get_thread_num();
        int nthetcombis=nbinstheta*nbinstheta;
        int nzcombis=nbinsz*nbinsz*nbinsz;
        int gam_thetshift=nbinsphi;
        int gam_zshift=nthetcombis*gam_thetshift;
        int gam_compshift=nzcombis*gam_zshift;
        double cosphi = cos(phi_centers[elphi]);
        double cos2phi = cos(2*phi_centers[elphi]);
        double sinphi = sin(phi_centers[elphi]);
        double complex expphi = cexp(I*phi_centers[elphi]);
        double complex expphi_c = conj(expphi);
        double complex expphi2 = expphi*expphi;
        double complex expphi2_c = conj(expphi2);
        int thetcombi, ind_gam, baseind_M3;
        double y1, dy1, y2, dy2, y1_2, y1_4, y2_2, y2_4, y13y2, y12y22, y1y23;
        double absq1s, absq2s, absq3s, absq123s, absq1q2q3_2;
        double complex q1q2q3starsq, q2q3q1starsq, q3q1q2starsq;
        double R_ap, R_ap_2, R_ap_4, R_ap_6, measures;
        double complex T0, T3_123, T3_231, T3_312;
        for (int itheta1=0; itheta1<nbinstheta; itheta1++){
            y1 = thetas_center_buffer[thisthread*nbinstheta+itheta1];
            dy1 = thetas_edges_buffer[thisthread*(nbinstheta+1)+itheta1+1]-thetas_edges_buffer[thisthread*(nbinstheta+1)+itheta1];
            y1_2 = y1*y1; y1_4=y1_2*y1_2;
            y13y2=y1_2*y1*y2; y12y22=y1_2*y2_2; y1y23=y1*y2*y2_2;
            for (int itheta2=0; itheta2<nbinstheta; itheta2++){
                thetcombi = itheta1*nbinstheta+itheta2;
                y2 = thetas_center_buffer[thisthread*nbinstheta+itheta2];
                dy2 = thetas_edges_buffer[thisthread*(nbinstheta+1)+itheta2+1]-thetas_edges_buffer[thisthread*(nbinstheta+1)+itheta1];
                y2_2 = y2*y2; y2_4=y2_2*y2_2;
                absq1s = 1./9.*(4*y1_2 - 4*y1*y2*cosphi + 1*y2_2);
                absq2s = 1./9.*(1*y1_2 - 4*y1*y2*cosphi + 4*y2_2);
                absq3s = 1./9.*(1*y1_2 + 2*y1*y2*cosphi + 1*y2_2);
                absq123s = 2./3. * (y1_2+y2_2-y1*y2*cosphi);
                absq1q2q3_2 = absq1s*absq2s*absq3s;
                for (int elR=0; elR<nmapradii; elR++){
                    R_ap = mapradii[elR];
                    R_ap_2=R_ap*R_ap; R_ap_4=R_ap_2*R_ap_2; R_ap_6=R_ap_2*R_ap_4;
                    q1q2q3starsq = -1./81*( 2*(y1_4 + y2_4 + y1_2*y2_2 * (2*cos2phi-5.)) - y1*y2*((y1_2+y2_2)*cosphi + 9I*(y1_2-y2_2)*sinphi));
                    q2q3q1starsq = -1./81*(-4*y1_4 + 2*y2_4 + y13y2*8*cosphi + y12y22*(8*expphi2-4-expphi2_c) + y1y23*(expphi_c-8*expphi));
                    q3q1q2starsq = -1./81*( 2*y1_4 - 4*y2_4 - y13y2*(8*expphi_c-expphi) - y12y22*(4+expphi2-8*expphi2_c) + 8*y1y23*cosphi);
                    measures = y1*dy1/R_ap_2 * y2*dy2/R_ap_2 * dphi/(2*M_PI);
                    T0 = 1./24. * measures * absq1q2q3_2/R_ap_6 * cexp(-absq123s/(2*R_ap_2));
                    T3_123 = measures * exp(-absq123s/(2*R_ap_2)) * (
                        1./24*absq1q2q3_2/R_ap_6 - 1./9.*q1q2q3starsq/R_ap_4 +
                        1./27*(q1q2q3starsq*q1q2q3starsq/(absq1q2q3_2*R_ap_2) + 2*q1q2q3starsq/(absq3s*R_ap_2)));
                    T3_231 = measures * exp(-absq123s/(2*R_ap_2)) * (
                        1./24*absq1q2q3_2/R_ap_6 - 1./9.*q2q3q1starsq/R_ap_4 +
                        1./27*(q2q3q1starsq*q2q3q1starsq/(absq1q2q3_2*R_ap_2) + 2*q2q3q1starsq/(absq1s*R_ap_2)));
                    T3_312 = measures * exp(-absq123s/(2*R_ap_2)) * (
                        1./24*absq1q2q3_2/R_ap_6 - 1./9.*q3q1q2starsq/R_ap_4 +
                        1./27*(q3q1q2starsq*q3q1q2starsq/(absq1q2q3_2*R_ap_2) + 2*q3q1q2starsq/(absq2s*R_ap_2)));
                    for (int zcombi=0;zcombi<nzcombis; zcombi++){
                        ind_gam = zcombi*gam_zshift+thetcombi*gam_thetshift+elphi;
                        baseind_M3 = thisthread*map3_threadshift + zcombi*nmapradii + elR;
                        M3correlators_buffer[baseind_M3+0*nzcombis*nmapradii] += T0*npcf[0*gam_compshift+ind_gam];
                        M3correlators_buffer[baseind_M3+1*nzcombis*nmapradii] += T0*npcf[1*gam_compshift+ind_gam];
                        M3correlators_buffer[baseind_M3+2*nzcombis*nmapradii] += T0*npcf[2*gam_compshift+ind_gam];
                        M3correlators_buffer[baseind_M3+3*nzcombis*nmapradii] += T0*npcf[3*gam_compshift+ind_gam];
                    }                    
                }
            }
        }
    }
    free(thetas_center_buffer);
    free(thetas_edges_buffer);
    
    // Accumulate M3 from buffer
    for (int elthread=0;elthread<nthreads;elthread++){
        for (int elc=0;elc<map3_threadshift;elc++){
            M3correlators[elc] += M3correlators_buffer[elthread*map3_threadshift+elc];
        }
    } 
    free(M3correlators_buffer);
}