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

// Convert shear 3pcf from x-projection to centroid projection
void _x2centroid_ggg(double complex *npcf, int nbinsz, 
                     double *theta_centers, int nbinstheta, double *phi_centers, int nbinsphi,
                     int nthreads){
    
    double *thetas_buffer = orpheus_calloc(nthreads*nbinstheta, sizeof(double));
    for (int elthread=0;elthread<nthreads; elthread++){
        for (int eltheta=0;eltheta<nbinstheta; eltheta++){
            thetas_buffer[elthread*nbinstheta+eltheta] = theta_centers[eltheta];
        }
    }
    #pragma omp parallel for num_threads(nthreads)
    for (int elphi=0; elphi<nbinsphi; elphi++){
        double bin1, bin2;
        double complex q1, q2, q3, q1_inv, q2_inv, q3_inv;
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
            q1 = (bin1 + bin2*phiexp_c)/(bin1 + bin2*phiexp);
            q2 = (2*bin1 - bin2*phiexp_c)/(2*bin1 - bin2*phiexp);
            q3 = (2*bin2*phiexp_c - bin1)/(2*bin2*phiexp - bin1);
            q1_inv = conj(q1)/cabs(q1);
            q2_inv = conj(q2)/cabs(q2);
            q3_inv = conj(q3)/cabs(q3);
            rot0 = q1*q2*q3*phiexp3;
            rot1 = q1_inv*q2*q3*phiexp;
            rot2 = q1*q2_inv*q3*phiexp3;
            rot3 = q1*q2*q3_inv*phiexp_c;
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

// Convert 3pcf from multipole-basis to real space for GGG, NGG and GNN. 
// * Assmes that nzbins2=nzbins3 such that the n-->-n symmetries work
// * modeweight, which is each correlator's own normalization convention
// * store_full_range: 0 reconstructs n<0 via the z2<->z3 transpose while
//   1 reads the full n=-nmax..nmax range directly.
// * modeweight carries dphi/(2pi), so the reconstructed norm is the multiplet count of
//   the bin. count_floor is a number of multiplets per resolution element, see below.
void multipoles2npcf_third_z1z23(double complex *Upsilon_n, double complex *N_n,
                                 int nmax, int ncomp_cf, int nbinsz1, int nbinsz23, int nbinstheta,
                                 double *phi_centers, int nbinsphi,
                                 int store_full_range, int *conjmap, double *modeweight,
                                 int is_edge_corrected, double count_floor,
                                 int nthreads,
                                 double complex *npcf, double complex *npcf_norm){

    int nzcombis = nbinsz1*nbinsz23*nbinsz23;
    int nmodes = store_full_range ? 2*nmax+1 : nmax+1;
    // For computation of the count threashold we need to take into account nmax and nbinsphi:
    // In each angular dimension, having nmax allows for 2*nmax+1 independent locations for the
    // peaks of the field but the pixelisation divides the [0,2*pi] interval into npix pixels. 
    // This means that a count, that in the ideal case is a delta function, will be recorded as 
    // a peak with weight one, but as this peak is distributed across nbinsphi/(2*nmax+1) pixels
    // its amplitude will be reduced. We set the count threashold as a scalar parameter (default 
    // 0.5 in python) times the average height of the smeared peak.
    double count_thr = count_floor*(2.*nmax + 1.)/nbinsphi;
    int nthetcombis = nbinstheta*nbinstheta;
    int ups_zshift = nthetcombis;
    int ups_nshift = nzcombis*ups_zshift;
    int ups_compshift = nmodes*ups_nshift;
    int gam_thetshift = nbinsphi;
    int gam_zshift = nthetcombis*gam_thetshift;
    int gam_compshift = nzcombis*gam_zshift;

    // Generate lookup table for exp s.t. we dont need it in the inner loop
    double complex *expphis = orpheus_calloc((2*nmax+1)*nbinsphi, sizeof(double complex));
    for (int nextn=0; nextn<=nmax; nextn++){
        for (int elphi=0; elphi<nbinsphi; elphi++){
            expphis[(nmax+nextn)*nbinsphi+elphi] = cexp(I*nextn*phi_centers[elphi]);
            expphis[(nmax-nextn)*nbinsphi+elphi] = conj(expphis[(nmax+nextn)*nbinsphi+elphi]);
        }
    }

    #pragma omp parallel for num_threads(nthreads)
    for (int itheta1=0; itheta1<nbinstheta; itheta1++){
        for (int itheta2=0; itheta2<nbinstheta; itheta2++){
            int thetcombi = itheta1*nbinstheta+itheta2;
            int thetcombi_t = itheta2*nbinstheta+itheta1;
            for (int zcombi=0; zcombi<nzcombis; zcombi++){
                int z1 = zcombi/(nbinsz23*nbinsz23);
                int z2 = (zcombi-z1*nbinsz23*nbinsz23)/nbinsz23;
                int z3 = zcombi%nbinsz23;
                int zcombi_t = z1*nbinsz23*nbinsz23+z3*nbinsz23+z2;

                // Check whether there are multiplets in this shell
                int base_ind = store_full_range ? nmax : 0;
                double complex N0 = modeweight[0]*N_n[base_ind*ups_nshift+zcombi*ups_zshift+thetcombi];
                int has_multiplets = shell_has_multiplets(N0);

                for (int elphi=0; elphi<nbinsphi; elphi++){
                    int ind_gam = zcombi*gam_zshift+thetcombi*gam_thetshift+elphi;
                    double complex norm_acc = 0;
                    double complex npcf_acc[ncomp_cf];
                    for (int elcf=0; elcf<ncomp_cf; elcf++){ npcf_acc[elcf] = 0; }
                    
                    // Multipoles only available for n>=0 
                    if (!store_full_range){
                        // Base case n=0.
                        int ind0 = zcombi*ups_zshift+thetcombi;
                        norm_acc += modeweight[0]*N_n[ind0];
                        for (int elcf=0; elcf<ncomp_cf; elcf++){
                            npcf_acc[elcf] += modeweight[0]*Upsilon_n[elcf*ups_compshift+ind0];
                        }
                        // Reconstruct n<0 via z2<->z3, theta transpose and conjmap.
                        for (int nextn=1; nextn<=nmax; nextn++){
                            int indp = nextn*ups_nshift+zcombi*ups_zshift+thetcombi;
                            int indm = nextn*ups_nshift+zcombi_t*ups_zshift+thetcombi_t;
                            double complex ep = expphis[(nmax+nextn)*nbinsphi+elphi];
                            double complex em = expphis[(nmax-nextn)*nbinsphi+elphi];
                            norm_acc += modeweight[nextn]*(N_n[indp]*ep+N_n[indm]*em);
                            for (int elcf=0; elcf<ncomp_cf; elcf++){
                                int elcf_conj = conjmap[elcf];
                                npcf_acc[elcf] += modeweight[nextn]*(
                                    Upsilon_n[elcf*ups_compshift+indp]*ep +
                                    Upsilon_n[elcf_conj*ups_compshift+indm]*em);
                            }
                        }
                    } 
                    // Multipoles available for all n
                    else {
                        for (int n=0; n<nmodes; n++){
                            int order = abs(n-nmax);
                            int ind = n*ups_nshift+zcombi*ups_zshift+thetcombi;
                            double complex e = expphis[n*nbinsphi+elphi];
                            norm_acc += modeweight[order]*N_n[ind]*e;
                            for (int elcf=0; elcf<ncomp_cf; elcf++){
                                npcf_acc[elcf] += modeweight[order]*Upsilon_n[elcf*ups_compshift+ind]*e;
                            }
                        }
                    }

                    npcf_norm[ind_gam] = norm_acc;
                    // Set the npcf to zero for two cases
                    // * shells without multiplets --> All elements set to zero
                    // * shells with few multiplets --> Here we expect ringing effects in bins
                    //   where no actual multiplets reside. To avoid near zero division we pick
                    //   a scale, count_thr, s.t. all elments with |N|<count_thr are zeroed. For 
                    //   more details on count_thr see the comment above
                    double complex divisor = is_edge_corrected ? N0 : norm_acc;
                    double dval = creal(divisor);
                    if (has_multiplets && fabs(dval) > count_thr){
                        for (int elcf=0; elcf<ncomp_cf; elcf++){
                            npcf[elcf*gam_compshift+ind_gam] = npcf_acc[elcf]/dval;
                        }
                    } else {
                        for (int elcf=0; elcf<ncomp_cf; elcf++){
                            npcf[elcf*gam_compshift+ind_gam] = 0;
                        }
                    }
                }
            }
        }
    }
    free(expphis);
}


///////////////////////////////////////
/// THIRD-ORDER APERTURE STATISTICS ///
///////////////////////////////////////

// For all 3pcf-to-aperture-statistics-conversions we parallelize over all
// radial bin combis for the 3pcf and then order the remaining loops as 
// phi --> apradii_combis --> zcombi. With this we can perform the heavy step
// i.e. the filter allocation before to the innermost loop.

// The Map3 filters divide by the |q_i|^2, each of which vanishes on a degenerate triangle;
// the same is true for the numerators which is why we get a 0/0 with an actual true finite
// limit. To recover this we nudge the phi-offset by small margin.
static inline double map3_safe_phi(double y1, double y2, double phi){
    double cphi = cos(phi);
    double thr = 1e-10/9.*(y1*y1 + y2*y2);
    double absq1s = 1./9.*(4*y1*y1 - 4*y1*y2*cphi + y2*y2);
    double absq2s = 1./9.*(y1*y1 - 4*y1*y2*cphi + 4*y2*y2);
    double absq3s = 1./9.*(y1*y1 + 2*y1*y2*cphi + y2*y2);
    if (absq1s < thr || absq2s < thr || absq3s < thr){ return phi + 1e-4; }
    return phi;
}

// Filter functions F_mu that convert between 3pcf and Map3 (single scale)
static inline void map3_filter_singleR_ggg(double y1, double y2, double dy1, double dy2,
    double phi, double dphi, double R_ap,
    double complex *T0, double complex *T3_123, double complex *T3_231, double complex *T3_312){

    phi = map3_safe_phi(y1, y2, phi);
    double cphi = cos(phi);
    double c2phi = cos(2*phi);
    double sphi = sin(phi);
    double complex ephi = cexp(I*phi);
    double complex ephic = conj(ephi);
    double complex e2phi = ephi*ephi;
    double complex e2phic = conj(e2phi);

    double R2 = R_ap*R_ap;
    double y1_2 = y1*y1, y2_2 = y2*y2;
    double y1_4 = y1_2*y1_2, y2_4 = y2_2*y2_2;
    double y13y2 = y1_2*y1*y2, y12y22 = y1_2*y2_2, y1y23 = y1*y2*y2_2;

    double absq1s = 1./9.*(4*y1_2 - 4*y1*y2*cphi + y2_2);
    double absq2s = 1./9.*(y1_2 - 4*y1*y2*cphi + 4*y2_2);
    double absq3s = 1./9.*(y1_2 + 2*y1*y2*cphi + y2_2);
    double absq123s = 2./3.*(y1_2+y2_2-y1*y2*cphi);
    double absq1q2q3_2 = absq1s*absq2s*absq3s;
    double measures = y1*dy1/R2 * y2*dy2/R2 * dphi/(2.*M_PI);
    double expfac = exp(-absq123s/(2.*R2));

    double nextT0 = absq1q2q3_2/(R2*R2*R2) * expfac;
    *T0 = 1./24. * measures * nextT0;

    double complex tmp1 = y1_4+y2_4+y1_2*y2_2*(2*c2phi-5.);
    double complex tmp2 = (y1_2+y2_2)*cphi + 9.*I*(y1_2-y2_2)*sphi;
    double complex q1q2q3starsq = -1./81.*(2*tmp1 - y1*y2*tmp2);
    double complex nextT3_123 = expfac * (1./24.*absq1q2q3_2/(R2*R2*R2) - 1./9.*q1q2q3starsq/(R2*R2) +
        1./27.*(q1q2q3starsq*q1q2q3starsq/(absq1q2q3_2*R2) + 2.*q1q2q3starsq/(absq3s*R2)));

    double complex inner231 = -4*y1_4 + 2*y2_4 + y13y2*8*cphi + y12y22*(8*e2phi-4-e2phic) + y1y23*(ephic-8*ephi);
    double complex q2q3q1starsq = -1./81.*inner231;
    double complex nextT3_231 = expfac * (1./24.*absq1q2q3_2/(R2*R2*R2) - 1./9.*q2q3q1starsq/(R2*R2) +
        1./27.*(q2q3q1starsq*q2q3q1starsq/(absq1q2q3_2*R2) + 2.*q2q3q1starsq/(absq1s*R2)));

    double complex inner312 = 2*y1_4 - 4*y2_4 - y13y2*(8*ephic-ephi) - y12y22*(4+e2phi-8*e2phic) + 8*y1y23*cphi;
    double complex q3q1q2starsq = -1./81.*inner312;
    double complex nextT3_312 = expfac * (1./24.*absq1q2q3_2/(R2*R2*R2) - 1./9.*q3q1q2starsq/(R2*R2) +
        1./27.*(q3q1q2starsq*q3q1q2starsq/(absq1q2q3_2*R2) + 2.*q3q1q2starsq/(absq2s*R2)));

    *T3_123 = measures * nextT3_123;
    *T3_231 = measures * nextT3_231;
    *T3_312 = measures * nextT3_312;
}

// Filter functions F_mu that convert between 3pcf and Map3 (multi scale)
static inline void map3_filter_multiR_ggg(double y1, double y2, double dy1, double dy2,
    double phi, double dphi, double R1, double R2, double R3,
    double complex *T0, double complex *T3_123, double complex *T3_231, double complex *T3_312){

    phi = map3_safe_phi(y1, y2, phi);
    double cphi = cos(phi);
    double c2phi = cos(2*phi);
    double sphi = sin(phi);
    double complex ephi = cexp(I*phi);
    double complex ephic = conj(ephi);
    double complex e2phi = ephi*ephi;
    double complex e2phic = conj(e2phi);

    double R1_2=R1*R1, R2_2=R2*R2, R3_2=R3*R3;
    double Theta2 = sqrt((R1_2*R2_2+R1_2*R3_2+R2_2*R3_2)/3.);
    double S = R1_2*R2_2*R3_2/(Theta2*Theta2*Theta2);

    double y1_2=y1*y1, y2_2=y2*y2;
    double y1_4=y1_2*y1_2, y2_4=y2_2*y2_2;
    double y13y2=y1_2*y1*y2, y12y22=y1_2*y2_2, y1y23=y1*y2*y2_2;

    double absq1s = 1./9.*(4*y1_2-4*y1*y2*cphi+y2_2);
    double absq2s = 1./9.*(y1_2-4*y1*y2*cphi+4*y2_2);
    double absq3s = 1./9.*(y1_2+2*y1*y2*cphi+y2_2);
    double absq1q2q3_2 = absq1s*absq2s*absq3s;

    double Z = ((-R1_2+2*R2_2+2*R3_2)*absq1s + (2*R1_2-R2_2+2*R3_2)*absq2s +
                (2*R1_2+2*R2_2-R3_2)*absq3s)/(6.*Theta2*Theta2);
    double complex frac231c = 1./3.*y2*(2*y1*ephi-y2)/absq1s;
    double complex frac312c = 1./3.*y1*(y1-2*y2*ephic)/absq2s;
    double complex frac123c = 1./3.*(y2_2-y1_2+2.*I*y1*y2*sphi)/absq3s;
    double complex f1 = (R2_2+R3_2)/(2.*Theta2) + frac231c*(R2_2-R3_2)/(6.*Theta2);
    double complex f2 = (R1_2+R3_2)/(2.*Theta2) + frac312c*(R3_2-R1_2)/(6.*Theta2);
    double complex f3 = (R1_2+R2_2)/(2.*Theta2) + frac123c*(R1_2-R2_2)/(6.*Theta2);
    double complex f1c = conj(f1), f2c = conj(f2), f3c = conj(f3);
    double complex g1c = conj(R2_2*R3_2/(Theta2*Theta2) + R1_2*(R3_2-R2_2)/(3.*Theta2*Theta2)*frac231c);
    double complex g2c = conj(R3_2*R1_2/(Theta2*Theta2) + R2_2*(R1_2-R3_2)/(3.*Theta2*Theta2)*frac312c);
    double complex g3c = conj(R1_2*R2_2/(Theta2*Theta2) + R3_2*(R2_2-R1_2)/(3.*Theta2*Theta2)*frac123c);
    double measures = y1*dy1/Theta2 * y2*dy2/Theta2 * dphi/(2.*M_PI);
    double complex expfac = cexp(-Z);

    double complex nextT0 = absq1q2q3_2/(Theta2*Theta2*Theta2) * f1c*f1c*f2c*f2c*f3c*f3c * expfac;
    *T0 = S/24. * measures * nextT0;

    double complex tmp1 = y1_4+y2_4+y1_2*y2_2*(2*c2phi-5.);
    double complex tmp2 = (y1_2+y2_2)*cphi + 9.*I*(y1_2-y2_2)*sphi;
    double complex q1q2q3starsq = -1./81.*(2*tmp1 - y1*y2*tmp2);
    double complex nextT3_123 = expfac * (1./24.*absq1q2q3_2/(Theta2*Theta2*Theta2) * f1c*f1c*f2c*f2c*f3*f3 -
        1./9.*q1q2q3starsq/(Theta2*Theta2) * f1c*f2c*f3*g3c +
        1./27.*(q1q2q3starsq*q1q2q3starsq/(absq1q2q3_2*Theta2) * g3c*g3c +
                2.*R1_2*R2_2/(Theta2*Theta2) * q1q2q3starsq/(absq3s*Theta2) * f1c*f2c));

    double complex inner231 = -4*y1_4+2*y2_4+y13y2*8*cphi+y12y22*(8*e2phi-4-e2phic)+y1y23*(ephic-8*ephi);
    double complex q2q3q1starsq = -1./81.*inner231;
    double complex nextT3_231 = expfac * (1./24.*absq1q2q3_2/(Theta2*Theta2*Theta2) * f2c*f2c*f3c*f3c*f1*f1 -
        1./9.*q2q3q1starsq/(Theta2*Theta2) * f2c*f3c*f1*g1c +
        1./27.*(q2q3q1starsq*q2q3q1starsq/(absq1q2q3_2*Theta2) * g1c*g1c +
                2.*R2_2*R3_2/(Theta2*Theta2) * q2q3q1starsq/(absq1s*Theta2) * f2c*f3c));

    double complex inner312 = 2*y1_4-4*y2_4-y13y2*(8*ephic-ephi)-y12y22*(4+e2phi-8*e2phic)+8*y1y23*cphi;
    double complex q3q1q2starsq = -1./81.*inner312;
    double complex nextT3_312 = expfac * (1./24.*absq1q2q3_2/(Theta2*Theta2*Theta2) * f3c*f3c*f1c*f1c*f2*f2 -
        1./9.*q3q1q2starsq/(Theta2*Theta2) * f3c*f1c*f2*g2c +
        1./27.*(q3q1q2starsq*q3q1q2starsq/(absq1q2q3_2*Theta2) * g2c*g2c +
                2.*R3_2*R1_2/(Theta2*Theta2) * q3q1q2starsq/(absq2s*Theta2) * f3c*f1c));

    *T3_123 = S * measures * nextT3_123;
    *T3_231 = S * measures * nextT3_231;
    *T3_312 = S * measures * nextT3_312;
}

// Convert GGG to Map3
void threepcf2M3correlators_ggg(double complex *npcf,
                                double *theta_edges, double *theta_centers, int nbinstheta,
                                double *phi_centers, int nbinsphi, int nzcombis,
                                double *radii1, double *radii2, double *radii3, int nrcombis,
                                int do_multiscale, int nthreads,
                                double complex *M3correlators){

    // If radial bins empty set them to arithmetic mean
    double *centers = orpheus_malloc(nbinstheta*sizeof(double));
    int haszero = 0;
    for (int i=0; i<nbinstheta; i++){ centers[i] = theta_centers[i]; if (centers[i]==0.){ haszero = 1; } }
    if (haszero){
        double ratiosum = 0.; int nratio = 0;
        for (int i=0; i<nbinstheta; i++){
            if (centers[i]!=0.){ ratiosum += centers[i]/theta_edges[i]; nratio++; }
        }
        double avratio = nratio>0 ? ratiosum/nratio : 0.;
        for (int i=0; i<nbinstheta; i++){
            if (centers[i]==0.){ centers[i] = avratio*theta_edges[i]; }
        }
    }

    int nthetcombis = nbinstheta*nbinstheta;
    int gam_thetshift = nbinsphi;
    int gam_zshift = nthetcombis*gam_thetshift;
    int gam_compshift = nzcombis*gam_zshift;
    int buf_compshift = nzcombis*nrcombis;
    int gam_threadshift = 4*buf_compshift;
    double complex *buf = orpheus_calloc((size_t)nthreads*gam_threadshift, sizeof(double complex));
    double dphi = phi_centers[1]-phi_centers[0];

    #pragma omp parallel for num_threads(nthreads)
    for (int thetcombi=0; thetcombi<nthetcombis; thetcombi++){
        int thisthread = omp_get_thread_num();
        int itheta1 = thetcombi/nbinstheta;
        int itheta2 = thetcombi%nbinstheta;
        double y1 = centers[itheta1];
        double dy1 = theta_edges[itheta1+1]-theta_edges[itheta1];
        double y2 = centers[itheta2];
        double dy2 = theta_edges[itheta2+1]-theta_edges[itheta2];
        for (int elphi=0; elphi<nbinsphi; elphi++){
            double phi = phi_centers[elphi];
            for (int elr=0; elr<nrcombis; elr++){
                double complex T0, T3_123, T3_231, T3_312;
                if (!do_multiscale){
                    map3_filter_singleR_ggg(y1, y2, dy1, dy2, phi, dphi, radii1[elr], &T0, &T3_123, &T3_231, &T3_312);
                } else {
                    map3_filter_multiR_ggg(y1, y2, dy1, dy2, phi, dphi, radii1[elr], radii2[elr], radii3[elr],
                                           &T0, &T3_123, &T3_231, &T3_312);
                }
                for (int zcombi=0; zcombi<nzcombis; zcombi++){
                    int ind_npcf = zcombi*gam_zshift+thetcombi*gam_thetshift+elphi;
                    int ind_buf = thisthread*gam_threadshift+zcombi*nrcombis+elr;
                    double complex term0 = T0*npcf[0*gam_compshift+ind_npcf];
                    double complex term1 = T3_123*npcf[1*gam_compshift+ind_npcf];
                    double complex term2 = T3_231*npcf[2*gam_compshift+ind_npcf];
                    double complex term3 = T3_312*npcf[3*gam_compshift+ind_npcf];
                    if (isfinite(cabs(term0))){ buf[0*buf_compshift+ind_buf] += term0; }
                    if (isfinite(cabs(term1))){ buf[1*buf_compshift+ind_buf] += term1; }
                    if (isfinite(cabs(term2))){ buf[2*buf_compshift+ind_buf] += term2; }
                    if (isfinite(cabs(term3))){ buf[3*buf_compshift+ind_buf] += term3; }
                }
            }
        }
    }
    free(centers);

    for (int elthread=0; elthread<nthreads; elthread++){
        for (int elc=0; elc<gam_threadshift; elc++){
            M3correlators[elc] += buf[elthread*gam_threadshift+elc];
        }
    }
    free(buf);
}

// Filter functions F_mu that convert between GNN 3pcf and MapNap2 (multi scale)
static inline double complex nnm_filter_gnn(double y1, double y2, double dy1, double dy2,
    double phi, double dphi, double R1, double R2, double R3){

    double cphi = cos(phi);
    double complex ephi = cexp(I*phi);
    double complex ephic = conj(ephi);

    double R1_2=R1*R1, R2_2=R2*R2, R3_2=R3*R3;
    double Theta4 = 1./3.*(R1_2*R2_2 + R1_2*R3_2 + R2_2*R3_2);
    double a2 = 2./3.*R1_2*R2_2*R3_2/Theta4;

    double b0 = y1*y1/(2*R1_2)+y2*y2/(2*R2_2) - a2/4.*(
        y1*y1/(R1_2*R1_2) + 2*y1*y2*cphi/(R1_2*R2_2) + y2*y2/(R2_2*R2_2));
    double complex g1 = y1 - a2/2.*(y1/R1_2 + y2*ephic/R2_2);
    double complex g2 = y2 - a2/2.*(y2/R2_2 + y1*ephi/R1_2);
    double complex g1c = conj(g1);
    double complex g2c = conj(g2);
    double complex F1 = 2*R1_2 - g1*g1c;
    double complex F2 = 2*R2_2 - g2*g2c;
    double pref = exp(-b0)/(72.*M_PI*Theta4*Theta4);
    double complex sum1 = (g1-y1)*(g2-y2) * (1./a2*F1*F2 - (F1+F2) + 2*a2 + g1c*g2*ephic + g1*g2c*ephi);
    double complex sum2 = ((g2-y2) + (g1-y1)*ephi) * (g1*(F2-2*a2) + g2*(F1-2*a2)*ephic);
    double complex sum3 = 2*g1*g2*a2;
    double measures = y1*dy1 * y2*dy2 * dphi;
    return measures * pref * (sum1-sum2+sum3);
}

// Convert GNN to MapNap2
void threepcf2NNMcorrelators_gnn(double complex *npcf,
                                 double *theta_edges, double *theta_centers, int nbinstheta,
                                 double *phi_centers, int nbinsphi, int nzcombis,
                                 double *radii1, double *radii2, double *radii3, int nrcombis,
                                 int nthreads,
                                 double complex *NNMcorrelators){

    int nthetcombis = nbinstheta*nbinstheta;
    int gam_thetshift = nbinsphi;
    int gam_zshift = nthetcombis*gam_thetshift;
    int gam_compshift = nzcombis*nrcombis;
    double complex *buf = orpheus_calloc((size_t)nthreads*gam_compshift, sizeof(double complex));
    double dphi = phi_centers[1]-phi_centers[0];

    #pragma omp parallel for num_threads(nthreads)
    for (int thetcombi=0; thetcombi<nthetcombis; thetcombi++){
        int thisthread = omp_get_thread_num();
        int itheta1 = thetcombi/nbinstheta;
        int itheta2 = thetcombi%nbinstheta;
        double y1 = theta_centers[itheta1];
        double dy1 = theta_edges[itheta1+1]-theta_edges[itheta1];
        double y2 = theta_centers[itheta2];
        double dy2 = theta_edges[itheta2+1]-theta_edges[itheta2];
        for (int elphi=0; elphi<nbinsphi; elphi++){
            double phi = phi_centers[elphi];
            for (int elr=0; elr<nrcombis; elr++){
                double complex A = nnm_filter_gnn(y1, y2, dy1, dy2, phi, dphi, radii1[elr], radii2[elr], radii3[elr]);
                for (int zcombi=0; zcombi<nzcombis; zcombi++){
                    int ind_npcf = zcombi*gam_zshift+thetcombi*gam_thetshift+elphi;
                    double complex term = A*npcf[ind_npcf];
                    if (isfinite(cabs(term))){
                        buf[thisthread*gam_compshift+zcombi*nrcombis+elr] += term;
                    }
                }
            }
        }
    }

    for (int elthread=0; elthread<nthreads; elthread++){
        for (int elc=0; elc<gam_compshift; elc++){
            NNMcorrelators[elc] += buf[elthread*gam_compshift+elc];
        }
    }
    free(buf);
}

// Filter functions F_mu that convert between NGG 3pcf and Map2Nap (multi scale)
static inline void nmm_filter_ngg(double y1, double y2, double dy1, double dy2,
    double phi, double dphi, double R1, double R2, double R3,
    double complex *A_MMN, double complex *A_MMstarN){

    double cphi = cos(phi);
    double complex ephi = cexp(I*phi);
    double complex ephic = conj(ephi);

    double R1_2=R1*R1, R2_2=R2*R2, R3_2=R3*R3;
    double Theta4 = 1./3.*(R1_2*R2_2 + R1_2*R3_2 + R2_2*R3_2);
    double a2 = 2./3.*R1_2*R2_2*R3_2/Theta4;

    double csq = a2*a2/4.*(y1*y1/(R1_2*R1_2) + y2*y2/(R2_2*R2_2) + 2*y1*y2*cphi/(R1_2*R2_2));
    double b0 = y1*y1/(2*R1_2)+y2*y2/(2*R2_2) - csq/a2;

    double complex g1 = y1 - a2/2.*(y1/R1_2 + y2*ephic/R2_2);
    double complex g2 = y2 - a2/2.*(y2/R2_2 + y1*ephi/R1_2);
    double complex g2c = conj(g2);
    double pref = exp(-b0)/(72.*M_PI*Theta4*Theta4);
    double complex h1 = 2.*(g2c*y1+g1*y2-2.*g1*g2c)*(g1*g2c+2.*a2*ephic);
    double complex h2 = 2.*a2*(2.*R3_2-csq-3.*a2)*ephic*ephic;
    double complex h3 = 4.*g1*g2c*(2.*R3_2-csq-2.*a2)*ephic;
    double complex h4 = (g1*g2c)*(g1*g2c)/a2 * (2.*R3_2-csq-a2);
    double complex sum_MMN = pref*g1*g2 * ((R3_2/R1_2+R3_2/R2_2-csq/a2)*g1*g2 + 2.*(g2*y1+g1*y2-2.*g1*g2));
    double complex sum_MMstarN = pref * (h1 + h2 + h3 + h4);
    double measures = y1*dy1 * y2*dy2 * dphi;

    *A_MMN = measures * sum_MMN;
    *A_MMstarN = measures * sum_MMstarN;
}

// Convert NGG to Map2Nap
void threepcf2NMMcorrelators_ngg(double complex *npcf,
                                 double *theta_edges, double *theta_centers, int nbinstheta,
                                 double *phi_centers, int nbinsphi, int nzcombis,
                                 double *radii1, double *radii2, double *radii3, int nrcombis,
                                 int nthreads,
                                 double complex *NMMcorrelators){

    int nthetcombis = nbinstheta*nbinstheta;
    int gam_thetshift = nbinsphi;
    int gam_zshift = nthetcombis*gam_thetshift;
    int gam_compshift = nzcombis*gam_zshift;
    int buf_compshift = nzcombis*nrcombis;
    int gam_threadshift = 2*buf_compshift;
    double complex *buf = orpheus_calloc((size_t)nthreads*gam_threadshift, sizeof(double complex));
    double dphi = phi_centers[1]-phi_centers[0];

    #pragma omp parallel for num_threads(nthreads)
    for (int thetcombi=0; thetcombi<nthetcombis; thetcombi++){
        int thisthread = omp_get_thread_num();
        int itheta1 = thetcombi/nbinstheta;
        int itheta2 = thetcombi%nbinstheta;
        double y1 = theta_centers[itheta1];
        double dy1 = theta_edges[itheta1+1]-theta_edges[itheta1];
        double y2 = theta_centers[itheta2];
        double dy2 = theta_edges[itheta2+1]-theta_edges[itheta2];
        for (int elphi=0; elphi<nbinsphi; elphi++){
            double phi = phi_centers[elphi];
            for (int elr=0; elr<nrcombis; elr++){
                double complex A_MMN, A_MMstarN;
                nmm_filter_ngg(y1, y2, dy1, dy2, phi, dphi, radii1[elr], radii2[elr], radii3[elr], &A_MMN, &A_MMstarN);
                for (int zcombi=0; zcombi<nzcombis; zcombi++){
                    int ind_npcf = zcombi*gam_zshift+thetcombi*gam_thetshift+elphi;
                    int ind_buf = thisthread*gam_threadshift+zcombi*nrcombis+elr;
                    double complex term0 = A_MMN*npcf[0*gam_compshift+ind_npcf];
                    double complex term1 = A_MMstarN*npcf[1*gam_compshift+ind_npcf];
                    if (isfinite(cabs(term0))){ buf[0*buf_compshift+ind_buf] += term0; }
                    if (isfinite(cabs(term1))){ buf[1*buf_compshift+ind_buf] += term1; }
                }
            }
        }
    }

    for (int elthread=0; elthread<nthreads; elthread++){
        for (int elc=0; elc<gam_threadshift; elc++){
            NMMcorrelators[elc] += buf[elthread*gam_threadshift+elc];
        }
    }
    free(buf);
}
