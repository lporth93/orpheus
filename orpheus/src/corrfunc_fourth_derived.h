#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <complex.h>

void getMultiplolesFromSymm(double complex *Upsn_in, double complex *Nn_in,
                            int nmax, int eltrafo, int *nindices, int len_nindices,
                            double complex *Upsn_out, double complex *Nn_out);
    
void multipoles2npcf_gggg(double complex *upsilon_n, double complex *N_n, double *rcenters, int projection,
                          int n_cfs, int nbinsr, int nmax, double *phis12, int nbinsphi12, double *phis13, int nbinsphi13,
                          int nthreads, double complex *npcf, double complex *npcf_norm);

void multipoles2npcf_gggg_rtriple(double complex *Upsilon_n, double complex *N_n, int n1max, int n2max,
                                  double *theta1, double *theta2, double *theta3, int nthetas,
                                  double *phis12, double *phis13, int nbinsphi12, int nbinsphi13,
                                  int projection, double complex *npcf, double complex *npcf_norm);

void multipoles2npcf_gggg_singletheta(double complex *Upsilon_n, double complex *N_n, int n1max, int n2max,
                                      double theta1, double theta2, double theta3,
                                      double *phis12, double *phis13, int nbinsphi12, int nbinsphi13,
                                      int projection, double complex *npcf, double complex *npcf_norm);

void multipoles2npcf_gggg_singletheta_nconvergence(
    double complex *Upsilon_n, double complex *N_n, int n1max, int n2max,
    double theta1, double theta2, double theta3,
    double *phis12, double *phis13, int nbinsphi12, int nbinsphi13,
    int projection, double complex *npcf, double complex *npcf_norm);

void fourpcf2M4correlators(int nzcombis,
                           double y1, double y2, double y3, double dy1, double dy2, double dy3,
                           double *phis1, double *phis2, double *dphis1, double *dphis2, int nbinsphi1, int nbinsphi2,
                           double complex *fourpcf, double complex *m4corr);

void fourpcf2M4correlators_parallel(int nzcombis,
                           double y1, double y2, double y3, double dy1, double dy2, double dy3,
                           double *phis1, double *phis2, double *dphis1, double *dphis2, int nbinsphi1, int nbinsphi2,
                           int nthreads, double complex *fourpcf, double complex *m4corr);

void alloc_notomoMap4_analytic(
    double rmin, double rmax, int nbinsr, double *phibins, double *dbinsphi, int nbinsphi, int nsubr,
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, double *mapradii, int nmapradii, 
    double *xip, double *xim, double thetamin_xi, double thetamax_xi, int nthetabins_xi, int nsubsample_filter,
    double complex *M4correlators);

void gauss4pcf_analytic(double theta1, double theta2, double theta3, double *phis, int nphis,
                        double *xip, double *xim, double thetamin_xis, double thetamax_xis, double dtheta_xis,
                        double complex *gaussfourpcf);

void gauss4pcf_analytic_integrated(
    int indbin1, int indbin2, int indbin3, int nsubr, double *rbin_edges, int nbinsr, double *phis, int nphis, 
    double *xip, double *xim, double thetamin_xi, double thetamax_xi, double dtheta_xi,
    double complex *gaussfourpcf);

void filter_Map4(double y1, double y2, double y3, double phi1, double phi2, double complex *output);