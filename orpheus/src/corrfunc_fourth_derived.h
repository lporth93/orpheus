#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <complex.h>

#include "multires_structs.h"

// GGGG //
void getMultipolesFromSymm(double complex *Upsn_in, double complex *Nn_in,
                            int nmax, int eltrafo, int *nindices, int len_nindices,
                            double complex *Upsn_out, double complex *Nn_out);
void multipoles2npcf_nnnn(double complex *N_n, const BinningParams *bin, const FourthParams *fourth,
                          double *theta_centers, double complex *npcf, int nthreads);
void multipoles2npcf_gggg_singletheta(double complex *Upsilon_n, double complex *N_n, int n1max, int n2max,
                                      double theta1, double theta2, double theta3,
                                      double *phis12, double *phis13, int nbinsphi12, int nbinsphi13,
                                      int projection, double complex *npcf, double complex *npcf_norm);

void multipoles2npcf_gggg_singletheta_nconvergence(
    double complex *Upsilon_n, double complex *N_n, int n1max, int n2max,
    double theta1, double theta2, double theta3,
    double *phis12, double *phis13, int nbinsphi12, int nbinsphi13,
    int projection, double complex *npcf, double complex *npcf_norm);

void multipoles2npcf_gggg(double complex *upsilon_n, double complex *N_n, double *rcenters,
                          const BinningParams *bin, const FourthParams *fourth,
                          int projection, int n_cfs, int nthreads,
                          double complex *npcf, double complex *npcf_norm);

void fourpcf2M4correlators(int nzcombis,
                           double y1, double y2, double y3, double dy1, double dy2, double dy3,
                           double *phis1, double *phis2, double *dphis1, double *dphis2, int nbinsphi1, int nbinsphi2,
                           double complex *fourpcf, double complex *m4corr);

void fourpcfmultipoles2M4correlators(
    int nmax, int nmax_trafo,
    double *theta_edges, double *theta_centers, int nthetas, 
    double *mapradii, int nmapradii,
    double *phis1, double *phis2, double *dphis1, double *dphis2, int nbinsphi1, int nbinsphi2,
    int projection, int nthreads, 
    double complex *Upsilon_n, double complex *N_n, double complex *m4corr);

void alloc_notomoMap4_analytic(
    double rmin, double rmax, int nbinsr, double *phibins, double *dbinsphi, int nbinsphi, int nsubr,
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, double *mapradii, int nmapradii, 
    double *xip, double *xim, double thetamin_xi, double thetamax_xi, int nthetabins_xi, int nsubsample_filter,
    double complex *M4correlators);

void gauss4pcf_analytic_integrated(
    int indbin1, int indbin2, int indbin3, int nsubr, double *rbin_edges, int nbinsr, double *phis, int nphis, 
    double *xip, double *xim, double thetamin_xi, double thetamax_xi, double dtheta_xi,
    double complex *gaussfourpcf);

void gauss4pcf_analytic(double theta1, double theta2, double theta3, double *phis, int nphis,
                        double *xip, double *xim, double thetamin_xis, double thetamax_xis, double dtheta_xis,
                        double complex *gaussfourpcf);


// GNNN //
void getMultipolesFromSymm_GNNN(double complex *Gtilden_in, double complex *Nn_in,
                                int nmax, int eltrafo, int *nindices, int len_nindices,
                                double complex *Gtilden_out, double complex *Nn_out);

double gnnn_clustering_corr(double theta1, double theta2, double theta3,
                            double phi12, double phi13,
                            double *xi_nn, double thetamin_xi, double thetamax_xi, double dtheta_xi, int has_xi,
                            double *zeta, double *zeta_rbins, int zeta_nr, double *zeta_phis, int zeta_nphi, int has_zeta);

void multipoles2npcf_gnnn_singletheta(double complex *Gtilde_n, double complex *N_n, int n1max, int n2max,
                                      double theta1, double theta2, double theta3,
                                      double *phis12, double *phis13, int nbinsphi12, int nbinsphi13,
                                      const ClustCorr *cc,
                                      double complex *npcf, double complex *npcf_norm);

void multipoles2npcf_gnnn_singletheta_nconvergence(
    double complex *Upsilon_n, double complex *N_n, int n1max, int n2max,
    double theta1, double theta2, double theta3,
    double *phis12, double *phis13, int nbinsphi12, int nbinsphi13,
    const ClustCorr *cc,
    double complex *npcf, double complex *npcf_norm);

void multipoles2npcf_gnnn(const double complex *Gtilde_n, const double complex *N_n,
                          const BinningParams *bin, const FourthParams *fourth, const ClustCorr *cc,
                          int nthreads, double complex *npcf, double complex *npcf_norm);

void fourpcfmultipoles2MN3correlators(
    int nmax, int nmax_trafo,
    double *theta_edges, double *theta_centers, int nthetas,
    double *apradii_N, double *apradii_M, int napradii,
    double *phis1, double *phis2, double *dphis1, double *dphis2, int nbinsphi1, int nbinsphi2,
    int projection, int nthreads, int verbose,
    const ClustCorr *cc,
    double complex *Upsilon_n, double complex *N_n, double complex *mn3corr);

void fourpcf2MN3correlator(int nzcombis,
                           double y1, double y2, double y3, double dy1, double dy2, double dy3,
                           double *phis1, double *phis2, double *dphis1, double *dphis2, int nbinsphi1, int nbinsphi2,
                           double complex *fourpcf, double complex *mn3corr);

void fourpcf2MN3correlatormulti(int nzcombis, double R1, double R2, double R3, double R4,
                           double theta1, double theta2, double theta3, double dtheta1, double dtheta2, double dtheta3,
                           double *phis1, double *phis2, double *dphis1, double *dphis2, int nbinsphi1, int nbinsphi2,
                           double complex *fourpcf, double complex *mn3corr);

void alloc_notomoMapNap3_corrections(
    double *theta_edges, double *theta_centers, int nthetas, double *phibins, double *dbinsphi, int nbinsphi, int nmax,
    int nthreads, double *apradii, int napradii, 
    double *xing, double complex *Gtilde_third, 
    int include_second, int include_third, double complex *MN3correlators);

void alloc_notomoMapNap3_analytic(
    double rmin, double rmax, int nbinsr, double *phibins, double *dbinsphi, int nbinsphi, int nsubr,
    int *thetacombis_batches, int *nthetacombis_batches, int *cumthetacombis_batches, int nthetbatches,
    int nthreads, double *apradii, int mapradii, 
    double *xing, double *xinn, double thetamin_xi, double thetamax_xi, int nthetabins_xi, int nsubsample_filter,
    double complex *MN3correlators);
    
void gtilde4pcf_analytic_integrated(
    int indbin1, int indbin2, int indbin3, int nsubr, double *rbin_edges, int nbinsr, double *phis, int nphis, 
    double *xing, double *xinn, double thetamin_xi, double thetamax_xi, double dtheta_xi,
    double complex *gaussfourpcf);

void gtilde4pcf_analytic(
    double theta1, double theta2, double theta3, double *phis, int nphis,
    double *xing, double *xinn, double thetamin_xis, double thetamax_xis, double dtheta_xis,
    double complex *gaussfourpcf);

void gtilde4pcf_corrections(
    int itheta1, int itheta2, int itheta3, int nthetas, double *phis, int nphis, int nmax, 
    int include_second, int include_third,  double *xi_ng, double complex *Gtilde_third,
    double complex *fourpcf_corr);

// NNNN //
void getMultipolesFromSymm_NNNN(double complex *Nn_in,
                                 int nmax, int eltrafo, int *nindices, int len_nindices,
                                 double complex *Nn_out);

void multipoles2npcf_nnnn_singletheta(double complex *N_n, int n1max, int n2max,
                                      double theta1, double theta2, double theta3,
                                      double *phis12, double *phis13, int nbinsphi12, int nbinsphi13,
                                      double complex *npcf);


// Generic //
void fourpcf2N4correlators(int nzcombis,
                           double y1, double y2, double y3, double dy1, double dy2, double dy3,
                           double *phis1, double *phis2, double *dphis1, double *dphis2, int nbinsphi1, int nbinsphi2,
                           double complex *fourpcf, double complex *n4corr);

