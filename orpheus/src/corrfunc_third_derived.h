#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <complex.h>

void _x2centroid_ggg(double complex *npcf, int nbinsz, 
                     double *theta_centers, int nbinstheta, double *phi_centers, int nbinsphi,
                     int nthreads);

void multipoles2npcf_third_z1z23(double complex *Upsilon_n, double complex *N_n,
                                 int nmax, int ncomp_cf, int nbinsz1, int nbinsz23, int nbinstheta,
                                 double *phi_centers, int nbinsphi,
                                 int store_full_range, int *conjmap, double *modeweight,
                                 int is_edge_corrected, int floor_use_abs, double *floor_thr,
                                 int nthreads,
                                 double complex *npcf, double complex *npcf_norm);

void threepcf2M3correlators_ggg(double complex *npcf,
                                double *theta_edges, double *theta_centers, int nbinstheta,
                                double *phi_centers, int nbinsphi, int nzcombis,
                                double *radii1, double *radii2, double *radii3, int nrcombis,
                                int do_multiscale, int nthreads,
                                double complex *M3correlators);

void threepcf2NNMcorrelators_gnn(double complex *npcf,
                                 double *theta_edges, double *theta_centers, int nbinstheta,
                                 double *phi_centers, int nbinsphi, int nzcombis,
                                 double *radii1, double *radii2, double *radii3, int nrcombis,
                                 int nthreads,
                                 double complex *NNMcorrelators);

void threepcf2NMMcorrelators_ngg(double complex *npcf,
                                 double *theta_edges, double *theta_centers, int nbinstheta,
                                 double *phi_centers, int nbinsphi, int nzcombis,
                                 double *radii1, double *radii2, double *radii3, int nrcombis,
                                 int nthreads,
                                 double complex *NMMcorrelators);