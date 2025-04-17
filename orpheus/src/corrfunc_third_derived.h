#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <complex.h>

void multipoles2npcf_ggg(double complex *Upsilon_n, double complex *N_n, int nmax, int nbinsz,
                         double *theta_centers, int nbinstheta, double *phi_centers, int nbinsphi,
                         int projection, int nthreads,
                         double complex *npcf, double complex *npcf_norm);

void _x2centroid_ggg(double complex *npcf, int nbinsz, 
                     double *theta_centers, int nbinstheta, double *phi_centers, int nbinsphi,
                     int nthreads);

void threepcf2M3correlators_singlescale(double *npcf, 
                                        double *theta_edges, double *theta_centers, int nbinstheta, 
                                        double *phi_centers, int nbinsphi, int nbinsz,
                                        int nthreads,
                                        double *mapradii, int nmapradii, double complex *M3correlators);