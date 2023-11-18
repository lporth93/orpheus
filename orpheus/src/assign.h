#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <complex.h>

// Assign fields in different zbins to same grid scale
// Fields are ordered as [field1, ..., fieldn]
// Result is ordered as [zbin0 --> [weight, grid1,..., gridn], ...]
void assign_fields(
    double *pos1, double *pos2, int *zbins, double *weight, double *fields, 
    int nzbins, int nfields, int ngal, int method, double min1, double min2, 
    double dpix, int n1, int n2, int nthreads, double *result);

// min1/max1 correspond to the min/max of the mesh edges, n1 corresponds to the number of pixels.
// We apply periodic boundary conditions to fulfill mass constraint 
// methods: 0:NGP, 1:CIC, 2:TSC
void assign_shapes2d(
    double *pos1, double *pos2, double *weight, double *e1, double *e2, int ngal, int method,
    double min1, double min2, double dpix, int n1, int n2,
    int nthreads, double *result);

void gen_weightgrid2d(
    double *pos1, double *pos2, int ngal, int method,
    double min1, double min2, double dpix, int n1, int n2,
    int nthreads, int *pixinds, double *pixweights);

// Applies periodic bc in one dimension
// Returns index in [0, ind_max-1]
int periodic(int ind, int ind_max);

// Gets weight of particle in a voxel based on a grid assignment scheme
// methods: 0:NGP, 1:CIC, 2:TSC
double getweight(int window, double reld);