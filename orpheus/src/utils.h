#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int binary_search(double *array, int len_arr, double target);
double linint(double *vec, double x, double xmin, double xmax, double dx);
void expand_arr(int *arr_long, int *arr_sel, int len_long, int len_sel, int *result);
double nexttomoterm(int order, int max_order, double *moments, int *zcombi, int elzcombi, int do_subtractions);
int sumintarr(int *arr, int len);
int countel(int el, int *arr, int len);
int maxarr(int *arr, int len);
void fillconsti(int *arr, int len_arr, int c);
void fillconstd(double *arr, int len_arr, double c);
void print_progress(int nregionsdone, int nfilledregions, int verbose);
double sphere_dist(double x1, double y1, double z1, double x2, double y2, double z2);
double sphere_bearing(double ra_a, double sindec_a, double cosdec_a,
                      double ra_b, double sindec_b, double cosdec_b);