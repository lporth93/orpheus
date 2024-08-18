#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int binary_search(double *array, int len_arr, double target);
double linint(double *vec, double x, double xmin, double xmax, double dx);
void expand_arr(int *arr_long, int *arr_sel, int len_long, int len_sel, int *result);
int sumintarr(int *arr, int len);
int countel(int el, int *arr, int len);
int maxarr(int *arr, int len);
void fillconsti(int *arr, int len_arr, int c);
void fillconstd(double *arr, int len_arr, double c);