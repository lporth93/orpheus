#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void getBellRecursive(int order, double *arguments, double *factorials, double *result);

void nextrgs(int *rgs, int *helper, int order);
int bell(int n);
void update_transformations(int *rgs, int len_rgs, double *powersums, double *fac_table, double *updates, int nupdates);
double update_powersum(int ind, int n, int k, double *fac_table, double *vals);
void update_powersums(int ind, int n, int k, double *fac_table, int nsums, double *vals, double *toadd);
int sel2cumind(int *sel, double *fac_table, int ntot);
int sel2ind(int *sel, double *fac_table, int ntot);
void ind2sel(int ind, int n, int k, double *fac_table, int *sel);
int minr(int ind, int n, int k, double *fac_table);
int maxr(int ind, int n, int k, double *fac_table);

int zcombis_order(int nbinsz, int order, int *fac_table);
int zcombis_tot(int nbinsz, int max_order, int *fac_table); 
void nextzcombination(int n_els, int len_list, int *combination);

double binom(int n , int k, double *fac_table);
void gen_fac_table(int order, double *out);
void gen_fac_table_int(int order, int *out);