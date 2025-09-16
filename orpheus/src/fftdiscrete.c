// Compile with: gcc -shared -fPIC -o binned_statistics_2d_inC.so binned_statistics_2d_inC.c -lm

#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "combinatorics.h"
#include "utils.h"

// Compute BellRecursive for all apertures (i.e. pixels)
void fftBellRec(int order, double *arguments, double *factorials, double *result, int bins_x, int bins_y){
    //double *helpres = malloc(sizeof(double)*(order+1));
    double *helparg = malloc(sizeof(double)*order);
    int pixnum = bins_x*bins_y;
    int flat_idx;
    for (int i=0; i<bins_x; i++){
        for (int j=0; j<bins_y; j++){
            flat_idx = i*bins_y + j;
            for (int k=0; k<order; k++){helparg[k] = arguments[flat_idx+k*pixnum];}
            double *helpres = calloc(order+1, sizeof(double));
            getBellRecursive(order, helparg, factorials, helpres);
            for (int k=0; k<order; k++){result[flat_idx+k*pixnum] = helpres[k+1]; if (k%2==0){result[flat_idx+k*pixnum] *= -1.;}}
            free(helpres);
        }
    }
    free(helparg);
}

void getcomplexMapmoments(int orderM, double *factorials, double *Msm_re, double *Msm_im, 
                          double *res_re, double *res_im, int bins_x, int bins_y){
    int *rgs, *helper;
    int nproducts;
    for (int ord=1; ord<=orderM; ord++){
        int *rgs = malloc(ord * sizeof(int)); 
        int *helper = malloc(ord * sizeof(int));
        for (int star=0; star<(int)(ceil(0.5*(1.+(double)ord))); star++){
            for (int el=0; el<bell(ord); el++){
                if (el==0){
                    rgs = calloc(ord, sizeof(int)); 
                    helper = calloc(ord, sizeof(int));
                }else{
                    nextrgs(rgs, helper, ord);
                }
                int nproducts = maxarr(rgs, ord) + 1;
                rgs2MsProd(rgs, ord, star, nproducts, factorials, Msm_re, Msm_im, 
                           res_re, res_im, bins_x, bins_y);
            }
        }
    }
    free(rgs);
    free(helper);
}

void rgs2MsProd(int *rgs, int ord, int star, int nproducts, double *factorials, double *Msm_re, double *Msm_im, 
                double *res_re, double *res_im, int bins_x, int bins_y){
    //printf("%d %d, rgs: ",ord,star);
    //for(int i=0; i<ord; i++){printf("%d ",rgs[i]);}
    //printf("\n");
    int *part = calloc(ord*nproducts, sizeof(int));
    for (int i=0; i<ord; i++){
        part[ord*rgs[i]+i] += 1;
    }
    int pixnum = bins_x*bins_y;
    double *tempres_re = malloc(pixnum * sizeof(double));
    double *tempres_im = malloc(pixnum * sizeof(double));
    for (int i=0; i<pixnum; i++){
            tempres_re[i] = 1.;
            tempres_im[i] = 1.;
    }
    double sign=1.;
    if ((ord+nproducts)%2!=0){sign = -1.;}
    double pref = 1.;
    int ordind, starind;
    for (int i=0; i<nproducts; i++){
        // Get prefactor
        pref *= factorials[numberofequalvals(rgs, ord, i) - 1];
        // Get expression
        ordind = 0;
        starind = 0;
        for (int j=0; j<ord; j++){
            ordind += part[ord*i+j];
            starind += part[ord*i+j];
        }
        for (int j=0; j<ord-star; j++){
            starind -= part[ord*i+j];
        }
        ordind -= 1;
        /*printf("ord%d star%d rgs ",ord,star);
        for(int j=0; j<ord; j++){printf("%d ",rgs[j]);}
        printf(", part ");
        for(int j=0; j<ord; j++){printf("%d ",part[ord*i+j]);}
        printf("\n");*/
        for (int i=0; i<pixnum; i++){
            tempres_re[i] *= Msm_re[i+pixnum*(ordind+ord*starind)];
            tempres_im[i] *= Msm_im[i+pixnum*(ordind+ord*starind)];
        }
    }
    for (int i=0; i<pixnum; i++){
        res_re[i+pixnum*(ord-1+ord*star)] += tempres_re[i]*sign*pref;
        res_im[i+pixnum*(ord-1+ord*star)] += tempres_im[i]*sign*pref;
    }
    free(part);
    free(tempres_re);
    free(tempres_im);
}

int findmax(int *arr, int len){
    int maxval=arr[0];
    for (int i=1; i<len; i++){
        if (maxval<arr[i]){
            maxval = arr[i];
        }
    }
    if (len>=maxval){
        return len;
    } else{
        return maxval;
    }
}

int numberofequalvals(int *arr, int len, int val){
    int n=0;
    for (int i=0; i<len; i++){
        if (val == arr[i]){n++;}
    }
    return n;
}

// Enum für die unterstützten Statistik-Typen
typedef enum {
    STAT_MEAN,
    STAT_SUM,
    STAT_COUNT,
    STAT_WEIGHTED_MEAN
} StatisticType;

// Funktion zur 2D-Binned-Statistik
void binned_statistic_2d(double *x, double *y, double *values, double *weights, int n,
                         int bins_x, int bins_y, double *range_x, double *range_y,
                         StatisticType stat_type, double *result, int *counts, int *binnumberx, int *binnumbery) {
    // Initialisieren der Ergebnis- und Zähl-Arrays mit 0
    double *pixweights = (double *)calloc(bins_x*bins_y, sizeof(double));

    // Berechnen der inversen Bin-Breiten
    double inv_bin_width_x = 1./(range_x[1] - range_x[0]);
    double inv_bin_width_y = 1./(range_y[1] - range_y[0]);
    
    // Iterieren über alle Datenpunkte
    for (int i = 0; i < n; i++) {
        //printf("%.10f ",values[i]);
        // Überprüfen, ob der Punkt innerhalb des Bereichs liegt
        if (x[i] >= range_x[0] && x[i] <= range_x[bins_x] &&
            y[i] >= range_y[0] && y[i] <= range_y[bins_y]) {

            // Berechnen des Bin-Index für die x- und y-Koordinate
            binnumberx[i] = (int)((x[i] - range_x[0])*inv_bin_width_x);
            binnumbery[i] = (int)((y[i] - range_y[0])*inv_bin_width_y);

            // Sicherstellen, dass die Indizes innerhalb der Grenzen liegen
            if (binnumberx[i] >= bins_x) {binnumberx[i] = bins_x - 1;}
            if (binnumbery[i] >= bins_y) {binnumbery[i] = bins_y - 1;}

            // Linearen Index für das flattened 2D-Array berechnen
            int flat_idx = binnumberx[i] * bins_y + binnumbery[i];

            // Statistik aktualisieren
            if (stat_type == STAT_MEAN || stat_type == STAT_SUM) {
                result[flat_idx] += values[i];
            }
            if (stat_type == STAT_WEIGHTED_MEAN) {
                result[flat_idx] += values[i]*weights[i];
                pixweights[flat_idx] += weights[i];
            }
            counts[flat_idx]++;
        }
    }

    // Für den Mittelwert: Summe durch Anzahl teilen
    if (stat_type == STAT_MEAN) {
        for (int i = 0; i < bins_x * bins_y; i++) {
            if (counts[i] > 0) {
                result[i] /= (double)counts[i];
            }
        }
    }

    if (stat_type == STAT_WEIGHTED_MEAN) {
        for (int i = 0; i < bins_x * bins_y; i++) {
            if (pixweights[i] > 0.) {
                result[i] /= pixweights[i];
            }
        }
    }

    // Für die Anzahl: Das Zähl-Array in das Ergebnis kopieren
    if (stat_type == STAT_COUNT) {
        for (int i = 0; i < bins_x * bins_y; ++i) {
            result[i] = (double)counts[i];
        }
    }
    free(pixweights);
}
