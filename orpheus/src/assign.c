//gcc -fopenmp $(gsl-config --cflags) -fPIC -ffast-math -O3 -shared -o assign.so assign.c $(gsl-config --libs) -std=c99

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "assign.h"

#define mymin(x,y) ((x) <= (y)) ? (x) : (y)


void assign_fields(
    double *pos1, double *pos2, int *zbins, double *weight, double *fields, 
    int nzbins, int nfields, int ngal, int method, double min1, double min2, 
    double dpix, int n1, int n2, int nthreads, double *result){
    
    int npix = n1*n2;
    int nfnpix = (nfields+1)*npix;
	//#pragma omp parallel for num_threads(nthreads)
	for (int ind_gal=0; ind_gal<ngal; ind_gal++){
		double w1, w12, wgal, zbin;
		double _d1, _d2, i1, i2, _i1, _i2;
        int elfield, zshift, index;
        
        i1 = (int) floor((pos1[ind_gal]-min1)/dpix);
        i2 = (int) floor((pos2[ind_gal]-min2)/dpix);
        wgal = weight[ind_gal];
        zbin = zbins[ind_gal];
        double *fieldvals = calloc(nfields, sizeof(double));
        for (elfield=0; elfield<nfields; elfield++){
            fieldvals[elfield] = fields[elfield*ngal+ind_gal];
        }
        zshift = zbin*nfnpix;
        
		// NGP
		if (method==0){
			int index = periodic(i2, n2)*n1+ periodic(i1,n1);
			//#pragma omp atomic
			result[zshift+index] += wgal;
            for (elfield=0; elfield<nfields; elfield++){
                result[zshift+(elfield+1)*npix+index] += wgal*fieldvals[elfield];
            }
		}
		// CIC or TSC
		else{
			for (int ind1=i1-method; ind1<=i1+method; ind1++){
				_d1 = fabs(pos1[ind_gal]-(min1+dpix/2. + ind1*dpix))/dpix;
				_i1 = periodic(ind1,n1);
				w1 = getweight(method, _d1);
				for (int ind2=i2-method; ind2<=i2+method; ind2++){
					_d2 = fabs(pos2[ind_gal]-(min2+dpix/2. + ind2*dpix))/dpix;
                    w12 = w1*getweight(method, _d2);
					if (w12 != 0){
						index = periodic(ind2, n2)*n1 + _i1;
						//#pragma omp atomic
						result[zshift+index] += w12;
                        for (elfield=0; elfield<nfields; elfield++){
                            result[zshift+(elfield+1)*npix+index] += w12*wgal*fieldvals[elfield];
                        }
					}
				}
			}
		}
        free(fieldvals);
	}
}

void assign_shapes2d(
    double *pos1, double *pos2, double *weight, double *e1, double *e2, int ngal, int method,
    double min1, double min2, double dpix, int n1, int n2,
    int nthreads, double *result){
    
    int npix = n1*n2;
	#pragma omp parallel for num_threads(nthreads)
	for (int ind_gal=0; ind_gal<ngal; ind_gal++){
		double w1, w12, wgal, e1gal, e2gal;
		double _d1, _d2, i1, i2, _i1, _i2;
        double toadd1, toadd2, toadd3;
        int index;
        
        i1 = (int) floor((pos1[ind_gal]-min1)/dpix);
        i2 = (int) floor((pos2[ind_gal]-min2)/dpix);
        wgal = weight[ind_gal];
        e1gal = e1[ind_gal];
        e2gal = e2[ind_gal];
        
		// NGP
		if (method==0){
			int index = periodic(i2, n2)*n1+ periodic(i1,n1);
			#pragma omp atomic
			result[index] += wgal;
            result[npix+index] += wgal*e1gal;
            result[2*npix+index] += wgal*e2gal;
		}
		// CIC or TSC
		else{
			for (int ind1=i1-method; ind1<=i1+method; ind1++){
				_d1 = fabs(pos1[ind_gal]-(min1+dpix/2. + ind1*dpix))/dpix;
				_i1 = periodic(ind1,n1);
				w1 = getweight(method, _d1);
				for (int ind2=i2-method; ind2<=i2+method; ind2++){
					_d2 = fabs(pos2[ind_gal]-(min2+dpix/2. + ind2*dpix))/dpix;
                    w12 = w1*getweight(method, _d2);
                    toadd1 = w12;
					toadd2 = w12*e1gal;
                    toadd3 = w12*e2gal;
					if (toadd1 != 0){
						index = periodic(ind2, n2)*n1 + _i1;
						#pragma omp atomic
						result[index] += toadd1;
                        result[npix+index] += toadd2;
                        result[2*npix+index] += toadd3;
					}
				}
			}
		}
	}
}

void gen_weightgrid2d(
    double *pos1, double *pos2, int ngal, int method,
    double min1, double min2, double dpix, int n1, int n2,
    int nthreads, int *pixinds, double *pixweights){
    
    int npix = n1*n2;
    int nwspergal_side = (2*method+1);
    int nwspergal = nwspergal_side*nwspergal_side;
	for (int ind_gal=0; ind_gal<ngal; ind_gal++){
		double w1, w12;
		double _d1, _d2, i1, i2, _i1, _i2;
        int indside1, indside2;
        int index;
        int gshift = ind_gal*nwspergal;
        i1 = (int) floor((pos1[ind_gal]-min1)/dpix);
        i2 = (int) floor((pos2[ind_gal]-min2)/dpix);
        
		// NGP
		if (method==0){
			int index = periodic(i2, n2)*n1+ periodic(i1,n1);
			pixinds[gshift] = index;
            pixweights[gshift] = 1;
		}
		// CIC or TSC
		else{
			for (int ind1=i1-method; ind1<=i1+method; ind1++){
                indside1 = ind1-(i1-method);
				_d1 = fabs(pos1[ind_gal]-(min1+dpix/2. + ind1*dpix))/dpix;
				_i1 = periodic(ind1,n1);
				w1 = getweight(method, _d1);
				for (int ind2=i2-method; ind2<=i2+method; ind2++){
                    indside2 = ind2-(i2-method);
					_d2 = fabs(pos2[ind_gal]-(min2+dpix/2. + ind2*dpix))/dpix;
                    w12 = w1*getweight(method, _d2);
                    index = periodic(ind2, n2)*n1 + _i1;
                    pixinds[gshift+indside1*nwspergal_side+indside2] = index;
                    pixweights[gshift+indside1*nwspergal_side+indside2] = w12;
				}
			}
		}
	}
}


// windows: 0:NGP, 1:CIC, 2:TSC
double getweight(int window, double reld){
	double weight = 0;
	if (window==0){weight = 1;}
	if (window==1){if (reld<=1){weight=1-reld;}}
	if (window==2){if (reld<=1.5){if (reld<=0.5){weight=0.75-reld*reld;} else{weight=0.5*pow(1.5-reld,2);}}}
	return weight;
}

int periodic(int ind, int ind_max){
	int result = ind % ind_max;
	if (result<0){result = ind_max + result;}
	return result;
}
