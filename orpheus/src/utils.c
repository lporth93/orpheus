#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

int binary_search(double *array, int len_arr, double target){
    int ind_min = 0;
    int ind_max = len_arr - 1;
    while (ind_min <= ind_max) {
        int mid = ind_min + (ind_max-ind_min)/2;
        if (target >= array[mid] && target < array[mid+1]){return mid;}
        if (target >= array[mid + 1]){ind_min = mid+1;}
        else {ind_max = mid-1;}
    }
    // Should not occur...
    return -1;
}


// As the number of pixels goes with R^2 most bins will be in the outer pixels and we need fewer if/else branching
int backsearch(double *array, int ind_min, int ind_max, double target) {
    int ind;
    for (ind=ind_max; ind>=ind_min; ind--) {
        if (target >= array[ind]) {break;}
    }
    return ind;
}



double linint(double *vec, double x, double xmin, double xmax, double dx){
    if (x<=xmin){return 0;}
    if (x>=(xmax-dx)){return 0;}
    int elb_lo = (int) ((x-xmin)/dx);
    double w = (x-(xmin+elb_lo*dx))/dx;
    double res = w*vec[elb_lo] + (1-w)*vec[elb_lo+1];
    //printf("%.9f %.9f %.9f %d %.9f %.9f %.9f %.9f\n",x,xmin,xmax,elb_lo,w,vec[elb_lo],vec[elb_lo+1],res);
    return res;
}
  
// Nested selection for binary array
// Example: arr_long = [0,1,1,0,1,0,1] ; arr_sel = [0,1,1,0] --> Returns [0,0,1,0,1,0,0]
// Note that the number of ones in arr_long must coincide with the length of arr_sel.
void expand_arr(int *arr_long, int *arr_sel, int len_long, int len_sel, int *result){
    int indsel = 0;
    for (int ind=0; ind<len_long; ind++){
        result[ind]=0;
        if (arr_long[ind]==1){
            if (arr_sel[indsel]==1){result[ind]=1;}
            indsel += 1;
            if (indsel==len_sel){break;}
        }
    }
}


int sumintarr(int *arr, int len){
    int res = 0;
    for (int i=0; i<len; i++){
        res += arr[i];
    }
    return res;
}

int countel(int el, int *arr, int len){
    int count=0;
    for (int i=0;i<len;i++){if (arr[i]==el){count += 1;}}
    return count;
}

int maxarr(int *arr, int len){
    int max = arr[0]; 
    for (int i=1; i<len; i++){if (arr[i]>max){max=arr[i];}}
    return max;
}

void fillconsti(int *arr, int len_arr, int c){
    for (int i=0; i<len_arr; i++){
        arr[i]=c;
    }
}

void fillconstd(double *arr, int len_arr, double c){
    for (int i=0; i<len_arr; i++){
        arr[i]=c;
    }
}