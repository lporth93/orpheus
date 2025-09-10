// This file has all the small combinatorical functions in it
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "combinatorics.h"
#include "utils.h"

#define mymin(x,y) ((x) <= (y)) ? (x) : (y)
#define mymax(x,y) ((x) >= (y)) ? (x) : (y)
////////////////////////////////////////////////////////////////////////
// COMBINATORICS RELATED TO TRAFO OF SINGLE-SCALE APERTURE STATISTICS //
////////////////////////////////////////////////////////////////////////

// This is a quick implementation, that works only up to order 170...
// Reason for that is the passed factorial table which enables quick computation
// of the binominal coefficients
void getBellRecursive(int order, double *arguments, double *factorials, double *result){
    double binom;
    int idxn, idxk;
    result[0]=1;
    result[1]=arguments[0];
    for (idxn=2; idxn<order+1; idxn++){
        for (idxk=0; idxk<idxn; idxk++){
            binom = factorials[idxn-1]/(factorials[idxk]*factorials[idxn-1-idxk]);
            result[idxn] += binom*result[idxn-1-idxk]*arguments[idxk];
        }
    }
}

///////////////////////////////////////////////////////////////////////
// COMBINATORICS RELATED TO TRAFO OF MULTI-SCALE APERTURE STATISTICS //
///////////////////////////////////////////////////////////////////////

// Build next rgs string from previous one
void nextrgs(int *rgs, int *helper, int order){
    int i=order-1;
    int j;
    while (i>=1){
        if (rgs[i] <= helper[i-1]){
            rgs[i] += 1; 
            helper[i] = mymax(helper[i], rgs[i]);
            for (j=i+1;j<order;j++){
                rgs[j] = rgs[0];
                helper[j] = helper[i]; 
            }
            break;
        }
        i-=1;
    }
}

// Compute nth bell number using bell triangles (https://en.wikipedia.org/wiki/Bell_triangle)
int bell(int n) { 
    int i,j;
    int res[n+1][n+1]; 
    res[0][0] = 1; 
    for (i=1; i<=n; i++) { 
        res[i][0] = res[i-1][i-1]; 
        for (j=1; j<=i; j++){
            res[i][j] = res[i-1][j-1] + res[i][j-1];}
    } 
    return res[n][0]; 
} 

// Computes product of mulltivariate power sums corresponding to a certain partition which is
// encoded in a restricted growth string
void update_transformations(int *rgs, int len_rgs, double *powersums, double *fac_table, 
    double *updates, int nupdates){
    int i, j;
    int nsums = pow(2,len_rgs) - 1; // Number of power sums per set
    int nproducts = maxarr(rgs, len_rgs) + 1; // Number of products in the update
    int *fullsel = calloc(nproducts*len_rgs, sizeof(int));
    int *nextsel = calloc(len_rgs, sizeof(int));
    // Select overall sign for all the updates. This is given by (-1)^x where 
    // x is the number of power sums of even order occuring in the partition
    // We can get this by setting sign=(-1)^(nradii+nproducts)
    // Proof: 
    // We just need to check that the equation above works for the four cases 
    // nradii \in [even/odd] and nproducts \in [even/odd]. For this we recall that
    // nradii = sum_{i=1}^{nproducts} order(powersum_i).
    // * Suppose we have an even number of radii and an even number of products.
    //   Then we must have an even number of odd order power sums (as otherwise nradii)
    //   would have to be odd). Therefore we must also have an even number of even order
    //   power sums, hence the overall sign should be "+". In our equation we have
    //   sign = (-1)^(even+even) = "+" --> Our equation works for this case
    // * The other three cases go along the same lines and all work out.
    for (i=0; i<nupdates; i++){
        updates[i] = pow(-1, (len_rgs+nproducts)%2);}
    //  Build the full selection corresponding to the rgs
    for (i=0; i<len_rgs; i++){fullsel[len_rgs*rgs[i]+i] += 1;}
    // Update the transformations
    for (i=0; i<nproducts; i++){
        for (j=0; j<len_rgs; j++){nextsel[j] = fullsel[i*len_rgs+j];}
        for (j=0; j<nupdates; j++){    
            updates[j] *= fac_table[countel(i, rgs, len_rgs) - 1];
            updates[j] *= powersums[j*nsums+sel2cumind(nextsel, fac_table, len_rgs)];
        }
    }
    free(fullsel);
    free(nextsel);
}


// n: Number of different aperture radii
// k: Number of selected aperture radii
// ind: Index of selection, given n and k
// fac_table: Helper table that has the factorials between 0 and n for fast evaluation of binomial coefficients
// vals: values of w_gal * e_t,gal * Q(d_gal;r_ap[i])
// Returns:
// toadd: Value to add to corresponding power sum 
double update_powersum(int ind, int n, int k, double *fac_table, double *vals){
    double y;
    double toadd;
    if (k==0){toadd=0;}
    else{
        toadd = 1;
        while (n>0){
            y=0;
            if (n>k && k>=0){
                y=binom(n-1, k, fac_table);
            }
            if (ind>=y){
                toadd *= vals[n-1];
                ind-=y; 
                k-=1;
            }
            n-=1;
        }
    }
    return toadd;
} 
    

// Basically a version of update_powersum that updates multiple power sums at once.
// This might be more useful as we only need to traverse the inner loops once and not separately
// for each power sum to be updated.
// Arguments:
// n: Number of different aperture radii
// k: Number of selected aperture radii
// ind: Index of selection, given n and k
// fac_table: Helper table that has the factorials between 0 and n for fast evaluation of binomial coefficients
// nsums: The number of different power sums we want to update
// vals: values of of the 'nsum' single evaluations. In our case of updating map, mx, w and w2 (nsums=4) this would be
//       [map[i], mx[i], wgal, w2gal] where each block consists of n components.
// toadd: Returns the nsums values to add to the corresponding power sums
void update_powersums(int ind, int n, int k, double *fac_table, int nsums, double *vals, double *toadd){
    double y;
    int i;
    int ntot = n;
    if (k>0){
        for (i=0; i<nsums; i++){toadd[i]=1;};
        while (n>0){
            y=0;
            if (n>k && k>=0){
                y=binom(n-1, k, fac_table);
            }
            if (ind>=y){
                for (i=0; i<nsums; i++){toadd[i] *= vals[i*ntot+n-1];}
                ind-=y; 
                k-=1;
            }
            n-=1;
        }
    }
}

int sel2cumind(int *sel, double *fac_table, int ntot){
    int nsel = sumintarr(sel, ntot);
    int ind = 0;
    for (int i=1; i<nsel; i++){
        ind += (int) binom(ntot, i, fac_table);}
    return ind + sel2ind(sel, fac_table, ntot);
}

int sel2ind(int *sel, double *fac_table, int ntot){
    int nsel = sumintarr(sel, ntot);
    int ind = 0;
    while (ntot>0){
        //printf("\n%i %i %i", ntot, nsel, ind);
        if (sel[ntot-1]==1){
            ind += binom(ntot-1, nsel, fac_table);
            nsel -= 1;
        }
        ntot -= 1;
    }
    return ind;
}

// Computes 'ind'th selection of binominial coefficient
void ind2sel(int ind, int n, int k, double *fac_table, int *sel){
    int y;
    for (int i=0; i<n; i++){sel[i]=0;}
    while (n>0){
        y=0;
        if (n>k && k>=0){
            y = (int) binom(n-1, k, fac_table);
        }
        if (ind>=y){
            sel[n-1]=1;
            ind -= y;
            k -= 1;
        }
        n -= 1;
    }
} 

// Find smallest radius of powersum corresponding to index
int minr(int ind, int n, int k, double *fac_table){
    while (n>0){
        if (ind>=(int) binom(n-1, k, fac_table)){break;}
        n-=1;
    }
    return n-1;
}

// Find largest radius of powersum corresponding to index
int maxr(int ind, int n, int k, double *fac_table){
    int indmax = 0;
    int *tmpsel = calloc(n, sizeof(double));
    ind2sel(ind, n, k, fac_table, tmpsel);
    for (int i=0; i<=n; i++){
        if (tmpsel[i] == 1){indmax=i;break;}
    }
    free(tmpsel);
    return indmax;
}

///////////////////////////////////////////////////////////////////////////////////////////
// COMBINATORICS RELATED TO TOMOGRAPHIC ENUMERATIONS OF SINGLE-SCALE APERTURE STATISTICS //
///////////////////////////////////////////////////////////////////////////////////////////

// Number of ways to arrange zbins for a specific order of Mapn.
// Due to the invariance of Mapn under permutations of the redshifts, we assume that the zbins are sorted.
int zcombis_order(int nbinsz, int order, double *fac_table){
    return nbinsz * (fac_table[nbinsz+order-1]/(fac_table[nbinsz]*fac_table[order]));
}

// Total number of redshift combinations up to order `max_order`
int zcombis_tot(int nbinsz, int max_order, double *fac_table){
    int res = 0;
    for (int order=1;order<=max_order;order++){
        res += zcombis_order(nbinsz, order, fac_table);
    }
    return res;
}

int zcombis_tot_two(int nbinsz_l, int nbinsz_s, int max_order, double *fac_table){
    int res = 0;
    int nextzcombis_s, nextzcombis_l;
    for (int order=1;order<=max_order;order++){
        for (int order_s=0;order_s<=order;order_s++){
            nextzcombis_s = zcombis_order(nbinsz_s, order_s, fac_table);
            nextzcombis_l = zcombis_order(nbinsz_l, order-order_s, fac_table);
            res += nextzcombis_s*nextzcombis_l;
        }
    }
    return res;
}

// Generates next zcombination 
// Abstractly, next iteration of sorted list of length `order` with `n` distinct elements.
// I.e. order=3, n=3 --> [[0,0,0],[0,0,1],[0,0,2],[0,1,1],[0,1,2],[0,2,2],[1,1,1],[1,1,2],[1,2,2],[2,2,2]]
void nextzcombination(int n_els, int len_list, int *combination){
    int i, j;
    // Find rightmost element that can be incremented
    for (i=len_list-1; i>=0; i--) {
        if (combination[i]<n_els-1) {
            combination[i]++;
            break;
        }
    }
    // Set all elements right of the incremented element to match it
    if (i>=0){
        for (j=i+1; j<len_list; j++) {
            combination[j] = combination[i];
        }
    }
}

/////////////////////////////////
// BASIC COMBINATORICS HELPERS //
/////////////////////////////////

double binom(int n, int k, double *fac_table){
    int res = 0;
    if (n>=k){res = fac_table[n]/(fac_table[k]*fac_table[n-k]);}
    return res;
}

void gen_fac_table_int(int order, int *out){
    out[0] = 1;
    out[1] = 1;
    for (int i=2; i<=order; i++){out[i]=i*out[i-1];}
}

void gen_fac_table(int order, double *out){
    out[0] = 1;
    out[1] = 1;
    for (int i=2; i<=order; i++){out[i]=i*out[i-1];}
}