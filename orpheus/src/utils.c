#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stddef.h>
#include <complex.h>
#include <math.h>

#include "multires_structs.h"

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
    //double res = w*vec[elb_lo] + (1-w)*vec[elb_lo+1];
    double res = (1-w)*vec[elb_lo] + w*vec[elb_lo+1];
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


// Builds product of terms based on tomo bin combination
// Example: zcombi = [z1,z1,z2,z5,z5,z5] --> res = moments[z1,2]*moments[z2,1]*moments[z5,3]
double nexttomoterm(int order, int max_order, double *moments, int *zcombi, int elzcombi, int do_subtractions){
    double res;
    int tmpzbin, tmporder;
    if (do_subtractions){
        if (order>1){
            res = 1;
            tmpzbin = zcombi[0];
            tmporder = 0;
            for (int elzbin=1; elzbin<order; elzbin++){
                if (zcombi[elzbin]==tmpzbin){tmporder+=1;}
                else{
                    res *= moments[tmpzbin*max_order+tmporder];
                    tmporder = 0;
                    tmpzbin = zcombi[elzbin];
                }
            }
            res *= moments[tmpzbin*max_order+tmporder];
        }
        else{
            res = moments[elzcombi*max_order+0];
        }
    }
    else{
        res = 1;
        for (int elzbin=0; elzbin<order; elzbin++){
            tmpzbin = zcombi[elzbin];
            res *= moments[tmpzbin*max_order+0];
        }
    }
    return res;
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

// Progress bar state, at file scope so it can be reset between runs
static int _progress_last_dot = 0;
static int _progress_last_pipe = 0;

// Reset the progress bar before a new run. Call once (single-threaded) before
// the parallel region, otherwise a second run in the same process sees the
// statics already at 100% and prints nothing.
void reset_progress(void) {
    _progress_last_dot = 0;
    _progress_last_pipe = 0;
}

// Prints progress bar consisting of dots and pipes (a . per 1%, a | per 10%).
// Safe to call at any granularity, including once per galaxy: the unlocked
// fast-path returns before the critical section unless a new dot/pipe is due.
void print_progress(int nregionsdone, int nfilledregions, int verbose) {
    if (verbose <= 0) return;

    int one_percent = nfilledregions / 100;
    if (one_percent == 0) one_percent = 1;  // ensure at least one
    int ten_percent = 10 * one_percent;

    int dot_now = nregionsdone / one_percent;
    int pipe_now = nregionsdone / ten_percent;

    // Relaxed unlocked read; a benign race at most repeats a cheap check
    if (dot_now <= _progress_last_dot && pipe_now <= _progress_last_pipe) return;

    #pragma omp critical
    {
        if (pipe_now > _progress_last_pipe) {printf("|");_progress_last_pipe = pipe_now;}
        for (int i = _progress_last_dot + 1; i <= dot_now; i++) {if (i % 10 != 0) printf(".");}
        _progress_last_dot = dot_now;
        fflush(stdout);
    }
}

// Geodesic angular separation (radians) between two unit position vectors.
// Robust atan2(|n1 x n2|, n1.n2) form: accurate at small separations (where the
// dot product alone loses precision), which is exactly the small-radial-bin regime.
double sphere_dist(double x1, double y1, double z1, double x2, double y2, double z2){
    double dot = x1*x2 + y1*y2 + z1*z2;
    double cx = y1*z2 - z1*y2;
    double cy = z1*x2 - x1*z2;
    double cz = x1*y2 - y1*x2;
    double cross = sqrt(cx*cx + cy*cy + cz*cz);
    return atan2(cross, dot);
}

// Bearing at point a=(ra_a,dec_a) toward b=(ra_b,dec_b), expressed in as 
// east-north tangent frame so -->  that it reduces to the flat-sky 
// atan2(d_north, d_east) in the small-angle limit.
double sphere_bearing(double ra_a, double sindec_a, double cosdec_a,
                      double ra_b, double sindec_b, double cosdec_b){
    double dlam = ra_b - ra_a;
    double e = cosdec_b * sin(dlam);
    double n = cosdec_a*sindec_b - sindec_a*cosdec_b*cos(dlam);
    return atan2(n, e);
}


///////////////////////////
// Struct layout exports //
///////////////////////////

// The structs in multires_structs.h are mirrored by hand in multires_structs.py.
// So if during an edit we change stuff in .py but not in .c this will only be seen 
// as memory corruption at runtime. So this just exports the current layout and asserts
// it against  types in tests/test_fast_abi.py. So mainly there to catch development 
// issues. Of course now this also needs to be updated during development!
int orpheus_struct_layout(int which, size_t *out, int len_out){
    int n = 0;
    #define PUT(v)    do { if (n<len_out){out[n] = (size_t)(v);} n++; } while (0)
    #define OFF(S, f) PUT(offsetof(S, f))
    switch (which){
        case 0:
            PUT(sizeof(MultiresoCatalog));
            OFF(MultiresoCatalog, metric);        OFF(MultiresoCatalog, nresos);
            OFF(MultiresoCatalog, ngal_resos);    OFF(MultiresoCatalog, nbinsz);
            OFF(MultiresoCatalog, isinner_resos); OFF(MultiresoCatalog, weight_resos);
            OFF(MultiresoCatalog, zbin_resos);    OFF(MultiresoCatalog, pos1_resos);
            OFF(MultiresoCatalog, pos2_resos);    OFF(MultiresoCatalog, pos3_resos);
            OFF(MultiresoCatalog, vx_resos);      OFF(MultiresoCatalog, vy_resos);
            OFF(MultiresoCatalog, vz_resos);      OFF(MultiresoCatalog, ra_resos);
            OFF(MultiresoCatalog, sindec_resos);  OFF(MultiresoCatalog, cosdec_resos);
            OFF(MultiresoCatalog, e1_resos);      OFF(MultiresoCatalog, e2_resos);
            OFF(MultiresoCatalog, weightsq_resos);
            break;
        case 1:
            PUT(sizeof(NavHash));
            OFF(NavHash, metric);             OFF(NavHash, index_matcher);
            OFF(NavHash, pixs_galind_bounds); OFF(NavHash, pix_gals);
            OFF(NavHash, pix1_start);         OFF(NavHash, pix1_d);
            OFF(NavHash, pix1_n);             OFF(NavHash, pix2_start);
            OFF(NavHash, pix2_d);             OFF(NavHash, pix2_n);
            OFF(NavHash, nregions);           OFF(NavHash, index_matcher_hash);
            OFF(NavHash, filledregions);      OFF(NavHash, nfilledregions);
            OFF(NavHash, slab_offsets);       OFF(NavHash, rshift_bounds);
            OFF(NavHash, nslabs);             OFF(NavHash, z0);
            OFF(NavHash, dpix_z);             OFF(NavHash, ncells_resos);
            OFF(NavHash, nside_nav);          OFF(NavHash, cell_pix);
            OFF(NavHash, cell_redbounds);     OFF(NavHash, rshift_red);
            OFF(NavHash, rshift_cellpix);     OFF(NavHash, rshift_cellbounds);
            break;
        case 2:
            PUT(sizeof(TreeResoParams));
            OFF(TreeResoParams, nresos);          OFF(TreeResoParams, nresos_grid);
            OFF(TreeResoParams, dpix1_resos);     OFF(TreeResoParams, dpix2_resos);
            OFF(TreeResoParams, reso_redges);     OFF(TreeResoParams, resoshift_leafs);
            OFF(TreeResoParams, minresoind_leaf); OFF(TreeResoParams, maxresoind_leaf);
            OFF(TreeResoParams, batch_membudget_mb);
            break;
        case 3:
            PUT(sizeof(BinningParams));
            OFF(BinningParams, rmin);   OFF(BinningParams, rmax);
            OFF(BinningParams, nbinsr); OFF(BinningParams, do_dc);
            OFF(BinningParams, nmax);   OFF(BinningParams, nmin);
            OFF(BinningParams, dccorr); OFF(BinningParams, Pi);
            OFF(BinningParams, rbins);
            break;
        case 4:
            PUT(sizeof(NPCFOutput));
            OFF(NPCFOutput, bin_centers); OFF(NPCFOutput, npcf);
            OFF(NPCFOutput, norm);        OFF(NPCFOutput, norm_mp);
            OFF(NPCFOutput, npair);       OFF(NPCFOutput, npair_cell);
            OFF(NPCFOutput, ncomp);       OFF(NPCFOutput, nmax);
            break;
        case 5:
            PUT(sizeof(FourthParams));
            OFF(FourthParams, nbinsphi1);    OFF(FourthParams, nbinsphi2);
            OFF(FourthParams, phibins1);     OFF(FourthParams, phibins2);
            OFF(FourthParams, dbinsphi1);    OFF(FourthParams, dbinsphi2);
            OFF(FourthParams, nindices);     OFF(FourthParams, len_nindices);
            OFF(FourthParams, nthetacombis); OFF(FourthParams, nthetbatches);
            OFF(FourthParams, thetacombis_batches);
            OFF(FourthParams, nthetacombis_batches);
            OFF(FourthParams, cumthetacombis_batches);
            OFF(FourthParams, count_floor);
            break;
        case 6:
            PUT(sizeof(ClustCorr));
            OFF(ClustCorr, count_floor); OFF(ClustCorr, xi_nn);
            OFF(ClustCorr, thetamin_xi); OFF(ClustCorr, thetamax_xi);
            OFF(ClustCorr, dtheta_xi);   OFF(ClustCorr, has_xi);
            OFF(ClustCorr, zeta);        OFF(ClustCorr, zeta_rbins);
            OFF(ClustCorr, zeta_nr);     OFF(ClustCorr, zeta_phis);
            OFF(ClustCorr, zeta_nphi);   OFF(ClustCorr, has_zeta);
            break;
        default:
            break;
    }
    #undef OFF
    #undef PUT
    return n;
}

// Same as above, but for the field names so this catches permutations of fields having
// the same shape
const char *orpheus_struct_fields(int which){
    switch (which){
        case 0: return "metric,nresos,ngal_resos,nbinsz,isinner_resos,weight_resos,zbin_resos,"
                       "pos1_resos,pos2_resos,pos3_resos,vx_resos,vy_resos,vz_resos,ra_resos,"
                       "sindec_resos,cosdec_resos,e1_resos,e2_resos,weightsq_resos";
        case 1: return "metric,index_matcher,pixs_galind_bounds,pix_gals,pix1_start,pix1_d,"
                       "pix1_n,pix2_start,pix2_d,pix2_n,nregions,index_matcher_hash,"
                       "filledregions,nfilledregions,slab_offsets,rshift_bounds,nslabs,z0,"
                       "dpix_z,ncells_resos,nside_nav,cell_pix,cell_redbounds,rshift_red,"
                       "rshift_cellpix,rshift_cellbounds";
        case 2: return "nresos,nresos_grid,dpix1_resos,dpix2_resos,reso_redges,resoshift_leafs,"
                       "minresoind_leaf,maxresoind_leaf,batch_membudget_mb";
        case 3: return "rmin,rmax,nbinsr,do_dc,nmax,nmin,dccorr,Pi,rbins";
        case 4: return "bin_centers,npcf,norm,norm_mp,npair,npair_cell,ncomp,nmax";
        case 5: return "nbinsphi1,nbinsphi2,phibins1,phibins2,dbinsphi1,dbinsphi2,nindices,"
                       "len_nindices,nthetacombis,nthetbatches,thetacombis_batches,"
                       "nthetacombis_batches,cumthetacombis_batches,count_floor";
        case 6: return "count_floor,xi_nn,thetamin_xi,thetamax_xi,dtheta_xi,has_xi,zeta,"
                       "zeta_rbins,zeta_nr,zeta_phis,zeta_nphi,has_zeta";
        default: return "";
    }
}

/////////////////////////
// Allocation guarding //
/////////////////////////

// Here we collect a few wrappers that make sure one can see where an allocation failed without
// segfaulting. In the python layer this is seen as a MemoryError.
static int _alloc_failed = 0;

void orpheus_clear_error(void){
    _alloc_failed = 0;
}

int orpheus_get_error(void){
    return _alloc_failed;
}

static void note_alloc_failure(size_t nbytes){
    fprintf(stderr, "orpheus: failed to allocate %zu bytes\n", nbytes);
    #pragma omp atomic write
    _alloc_failed = 1;
}

void *orpheus_malloc(size_t nbytes){
    void *p = malloc(nbytes);
    if (p==NULL && nbytes>0){note_alloc_failure(nbytes);}
    return p;
}

void *orpheus_calloc(size_t nmemb, size_t size){
    void *p = calloc(nmemb, size);
    if (p==NULL && nmemb>0 && size>0){note_alloc_failure(nmemb*size);}
    return p;
}


