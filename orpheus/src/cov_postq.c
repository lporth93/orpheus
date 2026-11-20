// Helpers for computation of tomographic xipm covariance

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "utils.h"
#include "cov_postq.h"

#define M_PI 3.14159265358979323846

// Interval i and weight w of x in a log-uniform grid, -1 outside of it
static inline int interp1d_locate(
    const double *x, int n, double logx0, double inv_dlogx, double xval, double *w){

    if (xval <= x[0] || xval >= x[n-1]) return -1;
    int i = (int)((log(xval)-logx0)*inv_dlogx);
    if (i < 0) i = 0;
    if (i > n-2) i = n-2;
    if (i < n-2 && x[i+1] <= xval) i++;
    else if (i > 0 && x[i] > xval) i--;
    *w = (xval-x[i])/(x[i+1]-x[i]);
    return i;
}

// Linear interpolation of y(x) on a log-uniform grid x, zero outside of it
static inline double interp1d_eval(
    const double *x, const double *y, int n,
    double logx0, double inv_dlogx, double xval){

    double w;
    int i = interp1d_locate(x, n, logx0, inv_dlogx, xval, &w);
    return i < 0 ? 0. : y[i] + w*(y[i+1]-y[i]);
}

//////////////////////////////////////
// Quadruplet density interpolation //
//////////////////////////////////////

// Metadata of the 4-D interpolation grid: the two radial axes are log-uniform
// and share `r_centers`, the two angular axes are uniform and periodic.
typedef struct {
    const double *r_centers, *phi_centers;
    int nr, nphi;
    double r_min, r_max, logr0, inv_dlogr, phi0, inv_dphi;
    long s_i2, s_i3; // long: nr^2*nphi^2 can exceed 2^31 at large grids
} Interp4dGrid;

static inline Interp4dGrid interp4d_grid(
    const double *r_centers, const double *phi_centers, int nr, int nphi,
    double r_min, double r_max, double logr0, double inv_dlogr,
    double phi0, double inv_dphi){

    Interp4dGrid g;
    g.r_centers = r_centers;
    g.phi_centers = phi_centers;
    g.nr = nr;
    g.nphi = nphi;
    g.r_min = r_min;
    g.r_max = r_max;
    g.logr0 = logr0;
    g.inv_dlogr = inv_dlogr;
    g.phi0 = phi0;
    g.inv_dphi = inv_dphi;
    g.s_i2 = (long)nr*nphi*nphi;
    g.s_i3 = (long)nphi*nphi;
    return g;
}

// One axis of the quadrilinear stencil, resolved on its own so that loop-invariant
// axes can be hoisted out of the q build of the SVA kernel
typedef struct { int i, ip; double w, mw; } Interp4dAxis;

// Log-uniform radial axis: zero above the grid, nearest neighbour below it
static inline __attribute__((always_inline)) Interp4dAxis interp4d_axis_r(
    const Interp4dGrid *g, double t){

    Interp4dAxis a;
    if (t > g->r_max){ a.i = 0; a.ip = 0; a.w = 0.; a.mw = 0.; return a; }

    const double *r_centers = g->r_centers;
    int nr = g->nr;
    int i = (int)((log(t)-g->logr0)*g->inv_dlogr);
    if (i < 0) i = 0;
    else if (i > nr-2) i = nr-2;
    if (i < nr-2 && r_centers[i+1] <= t) i++;
    else if (i > 0 && r_centers[i] > t) i--;

    double w = (t-r_centers[i])/(r_centers[i+1]-r_centers[i]);
    if (t < g->r_min){ i = 0; w = 0.; }
    a.i = i; a.ip = i+1; a.w = w; a.mw = 1.-w;
    return a;
}

// Uniform angular axis of nphi centers spanning 2*pi, periodic: angles below the first or
// above the last center interpolate between the last and the first one. (Clamping the stencil
// to the inner pairs instead extrapolated linearly there, which is where the second pair of
// the quadruplet integral sits for small theta2/phi.)
static inline __attribute__((always_inline)) Interp4dAxis interp4d_axis_phi(
    const Interp4dGrid *g, double a){

    int nphi = g->nphi;
    double u = (a-g->phi0)*g->inv_dphi; // in bin spacings from the first center
    double fl = floor(u);
    int i = (int)(fl - nphi*floor(fl/nphi)); // fl mod nphi, in [0, nphi)
    if (i >= nphi) i -= nphi; // rounding for fl just below a multiple of nphi

    Interp4dAxis ax;
    ax.w = u-fl;
    ax.mw = 1.-ax.w;
    ax.i = i; ax.ip = (i+1 < nphi) ? i+1 : 0;
    return ax;
}

// The 16-corner sum over the four resolved axes
static inline __attribute__((always_inline)) double interp4d_gather(
    const double *q_slice, const Interp4dGrid *g, const Interp4dAxis *a2,
    const Interp4dAxis *a3, const Interp4dAxis *a12, const Interp4dAxis *a13){

    int nphi = g->nphi;
    double w2 = a2->w, mw2 = a2->mw, w3 = a3->w, mw3 = a3->mw;
    double w12 = a12->w, mw12 = a12->mw, w13 = a13->w, mw13 = a13->mw;
    int i13 = a13->i, i13p = a13->ip;

    long b_i2 = (long)a2->i*g->s_i2;
    long b_i2p = (long)a2->ip*g->s_i2;
    long b_i3 = (long)a3->i*g->s_i3;
    long b_i3p = (long)a3->ip*g->s_i3;
    long b_i12 = (long)a12->i*nphi;
    long b_i12p = (long)a12->ip*nphi;

    return
        mw2*mw3*mw12*mw13*q_slice[b_i2+b_i3+b_i12+i13]
        + mw2*mw3*mw12*w13*q_slice[b_i2+b_i3+b_i12+i13p]
        + mw2*mw3*w12*mw13*q_slice[b_i2+b_i3+b_i12p+i13]
        + mw2*mw3*w12*w13*q_slice[b_i2+b_i3+b_i12p+i13p]
        + w2*mw3*mw12*mw13*q_slice[b_i2p+b_i3+b_i12+i13]
        + w2*mw3*mw12*w13*q_slice[b_i2p+b_i3+b_i12+i13p]
        + w2*mw3*w12*mw13*q_slice[b_i2p+b_i3+b_i12p+i13]
        + w2*mw3*w12*w13*q_slice[b_i2p+b_i3+b_i12p+i13p]
        + mw2*w3*mw12*mw13*q_slice[b_i2+b_i3p+b_i12+i13]
        + mw2*w3*mw12*w13*q_slice[b_i2+b_i3p+b_i12+i13p]
        + mw2*w3*w12*mw13*q_slice[b_i2+b_i3p+b_i12p+i13]
        + mw2*w3*w12*w13*q_slice[b_i2+b_i3p+b_i12p+i13p]
        + w2*w3*mw12*mw13*q_slice[b_i2p+b_i3p+b_i12+i13]
        + w2*w3*mw12*w13*q_slice[b_i2p+b_i3p+b_i12+i13p]
        + w2*w3*w12*mw13*q_slice[b_i2p+b_i3p+b_i12p+i13]
        + w2*w3*w12*w13*q_slice[b_i2p+b_i3p+b_i12p+i13p];
}

// Quadrilinear lookup q(t2, t3, a12, a13) on a (nr, nr, nphi, nphi) slice
static inline __attribute__((always_inline)) double interp4d_point(
    const double *q_slice, const Interp4dGrid *g,
    double t2, double t3, double a12, double a13){

    Interp4dAxis x2 = interp4d_axis_r(g, t2);
    Interp4dAxis x3 = interp4d_axis_r(g, t3);
    Interp4dAxis x12 = interp4d_axis_phi(g, a12);
    Interp4dAxis x13 = interp4d_axis_phi(g, a13);
    return interp4d_gather(q_slice, g, &x2, &x3, &x12, &x13);
}

// Quadrilinear interpolation of a (nr, nr, nphi, nphi) quadruplet-density slice at n points,
// log-uniform in the two radial and periodic in the two angular axes
void interp4d(
    const double *q_slice, const double *r_centers, const double *phi_centers,
    int nr, int nphi, double r_min, double r_max, double logr0, double inv_dlogr,
    double phi0, double inv_dphi,
    const double *t2f, const double *t3f, const double *a12f, const double *a13f,
    int n, double *out){

    Interp4dGrid g = interp4d_grid(r_centers, phi_centers, nr, nphi,
        r_min, r_max, logr0, inv_dlogr, phi0, inv_dphi);

    for (int idx = 0; idx < n; idx++)
        out[idx] = interp4d_point(q_slice, &g, t2f[idx], t3f[idx],
                                  a12f[idx], a13f[idx]);
}

///////////////////////////////
// Quadruplet (SVA) integral //
///////////////////////////////

// Cosmic-variance (r-term) integral of all callbacks over the phi grid, with the two sides
// psi_a = phi - theta1 e(V1) and psi_b = phi + theta2 e(V2) and the quadruplet density q
// (q_mode 0: tensor, 1: constant, 2: separable).
//
// Every weight is a sum of at most two products of a factor of each side, and a density tensor
// is multilinear in its nodes, so the quadrature sums over V1 and V2 are projected onto the
// nodes of q instead of evaluating q on the (V1, V2) grid: side a onto the angular nodes of
// V1 (A) and side b onto the (|psi_b|, varphi_b) cells its circle crosses (B). An element is
// then A^T q B, the same sum in another order.
//
// The r-terms are linear in q, so that a second density q2 on its own radial centers (same
// angular bins, q2_nr > 0) gives the Richardson extrapolation 2 R(q) - R(q2) in the same pass.
// The threads share the (theta1, theta2) elements and not the phi nodes, which needs a single
// accumulator and makes the result independent of their number.

// Factors of a side: 1, cos and sin of 4 psi, of 4 (V - psi) and of 4 V
#define SVA_NFAC 7
static const int SVA_NTERM[5] = {1, 2, 2, 2, 1};
static const int SVA_FAC_A[5][2] = {{0, 0}, {1, 2}, {3, 4}, {5, 6}, {0, 0}};
static const int SVA_FAC_B[5][2] = {{0, 0}, {1, 2}, {3, 4}, {5, 6}, {3, 0}};
static const double SVA_COEF[5][2] = {{1., 0.}, {1., 1.}, {1., -1.}, {1., 1.}, {2., 0.}};

// Index of the (xi table, factor) vector of a side, appended if new
static int sva_vector(int tab, int fac, int *tabs, int *facs, int *n){
    for (int i = 0; i < *n; i++)
        if (tabs[i] == tab && facs[i] == fac) return i;
    tabs[*n] = tab; facs[*n] = fac;
    return (*n)++;
}

// The factors and all xi tables at psi = (x, y) of the node angle with cos(4V) = c4v, sin(4V) = s4v
static inline void sva_side(
    double x, double y, double c4v, double s4v,
    const double *xi_x, const double *xi_y_tables, int xi_n, int n_xi,
    double xi_logx0, double xi_inv_dlogx,
    double *r, double *psi, double *fac, double *xi){

    *r = hypot(x, y);
    *psi = atan2(y, x);
    double c4 = cos(4.*(*psi)), s4 = sin(4.*(*psi));
    fac[0] = 1.; fac[1] = c4; fac[2] = s4;
    fac[3] = c4v*c4 + s4v*s4; fac[4] = s4v*c4 - c4v*s4;
    fac[5] = c4v; fac[6] = s4v;
    double w;
    int i = interp1d_locate(xi_x, xi_n, xi_logx0, xi_inv_dlogx, *r, &w);
    for (int t = 0; t < n_xi; t++) {
        const double *y = xi_y_tables + (long)t*xi_n;
        xi[t] = i < 0 ? 0. : y[i] + w*(y[i+1]-y[i]);
    }
}

// A density tensor with its slice at t1 = phi and the radial axis of every theta1 node
typedef struct {
    Interp4dGrid grid;
    const double *q;
    double coef; // weight of its integral in the result
    double *slice;
    Interp4dAxis *axis_t1;
} SvaDensity;

// Side b of one theta2 bin on a density: B (ncell, n_bv) on the cells its circle crosses and
// Y = q B (nr, n_bv, nphi) on the theta1 nodes that are needed
typedef struct {
    int *cellmap, *cells, ncell;
    char *need;
    double *B, *Y;
} SvaScratch;

static inline void sva_add_corner(SvaScratch *s, int cell, double w, const double *val, int n_bv){
    if (w == 0.) return;
    int c = s->cellmap[cell];
    if (c < 0) {
        c = s->ncell++;
        s->cellmap[cell] = c;
        s->cells[c] = cell;
        memset(s->B + (long)c*n_bv, 0, (size_t)n_bv*sizeof(double));
    }
    double *B = s->B + (long)c*n_bv;
    #pragma omp simd
    for (int b = 0; b < n_bv; b++) B[b] += w*val[b];
}

void quadruplet_sva_integrate(
    const double *phi_grid, const double *phi_w, int n_phi, int n_r,
    const double *phi1_nodes, const double *phi1_weights, int ns1,
    const double *phi2_nodes, const double *phi2_weights, int ns2,
    const double *v1_grid, const double *v1_w, int nv1,
    const double *v2_grid, const double *v2_w, int nv2,
    int q_mode, const double *q_array,
    const double *q_r_centers, const double *q_phi_centers, int q_nr, int q_nphi,
    const double *q2_array, const double *q2_r_centers, int q2_nr,
    double q_const, const double *sup_t1, const double *sup_t2, const double *sup_t3,
    const double *xi_x, int xi_n, double xi_logx0, double xi_inv_dlogx,
    const double *xi_y_tables, int n_xi_unique,
    int n_cb, const int *xi_a_idx, const int *xi_b_idx,
    const int *w_ids, const int *out_idx, const double *wick_fac,
    int n_out, const int *symmetric_mask, const int *pair_mask,
    double *results, int nthreads, int verbose){

    int n1 = n_r*ns1;
    int n_den = q_mode == 0 ? (q2_nr > 0 ? 2 : 1) : 0;
    int na_k = q_mode == 0 ? q_nphi : 1; // angular nodes of V1 carried by A

    // Vectors of the two sides and the terms a^T q b of every callback
    int ncb = n_cb > 0 ? n_cb : 1;
    int *av_tab = (int *)orpheus_malloc((size_t)2*ncb*sizeof(int));
    int *av_fac = (int *)orpheus_malloc((size_t)2*ncb*sizeof(int));
    int *bv_tab = (int *)orpheus_malloc((size_t)2*ncb*sizeof(int));
    int *bv_fac = (int *)orpheus_malloc((size_t)2*ncb*sizeof(int));
    int *term_a = (int *)orpheus_malloc((size_t)2*ncb*sizeof(int));
    int *term_b = (int *)orpheus_malloc((size_t)2*ncb*sizeof(int));
    int *term_out = (int *)orpheus_malloc((size_t)2*ncb*sizeof(int));
    double *term_coef = (double *)orpheus_malloc((size_t)2*ncb*sizeof(double));
    double *cos_v1 = (double *)orpheus_malloc((size_t)nv1*sizeof(double));
    double *sin_v1 = (double *)orpheus_malloc((size_t)nv1*sizeof(double));
    double *cos4v1 = (double *)orpheus_malloc((size_t)nv1*sizeof(double));
    double *sin4v1 = (double *)orpheus_malloc((size_t)nv1*sizeof(double));
    double *cos_v2 = (double *)orpheus_malloc((size_t)nv2*sizeof(double));
    double *sin_v2 = (double *)orpheus_malloc((size_t)nv2*sizeof(double));
    double *cos4v2 = (double *)orpheus_malloc((size_t)nv2*sizeof(double));
    double *sin4v2 = (double *)orpheus_malloc((size_t)nv2*sizeof(double));
    char *row_on = (char *)orpheus_calloc((size_t)n_r, 1);
    char *col_on = (char *)orpheus_calloc((size_t)n_r, 1);
    // Accumulator (theta2, theta1, output): an element is contiguous and owned by one thread
    double *acc = (double *)orpheus_calloc((size_t)n_r*n_r*n_out, sizeof(double));
    Interp4dAxis *axis_v1 = (Interp4dAxis *)orpheus_malloc(
        (size_t)(nv1 > 0 ? nv1 : 1)*sizeof(Interp4dAxis));
    int alloc_ok = (av_tab && av_fac && bv_tab && bv_fac && term_a && term_b && term_out
                    && term_coef && cos_v1 && sin_v1 && cos4v1 && sin4v1 && cos_v2 && sin_v2
                    && cos4v2 && sin4v2 && row_on && col_on && acc && axis_v1);

    SvaDensity den[2];
    for (int g = 0; g < n_den; g++) {
        const double *rc = g == 0 ? q_r_centers : q2_r_centers;
        int nr = g == 0 ? q_nr : q2_nr;
        den[g].grid = interp4d_grid(rc, q_phi_centers, nr, q_nphi, rc[0], rc[nr-1],
            log(rc[0]), 1./(log(rc[1])-log(rc[0])),
            q_phi_centers[0], 1./(q_phi_centers[1]-q_phi_centers[0]));
        den[g].q = g == 0 ? q_array : q2_array;
        den[g].coef = n_den == 1 ? 1. : (g == 0 ? 2. : -1.);
        den[g].slice = (double *)orpheus_malloc((size_t)den[g].grid.s_i2*nr*sizeof(double));
        den[g].axis_t1 = (Interp4dAxis *)orpheus_malloc((size_t)n1*sizeof(Interp4dAxis));
        if (!(den[g].slice && den[g].axis_t1)) alloc_ok = 0;
    }

    int n_av = 0, n_bv = 0, n_term = 0;
    if (alloc_ok) {
        for (int k = 0; k < n_cb; k++) {
            int w = (w_ids[k] >= 0 && w_ids[k] <= 4) ? w_ids[k] : 0;
            for (int j = 0; j < SVA_NTERM[w]; j++) {
                term_a[n_term] = sva_vector(xi_a_idx[k], SVA_FAC_A[w][j], av_tab, av_fac, &n_av);
                term_b[n_term] = sva_vector(xi_b_idx[k], SVA_FAC_B[w][j], bv_tab, bv_fac, &n_bv);
                term_out[n_term] = out_idx[k];
                term_coef[n_term++] = wick_fac[k]*SVA_COEF[w][j];
            }
        }
        for (int j = 0; j < nv1; j++) {
            cos_v1[j] = cos(v1_grid[j]); sin_v1[j] = sin(v1_grid[j]);
            cos4v1[j] = cos(4.*v1_grid[j]); sin4v1[j] = sin(4.*v1_grid[j]);
            if (n_den) axis_v1[j] = interp4d_axis_phi(&den[0].grid, v1_grid[j]);
        }
        for (int j = 0; j < nv2; j++) {
            cos_v2[j] = cos(v2_grid[j]); sin_v2[j] = sin(v2_grid[j]);
            cos4v2[j] = cos(4.*v2_grid[j]); sin4v2[j] = sin(4.*v2_grid[j]);
        }
        for (int g = 0; g < n_den; g++)
            for (int j = 0; j < n1; j++)
                den[g].axis_t1[j] = interp4d_axis_r(&den[g].grid, phi1_nodes[j]);
        for (int i = 0; i < n_r; i++)
            for (int j = 0; j < n_r; j++)
                if (pair_mask[i*n_r+j]) { row_on[i] = 1; col_on[j] = 1; }
    }
    int nvec = n_av > 0 ? n_av : 1, nbvec = n_bv > 0 ? n_bv : 1;
    // Side a of every theta1 node (n1, n_av, na_k), shared by the team
    double *A_all = alloc_ok ? (double *)orpheus_malloc(
        (size_t)n1*nvec*na_k*sizeof(double)) : NULL;
    if (!A_all) alloc_ok = 0;

    reset_progress();

    #pragma omp parallel num_threads(nthreads)
    {
        double fac[SVA_NFAC], r, psi;
        double *xi = (double *)orpheus_malloc((size_t)n_xi_unique*sizeof(double));
        double *val = (double *)orpheus_malloc((size_t)nbvec*sizeof(double));
        double *Bs = (double *)orpheus_malloc((size_t)nbvec*sizeof(double));
        double *Yc = (double *)orpheus_malloc((size_t)nbvec*na_k*sizeof(double));
        double *qk = (double *)orpheus_malloc((size_t)na_k*sizeof(double));
        int ok = (xi && val && Bs && Yc && qk);
        SvaScratch scr[2] = {{0}, {0}};
        for (int g = 0; g < n_den; g++) {
            int nr = den[g].grid.nr, ncell = nr*q_nphi;
            scr[g].cellmap = (int *)orpheus_malloc((size_t)ncell*sizeof(int));
            scr[g].cells = (int *)orpheus_malloc((size_t)ncell*sizeof(int));
            scr[g].need = (char *)orpheus_malloc((size_t)nr);
            scr[g].B = (double *)orpheus_malloc((size_t)ncell*nbvec*sizeof(double));
            scr[g].Y = (double *)orpheus_malloc((size_t)ncell*nbvec*sizeof(double));
            if (!(scr[g].cellmap && scr[g].cells && scr[g].need && scr[g].B && scr[g].Y)) ok = 0;
            else for (int c = 0; c < ncell; c++) scr[g].cellmap[c] = -1;
        }
        if (!ok) {
            #pragma omp atomic write
            alloc_ok = 0;
        }
        #pragma omp barrier

        // Every thread runs the phi loop and shares the work of each node
        int n_phi_run = alloc_ok ? n_phi : 0, n_r_run = alloc_ok ? n_r : 0;
        for (int kphi = 0; kphi < n_phi_run; kphi++) {
            double phi = phi_grid[kphi];
            double common = phi*phi_w[kphi];

            // Density slices at the first spoke t1 = phi: zero above the q grid, nearest
            // neighbour below it and linear in between
            int active[2] = {0, 0};
            for (int g = 0; g < n_den; g++) {
                const Interp4dGrid *gr = &den[g].grid;
                long slice_len = gr->s_i2*gr->nr;
                active[g] = phi < gr->r_max;
                if (!active[g]) continue;
                int i = 0;
                double w = 0.;
                if (phi > gr->r_min) {
                    while (i < gr->nr && gr->r_centers[i] < phi) i++;
                    i--;
                    if (i > gr->nr-2) i = gr->nr-2;
                    w = (phi-gr->r_centers[i])/(gr->r_centers[i+1]-gr->r_centers[i]);
                }
                const double *s0 = den[g].q + (long)i*slice_len;
                const double *s1 = w > 0. ? s0 + slice_len : s0;
                double *slice = den[g].slice;
                #pragma omp for schedule(static)
                for (long j = 0; j < slice_len; j++)
                    slice[j] = (1.-w)*s0[j] + w*s1[j];
            }
            double s1phi = q_mode == 2 ? sup_t1[kphi] : 1.;

            // Side a: psi_a = phi - theta1 e(V1), with the node and the V1 weights
            #pragma omp for schedule(static)
            for (int ia = 0; ia < n1; ia++) {
                if (!row_on[ia/ns1]) continue;
                double *A = A_all + (long)ia*nvec*na_k;
                memset(A, 0, (size_t)nvec*na_k*sizeof(double));
                double P = phi1_nodes[ia];
                for (int iv = 0; iv < nv1; iv++) {
                    sva_side(phi - P*cos_v1[iv], -P*sin_v1[iv], cos4v1[iv], sin4v1[iv],
                             xi_x, xi_y_tables, xi_n, n_xi_unique, xi_logx0, xi_inv_dlogx,
                             &r, &psi, fac, xi);
                    double wq = phi1_weights[ia]*v1_w[iv];
                    int k0 = 0, k1 = 0;
                    double w0 = 1., w1 = 0.;
                    if (n_den) {
                        k0 = axis_v1[iv].i; k1 = axis_v1[iv].ip;
                        w0 = axis_v1[iv].mw; w1 = axis_v1[iv].w;
                    }
                    for (int a = 0; a < n_av; a++) {
                        double v = wq*fac[av_fac[a]]*xi[av_tab[a]];
                        A[a*na_k + k0] += w0*v;
                        A[a*na_k + k1] += w1*v;
                    }
                }
            }

            #pragma omp for schedule(dynamic, 1)
            for (int b2 = 0; b2 < n_r; b2++) {
                if (!col_on[b2]) continue;

                // Side b: psi_b = phi + theta2 e(V2), projected onto the cells of each density
                for (int b = 0; b < n_bv; b++) Bs[b] = 0.;
                for (int ib = b2*ns2; ib < (b2+1)*ns2; ib++) {
                    double P = phi2_nodes[ib];
                    for (int iv = 0; iv < nv2; iv++) {
                        sva_side(phi + P*cos_v2[iv], P*sin_v2[iv], cos4v2[iv], sin4v2[iv],
                                 xi_x, xi_y_tables, xi_n, n_xi_unique, xi_logx0, xi_inv_dlogx,
                                 &r, &psi, fac, xi);
                        double wq = phi2_weights[ib]*v2_w[iv];
                        if (q_mode == 2) wq *= sup_t3[ib];
                        for (int b = 0; b < n_bv; b++)
                            val[b] = wq*fac[bv_fac[b]]*xi[bv_tab[b]];
                        for (int b = 0; b < n_bv && !n_den; b++) Bs[b] += val[b];
                        for (int g = 0; g < n_den; g++) {
                            if (!active[g]) continue;
                            Interp4dAxis x3 = interp4d_axis_r(&den[g].grid, r);
                            Interp4dAxis x13 = interp4d_axis_phi(&den[g].grid, psi);
                            sva_add_corner(&scr[g], x3.i*q_nphi + x13.i, x3.mw*x13.mw, val, n_bv);
                            sva_add_corner(&scr[g], x3.i*q_nphi + x13.ip, x3.mw*x13.w, val, n_bv);
                            sva_add_corner(&scr[g], x3.ip*q_nphi + x13.i, x3.w*x13.mw, val, n_bv);
                            sva_add_corner(&scr[g], x3.ip*q_nphi + x13.ip, x3.w*x13.w, val, n_bv);
                        }
                    }
                }

                // Y = q B on the radial nodes of the theta1 bins paired with this theta2 bin
                for (int g = 0; g < n_den; g++) {
                    if (!active[g]) continue;
                    const Interp4dGrid *gr = &den[g].grid;
                    SvaScratch *s = &scr[g];
                    memset(s->need, 0, (size_t)gr->nr);
                    for (int ia = 0; ia < n1; ia++) {
                        if (!pair_mask[(ia/ns1)*n_r + b2]) continue;
                        Interp4dAxis x2 = den[g].axis_t1[ia];
                        if (x2.mw != 0.) s->need[x2.i] = 1;
                        if (x2.w != 0.) s->need[x2.ip] = 1;
                    }
                    for (int i2 = 0; i2 < gr->nr; i2++) {
                        if (!s->need[i2]) continue;
                        double *Y = s->Y + (long)i2*n_bv*q_nphi;
                        memset(Y, 0, (size_t)n_bv*q_nphi*sizeof(double));
                        for (int c = 0; c < s->ncell; c++) {
                            const double *qc = den[g].slice + (long)i2*gr->s_i2
                                + (long)(s->cells[c]/q_nphi)*gr->s_i3 + s->cells[c]%q_nphi;
                            const double *B = s->B + (long)c*n_bv;
                            for (int k = 0; k < q_nphi; k++) qk[k] = qc[(long)k*q_nphi];
                            for (int b = 0; b < n_bv; b++) {
                                double *Yb = Y + (long)b*q_nphi, Bb = B[b];
                                #pragma omp simd
                                for (int k = 0; k < q_nphi; k++) Yb[k] += Bb*qk[k];
                            }
                        }
                    }
                    for (int c = 0; c < s->ncell; c++) s->cellmap[s->cells[c]] = -1;
                    s->ncell = 0;
                }

                // Elements of this theta2 bin: the terms a^T Yc of every theta1 node, with Yc
                // the densities at its radius times the phi weight
                for (int ia = 0; ia < n1; ia++) {
                    int b1 = ia/ns1;
                    if (!pair_mask[b1*n_r + b2]) continue;
                    if (n_den) {
                        memset(Yc, 0, (size_t)n_bv*na_k*sizeof(double));
                        for (int g = 0; g < n_den; g++) {
                            if (!active[g]) continue;
                            Interp4dAxis x2 = den[g].axis_t1[ia];
                            int i2s[2] = {x2.i, x2.ip};
                            double ws[2] = {x2.mw, x2.w};
                            for (int e = 0; e < 2; e++) {
                                if (ws[e] == 0.) continue;
                                double f = common*den[g].coef*ws[e];
                                const double *Y = scr[g].Y + (long)i2s[e]*n_bv*q_nphi;
                                #pragma omp simd
                                for (int j = 0; j < n_bv*q_nphi; j++) Yc[j] += f*Y[j];
                            }
                        }
                    } else {
                        double f = common*q_const*(q_mode == 2 ? s1phi*sup_t2[ia] : 1.);
                        for (int b = 0; b < n_bv; b++) Yc[b] = f*Bs[b];
                    }
                    const double *A = A_all + (long)ia*nvec*na_k;
                    double *dst = acc + ((long)b2*n_r + b1)*n_out;
                    for (int t = 0; t < n_term; t++) {
                        const double *a = A + (long)term_a[t]*na_k;
                        const double *y = Yc + (long)term_b[t]*na_k;
                        double s = 0.;
                        #pragma omp simd reduction(+:s)
                        for (int k = 0; k < na_k; k++) s += a[k]*y[k];
                        dst[term_out[t]] += term_coef[t]*s;
                    }
                }
            }

            #pragma omp single
            print_progress(kphi+1, n_phi, verbose);
        }

        // Symmetrise and reorder to (output, theta1, theta2)
        #pragma omp for schedule(static)
        for (int j = 0; j < n_r_run; j++)
            for (int i = 0; i < n_r; i++)
                for (int o = 0; o < n_out; o++) {
                    double v = acc[((long)j*n_r + i)*n_out + o];
                    if (symmetric_mask[o])
                        v = .5*(v + acc[((long)i*n_r + j)*n_out + o]);
                    results[((long)o*n_r + i)*n_r + j] += v;
                }

        free(xi); free(val); free(Bs); free(Yc); free(qk);
        for (int g = 0; g < n_den; g++) {
            free(scr[g].cellmap); free(scr[g].cells); free(scr[g].need);
            free(scr[g].B); free(scr[g].Y);
        }
    }
    if (verbose > 0) { printf("\n"); fflush(stdout); }

    for (int g = 0; g < n_den; g++) { free(den[g].slice); free(den[g].axis_t1); }
    free(av_tab); free(av_fac); free(bv_tab); free(bv_fac);
    free(term_a); free(term_b); free(term_out); free(term_coef);
    free(cos_v1); free(sin_v1); free(cos4v1); free(sin4v1);
    free(cos_v2); free(sin_v2); free(cos4v2); free(sin4v2);
    free(row_on); free(col_on); free(acc); free(axis_v1); free(A_all);
}


/////////////////////////////////
// Continuous-limit estimators //
/////////////////////////////////

// Continuous q-term (triplet) of one xi on a fine (alpha*n_r) grid, bin-averaged to (n_r, n_r).
// w_type: 0 = qpp (w = 1), 1 = qmm (cos(4*PHI)), 2 = qpm (cos(4*(PHI-phi_jk)))
void continuous_q_single(
    int n_r, int alpha, const double *fine_centers, const double *fine_weights,
    const double *phi_centers, const double *dphi, int n_fine, int n_phi,
    const double *xi_x, int xi_n, double xi_logx0, double xi_inv_dlogx, const double *xi_y,
    int w_type, double sigma_eps2, double area, double nbar,
    const double *Np, double *out){

    double prefactor = 2.*sigma_eps2/(M_PI*area*nbar);

    // Do fine-grid integral
    double *fine = (double *)orpheus_calloc((long)n_fine*n_fine, sizeof(double));
    if (!fine) return;

    for (int i1 = 0; i1 < n_fine; i1++) {
        double T1 = fine_centers[i1];
        for (int i2 = 0; i2 < n_fine; i2++) {
            double T2 = fine_centers[i2];
            double sum = 0.;
            for (int k = 0; k < n_phi; k++) {
                double PHI = phi_centers[k];
                double jk_x = T1 - T2*cos(PHI);
                double jk_y = -T2*sin(PHI);
                double r_jk = hypot(jk_x, jk_y);
                double xi_val = interp1d_eval(xi_x, xi_y, xi_n,
                    xi_logx0, xi_inv_dlogx, r_jk);
                double w;
                switch (w_type) {
                case 0: w = 1.; break;
                case 1: w = cos(4.*PHI); break;
                case 2: w = cos(4.*(PHI-atan2(jk_y, jk_x))); break;
                default: w = 0.; break;
                }
                sum += w*xi_val*dphi[k];
            }
            fine[(long)i1*n_fine + i2] = sum;
        }
    }

    // Apply bin averaging
    for (int ic = 0; ic < n_r; ic++) {
        for (int jc = 0; jc < n_r; jc++) {
            double num = 0., den = 0.;
            for (int ia = 0; ia < alpha; ia++) {
                int i1 = ic*alpha + ia;
                double w1 = fine_weights[i1];
                for (int ja = 0; ja < alpha; ja++) {
                    int i2 = jc*alpha + ja;
                    double w2 = fine_weights[i2];
                    double ww = w1*w2;
                    num += fine[(long)i1*n_fine + i2]*ww;
                    den += ww;
                }
            }
            out[(long)ic*n_r + jc] += prefactor*num/den;
        }
    }

    free(fine);
}


// Build q-terms for all tomo combos, in continuous approximation
void continuous_q_tomo(
    int n_r, int alpha, const double *fine_centers, const double *fine_weights,
    const double *phi_centers, const double *dphi, int n_fine, int n_phi,
    const double *xi_x, int xi_n, double xi_logx0, double xi_inv_dlogx,
    const double *xi_y_tables, int n_xi_unique,
    double sigma_eps2, double area, double nbar, const double *Np,
    int n_calls, const int *call_xi_idx, const int *call_w_type,
    const int *call_out_idx, const double *call_factor,
    int n_out, double *results, int nthreads, int verbose){

    long block_size = (long)n_r*n_r;
    reset_progress();
    int ndone = 0;

    #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for (int k = 0; k < n_calls; k++) {
        const double *xi_y = xi_y_tables + (long)call_xi_idx[k]*xi_n;
        double *block = (double *)orpheus_calloc(block_size, sizeof(double));
        if (!block) continue;

        continuous_q_single(n_r, alpha, fine_centers, fine_weights,
            phi_centers, dphi, n_fine, n_phi,
            xi_x, xi_n, xi_logx0, xi_inv_dlogx, xi_y, call_w_type[k],
            sigma_eps2, area, nbar, Np, block);

        // Accumulate into the output (needs critical section)
        double fac = call_factor[k];
        double *dst = results + (long)call_out_idx[k]*block_size;
        #pragma omp critical
        {
            for (long j = 0; j < block_size; j++)
                dst[j] += fac*block[j];
        }
        free(block);
        #pragma omp atomic
        ndone += 1;
        print_progress(ndone, n_calls, verbose);
    }
    if (verbose > 0) { printf("\n"); fflush(stdout); }
}


// Build r-terms for all tomo combos, in continuous approximation. The side integrals are
// shared across work units, the outer products parallelise over (combo, Wick pairing).
void continuous_rterms_tomo(
    int n_r, const double *theta_centers, const double *phi_grid, const double *phi_w, int n_phi,
    const double *v_grid, const double *v_w, int n_v,
    const double *v_half, const double *v_half_w, int n_vh, double area,
    const double *xi_x, int xi_n, double xi_logx0, double xi_inv_dlogx,
    const double *xi_y_tables, int n_xi_unique,
    int n_work, const int *work_xi_a_p, const int *work_xi_b_p,
    const int *work_xi_a_m, const int *work_xi_b_m, const int *work_out_idx,
    int n_out, double *results, int nthreads, int verbose){

    long block_size = (long)n_r*n_r;
    double inv_2piA = 1./(2.*M_PI*area);
    double inv_piA_times2 = 2./(M_PI*area);

    // Distinct (xi_p, xi_m) table pairs per side, so that each side integral is evaluated once
    int *a_p = (int *)orpheus_malloc((size_t)n_work*sizeof(int));
    int *a_m = (int *)orpheus_malloc((size_t)n_work*sizeof(int));
    int *b_p = (int *)orpheus_malloc((size_t)n_work*sizeof(int));
    int *b_m = (int *)orpheus_malloc((size_t)n_work*sizeof(int));
    int *a_of = (int *)orpheus_malloc((size_t)n_work*sizeof(int));
    int *b_of = (int *)orpheus_malloc((size_t)n_work*sizeof(int));
    int n_a = 0, n_b = 0;
    for (int wk = 0; wk < n_work; wk++) {
        int p = work_xi_a_p[wk], m = work_xi_a_m[wk], f = -1;
        for (int u = 0; u < n_a; u++)
            if (a_p[u] == p && a_m[u] == m) { f = u; break; }
        if (f < 0) { f = n_a; a_p[n_a] = p; a_m[n_a] = m; n_a++; }
        a_of[wk] = f;

        p = work_xi_b_p[wk]; m = work_xi_b_m[wk]; f = -1;
        for (int u = 0; u < n_b; u++)
            if (b_p[u] == p && b_m[u] == m) { f = u; break; }
        if (f < 0) { f = n_b; b_p[n_b] = p; b_m[n_b] = m; n_b++; }
        b_of[wk] = f;
    }

    // Side integrals over (phi, distinct table pair), stored as I[(kp*n_side + u)*n_r + ir]
    long a_stride = (long)n_a*n_r, b_stride = (long)n_b*n_r;
    long a_len = (long)n_phi*a_stride, b_len = (long)n_phi*b_stride;
    double *Ia_rp0 = (double *)orpheus_malloc(a_len*sizeof(double));
    double *Ia_rp1_c = (double *)orpheus_malloc(a_len*sizeof(double));
    double *Ia_rp1_s = (double *)orpheus_malloc(a_len*sizeof(double));
    double *Ia_rm0 = (double *)orpheus_malloc(a_len*sizeof(double));
    double *Ia_rm1 = (double *)orpheus_malloc(a_len*sizeof(double));
    double *Ia_rpm = (double *)orpheus_malloc(a_len*sizeof(double));
    double *Ib_rp0 = (double *)orpheus_malloc(b_len*sizeof(double));
    double *Ib_rp1_c = (double *)orpheus_malloc(b_len*sizeof(double));
    double *Ib_rp1_s = (double *)orpheus_malloc(b_len*sizeof(double));
    double *Ib_rm0 = (double *)orpheus_malloc(b_len*sizeof(double));
    double *Ib_rm1 = (double *)orpheus_malloc(b_len*sizeof(double));
    double *Ib_rpm = (double *)orpheus_malloc(b_len*sizeof(double));

    // Bail out if any of the side-integral tables failed to allocate
    if (!(a_p && a_m && b_p && b_m && a_of && b_of
          && Ia_rp0 && Ia_rp1_c && Ia_rp1_s && Ia_rm0 && Ia_rm1 && Ia_rpm
          && Ib_rp0 && Ib_rp1_c && Ib_rp1_s && Ib_rm0 && Ib_rm1 && Ib_rpm)) {
        free(a_p); free(a_m); free(b_p); free(b_m); free(a_of); free(b_of);
        free(Ia_rp0); free(Ia_rp1_c); free(Ia_rp1_s);
        free(Ia_rm0); free(Ia_rm1); free(Ia_rpm);
        free(Ib_rp0); free(Ib_rp1_c); free(Ib_rp1_s);
        free(Ib_rm0); free(Ib_rm1); free(Ib_rpm);
        return;
    }

    if (verbose > 0) printf("  side integrals ");
    reset_progress();
    int nsidedone = 0;
    int nsidetot = n_phi*(n_a+n_b);

    #pragma omp parallel for schedule(dynamic) collapse(2) num_threads(nthreads)
    for (int kp = 0; kp < n_phi; kp++) {
        for (int u = 0; u < n_a; u++) {
            double phi = phi_grid[kp];
            const double *xip = xi_y_tables + (long)a_p[u]*xi_n;
            const double *xim = xi_y_tables + (long)a_m[u]*xi_n;
            long off = ((long)kp*n_a + u)*n_r;

            for (int ir = 0; ir < n_r; ir++) {
                double T = theta_centers[ir];

                // rp0: V in [0, pi]
                double s_rp0 = 0.;
                for (int iv = 0; iv < n_vh; iv++) {
                    double abs_a = sqrt(phi*phi + T*T
                        - 2.*phi*T*cos(v_half[iv]));
                    s_rp0 += interp1d_eval(xi_x, xip, xi_n, xi_logx0,
                        xi_inv_dlogx, abs_a)*v_half_w[iv];
                }
                Ia_rp0[off+ir] = s_rp0;

                // rp1, rm0, rm1, rpm: V in [0, 2*pi]
                double s_c = 0., s_s = 0., s_rm0 = 0., s_rm1 = 0., s_rpm = 0.;
                for (int iv = 0; iv < n_v; iv++) {
                    double V = v_grid[iv];
                    double pax = phi - T*cos(V), pay = -T*sin(V);
                    double abs_a = hypot(pax, pay);
                    double va = atan2(pay, pax);
                    double xim_val = interp1d_eval(xi_x, xim, xi_n,
                        xi_logx0, xi_inv_dlogx, abs_a);
                    double xip_val = interp1d_eval(xi_x, xip, xi_n,
                        xi_logx0, xi_inv_dlogx, abs_a);
                    s_c += xim_val*cos(4.*va)*v_w[iv];
                    s_s += xim_val*sin(4.*va)*v_w[iv];
                    s_rm0 += xim_val*cos(4.*(V-va))*v_w[iv];
                    s_rm1 += xip_val*cos(4.*V)*v_w[iv];
                    s_rpm += xip_val*v_w[iv];
                }
                Ia_rp1_c[off+ir] = s_c;
                Ia_rp1_s[off+ir] = s_s;
                Ia_rm0[off+ir] = s_rm0;
                Ia_rm1[off+ir] = s_rm1;
                Ia_rpm[off+ir] = s_rpm;
            }
            #pragma omp atomic
            nsidedone += 1;
            print_progress(nsidedone, nsidetot, verbose);
        }
    }

    #pragma omp parallel for schedule(dynamic) collapse(2) num_threads(nthreads)
    for (int kp = 0; kp < n_phi; kp++) {
        for (int u = 0; u < n_b; u++) {
            double phi = phi_grid[kp];
            const double *xip = xi_y_tables + (long)b_p[u]*xi_n;
            const double *xim = xi_y_tables + (long)b_m[u]*xi_n;
            long off = ((long)kp*n_b + u)*n_r;

            for (int ir = 0; ir < n_r; ir++) {
                double T = theta_centers[ir];

                double s_rp0 = 0.;
                for (int iv = 0; iv < n_vh; iv++) {
                    double abs_b = sqrt(phi*phi + T*T
                        + 2.*phi*T*cos(v_half[iv]));
                    s_rp0 += interp1d_eval(xi_x, xip, xi_n, xi_logx0,
                        xi_inv_dlogx, abs_b)*v_half_w[iv];
                }
                Ib_rp0[off+ir] = s_rp0;

                double s_c = 0., s_s = 0., s_rm0 = 0., s_rm1 = 0., s_rpm = 0.;
                for (int iv = 0; iv < n_v; iv++) {
                    double V = v_grid[iv];
                    double pbx = phi + T*cos(V), pby = T*sin(V);
                    double abs_b = hypot(pbx, pby);
                    double vb = atan2(pby, pbx);
                    double xim_val = interp1d_eval(xi_x, xim, xi_n,
                        xi_logx0, xi_inv_dlogx, abs_b);
                    double xip_val = interp1d_eval(xi_x, xip, xi_n,
                        xi_logx0, xi_inv_dlogx, abs_b);
                    s_c += xim_val*cos(4.*vb)*v_w[iv];
                    s_s += xim_val*sin(4.*vb)*v_w[iv];
                    s_rm0 += xim_val*cos(4.*(V-vb))*v_w[iv];
                    s_rm1 += xip_val*cos(4.*V)*v_w[iv];
                    s_rpm += 2.*xim_val*cos(4.*(V-vb))*v_w[iv];
                }
                Ib_rp1_c[off+ir] = s_c;
                Ib_rp1_s[off+ir] = s_s;
                Ib_rm0[off+ir] = s_rm0;
                Ib_rm1[off+ir] = s_rm1;
                Ib_rpm[off+ir] = s_rpm;
            }
            #pragma omp atomic
            nsidedone += 1;
            print_progress(nsidedone, nsidetot, verbose);
        }
    }
    if (verbose > 0) { printf("\n  outer products "); fflush(stdout); }
    reset_progress();
    int nworkdone = 0;

    // Outer products
    #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for (int wk = 0; wk < n_work; wk++) {
        double *rp0 = (double *)orpheus_calloc(block_size, sizeof(double));
        double *rp1 = (double *)orpheus_calloc(block_size, sizeof(double));
        double *rm0 = (double *)orpheus_calloc(block_size, sizeof(double));
        double *rm1 = (double *)orpheus_calloc(block_size, sizeof(double));
        double *rpm = (double *)orpheus_calloc(block_size, sizeof(double));
        if (!(rp0 && rp1 && rm0 && rm1 && rpm)) {
            free(rp0); free(rp1); free(rm0); free(rm1); free(rpm);
            continue;
        }

        for (int kp = 0; kp < n_phi; kp++) {
            double pwphi = phi_w[kp]*phi_grid[kp];
            long ao = ((long)kp*n_a + a_of[wk])*n_r;
            long bo = ((long)kp*n_b + b_of[wk])*n_r;

            for (int i1 = 0; i1 < n_r; i1++) {
                for (int i2 = 0; i2 < n_r; i2++) {
                    long idx = (long)i1*n_r + i2;
                    rp0[idx] += pwphi*Ia_rp0[ao+i1]*Ib_rp0[bo+i2];
                    rp1[idx] += pwphi*(Ia_rp1_c[ao+i1]*Ib_rp1_c[bo+i2]
                        + Ia_rp1_s[ao+i1]*Ib_rp1_s[bo+i2]);
                    rm0[idx] += pwphi*Ia_rm0[ao+i1]*Ib_rm0[bo+i2];
                    rm1[idx] += pwphi*Ia_rm1[ao+i1]*Ib_rm1[bo+i2];
                    rpm[idx] += pwphi*Ia_rpm[ao+i1]*Ib_rpm[bo+i2];
                }
            }
        }

        // Apply prefactors and accumulate into output
        int base = work_out_idx[wk];
        #pragma omp critical
        {
            double *dst;
            dst = results + (long)(base+0)*block_size;
            for (long j = 0; j < block_size; j++) dst[j] += inv_piA_times2*rp0[j];
            dst = results + (long)(base+1)*block_size;
            for (long j = 0; j < block_size; j++) dst[j] += inv_2piA*rp1[j];
            dst = results + (long)(base+2)*block_size;
            for (long j = 0; j < block_size; j++) dst[j] += inv_2piA*rm0[j];
            dst = results + (long)(base+3)*block_size;
            for (long j = 0; j < block_size; j++) dst[j] += inv_2piA*rm1[j];
            dst = results + (long)(base+4)*block_size;
            for (long j = 0; j < block_size; j++) dst[j] += inv_2piA*rpm[j];
        }

        free(rp0); free(rp1); free(rm0); free(rm1); free(rpm);
        #pragma omp atomic
        nworkdone += 1;
        print_progress(nworkdone, n_work, verbose);
    }
    if (verbose > 0) { printf("\n"); fflush(stdout); }

    free(a_p); free(a_m); free(b_p); free(b_m); free(a_of); free(b_of);
    free(Ia_rp0); free(Ia_rp1_c); free(Ia_rp1_s);
    free(Ia_rm0); free(Ia_rm1); free(Ia_rpm);
    free(Ib_rp0); free(Ib_rp1_c); free(Ib_rp1_s);
    free(Ib_rm0); free(Ib_rm1); free(Ib_rpm);
}


//////////////////////////////////
// Phi-reduction of mixed terms //
//////////////////////////////////

// Sum of the triplet counts against weight*xi(r_jk) over phi, for one mixed term
void triplet_term_reduce(
    const double *Nt, int nr, int nphi, const double *tc, const double *phic, int weight_type,
    const double *xi_x, int xi_n, double xi_logx0, double xi_inv_dlogx,
    const double *xi_y, double *out, int nthreads, int verbose){

    double *cosphi = (double *)orpheus_malloc((size_t)nphi*sizeof(double));
    double *sinphi = (double *)orpheus_malloc((size_t)nphi*sizeof(double));
    double *cos4phi = (double *)orpheus_malloc((size_t)nphi*sizeof(double));
    if (!(cosphi && sinphi && cos4phi)) {
        free(cosphi); free(sinphi); free(cos4phi);
        return;
    }
    for (int k = 0; k < nphi; k++) {
        cosphi[k] = cos(phic[k]);
        sinphi[k] = sin(phic[k]);
        cos4phi[k] = cos(4.*phic[k]);
    }
    reset_progress();
    int ndone = 0;

    #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for (int i = 0; i < nr; i++) {
        double T1 = tc[i];
        for (int j = 0; j < nr; j++) {
            double T2 = tc[j];
            const double *row = Nt + ((size_t)i*nr + j)*nphi;
            double acc = 0.;
            for (int k = 0; k < nphi; k++) {
                double cphi = cosphi[k];
                double r2 = T1*T1 + T2*T2 - 2.*T1*T2*cphi;
                double rjk = r2 > 0. ? sqrt(r2) : 0.;
                double xi = interp1d_eval(xi_x, xi_y, xi_n,
                    xi_logx0, xi_inv_dlogx, rjk);
                double w;
                if (weight_type == 0) {
                    w = 1.;
                } else if (weight_type == 1) {
                    w = cos4phi[k];
                } else {
                    double jk_x = T1 - T2*cphi;
                    double jk_y = -T2*sinphi[k];
                    double phi_jk = atan2(jk_y, jk_x);
                    w = cos(4.*(phic[k]-phi_jk));
                }
                acc += row[k]*w*xi;
            }
            out[(size_t)i*nr + j] = acc;
        }
        #pragma omp atomic
        ndone += 1;
        print_progress(ndone, nr, verbose);
    }
    if (verbose > 0) { printf("\n"); fflush(stdout); }

    free(cosphi); free(sinphi); free(cos4phi);
}
