// Helpers for computation of tomographic xipm covariance

#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "utils.h"
#include "cov_postq.h"

#define M_PI 3.14159265358979323846

// ln(x) of a positive normal x to 1e-3, from its exponent and a cubic in its mantissa
static inline double log_approx(double x){

    uint64_t bits;
    memcpy(&bits, &x, sizeof(bits));
    int e = (int)((bits >> 52) & 0x7ff) - 1023;
    bits = (bits & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL;
    double t;
    memcpy(&t, &bits, sizeof(t));
    t -= 1.;
    return (e*0.6931471805599453 + 9.25032111e-4
        + t*(0.979753413 + t*(-0.393535802 + t*0.106684733)));
}

// Lower node of xval on a log-uniform grid x and its linear weight w, or -1 outside of the grid
static inline int interp1d_locate(
    const double *x, int n, double logx0, double inv_dlogx, double xval, double *w){

    if (xval<=x[0] || xval>=x[n-1]) return -1;
    // Lower node from the log-uniform spacing, then the exact one by comparison
    int i = (int)((log_approx(xval)-logx0)*inv_dlogx);
    if (i<0) i = 0;
    if (i>n-2) i = n-2;
    while (i<n-2 && x[i+1]<=xval) i++;
    while (i>0 && x[i]>xval) i--;
    *w = (xval-x[i])/(x[i+1]-x[i]);
    return i;
}

// Linear interpolation of y(x) on a log-uniform grid x, zero outside of it
static inline double interp1d_eval(
    const double *x, const double *y, int n,
    double logx0, double inv_dlogx, double xval){

    double w;
    int i = interp1d_locate(x, n, logx0, inv_dlogx, xval, &w);
    if (i<0) return 0.;
    return y[i] + w*(y[i+1]-y[i]);
}

// cos, sin, cos(4 x) and sin(4 x) of the n angles x, as four consecutive rows of one allocation,
// NULL if it fails
static double *trig_tables(const double *x, int n){

    double *tab = (double *)orpheus_malloc((size_t)4*n*sizeof(double));
    if (!tab) return NULL;
    for (int i=0; i<n; i++){
        tab[i] = cos(x[i]);
        tab[n+i] = sin(x[i]);
        tab[2*n+i] = cos(4.*x[i]);
        tab[3*n+i] = sin(4.*x[i]);
    }
    return tab;
}

// Real Fourier basis of the angle v with cos c1 and sin s1, m = 2 n + 1 entries, by recurrence, in
// the blocked order 1, cos(v), ..., cos(n v), sin(v), ..., sin(n v)
static inline void fourier_row(double c1, double s1, int m, double *row){
    double c = 1., s = 0.;
    row[0] = 1.;
    for (int k=1; 2*k<m; k++){
        double cn = c*c1 - s*s1;
        s = s*c1 + c*s1;
        c = cn;
        row[k] = c;
        row[m/2+k] = s;
    }
}

// Position of the mode of index i of the basis 1, cos(v), sin(v), ... in its blocked order
static inline int blocked_mode(int i, int m){
    return i%2 ? (i+1)/2 : (i ? m/2+i/2 : 0);
}

//////////////////////////////////////
// Quadruplet density interpolation //
//////////////////////////////////////

// Metadata of the radial interpolation of a density tensor, whose radial axes are log-uniform and
// share `r_centers`
typedef struct {
    const double *r_centers;
    int nr, cubic;
    double r_min, r_max, logr0, inv_dlogr;
} Interp4dGrid;

static inline Interp4dGrid interp4d_grid(const double *r_centers, int nr, int cubic){

    return (Interp4dGrid){r_centers, nr, cubic, r_centers[0], r_centers[nr-1], log(r_centers[0]),
        1./(log(r_centers[1])-log(r_centers[0]))};
}

// A radial axis of the stencil, n consecutive nodes from first, resolved on its own so that
// loop-invariant axes can be hoisted out of the SVA kernel
typedef struct { int first, n; double w[4]; } Interp4dRadial;

// Log-uniform radial axis: zero above the grid, nearest neighbour below it, and in between linear
// in t or Catmull-Rom in log t, which is linear in the first and the last interval
static inline __attribute__((always_inline)) Interp4dRadial interp4d_axis_r(
    const Interp4dGrid *g, double t){

    Interp4dRadial ax = {0, 0, {0., 0., 0., 0.}};
    if (t > g->r_max) return ax;
    if (t < g->r_min){
        ax.n = 2; ax.w[0] = 1.;
        return ax;
    }

    const double *r_centers = g->r_centers;
    int nr = g->nr;
    double log_t = log(t);
    int i = (int)((log_t-g->logr0)*g->inv_dlogr);
    if (i < 0) i = 0;
    else if (i > nr-2) i = nr-2;
    if (i < nr-2 && r_centers[i+1] <= t) i++;
    else if (i > 0 && r_centers[i] > t) i--;

    if (g->cubic && i > 0 && i < nr-2){
        double log_lo = log(r_centers[i]);
        double u = (log_t-log_lo)/(log(r_centers[i+1])-log_lo), u2 = u*u, u3 = u2*u;
        ax.first = i-1; ax.n = 4;
        ax.w[0] = .5*(-u3 + 2.*u2 - u);
        ax.w[1] = .5*(3.*u3 - 5.*u2 + 2.);
        ax.w[2] = .5*(-3.*u3 + 4.*u2 + u);
        ax.w[3] = .5*(u3 - u2);
        return ax;
    }
    double w = (t-r_centers[i])/(r_centers[i+1]-r_centers[i]);
    ax.first = i; ax.n = 2; ax.w[0] = 1.-w; ax.w[1] = w;
    return ax;
}

///////////////////////////////
// Quadruplet (SVA) integral //
///////////////////////////////

// Notation and steps follow the note on the SVA integral, docs/source/notes/sva_collapse.pdf:
// pair a has length theta1 and angle varphi1, pair b length theta2 and angle varphi2, and phi
// links the two pairs along the x axis. xi_a and xi_b are evaluated at the separations psi_a = phi
// - theta1 e(varphi1) and psi_b = phi + theta2 e(varphi2) of polar angles varphi_a and varphi_b
// (Eq. 5), and the density at the spokes (phi, theta1, |psi_b|) of angles (0, varphi1, varphi_b).
// In the names, t1, t2 stand for theta1, theta2 and v1, v2, va, vb for varphi1, varphi2, varphi_a,
// varphi_b.

// The angular weights w_ids 0-4 of rp0, rp1, rm0, rm1 and rpm, 1, cos(4*(va-vb)),
// cos(4*(v1+v2-va-vb)), cos(4*(v1-v2)) and 2*cos(4*(v2-vb)), as sums of sva_nterms (T_X) products
// c_t f^a_t f^b_t of a pair-a and a pair-b factor (step f, Table 3). The factors 0-6 of a pair of
// angle v and separation angle vpsi are 1, cos(4*vpsi), sin(4*vpsi), cos(4*(v-vpsi)),
// sin(4*(v-vpsi)), cos(4*v) and sin(4*v).
#define SVA_NFAC 7
static const int sva_nterms[5] = {1, 2, 2, 2, 1};
static const double sva_coef[5][2] = {{1., 0.}, {1., 1.}, {1., -1.}, {1., 1.}, {2., 0.}};
static const int sva_factor_a[5][2] = {{0, 0}, {1, 2}, {3, 4}, {5, 6}, {0, 0}};
static const int sva_factor_b[5][2] = {{0, 0}, {1, 2}, {3, 4}, {5, 6}, {3, 0}};

// Separations psi = phi + sgn*theta*e(v) of one pair on its n varphi nodes, psi_a for sgn = -1 and
// psi_b for sgn = 1 (Eq. 5): lengths, polar angles vpsi and the cos and sin of 4*vpsi
static void sva_separations(
    double phi, double theta, double sgn, const double *cos_v, const double *sin_v, int n,
    double *abs_psi, double *vpsi, double *cos4vpsi, double *sin4vpsi){

    for (int i=0; i<n; i++){
        double x = phi + sgn*theta*cos_v[i], y = sgn*theta*sin_v[i];
        abs_psi[i] = hypot(x, y);
        vpsi[i] = atan2(y, x);
        cos4vpsi[i] = cos(4.*vpsi[i]);
        sin4vpsi[i] = sin(4.*vpsi[i]);
    }
}

// The SVA_NFAC factors f_t (Table 3) of one pair on its n varphi nodes, rows (SVA_NFAC, n)
static void sva_factor_rows(
    const double *cos4v, const double *sin4v, const double *cos4vpsi, const double *sin4vpsi, int n,
    double *rows){

    for (int i=0; i<n; i++){
        rows[i] = 1.;
        rows[n+i] = cos4vpsi[i];
        rows[2*n+i] = sin4vpsi[i];
        rows[3*n+i] = cos4v[i]*cos4vpsi[i] + sin4v[i]*sin4vpsi[i];
        rows[4*n+i] = sin4v[i]*cos4vpsi[i] - cos4v[i]*sin4vpsi[i];
        rows[5*n+i] = cos4v[i];
        rows[6*n+i] = sin4v[i];
    }
}

// Cosmic-variance (r-term) integral of all callbacks, Eq. 6 (h) without its prefactor 2 pi A/(N_1
// N_2): per (theta1, theta2) node the phi sum of xi_a q xi_b times the angular weight over the
// (varphi1, varphi2) grid, added to results (n_out, n_r, n_r)
void quadruplet_sva_integrate(
    const double *phi_grid, const double *phi_w, int n_phi, int n_r,
    const double *theta1_nodes, const double *theta1_weights, int n_sub1,
    const double *theta2_nodes, const double *theta2_weights, int n_sub2,
    const double *v1_grid, const double *v1_w, int nv1,
    const double *v2_grid, const double *v2_w, int nv2,
    const double *q_array, const double *q_r_centers, int q_nr, int n_modes, int q_cubic,
    double q_const, const double *sup_phi, const double *sup_theta1, const double *sup_theta2,
    const double *xi_x, int xi_n, double xi_logx0, double xi_inv_dlogx,
    const double *xi_y_tables, int n_xi_unique,
    int n_cb, const int *xi_a_idx, const int *xi_b_idx,
    const int *w_ids, const int *out_idx, const double *wick_fac,
    int n_out, const int *symmetric_mask, double *results, int nthreads, int verbose){

    // Sizes of one output block, of the theta1 nodes and of a slice C_phi over (iota, kappa, beta,
    // alpha), the cosine modes, and the most (kappa, beta) knots that a pair-b node can reach
    long block_size = (long)n_r*n_r;
    long results_len = (long)n_out*block_size;
    int n_t1 = n_r*n_sub1;
    int nvmax = nv1 > nv2 ? nv1 : nv2;
    long n_modes2 = (long)n_modes*n_modes;
    int n_cos = n_modes/2 + 1;
    long n_iota_kappa = (long)q_nr*q_nr;
    long slice_len = n_iota_kappa*n_modes2;
    long n_kb_max = (long)q_nr*n_modes;

    // Density q = q_const s(phi) s(theta1) s(theta2) C, q_const = s = 1 for a measured one and
    // C = 1 for the idealised ones, with C = q_array (q_nr, q_nr, q_nr, n_modes, n_modes) the real
    // Fourier coefficients 1, cos(n a), sin(n a) of a = phi12, phi13 on the radial knots
    // q_r_centers (Eq. 4): its radial grid, linear or with q_cubic cubic, and the points N_st of
    // its stencil
    Interp4dGrid grid = interp4d_grid(q_r_centers, q_nr, q_cubic);
    int n_stencil = grid.cubic ? 4 : 2;
    if ((long)n_stencil*nv2 < q_nr) n_kb_max = (long)n_stencil*nv2*n_modes;

    // Trig of the two varphi grids, rows cos, sin, cos(4 v), sin(4 v): shared, read-only,
    // phi-independent.
    double *trig_v1 = trig_tables(v1_grid, nv1), *trig_v2 = trig_tables(v2_grid, nv2);

    // Radial stencils R_iota of the theta1 nodes and the basis Y_alpha of the varphi1 grid,
    // (nv1, n_modes), which do not depend on phi
    Interp4dRadial *axis_t1 = (Interp4dRadial *)orpheus_malloc(
        (size_t)n_t1*sizeof(Interp4dRadial));
    double *basis_v1 = (double *)orpheus_malloc((size_t)nv1*n_modes*sizeof(double));

    // The distinct pairs of xi table and factor of pair a and of pair b over all callbacks, F_a
    // and F_b, in order of appearance: the pair of each (table, factor) key, the table and the
    // factor of each pair, and the pairs of the at most two terms of each callback
    int *fa_of_key = (int *)orpheus_malloc((size_t)n_xi_unique*SVA_NFAC*sizeof(int));
    int *fb_of_key = (int *)orpheus_malloc((size_t)n_xi_unique*SVA_NFAC*sizeof(int));
    int *fa_xi = (int *)orpheus_malloc((size_t)n_xi_unique*SVA_NFAC*sizeof(int));
    int *fa_factor = (int *)orpheus_malloc((size_t)n_xi_unique*SVA_NFAC*sizeof(int));
    int *fb_xi = (int *)orpheus_malloc((size_t)n_xi_unique*SVA_NFAC*sizeof(int));
    int *fb_factor = (int *)orpheus_malloc((size_t)n_xi_unique*SVA_NFAC*sizeof(int));
    int *term_fa = (int *)orpheus_malloc((size_t)2*n_cb*sizeof(int));
    int *term_fb = (int *)orpheus_malloc((size_t)2*n_cb*sizeof(int));
    int *xi_used_a = (int *)orpheus_calloc((size_t)n_xi_unique, sizeof(int));
    int *xi_used_b = (int *)orpheus_calloc((size_t)n_xi_unique, sizeof(int));
    int n_fa = 0, n_fb = 0;

    // Radial stencils of the phi nodes, i.e. the slices of q_array at t1 = phi and their weights,
    // and the theta1 knots iota with weight on any theta1 node, iota_lo to iota_hi
    Interp4dRadial *axis_phi = (Interp4dRadial *)orpheus_calloc(
        (size_t)n_phi, sizeof(Interp4dRadial));
    int iota_lo = q_nr, iota_hi = -1;

    // The per-slot and per-thread arrays below are the largest allocations of the module: nothing
    // is touched before the whole team has allocated, and check_clib_error() raises otherwise
    int alloc_ok = (trig_v1 && trig_v2 && axis_t1 && basis_v1
        && fa_of_key && fb_of_key && fa_xi && fa_factor && fb_xi && fb_factor
        && term_fa && term_fb && xi_used_a && xi_used_b && axis_phi);
    if (alloc_ok){
        for (int key=0; key<n_xi_unique*SVA_NFAC; key++){
            fa_of_key[key] = -1;
            fb_of_key[key] = -1;
        }
        for (int icb=0; icb<n_cb; icb++){
            int wid = w_ids[icb];
            for (int t=0; t<sva_nterms[wid]; t++){
                int key_a = xi_a_idx[icb]*SVA_NFAC + sva_factor_a[wid][t];
                int key_b = xi_b_idx[icb]*SVA_NFAC + sva_factor_b[wid][t];
                if (fa_of_key[key_a] < 0){
                    fa_xi[n_fa] = xi_a_idx[icb]; fa_factor[n_fa] = sva_factor_a[wid][t];
                    fa_of_key[key_a] = n_fa++;
                }
                if (fb_of_key[key_b] < 0){
                    fb_xi[n_fb] = xi_b_idx[icb]; fb_factor[n_fb] = sva_factor_b[wid][t];
                    fb_of_key[key_b] = n_fb++;
                }
                term_fa[2*icb+t] = fa_of_key[key_a];
                term_fb[2*icb+t] = fb_of_key[key_b];
            }
            xi_used_a[xi_a_idx[icb]] = 1;
            xi_used_b[xi_b_idx[icb]] = 1;
        }
        for (int it1=0; it1<n_t1; it1++) axis_t1[it1] = interp4d_axis_r(&grid, theta1_nodes[it1]);
        for (int iv1=0; iv1<nv1; iv1++)
            fourier_row(cos(v1_grid[iv1]), sin(v1_grid[iv1]), n_modes,
                basis_v1 + (long)iv1*n_modes);
        for (int it1=0; it1<n_t1; it1++){
            for (int c=0; c<axis_t1[it1].n; c++){
                if (axis_t1[it1].w[c] == 0.) continue;
                int iota = axis_t1[it1].first+c;
                if (iota < iota_lo) iota_lo = iota;
                if (iota > iota_hi) iota_hi = iota;
            }
        }
        for (int iphi=0; iphi<n_phi; iphi++)
            axis_phi[iphi] = interp4d_axis_r(&grid, phi_grid[iphi]);
    }
    int n_iota = iota_hi - iota_lo + 1;

    // Phi nodes integrated at once, for about four (phi, column) items per thread, and the arrays
    // of each: the slice C_phi, g_hat of all theta1 nodes over (theta1 node, F_a, alpha) and, for
    // more than one, the outputs
    int n_slots = (4*nthreads + n_r - 1)/n_r;
    if (n_slots > nthreads) n_slots = nthreads;
    if (n_slots > n_phi) n_slots = n_phi;
    if (n_slots < 1) n_slots = 1;
    long g_hat_len = (long)n_t1*n_fa*n_modes;
    double *c_slices = (double *)orpheus_malloc((size_t)n_slots*slice_len*sizeof(double));
    double *g_hat = (double *)orpheus_malloc((size_t)n_slots*g_hat_len*sizeof(double));
    double *slot_results = (n_slots > 1
        ? (double *)orpheus_calloc((size_t)n_slots*results_len, sizeof(double)) : NULL);
    alloc_ok = alloc_ok && g_hat && c_slices && (n_slots == 1 || slot_results);
    reset_progress();
    int ndone = 0;

    #pragma omp parallel num_threads(nthreads)
    {
        // Per-thread scratch of one column: the separations, xi and the factors of a pair on its
        // varphi nodes; h_hat over (F_b, kappa beta), P over (F_b, iota, alpha) and P at one
        // theta1 node over (F_b, alpha); per varphi2 node the slots and weights of its knots kappa;
        // the slice row of each (kappa, beta) and the slot of each knot kappa; the basis Y_beta
        // of varphi_b, (nv2, n_modes); and the outputs of the column
        double *abs_psi = (double *)orpheus_malloc((size_t)nvmax*sizeof(double));
        double *vpsi = (double *)orpheus_malloc((size_t)nvmax*sizeof(double));
        double *cos4vpsi = (double *)orpheus_malloc((size_t)nvmax*sizeof(double));
        double *sin4vpsi = (double *)orpheus_malloc((size_t)nvmax*sizeof(double));
        double *xi_nodes = (double *)orpheus_malloc((size_t)n_xi_unique*nvmax*sizeof(double));
        double *factor_rows = (double *)orpheus_malloc((size_t)SVA_NFAC*nvmax*sizeof(double));
        double *h_hat = (double *)orpheus_malloc((size_t)n_fb*n_kb_max*sizeof(double));
        double *p_iota = (double *)orpheus_malloc((size_t)n_fb*q_nr*n_modes*sizeof(double));
        double *p_t1 = (double *)orpheus_malloc((size_t)n_fb*n_modes*sizeof(double));
        int *kappa_slot = (int *)orpheus_malloc((size_t)n_stencil*nv2*sizeof(int));
        double *kappa_w = (double *)orpheus_malloc((size_t)n_stencil*nv2*sizeof(double));
        long *kb_row = (long *)orpheus_malloc((size_t)n_kb_max*sizeof(long));
        int *slot_of_kappa = (int *)orpheus_malloc((size_t)q_nr*sizeof(int));
        double *basis_vb = (double *)orpheus_malloc((size_t)nv2*n_modes*sizeof(double));
        double *col_out = (double *)orpheus_malloc((size_t)n_out*n_r*sizeof(double));

        // A thread that failed to allocate clears alloc_ok for the whole team
        if (!(abs_psi && vpsi && cos4vpsi && sin4vpsi && xi_nodes && factor_rows
            && h_hat && p_iota && p_t1 && kappa_slot && kappa_w && kb_row
            && col_out && slot_of_kappa && basis_vb)){
            #pragma omp atomic write
            alloc_ok = 0;
        }
        if (slot_of_kappa)
            for (int kappa=0; kappa<q_nr; kappa++) slot_of_kappa[kappa] = -1;
        #pragma omp barrier

        // A failed team still enters each `omp for`, whose barrier is collective, with zero trips
        int n_phi_run = alloc_ok ? n_phi : 0;

        for (int phi0=0; phi0<n_phi_run; phi0+=n_slots){
            int n_run = n_phi-phi0 < n_slots ? n_phi-phi0 : n_slots;

            // Slice C_phi (Eq. 7, slice): q_array at t1 = phi, transposed to (iota, kappa, beta,
            // alpha) so that the modes alpha of varphi1 are contiguous, both in the blocked order
            // of fourier_row, in which the cosine and the sine modes each form a contiguous block
            #pragma omp for schedule(static)
            for (long item=0; item<n_run*n_iota_kappa; item++){
                int islot = item/n_iota_kappa, iphi = phi0+islot;
                long iota_kappa = item%n_iota_kappa;
                if (phi_grid[iphi] >= grid.r_max) continue;
                const Interp4dRadial *ax = &axis_phi[iphi];
                const double *src[4];
                for (int c=0; c<ax->n; c++)
                    src[c] = q_array + (ax->first+c)*slice_len + iota_kappa*n_modes2;
                double *dst = c_slices + islot*slice_len + iota_kappa*n_modes2;
                for (int beta=0; beta<n_modes; beta++){
                    for (int alpha=0; alpha<n_modes; alpha++){
                        long alpha_beta = (long)alpha*n_modes + beta;
                        double val = 0.;
                        for (int c=0; c<ax->n; c++) val += ax->w[c]*src[c][alpha_beta];
                        dst[(long)blocked_mode(beta, n_modes)*n_modes
                            + blocked_mode(alpha, n_modes)] = val;
                    }
                }
            }

            // Projections g_hat of pair a (step g, Eq. 7 g_hat): per theta1 node and pair of xi
            // table and factor, g = xi_a f^a_t times the varphi1 weights, summed onto the modes
            // Y_alpha(varphi1)
            #pragma omp for schedule(dynamic, 4)
            for (long item=0; item<(long)n_run*n_t1; item++){
                int islot = item/n_t1, it1 = item%n_t1;
                double phi = phi_grid[phi0+islot];
                if (phi >= grid.r_max) continue;
                sva_separations(phi, theta1_nodes[it1], -1., trig_v1, trig_v1+nv1, nv1,
                    abs_psi, vpsi, cos4vpsi, sin4vpsi);
                for (int itab=0; itab<n_xi_unique; itab++){
                    if (!xi_used_a[itab]) continue;
                    const double *xi_y = xi_y_tables + (long)itab*xi_n;
                    double *dst = xi_nodes + (long)itab*nv1;
                    for (int iv1=0; iv1<nv1; iv1++)
                        dst[iv1] = interp1d_eval(xi_x, xi_y, xi_n,
                            xi_logx0, xi_inv_dlogx, abs_psi[iv1]);
                }
                sva_factor_rows(trig_v1+2*nv1, trig_v1+3*nv1, cos4vpsi, sin4vpsi, nv1,
                    factor_rows);
                double *g_hat_t1 = g_hat + islot*g_hat_len + (long)it1*n_fa*n_modes;
                for (int ifa=0; ifa<n_fa; ifa++){
                    const double *xi_row = xi_nodes + (long)fa_xi[ifa]*nv1;
                    const double *factor_row = factor_rows + (long)fa_factor[ifa]*nv1;
                    double *dst = g_hat_t1 + (long)ifa*n_modes;
                    for (int alpha=0; alpha<n_modes; alpha++) dst[alpha] = 0.;
                    for (int iv1=0; iv1<nv1; iv1++){
                        double g = v1_w[iv1]*xi_row[iv1]*factor_row[iv1];
                        const double *basis = basis_v1 + (long)iv1*n_modes;
                        for (int alpha=0; alpha<n_modes; alpha++) dst[alpha] += g*basis[alpha];
                    }
                }
            }

            // Columns, split among the threads: pair b on the theta2 nodes of column ib2 at phi,
            // contracted with the shared slice and g_hat of all theta1 nodes. A column is
            // accumulated as a row of the transposed results, so that it is contiguous
            #pragma omp for schedule(dynamic)
            for (int item=0; item<n_run*n_r; item++){
                int islot = item/n_r, ib2 = item%n_r, iphi = phi0+islot;
                double phi = phi_grid[iphi];
                if (phi >= grid.r_max || n_iota < 1) continue;
                const double *c_slice = c_slices + islot*slice_len;
                const double *g_hat_phi = g_hat + islot*g_hat_len;

                memset(col_out, 0, (size_t)n_out*n_r*sizeof(double));
                for (int it2=ib2*n_sub2; it2<(ib2+1)*n_sub2; it2++){
                    sva_separations(phi, theta2_nodes[it2], 1., trig_v2, trig_v2+nv2, nv2,
                        abs_psi, vpsi, cos4vpsi, sin4vpsi);
                    for (int itab=0; itab<n_xi_unique; itab++){
                        if (!xi_used_b[itab]) continue;
                        const double *xi_y = xi_y_tables + (long)itab*xi_n;
                        double *dst = xi_nodes + (long)itab*nv2;
                        for (int iv2=0; iv2<nv2; iv2++)
                            dst[iv2] = interp1d_eval(xi_x, xi_y, xi_n,
                                xi_logx0, xi_inv_dlogx, abs_psi[iv2]);
                    }
                    sva_factor_rows(trig_v2+2*nv2, trig_v2+3*nv2, cos4vpsi, sin4vpsi, nv2,
                        factor_rows);

                    // The radial knots kappa that |psi_b| reaches on the varphi2 nodes, numbered
                    // in slots, with the slot and the weight R_kappa(|psi_b|) of each stencil point
                    // of a node, and the basis Y_beta(varphi_b): slot k holds the (kappa, beta)
                    // entries k*n_modes + beta, which lie in the slice rows kappa*n_modes + beta
                    int n_kappa = 0;
                    for (int iv2=0; iv2<nv2; iv2++){
                        Interp4dRadial axis_psib = interp4d_axis_r(&grid, abs_psi[iv2]);
                        int *slots = kappa_slot + n_stencil*iv2;
                        double *weights = kappa_w + n_stencil*iv2;
                        for (int c=0; c<axis_psib.n; c++){
                            int kappa = axis_psib.first+c;
                            double w_kappa = axis_psib.w[c];
                            if (w_kappa != 0. && slot_of_kappa[kappa] < 0){
                                slot_of_kappa[kappa] = n_kappa;
                                long *kb_row_k = kb_row + (long)n_kappa*n_modes;
                                for (int beta=0; beta<n_modes; beta++)
                                    kb_row_k[beta] = (long)kappa*n_modes + beta;
                                n_kappa++;
                            }
                            slots[c] = w_kappa != 0. ? slot_of_kappa[kappa] : 0;
                            weights[c] = w_kappa;
                        }
                        for (int c=axis_psib.n; c<n_stencil; c++){ slots[c] = 0; weights[c] = 0.; }
                        double inv_abs_psi = abs_psi[iv2] > 0. ? 1./abs_psi[iv2] : 0.;
                        double x = phi + theta2_nodes[it2]*trig_v2[iv2];
                        double y = theta2_nodes[it2]*trig_v2[nv2+iv2];
                        fourier_row(inv_abs_psi > 0. ? x*inv_abs_psi : 1., y*inv_abs_psi, n_modes,
                            basis_vb + (long)iv2*n_modes);
                    }
                    for (int k=0; k<n_kappa; k++)
                        slot_of_kappa[kb_row[(long)k*n_modes]/n_modes] = -1;
                    if (n_kappa == 0) continue;
                    int n_kb = n_kappa*n_modes;

                    // Projections h_hat of pair b (step g, Eq. 7 h_hat): per pair of xi table and
                    // factor, h = xi_b f^b_t times the varphi2 weights, summed onto the (kappa,
                    // beta) knots with R_kappa(|psi_b|) Y_beta(varphi_b)
                    for (int ifb=0; ifb<n_fb; ifb++){
                        const double *xi_row = xi_nodes + (long)fb_xi[ifb]*nv2;
                        const double *factor_row = factor_rows + (long)fb_factor[ifb]*nv2;
                        double *dst = h_hat + (long)ifb*n_kb;
                        for (int ikb=0; ikb<n_kb; ikb++) dst[ikb] = 0.;
                        for (int iv2=0; iv2<nv2; iv2++){
                            double h = v2_w[iv2]*xi_row[iv2]*factor_row[iv2];
                            const double *basis = basis_vb + (long)iv2*n_modes;
                            for (int c=0; c<n_stencil; c++){
                                double w_h = kappa_w[n_stencil*iv2+c]*h;
                                if (w_h == 0.) continue;
                                double *dst_kappa = dst + (long)kappa_slot[n_stencil*iv2+c]*n_modes;
                                for (int beta=0; beta<n_modes; beta++)
                                    dst_kappa[beta] += w_h*basis[beta];
                            }
                        }
                    }

                    // Contraction P of h_hat with the slice (step h, Eq. 7 P) on the theta1 knots
                    // iota_lo to iota_hi, over (F_b, iota, alpha). A reflection about the x axis
                    // leaves the r-terms unchanged and flips the sign of the cosine-sine blocks of
                    // C, so only the modes alpha of the parity of beta enter, exactly so on varphi
                    // grids symmetric under v -> -v
                    memset(p_iota, 0, (size_t)n_fb*n_iota*n_modes*sizeof(double));
                    for (int iota=iota_lo; iota<=iota_hi; iota++){
                        const double *c_iota = c_slice + (long)iota*q_nr*n_modes2;
                        for (int ikb=0; ikb<n_kb; ikb++){
                            const double *c_row = c_iota + kb_row[ikb]*n_modes;
                            int is_cos = ikb%n_modes < n_cos;
                            int alpha_lo = is_cos ? 0 : n_cos, alpha_hi = is_cos ? n_cos : n_modes;
                            for (int ifb=0; ifb<n_fb; ifb++){
                                double h_val = h_hat[(long)ifb*n_kb + ikb];
                                double *dst = p_iota + ((long)ifb*n_iota + iota-iota_lo)*n_modes;
                                for (int alpha=alpha_lo; alpha<alpha_hi; alpha++)
                                    dst[alpha] += h_val*c_row[alpha];
                            }
                        }
                    }

                    // Terms (step h, Eq. 7 terms): P at each theta1 node, the sum of R_iota P over
                    // iota, then per callback the sum of c_t g_hat P over its terms t and the modes
                    // alpha, times the radial weights omega_s omega_s' s(theta1) s(theta2)
                    for (int it1=0; it1<n_t1; it1++){
                        int ib1 = it1/n_sub1;
                        const Interp4dRadial *ax = &axis_t1[it1];
                        for (int ifb=0; ifb<n_fb; ifb++){
                            double *dst = p_t1 + (long)ifb*n_modes;
                            for (int alpha=0; alpha<n_modes; alpha++) dst[alpha] = 0.;
                            for (int c=0; c<ax->n; c++){
                                if (ax->w[c] == 0.) continue;
                                const double *src = (p_iota
                                    + ((long)ifb*n_iota + ax->first+c-iota_lo)*n_modes);
                                for (int alpha=0; alpha<n_modes; alpha++)
                                    dst[alpha] += ax->w[c]*src[alpha];
                            }
                        }
                        const double *g_hat_t1 = g_hat_phi + (long)it1*n_fa*n_modes;
                        double w_nodes = (theta1_weights[it1]*theta2_weights[it2]
                            *(sup_theta1[it1]*sup_theta2[it2]));
                        for (int icb=0; icb<n_cb; icb++){
                            int wid = w_ids[icb];
                            double sum = 0.;
                            for (int t=0; t<sva_nterms[wid]; t++){
                                const double *g_row = g_hat_t1 + (long)term_fa[2*icb+t]*n_modes;
                                const double *p_row = p_t1 + (long)term_fb[2*icb+t]*n_modes;
                                double dot = 0.;
                                for (int alpha=0; alpha<n_modes; alpha++)
                                    dot += g_row[alpha]*p_row[alpha];
                                sum += sva_coef[wid][t]*dot;
                            }
                            col_out[(long)out_idx[icb]*n_r + ib1] += wick_fac[icb]*sum*w_nodes;
                        }
                    }
                }

                // Phi quadrature (step e): add the column times w_p phi_p q_const s(phi) to row ib2
                // of the transposed results, or of the slot of phi. The product is taken apart from
                // the sum, which FMA contraction would otherwise round differently for the two.
                double phi_weight = phi*phi_w[iphi]*(q_const*sup_phi[iphi]);
                double *dst = n_slots > 1 ? slot_results + islot*results_len : results;
                for (long j=0; j<(long)n_out*n_r; j++) col_out[j] *= phi_weight;
                for (int oi=0; oi<n_out; oi++){
                    double *dst_row = dst + oi*block_size + (long)ib2*n_r;
                    const double *src_row = col_out + (long)oi*n_r;
                    for (int ib1=0; ib1<n_r; ib1++) dst_row[ib1] += src_row[ib1];
                }
            }

            // Add the slots to the results in phi order, so that every element is summed in phi
            // order whatever the number of threads
            if (n_slots > 1){
                #pragma omp for schedule(static)
                for (long j=0; j<results_len; j++){
                    for (int islot=0; islot<n_run; islot++){
                        results[j] += slot_results[islot*results_len + j];
                        slot_results[islot*results_len + j] = 0.;
                    }
                }
            }
            #pragma omp single nowait
            {
                ndone += n_run;
                print_progress(ndone, n_phi, verbose);
            }
        }

        free(abs_psi); free(vpsi); free(cos4vpsi); free(sin4vpsi);
        free(xi_nodes); free(factor_rows);
        free(h_hat); free(p_iota); free(p_t1);
        free(kappa_slot); free(kappa_w); free(kb_row); free(slot_of_kappa); free(basis_vb);
        free(col_out);
    }
    if (verbose > 0){ printf("\n"); fflush(stdout); }

    // Transpose the outputs back to (theta1, theta2), symmetrising the symmetric ones
    if (alloc_ok){
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for (int oi=0; oi<n_out; oi++){
            double *res = results + (long)oi*block_size;
            for (int i=0; i<n_r; i++){
                for (int j=i+1; j<n_r; j++){
                    long ij = (long)i*n_r + j, ji = (long)j*n_r + i;
                    double r_ij = res[ji], r_ji = res[ij];
                    if (symmetric_mask[oi]) r_ij = r_ji = .5*(r_ij + r_ji);
                    res[ij] = r_ij;
                    res[ji] = r_ji;
                }
            }
        }
    }

    free(c_slices); free(g_hat); free(slot_results);
    free(trig_v1); free(trig_v2); free(axis_t1); free(basis_v1);
    free(fa_of_key); free(fb_of_key); free(fa_xi); free(fa_factor); free(fb_xi); free(fb_factor);
    free(term_fa); free(term_fb); free(xi_used_a); free(xi_used_b);
    free(axis_phi);
}


/////////////////////////////////
// Continuous-limit estimators //
/////////////////////////////////

// Angular weight of qpp (w_type 0), qmm (1) or qpm (2) at the triplet angle phi, with (x, y) the
// third side r_jk of polar angle phi_jk, from cos(4 phi) and sin(4 phi). That of qpm,
// cos(4*(phi - phi_jk)) = Re[e^(4 i phi) (x - i y)^4]/|r_jk|^4, needs no trigonometric call.
static inline double q_weight(int w_type, double cos4phi, double sin4phi, double x, double y){

    if (w_type == 0) return 1.;
    if (w_type == 1) return cos4phi;
    double x2 = x*x, y2 = y*y, d = x2-y2, r2 = x2+y2;
    return r2 > 0. ? (cos4phi*(d*d - 4.*x2*y2) + sin4phi*4.*x*y*d)/(r2*r2) : cos4phi;
}

// Phi integral of one mixed term on the fine grid, fine[i1, i2] = sum_k w xi(r_jk) dphi_k. The
// progress counter ndone runs up to progtot rows, so that several calls fill one bar.
static void continuous_q_fine(
    const double *fine_centers, int n_fine, const double *phi_centers, const double *dphi,
    int n_phi, const double *xi_x, int xi_n, double xi_logx0, double xi_inv_dlogx,
    const double *xi_y, int w_type, double *fine, int *ndone, int progtot, int nthreads,
    int verbose){

    // Trig of the phi nodes
    double *trig_phi = trig_tables(phi_centers, n_phi);
    if (!trig_phi) return;
    const double *cosphi = trig_phi, *sinphi = trig_phi + n_phi;
    const double *cos4phi = trig_phi + 2*n_phi, *sin4phi = trig_phi + 3*n_phi;

    // One task per fine theta1 row
    #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for (int i1=0; i1<n_fine; i1++){
        double theta1 = fine_centers[i1];
        for (int i2=0; i2<n_fine; i2++){
            double theta2 = fine_centers[i2];
            // xi at r_jk = |theta1 - theta2 e(phi)|, weighted per w_type
            double acc = 0.;
            for (int iphi=0; iphi<n_phi; iphi++){
                double rjk_x = theta1 - theta2*cosphi[iphi];
                double rjk_y = -theta2*sinphi[iphi];
                double xi = interp1d_eval(xi_x, xi_y, xi_n,
                    xi_logx0, xi_inv_dlogx, hypot(rjk_x, rjk_y));
                double w = q_weight(w_type, cos4phi[iphi], sin4phi[iphi], rjk_x, rjk_y);
                acc += w*xi*dphi[iphi];
            }
            fine[(size_t)i1*n_fine + i2] = acc;
        }
        #pragma omp atomic
        *ndone += 1;
        print_progress(*ndone, progtot, verbose);
    }

    free(trig_phi);
}

// Average of the fine grid over the alpha x alpha sub-bins of each (n_r, n_r) bin, times prefactor
static void continuous_q_binavg(
    const double *fine, const double *fine_weights, int n_fine, int n_r, int alpha,
    double prefactor, double *out, int nthreads){

    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int ib1=0; ib1<n_r; ib1++){
        for (int ib2=0; ib2<n_r; ib2++){
            // Weighted mean over the fine sub-bins (is1, is2) of the coarse bin (ib1, ib2)
            double num = 0., den = 0.;
            for (int is1=0; is1<alpha; is1++){
                int i1 = ib1*alpha + is1;
                double w1 = fine_weights[i1];
                for (int is2=0; is2<alpha; is2++){
                    int i2 = ib2*alpha + is2;
                    double ww = w1*fine_weights[i2];
                    num += fine[(size_t)i1*n_fine + i2]*ww;
                    den += ww;
                }
            }
            out[(size_t)ib1*n_r + ib2] += prefactor*num/den;
        }
    }
}

// Continuous q-terms of all calls, each bin-averaged from a fine (alpha*n_r) grid, added to results
// (n_calls, n_r, n_r)
void continuous_q_tomo(
    int n_r, int alpha, const double *fine_centers, const double *fine_weights,
    const double *phi_centers, const double *dphi, int n_fine, int n_phi,
    const double *xi_x, int xi_n, double xi_logx0, double xi_inv_dlogx,
    const double *xi_y_tables, double sigma_eps2, double area, double nbar,
    int n_calls, const int *call_xi_idx, const int *call_w_type, double *results,
    int nthreads, int verbose){

    double *fine = (double *)orpheus_malloc((size_t)n_fine*n_fine*sizeof(double));
    if (!fine) return;
    reset_progress();
    int ndone = 0;
    // One call after the other, each parallelised over the fine grid
    for (int icall=0; icall<n_calls; icall++){
        continuous_q_fine(fine_centers, n_fine, phi_centers, dphi, n_phi,
            xi_x, xi_n, xi_logx0, xi_inv_dlogx, xi_y_tables + (long)call_xi_idx[icall]*xi_n,
            call_w_type[icall], fine, &ndone, n_calls*n_fine, nthreads, verbose);
        continuous_q_binavg(fine, fine_weights, n_fine, n_r, alpha,
            2.*sigma_eps2/(M_PI*area*nbar), results + (long)icall*n_r*n_r, nthreads);
    }
    if (verbose > 0){ printf("\n"); fflush(stdout); }
    free(fine);
}

// Continuous r-terms of all work units at a constant density, phi integrals of products of varphi
// integrals of either pair, added to results (n_out, n_r, n_r)
void continuous_rterms_tomo(
    int n_r, const double *theta_centers, const double *phi_grid, const double *phi_w, int n_phi,
    const double *v_grid, const double *v_w, int n_v, double area,
    const double *xi_x, int xi_n, double xi_logx0, double xi_inv_dlogx,
    const double *xi_y_tables, int n_xi_unique,
    int n_work, const int *work_xi_a_p, const int *work_xi_b_p,
    const int *work_xi_a_m, const int *work_xi_b_m, const int *work_out_idx,
    double *results, int nthreads, int verbose){

    // Prefactor 1/(2 pi A), which is the 2/(pi A) of rp0 times the 1/4 of its half circles
    long block_size = (long)n_r*n_r;
    double pre_r = 1./(2.*M_PI*area);

    // Integrals over varphi of each xi table that a work unit reads, as J[(iphi*n_xi_unique +
    // itab)*n_r + ir]: J_1 and J_c4v of xi_+, J_c4psi and J_c4vpsi of xi_-. The first pair sits at
    // psi = phi - theta e(varphi) and the second at psi(varphi + pi), so on a varphi grid that a
    // shift by pi maps onto itself both take the same integrals, and those of sin(4*varphi_psi)
    // and sin(4*(varphi - varphi_psi)) vanish by the symmetry varphi -> -varphi
    long j_len = (long)n_phi*n_xi_unique*n_r;
    int *tab_used = (int *)orpheus_calloc(n_xi_unique, sizeof(int));
    double *J_1 = (double *)orpheus_malloc(j_len*sizeof(double));
    double *J_c4v = (double *)orpheus_malloc(j_len*sizeof(double));
    double *J_c4psi = (double *)orpheus_malloc(j_len*sizeof(double));
    double *J_c4vpsi = (double *)orpheus_malloc(j_len*sizeof(double));
    double *trig_v = trig_tables(v_grid, n_v);
    if (!(tab_used && J_1 && J_c4v && J_c4psi && J_c4vpsi && trig_v)){
        free(tab_used); free(J_1); free(J_c4v); free(J_c4psi); free(J_c4vpsi); free(trig_v);
        return;
    }
    const double *cos_v = trig_v, *sin_v = trig_v + n_v, *cos4v = trig_v + 2*n_v;
    const double *sin4v = trig_v + 3*n_v;
    for (int iwork=0; iwork<n_work; iwork++){
        tab_used[work_xi_a_p[iwork]] = tab_used[work_xi_b_p[iwork]] = 1;
        tab_used[work_xi_a_m[iwork]] = tab_used[work_xi_b_m[iwork]] = 1;
    }

    reset_progress();
    int ndone = 0;
    int ntot = n_phi*n_r + n_work;

    // Varphi integrals per (phi, theta) node pair, whose separations all xi tables share, with
    // e(4*varphi_psi) = (psi/|psi|)^4
    #pragma omp parallel num_threads(nthreads)
    {
        // Lower xi node and its weight, cos(4*varphi_psi) and cos(4*(varphi - varphi_psi)) per
        // varphi node, with the node -1 outside of the xi grid
        int *ilo = (int *)orpheus_malloc((size_t)n_v*sizeof(int));
        double *wlo = (double *)orpheus_malloc((size_t)n_v*sizeof(double));
        double *c4psi = (double *)orpheus_malloc((size_t)n_v*sizeof(double));
        double *c4vpsi = (double *)orpheus_malloc((size_t)n_v*sizeof(double));
        int ok = ilo && wlo && c4psi && c4vpsi;

        #pragma omp for schedule(dynamic) collapse(2)
        for (int iphi=0; iphi<n_phi; iphi++){
            for (int ir=0; ir<n_r; ir++){
                if (!ok) continue;
                double phi = phi_grid[iphi], theta = theta_centers[ir];
                for (int iv=0; iv<n_v; iv++){
                    double x = phi - theta*cos_v[iv], y = -theta*sin_v[iv];
                    double abs_psi = hypot(x, y);
                    ilo[iv] = interp1d_locate(xi_x, xi_n, xi_logx0, xi_inv_dlogx, abs_psi, wlo+iv);
                    double c = abs_psi > 0. ? x/abs_psi : 1., s = abs_psi > 0. ? y/abs_psi : 0.;
                    double c2 = c*c - s*s, s2 = 2.*c*s;
                    double c4 = c2*c2 - s2*s2, s4 = 2.*c2*s2;
                    c4psi[iv] = c4;
                    c4vpsi[iv] = cos4v[iv]*c4 + sin4v[iv]*s4;
                }
                for (int itab=0; itab<n_xi_unique; itab++){
                    if (!tab_used[itab]) continue;
                    const double *xi_y = xi_y_tables + (long)itab*xi_n;
                    double s_1 = 0., s_c4v = 0., s_c4psi = 0., s_c4vpsi = 0.;
                    for (int iv=0; iv<n_v; iv++){
                        if (ilo[iv] < 0) continue;
                        int i = ilo[iv];
                        double xi_w = (xi_y[i] + wlo[iv]*(xi_y[i+1]-xi_y[i]))*v_w[iv];
                        s_1 += xi_w;
                        s_c4v += xi_w*cos4v[iv];
                        s_c4psi += xi_w*c4psi[iv];
                        s_c4vpsi += xi_w*c4vpsi[iv];
                    }
                    long off = ((long)iphi*n_xi_unique + itab)*n_r + ir;
                    J_1[off] = s_1;
                    J_c4v[off] = s_c4v;
                    J_c4psi[off] = s_c4psi;
                    J_c4vpsi[off] = s_c4vpsi;
                }
                #pragma omp atomic
                ndone += 1;
                print_progress(ndone, ntot, verbose);
            }
        }
        free(ilo); free(wlo); free(c4psi); free(c4vpsi);
    }

    // Outer products of the two pairs per work unit, i.e. (combo, Wick pairing), integrated over
    // phi with weight phi dphi
    #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for (int iwork=0; iwork<n_work; iwork++){
        // rp0, rp1, rm0, rm1 and rpm of the work unit, in the order of its outputs
        double *acc = (double *)orpheus_calloc((size_t)5*block_size, sizeof(double));
        if (!acc) continue;
        double *rp0 = acc, *rp1 = acc + block_size, *rm0 = acc + 2*block_size;
        double *rm1 = acc + 3*block_size, *rpm = acc + 4*block_size;

        for (int iphi=0; iphi<n_phi; iphi++){
            double phi_weight = phi_w[iphi]*phi_grid[iphi];
            long row = (long)iphi*n_xi_unique;
            const double *a_1 = J_1 + (row + work_xi_a_p[iwork])*n_r;
            const double *a_c4v = J_c4v + (row + work_xi_a_p[iwork])*n_r;
            const double *a_c4psi = J_c4psi + (row + work_xi_a_m[iwork])*n_r;
            const double *a_c4vpsi = J_c4vpsi + (row + work_xi_a_m[iwork])*n_r;
            const double *b_1 = J_1 + (row + work_xi_b_p[iwork])*n_r;
            const double *b_c4v = J_c4v + (row + work_xi_b_p[iwork])*n_r;
            const double *b_c4psi = J_c4psi + (row + work_xi_b_m[iwork])*n_r;
            const double *b_c4vpsi = J_c4vpsi + (row + work_xi_b_m[iwork])*n_r;

            for (int i1=0; i1<n_r; i1++){
                for (int i2=0; i2<n_r; i2++){
                    long idx = (long)i1*n_r + i2;
                    rp0[idx] += phi_weight*a_1[i1]*b_1[i2];
                    rp1[idx] += phi_weight*a_c4psi[i1]*b_c4psi[i2];
                    rm0[idx] += phi_weight*a_c4vpsi[i1]*b_c4vpsi[i2];
                    rm1[idx] += phi_weight*a_c4v[i1]*b_c4v[i2];
                    rpm[idx] += phi_weight*a_1[i1]*b_c4vpsi[i2];
                }
            }
        }

        // Apply the prefactors and add to the five consecutive outputs rp0, rp1, rm0, rm1 and rpm
        // from work_out_idx, with the 2 of the second pair of rpm
        double *dst = results + (long)work_out_idx[iwork]*block_size;
        #pragma omp critical
        {
            for (long j=0; j<4*block_size; j++) dst[j] += pre_r*acc[j];
            for (long j=4*block_size; j<5*block_size; j++) dst[j] += 2.*pre_r*acc[j];
        }

        free(acc);
        #pragma omp atomic
        ndone += 1;
        print_progress(ndone, ntot, verbose);
    }
    if (verbose > 0){ printf("\n"); fflush(stdout); }

    free(tab_used); free(J_1); free(J_c4v); free(J_c4psi); free(J_c4vpsi); free(trig_v);
}


//////////////////////////////////
// Phi-reduction of mixed terms //
//////////////////////////////////

// Integral over phi of the triplets per unit angle (nr, nr, n_modes), in the basis 1, cos(n phi),
// sin(n phi), against weight*xi(r_jk) for one mixed term, written to out (nr, nr)
void triplet_term_reduce(
    const double *triplets, int nr, int n_modes, const double *theta_edges, int n_sub, int n_ang,
    int weight_type, const double *xi_x, int xi_n, double xi_logx0, double xi_inv_dlogx,
    const double *xi_y, double *out, int nthreads, int verbose){

    // Radial sub-nodes at the midpoints of n_sub equal sub-intervals of each bin, over which
    // weight*xi is averaged with the triplets uniform in theta dtheta, and their weights theta
    // dtheta, normalised within each bin
    int n_nodes = nr*n_sub;
    double *theta_nodes = (double *)orpheus_malloc((size_t)n_nodes*sizeof(double));
    double *theta_weights = (double *)orpheus_malloc((size_t)n_nodes*sizeof(double));
    // Angular nodes at the midpoints of n_ang (even) equal intervals of 2 pi, of which those in
    // (0, pi) suffice as weight*xi is even in phi, their trig and Fourier basis (n_half, n_modes)
    int n_half = n_ang/2;
    double *phi_nodes = (double *)orpheus_malloc((size_t)n_half*sizeof(double));
    double *basis = (double *)orpheus_malloc((size_t)n_half*n_modes*sizeof(double));
    if (!(theta_nodes && theta_weights && phi_nodes && basis)){
        free(theta_nodes); free(theta_weights); free(phi_nodes); free(basis);
        return;
    }
    for (int i=0; i<nr; i++){
        double dt = (theta_edges[i+1]-theta_edges[i])/n_sub, norm = 0.;
        for (int isub=0; isub<n_sub; isub++){
            double lo = theta_edges[i] + isub*dt, hi = lo + dt;
            theta_nodes[i*n_sub+isub] = .5*(lo+hi);
            theta_weights[i*n_sub+isub] = hi*hi - lo*lo;
            norm += theta_weights[i*n_sub+isub];
        }
        for (int isub=0; isub<n_sub; isub++) theta_weights[i*n_sub+isub] /= norm;
    }
    for (int iphi=0; iphi<n_half; iphi++) phi_nodes[iphi] = (iphi+.5)*2.*M_PI/n_ang;
    double *trig_phi = trig_tables(phi_nodes, n_half);
    if (!trig_phi){
        free(theta_nodes); free(theta_weights); free(phi_nodes); free(basis);
        return;
    }
    const double *cosphi = trig_phi, *sinphi = trig_phi + n_half;
    const double *cos4phi = trig_phi + 2*n_half, *sin4phi = trig_phi + 3*n_half;
    for (int iphi=0; iphi<n_half; iphi++)
        fourier_row(cosphi[iphi], sinphi[iphi], n_modes, basis + (long)iphi*n_modes);
    // Each node in (0, pi) counts twice
    double dphi = 4.*M_PI/n_ang;
    reset_progress();
    int ndone = 0;

    #pragma omp parallel num_threads(nthreads)
    {
        // The triplets of one radial bin pair on the angular nodes
        double *triplets_phi = (double *)orpheus_malloc((size_t)n_half*sizeof(double));
        #pragma omp for schedule(dynamic)
        for (int i1=0; i1<nr; i1++){
            for (int i2=0; i2<nr && triplets_phi; i2++){
                const double *coef = triplets + ((size_t)i1*nr + i2)*n_modes;
                // Triplets on the nodes from their cosine coefficients, the only ones that enter
                // as weight*xi is even in phi: 1, cos(phi), cos(2 phi), ... sit at 0, 1, 3, ... of
                // the triplets and at 0 ... n_modes/2 of the blocked basis
                for (int iphi=0; iphi<n_half; iphi++){
                    const double *row = basis + (long)iphi*n_modes;
                    double val = coef[0];
                    for (int k=1; k<=n_modes/2; k++) val += coef[2*k-1]*row[k];
                    triplets_phi[iphi] = val;
                }
                double acc = 0.;
                for (int is1=0; is1<n_sub; is1++){
                    double theta1 = theta_nodes[i1*n_sub+is1];
                    for (int is2=0; is2<n_sub; is2++){
                        double theta2 = theta_nodes[i2*n_sub+is2];
                        double sum = 0.;
                        for (int iphi=0; iphi<n_half; iphi++){
                            double rjk2 = theta1*theta1 + theta2*theta2
                                - 2.*theta1*theta2*cosphi[iphi];
                            double rjk = rjk2 > 0. ? sqrt(rjk2) : 0.;
                            double xi = interp1d_eval(xi_x, xi_y, xi_n,
                                xi_logx0, xi_inv_dlogx, rjk);
                            sum += triplets_phi[iphi]*xi*q_weight(weight_type, cos4phi[iphi],
                                sin4phi[iphi], theta1 - theta2*cosphi[iphi], -theta2*sinphi[iphi]);
                        }
                        acc += theta_weights[i1*n_sub+is1]*theta_weights[i2*n_sub+is2]*sum;
                    }
                }
                out[(size_t)i1*nr + i2] = acc*dphi;
            }
            #pragma omp atomic
            ndone += 1;
            #pragma omp critical
            print_progress(ndone, nr, verbose);
        }
        free(triplets_phi);
    }
    if (verbose > 0){ printf("\n"); fflush(stdout); }

    free(theta_nodes); free(theta_weights); free(phi_nodes); free(basis); free(trig_phi);
}


////////////////////////////////////////////////////////
// Post-processing of the measured quadruplet density //
////////////////////////////////////////////////////////

// Density (nr, nr, nr, nm, nm) of the sparse radial bins from the smallest aligned block of 2^l
// bins per axis that holds count_min quadruplets per angular resolution cell, else the whole grid,
// in place
void quadruplets_merge_sparse(
    double *q, int nr, int nm, const double *v, double cell_norm, double q_ideal,
    double count_min, int nthreads){

    long nm2 = (long)nm*nm, n_cell = (long)nr*nr*nr;

    // Sparse bins, whose idealised count per cell, q_ideal cell_norm v_i v_j v_k, is below
    // count_min
    long n_sparse = 0;
    for (long cell=0; cell<n_cell; cell++)
        if (q_ideal*cell_norm*v[cell/((long)nr*nr)]*v[(cell/nr)%nr]*v[cell%nr] < count_min)
            n_sparse++;
    if (n_sparse == 0) return;
    long *sparse = (long *)orpheus_malloc((size_t)n_sparse*sizeof(long));
    double *cell_sum = (double *)orpheus_malloc((size_t)n_cell*sizeof(double));
    double *merged = (double *)orpheus_malloc((size_t)n_sparse*nm2*sizeof(double));
    if (!(sparse && cell_sum && merged)){
        free(sparse); free(cell_sum); free(merged);
        return;
    }
    n_sparse = 0;
    for (long cell=0; cell<n_cell; cell++)
        if (q_ideal*cell_norm*v[cell/((long)nr*nr)]*v[(cell/nr)%nr]*v[cell%nr] < count_min)
            sparse[n_sparse++] = cell;

    // Sum of q over the nm2 resolution cells of each radial bin, nm2 times its constant Fourier
    // coefficient
    for (long cell=0; cell<n_cell; cell++) cell_sum[cell] = q[cell*nm2]*nm2;

    // Block of each sparse bin and its density, the mean of q over its bins weighted by v_i v_j
    // v_k, written once all are known. The count of a block is cell_norm v_i v_j v_k q summed over
    // its bins and averaged over the nm^2 cells
    #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for (long is=0; is<n_sparse; is++){
        long cell = sparse[is];
        int bin[3] = {(int)(cell/((long)nr*nr)), (int)((cell/nr)%nr), (int)(cell%nr)};
        int lo[3], hi[3];
        for (int size=2; ; size*=2){
            for (int a=0; a<3; a++){
                lo[a] = bin[a]/size*size;
                hi[a] = lo[a]+size < nr ? lo[a]+size : nr;
            }
            double count = 0.;
            for (int i=lo[0]; i<hi[0]; i++)
                for (int j=lo[1]; j<hi[1]; j++)
                    for (int k=lo[2]; k<hi[2]; k++)
                        count += v[i]*v[j]*v[k]*cell_sum[((long)i*nr + j)*nr + k];
            if (cell_norm*count/nm2 >= count_min || size >= nr) break;
        }
        double *dst = merged + is*nm2;
        double v_sum = 0.;
        for (long ab=0; ab<nm2; ab++) dst[ab] = 0.;
        for (int i=lo[0]; i<hi[0]; i++){
            for (int j=lo[1]; j<hi[1]; j++){
                for (int k=lo[2]; k<hi[2]; k++){
                    double vol = v[i]*v[j]*v[k];
                    const double *src = q + (((long)i*nr + j)*nr + k)*nm2;
                    v_sum += vol;
                    for (long ab=0; ab<nm2; ab++) dst[ab] += vol*src[ab];
                }
            }
        }
        for (long ab=0; ab<nm2; ab++) dst[ab] /= v_sum;
    }
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (long is=0; is<n_sparse; is++)
        memcpy(q + sparse[is]*nm2, merged + is*nm2, (size_t)nm2*sizeof(double));

    free(sparse); free(cell_sum); free(merged);
}

// Knot values of the quadruplet density (nr, nr, nr, nm, nm) from its bin means m, in place, whose
// interpolant has these means
void quadruplets_to_knots(double *q, int nr, int nm, double coef, int nthreads){

    long nm2 = (long)nm*nm;
    for (int axis=0; axis<3; axis++){
        // Stride of the axis, and the start of its line through the bins (u, w) of the other two
        long stride = axis == 0 ? (long)nr*nr*nm2 : (axis == 1 ? (long)nr*nm2 : nm2);
        #pragma omp parallel num_threads(nthreads)
        {
            double *prev = (double *)orpheus_malloc((size_t)nm2*sizeof(double));
            double *cur = (double *)orpheus_malloc((size_t)nm2*sizeof(double));
            #pragma omp for schedule(static)
            for (long line=0; line<(long)nr*nr; line++){
                if (!(prev && cur)) continue;
                long u = line/nr, w = line%nr;
                long start = (axis == 0 ? (u*nr + w)*nm2
                    : (axis == 1 ? (u*nr*nr + w)*nm2 : (u*nr + w)*nr*nm2));
                memcpy(prev, q + start, (size_t)nm2*sizeof(double));
                // m_i - coef (m_(i+1) - 2 m_i + m_(i-1)) except at the first and the last bin, coef
                // 1/24 for the cubic and 1/8 for the linear interpolation
                for (int i=1; i<nr-1; i++){
                    double *node = q + start + i*stride;
                    const double *next = node + stride;
                    memcpy(cur, node, (size_t)nm2*sizeof(double));
                    for (long ab=0; ab<nm2; ab++)
                        node[ab] = cur[ab] - coef*(next[ab] - 2.*cur[ab] + prev[ab]);
                    double *swap = prev; prev = cur; cur = swap;
                }
            }
            free(prev); free(cur);
        }
    }
}

// Quadruplet density (nr, nr, nr, nm, nm), nr even and >= 4, extrapolated to vanishing radial bin
// width from its bin means q, in place: (4 q - q_2)/3 of the knot values of q and of q_2 on merged
// pairs of radial bins
void quadruplets_richardson(
    double *q, int nr, int nm, const double *weights, const double *centers, double coef,
    int cubic, int nthreads){

    int n2 = nr/2;
    long nm2 = (long)nm*nm, slab_len = (long)n2*nm2;
    double *w_merged = (double *)orpheus_malloc((size_t)n2*sizeof(double));
    double *centers_merged = (double *)orpheus_malloc((size_t)n2*sizeof(double));
    double *q_merged = (double *)orpheus_malloc((size_t)n2*n2*n2*nm2*sizeof(double));
    Interp4dRadial *axes = (Interp4dRadial *)orpheus_malloc((size_t)nr*sizeof(Interp4dRadial));
    if (!(w_merged && centers_merged && q_merged && axes)){
        free(w_merged); free(centers_merged); free(q_merged); free(axes);
        return;
    }

    // Weights of the merged bins and their centers, the weighted means of the centers
    for (int mi=0; mi<n2; mi++){
        w_merged[mi] = weights[2*mi] + weights[2*mi+1];
        centers_merged[mi] = ((weights[2*mi]*centers[2*mi] + weights[2*mi+1]*centers[2*mi+1])
            /w_merged[mi]);
    }

    // Merged density q_2, the same counts on pairs of radial bins: the mean over the 2 x 2 x 2
    // radial bins of a merged bin with the radial weights
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (long mijk=0; mijk<(long)n2*n2*n2; mijk++){
        int mi = mijk/((long)n2*n2), mj = (mijk/n2)%n2, mk = mijk%n2;
        double *dst = q_merged + mijk*nm2;
        double norm = 1./(w_merged[mi]*w_merged[mj]*w_merged[mk]);
        for (long ab=0; ab<nm2; ab++) dst[ab] = 0.;
        for (int d1=0; d1<2; d1++){
            for (int d2=0; d2<2; d2++){
                for (int d3=0; d3<2; d3++){
                    int i = 2*mi+d1, j = 2*mj+d2, k = 2*mk+d3;
                    double wt = weights[i]*weights[j]*weights[k]*norm;
                    const double *src = q + (((long)i*nr + j)*nr + k)*nm2;
                    for (long ab=0; ab<nm2; ab++) dst[ab] += wt*src[ab];
                }
            }
        }
    }

    // Knot values of both, quadruplets_to_knots with coef
    quadruplets_to_knots(q, nr, nm, coef, nthreads);
    quadruplets_to_knots(q_merged, n2, nm, coef, nthreads);

    // Stencil of each radial bin on the merged centers, as the integral interpolates (cubic or
    // linear), held beyond the first and the last of them
    Interp4dGrid grid_merged = interp4d_grid(centers_merged, n2, cubic);
    double last = centers_merged[n2-1];
    for (int i=0; i<nr; i++)
        axes[i] = interp4d_axis_r(&grid_merged, centers[i] < last ? centers[i] : last);

    // (4 q - q_2)/3 with q_2 interpolated per (theta1, theta2) bin first in theta1 and theta2 for
    // all merged theta3, then in theta3
    #pragma omp parallel num_threads(nthreads)
    {
        double *slab = (double *)orpheus_malloc((size_t)slab_len*sizeof(double));
        #pragma omp for schedule(static)
        for (long ij=0; ij<(long)nr*nr; ij++){
            if (!slab) continue;
            const Interp4dRadial *ai = axes + ij/nr, *aj = axes + ij%nr;
            for (long x=0; x<slab_len; x++) slab[x] = 0.;
            for (int d1=0; d1<ai->n; d1++){
                for (int d2=0; d2<aj->n; d2++){
                    double wt = ai->w[d1]*aj->w[d2];
                    if (wt == 0.) continue;
                    const double *src = (q_merged
                        + ((long)(ai->first+d1)*n2 + aj->first+d2)*slab_len);
                    for (long x=0; x<slab_len; x++) slab[x] += wt*src[x];
                }
            }
            for (int k=0; k<nr; k++){
                const Interp4dRadial *ak = axes + k;
                double *dst = q + (ij*nr + k)*nm2;
                for (long ab=0; ab<nm2; ab++){
                    double q_merged_k = 0.;
                    for (int d3=0; d3<ak->n; d3++)
                        q_merged_k += ak->w[d3]*slab[(long)(ak->first+d3)*nm2 + ab];
                    dst[ab] = (4.*dst[ab] - q_merged_k)/3.;
                }
            }
        }
        free(slab);
    }

    free(w_merged); free(centers_merged); free(q_merged); free(axes);
}
