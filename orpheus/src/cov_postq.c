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

static inline double interp1d_eval(
    const double *x, const double *y, int n,
    double logx0, double inv_dlogx, double xval){

    if (xval <= x[0] || xval >= x[n-1]) return 0.;
    int i = (int)((log(xval)-logx0)*inv_dlogx);
    if (i < 0) i = 0;
    if (i > n-2) i = n-2;
    if (i < n-2 && x[i+1] <= xval) i++;
    else if (i > 0 && x[i] > xval) i--;
    double w = (xval-x[i])/(x[i+1]-x[i]);
    return y[i] + w*(y[i+1]-y[i]);
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

// One axis of the quadrilinear stencil: the two bracketing indices and their
// weights. Resolving an axis costs a log or an fmod plus the clamp branches,
// which is three quarters of a lookup, and each of the four axes is loop
// invariant somewhere in the SVA kernel's q build -- hence the split.
typedef struct { int i, ip; double w, mw; } Interp4dAxis;

// Log-uniform radial axis. Above the grid both weights are zeroed, which makes
// the whole 16-corner sum vanish; below it the weight clamps to 0 (nearest
// neighbour on that axis).
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

// Uniform angular axis, wrapping periodically mod 2*pi.
static inline __attribute__((always_inline)) Interp4dAxis interp4d_axis_phi(
    const Interp4dGrid *g, double a){

    double twopi = 2.*M_PI;
    a = fmod(a, twopi);
    if (a < 0.) a += twopi;

    const double *phi_centers = g->phi_centers;
    int nphi = g->nphi;
    int i = (int)((a-g->phi0)*g->inv_dphi);
    if (i < 0) i = 0;
    else if (i > nphi-2) i = nphi-2;
    if (i < nphi-2 && phi_centers[i+1] <= a) i++;
    else if (i > 0 && phi_centers[i] > a) i--;

    Interp4dAxis ax;
    ax.w = (a-phi_centers[i])/(phi_centers[i+1]-phi_centers[i]);
    ax.mw = 1.-ax.w;
    ax.i = i; ax.ip = (i+1)%nphi;
    return ax;
}

// The 16-corner sum, once all four axes are resolved. The repeated products
// (mw2*mw3 and so on) are left written out: gcc already shares them, and
// pre-combining them by hand measures no faster.
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

// One quadrilinear lookup q(t2, t3, a12, a13) on a 4-D slice of shape
// (nr, nr, nphi, nphi), resolving all four axes on the spot. Callers that can
// hoist an axis out of a loop should use the three pieces above instead.
static inline __attribute__((always_inline)) double interp4d_point(
    const double *q_slice, const Interp4dGrid *g,
    double t2, double t3, double a12, double a13){

    Interp4dAxis x2 = interp4d_axis_r(g, t2);
    Interp4dAxis x3 = interp4d_axis_r(g, t3);
    Interp4dAxis x12 = interp4d_axis_phi(g, a12);
    Interp4dAxis x13 = interp4d_axis_phi(g, a13);
    return interp4d_gather(q_slice, g, &x2, &x3, &x12, &x13);
}

// interp4d
//
// Quadrilinear interpolation of a 4-D quadruplet-density slice q(t2, t3, a12, a13)
// on a grid with log-uniform radial spacing and uniform (periodic) angular spacing.
// Ported from the numba-jitted Python reference that used to live in covariance.py.
//
// q_slice      (nr, nr, nphi, nphi)  C-contiguous
// r_centers    (nr,)                 log-uniform radial grid
// phi_centers  (nphi,)               uniform angular grid, [0, 2pi)
// t2f/t3f/a12f/a13f  (n,)            flat query points
// out          (n,)                  output, overwritten
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

////////////////////////////////
// Quadruplet (SVA) integral   //
////////////////////////////////

// One callback over the (varphi1, varphi2) grid of a single (theta1, theta2)
// pair. wid is a compile-time constant in each instantiation, so the inner
// loop carries no branch and vectorises; every weight it needs is already
// tabulated on the grid, and the quadrature weights are folded into q.
#define SVA_SLICE_BODY(WEXPR)                                                 \
    for (int iv1 = 0; iv1 < nv1; iv1++) {                                     \
        const double *q = qslice + (long)iv1*nv2c;                            \
        const double *w1 = wtab1 + (long)iv1*nv2c;                            \
        const double *w2 = wtab2 + (long)iv1*nv2c;                            \
        const double *wv = w_v1v2 + (long)iv1*nv2c;                           \
        double s = 0.;                                                        \
        (void)w1; (void)w2; (void)wv;                                         \
        _Pragma("omp simd reduction(+:s)")                                    \
        for (int iv2 = 0; iv2 < nv2c; iv2++)                                  \
            s += q[iv2]*xib[iv2]*(WEXPR);                                     \
        sum += xia[iv1]*s;                                                    \
    }

static inline __attribute__((always_inline)) double sva_slice(
    int wid, const double *qslice, const double *xia, const double *xib,
    const double *wtab1, const double *wtab2, const double *w_v1v2,
    int nv1, int nv2c){

    double sum = 0.;
    switch (wid) {
    case 0: { SVA_SLICE_BODY(1.) } break;
    case 1: { SVA_SLICE_BODY(w1[iv2]) } break;
    case 2: { SVA_SLICE_BODY(w2[iv2]) } break;
    case 3: { SVA_SLICE_BODY(wv[iv2]) } break;
    default: { SVA_SLICE_BODY(1.) } break; // wid 4 is folded into xib
    }
    return sum;
}

// quadruplet_sva_integrate
//
// Cosmic-variance (r-term) integral over the whole phi grid in one call. Each
// phi node contributes an outer product of the two side integrals psi_a =
// phi - theta1 e(V1) and psi_b = phi + theta2 e(V2), weighted by the
// quadruplet density q and by xi_pm(|psi|); OpenMP parallelises over phi.
// Every thread keeps a private (n_out, n_r, n_r) accumulator and the
// accumulators are summed in thread order at the end, so the result is
// reproducible run to run and identical to the serial one at nthreads=1.
//
// phi_grid, phi_w   (n_phi,)        outer radial nodes and trapezoid weights
// phi1_nodes        (n_r*ns1,)      theta1 nodes, (bin, subgrid) flattened
// phi1_weights      (n_r*ns1,)      matching quadrature weights
// phi2_nodes, phi2_weights (n_r*ns2,)   the same on side b
// v1_grid, v1_w     (nv1,)          varphi1 nodes and weights
// v2_grid, v2_w     (nv2,)          varphi2 nodes and weights
// nr_block          tile size on the (theta1, theta2) axes
// v2_block          chunk size on the varphi2 axis
// tiles             (n_tiles, 2)    (i1, i2) tile origins, in evaluation order
// q_mode            0 = 5-D density tensor, 1 = constant q_const,
//                   2 = separable q_const*s(t1)*s(t2)*s(t3)
// q_array           (q_nr, q_nr, q_nr, q_nphi, q_nphi)   q_mode 0
// q_r_centers       (q_nr,)         log-uniform radial grid of q
// q_phi_centers     (q_nphi,)       uniform angular grid of q
// sup_t1            (n_phi,)                        q_mode 2: s(phi)
// sup_t2            (n_r*ns1,)                      q_mode 2: s(theta1 node)
// sup_t3            (n_phi, n_r*ns2, nv2)           q_mode 2: s(|psi_b|)
// xi_x              (xi_n,)         shared x grid for all xi splines
// xi_y_tables       (n_xi_unique, xi_n)   tabulated xi spline y values
// xi_a_idx, xi_b_idx (n_cb,)        table index for the psi_a / psi_b lookup
// w_ids             (n_cb,)         weight type 0-4
// out_idx           (n_cb,)         output block index
// wick_fac          (n_cb,)         Wick multiplicity (1. or 0.5)
// symmetric_mask    (n_out,)        1 = symmetrise the block at each phi node
// pair_mask         (n_r, n_r)      0 = drop the entry (all ones = no mask)
// results           (n_out, n_r, n_r)   output, accumulated in-place
void quadruplet_sva_integrate(
    const double *phi_grid, const double *phi_w, int n_phi, int n_r,
    const double *phi1_nodes, const double *phi1_weights, int ns1,
    const double *phi2_nodes, const double *phi2_weights, int ns2,
    const double *v1_grid, const double *v1_w, int nv1,
    const double *v2_grid, const double *v2_w, int nv2,
    int nr_block, int v2_block, const int *tiles, int n_tiles,
    int q_mode, const double *q_array,
    const double *q_r_centers, const double *q_phi_centers, int q_nr, int q_nphi,
    double q_const, const double *sup_t1, const double *sup_t2, const double *sup_t3,
    const double *xi_x, int xi_n, double xi_logx0, double xi_inv_dlogx,
    const double *xi_y_tables, int n_xi_unique,
    int n_cb, const int *xi_a_idx, const int *xi_b_idx,
    const int *w_ids, const int *out_idx, const double *wick_fac,
    int n_out, const int *symmetric_mask, const int *pair_mask,
    double *results, int nthreads, int verbose){

    long block = (long)n_r*n_r;
    long acc_len = (long)n_out*block;
    int mb = nr_block < n_r ? nr_block : n_r;
    int m1max = mb*ns1, m2max = mb*ns2;
    int nv2max = v2_block < nv2 ? v2_block : nv2;
    int m2tot = n_r*ns2;
    long slice_len = (long)q_nr*q_nr*q_nphi*q_nphi;

    Interp4dGrid grid = {0};
    if (q_mode == 0)
        grid = interp4d_grid(q_r_centers, q_phi_centers, q_nr, q_nphi,
            q_r_centers[0], q_r_centers[q_nr-1], log(q_r_centers[0]),
            1./(log(q_r_centers[1])-log(q_r_centers[0])),
            q_phi_centers[0], 1./(q_phi_centers[1]-q_phi_centers[0]));

    // Trig of the two varphi grids: shared, read-only, phi-independent.
    double *cos_v1 = (double *)orpheus_malloc((size_t)nv1*sizeof(double));
    double *sin_v1 = (double *)orpheus_malloc((size_t)nv1*sizeof(double));
    double *cos4v1 = (double *)orpheus_malloc((size_t)nv1*sizeof(double));
    double *sin4v1 = (double *)orpheus_malloc((size_t)nv1*sizeof(double));
    double *cos_v2 = (double *)orpheus_malloc((size_t)nv2*sizeof(double));
    double *sin_v2 = (double *)orpheus_malloc((size_t)nv2*sizeof(double));
    double *cos4v2f = (double *)orpheus_malloc((size_t)nv2*sizeof(double));
    double *sin4v2f = (double *)orpheus_malloc((size_t)nv2*sizeof(double));
    double *partials = (double *)orpheus_calloc((size_t)nthreads*acc_len, sizeof(double));

    // Axis lookups into the q grid that do not depend on phi: the theta1 nodes
    // and the whole varphi1 grid. Both are read n_r*ns2*nv2 times per phi node
    // in the q build below, so they are resolved once here.
    Interp4dAxis *axis_t1 = (q_mode == 0 ? (Interp4dAxis *)orpheus_malloc(
        (size_t)n_r*ns1*sizeof(Interp4dAxis)) : NULL);
    Interp4dAxis *axis_v1 = (q_mode == 0 ? (Interp4dAxis *)orpheus_malloc(
        (size_t)nv1*sizeof(Interp4dAxis)) : NULL);

    // Per-thread scratch below is the largest allocation in the module and
    // grows with nthreads, so a failure here is a realistic outcome rather
    // than a formality. Nothing may be dereferenced until the whole team has
    // agreed it allocated; orpheus_malloc has already reported the size that
    // failed, and the caller's check_clib_error() turns it into a MemoryError.
    int alloc_ok = (cos_v1 && sin_v1 && cos4v1 && sin4v1
                    && cos_v2 && sin_v2 && cos4v2f && sin4v2f && partials
                    && (q_mode != 0 || (axis_t1 && axis_v1)));
    if (alloc_ok && q_mode == 0) {
        for (int j = 0; j < n_r*ns1; j++)
            axis_t1[j] = interp4d_axis_r(&grid, phi1_nodes[j]);
        for (int j = 0; j < nv1; j++)
            axis_v1[j] = interp4d_axis_phi(&grid, v1_grid[j]);
    }
    if (alloc_ok) {
        for (int j = 0; j < nv1; j++) {
            cos_v1[j] = cos(v1_grid[j]);
            sin_v1[j] = sin(v1_grid[j]);
            cos4v1[j] = cos(4.*v1_grid[j]);
            sin4v1[j] = sin(4.*v1_grid[j]);
        }
        for (int j = 0; j < nv2; j++) {
            cos_v2[j] = cos(v2_grid[j]);
            sin_v2[j] = sin(v2_grid[j]);
            cos4v2f[j] = cos(4.*v2_grid[j]);
            sin4v2f[j] = sin(4.*v2_grid[j]);
        }
    }
    reset_progress();
    int ndone = 0;

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        double *part = partials + (long)tid*acc_len;

        double *abs_a = (double *)orpheus_malloc((size_t)m1max*nv1*sizeof(double));
        double *va = (double *)orpheus_malloc((size_t)m1max*nv1*sizeof(double));
        double *cos4va = (double *)orpheus_malloc((size_t)m1max*nv1*sizeof(double));
        double *sin4va = (double *)orpheus_malloc((size_t)m1max*nv1*sizeof(double));
        double *abs_b = (double *)orpheus_malloc((size_t)m2max*nv2max*sizeof(double));
        double *vb = (double *)orpheus_malloc((size_t)m2max*nv2max*sizeof(double));
        double *cos4vb = (double *)orpheus_malloc((size_t)m2max*nv2max*sizeof(double));
        double *sin4vb = (double *)orpheus_malloc((size_t)m2max*nv2max*sizeof(double));
        double *xi_a_tab = (double *)orpheus_malloc(
            (size_t)n_xi_unique*m1max*nv1*sizeof(double));
        double *xi_b_tab = (double *)orpheus_malloc(
            (size_t)n_xi_unique*m2max*nv2max*sizeof(double));
        double *w_v1v2 = (double *)orpheus_malloc((size_t)nv1*nv2max*sizeof(double));
        double *w_r2v2 = (double *)orpheus_malloc((size_t)m2max*nv2max*sizeof(double));
        double *wtab1 = (double *)orpheus_malloc((size_t)nv1*nv2max*sizeof(double));
        double *wtab2 = (double *)orpheus_malloc((size_t)nv1*nv2max*sizeof(double));
        double *xib_scr = (double *)orpheus_malloc((size_t)nv2max*sizeof(double));
        double *q_vals = (double *)orpheus_malloc(
            (size_t)m1max*m2max*nv1*nv2max*sizeof(double));
        double *c_accum = (double *)orpheus_malloc((size_t)n_out*m1max*m2max*sizeof(double));
        double *sub_accum = (double *)orpheus_malloc((size_t)acc_len*sizeof(double));
        double *q_slice = (q_mode == 0
            ? (double *)orpheus_malloc((size_t)slice_len*sizeof(double)) : NULL);
        Interp4dAxis *axis_b_r = (q_mode == 0 ? (Interp4dAxis *)orpheus_malloc(
            (size_t)m2max*nv2max*sizeof(Interp4dAxis)) : NULL);
        Interp4dAxis *axis_b_p = (q_mode == 0 ? (Interp4dAxis *)orpheus_malloc(
            (size_t)m2max*nv2max*sizeof(Interp4dAxis)) : NULL);

        if (!(abs_a && va && cos4va && sin4va && abs_b && vb && cos4vb && sin4vb
              && xi_a_tab && xi_b_tab && w_v1v2 && w_r2v2 && wtab1 && wtab2
              && xib_scr && q_vals && c_accum && sub_accum
              && (q_mode != 0 || (q_slice && axis_b_r && axis_b_p)))) {
            #pragma omp atomic write
            alloc_ok = 0;
        }
        #pragma omp barrier

        // Every thread must still encounter the `omp for` (its implicit
        // barrier is collective), so a failed team is turned into a zero trip
        // count rather than a skipped construct.
        int n_phi_run = alloc_ok ? n_phi : 0;

        #pragma omp for schedule(static)
        for (int kphi = 0; kphi < n_phi_run; kphi++) {
            double phi = phi_grid[kphi];

            // Density at the fixed first spoke t1 = phi. Boundary conventions:
            // phi above the q grid gives a hard zero, below it the innermost
            // slice (nearest neighbour), else a linear blend.
            if (q_mode == 0) {
                if (phi >= grid.r_max) {
                    memset(q_slice, 0, (size_t)slice_len*sizeof(double));
                } else if (phi <= grid.r_min) {
                    memcpy(q_slice, q_array, (size_t)slice_len*sizeof(double));
                } else {
                    int i = 0;
                    while (i < q_nr && q_r_centers[i] < phi) i++;
                    i--;
                    if (i > q_nr-2) i = q_nr-2;
                    double w = (phi-q_r_centers[i])
                        /(q_r_centers[i+1]-q_r_centers[i]);
                    const double *s0 = q_array + (long)i*slice_len;
                    const double *s1 = s0 + slice_len;
                    for (long j = 0; j < slice_len; j++)
                        q_slice[j] = (1.-w)*s0[j] + w*s1[j];
                }
            }
            double s1phi = q_mode == 2 ? sup_t1[kphi] : 1.;
            const double *sup3 = (q_mode == 2
                ? sup_t3 + (long)kphi*m2tot*nv2 : sup_t3);

            memset(sub_accum, 0, (size_t)acc_len*sizeof(double));

            for (int it = 0; it < n_tiles; it++) {
                int i1 = tiles[2*it], i2 = tiles[2*it+1];
                int nrb1 = nr_block < n_r-i1 ? nr_block : n_r-i1;
                int nrb2 = nr_block < n_r-i2 ? nr_block : n_r-i2;
                int m1 = nrb1*ns1, m2 = nrb2*ns2;
                const double *P1 = phi1_nodes + (long)i1*ns1;
                const double *P2 = phi2_nodes + (long)i2*ns2;
                int n_a = m1*nv1;

                // psi_a = phi - theta1 e(V1), independent of the varphi2 chunk
                for (int ia = 0; ia < m1; ia++) {
                    for (int iv1 = 0; iv1 < nv1; iv1++) {
                        double ax = phi - P1[ia]*cos_v1[iv1];
                        double ay = -P1[ia]*sin_v1[iv1];
                        int j = ia*nv1 + iv1;
                        abs_a[j] = hypot(ax, ay);
                        va[j] = atan2(ay, ax);
                        cos4va[j] = cos(4.*va[j]);
                        sin4va[j] = sin(4.*va[j]);
                    }
                }
                for (int t = 0; t < n_xi_unique; t++) {
                    const double *yi = xi_y_tables + (long)t*xi_n;
                    double *dst = xi_a_tab + (long)t*n_a;
                    for (int j = 0; j < n_a; j++)
                        dst[j] = interp1d_eval(xi_x, yi, xi_n,
                            xi_logx0, xi_inv_dlogx, abs_a[j]);
                }

                for (int j0 = 0; j0 < nv2; j0 += v2_block) {
                    int nv2c = v2_block < nv2-j0 ? v2_block : nv2-j0;
                    int n_b = m2*nv2c;
                    const double *cos4v2 = cos4v2f + j0;
                    const double *sin4v2 = sin4v2f + j0;
                    const double *v2_wc = v2_w + j0;

                    // psi_b = phi + theta2 e(V2)
                    for (int ib = 0; ib < m2; ib++) {
                        for (int iv2 = 0; iv2 < nv2c; iv2++) {
                            double bx = phi + P2[ib]*cos_v2[j0+iv2];
                            double by = P2[ib]*sin_v2[j0+iv2];
                            int j = ib*nv2c + iv2;
                            abs_b[j] = hypot(bx, by);
                            vb[j] = atan2(by, bx);
                            cos4vb[j] = cos(4.*vb[j]);
                            sin4vb[j] = sin(4.*vb[j]);
                        }
                    }
                    for (int t = 0; t < n_xi_unique; t++) {
                        const double *yi = xi_y_tables + (long)t*xi_n;
                        double *dst = xi_b_tab + (long)t*n_b;
                        for (int j = 0; j < n_b; j++)
                            dst[j] = interp1d_eval(xi_x, yi, xi_n,
                                xi_logx0, xi_inv_dlogx, abs_b[j]);
                    }
                    // The remaining two q axes: |psi_b| and varphi_b vary with
                    // phi, so unlike theta1 and varphi1 they are resolved here.
                    if (q_mode == 0) {
                        for (int j = 0; j < n_b; j++) {
                            axis_b_r[j] = interp4d_axis_r(&grid, abs_b[j]);
                            axis_b_p[j] = interp4d_axis_phi(&grid, vb[j]);
                        }
                    }

                    // The two weight tables that do not depend on (ir1, ir2):
                    // * w=3: cos(4*(v1-v2))
                    // * w=4: 2*cos(4*(v2-vb))
                    for (int iv1 = 0; iv1 < nv1; iv1++) {
                        for (int iv2 = 0; iv2 < nv2c; iv2++)
                            w_v1v2[iv1*nv2c + iv2] = cos4v1[iv1]*cos4v2[iv2]
                                + sin4v1[iv1]*sin4v2[iv2];
                    }
                    for (int ir2 = 0; ir2 < m2; ir2++)
                        for (int iv2 = 0; iv2 < nv2c; iv2++) {
                            int idx = ir2*nv2c + iv2;
                            w_r2v2[idx] = 2.*(cos4v2[iv2]*cos4vb[idx]
                                + sin4v2[iv2]*sin4vb[idx]);
                        }

                    // q on the (theta1, |psi_b|, V1, varphi_b) tile
                    long q_stride = (long)nv1*nv2c;
                    if (q_mode == 0) {
                        for (int ia = 0; ia < m1; ia++) {
                            const Interp4dAxis *x2 = &axis_t1[i1*ns1 + ia];
                            for (int ib = 0; ib < m2; ib++) {
                                double *dst = q_vals
                                    + ((long)ia*m2 + ib)*q_stride;
                                for (int iv1 = 0; iv1 < nv1; iv1++) {
                                    const Interp4dAxis *x12 = &axis_v1[iv1];
                                    double *row = dst + (long)iv1*nv2c;
                                    for (int iv2 = 0; iv2 < nv2c; iv2++) {
                                        int bj = ib*nv2c + iv2;
                                        row[iv2] = interp4d_gather(q_slice, &grid,
                                            x2, &axis_b_r[bj], x12, &axis_b_p[bj]);
                                    }
                                }
                            }
                        }
                    } else if (q_mode == 1) {
                        long ntot = (long)m1*m2*q_stride;
                        for (long j = 0; j < ntot; j++) q_vals[j] = q_const;
                    } else {
                        for (int ia = 0; ia < m1; ia++) {
                            double s12 = s1phi*sup_t2[i1*ns1 + ia];
                            for (int ib = 0; ib < m2; ib++) {
                                const double *s3 = sup3
                                    + (long)(i2*ns2 + ib)*nv2 + j0;
                                double *dst = q_vals
                                    + ((long)ia*m2 + ib)*q_stride;
                                for (int iv1 = 0; iv1 < nv1; iv1++) {
                                    double *row = dst + (long)iv1*nv2c;
                                    for (int iv2 = 0; iv2 < nv2c; iv2++)
                                        row[iv2] = q_const*(s12*s3[iv2]);
                                }
                            }
                        }
                    }

                    // Fold the varphi quadrature weights into q. They depend
                    // only on (iv1, iv2), and the callback loop below reads q
                    // n_cb times, so this pays for itself immediately.
                    for (long ip = 0; ip < (long)m1*m2; ip++) {
                        double *dst = q_vals + ip*q_stride;
                        for (int iv1 = 0; iv1 < nv1; iv1++) {
                            double w1 = v1_w[iv1];
                            double *row = dst + (long)iv1*nv2c;
                            for (int iv2 = 0; iv2 < nv2c; iv2++)
                                row[iv2] *= w1*v2_wc[iv2];
                        }
                    }

                    memset(c_accum, 0, (size_t)n_out*m1*m2*sizeof(double));

                    // Main callback loop. The (theta1, theta2) pair is the
                    // outer level so that the q slice and the two (ir1, ir2)
                    // dependent weight tables are built once and reused by all
                    // n_cb callbacks -- wid 2 alone is 14 flops per grid point
                    // and would otherwise be recomputed for every tomographic
                    // bin combination. With wid then fixed per instantiation
                    // of sva_slice the innermost loop is branch free and
                    // vectorises; leaving the switch in it blocks that
                    // ("control flow in loop") and measures 4x slower.
                    // Hoisting the callbacks the other way instead, outside
                    // the r and varphi2 tiling, is the variant that measured
                    // 2x slower -- it scatters into c_accum from inside the
                    // hot nest and streams q from memory once per callback.
                    for (int ir1 = 0; ir1 < m1; ir1++) {
                        for (int ir2 = 0; ir2 < m2; ir2++) {
                            const double *qslice = q_vals
                                + ((long)ir1*m2 + ir2)*q_stride;
                            const double *cvb = cos4vb + ir2*nv2c;
                            const double *svb = sin4vb + ir2*nv2c;

                            // w=1: cos(4*(va-vb)),  w=2: cos(4*(v1+v2-va-vb))
                            for (int iv1 = 0; iv1 < nv1; iv1++) {
                                double ca = cos4va[ir1*nv1 + iv1];
                                double sa = sin4va[ir1*nv1 + iv1];
                                double c1 = cos4v1[iv1], s1 = sin4v1[iv1];
                                double *tw1 = wtab1 + (long)iv1*nv2c;
                                double *tw2 = wtab2 + (long)iv1*nv2c;
                                for (int iv2 = 0; iv2 < nv2c; iv2++) {
                                    double cb = cvb[iv2], sb = svb[iv2];
                                    double cA = c1*cos4v2[iv2] - s1*sin4v2[iv2];
                                    double sA = s1*cos4v2[iv2] + c1*sin4v2[iv2];
                                    tw1[iv2] = ca*cb + sa*sb;
                                    tw2[iv2] = cA*(ca*cb - sa*sb)
                                        + sA*(sa*cb + ca*sb);
                                }
                            }

                            for (int k = 0; k < n_cb; k++) {
                                int wid = w_ids[k];
                                const double *xia = xi_a_tab
                                    + (long)xi_a_idx[k]*n_a + ir1*nv1;
                                const double *xib = xi_b_tab
                                    + (long)xi_b_idx[k]*n_b + ir2*nv2c;
                                if (wid == 4) {
                                    const double *wr = w_r2v2 + ir2*nv2c;
                                    for (int j = 0; j < nv2c; j++)
                                        xib_scr[j] = xib[j]*wr[j];
                                    xib = xib_scr;
                                }
                                c_accum[(long)out_idx[k]*m1*m2 + ir1*m2 + ir2]
                                    += wick_fac[k]*sva_slice(wid, qslice, xia,
                                        xib, wtab1, wtab2, w_v1v2, nv1, nv2c);
                            }
                        }
                    }

                    // Contract the phi subgrid nodes back onto the coarse bins
                    for (int oi = 0; oi < n_out; oi++) {
                        const double *ca = c_accum + (long)oi*m1*m2;
                        double *dst = sub_accum + (long)oi*block;
                        for (int a = 0; a < nrb1; a++) {
                            for (int b = 0; b < nrb2; b++) {
                                double s = 0.;
                                for (int p = 0; p < ns1; p++) {
                                    double w1 = phi1_weights[(i1+a)*ns1 + p];
                                    for (int q = 0; q < ns2; q++)
                                        s += ca[(long)(a*ns1+p)*m2 + b*ns2+q]
                                            *w1*phi2_weights[(i2+b)*ns2 + q];
                                }
                                dst[(long)(i1+a)*n_r + i2+b] += s;
                            }
                        }
                    }
                }
            }

            // Mask, symmetrise and fold this phi node into the thread total
            double common = phi*phi_w[kphi];
            double half = .5*common;
            for (int oi = 0; oi < n_out; oi++) {
                double *cur = sub_accum + (long)oi*block;
                double *p = part + (long)oi*block;
                for (long j = 0; j < block; j++)
                    if (!pair_mask[j]) cur[j] = 0.;
                if (symmetric_mask[oi])
                    for (int i = 0; i < n_r; i++)
                        for (int j = 0; j < n_r; j++)
                            p[i*n_r+j] += half*(cur[i*n_r+j] + cur[j*n_r+i]);
                else
                    for (long j = 0; j < block; j++)
                        p[j] += common*cur[j];
            }
            #pragma omp atomic
            ndone += 1;
            print_progress(ndone, n_phi, verbose);
        }

        free(abs_a); free(va); free(cos4va); free(sin4va);
        free(abs_b); free(vb); free(cos4vb); free(sin4vb);
        free(xi_a_tab); free(xi_b_tab);
        free(w_v1v2); free(w_r2v2); free(wtab1); free(wtab2); free(xib_scr);
        free(q_vals); free(c_accum); free(sub_accum); free(q_slice);
        free(axis_b_r); free(axis_b_p);
    }
    if (verbose > 0) { printf("\n"); fflush(stdout); }

    // Sum the per-thread accumulators in thread order
    for (int t = 0; alloc_ok && t < nthreads; t++) {
        const double *p = partials + (long)t*acc_len;
        for (long j = 0; j < acc_len; j++) results[j] += p[j];
    }

    free(partials);
    free(cos_v1); free(sin_v1); free(cos4v1); free(sin4v1);
    free(cos_v2); free(sin_v2); free(cos4v2f); free(sin4v2f);
    free(axis_t1); free(axis_v1);
}


/////////////////////////////////
// Continuous-limit estimators //
/////////////////////////////////

// Continuous q-term (triplet) for one xi function. Computes the bin-averaged
// integral sum_k weight(T1,T2,phi_k,r_jk) * xi(r_jk) on a fine (alpha*n_r)
// grid, then sums sub-bins back to (n_r, n_r).
// w_type: 0 = qpp (weight=1), 1 = qmm (cos(4*PHI)),
//         2 = qpm (cos(4*(PHI - atan2(jk_y, jk_x))))
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


// Build r-terms for all tomo combos, in continuous approximation.  The side
// integrals are shared across work units; the outer products parallelise over
// (combo, wick_pairing).
void continuous_rterms_tomo(
    int n_r, const double *theta_centers, const double *phi_grid, const double *phi_w, int n_phi,
    const double *v_grid, const double *v_w, int n_v, const double *v_half, const double *v_half_w, int n_vh,
    double area, const double *xi_x, int xi_n, double xi_logx0, double xi_inv_dlogx, const double *xi_y_tables, int n_xi_unique,
    int n_work, const int *work_xi_a_p, const int *work_xi_b_p, const int *work_xi_a_m, const int *work_xi_b_m, const int *work_out_idx,
    int n_out, double *results, int nthreads, int verbose){

    long block_size = (long)n_r*n_r;
    double inv_2piA = 1./(2.*M_PI*area);
    double inv_piA_times2 = 2./(M_PI*area);

    // Distinct xi table pairs per side. Every output is an outer product
    // Ia[.] (x) Ib[.], and each side depends only on its own (xi_p, xi_m)
    // table pair. The same pair recurs across O(n_bp) work units, so the
    // pairs are collected here and each side integral is evaluated exactly
    // once below.
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

    // Side integrals, one pass over (phi, distinct table pair).
    // Layout: I[(kp * n_side + u) * n_r + ir].
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

    // The side-integral tables scale as n_phi*n_side*n_r and are the only
    // large allocation here; bail out before touching them if any failed.
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
void triplet_term_reduce(
    const double *Nt, int nr, int nphi, const double *tc, const double *phic, int weight_type,
    const double *xi_x, int xi_n, double xi_logx0, double xi_inv_dlogx, const double *xi_y,
    double *out, int nthreads, int verbose){

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
