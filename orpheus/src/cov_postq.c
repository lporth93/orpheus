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

// Linear interpolation of y(x) on a log-uniform grid x, zero outside of it
static inline double interp1d_eval(
    const double *x, const double *y, int n,
    double logx0, double inv_dlogx, double xval){

    if (xval<=x[0] || xval>=x[n-1]) return 0.;
    // Lower node from the log-uniform spacing, corrected by one step against round-off
    int i = (int)((log(xval)-logx0)*inv_dlogx);
    if (i<0) i = 0;
    if (i>n-2) i = n-2;
    if (i<n-2 && x[i+1]<=xval) i++;
    else if (i>0 && x[i]>xval) i--;
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
    long s_i2, s_i3;
} Interp4dGrid;

// Grid from its axes, with the strides of the t2 and t3 axes of a (nr, nr, nphi, nphi) slice
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

    Interp4dAxis ax;
    if (t > g->r_max){ ax.i = 0; ax.ip = 0; ax.w = 0.; ax.mw = 0.; return ax; }

    const double *r_centers = g->r_centers;
    int nr = g->nr;
    int i = (int)((log(t)-g->logr0)*g->inv_dlogr);
    if (i < 0) i = 0;
    else if (i > nr-2) i = nr-2;
    if (i < nr-2 && r_centers[i+1] <= t) i++;
    else if (i > 0 && r_centers[i] > t) i--;

    double w = (t-r_centers[i])/(r_centers[i+1]-r_centers[i]);
    if (t < g->r_min){ i = 0; w = 0.; }
    ax.i = i; ax.ip = i+1; ax.w = w; ax.mw = 1.-w;
    return ax;
}

// Uniform angular axis, wrapping periodically mod 2*pi.
static inline __attribute__((always_inline)) Interp4dAxis interp4d_axis_phi(
    const Interp4dGrid *g, double ang){

    double twopi = 2.*M_PI;
    ang = fmod(ang, twopi);
    if (ang < 0.) ang += twopi;

    const double *phi_centers = g->phi_centers;
    int nphi = g->nphi;
    int i = (int)((ang-g->phi0)*g->inv_dphi);
    if (i < 0) i = 0;
    else if (i > nphi-2) i = nphi-2;
    if (i < nphi-2 && phi_centers[i+1] <= ang) i++;
    else if (i > 0 && phi_centers[i] > ang) i--;

    Interp4dAxis ax;
    ax.w = (ang-phi_centers[i])/(phi_centers[i+1]-phi_centers[i]);
    ax.mw = 1.-ax.w;
    ax.i = i; ax.ip = (i+1)%nphi;
    return ax;
}

// The 16-corner sum over the four resolved axes, as bilinear sums over the angular axes at the
// four radial corners
static inline __attribute__((always_inline)) double interp4d_gather(
    const double *q_slice, const Interp4dGrid *g, const Interp4dAxis *a2,
    const Interp4dAxis *a3, const Interp4dAxis *a12, const Interp4dAxis *a13){

    int nphi = g->nphi;
    double w2 = a2->w, mw2 = a2->mw, w3 = a3->w, mw3 = a3->mw;
    double w12 = a12->w, mw12 = a12->mw, w13 = a13->w, mw13 = a13->mw;
    int i13 = a13->i, i13p = a13->ip;

    // Offsets of the lower and upper nodes on the t2, t3 and a12 axes
    long b_i2 = (long)a2->i*g->s_i2;
    long b_i2p = (long)a2->ip*g->s_i2;
    long b_i3 = (long)a3->i*g->s_i3;
    long b_i3p = (long)a3->ip*g->s_i3;
    long b_i12 = (long)a12->i*nphi;
    long b_i12p = (long)a12->ip*nphi;

    // Products of the corner weights, shared by the four radial corners
    double m2m3 = mw2*mw3, w2m3 = w2*mw3, m2w3 = mw2*w3, w2w3 = w2*w3;
    double m12m13 = mw12*mw13, m12w13 = mw12*w13, w12m13 = w12*mw13, w12w13 = w12*w13;
    const double *q00 = q_slice+b_i2+b_i3, *q10 = q_slice+b_i2p+b_i3;
    const double *q01 = q_slice+b_i2+b_i3p, *q11 = q_slice+b_i2p+b_i3p;
    double s00 = m12m13*q00[b_i12+i13] + m12w13*q00[b_i12+i13p] + w12m13*q00[b_i12p+i13] + w12w13*q00[b_i12p+i13p];
    double s10 = m12m13*q10[b_i12+i13] + m12w13*q10[b_i12+i13p] + w12m13*q10[b_i12p+i13] + w12w13*q10[b_i12p+i13p];
    double s01 = m12m13*q01[b_i12+i13] + m12w13*q01[b_i12+i13p] + w12m13*q01[b_i12p+i13] + w12w13*q01[b_i12p+i13p];
    double s11 = m12m13*q11[b_i12+i13] + m12w13*q11[b_i12+i13p] + w12m13*q11[b_i12p+i13] + w12w13*q11[b_i12p+i13p];

    return m2m3*s00 + w2m3*s10 + m2w3*s01 + w2w3*s11;
}

// Quadrilinear lookup q(t2, t3, a12, a13) on a (nr, nr, nphi, nphi) slice
static inline __attribute__((always_inline)) double interp4d_point(
    const double *q_slice, const Interp4dGrid *g,
    double t2, double t3, double a12, double a13){

    Interp4dAxis ax2 = interp4d_axis_r(g, t2);
    Interp4dAxis ax3 = interp4d_axis_r(g, t3);
    Interp4dAxis ax12 = interp4d_axis_phi(g, a12);
    Interp4dAxis ax13 = interp4d_axis_phi(g, a13);
    return interp4d_gather(q_slice, g, &ax2, &ax3, &ax12, &ax13);
}

// Quadrilinear interpolation of a (nr, nr, nphi, nphi) quadruplet-density slice at n points,
// log-uniform in the two radial and periodic in the two angular axes
// * q_slice: (nr, nr, nphi, nphi) over (theta2, theta3, phi12, phi13)
// * t2f, t3f, a12f, a13f, out: (n), out is overwritten
void interp4d(
    const double *q_slice, const double *r_centers, const double *phi_centers,
    int nr, int nphi, double r_min, double r_max, double logr0, double inv_dlogr,
    double phi0, double inv_dphi,
    const double *t2f, const double *t3f, const double *a12f, const double *a13f,
    int n, double *out){

    Interp4dGrid g = interp4d_grid(r_centers, phi_centers, nr, nphi,
        r_min, r_max, logr0, inv_dlogr, phi0, inv_dphi);

    for (int idx=0; idx<n; idx++)
        out[idx] = interp4d_point(q_slice, &g, t2f[idx], t3f[idx],
            a12f[idx], a13f[idx]);
}

///////////////////////////////
// Quadruplet (SVA) integral //
///////////////////////////////

// Notation: pair a has length theta1 and angle varphi1, pair b length theta2 and angle varphi2, and
// phi links the two pairs along the x axis. xi_a and xi_b are evaluated at the separations
// psi_a = phi - theta1 e(varphi1) and psi_b = phi + theta2 e(varphi2), of polar angles varphi_a and
// varphi_b, and q at the spokes (phi, theta1, |psi_b|) of angles (0, varphi1, varphi_b). In the
// names, t1, t2 stand for theta1, theta2 (whose subgrid nodes are phi1_nodes, phi2_nodes) and
// v1, v2, va, vb for varphi1, varphi2, varphi_a, varphi_b.

// The varphi2 sums of one callback on the (varphi1, varphi2) grid of one (theta1, theta2) pair,
// one per varphi1. They depend on wid and xi_b only, so callbacks that differ in xi_a share them.
// wid is constant per instantiation, so the inner loop is branch free and vectorises.
#define SVA_V2SUM_BODY(WEXPR) \
    for (int iv1=0; iv1<nv1; iv1++){ \
        const double *qrow = q_node + (long)iv1*nv2c; \
        const double *w1row = wtab1 + (long)iv1*nv2c; \
        const double *w2row = wtab2 + (long)iv1*nv2c; \
        const double *w3row = wtab3 + (long)iv1*nv2c; \
        double s = 0.; \
        (void)w1row; (void)w2row; (void)w3row; \
        _Pragma("omp simd reduction(+:s)") \
        for (int iv2=0; iv2<nv2c; iv2++) \
            s += qrow[iv2]*xib[iv2]*(WEXPR); \
        v2sums[iv1] = s; \
    }

static inline __attribute__((always_inline)) void sva_v2sums(
    int wid, const double *q_node, const double *xib,
    const double *wtab1, const double *wtab2, const double *wtab3,
    int nv1, int nv2c, double *v2sums){

    switch (wid){
    case 0: { SVA_V2SUM_BODY(1.) } break;
    case 1: { SVA_V2SUM_BODY(w1row[iv2]) } break;
    case 2: { SVA_V2SUM_BODY(w2row[iv2]) } break;
    case 3: { SVA_V2SUM_BODY(w3row[iv2]) } break;
    default: { SVA_V2SUM_BODY(1.) } break; // wid 4 is folded into xib
    }
}

// Cosmic-variance (r-term) integral of all callbacks over the phi grid: outer products of the
// side integrals over psi_a and psi_b, weighted by the quadruplet density (q_mode 0: tensor,
// 1: constant, 2: separable). Output oi keeps the (theta1, theta2) elements of mask mask_idx[oi]
// of pair_masks, and a callback is evaluated only where its output keeps the element. The
// per-thread accumulators are summed in thread order, so the result is reproducible run to run.
// * phi_grid, phi_w: (n_phi). phi1_nodes, phi1_weights: (n_r, ns1). phi2_nodes, phi2_weights:
//   (n_r, ns2). v1_grid, v1_w: (nv1). v2_grid, v2_w: (nv2)
// * tiles: (n_tiles, 2), the first theta1 and theta2 bin of each tile of nr_block bins
// * q_mode 0: q_array (q_nr, q_nr, q_nr, q_nphi, q_nphi) over (theta1, theta2, theta3, phi12, phi13)
//   on the log-uniform q_r_centers and the uniform q_phi_centers
// * q_mode 1: q = q_const
// * q_mode 2: q = q_const sup_t1 sup_t2 sup_t3, with sup_t1 (n_phi), sup_t2 (n_r, ns1) and
//   sup_t3 (n_phi, n_r, ns2, nv2)
// * xi_y_tables: (n_xi_unique, xi_n) on the log-uniform xi_x
// * Callbacks, each (n_cb): xi tables xi_a_idx and xi_b_idx of the two pairs, angular weight w_ids
//   (0-4 for rp0, rp1, rm0, rm1, rpm), output out_idx and factor wick_fac
// * symmetric_mask, mask_idx: (n_out). pair_masks: (n_mask, n_r, n_r)
// * results: (n_out, n_r, n_r), added to, so the caller zeroes it
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
    int n_out, const int *symmetric_mask,
    int n_mask, const int *pair_masks, const int *mask_idx,
    double *results, int nthreads, int verbose){

    // Sizes of one output block, of the largest (theta1, theta2) tile and of the q slice at one phi
    long block_size = (long)n_r*n_r;
    long acc_len = (long)n_out*block_size;
    int nrbmax = nr_block < n_r ? nr_block : n_r;
    int nt1max = nrbmax*ns1, nt2max = nrbmax*ns2;
    int nv2max = v2_block < nv2 ? v2_block : nv2;
    int nt2tot = n_r*ns2;
    long slice_len = (long)q_nr*q_nr*q_nphi*q_nphi;

    // Interpolation grid of the q tensor, q_mode 0 only
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
    double *cos4v2 = (double *)orpheus_malloc((size_t)nv2*sizeof(double));
    double *sin4v2 = (double *)orpheus_malloc((size_t)nv2*sizeof(double));
    double *thread_accums = (double *)orpheus_calloc((size_t)nthreads*acc_len, sizeof(double));

    // q-grid axes of the theta1 nodes and of the varphi1 grid, which do not depend on phi
    Interp4dAxis *axis_t1 = (q_mode == 0 ? (Interp4dAxis *)orpheus_malloc(
        (size_t)n_r*ns1*sizeof(Interp4dAxis)) : NULL);
    Interp4dAxis *axis_v1 = (q_mode == 0 ? (Interp4dAxis *)orpheus_malloc(
        (size_t)nv1*sizeof(Interp4dAxis)) : NULL);

    // The per-thread scratch below is the largest allocation of the module: nothing is
    // touched before the whole team has allocated, and check_clib_error() raises otherwise
    int alloc_ok = (cos_v1 && sin_v1 && cos4v1 && sin4v1
        && cos_v2 && sin_v2 && cos4v2 && sin4v2 && thread_accums
        && (q_mode != 0 || (axis_t1 && axis_v1)));
    if (alloc_ok && q_mode == 0){
        for (int j=0; j<n_r*ns1; j++)
            axis_t1[j] = interp4d_axis_r(&grid, phi1_nodes[j]);
        for (int j=0; j<nv1; j++)
            axis_v1[j] = interp4d_axis_phi(&grid, v1_grid[j]);
    }
    if (alloc_ok){
        for (int j=0; j<nv1; j++){
            cos_v1[j] = cos(v1_grid[j]);
            sin_v1[j] = sin(v1_grid[j]);
            cos4v1[j] = cos(4.*v1_grid[j]);
            sin4v1[j] = sin(4.*v1_grid[j]);
        }
        for (int j=0; j<nv2; j++){
            cos_v2[j] = cos(v2_grid[j]);
            sin_v2[j] = sin(v2_grid[j]);
            cos4v2[j] = cos(4.*v2_grid[j]);
            sin4v2[j] = sin(4.*v2_grid[j]);
        }
    }
    reset_progress();
    int ndone = 0;

    // Each thread integrates its static share of the phi nodes into its own thread_accum
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        double *thread_accum = thread_accums + (long)tid*acc_len;

        // Per-thread scratch, sized for the largest tile and varphi2 chunk
        double *abs_psia = (double *)orpheus_malloc((size_t)nt1max*nv1*sizeof(double));
        double *varphi_a = (double *)orpheus_malloc((size_t)nt1max*nv1*sizeof(double));
        double *cos4va = (double *)orpheus_malloc((size_t)nt1max*nv1*sizeof(double));
        double *sin4va = (double *)orpheus_malloc((size_t)nt1max*nv1*sizeof(double));
        double *abs_psib = (double *)orpheus_malloc((size_t)nt2max*nv2max*sizeof(double));
        double *varphi_b = (double *)orpheus_malloc((size_t)nt2max*nv2max*sizeof(double));
        double *cos4vb = (double *)orpheus_malloc((size_t)nt2max*nv2max*sizeof(double));
        double *sin4vb = (double *)orpheus_malloc((size_t)nt2max*nv2max*sizeof(double));
        double *xi_a_tab = (double *)orpheus_malloc(
            (size_t)n_xi_unique*nt1max*nv1*sizeof(double));
        double *xi_b_tab = (double *)orpheus_malloc(
            (size_t)n_xi_unique*nt2max*nv2max*sizeof(double));
        double *wtab3 = (double *)orpheus_malloc((size_t)nv1*nv2max*sizeof(double));
        double *wtab4 = (double *)orpheus_malloc((size_t)nt2max*nv2max*sizeof(double));
        double *wtab1 = (double *)orpheus_malloc((size_t)nv1*nv2max*sizeof(double));
        double *wtab2 = (double *)orpheus_malloc((size_t)nv1*nv2max*sizeof(double));
        double *xib_w4 = (double *)orpheus_malloc((size_t)nv2max*sizeof(double));
        double *v2sums = (double *)orpheus_malloc((size_t)nv1*sizeof(double));
        double *q_vals = (double *)orpheus_malloc(
            (size_t)nt1max*nt2max*nv1*nv2max*sizeof(double));
        double *tile_accum = (double *)orpheus_malloc((size_t)n_out*nt1max*nt2max*sizeof(double));
        double *phi_accum = (double *)orpheus_malloc((size_t)acc_len*sizeof(double));
        double *q_slice = (q_mode == 0
            ? (double *)orpheus_malloc((size_t)slice_len*sizeof(double)) : NULL);
        Interp4dAxis *axis_psib = (q_mode == 0 ? (Interp4dAxis *)orpheus_malloc(
            (size_t)nt2max*nv2max*sizeof(Interp4dAxis)) : NULL);
        Interp4dAxis *axis_varphib = (q_mode == 0 ? (Interp4dAxis *)orpheus_malloc(
            (size_t)nt2max*nv2max*sizeof(Interp4dAxis)) : NULL);
        int *tile_mask = (int *)orpheus_malloc((size_t)n_mask*nt1max*nt2max*sizeof(int));
        int *tile_keep = (int *)orpheus_malloc((size_t)nt1max*nt2max*sizeof(int));

        // A thread that failed to allocate clears alloc_ok for the whole team
        if (!(abs_psia && varphi_a && cos4va && sin4va && abs_psib && varphi_b && cos4vb && sin4vb
            && xi_a_tab && xi_b_tab && wtab3 && wtab4 && wtab1 && wtab2
            && xib_w4 && v2sums && q_vals && tile_accum && phi_accum && tile_mask && tile_keep
            && (q_mode != 0 || (q_slice && axis_psib && axis_varphib)))){
            #pragma omp atomic write
            alloc_ok = 0;
        }
        #pragma omp barrier

        // A failed team still enters the `omp for`, whose barrier is collective, with zero trips
        int n_phi_run = alloc_ok ? n_phi : 0;

        #pragma omp for schedule(static)
        for (int iphi=0; iphi<n_phi_run; iphi++){
            double phi = phi_grid[iphi];

            // Get density slice at t1 = phi
            // zero above the q grid, q_array[0, ...] below it, linear in between
            if (q_mode == 0){
                if (phi >= grid.r_max){
                    memset(q_slice, 0, (size_t)slice_len*sizeof(double));
                } else if (phi <= grid.r_min){
                    memcpy(q_slice, q_array, (size_t)slice_len*sizeof(double));
                } else {
                    int i = 0;
                    while (i < q_nr && q_r_centers[i] < phi) i++;
                    i--;
                    if (i > q_nr-2) i = q_nr-2;
                    double w = (phi-q_r_centers[i])
                        /(q_r_centers[i+1]-q_r_centers[i]);
                    const double *q_lo = q_array + (long)i*slice_len;
                    const double *q_hi = q_lo + slice_len;
                    for (long j=0; j<slice_len; j++)
                        q_slice[j] = (1.-w)*q_lo[j] + w*q_hi[j];
                }
            }
            double sup_phi = q_mode == 2 ? sup_t1[iphi] : 1.;
            const double *sup_t3_phi = (q_mode == 2
                ? sup_t3 + (long)iphi*nt2tot*nv2 : sup_t3);

            memset(phi_accum, 0, (size_t)acc_len*sizeof(double));

            // Tiles of nrb1 x nrb2 bins from the bins (ib1_0, ib2_0), with nt1 x nt2 theta nodes
            for (int itile=0; itile<n_tiles; itile++){
                int ib1_0 = tiles[2*itile], ib2_0 = tiles[2*itile+1];
                int nrb1 = nr_block < n_r-ib1_0 ? nr_block : n_r-ib1_0;
                int nrb2 = nr_block < n_r-ib2_0 ? nr_block : n_r-ib2_0;
                int nt1 = nrb1*ns1, nt2 = nrb2*ns2;
                const double *t1_nodes = phi1_nodes + (long)ib1_0*ns1;
                const double *t2_nodes = phi2_nodes + (long)ib2_0*ns2;
                int npsia = nt1*nv1;

                // psi_a on the (theta1, varphi1) nodes of the tile, for all varphi2 chunks
                for (int it1=0; it1<nt1; it1++){
                    for (int iv1=0; iv1<nv1; iv1++){
                        double psia_x = phi - t1_nodes[it1]*cos_v1[iv1];
                        double psia_y = -t1_nodes[it1]*sin_v1[iv1];
                        int j = it1*nv1 + iv1;
                        abs_psia[j] = hypot(psia_x, psia_y);
                        varphi_a[j] = atan2(psia_y, psia_x);
                        cos4va[j] = cos(4.*varphi_a[j]);
                        sin4va[j] = sin(4.*varphi_a[j]);
                    }
                }
                // xi_a of every distinct table at |psi_a|
                for (int t=0; t<n_xi_unique; t++){
                    const double *yi = xi_y_tables + (long)t*xi_n;
                    double *dst = xi_a_tab + (long)t*npsia;
                    for (int j=0; j<npsia; j++)
                        dst[j] = interp1d_eval(xi_x, yi, xi_n,
                            xi_logx0, xi_inv_dlogx, abs_psia[j]);
                }

                // Masks of the tile on its (theta1, theta2) nodes, and the nodes that any
                // output keeps. The others are skipped.
                long nt12 = (long)nt1*nt2;
                for (int it1=0; it1<nt1; it1++){
                    for (int it2=0; it2<nt2; it2++){
                        long ielem = (long)(ib1_0 + it1/ns1)*n_r + ib2_0 + it2/ns2;
                        int it12 = it1*nt2 + it2, keep = 0;
                        for (int im=0; im<n_mask; im++){
                            tile_mask[im*nt12 + it12] = pair_masks[im*block_size + ielem];
                            keep |= pair_masks[im*block_size + ielem];
                        }
                        tile_keep[it12] = keep;
                    }
                }

                // Chunks of nv2c <= v2_block varphi2 nodes from iv2_0
                for (int iv2_0=0; iv2_0<nv2; iv2_0+=v2_block){
                    int nv2c = v2_block < nv2-iv2_0 ? v2_block : nv2-iv2_0;
                    int npsib = nt2*nv2c;
                    const double *cos4v2c = cos4v2 + iv2_0;
                    const double *sin4v2c = sin4v2 + iv2_0;
                    const double *v2_wc = v2_w + iv2_0;

                    // psi_b on the (theta2, varphi2) nodes of the chunk
                    for (int it2=0; it2<nt2; it2++){
                        for (int iv2=0; iv2<nv2c; iv2++){
                            double psib_x = phi + t2_nodes[it2]*cos_v2[iv2_0+iv2];
                            double psib_y = t2_nodes[it2]*sin_v2[iv2_0+iv2];
                            int j = it2*nv2c + iv2;
                            abs_psib[j] = hypot(psib_x, psib_y);
                            varphi_b[j] = atan2(psib_y, psib_x);
                            cos4vb[j] = cos(4.*varphi_b[j]);
                            sin4vb[j] = sin(4.*varphi_b[j]);
                        }
                    }
                    // xi_b of every distinct table at |psi_b|
                    for (int t=0; t<n_xi_unique; t++){
                        const double *yi = xi_y_tables + (long)t*xi_n;
                        double *dst = xi_b_tab + (long)t*npsib;
                        for (int j=0; j<npsib; j++)
                            dst[j] = interp1d_eval(xi_x, yi, xi_n,
                                xi_logx0, xi_inv_dlogx, abs_psib[j]);
                    }
                    // q-grid axes of |psi_b| and varphi_b
                    if (q_mode == 0){
                        for (int j=0; j<npsib; j++){
                            axis_psib[j] = interp4d_axis_r(&grid, abs_psib[j]);
                            axis_varphib[j] = interp4d_axis_phi(&grid, varphi_b[j]);
                        }
                    }

                    // Weight tables independent of (theta1, theta2): wtab3 = cos(4*(v1-v2)) and
                    // wtab4 = 2*cos(4*(v2-vb))
                    for (int iv1=0; iv1<nv1; iv1++){
                        for (int iv2=0; iv2<nv2c; iv2++)
                            wtab3[iv1*nv2c + iv2] = cos4v1[iv1]*cos4v2c[iv2]
                                + sin4v1[iv1]*sin4v2c[iv2];
                    }
                    for (int it2=0; it2<nt2; it2++)
                        for (int iv2=0; iv2<nv2c; iv2++){
                            int idx = it2*nv2c + iv2;
                            wtab4[idx] = 2.*(cos4v2c[iv2]*cos4vb[idx]
                                + sin4v2c[iv2]*sin4vb[idx]);
                        }

                    // q on the (theta1, |psi_b|, varphi1, varphi_b) tile at the nodes that any
                    // output keeps: from the tensor (q_mode 0), constant (1) or separable (2)
                    long q_stride = (long)nv1*nv2c;
                    if (q_mode == 0){
                        // Quadrilinear lookup on the axes resolved once per tile and per run
                        for (int it1=0; it1<nt1; it1++){
                            const Interp4dAxis *ax_t1 = &axis_t1[ib1_0*ns1 + it1];
                            for (int it2=0; it2<nt2; it2++){
                                if (!tile_keep[it1*nt2 + it2]) continue;
                                double *dst = q_vals
                                    + ((long)it1*nt2 + it2)*q_stride;
                                for (int iv1=0; iv1<nv1; iv1++){
                                    const Interp4dAxis *ax_v1 = &axis_v1[iv1];
                                    double *row = dst + (long)iv1*nv2c;
                                    for (int iv2=0; iv2<nv2c; iv2++){
                                        int j = it2*nv2c + iv2;
                                        row[iv2] = interp4d_gather(q_slice, &grid,
                                            ax_t1, &axis_psib[j], ax_v1, &axis_varphib[j]);
                                    }
                                }
                            }
                        }
                    } else if (q_mode == 1){
                        long ntot = (long)nt1*nt2*q_stride;
                        for (long j=0; j<ntot; j++) q_vals[j] = q_const;
                    } else {
                        // q_const s(phi) s(theta1) s(theta2), with sup_t3 already on varphi2
                        for (int it1=0; it1<nt1; it1++){
                            double sup12 = sup_phi*sup_t2[ib1_0*ns1 + it1];
                            for (int it2=0; it2<nt2; it2++){
                                if (!tile_keep[it1*nt2 + it2]) continue;
                                const double *sup_t3_row = sup_t3_phi
                                    + (long)(ib2_0*ns2 + it2)*nv2 + iv2_0;
                                double *dst = q_vals
                                    + ((long)it1*nt2 + it2)*q_stride;
                                for (int iv1=0; iv1<nv1; iv1++){
                                    double *row = dst + (long)iv1*nv2c;
                                    for (int iv2=0; iv2<nv2c; iv2++)
                                        row[iv2] = q_const*(sup12*sup_t3_row[iv2]);
                                }
                            }
                        }
                    }

                    // Fold the varphi quadrature weights into q
                    for (long it12=0; it12<(long)nt1*nt2; it12++){
                        if (!tile_keep[it12]) continue;
                        double *dst = q_vals + it12*q_stride;
                        for (int iv1=0; iv1<nv1; iv1++){
                            double w1 = v1_w[iv1];
                            double *row = dst + (long)iv1*nv2c;
                            for (int iv2=0; iv2<nv2c; iv2++)
                                row[iv2] *= w1*v2_wc[iv2];
                        }
                    }

                    memset(tile_accum, 0, (size_t)n_out*nt1*nt2*sizeof(double));

                    // Callback loop. The (theta1, theta2) pair is the outer level, so the q
                    // slice and the weight tables are built once for all n_cb callbacks;
                    // moving the callbacks outside the r and varphi2 tiling measured 2x slower.
                    // A callback is evaluated only where its output keeps the node. The
                    // callbacks come sorted by (wid, xi_b), and the varphi2 sums are recomputed
                    // only when that pair changes.
                    for (int it1=0; it1<nt1; it1++){
                        for (int it2=0; it2<nt2; it2++){
                            int it12 = it1*nt2 + it2;
                            if (!tile_keep[it12]) continue;
                            const double *q_node = q_vals + (long)it12*q_stride;
                            const double *cos4vb_row = cos4vb + it2*nv2c;
                            const double *sin4vb_row = sin4vb + it2*nv2c;

                            // wtab1 = cos(4*(va-vb)) and wtab2 = cos(4*(v1+v2-va-vb)) from the
                            // cos and sin of the 4-fold angles, with c12, s12 those of 4*(v1+v2)
                            for (int iv1=0; iv1<nv1; iv1++){
                                double ca = cos4va[it1*nv1 + iv1];
                                double sa = sin4va[it1*nv1 + iv1];
                                double c1 = cos4v1[iv1], s1 = sin4v1[iv1];
                                double *w1row = wtab1 + (long)iv1*nv2c;
                                double *w2row = wtab2 + (long)iv1*nv2c;
                                for (int iv2=0; iv2<nv2c; iv2++){
                                    double cb = cos4vb_row[iv2], sb = sin4vb_row[iv2];
                                    double c12 = c1*cos4v2c[iv2] - s1*sin4v2c[iv2];
                                    double s12 = s1*cos4v2c[iv2] + c1*sin4v2c[iv2];
                                    w1row[iv2] = ca*cb + sa*sb;
                                    w2row[iv2] = c12*(ca*cb - sa*sb)
                                        + s12*(sa*cb + ca*sb);
                                }
                            }

                            // v2sums of the current (wid, xi_b), the w=4 weight folded into xi_b
                            int v2sums_wid = -1, v2sums_xib = -1;
                            for (int icb=0; icb<n_cb; icb++){
                                if (!tile_mask[mask_idx[out_idx[icb]]*nt12 + it12]) continue;
                                if (w_ids[icb] != v2sums_wid || xi_b_idx[icb] != v2sums_xib){
                                    v2sums_wid = w_ids[icb];
                                    v2sums_xib = xi_b_idx[icb];
                                    const double *xib = xi_b_tab
                                        + (long)v2sums_xib*npsib + it2*nv2c;
                                    if (v2sums_wid == 4){
                                        const double *w4row = wtab4 + it2*nv2c;
                                        for (int j=0; j<nv2c; j++)
                                            xib_w4[j] = xib[j]*w4row[j];
                                        xib = xib_w4;
                                    }
                                    sva_v2sums(v2sums_wid, q_node, xib, wtab1, wtab2, wtab3,
                                        nv1, nv2c, v2sums);
                                }
                                // Contract with xi_a over varphi1
                                const double *xia = xi_a_tab
                                    + (long)xi_a_idx[icb]*npsia + it1*nv1;
                                double sum = 0.;
                                for (int iv1=0; iv1<nv1; iv1++)
                                    sum += xia[iv1]*v2sums[iv1];
                                tile_accum[out_idx[icb]*nt12 + it12] += wick_fac[icb]*sum;
                            }
                        }
                    }

                    // Contract the phi subgrid nodes back onto the coarse bins
                    for (int oi=0; oi<n_out; oi++){
                        const double *acc = tile_accum + (long)oi*nt1*nt2;
                        double *dst = phi_accum + (long)oi*block_size;
                        for (int ib1=0; ib1<nrb1; ib1++){
                            for (int ib2=0; ib2<nrb2; ib2++){
                                double s = 0.;
                                for (int is1=0; is1<ns1; is1++){
                                    double w1 = phi1_weights[(ib1_0+ib1)*ns1 + is1];
                                    for (int is2=0; is2<ns2; is2++)
                                        s += acc[(long)(ib1*ns1+is1)*nt2 + ib2*ns2+is2]
                                            *w1*phi2_weights[(ib2_0+ib2)*ns2 + is2];
                                }
                                dst[(long)(ib1_0+ib1)*n_r + ib2_0+ib2] += s;
                            }
                        }
                    }
                }
            }

            // Symmetrise and fold this phi node into the thread total, within the mask of each
            // output
            double phi_weight = phi*phi_w[iphi];
            double half_weight = .5*phi_weight;
            for (int oi=0; oi<n_out; oi++){
                const double *cur = phi_accum + (long)oi*block_size;
                const int *mask = pair_masks + (long)mask_idx[oi]*block_size;
                double *dst = thread_accum + (long)oi*block_size;
                if (symmetric_mask[oi]){
                    for (int i=0; i<n_r; i++)
                        for (int j=0; j<n_r; j++)
                            if (mask[i*n_r+j])
                                dst[i*n_r+j] += half_weight*(cur[i*n_r+j] + cur[j*n_r+i]);
                } else {
                    for (long j=0; j<block_size; j++)
                        if (mask[j]) dst[j] += phi_weight*cur[j];
                }
            }
            #pragma omp atomic
            ndone += 1;
            print_progress(ndone, n_phi, verbose);
        }

        free(abs_psia); free(varphi_a); free(cos4va); free(sin4va);
        free(abs_psib); free(varphi_b); free(cos4vb); free(sin4vb);
        free(xi_a_tab); free(xi_b_tab);
        free(wtab3); free(wtab4); free(wtab1); free(wtab2); free(xib_w4); free(v2sums);
        free(q_vals); free(tile_accum); free(phi_accum); free(q_slice);
        free(axis_psib); free(axis_varphib); free(tile_mask); free(tile_keep);
    }
    if (verbose > 0){ printf("\n"); fflush(stdout); }

    // Sum the per-thread accumulators in thread order
    for (int t=0; alloc_ok && t<nthreads; t++){
        const double *src = thread_accums + (long)t*acc_len;
        for (long j=0; j<acc_len; j++) results[j] += src[j];
    }

    free(thread_accums);
    free(cos_v1); free(sin_v1); free(cos4v1); free(sin4v1);
    free(cos_v2); free(sin_v2); free(cos4v2); free(sin4v2);
    free(axis_t1); free(axis_v1);
}


/////////////////////////////////
// Continuous-limit estimators //
/////////////////////////////////

// Phi integral of one mixed term on the fine grid, fine[i1, i2] = sum_k R w xi(r_jk) dphi_k,
// with R the measured over the idealised triplet counts, interpolated from (nr_ratio, nr_ratio,
// nphi_ratio) nodes, log-uniform in theta and uniform in phi, or R = 1 for nr_ratio = 0. The
// progress counter ndone runs up to progtot rows, so that several calls fill one bar.
static void continuous_q_fine(
    const double *fine_centers, int n_fine, const double *phi_centers, const double *dphi,
    int n_phi, const double *xi_x, int xi_n, double xi_logx0, double xi_inv_dlogx,
    const double *xi_y, int w_type, const double *ratio, int nr_ratio, int nphi_ratio,
    double ratio_logr0, double ratio_inv_dlogr, double ratio_phi0, double ratio_inv_dphi,
    double *fine, int *ndone, int progtot, int nthreads, int verbose){

    // Trig of the phi nodes
    double *cosphi = (double *)orpheus_malloc((size_t)n_phi*sizeof(double));
    double *sinphi = (double *)orpheus_malloc((size_t)n_phi*sizeof(double));
    double *cos4phi = (double *)orpheus_malloc((size_t)n_phi*sizeof(double));
    if (!(cosphi && sinphi && cos4phi)){
        free(cosphi); free(sinphi); free(cos4phi);
        return;
    }
    for (int iphi=0; iphi<n_phi; iphi++){
        cosphi[iphi] = cos(phi_centers[iphi]);
        sinphi[iphi] = sin(phi_centers[iphi]);
        cos4phi[iphi] = cos(4.*phi_centers[iphi]);
    }

    // Lower interpolation node and weight of every fine theta and phi, clamped to the ratio grid
    int *itheta_lo = NULL, *iphi_lo = NULL;
    double *wtheta = NULL, *wphi = NULL;
    if (nr_ratio > 0){
        itheta_lo = (int *)orpheus_malloc((size_t)n_fine*sizeof(int));
        wtheta = (double *)orpheus_malloc((size_t)n_fine*sizeof(double));
        iphi_lo = (int *)orpheus_malloc((size_t)n_phi*sizeof(int));
        wphi = (double *)orpheus_malloc((size_t)n_phi*sizeof(double));
        if (!(itheta_lo && wtheta && iphi_lo && wphi)){
            free(cosphi); free(sinphi); free(cos4phi);
            free(itheta_lo); free(wtheta); free(iphi_lo); free(wphi);
            return;
        }
        for (int i=0; i<n_fine; i++){
            double x = (log(fine_centers[i])-ratio_logr0)*ratio_inv_dlogr;
            x = x < 0. ? 0. : (x > nr_ratio-1 ? nr_ratio-1 : x);
            itheta_lo[i] = (int)x < nr_ratio-2 ? (int)x : nr_ratio-2;
            wtheta[i] = x-itheta_lo[i];
        }
        for (int iphi=0; iphi<n_phi; iphi++){
            double x = (phi_centers[iphi]-ratio_phi0)*ratio_inv_dphi;
            x = x < 0. ? 0. : (x > nphi_ratio-1 ? nphi_ratio-1 : x);
            iphi_lo[iphi] = (int)x < nphi_ratio-2 ? (int)x : nphi_ratio-2;
            wphi[iphi] = x-iphi_lo[iphi];
        }
    }

    // One task per fine theta1 row
    #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for (int i1=0; i1<n_fine; i1++){
        double theta1 = fine_centers[i1];
        for (int i2=0; i2<n_fine; i2++){
            double theta2 = fine_centers[i2];
            // Ratio rows at the four (theta1, theta2) corners and the weights between them
            const double *r00 = NULL, *r01 = NULL, *r10 = NULL, *r11 = NULL;
            double wt1 = 0., wt2 = 0.;
            if (nr_ratio > 0){
                r00 = ratio + ((size_t)itheta_lo[i1]*nr_ratio + itheta_lo[i2])*nphi_ratio;
                r01 = r00 + nphi_ratio;
                r10 = r00 + (size_t)nr_ratio*nphi_ratio;
                r11 = r10 + nphi_ratio;
                wt1 = wtheta[i1]; wt2 = wtheta[i2];
            }
            // xi at r_jk = |theta1 - theta2 e(phi)|, weighted per w_type as in continuous_q_single
            double acc = 0.;
            for (int iphi=0; iphi<n_phi; iphi++){
                double rjk_x = theta1 - theta2*cosphi[iphi];
                double rjk_y = -theta2*sinphi[iphi];
                double xi = interp1d_eval(xi_x, xi_y, xi_n,
                    xi_logx0, xi_inv_dlogx, hypot(rjk_x, rjk_y));
                double w;
                if (w_type == 0){
                    w = 1.;
                } else if (w_type == 1){
                    w = cos4phi[iphi];
                } else {
                    w = cos(4.*(phi_centers[iphi]-atan2(rjk_y, rjk_x)));
                }
                // Times R, linear in phi and bilinear in (theta1, theta2)
                if (nr_ratio > 0){
                    int ip = iphi_lo[iphi];
                    double wp = wphi[iphi];
                    double v00 = r00[ip] + wp*(r00[ip+1]-r00[ip]);
                    double v01 = r01[ip] + wp*(r01[ip+1]-r01[ip]);
                    double v10 = r10[ip] + wp*(r10[ip+1]-r10[ip]);
                    double v11 = r11[ip] + wp*(r11[ip+1]-r11[ip]);
                    w *= (1.-wt1)*((1.-wt2)*v00 + wt2*v01) + wt1*((1.-wt2)*v10 + wt2*v11);
                }
                acc += w*xi*dphi[iphi];
            }
            fine[(size_t)i1*n_fine + i2] = acc;
        }
        #pragma omp atomic
        *ndone += 1;
        print_progress(*ndone, progtot, verbose);
    }

    free(cosphi); free(sinphi); free(cos4phi);
    free(itheta_lo); free(wtheta); free(iphi_lo); free(wphi);
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

// Continuous q-term (triplet) of one xi on a fine (alpha*n_r) grid, bin-averaged to (n_r, n_r).
// w_type: 0 = qpp (w = 1), 1 = qmm (cos(4*PHI)), 2 = qpm (cos(4*(PHI-phi_jk)))
// * fine_centers, fine_weights: (n_fine), n_fine = alpha*n_r. phi_centers, dphi: (n_phi)
// * xi_y: (xi_n) on the log-uniform xi_x
// * ratio: (nr_ratio, nr_ratio, nphi_ratio) measured over idealised triplet counts on log-uniform
//   theta from ratio_logr0 and uniform phi in [0, pi) from ratio_phi0, unused for nr_ratio = 0
// * out: (n_r, n_r), added to
void continuous_q_single(
    int n_r, int alpha, const double *fine_centers, const double *fine_weights,
    const double *phi_centers, const double *dphi, int n_fine, int n_phi,
    const double *xi_x, int xi_n, double xi_logx0, double xi_inv_dlogx, const double *xi_y,
    int w_type, double sigma_eps2, double area, double nbar,
    const double *ratio, int nr_ratio, int nphi_ratio, double ratio_logr0, double ratio_inv_dlogr,
    double ratio_phi0, double ratio_inv_dphi, double *out, int nthreads, int verbose){

    double *fine = (double *)orpheus_malloc((size_t)n_fine*n_fine*sizeof(double));
    if (!fine) return;
    reset_progress();
    int ndone = 0;
    continuous_q_fine(fine_centers, n_fine, phi_centers, dphi, n_phi,
        xi_x, xi_n, xi_logx0, xi_inv_dlogx, xi_y, w_type, ratio, nr_ratio, nphi_ratio,
        ratio_logr0, ratio_inv_dlogr, ratio_phi0, ratio_inv_dphi, fine, &ndone, n_fine,
        nthreads, verbose);
    if (verbose > 0){ printf("\n"); fflush(stdout); }
    continuous_q_binavg(fine, fine_weights, n_fine, n_r, alpha,
        2.*sigma_eps2/(M_PI*area*nbar), out, nthreads);
    free(fine);
}


// Build q-terms for all tomo combos, in continuous approximation. The calls run one after the
// other, each parallelised over the fine grid.
// * Grids as in continuous_q_single. xi_y_tables: (n_xi_unique, xi_n)
// * Calls, each (n_calls): xi table call_xi_idx, w_type call_w_type, output call_out_idx and
//   factor call_factor
// * results: (n_out, n_r, n_r), added to
void continuous_q_tomo(
    int n_r, int alpha, const double *fine_centers, const double *fine_weights,
    const double *phi_centers, const double *dphi, int n_fine, int n_phi,
    const double *xi_x, int xi_n, double xi_logx0, double xi_inv_dlogx,
    const double *xi_y_tables, int n_xi_unique,
    double sigma_eps2, double area, double nbar,
    int n_calls, const int *call_xi_idx, const int *call_w_type,
    const int *call_out_idx, const double *call_factor,
    int n_out, double *results, int nthreads, int verbose){

    long block_size = (long)n_r*n_r;
    double *fine = (double *)orpheus_malloc((size_t)n_fine*n_fine*sizeof(double));
    double *block = (double *)orpheus_malloc((size_t)block_size*sizeof(double));
    if (!(fine && block)){
        free(fine); free(block);
        return;
    }
    reset_progress();
    int ndone = 0;
    // Each call fills the fine grid, bin-averages it and adds it with its factor to its output
    for (int icall=0; icall<n_calls; icall++){
        continuous_q_fine(fine_centers, n_fine, phi_centers, dphi, n_phi,
            xi_x, xi_n, xi_logx0, xi_inv_dlogx, xi_y_tables + (long)call_xi_idx[icall]*xi_n,
            call_w_type[icall], NULL, 0, 0, 0., 0., 0., 0., fine, &ndone, n_calls*n_fine,
            nthreads, verbose);
        memset(block, 0, (size_t)block_size*sizeof(double));
        continuous_q_binavg(fine, fine_weights, n_fine, n_r, alpha,
            2.*sigma_eps2/(M_PI*area*nbar), block, nthreads);
        double fac = call_factor[icall];
        double *dst = results + (long)call_out_idx[icall]*block_size;
        for (long j=0; j<block_size; j++)
            dst[j] += fac*block[j];
    }
    if (verbose > 0){ printf("\n"); fflush(stdout); }
    free(fine); free(block);
}


// Build r-terms for all tomo combos, in continuous approximation. The side integrals are
// shared across work units, the outer products parallelise over (combo, Wick pairing).
// * theta_centers: (n_r) radial nodes. phi_grid, phi_w: (n_phi). v_grid, v_w: (n_v) on [0, 2 pi).
//   v_half, v_half_w: (n_vh) on [0, pi]
// * xi_y_tables: (n_xi_unique, xi_n) on the log-uniform xi_x
// * Work units, each (n_work): xi_+ and xi_- tables of the two pairs, work_xi_a_p, work_xi_a_m,
//   work_xi_b_p and work_xi_b_m, and work_out_idx, the first of five consecutive outputs
//   rp0, rp1, rm0, rm1, rpm
// * results: (n_out, n_r, n_r), added to
void continuous_rterms_tomo(
    int n_r, const double *theta_centers, const double *phi_grid, const double *phi_w, int n_phi,
    const double *v_grid, const double *v_w, int n_v,
    const double *v_half, const double *v_half_w, int n_vh, double area,
    const double *xi_x, int xi_n, double xi_logx0, double xi_inv_dlogx,
    const double *xi_y_tables, int n_xi_unique,
    int n_work, const int *work_xi_a_p, const int *work_xi_b_p,
    const int *work_xi_a_m, const int *work_xi_b_m, const int *work_out_idx,
    int n_out, double *results, int nthreads, int verbose){

    // Prefactors 1/(2 pi A) of rp1, rm0, rm1, rpm and 2/(pi A) of rp0
    long block_size = (long)n_r*n_r;
    double pre_r = 1./(2.*M_PI*area);
    double pre_rp0 = 2./(M_PI*area);

    // Distinct (xi_p, xi_m) table pairs per side, so that each side integral is evaluated once
    int *a_xip = (int *)orpheus_malloc((size_t)n_work*sizeof(int));
    int *a_xim = (int *)orpheus_malloc((size_t)n_work*sizeof(int));
    int *b_xip = (int *)orpheus_malloc((size_t)n_work*sizeof(int));
    int *b_xim = (int *)orpheus_malloc((size_t)n_work*sizeof(int));
    int *a_of_work = (int *)orpheus_malloc((size_t)n_work*sizeof(int));
    int *b_of_work = (int *)orpheus_malloc((size_t)n_work*sizeof(int));
    int nside_a = 0, nside_b = 0;
    for (int iwork=0; iwork<n_work; iwork++){
        int ip = work_xi_a_p[iwork], im = work_xi_a_m[iwork], found = -1;
        for (int iside=0; iside<nside_a; iside++)
            if (a_xip[iside] == ip && a_xim[iside] == im){ found = iside; break; }
        if (found < 0){ found = nside_a; a_xip[nside_a] = ip; a_xim[nside_a] = im; nside_a++; }
        a_of_work[iwork] = found;

        ip = work_xi_b_p[iwork]; im = work_xi_b_m[iwork]; found = -1;
        for (int iside=0; iside<nside_b; iside++)
            if (b_xip[iside] == ip && b_xim[iside] == im){ found = iside; break; }
        if (found < 0){ found = nside_b; b_xip[nside_b] = ip; b_xim[nside_b] = im; nside_b++; }
        b_of_work[iwork] = found;
    }

    // Side integrals over (phi, distinct table pair), stored as I[(iphi*nside + iside)*n_r + ir]
    long a_stride = (long)nside_a*n_r, b_stride = (long)nside_b*n_r;
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
    if (!(a_xip && a_xim && b_xip && b_xim && a_of_work && b_of_work
        && Ia_rp0 && Ia_rp1_c && Ia_rp1_s && Ia_rm0 && Ia_rm1 && Ia_rpm
        && Ib_rp0 && Ib_rp1_c && Ib_rp1_s && Ib_rm0 && Ib_rm1 && Ib_rpm)){
        free(a_xip); free(a_xim); free(b_xip); free(b_xim); free(a_of_work); free(b_of_work);
        free(Ia_rp0); free(Ia_rp1_c); free(Ia_rp1_s);
        free(Ia_rm0); free(Ia_rm1); free(Ia_rpm);
        free(Ib_rp0); free(Ib_rp1_c); free(Ib_rp1_s);
        free(Ib_rm0); free(Ib_rm1); free(Ib_rpm);
        return;
    }

    if (verbose > 0) printf("  side integrals ");
    reset_progress();
    int nsidedone = 0;
    int nsidetot = n_phi*(nside_a+nside_b);

    // Side a: integrals over varphi at psi_a = phi - theta e(varphi), with xi_+ for rpm
    #pragma omp parallel for schedule(dynamic) collapse(2) num_threads(nthreads)
    for (int iphi=0; iphi<n_phi; iphi++){
        for (int iside=0; iside<nside_a; iside++){
            double phi = phi_grid[iphi];
            const double *xip = xi_y_tables + (long)a_xip[iside]*xi_n;
            const double *xim = xi_y_tables + (long)a_xim[iside]*xi_n;
            long off = ((long)iphi*nside_a + iside)*n_r;

            for (int ir=0; ir<n_r; ir++){
                double theta = theta_centers[ir];

                // rp0: varphi in [0, pi]
                double s_rp0 = 0.;
                for (int iv=0; iv<n_vh; iv++){
                    double abs_psia = sqrt(phi*phi + theta*theta
                        - 2.*phi*theta*cos(v_half[iv]));
                    s_rp0 += interp1d_eval(xi_x, xip, xi_n, xi_logx0,
                        xi_inv_dlogx, abs_psia)*v_half_w[iv];
                }
                Ia_rp0[off+ir] = s_rp0;

                // rp1, rm0, rm1, rpm: varphi in [0, 2*pi]
                double s_rp1_c = 0., s_rp1_s = 0., s_rm0 = 0., s_rm1 = 0., s_rpm = 0.;
                for (int iv=0; iv<n_v; iv++){
                    double varphi = v_grid[iv];
                    double psia_x = phi - theta*cos(varphi), psia_y = -theta*sin(varphi);
                    double abs_psia = hypot(psia_x, psia_y);
                    double varphi_a = atan2(psia_y, psia_x);
                    double xim_val = interp1d_eval(xi_x, xim, xi_n,
                        xi_logx0, xi_inv_dlogx, abs_psia);
                    double xip_val = interp1d_eval(xi_x, xip, xi_n,
                        xi_logx0, xi_inv_dlogx, abs_psia);
                    s_rp1_c += xim_val*cos(4.*varphi_a)*v_w[iv];
                    s_rp1_s += xim_val*sin(4.*varphi_a)*v_w[iv];
                    s_rm0 += xim_val*cos(4.*(varphi-varphi_a))*v_w[iv];
                    s_rm1 += xip_val*cos(4.*varphi)*v_w[iv];
                    s_rpm += xip_val*v_w[iv];
                }
                Ia_rp1_c[off+ir] = s_rp1_c;
                Ia_rp1_s[off+ir] = s_rp1_s;
                Ia_rm0[off+ir] = s_rm0;
                Ia_rm1[off+ir] = s_rm1;
                Ia_rpm[off+ir] = s_rpm;
            }
            #pragma omp atomic
            nsidedone += 1;
            print_progress(nsidedone, nsidetot, verbose);
        }
    }

    // Side b: the same at psi_b = phi + theta e(varphi), with xi_- and 2*cos(4*(varphi-varphi_b))
    // for rpm
    #pragma omp parallel for schedule(dynamic) collapse(2) num_threads(nthreads)
    for (int iphi=0; iphi<n_phi; iphi++){
        for (int iside=0; iside<nside_b; iside++){
            double phi = phi_grid[iphi];
            const double *xip = xi_y_tables + (long)b_xip[iside]*xi_n;
            const double *xim = xi_y_tables + (long)b_xim[iside]*xi_n;
            long off = ((long)iphi*nside_b + iside)*n_r;

            for (int ir=0; ir<n_r; ir++){
                double theta = theta_centers[ir];

                // rp0: varphi in [0, pi]
                double s_rp0 = 0.;
                for (int iv=0; iv<n_vh; iv++){
                    double abs_psib = sqrt(phi*phi + theta*theta
                        + 2.*phi*theta*cos(v_half[iv]));
                    s_rp0 += interp1d_eval(xi_x, xip, xi_n, xi_logx0,
                        xi_inv_dlogx, abs_psib)*v_half_w[iv];
                }
                Ib_rp0[off+ir] = s_rp0;

                // rp1, rm0, rm1, rpm: varphi in [0, 2*pi]
                double s_rp1_c = 0., s_rp1_s = 0., s_rm0 = 0., s_rm1 = 0., s_rpm = 0.;
                for (int iv=0; iv<n_v; iv++){
                    double varphi = v_grid[iv];
                    double psib_x = phi + theta*cos(varphi), psib_y = theta*sin(varphi);
                    double abs_psib = hypot(psib_x, psib_y);
                    double varphi_b = atan2(psib_y, psib_x);
                    double xim_val = interp1d_eval(xi_x, xim, xi_n,
                        xi_logx0, xi_inv_dlogx, abs_psib);
                    double xip_val = interp1d_eval(xi_x, xip, xi_n,
                        xi_logx0, xi_inv_dlogx, abs_psib);
                    s_rp1_c += xim_val*cos(4.*varphi_b)*v_w[iv];
                    s_rp1_s += xim_val*sin(4.*varphi_b)*v_w[iv];
                    s_rm0 += xim_val*cos(4.*(varphi-varphi_b))*v_w[iv];
                    s_rm1 += xip_val*cos(4.*varphi)*v_w[iv];
                    s_rpm += 2.*xim_val*cos(4.*(varphi-varphi_b))*v_w[iv];
                }
                Ib_rp1_c[off+ir] = s_rp1_c;
                Ib_rp1_s[off+ir] = s_rp1_s;
                Ib_rm0[off+ir] = s_rm0;
                Ib_rm1[off+ir] = s_rm1;
                Ib_rpm[off+ir] = s_rpm;
            }
            #pragma omp atomic
            nsidedone += 1;
            print_progress(nsidedone, nsidetot, verbose);
        }
    }
    if (verbose > 0){ printf("\n  outer products "); fflush(stdout); }
    reset_progress();
    int nworkdone = 0;

    // Outer products of the two sides per work unit, integrated over phi with weight phi dphi
    #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for (int iwork=0; iwork<n_work; iwork++){
        double *rp0 = (double *)orpheus_calloc(block_size, sizeof(double));
        double *rp1 = (double *)orpheus_calloc(block_size, sizeof(double));
        double *rm0 = (double *)orpheus_calloc(block_size, sizeof(double));
        double *rm1 = (double *)orpheus_calloc(block_size, sizeof(double));
        double *rpm = (double *)orpheus_calloc(block_size, sizeof(double));
        if (!(rp0 && rp1 && rm0 && rm1 && rpm)){
            free(rp0); free(rp1); free(rm0); free(rm1); free(rpm);
            continue;
        }

        for (int iphi=0; iphi<n_phi; iphi++){
            double phi_weight = phi_w[iphi]*phi_grid[iphi];
            long off_a = ((long)iphi*nside_a + a_of_work[iwork])*n_r;
            long off_b = ((long)iphi*nside_b + b_of_work[iwork])*n_r;

            for (int i1=0; i1<n_r; i1++){
                for (int i2=0; i2<n_r; i2++){
                    long idx = (long)i1*n_r + i2;
                    rp0[idx] += phi_weight*Ia_rp0[off_a+i1]*Ib_rp0[off_b+i2];
                    rp1[idx] += phi_weight*(Ia_rp1_c[off_a+i1]*Ib_rp1_c[off_b+i2]
                        + Ia_rp1_s[off_a+i1]*Ib_rp1_s[off_b+i2]);
                    rm0[idx] += phi_weight*Ia_rm0[off_a+i1]*Ib_rm0[off_b+i2];
                    rm1[idx] += phi_weight*Ia_rm1[off_a+i1]*Ib_rm1[off_b+i2];
                    rpm[idx] += phi_weight*Ia_rpm[off_a+i1]*Ib_rpm[off_b+i2];
                }
            }
        }

        // Apply prefactors and accumulate into output
        int out0 = work_out_idx[iwork];
        #pragma omp critical
        {
            double *dst;
            dst = results + (long)(out0+0)*block_size;
            for (long j=0; j<block_size; j++) dst[j] += pre_rp0*rp0[j];
            dst = results + (long)(out0+1)*block_size;
            for (long j=0; j<block_size; j++) dst[j] += pre_r*rp1[j];
            dst = results + (long)(out0+2)*block_size;
            for (long j=0; j<block_size; j++) dst[j] += pre_r*rm0[j];
            dst = results + (long)(out0+3)*block_size;
            for (long j=0; j<block_size; j++) dst[j] += pre_r*rm1[j];
            dst = results + (long)(out0+4)*block_size;
            for (long j=0; j<block_size; j++) dst[j] += pre_r*rpm[j];
        }

        free(rp0); free(rp1); free(rm0); free(rm1); free(rpm);
        #pragma omp atomic
        nworkdone += 1;
        print_progress(nworkdone, n_work, verbose);
    }
    if (verbose > 0){ printf("\n"); fflush(stdout); }

    free(a_xip); free(a_xim); free(b_xip); free(b_xim); free(a_of_work); free(b_of_work);
    free(Ia_rp0); free(Ia_rp1_c); free(Ia_rp1_s);
    free(Ia_rm0); free(Ia_rm1); free(Ia_rpm);
    free(Ib_rp0); free(Ib_rp1_c); free(Ib_rp1_s);
    free(Ib_rm0); free(Ib_rm1); free(Ib_rpm);
}


//////////////////////////////////
// Phi-reduction of mixed terms //
//////////////////////////////////

// Sum of the triplet counts against weight*xi(r_jk) over phi, for one mixed term
// * Nt: (nr, nr, nphi) over (theta1, theta2, phi). tc: (nr). phic: (nphi)
// * weight_type as w_type of continuous_q_single. xi_y: (xi_n) on the log-uniform xi_x
// * out: (nr, nr), overwritten
void triplet_term_reduce(
    const double *Nt, int nr, int nphi, const double *tc, const double *phic, int weight_type,
    const double *xi_x, int xi_n, double xi_logx0, double xi_inv_dlogx,
    const double *xi_y, double *out, int nthreads, int verbose){

    // Trig of the phi nodes
    double *cosphi = (double *)orpheus_malloc((size_t)nphi*sizeof(double));
    double *sinphi = (double *)orpheus_malloc((size_t)nphi*sizeof(double));
    double *cos4phi = (double *)orpheus_malloc((size_t)nphi*sizeof(double));
    if (!(cosphi && sinphi && cos4phi)){
        free(cosphi); free(sinphi); free(cos4phi);
        return;
    }
    for (int iphi=0; iphi<nphi; iphi++){
        cosphi[iphi] = cos(phic[iphi]);
        sinphi[iphi] = sin(phic[iphi]);
        cos4phi[iphi] = cos(4.*phic[iphi]);
    }
    reset_progress();
    int ndone = 0;

    #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for (int i1=0; i1<nr; i1++){
        double theta1 = tc[i1];
        for (int i2=0; i2<nr; i2++){
            double theta2 = tc[i2];
            const double *row = Nt + ((size_t)i1*nr + i2)*nphi;
            // xi at r_jk = |theta1 - theta2 e(phi)|, weighted by the triplet counts
            double acc = 0.;
            for (int iphi=0; iphi<nphi; iphi++){
                double cphi = cosphi[iphi];
                double rjk2 = theta1*theta1 + theta2*theta2 - 2.*theta1*theta2*cphi;
                double rjk = rjk2 > 0. ? sqrt(rjk2) : 0.;
                double xi = interp1d_eval(xi_x, xi_y, xi_n,
                    xi_logx0, xi_inv_dlogx, rjk);
                // Angular weight of qpp, qmm or qpm
                double w;
                if (weight_type == 0){
                    w = 1.;
                } else if (weight_type == 1){
                    w = cos4phi[iphi];
                } else {
                    double rjk_x = theta1 - theta2*cphi;
                    double rjk_y = -theta2*sinphi[iphi];
                    double phi_jk = atan2(rjk_y, rjk_x);
                    w = cos(4.*(phic[iphi]-phi_jk));
                }
                acc += row[iphi]*w*xi;
            }
            out[(size_t)i1*nr + i2] = acc;
        }
        #pragma omp atomic
        ndone += 1;
        print_progress(ndone, nr, verbose);
    }
    if (verbose > 0){ printf("\n"); fflush(stdout); }

    free(cosphi); free(sinphi); free(cos4phi);
}
