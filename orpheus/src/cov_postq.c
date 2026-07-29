// Helpers for computation of tomographic xipm covariance

#include <math.h>
#include <stdlib.h>
#include <string.h>

static inline double interp1d_eval(
    const double *x, const double *y, int n,
    double logx0, double inv_dlogx, double xval)
{
    if (xval <= x[0] || xval >= x[n - 1]) return 0.;
    int i = (int)((log(xval) - logx0) * inv_dlogx);
    if (i < 0)     i = 0;
    if (i > n - 2) i = n - 2;
    if (i < n - 2 && x[i + 1] <= xval) i++;
    else if (i > 0 && x[i] > xval)     i--;
    double w = (xval - x[i]) / (x[i + 1] - x[i]);
    return y[i] + w * (y[i + 1] - y[i]);
}

/*
 * postq_accumulate_tomo
 *
 * q_vals      (nrb1, nrb2, nv1, nv2c)  C-contiguous, from Python/numba
 * abs_a       (nrb1, nv1)              |psi_a| per (ir1, iv1)
 * va          (nrb1, nv1)              angle(psi_a)
 * abs_b       (nrb2, nv2c)             |psi_b| per (ir2, iv2)
 * vb          (nrb2, nv2c)             angle(psi_b)
 * v1_grid     (nv1,)
 * v2_chunk    (nv2c,)
 * xi_x        (xi_n,)                  shared x grid for all xi splines
 * xi_y_tables (n_xi_unique, xi_n)      tabulated xi spline y values
 * n_cb        total number of callbacks
 * xi_a_idx    (n_cb,)  index into xi_y_tables for psi_a lookup
 * xi_b_idx    (n_cb,)  index into xi_y_tables for psi_b lookup
 * w_ids       (n_cb,)  weight type 0-4
 * out_idx     (n_cb,)  output block index
 * wick_fac    (n_cb,)  multiplicative factor (1.0 or rarely 2.0)
 * n_out       number of output blocks
 * sub_accum   (n_out, nrb1, nrb2)      output, accumulated in-place
 */
void postq_accumulate_tomo(
    const double *q_vals,
    int nrb1, int nrb2, int nv1, int nv2c,
    const double *abs_a,
    const double *va,
    const double *abs_b,
    const double *vb,
    const double *v1_grid,
    const double *v2_chunk,
    const double *xi_x, int xi_n,
    double xi_logx0, double xi_inv_dlogx,
    int n_cb,
    const int *xi_a_idx,
    const int *xi_b_idx,
    const double *xi_y_tables,
    int n_xi_unique,
    const double *v1_w,
    const double *v2_wc,
    const int *w_ids,
    const int *out_idx,
    const double *wick_fac,
    int n_out,
    double *sub_accum)
{
    int n_a = nrb1 * nv1;
    int n_b = nrb2 * nv2c;

    /* Pre-tabulate xi on geometry grids */
    double *xi_a_tab = (double *)malloc((long)n_xi_unique * n_a * sizeof(double));
    for (int t = 0; t < n_xi_unique; t++) {
        const double *yi = xi_y_tables + (long)t * xi_n;
        double *out = xi_a_tab + (long)t * n_a;
        for (int j = 0; j < n_a; j++)
            out[j] = interp1d_eval(xi_x, yi, xi_n,
                                   xi_logx0, xi_inv_dlogx, abs_a[j]);
    }
    double *xi_b_tab = (double *)malloc((long)n_xi_unique * n_b * sizeof(double));
    for (int t = 0; t < n_xi_unique; t++) {
        const double *yi = xi_y_tables + (long)t * xi_n;
        double *out = xi_b_tab + (long)t * n_b;
        for (int j = 0; j < n_b; j++)
            out[j] = interp1d_eval(xi_x, yi, xi_n,
                                   xi_logx0, xi_inv_dlogx, abs_b[j]);
    }

    /* Pre-tabulate trig */
    double *cos4va = (double *)malloc(n_a * sizeof(double));
    double *sin4va = (double *)malloc(n_a * sizeof(double));
    for (int j = 0; j < n_a; j++) {
        cos4va[j] = cos(4. * va[j]);
        sin4va[j] = sin(4. * va[j]);
    }
    double *cos4vb = (double *)malloc(n_b * sizeof(double));
    double *sin4vb = (double *)malloc(n_b * sizeof(double));
    for (int j = 0; j < n_b; j++) {
        cos4vb[j] = cos(4. * vb[j]);
        sin4vb[j] = sin(4. * vb[j]);
    }
    double *cos4v1 = (double *)malloc(nv1 * sizeof(double));
    double *sin4v1 = (double *)malloc(nv1 * sizeof(double));
    for (int j = 0; j < nv1; j++) {
        cos4v1[j] = cos(4. * v1_grid[j]);
        sin4v1[j] = sin(4. * v1_grid[j]);
    }
    double *cos4v2 = (double *)malloc(nv2c * sizeof(double));
    double *sin4v2 = (double *)malloc(nv2c * sizeof(double));
    for (int j = 0; j < nv2c; j++) {
        cos4v2[j] = cos(4. * v2_chunk[j]);
        sin4v2[j] = sin(4. * v2_chunk[j]);
    }

    /* Integration weight grid */
    double *wgrid = (double *)malloc((long)nv1 * nv2c * sizeof(double));
    for (int iv1 = 0; iv1 < nv1; iv1++)
        for (int iv2 = 0; iv2 < nv2c; iv2++)
            wgrid[iv1 * nv2c + iv2] = v1_w[iv1] * v2_wc[iv2];

    /* Pre-compute weight tables that don't depend on ir1/ir2 */
    /* w=3: cos(4*(v1-v2)), depends on (iv1, iv2) only */
    double *w_v1v2 = (double *)malloc((long)nv1 * nv2c * sizeof(double));
    for (int iv1 = 0; iv1 < nv1; iv1++)
        for (int iv2 = 0; iv2 < nv2c; iv2++)
            w_v1v2[iv1 * nv2c + iv2] =
                cos4v1[iv1] * cos4v2[iv2] + sin4v1[iv1] * sin4v2[iv2];

    /* w=4: 2*cos(4*(v2-vb)), depends on (ir2, iv2) only */
    double *w_r2v2 = (double *)malloc(n_b * sizeof(double));
    for (int ir2 = 0; ir2 < nrb2; ir2++)
        for (int iv2 = 0; iv2 < nv2c; iv2++) {
            int idx = ir2 * nv2c + iv2;
            w_r2v2[idx] = 2. * (cos4v2[iv2] * cos4vb[idx]
                              + sin4v2[iv2] * sin4vb[idx]);
        }

    /* Main callback loop */
    for (int k = 0; k < n_cb; k++) {
        int wid = w_ids[k];
        const double *xia = xi_a_tab + (long)xi_a_idx[k] * n_a;
        const double *xib = xi_b_tab + (long)xi_b_idx[k] * n_b;
        double fac = wick_fac[k];
        double *acc = sub_accum + (long)out_idx[k] * nrb1 * nrb2;

        for (int ir1 = 0; ir1 < nrb1; ir1++) {
            for (int ir2 = 0; ir2 < nrb2; ir2++) {
                double sum = 0.;
                long qbase = ((long)ir1 * nrb2 + ir2) * (long)nv1 * nv2c;

                for (int iv1 = 0; iv1 < nv1; iv1++) {
                    double xia_val = xia[ir1 * nv1 + iv1];
                    int a_idx = ir1 * nv1 + iv1;
                    long qrow = qbase + (long)iv1 * nv2c;

                    for (int iv2 = 0; iv2 < nv2c; iv2++) {
                        double qval = q_vals[qrow + iv2];
                        int b_idx = ir2 * nv2c + iv2;
                        double xib_val = xib[b_idx];
                        double wg = wgrid[iv1 * nv2c + iv2];

                        double w;
                        switch (wid) {
                        case 0: w = 1.; break;
                        case 1:
                            w = cos4va[a_idx] * cos4vb[b_idx]
                              + sin4va[a_idx] * sin4vb[b_idx];
                            break;
                        case 2: {
                            double cA = cos4v1[iv1]*cos4v2[iv2]
                                      - sin4v1[iv1]*sin4v2[iv2];
                            double sA = sin4v1[iv1]*cos4v2[iv2]
                                      + cos4v1[iv1]*sin4v2[iv2];
                            double cB = cos4va[a_idx]*cos4vb[b_idx]
                                      - sin4va[a_idx]*sin4vb[b_idx];
                            double sB = sin4va[a_idx]*cos4vb[b_idx]
                                      + cos4va[a_idx]*sin4vb[b_idx];
                            w = cA * cB + sA * sB;
                            } break;
                        case 3: w = w_v1v2[iv1 * nv2c + iv2]; break;
                        case 4: w = w_r2v2[b_idx]; break;
                        default: w = 0.; break;
                        }

                        sum += fac * qval * xia_val * xib_val * w * wg;
                    }
                }
                acc[ir1 * nrb2 + ir2] += sum;
            }
        }
    }

    free(xi_a_tab); free(xi_b_tab);
    free(cos4va); free(sin4va); free(cos4vb); free(sin4vb);
    free(cos4v1); free(sin4v1); free(cos4v2); free(sin4v2);
    free(wgrid); free(w_v1v2); free(w_r2v2);
}


/* ================================================================== */
/* Continuous q-term (triplet) for one xi function.                   */
/*                                                                     */
/* Computes the bin-averaged integral:                                 */
/*   sum_k weight(T1,T2,phi_k,r_jk) * xi(r_jk)                      */
/* on a fine (alpha*n_r) grid, then sums sub-bins back to (n_r,n_r).  */
/*                                                                     */
/* w_type: 0 = qpp (weight=1), 1 = qmm (cos(4*PHI)),                */
/*         2 = qpm (cos(4*(PHI - atan2(jk_y, jk_x))))               */
/* ================================================================== */
void continuous_q_single(
    int n_r, int alpha,
    const double *fine_centers,
    const double *fine_weights,
    const double *phi_centers,
    const double *dphi,
    int n_fine, int n_phi,
    const double *xi_x, int xi_n,
    double xi_logx0, double xi_inv_dlogx,
    const double *xi_y,
    int w_type,
    double sigma_eps2, double area, double nbar,
    const double *Np,
    double *out)
{
    double prefactor = 2. * sigma_eps2 / (3.14159265358979323846 * area * nbar);

    // Do fine-grid integral
    double *fine = (double *)calloc((long)n_fine * n_fine, sizeof(double));

    for (int i1 = 0; i1 < n_fine; i1++) {
        double T1 = fine_centers[i1];
        for (int i2 = 0; i2 < n_fine; i2++) {
            double T2 = fine_centers[i2];
            double sum = 0.;
            for (int k = 0; k < n_phi; k++) {
                double PHI = phi_centers[k];
                double jk_x = T1 - T2 * cos(PHI);
                double jk_y = -T2 * sin(PHI);
                double r_jk = hypot(jk_x, jk_y);
                double xi_val = interp1d_eval(xi_x, xi_y, xi_n,
                                              xi_logx0, xi_inv_dlogx, r_jk);
                double w;
                switch (w_type) {
                case 0: w = 1.; break;
                case 1: w = cos(4. * PHI); break;
                case 2: w = cos(4. * (PHI - atan2(jk_y, jk_x))); break;
                default: w = 0.; break;
                }
                sum += w * xi_val * dphi[k];
            }
            fine[(long)i1 * n_fine + i2] = sum;
        }
    }

    // Apply bin averaging
    for (int ic = 0; ic < n_r; ic++) {
        for (int jc = 0; jc < n_r; jc++) {
            double num = 0., den = 0.;
            for (int ia = 0; ia < alpha; ia++) {
                int i1 = ic * alpha + ia;
                double w1 = fine_weights[i1];
                for (int ja = 0; ja < alpha; ja++) {
                    int i2 = jc * alpha + ja;
                    double w2 = fine_weights[i2];
                    double ww = w1 * w2;
                    num += fine[(long)i1 * n_fine + i2] * ww;
                    den += ww;
                }
            }
            out[(long)ic * n_r + jc] += prefactor * num / den;
        }
    }

    free(fine);
}


////////////////////////////////////////////////////////////////////
// Build q-terms for all tomo combos, in continuous approximation // 
// /////////////////////////////////////////////////////////////////
void continuous_q_tomo(
    int n_r, int alpha,
    const double *fine_centers, const double *fine_weights,
    const double *phi_centers, const double *dphi,
    int n_fine, int n_phi,
    const double *xi_x, int xi_n,
    double xi_logx0, double xi_inv_dlogx,
    const double *xi_y_tables, int n_xi_unique,
    double sigma_eps2, double area, double nbar,
    const double *Np,
    int n_calls,
    const int *call_xi_idx,
    const int *call_w_type,
    const int *call_out_idx,
    const double *call_factor,
    int n_out,
    double *results,
    int nthreads)
{
    long block_size = (long)n_r * n_r;

    #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for (int k = 0; k < n_calls; k++) {
        const double *xi_y = xi_y_tables + (long)call_xi_idx[k] * xi_n;
        double *block = (double *)calloc(block_size, sizeof(double));

        continuous_q_single(n_r, alpha, fine_centers, fine_weights,
                            phi_centers, dphi, n_fine, n_phi,
                            xi_x, xi_n, xi_logx0, xi_inv_dlogx,
                            xi_y, call_w_type[k],
                            sigma_eps2, area, nbar, Np, block);

        /* Accumulate into the output (needs critical section) */
        double fac = call_factor[k];
        double *dst = results + (long)call_out_idx[k] * block_size;
        #pragma omp critical
        {
            for (long j = 0; j < block_size; j++)
                dst[j] += fac * block[j];
        }
        free(block);
    }
}


////////////////////////////////////////////////////////////////////
// Build r-terms for all tomo combos, in continuous approximation // 
// /////////////////////////////////////////////////////////////////
// We parallelise over (combo, wick_pairing)
void continuous_rterms_tomo(
    int n_r,
    const double *theta_centers,
    const double *phi_grid,
    const double *phi_w,
    int n_phi,
    const double *v_grid,
    const double *v_w,
    int n_v,
    const double *v_half,
    const double *v_half_w, 
    int n_vh,
    double area,
    const double *xi_x, int xi_n,
    double xi_logx0, double xi_inv_dlogx,
    const double *xi_y_tables, int n_xi_unique,
    int n_work,
    const int *work_xi_a_p,
    const int *work_xi_b_p,
    const int *work_xi_a_m,
    const int *work_xi_b_m,
    const int *work_out_idx,
    int n_out,
    double *results,
    int nthreads)
{
    long bs = (long)n_r * n_r;
    double inv_2piA = 1. / (2. * 3.14159265358979323846 * area);
    double inv_piA_times2 = 2. / (3.14159265358979323846 * area);

    #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for (int wk = 0; wk < n_work; wk++) {
        const double *xip_a = xi_y_tables + (long)work_xi_a_p[wk] * xi_n;
        const double *xip_b = xi_y_tables + (long)work_xi_b_p[wk] * xi_n;
        const double *xim_a = xi_y_tables + (long)work_xi_a_m[wk] * xi_n;
        const double *xim_b = xi_y_tables + (long)work_xi_b_m[wk] * xi_n;

        double *rp0 = (double *)calloc(bs, sizeof(double));
        double *rp1 = (double *)calloc(bs, sizeof(double));
        double *rm0 = (double *)calloc(bs, sizeof(double));
        double *rm1 = (double *)calloc(bs, sizeof(double));
        double *rpm_b = (double *)calloc(bs, sizeof(double));

        for (int kp = 0; kp < n_phi; kp++) {
            double phi = phi_grid[kp];
            double pw = phi_w[kp];

            // Do rp0: V in [0, pi]
            for (int ir1 = 0; ir1 < n_r; ir1++) {
                double Ia = 0.;
                for (int iv = 0; iv < n_vh; iv++) {
                    double cosV = cos(v_half[iv]);
                    double abs_a = sqrt(phi*phi + theta_centers[ir1]*theta_centers[ir1]
                                        - 2.*phi*theta_centers[ir1]*cosV);
                    Ia += interp1d_eval(xi_x, xip_a, xi_n, xi_logx0, xi_inv_dlogx, abs_a)
                          * v_half_w[iv];
                }
                for (int ir2 = 0; ir2 < n_r; ir2++) {
                    double Ib = 0.;
                    for (int iv = 0; iv < n_vh; iv++) {
                        double cosV = cos(v_half[iv]);
                        double abs_b = sqrt(phi*phi + theta_centers[ir2]*theta_centers[ir2]
                                            + 2.*phi*theta_centers[ir2]*cosV);
                        Ib += interp1d_eval(xi_x, xip_b, xi_n, xi_logx0, xi_inv_dlogx, abs_b)
                              * v_half_w[iv];
                    }
                    rp0[ir1*n_r + ir2] += pw * phi * Ia * Ib;
                }
            }

            // Do rp1, rm0, rm1, rpm: V in [0, 2*pi]
            double *Ia_rp1_c = (double *)calloc(n_r, sizeof(double));
            double *Ia_rp1_s = (double *)calloc(n_r, sizeof(double));
            double *Ia_rm0   = (double *)calloc(n_r, sizeof(double));
            double *Ia_rm1   = (double *)calloc(n_r, sizeof(double));
            double *Ia_rpm   = (double *)calloc(n_r, sizeof(double));

            for (int ir = 0; ir < n_r; ir++) {
                double T = theta_centers[ir];
                for (int iv = 0; iv < n_v; iv++) {
                    double V = v_grid[iv];
                    double cV = cos(V), sV = sin(V);
                    double pax = phi - T*cV, pay = -T*sV;
                    double abs_a = hypot(pax, pay);
                    double va = atan2(pay, pax);

                    double xim_val = interp1d_eval(xi_x, xim_a, xi_n,
                                                   xi_logx0, xi_inv_dlogx, abs_a);
                    double xip_val = interp1d_eval(xi_x, xip_a, xi_n,
                                                   xi_logx0, xi_inv_dlogx, abs_a);

                    double c4a = cos(4.*va), s4a = sin(4.*va);
                    Ia_rp1_c[ir] += xim_val * c4a * v_w[iv];
                    Ia_rp1_s[ir] += xim_val * s4a * v_w[iv];
                    Ia_rm0[ir]   += xim_val * cos(4.*(V - va)) * v_w[iv];
                    Ia_rm1[ir]   += xip_val * cos(4.*V) * v_w[iv];
                    Ia_rpm[ir]   += xip_val * v_w[iv];
                }
            }

            double *Ib_rp1_c = (double *)calloc(n_r, sizeof(double));
            double *Ib_rp1_s = (double *)calloc(n_r, sizeof(double));
            double *Ib_rm0   = (double *)calloc(n_r, sizeof(double));
            double *Ib_rm1   = (double *)calloc(n_r, sizeof(double));
            double *Ib_rpm   = (double *)calloc(n_r, sizeof(double));

            for (int ir = 0; ir < n_r; ir++) {
                double T = theta_centers[ir];
                for (int iv = 0; iv < n_v; iv++) {
                    double V = v_grid[iv];
                    double cV = cos(V), sV = sin(V);
                    double pbx = phi + T*cV, pby = T*sV;
                    double abs_b = hypot(pbx, pby);
                    double vb = atan2(pby, pbx);

                    double xim_val = interp1d_eval(xi_x, xim_b, xi_n,
                                                   xi_logx0, xi_inv_dlogx, abs_b);
                    double xip_val = interp1d_eval(xi_x, xip_b, xi_n,
                                                   xi_logx0, xi_inv_dlogx, abs_b);

                    double c4b = cos(4.*vb), s4b = sin(4.*vb);
                    Ib_rp1_c[ir] += xim_val * c4b * v_w[iv];
                    Ib_rp1_s[ir] += xim_val * s4b * v_w[iv];
                    Ib_rm0[ir]   += xim_val * cos(4.*(V - vb)) * v_w[iv];
                    Ib_rm1[ir]   += xip_val * cos(4.*V) * v_w[iv];
                    Ib_rpm[ir]   += 2.*xim_val * cos(4.*(V - vb)) * v_w[iv];
                }
            }

            // Form outer products and accumulate
            for (int i1 = 0; i1 < n_r; i1++) {
                for (int i2 = 0; i2 < n_r; i2++) {
                    long idx = i1*n_r + i2;
                    rp1[idx]  += pw * phi * (Ia_rp1_c[i1]*Ib_rp1_c[i2]
                                           + Ia_rp1_s[i1]*Ib_rp1_s[i2]);
                    rm0[idx]  += pw * phi * Ia_rm0[i1] * Ib_rm0[i2];
                    rm1[idx]  += pw * phi * Ia_rm1[i1] * Ib_rm1[i2];
                    rpm_b[idx] += pw * phi * Ia_rpm[i1] * Ib_rpm[i2];
                }
            }

            free(Ia_rp1_c); free(Ia_rp1_s); free(Ia_rm0); free(Ia_rm1); free(Ia_rpm);
            free(Ib_rp1_c); free(Ib_rp1_s); free(Ib_rm0); free(Ib_rm1); free(Ib_rpm);
        }

        // Apply prefactors and accumulate into output
        int base = work_out_idx[wk];
        #pragma omp critical
        {
            double *dst;
            dst = results + (long)(base + 0) * bs;
            for (long j = 0; j < bs; j++) dst[j] += inv_piA_times2 * rp0[j];
            dst = results + (long)(base + 1) * bs;
            for (long j = 0; j < bs; j++) dst[j] += inv_2piA * rp1[j];
            dst = results + (long)(base + 2) * bs;
            for (long j = 0; j < bs; j++) dst[j] += inv_2piA * rm0[j];
            dst = results + (long)(base + 3) * bs;
            for (long j = 0; j < bs; j++) dst[j] += inv_2piA * rm1[j];
            dst = results + (long)(base + 4) * bs;
            for (long j = 0; j < bs; j++) dst[j] += inv_2piA * rpm_b[j];
        }

        free(rp0); free(rp1); free(rm0); free(rm1); free(rpm_b);
    }
}


//////////////////////////////////
// Phi-reduction of mixed terms //
//////////////////////////////////
void triplet_term_reduce(
    const double *Nt,
    int nr, int nphi,
    const double *tc,
    const double *phic,
    int weight_type,
    const double *xi_x, int xi_n,
    double xi_logx0, double xi_inv_dlogx,
    const double *xi_y,
    double *out,
    int nthreads)
{
    double *cosphi  = (double *)malloc((size_t)nphi * sizeof(double));
    double *sinphi  = (double *)malloc((size_t)nphi * sizeof(double));
    double *cos4phi = (double *)malloc((size_t)nphi * sizeof(double));
    for (int k = 0; k < nphi; k++) {
        cosphi[k]  = cos(phic[k]);
        sinphi[k]  = sin(phic[k]);
        cos4phi[k] = cos(4. * phic[k]);
    }

    #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for (int i = 0; i < nr; i++) {
        double T1 = tc[i];
        for (int j = 0; j < nr; j++) {
            double T2 = tc[j];
            const double *row = Nt + ((size_t)i * nr + j) * nphi;
            double acc = 0.;
            for (int k = 0; k < nphi; k++) {
                double cphi = cosphi[k];
                double r2 = T1 * T1 + T2 * T2 - 2. * T1 * T2 * cphi;
                double rjk = r2 > 0. ? sqrt(r2) : 0.;
                double xi = interp1d_eval(xi_x, xi_y, xi_n,
                                          xi_logx0, xi_inv_dlogx, rjk);
                double w;
                if (weight_type == 0) {
                    w = 1.;
                } else if (weight_type == 1) {
                    w = cos4phi[k];
                } else {
                    double jk_x = T1 - T2 * cphi;
                    double jk_y = -T2 * sinphi[k];
                    double phi_jk = atan2(jk_y, jk_x);
                    w = cos(4. * (phic[k] - phi_jk));
                }
                acc += row[k] * w * xi;
            }
            out[(size_t)i * nr + j] = acc;
        }
    }

    free(cosphi); free(sinphi); free(cos4phi);
}
