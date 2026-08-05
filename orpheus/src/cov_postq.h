// Helpers for computation of tomographic xipm covariance

#ifndef COV_POSTQ_H
#define COV_POSTQ_H

///////////////////////////////////
// Quadruplet (SVA) accumulation //
///////////////////////////////////

void postq_accumulate_tomo(const double *q_vals,
                           int nrb1, int nrb2, int nv1, int nv2c,
                           const double *abs_a, const double *va,
                           const double *abs_b, const double *vb,
                           const double *v1_grid, const double *v2_chunk,
                           const double *xi_x, int xi_n,
                           double xi_logx0, double xi_inv_dlogx,
                           int n_cb,
                           const int *xi_a_idx, const int *xi_b_idx,
                           const double *xi_y_tables, int n_xi_unique,
                           const double *v1_w, const double *v2_wc,
                           const int *w_ids, const int *out_idx,
                           const double *wick_fac,
                           int n_out, double *sub_accum);

/////////////////////////////////
// Continuous-limit estimators //
/////////////////////////////////

void continuous_q_single(int n_r, int alpha,
                         const double *fine_centers, const double *fine_weights,
                         const double *phi_centers, const double *dphi,
                         int n_fine, int n_phi,
                         const double *xi_x, int xi_n,
                         double xi_logx0, double xi_inv_dlogx,
                         const double *xi_y,
                         int w_type,
                         double sigma_eps2, double area, double nbar,
                         const double *Np, double *out);

void continuous_q_tomo(int n_r, int alpha,
                       const double *fine_centers, const double *fine_weights,
                       const double *phi_centers, const double *dphi,
                       int n_fine, int n_phi,
                       const double *xi_x, int xi_n,
                       double xi_logx0, double xi_inv_dlogx,
                       const double *xi_y_tables, int n_xi_unique,
                       double sigma_eps2, double area, double nbar,
                       const double *Np,
                       int n_calls,
                       const int *call_xi_idx, const int *call_w_type,
                       const int *call_out_idx, const double *call_factor,
                       int n_out, double *results,
                       int nthreads, int verbose);

void continuous_rterms_tomo(int n_r, const double *theta_centers,
                            const double *phi_grid, const double *phi_w, int n_phi,
                            const double *v_grid, const double *v_w, int n_v,
                            const double *v_half, const double *v_half_w, int n_vh,
                            double area,
                            const double *xi_x, int xi_n,
                            double xi_logx0, double xi_inv_dlogx,
                            const double *xi_y_tables, int n_xi_unique,
                            int n_work,
                            const int *work_xi_a_p, const int *work_xi_b_p,
                            const int *work_xi_a_m, const int *work_xi_b_m,
                            const int *work_out_idx,
                            int n_out, double *results,
                            int nthreads, int verbose);

//////////////////////////////////
// Phi-reduction of mixed terms //
//////////////////////////////////

void triplet_term_reduce(const double *Nt, int nr, int nphi,
                         const double *tc, const double *phic,
                         int weight_type,
                         const double *xi_x, int xi_n,
                         double xi_logx0, double xi_inv_dlogx,
                         const double *xi_y,
                         double *out, int nthreads, int verbose);

#endif
