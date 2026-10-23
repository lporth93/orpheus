// Helpers for computation of tomographic xipm covariance

#ifndef COV_POSTQ_H
#define COV_POSTQ_H

//////////////////////////////////////
// Quadruplet density interpolation //
//////////////////////////////////////

void interp4d(const double *q_slice, const double *r_centers, const double *phi_centers,
             int nr, int nphi,
             double r_min, double r_max, double logr0, double inv_dlogr,
             double phi0, double inv_dphi,
             const double *t2f, const double *t3f, const double *a12f, const double *a13f,
             int n, double *out);

////////////////////////////////
// Quadruplet (SVA) integral   //
////////////////////////////////

void quadruplet_sva_integrate(const double *phi_grid, const double *phi_w,
                              int n_phi, int n_r,
                              const double *phi1_nodes,
                              const double *phi1_weights, int ns1,
                              const double *phi2_nodes,
                              const double *phi2_weights, int ns2,
                              const double *v1_grid, const double *v1_w, int nv1,
                              const double *v2_grid, const double *v2_w, int nv2,
                              int nr_block, int v2_block,
                              const int *tiles, int n_tiles,
                              int q_mode, const double *q_array,
                              const double *q_r_centers,
                              const double *q_phi_centers, int q_nr, int q_nphi,
                              double q_const, const double *sup_t1,
                              const double *sup_t2, const double *sup_t3,
                              const double *xi_x, int xi_n,
                              double xi_logx0, double xi_inv_dlogx,
                              const double *xi_y_tables, int n_xi_unique,
                              int n_cb,
                              const int *xi_a_idx, const int *xi_b_idx,
                              const int *w_ids, const int *out_idx,
                              const double *wick_fac,
                              int n_out, const int *symmetric_mask,
                              const int *pair_mask,
                              double *results, int nthreads, int verbose);

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
