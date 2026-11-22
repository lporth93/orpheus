// Helpers for computation of tomographic xipm covariance

#ifndef COV_POSTQ_H
#define COV_POSTQ_H

///////////////////////////////
// Quadruplet (SVA) integral //
///////////////////////////////

// Cosmic-variance (r-term) integral of all callbacks over the phi grid, Eq. 6 (h) of the SVA note
void quadruplet_sva_integrate(const double *phi_grid, const double *phi_w,
                              int n_phi, int n_r,
                              const double *theta1_nodes,
                              const double *theta1_weights, int n_sub1,
                              const double *theta2_nodes,
                              const double *theta2_weights, int n_sub2,
                              const double *v1_grid, const double *v1_w, int nv1,
                              const double *v2_grid, const double *v2_w, int nv2,
                              const double *q_array,
                              const double *q_r_centers, int q_nr, int n_modes,
                              int q_cubic,
                              double q_const, const double *sup_phi,
                              const double *sup_theta1, const double *sup_theta2,
                              const double *xi_x, int xi_n,
                              double xi_logx0, double xi_inv_dlogx,
                              const double *xi_y_tables, int n_xi_unique,
                              int n_cb,
                              const int *xi_a_idx, const int *xi_b_idx,
                              const int *w_ids, const int *out_idx,
                              const double *wick_fac,
                              int n_out, const int *symmetric_mask,
                              double *results, int nthreads, int verbose);

/////////////////////////////////
// Continuous-limit estimators //
/////////////////////////////////

// Continuous q-terms of all calls, each bin-averaged from a fine (alpha*n_r) grid
void continuous_q_tomo(int n_r, int alpha,
                       const double *fine_centers, const double *fine_weights,
                       const double *phi_centers, const double *dphi,
                       int n_fine, int n_phi,
                       const double *xi_x, int xi_n,
                       double xi_logx0, double xi_inv_dlogx,
                       const double *xi_y_tables,
                       double sigma_eps2, double area, double nbar,
                       int n_calls,
                       const int *call_xi_idx, const int *call_w_type,
                       double *results,
                       int nthreads, int verbose);

// Continuous r-terms of all tomo combos
void continuous_rterms_tomo(int n_r, const double *theta_centers,
                            const double *phi_grid, const double *phi_w, int n_phi,
                            const double *v_grid, const double *v_w, int n_v,
                            double area,
                            const double *xi_x, int xi_n,
                            double xi_logx0, double xi_inv_dlogx,
                            const double *xi_y_tables, int n_xi_unique,
                            int n_work,
                            const int *work_xi_a_p, const int *work_xi_b_p,
                            const int *work_xi_a_m, const int *work_xi_b_m,
                            const int *work_out_idx,
                            double *results,
                            int nthreads, int verbose);

//////////////////////////////////
// Phi-reduction of mixed terms //
//////////////////////////////////

// Integral over phi of the triplets per unit angle against weight*xi(r_jk), for one mixed term
void triplet_term_reduce(
    const double *triplets, int nr, int n_modes, const double *theta_edges, int n_sub, int n_ang,
    int weight_type, const double *xi_x, int xi_n, double xi_logx0, double xi_inv_dlogx,
    const double *xi_y, double *out, int nthreads, int verbose);

////////////////////////////////////////////////////////
// Post-processing of the measured quadruplet density //
////////////////////////////////////////////////////////

// Density of the sparse radial bins from the smallest block of bins that holds count_min, in place
void quadruplets_merge_sparse(double *q, int nr, int nm, const double *v, double cell_norm,
                              double q_ideal, double count_min, int nthreads);

// Knot values of the quadruplet density whose interpolant has its bin means, in place
void quadruplets_to_knots(double *q, int nr, int nm, double coef, int nthreads);

// Quadruplet density extrapolated to vanishing radial bin width from its bin means, in place
void quadruplets_richardson(double *q, int nr, int nm, const double *weights,
                            const double *centers, double coef, int cubic, int nthreads);

#endif
