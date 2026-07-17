#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>

int binary_search(double *array, int len_arr, double target);
double linint(double *vec, double x, double xmin, double xmax, double dx);
void expand_arr(int *arr_long, int *arr_sel, int len_long, int len_sel, int *result);
double nexttomoterm(int order, int max_order, double *moments, int *zcombi, int elzcombi, int do_subtractions);
int sumintarr(int *arr, int len);
int countel(int el, int *arr, int len);
int maxarr(int *arr, int len);
void fillconsti(int *arr, int len_arr, int c);
void fillconstd(double *arr, int len_arr, double c);
void reset_progress(void);
void print_progress(int nregionsdone, int nfilledregions, int verbose);
double sphere_dist(double x1, double y1, double z1, double x2, double y2, double z2);
double sphere_bearing(double ra_a, double sindec_a, double cosdec_a,
                      double ra_b, double sindec_b, double cosdec_b);

/////////////////////////
// Curved-sky geometry //
/////////////////////////

// We use a speedy way to compute the bearing angles on the sphere. Lots of the credit for the
// equations go to the treecorr package, in particular the main/include/ProjectHelper.h which
// introduces the prescription in terms of the numerically more stabe chord distance.
// * Define spherical coords: r(ra,dec) = (cos(dec)cos(ra), cos(dec)sin(ra), sin(dec)) = (x,y,z)
// * We get the tangent basis  by differentiation:
//   East:  ∂r/∂ra = cos(dec) (-sin(ra),cos(ra),0) --> e_E = (∂r/∂ra)/|∂r/∂ra| = (-sin(ra),cos(ra),0)
//   North: ∂r/∂dec = (-sin(dec)cos(ra),-sin(dec)sin(ra),cos(dec)) --> e_N = ∂r/∂dec
// * Now define point a = (xa,ya,za) and point b = (xb,yb,zb) on the sphere with radius 1
// * The bearing at a, looking toward b, is obtained by projecting b onto a's own tangent basis,
//   i.e. (e_E, e_N evaluated at a):
//   E = e_E·b = -sin(ra_a)*xb + cos(ra_a)*yb = (xa*yb - ya*xb)/cd_a  =: E_ab/cd_a --> E_ab = xa*yb - ya*xb
//   N = e_N·b = -sd_a*(cos(ra_a)*xb+sin(ra_a)*yb) + cd_a*zb = [zb - za*dot]/cd_a =: N_ab/cd_a --> N_ab = zb - za*dot
//   where dot = xa*xb + ya*yb + za*zb is the 3D dot product of the two points
// * N_ab = zb - za*dot is prone to cancellation for nearby points where dot->1. Therefore we rewrite
//   using the squared chord distance dsq = |a-b|² = 2*(1-dot), i.e. 1-dot = dsq/2 such that
//   N_ab = zb - za*dot = (zb-za) + za*(1-dot) = (zb-za) + 0.5*za*dsq
//   This is exact and equivalent, but numerically stable, since dsq is built from small
//   pairwise coordinate differences rather than a near-cancelling dot product.
// * All together we now get the rotator e^{i*phi_ab} as
//   e^{i*phi_ab}  = (E_ab + i*N_ab)/sqrt(E_ab² + N_ab²)
// * Note that if we are only interested in the projection for polar fields,  e^{-2i*phi_ab},
//   we can get this without any square root.

typedef struct { double E, N; } BearingAB;

// Get bearing geometry from cartesian coordinates
static inline BearingAB bearing_AB_cart(double xa, double ya, double za,
                                         double xb, double yb, double zb) {
    double dx=xa-xb, dy=ya-yb, dz=za-zb;
    double dsq = dx*dx + dy*dy + dz*dz;
    BearingAB g;
    g.E =  xa*yb - ya*xb;
    g.N = (zb - za) + 0.5*za*dsq;
    return g;
}

// Get bearing geometry from radec coordinates
static inline BearingAB bearing_AB_radec(double ra_a, double sindec_a, double cosdec_a,
                                          double ra_b, double sindec_b, double cosdec_b) {
    double dlam = ra_b - ra_a;
    double cdlam = cos(dlam), sdlam = sin(dlam);
    BearingAB g;
    g.E = cosdec_a * cosdec_b * sdlam;
    g.N = cosdec_a * (cosdec_a*sindec_b - sindec_a*cosdec_b*cdlam);
    return g;
}

// Single rotator e^{i*phi}.
static inline double complex bearing_phirot(BearingAB g) {
    double hyp = sqrt(g.E*g.E + g.N*g.N);
    return (g.E + I*g.N) / hyp;
}

// Doubled, conjugated rotator e^{-2i*phi}
static inline double complex bearing_rc(BearingAB g) {
    double D = g.E*g.E + g.N*g.N;
    return (g.E*g.E - g.N*g.N - 2.0*I*g.E*g.N) / D;
}

#endif