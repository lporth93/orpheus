# Vendored HEALPix C++ sources

These files are copied verbatim from `healpix_cxx-3.83.0`, distributed as part of
[HEALPix](https://healpix.sourceforge.io/) and licensed under the GNU General Public
License, version 2 or later. Upstream authors: M. Reinecke, and the HEALPix collaboration
(Gorski, Hivon, Banday, Wandelt, Hansen, Reinecke, Bartelmann).

Only the subset needed by `T_Healpix_Base` is included -- the nested-pixel geometry used by
`orpheus/src/healpix_utils.cpp` for `query_disc` and `ang2pix`.

Do not edit these files. To move to a newer HEALPix, re-copy the same list from the upstream
`healpix_cxx` tarball and make sure they run as expected.

If you use orpheus in published work that relies on the curved-sky estimators, cite the
HEALPix paper: Gorski et al. 2005, ApJ 622, 759.