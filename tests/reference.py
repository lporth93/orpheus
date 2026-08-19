# In this file we collect all methods required to generate the deterministic 
# gaussian quadrupole shear field from the notes and to compute all relevant 
# theoretical expressions from it. We refer to the Equations in these notes.

from math import factorial

import numpy as np

__all__ = ["AnalyticField"]

class AnalyticField:
    """The quadrupole shear field and every statistic derived from it.

    Parameters
    ----------
    gamma0: float
        Amplitude of the shear field.
    r0: float
        Characteristic scale of the field.
    boxsize: float
        Sidelength of the square footprint.
    delta0: float
        Amplitude of the companion scalar field.
    chi: float
        Phase of the complex amplitude, i.e. the E/B mixing angle, see notes sect 6.

    Notes
    -----
    * For the binned correlation functions we always return two things, namely the
      function itself, as well as the finite field correction. The latter should be
      applied to the measurement in order to debias it.
    * The transformation to the x-basis or to the centroid basis is solely done at 
      at the discrete angular bin centers, mimicking what the estimator is doing.
    """

    def __init__(self, gamma0=.05, r0=3., boxsize=48., delta0=.3, chi=0.):
        self.gamma0 = float(gamma0)
        self.r0 = float(r0)
        self.boxsize = float(boxsize)
        self.delta0 = float(delta0)
        self.chi = float(chi)

    @property
    def ebmix(self):
        """The pair (cos chi, sin chi) that every E/B component is built from."""
        return np.cos(self.chi), np.sin(self.chi)

    @property
    def area(self):
        return self.boxsize**2

    ##########
    # FIELDS #
    ##########
    # Sect 2.1, Eq (9) in notes, with the phase of sect 6.1 eq (56)
    def shear(self, x, y):
        z = x + 1j*y
        return (-self.gamma0*np.exp(1j*self.chi)/self.r0**2*z**2
                *np.exp(-np.abs(z)**2/(2.*self.r0**2)))

     # Sect 2.2, Eq (13) in notes. Note that for chi != 0 this is the E-mode template
     # and the physical convergence is cos(chi) times it.
    def kappa(self, x, y):
        rsq = x*x + y*y
        return (self.gamma0/self.r0**2*(2.*self.r0**2 - rsq)
                *np.exp(-rsq/(2.*self.r0**2)))

    # Sect 2.3, Eq (16) in notes
    def delta(self, x, y):
        """Scalar tracer field, a rescaled copy of the convergence."""
        return self.delta0/self.gamma0*self.kappa(x, y)
    
    ###########################
    # MOCK CATALOG GENERATION #
    ###########################
    def positions(self, ngal, seed=1, stratified=True):
        """Sample the footprint, centred on the origin.

        Note that we use stratified sampling instead of poisson as this yields
        better convergence properties, see notes sect. 7.2 for details.
        """
        rng = np.random.default_rng(seed)
        # Uniform sampling
        if not stratified:
            return rng.uniform(-self.boxsize/2., self.boxsize/2., (2, ngal))
        # Stratified sampling, using the closest square to the target ngal
        n = int(round(np.sqrt(ngal)))
        step = self.boxsize/n
        g = (np.arange(n) + .5)*step - self.boxsize/2.
        gx, gy = np.meshgrid(g, g, indexing='ij')
        jit = rng.uniform(-.5, .5, (2, n, n))*step
        return (gx + jit[0]).ravel(), (gy + jit[1]).ravel()

    def grid_step(self, ngal):
        """Mean tracer spacing, which for a stratified sample is the cell size."""
        return self.boxsize/np.sqrt(ngal)

    # Notes sect. 7.2 for details
    def min_usable_sep(self, ngal, cells=5.):
        """Smallest separation that measures the field rather than the sampling."""
        return cells*self.grid_step(ngal)

    def ngal_for_min_sep(self, min_sep, cells=5.):
        """Inverse of :meth:`min_usable_sep`: the sampling a given theta_min demands."""
        return (cells*self.boxsize/min_sep)**2

    def catalogs(self, ngal, seed=1, stratified=True, delta_sign=1.):
        """Generate source and lens catalog according to setup. """
        from orpheus.catalog import ScalarTracerCatalog, SpinTracerCatalog
        x, y = self.positions(ngal, seed, stratified)
        e = self.shear(x, y)
        cat_shape = SpinTracerCatalog(spin=2, pos1=x, pos2=y,
                                      tracer_1=e.real, tracer_2=e.imag,
                                      weight=np.ones(len(x)), geometry='flat2d')
        cat_lens = ScalarTracerCatalog(pos1=x, pos2=y, tracer=np.ones(len(x)),
                                       weight=1.+delta_sign*self.delta(x, y),
                                       geometry='flat2d')
        return cat_shape, cat_lens

    @staticmethod
    def parity_combine(plus, minus, order_in_delta):
        """Build even or odd parity combination of arrays """
        plus, minus = np.asarray(plus), np.asarray(minus)
        return .5*(plus + minus) if order_in_delta % 2 == 0 else .5*(plus - minus)


    ################
    # SECOND ORDER #
    ################
    
    # Sect 4.1 eq (27) in notes
    def xi_plus(self, theta):
        t = theta/self.r0
        return (np.pi*self.gamma0**2*self.r0**2/self.area
                *(2. - t**2 + t**4/16.)*np.exp(-t**2/4.))

    # Sect 4.1 eq (29) in notes, with the phase correction of sect 6.2
    def xi_minus(self, theta):
        t = theta/self.r0
        base = np.pi*self.gamma0**2*self.r0**2/self.area*t**4/16.*np.exp(-t**2/4.)
        phase = np.exp(2j*self.chi)
        return base*phase

    # Sect 4.1 eq (31) in notes, no phase correction
    def omega(self, theta):
        return (self.delta0/self.gamma0)**2*self.xi_plus(theta)

    def xi_binned(self, edges, nsub=5):
        """Bin-averaged xi_pm and f_pair."""
        # Set up binning
        x, w = np.polynomial.legendre.leggauss(nsub)
        lo, hi = edges[:-1], edges[1:]
        th = .5*(lo + hi)[:, None] + .5*(hi - lo)[:, None]*x[None, :]
        ww = .5*(hi - lo)[:, None]*w[None, :]*th
        # Compute quantities
        den = np.sum(ww, axis=1)
        true_xip = np.sum(self.xi_plus(th)*ww, axis=1)/den
        true_xim = np.sum(self.xi_minus(th)*ww, axis=1)/den
        true_fpair = np.sum(self.f_pair(th)*ww, axis=1)/den
        return (true_xip, true_xim, true_fpair)

    ###############
    # THIRD ORDER #
    ###############
    def _betas(self, r1, r2, phi):
        """Vertex offsets from the triangle centroid, in units of r0."""
        r1, r2, phi = np.broadcast_arrays(*np.broadcast_arrays(r1, r2, phi))
        a = np.stack([np.zeros(phi.shape, dtype=complex), r1.astype(complex),
                      r2*np.exp(1j*phi)])
        return a - a.mean(axis=0)

    # Sect 4.2 Eqns (33,35) in notes, with the phases of sect 6.2
    def _gamma_raw(self, r1, r2, phi):
        """The natural components in the cartesian projection"""
        beta = self._betas(r1, r2, phi)/self.r0
        E = np.exp(-.5*np.sum(np.abs(beta)**2, axis=0))
        pref = -2.*np.pi*self.gamma0**3*self.r0**2/(3.*self.area)
        raw = np.empty((4,) + beta.shape[1:], dtype=complex)
        phase_0 = np.exp(3j*self.chi)
        raw[0] = pref*np.prod(beta, axis=0)**2*E * phase_0
        for k in range(3):
            l, m = [j for j in range(3) if j != k]
            bk, bl, bm = beta[k], beta[l], beta[m]
            phase_k = np.exp(1j*self.chi)
            raw[1+k] = pref*phase_k*(
                8./9.*(bk**2 + 2.*bl*bm) - 8./3.*np.abs(bk)**2*bl*bm
                + np.conj(bk)**2*bl**2*bm**2)*E
        return raw

    def _centroid_phases(self, r1, r2, phi):
        """The rotations taking the cartesian components into the centroid basis."""
        ph = self._betas(r1, r2, phi)
        ph = ph/np.abs(ph)
        out = np.empty((4,) + ph.shape[1:], dtype=complex)
        out[0] = -np.prod(np.conj(ph)**2, axis=0)
        for k in range(3):
            l, m = [j for j in range(3) if j != k]
            out[1+k] = -ph[k]**2*np.conj(ph[l])**2*np.conj(ph[m])**2
        return out

    def gamma_centroid(self, r1, r2, phi):
        """The natural components of the shear 3PCF using centroid-projection."""
        return self._gamma_raw(r1, r2, phi)*self._centroid_phases(r1, r2, phi)

    def gamma_binned(self, edges, phis, nsub=3, centers=None):
        """Bin-averaged gamma_centroid and f_triplet."""
        # Set up binning
        x, w = np.polynomial.legendre.leggauss(nsub)
        phis = np.asarray(phis)
        lo, hi = edges[:-1], edges[1:]
        rs = (.5*(lo + hi)[:, None] + .5*(hi - lo)[:, None]*x[None, :]).ravel()
        wr = (np.repeat(.5*(hi - lo), nsub)*np.tile(w, len(lo)))*rs
        nr, nphi = len(lo), len(phis)

        # Compute quantities
        # Note that as the centroid projection is dependent on the actual subbin,
        # we first perform the subbin averaging and then transform at the bin centers
        gam = np.empty((4, nr, nr, nphi), dtype=complex)
        win = np.empty((nr, nr, nphi))
        for i in range(nr):
            sl = slice(i*nsub, (i+1)*nsub)
            r1 = rs[sl][:, None, None]
            r2 = rs[None, :, None]
            ph = phis[None, None, :]
            ww = wr[sl][:, None, None]*wr[None, :, None]
            den = ww.reshape(nsub, nr, nsub).sum(axis=(0, 2))[:, None]
            gamma = self._gamma_raw(r1, r2, ph)
            if centers is None: gamma *= self._centroid_phases(r1, r2, ph)
            num = (gamma*ww[None]).reshape(4, nsub, nr, nsub, nphi).sum(axis=(1, 3))
            gam[:, i] = num/den
            win[i] = ((self.f_triplet(r1, r2, ph)*ww)
                      .reshape(nsub, nr, nsub, nphi).sum(axis=(0, 2))/den)
        if centers is not None:
            c = np.asarray(centers).ravel()
            gam *= self._centroid_phases(c[:, None, None], c[None, :, None],
                                         phis[None, None, :])
        return gam, win

    #####################
    # THIRD ORDER MIXED #
    #####################

    # Sect 4.4 Eq 42 in notes, with the phase of sect 6.2
    def _gnn_raw(self, r1, r2, phi):
        """G3L correlator in the cartesian basis."""
        beta = self._betas(r1, r2, phi)/self.r0
        b0, b1, b2 = beta
        E = np.exp(-.5*np.sum(np.abs(beta)**2, axis=0))
        pref = -2.*np.pi*self.gamma0*self.delta0**2*self.r0**2/(3.*self.area)
        Xi = (8./9.*(b1*b2 - b0**2)
              + 4./3.*b0*(b1*np.abs(b2)**2 + b2*np.abs(b1)**2)
              + 2./3.*b0**2*(np.abs(b1)**2 + np.abs(b2)**2 + b1*np.conj(b2) + np.conj(b1)*b2)
              + b0**2*(4. - 2.*(np.abs(b1)**2 + np.abs(b2)**2) + np.abs(b1)**2*np.abs(b2)**2))
        phase = np.exp(1j*self.chi)
        return pref*phase*Xi*E

    def gnn_x(self, r1, r2, phi):
        """Upsilon_GNN in orpheus's 'X' projection."""
        return self._gnn_raw(r1, r2, phi)*(-np.exp(-1j*np.asarray(phi, dtype=float)))

    def gnn_binned(self, edges, phis, nsub=3):
        """Bin-averaged gnn_x and f_triplet."""
        # Set up binning
        x, w = np.polynomial.legendre.leggauss(nsub)
        phis = np.asarray(phis)
        lo, hi = edges[:-1], edges[1:]
        rs = (.5*(lo + hi)[:, None] + .5*(hi - lo)[:, None]*x[None, :]).ravel()
        wr = (np.repeat(.5*(hi - lo), nsub)*np.tile(w, len(lo)))*rs
        nr, nphi = len(lo), len(phis)

        # Compute quantities
        gnn = np.empty((nr, nr, nphi), dtype=complex)
        win = np.empty((nr, nr, nphi))
        for i in range(nr):
            sl = slice(i*nsub, (i+1)*nsub)
            r1 = rs[sl][:, None, None]
            r2 = rs[None, :, None]
            ph = phis[None, None, :]
            ww = wr[sl][:, None, None]*wr[None, :, None]
            den = ww.reshape(nsub, nr, nsub).sum(axis=(0, 2))[:, None]
            num = (self.gnn_x(r1, r2, ph)*ww).reshape(nsub, nr, nsub, nphi).sum(axis=(0, 2))
            gnn[i] = num/den
            win[i] = ((self.f_triplet(r1, r2, ph)*ww)
                      .reshape(nsub, nr, nsub, nphi).sum(axis=(0, 2))/den)
        return gnn, win

    # Sect 4.5 Eq (43), (45) in notes, with the phases of sect 6.2
    def _ngg_raw(self, r1, r2, phi):
        """The two NGG natural components [G_-, G_+] before projection."""
        beta = self._betas(r1, r2, phi)/self.r0
        b0, b1, b2 = beta
        E = np.exp(-.5*np.sum(np.abs(beta)**2, axis=0))
        pref = 2.*np.pi*self.gamma0**2*self.delta0*self.r0**2/(3.*self.area)
        kappa0 = 2. - np.abs(b0)**2
        bb0, bb2 = np.conj(b0), np.conj(b2)
        phase_m = np.exp(2j*self.chi)
        phase_p = 1.
        gm = pref*phase_m*(
            kappa0*b1**2*b2**2 + 2./3.*(2.*b0**2*b1*b2 - b1**2*b2**2))*E
        gp = pref*phase_p*(
            kappa0*b1**2*bb2**2
            + 2./3.*(4.*kappa0*b1*bb2 - b1**2*bb2**2 - 2.*bb0*b1**2*bb2 - 2.*b0*b1*bb2**2)
            + 8./9.*(kappa0 - 4.*b1*bb2 - 2.*bb0*b1 - 2.*b0*bb2)
            - 16./9.)*E
        return np.stack([gm, gp])

    def ngg_x(self, r1, r2, phi):
        """The two NGG natural components in orpheus's 'X' projection, [G_-, G_+]."""
        gm, gp = self._ngg_raw(r1, r2, phi)
        phi = np.asarray(phi, dtype=float)
        return np.stack([gm*np.exp(-2j*phi), gp*np.exp(2j*phi)])

    def ngg_binned(self, edges, phis, nsub=3):
        """Bin-averaged ngg_x and f_triplet."""
        # Set up binning
        x, w = np.polynomial.legendre.leggauss(nsub)
        phis = np.asarray(phis)
        lo, hi = edges[:-1], edges[1:]
        rs = (.5*(lo + hi)[:, None] + .5*(hi - lo)[:, None]*x[None, :]).ravel()
        wr = (np.repeat(.5*(hi - lo), nsub)*np.tile(w, len(lo)))*rs
        nr, nphi = len(lo), len(phis)

        # Compute quantities
        ngg = np.empty((2, nr, nr, nphi), dtype=complex)
        win = np.empty((nr, nr, nphi))
        for i in range(nr):
            sl = slice(i*nsub, (i+1)*nsub)
            r1 = rs[sl][:, None, None]
            r2 = rs[None, :, None]
            ph = phis[None, None, :]
            ww = wr[sl][:, None, None]*wr[None, :, None]
            den = ww.reshape(nsub, nr, nsub).sum(axis=(0, 2))[:, None]
            num = (self.ngg_x(r1, r2, ph)*ww[None]).reshape(2, nsub, nr, nsub, nphi).sum(axis=(1, 3))
            ngg[:, i] = num/den
            win[i] = ((self.f_triplet(r1, r2, ph)*ww)
                      .reshape(nsub, nr, nsub, nphi).sum(axis=(0, 2))/den)
        return ngg, win

    ################
    # FOURTH ORDER #
    ################

    # Indices of the legs that are complex conjugated for GGGG natural components
    CONJ4 = ((), (0,), (1,), (2,), (3,), (0, 1), (0, 2), (0, 3))

    def _betas4(self, r1, r2, r3, phi12, phi13):
        """Vertex offsets from the quadrilateral centroid."""
        r1, r2, r3, phi12, phi13 = np.broadcast_arrays(
            *np.broadcast_arrays(r1, r2, r3, phi12, phi13))
        a = np.stack([np.zeros(phi12.shape, dtype=complex), r1.astype(complex),
                      r2*np.exp(1j*phi12), r3*np.exp(1j*phi13)])
        return a - a.mean(axis=0)

    @staticmethod
    def _esp(legs):
        """Elementary symmetric polynomials e_0, ..., e_m of the m arrays in legs."""
        e = [np.ones(np.broadcast_shapes(*[np.shape(c) for c in legs]), dtype=complex)]
        for c in legs:
            e = [e[0]] + [e[i] + c*e[i-1] for i in range(1, len(e))] + [c*e[-1]]
        return e

    # Sect 4.3, Eq (38) in notes 
    @staticmethod
    def _sympow(e, nu, r):
        """Coefficient of v^r in the square of prod_mu (v + c_mu)."""
        return sum(e[k]*e[2*nu - r - k] for k in range(max(0, nu - r), min(nu, 2*nu - r) + 1))

    # Sect 4.3, Eq (39) in notes for n=4 and the phases of sect 6.2
    def _gamma4_raw(self, r1, r2, r3, phi12, phi13):
        """The eight natural components before projection, i.e. in the cartesian basis."""
        c = self._betas4(r1, r2, r3, phi12, phi13)
        E = np.exp(-.5*np.sum(np.abs(c)**2, axis=0)/self.r0**2)
        raw = np.empty((8,) + c.shape[1:], dtype=complex)
        for mu, conj in enumerate(self.CONJ4):
            unco = [j for j in range(4) if j not in conj]
            nu, nubar = len(unco), len(conj)
            eu = self._esp([c[j] for j in unco])
            ec = self._esp([np.conj(c[j]) for j in conj])
            tot = sum(self._sympow(eu, nu, r)*self._sympow(ec, nubar, r)
                      *np.pi*factorial(r)*(self.r0**2/2.)**(r + 1)
                      for r in range(min(2*nu, 2*nubar) + 1))
            raw[mu] = (self.gamma0**4*np.exp(1j*self.chi*(nu - nubar))/self.r0**8
                       *E*tot/self.area)
        return raw

    # Phases in the 'X'-basis for each fourth-order natural component
    def _x_phases4(self, phi12, phi13):
        zeta = [phi13/2., np.zeros_like(phi13), phi12, phi13]
        return np.stack([np.exp(2j*sum((1. if j in conj else -1.)*zeta[j] for j in range(4)))
                         for conj in self.CONJ4])

    def gamma4_x(self, r1, r2, r3, phi12, phi13):
        """The eight fourth-order natural components in the 'X' basis."""
        b = np.broadcast_arrays(r1, r2, r3, phi12, phi13)
        return self._gamma4_raw(*b)*self._x_phases4(b[3], b[4])

    def gamma4_binned(self, edges, phis, comp=None, nsub=3):
        """Bin-averaged gamma4_x and f_quadruplet."""

        # Set up binning
        x, w = np.polynomial.legendre.leggauss(nsub)
        phis = np.asarray(phis)
        lo, hi = edges[:-1], edges[1:]
        rs = (.5*(lo + hi)[:, None] + .5*(hi - lo)[:, None]*x[None, :]).ravel()
        wr = (np.repeat(.5*(hi - lo), nsub)*np.tile(w, len(lo)))*rs
        nr, nphi = len(lo), len(phis)

        # Compute quantities
        gam = np.empty((8, nr, nr, nr, nphi, nphi), dtype=complex)
        win = np.empty((nr, nr, nr, nphi, nphi))
        p1 = phis[:, None]
        p2 = phis[None, :]
        for i in range(nr):
            for j in range(nr):
                for k in range(nr):
                    a = rs[i*nsub:(i+1)*nsub][:, None, None, None, None]
                    b = rs[j*nsub:(j+1)*nsub][None, :, None, None, None]
                    c = rs[k*nsub:(k+1)*nsub][None, None, :, None, None]
                    ww = (wr[i*nsub:(i+1)*nsub][:, None, None]
                          *wr[j*nsub:(j+1)*nsub][None, :, None]
                          *wr[k*nsub:(k+1)*nsub][None, None, :])[..., None, None]
                    den = ww.sum(axis=(0, 1, 2))
                    gam[:, i, j, k] = ((self.gamma4_x(a, b, c, p1, p2)*ww)
                                       .sum(axis=(1, 2, 3))/den)
                    win[i, j, k] = ((self.f_quadruplet(a, b, c, p1, p2)*ww)
                                    .sum(axis=(0, 1, 2))/den)
        return (gam if comp is None else gam[comp]), win

    ###################
    # APERTURE MASSES #
    ###################

    # Sect 5.2 Eq (50) in notes
    def map_n(self, n, theta0):
        """Equal-scale Mapn up until order 4."""
        rsq = self.r0**2 + np.asarray(theta0, dtype=float)**2
        Jn = {1: 0., 2: 3./4., 3: 176./243., 4: 2547./2048.}
        return (2.*np.pi*(2.*self.gamma0*self.r0**4*np.asarray(theta0)**2)**n
                /(self.area*rsq**(3*n-1))*Jn[n])

    # Sect 5.3 Eq (54) in notes
    def map_unequal(self, thetas):
        """Unqual-scale Mapn for arbitrary order."""

        # Basic setup
        thetas = np.atleast_1d(np.asarray(thetas, dtype=float))
        n = len(thetas)
        rsq = self.r0**2 + thetas**2
        b = 1./(2.*rsq)
        up, um = 2. + np.sqrt(2.), 2. - np.sqrt(2.)

        # Build elementry symmetric polynomials
        e = np.zeros(n + 1)
        e[0] = 1.
        for bi in b:
            e[1:] += bi*e[:-1]

        # Build c_m coefficients from eq (53)
        c = np.zeros(2*n + 1)
        for j in range(n + 1):
            for k in range(n + 1):
                c[j+k] += e[j]*e[k]*(-up)**(n-j)*(-um)**(n-k)

        # Evaluate the sum eq (54)
        m = np.arange(2*n + 1)
        mu = e[1]
        pref = np.prod(2.*self.gamma0*self.r0**4*thetas**2/rsq**3)
        fac = np.array([float(factorial(int(k))) for k in m])
        return np.pi/self.area*pref*np.sum(c*fac/mu**(m + 1.))

    # Sect 6.2 Eq (60) in notes
    def map_n_ebmodes(self, n, theta0):
        """The n+1 components ``<Map^(n-j) Mx^j>``, indexed by the number j of Mx legs."""
        j = np.arange(n + 1)
        c, s = self.ebmix
        fac = c**(n - j)*s**j
        return fac.reshape((n + 1,) + (1,)*np.ndim(theta0))*self.map_n(n, theta0)

    # Sect 6.2 Eq (61) in notes
    def nap_map_n(self, n_nap, n_map, theta0, ncross=0):
        """Mixed ``<Nap^m Map^(n-k) Mx^k>`` correlator, with ``k = ncross``."""
        c, s = self.ebmix
        return ((self.delta0/self.gamma0)**n_nap*c**(n_map - ncross)*s**ncross
                *self.map_n(n_nap + n_map, theta0))

    ################
    # EDGE WINDOWS #
    ################

    # Sect 7.1 Eq (65) in notes
    def f_pair(self, theta):
        """Fraction of the box available to a pair of separation ``theta``."""
        return np.clip(1. - 4.*theta/(np.pi*self.boxsize)
                       + theta**2/(np.pi*self.boxsize**2), 1e-12, None)

    # Sect 7.1 Eq (67) in notes
    def _f_window(self, b, nalpha=32):
        """Bounding-box overlap of the offsets ``b``, averaged over orientation."""
        acc = np.zeros(b.shape[1:])
        for al in np.linspace(0., np.pi/2., nalpha, endpoint=False):
            rot = b*np.exp(1j*al)
            dx = rot.real.max(axis=0) - rot.real.min(axis=0)
            dy = rot.imag.max(axis=0) - rot.imag.min(axis=0)
            acc += (np.clip(self.boxsize - dx, 0., None)
                    *np.clip(self.boxsize - dy, 0., None)/self.boxsize**2)
        return acc/nalpha

    def f_triplet(self, r1, r2, phi, nalpha=32):
        """Triplet analogue of :meth:`f_pair`: bounding-box overlap, orientation-averaged."""
        return self._f_window(self._betas(r1, r2, phi), nalpha)

    def f_quadruplet(self, r1, r2, r3, phi12, phi13, nalpha=32):
        """Quadruplet analogue of :meth:`f_triplet`."""
        return self._f_window(self._betas4(r1, r2, r3, phi12, phi13), nalpha)

    ###########
    # METRICS #
    ###########
    @staticmethod
    def deviation(measured, theory, kind='max'):
        """Deviation normalised by the peak of ``theory``, not pointwise.

        For second-order we can safely use max but for third-order rms this
        might be dominated by sampling noise."""
        d = np.abs(np.asarray(measured) - np.asarray(theory))
        norm = np.max(np.abs(theory))
        return (np.max(d) if kind == 'max' else np.sqrt(np.mean(d**2)))/norm