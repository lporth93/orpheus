import numpy as np
from scipy.signal import fftconvolve
from scipy.special import erf
import ctypes as ct
from numpy.ctypeslib import ndpointer
import glob
from pathlib import Path

class FlatPixelGrid(object):
    """ Grid with non-identity pixel size and offset.
    """

    def __init__(self, shape=None, pixsize=None, min1=None, min2=None,
                 max1=None, max2=None, n1=None, n2=None, units="arcmin"):
        """
        Precedences: min/max/n overrule pixsize
                     shape overrules n1/n2

        TODO implement possibility/precedancy and checks for setting up with min/max + pixsize
        """

        # Setup shape of data
        if shape is not None:
            assert(isinstance(shape, tuple))
            self.units = units
            self.shape = shape
            self.n1 = self.shape[1]
            self.n2 = self.shape[0]
        else:
            if n1 is None or n2 is None:
                raise ValueError("Insufficient information specified")
            else:
                self.n1 = n1
                self.n2 = n2
                self.shape = (n2, n1)

        # Setup extent of datagrid
        if pixsize is None:
            if min1 is not None and min2 is not None and max1 is not None and max2 is not None:
                _1 = (max1 - min1) / self.n1
                _2 = (max2 - min2) / self.n2
                assert(isclose(_1, _2, rel_tol=0.1 / max(self.n1, self.n2)))
                self.pixsize = _1
            else:
                self.pixsize = 1.
        else:
            self.pixsize = pixsize
        if min1 is None:
            self.min1 = 0.
        else:
            self.min1 = min1
        if min2 is None:
            self.min2 = 0.
        else:
            self.min2 = min2
        self.max1 = self.min1 + (self.n1 - 1) * self.pixsize
        self.max2 = self.min2 + (self.n2 - 1) * self.pixsize
        self.len1 = self.max1 - self.min1
        self.len2 = self.max2 - self.min2

        # Setup pixel bounds
        self.min1_bound = self.min1 - .5 * self.pixsize
        self.min2_bound = self.min2 - .5 * self.pixsize
        self.max1_bound = self.max1 + .5 * self.pixsize
        self.max2_bound = self.max2 + .5 * self.pixsize
        self.len1_bound = self.max1_bound - self.min1_bound
        self.len2_bound = self.max2_bound - self.min2_bound
    
    @staticmethod
    def cartesian(arrays):
        arrays = [np.asarray(a) for a in arrays]
        shape = (len(x) for x in arrays)
        ix = np.indices(shape, dtype=int)
        ix = ix.reshape(len(arrays), -1).T
        for n, arr in enumerate(arrays):
            ix[:, n] = arrays[n][ix[:, n]]
        return ix

    def torad(self, val):
        if self.units == "rad":
            fac = 1.
        elif self.units == "deg":
            fac = 360.
        elif self.units == "arcmin":
            fac = 360. * 60.
        elif self.units == "arcsec":
            fac = 360. * 3600.
        return 2. * np.pi / fac * val

    def get_pix_centers(self):
        c1 = np.linspace(self.min1, self.max1, self.n1, dtype=float)
        c2 = np.linspace(self.min2, self.max2, self.n2, dtype=float)
        return c1, c2

    def get_pix_bounds(self):
        b1 = np.linspace(self.min1_bound, self.max1_bound,
                         self.n1 + 1, dtype=float)
        b2 = np.linspace(self.min2_bound, self.max2_bound,
                         self.n2 + 1, dtype=float)
        return b1, b2

    def pos2pix(self, pos1, pos2):
        pix1 = np.floor((pos1 - self.min1_bound) / self.pixsize).astype(int)
        pix2 = np.floor((pos2 - self.min2_bound) / self.pixsize).astype(int)
        return pix1, pix2

    def rgrid(self, rmax=None, npix=None):
        """ Put a radially symmetric function on the grid.
        """
        if rmax is None and npix is None:
            npix = self.n1
        if rmax is not None:
            npix = rmax / self.pixsize
        npix = int(np.ceil(npix))
        _ = np.arange(-npix, npix + 1)
        ds = self.cartesian([_, _])
        rgrid = np.hypot(ds[:, 0], ds[:, 1]).reshape(
            (len(_), len(_))) * self.pixsize
        return rgrid

    def polargrid(self, rmax=None, npix=None):
        if rmax is None and npix is None:
            npix = self.n1
        if rmax is not None:
            npix = rmax / self.pixsize
        rgrid = self.rgrid(rmax, npix)
        npix = int(np.ceil(npix))
        _ = np.arange(-npix, npix + 1)
        y1 = np.repeat(_[np.newaxis, :], len(_), axis=0)
        y2 = np.repeat(_[:, np.newaxis], len(_), axis=1)
        phigrid = np.arctan2(y1, y2)
        return rgrid, phigrid

    def ellvec(self):
        return np.fft.fftfreq(n=self.n1, d=self.torad(self.pixsize) / (2. * np.pi))

    def ellgrid(self):
        ellgrid = np.sqrt(np.add.outer(self.ellvec()**2, self.ellvec()**2))
        ellgrid[0, 0] = 1.
        return ellgrid

    def kvec(self):
        return np.fft.fftfreq(self.n1, self.pixsize) * (2. * np.pi / self.pixsize)

    def kgrid(self):
        kgrid = np.sqrt(np.add.outer(self.kvec()**2, self.kvec()**2))
        kgrid[0, 0] = 1.
        return kgrid

    def gauss_herm_grid(self, seed):
        mag = np.random.normal(0, 1, size=[self.n1 + 1] * 2)
        phi = 2 * np.pi * np.random.uniform(size=[self.n1 + 1] * 2)
        revidx = (slice(None, None, -1),) * len(mag.shape)
        mag = (mag + mag[revidx]) / np.sqrt(2)
        phi = (phi - phi[revidx]) / 2 + np.pi
        gh = mag * (np.cos(phi) + 1j * np.sin(phi))
        if not self.n1 % 2:
            cutidx = (slice(None, -1),) * 2
            gh = gh[cutidx]
        return gh

    def downsample(self, data, factor, return_inst=None):
        """ Downsamples data by taking the mean of the smaller pixels.

        Notes:
        ------
        - The resulting grid has a slightly larger extent than the original one
        """

        assert(data.shape == self.shape)
        assert(return_inst in [None, DataPixelGrid, MaskPixelGrid])

        if factor <= 1.:
            warnings.warn("No downsampling possible")
            if return_inst is None:
                return data
            else:
                return return_inst(data, pixsize=self.pixsize, min1=self.min1, min2=self.min2)

        else:
            jump = int(np.floor(factor))
            n1 = int(np.ceil(float(self.n1) / jump))
            n2 = int(np.ceil(float(self.n2) / jump))
            pixsize = jump * self.pixsize
            min1 = self.min1 + (jump - 1) * self.pixsize / 2.
            min2 = self.min2 + (jump - 1) * self.pixsize / 2.
            max1 = min1 + (n1 - 1) * pixsize
            max2 = 2 + (n2 - 1) * pixsize
            extend_1 = (jump - self.n1 % jump) % jump
            extend_2 = (jump - self.n2 % jump) % jump
            res = np.zeros((self.n2 + extend_2, self.n1 + extend_1))
            res[:self.n2, :self.n1] = data.astype(float)
            # Create the downsampled baseline mask
            res = np.mean(np.mean(res.reshape(
                [n2, jump, n1, jump]), axis=3), axis=1)
            if return_inst is None:
                return res
            else:
                return return_inst(res, pixsize=pixsize, min1=min1, min2=min2)

    def mapto(self, data, other, method="interpolation", return_inst=None):
        """
        Maps data between two grids.

        Args:
            data (ndarray):
                The data grid to be remapped. We ssume that the data is given
                on the pixel centers.
            other (:obj:`FlatPixelGrid`):
                The instance that holds the specifications of the grid that
                data is being mapped onto
            method (:obj:`str`):
                Method to be employed for remapping.

        Notes:
            - We employ two different methods for the remapping, 'interpolation'
            and 'average'. The first one interpolates to the closest gridpoint
            and thus looses all the information of the surrounding points. This
            method should be used when the data on each pixel itsel is relevant
            (i.e. when mapping two aperture grids onto each other). The second
            method averages over all the pixels in the bin. This should be used
            when we want to maintain the overall topology of the datagrid (i.e.
            when downsampling a mask).

        TODO: Worry about handeling of grid boundaries s.t. one can define a shape
        to map to properly for downsampling. Not really pressing though...
        """
        assert(data.shape == self.shape)
        assert(method in ["interpolation", "average"])
        assert(return_inst in [None, DataPixelGrid, MaskPixelGrid])

        # If the centers match, all should be fine as the bounds should
        # match after rebinning.
        match_c, match_b = self.check_bounds(other)
        assert(match_c)

        if method == "interpolation":
            if self.nx < other.nx or self.ny < other.ny:
                warnings.warn(
                    "We need to upsample the grid which is physically not sensible.")
            new_data = self.__interpolate(data, other)
        elif method == "average":
            new_data = self.__downsample(data, other)

        if return_inst is None:
            return new_data
        else:
            return return_inst(new_data, pixsize=other.pixsize, min1=other.min1, min2=other.min2)

    # Try to incorporate compress_and_average from .utils here
    def __downsample(self, data, other, return_inst=None):
        raise NotImplementedError

    # Maybe revert to congid from .utils here which is more memory efficient
    def __interpolate(self, data, other, return_inst=None):
        sc1, sc2 = self.get_pix_centers()
        oc1, oc2 = other.get_pix_centers()
        spl = interp2d(sc1, sc2, data, kind='linear',
                       bounds_error=False, fill_value=0)
        return spl(oc1, oc2)

    def check_bounds(self, other, rel_tol=1e-9):
        """ Make sure that either bin centers or bin bounds of two instances match
        """
        match_c1 = isclose(self.min1, other.min1, rel_tol=rel_tol)
        match_c2 = isclose(self.min2, other.min2, rel_tol=rel_tol)
        match_b1 = isclose(
            self.min1_bound, other.min1_bound, rel_tol=rel_tol)
        match_b2 = isclose(
            self.min2_bound, other.min2_bound, rel_tol=rel_tol)
        assert((match_c1 and match_c2) or (match_b1 and match_b2))
        return match_c1 and match_c2, match_b1 and match_b2

    def get_shiftvectors(self, d_max, include_reflections=False):
        """ Returns all shifts on the grid whose length is less that d_max. 
        We return the following sets (shift0, shift1 > 0):
        * [shift0, 0]
        * [0, shift1]
        * [shift0, shift1]
        * [-shift0, shift1]
        This gives all the shift vectors from which pair counts can
        be computed, modulo reflection.

        If 'include_reflections' is set to True we return the whole set of
        shift vectors which is twice as long as the reduced one.
        """

        # Construct reduced shiftvector
        # We force them to be smaller than the extent of the grid.
        nshiftsx = min(self.n1, int(d_max / self.pixsize))
        nshiftsy = min(self.n2, int(d_max / self.pixsize))
        shifts = []
        radii = []
        for shiftx in range(nshiftsx):
            for shifty in range(nshiftsy):
                r = np.sqrt((shiftx * self.pixsize)**2 +
                            (shifty * self.pixsize)**2)
                if r > 0 and r <= d_max:
                    radii.append(r)
                    shifts.append([shifty, shiftx])
                    if shiftx != 0 and shifty != 0:
                        radii.append(r)
                        shifts.append([-shifty, shiftx])

        # Explicity include all the reflections
        if include_reflections:
            nshifts = len(shifts)
            for el in range(nshifts):
                radii.append(radii[el])
                shifts.append(list(-np.asarray(shifts[el])))

        # Sort wrt radii of the shiftvectors
        _sorter = np.argsort(np.asarray(radii))
        radii = np.asarray(radii)[_sorter]
        shifts = np.asarray(shifts)[_sorter]

        return radii, shifts

class FFTDirectEstimator(FlatPixelGrid):

    def __init__(self, pixsize, grid=None, min1=None, min2=None, max1=None, max2=None, n1=None, n2=None, units="arcmin"):
        self.pixsize = pixsize

        #############################
        ## Link compiled libraries ##
        #############################
        # Method that works for LP
        target_path = __import__('orpheus').__file__
        self.library_path = str(Path(__import__('orpheus').__file__).parent.parent.absolute())
        self.clib = ct.CDLL(glob.glob(self.library_path+"/orpheus_clib*.so")[0])
        
        # In case the environment is weird, compile code manually and load it here...
        #self.clib = ct.CDLL("/vol/euclidraid4/data/lporth/HigherOrderLensing/Estimator/orpheus/orpheus/src/discrete.so")
        
        # Method that works for RR (but not for LP with a local HPC install)
        #self.clib = ct.CDLL(search_file_in_site_package(get_site_packages_dir(),"orpheus_clib"))
        #self.library_path = str(Path(__import__('orpheus').__file__).parent.parent.absolute())
        #print(self.library_path)
        #print(self.clib)
        p_c128 = ndpointer(complex, flags="C_CONTIGUOUS")
        p_f64 = ndpointer(np.float64, flags="C_CONTIGUOUS")
        p_f32 = ndpointer(np.float32, flags="C_CONTIGUOUS")
        p_i32 = ndpointer(np.int32, flags="C_CONTIGUOUS")
        p_f64_nof = ndpointer(np.float64)

        if grid is not None:
            self.grid = grid
            super().__init__(pixsize=self.pixsize, shape=(self.grid["gamma"].shape[0],grid["gamma"].shape[1]))
    
    def makegrid(self, x, y, gamma1, gamma2, weight, zbingals, isinnermask, zlensbin, zsourcebin, 
                 lensfrac=1., dec=None, ra=None, orderN=1, orderM=1):
        galbinning = Galaxy_binning(x, y, gamma1, gamma2, weight, zbingals, isinnermask, dec, ra, orderN, orderM)
        galbinning.selectzbins(zlensbin=zlensbin, zsourcebin=zsourcebin, lensfrac=lensfrac)
        galbinning.galaxy_grid(orderM=orderM)
        super().__init__(pixsize=self.pixsize, shape=(self.grid["gamma"].shape[0],self.grid["gamma"].shape[1]))
    
    @staticmethod
    def getstrfromint(i):
        if i==1:
            return ""
        else:
            return str(i)
    
    def getstatnames(self, orderN, orderM):
        stats = []
        orderlist = []
        if orderN>0:
            for elN in range(1,orderN+1):
                stats.append("Nap"+self.getstrfromint(elN))
                orderlist.append((elN,0,0))
        if orderM>0:
            for elM in range(1,orderM+1):
                stats.append("Map"+self.getstrfromint(elM))
                orderlist.append((0,elM,0))
                stats.append("Mx"+self.getstrfromint(elM))
                orderlist.append((0,0,elM))
            for ord in range(2,orderM+1):
                for elM in range(1,ord):
                    stats.append("Map"+self.getstrfromint(elM)+"Mx"+self.getstrfromint(ord-elM))
                    orderlist.append((0,elM,ord-elM))
        if orderN>0 and orderM>0:
            for elN in range(1,orderN+1):
                for elM in range(1,orderM+1):
                    stats.append("Nap"+self.getstrfromint(elN)+"Map"+self.getstrfromint(elM))
                    orderlist.append((elN,elM,0))
                    stats.append("Nap"+self.getstrfromint(elN)+"Mx"+self.getstrfromint(elM))
                    orderlist.append((elN,0,elM))
                for ord in range(2,orderM+1):
                    for elM in range(1,ord):
                        stats.append("Nap"+self.getstrfromint(elN)+"Map"+self.getstrfromint(elM)+"Mx"+self.getstrfromint(ord-elM))
                        orderlist.append((elN,elM,ord-elM))
        return stats, orderlist
    
class Galaxy_binning():
    
    def __init__(self, pixsize, x, y, gamma1, gamma2, weight, zbingals, isinnermask,
                 zlensbin, zsourcebin, lensfrac=1., dec=None, ra=None, orderN=1, orderM=1):
        self.pixsize = pixsize
        self.x = x.astype(np.double)
        self.y = y.astype(np.double)
        self.gamma1 = gamma1.astype(np.double)
        self.gamma2 = gamma2.astype(np.double)
        self.weight = weight.astype(np.double)
        self.orderN = orderN
        self.orderM = orderM
        self.zbingals = zbingals
        self.isinnermask = isinnermask
        self.zlensbin = zlensbin
        self.zsourcebin = zsourcebin
        self.lensfrac = lensfrac
        self.dec = dec
        self.ra = ra
        self.orderM = orderM

        #############################
        ## Link compiled libraries ##
        #############################
        # Method that works for LP
        target_path = __import__('orpheus').__file__
        self.library_path = str(Path(__import__('orpheus').__file__).parent.parent.absolute())
        self.clib = ct.CDLL(glob.glob(self.library_path+"/orpheus_clib*.so")[0])
        
        c_args = [ndpointer(ct.c_double, flags="C_CONTIGUOUS"), ndpointer(ct.c_double, flags="C_CONTIGUOUS"), 
                  ndpointer(ct.c_double, flags="C_CONTIGUOUS"), ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
                  ct.c_int, ct.c_int, ct.c_int, 
                  ndpointer(ct.c_double, flags="C_CONTIGUOUS"), ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
                  ct.c_int,
                  ndpointer(ct.c_double, flags="C_CONTIGUOUS"), ndpointer(ct.c_int, flags="C_CONTIGUOUS"),
                  ndpointer(ct.c_int, flags="C_CONTIGUOUS"), ndpointer(ct.c_int, flags="C_CONTIGUOUS")]
        self.StatisticType = {"STAT_MEAN": 0, "STAT_SUM": 1, "STAT_COUNT": 2, "STAT_WEIGHTED_MEAN": 3}
        self.clib.binned_statistic_2d.restype = None
        self.clib.binned_statistic_2d.argtypes = c_args
    
    def makegrid(self):
        self.grid = {"pixsize":self.pixsize}
        self.selectzbins()
        self.galaxy_grid()
        return self.grid
    
    def selectzbins(self):
        self.z_lens = self.zbingals==self.zlensbin
        self.z_source = self.zbingals==self.zsourcebin
        if self.lensfrac<1.:
            n_true = int(self.lensfrac*len(self.z_lens[self.z_lens==True])) # number of lenses to use
            lens_reduce = np.zeros(len(self.z_lens[self.z_lens==True])-n_true, dtype=int)
            # reduced array with True and False according to number of used and not used lenses
            lens_reduce = np.concatenate((lens_reduce, np.ones(n_true, dtype=int))).astype(bool)
            rng = np.random.default_rng()
            rng.shuffle(lens_reduce)
            self.z_lens[self.z_lens==True] = lens_reduce # substitute lens mask (with all lenses) with reduced lens mask

    def galaxy_grid(self):
        # Compute bin edges
        self.bins = np.arange(min(np.min(self.x),np.min(self.y))-0.5*self.pixsize,
                         max(np.max(self.x),np.max(self.y))+1.5*self.pixsize, self.pixsize, dtype=np.double)
        self.nbins = len(self.bins)-1
        # Bin the positions and shears of source galaxies
        xs = self.x[self.z_source]
        ys = self.y[self.z_source]
        self.ngals = len(xs)
        N_source = np.zeros(self.nbins*self.nbins, dtype=np.double)
        counts = np.zeros(self.nbins*self.nbins, dtype=np.int32)
        binnumberx = np.zeros(self.ngals, dtype=np.int32)
        binnumbery = np.zeros(self.ngals, dtype=np.int32)
        self.clib.binned_statistic_2d(xs, ys, np.ones_like(xs), np.ones_like(xs), self.ngals, self.nbins, self.nbins, self.bins, self.bins,
                                     self.StatisticType["STAT_COUNT"], N_source, counts, binnumberx, binnumbery)
        N_source = np.reshape(N_source, (self.nbins,self.nbins), order="C")
        
        weights = np.zeros((self.nbins*self.nbins, self.orderM), dtype=np.double)
        nMMstar = int(np.ceil((self.orderM+1)/2))
        gamma_grid = np.zeros((self.nbins*self.nbins, self.orderM, nMMstar), dtype=complex)
        gamma_complex = self.gamma1[self.z_source] + 1j*self.gamma2[self.z_source]
        for ord in range(1,self.orderM+1):
            result = np.zeros(self.nbins*self.nbins, dtype=np.double)
            counts = np.zeros(self.nbins*self.nbins, dtype=np.int32)
            self.clib.binned_statistic_2d(xs, ys, self.weight[self.z_source]**ord, np.ones_like(xs), self.ngals, self.nbins, self.nbins, self.bins, self.bins,
                                         self.StatisticType["STAT_SUM"], result, counts, binnumberx, binnumbery)
            weights[...,ord-1] += result
            for star in range(int(np.ceil((ord+1)/2))):
                gammacomb = gamma_complex**(ord-star)*np.conj(gamma_complex)**star
                gammahelpreal = np.zeros(self.nbins*self.nbins, dtype=np.double)
                counts = np.zeros(self.nbins*self.nbins, dtype=np.int32)
                self.clib.binned_statistic_2d(xs, ys, np.ascontiguousarray(np.real(gammacomb)), self.weight[self.z_source]**ord, self.ngals, self.nbins, self.nbins, self.bins, self.bins,
                                             self.StatisticType["STAT_WEIGHTED_MEAN"], gammahelpreal, counts, binnumberx, binnumbery)
                gammahelpimag = np.zeros(self.nbins*self.nbins, dtype=np.double)
                counts = np.zeros(self.nbins*self.nbins, dtype=np.int32)
                self.clib.binned_statistic_2d(xs, ys, np.ascontiguousarray(np.imag(gammacomb)), self.weight[self.z_source]**ord, self.ngals, self.nbins, self.nbins, self.bins, self.bins,
                                             self.StatisticType["STAT_WEIGHTED_MEAN"], gammahelpimag, counts, binnumberx, binnumbery)
                gamma_grid[...,ord-1,star] = gammahelpreal+1j*gammahelpimag
        
        gamma_grid = np.reshape(gamma_grid, (self.nbins,self.nbins,self.orderM,nMMstar), order="C")
        weights = np.reshape(weights, (self.nbins,self.nbins,self.orderM), order="C")
                
        # Bin the positions of lens galaxies
        xl = self.x[self.z_lens]
        yl = self.y[self.z_lens]
        self.ngall = len(xl)
        result = np.zeros(self.nbins*self.nbins, dtype=np.double)
        counts = np.zeros(self.nbins*self.nbins, dtype=np.int32)
        binnumberx = np.zeros(self.ngall, dtype=np.int32)
        binnumbery = np.zeros(self.ngall, dtype=np.int32)
        self.clib.binned_statistic_2d(xl, yl, np.ones_like(xl), np.ones_like(xl), self.ngall, self.nbins, self.nbins, self.bins, self.bins,
                                     self.StatisticType["STAT_COUNT"], result, counts, binnumberx, binnumbery)
        N_lens = np.reshape(counts.astype(np.double), (self.nbins,self.nbins), order="C")
        self.binnumber = np.array([binnumberx, binnumbery])
        
        # Mask for the inner region of data to avoid boundary effects on Map
        if self.isinnermask is None:
            self.isinnermask = np.ones_like(self.x, dtype=int).astype(bool)
        result = np.zeros(self.nbins*self.nbins, dtype=np.double)
        counts = np.zeros(self.nbins*self.nbins, dtype=np.int32)
        binnumberx = np.zeros(len(self.x), dtype=np.int32)
        binnumbery = np.zeros(len(self.x), dtype=np.int32)
        self.clib.binned_statistic_2d(self.x, self.y, self.isinnermask.astype(np.double), np.ones_like(self.x), len(self.x), self.nbins, self.nbins, self.bins, self.bins,
                                     self.StatisticType["STAT_SUM"], result, counts, binnumberx, binnumbery)
        field_sel = np.reshape(result, (self.nbins,self.nbins), order="C")
        
        field_sel[field_sel>0] = 1
        field_sel = field_sel.astype(bool)
        # Fill up all gaps inside the inner region of field_sel (which are still False because there is no galaxy)
        for i in range(field_sel.shape[0]):
            trues = np.argwhere(field_sel[i,:]==True) # indices with an entry True in each line
            if trues.size>0:
                for j in range(np.min(trues),np.max(trues)):
                    field_sel[i,j]=True # set all entries to True inside the inner region
        
        # CIC sampling method:
        #N_lens = self.sample_galaxies_on_grid(xl,yl, method="CIC")
        self.grid.update({"N_lens":N_lens, "gamma":gamma_grid, "weight_Map":weights,
                "field_sel":field_sel, "N_lens_gal_isinner":np.sum(N_lens[field_sel]),
                "N_lens_gal_tot":np.sum(N_lens), "N_source_gal_isinner":np.sum(N_source[field_sel]),
                "N_source_gal_tot":np.sum(N_source), "A_inner":len(N_lens[field_sel])*self.pixsize**2})
        if self.dec is not None and self.ra is not None:
            self.grid["dec"] = np.mean(self.dec[self.isinnermask])
            self.grid["ra"] = np.mean(self.ra[self.isinnermask])
        
    @staticmethod
    def mass_assignment(dx, dy, method):
        if method=="CIC":
            w_x = np.where(dx<1., 1.-dx, 0.)
            w_y = np.where(dy<1., 1.-dy, 0.)
        elif method=="NGP":
            w_x = np.where(dx<0.5, 1., 0.)
            w_y = np.where(dy<0.5, 1., 0.)
        elif method=="TSC":
            w_x = np.where((dx>0.5) & (dx<1.5), 0.5*(1.5-dx)**2, 0.)
            w_x += np.where((dx<=0.5) & (dx>0.), 0.75-dx**2, 0.)
            w_y = np.where((dy>0.5) & (dy<1.5), 0.5*(1.5-dy)**2, 0.)
            w_y += np.where((dy<=0.5) & (dy>0.), 0.75-dy**2, 0.)
        return w_x, w_y

    def sample_galaxies_on_grid(self, xgal, ygal, method="CIC"):
        # Bin centers
        bins_center = 0.5*(self.bins[1:]+self.bins[:-1])
        # distance of each lens galaxy to nearest pixel in units of pixel size
        diff_x = (xgal-bins_center[self.binnumber[0,:]])/self.pixsize
        diff_y = (ygal-bins_center[self.binnumber[1,:]])/self.pixsize

        # Compute galaxy density field via CIC-method
        # Distribute galaxy weight on four nearest pixels
        # Check whether a part of the galaxy weight goes into the left or right (lower or upper) neighbouring pixel
        xstep = np.where(diff_x>0., 1,-1)
        ystep = np.where(diff_y>0., 1,-1)
        # edge cases: don't let x and y go out of bounds
        xstep = np.where((self.binnumber[0,:]>1)&(self.binnumber[0,:]<len(bins_center)), xstep, 0)
        ystep = np.where((self.binnumber[1,:]>1)&(self.binnumber[1,:]<len(bins_center)), ystep, 0)

        w_x, w_y = self.mass_assignment(np.abs(diff_x),np.abs(diff_y),method)

        Ngal = np.zeros((len(self.bins)-1, len(self.bins)-1))

        w_norm = w_x*w_y + (1-w_x)*w_y + w_x*(1.-w_y) + (1.-w_x)*(1.-w_y)
        Ngal += np.histogram2d(xgal, ygal, bins=self.bins, weights=w_x*w_y/w_norm)[0]
        Ngal += np.histogram2d(xgal+xstep*self.pixsize, ygal, bins=self.bins, weights=(1.-w_x)*w_y/w_norm)[0]
        Ngal += np.histogram2d(xgal, ygal+ystep*self.pixsize, bins=self.bins, weights=w_x*(1.-w_y)/w_norm)[0]
        Ngal += np.histogram2d(xgal+xstep*self.pixsize, ygal+ystep*self.pixsize, bins=self.bins, weights=(1.-w_x)*(1.-w_y)/w_norm)[0]
        return Ngal

class FFTprocess(FFTDirectEstimator):
    
    def __init__(self, grid):
        self.cuts = {"Schneider":1, "Crittenden":4}
        if "weight_Map" not in self.grid.keys():
            self.grid["weight_Map"] = np.ones_like(self.grid[list(self.grid)[0]].shape)
        
        self.clib.fftBellRec.restype = None
        self.clib.fftBellRec.argtype = [ct.c_int, ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
                                              ndpointer(ct.c_double, flags="C_CONTIGUOUS"), ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
                                              ct.c_int, ct.c_int]

    def _getFilter(self, radius, rmax, func="U_Schneider", *pars):
        assert(hasattr(self, func))
        func = getattr(self, func)
        if func == self.U_Schneider:
            args = [radius]
        if func == self.U_Crittenden:
            args = [radius, rmax]
        if func == self.Q_Schneider:
            args = [radius]
        if func == self.Q_Crittenden:
            args = [radius, rmax]
        return func(*args, *pars)

    def Q_Schneider(self, R):
        # Setup basic Q
        rgrid, phigrid = self.polargrid(R)
        Q = 6. / (np.pi * R**2) * (rgrid / R)**2 * (1. - (rgrid / R)**2)
        Q *= (rgrid < self.cuts["Schneider"]*R)
        # Get phase correction
        res = Q * np.e**(-2. * 1J * phigrid)
        return -res
    
    # We are using this one
    def Q_Crittenden(self, R, Rmax):
        # Setup basic Q
        rgrid, phigrid = self.polargrid(self.cuts["Crittenden"]*Rmax)
        if self.pixavgfilter:
            x=rgrid*np.cos(phigrid)
            y=rgrid*np.sin(phigrid)
            xerf = (erf((x-0.5*self.pixsize)/(np.sqrt(2)*R)) - erf((x+0.5*self.pixsize)/(np.sqrt(2)*R)))
            yerf = (erf((y-0.5*self.pixsize)/(np.sqrt(2)*R)) - erf((y+0.5*self.pixsize)/(np.sqrt(2)*R)))
            Q = np.exp(-0.5*(y/R)**2)*((y+0.5*self.pixsize)*np.exp(-0.5*y*self.pixsize/R**2) - (y-0.5*self.pixsize)*np.exp(0.5*y*self.pixsize/R**2))*xerf
            Q += np.exp(-0.5*(x/R)**2)*((x+0.5*self.pixsize)*np.exp(-0.5*x*self.pixsize/R**2) - (x-0.5*self.pixsize)*np.exp(0.5*x*self.pixsize/R**2))*yerf
            Q *= np.exp(-0.125*(self.pixsize/R)**2)/(4*np.sqrt(2*np.pi)*R*self.pixsize**2)
            Q += 0.25*xerf*yerf/self.pixsize**2
        else:
            Q = 1. / (4*np.pi * R**2) * (rgrid / R)**2 * np.exp(-rgrid**2/(2*R**2))
            Q *= (rgrid < self.cuts["Crittenden"]*R)
        # Get phase correction
        res = Q * np.e**(-2. * 1J * phigrid)
        return -res

    def U_Schneider(self, R):
        rgrid = self.rgrid(R)
        U = 9. / (np.pi * R**2) * (1. - (rgrid / R)**2) * \
            (1. / 3. - (rgrid / R)**2)
        U *= (rgrid < self.cuts["Schneider"]*R)
        return U / np.sum(rgrid < R)
    
    def U_Crittenden(self, R, Rmax):
        rgrid, phigrid = self.polargrid(self.cuts["Crittenden"]*Rmax)
        if self.pixavgfilter:
            x=rgrid*np.cos(phigrid)
            y=rgrid*np.sin(phigrid)
            U = np.exp(-0.5*(y/R)**2)*((y+0.5*self.pixsize)*np.exp(-0.5*y*self.pixsize/R**2) - (y-0.5*self.pixsize)*np.exp(0.5*y*self.pixsize/R**2)) * \
                (erf((x+0.5*self.pixsize)/(np.sqrt(2)*R)) - erf((x-0.5*self.pixsize)/(np.sqrt(2)*R)))
            U += np.exp(-0.5*(x/R)**2)*((x+0.5*self.pixsize)*np.exp(-0.5*x*self.pixsize/R**2) - (x-0.5*self.pixsize)*np.exp(0.5*x*self.pixsize/R**2)) * \
                (erf((y+0.5*self.pixsize)/(np.sqrt(2)*R)) - erf((y-0.5*self.pixsize)/(np.sqrt(2)*R)))
            U *= np.exp(-0.125*(self.pixsize/R)**2)/(4*np.sqrt(2*np.pi)*R*self.pixsize**2)
        else:
            U = 1. / (2*np.pi * R**2) * (1.-0.5*(rgrid / R)**2) * np.exp(-rgrid**2/(2*R**2))
            U *= (rgrid < self.cuts["Crittenden"]*R)
        return U
    
    def apcounts(self, r, rmax, filter):
        rgrid, _ = self.polargrid(self.cuts[filter]*rmax)
        return (rgrid < self.cuts[filter]*r)
        
    def compNapmoments(self, rN, rmax):
        filtNap = self._getFilter(rN, rmax, "U_"+self.filter)
        lenscountsfilt = self.apcounts(rN, rmax, self.filter)
        # 1 / Number of lenses in the the integration range of every aperture, for normalization
        N_lensinaps = fftconvolve(self.grid[self.fieldNap], lenscountsfilt, mode="same").real
        eps_norm = 1e-4
        N_lensinaps_inv = np.divide(1., N_lensinaps, out=np.zeros_like(N_lensinaps), where=N_lensinaps>eps_norm)
        N_pix = np.sum(lenscountsfilt)
                    
        Nbar = np.mean(self.grid[self.fieldNap][self.grid["field_sel"]])
        #starttime = time.time()
        # Compute Nsm fields, normalized by the m-th power of Nbar
        Nsm_Nnorm = np.zeros(self.shapeN)
        for order in range(1,self.orderN+1):
            Nsm_Nnorm[...,order-1] = fftconvolve(self.grid[self.fieldNap], filtNap**order, mode="same").real
            if self.Nbar_global:
                Nsm_Nnorm[...,order-1] *= (self.pixsize**2/Nbar)**order
            else:
                # Normalize density field, accounting for fluctuating density in different apertures
                Nsm_Nnorm[...,order-1] *= (self.pixsize**2*N_lensinaps_inv*N_pix)**order
            Nsm_Nnorm[...,order-1] *= -self.factorial(order-1) # For the getBellRecursive function
        #print("Nap convolutions: %.2f s"%(starttime-time.time()))

        # Compute all Napm moments for all apertures
        self.Napm = self.getBellRecursive(self.orderN, Nsm_Nnorm, self.Napm)
        # Use getBellRecursive from combinatorics.c, add iteration over apertures
        #factorials = self.factorial(np.arange(170))
        #self.clib.getBellRecursive(self.orderN, np.ravel(self.Napm), factorials, )
        # Compute all pure Nap moments with the help of the Nsm fields
        for ordN in range(1,self.orderN+1):
            # Compute the normalization factors for the Nap moments, and the aperture weights
            self.wNapm[...,ordN-1] *= N_lensinaps
            if ordN>1:
                for k in range(1,ordN):
                    self.wNapm[...,ordN-1] *= N_lensinaps-k
                norm = np.divide(N_lensinaps**ordN, self.wNapm[...,ordN-1], out=np.zeros_like(self.Napm[...,ordN-1]), where=self.wNapm[...,ordN-1]>eps_norm)
                self.Napm[...,ordN-1] *= norm
            self.wNapm[...,ordN-1] = np.where(self.wNapm[...,ordN-1]>eps_norm, self.wNapm[...,ordN-1], 0.)
    
    def compMapmoments(self, rM, rmax):
        filtMap = self._getFilter(rM, rmax, "Q_"+self.filter)
        shearcountsfilt = self.apcounts(rM, rmax, self.filter)  
        A_Map = np.pi*(self.cuts[self.filter]*rM)**2  
        Sm, norm = np.zeros_like(self.grid["weight_Map"]), np.zeros_like(self.grid["weight_Map"])
        for ord in range(1,self.orderM+1):
            norm[...,ord-1] = fftconvolve(self.grid["weight_Map"][...,ord-1], shearcountsfilt, mode="same").real
            Sm[...,ord-1] = np.divide(norm[...,ord-1],norm[...,0]**ord, out=np.zeros_like(norm[...,0]), where=norm[...,0]>0.)
            Sm[...,ord-1] *= -self.factorial(ord-1) # For the getBellRecursive function
        
        #starttime = time.time()
        # Compute Msm fields, normalized by the shear weights
        self.Msm = np.zeros(self.shapeM, dtype=complex)
        for ord in range(1,self.orderM+1):
            for star in range(int(np.ceil((ord+1)/2))):
                self.Msm[...,ord-1,star] = A_Map**ord*fftconvolve(self.grid["weight_Map"][...,ord-1]*self.grid[self.fieldMap][...,ord-1,star], 
                                                             filtMap**(ord-star)*np.conj(filtMap)**star, mode="same")
                self.Msm[...,ord-1,star] = np.divide(self.Msm[...,ord-1,star], norm[...,0]**ord, out=np.zeros_like(self.Msm[...,ord-1,star]), where=norm[...,0]>0.)
                if star!=ord-star and ord-star<=self.nMMstar-1:
                    self.Msm[...,ord-1,ord-star] = np.conj(self.Msm[...,ord-1,star])
        #print("Map convolutions: %.2f s"%(starttime-time.time()))
        #starttime = time.time()
        for ord in range(1,self.orderM+1):
            for star in range(int(np.ceil((ord+1)/2))):
                self.getMapmoment(ord, ord-star)
        #print("getMapmoments: %.2f s"%(starttime-time.time()))
        # Compute all Mapm moments for all apertures, and the corresponding weight normalisation
        norm_Mapm = np.zeros(self.shapewM, dtype=float)
        norm_Mapm = self.getBellRecursive(self.orderM, Sm, norm_Mapm)
        for ordM in range(1,self.orderM+1):
            norm_div = np.divide(1., norm_Mapm[...,ordM-1], out=np.zeros_like(norm_Mapm[...,ordM-1]), where=norm_Mapm[...,ordM-1]>0.)
            for star in range(self.nMMstar):
                self.Mapm[...,ordM-1,star] *= norm_div
            self.wMapm[...,ordM-1] = np.where(norm[...,0]>0., norm[...,0], 0.)
        
        

    def getStats(self, Rap, samplingrate, orderN=1, orderM=1,
                 fieldNap="N_lens", fieldMap="gamma", Nbar_global=False,
                 pixavgfilter=False, filter="Crittenden"):
        Stats = {"Rap":Rap}
        leaveout = [fieldNap,fieldMap,"weight_Map","field_sel"]
        for key in self.grid.keys():
            if key not in leaveout:
                Stats[key] = self.grid[key]
        nRap = Rap.shape[1]
        statnames, orderlist = self.getstatnames(orderN, orderM)
        nsampling = len(samplingrate)
        for key in statnames:
            Stats[key] = np.zeros((nsampling,nRap))
        Stats["stats"] = statnames
        Stats["orderlist"] = orderlist
        Stats["samplingrate"] = samplingrate
        
        assert(fieldNap in self.grid.keys())
        assert(fieldMap in self.grid.keys())
        #assert(orderN == self.grid["N_lens"].shape[2])
        if "field_sel" not in self.grid.keys():
            self.grid["field_sel"] = np.ones_like(self.grid[fieldNap][...,0])
        
        self.orderN = orderN
        self.orderM = orderM
        self.fieldNap = fieldNap
        self.fieldMap = fieldMap
        self.Nbar_global = Nbar_global
        self.pixavgfilter = pixavgfilter
        self.filter = filter

        self.shapeN = (self.grid[fieldNap].shape[0], self.grid[fieldNap].shape[1], orderN)
        #self.shapeM = (self.grid[fieldMap].shape[0], self.grid[fieldMap].shape[1], orderM)
        self.nMMstar = int(np.ceil((self.orderM+1)/2))
        self.shapeM = (self.grid[fieldMap].shape[0], self.grid[fieldMap].shape[1], orderM, self.nMMstar)
        self.shapewM = (self.grid[fieldMap].shape[0], self.grid[fieldMap].shape[1], orderM)
        
        for elr in range(nRap):
            rN = Rap[0,elr]
            rM = Rap[1,elr]
            rmax = max(rM, rN)
            #starttime=time.time()
            if orderN>0:
                self.Napm = np.zeros(self.shapeN, dtype=float)
                self.wNapm = np.ones(self.shapeN, dtype=float)
                self.compNapmoments(rN,rmax)
                #starttime=time.time()
                # Average the Nap moments over the whole survey field (aperture weighted)
                for elsr,sr in enumerate(samplingrate):
                    aperture_mask = self.getaperturemask(rN, sr)
                    for ordN in range(1,orderN+1):
                        Stats["Nap"+self.getstrfromint(ordN)][elsr,elr] = np.average(self.Napm[...,ordN-1][aperture_mask], 
                                                                                weights=self.wNapm[...,ordN-1][aperture_mask])
                #print("Nap averaging: %.2f s"%(starttime-time.time()))
            
            if orderM>0:
                self.Mapm = np.zeros(self.shapeM, dtype=complex)
                self.wMapm = np.ones(self.shapewM, dtype=float)
                #starttime=time.time()
                self.compMapmoments(rM,rmax)
                #print("Map moments: %.2f s"%(starttime-time.time()))
                #starttime=time.time()
                # Average the Map moments over the whole survey field (aperture weighted)                
                for ordM in range(1,self.orderM+1):
                    for m in range(ordM+1):
                        Ks = self.getKsnm(ordM,m)
                        MMstarfield = 0.
                        for elk,k in enumerate(Ks):
                            MMstarfield += k*self.Mapm[...,ordM-1,elk]
                        name = statnames[np.argwhere(np.sum(np.abs(np.array(orderlist)-[0,m,ordM-m]), axis=1)==0)[0,0]]
                        for elsr,sr in enumerate(samplingrate):
                            aperture_mask = self.getaperturemask(rM, sr)
                            if (ordM-m)%2==0:
                                Stats[name][elsr,elr] = np.average(MMstarfield[aperture_mask].real, 
                                                                    weights=self.wMapm[...,ordM-1][aperture_mask])
                            else:
                                Stats[name][elsr,elr] = np.average(MMstarfield[aperture_mask].imag, 
                                                                    weights=self.wMapm[...,ordM-1][aperture_mask])
                #print("Map averaging: %.2f s"%(starttime-time.time()))    
            
            #starttime=time.time()
            if orderN>0 and orderM>0:
                for ordN in range(1,orderN+1):
                    for ordM in range(1,self.orderM+1):
                        for m in range(ordM+1):
                            Ks = self.getKsnm(ordM,m)
                            NapMMstarfield = 0.
                            for elk,k in enumerate(Ks):
                                NapMMstarfield += k*self.Mapm[...,ordM-1,elk]
                            if (ordM-m)%2==0:
                                NapMMstarfield = self.Napm[...,ordN-1]*NapMMstarfield.real
                            else:
                                NapMMstarfield = self.Napm[...,ordN-1]*NapMMstarfield.imag
                            wNapMapfield = self.wNapm[...,ordN-1]*self.wMapm[...,ordM-1]
                            name = statnames[np.argwhere(np.sum(np.abs(np.array(orderlist)-[ordN,m,ordM-m]), axis=1)==0)[0,0]]
                            for elsr,sr in enumerate(samplingrate):
                                aperture_mask = self.getaperturemask(min(rM,rN), sr)
                                Stats[name][elsr,elr] = np.average(NapMMstarfield[aperture_mask], weights=wNapMapfield[aperture_mask])
            #print("NapMap averaging: %.2f s"%(starttime-time.time()))
            #print("Nap,Map,NapMap : %.2f s"%(starttime-time.time()))
        return Stats
    
    def getstatmap(self, orderN, orderM, nMap, rN, rM, samplingrate,
                   fieldNap="N_lens", fieldMap="gamma", Nbar_global=False, filter="Crittenden"):
        self.orderN = orderN
        self.orderM = orderM
        self.fieldNap = fieldNap
        self.fieldMap = fieldMap
        self.Nbar_global = Nbar_global
        self.filter = filter
        self.nMMstar = int(np.ceil((self.orderM+1)/2))
        self.shapeM = (self.grid[fieldMap].shape[0], self.grid[fieldMap].shape[1], orderM, self.nMMstar)
        self.shapewM = (self.grid[fieldMap].shape[0], self.grid[fieldMap].shape[1], orderM)
        mapdict = {"orderN":orderN, "orderM":orderM, "rN":rN, "rM":rM, "samplingrate":samplingrate}
        if orderN>0:
            self.shapeN = (self.grid[fieldNap].shape[0], self.grid[fieldNap].shape[1], orderN)
            self.Napm = np.zeros(self.shapeN, dtype=float)
            self.wNapm = np.ones(self.shapeN, dtype=float)
            self.compNapmoments(rN,max(rN,rM))
            if orderM==0:
                aperture_mask = self.getaperturemask(rN, samplingrate)
                mapdict["statmap"] = self.Napm[...,orderN-1]
                mapdict["aperturemask"] = aperture_mask
        if orderM>0:
            self.Mapm = np.zeros(self.shapeM, dtype=complex)
            self.wMapm = np.ones(self.shapewM, dtype=float)
            self.compMapmoments(rM,max(rM, rN))
            if orderN==0:
                Ks = self.getKsnm(orderM,nMap)
                MMstarfield = 0.
                for elk,k in enumerate(Ks):
                    MMstarfield += k*self.Mapm[...,orderM-1,elk]
                aperture_mask = self.getaperturemask(rM, samplingrate)
                if (orderM-nMap)%2==0:
                    mapdict["statmap"] = MMstarfield.real
                else:
                    mapdict["statmap"] = MMstarfield.imag
                mapdict["aperturemask"] = aperture_mask
        if orderN>0 and orderM>0:
            Ks = self.getKsnm(orderM,nMap)
            NapMMstarfield = 0.
            for elk,k in enumerate(Ks):
                NapMMstarfield += k*self.Mapm[...,orderM-1,elk]
            if (orderM-nMap)%2==0:
                NapMMstarfield = self.Napm[...,orderN-1]*NapMMstarfield.real
            else:
                NapMMstarfield = self.Napm[...,orderN-1]*NapMMstarfield.imag
            aperture_mask = self.getaperturemask(min(rM,rN), samplingrate)
            mapdict["statmap"] = NapMMstarfield
            mapdict["aperturemask"] = aperture_mask
        return mapdict
                    
    def getaperturemask(self,r,samplingrate):
        samplingsep = max(int(r/self.pixsize/samplingrate), 1)
        aperture_mask = np.zeros_like(self.grid["field_sel"],dtype=int)
        aperture_mask[::samplingsep,::samplingsep] = 1
        return (self.grid["field_sel"]==True)&(aperture_mask==1)

    def factorial(self, n):
        if n>1:
            return n*self.factorial(n-1)
        else:
            return 1
    
    def getBellRecursive(self, order, arguments, result):
        result[...,0]=arguments[...,0]
        for idxn in range(2,order+1):
            result[...,idxn-1] += arguments[...,idxn-1]
            for idxk in range(idxn-1):
                binom = self.factorial(idxn-1)/(self.factorial(idxk)*self.factorial(idxn-1-idxk))
                result[...,idxn-1] += binom*result[...,idxn-2-idxk]*arguments[...,idxk]
        for k in range(order):
            result[...,k] *= (-1)**(k+1)
        return result
    
    def getMapmoment(self, order, nM):
        for el in range(self.bell(order)):
            if el == 0:
                rgs, m = self.firstrgs(order)
            else:
                rgs, m = self.nextrgs(rgs, m)
            self.Mapm[...,order-1,order-nM] += self.rgs2MsProd(rgs,nM)

    def rgs2MsProd(self, rgs, nM):
        n = len(rgs)
        part = self.rgs2part(rgs)
        res = 1.
        # Get sign
        if (n+np.max(rgs))%2:
            sign = 1.
        else:
            sign = -1.
        pref = 1.
        for i in range(np.max(rgs)+1):
            # Get prefactor
            pref *= self.factorial(len(np.argwhere(rgs==i)) - 1)
            # Get expr
            rel_part = part[n*i:n*(i+1)]
            ord = int(np.sum(rel_part))
            star = ord - int(np.sum(rel_part[:nM]))
            res *= self.Msm[...,ord-1,star]
        res *= sign*pref
        return res

    @staticmethod
    def firstrgs(n):
        k = np.zeros(n).astype(np.int32)
        m = np.zeros(n).astype(np.int32)
        return k, m

    @staticmethod
    def nextrgs(k, m, order=None):
        """ Returns the next largest rgs """
        if order is None:
            order = len(k) 
        i = order-1

        while i>=1:
            if k[i] <= m[i-1]:
                k[i] += 1
                m[i] = max(m[i],k[i])
                for j in range(i+1, order):
                    k[j] = k[0]
                    m[j] = m[i]
                return (k,m)
            i -=1
        
    @staticmethod
    def rgs2part(rgs):
        """ Maps rgs to partition selecection
        i.e. (0,1,0,2) --> [1 0 1 0   0 1 0 0   0 0 0 1]  
        """
        lensel = len(rgs)
        nparts = np.max(rgs)+1
        res = np.zeros(lensel*nparts)
        for ind, el in enumerate(rgs):
            res[lensel*el+ind] += 1
        return res

    @staticmethod
    def bell(n): 
        bell = [[0 for i in range(n+1)] for j in range(n+1)] 
        bell[0][0] = 1
        for i in range(1, n+1): 
            # Explicitly fill for j = 0 
            bell[i][0] = bell[i-1][i-1] 
            # Fill for remaining values of j 
            for j in range(1, i+1): 
                bell[i][j] = bell[i-1][j-1] + bell[i][j-1] 
        return bell[n][0]

    def getbincoeff(self, n,k):
        return (self.factorial(n)/(self.factorial(k)*self.factorial(n-k)))

    def getpnms(self, n,m,s):
        res=0
        for k in range(n-s+1):
            for l in range(s+1):
                if k+l==m:
                    res += self.getbincoeff(n-s,k)*self.getbincoeff(s,l)*(-1)**(s-l)
        return res

    def getKsnm(self, n,m):
        nrow = np.ceil((n+1)/2).astype(int)
        evorodd = 0
        if (n-m)%2!=0:
            nrow = np.floor((n+1)/2).astype(int)
            evorodd = 1
        m_arr = np.zeros(nrow)
        b = np.zeros(nrow)
        if n%2==0:
            for i in range(nrow):
                m_arr[i] = 2*i+evorodd
                if m_arr[i]==m:
                    b[i] = (-1.)**((n-m-evorodd)/2)
        else:
            for i in range(nrow):
                m_arr[i] = 2*i+1-evorodd
                if m_arr[i]==m:
                    b[i] = (-1.)**((n-m-evorodd)/2)
        pmatrix = np.zeros((nrow,nrow))
        for elx,x in enumerate(m_arr):
            for s in range(nrow):
                pmatrix[elx,s] = self.getpnms(n,x,s)
        return np.linalg.solve(pmatrix, b)
