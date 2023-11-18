import ctypes as ct
import numpy as np 
from numpy.ctypeslib import ndpointer
from pathlib import Path
import glob
import sys
import time


#__all__ = ["Catalog", "ScalarCatalog", "PolarCatalog", "ScalarAndPolarCatalog",
#          "TwoDimGrid", "GriddedCatalog", "GriddedScalarCatalog", "GriddedScalarCatalog"]

__all__ = ["Catalog", "ScalarTracerCatalog", "SpinTracerCatalog", "MultiTracerCatalog"]
    
    
##############################################
## Classes that deal with discrete catalogs ##
##############################################
class Catalog:
    
    def __init__(self, pos1, pos2, weight=None, zbins=None, isinner=None):
        self.pos1 = pos1.astype(np.float64)
        self.pos2 = pos2.astype(np.float64)
        self.weight = weight
        self.zbins = zbins
        self.ngal = len(self.pos1)
        # Normalize weight s.t. <weight> = 1
        if self.weight is None:
            self.weight = np.ones(self.ngal)
        self.weight = self.weight.astype(np.float64)
        #self.weight /= np.mean(self.weight)
        # Require zbins to only contain elements in {0, 1, ..., nbinsz-1}
        if self.zbins is None:
            self.zbins = np.zeros(self.ngal)        
        self.zbins = self.zbins.astype(np.int)
        self.nbinsz = len(np.unique(self.zbins))
        assert(np.max(self.zbins)-np.min(self.zbins)==self.nbinsz-1)
        self.zbins -= (np.min( self.zbins))
        if isinner is None:
            isinner = np.ones(self.ngal, dtype=np.float64)
        self.isinner = np.asarray(isinner, dtype=np.float64)
        assert(np.min(self.isinner) >= 0.)
        assert(np.max(self.isinner) <= 1.)
        assert(len(self.isinner)==self.ngal)
        assert(len(self.pos2)==self.ngal)
        assert(len(self.weight)==self.ngal)
        assert(len(self.zbins)==self.ngal)
        
        self.min1 = np.min(self.pos1)
        self.min2 = np.min(self.pos2)
        self.max1 = np.max(self.pos1)
        self.max2 = np.max(self.pos2)
        self.len1 = self.max1-self.min1
        self.len2 = self.max2-self.min2
        
        self.spatialhash = None
        self.hasspatialhash = False
        self.index_matcher = None
        self.pixs_galind_bounds = None
        self.pix_gals = None
        self.pix1_start = None
        self.pix1_d = None
        self.pix1_n = None
        self.pix2_start = None
        self.pix2_d = None
        self.pix2_n = None
        
        self.assign_methods = {"NGP":0, "CIC":1, "TSC":2}
        
        ## Link compiled libraries ##
        target_path = __import__('orpheus').__file__
        #print(target_path)
        self.library_path = str(Path(__import__('orpheus').__file__).parent.parent.absolute())
        #print(self.library_path)
        self.clib = ct.CDLL(glob.glob(self.library_path+"/orpheus_clib*.so")[0])
        #self.library_path = str(Path(__file__).parent.absolute()) + "/src/"
        #self.clib = ct.CDLL(self.library_path + "clibrary.so")
        p_c128 = ndpointer(np.complex128, flags="C_CONTIGUOUS")
        p_f64 = ndpointer(np.float64, flags="C_CONTIGUOUS")
        p_f32 = ndpointer(np.float32, flags="C_CONTIGUOUS")
        p_i32 = ndpointer(np.int32, flags="C_CONTIGUOUS")
        p_f64_nof = ndpointer(np.float64)
        
        # Assigns a set of tomographic fields over a grid
        # Safely called within 'togrid' function
        self.clib.assign_fields.restype = ct.c_void_p
        self.clib.assign_fields.argtypes = [
            p_f64, p_f64, p_i32, p_f64, p_f64, ct.c_int32, ct.c_int32, ct.c_int32, 
            ct.c_int32, ct.c_double, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
            ct.c_int32, np.ctypeslib.ndpointer(dtype=np.float64)]
        
        # Assigns a set of tomographic fields over a grid
        # Safely called within 'togrid' function
        self.clib.gen_weightgrid2d.restype = ct.c_void_p
        self.clib.gen_weightgrid2d.argtypes = [
            p_f64, p_f64, ct.c_int32, ct.c_int32, 
            ct.c_double, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32, 
            np.ctypeslib.ndpointer(dtype=np.int32),
            np.ctypeslib.ndpointer(dtype=np.float64)]
        
        # Generate pixel --> galaxy mapping
        # Safely called within other wrapped functions
        self.clib.build_spatialhash.restype = ct.c_void_p
        self.clib.build_spatialhash.argtypes = [
            p_f64, p_f64, ct.c_int32, ct.c_double, ct.c_double, ct.c_double, ct.c_double,
            ct.c_int32, ct.c_int32,
            np.ctypeslib.ndpointer(dtype=np.int32)]
        
        self.clib.reducecat.restype = ct.c_void_p
        self.clib.reducecat.argtypes = [
            p_f64, p_f64, p_f64, p_f64, ct.c_int32, ct.c_int32, 
            ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
            p_f64_nof, p_f64_nof, p_f64_nof, p_f64_nof,ct.c_int32]
        
        self.clib.reducecat2.restype = ct.c_void_p
        self.clib.reducecat2.argtypes = [
            p_f64, p_f64, p_f64, p_f64, p_f64, ct.c_int32, 
            ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
            p_f64_nof, p_f64_nof, p_f64_nof, p_f64_nof,p_f64_nof]

        """
        # Allocates Gns and Gammans for discrete data such that one can evaluate all Gamman in [nmin, nmax]
        # Use 'alloc_Gns_discrete' to safely call this function
        self.clib.alloc_GnsGammans_discrete_basic.restype = ct.c_void_p
        self.clib.alloc_GnsGammans_discrete_basic.argtypes = [
            p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32,
            ct.c_int32, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, 
            p_i32, p_i32, p_i32,
            ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, 
            ct.c_int32, 
            np.ctypeslib.ndpointer(dtype=np.double), np.ctypeslib.ndpointer(dtype=np.int32), 
            np.ctypeslib.ndpointer(dtype=np.complex), np.ctypeslib.ndpointer(dtype=np.complex),
            np.ctypeslib.ndpointer(dtype=np.complex)] """
        
        """
        self.clib.alloc_Gammansingle_discretemixed_basic.restype = ct.c_void_p
        self.clib.alloc_Gammansingle_discretemixed_basic.argtypes = [
            p_i32, p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32,
            ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, 
            p_i32, p_i32, p_i32,
            ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, 
            ct.c_double, ct.c_double, 
            ct.c_int32, p_f64, p_f64, p_f64, p_i32, p_i32, p_i32, p_i32, p_i32, p_i32,
            p_f64, p_f64, p_f64, p_c128, ct.c_int32,
            np.ctypeslib.ndpointer(dtype=np.float64),
            np.ctypeslib.ndpointer(dtype=np.int32),
            np.ctypeslib.ndpointer(dtype=np.complex),
            np.ctypeslib.ndpointer(dtype=np.complex), ct.c_int32]
        
        # Allocates Gns for discrete data such that one can evaluate all Gamman in [nmin, nmax]
        # Use 'alloc_Gns_discrete' to safely call this function
        self.clib.alloc_Gns_discrete_basic.restype = ct.c_void_p
        self.clib.alloc_Gns_discrete_basic.argtypes = [
            p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32,
            ct.c_int32, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, 
            p_i32, p_i32, p_i32,
            ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, 
            ct.c_int32, 
            np.ctypeslib.ndpointer(dtype=np.double), np.ctypeslib.ndpointer(dtype=np.int32), 
            np.ctypeslib.ndpointer(dtype=np.complex), np.ctypeslib.ndpointer(dtype=np.complex)]    
        
        # Allocates Gns for discrete data such that one can evaluate all Gamman in [nmin, nmax]
        # Use 'alloc_Gns_discrete' to safely call this function
        self.clib.alloc_Gnsingle_discrete_basic.restype = ct.c_void_p
        self.clib.alloc_Gnsingle_discrete_basic.argtypes = [
            p_f64, p_f64, p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32,
            ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, 
            p_i32, p_i32, p_i32,
            ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, 
            ct.c_int32, 
            np.ctypeslib.ndpointer(dtype=np.double), 
            np.ctypeslib.ndpointer(dtype=np.int32), 
            np.ctypeslib.ndpointer(dtype=np.complex),
            np.ctypeslib.ndpointer(dtype=np.complex)]    
        
        # Allocate Gamman from Gns obtained via the discrete estimator
        # Use 'alloc_Gamman_discrete' to safely call this function    
        self.clib.alloc_Gamman_discrete_basic.restype = ct.c_void_p
        self.clib.alloc_Gamman_discrete_basic.argtypes = [
            p_c128, p_c128, p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32,
            ct.c_int32, ct.c_int32,
            np.ctypeslib.ndpointer(dtype=np.complex),
            np.ctypeslib.ndpointer(dtype=np.complex)]
        
        # Allocate Gamman for G3L via the discrete estimator
        # Use 'alloc_Gamman_G3L' to safely call this function 
        self.clib.alloc_Gammans_discrete_G3L.restype = ct.c_void_p
        self.clib.alloc_Gammans_discrete_G3L.argtypes = [
            p_f64, p_f64, p_f64, p_f64, p_f64, ct.c_int32, 
            p_f64, p_f64, p_f64, ct.c_int32, 
            ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, 
            p_i32, p_i32, p_i32,
            ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, 
            ct.c_int32, 
            np.ctypeslib.ndpointer(dtype=np.double),
            np.ctypeslib.ndpointer(dtype=np.complex),
            np.ctypeslib.ndpointer(dtype=np.complex)]
        
        # Allocate Gamman for G3L via the discrete estimator
        # Use 'alloc_Gamman_G3L' to safely call this function 
        self.clib.alloc_Gammans_discrete_SSL.restype = ct.c_void_p
        self.clib.alloc_Gammans_discrete_SSL.argtypes = [
            p_f64, p_f64, p_f64, p_f64, p_f64, ct.c_int32, 
            p_f64, p_f64, p_f64, ct.c_int32, 
            ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, 
            p_i32, p_i32, p_i32,
            ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32, 
            ct.c_int32, 
            np.ctypeslib.ndpointer(dtype=np.double),
            np.ctypeslib.ndpointer(dtype=np.complex),
            np.ctypeslib.ndpointer(dtype=np.complex)]
    """
        
    # Reduces catalog to smaller catalog where positions & quantities are
    # averaged over regular grid
    def _reduce(self, fields, dpix, tomo=False, normed=False, 
               extent=[None,None,None,None], forcedivide=1, 
               ret_inst=False):
        
        # Initialize grid
        start1, start2, n1, n2 = self._gengridprops(dpix, forcedivide, extent)
        
        # Prepare arguments
        if not tomo:
            zbinarr = np.zeros(self.ngal).astype(np.int32)
        else:
            zbinarr = self.zbins.astype(np.int32)
        nbinsz = len(np.unique(zbinarr))
        ncompfields = []
        scalarquants = []
        nfields = 0
        for field in fields:
            if type(field[0].item()) is float:
                scalarquants.append(field)
                nfields += 1
                ncompfields.append(1)
            if type(field[0].item()) is complex:
                scalarquants.append(field.real)
                scalarquants.append(field.imag)
                nfields += 2
                ncompfields.append(2)
        scalarquants = np.asarray(scalarquants)
        
        # Compute reduction (individually for each zbin)
        w_red = np.zeros(self.ngal, dtype=float)
        pos1_red = np.zeros(self.ngal, dtype=float)
        pos2_red = np.zeros(self.ngal, dtype=float)
        zbins_red = np.zeros(self.ngal, dtype=int)
        scalarquants_red = np.zeros((nfields, self.ngal), dtype=float)
        ind_start = 0
        for elz in range(nbinsz):
            sel_z = zbinarr==elz
            ngal_z = np.sum(sel_z)
            ngal_red_z = 0
            red_shape = (len(fields), ngal_z)
            w_red_z = np.zeros(ngal_z, dtype=np.float64)
            pos1_red_z = np.zeros(ngal_z, dtype=np.float64)
            pos2_red_z = np.zeros(ngal_z, dtype=np.float64)
            scalarquants_red_z = np.zeros(nfields*ngal_z, dtype=np.float64)
            self.clib.reducecat(self.weight[sel_z].astype(np.float64), 
                                self.pos1[sel_z].astype(np.float64), 
                                self.pos2[sel_z].astype(np.float64),
                                scalarquants[:,sel_z].flatten().astype(np.float64),
                                ngal_z, nfields,
                                dpix, dpix, start1, start2, n1, n2,
                                w_red_z, pos1_red_z, pos2_red_z, scalarquants_red_z, ngal_red_z)
            w_red[ind_start:ind_start+ngal_z] = w_red_z
            pos1_red[ind_start:ind_start+ngal_z] = pos1_red_z
            pos2_red[ind_start:ind_start+ngal_z] = pos2_red_z
            zbins_red[ind_start:ind_start+ngal_z] = elz*np.ones(ngal_z, dtype=int)
            scalarquants_red[:,ind_start:ind_start+ngal_z] = scalarquants_red_z.reshape((nfields, ngal_z))
            ind_start += ngal_z
            
        # Accumulate reduced atalog
        sel_nonzero = w_red>0
        w_red = w_red[sel_nonzero]
        pos1_red = pos1_red[sel_nonzero]
        pos2_red = pos2_red[sel_nonzero]
        zbins_red = zbins_red[sel_nonzero]
        scalarquants_red = scalarquants_red[:,sel_nonzero]
        fields_red = []
        tmpcomp = 0
        for elf in range(len(fields)):
            if ncompfields[elf]==1:
                fields_red.append(scalarquants_red[tmpcomp])
            if ncompfields[elf]==2:
                fields_red.append(scalarquants_red[tmpcomp]+1J*scalarquants_red[tmpcomp+1])
            tmpcomp += ncompfields[elf]
            
        if ret_inst:
            return Catalog(pos1=pos1_red, pos2=pos2_red, weight=w_red, zbins=zbins_red), fields_red
            
        return w_red, pos1_red, pos2_red, zbins_red, fields_red
    
    def _multihash(self, dpixs, fields, dpix_hash=None, tomo=False, normed=False, 
                  extent=[None,None,None,None], forcedivide=1):
        """ Builds spatialhash for a base catalog and its reductions. """
        
        dpixs = sorted(dpixs)
        if dpix_hash is None:
            dpix_hash = dpixs[-1]
        if extent[0] is None:
            extent = [self.min1-dpix_hash, self.max1+dpix_hash, self.min2-dpix_hash, self.max2+dpix_hash]
        
        # Initialize spatial hash for discrete catalog
        self.build_spatialhash(dpix=dpix_hash, extent=extent)
        ngals = [self.ngal]
        pos1s = [self.pos1]
        pos2s = [self.pos2]
        weights = [self.weight]
        zbins = [self.zbins*tomo]
        allfields = [fields]
        index_matchers = [self.index_matcher]
        pixs_galind_bounds = [self.pixs_galind_bounds]
        pix_gals = [self.pix_gals]
        # Build spatial hashes for reduced catalogs 
        for dpix in dpixs:
            print(dpix, len(self.pos1))
            nextcat, fields_red = self._reduce(fields=fields,
                                              dpix=dpix, 
                                              tomo=tomo, 
                                              normed=normed, 
                                              extent=extent, 
                                              forcedivide=forcedivide, 
                                              ret_inst=True)
            nextcat.build_spatialhash(dpix=dpix_hash, extent=extent)
            ngals.append(nextcat.ngal)
            pos1s.append(nextcat.pos1)
            pos2s.append(nextcat.pos2)
            weights.append(nextcat.weight)
            zbins.append(nextcat.zbins)
            allfields.append(fields_red)
            index_matchers.append(nextcat.index_matcher)
            pixs_galind_bounds.append(nextcat.pixs_galind_bounds)
            pix_gals.append(nextcat.pix_gals)
            
        return ngals, pos1s, pos2s, weights, zbins, allfields, index_matchers, pixs_galind_bounds, pix_gals
                        
    # Maps catalog to grid
    def togrid(self, fields, dpix, tomo=False, normed=False, 
               extent=[None,None,None,None], method="CIC", forcedivide=1, 
               asgrid=None, nthreads=1, ret_inst=False):
        """ 
        - field is a list with the weights as its last element.
        - Per default builds the grid around min/max of galaxy positions. If
          one wants this can be cast to a fixed larger grid by making use 
          of the 'extent' parameter [min1, max1, min2, max2]
        - forcedivide makes sure that n1 & n2 are divisible by at least some 
          factor. If 'extent' is set this might alter dpix by a bit
        - if one wants to mimic an already existing GriddedCatalog instance
          simply put this instance as 'asgrid'. This overrides all other 
          settings, i.e. only 'fields' needs to be given
        """
        
        if asgrid is not None:
            assert(isinstance(asgrid, GriddedCatalog))
            tomo = asgrid.nbinsz > 1
            extent = [asgrid.pos1start, asgrid.pos1start+asgrid.len1,
                      asgrid.pos2start, asgrid.pos2start+asgrid.len2]
            normed = asgrid.normed
            method = asgrid.method
            asgrid = None
            #print(asgrid.dpix, tomo, asgrid.normed, extent, asgrid.method, 
            #      forcedivide, nthreads)
            #return self.togrid(fields, asgrid.dpix, tomo=tomo, 
            #                   normed=asgrid.normed, extent=extent, 
            #                   method=asgrid.method, forcedivide=forcedivide,
            #                   asgrid=None, nthreads=nthreads)
        
        # Choose index of method for c wrapper
        assert(method in ["NGP", "CIC", "TSC"])
        elmethod = self.assign_methods[method]
        start1, start2, n1, n2 = self._gengridprops(dpix, forcedivide, extent)
        
        # Prepare arguments
        if not tomo:
            zbinarr = np.zeros(self.ngal).astype(np.int32)
        else:
            zbinarr = self.zbins.astype(np.int32)
        nbinsz = len(np.unique(zbinarr))
        nfields = len(fields)-1
        weightarr = fields[-1].astype(np.float64)
        fieldarr = np.zeros(nfields*self.ngal, dtype=np.float64)
        for _ in range(nfields):
            fieldarr[_*self.ngal:(1+_)*self.ngal] = fields[_]
            
        # Call wrapper and reshape output to (zbins, nfields, size_field)
        proj_shape = (nbinsz, len(fields), n2, n1)
        projectedfields = np.zeros((nbinsz*len(fields)*n2*n1), dtype=np.float64)
        self.clib.assign_fields(self.pos1.astype(np.float64), 
                                          self.pos2.astype(np.float64),
                                          zbinarr, weightarr, fieldarr,
                                          nbinsz, nfields, self.ngal,
                                          elmethod, start1, start2, dpix, 
                                          n1, n2, nthreads, projectedfields)
        projectedfields = projectedfields.reshape(proj_shape)
        if normed:
            projectedfields[:,1:] = np.nan_to_num(projectedfields[:,1:]/projectedfields[:,0])
            
        if not ret_inst:
            return projectedfields, start1, start2, dpix, normed, method
        
        return GriddedCatalog(projectedfields, 
                              start1, start2, dpix, normed, method)
    
    def gen_weightgrid2d(self, dpix, 
                         extent=[None,None,None,None], method="CIC", forcedivide=1, 
                         nthreads=1):
        
        # Choose index of method for c wrapper
        assert(method in ["NGP", "CIC", "TSC"])
        elmethod = self.assign_methods[method]
        start1, start2, n1, n2 = self._gengridprops(dpix, forcedivide, extent)
        
        #void gen_weightgrid2d(
        #    double *pos1, double *pos2, int ngal, int method,
        #    double min1, double min2, int dpix, int n1, int n2,
        #    int nthreads, int *pixinds, double *pixweights){
        
        self.ngal
        nsubs = 2*elmethod+1
        pixinds = np.zeros(nsubs*nsubs*self.ngal, dtype=np.int32)
        pixweights = np.zeros(nsubs*nsubs*self.ngal, dtype=np.float64)
        self.clib.gen_weightgrid2d(self.pos1.astype(np.float64), 
                                             self.pos2.astype(np.float64),
                                             self.ngal, elmethod,
                                             start1, start2, dpix, n1, n2,
                                             nthreads, pixinds, pixweights)
        return pixinds, pixweights
        
        
    
    def build_spatialhash(self, dpix=1., extent=[None, None, None, None]):
        
        # Build extent
        if extent[0] is None:
            thismin1 = self.min1
        else:
            thismin1 = extent[0]
            assert(thismin1 <= self.min1)
        if extent[1] is None:
            thismax1 = self.max1
        else:
            thismax1 = extent[1]
            assert(thismax1 >= self.max1)
        if extent[2] is None:
            thismin2 = self.min2
        else:
            thismin2 = extent[2]
            assert(thismin2 <= self.min2)
        if extent[3] is None:
            thismax2 = self.max2
        else:
            thismax2 = extent[3]
            assert(thismax2 >= self.max2)
            
            
        # Collect arguments
        # Note that the C function assumes the mask to start at zero, that's why we shift
        # the galaxy positions
        self.pix1_start = thismin1 - dpix/1.
        self.pix2_start = thismin2 - dpix/1.
        stop1 = thismax1 + dpix/1.
        stop2 = thismax2 + dpix/1.
        self.pix1_n = int(np.ceil((stop1-self.pix1_start)//dpix))
        self.pix2_n = int(np.ceil((stop2-self.pix2_start)//dpix))
        npix = self.pix1_n * self.pix2_n
        self.pix1_d = (stop1-self.pix1_start)/(self.pix1_n)
        self.pix2_d = (stop2-self.pix2_start)/(self.pix2_n)

        # Compute hashtable
        print(np.min(self.pos1), np.min(self.pos2), np.max(self.pos1), np.max(self.pos2))
        print(self.pix1_start, self.pix2_start, 
              self.pix1_start+self.pix1_n*self.pix1_d,
             self.pix2_start+self.pix2_n*self.pix2_d)
        print(self.pix1_n,self.pix2_n,self.pix1_d,self.pix2_d)
        result = np.zeros(2 * npix + 3 * self.ngal + 1).astype(np.int32)
        self.clib.build_spatialhash(self.pos1, self.pos2, self.ngal,
                                  self.pix1_d, self.pix2_d, 
                                  self.pix1_start, self.pix2_start, 
                                  self.pix1_n, self.pix2_n,
                                  result)

        # Allocate result
        start_isoutside = 0
        start_index_matcher = self.ngal
        start_pixs_galind_bounds = self.ngal + npix
        start_pixs_gals = self.ngal + npix + self.ngal + 1
        start_ngalinpix = self.ngal + npix + self.ngal + 1 + self.ngal
        self.index_matcher = result[start_index_matcher:start_pixs_galind_bounds]
        self.pixs_galind_bounds = result[start_pixs_galind_bounds:start_pixs_gals]
        self.pix_gals = result[start_pixs_gals:start_ngalinpix]
        self.hasspatialhash = True
        

    def _gengridprops(self, dpix, forcedivide, extent=[None,None,None,None]):
        
        # Define inner extent of the grid
        fixedsize = False
        if extent[0] is not None:
            fixedsize = True
        if extent[0] is None:
            thismin1 = self.min1
        else:
            thismin1 = extent[0]
            assert(thismin1 <= self.min1)
        if extent[1] is None:
            thismax1 = self.max1
        else:
            thismax1 = extent[1]
            assert(thismax1 >= self.max1)
        if extent[2] is None:
            thismin2 = self.min2
        else:
            thismin2 = extent[2]
            assert(thismin2 <= self.min2)
        if extent[3] is None:
            thismax2 = self.max2
        else:
            thismax2 = extent[3]
            assert(thismax2 >= self.max2)
            
        # Add buffer to grid and get associated pixelization
        if not fixedsize:
            start1 = thismin1 - 4*dpix
            start2 = thismin2 - 4*dpix
            n1 = int(np.ceil((thismax1+4*dpix - start1)/dpix))
            n2 = int(np.ceil((thismax2+4*dpix - start2)/dpix))
            n1 += (forcedivide - n1%forcedivide)%forcedivide
            n2 += (forcedivide - n2%forcedivide)%forcedivide
        else:
            start1=extent[0]
            start2=extent[2]
            n1 = int((thismax1-thismin1)/dpix)
            n2 = int((thismax2-thismin2)/dpix)
            assert(not n1%forcedivide)
            assert(not n2%forcedivide)
            
        return start1, start2, n1, n2
    
class ScalarTracerCatalog(Catalog):
    
    def __init__(self, pos1, pos2, tracer, **kwargs):
        super().__init__(pos1=pos1, pos2=pos2, **kwargs)
        self.tracer = tracer
        self.spin = 0
        
    def reduce(self, dpix, tomo=False, normed=False, 
               extent=[None,None,None,None], forcedivide=1, 
               ret_inst=False):
        res = super()._reduce(dpix=dpix, 
                             fields=[self.tracer], 
                             tomo=tomo, 
                             normed=normed, 
                             extent=extent,
                             forcedivide=forcedivide,
                            ret_inst=False)
        (w_red, pos1_red, pos2_red, zbins_red, fields_red) = res
        if ret_inst:
            return ScalarTracerCatalog(self.spin, pos1_red, pos2_red, 
                                       fields_red[0], 
                                       weight=w_red, zbins=zbins_red)
        return res
    
    def multireduce(self, dpixs, tomo=False, normed=False, 
                    extent=[None,None,None,None], forcedivide=1):
        pass
        
class SpinTracerCatalog(Catalog):
    
    def __init__(self, spin, pos1, pos2, tracer_1, tracer_2, **kwargs):
        super().__init__(pos1=pos1, pos2=pos2, **kwargs)
        self.tracer_1 = tracer_1.astype(float)
        self.tracer_2 = tracer_2.astype(float)
        self.spin = int(spin)
        
    def reduce(self, dpix, tomo=False, normed=False, 
               extent=[None,None,None,None], forcedivide=1, 
               ret_inst=False):
        res = super()._reduce(dpix=dpix, 
                             fields=[self.tracer_1, self.tracer_2], 
                             tomo=tomo, 
                             normed=normed, 
                             extent=extent,
                             forcedivide=forcedivide,
                             ret_inst=False)
        (w_red, pos1_red, pos2_red, zbins_red, fields_red) = res
        if ret_inst:
            return SpinTracerCatalog(spin=self.spin, pos1=pos1_red, pos2=pos2_red, 
                                     tracer_1=fields_red[0], tracer_2=fields_red[1], 
                                     weight=w_red, zbins=zbins_red)
        return res
    
    def multihash(self, dpixs, dpix_hash=None, tomo=False, normed=False, 
                  extent=[None,None,None,None], forcedivide=1):
        res = super()._multihash(dpixs=dpixs, 
                                fields=[self.tracer_1, self.tracer_2], 
                                dpix_hash=dpix_hash,
                                tomo=tomo, 
                                normed=normed, 
                                extent=extent,
                                forcedivide=forcedivide)
        return res
        
        
        
class JointTracerCatalog(Catalog):
    """ 
    In this class we assume that the ra/dec is always the same!
    Tracers is given as dict as follows
    {"<Name_1>": {"s":<spin>, "data":<array of float of complex>}}
    """
    
    def __init__(self, pos1, pos2, tracers, **kwargs):
        super().__init__(pos1=pos1, pos2=pos2, **kwargs)
        self.tracers = tracer.astype(complex)
        
class MultiTracerCatalog:
    """ All arguments are given as lists.
    Tracers are given as individual lists. I.e. scalar+polar catalog would have
    tracers = [[scalar],[polar_1, polar_2]]
    """
    def __init__(self, pos1, pos2, tracers, spins, **kwargs):
        self.ntracers = len(pos1)
        assert(len(pos1)==len(pos2))
        assert(len(pos1)==len(tracers))
        assert(len(pos1)==len(spins))
        for key in kwargs.keys():
            if kwargs[key] is None:
                kwargs[key] = [None]*self.ntracers
            else:
                assert(len(kwargs[key][0])==len(pos1))
        self.tracercats = []
        for eltracer in range(self.ntracers):
            thiskwargs = {}
            for key in kwargs.keys():
                thiskwargs[key] = kwargs[key][eltracer]
            if len(tracers[eltracer])==1:
                self.tracercats.append(ScalarTracerCatalog(pos1=pos1[eltracer], 
                                                           pos2=pos2[eltracer], 
                                                           tracer=tracer[eltracer][0],
                                                           **thiskwargs))
            elif len(tracers[eltracer])==2:
                self.tracercats.append(SpinTracerCatalog(pos1=pos1[eltracer], 
                                                         pos2=pos2[eltracer], 
                                                         tracer_1=tracer[eltracer][0],
                                                         tracer_2=tracer[eltracer][1],
                                                         **thiskwargs))
                
    def reduce(self, dpix, tomo=False, normed=False, 
               extent=[None,None,None,None], forcedivide=1, 
               ret_inst=False):
        
        allpos1_red = []
        allpos2_red = []
        alltracers_red = []
        allweights_red = []
        allzbins_red = []
        
        for eltracer in range(self.ntracers):
            res = self.tracercats[eltracer].reduce(dpix, 
                                                   tomo=tomo,
                                                   normed=normed, 
                                                   extent=extent, 
                                                   forcedivide=forcedivide,
                                                   ret_inst=False)
            (w_red, pos1_red, pos2_red, zbins_red, fields_red) = res
            allpos1_red.append(pos1_red)
            allpos2_red.append(pos2_red)
            alltracers_red.append(fields_red)
            allweights_red.append(w_red)
            allzbins_red.append(zbins_red)
        
        if ret_inst:
            return MultiTracerCatalog(pos1=allpos1_red,
                                      pos2=allpos2_red, 
                                      tracers=alltracers_red,
                                      spins=self.spins,
                                      weight=allweights_red, 
                                      zbins=allzbins_red)
        return allweights_red, allpos1_red, allpos2_red, allzbins_red, alltracers_red