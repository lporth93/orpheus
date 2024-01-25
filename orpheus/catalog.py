import ctypes as ct
import numpy as np 
from numpy.ctypeslib import ndpointer
from pathlib import Path
import glob
from .utils import get_site_packages_dir, search_file_in_site_package
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
        self.zbins = self.zbins.astype(np.int32)
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
        # Method that works for LP
        target_path = __import__('orpheus').__file__
        self.library_path = str(Path(__import__('orpheus').__file__).parent.parent.absolute())
        self.clib = ct.CDLL(glob.glob(self.library_path+"/orpheus_clib*.so")[0])
        # Method that works for RR (but not for LP with a local HPC install)
        #self.clib = ct.CDLL(search_file_in_site_package(get_site_packages_dir(),"orpheus_clib"))
        #self.library_path = str(Path(__import__('orpheus').__file__).parent.parent.absolute())
        #print(self.library_path)
        #self.clib = ct.CDLL(glob.glob(self.library_path+"/orpheus_clib*.so")[0])
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
            ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32,
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
            np.ctypeslib.ndpointer(dtype=complex), np.ctypeslib.ndpointer(dtype=complex),
            np.ctypeslib.ndpointer(dtype=complex)] """
        
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
            np.ctypeslib.ndpointer(dtype=complex),
            np.ctypeslib.ndpointer(dtype=complex), ct.c_int32]
        
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
            np.ctypeslib.ndpointer(dtype=complex), np.ctypeslib.ndpointer(dtype=complex)]    
        
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
            np.ctypeslib.ndpointer(dtype=complex),
            np.ctypeslib.ndpointer(dtype=complex)]    
        
        # Allocate Gamman from Gns obtained via the discrete estimator
        # Use 'alloc_Gamman_discrete' to safely call this function    
        self.clib.alloc_Gamman_discrete_basic.restype = ct.c_void_p
        self.clib.alloc_Gamman_discrete_basic.argtypes = [
            p_c128, p_c128, p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32,
            ct.c_int32, ct.c_int32,
            np.ctypeslib.ndpointer(dtype=complex),
            np.ctypeslib.ndpointer(dtype=complex)]
        
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
            np.ctypeslib.ndpointer(dtype=complex),
            np.ctypeslib.ndpointer(dtype=complex)]
        
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
            np.ctypeslib.ndpointer(dtype=complex),
            np.ctypeslib.ndpointer(dtype=complex)]
    """
        
    # Reduces catalog to smaller catalog where positions & quantities are
    # averaged over regular grid
    def _reduce(self, fields, dpix, dpix2=None, relative_to_hash=None, normed=False, shuffle=False,
               extent=[None,None,None,None], forcedivide=1, 
               ret_inst=False):
        
        # Initialize grid
        if relative_to_hash is None: 
            if dpix2 is None:
                dpix2 = dpix
            start1, start2, n1, n2 = self._gengridprops(dpix, dpix2, forcedivide, extent)
        else:
            assert(self.hasspatialhash)
            assert(isinstance(relative_to_hash,np.int32))
            start1 = self.pix1_start
            start2 = self.pix2_start
            dpix = self.pix1_d/np.float64(relative_to_hash)
            dpix2 = self.pix2_d/np.float64(relative_to_hash)
            n1 = self.pix1_n*relative_to_hash
            n2 = self.pix2_n*relative_to_hash
        
        # Prepare arguments
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
        w_red = np.zeros(self.ngal, dtype=np.float64)
        pos1_red = np.zeros(self.ngal, dtype=np.float64)
        pos2_red = np.zeros(self.ngal, dtype=np.float64)
        zbins_red = np.zeros(self.ngal, dtype=np.int32)
        scalarquants_red = np.zeros((nfields, self.ngal), dtype=np.float64)
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
                                dpix, dpix2, start1, start2, n1, n2, np.int32(shuffle),
                                w_red_z, pos1_red_z, pos2_red_z, scalarquants_red_z, ngal_red_z)
            w_red[ind_start:ind_start+ngal_z] = w_red_z
            pos1_red[ind_start:ind_start+ngal_z] = pos1_red_z
            pos2_red[ind_start:ind_start+ngal_z] = pos2_red_z
            zbins_red[ind_start:ind_start+ngal_z] = elz*np.ones(ngal_z, dtype=np.int32)
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
    
    def _multihash(self, dpixs, fields, dpix_hash=None, normed=False, shuffle=False,
                  extent=[None,None,None,None], forcedivide=1):
        """ Builds spatialhash for a base catalog and its reductions. """
        
        dpixs = sorted(dpixs)
        if dpix_hash is None:
            dpix_hash = dpixs[-1]
        if extent[0] is None:
            extent = [self.min1-dpix_hash, self.max1+dpix_hash, self.min2-dpix_hash, self.max2+dpix_hash]
            #extent = [self.min1, self.max1+dpix_hash-self.len1%dpix_hash, 
            #          self.min2, self.max2+dpix_hash-self.len2%dpix_hash]
        
        # Initialize spatial hash for discrete catalog
        print("First spatialhash")
        self.build_spatialhash(dpix=dpix_hash, extent=extent)
        ngals = [self.ngal]
        pos1s = [self.pos1]
        pos2s = [self.pos2]
        weights = [self.weight]
        zbins = [self.zbins]
        allfields = [fields]
        index_matchers = [self.index_matcher]
        pixs_galind_bounds = [self.pixs_galind_bounds]
        pix_gals = [self.pix_gals]

        # Build spatial hashes for reduced catalogs 
        fac_pix1 = self.pix1_d/dpix_hash
        fac_pix2 = self.pix2_d/dpix_hash
        dpixs1_true = np.zeros_like(np.asarray(dpixs))
        dpixs2_true = np.zeros_like(np.asarray(dpixs))
        for elreso in range(len(dpixs)):
            print("Doing reso %i"%elreso)
            dpixs1_true[elreso]=fac_pix1*dpixs[elreso]
            dpixs2_true[elreso]=fac_pix2*dpixs[elreso]
            #print(dpixs[elreso], dpixs1_true[elreso], dpixs2_true[elreso], len(self.pos1))
            nextcat, fields_red = self._reduce(fields=fields,
                                               dpix=dpixs1_true[elreso], 
                                               dpix2=dpixs2_true[elreso],
                                               relative_to_hash=np.int32(2**(len(dpixs)-elreso-1)),
                                               #relative_to_hash=None,
                                               normed=normed, 
                                               shuffle=shuffle,
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
            print("Done reso %i"%elreso)
            
        return ngals, pos1s, pos2s, weights, zbins, allfields, index_matchers, pixs_galind_bounds, pix_gals, dpixs1_true, dpixs2_true
    
    def _genmatched_multiresocats(self, dpixs, fields, flattened=True, 
                                  normed=False, extent=[None,None,None,None], forcedivide=1):
        """
        Shapes of the unflattened arrays/lists:
        - ngals/resos1/resos2: (nresos,)
        - ngalshifts_1d: (nresos+1,)
        - ngalshifts_3d: (nbinsz, nresos-1, nresos)
        - pos1s/pos1s/weights/zbins: (ngals_reso1 + ngal_reso2 + ... + ngal_nresos)==(cumgals,)
        - allfields: (nfields, cumgals)
        - index_matchers: (nresos, npix_spatialhash)
        - pixs_galind_bounds/pix_gals:  (cumgals,)
        - pixmatcher: (nresos*ngal_reso1 + (nresos-1)*ngal_reso2 + ... + 1*ngal_{nresos-1}, )
        """
        _ = self.__genmatched_multiresocats(dpixs, fields, change_renormsign=False, 
                                            normed=normed,
                                            extent=extent, forcedivide=forcedivide)
        failed, *res = _
        if failed:
            print("Pixelmapper creation failed first try....shuffling grid")
            _ = self.__genmatched_multiresocats(dpixs, fields, change_renormsign=True,
                                                normed=normed,
                                                extent=extent, forcedivide=forcedivide)
            failed, *res = _
            print(failed)
        if not flattened:
            return res
        else:
            ngals = res[0]
            nresos = len(ngals)
            ngals_zreso = np.zeros((self.nbinsz,nresos), dtype=np.int32)
            for elbinz in range(self.nbinsz):
                for elreso in range(nresos):
                    ngals_zreso[elbinz][elreso] = np.sum(res[4][elreso]==elbinz)
            ngalshifts_1d = np.append(np.array([0]), np.cumsum(ngals)).astype(np.int32)
            pos1s = np.hstack(res[1]).squeeze()
            pos2s = np.hstack(res[2]).squeeze()
            weights = np.hstack(res[3]).squeeze()
            zbins = np.hstack(res[4]).squeeze()
            allfields = [None]*len(res[5][0])
            for elf in range(len(res[5][0])):
                allfields[elf] = np.zeros(ngalshifts_1d[-1], dtype=type(res[5][0][elf][0]))
                for elreso in range(nresos):
                    #print(elf,elreso,len(res[elreso][elf]),ngalshifts_1d[elreso]-ngalshifts_1d[elreso+1])
                    allfields[elf][ngalshifts_1d[elreso]:ngalshifts_1d[elreso+1]] = res[5][elreso][elf]
            index_matchers = np.hstack(res[6]).squeeze() # length: nresos*self.pix1_n*self.pix2_n
            pixs_galind_bounds = np.hstack(res[7]).squeeze()
            pix_gals = np.hstack(res[8]).squeeze()
            resos1 = res[9]
            resos2 = res[10]
            _tmpshift = 0
            ngalshifts_3d = -1*np.ones((self.nbinsz, nresos-1, nresos), dtype=np.int32)
            for elbinz in range(self.nbinsz):
                for elreso in range(nresos-1):
                    ngalshifts_3d[elbinz][elreso][elreso] = _tmpshift
                    for elreso2 in range(elreso+1,nresos):
                        _tmpshift += ngals_zreso[elbinz][elreso]
                        ngalshifts_3d[elbinz][elreso][elreso2] = _tmpshift
            pixmatcher = np.zeros(ngalshifts_3d[-1][-1][-1])
            for elbinz in range(self.nbinsz):
                for elreso in range(nresos-1):
                    for elreso2 in range(elreso+1,nresos):
                        _start = ngalshifts_3d[elbinz][elreso][elreso2-1]
                        _toappend = res[11][elbinz][elreso][elreso2-elreso-1]
                        pixmatcher[_start:_start+len(_toappend)] = _toappend
                        #print(elreso,elreso2,len(res[11]),len(res[11][0]),_start,len(_toappend))
        return np.array(ngals, dtype=np.int32), resos1.astype(np.float64), resos2.astype(np.float64), \
               ngalshifts_1d, ngalshifts_3d.flatten(), \
               pos1s, pos2s, weights, zbins, allfields, \
               index_matchers, pixs_galind_bounds , pix_gals, pixmatcher

    def __genmatched_multiresocats(self, dpixs, fields, change_renormsign=False,
                                   normed=False, extent=[None,None,None,None], forcedivide=1):

        # Build multihashsh
        _min1 = self.min1-2.*dpixs[-1]
        _max1 = self.max1+2.*dpixs[-1]
        _min2 = self.min2-2.*dpixs[-1]
        _max2 = self.max2+2.*dpixs[-1]
        _renorm = (-1)**change_renormsign*dpixs[-1]*1e-5
        extent = [_min1, _max1-(_max1-_min1)%dpixs[-1]-_renorm, 
                  _min2, _max2-(_max2-_min2)%dpixs[-1]-_renorm]
        _ = self._multihash(dpixs=dpixs, fields=fields, extent=extent, shuffle=False, 
                            normed=normed, forcedivide=forcedivide)
        ngals, pos1s, pos2s, weights, zbins, allfields, index_matchers, pixs_galind_bounds, pix_gals, dpixs1_true, dpixs2_true = _ 

        # Match reduced pixelgrids between catalogs
        resos1 = np.append([0], dpixs1_true)
        resos2 = np.append([0], dpixs2_true)
        #print(resos1,resos2)
        ind_targetmatcher = [None]*(self.nbinsz)
        for elbinz in range(self.nbinsz):
            ind_targetmatcher[elbinz] = [None]*(len(resos1)-1)
            for elreso in range(1,len(resos1)):
                if elbinz==0:
                    ipixfull_1 = np.floor((pos1s[elreso] - self.pix1_start)/resos1[elreso]).astype(np.int32)
                    ipixfull_2 = np.floor((pos2s[elreso] - self.pix2_start)/resos2[elreso]).astype(np.int32)
                    indfull_targets = (ipixfull_1*thisn1 + ipixfull_2).astype(np.int32)
                sel_z = np.argwhere(zbins[elreso]==elbinz).flatten()
                thisn1 = int(self.pix1_n*self.pix1_d/resos1[elreso])
                thisn2 = int(self.pix2_n*self.pix2_d/resos2[elreso])
                ipix_1 = np.floor((pos1s[elreso][sel_z] - self.pix1_start)/resos1[elreso]).astype(np.int32)
                ipix_2 = np.floor((pos2s[elreso][sel_z] - self.pix2_start)/resos2[elreso]).astype(np.int32)
                ind_targets = (ipix_2*thisn1 + ipix_1).astype(np.int32)
                ind_targetmatcher[elbinz][elreso-1] = np.zeros(thisn1*thisn2, dtype=np.int32)
                ind_targetmatcher[elbinz][elreso-1][ind_targets] = sel_z.astype(np.int32)
                #print(elreso, len(ind_targetmatcher[elreso-1]), np.max(ind_targets[elreso-1]), np.max(ind_targetmatcher[elreso-1]))

        # Build pixelmatcher
        pixmatcher = [None]*(self.nbinsz)
        for elbinz in range(self.nbinsz):
            pixmatcher[elbinz] = [None]*(len(resos1)-1)
            for elreso in range(len(resos1)-1):
                pixmatcher[elbinz][elreso] = [None]*(len(resos1)-elreso-1)
                sel_z = np.argwhere(zbins[elreso]==elbinz).flatten()
                for elreso2 in range(elreso+1, len(resos1)):
                    ipix_1 = np.floor((pos1s[elreso][sel_z] - self.pix1_start)/resos1[elreso2]).astype(np.int32)
                    ipix_2 = np.floor((pos2s[elreso][sel_z] - self.pix2_start)/resos2[elreso2]).astype(np.int32)
                    indreso = (ipix_2*(self.pix1_n*self.pix1_d/resos1[elreso2]) + ipix_1).astype(np.int32)
                    pixmatcher[elbinz][elreso][elreso2-(elreso+1)] = ind_targetmatcher[elbinz][elreso2-1][indreso]
                    #print(elbinz,elreso,elreso2,np.min(pixmatcher[elbinz][elreso][elreso2-(elreso+1)]))
                    pixmatcher[elbinz][elreso][elreso2-(elreso+1)] -= np.min(pixmatcher[elbinz][elreso][elreso2-(elreso+1)])
        failed = pixmatcher[0][-1][0][0]!=0

        return failed, ngals, pos1s, pos2s, weights, zbins, allfields, \
               index_matchers, pixs_galind_bounds, pix_gals, resos1, resos2, pixmatcher


    # Maps catalog to grid
    def togrid(self, fields, dpix, normed=False, 
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
        start1, start2, n1, n2 = self._gengridprops(dpix, dpix, forcedivide, extent)
        
        # Prepare arguments
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
        start1, start2, n1, n2 = self._gengridprops(dpix, dpix, forcedivide, extent)
        
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
        self.pix1_n = int(np.ceil((stop1-self.pix1_start)/dpix))
        self.pix2_n = int(np.ceil((stop2-self.pix2_start)/dpix))
        npix = self.pix1_n * self.pix2_n
        self.pix1_d = (stop1-self.pix1_start)/(self.pix1_n)
        self.pix2_d = (stop2-self.pix2_start)/(self.pix2_n)

        # Compute hashtable
        #print(np.min(self.pos1), np.min(self.pos2), np.max(self.pos1), np.max(self.pos2))
        #print(self.pix1_start, self.pix2_start, 
        #      self.pix1_start+self.pix1_n*self.pix1_d,
        #     self.pix2_start+self.pix2_n*self.pix2_d)
        #print(self.pix1_n,self.pix2_n,self.pix1_d,self.pix2_d)
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
        

    def _gengridprops(self, dpix, dpix2=None, forcedivide=1, extent=[None,None,None,None]):
        
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

        if dpix2 is None:
            dpix2 = dpix
            
        # Add buffer to grid and get associated pixelization
        if not fixedsize:
            start1 = thismin1 - 4*dpix
            start2 = thismin2 - 4*dpix2
            n1 = int(np.ceil((thismax1+4*dpix - start1)/dpix))
            n2 = int(np.ceil((thismax2+4*dpix2 - start2)/dpix2))
            n1 += (forcedivide - n1%forcedivide)%forcedivide
            n2 += (forcedivide - n2%forcedivide)%forcedivide
        else:
            start1=extent[0]
            start2=extent[2]
            n1 = int((thismax1-thismin1)/dpix)
            n2 = int((thismax2-thismin2)/dpix2)
            assert(not n1%forcedivide)
            assert(not n2%forcedivide)
            
        return start1, start2, n1, n2
    
class ScalarTracerCatalog(Catalog):
    
    def __init__(self, pos1, pos2, tracer, **kwargs):
        super().__init__(pos1=pos1, pos2=pos2, **kwargs)
        self.tracer = tracer
        self.spin = 0
        
    def reduce(self, dpix, dpix2=None, relative_to_hash=None, normed=False, shuffle=False,
               extent=[None,None,None,None], forcedivide=1, 
               ret_inst=False):
        res = super()._reduce(
            dpix=dpix,
            dpix2=None, 
            relative_to_hash=None, 
            fields=[self.tracer], 
            normed=normed, 
            shuffle=shuffle,
            extent=extent,
            forcedivide=forcedivide,
            ret_inst=False)
        (w_red, pos1_red, pos2_red, zbins_red, fields_red) = res
        if ret_inst:
            return ScalarTracerCatalog(self.spin, pos1_red, pos2_red, 
                                       fields_red[0], 
                                       weight=w_red, zbins=zbins_red)
        return res
    
    def multihash(self, dpixs, dpix_hash=None, normed=False, shuffle=False,
                  extent=[None,None,None,None], forcedivide=1):
        res = super()._multihash(
            dpixs=dpixs.astype(np.float64), 
            fields=[self.tracer], 
            dpix_hash=dpix_hash,
            normed=normed, 
            shuffle=shuffle,
            extent=extent,
            forcedivide=forcedivide)
        return res
    
    def genmatched_multiresocats(self, dpixs, flattened=True, 
                                 normed=False, extent=[None,None,None,None], forcedivide=1):
        res = super()._multihash(
            dpixs=dpixs, 
            fields=[self.tracer], 
            flattened=flattened,
            normed=normed, 
            extent=extent,
            forcedivide=forcedivide)
        return res
        
class SpinTracerCatalog(Catalog):
    
    def __init__(self, spin, pos1, pos2, tracer_1, tracer_2, **kwargs):
        super().__init__(pos1=pos1, pos2=pos2, **kwargs)
        self.tracer_1 = tracer_1.astype(np.float64)
        self.tracer_2 = tracer_2.astype(np.float64)
        self.spin = int(spin)
        
    def reduce(self, dpix, dpix2=None, relative_to_hash=None, normed=False, shuffle=False,
               extent=[None,None,None,None], forcedivide=1, 
               ret_inst=False):
        res = super()._reduce(
            dpix=dpix, 
            dpix2=None, 
            relative_to_hash=None, 
            fields=[self.tracer_1, self.tracer_2], 
            normed=normed,
            shuffle=shuffle,
            extent=extent,
            forcedivide=forcedivide,
            ret_inst=False)
        (w_red, pos1_red, pos2_red, zbins_red, fields_red) = res
        if ret_inst:
            return SpinTracerCatalog(spin=self.spin, pos1=pos1_red, pos2=pos2_red, 
                                     tracer_1=fields_red[0], tracer_2=fields_red[1], 
                                     weight=w_red, zbins=zbins_red)
        return res
    
    def multihash(self, dpixs, dpix_hash=None, normed=False, shuffle=False,
                  extent=[None,None,None,None], forcedivide=1):
        res = super()._multihash(
            dpixs=dpixs, 
            fields=[self.tracer_1, self.tracer_2], 
            dpix_hash=dpix_hash,
            normed=normed, 
            shuffle=shuffle,
            extent=extent,
            forcedivide=forcedivide)
        return res
    
    def genmatched_multiresocats(self, dpixs, flattened=True,
                                 normed=False, extent=[None,None,None,None], forcedivide=1):
        res = super()._genmatched_multiresocats(
            dpixs=dpixs, 
            fields=[self.tracer_1, self.tracer_2], 
            flattened=flattened,
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
        self.tracers = tracers.astype(complex)
        
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
                                                           tracer=tracers[eltracer][0],
                                                           **thiskwargs))
            elif len(tracers[eltracer])==2:
                self.tracercats.append(SpinTracerCatalog(pos1=pos1[eltracer], 
                                                         pos2=pos2[eltracer], 
                                                         tracer_1=tracers[eltracer][0],
                                                         tracer_2=tracers[eltracer][1],
                                                         **thiskwargs))
                
    def reduce(self, dpix, normed=False, 
               extent=[None,None,None,None], forcedivide=1, 
               ret_inst=False):
        
        allpos1_red = []
        allpos2_red = []
        alltracers_red = []
        allweights_red = []
        allzbins_red = []
        
        for eltracer in range(self.ntracers):
            res = self.tracercats[eltracer].reduce(dpix, 
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