from abc import ABC, abstractmethod

import ctypes as ct
import numpy as np 
from numpy.ctypeslib import ndpointer
from pathlib import Path
import sys
from .catalog import Catalog, ScalarTracerCatalog
from .utils import flatlist, gen_thetacombis_fourthorder, gen_n2n3indices_Upsfourth


class GGCovarianceNoTomo:
    """ Computes covariance of second-order shear correlation functions
    """
    def __init__(self,   
                 bins_xipm,
                 method_pairs='analytic', method_triplets='analytic', method_quadruplets='analytic',
                 spline_xip=None, spline_xim=None,
                 nbar=None, area=None, weights_mean=1., weights_std=0, spl_supression_ww=None,
                 pairs=None, triplets=None, quadruplets=None,
                 ):
        
        ## Pass arguments from init ##
        self.bins_xipm=bins_xipm

        self.method_pairs = method_pairs
        self.method_triplets = method_triplets
        self.method_quadruplets = method_quadruplets
        self.xip = spline_xip
        self.xim = spline_xim

        self.nbar=None 
        self.area=None
        self.weights_mean = weights_mean
        self.weights_std = weights_std
        self.spl_supression_ww = spl_supression_ww

        self.pairs = pairs
        self.triplets = triplets
        self.quadruplets = quadruplets

        # Init different contributions to covariance (cf SKL 2002)
        self.D = None
        self.qpp = None
        self.qmm = None
        self.qpm = None
        self.rp0 = None
        self.rp1 = None
        self.rm0 = None
        self.rm1 = None
        self.rpm = None
        self.cov_pp = None
        self.cov_mm = None
        self.cov_pm = None

        ## PREPARE ALL DATA RELATED TO MULTIPLET COUNTS ##
        # If required, load the multiplet counts from file or compute
        if self.method_pairs=='discrete' and type(self.pairs) is str:
            self.pairs = self.__loadpairs(self.pairs)
        elif self.method_pairs=='discrete' and self.pairs is None:
            self.pairs = self.computeDiscrete('pairs')
        elif self.method_pairs=='analytic':
            self.pairs = self.pairs_analytic 
        else:
            raise ValueError('Incorrect parameters for pair contribution')
        if self.method_triplets=='discrete' and type(self.triplets) is str:
            self.triplets = self.__loadtriplets(self.triplets)
        elif self.method_triplets=='discrete' and self.triplets is None:
            self.triplets = self.computeDiscrete('triplets')
        elif self.method_triplets=='analytic':
            self.triplets = self.triplets_analytic 
        else:
            raise ValueError('Incorrect parameters for triplets contribution')
        if self.method_quadruplets=='discrete' and type(self.quadruplets) is str:
            self.triplequadrupletsts = self.__loadquadruplets(self.quadruplets)
        elif self.method_quadruplets=='discrete' and self.quadruplets is None:
            self.quadruplets = self.computeDiscrete('quadruplets')
        elif self.method_quadruplets=='analytic':
            self.quadruplets = self.quadruplets_analytic 
        else:
            raise ValueError('Incorrect parameters for quadruplets contribution')
        
    ## COMPUTATION OF INDIVIDUAL TERMS ##
    # TODO (MARTINA): IMPLEMENT THIS
    def getD(self):
        if self.method_pairs=='analytic':
            # self.D = ...
            raise NotImplementedError
        if self.method_pairs=='discrete':
            # self.D = ...
            raise NotImplementedError
        
    def getNpairs(self):
        raise NotImplementedError
    
    def getqpp(self):
        raise NotImplementedError

    ## HELPERS ##
    #  TODO (LUCAS): Link this to orpheus multiplet computatoin
    def computeDiscrete(self, contribution):
        if contribution=='pairs':
            raise NotImplementedError
        if contribution=='triplets':
            raise NotImplementedError
        if contribution=='quadruplets':
            raise NotImplementedError
    
    def get_pairs(self):
        ''' Returns all paircounts '''
        if self.method_pairs=='analytic':
            return self.analyticpairs()
        else:
            raise NotImplementedError
    
    def get_triplets(self):
        ''' Returns all the triplets '''
        if self.method_triplets=='analytic':
            return self.analytictriplets()
        else:
            raise NotImplementedError

    # TODO (MARTINA): Make sure the selection is consistent with self.analyticquadruplets
    def get_quadruplets(self):
        ''' Returns selection of quadruplets  '''
        if self.method_quadruplets=='analytic':
            return self.analyticquadruplets()
        else:
            raise NotImplementedError
        
    # TODO (MARTINA): Add your implementation-- output the full tensor
    def analyticpairs(self):
        raise NotImplementedError
    
    # TODO (MARTINA): Add your implementation -- output the full tensor
    def analytictriplets(self):
        raise NotImplementedError
    
    # TODO (MARTINA): Add your implementation-- output part of tensor that fits most to your parametrisation
    def analyticquadruplets(self):
        raise NotImplementedError
    
    def __loadpairs(self, fname):
        # TODO (LUCAS): Check consistency with class parameters
        raise NotImplementedError
    
    def __loadtriplets(self, fname):
        # TODO (LUCAS): Check consistency with class parameters
        raise NotImplementedError
    
    def __loadquadruplets(self, fname):
        # TODO (LUCAS): Check consistency with class parameters
        raise NotImplementedError