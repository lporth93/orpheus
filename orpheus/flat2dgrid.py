import numpy as np
from scipy.interpolate import RegularGridInterpolator

__all__ = ["FlatPixelGrid_2D","FlatDataGrid_2D"]

class FlatPixelGrid_2D(object):
    
    def __init__(self, start_1, start_2, npix_1, npix_2, dpix_1, dpix_2):
        self.start_1 = start_1
        self.start_2 = start_2
        self.npix_1 = npix_1
        self.npix_2 = npix_2
        self.dpix_1 = dpix_1
        self.dpix_2 = dpix_2
        self.pix1_lbounds = self.start_1 + self.dpix_1*np.arange(self.npix_1)
        self.pix2_lbounds = self.start_2 + self.dpix_2*np.arange(self.npix_2)
        self.pix1_centers = self.pix1_lbounds + self.dpix_1/2.
        self.pix2_centers = self.pix2_lbounds + self.dpix_2/2.
        
    def todatagrid(self, data):
        return FlatDataGrid_2D(data, self.start_1, self.start_2, self.dpix_1, self.dpix_2)
        
    def _regrid(self, other, data):
        assert(isinstance(other,FlatPixelGrid_2D)) 
        assert(data.shape == (self.npix_2,self.npix_1))
        data_int = RegularGridInterpolator((self.pix2_centers, self.pix1_centers), data,
                                           method="linear", bounds_error=False, fill_value=0)
        oc1, oc2 = np.meshgrid(other.pix2_centers,other.pix1_centers, indexing='ij')
        centers_mapgrid = np.array([oc1,oc2]).reshape((2,oc1.shape[1]*oc1.shape[0])).transpose()
        data_regridded = data_int(centers_mapgrid).reshape((oc1.shape[0],oc1.shape[1]))
        return other.todatagrid(data_regridded)
                
        
class FlatDataGrid_2D(FlatPixelGrid_2D):
    
    def __init__(self, data, start_1, start_2, dpix_1, dpix_2):
        """
        Convention: O (unmasked) --> 1 (fully masked)
        """
        super().__init__(start_1, start_2, data.shape[1], data.shape[0], dpix_1, dpix_2)
        self.data = data
        
    def regrid(self, other_grid):
        return super()._regrid(other_grid, self.data)