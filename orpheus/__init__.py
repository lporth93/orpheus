from ._version import __version__

from .catalog import Catalog, ScalarTracerCatalog, SpinTracerCatalog
from .npcf_base import BinnedNPCF
from .npcf_second import GGCorrelation, NGCorrelation, NNCorrelation
from .npcf_third import GGGCorrelation, GNNCorrelation, NGGCorrelation, NNNCorrelation
from .npcf_fourth import (GGGGCorrelation_NoTomo, GNNNCorrelation_NoTomo,
                          NNNNCorrelation_NoTomo)
from .direct import (DirectEstimator, Direct_Map3Unequal, Direct_MapnEqual,
                     Direct_NapnEqual, MapCombinatorics)
from .flat2dgrid import FlatDataGrid_2D, FlatPixelGrid_2D
from .patchutils import (cat2hpx, frompatchindices_preparerot, gen_cat_patchindices,
                         pickle_load, pickle_save, toorigin)
from .utils import (check_clib_error, convertunits, flatlist,
                    gen_n2n3indices_Gtildefourth, gen_n2n3indices_Upsfourth,
                    gen_thetacombis_fourthorder, get_site_packages_dir, map_ztuples,
                    search_file_in_site_package, symmetrize_map3_multiscale)

# TODO: Put in once public
try:
    from .covariance import *
except ModuleNotFoundError:
    pass

__all__ = [
    "__version__",
    # Catalogs
    "Catalog", "ScalarTracerCatalog", "SpinTracerCatalog",
    # Correlation functions
    "BinnedNPCF",
    "NNCorrelation", "GGCorrelation", "NGCorrelation",
    "NNNCorrelation", "GGGCorrelation", "GNNCorrelation", "NGGCorrelation",
    "NNNNCorrelation_NoTomo", "GGGGCorrelation_NoTomo", "GNNNCorrelation_NoTomo",
    # Direct estimators
    "DirectEstimator", "Direct_MapnEqual", "Direct_NapnEqual", "Direct_Map3Unequal",
    "MapCombinatorics",
    # Grids and helpers
    "FlatDataGrid_2D", "FlatPixelGrid_2D",
    "cat2hpx", "frompatchindices_preparerot", "gen_cat_patchindices", "toorigin",
    "pickle_load", "pickle_save",
    "check_clib_error", "convertunits", "flatlist", "get_site_packages_dir",
    "map_ztuples", "search_file_in_site_package", "symmetrize_map3_multiscale",
    "gen_n2n3indices_Gtildefourth", "gen_n2n3indices_Upsfourth",
    "gen_thetacombis_fourthorder",
]
