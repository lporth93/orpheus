# Here we make sure that the structs in orpheus/src/multires_structs.h and in 
# orpheus/multires_structs.py are consistent with each other.

import ctypes
import glob
import os

import pytest

import orpheus
from orpheus import multires_structs as ms


# Tags are the declaration order in multires_structs.h, which is the order the C side
# exports them under
STRUCTS = [(0, ms.MultiresoCatalog),
           (1, ms.NavHash),
           (2, ms.TreeResoParams),
           (3, ms.BinningParams),
           (4, ms.NPCFOutput),
           (5, ms.FourthParams),
           (6, ms.ClustCorr),]


@pytest.fixture(scope="module")
def clib():
    pkgdir = os.path.dirname(os.path.abspath(orpheus.__file__))
    lib = ctypes.CDLL(glob.glob(os.path.join(pkgdir, "orpheus_clib*.so"))[0])
    lib.orpheus_struct_layout.restype = ctypes.c_int
    lib.orpheus_struct_layout.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
                                          ctypes.c_int]
    lib.orpheus_struct_fields.restype = ctypes.c_char_p
    lib.orpheus_struct_fields.argtypes = [ctypes.c_int]
    return lib


def c_layout(clib, tag):
    """sizeof followed by every field offset, as the compiler laid the struct out."""
    nentries = clib.orpheus_struct_layout(tag, None, 0)
    buf = (ctypes.c_size_t * nentries)()
    clib.orpheus_struct_layout(tag, buf, nentries)
    return list(buf)


@pytest.mark.parametrize("tag,cls", STRUCTS, ids=[cls.__name__ for _, cls in STRUCTS])
def test_struct_layout_matches_c(clib, tag, cls):
    layout = c_layout(clib, tag)
    names = [field[0] for field in cls._fields_]
    cnames = clib.orpheus_struct_fields(tag).decode().split(",")

    assert layout, "the C side exports no layout for %s"%cls.__name__
    assert ctypes.sizeof(cls) == layout[0], (
        "sizeof(%s) is %d in python and %d in C"%(cls.__name__, ctypes.sizeof(cls), layout[0]))
    # First make sure that the names match and then that the offsets match for each name
    assert names == cnames, (
        "%s declares its fields in a different order in python than in C:\n  python: %s\n  C:      %s"%(
            cls.__name__, names, cnames))
    for name, offset in zip(names, layout[1:]):
        assert getattr(cls, name).offset == offset, (
            "%s.%s sits at byte %d in python and %d in C"%(
                cls.__name__, name, getattr(cls, name).offset, offset))


def test_all_mirrored_structs_are_covered():
    """Every ctypes.Structure in the module is checked, so a new one cannot slip through."""
    mirrored = {name for name, obj in vars(ms).items()
                if isinstance(obj, type) and issubclass(obj, ctypes.Structure)}
    assert mirrored == {cls.__name__ for _, cls in STRUCTS}