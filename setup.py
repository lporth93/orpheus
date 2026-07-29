import os
import sys
import shutil
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Helper to see if some compiler setup works (small test compile+link)
def try_compile(code, compiler, cflags=None, lflags=None, include_dirs=None, library_dirs=None,
                suffix=".c"):
    import tempfile
    cflags = cflags or []
    lflags = lflags or []
    include_dirs = include_dirs or []
    library_dirs = library_dirs or []

    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
        f.write(code)
        src_name = f.name
    obj_name = src_name.replace(suffix, ".o")
    exe_name = src_name.replace(suffix, "")

    try:
        compile_cmd = [compiler] + cflags + ["-c", src_name, "-o", obj_name]
        for d in include_dirs:
            compile_cmd.extend(["-I", d])
        subprocess.check_call(compile_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        link_cmd = [compiler] + [obj_name] + lflags + ["-o", exe_name]
        for d in library_dirs:
            link_cmd.extend(["-L", d])
        subprocess.check_call(link_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    finally:
        for fname in (src_name, obj_name, exe_name):
            if os.path.exists(fname):
                os.remove(fname)

CXX_MISSING = """
%(rule)s
orpheus needs a C++ compiler to build orpheus/src/healpix_utils.cpp. None of the
following could compile a trivial C++ program:

%(tried)s

%(rule)s
"""

# Find the C++ driver needed for for healpix_utils.cpp
def detect_cxx(cc_path=None):
    candidates = []
    if os.environ.get("CXX"):
        candidates.append(os.environ["CXX"])
    if cc_path:
        name = os.path.basename(cc_path)
        for c_name, cxx_name in (("gcc", "g++"), ("clang", "clang++"), ("icc", "icpc")):
            if name.startswith(c_name):
                candidates.append(os.path.join(os.path.dirname(cc_path),
                                               name.replace(c_name, cxx_name, 1)))
    candidates += ["c++", "g++", "clang++"]

    probe = "#include <vector>\nint main(){ std::vector<int> v(1); return (int)v.size()-1; }\n"
    tried = []
    for cxx in candidates:
        path = shutil.which(cxx)
        if path and path not in tried:
            if try_compile(probe, path, cflags=["-std=c++14"], suffix=".cpp"):
                return path
            tried.append(path)
    raise RuntimeError(CXX_MISSING % {"rule": "=" * 78,
                                      "tried": "\n".join("    " + t for t in tried) or "    (none found)"})


# Find first available compiler
def detect_compiler(preferred=("gcc-15", "gcc-14", "gcc-13", "gcc-12", "gcc-11", "gcc", "icc")):
    # On macOS the whole toolchain must be clang: the SDK and the extension link
    # flags both assume it. OpenMP comes from libomp.
    if sys.platform == "darwin":
        preferred = ("clang",)
    for cc in preferred:
        path = shutil.which(cc)
        if not path:
            continue
        try:
            version_output = subprocess.check_output([path, "--version"], stderr=subprocess.STDOUT).decode()
            version_string = version_output.splitlines()[0].strip()
            out_lower = version_output.lower()
        except Exception:
            continue

        if "gcc" in out_lower and "apple" not in out_lower:
            return path, "gcc", version_string
        elif "clang" in out_lower:
            # test OpenMP support for this clang
            omp_code = "#include <omp.h>\nint main(){ return omp_get_max_threads(); }"
            cflags_omp = ["-fopenmp", "-O3", "-std=c99"]
            lflags_omp = ["-fopenmp", "-lm"]
            if try_compile(omp_code, path, cflags=cflags_omp, lflags=lflags_omp):
                return path, "clang-omp", version_string
            return path, "clang", version_string
        elif "intel" in out_lower or "icc" in out_lower:
            return path, "icc", version_string

    # Fallback to default 'cc'
    cc_path = shutil.which("cc") or "cc"
    return cc_path, "unknown", "Version not detected for fallback compiler."


class BuildExtWithDetect(build_ext):
    def build_extensions(self):
        # Prefer Apple clang on macOS to avoid mixing Homebrew GCC with Xcode SDK headers
        if sys.platform == "darwin":
            clang = shutil.which("clang")
            clangpp = shutil.which("clang++") or clang
            if clang:
                try:
                    for key, exe in (
                        ("compiler_so", clang),
                        ("compiler_cxx", clangpp),
                        ("linker_so", clang),
                        ("linker_exe", clangpp),
                    ):
                        try:
                            self.compiler.set_executable(key, exe)
                        except Exception:
                            try:
                                setattr(self.compiler, key, [exe])
                            except Exception:
                                pass
                except Exception:
                    pass

        # Detect compiler (may override earlier values)
        cc_path, cc_type, cc_version = detect_compiler()
        if cc_path:
            try:
                self.compiler.set_executable("compiler_so", cc_path)
                self.compiler.set_executable("linker_so", cc_path)
            except Exception:
                try:
                    setattr(self.compiler, "compiler_so", [cc_path])
                    setattr(self.compiler, "linker_so", [cc_path])
                except Exception:
                    pass

        # Check whether the selected compiler supports OpenMP.
        # macOS links extensions as bundles, and setuptools passes -bundle itself;
        # adding -shared there means -dynamiclib, which conflicts with it.
        link_shared = [] if sys.platform == "darwin" else ["-shared"]
        omp_cflags = ["-fopenmp", "-O3", "-ffast-math", "-std=c99", "-fPIC"]
        omp_lflags = link_shared + ["-fopenmp", "-lm"]
        use_openmp = False
        applied_alternative_clang_flags = False

        omp_test_code = "#include <omp.h>\nint main(){ return 0; }"
        if try_compile(omp_test_code, cc_path, cflags=["-fopenmp"], lflags=["-fopenmp"]):
            use_openmp = True
        else:
            if sys.platform == "darwin":
                brew_prefixes = ["/opt/homebrew", "/usr/local"]
                libomp_include = None
                libomp_lib = None
                for p in brew_prefixes:
                    inc = os.path.join(p, "opt", "libomp", "include")
                    lib = os.path.join(p, "opt", "libomp", "lib")
                    if os.path.isdir(inc) and os.path.isdir(lib):
                        libomp_include = inc
                        libomp_lib = lib
                        break
                if not libomp_include:
                    for p in brew_prefixes:
                        inc = os.path.join(p, "include")
                        lib = os.path.join(p, "lib")
                        if os.path.isdir(inc) and os.path.isdir(lib) and os.path.exists(os.path.join(lib, "libomp.dylib")):
                            libomp_include = inc
                            libomp_lib = lib
                            break

                if libomp_include and libomp_lib:
                    clang_alt_cflags = ["-Xpreprocessor", "-fopenmp", "-O3", "-ffast-math", "-std=c99", "-fPIC"]
                    clang_alt_lflags = ["-lomp", "-lm"]
                    if try_compile(omp_test_code, cc_path, cflags=clang_alt_cflags, lflags=clang_alt_lflags,
                                   include_dirs=[libomp_include], library_dirs=[libomp_lib]):
                        use_openmp = True
                        applied_alternative_clang_flags = True
                        omp_cflags = clang_alt_cflags
                        omp_lflags = ["-L" + libomp_lib, "-lomp", "-lm"]
                    else:
                        use_openmp = False
                else:
                    use_openmp = False

        if not use_openmp:
            print("WARNING: OpenMP support not detected for the selected compiler.")

        # Configure the build flags for each extension.
        for ext in self.extensions:
            if use_openmp:
                ext.extra_compile_args = omp_cflags
                ext.extra_link_args = omp_lflags
                if applied_alternative_clang_flags:
                    for p in ["/opt/homebrew", "/usr/local"]:
                        inc = os.path.join(p, "opt", "libomp", "include")
                        lib = os.path.join(p, "opt", "libomp", "lib")
                        if os.path.isdir(inc) and os.path.isdir(lib):
                            ext.include_dirs = list(ext.include_dirs or []) + [inc]
                            ext.library_dirs = list(getattr(ext, "library_dirs", []) or []) + [lib]
                            break
            else:
                ext.extra_compile_args = ["-O3", "-ffast-math", "-std=c99", "-fPIC"]
                ext.extra_link_args = link_shared + ["-lm"]

        # Most files are C; healpix_utils.cpp and the vendored HEALPix sources need
        # C++ flags instead. setuptools compiles those with its own driver, which need
        # not be the OpenMP-capable compiler picked for the C sources, and Apple clang
        # rejects -fopenmp outright. No C++ source here uses OpenMP, so the flag is
        # simply dropped for them.
        cxx_path = detect_cxx(cc_path)
        orig_compile = self.compiler._compile

        # The C++ sources are compiled with the C++ driver rather than with the
        # compiler_so set above, which may be a gcc without a C++ frontend.
        def _compile_per_lang(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if os.path.splitext(src)[1] in (".cpp", ".cc", ".cxx"):
                postargs = [a for a in extra_postargs
                            if a not in ("-std=c99", "-fopenmp", "-Xpreprocessor")]
                postargs = postargs + ["-std=c++14"]
                saved_cc = self.compiler.compiler_so
                self.compiler.compiler_so = [cxx_path] + list(saved_cc[1:])
                try:
                    return orig_compile(obj, src, ext, cc_args, postargs, pp_opts)
                finally:
                    self.compiler.compiler_so = saved_cc
            return orig_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        self.compiler._compile = _compile_per_lang
        # macOS links C++ through clang, whose runtime is libc++; recent SDKs no
        # longer ship libstdc++ at all.
        cxx_runtime = "-lc++" if sys.platform == "darwin" else "-lstdc++"
        for ext in self.extensions:
            ext.extra_link_args = list(ext.extra_link_args or []) + [cxx_runtime]

        super().build_extensions()


# All external modules from orpheus
clib_sources = [
    "orpheus/src/utils.c",
    "orpheus/src/assign.c",
    "orpheus/src/healpix_utils.cpp",
    "orpheus/src/spatialhash.c",
    "orpheus/src/combinatorics.c",
    "orpheus/src/directestimator.c",
    "orpheus/src/corrfunc_second.c",
    "orpheus/src/corrfunc_third.c",
    "orpheus/src/corrfunc_third_derived.c",
    "orpheus/src/corrfunc_fourth.c",
    "orpheus/src/corrfunc_fourth_derived.c",]

# Vendored HEALPix subset backing healpix_utils.cpp, see orpheus/src/healpix/README.md
clib_sources += [
    "orpheus/src/healpix/healpix_base.cc",
    "orpheus/src/healpix/healpix_tables.cc",
    "orpheus/src/healpix/error_handling.cc",
    "orpheus/src/healpix/geom_utils.cc",
    "orpheus/src/healpix/pointing.cc",
    "orpheus/src/healpix/string_utils.cc",]

# The covariance kernels are not part of the distribution for now
if os.path.exists("orpheus/src/cov_postq.c"):
    clib_sources.append("orpheus/src/cov_postq.c")

ext_modules = [
    Extension(
        "orpheus.orpheus_clib",
        sources=clib_sources,
        include_dirs=["orpheus/src", "orpheus/src/healpix"],
    ),
]

# read long description from README
thisfile = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(thisfile, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="orpheus-npcf",
    version="0.3.0",
    description="Compute N-point correlation functions of spin-s fields.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="GPL-3.0-or-later",
    url="https://github.com/lporth93/orpheus",
    author="Lucas Porth",
    packages=["orpheus"],
    python_requires=">=3.10",
    install_requires=[
        "astropy>=6",
        "healpy>=1.18",
        "numba>=0.61,<=0.62.1",
        "numpy>=1.24",
        "scipy>=1.15",
        "scikit-learn",],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtWithDetect},
    include_package_data=False,
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    ],
)