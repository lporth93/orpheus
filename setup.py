import os
import sys
import shutil
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Helper to see if some compiler setup works (small test compile+link)
def try_compile(code, compiler, cflags=None, lflags=None, include_dirs=None, library_dirs=None):
    import tempfile
    cflags = cflags or []
    lflags = lflags or []
    include_dirs = include_dirs or []
    library_dirs = library_dirs or []

    with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False) as f:
        f.write(code)
        src_name = f.name
    obj_name = src_name.replace(".c", ".o")
    exe_name = src_name.replace(".c", "")

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

# Find first available compiler
def detect_compiler(preferred=("gcc-15", "gcc-14", "gcc-13", "gcc-12", "gcc-11", "gcc", "icc")):
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

        # Determine OpenMP support and appropriate flags
        omp_cflags = ["-fopenmp", "-O3", "-ffast-math", "-std=c99", "-fPIC"]
        omp_lflags = ["-shared", "-fopenmp", "-lm"]
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
                    clang_alt_cflags = ["-Xpreprocessor", "-fopenmp", "-O0", "-ffast-math", "-std=c99", "-fPIC"]
                    clang_alt_lflags = ["-lomp", "-lm"]
                    if try_compile(omp_test_code, cc_path, cflags=clang_alt_cflags, lflags=clang_alt_lflags,
                                   include_dirs=[libomp_include], library_dirs=[libomp_lib]):
                        use_openmp = True
                        applied_alternative_clang_flags = True
                        omp_cflags = clang_alt_cflags
                        # macOS: link as a Python extension bundle and link libomp
                        omp_lflags = ["-bundle", "-undefined", "dynamic_lookup", "-L" + libomp_lib, "-lomp", "-lm"]
                    else:
                        use_openmp = False
                else:
                    use_openmp = False

        if not use_openmp:
            print("WARNING: OpenMP support not detected for the selected compiler.")

        # Apply compile/link args per-extension
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
                ext.extra_link_args = ["-shared", "-lm"]

        super().build_extensions()


# All external modules from orpheus
ext_modules = [
    Extension(
        "orpheus.orpheus_clib",
        sources=[
            "orpheus/src/utils.c",
            "orpheus/src/assign.c",
            "orpheus/src/spatialhash.c",
            "orpheus/src/combinatorics.c",
            "orpheus/src/directestimator.c",
            "orpheus/src/corrfunc_second.c",
            "orpheus/src/corrfunc_third.c",
            "orpheus/src/corrfunc_third_derived.c",
            "orpheus/src/corrfunc_fourth.c",
            "orpheus/src/corrfunc_fourth_derived.c",],
        include_dirs=["orpheus/src"],
    )
]

# read long description from README
thisfile = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(thisfile, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="orpheus-npcf",
    version="0.2.2",
    description="Compute N-point correlation functions of spin-s fields.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/lporth93/orpheus",
    author="Lucas Porth",
    packages=["orpheus"],
    python_requires=">=3.9",
    install_requires=[
        "astropy>=6",
        "healpy>=1.17",
        "coverage>=7.6.1",
        "numba>=0.58,<=0.62.1",
        "numpy>=1.22,<1.27",
        "scipy>=1.15",
        "scikit-learn",],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtWithDetect},
    include_package_data=False,
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)