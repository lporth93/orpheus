import os
import sys
import shutil
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Helper to see if some compiler setup works
def try_compile(code, compiler, cflags, lflags):
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
        f.write(code)
        src_name = f.name
    obj_name = src_name.replace('.c', '.o')
    exe_name = src_name.replace('.c', '')

    try:
        # Compile step
        compile_cmd = [compiler] + cflags + ['-c', src_name, '-o', obj_name]
        subprocess.check_call(compile_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Link step
        link_cmd = [compiler] + [obj_name] + lflags + ['-o', exe_name]
        subprocess.check_call(link_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    finally:
        for f in [src_name, obj_name, exe_name]:
            if os.path.exists(f):
                os.remove(f)

# Find first available compiler
def detect_compiler(preferred=("gcc-12", "gcc-11", "gcc", "clang", "icc")):
    
    for cc in preferred:
        path = shutil.which(cc)
        if not path:
            continue
        try:
            # Get the full version output
            version_output = subprocess.check_output([path, "--version"], stderr=subprocess.STDOUT).decode()
            # The first line is usually the version info
            version_string = version_output.splitlines()[0].strip()
            out_lower = version_output.lower()
        except Exception:
            continue
        
        if "gcc" in out_lower:
            return path, "gcc", version_string
        elif "clang" in out_lower:
            # Check if OpenMP works
            omp_code = "#include <omp.h>\nint main() { return omp_get_max_threads(); }"
            cflags_omp = ["-Xpreprocessor", "-fopenmp", "-O3", "-ffast-math", "-std=c99"]
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

        # Get compiler type and path
        cc_path, cc_type, cc_version = detect_compiler()
        if cc_path:
            self.compiler.set_executable("compiler_so", cc_path)
            self.compiler.set_executable("linker_so", cc_path)

        # Set compile and link arguments
        for ext in self.extensions:
            ext.extra_compile_args = ["-fopenmp", "-O3", "-ffast-math", "-std=c99", "-fPIC"]
            ext.extra_link_args    = ["-shared", "-fopenmp", "-lm"]

        super().build_extensions()

ext_modules = [
    Extension(
        "orpheus_clib",
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
            "orpheus/src/corrfunc_fourth_derived.c",
        ],
        include_dirs=["orpheus/src"],
    )
]

setup(
    name="orpheus",
    version="0.0.1",
    description="Compute N-point correlation functions of spin-s fields.",
    license="MIT",
    packages=["orpheus"],
    install_requires=[
        "astropy",
        "healpy",
        "numba",
        "numpy",
        "pathlib",
        "scipy",
        "scikit-learn",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtWithDetect},
    include_package_data=False,
    zip_safe=False,
)