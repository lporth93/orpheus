from setuptools import find_packages, Extension
from distutils.core import setup
from distutils.sysconfig import get_config_var
import os
import sys

__version__ = "0.0.1"

if (sys.platform[:6] == "darwin"
        and ("clang" in get_config_var("CC")
                or "clang" in os.environ.get("CC", ""))):
    compiler_args = ["-Xpreprocessor"]
    linker_args = ["-mlinker-version=305", "-Xpreprocessor"]
else:
    # Force this for now as anacoda default compiler incompatible with newer gcc versions
    # See https://github.com/ContinuumIO/anaconda-issues/issues/11152
    os.environ["CC"] = "/usr/bin/gcc" 
    compiler_args = ["-shared"]
    linker_args = ["-shared"]

compiler_args += ["-fopenmp", "-fPIC", "-Wall", "-lm", "-O3", "-ffast-math", "-std=c99", "-lgomp" "-lrt"]
linker_args +=   ["-fopenmp", "-fPIC", "-Wall", "-lm", "-O3", "-ffast-math", "-std=c99", "-lgomp" "-lrt"]
    
ext_modules = [
Extension(
    "orpheus_clib",
    sources=["orpheus/src/assign.c", "orpheus/src/spatialhash.c", "orpheus/src/discrete.c"],#, "orpheus/src/discrete.c"],#, "orpheus/src/spatialhash.c"],
    include_dirs=["orpheus/src/assign.h", "orpheus/src/spatialhash.h", "orpheus/src/discrete.h"],#, "orpheus/src/spatialhash.h"],
    extra_compile_args = compiler_args,
    extra_linker_args = linker_args,
    ),
]

# Actual setup script
setup(name='orpheus',
      version=__version__,
      description='Compute N-point correlation functions of spin-s fields.',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.5',
      ],
      keywords='Weak lensing, ggl',
      # url='http://bitbucket.org/lporth/directmap',
      author='Lucas Porth',
      # author_email='lporth@sussex.ac.uk',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy', 'pathlib'],
      ext_modules=ext_modules,
      #cmdclass={'compile_ext': compile_ext},
      include_package_data=True,
      zip_safe=False)
