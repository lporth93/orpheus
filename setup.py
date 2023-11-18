from setuptools import find_packages, Extension
from distutils.core import setup
from distutils.command.install import install as DistutilsInstall
import os
import glob
import sys
import subprocess
import shutil

"""
compiler = "gcc" #mpicc
flags = "-fopenmp -fPIC -Wall -O3"
#pythonVersion= "python3.5"
pythonVersion= "python3.10"

# Somehow I am too stupid to merge this in the basic install method. Right now
# is a placeholder for compiling the external .c code s.t. it binds to ctypes.
# Thus for full building we need to use `python setup.py install compile_ext`...
class compile_ext(DistutilsInstall):

    def run(self):
        filenames = ["spatialhash","discrete","assign"]
        base = sys.exec_prefix + "/lib/"+pythonVersion+"/site-packages/"
        latest_dir = max(glob.glob(os.path.join(
            base, 'orpheus*/')), key=os.path.getmtime)
        for fname in filenames:
            compile_single(fname, latest_dir)
        compile_dll(filenames, "clibrary", latest_dir)
        

def compile_single(fname, latest_dir):
    sys.stdout.write("Compiling file %s...." % fname)
    full_fname = latest_dir + "orpheus/src/" + fname
    run_cmd("%s %s -c %s.c -std=c99" % (compiler, flags, full_fname))
    print(latest_dir)
    print("%s %s -c %s.c -std=c99" % (compiler, flags, full_fname))
    shutil.move(os.getcwd() + "/" + fname + ".o", full_fname + ".o")
    sys.stdout.write("Done\n")

def compile_dll(fnames, dll_name, latest_dir):
    sys.stdout.write("Compiling dll %s...." % dll_name)
    str_fnames = ""
    full_dllname = latest_dir + "orpheus/src/%s" % dll_name
    for fname in fnames:
        str_fnames += " " + latest_dir + "orpheus/src/%s.o" % fname
    run_cmd("%s %s -shared -o %s.so %s" %
            (compiler, flags, full_dllname, str_fnames))
    #run_cmd("gcc -fPIC -shared -o %s.so %s" %
    #       (full_dllname, str_fnames))
    sys.stdout.write("Done\n")
    
def run_cmd(cmd):
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
"""

# Dummy
def readme():
    with open('README.rst') as f:
        return f.read()
    
ext_modules = [
Extension(
    "orpheus_clib",
    sources=["orpheus/src/assign.c", "orpheus/src/spatialhash.c", "orpheus/src/discrete.c"],#, "orpheus/src/discrete.c"],#, "orpheus/src/spatialhash.c"],
    include_dirs=["orpheus/src/assign.h", "orpheus/src/spatialhash.h", "orpheus/src/discrete.h"],#, "orpheus/src/spatialhash.h"],
    extra_compile_args = ["-fopenmp", "-fPIC", "-Wall", "-O3", "-std=c99"],
    extra_linker_args = ["-fopenmp", "-fPIC", "-Wall", "-O3", "-std=c99"],
    ),
]

# Actual setup script
setup(name='orpheus',
      version='0.1',
      description='Compute N-point correlation functions of spin-s fields.',
      long_description=readme(),
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
          'numpy'],
      ext_modules=ext_modules,
      #cmdclass={'compile_ext': compile_ext},
      include_package_data=True,
      zip_safe=False)