.. _installation:

Installation
============

``orpheus`` compiles a parallelised C/C++ extension at install time, so the
installation needs a working compiler toolchain in addition to the python
dependencies. No external C libraries are required.

Quick install
-------------

As orpheus is hosted on pypi a simple

.. code-block:: shell

  pip install orpheus-npcf

should be sufficient. Alternatively you can clone the github repo and build from
source

.. code-block:: shell

   git clone https://github.com/lporth93/orpheus.git
   cd orpheus
   pip install .

Known version constraints
-------------------------


* ``python>=3.10``, inherited from ``scipy>=1.15`` and ``healpy>=1.18``.
* ``numba<=0.62.1``, which in turn caps ``numpy`` below 2.4. Neither bound
  originates in the orpheus code itself.

Troubleshooting
---------------

``WARNING: OpenMP support not detected for the selected compiler.``
    The build proceeds, but the estimators will run on a single thread. On macOS
    install ``libomp`` (``brew install libomp``), on Linux make sure a GCC
    installation with ``libgomp`` is available.

``RuntimeWarning: Found N OpenMP runtimes in this process``, or a segmentation fault on macOS
    Python wheels on macOS tend to bundle their own copy of ``libomp.dylib``, and
    both healpy and scikit-learn ship one. The copies coalesce their weak symbols
    into a single definition, so a worker thread created by one runtime can end up
    suspending itself in the state of another one. The process then segfaults as
    soon as a kernel opens a parallel region whenever ``nthreads>1``.

    The Apple silicon wheels of orpheus bind to the copy healpy already provides
    rather than adding another one. The x86_64 wheels still carry their own, as the
    copies vendored there predate symbols that the kernels need. orpheus emits the
    warning above whenever it loads its C library and finds more than one runtime:
    two coexisting copies are the norm on macOS and generally harmless, three are
    what might crash. To list them by hand use

    .. code-block:: python

       import ctypes
       dyld = ctypes.CDLL(None)
       dyld._dyld_get_image_name.restype = ctypes.c_char_p
       for i in range(dyld._dyld_image_count()):
           name = dyld._dyld_get_image_name(i).decode()
           if "omp" in name:
               print(name)

    The clean fix is an environment in which every package links the same
    ``libomp``, which conda-forge provides.

    .. code-block:: shell

       conda create -n orpheus -c conda-forge --override-channels python=3.12
       conda activate orpheus
       conda install -c conda-forge --override-channels healpy scikit-learn scipy numpy astropy llvm-openmp
       pip install --no-binary orpheus-npcf orpheus-npcf

    In an existing environment the copies can instead be pointed at a single file.
    This has to be repeated whenever pip reinstalls one of the packages

    .. code-block:: shell

       SP=$(python -c "import site; print(site.getsitepackages()[0])")
       for p in sklearn orpheus; do
         [ -e "$SP/$p/.dylibs/libomp.dylib" ] || continue
         mv "$SP/$p/.dylibs/libomp.dylib" "$SP/$p/.dylibs/libomp.dylib.backup"
         ln -s "$SP/healpy/.dylibs/libomp.dylib" "$SP/$p/.dylibs/libomp.dylib"
       done
