.. _installation:

Installation
============

``orpheus`` compiles a parallelised C/C++ extension at install time, so the
installation needs a working compiler toolchain in addition to the python
dependencies.

Quick install
-------------

As orpheus is hosted on pypi a simple

.. code-block:: shell

  pip install orpheus-npcf

should be sufficient. The pypi wheels ship with ``healpix_cxx`` bundled, so this
route needs no system library. Alternatively you can first clone the github repo
and build from source, which does need ``healpix_cxx`` (see :ref:`healpix_cxx`)


.. code-block:: shell

   conda install -c conda-forge healpix_cxx pkg-config   # or: sudo apt install libhealpix-cxx-dev pkg-config
   git clone https://github.com/lporth93/orpheus.git
   cd orpheus
   pip install .


.. _healpix_cxx:

The healpix_cxx dependency
--------------------------

The curved-sky estimators query the pixels within a disc on a nested HEALPix map
from inside the parallelised C kernels, for which orpheus links against the
``healpix_cxx`` C++ library. The ``healpy`` python package does not expose this
at the C level and is therefore not a substitute. A source build needs both the
headers and the shared library; if they live in a prefix that ``pkg-config``
cannot see, point at it with ``HEALPIX_CXX_DIR``.

Known version constraints
-------------------------


* ``python>=3.10``, inherited from ``scipy>=1.15`` and ``healpy>=1.18``.
* ``numba<=0.62.1``, which in turn caps ``numpy`` below 2.4. Neither bound
  originates in the orpheus code itself.

Troubleshooting
---------------

``fatal error: healpix_base.h: No such file or directory``
    ``healpix_cxx`` is missing or not discoverable. Install it as described in
    :ref:`healpix_cxx`, or set ``HEALPIX_CXX_DIR`` to its prefix. Note that
    installing ``healpy`` does not help here.

``ImportError: libhealpix_cxx.so.3: cannot open shared object file``
    The extension built correctly, but the ``healpix_cxx`` shared library cannot
    be found at import time. This usually means it was installed into a prefix
    that is not on the loader path -- add ``$PREFIX/lib`` to
    ``LD_LIBRARY_PATH`` (``DYLD_LIBRARY_PATH`` on macOS), or install
    ``healpix_cxx`` into the same conda environment as ``orpheus``.

``WARNING: OpenMP support not detected for the selected compiler.``
    The build proceeds, but the estimators will run on a single thread. On macOS
    install ``libomp`` (``brew install libomp``), on Linux make sure a GCC
    installation with ``libgomp`` is available.
