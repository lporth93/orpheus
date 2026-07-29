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
