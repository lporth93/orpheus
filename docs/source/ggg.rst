GGGCorrelation: Shear–Shear–Shear Correlations
----------------------------------------------

The ``GGGCorrelation`` class computes the three‑point correlation of spin-2 fields.

Example
~~~~~~~

.. code-block:: python

    import orpheus

    cat = orpheus.SpinTracerCatalog(
        spin=2, pos1=x, pos2=y, tracer_1=g1, tracer_2=g2)
    ggg = orpheus.GGGCorrelation(
        n_cfs=4, min_sep=1., max_sep=128., binsize=0.1, nthreads=nthreads)
    ggg.process(cat)      # Compute 3PCF in multipole basis
    ggg.multipoles2npcf() # Transform to real-space basis


.. autoclass:: orpheus.GGGCorrelation
   :members:
   :inherited-members:
   :show-inheritance: