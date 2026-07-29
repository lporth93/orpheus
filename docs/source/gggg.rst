GGGGCorrelation: Shear-Shear-Shear-Shear Correlations
-----------------------------------------------------

The ``GGGGCorrelation`` class computes the four-point correlation of spin-2 fields.

Example
~~~~~~~

.. code-block:: python

    import orpheus

    cat = orpheus.SpinTracerCatalog(
        spin=2, pos1=x, pos2=y, tracer_1=g1, tracer_2=g2)
    gggg = orpheus.GGGGCorrelation_NoTomo(
        min_sep=1., max_sep=128., binsize=0.1, nthreads=nthreads)
    gggg.process(cat)      # Compute 4PCF in multipole basis
    gggg.multipoles2npcf() # Transform to real-space basis


.. autoclass:: orpheus.GGGGCorrelation_NoTomo
   :members:
   :inherited-members:
   :show-inheritance: