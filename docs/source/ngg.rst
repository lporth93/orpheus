NGGCorrelation: Lens-Shear-Shear Correlations
----------------------------------------------

The ``NGGCorrelation`` class computes the correlation of one scalar field and two spin-2 fields.

Example
~~~~~~~

.. code-block:: python

    import orpheus

    scat = orpheus.SpinTracerCatalog(
        spin=2, pos1=x, pos2=y, tracer_1=g1, tracer_2=g2)
    lcat = orpheus.ScalarTracerCatalog(
        pos1=x, pos2=y, tracer=weight)
    ngg = orpheus.NGGCorrelation(
        min_sep=1., max_sep=128., binsize=0.1, nthreads=nthreads)
    ngg.process(scat,lcat)  # Compute 3PCF in multipole basis
    ngg.multipoles2npcf()   # Transform to real-space basis


.. autoclass:: orpheus.NGGCorrelation
   :members:
   :inherited-members:
   :show-inheritance: