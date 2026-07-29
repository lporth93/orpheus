GNNNCorrelation: Lens-Lens-Lens-Shear Correlations
--------------------------------------------------

The ``GNNNCorrelation`` class computes the correlation of one spin-2 and three scalar fields.

Example
~~~~~~~

.. code-block:: python

    import orpheus

    scat = orpheus.SpinTracerCatalog(
        spin=2, pos1=x, pos2=y, tracer_1=g1, tracer_2=g2)
    lcat = orpheus.ScalarTracerCatalog(
        pos1=x, pos2=y, tracer=weight)
    gnnn = orpheus.GNNNCorrelation_NoTomo(
        min_sep=1., max_sep=128., binsize=0.1, nthreads=nthreads)
    gnnn.process(scat,lcat)  # Compute 4PCF in multipole basis
    gnnn.multipoles2npcf()   # Transform to real-space basis


.. autoclass:: orpheus.GNNNCorrelation_NoTomo
   :members:
   :inherited-members:
   :show-inheritance: