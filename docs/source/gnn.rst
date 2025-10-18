GNNCorrelation: Lens-Lens-Shear Correlations
----------------------------------------------

The ``GNNCorrelation`` class computes the correlation of two spin-2 and one scalar field.

Example
~~~~~~~

.. code-block:: python

    import orpheus

    scat = orpheus.SpinTracerCatalog(
        spin=2, pos1=x, pos2=y, tracer_1=g1, tracer_2=g2)
    lcat = orpheus.ScalarTracerCatalog(
        pos1=x, pos2=y, tracer=weight)
    gnn = orpheus.GNNCorrelation(
        min_sep=1., max_sep=128., binsize=0.1, nthreads=nthreads)
    gnn.process(scat,lcat)  # Compute 3PCF in multipole basis
    gnn.multipoles2npcf()   # Transform to real-space basis


.. autoclass:: orpheus.GNNCorrelation
   :members:
   :inherited-members:
   :show-inheritance: