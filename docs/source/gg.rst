GGCorrelation: Shear-Shear Correlations
---------------------------------------

The ``GGCorrelation`` class computes the two-point correlation of spin-2 fields.

Example
~~~~~~~

.. code-block:: python

    import orpheus

    cat = orpheus.SpinTracerCatalog(
        spin=2, pos1=x, pos2=y, tracer_1=g1, tracer_2=g2)
    gg = orpheus.GGGCorrelation(
        min_sep=1., max_sep=128., binsize=0.1, nthreads=nthreads)
    gg.process(cat)      # Compute 2pcf


.. autoclass:: orpheus.GGCorrelation
   :members:
   :inherited-members:
   :show-inheritance: