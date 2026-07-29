NGCorrelation: Lens-Shear Correlations
--------------------------------------

The ``NGCorrelation`` class computes the correlation of one scalar field and one spin-2 field.

Example
~~~~~~~

.. code-block:: python

    import orpheus

    scat = orpheus.SpinTracerCatalog(
        spin=2, pos1=x, pos2=y, tracer_1=g1, tracer_2=g2)
    lcat = orpheus.ScalarTracerCatalog(
        pos1=x, pos2=y, tracer=weight)
    ng = orpheus.NGCorrelation(
        min_sep=1., max_sep=128., binsize=0.1, nthreads=nthreads)
    ng.process(scat,lcat)  # Compute 2PCF in multipole basis


.. autoclass:: orpheus.NGCorrelation
   :members:
   :inherited-members:
   :show-inheritance: