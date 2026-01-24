NNCorrelation: Projected clustering two-point correlations
----------------------------------------------------------

The ``NNCorrelation`` class computes the paircounts between various point catalogs. Given a random catalog, it computes the two-point clustering correlation function.

Example
~~~~~~~

.. code-block:: python

    import orpheus

    data = orpheus.ScalarTracerCatalog(
        spin=2, pos1=xdat, pos2=ydat, tracer=np.ones_like(xdat))
    rand = orpheus.ScalarTracerCatalog(
        spin=2, pos1=xrand, pos2=yrand, tracer=np.ones_like(xrand))
    nn = orpheus.NNCorrelation(
        n_cfs=4, min_sep=1., max_sep=128., binsize=0.1, nthreads=nthreads)
    nn.process(cat=data, cat_random=rand)  # Compute xi via Landy-Szalay estimator


.. autoclass:: orpheus.NNCorrelation
   :members:
   :inherited-members:
   :show-inheritance: