NNNCorrelation: Projected clustering three-point correlations
-------------------------------------------------------------

The ``NNNCorrelation`` class computes the triplet between various point catalogs. Given a random catalog, it computes the three-point clustering correlation function.

Example
~~~~~~~

.. code-block:: python

    import numpy as np
    import orpheus

    data = orpheus.ScalarTracerCatalog(
        pos1=xdat, pos2=ydat, tracer=np.ones_like(xdat))
    rand = orpheus.ScalarTracerCatalog(
        pos1=xrand, pos2=yrand, tracer=np.ones_like(xrand))
    nnn = orpheus.NNCorrelation(
        min_sep=1., max_sep=128., binsize=0.1, nthreads=nthreads)
    nnn.process(cat=data, cat_random=rand)  # Compute zeta via Landy-Szalay estimator
    nnn.multipoles2npcf()


.. autoclass:: orpheus.NNNCorrelation
   :members:
   :inherited-members:
   :show-inheritance: