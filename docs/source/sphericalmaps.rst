Aperture mass maps on the celestial sphere
==========================================

While the direct estimators compress the shear field into aperture statistics at a set of radii,
:class:`orpheus.SphericalMap` keeps the map itself.

.. note::

   The estimator is first order in the aperture mass. Higher moments :math:`M_\mathrm{ap}^n`
   require the multiple-counting corrections of the direct estimators and are computed by
   :doc:`Direct_MapnEqual <directmapn>` instead.

For a worked example on a full-sky simulated catalog see the
:doc:`tutorial notebook <notebooks/spherical_maps>`.

.. autoclass:: orpheus.SphericalMap
    :members:
    :special-members: __add__
    :show-inheritance:
