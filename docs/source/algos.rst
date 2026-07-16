How orpheus works
=================

The main reason for the efficiency of ``orpheus`` is twofold. First, it computes
the higher-order correlators in their multipole basis and second, it uses tree-based
methods to speed up the two different parts of this computation.

Multipole decomposition
-----------------------

In real space, a component of the :math:`N\mathrm{PCF}` is a function of the arguments 
:math:`(x_1, x_2, \cdots, x_{N-1}, \phi_{1,2}, \cdots, \phi_{1,N-1})` and the estimators need to 
search for :math:`N`-tuplets of points which are then assigned to their corresponding bin. While
tree-based methods can make the search for the :math:`N`-tuplets more efficient, the fundamental
scaling of the algorithm will remain dependent on the order of the correlator.

In `orpheus` we make use of the multipole decomposition of the :math:`N\mathrm{PCF}` that has been developed
by Chen & Szapudi (2005, ApJ, 635, 743), Slepian & Eisenstein (2015, MNRAS, 454, 4142),
and Philcox et al. (2022, MNRAS, 509, 2457). In particular, we also include the expressions for the 
correlations of non-spin-0 fields as introduced in Porth et al (2024, A&A, 689, 227) and extended in
Porth et al (2025, arXiv:xxxx.xxxx). The multipole components of some hypothetic 
*N* point correlator, :math:`\mathscr{C}`, are related to the real-space components as 

.. math::
    \mathscr{C}^{\mathcal{P}}\left(\Theta_1, \cdots, \Theta_{N-1},\phi_{1 \, 2},\cdots,
    \phi_{1 \, N-1}\right) 
    \sim \sum_{\mathbf{n}=-\infty}^\infty 
     \mathscr{C}^{\mathcal{P}}_{\mathbf{n}_{N-2}}(\Theta_1,\cdots, \Theta_{N-1}) \ 
     \mathrm{e}^{\mathrm{i} n_{2}\phi_{1,2}} \cdots \mathrm{e}^{\mathrm{i} n_{N-1}\phi_{1,N-1}} \ ,

where the :math:`\Theta_k` denote the radial bins, :math:`\phi_{1,j}` is the polar angle
between the vertices :math:`\vartheta_1` and :math:`\vartheta_j`, and the :math:`\mathcal{P}` 
denotes a certain projection applied to the field of tracers; the latter is only relevant for 
correlators containing non-spin-0 objects. One can invert this relation to obtain an expression
for the multipole components in terms of the field of tracers. If one chooses a suitable
projection for the non-spin-0 fields this relation can be brought to the form

.. math::

   \mathscr{C}_{\mathbf{n}_{N-2}}(\Theta_1, \cdots, \Theta_{N-1}) 
    \sim
    \sum_{i=1}^{N_{\rm{disc}}} x\left(\vec{\vartheta_i}\right)  
    \ X_{n'_2}^{\rm{disc}} \left( \Theta_1; \vec{\vartheta_i}\right) 
    \ \cdots \ X_{n'_{N}}^{\rm{disc}} \left( \Theta_{N-1}; \vec{\vartheta_i}\right) \ ,

where the :math:`x` denotes the value of the tracer in question (i.e. :math:`w` for number counts 
or :math:`we_\mathrm{c}` for ellipticities), the :math:`X_{n'_k}^{\rm disc}` are the building blocks
(i.e. :math:`W_n` for number counts or :math:`G_n` for ellipticities) and the :math:`n'_k` are a 
linear combination of the multipole components :math:`n_k, \ k\in\{2,\cdots,N-1\}`. Schematically, the
shape of the :math:`X_n` reads

.. math::

    X_{n}^{\rm{disc}} \left( \Theta; \vec{\vartheta_i}\right) =
    \sum_{j=1}^{N_{\rm{disc}}} x\left(\vec{\vartheta_j}\right) \ 
    \mathrm{e}^{\mathrm{i} n \varphi_{ij}} \ \mathcal{B}(\theta_{ij} \in \Theta) \ ,

which can formally be seen as a range search problem. Looking at the previous two expressions, we see 
that the multipole-based estimation of :math:`\mathscr{C}_{\mathbf{n}_{N-2}}` consists of two steps

::

    multipoles = initialise_multipoles()
    for every tracer in tracers:
        nextXn = allocate_Xn(tracer)                   # scales as O(N_{\mathrm{tracers}})
        update_multipoles(multipoles, tracer,nextXn)   # scales as O(N_{\mathrm{bins}})

Focusing at the two different steps we infer that the estimator has a time complexity of 
:math:`\mathcal{O}(N_{\mathrm{tracers}}^2)+\mathcal{O}(N_{\mathrm{tracers}} \, N_{\mathrm{bins}})`.
While this scaling is much more beneficial than for brute-force estimators, it can nevertheless
become computationally impractical. In the next subsection we show how ``orpheus`` further reduces
this scaling.


.. note::

    In case of a tomographic survey we would also need a seperate set of indices for the different
    tomographic bins. This is omitted for notational convenience but it is implemented within ``orpheus``.

Hierarchical spatial hashing
----------------------------
The core pair-finding algorithm in ``orpheus`` is built on spatial hashing and we apply this to all present
geometries. While the implementation for each geometry differs a bit, the core idea is the same. We divide
the geometry by grids G of a predefined shape. We then use a hash function to map the galaxy coordinates to
a grid index and then store references to the galaxies inside the cell they occupy. Such a data layout implies
that all galaxies within a given cell can be looked up in :math:`\mathcal{O}(1)` time, so that a range search
only has to visit the cells overlapping the search region.

We can further use these grid cells to construct a `reduced` catalog by merging all galaxies within a given grid 
cell into one where we average the tracer quantities (such as the shapes) and sum up the tracer weights. By 
constructing a hierarchy of such grids  and increasing the sidelength of each cell by powers of two, we can build 
connections between the galaxies residing in the hash cells of the various resolutions. 

Flat two-dimensional geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Here the grid cells are simple square pixels laid over the footprint. Each galaxy is mapped to a
flat cell index :math:`p = i_x + i_y\, n_1` and the members of a cell are recovered through a
compressed layout consisting of three arrays: ``index_matcher`` points from a cell to its slot in
the sparse structure, ``pixs_galind_bounds`` holds the start/stop offsets of that slot,
and ``pix_gals`` lists the catalogue indices of the galaxies it contains. Empty cells map to ``-1``
and are skipped. You can have a look yourself by hovering over a pixel in the figure how this 
lookup traces.

.. raw:: html

   <iframe
     src="_static/spatialhash_flat.html"
     width="100%"
     height="700px"
     frameborder="0"
     style="border: none;"
   ></iframe>


Two-dimensional slabs within a 3D box
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For a projected statistic in a 3D box projection length :math:`\pm\Pi` we first slice the box along the 
line of sight into slabs of width ``dpix_z`` and assign every galaxy to the slab containing its line-of-sight 
coordinate. Within each slab we then build one two-dimensional spatial hash of the transverse positions, 
all sharing a same transverse grid so that the slabs stay aligned. In order to only having to iterate over 
just a few of those slabs when searching for pairs we choose the value of ``dpix_z`` to be similar to 
:math:`\Pi`.

Data on the celestial sphere
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When a catalog is processed on the sphere we hash the galaxies onto a nested ``healpix`` scheme. Each galaxy 
is mapped to a nested pixel index via ``ang2pix`` and the galaxies are sorted so that every occupied pixel 
owns a contiguous slice of the reordered arrays. Only the occupied pixels are kept: their identifiers are 
stored in ``cell_pix`` and their bounds in ``cell_redbounds``, so an empty pixel simply never appears. Sorting is what makes this contiguous, sparse layout possible: since
only the occupied pixels are stored, the memory scales with the number of populated cells rather than with the
full :math:`12\, N_{\rm side}^2` ``healpix`` map, so that even a fine ``nside`` over a small footprint stays cheap.
When a cell
merges several galaxies into a reduced one, the associated shears are parallel-transported along the connecting 
geodesic before being averaged, so that spin-2 quantities remain well defined on the curved sky.

.. raw:: html

   <iframe
     src="_static/spatialhash_sphere.html"
     width="100%"
     height="700px"
     frameborder="0"
     style="border: none;"
   ></iframe>


Approximation schemes used in ``orpheus``
-----------------------------------------
Besides the exact discrete estimator, ``orpheus`` implements three tree-based approximation schemes to
accelerate the NPCF estimation, each of which might be suited for different usecases. As each of them uses
a variety of tuning parameters we include a widget that allows to visualize how the different
approximations of each scheme will look like for a given survey region.

The Discrete Estimator
~~~~~~~~~~~~~~~~~~~~~~
This uses no approximation whatsoever. It is already pretty efficient for smallish datasets and can
be used to benchmark the accuracy of the various other approximation schemes.

The Tree-Approximation
~~~~~~~~~~~~~~~~~~~~~~
In the tree approximation we aim to speed up the allocation of the :math:`X_n`. We still use every
tracer as a base point, but allocate the leaf points using a hierarchy of reduced catalogs; schematically 
we have 

.. math::

    X_{n}^{(\Delta_{\rm leaf})} \left( \Theta; \vec{\vartheta_i}\right) =
    \sum_{j=1}^{N_{\Delta_{\rm leaf}}} x^{(\Delta_{\rm leaf})}\left(\vec{\vartheta_j}\right) \ 
    \mathrm{e}^{\mathrm{i} n \varphi_{ij}} \ \mathcal{B}(\theta_{ij} \in \Theta) \ ,

while the allocation of the :math:`\mathscr{C}_{\mathbf{n}_{N-2}}` remains untouched. We choose the cell size
of the reduced catalog by the variable :math:`r_{\mathrm{min},\Delta}` which is defined as the ratio of the 
radius :math:`R` of a circle by the cell sidelength :math:`\Delta`. For our case of a hierarchy consisting of
resolutions :math:`\Delta_d \in \{0,\Delta,2\Delta,\cdots,2^{n_\mathrm{reso}-1}\Delta\}` and a radial binning 
scheme we fix the resolution at each bin to be the largest resolution :math:`\Delta' \in \Delta_d` for which 
:math:`\Theta_\mathrm{low}/\Delta' \geq r_{\mathrm{min},\Delta}`. 

The speedup of this method can primarily be tuned by setting the :math:`r_{\mathrm{min},\Delta}` parameter. 
Choosing a small value for :math:`r_{\mathrm{min},\Delta}`, however, limits the angular resolution
of the multiplet counts which might 'smear out' extreme multiplet configurations and hence yield biased
estimates for the :math:`N\mathrm{PCF}` of non-spin-0 tracers. Furthermore, the multiple-counting corrections
are only well-defined for the discrete catalog such that the diagonal elements of the :math:`\mathscr{C}_{\mathbf{n}_{N-2}}`
become less trustworthy.  


The BaseTree-Approximation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The tree approximation still visits every discrete tracer as a base point, which can dominate the runtime
for dense surveys. In the ``BaseTree`` scheme we therefore also draw the base points from the reduced
catalogs, using at each radial bin the same resolution :math:`\Delta'` that is applied to the corresponding
leaf points, so that the building blocks are now evaluated at the reduced base points
:math:`\vec{\vartheta^{(\Delta_{\rm leaf})}_i}`,

.. math::

    X_{n}^{(\Delta_{\rm leaf})} \left( \Theta; \vec{\vartheta^{(\Delta_{\rm leaf})}_i}\right) =
    \sum_{j=1}^{N_{\Delta_{\rm leaf}}} x^{(\Delta_{\rm leaf})}\left(\vec{\vartheta}_j\right) \
    \mathrm{e}^{\mathrm{i} n \varphi_{ij}} \ \mathcal{B}(\theta_{ij} \in \Theta) \ .

The multipole components :math:`\mathscr{C}_{\mathbf{n}_{N-2}}` are still allocated directly for
each (reduced) base point, i.e. without the recursion introduced below. This yields a further speedup at the
cost of the same loss of angular resolution as the tree approximation, now affecting the base points as well,
and is again controlled by the :math:`r_{\mathrm{min},\Delta}` parameter.


The DoubleTree-Approximation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``DoubleTree`` scheme starts from the same base allocation as the BaseTree approximation, i.e. both the
base and the leaf points are drawn from the reduced hierarchy,

.. math::

    X_{n}^{(\Delta_{\rm leaf})} \left( \Theta; \vec{\vartheta^{(\Delta_{\rm leaf})}_i}\right) =
    \sum_{j=1}^{N_{\Delta_{\rm leaf}}} x^{(\Delta_{\rm leaf})}\left(\vec{\vartheta}_j\right) \
    \mathrm{e}^{\mathrm{i} n \varphi_{ij}} \ \mathcal{B}(\theta_{ij} \in \Theta) \ .

On top of this it also speeds up the allocation of the :math:`\mathscr{C}_{\mathbf{n}_{N-2}}` themselves: instead
of allocating them directly for each base point, we build caches for the :math:`X_n` that are distributed across
the different regions of the hierarchy and use those to recursively allocate the multipole components, see eq.
(F.6) in P25 for the explicit recursion. Its overall accuracy is again steered by the
:math:`r_{\mathrm{min},\Delta}` parameter.

Whether the allocation of the :math:`X_n` or the allocation of the :math:`\mathscr{C}_{\mathbf{n}_{N-2}}` dominates depends
on both the survey itself (in particular the source density :math:`\overline{n}`) and the bin density of the NPCF. As a
heuristic we can define the ratio 
:math:`R\equiv\frac{\mathcal{O}(\mathrm{allocate } \, X_n)}{\mathcal{O}(\mathrm{allocate } \, \mathscr{C}_{\mathbf{n}_{N-2}})} \approx \frac{n_{\mathrm{max}} \, N_{\rm{gal,ap}}}{N_{\mathrm{bins}}} \approx \frac{\overline{n}_\mathrm{eff} \, \Theta^2}{n_{\mathrm{max}}^{N-3} \ n_\Theta^{N-1} n_\mathrm{tomo}^N}`
to determine which part dominates. As a rule of thumb, for a non-tomographic survey :math:`R > 1` while for a tomographic survey :math:`R < 1`. In the latter case we can then boost the accuracy by choosing a finer resolution
of the leaf cells as compared to the base. This is achieved by adjusting the ``resoshift_leafs`` and the ``maxresoind_leaf`` parameters.
The former sets an effective :math:`r_{\mathrm{min,leaf},\Delta} = 2^{-\mathrm{resoshift\_leafs}} \, r_{\mathrm{min},\Delta}`
while the latter further imposes a hard bound on the largest allowed leaf cellsize.

.. raw:: html

   <iframe
     src="_static/tree_widget.html"
     width="100%"
     height="1200px"
     frameborder="0"
     style="border: none;"
   ></iframe>

|
|

Dealing with data on the celestial sphere
-----------------------------------------
While the framework for the multiple decomposition is strictly only valid in the flat sky
approximation, we note that for current cosmic shear surveys the information content saturates
on scales on which this approximation is still valid.

For a few selected correlators (currently  ``NN``, ``GG``, ``GGG`` and ``NNNN``) the statistics can be 
computed directly on the sphere using the appropriate distances and projections. However, all 
correlators instead handle a spherical catalog through a patch decomposition: ``orpheus`` splits the 
catalog into patches, maps those to the flat sky, computes the statistics on each patch, and finally 
accumulates the result across all patches. To make sure to not miss multiplet counts across different 
patches we allow the patches to overlap such that all counts are accounted for. For a complete worked 
example we refer to the corresponding
:doc:`tutorial notebook <../notebooks/catalog_tutorial_patches>`.

