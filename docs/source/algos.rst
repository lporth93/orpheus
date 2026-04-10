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
The core pair-finding algorithm in ``orpheus`` is built on spatial hashing. For our implementation we 
assume the data to be distributed on a two-dimensional plane, which we divide into a grid of fixed-size 
cells. We use a hash function to map the 2D-coordinates (x,y) to a cell index and then store references 
to the objects inside the cell they occupy. By constructing a hierarchy of such grid cells with fixed 
bounds and increasing the sidelength of each cell by powers of two, we can further build connections 
between the objects residing in the hash cells of the various resolutions. 

.. TODO: Insert figure here.

In ``orpheus`` we parametrize the pixelsize by the variable :math:`r_{\mathrm{min},\Delta}` which is defined as the 
ratio of the radius :math:`R` of a circle by the pixel sidelength :math:`\Delta`. In case of a hierarchy
consisting of resolutions :math:`\Delta_d \in \{0,\Delta,2\Delta,\cdots,2^{n_\mathrm{reso}-1}\Delta\}` and a radial binning
scheme we fix the resolution at each bin to be the largest resolution :math:`\Delta' \in \Delta_d` for which :math:`\Theta_\mathrm{low}/\Delta' \geq r_{\mathrm{min},\Delta}`.
Mapping the catalog on such a hierarchy of grids we create a corresponding hierarchy of `reduced` catalogs.


Approximation schemes used in ``orpheus``
-----------------------------------------
There are three different schemes implemented in ``orpheus``, each of which might be 
suited for different usecases. As each of them use a variety of tuning parameters we 
include a widget that allows to visualize how the different approximations of each 
scheme will look like for a given survey region.

The Discrete Estimator
~~~~~~~~~~~~~~~~~~~~~~
This uses no approximation whatsoever. It is already pretty efficient for smallish datasets and can
be used to benchmark the accuracy of the various other approximation schemes.

The Tree-Approximation
~~~~~~~~~~~~~~~~~~~~~~
In the tree approximation we aim to speed up the allocation of the :math:`X_n`. We still use every
tracer as a base point, but allocate the leaf points using the hierarchy of reduced catalogs; schematically 
we have 

.. math::

    X_{n}^{(\Delta_{\rm leaf})} \left( \Theta; \vec{\vartheta_i}\right) =
    \sum_{j=1}^{N_{\Delta_{\rm leaf}}} x^{(\Delta_{\rm leaf})}\left(\vec{\vartheta_j}\right) \ 
    \mathrm{e}^{\mathrm{i} n \varphi_{ij}} \ \mathcal{B}(\theta_{ij} \in \Theta) \ ,

while the allocation of the :math:`\mathscr{C}_{\mathbf{n}_{N-2}}` remains untouched. The speedup
of this method can primarily be tuned by setting the :math:`r_{\mathrm{min},\Delta}` parameter. 
Choosing a small value for :math:`r_{\mathrm{min},\Delta}`, however, limits the angular resolution
of the multiplet counts which might 'smear out' extreme multiplet configurations and hence yield biased
estimates for the :math:`N\mathrm{PCF}` of non-spin-0 tracers. Furthermore, the multiple-counting corrections
are only well-defined for the discrete catalog such that the diagonal elements of the :math:`\mathscr{C}_{\mathbf{n}_{N-2}}`
become less trustworthy.  


The DoubleTree-Approximation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To further speed up the allocation of the :math:`\mathscr{C}_{\mathbf{n}_{N-2}}` we can utilise a similar hierachy
for the tracers sitting at the multiplets base. In a first step, we allocate the :math:`X_n` similarly to the 
Tree-approximation, but use the tracers from the hierarchy of the corresponding resolution:

.. math::

    X_{n}^{(\Delta_{\rm leaf})} \left( \Theta; \vec{\vartheta^{(\Delta_{\rm leaf})}_i}\right) =
    \sum_{j=1}^{N_{\Delta_{\rm leaf}}} x^{(\Delta_{\rm leaf})}\left(\vec{\vartheta}_j\right) \ 
    \mathrm{e}^{\mathrm{i} n \varphi_{ij}} \ \mathcal{B}(\theta_{ij} \in \Theta) \ .

We then build caches for the :math:`X_n` that are distributed across the different regions within the hierarchy use 
those to recusively allocate the :math:`\mathscr{C}_{\mathbf{n}_{N-2}}`, see eq (F.6) in P25 for the explicit recursion.
The general accuracy of this method is steered again by the :math:`r_{\mathrm{min},\Delta}` parameter. 

Wheter the allocation of the :math:`X_n` or the allocation of the :math:`\mathscr{C}_{\mathbf{n}_{N-2}}` dominates depends
on both, the survey itself (in particular the source density :math:`\overline{n}`) or the bin density of the NPCF. As a
heuristic we can define the ratio 
:math:`R\equiv\frac{\mathcal{O}(\mathrm{allocate } \, X_n)}{\mathcal{O}(\mathrm{allocate } \, \mathscr{C}_{\mathbf{n}_{N-2}})} \approx \frac{n_{\mathrm{max}} \, N_{\rm{gal,ap}}}{N_{\mathrm{bins}}} \approx \frac{\overline{n}_\mathrm{eff} \, \Theta^2}{n_{\mathrm{max}}^{N-3} \ n_\Theta^{N-1} n_\mathrm{tomo}^N}`
to determine which part dominates. As a rule of thumb, for a non-tomographic survey :math:`R > 1` while for a tomographic survey :math:`R < 1`. In the latter case we can then boost the accuracy by choosing a finer resolution
of the leaf cells as compared to the base. This is achieved by adjusting the ``resoshift_leaf`` and the ``maxresoind_leaf`` parameters.
The former sets an effective :math:`r_{\mathrm{min,leaf},\Delta} = 2^{-\mathrm{resoshift\_leaf}} \, r_{\mathrm{min},\Delta}` 
while the latter further imposes a hard bound on the largest allowed leaf cellsize.

.. raw:: html

   <iframe
     src="_static/tree_widget.html"
     width="100%"
     height="900px"
     frameborder="0"
     style="border: none;"
   ></iframe>

|
|

Dealing with data on the celestial sphere
-----------------------------------------
While the framework for the multiple decomposition is strictly only valid in the flat sky
approximation, we note that for current cosmic shear surveys the information content saturates
on scales on which this approximation is still valid. In case the catalog is given on the
celestial sphere, ``orpheus`` first decomposes the catalog into patches, maps those to the
flat sky, computes the statistics on each patch, and finally accumulates the result across all
patches. To make sure to not miss multiplet counts across different patches we allow the patches
to overlap such that all counts are accounted for. For a complete worked example we refer the
corresponding :doc:`tutorial notebook <../notebooks/catalog_tutorial_patches>`.

