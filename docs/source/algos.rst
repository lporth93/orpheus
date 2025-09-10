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

In `orpheus` we make use of the multipole decomposion of the :math:`N\mathrm{PCF}` that has been developed
by Chen & Szapudi (2005, ApJ, 635, 743), Slepian & Eisenstein (2015, MNRAS, 454, 4142), 
and Philcox et al (2022, MNRAS, 509, 2457). In particular, we also include the expressions for the 
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
correlators containing non-spin-0 objects. One can invert this relation to obtain a expression
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
linear combination of the multipole components :math:`n_k, \ k\in\{2,\cdots,N-1\}`. In particular,
we see that this form allows for an estimation of :math:`\mathscr{C}_{\mathbf{n}_{N-2}}` consisting
of two steps

::

    multipoles = initialise_multipoles()
    for every tracer in tracers:
        nextXn = allocate_Xn(tracer)                   # scales as O(N_{\mathrm{tracers}})
        update_multipoles(multipoles, tracer,nextXn)   # scales as O(N_{\mathrm{bins}})

Looking at the two different steps we see that the estimator has a time complexity of 
:math:`\mathcal{O}(N_{\mathrm{tracers}}^2)+\mathcal{O}(N_{\mathrm{tracers}} \, N_{\mathrm{bins}})`. While this scaling is
much more beneficial than for brute-force estimators, it can nevertheless become computationally
impractible. In the next subsection we show how ``orpheus`` further reduces this scaling.


Hierarchical spatial hashing
----------------------------
The core pair-finding algorithm in ``orpheus`` is built on spatial hashing. For our implementation we assume
the data to be distributed on a two-dimensional plane which we divide into a grid of fixed-size cells. We use 
a hash function to map the 2D-coordinates (x,y) to a cell index and then store references to the objects inside 
the cell they occupy. By constructing a hierarchy of such grids-cells with fixed bounds and an increase the sidelength of each cell by 
powers of two we can further build connections between the objects residing in the hash cells of the various
resolutions. 

<Insert Figure here>

In ``orpheus`` we parametrize the pixelsize by the variable :math:`r_{\mathrm{min},\Delta}` which is defined as the 
ratio of the radius :math:`R` of a circle by the pixel sidelength :math:`\Delta`. In case of a hierarchy
consisting of resolutions :math:`\Delta_d \in \{0,\Delta,2\Delta,\cdots,2^{n_\mathrm{reso}-1}\Delta\}` and a radial binning
scheme we fix the resolution at each bin to be the largest resolution :math:`\Delta' \in \Delta_d` for which :math:`\Theta_\mathrm{low}/\Delta' \geq r_{\mathrm{min},\Delta}`.

The Tree-Approximation
~~~~~~~~~~~~~~~~~~~~~~
In the tree approximation we 


The BaseTree-Approximation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The DoubleTree-Approximation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~




