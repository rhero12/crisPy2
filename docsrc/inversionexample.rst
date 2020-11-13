Example of Inversion Data in crispy
===================================

The data structure for inversion data in ``crispy`` is built to work with inversions from the `RADYNVERSION <https://iopscience.iop.org/article/10.3847/1538-4357/ab07b4>`_ code, specificaly whole image inversions (WII). The code can be easily inherited to work for a variety of inversion codes, however. The main things we need is the inversion itself, the corresponding observation, and the height grid that the inversions are calculated on which is stored in the ``crispy2.Radynversion`` submodule.

.. note:: The ``Inversion`` objects can also be indexed in a similar manner to the other data structures to making plotting easier and working with the data simpler.

.. jupyter-execute::

   from crispy.inversions import Inversion
   from crispy.Radynversion.utils import z
   from crispy.crisp import CRISP

   crisp = CRISP("../examples/2014/crisp_l2_20140906_152724_8542_r00447.fits")
   inversion = Inversion("../examples/inversions/inversion_0450.hdf5", z=z, header=crisp.file.header)
   print(inversion)

Now that the object exists we can use the built-in plotting methods to first show the line-of-sight fluid velocity at a height roughly corresponding to the formation height of the core of both H :math:`\alpha` and Ca II 8542

.. jupyter-execute::
   :hide-output:

   inversion[15].vel_map()

.. figure:: ../docs/images/InversionVel.png
   :align: center
   :figclass: align-center
   :width: 100%

Next we look at the line-of-sight velocity for a specific point on the eastern flare ribbon at (-755,-330)

.. jupyter-execute::
   :hide-output:

   inversion.from_lonlat(-755,-330)

.. jupyter-execute::
   :hide-output:

   inversion[:,408,298].plot_vel()

.. figure:: ../docs/images/InversionVel_Height.png
   :align: center
   :figclass: align-center
   :width: 100%