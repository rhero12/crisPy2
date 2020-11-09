Examples of the Sequence Classes
================================

Along with the data structures for single CRISP observations we also introduce sequence data structures that allow us to use one object for many CRISP observations. This is very simply done by the structure essentially being a fancy list of the normal objects, depending which structure is used.

There are three sequence classes: ``CRISPSequence``, ``CRISPNonUSequence``, and ``CRISPWidebandSequence`` to be used depending on the observations you have. The motivation behind these structures is that you will often find there is more than one spectral line observation for a given time e.g. H :math:`\alpha` and Ca II 8542, or we may want to group chronological observations together into one structure. Both can be accomplished using the sequences as shown below.

.. note:: To construct the sequence object, we need a list of the keyword arguments for the underlying data structures as the argument. To avoid confusion and messy code, we provide the ``crispy2.utils.CRISP_sequence_constructor`` function to create this for the user.

.. jupyter-execute::

   from crispy.crisp import CRISPSequence, CRISPNonUSequence, CRISPWidebandSequence
   from crispy.utils import CRISP_sequence_constructor

CRISPSequence
-------------

Again, for the ``CRISPSequence``, we use an example from the `F-CHROMA database <https://star.pst.qub.ac.uk/wiki/doku.php/public/solarflares/start>`_ of the M1.1 solar flare SOL20140906T17:09 to demonstrate its functionality.

.. jupyter-execute::

   crisps = CRISPSequence(CRISP_sequence_constructor(["../examples/2014/crisp_l2_20140906_152724_8542_r00447.fits","../examples/2014/crisp_l2_20140906_152724_6563_r00447.fits"]))
   print(crisps)

Each observation can be accessed through the ``.list`` class attribute e.g.

.. jupyter-execute::

   print(crisps.list[0])

The plotting methods from before also exist for the sequence methods which will just loop of the ``.list`` attribute and call the native class methods for each. For combined plots of the data please see the examples and API documentation for the viewers.

CRISPNonUSequence
-----------------

Similarly, for the ``CRISPNonUSequence`` we use the example from the X2.2 solar flare SOL20170906T09:10:

.. note:: When using ``CRISP_sequence_constructor`` for the non-uniformly sampled spectra set the keyword argument ``nonu = True`` in the function.

.. jupyter-execute::

   crispsnonu = CRISPNonUSequence(CRISP_sequence_constructor(["../examples/2017/ca8542/00000.h5", "../examples/2017/Halpha/00000.h5"], nonu=True))
   print(crispsnonu)

CRISPWidebandSequence
---------------------

The ``CRISPWidebandSequence`` is good for grouping together a time sequence of wideband context images. We again use the wideband context images from the X2.2 solar flare SOL20170906T09:10

.. jupyter-execute::

   crispswideband = CRISPWidebandSequence(CRISP_sequence_constructor(["../examples/2017/ca8542/wideband/00000.h5", "../examples/2017/ca8542/wideband/00002.h5"]))
   print(crispswideband)