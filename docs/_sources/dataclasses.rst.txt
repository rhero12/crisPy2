Examples Using the Main Data Wrappers from crispy
=================================================

Here, we provide a brief introduction on how to use the main data structures defined in crispy.crisp. Specifically, these examples will cover the CRISP, CRISPNonU, and CRISPWideband structures.

.. jupyter-execute::

   from crispy.crisp import CRISP, CRISPWideband, CRISPNonU

CRISP
-----

To demonstrate the CRISP class we use an example from the M1.1 solar flare SOL20140906T17:09 observed by SST/CRISP in H :math:`\alpha` and Ca II 8542. The data is publicly available through the F-CHROMA project `here <https://star.pst.qub.ac.uk/wiki/doku.php/public/solarflares/start>`_. Below we show the data for H :math:`\alpha`.

.. jupyter-execute::

   crisp = CRISP("../examples/2014/crisp_l2_20140906_152724_6563_r00447.fits")
   print(crisp)

The data structures use slicing derived from astropy's `NDData <https://docs.astropy.org/en/stable/nddata/>`_ meaning the objects can be directly sliced to produce new objects of just the slice e.g. for only looking at the far blue wing data in our observation

.. jupyter-execute::

   print(crisp[0])

This is also the easiest way to use the plotting methods. For this observation, to view an image at a specific wavelength we would use the ``intensity_map`` class method as shown below for the line core of the H :math:`\alpha` observation

.. jupyter-execute::
   :hide-output:

   crisp[7].intensity_map()

.. figure:: ../docs/images/CRISP.png
   :align: center
   :figclass: align-center
   :width: 100%

We can than use slicing and the ``plot_spectrum`` class method to plot the spectrum at a certain point. Firstly though to identify the slice we need we use the ``from_lonlat`` class method, in this example we take the point (-720, -310) in the Helioprojective plane:

.. jupyter-execute::

    crisp.from_lonlat(-720, -310)

.. jupyter-execute::
   :hide-output:

   crisp[:,759,912].plot_spectrum()

.. figure:: ../docs/images/CRISP_Spectrum.png
   :align: center
   :figclass: align-center
   :width: 100%

CRISPNonU
---------

For the CRISPNonU class, we choose an imaging spectropolarimetric Ca II 8542 observation of the X2.2 solar flare SOL20170906T09:10.

.. jupyter-execute::

   crispnonu = CRISPNonU("../examples/2017/ca8542/00000.h5")
   print(crispnonu)

The ``intensity_map`` and ``plot_spectrum`` methods will also work here with the correct slicing. The main difference from the CRISPNonU class can be seen in the wavelengths sampled section: the wavelengths are sampled non-uniformly but the class deals with this for us. Here, we will show the polarimetric class methods. Firstly is ``stokes_map`` for the line core:

.. jupyter-execute::
   :hide-output:

   crispnonu[:,5].stokes_map(stokes="all")

.. figure:: ../docs/images/CRISPNonU.png
   :align: center
   :figclass: align-center
   :width: 100%

We can then identify a point to view the Stokes profiles for using the ``from_lonlat`` as before and plot the Stokes profiles using the ``plot_stokes`` class method:

.. jupyter-execute::
   :hide-output:

   crispnonu[:,:,38,257].plot_stokes(stokes="all")

.. figure:: ../docs/images/CRISPNonU_Spectrum.png
   :align: center
   :figclass: align-center
   :width: 100%

CRISPWideband
-------------

For the CRISPWideband class, we use the complimentary wideband Ca II 8542 for the observation shown as an example for the CRISPNonU class, above.

.. jupyter-execute::

   crispwideband = CRISPWideband("../examples/2017/ca8542/wideband/00000.h5")
   print(crispwideband)

The CRISPWideband has one useful class method, that is ``intensity_map``:

.. jupyter-execute::
   :hide-output:

   crispwideband.intensity_map()

.. figure:: ../docs/images/CRISPWideband.png
   :align: center
   :figclass: align-center
   :width: 100%