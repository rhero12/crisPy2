Data Structures
===============
The following describes the objects created for the different types of CRISP data that is used throughout the package.

Narrowband Data
---------------

The first type is the generic data object that can be used for a CRISP observation in a single spectral line. A filename or ObjDict can be passed to this class to initialise it.

.. autoclass:: crispy.crisp.CRISP
   :members:

Often, CRISP observations are taken in more than one spectral line. As such, the ``CRISPSequence`` class can be used to combine multiple spectral line observations. This is just a container which creates a list of ``CRISP`` objects which can be accessed individually or as a group (under the assumption that they all share the same world coordinate system).

.. autoclass:: crispy.crisp.CRISPSequence
   :members:

The spacing between the wavelength positions of CRISP's Fabry-Pérot interferometer is variable. This means that the line core can be more densely sampled than the line wings and vice versa. As a result, the wavelength axis cannot be represented by the world coordinate system (however, the spatial coordinates **are** described by the world coordinate system). To combat this, use the ``CRISPNonU`` class which will not assume that the wavelength axis in the header information is throwaway and look for specific wavelength positions elsewhere in the file (e.g. for a fits file, the code looks into the 1st non-PrimaryHDU for these wavelength positions, if this is not common conventions please let me know). This leads to an attribute containing these wavelengths known as ``wvls`` but the class method ``wave`` will still work perfectly fine.

``CRISPNonU`` works identically to the ``CRISP`` class in terms of class methods and interface with other parts of this package.

Similarly to ``CRISPSequence``, there is a ``CRISPNonUSequence`` class which again works identically but has the subtle difference of the wavelengths not being taken from the world coordinate system.

.. autoclass:: crispy.crisp.CRISPNonU
   :members:

.. autoclass:: crispy.crisp.CRISPNonUSequence
   :members:

Broadband Data
--------------

As well as narrowband observations, CRISP also images using a broadband filter for continuum images to be used as context images. The ``CRISPWideband`` class can be used as a container for such images with ``CRISPWidebandSequence`` being available for a time series of these images.

.. autoclass:: crispy.crisp.CRISPWideband
   :members:

.. autoclass:: crispy.crisp.CRISPWidebandSequence
   :members: