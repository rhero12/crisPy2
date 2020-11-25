Examples of the Data Visualisers
================================

The following contains examples of how to use the interactive data visualiser packaged with crisPy. These visualisers require `ipywidgets <https://ipywidgets.readthedocs.io/en/latest/>`_ which will be installed as a dependency for crisPy and then activated in Jupyter notebook using

.. code-block:: bash

   jupyter nbextension enable --py widgetsnbextension

or for Jupyter lab using

.. code-block:: bash

   pip install ipympl
   jupyter labextension install @jupyter-widgets/jupyterlab-manager

.. important:: To set up the widgets to work correctly in Jupyter lab will require ``node`` v14+.

.. note:: These visualisers **only** work with Jupyter notebooks.

The visualisers can be imported from the ``crispy.visualisation`` submodule:

.. jupyter-execute::

   from crispy.visualisation import *

Image Viewer
------------
The first visualiser is the ``ImageViewer`` which is used when only wanting to look at the images for a single instant in time. This takes as argument a string, a list of strings or one of the dataclasses to produce an interactive image visualisation tool. Below is an illustration of what the user interface looks like:

.. jupyter-execute::
   :hide-output:

   iv = ImageViewer(["../examples/2014/crisp_l2_20140906_152724_6563_r00447.fits", "../examples/2014/crisp_l2_20140906_152724_8542_r00447.fits"])

.. figure:: ../docs/images/imageviewer.png
   :align: center
   :figclass: align-center
   :width: 100%

In this example, we have chosen to look at observations of both Ca II 8542 and H :math:`\alpha`. There are two sliders to vary the wavelength in the images above along with the ability to save the current snapshot of the figure to a filename typed into the text box.

Spectral Viewer
---------------

.. jupyter-execute::
   :hide-output:

   sv = SpectralViewer(["../examples/2014/crisp_l2_20140906_152724_6563_r00447.fits", "../examples/2014/crisp_l2_20140906_152724_8542_r00447.fits"])

.. figure:: ../docs/images/spectralviewer.png
   :align: center
   :figclass: align-center
   :width: 100%