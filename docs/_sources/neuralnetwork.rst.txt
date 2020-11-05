Neural Network Building Blocks
==============================

.. note:: These neural network layers are built using the `PyTorch <https://pytorch.org/>`_ framework and are only compatible with v1.1.0+.

Neural networks have seen a rise in popularity in solar physics (see `Armstrong & Fletcher, 2019 <https://link.springer.com/article/10.1007/s11207-019-1473-z>`_ and `Armstrong et al., 2020 <https://doi.org/10.1093/astrogeo/ataa044>`_ for an overview), however, the application of such models can be a tricky subject to get into. As a result, the following class should aid anyone wanting to build a deep neural network by taking away a lot of the complexity involved (such as structuring).

Neural networks are composed of non-linear *layers* made up of a linear operation, a normalisation technique and an non-linear (known as activation) operation. Like Lego, neural network frameworks typically give each of these pieces individually and a mix and match process can ensue. The approach we take however is that these layers will always follow the linear -> normalisation -> non-linear path and therefore we combine these key building blocks together with the interchangeability being encapsulated by keyword arguments.

The most commonly used type of neural network is known as a convolutional neural network (CNN). CNNs have layers as described above and so will utilise the ``ConvBlock`` object to build the layers.

.. autoclass:: crispy.neural_network.ConvBlock

Another popular type of neural network is known as a Residual Network. Residual networks differ from CNNs because rather than each layer learning some function :math:`f`, residual layers learn the residual to that function :math:`H` such that

.. math::
   H(x) = f(x) - x

where :math:`x` is the input to the layer. Residual networks are typically used when the problem requires are very deep network to learn or if a model encounters the vanishing gradient problem (for more on residual layers check out `He et al., 2015 <https://arxiv.org/abs/1512.03385>`_).

The inner structure of residual layers is also different from a normal convolutional layer as they contain two convolutions, two normalisations and two activations as shown in the figure below. The input to the layer is added after the second normalisation and before the second activation as shown by the arrow in the figure below.

.. figure:: ../docs/images/resblock.png
   :align: center
   :figclass: align-center
   :width: 100%

   A schematic diagram of a residual layer.

.. autoclass:: crispy.neural_network.ResBlock

In popular network architectures such as Autoencoders (AEs) or Variational Autoencoders (VAEs), the data is often downsampled to a latent representation of the data to be upsampled to the desired result. There are typically two approaches to this upsampling.
 1. Using a fixed interpolation by the desired scale factor which can be achieved by setting ``upsample=True`` in ``ConvBlock``/``ResBlock``
 2. Using *transpose* convolution which is a kind of learned upsampling. (For a better explanation of transpose convolution than I could provide see `here <https://medium.com/apache-mxnet/transposed-convolutions-explained-with-ms-excel-52d13030c7e8>`_). Transpose convolution is an interpolation using a sparse convolutional kernel where the non-zero numbers in that kernel are learnable parameters. This means that in training a network, the model should learn to do the optimal upsampling for the reconstruction of the data.

 .. autoclass:: crispy.neural_network.ConvTransBlock