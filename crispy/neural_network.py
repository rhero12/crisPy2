import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    A modifiable convolutional layer for deep networks.

    Parameters
    ----------
    in_channels : int
        The number of channels fed into the convolutional layer.
    out_channels : int
        The number of channels fed out of the convolutional layer.
    kernel : int, optional
        The size of the convolutional kernel. Default is 3 e.g. 3x3 convolutional kernel.
    stride : int, optional
        The stride of the convolution. Default is 1.
    pad : str, optional
        The type of padding to use when calculating the convolution. Default is "reflect".
    bias : bool, optional
        Whether or not to include a bias in the linear transformation. Default is False.
    normal : str, optional
        The type of normalisation layer to use. Default is "batch".
    activation : str, optional
        The activation function to use. Default is "relu" to use the Rectified Linear Unit (ReLU) activation function.
    upsample : bool, optional
        Whether or not to upsample the input to the layer. This is useful in decoder layers in autoencoders. Upsampling is done via a factor of 2 interpolation (it is only currently implemented assuming the size of the input is to be doubled, will be retconned to work for me if there is demand). Default is False.
    """

    def __init__(self, in_channels, out_channels, kernel=3, stride=1, pad="reflect", bias=False, normal="batch", activation="relu", upsample=False, **kwargs):
        super(ConvBlock, self).__init__()

        self.upsample = upsample
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, bias=bias, padding=(kernel-1)//2, padding_mode=pad, **kwargs)
        if normal == "batch":
            self.norm = nn.BatchNorm2d(out_channels)
        elif normal == "instance":
            self.norm = nn.InstanceNorm2d(out_channels)
        elif normal == None:
            self.norm = None
        if activation.lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            raise NotImplementedError("It'll be there soon.")

        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0.01)

    def forward(self, inp):
        if self.upsample:
            inp = F.interpolate(inp, scale_factor=2)
        out = self.conv(inp)
        if self.norm is not None:
            out = self.norm(out)
        out = self.act(out)

        return out

class ConvTransBlock(nn.Module):
    """
    A modifiable transpose conovlutional layer.

    Parameters
    ----------
    in_channels : int
        The number of channels fed into the convolutional layer.
    out_channels : int
        The number of channels fed out of the convolutional layer.
    kernel : int, optional
        The size of the convolutional kernel. Default is 3 e.g. 3x3 convolutional kernel.
    stride : int, optional
        The stride of the convolution. Default is 1.
    pad : str, optional
        The type of padding to use when calculating the convolution. Default is "reflect".
    bias : bool, optional
        Whether or not to include a bias in the linear transformation. Default is False.
    normal : str, optional
        The type of normalisation layer to use. Default is "batch".
    activation : str, optional
        The activation function to use. Default is "relu" to use the Rectified Linear Unit (ReLU) activation function.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=3,
        stride=1,
        bias=False,
        pad="reflect",
        normal="batch",
        activation="relu",
        **kwargs
    ):
        super(ConvTransBlock, self).__init__()

        self.convtrans = nn.ConvTranspose2d(
            in_channles,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            bias=bias,
            pad=(kernel-1)//2,
            padding_mode=pad,
            output_padding=(kernel-1)//2,
            **kwargs
            )
        if normal == "batch":
            self.norm = nn.BatchNorm2d(out_channels)
        elif normal == "instance":
            self.norm = nn.InstanceNorm2d(out_channels)
        elif normal == None:
            self.norm = None

        if activation.lower() == "relu":
            self.act = nn.ReLU()
        else:
            raise NotImplementedError("Soon....")

        nn.init.kaiming_normal_(self.convtrans.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0.01)

    def forward(self, inp):
        out = self.convtrans(inp)
        if self.norm != None:
            out = self.norm(out)
        out = self.act(out)

        return out

class ResBlock(nn.Module):
    """
    A modifiable residual block for deep neural networks.

    Parameters
    ----------
    in_channels : int
        The number of channels fed into the residual layer.
    out_channels : int
        The number of channels fed out of the residual layer.
    kernel : int, optional
        The size of the convolutional kernel. Default is 3 e.g. 3x3 convolutional kernel.
    stride : int, optional
        The stride of the convolution. Default is 1.
    pad : str, optional
        The type of padding to use when calculating the convolution. Default is "reflect".
    bias : bool, optional
        Whether or not to include a bias in the linear transformation. Defulat is False.
    normal : str, optional
        The type of normalisation layer to use. Default is "batch".
    activation : str, optional
        The activation function to use. Default is "relu" to use the Rectified Linear Unit (ReLU) activation function.
    upsample : bool, optional
        Whether or not to upsample the input to the layer. This is useful in decoder layers in autoencoders. Upsampling is done via a factor of 2 interpolation (it is only currently implemented assuming the size of the input is to be doubled, will be retconned to work for me if there is demand). Default is False.
    """

    def __init__(self, in_channels, out_channels, kernel=3, stride=1, pad="reflect", bias=False, normal="batch", activation="relu", upsample=False, use_dropout=False):
        super(ResBlock, self).__init__()

        self.upsample = upsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, bias=bias, padding=(kernel-1)//2, padding_mode=pad)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=1, bias=bias, padding=(kernel-1)//2, padding_mode=pad)
        if normal == "batch":
            self.norm1 = nn.BatchNorm2d(out_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        elif normal == "instance":
            self.norm1 = nn.InstanceNorm2d(out_channels)
            self.norm2 = nn.InstanceNorm2d(out_channels)
        elif normal == None:
            self.norm1 = None
            self.norm2 = None
        if activation.lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            raise NotImplementedError("Soon....")

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        if bias:
            nn.init.constant_(self.conv1.bias, 0.01)
            nn.init.constant_(self.conv2.bias, 0.01)

        if in_channels != out_channels and not upsample:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        elif in_channels != out_channels and upsample:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        else:
            self.downsample = None
            
        if use_dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = None

    def forward(self, inp):
        identity = inp.clone()

        if self.upsample:
            identity = F.interpolate(identity, scale_factor=2)
            inp = F.interpolate(inp, scale_factor=2)

        out = self.conv1(inp)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = self.act(out)
        
        if self.dropout:
            out = self.dropout(out)

        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.act(out)

        return out