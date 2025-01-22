import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from mmcv.cnn import ConvModule
from mmseg.ops import resize


def build_ms2one(config):
    config = copy.deepcopy(config)
    t = config.pop('type')
    if t == 'Naive':
        return Naive(**config)
    elif t == 'DilateNaive':
        return DilateNaive(**config)


class Naive(nn.Module):
    """_summary_
    Naive Class:
    Purpose: This class implements a basic convolutional layer with 1x1 kernel size.
    It is designed to operate on multi-scale feature maps.
    Attributes:
    inc: Number of input channels.
    outc: Number of output channels.
    kernel_size: Size of the convolutional kernel (default is 1).
    Methods:
    __init__: Initializes the convolutional layer.
    forward: Defines the forward pass of the layer, which involves interpolating and 
    concatenating feature maps before passing them through the convolutional layer.
    """
    def __init__(self, inc, outc, kernel_size=1):
        super().__init__()
        self.layer = nn.Conv2d(inc, outc, kernel_size=1)

    def forward(self, ms_feats):
        out = self.layer(torch.cat([
            F.interpolate(tmp, ms_feats[0].shape[-2:],
                          mode='bilinear') for tmp in ms_feats], dim=1))
        return out


class DilateNaive(nn.Module):
    """_summary_
    DilateNaive Class:
    Purpose: This class implements a network with dilated convolutions, allowing for an increased receptive field without downsampling the input.
    Attributes:
    inc: Number of input channels for each scale
    outc: Number of output channels.
    num_scales: Number of scales (default is 4).
    dilations: A tuple specifying the dilation rates for each scale.
    merge: Whether to merge the outputs from different scales (default is True).
    fpn: Whether to use feature pyramid network-like merging (default is False).
    target_shape: If specified, the target shape for the output.
    one_layer_before: Whether to apply one additional layer before dilated convolutions (default is False).
    Methods:
    __init__: Initializes the DilateNaive module.
    forward: Defines the forward pass, which involves applying dilated convolutions on each scale and optionally merging the results.
    """
    def __init__(self, inc, outc, num_scales=4,
                 dilations=(1, 2, 5, 9),
                 merge=True, fpn=False,
                 target_shape=None,
                 one_layer_before=False):
        
        super().__init__()

        # [1, 2, 5 , 9]
        self.dilations = dilations
        self.num_scales = num_scales

        if not isinstance(inc, (tuple, list)):
            inc = [inc for _ in range(num_scales)]

        self.inc = inc
        self.outc = outc
        self.merge = merge
        self.fpn = fpn
        self.target_shape = target_shape
        self.layers = nn.ModuleList()

        """
        # Implementation of a neural network module using dilated convolutions
        # to capture spatial features at different scales. The model consists
        # of multiple scales, each containing dilated convolution layers with
        # varying dilation rates. The `num_scales` parameter determines the
        # number of scales. Each scale may have multiple dilated convolution
        # layers (`j` iterations), and the dilation rates decrease with depth.
        # The chosen dilation values (1, 2, 5, 9) increase gradually, enhancing
        # the model's ability to recognize features at different spatial scales.
        # If `one_layer_before` is True, an additional convolution layer with
        # batch normalization and ReLU activation is applied before dilated
        # convolutions to reduce input size. The model can merge outputs from
        # different scales (`merge=True`). If `fpn` is also True, feature pyramid
        # network-like merging is employed. The `target_shape` parameter can be
        # used to specify the desired output shape. The final layer, if merging is
        # enabled, combines the scaled outputs. Overall, this architecture enables
        # the network to efficiently capture spatial information across multiple
        # scales without downsampling the input.
        """

        for i in range(num_scales):
            layers = []
            if one_layer_before:
                layers.extend([
                    nn.Conv2d(inc[i], outc, kernel_size=1, bias=False),
                    nn.BatchNorm2d(outc),
                    nn.ReLU(True)
                ])

            for j in range(len(dilations[:-i])):
                d = dilations[j]
                layers.append(nn.Sequential(
                    nn.Conv2d(inc[i] if j == 0 and not one_layer_before else outc, outc,
                              kernel_size=1 if d == 1 else 3,
                              stride=1,
                              padding=0 if d == 1 else d,
                              dilation=d,
                              bias=False),
                    nn.BatchNorm2d(outc),
                    nn.ReLU(True)))
                
            self.layers.append(nn.Sequential(*layers))

        if self.merge:
            self.final_layer = nn.Sequential(
                nn.Conv2d(outc, outc, 3, 1, padding=1, bias=False),
                nn.BatchNorm2d(outc),
                nn.ReLU(True),
                nn.Conv2d(outc, outc, 1))

    def forward(self, x):
        outs = []

        for i in range(self.num_scales - 1, -1, -1):
            if self.fpn and i < self.num_scales - 1:
                tmp = self.layers[i](x[i] + F.interpolate(
                    x[i + 1], x[i].shape[2:],
                    mode='bilinear', align_corners=True))
            else:
                tmp = self.layers[i](x[i])

            if self.target_shape is None:
                if i > 0 and self.merge:
                    tmp = F.interpolate(tmp, x[0].shape[2:],
                        mode='bilinear', align_corners=True)
            else:
                tmp = F.interpolate(tmp, self.target_shape,
                        mode='bilinear', align_corners=True)
            outs.append(tmp)
        if self.merge:
            out = torch.sum(torch.stack(outs, dim=-1), dim=-1)
            out = self.final_layer(out)
            
            return out
        else:
            return outs
