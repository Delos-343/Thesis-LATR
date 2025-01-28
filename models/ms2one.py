import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from mmcv.cnn import ConvModule
from mmseg.ops import resize


def build_ms2one(config):

    """

        # Implementation of a neural network module using dilated convolutions to capture spatial features at different scales.
        # The model consists of multiple scales, each containing dilated convolution layers with varying dilation rates.
        # The `num_scales` parameter determines the number of scales.
        # Each scale may have multiple dilated convolution layers (`j` iterations), and the dilation rates decrease with depth.
        # The chosen dilation values (1, 2, 5, 9) increase gradually, enhancing the model's ability to recognize features at different spatial scales.
        # If `one_layer_before` is True, an additional convolution layer with batch normalization and ReLU activation is applied before dilated convolutions to reduce input size.
        # The model can merge outputs from different scales (`merge=True`).
        # If `fpn` is also True, feature pyramid network-like merging is employed.
        # The `target_shape` parameter can be used to specify the desired output shape.
        # The final layer, if merging is enabled, combines the scaled outputs.
        # Overall, this architecture enables the network to efficiently capture spatial information across multiple scales without downsampling the input.

    """

    """Builds the ms2one module based on the provided configuration."""
    config = copy.deepcopy(config)
    module_type = config.pop('type')

    if module_type == 'Naive':
        return Naive(**config)
    elif module_type == 'DilateNaive':
        return DilateNaive(**config)
    else:
        raise ValueError(f"Unknown ms2one type: {module_type}")


class Naive(nn.Module):

    """
    _summary_

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

    """Simple feature aggregation using a single 1x1 convolution."""
    def __init__(self, inc, outc, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size)

    def forward(self, ms_feats):
        """Aggregates multi-scale features into a single tensor."""
        interpolated_feats = [
            F.interpolate(feat, size=ms_feats[0].shape[-2:], mode='bilinear', align_corners=False)
            for feat in ms_feats
        ]
        concatenated_feats = torch.cat(interpolated_feats, dim=1)
        return self.conv(concatenated_feats)


class DilateNaive(nn.Module):

    """
    _summary_

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

    """Multi-scale feature processing with dilated convolutions."""
    def __init__(self, inc, outc, num_scales=4, dilations=(1, 2, 5, 9),
                 merge=True, fpn=False, target_shape=None, one_layer_before=False):
        super().__init__()
        
        self.dilations = dilations
        self.num_scales = num_scales
        self.merge = merge
        self.fpn = fpn
        self.target_shape = target_shape

        # Ensure `inc` is a list of correct length
        if not isinstance(inc, (tuple, list)):
            inc = [inc] * num_scales
        self.inc = inc
        self.outc = outc

        # Build convolutional layers for each scale
        self.layers = nn.ModuleList([
            self._build_dilated_layers(inc[i], outc, dilations, one_layer_before)
            for i in range(num_scales)
        ])

        # Final merging layer
        if self.merge:
            self.final_layer = nn.Sequential(
                nn.Conv2d(outc, outc, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(outc),
                nn.ReLU(inplace=True),
                nn.Conv2d(outc, outc, kernel_size=1)
            )

    def _build_dilated_layers(self, inc, outc, dilations, one_layer_before):
        """Creates a sequential block of dilated convolutional layers."""
        layers = []

        # Initial layer (if one_layer_before is True)
        if one_layer_before:
            layers += [
                nn.Conv2d(inc, outc, kernel_size=1, bias=False),
                nn.BatchNorm2d(outc),
                nn.ReLU(inplace=True)
            ]
            inc = outc  # Ensure consistency for subsequent layers

        # Stacked dilated convolutions
        for dilation in dilations:
            layers.append(nn.Sequential(
                nn.Conv2d(inc, outc, kernel_size=1 if dilation == 1 else 3,
                          padding=0 if dilation == 1 else dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(outc),
                nn.ReLU(inplace=True)
            ))
            inc = outc  # Maintain consistent channel size

        return nn.Sequential(*layers)

    def forward(self, x):
        """Processes multi-scale feature maps and merges them if required."""
        outputs = []

        for i in reversed(range(self.num_scales)):
            if self.fpn and i < self.num_scales - 1:
                fused_input = x[i] + F.interpolate(
                    x[i + 1], size=x[i].shape[2:], mode='bilinear', align_corners=True)
            else:
                fused_input = x[i]

            processed_feat = self.layers[i](fused_input)

            if self.target_shape:
                processed_feat = F.interpolate(processed_feat, size=self.target_shape,
                                               mode='bilinear', align_corners=True)
            elif self.merge and i > 0:
                processed_feat = F.interpolate(processed_feat, size=x[0].shape[2:],
                                               mode='bilinear', align_corners=True)

            outputs.append(processed_feat)

        if self.merge:
            merged_out = torch.sum(torch.stack(outputs, dim=-1), dim=-1)
            return self.final_layer(merged_out)
        
        return outputs
