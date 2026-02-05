"""The UNet CNN used by the vines thermal project.
"""

import torch

#===============================================================================
#U-net code
# Dependencies
def convrelu(in_channels, out_channels,
             kernel=3, padding=None, stride=1, bias=True, inplace=True):
    """Shortcut for creating a PyTorch 2D convolution followed by a ReLU.

    Parameters
    ----------
    in_channels : int
        The number of input channels in the convolution.
    out_channels : int
        The number of output channels in the convolution.
    kernel : int, optional
        The kernel size for the convolution (default: 3).
    padding : int or None, optional
        The padding size for the convolution; if `None` (the default), then
        chooses a padding size that attempts to maintain the image-size.
    stride : int, optional
        The stride to use in the convolution (default: 1).
    bias : boolean, optional
        Whether the convolution has a learnable bias (default: True).
    inplace : boolean, optional
        Whether to perform the ReLU operation in-place (default: True).

    Returns
    -------
    torch.nn.Sequential
        The model of a 2D-convolution followed by a ReLU operation.
    """
    if padding is None:
        padding = kernel_default_padding(kernel)
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel,
                        padding=padding, bias=bias),
        torch.nn.ReLU(inplace=inplace))

#===============================================================================
# Image-based CNN Model Code

class UNet(torch.nn.Module):
    """a U-Net with a ResNet18 backbone for learning visual area labels.

    The `UNet` class implements a ["U-Net"](https://arxiv.org/abs/1505.04597)
    with a [ResNet-18](https://pytorch.org/hub/pytorch_vision_resnet/) bacbone.
    The class inherits from `torch.nn.Module`.
    
    The original implementation of this class was by Shaoling Chen
    (sc6995@nyu.edu), and additional modifications have been made by Noah C.
    Benson (nben@uw.edu).

    Parameters
    ----------
    feature_count : int
        The number of channels (features) in the input image. When using an
        `HCPVisualDataset` object for training, this value should be set to 4
        if the dataset uses the `'anat'` or `'func'` features and 8 if it uses
        the `'both'` features.
    segment_count : int
        The number of segments (AKA classes, labels) in the output data. For
        V1-V3 this is typically either 3 (V1, V2, V3) or 6 (LV1, LV2, LV3, RV1,
        RV2, RV3).
    base_model : model name or tuple, optional
        The name of the model that is to be used as the base/backbone of the
        UNet. The default is `'resnet18'`, but 
    pretrained : boolean, optional
        Whether to use a pretrained base model for the backbone (`True`) or not
        (`False`). The default is `False`.
    logits : boolean, optional
        Whether the model should return logits (`True`) or probabilities
        (`False`). The default is `True`.

    Attributes
    ----------
    pretrained_base : boolean
        `True` if the base model used in this `UNet` was originally pre-trained
        and `False` otherwise.
    base_model : PyTorch Module
        The ResNet-18 model that is used as the backbone of the `UNet` model.
    base_layers : list of PyTorch Modules
        The ResNet-18 layers that are used in the backbone of the `UNet` model.
    feature_count : int
        The number of input channels (features) that the model expects in input
        images.
    segment_count : int
        The number of segments (labels) predicted by the model.
    logits : bool
        `True` if the output of the model is in logits and `False` if its output
        is in probabilities.
    """
    def __init__(self, feature_count=3, segment_count=1,
                 base_model='resnet18',
                 pretrained=True,
                 logits=True):
        import torch.nn as nn
        # Initialize the super-class.
        super().__init__()
        # Store some basic attributes.
        self.feature_count = feature_count
        self.segment_count = segment_count
        self.pretrained = pretrained
        self.logits = logits
        # Set up the base model and base layers for the model.
        if pretrained:
            weights = 'IMAGENET1K_V1'
        else:
            weights = None
        import torchvision.models as mdls
        base_model = getattr(mdls, base_model)
        try:
            base_model = base_model(weights=weights)
        except TypeError:
            base_model = base_model(pretrained=pretrained)
        # Not sure we should store the base model; seems like a good idea, but
        # does it get caught up in PyTorch's Module data when we do?
        #self.base_model = resnet18(pretrained=pretrained)
        # Because the input size may not be 3 and the output size may not be 3,
        # we want to add an additional 
        if feature_count != 3:
            # Adjust the first convolution's number of input channels.
            c1 = base_model.conv1
            base_model.conv1 = nn.Conv2d(
                feature_count, c1.out_channels,
                kernel_size=c1.kernel_size, stride=c1.stride,
                padding=c1.padding, bias=c1.bias)
        base_layers = list(base_model.children())
        #self.base_layers = base_layers
        # Make the U-Net layers out of the base-layers.
        # size = (N, 64, H/2, W/2)
        self.layer0 = nn.Sequential(*base_layers[:3]) 
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        # size = (N, 64, H/4, W/4)
        self.layer1 = nn.Sequential(*base_layers[3:5])
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        # size = (N, 128, H/8, W/8)        
        self.layer2 = base_layers[5]
        self.layer2_1x1 = convrelu(128, 128, 1, 0)  
        # size = (N, 256, H/16, W/16)
        self.layer3 = base_layers[6]  
        self.layer3_1x1 = convrelu(256, 256, 1, 0)  
        # size = (N, 512, H/32, W/32)
        self.layer4 = base_layers[7]
        self.layer4_1x1 = convrelu(512, 512, 1, 0)
        # The up-swing of the UNet; we will need to upsample the image.
        self.upsample = nn.Upsample(scale_factor=2,
                                    mode='bilinear',
                                    align_corners=True)
        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        self.conv_original_size0 = convrelu(feature_count, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        self.conv_last = nn.Conv2d(64, segment_count, 1)
    def forward(self, input):
        # Do the original size convolutions.
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        # Now the front few layers, which we save for adding back in on the UNet
        # up-swing below.
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)        
        layer4 = self.layer4(layer3)
        # Now, we start the up-swing; each step must upsample the image.
        layer4 = self.layer4_1x1(layer4)
        # Up-swing Step 1
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
        # Up-swing Step 2
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)
        # Up-swing Step 3
        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)
        # Up-swing Step 4
        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        # Up-swing Step 5
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)        
        # And the final convolution.
        out = self.conv_last(x)
        if not self.logits:
            out = torch.sigmoid(out)
        return out
