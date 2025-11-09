# ==============================================================================
# Copyright 2025 Florent Mahoudeau. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Inspired from the work by Y. Zhou, X. Sun, Z-J Zha and W. Zeng:
# MiCT: Mixed 3D/2D Convolutional Tube for Human Action Recognition
# ==============================================================================
# More details about this implementation can be found here:
# https://github.com/fmahoudeau/MiCT-Net-PyTorch
# https://towardsdatascience.com/mict-net-for-human-action-recognition-in-videos-3a18e4f97342
# ==============================================================================

import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['MiCTResNet', 'MiCTBlock', 'get_mictresnet']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}


def _to_4d_tensor(x):
    """
    Converts a 5d tensor to 4d by flattening
    the batch and depth dimensions:
    [B, C, D, H, W] --> [(B*D), C, H, W]
    Returns (x4d, D_after_stride)
    """
    # Move D next to B, then merge (B, D) -> (B*D)
    # contiguous() ensures good layout for conv2d afterwards
    x4d = x.movedim(2, 1).contiguous().flatten(0, 1)   # (B*D, C, H, W)
    return x4d, x.size(2)


def _to_5d_tensor(x, depth):
    """
    Converts a 4D tensor back to 5D by splitting/unflattening
    the batch dimension to restore the depth dimension:
    [(B*D), C, H, W] --> [B, C, D, H, W]
    Returns x5d
    """
    B = x.size(0) // depth
    x5d = x.unflatten(0, (B, depth)).movedim(1, 2).contiguous()
    return x5d


class MiCTBlock(nn.Module):
    """Mixes a ResNet BasicBlock with a residual 3D channel separated grouped block."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None,
                 block_kernels=((3, 3, 3), (3, 3, 3)), groups=1):
        super(MiCTBlock, self).__init__()

        # The 2D path is a ResNet BasicBlock
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride[1], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 3D convolutions
        self.n_conv_3d = self._check_kernels(block_kernels)
        self.conv21 = nn.Conv3d(inplanes, planes, kernel_size=block_kernels[0],
                                stride=(stride[0], stride[1], stride[1]),
                                padding=(block_kernels[0][0]//2, 
                                         block_kernels[0][1]//2, 
                                         block_kernels[0][2]//2),
                                groups=groups,
                                bias=False)
        self.bn21 = nn.BatchNorm3d(planes)
        self.conv22 = nn.Conv3d(planes, planes, kernel_size=block_kernels[1],
                                stride=(stride[0], 1, 1),
                                padding=(block_kernels[1][0]//2, 
                                        block_kernels[1][1]//2, 
                                        block_kernels[1][2]//2),
                                groups=groups,
                                bias=False)
        self.bn22 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.block_kernels = block_kernels

    @staticmethod
    def _check_kernels(block_kernels):
        if not isinstance(block_kernels, (list, tuple)):
            raise TypeError("block_kernels must be a list/tuple of kernel tuples.")
        n = len(block_kernels)
        if n < 1 or n > 2:
            raise ValueError("block_kernels must contain 1 or 2 kernel tuples.")
        for k in block_kernels:
            if not (isinstance(k, (list, tuple)) and len(k) == 3):
                raise ValueError("Each kernel must be a 3-tuple like (Kt, Kh, Kw).")
        return n

    def forward(self, x):
        out1 = self.conv21(x)
        out1 = self.bn21(out1)
        out1 = self.relu(out1)
        out1 = self.conv22(out1)
        out1 = self.bn22(out1)
        out1 = self.relu(out1)

        x, depth = _to_4d_tensor(x)
        residual = x
        out2 = self.conv1(x)
        out2 = self.bn1(out2)
        out2 = self.relu(out2)
        out2 = self.conv2(out2)
        out2 = self.bn2(out2)

        if self.downsample is not None:
            residual = self.downsample(x)

        out2 += residual
        out2 = self.relu(out2)
        out2 = _to_5d_tensor(out2, depth)
        out = out1 + out2
        return out


class MiCTStem(nn.Module):
    def __init__(self, channels, stem_kernel=(3, 3, 3)):
        super(MiCTStem, self).__init__()

        self.stem_kernel = stem_kernel
        self.channels = channels

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7),
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,
                                     stride=2, padding=1)

        padding = (stem_kernel[0]//2, stem_kernel[1]//2, stem_kernel[2]//2)
        self.conv21 = nn.Conv3d(3, channels[0], kernel_size=stem_kernel,
                                stride=(1, 2, 2), 
                                padding=padding, bias=False)
        self.bn21 = nn.BatchNorm3d(channels[0])
        self.conv22 = nn.Conv3d(channels[0], channels[1], kernel_size=stem_kernel,
                                stride=1, padding=padding, bias=False)
        self.bn22 = nn.BatchNorm3d(channels[1])
        self.conv23 = nn.Conv3d(channels[1], channels[2], kernel_size=stem_kernel,
                                stride=1, padding=padding, bias=False)
        self.bn23 = nn.BatchNorm3d(channels[2])
        self.maxpool2 = nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.transpose(1, 2)  # BxDxCxHxW => BxCxDxHxW
        out1 = self.conv21(x)
        out1 = self.bn21(out1)
        out1 = self.relu(out1)
        out1 = self.conv22(out1)
        out1 = self.bn22(out1)
        out1 = self.relu(out1)
        out1 = self.conv23(out1)
        out1 = self.bn23(out1)
        out1 = self.relu(out1)
        out1 = self.maxpool2(out1)

        x, depth = _to_4d_tensor(x)
        out2 = self.conv1(x)
        out2 = self.bn1(out2)
        out2 = self.relu(out2)
        out2 = self.maxpool1(out2)
        out2 = _to_5d_tensor(out2, depth)
        out = out1 + out2
        return out


class MiCTResNet(nn.Module):
    """
    MiCTResNet is a ResNet backbone augmented with five 3D cross-domain
    residual convolutions.

    The model operates on 5D tensors but since 2D CNNs expect 4D input,
    the data is transformed many times to 4D and then transformed back
    to 5D when necessary. For efficiency only one 2D convolution is
    performed for each kernel by vertically stacking the features maps
    of each video clip contained in the batch.

    This models is inspired from the work by Y. Zhou, X. Sun, Z-J Zha
    and W. Zeng: MiCT: Mixed 3D/2D Convolutional Tube for Human Action
    Recognition.
    """

    def __init__(self, block, layers, stem_kernels, 
                 stages_kernels, groups, **kwargs):
        """
        :param block: the block class, either BasicBlock or Bottleneck.
        :param layers: the number of blocks for each for each of the
            four feature depth.
        """
        super(MiCTResNet, self).__init__(**kwargs)

        self.inplanes = 64
        self.stem = MiCTStem([16, 32, 64], stem_kernels)

        self.layer1 = MiCTStage(block, self.inplanes, 64, layers[0], stride=(1, 1),
                                block_kernels=stages_kernels[0], groups=groups[0])
        self.layer2 = MiCTStage(block, self.layer1.inplanes, 128, layers[1], stride=(1, 2),
                                block_kernels=stages_kernels[1], groups=groups[1])
        self.layer3 = MiCTStage(block, self.layer2.inplanes, 256, layers[2], stride=(1, 2),
                                block_kernels=stages_kernels[2], groups=groups[2])
        self.layer4 = MiCTStage(block, self.layer3.inplanes, 512, layers[3], stride=(1, 1),
                                block_kernels=stages_kernels[3], groups=groups[3])

    def transfer_weights(self, state_dict):
        """
        Transfers ResNet weights pre-trained on the ImageNet dataset.

        :param state_dict: the state dictionary of the loaded ResNet model.
        """
        for key in state_dict.keys():
            if key.startswith('conv') | key.startswith('bn'):
                eval('self.stem.' + key + '.data.copy_(state_dict[\'' + key + '\'])')
            if key.startswith('layer'):
                var = key.split('.')
                if var[2] == 'downsample':
                    eval('self.' + var[0] + '.bottlenecks[' + var[1] + '].downsample[' + var[3] + '].' +
                        var[4] + '.data.copy_(state_dict[\'' + key + '\'])')
                else:
                    eval('self.' + var[0] + '.bottlenecks[' + var[1] + '].' + var[2] + '.' + var[3] +
                         '.data.copy_(state_dict[\'' + key + '\'])')

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class MiCTStage(nn.Module):
    """
    The MiCTStage groups several MiCTBlocks at a given feature depth. 
    """
    def __init__(self, block, inplanes, planes, n_blocks, stride=(1, 1),
                 block_kernels=((3, 3, 3), (3, 3, 3)), groups=1):
        """
        :param block: the block class, either BasicBlock or Bottleneck.
        :param inplanes: the number of input plances.
        :param planes: the number of output planes.
        :param n_blocks: the number of blocks.
        :param stride: (temporal, spatial) stride.
        """
        super(MiCTStage, self).__init__()

        expansion = block.expansion
        downsample = None
        if stride[1] != 1 or inplanes != planes * expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * expansion, kernel_size=1, 
                          stride=stride[1], bias=False),
                nn.BatchNorm2d(planes * expansion),
            )

        self.n_blocks = n_blocks
        self.stride = stride
        self.block_kernels = block_kernels
        self.bottlenecks = nn.ModuleList()

        self.bottlenecks.append(            
            block(
                inplanes, planes, stride,
                downsample=downsample, 
                block_kernels=block_kernels, 
                groups=groups
            )
        )

        self.inplanes = planes * expansion
        for _ in range(1, n_blocks):
            self.bottlenecks.append(
                block(
                    self.inplanes, planes,
                    block_kernels=block_kernels, 
                    groups=groups
                )
            )

    def forward(self, x):
        out = x
        for i in range(0, self.n_blocks):
            out = self.bottlenecks[i](out)
        return out


def get_mictresnet(backbone, stem_kernels, stages_kernels, 
                   groups, pretrained=True, **kwargs):
    """
    Constructs a MiCT-Net model with a ResNet backbone.

    :param backbone: the ResNet backbone, only `resnet18` is supported.
    :param pretrained: if True, returns a model pre-trained on ImageNet.
    """
    if backbone == 'resnet18':
        model = MiCTResNet(
            MiCTBlock, [2, 2, 2, 2], stem_kernels, 
            stages_kernels, groups, **kwargs)
        if pretrained:
            model.transfer_weights(model_zoo.load_url(model_urls['resnet18']))
    else:
        raise ValueError('Unknown backbone: {}'.format(backbone))

    return model
