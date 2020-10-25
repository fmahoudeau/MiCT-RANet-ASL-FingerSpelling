# ==============================================================================
# Copyright 2020 Florent Mahoudeau. All Rights Reserved.
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

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F


__all__ = ['MiCTResNet', 'MiCTBlock', 'get_mictresnet']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}


def _to_4d_tensor(x, depth_stride=None):
    """Converts a 5d tensor to 4d by stacking
    the batch and depth dimensions."""
    x = x.transpose(0, 2)  # swap batch and depth dimensions: BxCxDxHxW => DxCxBxHxW
    if depth_stride:
        x = x[::depth_stride]  # downsample feature maps along depth dimension
    depth = x.size()[0]
    x = x.permute(2, 0, 1, 3, 4)  # DxCxBxHxW => BxDxCxHxW
    x = torch.split(x, 1, dim=0)  # split along batch dimension: BxDxCxHxW => B*[1xDxCxHxW]
    x = torch.cat(x, 1)  # concatenate along depth dimension: B*[1xDxCxHxW] => 1x(B*D)xCxHxW
    x = x.squeeze(0)  # 1x(B*D)xCxHxW => (B*D)xCxHxW
    return x, depth


def _to_5d_tensor(x, depth):
    """Converts a 4d tensor back to 5d by splitting
    the batch dimension to restore the depth dimension."""
    x = torch.split(x, depth)  # (B*D)xCxHxW => B*[DxCxHxW]
    x = torch.stack(x, dim=0)  # re-instate the batch dimension: BxDxCxHxW
    x = x.transpose(1, 2)  # swap back depth and channel dimensions: BxDxCxHxW => BxCxDxHxW
    return x


class BasicBlock(nn.Module):
    """ResNet BasicBlock"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

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

    def __init__(self, block, layers, dropout, n_classes, no_top=False,
                 first_3d_conv_depth=5, mict_3d_conv_depth=5,
                 t_strides=(1, 1, 1, 1, 1), t_padding='center', **kwargs):
        """
        :param block: the block class, either BasicBlock or Bottleneck.
        :param layers: the number of blocks for each for each of the
            four feature depth.
        :param dropout: dropout rate applied during training.
        :param n_classes: the number of classes in the dataset.
        :param no_top: remove pooling and classification layer.
        :param first_3d_conv_depth: the depth of the first 3D kernel.
        :param mict_3d_conv_depth: the depth of the 3D kernel in each MiCT block.
        :param t_strides: temporal strides for each of the 3D convolutions.
        :param t_padding: temporal padding method, either 'center' or 'forward'.
        """
        super(MiCTResNet, self).__init__(**kwargs)

        if t_padding not in ('forward', 'center'):
            raise ValueError('Invalid value for parameter `t_padding`: {}'.format(t_padding))

        self.inplanes = 64
        self.dropout = dropout
        self.t_strides = t_strides
        self.first_3d_conv_depth = first_3d_conv_depth
        self.t_padding = t_padding
        self.n_classes = n_classes
        self.no_top = no_top

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7),
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,
                                     stride=2, padding=1)

        self.conv2 = nn.Conv3d(3, 64, kernel_size=(first_3d_conv_depth, 7, 7),
                               stride=(t_strides[0], 2, 2),
                               padding=(first_3d_conv_depth//2, 3, 3) if t_padding == 'center' else 0,
                               bias=False)
        self.bn2 = nn.BatchNorm3d(64)
        self.maxpool2 = nn.MaxPool3d(kernel_size=3, stride=(t_strides[1], 2, 2), padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = MiCTBlock(block, self.inplanes, 64, layers[0], stride=(1, 1),
                                mict_3d_conv_depth=mict_3d_conv_depth, t_padding=t_padding)
        self.layer2 = MiCTBlock(block, self.layer1.inplanes, 128, layers[1], stride=(t_strides[2], 2),
                                mict_3d_conv_depth=mict_3d_conv_depth, t_padding=t_padding)
        self.layer3 = MiCTBlock(block, self.layer2.inplanes, 256, layers[2], stride=(t_strides[3], 2),
                                mict_3d_conv_depth=mict_3d_conv_depth, t_padding=t_padding)
        self.layer4 = MiCTBlock(block, self.layer3.inplanes, 512, layers[3], stride=(t_strides[4], 1),
                                mict_3d_conv_depth=mict_3d_conv_depth, t_padding=t_padding)

        if not self.no_top:
            self.avgpool1 = nn.AdaptiveAvgPool3d((None, 1, 1))
            self.avgpool2 = nn.AdaptiveAvgPool1d(1)
            self.drop = nn.Dropout3d(dropout)
            self.fc = nn.Linear(512 * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def transfer_weights(self, state_dict):
        """
        Transfers ResNet weights pre-trained on the ImageNet dataset.

        :param state_dict: the state dictionary of the loaded ResNet model.
        :return: None
        """
        for key in state_dict.keys():
            if key.startswith('conv1') | key.startswith('bn1'):
                eval('self.' + key + '.data.copy_(state_dict[\'' + key + '\'])')
            if key.startswith('layer'):
                var = key.split('.')
                if var[2] == 'downsample':
                    eval('self.' + var[0] + '.bottlenecks[' + var[1] + '].downsample[' + var[3] + '].' +
                         var[4] + '.data.copy_(state_dict[\'' + key + '\'])')
                else:
                    eval('self.' + var[0] + '.bottlenecks[' + var[1] + '].' + var[2] + '.' + var[3] +
                         '.data.copy_(state_dict[\'' + key + '\'])')

    def forward(self, x):
        x = x.transpose(1, 2)  # BxDxCxHxW => BxCxDxHxW
        if self.t_padding == 'forward':
            out1 = F.pad(x, [3, 3, 3, 3, 0, 2*(self.first_3d_conv_depth//2)], 'constant', 0)
            out1 = self.conv2(out1)
        else:
            out1 = self.conv2(x)  # center padding
        out1 = self.bn2(out1)
        out1 = self.relu(out1)
        out1 = self.maxpool2(out1)

        x, depth = _to_4d_tensor(x, depth_stride=self.t_strides[1])
        out2 = self.conv1(x)
        out2 = self.bn1(out2)
        out2 = self.relu(out2)
        out2 = self.maxpool1(out2)
        out2 = _to_5d_tensor(out2, depth)
        out = out1 + out2

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if not self.no_top:
            out = self.avgpool1(out)
            out = out.squeeze(4).squeeze(3)
            out = self.drop(out)
            out_fc = []
            for i in range(out.size()[-1]):
                out_fc.append(self.fc(out[:, :, i]).unsqueeze(2))
            out_fc = torch.cat(out_fc, 2)
            out = self.avgpool2(out_fc).squeeze(2)

        return out


class MiCTBlock(nn.Module):
    """
    The MiCTBlock groups all ResNet basic/bottleneck blocks at
    a given feature depth. It performs a parallel 3D convolution
    on the input and then merges the output with the output of
    the first 2D CNN block using point-wise summation to form
    a residual cross-domain connection.
    """
    def __init__(self, block, inplanes, planes, blocks, stride=(1, 1),
                 mict_3d_conv_depth=5, t_padding='center'):
        """
        :param block: the block class, either BasicBlock or Bottleneck.
        :param inplanes: the number of input plances.
        :param planes: the number of output planes.
        :param blocks: the number of blocks.
        :param stride: (temporal, spatial) stride.
        :param mict_3d_conv_depth: the depth of the 3D kernel in each MiCT block.
        :param t_padding: temporal padding method, either 'center' or 'forward'.
        """
        super(MiCTBlock, self).__init__()

        if t_padding not in ('forward', 'center'):
            raise ValueError('Invalid value for parameter `t_padding`: {}'.format(t_padding))

        downsample = None
        if stride[1] != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride[1], bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        self.blocks = blocks
        self.stride = stride
        self.mict_3d_conv_depth = mict_3d_conv_depth
        self.t_padding = t_padding
        self.bottlenecks = nn.ModuleList()
        self.bottlenecks.append(block(inplanes, planes, stride[1],
                                      downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            self.bottlenecks.append(block(self.inplanes, planes))

        self.conv = nn.Conv3d(inplanes, planes, kernel_size=(mict_3d_conv_depth, 3, 3),
                              stride=(stride[0], stride[1], stride[1]),
                              padding=(mict_3d_conv_depth//2, 1, 1) if t_padding == 'center' else 0,
                              bias=False)
        self.bn = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.t_padding == 'forward':
            out1 = F.pad(x, [1, 1, 1, 1, 0, 2*(self.mict_3d_conv_depth//2)], 'constant', 0)
            out1 = self.conv(out1)
        else:
            out1 = self.conv(x)
        out1 = self.bn(out1)
        out1 = self.relu(out1)

        x, depth = _to_4d_tensor(x, depth_stride=self.stride[0])
        out2 = self.bottlenecks[0](x)
        out2 = _to_5d_tensor(out2, depth)
        out = out1 + out2

        out, depth = _to_4d_tensor(out)
        for i in range(1, self.blocks):
            out = self.bottlenecks[i](out)
        out = _to_5d_tensor(out, depth)

        return out


def get_mictresnet(backbone, dropout=0.5, n_classes=101,
                   no_top=False, pretrained=True, **kwargs):
    """
    Constructs a MiCT-Net model with a ResNet backbone.

    :param backbone: the ResNet backbone, either `resnet18` or `resnet34`.
    :param dropout: the dropout rate applied before the FC layer.
    :param n_classes: the number classes for classification datasets.
        Defaults to 101 for UCF-101 human action recognition.
    :param no_top: remove pooling and classification layer.
    :param pretrained: if True, returns a model pre-trained on ImageNet.
    """
    if backbone == 'resnet18':
        model = MiCTResNet(BasicBlock, [2, 2, 2, 2], dropout,
                           n_classes, no_top, **kwargs)
        if pretrained:
            model.transfer_weights(model_zoo.load_url(model_urls['resnet18']))
    elif backbone == 'resnet34':
        model = MiCTResNet(BasicBlock, [3, 4, 6, 3], dropout,
                           n_classes, no_top, **kwargs)
        if pretrained:
            model.transfer_weights(model_zoo.load_url(model_urls['resnet34']))
    else:
        raise ValueError('Unknown backbone: {}'.format(backbone))

    return model
