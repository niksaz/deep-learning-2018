"""
Based on https://arxiv.org/pdf/1611.05431.pdf.
"""

import torch.nn as nn
import torch
import math


class ResNeXtBlock(nn.Module):

    def __init__(self, downsample):
        super(ResNeXtBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.comp_block = None
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.comp_block(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class TwoLayeredBlock(ResNeXtBlock):
    expansion = 1

    def __init__(self, card, inplanes, planes, stride=1, downsample=None):
        super(TwoLayeredBlock, self).__init__(downsample)
        # For two layered block it is just wider convolutions.
        self.comp_block = nn.Sequential(
            conv3x3(inplanes, planes * 2, stride),
            nn.BatchNorm2d(planes * 2),
            nn.ReLU(inplace=True),
            conv3x3(planes * 2, planes),
            nn.BatchNorm2d(planes),
        )


class ThreeLayeredBlock(ResNeXtBlock):
    expansion = 4

    def __init__(self, card, inplanes, planes, stride=1, downsample=None):
        super(ThreeLayeredBlock, self).__init__(downsample)
        # For three layered block it is wider convolutions and they are also
        # grouped on the second step.
        self.comp_block = nn.Sequential(
            nn.Conv2d(inplanes, planes*2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes*2, planes*2, kernel_size=3, stride=stride,
                      padding=1, bias=False, groups=card),
            nn.BatchNorm2d(planes*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes*2, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )


class ResNeXt(nn.Module):
    card = 32

    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(3, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.card, self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.card, self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnext18(weights_path=None):
    """Constructs a ResNeXt-18 model."""
    model = ResNeXt(TwoLayeredBlock, [2, 2, 2, 2])
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
    return model


def resnext34(weights_path=None):
    """Constructs a ResNeXt-34 model."""
    model = ResNeXt(TwoLayeredBlock, [3, 4, 6, 3])
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
    return model


def resnext50(weights_path=None):
    """Constructs a ResNeXt-50 model."""
    model = ResNeXt(ThreeLayeredBlock, [3, 4, 6, 3])
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
    return model


def resnext101(weights_path=None):
    """Constructs a ResNeXt-101 model."""
    model = ResNeXt(ThreeLayeredBlock, [3, 4, 23, 3])
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
    return model


def resnext152(weights_path=None):
    """Constructs a ResNeXt-152 model."""
    model = ResNeXt(ThreeLayeredBlock, [3, 8, 36, 3])
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
    return model
