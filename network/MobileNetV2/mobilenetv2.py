import pdb

import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101
from copy import deepcopy
from utils import load_ckpt


__all__ = ['mobilenetv2']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.inp = inp
        self.oup = oup
        self.stride = stride
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

    def __repr__(self):
        if self.stride == 1:
            main_str = self._get_name() + f'({self.inp}, {self.oup})'
        else:
            main_str = self._get_name() + f'({self.inp}, {self.oup}, stride={self.stride})'
        return main_str


class MobileNetV2(nn.Module):
    def __init__(self, 
                 num_classes=None, 
                 width_mult=1., 
                 out_layers=4, 
                 num_input_images=1, 
                 **kwargs):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        assert out_layers in [3, 4, 5]
        self.out_layers = out_layers
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],  # 1, stride 4
            [6,  32, 3, 2],  # 2, stride 8
            [6,  64, 4, 2],
            [6,  96, 3, 1],  # 4, stride 16
        ]
        if out_layers >= 4:
            self.cfgs.extend([
                [6, 160, 3, 2],
                [6, 320, 1, 1]
            ])  # 6, stride 32)

        self.num_classes = num_classes
        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        self.conv1 = conv_3x3_bn(3 * num_input_images, input_channel, 2)
        # layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        self.layers = nn.ModuleList()
        self.num_layers = len(self.cfgs)
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            layers = []
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
            self.layers.append(nn.Sequential(*layers))
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        if self.num_classes is not None:
            self.conv = conv_1x1_bn(input_channel, output_channel)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(output_channel, num_classes)

        self.init_weights()

    def forward(self, x):
        out_layers = [1, 2, 4, 6] if self.out_layers >= 4 else [1, 2, 4]
        x = self.conv1(x)
        outs = []

        if self.out_layers == 5:
            outs.append(x)

        for i in range(self.num_layers):
            x = self.layers[i](x)
            if i in out_layers:
                outs.append(x)
        if self.num_classes is None:
            return outs
        else:
            x = self.conv(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(**kwargs)


# class Classifier(nn.Module):
#     def __init__(self, opt, config):
#         backbone_config = deepcopy(config.model.backbone)
#         arch = backbone_config.pop('type')
#         pretrained = backbone_config.pop('pretrained')
#         softlabel = config.data.soft_label
#         class_names = config.data.class_names
#         if softlabel:
#             num_classes = 1
#         else:
#             num_classes = len(class_names)

#         backbone_config['num_classes'] = num_classes 

#         super(Classifier, self).__init__()
#         if arch == 'MobileNetV2':
#             import ipdb
#             ipdb.set_trace()

#             self.network = mobilenetv2(**backbone_config)
#             if pretrained is not None:
#                 state_dict = torch.load(pretrained, map_location='cpu')
#                 load_ckpt(self.network, state_dict)
#         else:
#             raise NotImplementedError(f'arch "{arch}" not implemented error.')

#     def forward(self, input):
#         x = input
#         return self.network(x)


