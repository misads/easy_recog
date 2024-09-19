import numpy as np
import torch
import torch.nn as nn

from .resnest_mmdet import ResNeSt
from misc_utils import color_print
from utils import is_first_gpu, is_distributed

# classes = opt.num_classes

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)


class Classifier(nn.Module):
    def __init__(self, opt, config):
        arch = config.model.backbone.type
        pretrained = config.model.backbone.pretrained
        softlabel = config.data.soft_label
        class_names = config.data.class_names
        if softlabel:
            num_classes = 1
        else:
            num_classes = len(class_names)

        super(Classifier, self).__init__()
        if is_distributed(opt):
            if is_first_gpu(opt):
                color_print('多卡训练, norm方式设为SyncBN', 5)
            norm_cfg = dict(type='SyncBN', requires_grad=True)
        else:
            color_print('单卡训练, norm方式设为BN', 5)
            norm_cfg = dict(type='BN', requires_grad=True)

        if arch == 'ResNeSt50':
            
            if pretrained:
                init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnest50')
            else:
                init_cfg = None

            self.network = ResNeSt(
                in_channels=3,
                stem_channels=64,
                depth=50,
                radix=2,
                reduction_factor=4,
                avg_down_stride=True,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=-1,
                norm_cfg=norm_cfg,
                norm_eval=False,
                style='pytorch',
                init_cfg=init_cfg   
            )
        elif arch == 'ResNeSt101':
            if pretrained:
                init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnest101')
            else:
                init_cfg = None

            self.network = ResNeSt(
                in_channels=3,
                stem_channels=128,
                depth=101,
                radix=2,
                reduction_factor=4,
                avg_down_stride=True,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=-1,
                norm_cfg=norm_cfg,
                norm_eval=False,
                style='pytorch',
                # dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
                # stage_with_dcn=(False, True, True, True),
                init_cfg=init_cfg
            )
        else:
            raise NotImplementedError(f'arch "{arch}" not implemented error.')
        
        self.network.init_weights()
        
        num_feats = 2048
        self.avgpool = GlobalAvgPool2d()
        self.fc = nn.Linear(num_feats, num_classes)

    def forward(self, input):
        x = input
        x = self.network(x)
        layer1, layer2, layer3, layer4, *layer5 = x
        
        x = self.avgpool(layer4)
        #x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

