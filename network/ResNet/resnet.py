import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101

"""
这个文件定义分类器的具体结构
"""

class Classifier(nn.Module):
    def __init__(self, opt, config):
        arch = config.model.backbone.type
        softlabel = config.data.soft_label
        class_names = config.data.class_names
        if softlabel:
            num_classes = 1
        else:
            num_classes = len(class_names)

        super(Classifier, self).__init__()
        if arch == 'ResNet50':
            self.network = resnet50(pretrained=True)
        elif arch == 'ResNet101':
            self.network = resnet101(pretrained=True)
        else:
            raise NotImplementedError(f'arch "{arch}" not implemented error.')

        num_feats = self.network.fc.in_features
        self.network.fc = nn.Linear(num_feats, num_classes)

    def forward(self, input):
        x = input
        return self.network(x)


# a = Classifier()
# img = torch.randn([1, 3, 256, 256])

