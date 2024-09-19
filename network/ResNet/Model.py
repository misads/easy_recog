import torch
import os

import torch.nn.functional as F
from collections import OrderedDict

from network.base_model import BaseModel

import misc_utils as utils

from .resnet import Classifier


class Model(BaseModel):
    def __init__(self, opt, config):
        super(Model, self).__init__(opt, config)
        self.opt = opt
        self._classifier = Classifier(opt.model)
        self.init_common()
