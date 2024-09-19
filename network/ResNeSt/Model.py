import pdb

import numpy as np
import torch
import torch.nn as nn
import os

from network.base_model import BaseModel

class Model(BaseModel):
    def __init__(self, opt, config):
        from .resnest_wrapper import Classifier
        super(Model, self).__init__(opt, config)
        self.opt = opt
        self._classifier = Classifier(opt, config)  

        self.init_common()
