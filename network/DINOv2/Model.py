import pdb

import numpy as np
import torch
import torch.nn as nn
import os

from network.base_model import BaseModel

from .dinov2_wrapper import Classifier

class Model(BaseModel):
    def __init__(self, opt, config):
        super(Model, self).__init__(opt, config)
        self.opt = opt
        self._classifier = Classifier(opt, config)  

        self.init_common()

    def update(self, input, label):
        predicted = self.classifier(input)
        loss = self.loss(predicted, label, avg_meters=self.avg_meters)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return None