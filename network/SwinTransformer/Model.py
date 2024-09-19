import torch.nn as nn

from torch.cuda.amp import autocast, GradScaler
from network.base_model import BaseModel


class Model(BaseModel):
    def __init__(self, opt):
        from .swin_wrapper import Classifier
        super(Model, self).__init__()
        self.opt = opt
        self._classifier = Classifier(opt.model)  
        # self.classifier.apply(weights_init)  # 初始化权重

        # print_network(self.classifier)
        if opt.fp16:
            self.loss_scaler = GradScaler(init_scale=512.)

        self.init_common()


    def update(self, input, label):
        if opt.fp16:
            with autocast():
                predicted = self.classifier(input)
                loss = get_loss(predicted, label, avg_meters=self.avg_meters)
        else:
            predicted = self.classifier(input)
            loss = get_loss(predicted, label, avg_meters=self.avg_meters)

        self.optimizer.zero_grad()

        if opt.fp16:
            self.loss_scaler.scale(loss).backward()
            self.loss_scaler.unscale_(self.optimizer)
        else:
            loss.backward()

        params = list(filter(lambda p: p.requires_grad and p.grad is not None, self.classifier.parameters()))
        nn.utils.clip_grad_norm_(params, 5.)

        if opt.fp16:
            self.loss_scaler.step(self.optimizer)
            self.loss_scaler.update()
        else:
            self.optimizer.step()

        return None

    def forward(self, x):
        if opt.fp16:
            with autocast():
                return self.classifier(x)
        return self.classifier(x)
