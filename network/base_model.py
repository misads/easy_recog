import os
from abc import abstractmethod

import torch

import misc_utils as utils
from misc_utils import color_print

from mscv import ExponentialMovingAverage
from optimizer import get_optimizer
from utils import is_distributed, convert_ckpt
from scheduler import get_scheduler
from loss import build_loss
from collections import OrderedDict


class BaseModel(torch.nn.Module):
    def __init__(self, opt, config):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.config = config
        self.avg_meters = ExponentialMovingAverage(0.95)
        self.save_dir = opt.save_dir

    def init_common(self):
        opt = self.opt
        config = self.config
        self._classifier.to(opt.device)
        find_unused_parameters = config.get('find_unused_parameters', False)

        if is_distributed(opt):
            from misc_utils import color_print
            color_print(f'ddp device: {opt.local_rank}', 5)
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self._classifier)
            self._classifier = torch.nn.parallel.DistributedDataParallel(self._classifier, find_unused_parameters=find_unused_parameters,
                    device_ids=[opt.local_rank], output_device=opt.local_rank)

        self._optimizer = get_optimizer(opt, config, self._classifier)
        self._scheduler = get_scheduler(opt, config, self._optimizer)

        self._loss = build_loss(opt, config)

        # 冻结部分网路权重
        self.freeze_parameters()

    @property
    def loss(self):
        return self._loss
    
    @property
    def classifier(self):
        return self._classifier

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def scheduler(self):
        return self._scheduler

    def forward(self, x):
        return self.classifier(x)

    def update(self, input, label):
        predicted = self.classifier(input)
        loss = self.loss(predicted, label, avg_meters=self.avg_meters)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return None

    def load(self, ckpt_path):
        load_dict = torch.load(ckpt_path, map_location='cpu')
        state_dict = convert_ckpt(load_dict['classifier'])
        self.classifier.load_state_dict(state_dict)
        if self.config.model.resume:
            self.optimizer.load_state_dict(load_dict['optimizer'])
            self.scheduler.load_state_dict(load_dict['scheduler'])
            self.scheduler.step()
            
            epoch = load_dict['epoch']
            utils.color_print('Load checkpoint from %s, resume training.' % ckpt_path, 3)
        else:
            epoch = load_dict['epoch']
            utils.color_print('Load checkpoint from %s.' % ckpt_path, 3)

        return epoch

    def save(self, which_epoch):
        save_filename = f'{which_epoch}_{self.config.model.backbone.type}.pt'
        save_path = os.path.join(self.save_dir, save_filename)
        save_dict = OrderedDict()
        save_dict['classifier'] = self.classifier.state_dict()

        save_dict['optimizer'] = self.optimizer.state_dict()
        save_dict['scheduler'] = self.scheduler.state_dict()
        save_dict['epoch'] = which_epoch
        torch.save(save_dict, save_path)
        utils.color_print(f'Save checkpoint "{save_path}".', 3)

    def freeze_parameters(self):
        # 根据config冻结部分权重
        train_set = set()
        frozen_set = set()

        frozen_parameters = self.config.model.frozen_parameters
        if frozen_parameters is not None:
            for name, param in self.classifier.named_parameters():
                for frozen_prefix in frozen_parameters:
                    if name.startswith(frozen_prefix):
                        param.requires_grad = False

                        if frozen_prefix not in frozen_set:
                            color_print(f'Frozen parameters: {frozen_prefix}', 1)
                            frozen_set.add(frozen_prefix)

            if param.requires_grad:
                name_prefix = name.split('.')[0]
                if name_prefix not in train_set:
                    color_print(f'Training parameters: {name_prefix}', 2)
                    train_set.add(name_prefix)  

    def export_onnx(self, export_path):
        raise NotImplementedError
    
    def check_onnx(self, onnx_path):
        raise NotImplementedError