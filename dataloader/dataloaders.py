# encoding=utf-8
import albumentations as A
import torch

from os.path import join
from dataloader.text_list import TextListTrainValDataset
from dataloader.ele_blur import EleBlurDataset
from dataloader.transforms import get_transform
from torch.utils.data import DataLoader, DistributedSampler
from utils import is_distributed
from os.path import join
from copy import deepcopy

dataset_dict = {
    "text_list": TextListTrainValDataset,
    "ele_blur": EleBlurDataset
}


def get_dataloader(opt, config, phase='train'):
    assert phase in {'train', 'val'}

    data_config = deepcopy(config.data[phase]._dict)
    data_dir = data_config.pop('data_dir')
    ann_file = join(data_dir, data_config.pop('ann_file'))
    batch_dize = data_config.pop('batch_size')
    _ = data_config.pop('transform')
    num_workers = config.runner.num_workers if not opt.debug else 0
    transforms = get_transform(config, phase=phase)

    max_size = batch_dize * 50 if opt.debug else float('inf')  # debug模式时dataset的最大大小

    dataset_class = dataset_dict[config.data.type]
    dataset = dataset_class(opt, config, ann_file, transforms=transforms, phase=phase, max_size=max_size, **data_config)

    if phase == 'train':
        sampler = DistributedSampler(dataset) if is_distributed(opt) else None
        shuffle = not is_distributed(opt) 
        dataloader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_dize,
            num_workers=num_workers,
            drop_last=True,
            sampler=sampler
        )
    else:
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_dize, 
            shuffle=False, 
            num_workers=num_workers
        )
    return dataloader

