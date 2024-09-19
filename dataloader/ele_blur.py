# encoding=utf-8
import torch.utils.data.dataset as dataset
import os
import cv2
import numpy as np
from misc_utils import  get_file_name, color_print, file_lines, load_json
from utils import is_first_gpu, read_image
from collections import defaultdict
from .text_list import TextListTrainValDataset

class EleBlurDataset(TextListTrainValDataset):
    """ImageDataset for training.

    Args:
        anno_txt: txt文件
        transforms: 变换
        max_size: 最多读取的数据量

    Example:
        train_dataset = ImageDataset('train.txt', aug=False)
        for i, data in enumerate(train_dataset):
            input, label = data['input']. data['label']

    """

    def __init__(self, opt, config, anno_txt, transforms, phase='train', max_size=float('inf'), psnr_path=None):
        super(EleBlurDataset, self).__init__(opt, config, anno_txt, transforms, phase, max_size)

        if psnr_path is not None:
            self._psnr_dict = load_json(psnr_path)
        else:
            self._psnr_dict = None

    def __getitem__(self, index):
        """
        Args:
            index(int): index

        Returns:
            {'input': input,
             'label': label,
             'poi_id': poi_id
            }

        """
        filepath, label = self._data[index]

        filename = get_file_name(filepath)
        if self._psnr_dict is not None and filename in self._psnr_dict:
            psnr = self._psnr_dict[filename] / 14 - 0.5
            psnr = min(max(psnr, 0.), 1.)
        else:
            psnr = 1.0

        filename = os.path.basename(filepath)
        image = read_image(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0  # 转成0~1之间

        sample = self.transforms(**{
            'image': image,
        })

        input = sample['image']

        if self._one_hot_label or self._soft_label:
            label = np.array([label, psnr], np.float32)

        sample = {
            'input': input,
            'label': label,
            'filename': filename,
            'filepath': filepath
        }

        return sample

