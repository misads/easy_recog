# encoding=utf-8
import torch.utils.data.dataset as dataset
import os
import cv2
import numpy as np
from misc_utils import  get_file_name, color_print, file_lines
from utils import is_first_gpu, read_image
from collections import defaultdict


class TextListTrainValDataset(dataset.Dataset):
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

    def __init__(self, opt, config, anno_txt, transforms, phase='train', max_size=float('inf')):
        self.transforms = transforms
        self.max_size = max_size
        self._data = [] 
        self.class_names = config.data.class_names
        self._one_hot_label = config.data.one_hot_label
        self._soft_label = config.data.soft_label and phase == 'train'
        assert not (self._one_hot_label and self._soft_label), 'soft_label and one_hot_label can not be both set'

        dataset_name = get_file_name(anno_txt)

        lines = file_lines(anno_txt)
        total_num = 0
        counter = defaultdict(int)
        for line in lines:
            if ',' in line:
                filepath, label = line.split(',')
            else:
                filepath, label = line.split(' ')

            assert os.path.isfile(filepath), f'{filepath} not found.'
            if len(label) == 0:
                continue

            total_num += 1

            if self._soft_label:
                label = float(label)
                counter[round(label, 1)] += 1
            else:
                label = int(label)
                counter[label] += 1

            if self._one_hot_label:
                one_hot_label = [0] * len(self.class_names)
                one_hot_label[label] = 1
                self._data.append((filepath, one_hot_label))
            else:
                self._data.append((filepath, label))

        if is_first_gpu(opt):
            color_print(f'dataset "{dataset_name}" build ok.', 3)
            print(f'total_num: {total_num}')
            for label in range(len(self.class_names)):
                num = counter[label]
                ratio = num / total_num * 100
                color_print(f'class: "{self.class_names[label]}", num: {num} ({ratio:.2f}%)', 4)

        self.anno_txt = anno_txt


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
        filename = os.path.basename(filepath)
        image = read_image(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0  # 转成0~1之间

        sample = self.transforms(**{
            'image': image,
        })

        input = sample['image']

        if self._one_hot_label or self._soft_label:
            label = np.array(label, np.float32)

        sample = {
            'input': input,
            'label': label,
            'filename': filename,
            'filepath': filepath
        }

        return sample

    def __len__(self):            
        return min(self.max_size, len(self._data))
