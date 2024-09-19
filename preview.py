# encoding: utf-8
import torch
import ipdb
import cv2
import os
import numpy as np
from options import opt
# from dataloader import paired_dataset
from mscv.summary import create_summary_writer, write_image
from mscv.image import tensor2im
from os.path import join

from dataloader.dataloaders import train_dataloader, val_dataloader
import cv2

import misc_utils as utils

import random

"""
source domain 是clear的
"""

"""
这个改成需要预览的数据集
"""
previewed = val_dataloader  # train_dataloader, val_dataloader
opt.workers = 0

from PIL import Image, ImageDraw, ImageFont
 

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  
        img = Image.fromarray(img)  # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
    draw = ImageDraw.Draw(img)
    # 字体
    fontStyle = ImageFont.truetype(
        "MSYHBD.TTC", textSize, encoding="utf-8")
 
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
 
    return np.asarray(img)


mean = (0.485, 0.456, 0.406, 0.5)
std = (0.229, 0.224, 0.225, 0.5)

def denormalize(img, max_pixel_value=1.):
    global mean, std
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    img *= std
    img += mean
    return img

CanChinese = False

preview_root = 'preview'
os.makedirs(preview_root, exist_ok=True)

for i, sample in enumerate(previewed):
    # if i > 30:
    #     break
    utils.progress_bar(i, len(previewed), 'Handling...')

    if opt.debug:
        ipdb.set_trace()
    label = sample['label'][0].item()
    poi_id = sample['poi_id'][0]
    input = sample['input'][0].detach().cpu().numpy().transpose([1,2,0])
    # image = (image.copy()
    input = (denormalize(input, max_pixel_value=1.0)*255).astype(np.uint8).copy()

    image = input[:, :, :3]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    heatmap = input[:, :, 3]
    cv2.imwrite(join(preview_root, f'{poi_id}_{label}.jpg'), image)
    cv2.imwrite(join(preview_root, f'{poi_id}_heat.jpg'), heatmap)

    # label = sample['label'][0].item()

    # if CanChinese:
    #     name = names[str(label)]
    #     image = cv2ImgAddText(image, name, 7, 3, (255, 0, 0), textSize=24)
    # else:
    #     cv2.putText(image, 'label: ' + str(label), (10, 30), 0, 1, (255, 0, 0), 2)

    # write_image(writer, f'preview_{opt.dataset}/{i}', '0_input', image, 0, 'HWC')


# writer.flush()