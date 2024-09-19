# python 3.5, pytorch 1.14

import os
import cv2
from network import get_model

import ipdb
import torch
import numpy as np
from utils import raise_exception, Context
from os.path import join
from multiprocessing.pool import ThreadPool
from misc_utils import color_print, file_lines, get_file_name, progress_bar, save_pickle
from utils import image_from_url


total = 0
cur = 0    

def worker(line):
    global cur, total
    delimeter = '\t'
    line = line.split(delimeter)
    if len(line) == 1:
        url_or_filepath = line[0]
        track_id = ''
        image_id = get_file_name(url_or_filepath)
    elif len(line) == 2:
        track_id, url_or_filepath = line
        image_id = get_file_name(url_or_filepath)
    elif len(line) == 3:
        track_id, image_id, url_or_filepath = line
    else:
        raise NotImplementedError
    
    progress_bar(cur, total, None, image_id)
    cur += 1

    if url_or_filepath.startswith('http'):
        image = image_from_url(url_or_filepath)
    elif os.path.isfile(url_or_filepath):
        image = cv2.imread(url_or_filepath)
    else:
        return

    image_id = get_file_name(url_or_filepath)

    if image is None:
        print(f'W: image "{image_id}" request failed.')
        return

    orig_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0  # 转成0~1之间

    sample = val_transform(**{
        'image': image,
    })

    image = sample['image']
    image = image.unsqueeze(0)
    image = image.to(device=opt.device)

    feats = model.classifier.transformer.forward_intermediate_layers(image)
    last_feat = feats[-1][0].detach().cpu().numpy().astype(np.float16)

    save_pickle(f'dino_feat/{image_id}.pkl', last_feat)

if __name__ == '__main__':
    from options import base_options, parse_args, set_config
    from network import get_model
    from dataloader.transforms import get_transform

    opt = base_options()
    opt.add_argument('--input', type=str, required=True, help='指定测试路径,可以是文件夹/list text file')
    opt = parse_args(opt)
    config = set_config(opt)

    with Context('init model'):
        model = get_model(opt, config)
        model = model.to(device=opt.device)
        model.eval()

    with Context('init data'):
        val_transform = get_transform(config, phase='val')

    with Context('handle input'):
        if os.path.isfile(opt.input):
            lines = file_lines(opt.input)
        elif os.path.isdir(opt.input):
            files = os.listdir(opt.input)
            lines = [join(opt.input, file) for file in files]
        else:
            raise FileNotFoundError(f'{opt.input} not found.')

    with Context('handle output'):
        os.makedirs('dino_feat', exist_ok=True)

    jobs = lines
    total = len(jobs)
    if opt.debug:
        for job in jobs:
            worker(job)
    else:
        pool = ThreadPool(processes=15)
        job_out = pool.map(worker, jobs)
        pool.close()
        pool.join()

