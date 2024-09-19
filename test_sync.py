# python 3.5, pytorch 1.14

import os
import cv2
from network import get_model

import ipdb
import torch
import numpy as np
import multiprocessing
from utils import raise_exception, Context
from os.path import join
from multiprocessing.pool import ThreadPool
from misc_utils import color_print, file_lines, get_file_name, progress_bar
from utils import image_from_url

lock = multiprocessing.Manager().Lock()
total = 0
cur = 0    

def worker(line):
    global cur, total, lock
    delimeter = '\t' if '\t' in line else ','
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
    if image_id in already_set:
        return

    if url_or_filepath.startswith('http'):
        image = image_from_url(url_or_filepath)
    elif os.path.isfile(url_or_filepath):
        image = cv2.imread(url_or_filepath)
    else:
        return

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

    logits = model(image)

    if _soft_label:
        logits = torch.sigmoid(logits)
    else:
        logits = torch.softmax(logits, dim=1)

    if _only_one_class or _soft_label:
        score_list = logits.squeeze().detach().cpu().numpy().tolist()
        score = logits[:, 0]
        predicted = (score > opt.thresh).long()
    else:
        score, predicted = torch.max(logits, 1)
        score_list = [score.item()]

    predicted = predicted.item()
    score = score.item()

    predicted_name = class_names[predicted]

    lock.acquire()
    with open(out_path, 'a', encoding='utf-8') as f:
        # json.dump(ret, f, ensure_ascii=False, indent=2, cls=MyEncoder)
        score_str = ','.join(map(lambda x: f'{x:.4f}', score_list))
        f.writelines(f'{track_id},{url_or_filepath},{predicted_name},{score_str}' + '\n')
    lock.release()

    if opt.vis:
        os.makedirs(f'vis/{predicted_name}', exist_ok=True)
        cv2.imwrite(f'vis/{predicted_name}/{image_id}.jpg', orig_image)

if __name__ == '__main__':
    from options import base_options, parse_args, set_config
    from network import get_model
    from dataloader.transforms import get_transform

    opt = base_options()
    opt.add_argument('--load', type=str, default=None, help='指定载入checkpoint的路径')
    opt.add_argument('--input', type=str, required=True, help='指定测试路径,可以是文件夹/list text file')
    opt.add_argument('--start', type=int, default=0, help='开始测试的条目')
    opt.add_argument('--to', type=int, default=99999999, help='结束测试的条目')
    opt.add_argument('--thresh', '-t', type=float, default=0.5, help='二分类的阈值')
    opt = parse_args(opt)
    config = set_config(opt)

    if not opt.load:
        print('Usage: eval.py [--tag TAG] --load LOAD')
        raise_exception('eval.py: the following arguments are required: --load')

    with Context('init model'):
        model = get_model(opt, config)
        model = model.to(device=opt.device)

        load_epoch = model.load(opt.load)
        if load_epoch is not None:
            opt.which_epoch = load_epoch

        model.eval()

    with Context('init data'):
        class_names = config.data.class_names
        _soft_label = config.data.soft_label
        _only_one_class = len(class_names) == 2
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
        if opt.vis:
            os.makedirs('vis', exist_ok=True)

        config_name = get_file_name(opt.config)
        input_name = get_file_name(opt.input.rstrip('/'))
        out_path = f'result_{config_name}_{input_name}.txt'

        already_set = set()
        color_print(f'save result at "{out_path}"', 2)
        if os.path.isfile(out_path):
            with open(out_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    url = line.split(',')[1]
                    image_id = get_file_name(url)
                    already_set.add(image_id)
        else:
            pass

    jobs = lines
    jobs = jobs[opt.start: opt.to]
    total = len(jobs)
    if opt.debug:
        for job in jobs:
            worker(job)
    else:
        pool = ThreadPool(processes=15)
        job_out = pool.map(worker, jobs)
        pool.close()
        pool.join()

