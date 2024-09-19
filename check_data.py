import os
from tabnanny import check
import cv2
from os.path import join
from misc_utils import file_lines, progress_bar
from multiprocessing.pool import ThreadPool
from PIL import Image

CHECK_IMAGE_CORRUPTED = False

def check_image(path):
    try:
        Image.open(path).load()
    except InterruptedError:
        exit()
    except Exception:
        print('ERROR: %s' % path)
        return False
    else:
        return True

check_list = ['train.txt', 'val.txt']

num_not_exist = 0
num_corrupted = 0
cur = 0
total = 0
def worker(filepath):
    global num_not_exist, num_corrupted, cur, total
    progress_bar(cur, total)
    cur += 1
    if not os.path.isfile(filepath):
        num_not_exist += 1
        print(f'{filepath} not exist.')
        return False
    if CHECK_IMAGE_CORRUPTED:
        if not check_image(filepath):
            print(f'{i}/{len(lines)} {filepath} corrupted.')
            num_corrupted += 1
            # os.remove(filepath)
            with open('bad_images.txt', 'a') as f2:
                f2.writelines(filepath + '\n')
            return False

    return True

def wget_image(url, image_path):
    global cur, total
    print(f'downloading...{cur}/{total} {image_path}')
    cur += 1
    if os.path.isfile(image_path):
        return
    cmd = f'wget {url} -O {image_path} -q'
    os.system(cmd)

num = 0
for file in check_list:
    jobs = []
    cur = 0
    file = join('datasets', file)
    print(f'check "{file}"')
    lines = file_lines(file)
    for i, line in enumerate(lines):
        filepath, label = line.split(' ')
        jobs.append(filepath)

    total = len(jobs)
    pool = ThreadPool(processes=15)
    job_out = pool.map(worker, jobs)
    pool.close()
    pool.join()


if not CHECK_IMAGE_CORRUPTED:
    num_corrupted = 'not checked'

print(f'not exist: {num_not_exist}, num corrupted: {num_corrupted}.')