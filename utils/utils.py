import numpy as np
import yaml
import os
import sys
import cv2
from misc_utils import try_make_dir, get_logger, get_time_str, color_print
from collections import OrderedDict

def raise_exception(msg, error_code=1):
    color_print('Exception: ' + msg, 1)
    exit(error_code)

def raise_warning(msg):
    color_print('Warning: ' + msg, 1)

def raise_info(msg):
    color_print('Info: ' + msg, 4)

class EasyDict:
    def __init__(self, data: dict):
        self._dict = data

    def update(self, data):
        self._dict.update(data)

    def get(self, data, default):
        return self._dict.get(data, default)

    def __iter__(self):
        return self._dict.__iter__()

    def __setattr__(self, attrname, value):
        if attrname == '_dict':
            return super(EasyDict, self).__setattr__(attrname, value)

        self._dict[attrname] = value

    def __getattr__(self, attrname):
        if attrname in self._dict:
            attvalue = self._dict[attrname]
            if isinstance(attvalue, dict):
                return EasyDict(attvalue)
            else:
                return attvalue

        return None

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __setitem__(self, key, value):
        return self.__setattr__(key, value)

    def __repr__(self):
        return str(self._dict)

class Context(object): 
    def __init__(self, msg=''):
        pass

    def __enter__(self):
        pass
 
    def __exit__(self, type, value, trace):
        pass

def parse_config(yml_path):
    if not os.path.isfile(yml_path):
        raise_exception(f'{yml_path} not exists.')

    with open(yml_path, 'r') as f:
        try:
            configs = yaml.safe_load(f.read())
        except yaml.YAMLError:
            raise_exception('Error parsing YAML file:' + yml_path)

    parsed_configs = EasyDict(configs)
    return parsed_configs

def get_command_run():
    args = sys.argv.copy()
    args[0] = args[0].split('/')[-1]

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_id = os.environ['CUDA_VISIBLE_DEVICES']
        command = f'CUDA_VISIBLE_DEVICES={gpu_id} '
    else:
        command = ''

    if sys.version[0] == '3':
        command += 'python3'
    else:
        command += 'python'

    for i in args:
        command += ' ' + i
    return command

def init_log(opt, dir_suffix=None):
    time_str = get_time_str(fmt="%Y-%m-%d_%H:%M:%S")

    if dir_suffix is None:
        log_dir = os.path.join(opt.save_dir, time_str)
    else:
        log_dir = os.path.join(opt.save_dir, time_str + dir_suffix)

    try_make_dir(log_dir)

    color_print(f'save result to "{log_dir}"', 3)

    copy_cmd = f'cp {opt.config} "{log_dir}"'
    os.system(copy_cmd)

    logger = get_logger(f=os.path.join(log_dir, 'log.txt'), level='info', mode='a')

    logger.info('==================Options==================')
    for k, v in opt._get_kwargs():
        logger.info(str(k) + '=' + str(v))
    logger.info('===========================================')
    return logger

def get_gpu_id(opt):
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_id = os.environ['CUDA_VISIBLE_DEVICES']
        gpu_id = str(gpu_id)
    else:
        gpu_id = str(opt.gpu_id)

    return gpu_id

def is_first_gpu(opt):
    # used in distributed mode
    return not opt.local_rank

def is_distributed(opt):
    return opt.local_rank is not None

# 设置多卡训练
def setup_multi_processes(opt):
    import torch.distributed as dist
    import torch
    import cv2
    import os

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(opt.local_rank)

    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    if 'OMP_NUM_THREADS' not in os.environ:
        omp_num_threads = 1
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in os.environ:
        mkl_num_threads = 1
        os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)

def read_image(image_path, flag=cv2.IMREAD_COLOR):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'{image_path} not found.')

    image = cv2.imread(image_path, flag)
    return image

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

def denormalize(img, max_pixel_value=1.):
    global mean, std
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    img *= std
    img += mean
    return img

def load_ckpt(model, state_dict, submodule=None, exclude=None, include=None, vis=True):
    ''' Load model state dict from a pretrained state dict.
    Args:
        model: network
        state_dict: loaded state dict, OrderedDict
        submodule: a string or None, indicates which module of model to load
            parameters, e.g., submodule='backbone' for RetinaNet, means to load
            retinanet.backbone(usually resnet50/resnet101) parameters.
        exclude: a list of strings or None, if None, load all model params,
            otherwise, only load model params whose name not startswith one of
            excluded string, e.g., exclude=['head'] means do not load model.head
            parameters.
        include: a list of strings or None, if None, load 0 model params
    Returns:
        An int number indicating the total count of updated parameters.
    '''
    if submodule:
        for module in submodule.split('.'):
            model = getattr(model, module)

    model_dict = model.state_dict()

    assert len(model_dict) > 0 and len(state_dict) > 0
    updated_keys = model_dict.keys()

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    if include:
        if not isinstance(include, list):
            raise TypeError(f'include {include} should be a list')
        updated_keys = filter(lambda x: any(map(x.startswith, include)), updated_keys)
    if exclude:
        if not isinstance(exclude, list):
            raise TypeError(f'exclude {exclude} should be a list')
        updated_keys = filter(lambda x: not any(map(x.startswith, exclude)), updated_keys)
    succ = 0
    flags = []

    loaded_keys = set()

    for key in updated_keys:
        if key in state_dict:
            if model_dict[key].shape == state_dict[key].shape:
                model_dict[key] = state_dict[key].type_as(model_dict[key])
                succ += 1
                loaded_keys.add(key)
                flags.append('succeed')
            else:
                flags.append('size mismatch')
                if vis:
                    print(f'{key} size mismath')
        else:
            flags.append('fail')
        
    if vis:
        color_print(f'load succeed: {succ}, model params: {len(model_dict)}, ckpt params: {len(state_dict)}', 3)

        if succ != len(model_dict) and succ / len(model_dict) > 0.8:
            print('not loaded:')
            for key in model_dict:
                if key not in loaded_keys:
                    print(f'    {key}')

        step = len(flags) // 100 + 1
        flags = flags[::step]
        for flag in flags:
            if flag == 'succeed':
                print('\033[0;32m■', end='')
            elif flag == 'fail':
                print('\033[0;31m■', end='')
            else:
                print('\033[0;35m■', end='')

        print('\033[0m')
    # assert succ == len(model_dict), 'model not matched with checkpoint.'
    model.load_state_dict(model_dict)

    return succ

def convert_ckpt(state_dict):
    converted_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k.replace('module.', '')
            converted_dict[k] = v

    return converted_dict