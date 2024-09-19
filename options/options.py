import argparse
import os
import torch
from misc_utils import get_file_name, get_time_str
from utils import parse_config, raise_exception, get_command_run
from os.path import join

def base_options():
    # experiment specifics
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str, default=None,
                        help='指定保存结果的路径, 如果未指定, 结果会保存在与配置文件文件名相同路径')

    parser.add_argument('--config', type=str, required=True, help='(必须指定) yml配置文件')

    # parser.add_argument('--gpu_id', '--gpu', type=int, default=0, help='gpu id: e.g. 0 . use -1 for CPU')
    parser.add_argument("--local_rank", type=int, default=None, help='only used in dist train mode')
    
    # # training options
    parser.add_argument('--debug', action='store_true', help='debug模式')
    parser.add_argument('--vis', action='store_true', help='可视化测试结果')

    return parser


def parse_args(options):
    return options.parse_args()


def set_config(opt):
    if opt.config:
        config = parse_config(opt.config)
    else:
        raise_exception('--config must be specified.')

    if not opt.save_dir:
        opt.save_dir = join('work_dir', get_file_name(opt.config))

    gpu_id = 0 if opt.local_rank is None else opt.local_rank
    opt.device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'

    if opt.debug:
        config.misc.save_freq = 1
        config.misc.val_freq = 1

    if not opt.debug:
        pid = f'[PID:{os.getpid()}]'
        with open('run_log.txt', 'a') as f:
            f.writelines(get_time_str(fmt="%Y-%m-%d %H:%M:%S") + ' ' + pid + ' ' + get_command_run() + '\n')
        
    return config


