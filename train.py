# encoding = utf-8
"""
    Author: misads
"""
import time

import torch
import torch.distributed as dist
import traceback

from options import base_options, parse_args, set_config
from utils import init_log, is_distributed, is_first_gpu, setup_multi_processes, Context

from dataloader import get_dataloader

from network import get_model
from eval import evaluate

from mscv.summary import create_summary_writer, write_meters_loss

from misc_utils import format_time, progress_bar

# 初始化
with Context():
    opt = base_options()
    opt = parse_args(opt)
    config = set_config(opt)

    # 在dataloader初始化前要先初始化多卡同步
    if is_distributed(opt):
        setup_multi_processes(opt)

    # 初始化路径
    save_root = opt.save_dir
    train_dataloader = get_dataloader(opt, config, 'train')
    val_dataloader = get_dataloader(opt, config, 'val')

    # 初始化日志
    if is_first_gpu(opt):
        logger = init_log(opt)

    # 初始化模型
    model = get_model(opt, config)

    # model = model.to(device=opt.device)

    # 加载预训练模型，恢复中断的训练
    if config.model.load_from:
        load_epoch = model.load(config.model.load_from)
        start_epoch = load_epoch + 1 if config.model.resume else 1
    else:
        start_epoch = 1

    # 开始训练
    model.train()

    # 计算开始和总共的step
    if is_first_gpu(opt):
        print('Start training...')
    start_step = (start_epoch - 1) * len(train_dataloader)
    global_step = start_step
    total_steps = config.runner.epochs * len(train_dataloader)
    start = time.time()

    #  定义scheduler
    optimizer = model.optimizer
    scheduler = model.scheduler

    start_time = time.time()
    
    if is_first_gpu(opt):
        # 在日志记录transforms
        logger.info('train_trasforms: ' +str(train_dataloader.dataset.transforms))
        logger.info('===========================================')
        if val_dataloader is not None:
            logger.info('val_trasforms: ' +str(val_dataloader.dataset.transforms))
        logger.info('===========================================')


try:
    # 训练循环
    for epoch in range(start_epoch, config.runner.epochs + 1):
        for iteration, data in enumerate(train_dataloader):
            global_step += 1
            
            # 计算剩余时间
            rate = (global_step - start_step) / (time.time() - start)
            remaining = (total_steps - global_step) / rate

            img, label = data['input'], data['label']  # ['label'], data['image']  #

            img = img.to(device=opt.device)
            label = label.to(device=opt.device)

            # 更新模型参数
            model.update(img, label)

            pre_msg = 'Epoch:%d' % epoch

            # 显示进度条
            cur_lr = round(scheduler.get_lr()[0], 6)
            eta_time = format_time(remaining)
            msg = f'lr: {cur_lr:.6f} (loss) {str(model.avg_meters)} ETA: {eta_time}'
            if is_first_gpu(opt):
                progress_bar(iteration, len(train_dataloader), pre_msg, msg)
        
                # 训练时每100个step记录一下loss
                if global_step % config.misc.log_iter == 0:
                    loss_str = str(model.avg_meters)
                    logger.info(f'step: {global_step} lr: {cur_lr:.6f} (loss) {loss_str} Eta: {eta_time}')


        if is_first_gpu(opt):
            # 每个epoch结束后的显示信息
            logger.info(f'Train epoch: {epoch}, lr: {round(scheduler.get_lr()[0], 6) : .6f}, (loss) ' + str(model.avg_meters))

            if epoch % config.misc.save_freq == 0 or epoch == config.runner.epochs:  # 最后一个epoch要保存一下
                model.save(epoch)

        # 训练中验证
        if not config.misc.no_eval and epoch % config.misc.val_freq == 0:
            if is_first_gpu(opt):
                model.eval()
                evaluate(model, opt, config, val_dataloader, epoch, logger)
                model.train()

        if scheduler is not None:
            scheduler.step()

    if is_distributed(opt):
        dist.destroy_process_group()

except KeyboardInterrupt:
    with open('run_log.txt', 'a') as f:
        f.writelines('    KeyboardInterrupt.\n')
    print('press Ctrl+C, exit.')
    exit()

except Exception as e:
    if not opt.debug:
        with open('run_log.txt', 'a') as f:
            for i, line in enumerate(traceback.format_exc().split('\n')):
                if i == 0:
                    f.writelines('    Error: ' + line + '\n')
                else:
                    f.writelines('    ' + line + '\n')

    raise e  # 再引起一个异常，这样才能打印之前的错误信息
