# python 3.5, pytorch 1.14

import os
from collections import defaultdict

# import dataloader as dl
# from options import opt, init_log
# from dataloader.example import class_names

import torch
import numpy as np
import csv
from utils import raise_exception
from misc_utils import try_make_dir, progress_bar
from prettytable import PrettyTable
from sklearn.metrics import roc_auc_score

def evaluate(model, opt, config, dataloader, epochs, logger, data_name='val'):
    class_names = config.data.class_names
    _soft_label = config.data.soft_label
    _one_hot_label = config.data.one_hot_label
    _only_one_class = len(class_names) == 2
    correct = 0
    ct_num = 0
    acc_dict = defaultdict(float)
    total_nums = defaultdict(float)
    pred_nums = defaultdict(int)
    all_scores = []
    all_labels = []
    # print('Start testing ' + tag + '...')
    # if opt.export:
    #     f2 = open('results.csv', 'w', encoding='utf-8')
    #     csv_writer = csv.writer(f2)

    #     csv_writer.writerow(['filename', '真值', '预测结果', 'score'])

    for i, data in enumerate(dataloader):
        if data_name == 'val':
            input, labels, filenames = data['input'], data['label'], data['filename']
            progress_bar(i, len(dataloader), 'Eva... ')

            with torch.no_grad():
                image = input.to(device=opt.device)
                labels = labels.to(device=opt.device)
                logits = model(image)

                if _soft_label:
                    scores = torch.sigmoid(logits)[:, 0]
                    predicted = (scores > 0.5).long()
                else:
                    logits = torch.softmax(logits, dim=1)
                    scores, predicted = torch.max(logits, 1)

                    if _only_one_class:
                        scores = logits[:, -1]

                if _one_hot_label:
                    _, labels = torch.max(labels, 1)

                predicted = predicted.cpu().detach().numpy()
                scores = scores.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                for predict, label, score, filename in zip(predicted, labels, scores, filenames):
                    predict_tag = class_names[predict]
                    label_tag = class_names[label]
                    all_scores.append(score)
                    all_labels.append(label)
                    # if opt.export:
                    #     csv_writer.writerow([filename, label_tag, predict_tag, score])

                    pred_nums[predict] += 1
                    total_nums[label] += 1
                    ct_num += 1

                    if predict == label:
                        acc_dict[predict] += 1
                        correct += 1
        else:
            raise Exception('Unknown dataset name: %s.' % data_name)

    # if opt.export:
    #     f2.close()

    if data_name == 'val':

        class_table = PrettyTable(field_names=['class_name', 'acc', 'recall'])
        mean_acc = 0.
        mean_recall = 0.
        logger.info('Eva(%s) epoch %d, ' % (data_name, epochs))
        for label in range(len(class_names)):
            label_name = class_names[label]
            acc = 0.
            recall = 0.
            if pred_nums[label] and total_nums[label]:
                acc = acc_dict[label] / pred_nums[label]
                # print(f'{label_name}\tacc: {acc}')
                recall = acc_dict[label] / total_nums[label]
                # logger.info(f'{label_name}\tacc: {acc}\trecall: {recall}')
                class_table.add_row([label_name, f'{acc:.4f}', f'{recall:.4f}'])
            mean_acc += acc
            mean_recall += recall

        mean_acc = mean_acc / len(class_names)
        mean_recall = mean_recall / len(class_names)
        
        logger.info('per class acc/recall: \n' + class_table.get_string())

        class_table = PrettyTable(field_names=None)  # ['', 'acc', 'recall']

        class_table.add_row(['total acc', f'{correct / float(ct_num):.4f}'])
        class_table.add_row(['mean acc', f'{mean_acc:.4f}'])
        class_table.add_row(['mean recall', f'{mean_recall:.4f}'])
        if _only_one_class:
            auc = roc_auc_score(all_labels, all_scores)
            class_table.add_row(['auc', f'{auc:.4f}'])

            sorted_all_scores = sorted(all_scores)
            _0_count = all_labels.count(0)
            best_thresh = 0.5 * (sorted_all_scores[_0_count - 1] + sorted_all_scores[_0_count])
            class_table.add_row(['best thresh', f'{best_thresh:.4f}'])

        logger.info('total acc/recall: \n' + class_table.get_string(header=False))
        # logger.info('Eva(%s) epoch %d, ' % (data_name, epochs) + 'conf_matrix: \n' + str(conf_matrix) + '.')
        # logger.info('Eva(%s) epoch %d, ' % (data_name, epochs) + 'Acc: ' + str(correct / float(ct_num)) + '.')
        # logger.info('Eva(%s) epoch %d, ' % (data_name, epochs) + f'Mean Acc: {mean_acc: .5f}.')

        return ''
    else:
        return ''


if __name__ == '__main__':
    from options import base_options, parse_args, set_config
    from dataloader import get_dataloader
    from network import get_model
    from utils import init_log

    opt = base_options()
    opt.add_argument('--load', type=str, default=None, help='指定载入checkpoint的路径')
    opt = parse_args(opt)
    config = set_config(opt)

    if not opt.load:
        print('Usage: eval.py [--tag TAG] --load LOAD')
        raise_exception('eval.py: the following arguments are required: --load')

    model = get_model(opt, config)
    model = model.to(device=opt.device)

    load_epoch = model.load(opt.load)
    if load_epoch is not None:
        opt.which_epoch = load_epoch

    model.eval()

    val_dataloader = get_dataloader(opt, config, 'val')
    logger = init_log(opt, dir_suffix='(val)')

    evaluate(model, opt, config, val_dataloader, load_epoch, logger, 'val')

