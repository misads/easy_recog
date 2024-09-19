from .LookAhead import Lookahead
from .RAdam import RAdam
from .Ranger import Ranger
from torch import optim

def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # if len(param.shape) == 1:
            #     color_print(f'{name} decay: 0', 4)
            # else:
            #     color_print(f'{name} decay: 0', 1)
        else:
            has_decay.append(param)
            # color_print(f'{name} has_decay', 3)

    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def get_optimizer(opt, config, module):
    optimizer_type = config.optimizer.type
    lr = config.optimizer.lr

    if 'Swin' in config.model.backbone.type:
        skip_keywords = {'.absolute_pos_embed', '.relative_position_bias_table', '.norm'}
        parameters = set_weight_decay(module, skip_list={}, skip_keywords=skip_keywords)
        optimizer = optim.AdamW(parameters,
                                lr=lr,  # 5e-4 * batch_size * gpu_num / 512
                                weight_decay=0.05,
                                eps=1e-8,
                                betas=(0.9, 0.999))
        return optimizer

    if optimizer_type == 'adam':
        optimizer = optim.Adam(module.parameters(), lr=lr, betas=(0.95, 0.999))
    elif optimizer_type == 'sgd':  # 从头训练 lr=0.1 fine_tune lr=0.01
        optimizer = optim.SGD(module.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    elif optimizer_type == 'radam':
        optimizer = RAdam(module.parameters(), lr=lr, betas=(0.95, 0.999))
    elif optimizer_type == 'lookahead':
        optimizer = Lookahead(module.parameters())
    elif optimizer_type == 'ranger':
        optimizer = Ranger(module.parameters(), lr=lr)
    else:
        raise NotImplementedError

    return optimizer