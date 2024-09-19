from torch import optim

def get_scheduler(opt, config, optimizer):
    schedulr_type = config.scheduler.type
    epochs = config.runner.epochs
    lr = config.optimizer.lr
    if schedulr_type == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)
    elif schedulr_type is None:
        def lambda_decay(step) -> float:
            return 1.
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda_decay)
    return scheduler