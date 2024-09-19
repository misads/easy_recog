from .label_smooth import LabelSmoothing
from .focal_loss import FocalLoss
import torch
from torch import nn
from torch.nn import functional as F

loss_name_dict = {
    'ce': nn.CrossEntropyLoss(),
    'bce': nn.BCEWithLogitsLoss(),
    'focal': FocalLoss(alpha=0.75, gamma=2.0)
}

# if opt.smooth != 0:
#     label_smooth_loss = LabelSmoothing(smoothing=opt.smooth)
# else:
#     label_smooth_loss = 0.

class Loss(nn.Module):
    def __init__(self, opt, config):
        super(Loss, self).__init__()
        self.loss_config = config.loss
        self.loss_dict = {}
        for loss_name in self.loss_config:
            self.loss_dict[loss_name] = loss_name_dict[loss_name]

    def forward(self, predicted, label, avg_meters):
        predicted = predicted.squeeze(1)

        loss_dict = {}
        for loss_name, loss_impl in self.loss_dict.items():
            loss_dict[loss_name] = loss_impl(predicted, label)

        total_loss = sum(loss_dict.values())

        loss_display = {k: v.item() for k, v in loss_dict.items()}
        loss_display.update({'total': total_loss.item()})
        avg_meters.update(loss_display)

        return total_loss


def build_loss(opt, config):
    return Loss(opt, config)


# def get_loss(predicted, label, avg_meters, *args):
#     ce_loss = calc_cross_entropy(predicted, label)

#     loss = ce_loss

#     avg_meters.update({'ce_loss': ce_loss.item()})
#     return loss