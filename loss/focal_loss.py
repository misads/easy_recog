import torch
from torch import nn
from torch.nn import functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        """
        label = 1 数量少时, alpha = 0.75
        label = 0 数量少时, alpha = 0.25
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predicted, label):
        predicted = torch.sigmoid(predicted)
        predicted = torch.clamp(predicted, 1e-4, 1.0 - 1e-4)

        bce = F.binary_cross_entropy(predicted, label, reduction='none')
        
        alpha_factor = torch.ones_like(label) * self.alpha

        alpha_factor = torch.where(torch.eq(label, 1.), alpha_factor, 1. - alpha_factor)

        focal_weight = torch.where(torch.eq(label, 1.), 1. - predicted, predicted)
        focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

        focal_loss = focal_weight * bce

        bce = bce.mean()
        focal_loss = focal_loss.mean() * 10
        return focal_loss