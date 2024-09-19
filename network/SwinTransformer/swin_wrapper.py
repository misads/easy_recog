import numpy as np
import torch
import torch.nn as nn

from .swin_transformer import SwinTransformer
from misc_utils import color_print
from utils import is_first_gpu

checkpoint_url = {
    'Swin-Tiny': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
    'Swin-Small': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth',
    'Swin-Base': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth'
}

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)


class Classifier(nn.Module):
    def __init__(self, arch):
        super(Classifier, self).__init__()
        pretrained_url = checkpoint_url[arch]
        if arch == 'Swin-Tiny':
            num_feats = 768
            self.network = SwinTransformer(
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.2,
                patch_norm=True,
                out_indices=(0, 1, 2, 3),
                with_cp=False,
                convert_weights=True,
                init_cfg=dict(type='Pretrained', checkpoint=pretrained_url)
            )
        elif arch == 'Swin-Small':
            num_feats = 768
            self.network = SwinTransformer(
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.2,
                patch_norm=True,
                out_indices=(0, 1, 2, 3),
                with_cp=False,
                convert_weights=True,
                init_cfg=dict(type='Pretrained', checkpoint=pretrained_url)
            )
        elif arch == 'Swin-Base':
            num_feats = 1024
            self.network = SwinTransformer(
                embed_dims=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.5,
                patch_norm=True,
                out_indices=(0, 1, 2, 3),
                with_cp=False,
                convert_weights=True,
                init_cfg=dict(type='Pretrained', checkpoint=pretrained_url)
            )
        else:
            raise NotImplementedError(f'arch "{arch}" not implemented error.')
        
        self.network.init_weights()
        
        self.avgpool = GlobalAvgPool2d()
        self.fc = nn.Linear(num_feats, classes)

    def forward(self, input):
        x = input
        x = self.network(x)

        layer1, layer2, layer3, layer4, *layer5 = x

        x = self.avgpool(layer4)
        #x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

