from utils import raise_info
from .vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from misc_utils import color_print
import torch.nn as nn
import torch

# model_dict = {
#     'DINOv2_ViT-S_14': 'dinov2_vits14',  # 21M
#     'DINOv2_ViT-B_14': 'dinov2_vitb14',  # 86M
#     'DINOv2_ViT-L_14': 'dinov2_vitl14'   # 300M
# }

init_args = {
    "img_size": 518,
    "patch_size": 14,
    "init_values": 1.0,
    "ffn_layer": 'mlp',
    "block_chunks": 0
}

model_dict = {
    'DINOv2_ViT-S_14': (vit_small, init_args),  # 21M
    'DINOv2_ViT-B_14': (vit_base, init_args),  # 86M
    'DINOv2_ViT-L_14': (vit_large, init_args)   # 300M
}


class Classifier(nn.Module):
    def __init__(self, opt, config, zero_head=False, vis=False):
        super(Classifier, self).__init__()

        softlabel = config.data.soft_label
        class_names = config.data.class_names
        if softlabel:
            self.num_classes = 1
        else:
            self.num_classes = len(class_names)

        self.zero_head = zero_head

        model_config = config.model.backbone
        model_type = model_config.type
        pretrained = model_config.pretrained
        # img_size = model_config.img_size
        hidden_size = model_config.hidden_size
        # transformer_config = model_config.transformer

        # model_hub_type = model_dict[model_type]
        # raise_info(f'torch.hub.load: {model_hub_type}')
        # self.transformer = torch.hub.load('facebookresearch/dinov2', model_hub_type, pretrained=pretrained)

        model_func, model_args = model_dict[model_type]
        self.transformer = model_func(**model_args)

        if pretrained:
            state_dict = torch.load(pretrained, map_location='cpu')
            self.transformer.load_state_dict(state_dict)
            color_print(f'load pretrained from {pretrained}.', 2)


        self.head = nn.Linear(hidden_size, self.num_classes)

    def forward(self, x, labels=None):
        x = self.transformer(x)
        logits = self.head(x)

        return logits
