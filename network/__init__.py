from .ResNet.Model import Model as ResNet
from .ResNeSt.Model import Model as ResNeSt
from .SwinTransformer.Model import Model as SwinTransformer
from .ViT.Model import Model as ViT
from .DINOv2.Model import Model as DINOv2
from .MobileNetV2.Model import Model as MobileNetV2
from .base_model import BaseModel

models = {
    'MobileNetV2': MobileNetV2,
    'ResNet50': ResNet,
    'ResNet101': ResNet,
    'ResNeSt50': ResNeSt,
    'ResNeSt101': ResNeSt,
    'Swin-Tiny': SwinTransformer,
    'Swin-Small': SwinTransformer,
    'Swin-Base': SwinTransformer,
    'ViT-B_16': ViT,
    'ViT-B_32': ViT,
    'DINOv2_ViT-S_14': DINOv2,  # 21M
    'DINOv2_ViT-B_14': DINOv2,  # 86M
    'DINOv2_ViT-L_14': DINOv2   # 300M
}


def get_model(opt, config) -> BaseModel:
    model_type = config.model.backbone.type
    if model_type is None:
        raise AttributeError('--model MUST be specified now, available: {%s}.' % ('|'.join(models.keys())))

    if model_type in models:
        ModelClass = models[model_type]
        model = ModelClass(opt, config)
        return model
    else:
        raise Exception('No such model: "%s", available: {%s}.' % (model_type, '|'.join(models.keys())))

