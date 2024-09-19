import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def random_sized_crop(scale):
    return A.RandomResizedCrop(height=scale, width=scale, scale=(0.7, 1.0), p=1.0)

def resize(scale):
    return A.Resize(height=scale, width=scale, p=1.0)

def normalize():
    return A.Normalize(max_pixel_value=1.0, p=1.0)

def color_shift(p=0.1):
    return A.OneOf([
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2,
                                val_shift_limit=0.2, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.1,
                                    contrast_limit=0.1, p=1.0),
    ], p=p)

def to_tensor():
    return ToTensorV2(p=1.0)

def get_transform(config, phase='train'):
    assert phase in {'train', 'val'}

    transform_config = config.data[phase].transform
    transforms = []
    for transform in transform_config:
        if isinstance(transform, dict):
            for k, v in transform.items():
                transforms.append(globals()[k](v))
        else:
            transforms.append(globals()[transform]())

    return A.Compose(transforms, p=1.0)

