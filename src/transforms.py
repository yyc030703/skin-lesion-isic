"""
Albumentations transforms for ISIC 2018.

Usage:
    from transforms import get_clf_transforms, get_seg_transforms

    train_tf = get_clf_transforms("train", input_size=224)
    val_tf   = get_clf_transforms("val",   input_size=224)

    train_seg_tf = get_seg_transforms("train", input_size=256)
    val_seg_tf   = get_seg_transforms("val",   input_size=256)

Then pass directly as transform= to ISICClassificationDataset or
ISICSegmentationDataset — both expect numpy arrays in, numpy arrays out,
which Albumentations does by default.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalization stats
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


def get_clf_transforms(split: str, input_size: int = 224) -> A.Compose:
    """
    Transforms for ISICClassificationDataset.

    Args:
        split:      "train" or "val" / "test"
        input_size: target square resolution (224 for ResNet-50,
                    380 for EfficientNet-B4)

    Returns:
        An A.Compose callable. Call as:
            transformed = transform(image=np_image)
            image_tensor = transformed["image"]   # float32 CHW tensor
    """
    if split == "train":
        return A.Compose([
            A.RandomResizedCrop(
                size=(input_size, input_size),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
                p=0.5,
            ),
            # Simulate hair and ruler artifacts common in dermoscopy images
            A.CoarseDropout(
                max_holes=8,
                max_height=input_size // 16,
                max_width=input_size // 16,
                fill_value=0,
                p=0.3,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ])
    else:  # val / test
        return A.Compose([
            A.Resize(input_size, input_size),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ])


def get_seg_transforms(split: str, input_size: int = 256) -> A.Compose:
    """
    Transforms for ISICSegmentationDataset.
    Image and mask are augmented jointly so they stay aligned.

    Args:
        split:      "train" or "val" / "test"
        input_size: target square resolution (256 or 384)

    Returns:
        An A.Compose callable. Call as:
            out = transform(image=np_image, mask=np_mask)
            image_tensor = out["image"]   # float32 CHW tensor
            mask_tensor  = out["mask"]    # float32 HW tensor, values in {0, 1}
    """
    if split == "train":
        return A.Compose([
            A.RandomResizedCrop(
                size=(input_size, input_size),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ElasticTransform(
                alpha=120,
                sigma=6,
                p=0.3,
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ])
    else:  # val / test
        return A.Compose([
            A.Resize(input_size, input_size),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ])
