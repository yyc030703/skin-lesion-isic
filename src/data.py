"""
PyTorch Dataset classes for ISIC 2018.

- ISICClassificationDataset: reads from a splits CSV (train/val/test) and
  returns (image, label) pairs for the 7-class classification task.
- ISICSegmentationDataset: reads paired image+mask files for the
  binary lesion segmentation task.

Both accept a transform callable. For Albumentations transforms, pass them
directly — they handle numpy arrays. For torchvision transforms, you'd
adapt the __getitem__ to use PIL images.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ISICClassificationDataset(Dataset):
    """7-class skin lesion classification (ISIC 2018 Task 3)."""

    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = np.array(Image.open(row["image_path"]).convert("RGB"))
        label = int(row["label"])

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label


class ISICSegmentationDataset(Dataset):
    """Binary lesion segmentation (ISIC 2018 Task 1)."""

    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        # Match images to masks by ISIC ID. Mask naming convention:
        # image  ISIC_0000000.jpg  ->  mask  ISIC_0000000_segmentation.png
        self.image_files = sorted(
            f for f in self.images_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg"}
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.masks_dir / f"{image_path.stem}_segmentation.png"

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.float32)  # binarize to {0, 1}

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        return image, mask